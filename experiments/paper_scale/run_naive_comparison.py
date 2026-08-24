"""Trains the 4 winning configurations (seed 42 only) on the NAIVE random split
instead of the duplicate-aware one, to measure the direct accuracy effect of the
leakage the audit quantified. Entirely separate checkpoint directory and result-key
prefix from the main sweep, so there is zero risk of collision with it."""
import os
os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc, json, re, shutil, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed)
from datasets import Dataset

FINAL_DIR   = Path(__file__).resolve().parent
TABLES_DIR  = FINAL_DIR.parents[1] / 'tables'
WORK_DIR    = FINAL_DIR / 'work'
RESULTS_DIR = FINAL_DIR / 'results'
PROBS_DIR   = FINAL_DIR / 'probs'
CKPT_DIR    = Path('/media/filwel/MLProject1/nlp_paper_ckpt_naive')  # SEPARATE from the main job's ckpt dir
for d in (RESULTS_DIR, PROBS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_LEN, EPOCHS, WARMUP_RATIO, PATIENCE = 128, 5, 0.1, 2
MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}
WINNERS = {
    ('D1', 'BERT'):    dict(lr=3e-05, bs=32, wd=0.1),
    ('D1', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.01),
    ('D2', 'BERT'):    dict(lr=2e-05, bs=16, wd=0.1),
    ('D2', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.1),
}

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def load_naive_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}_naive.npz')
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}

def weighted_metrics(y, p):
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a,4), round(pr,4), round(rc,4), round(f,4)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, pre, rec, f1 = weighted_metrics(labels, preds)
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}

def atomic_write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def atomic_write_npz(path, **arrays):
    path = Path(path); tmp = path.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays); fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def find_checkpoint(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists(): return None
    cks = [p for p in run_dir.glob('checkpoint-*') if (p / 'trainer_state.json').exists()]
    return str(max(cks, key=lambda p: int(p.name.split('-')[1]))) if cks else None

_TOKCACHE, _DATACACHE = {}, {}
def get_tokenizer(mk):
    if mk not in _TOKCACHE: _TOKCACHE[mk] = AutoTokenizer.from_pretrained(MODELS[mk])
    return _TOKCACHE[mk]

def get_tokenized(tag, mk):
    key = (tag, mk)
    if key in _DATACACHE: return _DATACACHE[key]
    df, splits = load_naive_split(tag)
    tok = get_tokenizer(mk)
    parts = {}
    for split, idx in splits.items():
        sub = df.loc[idx]
        ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                                'labels': [int(v) for v in sub['label']]})
        parts[split] = ds.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
                              batched=True, remove_columns=['text'])
    _DATACACHE[key] = (parts, splits)
    del df; gc.collect()
    return _DATACACHE[key]

def run_one_naive(tag, mk, seed=42, verbose=True):
    cfg = WINNERS[(tag, mk)]
    lr, bs, wd = cfg['lr'], cfg['bs'], cfg['wd']
    key = 'naive_%s_%s_lr%g_bs%d_wd%g_s%d' % (tag, mk, lr, bs, wd, seed)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        if verbose: print('[skip] %s test_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    run_dir = CKPT_DIR / key
    resume = find_checkpoint(run_dir)
    parts, splits = get_tokenized(tag, mk)
    tok = get_tokenizer(mk)

    attempts = []
    cur = bs
    while cur >= 4:
        attempts.append((cur, bs // cur)); cur //= 2
    if not attempts: attempts = [(bs, 1)]

    for attempt_i, (per_device_bs, accum) in enumerate(attempts):
      try:
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[mk], num_labels=2)
        model.config.id2label = {0: 'human', 1: 'ai'}; model.config.label2id = {'human': 0, 'ai': 1}
        args = TrainingArguments(
            output_dir=str(run_dir), learning_rate=lr,
            per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=accum,
            per_device_eval_batch_size=min(32, per_device_bs*2), weight_decay=wd,
            num_train_epochs=EPOCHS, warmup_ratio=WARMUP_RATIO, lr_scheduler_type='linear',
            optim='adamw_torch', bf16=True, eval_strategy='epoch', save_strategy='epoch',
            save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='eval_f1',
            greater_is_better=True, logging_steps=200, seed=seed, data_seed=seed,
            dataloader_num_workers=0, report_to='none', disable_tqdm=not verbose)
        trainer = Trainer(model=model, args=args, train_dataset=parts['train'],
                          eval_dataset=parts['val'], data_collator=DataCollatorWithPadding(tok),
                          compute_metrics=compute_metrics,
                          callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume)
        train_secs = time.time() - t0
        break
      except torch.cuda.OutOfMemoryError:
        print('    OOM at batch %d, falling back' % per_device_bs)
        try: del trainer
        except NameError: pass
        try: del model
        except NameError: pass
        gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(run_dir, ignore_errors=True)
        resume = None
        if attempt_i == len(attempts)-1: raise

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag], 'model': mk,
           'checkpoint': MODELS[mk], 'lr': lr, 'batch_size': bs, 'weight_decay': wd, 'seed': seed,
           'max_len': MAX_LEN, 'scale': 'full_balanced', 'split': 'naive_random',
           'n_train': len(splits['train']), 'n_val': len(splits['val']), 'n_test': len(splits['test']),
           'train_seconds': round(train_secs, 1), 'epochs_run': int(trainer.state.epoch or 0),
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2)}

    probs = {}
    for split in ('val', 'test'):
        pred = trainer.predict(parts[split])
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        p = torch.softmax(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}
        out[split + '_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs['%s_probs' % split] = p; probs['%s_labels' % split] = y

    atomic_write_npz(ppath, **probs)
    atomic_write_json(jpath, out)
    del trainer, model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(run_dir, ignore_errors=True)
    if verbose:
        print('[done] %-46s test_f1=%.4f  %.1f min' % (key, out['test']['f1'], train_secs/60))
    return out

if __name__ == '__main__':
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available())
    t0 = time.time()
    for i, (tag, mk) in enumerate(WINNERS, 1):
        print('--- naive-split run %d/4  %s %s ---' % (i, tag, mk))
        run_one_naive(tag, mk, seed=42)
        _DATACACHE.clear(); gc.collect()
    print('NAIVE-SPLIT COMPARISON COMPLETE in %.2f hours' % ((time.time()-t0)/3600))

    # print the before/after delta immediately for convenience
    print('\n=== BEFORE (naive split, with leakage) vs AFTER (duplicate-aware split) ===')
    for tag in ('D1','D2'):
        for mk in ('BERT','DeBERTa'):
            cfg = WINNERS[(tag,mk)]
            nkey = 'naive_%s_%s_lr%g_bs%d_wd%g_s42' % (tag,mk,cfg['lr'],cfg['bs'],cfg['wd'])
            fkey = 'full_%s_%s_lr%g_bs%d_wd%g_s42' % (tag,mk,cfg['lr'],cfg['bs'],cfg['wd'])
            npath, fpath = RESULTS_DIR/(nkey+'.json'), RESULTS_DIR/(fkey+'.json')
            if npath.exists() and fpath.exists():
                nf1 = json.load(open(npath))['test']['f1']
                ff1 = json.load(open(fpath))['test']['f1']
                print('  %s %-8s naive=%.4f  dedup-aware=%.4f  delta=%+.4f' % (tag, mk, nf1, ff1, nf1-ff1))
