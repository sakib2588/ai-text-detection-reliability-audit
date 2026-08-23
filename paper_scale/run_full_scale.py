import os
os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc, json, re, shutil, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed)
from datasets import Dataset
warnings.filterwarnings('ignore')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')
FINAL_DIR   = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
PS_DIR      = FINAL_DIR / 'paper_scale'
WORK_DIR    = PS_DIR / 'work'
RESULTS_DIR = PS_DIR / 'results'
PROBS_DIR   = PS_DIR / 'probs'
MODELS_DIR  = PS_DIR / 'models'
CKPT_DIR    = Path('/media/filwel/MLProject1/nlp_paper_ckpt')
for d in (WORK_DIR, RESULTS_DIR, PROBS_DIR, MODELS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_LEN, EPOCHS, WARMUP_RATIO, PATIENCE = 128, 5, 0.1, 2
MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}

# the 4 winning configurations, carried over from the 6000-row coursework sweep
WINNERS = {
    ('D1', 'BERT'):    dict(lr=3e-05, bs=32, wd=0.1),
    ('D1', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.01),
    ('D2', 'BERT'):    dict(lr=2e-05, bs=16, wd=0.1),
    ('D2', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.1),
}

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

# The split is FIXED (built once by build_full_splits.py with a duplicate-group-aware
# partition, seed=42) and reused across every training seed. Only model initialisation
# and data shuffling order during training vary with `seed` -- never which rows are in
# which partition. Re-splitting per seed would silently change the evaluation set between
# runs, making the 3-seed comparison invalid.
def load_fixed_split(tag):
    df = pd.read_parquet(WORK_DIR / ('data_%s.parquet' % tag))
    sp = np.load(WORK_DIR / ('split_%s.npz' % tag))
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}

LOADERS = {'D1': lambda: load_fixed_split('D1')[0], 'D2': lambda: load_fixed_split('D2')[0]}

def build_split(tag):
    return load_fixed_split(tag)

def weighted_metrics(y, p):
    acc = accuracy_score(y, p)
    pre, rec, f1, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return acc, pre, rec, f1

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

def get_tokenizer(model_key):
    if model_key not in _TOKCACHE:
        _TOKCACHE[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return _TOKCACHE[model_key]

def get_tokenized(tag, model_key):
    key = (tag, model_key)
    if key in _DATACACHE:
        return _DATACACHE[key]
    df, splits = build_split(tag)
    tok = get_tokenizer(model_key)
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

def run_one(tag, model_key, seed, save_model=False, force=False, verbose=True):
    cfg = WINNERS[(tag, model_key)]
    lr, bs, wd = cfg['lr'], cfg['bs'], cfg['wd']
    key = 'full_%s_%s_lr%g_bs%d_wd%g_s%d' % (tag, model_key, lr, bs, wd, seed)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    # force=True retrains even when metrics already exist on disk -- needed when the
    # metrics were saved but the model weights were not (seeds 123/456 originally ran
    # with save_model=False), and we now need the weights for a downstream eval.
    if not force and jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        if verbose:
            print('[skip] %-46s test_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    run_dir = CKPT_DIR / key
    resume = find_checkpoint(run_dir)
    parts, splits = get_tokenized(tag, model_key)
    tok = get_tokenizer(model_key)

    attempts = [(bs, 1)]
    if bs > 16:
        attempts.append((bs // 2, 2))
    for attempt_i, (per_device_bs, accum) in enumerate(attempts):
      try:
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_key], num_labels=2)
        model.config.id2label = {0: 'human', 1: 'ai'}
        model.config.label2id = {'human': 0, 'ai': 1}
        args = TrainingArguments(
            output_dir=str(run_dir), learning_rate=lr,
            per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=accum,
            per_device_eval_batch_size=64, weight_decay=wd, num_train_epochs=EPOCHS,
            warmup_ratio=WARMUP_RATIO, lr_scheduler_type='linear', optim='adamw_torch', bf16=True,
            eval_strategy='epoch', save_strategy='epoch', save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model='eval_f1', greater_is_better=True,
            logging_steps=200, seed=seed, data_seed=seed, dataloader_num_workers=0,
            report_to='none', disable_tqdm=not verbose,
        )
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
        print('    OOM at per-device batch %d, falling back' % per_device_bs)
        try: del trainer
        except NameError: pass
        try: del model
        except NameError: pass
        gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(run_dir, ignore_errors=True)
        resume = None
        if attempt_i == len(attempts) - 1:
            raise

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag], 'model': model_key,
           'checkpoint': MODELS[model_key], 'lr': lr, 'batch_size': bs, 'weight_decay': wd,
           'seed': seed, 'max_len': MAX_LEN, 'scale': 'full_balanced',
           'n_total': len(splits['train'])+len(splits['val'])+len(splits['test']), 'n_train': len(splits['train']), 'n_val': len(splits['val']),
           'n_test': len(splits['test']), 'train_seconds': round(train_secs, 1),
           'epochs_run': int(trainer.state.epoch or 0),
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2)}
    probs = {}
    for split in ('val', 'test'):
        pred = trainer.predict(parts[split])
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        logits = torch.tensor(raw, dtype=torch.float32)
        p = torch.softmax(logits, dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': round(acc,4), 'precision': round(pre,4), 'recall': round(rec,4), 'f1': round(f1,4)}
        out[split + '_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs['%s_probs' % split] = p; probs['%s_labels' % split] = y

    atomic_write_npz(ppath, **probs)
    atomic_write_json(jpath, out)

    if save_model:
        # seed 42's dir name has no suffix (matches what cross_dataset_eval.py already
        # expects); other seeds get a _seed{N} suffix so they never collide with it.
        mdir = MODELS_DIR / (('%s_%s' % (tag, model_key)) if seed == 42
                             else ('%s_%s_seed%d' % (tag, model_key, seed)))
        trainer.save_model(str(mdir)); tok.save_pretrained(str(mdir))
        json.dump(out, open(mdir / 'run_info.json', 'w'), indent=2)
        if verbose:
            print('    model saved -> %s' % mdir)

    del trainer, model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(run_dir, ignore_errors=True)

    if verbose:
        print('[done] %-46s val_f1=%.4f test_f1=%.4f  %.1f min  peak %.2f GiB  n_train=%d' % (
            key, out['val']['f1'], out['test']['f1'], train_secs/60, out['peak_vram_gib'], out['n_train']))
    return out

if __name__ == '__main__':
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| bf16', torch.cuda.is_bf16_supported())
    SEEDS = [42, 123, 456]
    total = len(WINNERS) * len(SEEDS)
    done = 0
    t_start = time.time()
    for (tag, mk) in WINNERS:
        for seed in SEEDS:
            done += 1
            print('--- run %d/%d  %s %s seed=%d ---' % (done, total, tag, mk, seed))
            run_one(tag, mk, seed, save_model=(seed == 42))
            e = time.time() - t_start
            print('    elapsed %.1f min, projected remaining %.1f min\n' % (e/60, e/done*(total-done)/60))
        _DATACACHE.clear(); gc.collect()
    print('FULL-SCALE SWEEP COMPLETE in %.2f hours' % ((time.time()-t_start)/3600))
