#!/usr/bin/env python3
"""
NLP Final Term Project -- Your Half of the BERT/DeBERTa Sweep
Group 02, Section B

The 32-configuration hyperparameter sweep (8 configs x BERT/DeBERTa x DAIGT/HC3) is
split between two machines to finish faster. Sakib's machine already ran the 4
"winning" configurations (found from an earlier smaller-scale sweep) plus his own
half of the remaining 28. This package contains YOUR assigned half: 14 configurations,
listed explicitly below.

You do NOT need to run every config in the assignment sheet -- only the 14 in
MY_HALF below. The other 18 (4 winners + 14 on Sakib's side) are being run
elsewhere and will be merged with your results afterward.

USAGE
-----
1. Install torch for your GPU FIRST, separately from requirements.txt (see below).
2. pip install -r requirements.txt
3. python run_my_half.py

Expected time: with an RTX 4060 (8-12 GB VRAM depending on model), roughly 6-8 hours
for all 14 configs. Leave it running -- it is checkpointed and safe to interrupt and
resume, exactly like the classical-baseline script you already ran.

INSTALLING TORCH -- READ THIS BEFORE requirements.txt
--------------------------------------------------------
Do NOT use a frozen pip-freeze-style requirements.txt for torch. The one you may
have seen earlier failed with an "nvidia-cufile-cu12" error because that package is
Linux-only and was captured from a Linux machine's pip freeze -- it has no Windows
wheel and never will. Instead, install torch using the official index, which
resolves the correct Windows-specific dependencies automatically:

    pip install torch --index-url https://download.pytorch.org/whl/cu128

Then install everything else:

    pip install -r requirements.txt

WHAT YOU NEED, WHAT YOU DON'T
--------------------------------
You need: an NVIDIA GPU with a reasonably recent driver (CUDA 12.8-compatible;
if you're not sure, the torch install command above will simply tell you if your
driver is too old). You do NOT need the raw daigt.csv / hc3.jsonl files -- this
package includes the exact balanced, duplicate-aware-split data already prepared
(data_D1.parquet, data_D2.parquet, split_D1.npz, split_D2.npz), so your results
land on the identical train/validation/test split as everyone else's. If you
rebuilt the split yourself from the raw datasets, even a small library version
difference could shuffle it slightly differently, and then your results would not
be combinable with everyone else's into the same table.

WHAT TO SEND BACK
--------------------
The whole `results/` folder and the whole `probs/` folder. Each file is named
uniquely by its exact configuration, so there is no risk of overwriting anyone
else's files when everything gets merged back together.
"""
import os, sys, re, gc, json, time, shutil
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).parent / 'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK_DIR    = HERE
RESULTS_DIR = HERE / 'results'
PROBS_DIR   = HERE / 'probs'
CKPT_DIR    = Path(os.environ.get('NLP_CKPT_DIR', HERE / 'ckpt_scratch'))
for d in (RESULTS_DIR, PROBS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_LEN, EPOCHS, WARMUP_RATIO, PATIENCE = 128, 5, 0.1, 2
MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}

# ============================================================================
# YOUR ASSIGNED 14 CONFIGURATIONS -- this is the entire point of this package.
# ============================================================================
MY_HALF = [
    ('D1', 'BERT',    3e-05, 16, 0.01),
    ('D1', 'BERT',    3e-05, 32, 0.01),
    ('D1', 'BERT',    3e-05, 16, 0.1),
    ('D1', 'DeBERTa', 2e-05, 16, 0.01),
    ('D1', 'DeBERTa', 3e-05, 32, 0.01),
    ('D1', 'DeBERTa', 3e-05, 16, 0.1),
    ('D1', 'DeBERTa', 3e-05, 32, 0.1),
    ('D2', 'BERT',    3e-05, 16, 0.01),
    ('D2', 'BERT',    3e-05, 32, 0.01),
    ('D2', 'BERT',    2e-05, 32, 0.1),
    ('D2', 'DeBERTa', 2e-05, 16, 0.01),
    ('D2', 'DeBERTa', 2e-05, 32, 0.01),
    ('D2', 'DeBERTa', 2e-05, 16, 0.1),
    ('D2', 'DeBERTa', 3e-05, 32, 0.1),
]

def check_files():
    missing = [f for f in ('data_D1.parquet', 'data_D2.parquet', 'split_D1.npz', 'split_D2.npz')
               if not (HERE / f).exists()]
    if missing:
        print('ERROR: missing required file(s):', ', '.join(missing))
        print('Make sure you extracted the whole package, not just this script.')
        raise SystemExit(1)

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def load_fixed_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}

def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a, 4), round(pr, 4), round(rc, 4), round(f, 4)

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
    if not run_dir.exists():
        return None
    cks = [p for p in run_dir.glob('checkpoint-*') if (p / 'trainer_state.json').exists()]
    return str(max(cks, key=lambda p: int(p.name.split('-')[1]))) if cks else None

_TOKCACHE, _DATACACHE = {}, {}

def get_tokenizer(model_key):
    from transformers import AutoTokenizer
    if model_key not in _TOKCACHE:
        _TOKCACHE[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return _TOKCACHE[model_key]

def get_tokenized(tag, model_key):
    key = (tag, model_key)
    if key in _DATACACHE:
        return _DATACACHE[key]
    from datasets import Dataset
    df, splits = load_fixed_split(tag)
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

def run_one(tag, model_key, lr, bs, wd, seed=42, verbose=True):
    import torch
    from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                              Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed)
    from sklearn.metrics import confusion_matrix

    key = 'full_%s_%s_lr%g_bs%d_wd%g_s%d' % (tag, model_key, lr, bs, wd, seed)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        if verbose:
            print('[skip] %-46s test_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    run_dir = CKPT_DIR / key
    resume = find_checkpoint(run_dir)
    parts, splits = get_tokenized(tag, model_key)
    tok = get_tokenizer(model_key)

    # Descending batch-size ladder: if your GPU doesn't have enough VRAM for the
    # requested batch size, this automatically retries at half the batch with
    # double the gradient accumulation (so the effective batch size stays the
    # same), down to a floor of 4. You should not need to do anything manually
    # even if you see an "OOM at batch..." message -- it will keep going.
    attempts = []
    cur = bs
    while cur >= 4:
        attempts.append((cur, bs // cur)); cur //= 2
    if not attempts:
        attempts = [(bs, 1)]

    for attempt_i, (per_device_bs, accum) in enumerate(attempts):
      try:
        set_seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_key], num_labels=2)
        model.config.id2label = {0: 'human', 1: 'ai'}
        model.config.label2id = {'human': 0, 'ai': 1}
        args = TrainingArguments(
            output_dir=str(run_dir), learning_rate=lr,
            per_device_train_batch_size=per_device_bs, gradient_accumulation_steps=accum,
            per_device_eval_batch_size=min(32, per_device_bs * 2), weight_decay=wd,
            num_train_epochs=EPOCHS, warmup_ratio=WARMUP_RATIO, lr_scheduler_type='linear',
            optim='adamw_torch', bf16=True, eval_strategy='epoch', save_strategy='epoch',
            save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='eval_f1',
            greater_is_better=True, logging_steps=200, seed=seed, data_seed=seed,
            dataloader_num_workers=0, report_to='none', disable_tqdm=not verbose,
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
        print('    OOM at batch %d, falling back to a smaller batch' % per_device_bs)
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
           'n_train': len(splits['train']), 'n_val': len(splits['val']), 'n_test': len(splits['test']),
           'train_seconds': round(train_secs, 1), 'epochs_run': int(trainer.state.epoch or 0),
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2)}

    def safe_predict(split_name, eval_bs):
        b = eval_bs
        while b >= 1:
            try:
                trainer.args.per_device_eval_batch_size = b
                return trainer.predict(parts[split_name])
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if b == 1:
                    raise
                b //= 2

    probs = {}
    for split in ('val', 'test'):
        pred = safe_predict(split, args.per_device_eval_batch_size)
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        p = torch.softmax(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}
        out[split + '_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs['%s_probs' % split] = p; probs['%s_labels' % split] = y

    atomic_write_npz(ppath, **probs)
    atomic_write_json(jpath, out)
    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(run_dir, ignore_errors=True)

    if verbose:
        print('[done] %-46s val_f1=%.4f test_f1=%.4f  %.1f min  peak %.2f GiB' % (
            key, out['val']['f1'], out['test']['f1'], train_secs / 60, out['peak_vram_gib']))
    return out


def main():
    check_files()
    import torch
    print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
    if not torch.cuda.is_available():
        print('ERROR: no CUDA GPU detected. This script needs a GPU to run the sweep.')
        print('Check that you installed torch with:')
        print('  pip install torch --index-url https://download.pytorch.org/whl/cu128')
        raise SystemExit(1)
    print('device:', torch.cuda.get_device_name(0), '| bf16 supported:', torch.cuda.is_bf16_supported())

    total = len(MY_HALF)
    print('\nrunning your assigned %d configurations' % total)
    done = 0
    t0 = time.time()
    for tag, mk, lr, bs, wd in MY_HALF:
        done += 1
        print('--- run %d/%d  %s %s lr=%g bs=%d wd=%g ---' % (done, total, tag, mk, lr, bs, wd))
        run_one(tag, mk, lr, bs, wd, seed=42)
        e = time.time() - t0
        print('    elapsed %.1f min, projected remaining %.1f min\n' % (e / 60, e / done * (total - done) / 60))
    print('YOUR HALF COMPLETE in %.2f hours' % ((time.time() - t0) / 3600))
    print('\nSend back the whole results/ folder and the whole probs/ folder.')


if __name__ == '__main__':
    main()
