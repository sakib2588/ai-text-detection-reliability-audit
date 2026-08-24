#!/usr/bin/env python3
"""
NLP Final Term Project -- Full-Dataset Pipeline
Group 02, Section B

Runs everything on the COMPLETE DAIGT V2 and HC3 corpora (not the 6,000-row
midterm-matched sample): duplicate-aware splits, the full 8-configuration x 2-model
x 2-dataset hyperparameter sweep, 3-seed robustness on the winning configurations,
classical baselines (Naive Bayes / Logistic Regression / SVM) refit at the same
scale, the ensemble, and the two result tables.

USAGE
-----
1. Put daigt.csv and hc3.jsonl next to this script (or set NLP_DATA_DIR to their folder).
2. pip install -r requirements.txt
3. python run_full_pipeline.py

That's it. The whole run is checkpointed and resumable: if it is interrupted for any
reason (power loss, closed terminal, out of memory), just run the same command again.
Every finished piece of work -- each training run, each classical baseline -- is
skipped on the next pass rather than redone, so re-running after an interruption costs
seconds for whatever is already done and continues exactly where it left off.

Expected time: roughly 14-15 hours on a single consumer GPU (tested on an RTX 3060 Ti,
8 GB VRAM). A GPU is required for the transformer sweep; the classical baselines and
table generation do not need one. You do not need to babysit this -- start it, walk
away, and check back. If it dies partway through (crash, reboot, anything), just run
the same command again and it picks up from the last completed run.

WHY THIS DIFFERS FROM THE MIDTERM SAMPLE
-----------------------------------------
The midterm and the first draft of this project used a 6,000-row balanced sample
(3,000 per class) from each dataset, matched exactly so the classical and transformer
results were comparable. This script instead balances each dataset to its full
available size (DAIGT: 17,497 per class = 34,994 rows; HC3: 26,903 per class = 53,806
rows) per your faculty's instruction to use the complete dataset.

One methodological fix is applied that the small-sample version did not need: HC3 was
audited and found to contain 7.16% duplicate or near-duplicate rows (see
Final/audit/hc3_full_audit.json), which leaks 11.2-11.3% of a naively random test split.
This script therefore splits HC3 (and DAIGT, for consistency, though its duplication
rate is negligible at 0.01%) so that no duplicate-content group crosses the
train/validation/test boundary. This is stricter than a plain random split and is the
scientifically correct choice, not an arbitrary one.
"""
import os, sys, re, gc, json, time, shutil, hashlib, warnings, argparse
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault('HF_HOME', str(Path(__file__).parent / 'hf_cache'))
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')  # safe: dataloader_num_workers stays 0, no fork-after-parallelism conflict
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths -- resolved relative to this script, with dataset auto-discovery.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
WORK_DIR    = HERE / 'work'
RESULTS_DIR = HERE / 'results'
PROBS_DIR   = HERE / 'probs'
MODELS_DIR  = HERE / 'models'
FIG_DIR     = HERE / 'figures'
CKPT_DIR    = Path(os.environ.get('NLP_CKPT_DIR', HERE / 'ckpt_scratch'))
for d in (WORK_DIR, RESULTS_DIR, PROBS_DIR, MODELS_DIR, FIG_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

_candidates = [Path(os.environ['NLP_DATA_DIR'])] if os.environ.get('NLP_DATA_DIR') else []
_candidates += [HERE, HERE / 'data', HERE.parent]
DATA_DIR = None
for _c in _candidates:
    if (_c / 'daigt.csv').exists() and (_c / 'hc3.jsonl').exists():
        DATA_DIR = _c
        break
if DATA_DIR is None:
    print('ERROR: could not find daigt.csv and hc3.jsonl.')
    print('Put both files beside this script, or set NLP_DATA_DIR to the folder that has them.')
    print('Looked in:', ', '.join(str(x) for x in _candidates))
    sys.exit(1)
MAX_LEN, EPOCHS, WARMUP_RATIO, PATIENCE, SPLIT_SEED = 128, 5, 0.1, 2, 42
GRID = [(2e-05,16,0.01), (3e-05,16,0.01), (2e-05,32,0.01), (3e-05,32,0.01),
        (2e-05,16,0.1),  (3e-05,16,0.1),  (2e-05,32,0.1),  (3e-05,32,0.1)]
MODELS  = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}
ROBUSTNESS_SEEDS = [42, 123, 456]
# Capped at 8 regardless of core count: each worker process holds its own copy of
# the NLTK lemmatizer/stopword objects plus its slice of text, and the classical
# preprocessing step runs BEFORE any GPU training starts (never concurrently with
# it), but an uncapped worker count on a high-core-count machine could still push
# memory usage higher than necessary for a CPU task this light. Override with
# NLP_WORKERS=<n> if you know your machine can handle more, or set it to 1 to
# disable parallelism entirely if you see a MemoryError.
NUM_WORKERS = max(1, min(8, int(os.environ.get('NLP_WORKERS', max(1, (os.cpu_count() or 2) - 1)))))

print('datasets found in:', DATA_DIR)
print('CPU cores available: %d, using %d worker process(es) for classical preprocessing '
      '(override with NLP_WORKERS=<n>)' % (os.cpu_count() or 1, NUM_WORKERS))

# ---------------------------------------------------------------------------
# Section 1 -- data loading, balancing, duplicate-aware splitting
# ---------------------------------------------------------------------------

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def content_hash(series):
    return series.map(lambda t: hashlib.md5(normalise(t).lower().encode()).hexdigest())

def balance_full(df, seed=SPLIT_SEED):
    n = int(df['label'].value_counts().min())
    parts = []
    for value in sorted(df['label'].unique()):
        subset = df[df['label'] == value]
        parts.append(subset.sample(n=n, random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_D1():
    raw = pd.read_csv(DATA_DIR / 'daigt.csv')
    df = raw[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    del raw; gc.collect()
    return balance_full(df)

def load_D2():
    raw = pd.read_json(DATA_DIR / 'hc3.jsonl', lines=True)
    human = raw[['human_answers']].explode('human_answers').rename(columns={'human_answers': 'text'})
    human['label'] = 0
    bot = raw[['chatgpt_answers']].explode('chatgpt_answers').rename(columns={'chatgpt_answers': 'text'})
    bot['label'] = 1
    df = pd.concat([human, bot], ignore_index=True).dropna()
    df['text'] = df['text'].astype(str)
    del raw, human, bot; gc.collect()
    return balance_full(df)

LOADERS = {'D1': load_D1, 'D2': load_D2}

def group_split(df, seed=SPLIT_SEED):
    """80/10/10 train/val/test that never splits a duplicate-content group across
    partitions. See the module docstring for why this matters for HC3."""
    from sklearn.model_selection import GroupShuffleSplit
    groups = df['hash'].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_full_idx, te_idx = next(gss1.split(df, df['label'], groups))
    sub = df.iloc[tr_full_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_idx_rel, val_idx_rel = next(gss2.split(sub, sub['label'], sub['hash'].values))
    idx_tr, idx_val = sub.index.values[tr_idx_rel], sub.index.values[val_idx_rel]
    idx_te = df.index.values[te_idx]
    g_tr, g_val, g_te = set(df.loc[idx_tr,'hash']), set(df.loc[idx_val,'hash']), set(df.loc[idx_te,'hash'])
    assert not (g_tr & g_val) and not (g_tr & g_te) and not (g_val & g_te), \
        'group leakage detected across the split -- this must never happen, stopping'
    return idx_tr, idx_val, idx_te

def build_splits_once():
    """Builds and caches the fixed full-scale split for each dataset. Idempotent:
    if the split files already exist on disk (e.g. from a previous run), they are
    reused rather than rebuilt, so the split never silently changes between runs."""
    for tag in ('D1', 'D2'):
        pq, npzp = WORK_DIR / f'data_{tag}.parquet', WORK_DIR / f'split_{tag}.npz'
        if pq.exists() and npzp.exists():
            print(f'[skip] {tag} split already built')
            continue
        print(f'building {tag} split (this reads and balances the full corpus, a few minutes)...')
        df = LOADERS[tag]()
        df['hash'] = content_hash(df['text'])
        idx_tr, idx_val, idx_te = group_split(df)
        print(f'  {tag}: balanced={len(df)}  train={len(idx_tr)} val={len(idx_val)} test={len(idx_te)}')
        print(f'  test label balance: {df.loc[idx_te,"label"].value_counts().sort_index().to_dict()}')
        df[['text','label']].to_parquet(pq, index=True)
        np.savez(npzp, train=idx_tr, val=idx_val, test=idx_te)
        del df; gc.collect()

def load_fixed_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}

# ---------------------------------------------------------------------------
# Section 2 -- transformer fine-tuning harness (checkpointed, resumable)
# ---------------------------------------------------------------------------

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

def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a,4), round(pr,4), round(rc,4), round(f,4)

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

def run_one(tag, model_key, lr, bs, wd, seed=42, save_model=False, verbose=True):
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

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc, pre, rec, f1 = weighted_metrics(labels, preds)
        return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}

    run_dir = CKPT_DIR / key
    resume = find_checkpoint(run_dir)
    parts, splits = get_tokenized(tag, model_key)
    tok = get_tokenizer(model_key)

    # Descending batch-size ladder: every starting batch size gets a real fallback
    # path down to a floor of 4, with gradient accumulation scaled up each step so
    # the EFFECTIVE batch size (per_device_bs * accum) always equals the original
    # requested batch size. This means a small-VRAM GPU degrades gracefully instead
    # of hard-crashing -- previously only bs=32 configs had any fallback at all,
    # bs=16 configs had none.
    attempts = []
    cur = bs
    while cur >= 4:
        attempts.append((cur, bs // cur))
        cur //= 2
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
            per_device_eval_batch_size=min(32, per_device_bs*2), weight_decay=wd, num_train_epochs=EPOCHS,
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
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        trainer.train(resume_from_checkpoint=resume)
        train_secs = time.time() - t0
        break
      except torch.cuda.OutOfMemoryError:
        print('    OOM at per-device batch %d, falling back to a smaller batch with gradient accumulation' % per_device_bs)
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
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else 0.0}

    def _safe_predict(split_name, eval_bs):
        """Retries prediction at progressively smaller batch sizes on CUDA OOM.
        Eval-time OOM is a separate failure mode from training-time OOM (different
        batch size, no gradients to free), and previously was not caught at all."""
        b = eval_bs
        while b >= 1:
            try:
                trainer.args.per_device_eval_batch_size = b
                return trainer.predict(parts[split_name])
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
                if b == 1:
                    raise
                print('    eval OOM at batch %d on %s, retrying at %d' % (b, split_name, b // 2))
                b //= 2

    probs = {}
    for split in ('val', 'test'):
        pred = _safe_predict(split, args.per_device_eval_batch_size)
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        logits = torch.tensor(raw, dtype=torch.float32)
        p = torch.softmax(logits, dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}
        out[split + '_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs['%s_probs' % split] = p; probs['%s_labels' % split] = y

    atomic_write_npz(ppath, **probs)
    atomic_write_json(jpath, out)

    if save_model and seed == 42:
        mdir = MODELS_DIR / ('%s_%s' % (tag, model_key))
        trainer.save_model(str(mdir)); tok.save_pretrained(str(mdir))
        json.dump(out, open(mdir / 'run_info.json', 'w'), indent=2)

    del trainer, model; gc.collect(); torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(run_dir, ignore_errors=True)

    if verbose:
        print('[done] %-46s val_f1=%.4f test_f1=%.4f  %.1f min  peak %.2f GiB' % (
            key, out['val']['f1'], out['test']['f1'], train_secs/60, out['peak_vram_gib']))
    return out

# ---------------------------------------------------------------------------
# Section 3 -- classical baselines at full scale, same split
# ---------------------------------------------------------------------------

def run_classical():
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    for pkg in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
        try:
            nltk.data.find(f'tokenizers/{pkg}' if 'punkt' in pkg else f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        tokens = word_tokenize(text)
        return ' '.join(lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1)

    def _preprocess_chunk(texts):
        lem = WordNetLemmatizer()
        sw = set(stopwords.words('english'))
        out = []
        for t in texts:
            t = re.sub(r'[^a-z\s]', ' ', str(t).lower())
            toks = word_tokenize(t)
            out.append(' '.join(x2 for x2 in (lem.lemmatize(x) for x in toks if x not in sw and len(x) > 1)))
        return out

    def preprocess_parallel(series, workers=NUM_WORKERS):
        texts = series.tolist()
        if workers <= 1 or len(texts) < 2000:
            return pd.Series(_preprocess_chunk(texts), index=series.index)
        chunk_size = (len(texts) + workers - 1) // workers
        chunks = [texts[i:i+chunk_size] for i in range(0, len(texts), chunk_size)]
        with Pool(processes=workers) as pool:
            results = pool.map(_preprocess_chunk, chunks)
        flat = [x for chunk in results for x in chunk]
        return pd.Series(flat, index=series.index)

    specs = (
        ('Naive Bayes', 'BoW', CountVectorizer, lambda: MultinomialNB()),
        ('Logistic Regression', 'BoW', CountVectorizer, lambda: LogisticRegression(max_iter=1000)),
        ('Support Vector Machine', 'TF-IDF', TfidfVectorizer, lambda: LinearSVC()),
    )
    for tag in ('D1', 'D2'):
        df, splits = load_fixed_split(tag)
        for name, rep, Vec, build in specs:
            key = 'full_%s_%s_%s' % (tag, name.replace(' ', ''), rep)
            jpath = RESULTS_DIR / (key + '.json')
            if jpath.exists():
                print('[skip] %s' % key)
                continue
            print('fitting %s %s %s (this takes a few minutes for preprocessing)...' % (tag, name, rep))
            clean = preprocess_parallel(df['text'])
            ytr, yte = df.loc[splits['train'],'label'].values, df.loc[splits['test'],'label'].values
            vec = Vec()
            Xtr = vec.fit_transform(clean.loc[splits['train']]); Xte = vec.transform(clean.loc[splits['test']])
            clf = build(); clf.fit(Xtr, ytr)
            acc, pre, rec, f1 = weighted_metrics(yte, clf.predict(Xte))
            out = {'key': key, 'dataset': tag, 'model': name, 'representation': rep,
                   'n_train': len(splits['train']), 'n_test': len(splits['test']),
                   'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}}
            atomic_write_json(jpath, out)
            print('  %-24s %-7s Acc %.4f F1 %.4f' % (name, rep, acc, f1))
            del clean, Xtr, Xte
        del df; gc.collect()

# ---------------------------------------------------------------------------
# Section 4 -- orchestration
# ---------------------------------------------------------------------------

def run_sweep():
    total = len(GRID) * 2 * 2
    done = 0
    t0 = time.time()
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            for lr, bs, wd in GRID:
                done += 1
                print('--- sweep %d/%d  %s %s lr=%g bs=%d wd=%g ---' % (done, total, tag, mk, lr, bs, wd))
                run_one(tag, mk, lr, bs, wd, seed=42)
                e = time.time() - t0
                print('    elapsed %.1f min, projected remaining %.1f min\n' % (e/60, e/done*(total-done)/60))
        _DATACACHE.clear(); gc.collect()
    print('SWEEP COMPLETE in %.2f hours' % ((time.time()-t0)/3600))

def run_seed_robustness():
    import json as _json
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            cand = [_json.load(open(p)) for p in RESULTS_DIR.glob(f'full_{tag}_{mk}_*_s42.json')]
            if not cand:
                print('WARNING: no seed-42 sweep result found for %s %s, run the sweep first' % (tag, mk))
                continue
            best = max(cand, key=lambda r: r['val']['f1'])
            print('winner for %s %s: lr=%g bs=%d wd=%g (val_f1=%.4f)' % (
                tag, mk, best['lr'], best['batch_size'], best['weight_decay'], best['val']['f1']))
            for seed in ROBUSTNESS_SEEDS:
                run_one(tag, mk, best['lr'], best['batch_size'], best['weight_decay'],
                       seed=seed, save_model=True)

def build_tables():
    import json as _json
    all_results = [_json.load(open(p)) for p in RESULTS_DIR.glob('*.json')]
    sweep = [r for r in all_results if r.get('seed') == 42 and 'model' in r
             and r['model'] in MODELS and r.get('scale') == 'full_balanced']
    classical = {(r['dataset'], r['model'], r['representation']): r
                 for r in all_results if 'representation' in r}

    def fmt(v): return '%.4f' % v

    # ensemble: best BERT + best DeBERTa per dataset, weighted soft vote tuned on validation
    ensemble = {}
    for tag in ('D1', 'D2'):
        picks = {}
        for mk in ('BERT', 'DeBERTa'):
            cand = [r for r in sweep if r['dataset'] == tag and r['model'] == mk]
            if not cand:
                continue
            picks[mk] = max(cand, key=lambda r: r['val']['f1'])
        if len(picks) < 2:
            continue
        zb = np.load(PROBS_DIR / (picks['BERT']['key'] + '.npz'))
        zd = np.load(PROBS_DIR / (picks['DeBERTa']['key'] + '.npz'))
        ws = np.arange(0, 1.0001, 0.05)
        vf1 = [weighted_metrics(zb['val_labels'], (w*zb['val_probs']+(1-w)*zd['val_probs']).argmax(1))[3] for w in ws]
        w = float(ws[int(np.argmax(vf1))])
        mix = w*zb['test_probs'] + (1-w)*zd['test_probs']
        acc, pre, rec, f1 = weighted_metrics(zb['test_labels'], mix.argmax(1))
        ensemble[tag] = (acc, pre, rec, f1, w)
        print('ensemble %s: w_bert=%.2f -> Acc %.4f F1 %.4f' % (tag, w, acc, f1))

    # Table 1
    rows = []
    for mk, label in (('BERT','BERT'), ('DeBERTa','DeBERTa')):
        for lr, bs, wd in GRID:
            row = {'Model': label, 'Learning Rate': '%.5f' % lr, 'Batch Size': bs, 'Weight Decay': wd}
            for tag in ('D1', 'D2'):
                m = next((r for r in sweep if r['dataset']==tag and r['model']==mk and
                         r['lr']==lr and r['batch_size']==bs and r['weight_decay']==wd), None)
                for col, k in (('Acc','accuracy'),('Prec','precision'),('Rec','recall'),('F1','f1')):
                    row['%s %s' % (tag, col)] = fmt(m['test'][k]) if m else ''
            rows.append(row)
    erow = {'Model': 'ENSEMBLE', 'Learning Rate': '', 'Batch Size': '', 'Weight Decay': ''}
    for tag in ('D1', 'D2'):
        if tag in ensemble:
            acc, pre, rec, f1, w = ensemble[tag]
            for col, v in zip(('Acc','Prec','Rec','F1'), (acc,pre,rec,f1)):
                erow['%s %s' % (tag, col)] = fmt(v)
    rows.append(erow)
    t1 = pd.DataFrame(rows)
    t1.to_csv(HERE / 'table1_experiments_full.csv', index=False)

    # Table 2
    BEST_REP = {'Naive Bayes': 'BoW', 'Logistic Regression': 'BoW', 'Support Vector Machine': 'TF-IDF'}
    SPEC_LABEL = {'Naive Bayes': 'Naïve Bayes', 'Logistic Regression': 'Logistic Regression',
                  'Support Vector Machine': 'Support Vector Machine'}
    rows2 = []
    for name, rep in BEST_REP.items():
        r = classical.get(('D1', name, rep)), classical.get(('D2', name, rep))
        row = {'Model': SPEC_LABEL[name]}
        for tag, cr in zip(('D1','D2'), r):
            for col, k in (('Acc','accuracy'),('Prec','precision'),('Rec','recall'),('F1','f1')):
                row['%s %s' % (tag, col)] = fmt(cr['test'][k]) if cr else ''
        rows2.append(row)
    for mk, label in (('BERT','BERT'), ('DeBERTa','DeBERTa')):
        row = {'Model': label}
        for tag in ('D1', 'D2'):
            cand = [r for r in sweep if r['dataset']==tag and r['model']==mk]
            best = max(cand, key=lambda r: r['val']['f1']) if cand else None
            for col, k in (('Acc','accuracy'),('Prec','precision'),('Rec','recall'),('F1','f1')):
                row['%s %s' % (tag, col)] = fmt(best['test'][k]) if best else ''
        rows2.append(row)
    erow2 = {'Model': 'ENSEMBLE'}
    for tag in ('D1', 'D2'):
        if tag in ensemble:
            acc, pre, rec, f1, w = ensemble[tag]
            for col, v in zip(('Acc','Prec','Rec','F1'), (acc,pre,rec,f1)):
                erow2['%s %s' % (tag, col)] = fmt(v)
    rows2.append(erow2)
    t2 = pd.DataFrame(rows2)
    t2.to_csv(HERE / 'table2_combined_full.csv', index=False)

    print('\nTable 1 -> %s' % (HERE / 'table1_experiments_full.csv'))
    print(t1.to_string(index=False))
    print('\nTable 2 -> %s' % (HERE / 'table2_combined_full.csv'))
    print(t2.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--skip-sweep', action='store_true', help='skip the 32-run hyperparameter sweep')
    ap.add_argument('--skip-robustness', action='store_true', help='skip the 3-seed robustness runs')
    ap.add_argument('--skip-classical', action='store_true', help='skip the classical baselines')
    args = ap.parse_args()

    import torch
    print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('device:', torch.cuda.get_device_name(0), '| bf16:', torch.cuda.is_bf16_supported())
    else:
        print('WARNING: no CUDA device found. The transformer sweep requires a GPU; '
              'classical baselines and table generation will still work if the sweep '
              'was already run elsewhere and results/ + probs/ are present.')

    print('\n=== Section 1: building duplicate-aware full-scale splits ===')
    build_splits_once()

    if not args.skip_classical:
        print('\n=== Section 3: classical baselines at full scale ===')
        run_classical()

    if not args.skip_sweep:
        print('\n=== Section 2: full hyperparameter sweep (32 runs) ===')
        run_sweep()

    if not args.skip_robustness:
        print('\n=== Section 2b: 3-seed robustness on the winning configurations ===')
        run_seed_robustness()

    print('\n=== Section 4: building result tables ===')
    build_tables()
    print('\nDone.')

if __name__ == '__main__':
    main()
