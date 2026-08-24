"""3-seed version of cross_dataset_eval.py: evaluates the cross-dataset generalization
gap at all 3 seeds (42, 123, 456) instead of just seed 42, so the gap itself can be
reported as mean +/- range like every other headline result in this project, instead
of a single point estimate.

Depends on retrain_seeds_for_cross.py having been run first (that produces the seed
123/456 checkpoints at models/{tag}_{model}_seed{seed}/; seed 42's checkpoint already
existed at models/{tag}_{model}/ from the original sweep).

Inference only, CPU by default -- same reasoning as cross_dataset_eval.py: this must
never contend with a concurrent GPU training job for VRAM. Set NLP_CROSS_DEVICE=cuda
once no training job is running if you want it faster.
"""
import os, re, gc, json, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

FINAL_DIR   = Path(__file__).resolve().parent
TABLES_DIR  = FINAL_DIR.parents[1] / 'tables'
WORK_DIR    = FINAL_DIR / 'work'
MODELS_DIR  = FINAL_DIR / 'models'
RESULTS_DIR = FINAL_DIR / 'results'
PROBS_DIR   = FINAL_DIR / 'probs'
MAX_LEN = 128
SEEDS = [42, 123, 456]
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}

# the 4 winning configs -- needed here to look up each seed's matching in-domain
# result file (which already exists on disk from the original 3-seed sweep).
WINNERS = {
    ('D1', 'BERT'):    dict(lr=3e-05, bs=32, wd=0.1),
    ('D1', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.01),
    ('D2', 'BERT'):    dict(lr=2e-05, bs=16, wd=0.1),
    ('D2', 'DeBERTa'): dict(lr=3e-05, bs=16, wd=0.1),
}

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def model_dir(tag, model_key, seed):
    return MODELS_DIR / (('%s_%s' % (tag, model_key)) if seed == 42
                         else ('%s_%s_seed%d' % (tag, model_key, seed)))

def cross_key(train_tag, test_tag, model_key, seed):
    base = 'cross_train%s_test%s_%s' % (train_tag, test_tag, model_key)
    return base if seed == 42 else (base + '_s%d' % seed)

def load_test_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df.loc[sp['test']]

def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a, 4), round(pr, 4), round(rc, 4), round(f, 4)

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

def run_cross(train_tag, model_key, seed, device='cpu'):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import confusion_matrix

    test_tag = 'D2' if train_tag == 'D1' else 'D1'
    key = cross_key(train_tag, test_tag, model_key, seed)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print('[skip] %-46s cross_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    mdir = model_dir(train_tag, model_key, seed)
    if not (mdir / 'model.safetensors').exists():
        print('[missing] no checkpoint at %s -- run retrain_seeds_for_cross.py first, skipping' % mdir)
        return None

    print('loading %s seed=%d (trained on %s), evaluating on %s test set...' % (
        model_key, seed, DATASET_NAMES[train_tag], DATASET_NAMES[test_tag]))
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).to(device)
    model.eval()

    test_df = load_test_split(test_tag)
    texts = [normalise(t) for t in test_df['text']]
    y = test_df['label'].values

    t0 = time.time()
    all_probs = []
    batch = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if i % (batch * 20) == 0:
                print('  %d / %d  (%.1f min elapsed)' % (i, len(texts), (time.time() - t0) / 60))
    p = np.concatenate(all_probs, axis=0)
    secs = time.time() - t0

    acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
    out = {'key': key, 'train_dataset': train_tag, 'test_dataset': test_tag,
           'train_dataset_name': DATASET_NAMES[train_tag], 'test_dataset_name': DATASET_NAMES[test_tag],
           'model': model_key, 'seed': seed, 'n_test': len(y), 'eval_seconds': round(secs, 1),
           'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1},
           'test_confusion': confusion_matrix(y, p.argmax(1)).tolist()}
    atomic_write_npz(ppath, test_probs=p, test_labels=y)
    atomic_write_json(jpath, out)
    print('[done] %-46s cross_f1=%.4f  (%.1f min, %d examples)' % (key, f1, secs / 60, len(y)))

    del model, tok; gc.collect()
    return out

def in_domain_f1_for(tag, mk, seed):
    cfg = WINNERS[(tag, mk)]
    key = 'full_%s_%s_lr%g_bs%d_wd%g_s%d' % (tag, mk, cfg['lr'], cfg['bs'], cfg['wd'], seed)
    p = RESULTS_DIR / (key + '.json')
    if not p.exists():
        return None
    return json.load(open(p))['test']['f1']

def build_multiseed_table():
    rows = []
    for train_tag in ('D1', 'D2'):
        test_tag = 'D2' if train_tag == 'D1' else 'D1'
        for mk in ('BERT', 'DeBERTa'):
            gaps, in_doms, crosses = [], [], []
            for seed in SEEDS:
                ind = in_domain_f1_for(train_tag, mk, seed)
                cp = RESULTS_DIR / (cross_key(train_tag, test_tag, mk, seed) + '.json')
                cross = json.load(open(cp))['test']['f1'] if cp.exists() else None
                if ind is not None and cross is not None:
                    in_doms.append(ind); crosses.append(cross); gaps.append(round(ind - cross, 4))
            if not gaps:
                continue
            rows.append({
                'trained_on': DATASET_NAMES[train_tag], 'model': mk, 'n_seeds': len(gaps),
                'in_domain_f1_mean': round(float(np.mean(in_doms)), 4),
                'cross_domain_f1_mean': round(float(np.mean(crosses)), 4),
                'cross_domain_f1_min': round(float(np.min(crosses)), 4),
                'cross_domain_f1_max': round(float(np.max(crosses)), 4),
                'generalization_gap_mean': round(float(np.mean(gaps)), 4),
                'generalization_gap_min': round(float(np.min(gaps)), 4),
                'generalization_gap_max': round(float(np.max(gaps)), 4),
            })
    t = pd.DataFrame(rows)
    out_csv = TABLES_DIR / 'table_cross_dataset_generalization_3seed.csv'
    t.to_csv(out_csv, index=False)
    print('\n' + t.to_string(index=False))
    print('\nwritten to:', out_csv)

if __name__ == '__main__':
    import torch
    device = os.environ.get('NLP_CROSS_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: cuda requested but not available, falling back to cpu')
        device = 'cpu'
    print('running 3-seed cross-dataset eval on device:', device)
    for train_tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            for seed in SEEDS:
                run_cross(train_tag, mk, seed, device=device)
    build_multiseed_table()
