"""Cross-dataset generalization test: each model trained on one dataset is evaluated
on the OTHER dataset's held-out test set. This is the piece that turns "we found
contamination" into "here's why it matters" -- everything else in this project is
in-domain (a model tested on the same corpus it trained on). This measures whether
a model actually learned "AI vs human writing" or just learned quirks specific to
its training corpus's domain (DAIGT essays vs HC3 QA answers).

Inference only -- no training, no gradients, no optimizer state -- so this runs on
CPU without conflict against a concurrent GPU training job, at the cost of being
slower than a GPU run would be.
"""
import os, re, gc, json, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

FINAL_DIR   = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/paper_scale')
WORK_DIR    = FINAL_DIR / 'work'
MODELS_DIR  = FINAL_DIR / 'models'
RESULTS_DIR = FINAL_DIR / 'results'
PROBS_DIR   = FINAL_DIR / 'probs'
MAX_LEN = 128
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def load_test_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df.loc[sp['test']]

def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a,4), round(pr,4), round(rc,4), round(f,4)

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

def run_cross(train_tag, model_key, device='cpu'):
    """Loads the model trained on `train_tag`, evaluates it on the OTHER dataset's
    test set. train_tag='D1' means the model was trained on DAIGT; it is evaluated
    on HC3's test set (test_tag='D2'), and vice versa."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import confusion_matrix

    test_tag = 'D2' if train_tag == 'D1' else 'D1'
    key = 'cross_train%s_test%s_%s' % (train_tag, test_tag, model_key)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print('[skip] %-40s cross_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    mdir = MODELS_DIR / ('%s_%s' % (train_tag, model_key))
    if not (mdir / 'model.safetensors').exists():
        print('[missing] no checkpoint at %s, skipping' % mdir)
        return None

    print('loading %s (trained on %s), evaluating on %s test set...' % (
        model_key, DATASET_NAMES[train_tag], DATASET_NAMES[test_tag]))
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
            chunk = texts[i:i+batch]
            enc = tok(chunk, truncation=True, max_length=MAX_LEN, padding=True, return_tensors='pt').to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            if i % (batch*20) == 0:
                print('  %d / %d  (%.1f min elapsed)' % (i, len(texts), (time.time()-t0)/60))
    p = np.concatenate(all_probs, axis=0)
    secs = time.time() - t0

    acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
    out = {'key': key, 'train_dataset': train_tag, 'test_dataset': test_tag,
           'train_dataset_name': DATASET_NAMES[train_tag], 'test_dataset_name': DATASET_NAMES[test_tag],
           'model': model_key, 'n_test': len(y), 'eval_seconds': round(secs, 1),
           'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1},
           'test_confusion': confusion_matrix(y, p.argmax(1)).tolist()}
    atomic_write_npz(ppath, test_probs=p, test_labels=y)
    atomic_write_json(jpath, out)
    print('[done] %-40s cross_f1=%.4f  (%.1f min, %d examples)' % (key, f1, secs/60, len(y)))

    del model, tok; gc.collect()
    return out

def build_table():
    """Compares in-domain (already on disk from the main sweep) against cross-domain
    (just computed) for all four models, so the generalization gap is visible directly."""
    rows = []
    for train_tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            # in-domain: best config's seed-42 result from the main sweep
            in_domain_files = sorted(RESULTS_DIR.glob(f'full_{train_tag}_{mk}_*_s42.json'))
            in_domain_f1 = None
            best_val = -1
            for f in in_domain_files:
                r = json.load(open(f))
                if r.get('val', {}).get('f1', -1) > best_val:
                    best_val = r['val']['f1']; in_domain_f1 = r['test']['f1']

            cross_key = 'cross_train%s_test%s_%s' % (train_tag, 'D2' if train_tag=='D1' else 'D1', mk)
            cross_path = RESULTS_DIR / (cross_key + '.json')
            cross_f1 = json.load(open(cross_path))['test']['f1'] if cross_path.exists() else None

            rows.append({
                'trained_on': DATASET_NAMES[train_tag], 'model': mk,
                'in_domain_f1': in_domain_f1,
                'cross_domain_f1': cross_f1,
                'generalization_gap': round(in_domain_f1 - cross_f1, 4) if (in_domain_f1 and cross_f1) else None,
            })
    t = pd.DataFrame(rows)
    out_csv = FINAL_DIR.parent / 'table_cross_dataset_generalization.csv'
    t.to_csv(out_csv, index=False)
    print('\n' + t.to_string(index=False))
    print('\nwritten to:', out_csv)

if __name__ == '__main__':
    import torch
    # Defaults to CPU. This is inference-only (no gradients), so CPU is viable and
    # -- more importantly -- it means this can run safely alongside the concurrent
    # GPU training job without contending for VRAM. Set NLP_CROSS_DEVICE=cuda to
    # use the GPU once the training sweep has finished.
    device = os.environ.get('NLP_CROSS_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: cuda requested but not available, falling back to cpu')
        device = 'cpu'
    print('running cross-dataset eval on device:', device,
          '(set NLP_CROSS_DEVICE=cuda to use the GPU once it is free)')
    for train_tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            run_cross(train_tag, mk, device=device)
    build_table()
