"""Artifact-cleaning ablation, ZERO-SHOT interim signal (Gap 4, stage 3a):
score the EXISTING raw-trained checkpoints on both raw and artifact-cleaned
test text, no retraining. Measures how much of the existing model's accuracy
depends on the artifact surviving at test time -- a real but weaker claim
than the full retrain-based ablation (run_artifact_cleaning_full.py, queued
until the GPU is free), since a model that memorized the artifact during
training may still score correctly on cleaned text if artifact-independent
signal was also learned. Both numbers are reported; this one is the
secondary/discussion figure, not the headline.

Cleaning applied per dataset (see text_perturbations.py for the exact
regexes/ranges, grounded in the 2026-08-22 data recon):
  - D2 (HC3): remove space-before-punctuation (human 88.7% vs ChatGPT 0.28%).
  - D1 (DAIGT V2): strip emoji/pictograph signal (human 0% vs AI 3.2%).
Both datasets also get normalize_nbsp() applied to BOTH raw and cleaned
conditions (encoding-noise repair, not part of the artifact being measured).
Length-matching is NOT applied here (it only matters at the class level for
training-time exploitation; zero-shot eval scores an already-fixed model on
per-row perturbed text, so per-row truncation would just be an additional
confound at this stage -- reserved for the full ablation in stage 3b).

Inference only, CPU by default, same convention as cross_dataset_eval.py.
"""
import os, gc, json, time
from pathlib import Path
import numpy as np, pandas as pd

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from text_perturbations import normalize_nbsp, clean_hc3_whitespace, clean_daigt_unicode

FINAL_DIR   = Path(__file__).resolve().parent
TABLES_DIR  = FINAL_DIR.parents[1] / 'tables'
WORK_DIR    = FINAL_DIR / 'work'
MODELS_DIR  = FINAL_DIR / 'models'
RESULTS_DIR = FINAL_DIR / 'results'
PROBS_DIR   = FINAL_DIR / 'probs'
MAX_LEN = 128
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}


def load_test_split(tag):
    df = pd.read_parquet(WORK_DIR / f'data_{tag}.parquet')
    sp = np.load(WORK_DIR / f'split_{tag}.npz')
    return df.loc[sp['test']]


def apply_condition(texts, tag, condition):
    """condition in {'raw', 'cleaned'}. Both get nbsp normalization (encoding
    hygiene, not the artifact under test); 'cleaned' additionally strips the
    dataset-specific, label-correlated artifact."""
    texts = [normalize_nbsp(t) for t in texts]
    if condition == 'raw':
        return texts
    if tag == 'D2':
        return [clean_hc3_whitespace(t) for t in texts]
    return [clean_daigt_unicode(t) for t in texts]


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


def run_cell(tag, model_key, condition, device='cpu'):
    key = 'artclean_zeroshot_%s_%s_%s' % (tag, model_key, condition)
    jpath, ppath = RESULTS_DIR / (key + '.json'), PROBS_DIR / (key + '.npz')
    if jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print('[skip] %-46s f1=%.4f' % (key, rec['test']['f1']))
        return rec

    mdir = MODELS_DIR / ('%s_%s' % (tag, model_key))
    if not (mdir / 'model.safetensors').exists():
        print('[missing] no checkpoint at %s, skipping' % mdir)
        return None

    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from sklearn.metrics import confusion_matrix

    test_df = load_test_split(tag)
    texts = apply_condition(list(test_df['text']), tag, condition)
    y = test_df['label'].values

    print('scoring %s (trained on %s, raw) under condition=%s ...' % (model_key, DATASET_NAMES[tag], condition))
    tok = AutoTokenizer.from_pretrained(str(mdir))
    model = AutoModelForSequenceClassification.from_pretrained(str(mdir)).to(device)
    model.eval()

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
    p = np.concatenate(all_probs, axis=0)
    secs = time.time() - t0

    acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag], 'model': model_key,
           'condition': condition, 'stage': 'zeroshot', 'n_test': len(y), 'eval_seconds': round(secs, 1),
           'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1},
           'test_confusion': confusion_matrix(y, p.argmax(1)).tolist()}
    atomic_write_npz(ppath, test_probs=p, test_labels=y)
    atomic_write_json(jpath, out)
    print('[done] %-46s f1=%.4f  (%.1f min)' % (key, f1, secs / 60))

    del model, tok; gc.collect()
    return out


def build_table():
    rows = []
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            raw_p = RESULTS_DIR / ('artclean_zeroshot_%s_%s_raw.json' % (tag, mk))
            clean_p = RESULTS_DIR / ('artclean_zeroshot_%s_%s_cleaned.json' % (tag, mk))
            raw_f1 = json.load(open(raw_p))['test']['f1'] if raw_p.exists() else None
            clean_f1 = json.load(open(clean_p))['test']['f1'] if clean_p.exists() else None
            rows.append({'dataset': DATASET_NAMES[tag], 'model': mk, 'raw_f1': raw_f1,
                         'cleaned_f1': clean_f1,
                         'delta': round(raw_f1 - clean_f1, 4) if (raw_f1 is not None and clean_f1 is not None) else None})
    t = pd.DataFrame(rows)
    out_csv = TABLES_DIR / 'table_artifact_cleaning_zeroshot.csv'
    t.to_csv(out_csv, index=False)
    print('\n' + t.to_string(index=False))
    print('\nwritten to:', out_csv)
    print('\nSanity check: raw_f1 should match the in-domain baseline already in '
          'table2_combined_full.csv (re-scoring the same checkpoint on unperturbed '
          'text should reproduce the known number).')


if __name__ == '__main__':
    import torch
    device = os.environ.get('NLP_CROSS_DEVICE', 'cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: cuda requested but not available, falling back to cpu')
        device = 'cpu'
    print('running artifact-cleaning zero-shot eval on device:', device,
          '(set NLP_CROSS_DEVICE=cuda to use the GPU once it is free)')
    for tag in ('D1', 'D2'):
        for mk in ('BERT', 'DeBERTa'):
            for condition in ('raw', 'cleaned'):
                run_cell(tag, mk, condition, device=device)
    build_table()
