"""Artifact-cleaning ablation, FULL retrain-based (Gap 4, stage 3b) -- the
review-recommended design: "run every cell twice... raw and cleaned." Unlike
the zero-shot interim signal (run_artifact_cleaning_zeroshot.py), this
retrains BERT and DeBERTa from scratch on CLEANED train+val+test data, at
the same already-identified winning hyperparameter config, so the model
never has the chance to learn the artifact as a shortcut in the first place
-- not just whether it can still classify once the artifact is removed at
test time only. This is the number that belongs in the paper's main results
table; the zero-shot number is a secondary/discussion point.

Reuses run_full_scale.py's run_one() directly (not reimplemented) via a
monkey-patch: writes a cleaned parquet + a copied split file under a
synthetic tag ('D1c'/'D2c'), registers matching WINNERS/DATASET_NAMES
entries, then calls run_one('D1c', ...) exactly like a normal full-scale
run. This keeps the atomic-write / skip-if-done / OOM-retry-ladder behavior
already built for GPU jobs, with zero duplicated training logic.

GPU job -- do not run concurrently with any other GPU consumer. Per the
project plan, launch this only after the seed-robustness retrain chain's
GPU stage (stage 1/2) has finished; stage 2 (cross-dataset eval) is CPU-only
so it does not block this.
"""
import time
from pathlib import Path
import numpy as np, pandas as pd

import run_full_scale as rfs
from text_perturbations import normalize_nbsp, clean_hc3_whitespace, clean_daigt_unicode, length_match

CLEAN_TAGS = {'D1': 'D1c', 'D2': 'D2c'}


def build_cleaned_data(tag):
    """Writes work/data_{tag}c.parquet (cleaned text, same row index / split
    membership as the raw data) and copies split_{tag}.npz -> split_{tag}c.npz
    unchanged, so the cleaned run uses an IDENTICAL train/val/test partition
    to the raw run -- only the text content differs, which is what makes the
    raw-vs-cleaned comparison valid."""
    ctag = CLEAN_TAGS[tag]
    out_parquet = rfs.WORK_DIR / f'data_{ctag}.parquet'
    out_split = rfs.WORK_DIR / f'split_{ctag}.npz'

    if out_parquet.exists() and out_split.exists():
        print('[skip] cleaned data for %s already built at %s' % (tag, out_parquet))
        return

    df = pd.read_parquet(rfs.WORK_DIR / f'data_{tag}.parquet')
    texts = [normalize_nbsp(t) for t in df['text']]
    if tag == 'D2':
        texts = [clean_hc3_whitespace(t) for t in texts]
    else:
        texts = [clean_daigt_unicode(t) for t in texts]
    df = df.assign(text=texts)
    df['text'] = length_match(df, label_col='label', text_col='text', unit='words')

    df.to_parquet(out_parquet, index=True)

    sp = np.load(rfs.WORK_DIR / f'split_{tag}.npz')
    np.savez(out_split, train=sp['train'], val=sp['val'], test=sp['test'])
    print('[built] cleaned data for %s -> %s (%d rows)' % (tag, out_parquet, len(df)))


def register_synthetic_tag(tag):
    ctag = CLEAN_TAGS[tag]
    rfs.DATASET_NAMES[ctag] = rfs.DATASET_NAMES[tag] + ' (cleaned)'
    for mk in ('BERT', 'DeBERTa'):
        rfs.WINNERS[(ctag, mk)] = rfs.WINNERS[(tag, mk)]


if __name__ == '__main__':
    import torch
    print('torch', torch.__version__, '| cuda', torch.cuda.is_available())

    for tag in ('D1', 'D2'):
        build_cleaned_data(tag)
        register_synthetic_tag(tag)

    total = 4  # 2 datasets x 2 models, seed 42 only (this ablation asks "does
               # cleaning change accuracy at the already-best config", not a
               # new hyperparameter search or a new seed-robustness study)
    done = 0
    t_start = time.time()
    for tag in ('D1', 'D2'):
        ctag = CLEAN_TAGS[tag]
        for mk in ('BERT', 'DeBERTa'):
            done += 1
            print('--- cleaning-ablation retrain %d/%d  %s %s (cleaned) ---' % (done, total, tag, mk))
            rfs.run_one(ctag, mk, seed=42, save_model=False, force=False)
            e = time.time() - t_start
            print('    elapsed %.1f min, projected remaining %.1f min\n' % (e / 60, e / done * (total - done) / 60))
        rfs._DATACACHE.clear()
    print('ARTIFACT-CLEANING FULL ABLATION COMPLETE in %.2f hours' % ((time.time() - t_start) / 3600))

    # build the raw-vs-cleaned comparison table
    rows = []
    for tag in ('D1', 'D2'):
        ctag = CLEAN_TAGS[tag]
        for mk in ('BERT', 'DeBERTa'):
            raw_files = sorted(rfs.RESULTS_DIR.glob(f'full_{tag}_{mk}_*_s42.json'))
            raw_f1, best_val = None, -1
            for f in raw_files:
                import json
                r = json.load(open(f))
                if r.get('val', {}).get('f1', -1) > best_val:
                    best_val = r['val']['f1']; raw_f1 = r['test']['f1']
            clean_files = sorted(rfs.RESULTS_DIR.glob(f'full_{ctag}_{mk}_*_s42.json'))
            clean_f1 = None
            if clean_files:
                import json
                clean_f1 = json.load(open(clean_files[0]))['test']['f1']
            rows.append({'dataset': rfs.DATASET_NAMES[tag], 'model': mk, 'raw_f1': raw_f1,
                         'cleaned_f1': clean_f1,
                         'delta': round(raw_f1 - clean_f1, 4) if (raw_f1 is not None and clean_f1 is not None) else None})
    t = pd.DataFrame(rows)
    out_csv = rfs.FINAL_DIR / 'table_artifact_cleaning_full.csv'
    t.to_csv(out_csv, index=False)
    print('\n' + t.to_string(index=False))
    print('\nwritten to:', out_csv)
