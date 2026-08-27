"""Does the HC3 null survive choosing each arm's regularisation on validation?

Round-2 review objection M1. Both decomposition arms are fitted at the library
default inverse regularisation strength. Sharing it makes the arms comparable to
each other, not to their own best, and they are not symmetric in dimension, 47
standardised features against roughly 47,000 raw counts. The HC3 gap being
defended is 0.06 points, so the objection is that a sweep could erase it.

Phase A sweeps C on the validation partition, independently per arm and per
corpus, and refits at the selected value on the same training partition the paper
uses. Phase B re-runs the five group-aware partitions at the selected values so
the null is re-tested where the paper tests it, not only on one split.

Nothing here changes the split, the feature maps or the classifier family.
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit

FINAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINAL / 'experiments' / 'audit'))
from surface_content_decomposition import surface_features, content_normalise  # noqa: E402

WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'regularisation_sweep.json'
DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SPLIT_SEEDS = [42, 123, 456, 789, 1337]
FIT_SEED = 42
BOOT = 10000


def wf1(y, p):
    _, _, v, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return float(v)


def fit(Xtr, ytr, Xte, C):
    clf = LogisticRegression(max_iter=1000, random_state=FIT_SEED, C=C).fit(Xtr, ytr)
    return clf.predict(Xte), bool(np.max(clf.n_iter_) < 1000)


def paired(y, pa, pb, rng):
    """McNemar exact binomial plus a paired bootstrap on the error difference."""
    wa, wb = pa != y, pb != y
    b = int((~wa & wb).sum())
    c = int((wa & ~wb).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) else 1.0
    idx = rng.integers(0, len(y), size=(BOOT, len(y)))
    boot = (wa[idx].mean(1) - wb[idx].mean(1)) * 100
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'b': b, 'c': c, 'mcnemar_exact_p': round(p, 8),
            'err_diff_pp': round(float((wa.mean() - wb.mean()) * 100), 3),
            'ci95_pp': [round(float(lo), 3), round(float(hi), 3)],
            'excludes_zero': bool(lo > 0 or hi < 0)}


def build_arms(texts, tr, te):
    """Surface z-scored on train, content raw counts vectorised on train."""
    X = np.nan_to_num(np.array([surface_features(t) for t in texts], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    norm = np.array([content_normalise(t) for t in texts], dtype=object)
    vec = CountVectorizer()
    Ctr = vec.fit_transform(norm[tr])
    Cte = vec.transform(norm[te])
    return {'surface': (Xz[tr], Xz[te]), 'content': (Ctr, Cte)}


def group_split(df, seed):
    groups = df['hash'].values
    tr_full, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                         random_state=seed).split(df, df['label'], groups))
    sub = df.iloc[tr_full]
    rel_tr, rel_val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                           .split(sub, sub['label'], sub['hash'].values))
    return tr_full[rel_tr], tr_full[rel_val], te


def main():
    rng = np.random.default_rng(FIT_SEED)
    report = {'objection': 'round-2 M1, both arms fitted at the default C',
              'c_grid': C_GRID, 'selection': 'validation weighted F1, per arm, per corpus',
              'fit_seed': FIT_SEED, 'bootstrap': BOOT,
              'phase_a_paper_split': {}, 'phase_b_five_partitions': {}}

    selected = {}
    for tag, name in DATASETS.items():
        t0 = time.time()
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        itr = df.index.get_indexer(sp['train'])
        ival = df.index.get_indexer(sp['val'])
        ite = df.index.get_indexer(sp['test'])
        y = df['label'].values
        texts = df['text'].values
        print(f'[{tag}] {name} train={len(itr)} val={len(ival)} test={len(ite)}', flush=True)

        # fit on train, select on val, then score the selected model on test
        arms_val = build_arms(texts, itr, ival)
        arms_te = build_arms(texts, itr, ite)
        rec, preds = {}, {}
        for arm in ('surface', 'content'):
            Xtr, Xval = arms_val[arm]
            sweep = {}
            for C in C_GRID:
                pv, conv = fit(Xtr, y[itr], Xval, C)
                sweep[str(C)] = {'val_weighted_f1': round(wf1(y[ival], pv), 4),
                                 'val_error_pp': round(float((1 - accuracy_score(y[ival], pv)) * 100), 3),
                                 'converged': conv}
                print(f'[{tag}] {arm:8s} C={C:<7g} val wF1={sweep[str(C)]["val_weighted_f1"]:.4f}'
                      f'{"" if conv else "  [DID NOT CONVERGE]"}', flush=True)
            best = max(C_GRID, key=lambda c: sweep[str(c)]['val_weighted_f1'])
            Xtr2, Xte2 = arms_te[arm]
            pt, conv = fit(Xtr2, y[itr], Xte2, best)
            preds[arm] = pt
            rec[arm] = {'sweep': sweep, 'selected_C': best,
                        'selected_is_default': bool(best == 1.0),
                        'test_error_pp': round(float((1 - accuracy_score(y[ite], pt)) * 100), 3),
                        'test_weighted_f1': round(wf1(y[ite], pt), 4),
                        'converged': conv}
            print(f'[{tag}] {arm:8s} SELECTED C={best} test err='
                  f'{rec[arm]["test_error_pp"]:.3f}pp', flush=True)
        rec['surface_minus_content'] = paired(y[ite], preds['surface'], preds['content'], rng)
        rec['seconds'] = round(time.time() - t0, 1)
        selected[tag] = {a: rec[a]['selected_C'] for a in ('surface', 'content')}
        report['phase_a_paper_split'][tag] = {'name': name, **rec}
        OUT.write_text(json.dumps(report, indent=1))

    # ---- Phase B, the five group-aware partitions at the selected values ----
    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        df = df.reset_index(drop=True)
        df['hash'] = df['text'].map(
            lambda t: hashlib.md5(re.sub(r'\s+', ' ', str(t)).strip().lower().encode()).hexdigest())
        y = df['label'].values
        texts = df['text'].values
        rows = []
        for seed in SPLIT_SEEDS:
            tr, _val, te = group_split(df, seed)
            arms = build_arms(texts, tr, te)
            p = {}
            for arm in ('surface', 'content'):
                Xtr, Xte = arms[arm]
                p[arm], _ = fit(Xtr, y[tr], Xte, selected[tag][arm])
            r = paired(y[te], p['surface'], p['content'], rng)
            r['split_seed'] = seed
            r['n_test'] = int(len(te))
            r['surface_err_pp'] = round(float((p['surface'] != y[te]).mean() * 100), 3)
            r['content_err_pp'] = round(float((p['content'] != y[te]).mean() * 100), 3)
            rows.append(r)
            print(f'[{tag}] partition {seed}: surface {r["surface_err_pp"]:.2f} '
                  f'content {r["content_err_pp"]:.2f} diff {r["err_diff_pp"]:+.2f} '
                  f'p={r["mcnemar_exact_p"]:.4g}', flush=True)
        signs = {int(np.sign(r['err_diff_pp'])) for r in rows}
        report['phase_b_five_partitions'][tag] = {
            'name': name, 'selected_C': selected[tag], 'partitions': rows,
            'n_significant': sum(1 for r in rows if r['mcnemar_exact_p'] < 0.05),
            'sign_changes': bool(len(signs) > 1)}
        OUT.write_text(json.dumps(report, indent=1))

    print('written to', OUT, flush=True)


if __name__ == '__main__':
    main()
