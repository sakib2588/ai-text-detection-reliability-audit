"""Split-level variance for the surface-content decomposition.

Every number in the ICCIT paper rests on one group-aware partition, seed 42. The
bootstrap intervals in audit/verify_paper_claims.py resample the TEST SET, so they
capture sampling noise and say nothing about how much the result moves if the
corpus is partitioned differently. This project's own history shows that matters:
a three-seed spread estimate moved 31% when its runs were repeated.

So: hold the balanced sample fixed (same corpus, same rows) and vary only the
partition seed, 42/123/456/789/1337. That isolates split variance from sampling
variance and from class-balance variance. Same construction as
paper_scale/build_full_splits.py:group_split, so no duplicate-content group ever
crosses a boundary.

Arms re-fitted per split are the four that can be re-fitted on CPU. The transformer
reference cannot -- it would need retraining per split -- so it is absent here by
construction, and the claim this script supports is about the surface-vs-content
comparison only, which is exactly the claim that needed it.

Output: audit/multisplit_decomposition.json
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent))
from surface_content_decomposition import (LENGTH_IDX, PURE_LENGTH_IDX,
                                           content_normalise, score,
                                           surface_features)

FINAL = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK = FINAL / 'paper_scale' / 'work'
OUT = FINAL / 'audit' / 'multisplit_decomposition.json'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
SPLIT_SEEDS = [42, 123, 456, 789, 1337]
FIT_SEED = 42


def content_hash(series):
    return series.map(
        lambda t: hashlib.md5(re.sub(r'\s+', ' ', str(t)).strip().lower().encode()).hexdigest())


def group_split(df, seed):
    """72/8/20, never splitting a duplicate-content group. Same as build_full_splits.py."""
    groups = df['hash'].values
    tr_full, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                         random_state=seed).split(df, df['label'], groups))
    sub = df.iloc[tr_full]
    rel_tr, rel_val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
                           .split(sub, sub['label'], sub['hash'].values))
    tr, val = tr_full[rel_tr], tr_full[rel_val]
    g = [set(df['hash'].values[i]) for i in (tr, val, te)]
    assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2]), 'GROUP LEAKAGE'
    return tr, val, te


def mcnemar(ca, cb):
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    return {'b': b, 'c': c, 'exact_p': round(p, 8),
            'err_diff_a_minus_b': round(float((1 - ca.mean()) - (1 - cb.mean())), 6)}


def run_split(texts, y, tr, te):
    ytr, yte = y[tr], y[te]
    arms, corr = {}, {}

    def fit(label, Xtr, Xte, nfeat):
        clf = LogisticRegression(max_iter=1000, random_state=FIT_SEED).fit(Xtr, ytr)
        pred = clf.predict(Xte)
        rec = score(yte, pred)
        rec['n_features'] = int(nfeat)
        rec['converged'] = bool(np.max(clf.n_iter_) < 1000)
        arms[label] = rec
        corr[label] = (pred == yte)

    X = np.nan_to_num(np.array([surface_features(t) for t in texts], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    keep = [i for i in range(X.shape[1]) if i not in LENGTH_IDX]
    fit('surface_only', Xz[tr], Xz[te], X.shape[1])
    fit('surface_only_nolength', Xz[tr][:, keep], Xz[te][:, keep], len(keep))
    fit('length_only', Xz[tr][:, PURE_LENGTH_IDX], Xz[te][:, PURE_LENGTH_IDX],
        len(PURE_LENGTH_IDX))

    norm = np.array([content_normalise(t) for t in texts], dtype=object)
    vec = CountVectorizer()
    Ctr, Cte = vec.fit_transform(norm[tr]), vec.transform(norm[te])
    fit('content_only', Ctr, Cte, Ctr.shape[1])
    scale = float(Ctr.sum(axis=1).mean())
    fit('content_only_l1norm_scaled', normalize(Ctr, norm='l1') * scale,
        normalize(Cte, norm='l1') * scale, Ctr.shape[1])

    tests = {
        'surface_vs_content': mcnemar(corr['surface_only'], corr['content_only']),
        'surface_vs_content_lengthfree': mcnemar(corr['surface_only_nolength'],
                                                 corr['content_only_l1norm_scaled']),
    }
    return arms, tests


def main():
    report = {'split_seeds': SPLIT_SEEDS, 'fit_seed': FIT_SEED,
              'design': 'balanced sample held fixed, only the partition seed varies',
              'datasets': {}}
    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet').reset_index(drop=True)
        df['hash'] = content_hash(df['text'])
        texts, y = df['text'].values, df['label'].values
        per_split = {}
        for sd in SPLIT_SEEDS:
            tr, val, te = group_split(df, sd)
            arms, tests = run_split(texts, y, tr, te)
            per_split[str(sd)] = {'n_train': int(len(tr)), 'n_val': int(len(val)),
                                  'n_test': int(len(te)), 'arms': arms, 'tests': tests}
            print(f'[{tag} seed {sd}] surface={arms["surface_only"]["error_rate"]:.4f} '
                  f'content={arms["content_only"]["error_rate"]:.4f} '
                  f'p={tests["surface_vs_content"]["exact_p"]:.4f} | '
                  f'lenfree surf={arms["surface_only_nolength"]["error_rate"]:.4f} '
                  f'cont={arms["content_only_l1norm_scaled"]["error_rate"]:.4f}', flush=True)

        summary = {}
        for arm in ('surface_only', 'content_only', 'surface_only_nolength',
                    'content_only_l1norm_scaled', 'length_only'):
            v = [per_split[str(s)]['arms'][arm]['error_rate'] for s in SPLIT_SEEDS]
            summary[arm] = {'error_by_split': v, 'mean': round(float(np.mean(v)), 6),
                            'min': min(v), 'max': max(v), 'range': round(max(v) - min(v), 6)}
        for t in ('surface_vs_content', 'surface_vs_content_lengthfree'):
            d = [per_split[str(s)]['tests'][t]['err_diff_a_minus_b'] for s in SPLIT_SEEDS]
            p = [per_split[str(s)]['tests'][t]['exact_p'] for s in SPLIT_SEEDS]
            summary[t] = {'err_diff_by_split': d, 'mean_err_diff': round(float(np.mean(d)), 6),
                          'diff_range': round(max(d) - min(d), 6),
                          'exact_p_by_split': p, 'max_p': max(p), 'min_p': min(p),
                          'all_splits_significant_at_0.05': bool(max(p) < 0.05),
                          'no_split_significant_at_0.05': bool(min(p) >= 0.05),
                          'sign_consistent': bool(all(x > 0 for x in d) or all(x < 0 for x in d))}
        report['datasets'][tag] = {'name': name, 'per_split': per_split, 'summary': summary}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, 'w'), indent=2)
    print('\nwritten to:', OUT)


if __name__ == '__main__':
    main()
