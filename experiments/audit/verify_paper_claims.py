"""Verifies, against artifacts already on disk, every claim in iccit/ that the
2026-08-24 review found unsourced, misstated, or asserted without a test.

Nothing here retrains anything. Every number is derived from files the pipeline
already wrote, so this script is cheap enough to re-run before any submission.

Six blocks, each closing one review finding:

  F1  splits          the paper says 80/10/10; measure what build_full_splits.py
                      actually produced
  F4  leakage         exact-text leakage, duplicate-group-aware split versus the
                      naive split, per dataset. Resolves the OPEN 11.2-11.3%
                      item in NUMBERS_SSOT.md Section 5
  F3b seed_spreads    every measured seed spread, keyed by dataset/model/config,
                      so the noise band quoted in prose has a source and belongs
                      to the cell it is applied to
  F6  whitespace_cue  the space-before-punctuation rate per human and machine
                      document, which prose quotes as 10.745 / 0.013 with no
                      source file anywhere in the repo
  F3a table1_stats    bootstrap 95% CI on error rate for all sixteen Table 1
                      cells, plus paired McNemar and a paired bootstrap on the
                      error difference for the comparisons the paper draws
                      conclusions from
  F3a decomposition   the same paired tests between the surface-only and
                      content-only arms, which is what "indistinguishable" needs
                      and does not currently have

Output: audit/paper_claim_verification.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

FINAL = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK = FINAL / 'paper_scale' / 'work'
RESULTS = FINAL / 'paper_scale' / 'results'
AUDIT = FINAL / 'audit'
OUT = AUDIT / 'paper_claim_verification.json'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
SEED = 42
N_BOOT = 10000

# identical to surface_content_decomposition.py, so the cue is measured the same
# way the surface arm measures it
WS_PUNCT = re.compile(r' +[.,;:!?]')


# --------------------------------------------------------------------------
# F1 + F4: what the splits actually are, and what leaks through them
# --------------------------------------------------------------------------
def check_splits_and_leakage():
    out = {}
    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        n = len(df)
        entry = {'name': name, 'n_balanced': int(n), 'variants': {}}
        for kind, fname in (('group_aware', f'split_{tag}.npz'),
                            ('naive', f'split_{tag}_naive.npz')):
            p = WORK / fname
            if not p.exists():
                entry['variants'][kind] = {'error': f'missing {fname}'}
                continue
            sp = np.load(p)
            tr, va, te = sp['train'], sp['val'], sp['test']
            seen = set(df.loc[tr, 'text']) | set(df.loc[va, 'text'])
            leaked = int(df.loc[te, 'text'].isin(seen).sum())
            entry['variants'][kind] = {
                'n_train': int(len(tr)), 'n_val': int(len(va)), 'n_test': int(len(te)),
                'pct_train': round(100 * len(tr) / n, 2),
                'pct_val': round(100 * len(va) / n, 2),
                'pct_test': round(100 * len(te) / n, 2),
                'ratio_string': '%.0f/%.0f/%.0f' % (100 * len(tr) / n, 100 * len(va) / n,
                                                    100 * len(te) / n),
                'exact_text_leak_rows': leaked,
                'exact_text_leak_pct': round(100 * leaked / len(te), 4),
            }
        out[tag] = entry
    return out


# --------------------------------------------------------------------------
# F3b: every seed spread that exists, so none has to be imported from
#      another dataset to serve as a noise band
# --------------------------------------------------------------------------
def check_seed_spreads():
    by_cfg = {}
    for p in sorted(RESULTS.glob('full_*_s*.json')):
        stem = p.stem
        if '_s' not in stem:
            continue
        cfg, seed = stem.rsplit('_s', 1)
        if not seed.isdigit():
            continue
        try:
            rec = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        f1 = rec.get('test', {}).get('f1')
        if f1 is None:
            continue
        by_cfg.setdefault(cfg, {})[seed] = float(f1)

    out = {}
    for cfg, seeds in sorted(by_cfg.items()):
        if len(seeds) < 2:
            continue
        vals = [seeds[s] for s in sorted(seeds)]
        parts = cfg.split('_')
        out[cfg] = {
            'dataset': parts[1] if len(parts) > 1 else None,
            'model': parts[2] if len(parts) > 2 else None,
            'n_seeds': len(vals),
            'seeds': sorted(seeds),
            'test_f1_by_seed': {s: seeds[s] for s in sorted(seeds)},
            'spread_range': round(max(vals) - min(vals), 4),
            'mean': round(sum(vals) / len(vals), 4),
        }
    return out


# --------------------------------------------------------------------------
# F6: source the space-before-punctuation rate
# --------------------------------------------------------------------------
def check_whitespace_cue():
    out = {}
    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        counts = df['text'].map(lambda t: len(WS_PUNCT.findall(str(t))))
        sp = np.load(WORK / f'split_{tag}.npz')
        entry = {'name': name}
        for scope, idx in (('balanced_corpus', df.index),
                           ('test_split', pd.Index(sp['test']))):
            sub_c, sub_y = counts.loc[idx], df.loc[idx, 'label']
            entry[scope] = {
                'n': int(len(idx)),
                'mean_per_human_doc': round(float(sub_c[sub_y == 0].mean()), 4),
                'mean_per_machine_doc': round(float(sub_c[sub_y == 1].mean()), 4),
                'pct_human_docs_with_cue': round(float((sub_c[sub_y == 0] > 0).mean() * 100), 2),
                'pct_machine_docs_with_cue': round(float((sub_c[sub_y == 1] > 0).mean() * 100), 2),
            }
        out[tag] = entry
    return out


# --------------------------------------------------------------------------
# Paired statistics. Bootstrap because the project protocol forbids parametric
# tests at n<=30 and prefers percentile CIs everywhere; exact-binomial McNemar
# because the discordant counts here are small enough that the chi-square
# approximation is the wrong instrument.
# --------------------------------------------------------------------------
def bootstrap_err_ci(correct, rng, n_boot=N_BOOT):
    n = len(correct)
    idx = rng.integers(0, n, size=(n_boot, n))
    errs = 1.0 - correct[idx].mean(axis=1)
    lo, hi = np.percentile(errs, [2.5, 97.5])
    return {'error_rate': round(float(1 - correct.mean()), 6),
            'ci95_lo': round(float(lo), 6), 'ci95_hi': round(float(hi), 6),
            'n_boot': n_boot, 'n_test': int(n)}


def paired_compare(correct_a, correct_b, label_a, label_b, rng, n_boot=N_BOOT):
    """McNemar (exact binomial) plus a paired bootstrap on the error difference.

    b = a right, b wrong; c = a wrong, b right. Positive mean_error_diff means
    a makes MORE errors than b.
    """
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    n = len(correct_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diff = (1 - correct_a[idx].mean(axis=1)) - (1 - correct_b[idx].mean(axis=1))
    lo, hi = np.percentile(diff, [2.5, 97.5])
    return {
        'a': label_a, 'b': label_b,
        'mcnemar_b_a_right_b_wrong': b,
        'mcnemar_c_a_wrong_b_right': c,
        'mcnemar_exact_p': round(p, 6),
        'error_diff_a_minus_b': round(float((1 - correct_a.mean()) - (1 - correct_b.mean())), 6),
        'error_diff_ci95_lo': round(float(lo), 6),
        'error_diff_ci95_hi': round(float(hi), 6),
        'ci_excludes_zero': bool(lo > 0 or hi < 0),
    }


# --------------------------------------------------------------------------
# F3a, Table 1
# --------------------------------------------------------------------------
def check_table1():
    npz_path = AUDIT / 'full_model_scores.npz'
    if not npz_path.exists():
        return {'error': 'audit/full_model_scores.npz missing'}
    z = np.load(npz_path, allow_pickle=True)
    recorded = {}
    ev = AUDIT / 'full_model_evaluation.json'
    if ev.exists():
        recorded = json.load(open(ev)).get('datasets', {})

    rng = np.random.default_rng(SEED)
    out = {}
    for tag, name in DATASETS.items():
        y = z[f'{tag}|y_true']
        models = [k.split('|', 1)[1] for k in z.files
                  if k.startswith(f'{tag}|') and not k.endswith('y_true')]
        cells, correct_by_model = {}, {}
        for m in models:
            s = z[f'{tag}|{m}'].astype(float)
            # SVM stores decision_function, everything else P(machine).
            thresh = 0.0 if 'Support Vector Machine' in m else 0.5
            pred = (s > thresh).astype(int)
            correct = (pred == y)
            correct_by_model[m] = correct
            cell = bootstrap_err_ci(correct, rng)
            # self-check: the derived predictions must reproduce the recorded
            # error rate, else the threshold assumption above is wrong
            rec = recorded.get(tag, {}).get('models', {}).get(m, {})
            rec_err = rec.get('error_rate')
            if rec_err is not None:
                cell['recorded_error_rate'] = rec_err
                cell['matches_recorded'] = bool(abs(cell['error_rate'] - rec_err) < 5e-4)
            cells[m] = cell

        best_classical = min((m for m in models if m not in ('BERT', 'DeBERTa')),
                             key=lambda m: cells[m]['error_rate'])
        best_transformer = min(('BERT', 'DeBERTa'), key=lambda m: cells[m]['error_rate'])
        comps = [paired_compare(correct_by_model[best_classical],
                                correct_by_model[best_transformer],
                                best_classical, best_transformer, rng),
                 paired_compare(correct_by_model['BERT'], correct_by_model['DeBERTa'],
                                'BERT', 'DeBERTa', rng)]
        out[tag] = {'name': name, 'best_classical': best_classical,
                    'best_transformer': best_transformer,
                    'cells': cells, 'paired_comparisons': comps}
    return out


# --------------------------------------------------------------------------
# F3a, the decomposition. Needs per-document predictions, which
# surface_content_decomposition.py only persists after the review patch.
# --------------------------------------------------------------------------
def check_decomposition():
    p = AUDIT / 'surface_content_predictions.npz'
    if not p.exists():
        return {'status': 'pending',
                'reason': 'run the patched audit/surface_content_decomposition.py first; '
                          'it must write surface_content_predictions.npz'}
    z = np.load(p, allow_pickle=True)
    rng = np.random.default_rng(SEED)
    out = {}
    for tag, name in DATASETS.items():
        y = z[f'{tag}|y_true']
        arms = sorted(k.split('|', 1)[1] for k in z.files
                      if k.startswith(f'{tag}|') and not k.endswith('y_true'))
        corr = {a: (z[f'{tag}|{a}'] == y) for a in arms}
        entry = {'name': name,
                 'arms': {a: bootstrap_err_ci(corr[a], rng) for a in arms},
                 'paired_comparisons': []}
        for a, b in (('surface_only', 'content_only'),
                     ('surface_only_nolength', 'content_only_l1norm_scaled'),
                     ('content_only', 'content_only_l1norm_scaled'),
                     ('surface_only', 'surface_only_nolength')):
            if a in corr and b in corr:
                entry['paired_comparisons'].append(paired_compare(corr[a], corr[b], a, b, rng))
        out[tag] = entry
    return out


def main():
    report = {
        'seed': SEED,
        'purpose': 'verifies iccit/ claims flagged in the 2026-08-24 review',
        'splits_and_leakage': check_splits_and_leakage(),
        'seed_spreads': check_seed_spreads(),
        'whitespace_cue': check_whitespace_cue(),
        'table1_stats': check_table1(),
        'decomposition_stats': check_decomposition(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, 'w'), indent=2)
    print('written to:', OUT)
    return report


if __name__ == '__main__':
    main()
