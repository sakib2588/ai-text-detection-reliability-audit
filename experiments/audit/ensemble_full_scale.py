"""Soft-vote ensemble of the two deployed transformer checkpoints, at full scale.

NUMBERS_SSOT.md Section 1 records an ensemble row (0.9949 / 0.9982) taken from the
same grid-maximum table that Section 13 supersedes, so it is not comparable to the
checkpoint-matched numbers the rest of the paper reports. This script recomputes the
ensemble from the DEPLOYED checkpoints' own saved probabilities, selects the mixing
weight on validation only, and tests the result against its stronger member with the
same paired machinery every other comparison in the paper uses.

Reads   experiments/paper_scale/models/{tag}_{model}/run_info.json   (which checkpoint is deployed)
        experiments/paper_scale/probs/{key}.npz                     (its val and test probabilities)
Writes  experiments/audit/ensemble_full_scale.json
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

FINAL = Path(__file__).resolve().parents[2]
PS = FINAL / 'experiments' / 'paper_scale'
OUT = FINAL / 'experiments' / 'audit' / 'ensemble_full_scale.json'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
MEMBERS = ('BERT', 'DeBERTa')
WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
BOOT = 10000
SEED = 42


def scores(y, pred):
    acc = accuracy_score(y, pred)
    _, _, wf1, _ = precision_recall_fscore_support(y, pred, average='weighted',
                                                   zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(y, pred, average='macro',
                                                   zero_division=0)
    return {'accuracy': round(float(acc), 4),
            'error_rate': round(float(1 - acc), 4),
            'weighted_f1': round(float(wf1), 4),
            'macro_f1': round(float(mf1), 4),
            'confusion_tn_fp_fn_tp': [int(v) for v in
                                      confusion_matrix(y, pred, labels=[0, 1]).ravel()]}


def paired(y, pred_a, pred_b, rng):
    """McNemar exact plus a paired bootstrap on the error difference, a minus b."""
    wa, wb = pred_a != y, pred_b != y
    b = int((~wa & wb).sum())      # a right, b wrong
    c = int((wa & ~wb).sum())      # a wrong, b right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    diff = float(wa.mean() - wb.mean())
    n = len(y)
    idx = rng.integers(0, n, size=(BOOT, n))
    boot = wa[idx].mean(1) - wb[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mcnemar_b_a_right_b_wrong': b, 'mcnemar_c_a_wrong_b_right': c,
            'mcnemar_exact_p': round(float(p), 6),
            'error_diff_a_minus_b': round(diff, 6),
            'error_diff_ci95_lo': round(float(lo), 6),
            'error_diff_ci95_hi': round(float(hi), 6),
            'ci_excludes_zero': bool(lo > 0 or hi < 0)}


def main():
    rng = np.random.default_rng(SEED)
    report = {'seed': SEED, 'bootstrap_resamples': BOOT,
              'weight_grid': [float(w) for w in WEIGHTS],
              'selection': 'mixing weight chosen on VALIDATION weighted F1 only',
              'members': 'deployed checkpoints, never grid maxima',
              'supersedes': 'NUMBERS_SSOT.md Section 1 ensemble row, which used grid maxima',
              'datasets': {}}

    for tag, name in DATASETS.items():
        info, probs = {}, {}
        for mk in MEMBERS:
            ri = json.load(open(PS / 'models' / f'{tag}_{mk}' / 'run_info.json'))
            z = np.load(PS / 'probs' / f"{ri['key']}.npz")
            info[mk] = ri
            probs[mk] = {k: z[k] for k in z.files}

        yv = probs['BERT']['val_labels']
        yt = probs['BERT']['test_labels']
        assert np.array_equal(yv, probs['DeBERTa']['val_labels']), 'val labels differ'
        assert np.array_equal(yt, probs['DeBERTa']['test_labels']), 'test labels differ'

        # Select w on validation only.
        val_f1 = []
        for w in WEIGHTS:
            mix = w * probs['BERT']['val_probs'] + (1 - w) * probs['DeBERTa']['val_probs']
            _, _, f1, _ = precision_recall_fscore_support(
                yv, mix.argmax(1), average='weighted', zero_division=0)
            val_f1.append(float(f1))
        best_w = float(WEIGHTS[int(np.argmax(val_f1))])

        test_mix = best_w * probs['BERT']['test_probs'] + \
            (1 - best_w) * probs['DeBERTa']['test_probs']
        ens_pred = test_mix.argmax(1)
        member_pred = {mk: probs[mk]['test_probs'].argmax(1) for mk in MEMBERS}

        ens = scores(yt, ens_pred)
        mem = {mk: scores(yt, member_pred[mk]) for mk in MEMBERS}
        stronger = max(MEMBERS, key=lambda mk: mem[mk]['weighted_f1'])

        report['datasets'][tag] = {
            'name': name,
            'n_val': int(len(yv)), 'n_test': int(len(yt)),
            'members': {mk: {'key': info[mk]['key'], 'lr': info[mk]['lr'],
                             'batch_size': info[mk]['batch_size'],
                             'weight_decay': info[mk]['weight_decay'],
                             'test': mem[mk]} for mk in MEMBERS},
            'val_f1_by_weight': {str(w): round(f, 4) for w, f in zip(WEIGHTS, val_f1)},
            'weight_on_bert': best_w,
            'weight_on_deberta': round(1 - best_w, 2),
            'degenerate': bool(best_w in (0.0, 1.0)),
            'ensemble_test': ens,
            'stronger_member': stronger,
            'ensemble_minus_stronger_f1': round(
                ens['weighted_f1'] - mem[stronger]['weighted_f1'], 4),
            'paired_ensemble_vs_stronger': paired(
                yt, ens_pred, member_pred[stronger], rng),
        }

        verdict = ('beats' if ens['weighted_f1'] > mem[stronger]['weighted_f1']
                   else 'does not beat')
        print(f'[{tag}] {name}: w_BERT={best_w:.2f}  ensemble wF1={ens["weighted_f1"]:.4f}  '
              f'{stronger}={mem[stronger]["weighted_f1"]:.4f}  -> {verdict} its stronger member'
              + ('   [DEGENERATE, collapsed onto one member]' if best_w in (0.0, 1.0) else ''))
        pr = report['datasets'][tag]['paired_ensemble_vs_stronger']
        print(f'      McNemar p={pr["mcnemar_exact_p"]:.4g}  '
              f'error diff {pr["error_diff_a_minus_b"]*100:+.3f} pp  '
              f'95% CI [{pr["error_diff_ci95_lo"]*100:+.3f}, {pr["error_diff_ci95_hi"]*100:+.3f}]')

    json.dump(report, open(OUT, 'w'), indent=1)
    print('\nwritten to', OUT.relative_to(FINAL))


if __name__ == '__main__':
    main()
