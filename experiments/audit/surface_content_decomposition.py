"""Surface/content separability decomposition.

Measures how much of a detection benchmark's human-vs-machine separability is
carried by SURFACE FORM (orthography, punctuation, casing, length) versus
CONTENT (lexical choice, with surface normalised away).

Three arms per dataset, all on the same fixed splits:

  surface-only  hand-built orthographic features, NO lexical content at all
  content-only  bag-of-words over surface-normalised text, the same
                lower()/[^a-z\\s] pipeline experiments/paper_scale/classical_full.py uses,
                so casing, punctuation and non-ASCII cannot leak in.
                NOTE: this arm applies NO stopword removal and NO lemmatisation.
                Those steps belong to experiments/paper_scale/classical_full.py, which feeds
                the Table 1 classical models, not this decomposition. The content
                arm here is therefore an UNFILTERED bag-of-words, a stronger
                content model than a filtered one, not a weaker one.
  full          fine-tuned transformer on raw text (read from existing results,
                not refit here)

Two length-controlled variants are also fitted, because the two primary arms are
not disjoint on document length: the surface arm reads character, word and
sentence counts directly, and raw CountVectorizer rows sum to document length, so
the content arm can recover it too. Length is a documented confound on both
corpora (NUMBERS_SSOT.md Section 9; Baidya et al. on HC3), so a decomposition
that claims to separate surface from content has to show what happens when the
shared channel is closed:

  surface-only-nolength   the 42 features that do not scale with document size
  content-only-l1norm     the same bag-of-words with rows L1-normalised

Per-document predictions for all four arms are written alongside the JSON so the
"indistinguishable" claim can be tested with paired McNemar rather than asserted
from a two-decimal gap (see experiments/audit/verify_paper_claims.py).

surface-only and content-only share one classifier family (logistic regression)
so the two arms are directly comparable. The transformer is a reference point,
not a matched arm, and is labelled as such.

Every figure the paper quotes for this decomposition comes from this file's
JSON output. Nothing is computed ad hoc.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

FINAL = Path(__file__).resolve().parents[2]
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
RESULTS = FINAL / 'experiments' / 'paper_scale' / 'results'
OUT = FINAL / 'experiments' / 'audit' / 'surface_content_decomposition.json'
OUT_PRED = FINAL / 'experiments' / 'audit' / 'surface_content_predictions.npz'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
SEED = 42

WS_PUNCT = re.compile(r' +[.,;:!?]')
PUNCT_CHARS = '.,;:!?\'"-()[]{}/*&%$#@+=_~`^<>|\\'
EMOJI = re.compile('[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]')
SENT_END = re.compile(r'[.!?]+')


def surface_features(text):
    """Orthographic statistics only. No word identity is ever read, so this
    arm cannot access lexical content by construction."""
    t = str(text)
    n = max(len(t), 1)
    words = t.split()
    nw = max(len(words), 1)
    alpha = [c for c in t if c.isalpha()]
    na = max(len(alpha), 1)

    feats = [
        len(WS_PUNCT.findall(t)),                 # the documented HC3 cue, raw count
        len(WS_PUNCT.findall(t)) / n * 1000.0,    # and as a rate
        len(t),                                   # length, chars
        len(words),                               # length, words
        float(np.mean([len(w) for w in words])) if words else 0.0,
        sum(1 for c in alpha if c.isupper()) / na * 100.0,
        sum(1 for w in words if w[:1].isupper()) / nw * 100.0,
        sum(1 for w in words if len(w) > 1 and w.isupper()) / nw * 100.0,
        sum(1 for c in t if ord(c) > 127) / n * 1000.0,
        len(EMOJI.findall(t)),
        sum(1 for c in t if c.isdigit()) / n * 1000.0,
        t.count('  ') / n * 1000.0,               # double spaces
        t.count('\n') / n * 1000.0,
        len(SENT_END.findall(t)),
        len(words) / max(len(SENT_END.findall(t)), 1),   # mean sentence length
    ]
    # per-punctuation-character rates
    feats += [t.count(c) / n * 1000.0 for c in PUNCT_CHARS]
    return feats


# Indices into surface_features() whose magnitude scales with document size:
# the raw cue count (0), characters (2), words (3), the raw emoji count (9) and
# the sentence count (13). Every one of them has either a rate twin in the same
# vector or a rate stand-in, so dropping them removes the length channel without
# removing the cue itself. The remaining 42 features are rates and ratios.
LENGTH_IDX = [0, 2, 3, 9, 13]

# The three that are pure document size, used to fit a length-only arm.
PURE_LENGTH_IDX = [2, 3, 13]


SURFACE_NORM = re.compile(r'[^a-z\s]')


def content_normalise(text):
    """Identical surface normalisation to experiments/paper_scale/classical_full.py, so the
    content arm provably cannot see punctuation, casing, or non-ASCII."""
    t = SURFACE_NORM.sub(' ', str(text).lower())
    return ' '.join(t.split())


def score(y_true, y_pred):
    _, _, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0)
    per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_true, y_pred)
    return {
        'accuracy': round(float(acc), 4),
        'error_rate': round(float(1.0 - acc), 4),   # reported so ceiling compression cannot flatter
        'weighted_f1': round(float(wf1), 4),
        'macro_f1': round(float(mf1), 4),
        'f1_human': round(float(per_class[2][0]), 4),
        'f1_machine': round(float(per_class[2][1]), 4),
        'fpr_human_called_machine': round(float(fp / max(tn + fp, 1)), 4),
        'confusion_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'n_predicted_machine': int(fp + tp),
        'n_test': int(len(y_true)),
    }


def transformer_reference(tag):
    """Deployed-checkpoint results only. Reading run_info.json from the saved
    model directory guarantees the number belongs to a checkpoint that exists,
    which is exactly the grid-maximum bug this project already shipped twice."""
    out = {}
    for mk in ('BERT', 'DeBERTa'):
        p = FINAL / 'experiments' / 'paper_scale' / 'models' / f'{tag}_{mk}' / 'run_info.json'
        if p.exists():
            r = json.load(open(p))
            out[mk] = {
                'key': r['key'],
                'test_f1_weighted': r['test']['f1'],
                'test_accuracy': r['test']['accuracy'],
                'error_rate': round(1.0 - r['test']['accuracy'], 4),
                'source': str(p.relative_to(FINAL)),
                'note': 'deployed checkpoint, not grid maximum',
            }
    return out


def main():
    report = {'seed': SEED, 'classifier': 'LogisticRegression(max_iter=1000)',
              'content_arm_preprocessing': {
                  'lowercase': True, 'strip_non_alpha': True,
                  'stopword_removal': False, 'lemmatisation': False,
                  'vectoriser': 'CountVectorizer() defaults, raw counts',
                  'why_recorded': 'iccit/sections/02_methods.tex claimed stopword removal '
                                  'and lemmatisation for this arm. Those steps run in '
                                  'experiments/paper_scale/classical_full.py, not here.'},
              'arms_are_not_disjoint_on': 'document length; see the length_controlled block',
              'datasets': {}}
    preds = {}

    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        itr = df.index.get_indexer(sp['train'])
        ite = df.index.get_indexer(sp['test'])
        y = df['label'].values
        ytr, yte = y[itr], y[ite]
        texts = df['text'].values

        print(f'[{tag}] {name}: n_train={len(itr)} n_test={len(ite)}', flush=True)

        arms = {}

        def fit_arm(label, Xtr, Xte, n_features, note=None):
            """One logistic regression, scored and with its predictions kept.

            n_iter_ is recorded because these arms run at up to 54k features and a
            baseline that quietly stopped at max_iter is not a baseline."""
            clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
            pred = clf.predict(Xte)
            rec = score(yte, pred)
            rec['n_features'] = int(n_features)
            rec['n_iter'] = int(np.max(clf.n_iter_))
            rec['converged'] = bool(np.max(clf.n_iter_) < 1000)
            if note:
                rec['note'] = note
            arms[label] = rec
            preds[f'{tag}|{label}'] = pred.astype(np.int8)
            flag = '' if rec['converged'] else '  [DID NOT CONVERGE]'
            print(f'[{tag}] {label:24s} wF1={rec["weighted_f1"]:.4f} '
                  f'mF1={rec["macro_f1"]:.4f} err={rec["error_rate"]:.4f}{flag}', flush=True)
            return rec

        # ---- surface-only, and the same arm with the length channel closed ----
        print(f'[{tag}] building surface features...', flush=True)
        X = np.array([surface_features(t) for t in texts], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = X[itr].mean(0), X[itr].std(0)
        sd[sd == 0] = 1.0
        Xz = (X - mu) / sd
        fit_arm('surface_only', Xz[itr], Xz[ite], X.shape[1])

        keep = [i for i in range(X.shape[1]) if i not in LENGTH_IDX]
        fit_arm('surface_only_nolength', Xz[itr][:, keep], Xz[ite][:, keep], len(keep),
                note='document-size features dropped: %s' % LENGTH_IDX)

        # How much does document size separate the corpus on its own? Reported so
        # the length channel is measured rather than inferred from what removing
        # it costs the other arms.
        fit_arm('length_only', Xz[itr][:, PURE_LENGTH_IDX], Xz[ite][:, PURE_LENGTH_IDX],
                len(PURE_LENGTH_IDX),
                note='characters, words and sentences only, nothing else')

        # ---- content-only, raw counts and length-normalised ----
        print(f'[{tag}] normalising for content arm...', flush=True)
        norm = np.array([content_normalise(t) for t in texts], dtype=object)
        vec = CountVectorizer()
        Ctr = vec.fit_transform(norm[itr])
        Cte = vec.transform(norm[ite])
        fit_arm('content_only', Ctr, Cte, Ctr.shape[1],
                note='no stopword removal, no lemmatisation; raw counts, so row sums '
                     'carry document length')
        fit_arm('content_only_l1norm', normalize(Ctr, norm='l1'), normalize(Cte, norm='l1'),
                Ctr.shape[1], note='rows L1-normalised, so document length is removed. '
                                   'Confounded: L1 rows sum to 1, which shrinks every '
                                   'feature by roughly the mean document length, so at a '
                                   'fixed C the penalty is effectively far stronger than '
                                   'in the raw-count arm. Read the _scaled arm instead.')
        # Same normalisation, feature scale restored to the raw-count arm's, so C=1.0
        # penalises the two arms comparably and the only difference left is length.
        scale = float(Ctr.sum(axis=1).mean())
        fit_arm('content_only_l1norm_scaled',
                normalize(Ctr, norm='l1') * scale, normalize(Cte, norm='l1') * scale,
                Ctr.shape[1],
                note='rows L1-normalised then rescaled by the mean training document '
                     'length (%.1f tokens), isolating length removal from the change '
                     'in regularisation scale' % scale)

        preds[f'{tag}|y_true'] = np.asarray(yte, dtype=np.int8)

        report['datasets'][tag] = {
            'name': name,
            'n_train': int(len(itr)), 'n_test': int(len(ite)),
            'class_balance_train': {
                'human': round(float(np.mean(ytr == 0)), 4),
                'machine': round(float(np.mean(ytr == 1)), 4)},
            'surface_only': arms['surface_only'],
            'content_only': arms['content_only'],
            'length_controlled': {
                'length_only': arms['length_only'],
                'surface_only_nolength': arms['surface_only_nolength'],
                'content_only_l1norm': arms['content_only_l1norm'],
                'content_only_l1norm_scaled': arms['content_only_l1norm_scaled'],
            },
            'full_transformer_reference': transformer_reference(tag),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(report, fh, indent=2)
    np.savez_compressed(OUT_PRED, **preds)
    print('\nwritten to:', OUT)
    print('written to:', OUT_PRED)
    return report


if __name__ == '__main__':
    main()
