"""Surface/content separability decomposition.

Measures how much of a detection benchmark's human-vs-machine separability is
carried by SURFACE FORM (orthography, punctuation, casing, length) versus
CONTENT (lexical choice, with surface normalised away).

Three arms per dataset, all on the same fixed splits:

  surface-only  hand-built orthographic features, NO lexical content at all
  content-only  bag-of-words over surface-normalised text, the same
                lower()/[^a-z\\s] pipeline paper_scale/classical_full.py uses,
                so casing, punctuation and non-ASCII cannot leak in
  full          fine-tuned transformer on raw text (read from existing results,
                not refit here)

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
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

FINAL = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK = FINAL / 'paper_scale' / 'work'
RESULTS = FINAL / 'paper_scale' / 'results'
OUT = FINAL / 'audit' / 'surface_content_decomposition.json'

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


SURFACE_NORM = re.compile(r'[^a-z\s]')


def content_normalise(text):
    """Identical surface normalisation to paper_scale/classical_full.py, so the
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
        p = FINAL / 'paper_scale' / 'models' / f'{tag}_{mk}' / 'run_info.json'
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
              'datasets': {}}

    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        itr = df.index.get_indexer(sp['train'])
        ite = df.index.get_indexer(sp['test'])
        y = df['label'].values
        ytr, yte = y[itr], y[ite]
        texts = df['text'].values

        print(f'[{tag}] {name}: n_train={len(itr)} n_test={len(ite)}', flush=True)

        # ---- surface-only ------------------------------------------------
        print(f'[{tag}] building surface features...', flush=True)
        X = np.array([surface_features(t) for t in texts], dtype=float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        mu, sd = X[itr].mean(0), X[itr].std(0)
        sd[sd == 0] = 1.0
        Xz = (X - mu) / sd
        clf_s = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xz[itr], ytr)
        surf = score(yte, clf_s.predict(Xz[ite]))
        surf['n_features'] = int(X.shape[1])
        print(f'[{tag}] surface-only  wF1={surf["weighted_f1"]:.4f} '
              f'mF1={surf["macro_f1"]:.4f} err={surf["error_rate"]:.4f}', flush=True)

        # ---- content-only ------------------------------------------------
        print(f'[{tag}] normalising for content arm...', flush=True)
        norm = np.array([content_normalise(t) for t in texts], dtype=object)
        vec = CountVectorizer()
        Ctr = vec.fit_transform(norm[itr])
        Cte = vec.transform(norm[ite])
        clf_c = LogisticRegression(max_iter=1000, random_state=SEED).fit(Ctr, ytr)
        cont = score(yte, clf_c.predict(Cte))
        cont['n_features'] = int(Ctr.shape[1])
        print(f'[{tag}] content-only  wF1={cont["weighted_f1"]:.4f} '
              f'mF1={cont["macro_f1"]:.4f} err={cont["error_rate"]:.4f}', flush=True)

        report['datasets'][tag] = {
            'name': name,
            'n_train': int(len(itr)), 'n_test': int(len(ite)),
            'class_balance_train': {
                'human': round(float(np.mean(ytr == 0)), 4),
                'machine': round(float(np.mean(ytr == 1)), 4)},
            'surface_only': surf,
            'content_only': cont,
            'full_transformer_reference': transformer_reference(tag),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as fh:
        json.dump(report, fh, indent=2)
    print('\nwritten to:', OUT)
    return report


if __name__ == '__main__':
    main()
