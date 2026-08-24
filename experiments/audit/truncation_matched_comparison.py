"""Truncation-matched classical-vs-transformer comparison.

The ICCIT paper concludes that transformers add nothing measurable on DAIGT V2,
comparing the best classical model against the best transformer. That comparison
is confounded as it stands. The classical models read the WHOLE document
(experiments/paper_scale/classical_full.py and full_model_evaluation.py apply no truncation),
while the transformers read 128 tokens, which is about 31% of a median DAIGT V2
essay -- 99.7% of DAIGT V2 documents exceed the window (NUMBERS_SSOT.md Section 9).
So the classical model is reading roughly three times as much text as the model it
is said to match.

This script removes the confound the cheap way, by handicapping the classical
models identically rather than by retraining transformers at a longer window. For
each deployed checkpoint we take the exact character span its own tokenizer kept
under truncation=True, max_length=128, refit the full six-cell classical grid on
that span, and re-run the paired comparison against that checkpoint's stored
predictions.

Character spans come from the fast tokenizer's offset mapping, not from decoding
token ids back to text, so no round-trip artefact is introduced.

HC3 is run identically and acts as the control: its median document already fits
the window, so truncation should barely move it. If HC3 moved as much as DAIGT V2
did, the manipulation would be doing something other than what it claims.

Output: audit/truncation_matched_comparison.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer

FINAL = Path(__file__).resolve().parents[2]
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
MODELS = FINAL / 'experiments' / 'paper_scale' / 'models'
AUDIT = FINAL / 'experiments' / 'audit'
OUT = AUDIT / 'truncation_matched_comparison.json'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
TRANSFORMERS = ('BERT', 'DeBERTa')
MAX_LEN = 128          # identical to paper_scale/run_full_scale.py
SEED = 42
N_BOOT = 10000

# byte-for-byte the pipeline full_model_evaluation.py uses, so truncated and
# full-text classical numbers differ only in the input span
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = re.sub(r'[^a-z\s]', ' ', str(text).lower())
    return ' '.join(lemmatizer.lemmatize(t) for t in word_tokenize(text)
                    if t not in stop_words and len(t) > 1)


CLASSICAL = [
    ('Naive Bayes', lambda: MultinomialNB()),
    ('Logistic Regression', lambda: LogisticRegression(max_iter=1000, random_state=SEED)),
    ('Support Vector Machine', lambda: LinearSVC(random_state=SEED, max_iter=20000)),
]
REPS = [('BoW', CountVectorizer), ('TF-IDF', TfidfVectorizer)]


def truncate_to_window(texts, tok):
    """The exact character span the checkpoint's tokenizer kept at max_length=128."""
    out, kept = [], []
    B = 512
    for i in range(0, len(texts), B):
        batch = [str(t) for t in texts[i:i + B]]
        enc = tok(batch, truncation=True, max_length=MAX_LEN,
                  return_offsets_mapping=True, add_special_tokens=True)
        for txt, offs in zip(batch, enc['offset_mapping']):
            ends = [e for s, e in offs if e > s]
            cut = max(ends) if ends else 0
            out.append(txt[:cut])
            kept.append(cut / max(len(txt), 1))
    return np.array(out, dtype=object), float(np.mean(kept))


def metrics(y, pred):
    _, _, wf1, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
    acc = accuracy_score(y, pred)
    return {'weighted_f1': round(float(wf1), 4), 'accuracy': round(float(acc), 4),
            'error_rate': round(float(1 - acc), 4)}


def paired(ca, cb, la, lb, rng):
    b = int(np.sum(ca & ~cb)); c = int(np.sum(~ca & cb))
    p = float(binomtest(b, b + c, 0.5).pvalue) if (b + c) > 0 else 1.0
    n = len(ca)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = (1 - ca[idx].mean(axis=1)) - (1 - cb[idx].mean(axis=1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {'a': la, 'b': lb, 'mcnemar_b': b, 'mcnemar_c': c,
            'mcnemar_exact_p': round(p, 8),
            'err_diff_a_minus_b': round(float((1 - ca.mean()) - (1 - cb.mean())), 6),
            'ci95_lo': round(float(lo), 6), 'ci95_hi': round(float(hi), 6),
            'ci_excludes_zero': bool(lo > 0 or hi < 0)}


def main():
    scores = np.load(AUDIT / 'full_model_scores.npz', allow_pickle=True)
    rng = np.random.default_rng(SEED)
    report = {'max_len': MAX_LEN, 'seed': SEED,
              'method': 'classical grid refit on the exact character span each checkpoint '
                        'tokenizer kept at max_length=128 (offset mapping, no decode round-trip)',
              'datasets': {}}

    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        ytr = df.loc[sp['train'], 'label'].values
        yte = df.loc[sp['test'], 'label'].values
        raw = df['text'].values
        entry = {'name': name, 'n_test': int(len(yte)), 'windows': {}}

        for mk in TRANSFORMERS:
            tok = AutoTokenizer.from_pretrained(str(MODELS / f'{tag}_{mk}'), use_fast=True)
            cache = WORK / f'trunc_{tag}_{mk}.parquet'
            if cache.exists():
                cdf = pd.read_parquet(cache)
                clean, frac = cdf['clean'], float(cdf['frac'].iloc[0])
                print(f'[{tag}/{mk}] cached truncated corpus', flush=True)
            else:
                print(f'[{tag}/{mk}] truncating to the {MAX_LEN}-token window...', flush=True)
                trunc, frac = truncate_to_window(raw, tok)
                print(f'[{tag}/{mk}] mean characters kept = {frac:.3f}; preprocessing...',
                      flush=True)
                clean = pd.Series(trunc, index=df.index).apply(preprocess)
                pd.DataFrame({'clean': clean, 'frac': frac}).to_parquet(cache)
            Xtr_text, Xte_text = clean.loc[sp['train']], clean.loc[sp['test']]

            tf_scores = scores[f'{tag}|{mk}'].astype(float)
            tf_correct = ((tf_scores > 0.5).astype(int) == yte)

            cells, corr = {}, {}
            for mname, build in CLASSICAL:
                for rep, Vec in REPS:
                    vec = Vec()
                    clf = build().fit(vec.fit_transform(Xtr_text), ytr)
                    pred = clf.predict(vec.transform(Xte_text))
                    key = f'{mname} ({rep})'
                    cells[key] = metrics(yte, pred)
                    corr[key] = (pred == yte)
                    print(f'   {key:34s} F1={cells[key]["weighted_f1"]:.4f} '
                          f'err={cells[key]["error_rate"]:.4f}', flush=True)

            best = min(cells, key=lambda k: cells[k]['error_rate'])
            entry['windows'][mk] = {
                'mean_fraction_of_characters_kept': round(frac, 4),
                'transformer_error_rate': round(float(1 - tf_correct.mean()), 6),
                'classical_truncated': cells,
                'best_classical_truncated': best,
                'paired_best_classical_vs_transformer': paired(
                    corr[best], tf_correct, f'{best} [128-token window]', mk, rng),
            }
        report['datasets'][tag] = entry

    json.dump(report, open(OUT, 'w'), indent=2)
    print('\nwritten to:', OUT)


if __name__ == '__main__':
    main()
