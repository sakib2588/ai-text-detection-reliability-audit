#!/usr/bin/env python3
"""
NLP Final Term Project -- Classical Baselines at Full Dataset Scale
Group 02, Section B -- THE MIDTERM MODELS ONLY

This trains and evaluates Naive Bayes, Logistic Regression, and Support Vector
Machine on the complete DAIGT V2 and HC3 corpora. It is the "midterm part" of the
project, re-run at full dataset scale instead of the small midterm sample.

It does NOT touch BERT, DeBERTa, or the ensemble -- those are the "final term part"
and are being run separately, on a different machine, with the transformer stack.
This script needs NO GPU, NO torch, NO transformers, and NO CUDA. Just plain
Python packages: pandas, numpy, scikit-learn, nltk, pyarrow.

USAGE
-----
1. pip install -r requirements.txt
2. python run_classical_full.py

That's it. Expected time: roughly 15-30 minutes total on a normal laptop CPU,
depending on core count -- not hours, and no GPU is used at any point.

WHY THIS IS A SEPARATE, SMALLER PACKAGE
-----------------------------------------
The full project package (with BERT/DeBERTa/the transformer sweep) requires torch,
CUDA, and several GB of downloads, and takes 14+ hours on a GPU. If you only need
the three classical models, none of that is necessary, and installing it anyway
is a common source of errors that have nothing to do with your actual task (for
example, torch's CUDA dependencies include Linux-only packages that fail to
install on Windows even when everything else is fine).

WHY THE SPLIT IS INCLUDED, NOT REBUILT FROM THE RAW DATASETS
---------------------------------------------------------------
data_D1.parquet / data_D2.parquet / split_D1.npz / split_D2.npz are included in this
package rather than regenerated from daigt.csv and hc3.jsonl. This guarantees your
results land on the EXACT SAME train/validation/test split as the BERT/DeBERTa run
happening elsewhere -- which is required for the two halves of the project (this
one and the transformer one) to combine into one valid comparison table. If each
side rebuilt its own split independently, even a tiny difference in library
versions could produce a different split and the two sets of results would not be
comparable.

The split itself is duplicate-aware: HC3 was found to contain 7.16% duplicate or
near-duplicate rows, which would leak into the test set under a plain random split.
This split guarantees no duplicate-content group crosses the train/validation/test
boundary. You do not need to do anything about this -- it is already built into the
files included here.
"""
import os, re, gc, json, time
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

NUM_WORKERS = max(1, min(8, int(os.environ.get('NLP_WORKERS', max(1, (os.cpu_count() or 2) - 1)))))
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}


def check_split_files():
    missing = [f for f in ('data_D1.parquet', 'data_D2.parquet', 'split_D1.npz', 'split_D2.npz')
               if not (HERE / f).exists()]
    if missing:
        print('ERROR: missing required file(s):', ', '.join(missing))
        print('These should have shipped alongside this script. Make sure you extracted')
        print('the whole package rather than just copying run_classical_full.py by itself.')
        raise SystemExit(1)


def load_split(tag):
    df = pd.read_parquet(HERE / f'data_{tag}.parquet')
    sp = np.load(HERE / f'split_{tag}.npz')
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}


def ensure_nltk():
    import nltk
    for pkg, path in (('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'),
                      ('stopwords', 'corpora/stopwords'), ('wordnet', 'corpora/wordnet'),
                      ('omw-1.4', 'corpora/omw-1.4')):
        try:
            nltk.data.find(path)
        except LookupError:
            print('downloading NLTK resource: %s' % pkg)
            nltk.download(pkg, quiet=True)


def _preprocess_chunk(texts):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    lem = WordNetLemmatizer()
    sw = set(stopwords.words('english'))
    out = []
    for t in texts:
        t = re.sub(r'[^a-z\s]', ' ', str(t).lower())
        toks = word_tokenize(t)
        out.append(' '.join(x for x in (lem.lemmatize(w) for w in toks if w not in sw and len(w) > 1)))
    return out


def preprocess_parallel(series, workers=NUM_WORKERS):
    texts = series.tolist()
    if workers <= 1 or len(texts) < 2000:
        return pd.Series(_preprocess_chunk(texts), index=series.index)
    chunk_size = (len(texts) + workers - 1) // workers
    chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    with Pool(processes=workers) as pool:
        results = pool.map(_preprocess_chunk, chunks)
    flat = [x for chunk in results for x in chunk]
    return pd.Series(flat, index=series.index)


def weighted_metrics(y, p):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a, 4), round(pr, 4), round(rc, 4), round(f, 4)


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def run_classical():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    specs = (
        ('Naive Bayes', 'BoW', CountVectorizer, lambda: MultinomialNB()),
        ('Logistic Regression', 'BoW', CountVectorizer, lambda: LogisticRegression(max_iter=1000)),
        ('Support Vector Machine', 'TF-IDF', TfidfVectorizer, lambda: LinearSVC()),
    )

    for tag in ('D1', 'D2'):
        df, splits = load_split(tag)
        print('\n%s (%s): %d train / %d val / %d test' % (
            tag, DATASET_NAMES[tag], len(splits['train']), len(splits['val']), len(splits['test'])))

        need_any = any(not (RESULTS_DIR / ('full_%s_%s_%s.json' % (
            tag, name.replace(' ', ''), rep))).exists() for name, rep, _, _ in specs)
        if not need_any:
            print('  all three models already done for %s, skipping preprocessing' % tag)
            continue

        print('  preprocessing %d documents across %d worker process(es) (a few minutes)...' % (
            len(df), NUM_WORKERS))
        t0 = time.time()
        clean = preprocess_parallel(df['text'])
        print('  preprocessing done in %.1f min' % ((time.time() - t0) / 60))

        ytr = df.loc[splits['train'], 'label'].values
        yte = df.loc[splits['test'], 'label'].values
        Xtr_text, Xte_text = clean.loc[splits['train']], clean.loc[splits['test']]

        for name, rep, Vec, build in specs:
            key = 'full_%s_%s_%s' % (tag, name.replace(' ', ''), rep)
            jpath = RESULTS_DIR / (key + '.json')
            if jpath.exists():
                print('  [skip] %s' % key)
                continue
            vec = Vec()
            Xtr = vec.fit_transform(Xtr_text)
            Xte = vec.transform(Xte_text)
            clf = build()
            clf.fit(Xtr, ytr)
            acc, pre, rec, f1 = weighted_metrics(yte, clf.predict(Xte))
            out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag],
                   'model': name, 'representation': rep, 'scale': 'full_balanced',
                   'n_train': len(splits['train']), 'n_test': len(splits['test']),
                   'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}}
            atomic_write_json(jpath, out)
            print('  %-24s %-7s  Acc %.4f  Prec %.4f  Rec %.4f  F1 %.4f' % (
                name, rep, acc, pre, rec, f1))
            del vec, Xtr, Xte
        del clean, Xtr_text, Xte_text
        gc.collect()


def build_table():
    BEST_REP = {'Naive Bayes': 'BoW', 'Logistic Regression': 'BoW', 'Support Vector Machine': 'TF-IDF'}
    SPEC_LABEL = {'Naive Bayes': 'Naïve Bayes', 'Logistic Regression': 'Logistic Regression',
                  'Support Vector Machine': 'Support Vector Machine'}
    rows = []
    for name, rep in BEST_REP.items():
        row = {'Model': SPEC_LABEL[name]}
        for tag in ('D1', 'D2'):
            key = 'full_%s_%s_%s' % (tag, name.replace(' ', ''), rep)
            jpath = RESULTS_DIR / (key + '.json')
            r = json.load(open(jpath)) if jpath.exists() else None
            for col, k in (('Acc', 'accuracy'), ('Prec', 'precision'), ('Rec', 'recall'), ('F1', 'f1')):
                row['%s %s' % (tag, col)] = ('%.4f' % r['test'][k]) if r else ''
        rows.append(row)
    t = pd.DataFrame(rows)
    out_csv = HERE / 'table2_classical_rows_full.csv'
    t.to_csv(out_csv, index=False)
    print('\nClassical-model rows for Table 2, full dataset scale:')
    print(t.to_string(index=False))
    print('\nWritten to:', out_csv)
    print('Send this CSV file and the results/ folder back -- these three rows plug')
    print('directly into the same Table 2 as the BERT/DeBERTa/ENSEMBLE rows.')


def main():
    check_split_files()
    ensure_nltk()
    print('CPU cores available: %d, using %d worker process(es) for preprocessing '
          '(override with NLP_WORKERS=<n>)' % (os.cpu_count() or 1, NUM_WORKERS))
    run_classical()
    build_table()
    print('\nDone. No GPU was used at any point.')


if __name__ == '__main__':
    main()
