"""Answers two questions raised by the 2026-08-26 hostile review.

Q3  Does the HC3 surface-content null survive a content arm built with the SAME
    preprocessing Table III's classical models use, namely stopword removal and
    lemmatisation? The paper's content arm is deliberately unfiltered, so a
    reviewer can ask whether the null is an artefact of that choice.

Q6  Does the surface arm's standing on HC3 survive Tian et al.'s cleaning kit,
    which removes the space-before-punctuation artefact their appendix identifies?

Both arms are logistic regressions on the fixed splits, so this is CPU work.

Q6 deliberately rebuilds the cleaned text rather than reusing work/data_D2c.parquet.
That file applies clean_hc3_whitespace AND length_match(unit='words') to the cleaned
arm only, which NUMBERS_SSOT.md records as a confound. Applying the cleaning alone
isolates what the review actually asks about.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

FINAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINAL / 'experiments' / 'audit'))
sys.path.insert(0, str(FINAL / 'experiments' / 'paper_scale'))
from surface_content_decomposition import (content_normalise,  # noqa: E402
                                           surface_features)
from text_perturbations import clean_hc3_whitespace  # noqa: E402

WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
OUT = FINAL / 'experiments' / 'audit' / 'review_question_answers.json'
SEED, BOOT = 42, 10000
DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}


def load(tag):
    df = pd.read_parquet(WORK / f'data_{tag}.parquet')
    sp = np.load(WORK / f'split_{tag}.npz')
    itr = df.index.get_indexer(sp['train'])
    ite = df.index.get_indexer(sp['test'])
    return df['text'].values, df['label'].values, itr, ite


def fit(Xtr, ytr, Xte):
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)
    return clf.predict(Xte), int(np.max(clf.n_iter_))


def score(y, p):
    acc = accuracy_score(y, p)
    _, _, f1, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return {'error_rate': round(float(1 - acc), 4), 'weighted_f1': round(float(f1), 4)}


def paired(y, pa, pb, rng):
    wa, wb = pa != y, pb != y
    b = int((~wa & wb).sum())
    c = int((wa & ~wb).sum())
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    idx = rng.integers(0, len(y), size=(BOOT, len(y)))
    boot = wa[idx].mean(1) - wb[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'b': b, 'c': c, 'mcnemar_p': round(float(p), 6),
            'err_diff_pp': round(float((wa.mean() - wb.mean()) * 100), 3),
            'ci95_pp': [round(float(lo * 100), 3), round(float(hi * 100), 3)],
            'excludes_zero': bool(lo > 0 or hi < 0)}


def surface_arm(texts, ytr, itr, ite, tag_note=''):
    X = np.nan_to_num(np.array([surface_features(t) for t in texts], dtype=float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[itr].mean(0), X[itr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    return fit(Xz[itr], ytr, Xz[ite])


def main():
    rng = np.random.default_rng(SEED)
    out = {'seed': SEED, 'bootstrap': BOOT, 'Q3': {}, 'Q6': {}}

    # ---------------- Q3 ----------------
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    for pkg in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    lem, stop = WordNetLemmatizer(), set(stopwords.words('english'))

    def filtered(text):
        """Exactly Table III's classical pipeline."""
        toks = word_tokenize(content_normalise(text))
        return ' '.join(lem.lemmatize(t) for t in toks if t not in stop and len(t) > 1)

    print('=== Q3  content arm WITH stopword removal and lemmatisation ===', flush=True)
    for tag, name in DATASETS.items():
        texts, y, itr, ite = load(tag)
        yte = y[ite]
        ps, _ = surface_arm(texts, y[itr], itr, ite)
        print(f'[{tag}] filtering {len(texts)} documents...', flush=True)
        norm = np.array([filtered(t) for t in texts], dtype=object)
        vec = CountVectorizer()
        Ctr, Cte = vec.fit_transform(norm[itr]), vec.transform(norm[ite])
        pc, n_iter = fit(Ctr, y[itr], Cte)
        rec = {'surface': score(yte, ps), 'content_filtered': score(yte, pc),
               'n_features': int(Ctr.shape[1]), 'converged': n_iter < 1000,
               'paired_surface_minus_content': paired(yte, ps, pc, rng)}
        out['Q3'][tag] = rec
        pr = rec['paired_surface_minus_content']
        print(f'  {name}: surface {rec["surface"]["error_rate"]*100:.2f}%  '
              f'content(filtered) {rec["content_filtered"]["error_rate"]*100:.2f}%  '
              f'diff {pr["err_diff_pp"]:+.2f} pp  p={pr["mcnemar_p"]:.4g}  '
              f'CI {pr["ci95_pp"]}', flush=True)

    # ---------------- Q6 ----------------
    print('\n=== Q6  HC3 with Tian et al. whitespace cleaning, no length matching ===', flush=True)
    texts, y, itr, ite = load('D2')
    yte = y[ite]
    cleaned = np.array([clean_hc3_whitespace(t) for t in texts], dtype=object)
    changed = int(sum(1 for a, b in zip(texts, cleaned) if a != b))
    ps_raw, _ = surface_arm(texts, y[itr], itr, ite)
    ps_cln, _ = surface_arm(cleaned, y[itr], itr, ite)
    norm = np.array([content_normalise(t) for t in cleaned], dtype=object)
    vec = CountVectorizer()
    pc_cln, _ = fit(vec.fit_transform(norm[itr]), y[itr], vec.transform(norm[ite]))
    out['Q6'] = {
        'documents_changed_by_cleaning': changed,
        'share_changed': round(changed / len(texts), 4),
        'surface_raw': score(yte, ps_raw),
        'surface_cleaned': score(yte, ps_cln),
        'content_cleaned': score(yte, pc_cln),
        'paired_surface_raw_vs_cleaned': paired(yte, ps_raw, ps_cln, rng),
        'paired_cleaned_surface_vs_cleaned_content': paired(yte, ps_cln, pc_cln, rng),
    }
    q6 = out['Q6']
    print(f'  cleaning changed {changed} of {len(texts)} documents ({changed/len(texts):.1%})')
    print(f'  surface raw     {q6["surface_raw"]["error_rate"]*100:.2f}%')
    print(f'  surface cleaned {q6["surface_cleaned"]["error_rate"]*100:.2f}%   '
          f'diff {q6["paired_surface_raw_vs_cleaned"]["err_diff_pp"]:+.2f} pp  '
          f'p={q6["paired_surface_raw_vs_cleaned"]["mcnemar_p"]:.4g}')
    print(f'  content cleaned {q6["content_cleaned"]["error_rate"]*100:.2f}%   '
          f'surface-vs-content after cleaning '
          f'{q6["paired_cleaned_surface_vs_cleaned_content"]["err_diff_pp"]:+.2f} pp  '
          f'p={q6["paired_cleaned_surface_vs_cleaned_content"]["mcnemar_p"]:.4g}')

    json.dump(out, open(OUT, 'w'), indent=1)
    print('\nwritten to', OUT.relative_to(FINAL))


if __name__ == '__main__':
    main()
