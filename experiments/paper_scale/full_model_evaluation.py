"""Complete per-model evaluation for every model family, both datasets.

Addresses three reporting gaps raised in supervisor review:
  1. only one model family was reported in detail
  2. the model x representation grid was never stated at full scale
  3. no ROC/AUC or per-model confusion matrices existed

Runs the full classical grid (3 models x {BoW, TF-IDF}), which at full scale had
only ever been run for the one winning representation per model, and pairs it
with the deployed transformer checkpoints read from their saved probability
arrays. Emits per-model scores so ROC, AUC and confusion matrices can all be
drawn from one committed source.

LinearSVC exposes no predict_proba; its decision_function is used as the ranking
score, which is what ROC requires. This is recorded per model in the output so
no reader mistakes it for a calibrated probability.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score,
                             roc_curve)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

FINAL = Path(__file__).resolve().parents[2]
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
PROBS = FINAL / 'experiments' / 'paper_scale' / 'probs'
MODELS = FINAL / 'experiments' / 'paper_scale' / 'models'
OUT_JSON = FINAL / 'experiments' / 'audit' / 'full_model_evaluation.json'
OUT_NPZ = FINAL / 'experiments' / 'audit' / 'full_model_scores.npz'

DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
SEED = 42

# identical preprocessing to paper_scale/classical_full.py, kept byte-for-byte so
# these numbers stay comparable with the ones already reported
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


CLASSICAL = [
    ('Naive Bayes', MultinomialNB, 'proba'),
    ('Logistic Regression', lambda: LogisticRegression(max_iter=1000, random_state=SEED), 'proba'),
    # max_iter raised from the sklearn default of 1000: at the default LinearSVC
    # fails to converge on HC3 and a non-converged baseline is not a fair one.
    ('Support Vector Machine', lambda: LinearSVC(random_state=SEED, max_iter=20000), 'decision'),
]
REPS = [('BoW', CountVectorizer), ('TF-IDF', TfidfVectorizer)]

# deployed checkpoints only, never a grid maximum
DEPLOYED = {
    ('D1', 'BERT'): 'full_D1_BERT_lr3e-05_bs32_wd0.1_s42',
    ('D1', 'DeBERTa'): 'full_D1_DeBERTa_lr3e-05_bs16_wd0.01_s42',
    ('D2', 'BERT'): 'full_D2_BERT_lr2e-05_bs16_wd0.1_s42',
    ('D2', 'DeBERTa'): 'full_D2_DeBERTa_lr3e-05_bs16_wd0.1_s42',
}


def metrics(y, pred, scores):
    _, _, wf1, _ = precision_recall_fscore_support(y, pred, average='weighted', zero_division=0)
    _, _, mf1, _ = precision_recall_fscore_support(y, pred, average='macro', zero_division=0)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, pred)
    return {
        'accuracy': round(float(acc), 4),
        'error_rate': round(float(1 - acc), 4),
        'weighted_f1': round(float(wf1), 4),
        'macro_f1': round(float(mf1), 4),
        # six decimals: at this ceiling several models round to 1.0000 at four,
        # which would hide real ordering between them
        'roc_auc': round(float(roc_auc_score(y, scores)), 6),
        'confusion_tn_fp_fn_tp': [int(tn), int(fp), int(fn), int(tp)],
        'fpr_human_called_machine': round(float(fp / max(tn + fp, 1)), 4),
    }


def main():
    report = {'seed': SEED, 'note': 'all baselines are deployed checkpoints, never grid maxima',
              'datasets': {}}
    store = {}

    for tag, name in DATASETS.items():
        df = pd.read_parquet(WORK / f'data_{tag}.parquet')
        sp = np.load(WORK / f'split_{tag}.npz')
        ytr = df.loc[sp['train'], 'label'].values
        yte = df.loc[sp['test'], 'label'].values
        # preprocessing dominates runtime, so cache it: identical output, and
        # re-runs after a model-side change become near-instant
        cache = WORK / f'clean_{tag}.parquet'
        if cache.exists():
            print(f'[{tag}] {name}: loading cached preprocessing', flush=True)
            clean = pd.read_parquet(cache)['clean']
        else:
            print(f'[{tag}] {name}: preprocessing {len(df)} docs (slow, once)...', flush=True)
            clean = df['text'].apply(preprocess)
            pd.DataFrame({'clean': clean}).to_parquet(cache)
        Xtr_text, Xte_text = clean.loc[sp['train']], clean.loc[sp['test']]

        entries = {}
        for mname, build, score_kind in CLASSICAL:
            for rep, Vec in REPS:
                vec = Vec()
                Xtr = vec.fit_transform(Xtr_text)
                Xte = vec.transform(Xte_text)
                clf = build()
                clf.fit(Xtr, ytr)
                pred = clf.predict(Xte)
                if score_kind == 'proba':
                    sc = clf.predict_proba(Xte)[:, 1]
                else:
                    sc = clf.decision_function(Xte)
                m = metrics(yte, pred, sc)
                m['representation'] = rep
                m['score_type'] = 'predict_proba' if score_kind == 'proba' else 'decision_function'
                m['n_features'] = int(Xtr.shape[1])
                key = f'{mname} ({rep})'
                entries[key] = m
                store[f'{tag}|{key}'] = sc
                print(f'  {key:34s} F1={m["weighted_f1"]:.4f} AUC={m["roc_auc"]:.4f}', flush=True)

        for mk in ('BERT', 'DeBERTa'):
            npz = PROBS / (DEPLOYED[(tag, mk)] + '.npz')
            if not npz.exists():
                print(f'  [missing] {npz.name}')
                continue
            z = np.load(npz)
            p = z['test_probs']
            yy = z['test_labels']
            sc = p[:, 1]
            m = metrics(yy, p.argmax(1), sc)
            m['representation'] = 'raw text (subword)'
            m['score_type'] = 'softmax'
            m['checkpoint_key'] = DEPLOYED[(tag, mk)]
            entries[mk] = m
            store[f'{tag}|{mk}'] = sc
            print(f'  {mk:34s} F1={m["weighted_f1"]:.4f} AUC={m["roc_auc"]:.4f}', flush=True)

        store[f'{tag}|y_true'] = yte
        report['datasets'][tag] = {'name': name, 'n_test': int(len(yte)), 'models': entries}
        del clean, Xtr_text, Xte_text

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT_JSON, 'w'), indent=2)
    np.savez_compressed(OUT_NPZ, **store)
    print('\nwritten to:', OUT_JSON, 'and', OUT_NPZ)


if __name__ == '__main__':
    main()
