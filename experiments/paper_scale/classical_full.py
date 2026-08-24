"""Refits Naive Bayes / Logistic Regression / SVM at full scale, on the SAME
duplicate-group-aware split used for the transformer runs, so Table 2 compares
full-scale models against full-scale models -- no scale mismatch."""
import re, json, pathlib, warnings
import pandas as pd, numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
warnings.filterwarnings('ignore')

WORK = pathlib.Path(__file__).resolve().parent / 'work'
OUT  = pathlib.Path(__file__).resolve().parent / 'results'
OUT.mkdir(parents=True, exist_ok=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return ' '.join(tokens)

def weighted_metrics(y, p):
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return round(a,4), round(pr,4), round(rc,4), round(f,4)

MODELS_ = (
    ('Naive Bayes', 'BoW', CountVectorizer, lambda: MultinomialNB()),
    ('Logistic Regression', 'BoW', CountVectorizer, lambda: LogisticRegression(max_iter=1000)),
    ('Support Vector Machine', 'TF-IDF', TfidfVectorizer, lambda: LinearSVC()),
)

results = {}
for tag in ('D1', 'D2'):
    df = pd.read_parquet(WORK / ('data_%s.parquet' % tag))
    sp = np.load(WORK / ('split_%s.npz' % tag))
    print('%s preprocessing %d docs (this takes a few minutes)...' % (tag, len(df)))
    clean = df['text'].apply(preprocess)
    ytr = df.loc[sp['train'], 'label'].values
    yte = df.loc[sp['test'], 'label'].values
    Xtr_text, Xte_text = clean.loc[sp['train']], clean.loc[sp['test']]

    for name, rep, Vec, build in MODELS_:
        vec = Vec()
        Xtr = vec.fit_transform(Xtr_text); Xte = vec.transform(Xte_text)
        clf = build(); clf.fit(Xtr, ytr)
        yhat = clf.predict(Xte)
        acc, pre, rec, f1 = weighted_metrics(yte, yhat)
        key = 'full_%s_%s_%s' % (tag, name.replace(' ', ''), rep)
        rec_out = {'key': key, 'dataset': tag, 'model': name, 'representation': rep,
                   'n_train': len(sp['train']), 'n_test': len(sp['test']),
                   'test': {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1},
                   'test_confusion': confusion_matrix(yte, yhat).tolist()}
        json.dump(rec_out, open(OUT / (key + '.json'), 'w'), indent=2)
        results[(tag, name, rep)] = rec_out
        print('  %-24s %-7s  Acc %.4f  F1 %.4f' % (name, rep, acc, f1))
    del clean, Xtr_text, Xte_text; import gc; gc.collect()

print('\nwritten to', OUT)
