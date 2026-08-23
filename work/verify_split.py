import gc, json, re, sys, warnings
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
warnings.filterwarnings('ignore')

DATA_DIR = '/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project '
OUT_DIR  = '/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/work'
SAMPLE_PER_CLASS = 3000

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]
    return ' '.join(tokens)

def balance(df, n, seed=42):
    parts = []
    for value in sorted(df['label'].unique()):
        subset = df[df['label'] == value]
        parts.append(subset.sample(n=min(n, len(subset)), random_state=seed))
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_d1():
    daigt_raw = pd.read_csv(DATA_DIR + '/daigt.csv')
    df1 = daigt_raw[['text', 'label']].dropna()
    df1['label'] = df1['label'].astype(int)
    del daigt_raw; gc.collect()
    return balance(df1, SAMPLE_PER_CLASS)

def load_d2():
    hc3_raw = pd.read_json(DATA_DIR + '/hc3.jsonl', lines=True)
    human = hc3_raw[['human_answers']].explode('human_answers').rename(columns={'human_answers': 'text'})
    human['label'] = 0
    chatgpt = hc3_raw[['chatgpt_answers']].explode('chatgpt_answers').rename(columns={'chatgpt_answers': 'text'})
    chatgpt['label'] = 1
    df2 = pd.concat([human, chatgpt], ignore_index=True).dropna()
    df2['text'] = df2['text'].astype(str)
    del hc3_raw, human, chatgpt; gc.collect()
    return balance(df2, SAMPLE_PER_CLASS)

# midterm recorded metrics, from executed cell outputs of nlp_mid_project_group_02.ipynb
MIDTERM = {
 ('D1','Naive Bayes','BoW'):        (0.9567,0.9580,0.9567,0.9566),
 ('D1','Naive Bayes','TF-IDF'):     (0.9508,0.9515,0.9508,0.9508),
 ('D1','Logistic Regression','BoW'):(0.9825,0.9826,0.9825,0.9825),
 ('D1','Logistic Regression','TF-IDF'):(0.9758,0.9761,0.9758,0.9758),
 ('D1','SVM','BoW'):                (0.9775,0.9776,0.9775,0.9775),
 ('D1','SVM','TF-IDF'):             (0.9875,0.9877,0.9875,0.9875),
 ('D2','Naive Bayes','BoW'):        (0.8583,0.8587,0.8583,0.8583),
 ('D2','Naive Bayes','TF-IDF'):     (0.8275,0.8387,0.8275,0.8261),
 ('D2','Logistic Regression','BoW'):(0.9333,0.9346,0.9333,0.9333),
 ('D2','Logistic Regression','TF-IDF'):(0.8925,0.8926,0.8925,0.8925),
 ('D2','SVM','BoW'):                (0.9117,0.9132,0.9117,0.9116),
 ('D2','SVM','TF-IDF'):             (0.9083,0.9084,0.9083,0.9083),
}

def metrics(yt, yp):
    a = accuracy_score(yt, yp)
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
    return (a, p, r, f)

report = {}
all_ok = True

for tag, loader in (('D1', load_d1), ('D2', load_d2)):
    print(f'\n{"="*66}\n{tag}\n{"="*66}', flush=True)
    df = loader()
    print(f'balanced rows={len(df)}  label counts={df["label"].value_counts().to_dict()}', flush=True)
    print('preprocessing (this takes a minute)...', flush=True)
    df['clean_text'] = df['text'].apply(preprocess)

    # (A) the literal midterm split, on clean_text
    Xtr_lit, Xte_lit, ytr_lit, yte_lit = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    # (B) the index-based split used by the final notebook
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        df.index.values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'].values)

    same_tr = np.array_equal(np.asarray(Xtr_lit.index), idx_tr)
    same_te = np.array_equal(np.asarray(Xte_lit.index), idx_te)
    print(f'SPLIT ASSERTION  train identical={same_tr}  test identical={same_te}  '
          f'(order-sensitive)', flush=True)
    assert same_tr and same_te, f'{tag}: index split does NOT reproduce the midterm split'
    print(f'sizes  train={len(idx_tr)}  test={len(idx_te)}  '
          f'test per-class={pd.Series(y_te).value_counts().to_dict()}', flush=True)

    # (C) reproduce all six midterm model results on this split
    for repname, Vec in (('BoW', CountVectorizer), ('TF-IDF', TfidfVectorizer)):
        vec = Vec()
        Xtr = vec.fit_transform(Xtr_lit)
        Xte = vec.transform(Xte_lit)
        for mname, mk in (('Naive Bayes', MultinomialNB),
                          ('Logistic Regression', lambda: LogisticRegression(max_iter=1000)),
                          ('SVM', LinearSVC)):
            clf = mk() if not isinstance(mk, type) else mk()
            clf.fit(Xtr, ytr_lit)
            got = metrics(yte_lit, clf.predict(Xte))
            exp = MIDTERM[(tag, mname, repname)]
            ok = all(abs(round(g,4)-e) <= 1e-4 for g, e in zip(got, exp))
            all_ok &= ok
            flag = 'MATCH  ' if ok else 'MISMATCH'
            print(f'  {flag} {mname:<20} {repname:<7} '
                  f'got={tuple(round(g,4) for g in got)} expected={exp}', flush=True)
            report[f'{tag}|{mname}|{repname}'] = {
                'got': [round(g,4) for g in got], 'expected': list(exp), 'match': bool(ok)}

    # (D) persist the split + raw text for the transformer notebook
    sub_tr, sub_val = train_test_split(idx_tr, test_size=0.1, random_state=42,
                                       stratify=df.loc[idx_tr,'label'].values)
    print(f'validation carve  train={len(sub_tr)}  val={len(sub_val)}  test={len(idx_te)}', flush=True)
    np.savez(f'{OUT_DIR}/split_{tag}.npz', train=sub_tr, val=sub_val, test=idx_te,
             full_train=idx_tr)
    df[['text','label']].to_parquet(f'{OUT_DIR}/data_{tag}.parquet', index=True)
    del df, Xtr_lit, Xte_lit, Xtr, Xte, vec; gc.collect()

json.dump(report, open(f'{OUT_DIR}/midterm_reproduction.json','w'), indent=2)
print(f'\n{"="*66}')
print('ALL 24 MIDTERM NUMBERS REPRODUCED' if all_ok else 'REPRODUCTION FAILED -- SEE MISMATCHES ABOVE')
print(f'{"="*66}')
sys.exit(0 if all_ok else 1)
