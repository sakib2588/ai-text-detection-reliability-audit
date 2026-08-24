import json, os

CELLS = []
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def code(s):
    lines = s.strip("\n").split("\n")
    CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                  "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]})

md(r"""
# NLP Final Term Project — Group 02, Section B

**Domain:** AI-generated content detection (binary: human vs machine)

**Models:** BERT (`bert-base-uncased`), BERT variation (`microsoft/deberta-v3-base`), and a
validation-weighted soft-vote ensemble of the two.

**Datasets:** Dataset 1 = DAIGT V2 Train Dataset. Dataset 2 = HC3 (Human ChatGPT Comparison Corpus).

**Continuity with the midterm.** The classical models (Naive Bayes, Logistic Regression, Support
Vector Machine) were evaluated at midterm on a balanced 3,000-per-class sample of each dataset,
split 80/20 with `random_state=42` and stratification. This notebook rebuilds that identical split
and scores the transformers on the identical 1,200-row test set, so that the final combined table
compares models rather than comparing splits. Section 1 verifies this by re-deriving all twenty-four
midterm numbers from scratch and asserting they match.
""")

md(r"""
## Section 0 — Environment

Cache locations are pinned before any HuggingFace import. The root filesystem on this machine has
under 4 GB free, so the model cache and the training checkpoints are both directed to a volume with
room. `HF_HUB_DISABLE_SYMLINKS` is required because that volume is NTFS, which has no POSIX symlinks.
""")

code(r"""
import os
os.environ['HF_HOME'] = '/media/filwel/MLProject1/hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
print('HF_HOME =', os.environ['HF_HOME'])
""")

code(r"""
import gc, json, re, shutil, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import transformers
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed)
from datasets import Dataset
warnings.filterwarnings('ignore')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print('torch          ', torch.__version__)
print('transformers   ', transformers.__version__)
print('cuda available ', torch.cuda.is_available())
assert torch.cuda.is_available(), 'CUDA unavailable, halting'
print('device         ', torch.cuda.get_device_name(0))
print('capability     ', torch.cuda.get_device_capability(0))
print('bf16 supported ', torch.cuda.is_bf16_supported())
assert torch.cuda.is_bf16_supported(), 'bf16 unsupported, halting'
free_b, total_b = torch.cuda.mem_get_info()
print('VRAM free/total %.2f / %.2f GiB' % (free_b/1024**3, total_b/1024**3))
""")

code(r"""
PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')
FINAL_DIR   = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK_DIR    = FINAL_DIR / 'experiments' / 'midterm' / 'work'
RESULTS_DIR = FINAL_DIR / 'experiments' / 'midterm' / 'results'
PROBS_DIR   = FINAL_DIR / 'experiments' / 'midterm' / 'probs'
FIG_DIR     = FINAL_DIR / 'experiments' / 'midterm' / 'figures'
CKPT_DIR    = Path('/media/filwel/MLProject1/nlp_final_ckpt')
for d in (WORK_DIR, RESULTS_DIR, PROBS_DIR, FIG_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_CLASS = 3000
MAX_LEN          = 128
EPOCHS           = 5
WARMUP_RATIO     = 0.1
PATIENCE         = 2
SEED             = 42

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
GRID = [(2e-5,16,0.01), (3e-5,16,0.01), (2e-5,32,0.01), (3e-5,32,0.01),
        (2e-5,16,0.1),  (3e-5,16,0.1),  (2e-5,32,0.1),  (3e-5,32,0.1)]
DATASETS = ['D1', 'D2']
print('configurations per model per dataset:', len(GRID))
print('total fine-tuning runs:', len(GRID) * len(MODELS) * len(DATASETS))
""")

md(r"""
## Section 1 — Data, split reconstruction, and midterm verification

The midterm pipeline is reproduced exactly: load, balance to 3,000 per class with `seed=42`, apply
the midterm text cleaning, then split 80/20 stratified with `random_state=42`.

Two things then happen that the midterm did not do.

First, the split is recomputed on the dataframe **index** rather than on a text column. With the same
`random_state` and the same stratification array, scikit-learn's partition does not depend on what
`X` contains, so index splitting recovers the identical partition while leaving both the raw text
and the cleaned text reachable. The cell below asserts order-sensitive equality between the two.

Second, a validation set is carved out of the training portion only (4,320 train / 480 validation).
The 1,200-row test set is never touched during training or model selection. Early stopping reads
validation; the ensemble weight is tuned on validation; test is scored exactly once per run.

**Why the transformers receive raw text.** The midterm cleaning lowercases, deletes every
non-alphabetic character, removes stopwords, and lemmatises. Punctuation rhythm, casing, and
function-word usage are among the strongest signals for distinguishing machine-generated text, and
that pipeline destroys all three. Applying it to a transformer would discard the information the
model exists to exploit. The two model families therefore see different inputs, each appropriate to
its class, and this is stated in the report.
""")

code(r"""
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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

def load_D1():
    raw = pd.read_csv(PROJECT_DIR / 'daigt.csv')
    df = raw[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    del raw; gc.collect()
    return balance(df, SAMPLE_PER_CLASS)

def load_D2():
    raw = pd.read_json(PROJECT_DIR / 'hc3.jsonl', lines=True)
    human = raw[['human_answers']].explode('human_answers').rename(columns={'human_answers': 'text'})
    human['label'] = 0
    bot = raw[['chatgpt_answers']].explode('chatgpt_answers').rename(columns={'chatgpt_answers': 'text'})
    bot['label'] = 1
    df = pd.concat([human, bot], ignore_index=True).dropna()
    df['text'] = df['text'].astype(str)
    del raw, human, bot; gc.collect()
    return balance(df, SAMPLE_PER_CLASS)

LOADERS = {'D1': load_D1, 'D2': load_D2}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}
print('loaders ready')
""")

code(r"""
MIDTERM_RECORDED = {
 ('D1','Naive Bayes','BoW'):           (0.9567,0.9580,0.9567,0.9566),
 ('D1','Naive Bayes','TF-IDF'):        (0.9508,0.9515,0.9508,0.9508),
 ('D1','Logistic Regression','BoW'):   (0.9825,0.9826,0.9825,0.9825),
 ('D1','Logistic Regression','TF-IDF'):(0.9758,0.9761,0.9758,0.9758),
 ('D1','Support Vector Machine','BoW'):(0.9775,0.9776,0.9775,0.9775),
 ('D1','Support Vector Machine','TF-IDF'):(0.9875,0.9877,0.9875,0.9875),
 ('D2','Naive Bayes','BoW'):           (0.8583,0.8587,0.8583,0.8583),
 ('D2','Naive Bayes','TF-IDF'):        (0.8275,0.8387,0.8275,0.8261),
 ('D2','Logistic Regression','BoW'):   (0.9333,0.9346,0.9333,0.9333),
 ('D2','Logistic Regression','TF-IDF'):(0.8925,0.8926,0.8925,0.8925),
 ('D2','Support Vector Machine','BoW'):(0.9117,0.9132,0.9117,0.9116),
 ('D2','Support Vector Machine','TF-IDF'):(0.9083,0.9084,0.9083,0.9083),
}

def weighted_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return acc, pre, rec, f1
print('midterm reference table loaded:', len(MIDTERM_RECORDED), 'entries')
""")

code(r"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

DATA, SPLITS, MIDTERM_ROWS = {}, {}, {}
all_match = True

for tag in DATASETS:
    df = LOADERS[tag]()
    df['clean_text'] = df['text'].apply(preprocess)

    Xtr_lit, Xte_lit, ytr_lit, yte_lit = train_test_split(
        df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])
    idx_tr, idx_te, _, _ = train_test_split(
        df.index.values, df['label'].values, test_size=0.2, random_state=42, stratify=df['label'].values)

    assert np.array_equal(np.asarray(Xtr_lit.index), idx_tr), tag + ': train split diverged'
    assert np.array_equal(np.asarray(Xte_lit.index), idx_te), tag + ': test split diverged'

    sub_tr, sub_val = train_test_split(idx_tr, test_size=0.1, random_state=42,
                                       stratify=df.loc[idx_tr, 'label'].values)

    for rep, Vec in (('BoW', CountVectorizer), ('TF-IDF', TfidfVectorizer)):
        vec = Vec()
        Xtr, Xte = vec.fit_transform(Xtr_lit), vec.transform(Xte_lit)
        for name, build in (('Naive Bayes', lambda: MultinomialNB()),
                            ('Logistic Regression', lambda: LogisticRegression(max_iter=1000)),
                            ('Support Vector Machine', lambda: LinearSVC())):
            clf = build(); clf.fit(Xtr, ytr_lit)
            got = weighted_metrics(yte_lit, clf.predict(Xte))
            exp = MIDTERM_RECORDED[(tag, name, rep)]
            ok = all(abs(round(g, 4) - e) <= 1e-4 for g, e in zip(got, exp))
            all_match &= ok
            MIDTERM_ROWS[(tag, name, rep)] = tuple(round(g, 4) for g in got)
            print('%-8s %-24s %-7s %s  %s' % (tag, name, rep,
                  tuple(round(g,4) for g in got), 'MATCH' if ok else 'MISMATCH'))
        del vec, Xtr, Xte

    DATA[tag] = df[['text', 'label']].copy()
    SPLITS[tag] = {'train': sub_tr, 'val': sub_val, 'test': idx_te}
    print('%s split  train=%d  val=%d  test=%d  test per class=%s\n' % (
        tag, len(sub_tr), len(sub_val), len(idx_te),
        pd.Series(df.loc[idx_te, 'label'].values).value_counts().sort_index().to_dict()))
    del df; gc.collect()

assert all_match, 'midterm reproduction failed, the combined table would be invalid'
print('=' * 70)
print('VERIFIED: all 24 midterm numbers reproduced, split is identical to the midterm split')
print('=' * 70)
""")

json.dump(CELLS, open(Path(__file__).resolve().parent / 'cells_part1.json','w'))
print('part1 cells:', len(CELLS))
