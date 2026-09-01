"""Builds notebooks/nlp_final_project_group_02_annotated_master.ipynb.

One notebook holding all four submission notebooks end to end (preprocessing, BERT,
DeBERTa, ensemble), for viva and self-study rather than for grading. The code cells
are byte-identical to notebooks/submission/{01..04}*.ipynb -- copied, not retyped, so
nothing here can silently drift from the numbers those notebooks produced -- and every
code cell is followed by a "Line by line" markdown cell that walks through what each
notable line does. The printed code appendix (docs/nlp_final_submission_code.pdf) stays
the terse, uncommented, four-section document the supervisor asked for; this is the
opposite document, meant to be read, not printed for grading.

After writing the notebook this script executes it in place with nbclient, using the
same disk-cache the submission notebooks rely on, so it re-runs in well under a minute
and finishes with real output in every cell rather than empty ones.

Run:  python notebooks/builders/build_annotated_master_nb.py
"""
import subprocess
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

OUT_DIR = Path(__file__).resolve().parents[1]
OUT = OUT_DIR / 'nlp_final_project_group_02_annotated_master.ipynb'


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.rstrip('\n'))


CELLS = []

# ============================================================== front matter
CELLS += [
    md('''# NLP Final Term Project, Group 02 --- Annotated Master Notebook

**Detecting machine-generated text.** One file holding all four submission notebooks
end to end -- preprocessing, BERT, DeBERTa, ensemble -- for study and viva defence
rather than for grading.

**How this relates to the actual submission.** Every code cell below is copied
unchanged from `notebooks/submission/01_preprocessing.ipynb`,
`02_bert_best_config.ipynb`, `03_deberta_best_config.ipynb` and `04_ensemble.ipynb`.
Nothing was retyped, so nothing here can silently disagree with the numbers those four
notebooks produced and the report quotes. What is new is a **"Line by line"** markdown
cell after every code cell, walking through what each notable line does. The printed
code appendix (`docs/nlp_final_submission_code.pdf`) is the opposite document on
purpose -- four sections, no comments, hyperparameters as literal values, because that
is what the supervisor asked to be handed on paper. This notebook is for understanding
the same code, not for printing it.

**Why it re-executes in seconds, not hours.** Every training call below checks
`experiments/paper_scale/results/` and `probs/` first. All eight deployed
configurations (BERT and DeBERTa, both datasets) are already trained and saved, so
`train_one(...)` finds its two files and returns the stored record instead of
re-training. Delete a result file, or pass `force=True`, to force a real re-run of
that one configuration.

| Part | Notebook | What it does |
|---|---|---|
| 1 | Preprocessing | load, balance, group-aware split, both cleaning paths |
| 2 | BERT | fine-tune `bert-base-uncased` at its best configuration, per dataset |
| 3 | DeBERTa | fine-tune `microsoft/deberta-v3-base` (the BERT variant), per dataset |
| 4 | Ensemble | weighted soft vote of the two, validation-selected weight |
'''),
]

# ============================================================== PART 1: preprocessing
CELLS += [
    md('''---
# Part 1. Data preprocessing

Turns the two raw corpora into the one fixed, balanced, leakage-checked split every
later part reuses. Nothing in this part trains a model.

| | source | task |
|---|---|---|
| D1 | DAIGT V2 (`daigt.csv`) | student essays, human against machine |
| D2 | HC3 (`hc3.jsonl`) | question answering, human against ChatGPT |
'''),
    md('## 1.1 Environment and paths'),
    code('''import os

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Raw corpora live outside the repository; the repository holds the derived splits.
PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')

FINAL_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
if not FINAL_DIR.exists():
    p = Path.cwd().resolve()
    while p.name != 'Final' and p != p.parent:
        p = p.parent
    FINAL_DIR = p

PS_DIR = FINAL_DIR / 'experiments' / 'paper_scale'
WORK_DIR = PS_DIR / 'work'
RESULTS_DIR = PS_DIR / 'results'
PROBS_DIR = PS_DIR / 'probs'
MODELS_DIR = PS_DIR / 'models'
CKPT_DIR = Path('/media/filwel/MLProject1/nlp_paper_ckpt')

MAX_LEN = 128
EPOCHS = 5
WARMUP_RATIO = 0.1
PATIENCE = 2
SPLIT_SEED = 42
TRAIN_SEED = 42

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}

WORK_DIR.mkdir(parents=True, exist_ok=True)
print('work directory:', WORK_DIR)'''),
    md('''**Line by line.**

- `os.environ.setdefault(...)` (4 lines) -- set four environment variables *before*
  `transformers`/`torch` are imported anywhere, because those libraries read their
  cache locations and threading behaviour once, at import time. `HF_HOME` moves the
  multi-gigabyte HuggingFace model cache off the small root filesystem.
  `HF_HUB_DISABLE_SYMLINKS=1` is required because that cache volume is NTFS, which has
  no POSIX symlinks. `setdefault` (not `=`) means a value already exported by the shell
  wins, so the same cell works unmodified on a teammate's machine with a different
  cache path.
- `warnings.filterwarnings('ignore')` -- silences the routine
  `transformers`/`sklearn` deprecation chatter so real output is not buried in noise.
- `PROJECT_DIR` -- where the two raw corpus files (`daigt.csv`, `hc3.jsonl`) live.
  Kept outside the git repository on purpose; a corpus this size does not belong in
  version control.
- `FINAL_DIR` and the `while` loop under it -- resolves the repository root regardless
  of which directory the notebook happens to be launched from, by walking up parent
  directories until one is literally named `Final`. This is what lets the same
  notebook run correctly whether opened from the repo root or from
  `notebooks/submission/`.
- `PS_DIR`, `WORK_DIR`, `RESULTS_DIR`, `PROBS_DIR`, `MODELS_DIR`, `CKPT_DIR` -- the six
  paths every later part reads from or writes to. `CKPT_DIR` is on a different,
  larger volume, because Hugging Face `Trainer` checkpoints are large and are deleted
  again after each run (see `train_one` in Part 2).
- `MAX_LEN = 128` -- the token truncation length every transformer run uses; the cost
  of this choice is measured later in 1.6.
- `EPOCHS = 5`, `WARMUP_RATIO = 0.1`, `PATIENCE = 2` -- the training-loop constants:
  at most 5 passes over the data, a linear warm-up over the first 10% of steps, and
  early stopping if validation F1 has not improved for 2 evaluation rounds.
- `SPLIT_SEED = 42`, `TRAIN_SEED = 42` -- kept as two separate names on purpose, even
  though both equal 42 here, because they control different things: `SPLIT_SEED` fixes
  which rows land in which partition, `TRAIN_SEED` fixes model initialisation and batch
  order. A seed-robustness check (outside this notebook) reruns training only, at
  `TRAIN_SEED` 123 and 456, while `SPLIT_SEED` never changes.
- `MODELS` and `DATASET_NAMES` -- the two lookup dictionaries used everywhere below to
  turn a short key (`'BERT'`, `'D1'`) into the string a library call or a print
  statement needs.
'''),
    md('''## 1.2 Loading and class balancing

Both corpora arrive unbalanced. HC3 additionally arrives one row per *question*, with
a list of human answers and a list of ChatGPT answers in that row, so it is exploded
to one answer per row before anything else happens.'''),
    code('''def normalise(t):
    """Collapse runs of whitespace. The only text change applied before hashing,
    splitting, or transformer tokenisation."""
    return re.sub(r'\\s+', ' ', str(t)).strip()


def balance(df, seed=SPLIT_SEED):
    n = int(df['label'].value_counts().min())
    parts = [df[df['label'] == v].sample(n=n, random_state=seed)
             for v in sorted(df['label'].unique())]
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


def load_D1():
    """DAIGT V2: already one document per row, with a 0/1 label column."""
    raw = pd.read_csv(PROJECT_DIR / 'daigt.csv')
    df = raw[['text', 'label']].dropna()
    df['label'] = df['label'].astype(int)
    del raw
    gc.collect()
    return balance(df)


def load_D2():
    """HC3: one row per question holding two answer lists. Explode both, label,
    and concatenate."""
    raw = pd.read_json(PROJECT_DIR / 'hc3.jsonl', lines=True)
    human = raw[['human_answers']].explode('human_answers').rename(
        columns={'human_answers': 'text'})
    human['label'] = 0
    bot = raw[['chatgpt_answers']].explode('chatgpt_answers').rename(
        columns={'chatgpt_answers': 'text'})
    bot['label'] = 1
    df = pd.concat([human, bot], ignore_index=True).dropna()
    df['text'] = df['text'].astype(str)
    del raw, human, bot
    gc.collect()
    return balance(df)


LOADERS = {'D1': load_D1, 'D2': load_D2}'''),
    md('''**Line by line.**

- `normalise(t)` -- `re.sub(r'\\s+', ' ', str(t)).strip()` replaces every run of
  whitespace (spaces, tabs, newlines) with a single space, then trims the ends. This
  is the *only* text transformation applied before hashing or transformer
  tokenisation; punctuation, casing and every other character survive untouched,
  because those carry signal a transformer can use.
- `balance(df, seed)` -- `df['label'].value_counts().min()` finds the size of the
  smaller class. The list comprehension takes one `.sample(n=n, ...)` per class
  (human, machine), so both classes end up exactly that size. `pd.concat(parts)`
  stacks them back together, `.sample(frac=1, ...)` then shuffles the *whole* combined
  frame (sampling 100% of rows in random order is the standard pandas idiom for a full
  shuffle), and `.reset_index(drop=True)` renumbers rows 0..n-1 so the old,
  now-meaningless row numbers do not leak through.
- `load_D1()` -- `pd.read_csv` reads the whole file; `raw[['text','label']]` keeps
  only the two columns needed; `.dropna()` drops any row missing either one;
  `.astype(int)` guarantees the label is a clean 0/1 integer, not a float or string.
  `del raw; gc.collect()` frees the full raw dataframe's memory immediately rather
  than waiting for Python's garbage collector to get to it on its own schedule, which
  matters because the raw file is tens of megabytes and this machine has limited free
  RAM alongside the GPU job.
- `load_D2()` -- `pd.read_json(..., lines=True)` reads one JSON object per line
  (JSONL format). `.explode('human_answers')` turns a row holding a *list* of answers
  into one row per list element, duplicating every other column; the same happens for
  `chatgpt_answers`. `.rename(columns={...})` gives both resulting frames a common
  `text` column so they can be concatenated. `human['label'] = 0` and
  `bot['label'] = 1` assign the two classes explicitly, matching DAIGT's
  0-is-human/1-is-machine convention.
- `LOADERS = {'D1': load_D1, 'D2': load_D2}` -- a dispatch table so later code can
  write `LOADERS[tag]()` instead of an `if/elif` chain on the dataset tag.
'''),
    md('''## 1.3 Duplicate-group-aware split

HC3 contains a substantial number of near-identical answers. A plain stratified split
puts copies of the same answer on both sides of the train/test boundary, and the test
score then partly measures memorisation rather than generalisation. So rows are
grouped by the MD5 hash of their normalised, lowercased text, and `GroupShuffleSplit`
is used instead of a plain `train_test_split`, which keeps every row in a duplicate
group on the same side of every boundary.'''),
    code('''import hashlib

from sklearn.model_selection import GroupShuffleSplit


def content_hash(series):
    return series.map(lambda t: hashlib.md5(normalise(t).lower().encode()).hexdigest())


def group_split(df, seed=SPLIT_SEED):
    groups = df['hash'].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_full, te = next(gss1.split(df, df['label'], groups))
    sub = df.iloc[tr_full]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_rel, val_rel = next(gss2.split(sub, sub['label'], sub['hash'].values))
    idx_tr, idx_val = sub.index.values[tr_rel], sub.index.values[val_rel]
    idx_te = df.index.values[te]
    g_tr = set(df.loc[idx_tr, 'hash'])
    g_val = set(df.loc[idx_val, 'hash'])
    g_te = set(df.loc[idx_te, 'hash'])
    assert not (g_tr & g_val) and not (g_tr & g_te) and not (g_val & g_te), \\
        'GROUP LEAKAGE ACROSS SPLIT'
    return idx_tr, idx_val, idx_te


def build_or_load_splits(tag, rebuild=False):
    """Build the split once and cache it. Every later notebook loads this file."""
    data_p = WORK_DIR / f'data_{tag}.parquet'
    split_p = WORK_DIR / f'split_{tag}.npz'
    if not rebuild and data_p.exists() and split_p.exists():
        df = pd.read_parquet(data_p)
        sp = np.load(split_p)
        print(f'{tag}: loaded cached split from disk')
        return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}
    df = LOADERS[tag]()
    df['hash'] = content_hash(df['text'])
    n_groups = df['hash'].nunique()
    print(f'{tag}: balanced rows={len(df)}  unique content groups={n_groups}  '
          f'duplicate rows={len(df) - n_groups} '
          f'({100 * (len(df) - n_groups) / len(df):.2f} percent)')
    idx_tr, idx_val, idx_te = group_split(df)
    df[['text', 'label']].to_parquet(data_p, index=True)
    np.savez(split_p, train=idx_tr, val=idx_val, test=idx_te)
    print(f'{tag}: written {data_p.name} and {split_p.name}')
    return df[['text', 'label']], {'train': idx_tr, 'val': idx_val, 'test': idx_te}


DATA, SPLITS = {}, {}
for tag in ('D1', 'D2'):
    DATA[tag], SPLITS[tag] = build_or_load_splits(tag)'''),
    md('''**Line by line.**

- `content_hash(series)` -- for every row, lowercases and whitespace-normalises the
  text, encodes it to bytes, and MD5-hashes it. Two documents that differ only in
  casing or in how many spaces separate words hash identically, so they land in the
  same duplicate *group*, which is exactly the definition of "near-identical" this
  project uses.
- `group_split`, first block -- `groups = df['hash'].values` is the array
  `GroupShuffleSplit` uses to decide what may never be separated. `gss1` with
  `test_size=0.2` peels off 20% of the *groups* (not 20% of raw rows) as test;
  `next(gss1.split(...))` pulls the single split out of the generator
  `GroupShuffleSplit` returns. `sub = df.iloc[tr_full]` is everything left after test
  is removed.
- second block -- `gss2` with `test_size=0.1` splits the *remaining* 80% again, taking
  a tenth of it as validation. Combined with the first split this gives 72% train,
  8% validation, 20% test of the original whole -- not 80/10/10, because the second
  cut is a tenth of the *remainder*, not of the original total.
- the three `assert` lines -- after computing the three index sets, this recomputes
  their hash groups (`g_tr`, `g_val`, `g_te`) and checks all three pairwise
  intersections are empty. If any duplicate group ended up split across two
  partitions this line raises `AssertionError: GROUP LEAKAGE ACROSS SPLIT` and stops
  the notebook rather than silently reporting an inflated score.
- `build_or_load_splits` -- the cache-or-build pattern used throughout this project.
  If both output files already exist it loads and returns them in a few
  milliseconds; only on a genuinely fresh machine does it call `LOADERS[tag]()`,
  hash every row, run `group_split`, and write the two files
  (`data_{tag}.parquet` for the text/label columns, `split_{tag}.npz` for the three
  index arrays) so every later notebook in this project sees the identical
  partition.
- final loop -- runs the whole thing once for `'D1'` and once for `'D2'`, populating
  the two module-level dictionaries `DATA` and `SPLITS` that every following cell in
  this part, and every following part, reads from.
'''),
    md('''## 1.4 Split integrity

Three things are checked here, because each of them would quietly inflate the
reported scores if it went wrong unnoticed: partition sizes, class balance inside
every partition, and whether any normalised document appears in more than one
partition.'''),
    code('''rows = []
for tag in ('D1', 'D2'):
    df, sp = DATA[tag], SPLITS[tag]
    seen = {}
    for split in ('train', 'val', 'test'):
        sub = df.loc[sp[split]]
        counts = sub['label'].value_counts().to_dict()
        rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}', 'split': split,
                     'n': len(sub), 'n_human': counts.get(0, 0), 'n_ai': counts.get(1, 0),
                     'ai_fraction': round(counts.get(1, 0) / len(sub), 4)})
        seen[split] = set(content_hash(sub['text']))
    overlap = (len(seen['train'] & seen['test']), len(seen['train'] & seen['val']),
               len(seen['val'] & seen['test']))
    print(f'{tag} {DATASET_NAMES[tag]:9s} shared normalised documents '
          f'train-test={overlap[0]}  train-val={overlap[1]}  val-test={overlap[2]}')
    assert sum(overlap) == 0, f'{tag}: content leaked across partitions'

print()
print(pd.DataFrame(rows).to_string(index=False))'''),
    md('''**Line by line.**

- `df.loc[sp[split]]` -- selects exactly the rows whose index appears in that
  partition's saved index array.
- `sub['label'].value_counts().to_dict()` -- a `{0: count, 1: count}` dictionary,
  used to report how many human and how many machine documents landed in this
  partition. `.get(1, 0)` guards against a partition that happens to contain zero of
  one class (would raise `KeyError` without the default).
- `seen[split] = set(content_hash(sub['text']))` -- the set of this partition's
  content hashes, computed independently of the hashes stored during the split, as an
  end-to-end re-check rather than trusting the earlier assertion alone.
- `overlap = (...)` -- three set intersections: any hash appearing in both train and
  test, both train and validation, both validation and test. Each should be empty.
- the final `assert` -- fails loudly if it is not, which is the second, independent
  leakage check in this notebook (the first is inside `group_split` itself in 1.3).
- last two lines -- print a small table of `n`, `n_human`, `n_ai` and `ai_fraction`
  per partition per dataset, which is what confirms the balancing from 1.2 held
  through the split (every `ai_fraction` should sit very close to 0.5).
'''),
    md('''## 1.5 Classical cleaning path

The classical baselines (Naive Bayes, logistic regression, linear SVM over
bag-of-words and TF-IDF) need a heavier normalisation than the transformers do:
lowercase, strip everything that is not a letter, drop English stopwords and
one-character tokens, and lemmatise. This path is **not** applied before the
transformers -- casing, punctuation and function words carry signal a subword model
can use, and discarding them before fine-tuning would throw away part of what the
model is being asked to detect.'''),
    code('''import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

for pkg in ('punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4'):
    nltk.download(pkg, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_classical(text):
    text = re.sub(r'[^a-z\\s]', ' ', str(text).lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 1]
    return ' '.join(tokens)


demo = DATA['D2']['text'].iloc[0]
print('raw       ', repr(demo[:150]))
print()
print('normalised', repr(normalise(demo)[:150]))
print()
print('classical ', repr(preprocess_classical(demo)[:150]))'''),
    md('''**Line by line.**

- `for pkg in (...): nltk.download(pkg, quiet=True)` -- fetches the five NLTK data
  packages this cell needs (`punkt`/`punkt_tab` for sentence/word tokenisation,
  `stopwords` for the English stopword list, `wordnet`/`omw-1.4` for lemmatisation).
  Already-downloaded packages are skipped instantly, so this is cheap on every
  re-run.
- `lemmatizer = WordNetLemmatizer()` and `stop_words = set(stopwords.words('english'))`
  -- built once, outside the function, so they are not rebuilt on every call; using a
  `set` for `stop_words` makes the `in` check in the list comprehension below O(1)
  instead of O(n).
- `preprocess_classical(text)` -- `re.sub(r'[^a-z\\s]', ' ', str(text).lower())`
  lowercases first, then replaces every character that is *not* a lowercase letter or
  whitespace with a space, which strips digits, punctuation and any non-ASCII
  character in one pass. `word_tokenize` splits the cleaned string into word tokens.
  The list comprehension keeps a token only if it is not a stopword and longer than
  one character, and lemmatises what remains (`"running"` to `"run"`,
  `"better"` stays `"better"` without a POS tag, since no part-of-speech is passed
  here). `' '.join(tokens)` reassembles the surviving tokens into one string, which is
  what `CountVectorizer`/`TfidfVectorizer` consume downstream.
- the three `print` calls -- show the same HC3 document through three stages: raw
  text, `normalise()` (whitespace only), and `preprocess_classical()` (heavy
  cleaning), so the difference between the transformer path and the classical path is
  visible directly rather than only described in prose.
'''),
    md('''## 1.6 Transformer tokenisation

Each transformer uses its own subword vocabulary, so tokenisation happens per model:
WordPiece for BERT, SentencePiece for DeBERTa-v3. Sequences are truncated at 128
tokens and padded per batch rather than to a global maximum, which keeps the padded
fraction low. 128 is a project constraint carried over from the midterm, not a free
choice; the second cell below reports what it costs.'''),
    code('''from datasets import Dataset
from transformers import AutoTokenizer

_TOKCACHE = {}


def get_tokenizer(model_key):
    if model_key not in _TOKCACHE:
        _TOKCACHE[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return _TOKCACHE[model_key]


def tokenise_split(tag, model_key, split):
    """Tokenise one partition into the exact form the Trainer consumes."""
    tok = get_tokenizer(model_key)
    sub = DATA[tag].loc[SPLITS[tag][split]]
    ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                            'labels': [int(v) for v in sub['label']]})
    return ds.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=['text'])


example = tokenise_split('D1', 'BERT', 'val')
print(example)
print()
print('first row input_ids (truncated view):', example[0]['input_ids'][:24], '...')
print('decoded:', get_tokenizer('BERT').decode(example[0]['input_ids'][:24]))'''),
    md('''**Line by line.**

- `_TOKCACHE = {}` and `get_tokenizer(model_key)` -- loading a tokenizer from
  Hugging Face touches disk (or network, on a cold cache) and is not free, so it is
  loaded once per model key and reused; the `if model_key not in _TOKCACHE` guard is
  the whole caching logic.
- `tokenise_split(tag, model_key, split)` -- `sub = DATA[tag].loc[SPLITS[tag][split]]`
  selects one partition of one dataset. `Dataset.from_dict({...})` builds a Hugging
  Face `datasets.Dataset` (not a plain dict) because that is the object type
  `Trainer` expects; note `normalise(t)` is applied here too, so the transformer sees
  whitespace-collapsed but otherwise untouched text. `ds.map(lambda b: tok(...),
  batched=True, remove_columns=['text'])` runs the tokenizer over the whole dataset in
  batches (far faster than one call per row), and drops the now-redundant raw `text`
  column from the output, keeping only `input_ids`, `attention_mask` and `labels`.
- `example = tokenise_split('D1', 'BERT', 'val')` -- a concrete demonstration run,
  not something later cells depend on.
- the two `print` calls after it -- show the dataset's schema, then decode the first
  24 token ids of the first row back into text with the same tokenizer, so the
  input/output correspondence of tokenisation is visible rather than only asserted.
'''),
    code('''# Length diagnostic on a fixed 4000-row sample per dataset per tokenizer.
SAMPLE_N = 4000
rows = []
for tag in ('D1', 'D2'):
    sub = DATA[tag].loc[SPLITS[tag]['train']]
    sub = sub.sample(n=min(SAMPLE_N, len(sub)), random_state=SPLIT_SEED)
    texts = [normalise(t) for t in sub['text']]
    for model_key in ('BERT', 'DeBERTa'):
        tok = get_tokenizer(model_key)
        lens = np.array([len(x) for x in
                         tok(texts, add_special_tokens=True, truncation=False)['input_ids']])
        rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}', 'tokenizer': model_key,
                     'median': int(np.median(lens)),
                     'p90': int(np.percentile(lens, 90)),
                     'p99': int(np.percentile(lens, 99)),
                     'max': int(lens.max()),
                     f'pct_over_{MAX_LEN}': round(100 * float((lens > MAX_LEN).mean()), 2)})

print(pd.DataFrame(rows).to_string(index=False))'''),
    md('''**Line by line.**

- `SAMPLE_N = 4000` -- this diagnostic runs on a fixed sample, not the full training
  set, purely for speed; 4,000 rows is large enough for the percentiles below to be
  stable.
- `sub.sample(n=min(SAMPLE_N, len(sub)), random_state=SPLIT_SEED)` -- the `min(...)`
  guards against a dataset smaller than 4,000 rows.
- inner loop over `('BERT', 'DeBERTa')` -- runs the diagnostic once per tokenizer,
  because the two vocabularies segment the same text into different numbers of
  tokens (BERT's WordPiece against DeBERTa's SentencePiece).
- `tok(texts, add_special_tokens=True, truncation=False)['input_ids']` -- deliberately
  **not** truncated here, unlike everywhere else this project tokenises text, because
  the whole point of this cell is to measure how long the *untruncated* sequences
  actually are.
- `lens = np.array([len(x) for x in ...])` -- one integer per document, its true
  token count including the `[CLS]`/`[SEP]` special tokens.
- the dictionary built per row -- median, 90th and 99th percentile, and maximum
  length, plus `pct_over_128`, the share of documents that would be cut by this
  project's `MAX_LEN = 128`. This is the number the report cites when justifying
  128 as a length budget rather than a number chosen without evidence.
'''),
    md('''## 1.7 What this notebook produced

`data_D1.parquet`, `split_D1.npz`, `data_D2.parquet`, `split_D2.npz` in
`experiments/paper_scale/work/`. Parts 2, 3 and 4 all read these same four files, so
the two models are scored on byte-identical test rows.'''),
    code('''for tag in ('D1', 'D2'):
    for name in (f'data_{tag}.parquet', f'split_{tag}.npz'):
        p = WORK_DIR / name
        print(f'{name:24s} {p.stat().st_size / 1024 ** 2:8.2f} MB   {p}')'''),
    md('''**Line by line.**

- nested loop -- for each dataset tag, for each of its two output files, look the
  file up under `WORK_DIR` and print its name, size in megabytes
  (`p.stat().st_size / 1024 ** 2`), and full path. A closing sanity check: if either
  file is missing or implausibly small, this cell shows it immediately rather than
  letting Part 2 fail later with a less obvious error.
'''),
]

# ============================================================== PART 2: BERT
CELLS += [
    md('''---
# Part 2. BERT at its best configuration

Fine-tunes `bert-base-uncased` on both datasets, at the configuration that scored
highest on **validation** weighted F1 in an eight-configuration grid (learning rate in
{2e-5, 3e-5}, batch size in {16, 32}, weight decay in {0.01, 0.1}; five epochs, early
stopping on validation F1 with patience 2, max sequence length 128). The winner is
picked on validation and only then run against test, once.

| dataset | learning rate | batch size | weight decay |
|---|---|---|---|
| D1 DAIGT V2 | 3e-5 | 32 | 0.1 |
| D2 HC3 | 2e-5 | 16 | 0.1 |
'''),
    md('## 2.1 Environment and paths\n\nRe-declared so this part runs standalone; identical to 1.1.'),
    code('''import os

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Raw corpora live outside the repository; the repository holds the derived splits.
PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')

FINAL_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
if not FINAL_DIR.exists():
    p = Path.cwd().resolve()
    while p.name != 'Final' and p != p.parent:
        p = p.parent
    FINAL_DIR = p

PS_DIR = FINAL_DIR / 'experiments' / 'paper_scale'
WORK_DIR = PS_DIR / 'work'
RESULTS_DIR = PS_DIR / 'results'
PROBS_DIR = PS_DIR / 'probs'
MODELS_DIR = PS_DIR / 'models'
CKPT_DIR = Path('/media/filwel/MLProject1/nlp_paper_ckpt')

MAX_LEN = 128
EPOCHS = 5
WARMUP_RATIO = 0.1
PATIENCE = 2
SPLIT_SEED = 42
TRAIN_SEED = 42

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}'''),
    md('Same constants as 1.1, see the notes there. Kept identical on purpose: `MAX_LEN`, the seeds and the two model checkpoints must match across every part or the four notebooks would silently stop being comparable.'),
    md('## 2.2 The fixed split from Part 1'),
    code('''def load_fixed_split(tag):
    """Load the one fixed split written by notebook 01.

    The split is built once, with seed 42, and reused by every training run.
    Only model initialisation and batch order vary with the training seed, never
    which rows sit in which partition; re-splitting per seed would silently move
    the evaluation set between runs and invalidate any across-seed comparison.
    """
    data_p = WORK_DIR / f'data_{tag}.parquet'
    split_p = WORK_DIR / f'split_{tag}.npz'
    if not (data_p.exists() and split_p.exists()):
        raise FileNotFoundError(
            f'missing {data_p.name} / {split_p.name}. Run 01_preprocessing.ipynb first.')
    df = pd.read_parquet(data_p)
    sp = np.load(split_p)
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}


DATA, SPLITS = {}, {}
for tag in ('D1', 'D2'):
    DATA[tag], SPLITS[tag] = load_fixed_split(tag)
    n = {k: len(v) for k, v in SPLITS[tag].items()}
    print(f'{tag} {DATASET_NAMES[tag]:9s} train={n["train"]:6d}  val={n["val"]:5d}  '
          f'test={n["test"]:6d}  total={sum(n.values()):6d}')'''),
    md('''**Line by line.**

- `load_fixed_split(tag)` -- unlike Part 1's `build_or_load_splits`, this version has
  no "build" branch at all: if the two files are missing it raises
  `FileNotFoundError` immediately with an instruction to run Part 1 first, rather
  than silently rebuilding a split with a different random draw. That is what
  guarantees this part can never accidentally train on a different partition than
  Part 1 produced.
- `pd.read_parquet` / `np.load` -- read the dataframe and the three index arrays back
  from disk exactly as Part 1 wrote them.
- final loop -- populates `DATA`/`SPLITS` for this part and prints the row counts per
  partition per dataset as a visible confirmation that the expected split loaded (compare
  against Part 1's own printed counts).
'''),
    md('## 2.3 Tokenisation'),
    code('''import torch
from datasets import Dataset
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_TOKCACHE, _DATACACHE = {}, {}


def normalise(t):
    return re.sub(r'\\s+', ' ', str(t)).strip()


def get_tokenizer(model_key):
    if model_key not in _TOKCACHE:
        _TOKCACHE[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return _TOKCACHE[model_key]


def get_tokenized(tag, model_key):
    """Tokenise the three partitions of one dataset for one model, cached in memory."""
    key = (tag, model_key)
    if key in _DATACACHE:
        return _DATACACHE[key]
    df, splits = DATA[tag], SPLITS[tag]
    tok = get_tokenizer(model_key)
    parts = {}
    for split, idx in splits.items():
        sub = df.loc[idx]
        ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                                'labels': [int(v) for v in sub['label']]})
        parts[split] = ds.map(
            lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
            batched=True, remove_columns=['text'])
    _DATACACHE[key] = (parts, splits)
    gc.collect()
    return _DATACACHE[key]


print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())'''),
    md('''**Line by line.**

- `torch.backends.cuda.matmul.allow_tf32 = True` and the cuDNN equivalent -- opt in
  to TensorFloat-32 matrix multiplication on the RTX 3060 Ti, which trades a small
  amount of numerical precision for a real speed-up on Ampere-generation GPUs; this
  has no effect at all if CUDA is unavailable.
- `_TOKCACHE, _DATACACHE = {}, {}` -- two caches: one for loaded tokenizers (as in
  Part 1), one for *already-tokenised datasets*, keyed by `(tag, model_key)`, so
  re-tokenising the same partition twice inside one notebook session costs nothing.
- `get_tokenized(tag, model_key)` -- the cache-check pattern again:
  `if key in _DATACACHE: return _DATACACHE[key]` short-circuits before any real work.
  Otherwise it tokenises all three partitions of that dataset for that model (the
  `for split, idx in splits.items()` loop) and stores the result before returning it.
  `gc.collect()` after the loop reclaims memory from the intermediate raw-text lists.
- final `print` -- reports the installed torch version, whether CUDA is visible to
  this process, and whether bf16 (bfloat16) mixed precision is supported, which is
  what `train_one` below reads to decide `bf16=True/False` in `TrainingArguments`.
'''),
    md('''## 2.4 Training harness

Weighted precision, recall and F1 are used throughout: the classes are balanced by
construction, so weighted and macro averaging agree closely, but weighted is what the
midterm reported and the two halves of the project stay comparable this way. Early
stopping monitors validation F1 with patience 2, and the best checkpoint by that
metric is what gets evaluated, never the last epoch.'''),
    code('''import shutil
import time

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

for d in (RESULTS_DIR, PROBS_DIR, MODELS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def weighted_metrics(y, p):
    acc = accuracy_score(y, p)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y, p, average='weighted', zero_division=0)
    return acc, pre, rec, f1


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, pre, rec, f1 = weighted_metrics(labels, preds)
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}


def run_key(tag, model_key, cfg, seed=TRAIN_SEED):
    return (f'full_{tag}_{model_key}_lr{cfg["lr"]:g}_bs{cfg["bs"]}'
            f'_wd{cfg["wd"]:g}_s{seed}')


def train_one(tag, model_key, cfg, seed=TRAIN_SEED, save_model=True, force=False):
    """Fine-tune one configuration and write its metrics and probabilities.

    If both artefacts already exist and force is False the run is skipped and the
    stored record returned, so re-executing this notebook costs seconds instead of
    hours while still reporting the exact numbers the report quotes.
    """
    key = run_key(tag, model_key, cfg, seed)
    jpath, ppath = RESULTS_DIR / f'{key}.json', PROBS_DIR / f'{key}.npz'
    if not force and jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print(f'[cached] {key}  val_f1={rec["val"]["f1"]:.4f}  test_f1={rec["test"]["f1"]:.4f}')
        return rec

    run_dir = CKPT_DIR / key
    parts, splits = get_tokenized(tag, model_key)
    tok = get_tokenizer(model_key)

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_key], num_labels=2)
    model.config.id2label = {0: 'human', 1: 'ai'}
    model.config.label2id = {'human': 0, 'ai': 1}

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=cfg['lr'],
        per_device_train_batch_size=cfg['bs'],
        per_device_eval_batch_size=64,
        weight_decay=cfg['wd'],
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type='linear',
        optim='adamw_torch',
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        logging_steps=200,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        report_to='none')

    trainer = Trainer(
        model=model, args=args, train_dataset=parts['train'],
        eval_dataset=parts['val'], data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag],
           'model': model_key, 'checkpoint': MODELS[model_key],
           'lr': cfg['lr'], 'batch_size': cfg['bs'], 'weight_decay': cfg['wd'],
           'seed': seed, 'max_len': MAX_LEN,
           'n_train': len(splits['train']), 'n_val': len(splits['val']),
           'n_test': len(splits['test']), 'train_seconds': round(train_secs, 1),
           'epochs_run': int(trainer.state.epoch or 0)}

    probs = {}
    for split in ('val', 'test'):
        pred = trainer.predict(parts[split])
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        p = torch.softmax(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': round(acc, 4), 'precision': round(pre, 4),
                      'recall': round(rec, 4), 'f1': round(f1, 4)}
        out[f'{split}_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs[f'{split}_probs'] = p
        probs[f'{split}_labels'] = y

    np.savez(ppath, **probs)
    json.dump(out, open(jpath, 'w'), indent=2)

    if save_model:
        mdir = MODELS_DIR / f'{tag}_{model_key}'
        trainer.save_model(str(mdir))
        tok.save_pretrained(str(mdir))
        json.dump(out, open(mdir / 'run_info.json', 'w'), indent=2)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f'[trained] {key}  val_f1={out["val"]["f1"]:.4f}  '
          f'test_f1={out["test"]["f1"]:.4f}  {train_secs / 60:.1f} min')
    return out'''),
    md('''**Line by line, the harness every model in this project trains through.**

- `weighted_metrics(y, p)` -- one shared definition of accuracy, precision, recall
  and F1, `average='weighted'` (each class's score weighted by how many true examples
  it has), `zero_division=0` so a class with no predicted members scores 0 instead of
  raising a warning.
- `compute_metrics(eval_pred)` -- the callback signature `Trainer` calls after every
  evaluation pass. `logits, labels = eval_pred` unpacks the tuple `Trainer` hands it;
  `np.argmax(logits, axis=-1)` turns two raw logits per row into a single 0/1
  prediction; the returned dict's keys (`'accuracy'`, `'f1'`, ...) are what
  `metric_for_best_model='eval_f1'` below refers to.
- `run_key(tag, model_key, cfg, seed)` -- builds one canonical string, e.g.
  `full_D1_BERT_lr3e-05_bs32_wd0.1_s42`, that names every artefact this run
  produces. Because the string is built *from the configuration itself*
  (`cfg["lr"]`, `cfg["bs"]`, `cfg["wd"]`), the file on disk is self-describing --
  reading the filename alone tells you exactly what hyperparameters produced it.
- `train_one`, cache check -- `jpath`/`ppath` are the two files this exact
  configuration would produce. If both exist and `force` is not set, it loads and
  returns the stored JSON record instead of retraining, printing `[cached] ...` --
  this is the mechanism that makes the whole four-notebook set re-run in seconds.
- `set_seed(seed)` -- seeds Python's `random`, NumPy and PyTorch (CPU and CUDA) in one
  call, so model initialisation and dropout are reproducible.
- `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` -- loads
  the pretrained encoder and attaches a **freshly, randomly initialised**
  `Dense(hidden_size, 2)` classification head on top; everything below the head
  carries pretrained weights, the head itself does not.
- `model.config.id2label` / `label2id` -- purely cosmetic metadata so a saved
  checkpoint's predictions are labelled `"human"`/`"ai"` instead of `0`/`1` when
  someone inspects it later; it does not change how the model computes anything.
- `TrainingArguments(...)` -- one call, with every training-loop knob visible at
  once: `learning_rate`, `per_device_train_batch_size` and `weight_decay` come from
  `cfg`, i.e. from the specific grid cell being run; `per_device_eval_batch_size=64`
  is fixed and independent of the training batch size, because no gradient is
  computed during evaluation, so it only affects speed and memory, never the result;
  `num_train_epochs=EPOCHS` is a ceiling, not a target, because
  `load_best_model_at_end=True` combined with `EarlyStoppingCallback` below can stop
  training earlier and still restore the best checkpoint; `bf16=...` turns on
  bfloat16 mixed precision only when the GPU supports it; `eval_strategy='epoch'` and
  `save_strategy='epoch'` must match for `load_best_model_at_end` to work at all;
  `metric_for_best_model='eval_f1'` names the exact key `compute_metrics` returns,
  prefixed with `eval_`; `dataloader_num_workers=0` avoids multiprocessing
  overhead that is not worth it at this batch size; `report_to='none'` disables
  logging integrations (Weights & Biases, etc.) that are not configured here.
- `Trainer(...)` -- wires the model, the training arguments, the two tokenised
  partitions, a `DataCollatorWithPadding` (pads each *batch* to its own longest
  sequence rather than to a fixed global length, which is more efficient), the
  metrics function, and the early-stopping callback (`patience=PATIENCE`, i.e. 2
  evaluation rounds without improvement) together into one object.
- `t0 = time.time(); trainer.train(); train_secs = time.time() - t0` -- times the
  actual training call, recorded in the output record for later reporting.
- `out = {...}` -- the metadata half of the result record: which configuration,
  which dataset, how many rows in each partition, how long training took, and how
  many epochs actually ran (`trainer.state.epoch`, which can be less than `EPOCHS` if
  early stopping fired).
- `for split in ('val', 'test'): ...` -- runs prediction on both partitions with the
  *best* restored checkpoint. `pred.predictions[0] if isinstance(..., tuple) else ...`
  guards against a model that returns a tuple of outputs instead of a bare logits
  array (some architectures do). `torch.softmax(..., dim=-1)` converts the two raw
  logits per row into two probabilities that sum to 1. `weighted_metrics` scores the
  argmax of those probabilities against the true labels; the confusion matrix and the
  raw probability arrays are both kept, the former for the printed report, the latter
  (`probs['{split}_probs']`) specifically so Part 4 can build an ensemble without
  ever re-running the model.
- `np.savez(ppath, **probs)` / `json.dump(out, open(jpath, 'w'), indent=2)` -- the
  two files whose existence is what the cache check at the top of this function
  looks for.
- `if save_model: ...` -- optionally also persists the fine-tuned weights and
  tokenizer to `models/{tag}_{model_key}/`, plus a copy of the same JSON record as
  `run_info.json` right next to the weights, so the deployed checkpoint is
  self-describing too.
- cleanup block -- `del trainer, model; gc.collect(); torch.cuda.empty_cache()`
  frees GPU memory before the *next* configuration in a grid loads its own model;
  `shutil.rmtree(run_dir, ignore_errors=True)` deletes the large intermediate
  `Trainer` checkpoint directory in `CKPT_DIR`, since only the final saved model and
  the JSON/`.npz` pair are meant to persist.
- final `print` and `return out` -- a one-line training summary, and the same record
  a cached call would have returned, so callers of `train_one` never need to know
  whether this run trained fresh or loaded from cache.
'''),
    md('## 2.5 Best BERT configuration, per dataset'),
    code('''BERT_BEST = {
    'D1': {'lr': 3e-5, 'bs': 32, 'wd': 0.1},
    'D2': {'lr': 2e-5, 'bs': 16, 'wd': 0.1},
}

BERT_RESULT = {}
for tag in ('D1', 'D2'):
    cfg = BERT_BEST[tag]
    print(f'{tag} {DATASET_NAMES[tag]:9s} BERT  learning_rate={cfg["lr"]:g}  '
          f'batch_size={cfg["bs"]}  weight_decay={cfg["wd"]:g}')
    BERT_RESULT[tag] = train_one(tag, 'BERT', cfg, save_model=True)'''),
    md('''**Line by line.**

- `BERT_BEST` -- the two configurations this cell actually runs, one per dataset,
  as literal values -- these are the numbers that were chosen on validation F1 by an
  earlier, separate eight-configuration grid search (`experiments/paper_scale/
  run_full_grid.py`), not tuned inside this cell. This is the exact dictionary the
  printed code appendix also shows, so the deployed hyperparameters are never hidden
  behind a lookup at print time.
- the `for` loop -- calls `train_one` once per dataset with that dataset's best
  configuration, storing each returned record in `BERT_RESULT[tag]`; thanks to the
  cache check inside `train_one`, both calls return in well under a second here since
  both configurations are already trained and saved on disk.
'''),
    md('## 2.6 Results'),
    code('''def result_table(records, title):
    rows = []
    for tag, rec in records.items():
        rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}',
                     'learning_rate': rec['lr'], 'batch_size': rec['batch_size'],
                     'weight_decay': rec['weight_decay'],
                     'epochs_run': rec.get('epochs_run'),
                     'val_f1': rec['val']['f1'],
                     'test_accuracy': rec['test']['accuracy'],
                     'test_precision': rec['test']['precision'],
                     'test_recall': rec['test']['recall'],
                     'test_f1': rec['test']['f1']})
    tab = pd.DataFrame(rows)
    print(title)
    print(tab.to_string(index=False))
    return tab


def show_confusion(records):
    for tag, rec in records.items():
        cm = np.array(rec['test_confusion'])
        print(f'{tag} {DATASET_NAMES[tag]} test confusion (rows true, cols predicted, '
              f'order human, ai)')
        print(pd.DataFrame(cm, index=['true_human', 'true_ai'],
                           columns=['pred_human', 'pred_ai']).to_string())
        tn, fp, fn, tp = cm.ravel()
        print(f'   false positives (human called ai) = {fp},  '
              f'false negatives (ai called human) = {fn}\\n')

bert_table = result_table(BERT_RESULT, 'BERT, best configuration per dataset\\n')'''),
    md('''**Line by line.**

- `result_table(records, title)` -- flattens the nested `BERT_RESULT` dictionary
  (which has `rec['val']['f1']`, `rec['test']['accuracy']`, etc., nested one level
  deep) into one flat row per dataset, wraps the rows in a `pandas.DataFrame`, prints
  it with `.to_string(index=False)` (no row-number column, since `dataset` already
  identifies each row), and returns it so it can also be inspected as data, not only
  as printed text.
- `show_confusion(records)` -- `np.array(rec['test_confusion'])` restores the
  2x2 confusion matrix from the plain nested list `json.dump` had to use (JSON has
  no native array type). `cm.ravel()` flattens the 2x2 matrix in row-major order,
  which for a matrix stored `[[TN, FP], [FN, TP]]` unpacks cleanly into
  `tn, fp, fn, tp = cm.ravel()`. The final print names the two error directions in
  words, false positive (human wrongly called AI) and false negative (AI wrongly
  called human), because those two mistakes are not equally costly to a real
  deployment.
- last line -- calls `result_table` once, on `BERT_RESULT`, and keeps the returned
  table as `bert_table` for anything downstream that wants it as data.
'''),
    code('show_confusion(BERT_RESULT)'),
    md('''**Line by line.** One call, printing both datasets' confusion matrices and false-positive/false-negative counts using the function defined in the cell above.'''),
    md('''## 2.7 What this notebook produced

For each dataset, three artefacts keyed by the configuration string
`full_{tag}_BERT_lr{lr}_bs{bs}_wd{wd}_s42`: the metrics JSON, the probabilities
`.npz`, and the saved model directory. Part 4 reads the `.npz` probabilities; nothing
is recomputed there, so the ensemble is built from exactly these predictions.'''),
    code('''for tag, rec in BERT_RESULT.items():
    print(f'{tag}  key={rec["key"]}')
    print(f'    results  {(RESULTS_DIR / (rec["key"] + ".json")).exists()}'
          f'   probs {(PROBS_DIR / (rec["key"] + ".npz")).exists()}'
          f'   model {(MODELS_DIR / f"{tag}_BERT").exists()}')'''),
    md('''**Line by line.**

- loop over `BERT_RESULT.items()` -- for each dataset, print the exact configuration
  key that was used, then three booleans confirming the results JSON, the
  probabilities `.npz`, and the saved model directory all actually exist on disk
  under that key -- the same closing existence check pattern as 1.7, one level up
  the pipeline.
'''),
]

# ============================================================== PART 3: DeBERTa
CELLS += [
    md('''---
# Part 3. DeBERTa-v3, the BERT variant, at its best configuration

`microsoft/deberta-v3-base` differs from BERT in three ways that matter here:
disentangled attention, which scores content and relative position separately rather
than adding a position embedding into the token embedding; an enhanced mask decoder
that reinstates absolute position at the output layer; and replaced-token-detection
pretraining instead of masked language modelling. It is the same parameter scale as
`bert-base-uncased`, so the comparison in Part 4 is a like-for-like one of
pretraining and attention design rather than of model size. The grid, selection rule
and split are identical to Part 2.

| dataset | learning rate | batch size | weight decay |
|---|---|---|---|
| D1 DAIGT V2 | 3e-5 | 16 | 0.01 |
| D2 HC3 | 3e-5 | 16 | 0.1 |
'''),
    md('## 3.1 Environment and paths\n\nIdentical to 1.1 / 2.1, re-declared so this part is standalone.'),
    code('''import os

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Raw corpora live outside the repository; the repository holds the derived splits.
PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')

FINAL_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
if not FINAL_DIR.exists():
    p = Path.cwd().resolve()
    while p.name != 'Final' and p != p.parent:
        p = p.parent
    FINAL_DIR = p

PS_DIR = FINAL_DIR / 'experiments' / 'paper_scale'
WORK_DIR = PS_DIR / 'work'
RESULTS_DIR = PS_DIR / 'results'
PROBS_DIR = PS_DIR / 'probs'
MODELS_DIR = PS_DIR / 'models'
CKPT_DIR = Path('/media/filwel/MLProject1/nlp_paper_ckpt')

MAX_LEN = 128
EPOCHS = 5
WARMUP_RATIO = 0.1
PATIENCE = 2
SPLIT_SEED = 42
TRAIN_SEED = 42

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}'''),
    md('Byte-identical to 2.1. See the notes there.'),
    md('## 3.2 The fixed split from Part 1\n\nThe same partitions BERT saw, loaded from the same two files. This is what allows the two models to be compared row by row in Part 4.'),
    code('''def load_fixed_split(tag):
    """Load the one fixed split written by notebook 01.

    The split is built once, with seed 42, and reused by every training run.
    Only model initialisation and batch order vary with the training seed, never
    which rows sit in which partition; re-splitting per seed would silently move
    the evaluation set between runs and invalidate any across-seed comparison.
    """
    data_p = WORK_DIR / f'data_{tag}.parquet'
    split_p = WORK_DIR / f'split_{tag}.npz'
    if not (data_p.exists() and split_p.exists()):
        raise FileNotFoundError(
            f'missing {data_p.name} / {split_p.name}. Run 01_preprocessing.ipynb first.')
    df = pd.read_parquet(data_p)
    sp = np.load(split_p)
    return df, {'train': sp['train'], 'val': sp['val'], 'test': sp['test']}


DATA, SPLITS = {}, {}
for tag in ('D1', 'D2'):
    DATA[tag], SPLITS[tag] = load_fixed_split(tag)
    n = {k: len(v) for k, v in SPLITS[tag].items()}
    print(f'{tag} {DATASET_NAMES[tag]:9s} train={n["train"]:6d}  val={n["val"]:5d}  '
          f'test={n["test"]:6d}  total={sum(n.values()):6d}')'''),
    md('Byte-identical to 2.2. See the notes there.'),
    md('''## 3.3 Tokenisation

DeBERTa-v3 uses a SentencePiece vocabulary of about 128k pieces, against BERT's 30k
WordPiece. The same document therefore becomes a different number of tokens under
the two tokenizers, which is why Part 1's length diagnostic (1.6) reports each model
separately.'''),
    code('''import torch
from datasets import Dataset
from transformers import AutoTokenizer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

_TOKCACHE, _DATACACHE = {}, {}


def normalise(t):
    return re.sub(r'\\s+', ' ', str(t)).strip()


def get_tokenizer(model_key):
    if model_key not in _TOKCACHE:
        _TOKCACHE[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return _TOKCACHE[model_key]


def get_tokenized(tag, model_key):
    """Tokenise the three partitions of one dataset for one model, cached in memory."""
    key = (tag, model_key)
    if key in _DATACACHE:
        return _DATACACHE[key]
    df, splits = DATA[tag], SPLITS[tag]
    tok = get_tokenizer(model_key)
    parts = {}
    for split, idx in splits.items():
        sub = df.loc[idx]
        ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                                'labels': [int(v) for v in sub['label']]})
        parts[split] = ds.map(
            lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
            batched=True, remove_columns=['text'])
    _DATACACHE[key] = (parts, splits)
    gc.collect()
    return _DATACACHE[key]


print('torch', torch.__version__, '| cuda', torch.cuda.is_available(),
      '| bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())'''),
    md('Byte-identical to 2.3. See the notes there. Note that `get_tokenizer` reads `MODELS[model_key]`, so when this cell is called with `model_key=\'DeBERTa\'` in the training cells below, it loads `microsoft/deberta-v3-base`\'s own SentencePiece tokenizer, not BERT\'s.'),
    md('## 3.4 Training harness\n\nByte-identical to the harness in Part 2, including the early-stopping rule and the cache check. Only the model key and the configuration passed into it change.'),
    code('''import shutil
import time

from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification,
                          DataCollatorWithPadding, EarlyStoppingCallback,
                          Trainer, TrainingArguments, set_seed)

for d in (RESULTS_DIR, PROBS_DIR, MODELS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def weighted_metrics(y, p):
    acc = accuracy_score(y, p)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y, p, average='weighted', zero_division=0)
    return acc, pre, rec, f1


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, pre, rec, f1 = weighted_metrics(labels, preds)
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}


def run_key(tag, model_key, cfg, seed=TRAIN_SEED):
    return (f'full_{tag}_{model_key}_lr{cfg["lr"]:g}_bs{cfg["bs"]}'
            f'_wd{cfg["wd"]:g}_s{seed}')


def train_one(tag, model_key, cfg, seed=TRAIN_SEED, save_model=True, force=False):
    """Fine-tune one configuration and write its metrics and probabilities.

    If both artefacts already exist and force is False the run is skipped and the
    stored record returned, so re-executing this notebook costs seconds instead of
    hours while still reporting the exact numbers the report quotes.
    """
    key = run_key(tag, model_key, cfg, seed)
    jpath, ppath = RESULTS_DIR / f'{key}.json', PROBS_DIR / f'{key}.npz'
    if not force and jpath.exists() and ppath.exists():
        rec = json.load(open(jpath))
        print(f'[cached] {key}  val_f1={rec["val"]["f1"]:.4f}  test_f1={rec["test"]["f1"]:.4f}')
        return rec

    run_dir = CKPT_DIR / key
    parts, splits = get_tokenized(tag, model_key)
    tok = get_tokenizer(model_key)

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODELS[model_key], num_labels=2)
    model.config.id2label = {0: 'human', 1: 'ai'}
    model.config.label2id = {'human': 0, 'ai': 1}

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=cfg['lr'],
        per_device_train_batch_size=cfg['bs'],
        per_device_eval_batch_size=64,
        weight_decay=cfg['wd'],
        num_train_epochs=EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type='linear',
        optim='adamw_torch',
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        logging_steps=200,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        report_to='none')

    trainer = Trainer(
        model=model, args=args, train_dataset=parts['train'],
        eval_dataset=parts['val'], data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])

    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag],
           'model': model_key, 'checkpoint': MODELS[model_key],
           'lr': cfg['lr'], 'batch_size': cfg['bs'], 'weight_decay': cfg['wd'],
           'seed': seed, 'max_len': MAX_LEN,
           'n_train': len(splits['train']), 'n_val': len(splits['val']),
           'n_test': len(splits['test']), 'train_seconds': round(train_secs, 1),
           'epochs_run': int(trainer.state.epoch or 0)}

    probs = {}
    for split in ('val', 'test'):
        pred = trainer.predict(parts[split])
        raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        p = torch.softmax(torch.tensor(raw, dtype=torch.float32), dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': round(acc, 4), 'precision': round(pre, 4),
                      'recall': round(rec, 4), 'f1': round(f1, 4)}
        out[f'{split}_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        probs[f'{split}_probs'] = p
        probs[f'{split}_labels'] = y

    np.savez(ppath, **probs)
    json.dump(out, open(jpath, 'w'), indent=2)

    if save_model:
        mdir = MODELS_DIR / f'{tag}_{model_key}'
        trainer.save_model(str(mdir))
        tok.save_pretrained(str(mdir))
        json.dump(out, open(mdir / 'run_info.json', 'w'), indent=2)

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    shutil.rmtree(run_dir, ignore_errors=True)
    print(f'[trained] {key}  val_f1={out["val"]["f1"]:.4f}  '
          f'test_f1={out["test"]["f1"]:.4f}  {train_secs / 60:.1f} min')
    return out'''),
    md('Byte-identical to 2.4. See the full line-by-line notes there -- every line of this harness behaves the same way regardless of which model key is passed in; the only thing that changes is which pretrained checkpoint `AutoModelForSequenceClassification.from_pretrained(MODELS[model_key], ...)` downloads and fine-tunes.'),
    md('## 3.5 Best DeBERTa configuration, per dataset'),
    code('''DEBERTA_BEST = {
    'D1': {'lr': 3e-5, 'bs': 16, 'wd': 0.01},
    'D2': {'lr': 3e-5, 'bs': 16, 'wd': 0.1},
}

DEBERTA_RESULT = {}
for tag in ('D1', 'D2'):
    cfg = DEBERTA_BEST[tag]
    print(f'{tag} {DATASET_NAMES[tag]:9s} DeBERTa  learning_rate={cfg["lr"]:g}  '
          f'batch_size={cfg["bs"]}  weight_decay={cfg["wd"]:g}')
    DEBERTA_RESULT[tag] = train_one(tag, 'DeBERTa', cfg, save_model=True)'''),
    md('''**Line by line.**

- `DEBERTA_BEST` -- DeBERTa's own winning configuration per dataset, from the same
  eight-configuration grid search, independent of BERT's winners in 2.5; note the
  weight decay for D1 is 0.01 here against BERT's 0.1, and the batch size is 16 for
  both DeBERTa runs against BERT's 32 on D1 -- the two models were not forced onto
  the same hyperparameters, each was tuned on its own validation F1.
- the `for` loop -- same pattern as 2.5, but calling `train_one` with
  `model_key='DeBERTa'`, which is what makes `MODELS[model_key]` resolve to
  `'microsoft/deberta-v3-base'` inside the shared harness.
'''),
    md('## 3.6 Results'),
    code('''def result_table(records, title):
    rows = []
    for tag, rec in records.items():
        rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}',
                     'learning_rate': rec['lr'], 'batch_size': rec['batch_size'],
                     'weight_decay': rec['weight_decay'],
                     'epochs_run': rec.get('epochs_run'),
                     'val_f1': rec['val']['f1'],
                     'test_accuracy': rec['test']['accuracy'],
                     'test_precision': rec['test']['precision'],
                     'test_recall': rec['test']['recall'],
                     'test_f1': rec['test']['f1']})
    tab = pd.DataFrame(rows)
    print(title)
    print(tab.to_string(index=False))
    return tab


def show_confusion(records):
    for tag, rec in records.items():
        cm = np.array(rec['test_confusion'])
        print(f'{tag} {DATASET_NAMES[tag]} test confusion (rows true, cols predicted, '
              f'order human, ai)')
        print(pd.DataFrame(cm, index=['true_human', 'true_ai'],
                           columns=['pred_human', 'pred_ai']).to_string())
        tn, fp, fn, tp = cm.ravel()
        print(f'   false positives (human called ai) = {fp},  '
              f'false negatives (ai called human) = {fn}\\n')

deberta_table = result_table(DEBERTA_RESULT, 'DeBERTa-v3, best configuration per dataset\\n')'''),
    md('Byte-identical to 2.6, run on `DEBERTA_RESULT` instead of `BERT_RESULT`. See the notes there.'),
    code('show_confusion(DEBERTA_RESULT)'),
    md('One call, printing both datasets\' confusion matrices for the DeBERTa runs.'),
    md('''## 3.7 Side by side with BERT

Read from the stored BERT records on disk, so this cell does not depend on Part 2
still being in memory in the same kernel session -- only on Part 2 having been run at
least once, ever.'''),
    code('''BERT_BEST = {
    'D1': {'lr': 3e-5, 'bs': 32, 'wd': 0.1},
    'D2': {'lr': 2e-5, 'bs': 16, 'wd': 0.1},
}

rows = []
for tag in ('D1', 'D2'):
    bkey = (f'full_{tag}_BERT_lr{BERT_BEST[tag]["lr"]:g}_bs{BERT_BEST[tag]["bs"]}'
            f'_wd{BERT_BEST[tag]["wd"]:g}_s{TRAIN_SEED}')
    bpath = RESULTS_DIR / f'{bkey}.json'
    if not bpath.exists():
        print(f'{tag}: BERT record not found, run 02_bert_best_config.ipynb first')
        continue
    b = json.load(open(bpath))
    d = DEBERTA_RESULT[tag]
    rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}',
                 'BERT_test_f1': b['test']['f1'],
                 'DeBERTa_test_f1': d['test']['f1'],
                 'delta_pp': round(100 * (d['test']['f1'] - b['test']['f1']), 3)})

if rows:
    print(pd.DataFrame(rows).to_string(index=False))'''),
    md('''**Line by line.**

- `BERT_BEST` -- re-declared here (rather than imported from Part 2) so this cell
  works even if only Part 3 has been run in the current kernel session; the values
  are identical to 2.5's.
- `bkey = (...)` -- rebuilds BERT's run key by hand, using the same f-string pattern
  as `run_key()`, rather than calling that function, so this cell has no hidden
  dependency on Part 2's Python objects, only on the file it wrote.
- `if not bpath.exists(): print(...); continue` -- a graceful skip, not a crash, if
  Part 2 genuinely has never been run: the loop still finishes and shows results for
  whichever datasets it could find a BERT record for.
- `b = json.load(open(bpath))` -- reads BERT's stored JSON directly from disk, the
  same file `train_one` wrote in Part 2.
- `d = DEBERTA_RESULT[tag]` -- the DeBERTa record still in memory from 3.5, in this
  session.
- `delta_pp` -- DeBERTa's test F1 minus BERT's, in percentage points (`* 100`), the
  single number this cell exists to compute.
'''),
    md('## 3.8 What this notebook produced\n\n`results/{key}.json`, `probs/{key}.npz` and `models/{tag}_DeBERTa/` for each dataset, under the key `full_{tag}_DeBERTa_lr{lr}_bs{bs}_wd{wd}_s42`. Part 4 consumes the probability files.'),
    code('''for tag, rec in DEBERTA_RESULT.items():
    print(f'{tag}  key={rec["key"]}')
    print(f'    results  {(RESULTS_DIR / (rec["key"] + ".json")).exists()}'
          f'   probs {(PROBS_DIR / (rec["key"] + ".npz")).exists()}'
          f'   model {(MODELS_DIR / f"{tag}_DeBERTa").exists()}')'''),
    md('Byte-identical pattern to 2.7, over `DEBERTA_RESULT` instead of `BERT_RESULT`.'),
]

# ============================================================== PART 4: ensemble
CELLS += [
    md('''---
# Part 4. Ensemble of BERT and DeBERTa

A weighted soft vote over the two fine-tuned models:

$$P_{\\text{ensemble}} = w \\cdot P_{\\text{BERT}} + (1 - w) \\cdot P_{\\text{DeBERTa}}$$

Soft rather than hard voting, because with only two members a majority vote has no
way to break a one-to-one tie. This part trains nothing; it reads only the saved
probability files Parts 2 and 3 wrote.

Procedure, and the order matters: take the two best configurations already selected
on validation; sweep the mixing weight `w` from 0 to 1 in steps of 0.05 and keep
whichever maximises **validation** weighted F1; apply that single fixed `w` to the
test set exactly once; compare the ensemble against its stronger member with a paired
test, since both models predicted the same test rows.'''),
    md('## 4.1 Environment and paths\n\nIdentical to 1.1 / 2.1 / 3.1, re-declared so this part is standalone.'),
    code('''import os

os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Raw corpora live outside the repository; the repository holds the derived splits.
PROJECT_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Project ')

FINAL_DIR = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
if not FINAL_DIR.exists():
    p = Path.cwd().resolve()
    while p.name != 'Final' and p != p.parent:
        p = p.parent
    FINAL_DIR = p

PS_DIR = FINAL_DIR / 'experiments' / 'paper_scale'
WORK_DIR = PS_DIR / 'work'
RESULTS_DIR = PS_DIR / 'results'
PROBS_DIR = PS_DIR / 'probs'
MODELS_DIR = PS_DIR / 'models'
CKPT_DIR = Path('/media/filwel/MLProject1/nlp_paper_ckpt')

MAX_LEN = 128
EPOCHS = 5
WARMUP_RATIO = 0.1
PATIENCE = 2
SPLIT_SEED = 42
TRAIN_SEED = 42

MODELS = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
DATASET_NAMES = {'D1': 'DAIGT V2', 'D2': 'HC3'}'''),
    md('Byte-identical to 3.1. See the notes at 1.1. This part never fine-tunes a model, so most of these constants (`MAX_LEN`, `EPOCHS`, ...) are unused here; they are kept for consistency across all four parts rather than trimmed.'),
    md('''## 4.2 Load the two members' saved probabilities

Keys are rebuilt from the same configuration dictionaries Parts 2 and 3 used, so the
ensemble is guaranteed to be built from the deployed checkpoints and not from some
other cell of the grid. The label vectors are asserted equal, which is the check
that the two members really were scored on the same rows.'''),
    code('''BERT_BEST = {
    'D1': {'lr': 3e-5, 'bs': 32, 'wd': 0.1},
    'D2': {'lr': 2e-5, 'bs': 16, 'wd': 0.1},
}
DEBERTA_BEST = {
    'D1': {'lr': 3e-5, 'bs': 16, 'wd': 0.01},
    'D2': {'lr': 3e-5, 'bs': 16, 'wd': 0.1},
}
BEST = {'BERT': BERT_BEST, 'DeBERTa': DEBERTA_BEST}
MEMBERS = ('BERT', 'DeBERTa')


def run_key(tag, model_key, seed=TRAIN_SEED):
    cfg = BEST[model_key][tag]
    return (f'full_{tag}_{model_key}_lr{cfg["lr"]:g}_bs{cfg["bs"]}'
            f'_wd{cfg["wd"]:g}_s{seed}')


def load_probs(key):
    p = PROBS_DIR / f'{key}.npz'
    if not p.exists():
        raise FileNotFoundError(
            f'{p.name} not found. Run 02_bert_best_config.ipynb and '
            f'03_deberta_best_config.ipynb first.')
    z = np.load(p)
    return {k: z[k] for k in z.files}


PROBS, LABELS = {}, {}
for tag in ('D1', 'D2'):
    PROBS[tag] = {mk: load_probs(run_key(tag, mk)) for mk in MEMBERS}
    assert np.array_equal(PROBS[tag]['BERT']['val_labels'],
                          PROBS[tag]['DeBERTa']['val_labels']), 'validation labels differ'
    assert np.array_equal(PROBS[tag]['BERT']['test_labels'],
                          PROBS[tag]['DeBERTa']['test_labels']), 'test labels differ'
    LABELS[tag] = {'val': PROBS[tag]['BERT']['val_labels'],
                   'test': PROBS[tag]['BERT']['test_labels']}
    print(f'{tag} {DATASET_NAMES[tag]:9s} val={len(LABELS[tag]["val"]):5d}  '
          f'test={len(LABELS[tag]["test"]):5d}   '
          f'members: {run_key(tag, "BERT")} | {run_key(tag, "DeBERTa")}')'''),
    md('''**Line by line.**

- `BERT_BEST`, `DEBERTA_BEST` -- the two models' deployed configurations, copied
  again from 2.5/3.5 so this part is standalone; `BEST = {'BERT': ..., 'DeBERTa': ...}`
  wraps both under one dictionary keyed by model name, `MEMBERS = ('BERT', 'DeBERTa')`
  names the two ensemble members explicitly.
- `run_key(tag, model_key, seed)` -- this version reconstructs a configuration key
  by first looking it up in `BEST[model_key][tag]`, rather than taking a `cfg` dict as
  an argument the way Part 2/3's `train_one`-facing `run_key` did; same output string,
  different calling convention, because this part never trains anything and only
  needs to name existing files.
- `load_probs(key)` -- looks for `{key}.npz` under `PROBS_DIR`; raises with an
  actionable message naming the two notebooks to run first if it is missing, rather
  than a bare `FileNotFoundError`. `{k: z[k] for k in z.files}` converts the lazy
  `NpzFile` object into a plain dict of arrays (`val_probs`, `val_labels`,
  `test_probs`, `test_labels`), which is easier to work with below.
- the `for tag in ('D1', 'D2')` loop -- `PROBS[tag] = {mk: load_probs(...) for mk in
  MEMBERS}` loads both members' saved arrays for this dataset in one dict
  comprehension. The two `assert np.array_equal(...)` lines are the load-bearing
  safety check of this whole part: they confirm BERT's and DeBERTa's saved
  `val_labels`/`test_labels` arrays are identical element-for-element, which is only
  true if both were evaluated on the exact same rows in the exact same order --
  exactly what the shared, fixed split from Part 1 guarantees, and what this
  assertion verifies rather than assumes. `LABELS[tag]` then keeps one copy of the
  (now proven-shared) label arrays, since either model's copy is equally valid.
'''),
    md('''## 4.3 Metrics and the paired test

McNemar's exact test asks only about the rows where the two systems disagree: of
those, is the split between them further from even than chance would allow. The
paired bootstrap resamples test rows 10,000 times and reports a 95 percent interval
on the difference in error rate. Both are paired tests, which is the right family
here because the same test rows went through both systems.'''),
    code('''from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

N_BOOT = 10000


def weighted_metrics(y, p):
    acc = accuracy_score(y, p)
    pre, rec, f1, _ = precision_recall_fscore_support(
        y, p, average='weighted', zero_division=0)
    return acc, pre, rec, f1


def paired_test(y, pred_a, pred_b, n_boot=N_BOOT, seed=SPLIT_SEED):
    """McNemar exact plus a paired bootstrap on the error difference, a minus b."""
    rng = np.random.default_rng(seed)
    wa, wb = pred_a != y, pred_b != y
    b = int((~wa & wb).sum())      # a right, b wrong
    c = int((wa & ~wb).sum())      # a wrong, b right
    p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    boot = wa[idx].mean(1) - wb[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'a_right_b_wrong': b, 'a_wrong_b_right': c, 'mcnemar_exact_p': float(p),
            'error_diff_pp': float((wa.mean() - wb.mean()) * 100),
            'ci_lo_pp': float(lo * 100), 'ci_hi_pp': float(hi * 100),
            'ci_excludes_zero': bool(lo > 0 or hi < 0)}'''),
    md('''**Line by line.**

- `N_BOOT = 10000` -- number of bootstrap resamples the paired test below draws;
  10,000 is a standard choice, large enough that the 2.5th/97.5th percentiles are
  stable across repeated runs.
- `weighted_metrics` -- the same definition as in Parts 2/3, redeclared here since
  this part is standalone.
- `paired_test(y, pred_a, pred_b, n_boot, seed)`, first two lines -- `wa = pred_a !=
  y` and `wb = pred_b != y` are two boolean arrays, `True` wherever that model's
  prediction is wrong. Everything below is built from these two arrays.
- `b = int((~wa & wb).sum())` -- counts rows where `a` is right (`~wa`) and `b` is
  wrong (`wb`): "a beats b" cases. `c = int((wa & ~wb).sum())` is the mirror image,
  "b beats a" cases. These `b`/`c` names follow the standard 2x2 McNemar contingency
  table notation, not the model names.
- `p = binomtest(b, b + c, 0.5).pvalue if (b + c) else 1.0` -- McNemar's exact test:
  under the null hypothesis that the two models are equally likely to be the one that
  is right when they disagree, `b` should be roughly half of `b + c`. `binomtest`
  computes the exact two-sided p-value for that; if `b + c == 0` (the two models never
  disagree at all) the ternary falls back to `p = 1.0` rather than dividing by zero.
- `idx = rng.integers(0, len(y), size=(n_boot, len(y)))` -- one matrix of resampled
  row indices, shape `(10000, n_test_rows)`, each row a fresh sample-with-replacement
  of the test set. `boot = wa[idx].mean(1) - wb[idx].mean(1)` uses NumPy fancy
  indexing to apply all 10,000 resamples at once: `wa[idx]` has the same
  `(10000, n)` shape, `.mean(1)` averages each resampled row into one error rate per
  bootstrap replicate, and the subtraction gives 10,000 draws of "a's error rate
  minus b's error rate".
- `lo, hi = np.percentile(boot, [2.5, 97.5])` -- the empirical 95% confidence
  interval on that difference, read directly off the bootstrap distribution rather
  than assuming it is normal.
- the returned dict -- `error_diff_pp` and the interval bounds are converted to
  percentage points (`* 100`) for readability; `ci_excludes_zero` is a convenience
  boolean, `True` only when the whole interval sits on one side of zero, i.e. when the
  bootstrap agrees the difference is not noise.
'''),
    md('## 4.4 Select the mixing weight on validation, then test once'),
    code('''WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)
ENSEMBLE, VAL_CURVE = {}, {}

for tag in ('D1', 'D2'):
    pb, pdb = PROBS[tag]['BERT'], PROBS[tag]['DeBERTa']
    yv, yt = LABELS[tag]['val'], LABELS[tag]['test']

    val_f1 = [weighted_metrics(yv, (w * pb['val_probs']
                                    + (1 - w) * pdb['val_probs']).argmax(1))[3]
              for w in WEIGHTS]
    best_w = float(WEIGHTS[int(np.argmax(val_f1))])
    VAL_CURVE[tag] = dict(zip(WEIGHTS.tolist(), [round(v, 4) for v in val_f1]))

    ens_pred = (best_w * pb['test_probs'] + (1 - best_w) * pdb['test_probs']).argmax(1)
    acc, pre, rec, f1 = weighted_metrics(yt, ens_pred)

    member_pred = {mk: PROBS[tag][mk]['test_probs'].argmax(1) for mk in MEMBERS}
    member_f1 = {mk: weighted_metrics(yt, v)[3] for mk, v in member_pred.items()}
    stronger = max(member_f1, key=member_f1.get)

    ENSEMBLE[tag] = {
        'weight_bert': best_w, 'weight_deberta': round(1 - best_w, 2),
        'degenerate': best_w in (0.0, 1.0),
        'test_accuracy': round(acc, 4), 'test_precision': round(pre, 4),
        'test_recall': round(rec, 4), 'test_f1': round(f1, 4),
        'member_f1': {mk: round(v, 4) for mk, v in member_f1.items()},
        'stronger_member': stronger,
        'ensemble_minus_stronger_f1': round(f1 - member_f1[stronger], 4),
        'confusion': confusion_matrix(yt, ens_pred).tolist(),
        'paired_vs_stronger': paired_test(yt, ens_pred, member_pred[stronger])}

    e = ENSEMBLE[tag]
    pt = e['paired_vs_stronger']
    print(f'{tag} {DATASET_NAMES[tag]}')
    print(f'   weight BERT={best_w:.2f}  weight DeBERTa={1 - best_w:.2f}'
          + ('   [weight sits at an endpoint]' if e['degenerate'] else ''))
    print(f'   BERT {member_f1["BERT"]:.4f}   DeBERTa {member_f1["DeBERTa"]:.4f}   '
          f'ensemble {f1:.4f}   (ensemble minus stronger member '
          f'{e["ensemble_minus_stronger_f1"]:+.4f})')
    print(f'   against {stronger}: McNemar p={pt["mcnemar_exact_p"]:.4g}  '
          f'error difference {pt["error_diff_pp"]:+.3f} pp  '
          f'95 percent CI [{pt["ci_lo_pp"]:+.3f}, {pt["ci_hi_pp"]:+.3f}]')
    print()'''),
    md('''**Line by line, the cell that actually builds the ensemble.**

- `WEIGHTS = np.round(np.arange(0.0, 1.0001, 0.05), 2)` -- 21 candidate weights, 0.00,
  0.05, ..., 1.00; the upper bound is `1.0001` rather than `1.0` because
  `np.arange`'s stop value is exclusive and floating-point step accumulation could
  otherwise drop the final 1.00; `np.round(..., 2)` then cleans up any floating-point
  residue so the weights print as exact two-decimal values.
- `pb, pdb = PROBS[tag]['BERT'], PROBS[tag]['DeBERTa']` -- shorthand for this
  dataset's two members' saved probability dictionaries.
- `val_f1 = [... for w in WEIGHTS]` -- for every candidate weight, mixes the two
  models' *validation* probabilities (`w * pb['val_probs'] + (1 - w) *
  pdb['val_probs']`), takes the argmax to get a 0/1 prediction per row, scores it
  with `weighted_metrics`, and keeps only the F1 (index `[3]` of the returned
  4-tuple). This is a list of 21 numbers, one F1 score per candidate weight.
- `best_w = float(WEIGHTS[int(np.argmax(val_f1))])` -- picks the weight whose
  validation F1 was highest; `np.argmax` returns the position of the first maximum,
  so if several weights tie exactly, the lowest such weight wins (relevant to the HC3
  case discussed in 4.7).
- `VAL_CURVE[tag] = dict(zip(WEIGHTS.tolist(), ...))` -- stores the full 21-point
  curve, not just the winning weight, so 4.5 can print the whole sweep and the reader
  can see how flat or peaked it was.
- `ens_pred = (...).argmax(1)` -- the actual ensemble prediction, computed exactly
  once, using `best_w` on the **test** probabilities. This is the only place test
  probabilities are combined; everything above this line only touched validation.
- `member_pred`, `member_f1` -- each individual model's own test-set predictions and
  F1, computed independently of the ensemble, so the ensemble can be compared against
  each member on equal footing.
- `stronger = max(member_f1, key=member_f1.get)` -- the name (`'BERT'` or
  `'DeBERTa'`) of whichever member scored higher F1 on this dataset's test set.
- the `ENSEMBLE[tag] = {...}` dictionary -- the full record for this dataset:
  `degenerate` flags a weight that landed exactly on 0.0 or 1.0 (meaning the
  "ensemble" is really just one member); `ensemble_minus_stronger_f1` is the single
  number that answers "did mixing help at all"; `paired_vs_stronger` calls the
  `paired_test` function from 4.3 to compare the ensemble's predictions against the
  stronger member's, on the same test rows.
- the four `print` lines -- a human-readable summary of everything just computed:
  which weight won (and whether it was degenerate), both members' F1 against the
  ensemble's, and the McNemar p-value with its bootstrap interval, so the numbers in
  4.7's prose can be checked directly against this cell's own output.
'''),
    md('''## 4.5 The validation weight sweep

Printed in full because the shape of this curve is the evidence for how the weight
was chosen, and on HC3 it is also the evidence that the flat region is wide.'''),
    code('''curve = pd.DataFrame(VAL_CURVE)
curve.index.name = 'w_bert'
curve.columns = [f'{c} {DATASET_NAMES[c]} val_f1' for c in curve.columns]
print(curve.to_string())'''),
    md('''**Line by line.**

- `pd.DataFrame(VAL_CURVE)` -- `VAL_CURVE` is a dict of dicts, `{tag: {weight:
  f1, ...}, ...}`; passed to `pd.DataFrame` this becomes a table with one row per
  weight and one column per dataset, automatically aligned on the shared weight
  values.
- `curve.index.name = 'w_bert'` -- labels the row index (which holds the 21 candidate
  weights) for a clearer printed table.
- `curve.columns = [...]` -- renames the two dataset columns from bare tags
  (`'D1'`, `'D2'`) to full descriptive labels before printing.
'''),
    md('## 4.6 Ensemble results'),
    code('''rows = []
for tag in ('D1', 'D2'):
    e = ENSEMBLE[tag]
    rows.append({'dataset': f'{tag} {DATASET_NAMES[tag]}',
                 'w_bert': e['weight_bert'], 'w_deberta': e['weight_deberta'],
                 'BERT_f1': e['member_f1']['BERT'],
                 'DeBERTa_f1': e['member_f1']['DeBERTa'],
                 'ensemble_f1': e['test_f1'],
                 'ensemble_accuracy': e['test_accuracy'],
                 'vs_stronger': e['ensemble_minus_stronger_f1'],
                 'mcnemar_p': round(e['paired_vs_stronger']['mcnemar_exact_p'], 4)})

print(pd.DataFrame(rows).to_string(index=False))
print()
for tag in ('D1', 'D2'):
    cm = np.array(ENSEMBLE[tag]['confusion'])
    print(f'{tag} {DATASET_NAMES[tag]} ensemble test confusion')
    print(pd.DataFrame(cm, index=['true_human', 'true_ai'],
                       columns=['pred_human', 'pred_ai']).to_string())
    print()'''),
    md('''**Line by line.**

- first loop -- flattens the `ENSEMBLE[tag]` dictionaries built in 4.4 into one
  summary row per dataset (weights, each member's F1, the ensemble's F1 and
  accuracy, the gap against the stronger member, and the McNemar p-value), printed as
  one table.
- second loop -- prints each dataset's ensemble confusion matrix, in the same
  `true_human`/`true_ai` by `pred_human`/`pred_ai` layout `show_confusion` uses in
  Parts 2 and 3, so the ensemble's error pattern is directly comparable to either
  member's.
'''),
    md('''## 4.7 Reading the result

An ensemble helps only when two conditions hold together: the members are comparable
in strength, and they make different mistakes. The two datasets sit on opposite sides
of the first condition, which is why they behave differently here.

**DAIGT V2.** The members are almost exactly matched, 0.9916 against 0.9917, and they
do make different mistakes, so the validation sweep settles on an even split and the
mixture improves on both of them: 0.9936, which is 0.19 percentage points above the
stronger member. The two systems disagree on 49 test rows and the ensemble wins 31 of
them. That is a favourable direction, but McNemar's exact test returns p = 0.085 and
the bootstrap interval on the error difference runs from -0.39 to +0.01 percentage
points, so it spans zero. The honest statement is that the ensemble is not worse and
is probably slightly better, not that the gain is established on this test set.

**HC3.** DeBERTa is ahead by 0.56 percentage points, which is a wide margin at this
error rate, and letting BERT vote can only pull predictions toward the weaker model.
Validation F1 is flat at 0.9979 for every weight from 0 up to 0.50 and falls after
that, so the argmax lands on the first of the tied weights, 0. The resulting test
predictions are identical to DeBERTa's, the two disagree on zero rows, and McNemar
has nothing left to test.

A weight at an endpoint is not a failed run and should not be described as the
ensemble collapsing. It is the selection rule working: given a member this dominant,
the best available mixture leans entirely on it, and the flat validation band exists
precisely because so few predictions change across most of the range. What would
improve on this is more diversity between members, not a different mixing rule --
a third family with a different inductive bias, or the same architecture trained on
different folds of the data.

`docs/ENSEMBLE_EXPLAINED.md` carries the longer version of this argument.'''),
    code('print(json.dumps(ENSEMBLE, indent=2, default=float))'),
    md('''**Line by line.**

- `json.dumps(ENSEMBLE, indent=2, default=float)` -- pretty-prints the entire
  `ENSEMBLE` dictionary built in 4.4, both datasets, every field, as formatted JSON.
  `default=float` is a fallback converter for any value `json` does not natively know
  how to serialise (a leftover NumPy scalar, for instance) by casting it to a plain
  Python `float` first; every value already stored in `ENSEMBLE` was rounded to a
  plain Python type when it was built, so in practice this fallback is rarely
  triggered, but it is there as a safety net rather than letting the whole cell raise
  `TypeError` over one stray value.
'''),
]

# ------------------------------------------------------------------- write & run
def main():
    nb = nbf.v4.new_notebook(cells=CELLS)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                                 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python'}
    nbf.write(nb, str(OUT))
    n_code = sum(1 for c in CELLS if c['cell_type'] == 'code')
    n_md = sum(1 for c in CELLS if c['cell_type'] == 'markdown')
    print(f'wrote {OUT} ({len(CELLS)} cells: {n_code} code, {n_md} markdown)')

    client = NotebookClient(nb, timeout=1800, kernel_name='python3',
                            resources={'metadata': {'path': str(OUT.parent)}})
    client.execute()
    nbf.write(nb, str(OUT))
    n_err = sum(1 for c in nb.cells if c.get('cell_type') == 'code'
               for o in c.get('outputs', []) if o.get('output_type') == 'error')
    print(f'executed. {n_err} cell(s) errored.' if n_err else 'executed cleanly, 0 errors.')
    if n_err:
        sys.exit(1)


if __name__ == '__main__':
    main()
