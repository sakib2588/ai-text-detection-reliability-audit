"""Builds the four separate submission notebooks required by the final report.

  01_preprocessing.ipynb          data loading, balancing, group-aware split, cleaning, tokenisation
  02_bert_best_config.ipynb       BERT at its best configuration, per dataset
  03_deberta_best_config.ipynb    DeBERTa-v3 (the BERT variant) at its best configuration
  04_ensemble.ipynb               weighted soft-vote ensemble of the two

Each notebook is self-contained: it re-declares its own paths and constants and
does not import from the others. Notebooks 02 and 03 skip training when the
metrics and probabilities are already on disk, so the whole set re-executes in
minutes rather than GPU-hours. Notebook 04 reads only what 02 and 03 wrote.

This mirrors experiments/paper_scale/{build_full_splits,run_full_scale}.py and
experiments/audit/ensemble_full_scale.py; it is the notebook presentation of the
same pipeline, not a second implementation with different numbers.
"""
import json
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / 'submission'
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- shared text

PREAMBLE = '''import os

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
'''

LOAD_SPLIT = '''def load_fixed_split(tag):
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
          f'test={n["test"]:6d}  total={sum(n.values()):6d}')
'''

TOKENISE = '''import torch
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
      '| bf16', torch.cuda.is_available() and torch.cuda.is_bf16_supported())
'''

TRAIN_HARNESS = '''import shutil
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
    return out
'''

REPORT_TABLE = '''def result_table(records, title):
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
'''


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.rstrip('\n'))


def write(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python',
                                 'name': 'python3'}
    nb.metadata['language_info'] = {'name': 'python'}
    path = OUT / name
    nbf.write(nb, str(path))
    print('wrote', path.relative_to(OUT.parents[2]), f'({len(cells)} cells)')


# ------------------------------------------------------- 01 preprocessing

nb1 = [
    md('''# 1. Data preprocessing

**NLP Final Term Project, Group 02 — detecting machine-generated text.**

This notebook turns two raw corpora into the one fixed, balanced, leakage-checked
split that every later notebook reuses. Nothing here trains a model.

| | source | task |
|---|---|---|
| D1 | DAIGT V2 (`daigt.csv`) | student essays, human against machine |
| D2 | HC3 (`hc3.jsonl`) | question answering, human against ChatGPT |

Steps, in order:

1. load each corpus and reduce it to `text` and a binary `label` (0 human, 1 machine),
2. balance the classes by downsampling the majority class,
3. hash the normalised text so near-identical documents form groups,
4. split 72 / 8 / 20 into train / validation / test **without letting a duplicate group
   cross a partition boundary**,
5. assert that no group crossed, and record the label balance of each partition,
6. build the two cleaning paths the models need: a classical lemmatised bag-of-words
   path, and the subword tokenisation the transformers consume.

Writes `experiments/paper_scale/work/data_{tag}.parquet` and `split_{tag}.npz`.
Notebooks 02, 03 and 04 read those files and never rebuild them, so the evaluation
set is identical across every result in the report.'''),
    md('''## 1.1 Environment and paths'''),
    code(PREAMBLE + '''
WORK_DIR.mkdir(parents=True, exist_ok=True)
print('work directory:', WORK_DIR)'''),
    md('''## 1.2 Loading and class balancing

Both corpora arrive unbalanced. HC3 additionally arrives one row per *question*, with
a list of human answers and a list of ChatGPT answers in that row, so it is exploded
to one answer per row before anything else happens.

Balancing is a downsample of the majority class to the minority count, at a fixed
seed. Accuracy on a balanced test set is then directly interpretable: 0.5 is chance.'''),
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
    md('''## 1.3 Duplicate-group-aware split

HC3 contains a substantial number of near-identical answers: 7.16 percent of the
corpus is a repeat of some other row once whitespace and case are normalised
(measured in `experiments/audit/hc3_full_audit.json`). A plain stratified split puts
copies of the same answer on both sides of the train/test boundary, and the test
score then partly measures memorisation.

So rows are grouped by the MD5 of their normalised lowercased text, and
`GroupShuffleSplit` keeps a whole group on one side. Measured effect on HC3: the
group-aware split leaks **0 of 10,732** test rows, the naive split leaks **570 of
10,762**, which is 5.30 percent. DAIGT's duplication rate is 0.01 percent, so
grouping is very nearly a no-op there, but it is applied to both datasets for
consistency.

The shares are 72 / 8 / 20, not 80 / 10 / 10: the first split takes a fifth for
test, the second takes a tenth of the remaining 80 percent for validation.'''),
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
    md('''## 1.4 Split integrity

Three things are checked, because each of them would quietly inflate the reported
scores if it were wrong: partition sizes, class balance inside every partition, and
whether any normalised document appears in more than one partition.'''),
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
    md('''## 1.5 Classical cleaning path

The classical baselines (Naive Bayes, logistic regression, linear SVM over
bag-of-words and TF-IDF) need a heavier normalisation than the transformers do:
lowercase, strip everything that is not a letter, drop English stopwords and
one-character tokens, and lemmatise.

This path is **not** applied before the transformers. Casing, punctuation and
function words carry signal that a subword model can use, and discarding them
before fine-tuning would throw away part of what the model is being asked to
detect.'''),
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
    md('''## 1.6 Transformer tokenisation

Each transformer uses its own subword vocabulary, so tokenisation happens per model:
WordPiece for BERT, SentencePiece for DeBERTa-v3. Sequences are truncated at 128
tokens and padded per batch rather than to a global maximum, which keeps the padded
fraction low.

128 is a project constraint carried over from the midterm, not a free choice. The
cell below reports what it costs: the share of documents that reach the limit and
are therefore cut.'''),
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
    md('''## 1.7 What this notebook produced

`data_D1.parquet`, `split_D1.npz`, `data_D2.parquet`, `split_D2.npz` in
`experiments/paper_scale/work/`.

Notebook 02 (BERT), notebook 03 (DeBERTa) and notebook 04 (ensemble) all read these
same four files. Because the split is fixed and shared, the two models are scored on
byte-identical test rows, which is what makes the paired ensemble comparison in
notebook 04 legitimate.'''),
    code('''for tag in ('D1', 'D2'):
    for name in (f'data_{tag}.parquet', f'split_{tag}.npz'):
        p = WORK_DIR / name
        print(f'{name:24s} {p.stat().st_size / 1024 ** 2:8.2f} MB   {p}')'''),
]

# ------------------------------------------------------------ 02 BERT

nb2 = [
    md('''# 2. BERT at its best configuration

**NLP Final Term Project, Group 02.**

Fine-tunes `bert-base-uncased` on both datasets, at the configuration that scored
highest on **validation** weighted F1 in the hyperparameter grid.

The grid was eight configurations per model per dataset: learning rate in
{2e-5, 3e-5}, batch size in {16, 32}, weight decay in {0.01, 0.1}; five epochs with
early stopping on validation F1, patience 2; maximum sequence length 128. The winner
was picked on validation and only then run against test, once. The test set selects
nothing.

| dataset | learning rate | batch size | weight decay |
|---|---|---|---|
| D1 DAIGT V2 | 3e-5 | 32 | 0.1 |
| D2 HC3 | 2e-5 | 16 | 0.1 |

**Requires** `01_preprocessing.ipynb` to have run. If the metrics and probabilities
for a configuration already exist on disk, the run is loaded rather than repeated,
so this notebook re-executes in seconds; delete the matching files in
`experiments/paper_scale/results/` and `probs/`, or pass `force=True`, to retrain.'''),
    md('''## 2.1 Environment and paths'''),
    code(PREAMBLE),
    md('''## 2.2 The fixed split from notebook 01'''),
    code(LOAD_SPLIT),
    md('''## 2.3 Tokenisation'''),
    code(TOKENISE),
    md('''## 2.4 Training harness

Weighted precision, recall and F1 are used throughout. The classes are balanced by
construction, so weighted and macro averaging agree closely, but weighted is what the
midterm reported and the two halves of the project stay comparable this way.

Early stopping monitors validation F1 with patience 2, and the best checkpoint by
that metric is what gets evaluated, never the last epoch.'''),
    code(TRAIN_HARNESS),
    md('''## 2.5 Best BERT configuration, per dataset'''),
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
    md('''## 2.6 Results'''),
    code(REPORT_TABLE + '''
bert_table = result_table(BERT_RESULT, 'BERT, best configuration per dataset\\n')'''),
    code('''show_confusion(BERT_RESULT)'''),
    md('''## 2.7 What this notebook produced

For each dataset, three artefacts keyed by the configuration string
`full_{tag}_BERT_lr{lr}_bs{bs}_wd{wd}_s42`:

* `experiments/paper_scale/results/{key}.json` — the metrics quoted in the report,
* `experiments/paper_scale/probs/{key}.npz` — validation and test class probabilities,
* `experiments/paper_scale/models/{tag}_BERT/` — the fine-tuned weights and tokenizer.

Notebook 04 reads the `.npz` probabilities. Nothing is recomputed there, so the
ensemble is built from exactly these predictions.'''),
    code('''for tag, rec in BERT_RESULT.items():
    print(f'{tag}  key={rec["key"]}')
    print(f'    results  {(RESULTS_DIR / (rec["key"] + ".json")).exists()}'
          f'   probs {(PROBS_DIR / (rec["key"] + ".npz")).exists()}'
          f'   model {(MODELS_DIR / f"{tag}_BERT").exists()}')'''),
]

# --------------------------------------------------------- 03 DeBERTa

nb3 = [
    md('''# 3. DeBERTa-v3, the BERT variant, at its best configuration

**NLP Final Term Project, Group 02.**

The variant is `microsoft/deberta-v3-base`. It differs from BERT in three ways that
matter here: disentangled attention, which scores content and relative position
separately rather than adding a position embedding into the token embedding; an
enhanced mask decoder that reinstates absolute position at the output layer; and
replaced-token-detection pretraining instead of masked language modelling. It is the
same parameter scale as `bert-base-uncased`, so the comparison is a like-for-like one
of pretraining and attention design rather than of model size.

The grid, the selection rule and the split are identical to notebook 02: eight
configurations, chosen on **validation** weighted F1, tested once.

| dataset | learning rate | batch size | weight decay |
|---|---|---|---|
| D1 DAIGT V2 | 3e-5 | 16 | 0.01 |
| D2 HC3 | 3e-5 | 16 | 0.1 |

**Requires** `01_preprocessing.ipynb`. Completed runs are loaded from disk rather
than repeated.'''),
    md('''## 3.1 Environment and paths'''),
    code(PREAMBLE),
    md('''## 3.2 The fixed split from notebook 01

The same partitions BERT saw, loaded from the same two files. This is what allows the
two models to be compared row by row in notebook 04.'''),
    code(LOAD_SPLIT),
    md('''## 3.3 Tokenisation

DeBERTa-v3 uses a SentencePiece vocabulary of about 128k pieces, against BERT's 30k
WordPiece. The same document therefore becomes a different number of tokens under the
two tokenizers, which is why notebook 01 reports the length diagnostic separately for
each.'''),
    code(TOKENISE),
    md('''## 3.4 Training harness

Byte-identical to the harness in notebook 02, including the early-stopping rule and
the cache check. Only the model key and the configuration change.'''),
    code(TRAIN_HARNESS),
    md('''## 3.5 Best DeBERTa configuration, per dataset'''),
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
    md('''## 3.6 Results'''),
    code(REPORT_TABLE + '''
deberta_table = result_table(DEBERTA_RESULT, 'DeBERTa-v3, best configuration per dataset\\n')'''),
    code('''show_confusion(DEBERTA_RESULT)'''),
    md('''## 3.7 Side by side with BERT

Read from the stored BERT records, so this cell does not depend on notebook 02 still
being in memory. It only runs if notebook 02 has been executed at least once.'''),
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
    md('''## 3.8 What this notebook produced

`results/{key}.json`, `probs/{key}.npz` and `models/{tag}_DeBERTa/` for each dataset,
under the key `full_{tag}_DeBERTa_lr{lr}_bs{bs}_wd{wd}_s42`. Notebook 04 consumes the
probability files.'''),
    code('''for tag, rec in DEBERTA_RESULT.items():
    print(f'{tag}  key={rec["key"]}')
    print(f'    results  {(RESULTS_DIR / (rec["key"] + ".json")).exists()}'
          f'   probs {(PROBS_DIR / (rec["key"] + ".npz")).exists()}'
          f'   model {(MODELS_DIR / f"{tag}_DeBERTa").exists()}')'''),
]

# -------------------------------------------------------- 04 ensemble

nb4 = [
    md('''# 4. Ensemble of BERT and DeBERTa

**NLP Final Term Project, Group 02.**

A weighted soft-vote over the two fine-tuned models:

$$P_{\\text{ensemble}} = w \\cdot P_{\\text{BERT}} + (1 - w) \\cdot P_{\\text{DeBERTa}}$$

Soft rather than hard voting, because with only two members a majority vote has no
way to break a one-to-one tie.

Procedure, and the order matters:

1. take the two best configurations, each already selected on validation in notebooks
   02 and 03,
2. sweep `w` from 0 to 1 in steps of 0.05 and keep whichever maximises **validation**
   weighted F1,
3. apply that single fixed `w` to the test set exactly once,
4. compare the ensemble against its stronger member with a paired test, since both
   models predicted the same test rows.

The test set is used once, for reporting. It selects nothing.

**Requires** notebooks 02 and 03 to have run; this notebook trains nothing and reads
only their saved probability files.'''),
    md('''## 4.1 Environment and paths'''),
    code(PREAMBLE),
    md('''## 4.2 Load the two members' saved probabilities

Keys are rebuilt from the same configuration dictionaries that notebooks 02 and 03
used, so the ensemble is guaranteed to be built from the deployed checkpoints and not
from some other cell of the grid. The label vectors are asserted equal, which is the
check that the two members really were scored on the same rows.'''),
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
    md('''## 4.3 Metrics and the paired test

McNemar's exact test asks only about the rows where the two systems disagree: of
those, is the split between them further from even than chance would allow. The
paired bootstrap resamples test rows 10,000 times and reports a 95 percent interval
on the difference in error rate. Both are paired, which is the right family of test
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
    md('''## 4.4 Select the mixing weight on validation, then test once'''),
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
    md('''## 4.5 The validation weight sweep

Printed in full because the shape of this curve is the evidence for how the weight
was chosen, and on HC3 it is also the evidence that the flat region is wide.'''),
    code('''curve = pd.DataFrame(VAL_CURVE)
curve.index.name = 'w_bert'
curve.columns = [f'{c} {DATASET_NAMES[c]} val_f1' for c in curve.columns]
print(curve.to_string())'''),
    md('''## 4.6 Ensemble results'''),
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
improve on this is more diversity between members, not a different mixing rule -- a
third family with a different inductive bias, or the same architecture trained on
different folds of the data.

`docs/ENSEMBLE_EXPLAINED.md` carries the longer version of this argument. It was
written against the earlier small-scale sweep, where DAIGT selected a 0.35 / 0.65 mix
and the ensemble sat just below DeBERTa; its reasoning still holds, but the DAIGT
numbers there are superseded by the full-scale ones reported above.'''),
    code('''print(json.dumps(ENSEMBLE, indent=2, default=float))'''),
]

write('01_preprocessing.ipynb', nb1)
write('02_bert_best_config.ipynb', nb2)
write('03_deberta_best_config.ipynb', nb3)
write('04_ensemble.ipynb', nb4)
