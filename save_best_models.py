import os
os.environ.setdefault('HF_HOME', '/media/filwel/MLProject1/hf_cache')
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import gc, json, re, shutil, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, EarlyStoppingCallback, DataCollatorWithPadding, set_seed)
from datasets import Dataset
warnings.filterwarnings('ignore')

FINAL = Path('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final')
WORK, RESULTS = FINAL/'work', FINAL/'results'
MODELS_DIR = FINAL/'models'; MODELS_DIR.mkdir(exist_ok=True)
CKPT = Path('/media/filwel/MLProject1/nlp_final_ckpt/save_best')
NAMES = {'BERT': 'bert-base-uncased', 'DeBERTa': 'microsoft/deberta-v3-base'}
MAX_LEN, EPOCHS, WARMUP, PATIENCE = 128, 5, 0.1, 2

def wm(y, p):
    a = accuracy_score(y, p)
    pr, rc, f, _ = precision_recall_fscore_support(y, p, average='weighted', zero_division=0)
    return a, pr, rc, f

def compute_metrics(ep):
    lg, lb = ep
    a, pr, rc, f = wm(lb, np.argmax(lg, axis=-1))
    return {'accuracy': a, 'precision': pr, 'recall': rc, 'f1': f}

def normalise(t): return re.sub(r'\s+', ' ', str(t)).strip()

runs = [json.load(open(p)) for p in RESULTS.glob('*.json')]
runs = [r for r in runs if r['seed'] == 42 and '_ep' not in r['key'] and '_len' not in r['key']]
print('sweep runs available: %d' % len(runs))

winners = []
for tag in ('D1', 'D2'):
    for mk in ('BERT', 'DeBERTa'):
        c = [r for r in runs if r['dataset'] == tag and r['model'] == mk]
        winners.append(max(c, key=lambda r: r['val']['f1']))

print('\nwinning configurations, selected on validation F1:')
for w in winners:
    print('  %s %-8s lr=%g bs=%d wd=%g  recorded test F1=%.4f' % (
        w['dataset'], w['model'], w['lr'], w['batch_size'], w['weight_decay'], w['test']['f1']))

manifest = []
for w in winners:
    tag, mk = w['dataset'], w['model']
    outdir = MODELS_DIR / ('%s_%s' % (tag, mk))
    if (outdir/'config.json').exists():
        print('\n[skip] %s already saved' % outdir.name); continue
    print('\n=== training %s %s for saving ===' % (tag, mk))
    df = pd.read_parquet(WORK/('data_%s.parquet' % tag))
    sp = np.load(WORK/('split_%s.npz' % tag))
    tok = AutoTokenizer.from_pretrained(NAMES[mk])
    parts = {}
    for split in ('train', 'val', 'test'):
        sub = df.loc[sp[split]]
        ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                                'labels': [int(v) for v in sub['label']]})
        parts[split] = ds.map(lambda b: tok(b['text'], truncation=True, max_length=MAX_LEN),
                              batched=True, remove_columns=['text'])
    set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained(NAMES[mk], num_labels=2)
    model.config.id2label = {0: 'human', 1: 'ai'}
    model.config.label2id = {'human': 0, 'ai': 1}
    args = TrainingArguments(
        output_dir=str(CKPT/outdir.name), learning_rate=w['lr'],
        per_device_train_batch_size=w['batch_size'], per_device_eval_batch_size=64,
        weight_decay=w['weight_decay'], num_train_epochs=EPOCHS, warmup_ratio=WARMUP,
        lr_scheduler_type='linear', optim='adamw_torch', bf16=True,
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model='eval_f1', greater_is_better=True,
        logging_steps=100, seed=42, data_seed=42, dataloader_num_workers=0, report_to='none')
    tr = Trainer(model=model, args=args, train_dataset=parts['train'], eval_dataset=parts['val'],
                 data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
                 callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)])
    t0 = time.time(); tr.train(); secs = time.time()-t0

    pred = tr.predict(parts['test'])
    raw = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
    yhat = np.asarray(raw).argmax(1)
    a, pr, rc, f = wm(np.asarray(pred.label_ids), yhat)

    outdir.mkdir(parents=True, exist_ok=True)
    tr.save_model(str(outdir)); tok.save_pretrained(str(outdir))
    rec = {'dir': outdir.name, 'dataset': tag, 'dataset_name': w['dataset_name'], 'model': mk,
           'checkpoint': NAMES[mk], 'lr': w['lr'], 'batch_size': w['batch_size'],
           'weight_decay': w['weight_decay'], 'seed': 42, 'max_len': MAX_LEN,
           'recorded_test_f1': w['test']['f1'], 'reproduced_test_f1': round(f, 4),
           'reproduced_test_accuracy': round(a, 4),
           'matches_recorded': abs(round(f, 4) - w['test']['f1']) < 1e-4,
           'train_seconds': round(secs, 1), 'labels': {'0': 'human', '1': 'ai'}}
    json.dump(rec, open(outdir/'run_info.json', 'w'), indent=2)
    manifest.append(rec)
    print('  saved -> %s  reproduced test F1 %.4f vs recorded %.4f  %s' % (
        outdir, f, w['test']['f1'], 'MATCH' if rec['matches_recorded'] else 'DIFFERS'))
    del tr, model; gc.collect(); torch.cuda.empty_cache()
    shutil.rmtree(CKPT/outdir.name, ignore_errors=True)

if manifest:
    json.dump(manifest, open(MODELS_DIR/'manifest.json', 'w'), indent=2)
print('\ndone. models in %s' % MODELS_DIR)
