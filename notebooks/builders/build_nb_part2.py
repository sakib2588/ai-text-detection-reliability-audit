import json

CELLS = []
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def code(s):
    lines = s.strip("\n").split("\n")
    CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                  "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]})

md(r"""
## Section 2 — Tokenisation and sequence-length diagnostics

Maximum sequence length is fixed at 128 by the project specification. DAIGT essays average roughly
2,200 characters, which is well beyond that, so the figure and table below quantify how much of each
document actually reaches the model. This is the leading limitation of the study and is reported
rather than worked around.
""")

code(r"""
TOKENIZERS, TOKENIZED = {}, {}

def get_tokenizer(model_key):
    if model_key not in TOKENIZERS:
        TOKENIZERS[model_key] = AutoTokenizer.from_pretrained(MODELS[model_key])
    return TOKENIZERS[model_key]

def normalise(t):
    return re.sub(r'\s+', ' ', str(t)).strip()

def get_tokenized(tag, model_key, max_len=MAX_LEN):
    cache_key = (tag, model_key, max_len)
    if cache_key in TOKENIZED:
        return TOKENIZED[cache_key]
    tok = get_tokenizer(model_key)
    df = DATA[tag]
    parts = {}
    for split, idx in SPLITS[tag].items():
        sub = df.loc[idx]
        ds = Dataset.from_dict({'text': [normalise(t) for t in sub['text']],
                                'labels': [int(v) for v in sub['label']]})
        ds = ds.map(lambda b: tok(b['text'], truncation=True, max_length=max_len),
                    batched=True, remove_columns=['text'])
        parts[split] = ds
    TOKENIZED[cache_key] = parts
    return parts
print('tokenisation helpers ready')
""")

code(r"""
rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, tag in zip(axes, DATASETS):
    tok = get_tokenizer('BERT')
    texts = [normalise(t) for t in DATA[tag]['text']]
    lens = np.array([len(tok(t, truncation=False)['input_ids']) for t in texts])
    ax.hist(np.clip(lens, 0, 1024), bins=60, color='#4477aa', edgecolor='none')
    ax.axvline(MAX_LEN, color='#cc3311', linestyle='--', label='max_length = %d' % MAX_LEN)
    ax.set_title('%s (%s)' % (tag, DATASET_NAMES[tag]))
    ax.set_xlabel('WordPiece tokens (clipped at 1024)'); ax.set_ylabel('documents'); ax.legend()
    rows.append({'dataset': tag, 'name': DATASET_NAMES[tag], 'median': int(np.median(lens)),
                 'mean': round(float(lens.mean()), 1), 'p95': int(np.percentile(lens, 95)),
                 'max': int(lens.max()),
                 'pct_truncated': round(float((lens > MAX_LEN).mean() * 100), 1),
                 'median_pct_kept': round(float(np.median(np.minimum(lens, MAX_LEN) / lens) * 100), 1)})
    del texts, lens; gc.collect()
plt.tight_layout(); plt.savefig(FIG_DIR / 'token_length_distribution.png', dpi=150)
plt.show()
LENGTH_STATS = pd.DataFrame(rows)
LENGTH_STATS.to_csv(WORK_DIR / 'length_stats.csv', index=False)
display(LENGTH_STATS)
""")

md(r"""
## Section 3 — Run harness with checkpointing

The sweep is thirty-two fine-tuning runs on a desktop GPU that also drives the display, so it has to
survive interruption. Three independent mechanisms provide that.

**Run-level resumption.** Every finished run writes a metrics JSON keyed by its full configuration.
On re-entry, a run whose JSON already exists returns immediately without touching the GPU. Re-running
the whole sweep cell after a completed pass therefore costs seconds, not hours.

**Within-run resumption.** The `Trainer` saves a checkpoint at the end of every epoch into a directory
private to that run. If the process dies mid-run, the next invocation detects the surviving
`checkpoint-*` directories and passes `resume_from_checkpoint`, so training restarts from the last
completed epoch with its optimizer state, learning-rate schedule, and early-stopping counters intact
rather than starting over.

**Atomic writes.** Metrics and probability arrays are written to a temporary file and then moved into
place with `os.replace`, which is atomic on a single filesystem. A process killed during the write
leaves the temporary file behind and no partial JSON, so a half-written run can never be mistaken for
a completed one on the next pass.

The checkpoint directory is deleted only after the JSON has landed, in that order. Deleting first
would create a window in which the run is neither resumable nor recorded. A `save_total_limit` of two
bounds the cost at roughly 4 GB for a DeBERTa run, and only one run is ever active.

A ledger file records the outcome of each run as it completes, giving a durable record of sweep
progress that survives a kernel restart.
""")

code(r"""
def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def atomic_write_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix('.npz.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrays)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)

def run_key(tag, model_key, lr, bs, wd, seed, max_len):
    base = '%s_%s_lr%g_bs%d_wd%g_s%d' % (tag, model_key, lr, bs, wd, seed)
    return base if max_len == MAX_LEN else base + '_len%d' % max_len

def find_checkpoint(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    cks = [p for p in run_dir.glob('checkpoint-*') if (p / 'trainer_state.json').exists()]
    if not cks:
        return None
    return str(max(cks, key=lambda p: int(p.name.split('-')[1])))

def ledger_append(entry):
    path = WORK_DIR / 'sweep_ledger.json'
    log = json.load(open(path)) if path.exists() else []
    log = [e for e in log if e['key'] != entry['key']] + [entry]
    atomic_write_json(path, log)

def cleanup_stale_tmp():
    n = 0
    for d in (RESULTS_DIR, PROBS_DIR):
        for p in list(d.glob('*.tmp')):
            p.unlink(); n += 1
    if n:
        print('removed %d stale temporary file(s) from an interrupted write' % n)
cleanup_stale_tmp()
print('checkpointing helpers ready')
""")

code(r"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc, pre, rec, f1 = weighted_metrics(labels, preds)
    return {'accuracy': acc, 'precision': pre, 'recall': rec, 'f1': f1}

def run_one(tag, model_key, lr, bs, wd, seed=SEED, epochs=EPOCHS, max_len=MAX_LEN, verbose=True):
    key = run_key(tag, model_key, lr, bs, wd, seed, max_len)
    jpath = RESULTS_DIR / (key + '.json')
    if jpath.exists():
        rec = json.load(open(jpath))
        if verbose:
            print('[skip] %-42s test_f1=%.4f' % (key, rec['test']['f1']))
        return rec

    run_dir = CKPT_DIR / key
    resume = find_checkpoint(run_dir)
    if resume and verbose:
        print('[resume] %s from %s' % (key, Path(resume).name))

    set_seed(seed)
    parts = get_tokenized(tag, model_key, max_len)
    tok = get_tokenizer(model_key)
    model = AutoModelForSequenceClassification.from_pretrained(MODELS[model_key], num_labels=2)

    per_device_bs, accum = bs, 1
    if model_key == 'DeBERTa' and bs == 32:
        per_device_bs, accum = 16, 2

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=lr,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=accum,
        per_device_eval_batch_size=64,
        weight_decay=wd,
        num_train_epochs=epochs,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type='linear',
        optim='adamw_torch',
        bf16=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        greater_is_better=True,
        logging_steps=50,
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
        report_to='none',
        disable_tqdm=not verbose,
    )
    trainer = Trainer(
        model=model, args=args,
        train_dataset=parts['train'], eval_dataset=parts['val'],
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    train_secs = time.time() - t0

    out = {'key': key, 'dataset': tag, 'dataset_name': DATASET_NAMES[tag], 'model': model_key,
           'checkpoint': MODELS[model_key], 'lr': lr, 'batch_size': bs,
           'per_device_batch_size': per_device_bs, 'gradient_accumulation_steps': accum,
           'weight_decay': wd, 'seed': seed, 'max_len': max_len, 'epochs_requested': epochs,
           'train_seconds': round(train_secs, 1),
           'epochs_run': int(trainer.state.epoch or 0),
           'best_checkpoint_metric': trainer.state.best_metric,
           'peak_vram_gib': round(torch.cuda.max_memory_allocated() / 1024**3, 2),
           'resumed': bool(resume)}

    probs = {}
    for split in ('val', 'test'):
        pred = trainer.predict(parts[split])
        logits = torch.tensor(pred.predictions, dtype=torch.float32)
        p = torch.softmax(logits, dim=-1).numpy()
        y = np.asarray(pred.label_ids)
        acc, pre, rec, f1 = weighted_metrics(y, p.argmax(1))
        out[split] = {'accuracy': round(acc,4), 'precision': round(pre,4),
                      'recall': round(rec,4), 'f1': round(f1,4)}
        out[split + '_confusion'] = confusion_matrix(y, p.argmax(1)).tolist()
        _, _, mf1, _ = precision_recall_fscore_support(y, p.argmax(1), average='macro', zero_division=0)
        out[split + '_macro_f1'] = round(float(mf1), 4)
        probs['%s_probs' % split] = p
        probs['%s_labels' % split] = y

    atomic_write_npz(PROBS_DIR / (key + '.npz'), **probs)
    atomic_write_json(jpath, out)
    ledger_append({'key': key, 'test_f1': out['test']['f1'], 'val_f1': out['val']['f1'],
                   'train_seconds': out['train_seconds'], 'epochs_run': out['epochs_run']})

    del trainer, model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    shutil.rmtree(run_dir, ignore_errors=True)

    if verbose:
        print('[done] %-42s val_f1=%.4f test_f1=%.4f  %.1f min  peak %.2f GiB' % (
            key, out['val']['f1'], out['test']['f1'], train_secs/60, out['peak_vram_gib']))
    return out
print('run_one ready')
""")

md(r"""
### Section 3.1 — Smoke test

One configuration per model on Dataset 2 for a single epoch. This confirms the DeBERTa-v3 tokenizer
converts (it requires `sentencepiece` and `protobuf`), that bf16 training is numerically stable, that
the batch-32 memory path fits, and it produces a measured seconds-per-step figure to replace the
planning estimate. The smoke-test artefacts are deleted afterwards so that the full sweep does not
mistake them for finished runs.
""")

code(r"""
SMOKE = []
for mk in ('BERT', 'DeBERTa'):
    r = run_one('D2', mk, 2e-5, 32, 0.01, epochs=1)
    SMOKE.append(r)
    print('   %-8s peak VRAM %.2f GiB, %.1f s, val_f1 %.4f' % (
        mk, r['peak_vram_gib'], r['train_seconds'], r['val']['f1']))

est = sum(r['train_seconds'] for r in SMOKE) * EPOCHS * len(GRID) * len(DATASETS) / 3600
print('\nmeasured projection for the full 32-run sweep: %.1f hours (upper bound, early stopping will cut it)' % est)

for r in SMOKE:
    (RESULTS_DIR / (r['key'] + '.json')).unlink(missing_ok=True)
    (PROBS_DIR / (r['key'] + '.npz')).unlink(missing_ok=True)
(WORK_DIR / 'sweep_ledger.json').unlink(missing_ok=True)
print('smoke-test artefacts cleared, sweep will run these configurations properly')
""")

json.dump(CELLS, open('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/work/cells_part2.json','w'))
print('part2 cells:', len(CELLS))
