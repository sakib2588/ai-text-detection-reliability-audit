import json

CELLS = []
def md(s):   CELLS.append({"cell_type":"markdown","metadata":{},"source":s.strip("\n").split("\n")})
def code(s):
    lines = s.strip("\n").split("\n")
    CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],
                  "source":[l+"\n" for l in lines[:-1]]+[lines[-1]]})

md(r"""
## Section 4 — Full hyperparameter sweep

Eight configurations for each of two models on each of two datasets, thirty-two runs. BERT runs
first, so that an interruption still leaves one complete model family. Because every finished run is
recorded on disk, this cell is safe to re-run: it resumes wherever the previous pass stopped.
""")

code(r"""
SWEEP_START = time.time()
total = len(DATASETS) * len(MODELS) * len(GRID)
done = 0
for tag in DATASETS:
    for model_key in ('BERT', 'DeBERTa'):
        for lr, bs, wd in GRID:
            done += 1
            print('--- run %d/%d  %s %s lr=%g bs=%d wd=%g ---' % (done, total, tag, model_key, lr, bs, wd))
            run_one(tag, model_key, lr, bs, wd)
            elapsed = time.time() - SWEEP_START
            if done:
                print('    elapsed %.1f min, projected remaining %.1f min\n' % (
                    elapsed/60, elapsed/done*(total-done)/60))
print('sweep complete in %.2f hours' % ((time.time()-SWEEP_START)/3600))
""")

code(r"""
found = sorted(RESULTS_DIR.glob('*.json'))
sweep_runs = [json.load(open(p)) for p in found if '_len' not in p.stem]
sweep_runs = [r for r in sweep_runs if r['seed'] == SEED]
print('completed sweep runs on disk: %d (expected 32)' % len(sweep_runs))
assert len(sweep_runs) == 32, 'sweep incomplete, re-run the cell above to resume'
SWEEP = pd.DataFrame([{
    'dataset': r['dataset'], 'model': r['model'], 'lr': r['lr'], 'batch_size': r['batch_size'],
    'weight_decay': r['weight_decay'], 'epochs_run': r['epochs_run'],
    'val_f1': r['val']['f1'],
    'acc': r['test']['accuracy'], 'prec': r['test']['precision'],
    'rec': r['test']['recall'], 'f1': r['test']['f1'],
    'minutes': round(r['train_seconds']/60, 1), 'peak_vram': r['peak_vram_gib'],
} for r in sweep_runs])
SWEEP.to_csv(WORK_DIR / 'sweep_results.csv', index=False)
display(SWEEP.sort_values(['dataset','model','val_f1'], ascending=[True,True,False]))
""")

md(r"""
## Section 5 — Ensemble

The ensemble combines the two model families by averaging their predicted class probabilities. The
member configurations are chosen by **validation** weighted F1, and the mixing weight is likewise
tuned on validation only. The test set is used once, after both choices are frozen.

A weight sweep that terminates at zero or one means the ensemble has collapsed onto a single member.
That is reported as such rather than presented as an ensemble result.
""")

code(r"""
def load_probs(key):
    z = np.load(PROBS_DIR / (key + '.npz'))
    return {k: z[k] for k in z.files}

ENSEMBLE_ROWS, ENSEMBLE_DETAIL = {}, {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, tag in zip(axes, DATASETS):
    picks = {}
    for mk in ('BERT', 'DeBERTa'):
        cand = [r for r in sweep_runs if r['dataset'] == tag and r['model'] == mk]
        picks[mk] = max(cand, key=lambda r: r['val']['f1'])
    pb, pd_ = load_probs(picks['BERT']['key']), load_probs(picks['DeBERTa']['key'])
    assert np.array_equal(pb['val_labels'], pd_['val_labels'])
    assert np.array_equal(pb['test_labels'], pd_['test_labels'])

    ws = np.arange(0, 1.0001, 0.05)
    val_f1 = [weighted_metrics(pb['val_labels'],
              (w*pb['val_probs'] + (1-w)*pd_['val_probs']).argmax(1))[3] for w in ws]
    best_w = float(ws[int(np.argmax(val_f1))])
    test_mix = best_w*pb['test_probs'] + (1-best_w)*pd_['test_probs']
    acc, pre, rec, f1 = weighted_metrics(pb['test_labels'], test_mix.argmax(1))
    ENSEMBLE_ROWS[tag] = (round(acc,4), round(pre,4), round(rec,4), round(f1,4))
    ENSEMBLE_DETAIL[tag] = {
        'weight_bert': best_w, 'weight_deberta': round(1-best_w, 2),
        'member_bert': picks['BERT']['key'], 'member_deberta': picks['DeBERTa']['key'],
        'member_bert_test_f1': picks['BERT']['test']['f1'],
        'member_deberta_test_f1': picks['DeBERTa']['test']['f1'],
        'ensemble_test_f1': round(f1,4),
        'degenerate': bool(best_w in (0.0, 1.0)),
        'confusion': confusion_matrix(pb['test_labels'], test_mix.argmax(1)).tolist()}

    ax.plot(ws, val_f1, marker='o', ms=3, color='#4477aa')
    ax.axvline(best_w, color='#cc3311', ls='--', label='chosen w = %.2f' % best_w)
    ax.set_title('%s (%s)' % (tag, DATASET_NAMES[tag]))
    ax.set_xlabel('weight on BERT (1 - w on DeBERTa)'); ax.set_ylabel('validation weighted F1'); ax.legend()

    print('%s  members BERT=%.4f DeBERTa=%.4f -> ensemble test F1=%.4f  (w=%.2f)%s' % (
        tag, picks['BERT']['test']['f1'], picks['DeBERTa']['test']['f1'], f1, best_w,
        '   DEGENERATE: collapsed onto a single member' if best_w in (0.0,1.0) else ''))
plt.tight_layout(); plt.savefig(FIG_DIR / 'ensemble_weight_sweep.png', dpi=150); plt.show()
atomic_write_json(WORK_DIR / 'ensemble_detail.json', ENSEMBLE_DETAIL)
""")

md(r"""
## Section 6 — Seed robustness

The sweep runs one seed. With eight configurations differing by fractions of a percent, some of the
ordering will be seed noise rather than a real effect. The winning configuration for each model and
dataset is therefore repeated at two further seeds and reported as a mean with a range.

At three seeds this supports descriptive statistics only. No significance test is applied, and none
should be: the purpose is to show how wide the noise band is, so that a reader can judge which
differences in the main table are meaningful.
""")

code(r"""
SEED_ROWS = []
for tag in DATASETS:
    for mk in ('BERT', 'DeBERTa'):
        cand = [r for r in sweep_runs if r['dataset'] == tag and r['model'] == mk]
        best = max(cand, key=lambda r: r['val']['f1'])
        f1s = [best['test']['f1']]
        for s in (123, 456):
            f1s.append(run_one(tag, mk, best['lr'], best['batch_size'], best['weight_decay'], seed=s)['test']['f1'])
        SEED_ROWS.append({'dataset': tag, 'model': mk,
                          'config': 'lr=%g bs=%d wd=%g' % (best['lr'], best['batch_size'], best['weight_decay']),
                          'seeds': '42/123/456', 'f1_values': [round(v,4) for v in f1s],
                          'mean_f1': round(float(np.mean(f1s)), 4),
                          'min_f1': round(float(np.min(f1s)), 4),
                          'max_f1': round(float(np.max(f1s)), 4),
                          'spread': round(float(np.max(f1s) - np.min(f1s)), 4)})
SEED_DF = pd.DataFrame(SEED_ROWS)
SEED_DF.to_csv(WORK_DIR / 'seed_robustness.csv', index=False)
display(SEED_DF)
print('\nWidest seed spread: %.4f F1. Differences in the main table smaller than this '
      'should not be read as real.' % SEED_DF['spread'].max())
""")

md(r"""
## Section 7 — Result tables

**Table 1** is the experiment table from the project specification: eight configurations per model
per dataset, plus the ensemble row.

**Table 2** is the final combined table. The classical rows are the midterm results, recomputed here
from scratch rather than copied. Each classical model is represented by its stronger text
representation; the transformer rows use the configuration with the best validation F1. Every cell in
both tables is read from a file on disk, none are entered by hand.
""")

code(r"""
def fmt(v): return '%.4f' % v

t1 = []
for model_key in ('BERT', 'DeBERTa'):
    label = 'BERT' if model_key == 'BERT' else 'Your BERT Variation (DeBERTa-v3)'
    for lr, bs, wd in GRID:
        row = {'Model': label, 'Learning Rate': '%.5f' % lr, 'Batch Size': bs, 'Weight Decay': wd}
        for tag in DATASETS:
            m = SWEEP[(SWEEP.dataset==tag) & (SWEEP.model==model_key) & (SWEEP.lr==lr) &
                      (SWEEP.batch_size==bs) & (SWEEP.weight_decay==wd)].iloc[0]
            for col, src in (('Acc','acc'), ('Prec','prec'), ('Rec','rec'), ('F1','f1')):
                row['%s %s' % (tag, col)] = fmt(m[src])
        t1.append(row)
erow = {'Model': 'ENSEMBLE', 'Learning Rate': '', 'Batch Size': '', 'Weight Decay': ''}
for tag in DATASETS:
    for col, v in zip(('Acc','Prec','Rec','F1'), ENSEMBLE_ROWS[tag]):
        erow['%s %s' % (tag, col)] = fmt(v)
t1.append(erow)
TABLE1 = pd.DataFrame(t1)
TABLE1.to_csv(FINAL_DIR / 'table1_experiments.csv', index=False)
display(TABLE1)
""")

code(r"""
BEST_REP = {'Naive Bayes': 'BoW', 'Logistic Regression': 'BoW', 'Support Vector Machine': 'TF-IDF'}
t2 = []
for name, rep in BEST_REP.items():
    row = {'Model': '%s (%s)' % (name, rep)}
    for tag in DATASETS:
        for col, v in zip(('Acc','Prec','Rec','F1'), MIDTERM_ROWS[(tag, name, rep)]):
            row['%s %s' % (tag, col)] = fmt(v)
    t2.append(row)
for model_key, label in (('BERT','BERT'), ('DeBERTa','Your BERT Variation (DeBERTa-v3)')):
    row = {'Model': label}
    for tag in DATASETS:
        cand = [r for r in sweep_runs if r['dataset']==tag and r['model']==model_key]
        best = max(cand, key=lambda r: r['val']['f1'])
        row['_cfg_%s' % tag] = 'lr=%g bs=%d wd=%g' % (best['lr'], best['batch_size'], best['weight_decay'])
        for col, k in (('Acc','accuracy'), ('Prec','precision'), ('Rec','recall'), ('F1','f1')):
            row['%s %s' % (tag, col)] = fmt(best['test'][k])
    t2.append(row)
row = {'Model': 'ENSEMBLE'}
for tag in DATASETS:
    for col, v in zip(('Acc','Prec','Rec','F1'), ENSEMBLE_ROWS[tag]):
        row['%s %s' % (tag, col)] = fmt(v)
t2.append(row)
TABLE2 = pd.DataFrame(t2)
cfg_cols = [c for c in TABLE2.columns if c.startswith('_cfg_')]
TABLE2_DISPLAY = TABLE2.drop(columns=cfg_cols)
TABLE2_DISPLAY.to_csv(FINAL_DIR / 'table2_combined.csv', index=False)
display(TABLE2_DISPLAY)
print('\nSelected transformer configurations (chosen on validation F1):')
for _, r in TABLE2[TABLE2['Model'].str.contains('BERT')].iterrows():
    print('  %-34s D1: %-24s D2: %s' % (r['Model'], r.get('_cfg_D1',''), r.get('_cfg_D2','')))
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, tag in zip(axes, DATASETS):
    piv = SWEEP[SWEEP.dataset==tag].pivot_table(
        index='model', columns=['lr','batch_size','weight_decay'], values='val_f1')
    im = ax.imshow(piv.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(['%g\n%d\n%g' % c for c in piv.columns], fontsize=7)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_title('%s (%s) validation F1' % (tag, DATASET_NAMES[tag]))
    ax.set_xlabel('lr / batch / weight decay', fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, '%.3f' % piv.values[i,j], ha='center', va='center',
                    color='w', fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig(FIG_DIR / 'validation_f1_heatmap.png', dpi=150); plt.show()
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, tag in zip(axes, DATASETS):
    cm = np.array(ENSEMBLE_DETAIL[tag]['confusion'])
    ax.imshow(cm, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                    color='w' if cm[i,j] > cm.max()/2 else 'k', fontsize=13)
    ax.set_xticks([0,1]); ax.set_xticklabels(['human','ai'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['human','ai'])
    ax.set_xlabel('predicted'); ax.set_ylabel('actual')
    ax.set_title('ENSEMBLE on %s (%s)' % (tag, DATASET_NAMES[tag]))
plt.tight_layout(); plt.savefig(FIG_DIR / 'ensemble_confusion.png', dpi=150); plt.show()
""")

code(r"""
summary = {
    'split': {tag: {k: int(len(v)) for k, v in SPLITS[tag].items()} for tag in DATASETS},
    'midterm_reproduced': True,
    'length_stats': LENGTH_STATS.to_dict('records'),
    'ensemble': ENSEMBLE_DETAIL,
    'seed_robustness': SEED_ROWS,
    'n_sweep_runs': len(sweep_runs),
    'transformers_version': transformers.__version__,
    'torch_version': torch.__version__,
    'gpu': torch.cuda.get_device_name(0),
}
atomic_write_json(FINAL_DIR / 'run_summary.json', summary)
print('Artefacts written to', FINAL_DIR)
for p in sorted(FINAL_DIR.glob('*.csv')) + sorted(FINAL_DIR.glob('*.json')):
    print('  ', p.name)
print('Figures:')
for p in sorted(FIG_DIR.glob('*.png')):
    print('  ', p.name)
print('\nBest test F1 per dataset:')
for tag in DATASETS:
    b = SWEEP[SWEEP.dataset==tag].sort_values('f1', ascending=False).iloc[0]
    print('  %s  %s lr=%g bs=%d wd=%g -> %.4f  (ensemble %.4f)' % (
        tag, b['model'], b['lr'], b['batch_size'], b['weight_decay'], b['f1'], ENSEMBLE_ROWS[tag][3]))
""")

json.dump(CELLS, open('/media/filwel/All/Sakib/Semester 10/ NATURAL LANGUAGE PROCESSING /Final/work/cells_part3.json','w'))
print('part3 cells:', len(CELLS))
