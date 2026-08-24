"""Publication figures for the ICCIT submission.

Reads only committed artefacts, so every figure traces to a file on disk:
  audit/full_model_evaluation.json / full_model_scores.npz   (all models, ROC)
  audit/surface_content_decomposition.json                   (decomposition)

Emits PDF at IEEE column widths into paper/iccit/figures/. Colours are chosen from a
colourblind-safe qualitative set and every series is additionally distinguished
by line style, so the figures survive greyscale printing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

FINAL = Path(__file__).resolve().parents[2]
AUDIT = FINAL / 'experiments' / 'audit'
FIGS = FINAL / 'paper' / 'iccit' / 'figures'

COL_W, DBL_W = 3.5, 7.16          # IEEE single / double column, inches
DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}

# colourblind-safe (Okabe-Ito), plus distinct dashes for greyscale
STYLE = {
    'Naive Bayes (BoW)':            ('#E69F00', (0, (1, 1))),
    'Naive Bayes (TF-IDF)':         ('#E69F00', (0, (3, 1, 1, 1))),
    'Logistic Regression (BoW)':    ('#56B4E9', (0, (1, 1))),
    'Logistic Regression (TF-IDF)': ('#56B4E9', (0, (3, 1, 1, 1))),
    'Support Vector Machine (BoW)': ('#009E73', (0, (1, 1))),
    'Support Vector Machine (TF-IDF)': ('#009E73', (0, (3, 1, 1, 1))),
    'BERT':                         ('#D55E00', (0, ())),
    'DeBERTa':                      ('#0072B2', (0, ())),
}
SHORT = {
    'Naive Bayes (BoW)': 'NB/BoW', 'Naive Bayes (TF-IDF)': 'NB/TF-IDF',
    'Logistic Regression (BoW)': 'LR/BoW', 'Logistic Regression (TF-IDF)': 'LR/TF-IDF',
    'Support Vector Machine (BoW)': 'SVM/BoW',
    'Support Vector Machine (TF-IDF)': 'SVM/TF-IDF',
    'BERT': 'BERT', 'DeBERTa': 'DeBERTa',
}

plt.rcParams.update({
    'font.size': 7, 'axes.labelsize': 7, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})


def fig_roc(ev, sc):
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.45))
    for ax, (tag, name) in zip(axes, DATASETS.items()):
        y = sc[f'{tag}|y_true']
        for mname, m in ev['datasets'][tag]['models'].items():
            key = f'{tag}|{mname}'
            if key not in sc:
                continue
            fpr, tpr, _ = roc_curve(y, sc[key])
            c, ls = STYLE.get(mname, ('#888888', (0, ())))
            lw = 1.3 if mname in ('BERT', 'DeBERTa') else 0.9
            ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=lw,
                    label=f'{SHORT.get(mname, mname)} ({m["roc_auc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='0.6', linewidth=0.6, linestyle=(0, (4, 3)))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
        ax.set_xlabel('False positive rate')
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}', loc='left')
        ax.legend(loc='lower right', frameon=False, handlelength=2.4)
    axes[0].set_ylabel('True positive rate')
    out = FIGS / 'fig_roc.pdf'
    fig.savefig(out); plt.close(fig)
    print('wrote', out)

    # zoomed companion: everything saturates near the corner
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.45))
    for ax, (tag, name) in zip(axes, DATASETS.items()):
        y = sc[f'{tag}|y_true']
        for mname, m in ev['datasets'][tag]['models'].items():
            key = f'{tag}|{mname}'
            if key not in sc:
                continue
            fpr, tpr, _ = roc_curve(y, sc[key])
            c, ls = STYLE.get(mname, ('#888888', (0, ())))
            lw = 1.3 if mname in ('BERT', 'DeBERTa') else 0.9
            ax.plot(fpr, tpr, color=c, linestyle=ls, linewidth=lw,
                    label=f'{SHORT.get(mname, mname)} ({m["roc_auc"]:.4f})')
        ax.set_xlim(0, 0.15); ax.set_ylim(0.85, 1.002)
        ax.set_xlabel('False positive rate')
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}, detail', loc='left')
        ax.legend(loc='lower right', frameon=False, handlelength=2.4)
    axes[0].set_ylabel('True positive rate')
    out = FIGS / 'fig_roc_zoom.pdf'
    fig.savefig(out); plt.close(fig)
    print('wrote', out)


def fig_confusion(ev):
    order = ['Naive Bayes (BoW)', 'Logistic Regression (BoW)',
             'Support Vector Machine (TF-IDF)', 'BERT', 'DeBERTa']
    fig, axes = plt.subplots(2, len(order), figsize=(DBL_W, 2.95))
    for r, (tag, name) in enumerate(DATASETS.items()):
        for c, mname in enumerate(order):
            ax = axes[r, c]
            m = ev['datasets'][tag]['models'].get(mname)
            if m is None:
                ax.axis('off'); continue
            tn, fp, fn, tp = m['confusion_tn_fp_fn_tp']
            cm = np.array([[tn, fp], [fn, tp]], float)
            cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap='Blues', vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f'{int(cm[i,j])}\n{cmn[i,j]*100:.1f}%',
                            ha='center', va='center', fontsize=5.4,
                            color='white' if cmn[i, j] > 0.5 else 'black')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['hum', 'mach'], fontsize=5.5)
            ax.set_yticklabels(['hum', 'mach'], fontsize=5.5)
            ax.grid(False)
            if r == 0:
                ax.set_title(SHORT.get(mname, mname), fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f'{name}\ntrue', fontsize=6.5)
            if r == 1:
                ax.set_xlabel('predicted', fontsize=6)
    out = FIGS / 'fig_confusion.pdf'
    fig.savefig(out); plt.close(fig)
    print('wrote', out)


def fig_decomposition(dec):
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    arms = [('surface_only', 'surface-only', '#E69F00'),
            ('content_only', 'content-only', '#0072B2')]
    x = np.arange(len(DATASETS)); w = 0.34
    for i, (k, lbl, col) in enumerate(arms):
        vals = [dec['datasets'][t][k]['error_rate'] * 100 for t in DATASETS]
        b = ax.bar(x + (i - 0.5) * w, vals, w, label=lbl, color=col, edgecolor='none')
        ax.bar_label(b, fmt='%.2f%%', fontsize=6, padding=1.5)
    for i, t in enumerate(DATASETS):
        ref = dec['datasets'][t]['full_transformer_reference']
        if ref:
            best = min(r['error_rate'] for r in ref.values()) * 100
            ax.hlines(best, x[i] - 0.55, x[i] + 0.55, color='#D55E00',
                      linewidth=1.2, linestyle=(0, (3, 2)),
                      label='full transformer' if i == 0 else None)
    ax.set_xticks(x); ax.set_xticklabels([d['name'] for d in dec['datasets'].values()])
    ax.set_ylabel('test error rate (%)')
    ax.set_ylim(0, 11.4)
    ax.legend(frameon=False, loc='upper center', ncol=3,
              columnspacing=1.0, handlelength=1.6, borderpad=0.2)
    out = FIGS / 'fig_decomposition.pdf'
    fig.savefig(out); plt.close(fig)
    print('wrote', out)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    ev = json.load(open(AUDIT / 'full_model_evaluation.json'))
    sc = dict(np.load(AUDIT / 'full_model_scores.npz', allow_pickle=False))
    dec = json.load(open(AUDIT / 'surface_content_decomposition.json'))
    fig_roc(ev, sc)
    fig_confusion(ev)
    fig_decomposition(dec)


if __name__ == '__main__':
    main()
