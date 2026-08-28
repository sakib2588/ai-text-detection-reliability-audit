"""Complete study pipeline diagram, fig_pipeline.pdf.

The earlier schematic drew only the surface-content decomposition and named one
classifier, logistic regression, which misrepresented a study that fits twenty
classical families over four representations plus two fine-tuned transformers.
This script draws the pipeline actually executed, end to end.

Counts on the diagram are taken from the committed artefacts:
  notebooks/tables/notebook_all_models_comparison.csv   40 rows per corpus
  notebooks/builders/build_analysis_notebook.py         the model zoo and reps
  paper corpus statistics                               row counts after balancing
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FINAL = Path(__file__).resolve().parents[2]
TABLES = FINAL / 'notebooks' / 'tables'
OUT = FINAL / 'paper' / 'iccit_profstyle'

DBL_W = 7.16  # IEEE double-column width, inches

CB = {'blue': '#4477AA', 'red': '#EE6677', 'green': '#228833',
      'yellow': '#CCBB44', 'purple': '#AA3377', 'grey': '#BBBBBB'}

plt.rcParams.update({
    'font.size': 7, 'figure.dpi': 400, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
})


def grid_counts():
    """Configurations per corpus, read from the results table rather than typed."""
    rows = list(csv.DictReader(open(TABLES / 'notebook_all_models_comparison.csv')))
    per = {}
    for d in ('DAIGT V2', 'HC3'):
        sub = [r for r in rows if r['dataset'] == d]
        per[d] = {
            'total': len(sub),
            'classical': len([r for r in sub if r['kind'] == 'classical']),
            'transformer': len([r for r in sub if r['kind'] == 'transformer']),
            'families': len({r['model'] for r in sub if r['kind'] == 'classical'}),
        }
    assert per['DAIGT V2'] == per['HC3'], per
    return per['DAIGT V2']


def main():
    g = grid_counts()

    fig, ax = plt.subplots(figsize=(DBL_W, 6.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, header, detail, fc, ec,
            header_fs=6.9, detail_fs=5.85, lw=0.9):
        """Draw a stage box. Text is laid out from the top down, so a box only
        has to be tall enough for its own lines."""
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.4,rounding_size=1.0',
                                    fc=fc, ec=ec, lw=lw))
        cx = x + w / 2
        nlines = header.count('\n') + 1
        top = y + h - 1.6
        ax.text(cx, top, header, ha='center', va='top', fontsize=header_fs,
                fontweight='bold', color='#141414', linespacing=1.3)
        if detail:
            ax.text(cx, top - nlines * 2.8 - 0.9, detail, ha='center', va='top',
                    fontsize=detail_fs, color='#2e2e2e', linespacing=1.5)

    def band(y, text):
        ax.text(-1.0, y, text, ha='center', va='center', fontsize=6.0,
                color='#8a8a8a', style='italic', rotation=90)

    def arrow(x1, y1, x2, y2, color='#6f6f6f', lw=0.85, rad=0.0, ls='-'):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=6.5, lw=lw, color=color,
                                     linestyle=ls, shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f'arc3,rad={rad}'))

    # ------------------------------------------------------------------ band 1
    band(89.5, 'data')
    box(3, 82, 21, 15, 'Corpora',
        'DAIGT V2, 44,868 rows\nHC3, 85,449 rows\nEnglish, human vs machine',
        '#F3F3F3', '#8a8a8a')
    box(27.5, 82, 22, 15, 'Contamination audit',
        'exact-match duplicates,\n6 and 6,118 rows\n483 artefact rows in HC3',
        '#F3F3F3', '#8a8a8a')
    box(53, 82, 21, 15, 'Class balancing',
        'downsample majority class\n34,994 and 53,806 rows',
        '#F3F3F3', '#8a8a8a')
    box(77.5, 82, 21, 15, 'Group-aware split',
        'MD5 groups kept whole\n72 / 8 / 20, leakage 0',
        '#F3F3F3', '#8a8a8a')
    arrow(24, 89.5, 27.5, 89.5)
    arrow(49.5, 89.5, 53, 89.5)
    arrow(74, 89.5, 77.5, 89.5)

    # ------------------------------------------------------------------ band 2
    band(70.5, 'representation')
    box(3, 63, 30, 15, 'Cleaned word view',
        'lowercased, stopwords removed,\nlemmatised, non-Latin dropped\n'
        'BoW 60k, TF-IDF 60k,\nreduced TF-IDF-s 6k',
        '#E7EEF6', CB['blue'])
    box(36.5, 63, 27, 15, 'Raw character view',
        'no cleaning at all\nTF-IDF over char_wb\n3-5 grams, 60k features',
        '#F6EAEA', CB['red'])
    box(67, 63, 31.5, 15, 'Raw subword view',
        'whitespace normalised only\nWordPiece and SentencePiece\n'
        'truncated at 128 tokens',
        '#EFE9F4', CB['purple'])
    # One bus out of the split, so the fan-out never crosses an upstream box.
    ax.plot([88, 88], [82, 80.2], color='#6f6f6f', lw=0.85,
            solid_capstyle='butt', zorder=1)
    ax.plot([18, 88], [80.2, 80.2], color='#6f6f6f', lw=0.85,
            solid_capstyle='butt', zorder=1)
    for x in (18, 50, 82.5):
        arrow(x, 80.2, x, 78)

    # ------------------------------------------------------------------ band 3
    band(50, 'models')
    box(3, 40, 60.5, 19,
        'Classical sweep, %d families, %d configurations per corpus'
        % (g['families'], g['classical']),
        'MultinomialNB, BernoulliNB, ComplementNB, LogReg [lbfgs, liblinear, saga],\n'
        'LinearSVC, RidgeClassifier, SGD [hinge, log_loss, modified huber],\n'
        'PassiveAggressive, DecisionTree [gini, entropy], RandomForest,\n'
        'ExtraTrees, AdaBoost, LightGBM, MLP [adam, sgd], each family fitted\n'
        'on every representation it supports',
        '#FBFBFB', '#5f5f5f', header_fs=6.8, detail_fs=5.7)
    box(67, 40, 31.5, 19, 'Transformer sweep',
        'bert-base-uncased and\ndeberta-v3-base, AdamW\n'
        'lr, batch size, weight decay,\n16 runs per corpus\n'
        'deployed checkpoints at 3 seeds',
        '#EFE9F4', CB['purple'])
    arrow(18, 63, 22, 59)
    arrow(50, 63, 44, 59)
    arrow(82.5, 63, 82.5, 59)

    # ------------------------------------------------------------------ band 4
    band(28, 'analysis')
    box(3, 18, 29, 19, 'Surface-content decomposition',
        'surface arm, 47 orthographic\nfeatures, never a word\n'
        'content arm, bag-of-words with\npunctuation and casing stripped\n'
        'length-only arm, 3 features',
        '#E7EEF6', CB['blue'], detail_fs=5.65)
    box(35.5, 18, 28, 19, 'Attribution and diagnostics',
        'SHAP over the surface arm and\nover LightGBM, linear coefficients,\n'
        'log-odds terms, learning curves,\noverfitting gap, class balance,\n'
        'length distributions, word clouds',
        '#EAF3EA', CB['green'], detail_fs=5.65)
    box(67, 18, 31.5, 19, 'Controls and stress tests',
        'tokenisation, pipeline-fires and\nlabel-free controls, whitespace\n'
        'cleaning, matched text budget,\ncross-corpus transfer, soft-vote\n'
        'ensemble, published detector',
        '#FCF4E0', CB['yellow'], detail_fs=5.65)
    arrow(17.5, 40, 17.5, 37)
    arrow(49.5, 40, 49.5, 37)
    arrow(82.5, 40, 82.5, 37)

    # ------------------------------------------------------------------ band 5
    band(8, 'evaluation')
    box(3, 1, 46, 13, 'Metrics on one fixed test partition',
        'accuracy, error rate, weighted and macro F1,\n'
        'per-class precision and recall, ROC-AUC,\nPR-AUC, false-positive rate, fit time',
        '#F3F3F3', '#5f5f5f')
    box(52.5, 1, 46, 13, 'Paired statistics',
        'exact-binomial McNemar on the same documents,\n'
        'paired bootstrap 95% interval, B = 10,000,\n'
        'decomposition repeated over 5 split seeds',
        '#F3F3F3', '#5f5f5f')
    arrow(17.5, 18, 20, 14)
    arrow(49.5, 18, 40, 14)
    arrow(82.5, 18, 78, 14)
    arrow(49, 7.5, 52.5, 7.5)

    fig.savefig(OUT / 'fig_pipeline.pdf')
    plt.close(fig)
    print('wrote', OUT / 'fig_pipeline.pdf')
    print('grid read from artefacts:', g)


if __name__ == '__main__':
    main()
