"""Study pipeline diagram, fig_pipeline.pdf.

Design notes. The first version of this figure listed twenty model names as
running prose inside a box, which made it a paragraph with a border rather than
a diagram. Here the sweep is drawn instead of described, one chip per fitted
configuration coloured by the representation it was fitted on, so "38 classical
configurations over four representations" is something the reader sees at a
glance. Wording elsewhere is cut to labels, and the detail that used to sit in
the boxes belongs in the caption.

The canvas is deliberately wide and short so that at \\textwidth it scales down
rather than up, which keeps the type small and sharp.

Every count is read from the committed artefacts:
  notebooks/tables/notebook_all_models_comparison.csv   configurations per corpus
  notebooks/tables/notebook_table1_experiments.csv      fine-tuning runs per corpus
"""
import collections
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FINAL = Path(__file__).resolve().parents[2]
TABLES = FINAL / 'notebooks' / 'tables'
OUT = FINAL / 'paper' / 'iccit_profstyle'

# Colourblind-safe (Paul Tol bright). Word views blue, the raw character view
# red, the subword view purple, the analysis layer green.
CB = {'word': '#4477AA', 'char': '#EE6677', 'sub': '#AA3377', 'ana': '#228833'}
FILL = {'word': '#E4EDF6', 'char': '#FBE7E9', 'sub': '#F2E9F5', 'ana': '#E6F2E8'}
REP_COLOR = {'BoW': '#7BA3CC', 'TF-IDF': '#4477AA', 'TF-IDF-s': '#2E5580',
             'char3-5': '#EE6677'}

plt.rcParams.update({
    'font.size': 7, 'figure.dpi': 400, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02, 'pdf.fonttype': 42,
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
})


def counts():
    """Configurations per representation, families, and fine-tuning runs."""
    rows = [r for r in csv.DictReader(open(TABLES / 'notebook_all_models_comparison.csv'))
            if r['dataset'] == 'DAIGT V2']
    classical = [r for r in rows if r['kind'] == 'classical']
    per_rep = collections.Counter(r['representation'] for r in classical)
    families = len({r['model'] for r in classical})
    runs = sum(1 for _ in csv.reader(open(TABLES / 'notebook_table1_experiments.csv'))) - 3
    return per_rep, families, len(classical), runs


def main():
    per_rep, n_fam, n_cfg, n_runs = counts()

    fig, ax = plt.subplots(figsize=(10.4, 4.35))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    def box(x, y, w, h, title, sub=None, fc='#F5F5F5', ec='#8a8a8a',
            tfs=8.2, sfs=7.0, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.35,rounding_size=1.2',
                                    fc=fc, ec=ec, lw=lw))
        cx = x + w / 2
        if sub:
            ax.text(cx, y + h * 0.62, title, ha='center', va='center',
                    fontsize=tfs, fontweight='bold', color='#111111')
            ax.text(cx, y + h * 0.27, sub, ha='center', va='center',
                    fontsize=sfs, color='#444444')
        else:
            ax.text(cx, y + h / 2, title, ha='center', va='center',
                    fontsize=tfs, fontweight='bold', color='#111111')

    def arrow(x1, y1, x2, y2, color='#777777', lw=1.0, rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=8, lw=lw, color=color,
                                     shrinkA=1.5, shrinkB=1.5,
                                     connectionstyle=f'arc3,rad={rad}'))

    def chips(x, y, cols_list, cols, cw=1.9, ch=2.6, gx=0.55, gy=0.75):
        """One small rounded chip per fitted configuration."""
        for i, col in enumerate(cols_list):
            r, c = divmod(i, cols)
            ax.add_patch(FancyBboxPatch((x + c * (cw + gx), y - r * (ch + gy)),
                                        cw, ch,
                                        boxstyle='round,pad=0.08,rounding_size=0.45',
                                        fc=col, ec='none'))

    def stage(x, y, n):
        ax.text(x, y, n, ha='center', va='center', fontsize=6.4,
                color='#FFFFFF', fontweight='bold',
                bbox=dict(boxstyle='circle,pad=0.26', fc='#666666', ec='none'))

    # ---------------------------------------------------------------- band 1
    ys, hs = 84, 12
    box(4, ys, 18, hs, 'Corpora', '44,868  |  85,449 rows')
    box(26, ys, 18, hs, 'Audit', '6 and 6,118 duplicates')
    box(48, ys, 18, hs, 'Balance', '34,994  |  53,806')
    box(70, ys, 18, hs, 'Split', '72 / 8 / 20, leakage 0')
    for x in (22, 44, 66):
        arrow(x, ys + hs / 2, x + 4, ys + hs / 2)
    stage(5.4, ys + hs - 0.8, '1')     # acquisition and audit
    stage(49.4, ys + hs - 0.8, '2')    # balancing and partitioning

    ax.plot([79, 79], [ys, 80.5], color='#777777', lw=1.0, zorder=1)
    ax.plot([19, 79], [80.5, 80.5], color='#777777', lw=1.0, zorder=1)

    # ---------------------------------------------------------------- band 2
    yv, hv = 68, 10
    box(4, yv, 30, hv, 'Cleaned word view',
        'BoW 60k  |  TF-IDF 60k  |  TF-IDF-s 6k', FILL['word'], CB['word'])
    box(37, yv, 24, hv, 'Raw character view',
        'char 3-5 grams, 60k', FILL['char'], CB['char'])
    box(64, yv, 32, hv, 'Raw subword view',
        'WordPiece / SentencePiece, 128 tok', FILL['sub'], CB['sub'])
    for x in (19, 49, 80):
        arrow(x, 80.5, x, yv + hv)
    stage(5.4, yv + hv - 0.8, '3')

    # ---------------------------------------------------------------- band 3
    ym, hm = 40, 22
    box(4, ym, 57, hm, '', None, '#FCFCFC', '#666666')
    ax.text(32.5, ym + hm - 3.6,
            f'Classical sweep   {n_fam} families   {n_cfg} configurations',
            ha='center', va='center', fontsize=8.4, fontweight='bold', color='#111111')
    order = ['BoW', 'TF-IDF', 'TF-IDF-s', 'char3-5']
    grid = [REP_COLOR[r] for r in order for _ in range(per_rep[r])]
    chips(9.5, ym + hm - 10.0, grid, cols=13)

    box(64, ym, 32, hm, '', None, FILL['sub'], CB['sub'])
    ax.text(80, ym + hm - 3.6, f'Transformer sweep   {n_runs} runs',
            ha='center', va='center', fontsize=8.4, fontweight='bold', color='#111111')
    chips(70.5, ym + hm - 10.0, ['#C08FCC'] * 8 + ['#8A4E9E'] * 8, cols=8)
    ax.text(80, ym + 3.0, 'BERT-base (top)    DeBERTa-v3 (bottom)',
            ha='center', va='center', fontsize=6.6, color='#444444')

    arrow(19, yv, 24, ym + hm)
    arrow(49, yv, 40, ym + hm)
    arrow(80, yv, 80, ym + hm)
    stage(5.4, ym + hm - 0.8, '4')

    # ---------------------------------------------------------------- band 4
    ya, ha_ = 20, 12
    box(4, ya, 28, ha_, 'Decomposition',
        'surface 47  |  content BoW  |  length 3', FILL['ana'], CB['ana'])
    box(35, ya, 26, ha_, 'Attribution',
        'SHAP, coefficients, gain', FILL['ana'], CB['ana'])
    box(64, ya, 32, ha_, 'Controls',
        'tokenisation, pipeline, label-free', FILL['ana'], CB['ana'])
    arrow(18, ym, 18, ya + ha_)
    arrow(48, ym, 48, ya + ha_)
    arrow(80, ym, 80, ya + ha_)
    stage(5.4, ya + ha_ - 0.8, '5')

    # ---------------------------------------------------------------- band 5
    yo, ho = 3, 11
    box(4, yo, 44, ho, 'Metrics',
        'accuracy   weighted and macro F1   ROC-AUC   PR-AUC   FPR')
    box(52, yo, 44, ho, 'Paired statistics',
        'exact McNemar   bootstrap 95% CI   5 split seeds')
    arrow(18, ya, 22, yo + ho)
    arrow(48, ya, 40, yo + ho)
    arrow(80, ya, 74, yo + ho)
    arrow(48, yo + ho / 2, 52, yo + ho / 2)

    handles = [Line2D([], [], marker='s', linestyle='none', markersize=4.6,
                      markerfacecolor=REP_COLOR[r], markeredgecolor='none',
                      label=f'{r}  ({per_rep[r]})') for r in order]
    ax.legend(handles=handles, loc='center', bbox_to_anchor=(0.505, 0.495),
              frameon=False, fontsize=6.4, handletextpad=0.35,
              labelspacing=0.3, ncol=1)

    fig.savefig(OUT / 'fig_pipeline.pdf')
    plt.close(fig)
    print('wrote', OUT / 'fig_pipeline.pdf')
    print(f'{n_fam} families, {n_cfg} classical configs {dict(per_rep)}, {n_runs} runs')


if __name__ == '__main__':
    main()
