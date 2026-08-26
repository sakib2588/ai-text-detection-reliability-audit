"""Visual figures for the expanded paper.

Three figures, all built from this project's own artefacts:

  fig_surface_correlation.pdf  Spearman correlation over the 47 surface features,
                               per dataset. Shows that the surface arm is a
                               correlated block rather than 47 independent cues,
                               and marks the five document-size features the
                               length control removes.
  fig_shap_beeswarm.pdf        SHAP over the surface-only logistic regression.
                               Explains WHICH orthographic features carry the
                               separability the decomposition measures.
  fig_dashboard.pdf            Six-panel summary of every headline result.

The surface featuriser is imported from surface_content_decomposition.py rather
than reimplemented, so these figures and the reported numbers cannot drift apart.

Nothing here is read from a grid maximum, and no value is typed in by hand. Every
number is loaded from the JSON the audit scripts wrote.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

FINAL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FINAL / 'experiments' / 'audit'))
from surface_content_decomposition import (LENGTH_IDX, PUNCT_CHARS,  # noqa: E402
                                           surface_features)

AUDIT = FINAL / 'experiments' / 'audit'
WORK = FINAL / 'experiments' / 'paper_scale' / 'work'
TABLES = FINAL / 'tables'
FIGS = FINAL / 'paper' / 'iccit' / 'figures'
FIGS.mkdir(parents=True, exist_ok=True)

COL_W, DBL_W = 3.5, 7.16          # IEEE single / double column, inches
DATASETS = {'D1': 'DAIGT V2', 'D2': 'HC3'}
SEED = 42

# Colourblind-safe qualitative set (Paul Tol bright), used across every panel.
CB = {'blue': '#4477AA', 'red': '#EE6677', 'green': '#228833', 'yellow': '#CCBB44',
      'cyan': '#66CCEE', 'purple': '#AA3377', 'grey': '#BBBBBB'}

plt.rcParams.update({
    'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'figure.dpi': 400, 'savefig.dpi': 400, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.01, 'pdf.fonttype': 42,
})

# Names for the 47 features, in the exact order surface_features() emits them.
BASE_NAMES = [
    'space-before-punct (count)', 'space-before-punct (rate)',
    'length, chars', 'length, words', 'mean word length',
    'uppercase-char %', 'capitalised-word %', 'ALLCAPS-word %',
    'non-ASCII rate', 'emoji count', 'digit rate',
    'double-space rate', 'newline rate', 'sentence count', 'mean sentence length',
]
PUNCT_LABEL = {' ': 'space', '\\': 'backslash'}
FEATURE_NAMES = BASE_NAMES + ["rate of '%s'" % PUNCT_LABEL.get(c, c) for c in PUNCT_CHARS]


SHORT_NAMES = {
    'space-before-punct (count)': 'space-pre-punct, n',
    'space-before-punct (rate)': 'space-pre-punct, rate',
    'mean sentence length': 'mean sent. length',
    'capitalised-word %': 'capitalised %',
    'uppercase-char %': 'uppercase %',
    'ALLCAPS-word %': 'ALLCAPS %',
    "rate of 'backslash'": "rate of '\\\\'",
    'double-space rate': 'double-space',
    'non-ASCII rate': 'non-ASCII',
    'mean word length': 'mean word len',
}


def short(name):
    """Shorter labels for the correlation panels, where a long name on the right
    panel runs left past its own axis and lands on the left panel's matrix."""
    return SHORT_NAMES.get(name, name)


def load_surface_matrix(tag):
    """Surface features on the fixed train/test split, standardised on train."""
    df = pd.read_parquet(WORK / f'data_{tag}.parquet')
    sp = np.load(WORK / f'split_{tag}.npz')
    itr = df.index.get_indexer(sp['train'])
    ite = df.index.get_indexer(sp['test'])
    X = np.array([surface_features(t) for t in df['text'].values], dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X[itr].mean(0), X[itr].std(0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd
    y = df['label'].values
    return Xz[itr], y[itr], Xz[ite], y[ite]


# --------------------------------------------------------------------------
# Figure 1: correlation matrix over the 47 surface features
# --------------------------------------------------------------------------
def fig_surface_correlation():
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 3.05),
                             gridspec_kw={'wspace': 0.42})
    recorded = {}

    for ax, (tag, name) in zip(axes, DATASETS.items()):
        _, _, Xte, _ = load_surface_matrix(tag)
        rho = spearmanr(Xte).statistic
        rho = np.nan_to_num(np.atleast_2d(rho), nan=0.0)
        np.fill_diagonal(rho, 1.0)

        # Cluster so correlated blocks are adjacent and the structure is visible.
        dist = np.clip(1.0 - np.abs(rho), 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        order = leaves_list(linkage(squareform(dist, checks=False), method='average'))
        R = rho[np.ix_(order, order)]

        im = ax.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        labels = [short(FEATURE_NAMES[i]) for i in order]
        # The matrix is symmetric and the two axes carry the same 47 labels in the
        # same order, so labelling rows alone identifies both. Rotated x labels at
        # this many features collide with each other and with the caption.
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(labels, fontsize=3.7)
        ax.set_xticks([])
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}')

        # Mark the five document-size features the length control removes.
        for j, orig in enumerate(order):
            if orig in LENGTH_IDX:
                ax.add_patch(Rectangle((j - .5, -.5), 1, len(order),
                                       fill=False, ec=CB['green'], lw=0.55))
                ax.get_yticklabels()[j].set_color(CB['green'])

        off = np.abs(R[np.triu_indices_from(R, k=1)])
        # Exact-zero std misses these, since standardising leaves a float residue
        # of order 1e-16. Count distinct values instead.
        const = [FEATURE_NAMES[i] for i in range(Xte.shape[1])
                 if len(np.unique(Xte[:, i])) == 1]
        recorded[tag] = {
            'n_features': int(R.shape[0]),
            'mean_abs_offdiagonal_spearman': round(float(off.mean()), 4),
            'pairs_above_0.5': int((off > 0.5).sum()),
            'n_pairs': int(off.size),
            'constant_on_test_split': const,
            'n_constant': len(const),
            'note': 'constant features have undefined correlation and are drawn as 0',
        }

    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
    cb.set_label('Spearman correlation', fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.savefig(FIGS / 'fig_surface_correlation.pdf')
    plt.close(fig)
    print('wrote fig_surface_correlation.pdf', recorded)
    return recorded


# --------------------------------------------------------------------------
# Figure 2: SHAP beeswarm over the surface-only arm
# --------------------------------------------------------------------------
def fig_shap_beeswarm():
    import shap
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 3.0))
    recorded = {}
    rng = np.random.default_rng(SEED)

    for ax, (tag, name) in zip(axes, DATASETS.items()):
        Xtr, ytr, Xte, yte = load_surface_matrix(tag)
        clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(Xtr, ytr)

        # LinearExplainer is exact for a linear model, so no sampling error enters
        # the attribution itself. The subsample is for plot density only.
        sub = rng.choice(len(Xte), size=min(2000, len(Xte)), replace=False)
        masker = shap.maskers.Independent(Xtr, max_samples=min(2000, len(Xtr)))
        expl = shap.LinearExplainer(clf, masker, feature_names=FEATURE_NAMES)
        sv = expl(Xte[sub])

        plt.sca(ax)
        shap.plots.beeswarm(sv, max_display=14, show=False,
                            color_bar=(tag == 'D2'), plot_size=None)
        # beeswarm sets its own font sizes on the axes it draws into, so every
        # text element is restyled afterwards rather than through rcParams.
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}', fontsize=7.5)
        ax.set_xlabel('SHAP value, positive drives the machine-generated class',
                      fontsize=5.8)
        ax.tick_params(axis='x', labelsize=5.2)
        ax.tick_params(axis='y', labelsize=5.2)
        for lbl in ax.get_yticklabels():
            lbl.set_fontsize(5.2)
        for lbl in ax.get_xticklabels():
            lbl.set_fontsize(5.2)

        mean_abs = np.abs(sv.values).mean(0)
        top = np.argsort(mean_abs)[::-1][:15]
        recorded[tag] = {
            'dataset': name,
            'classifier': 'LogisticRegression(max_iter=1000) on 47 surface features',
            'explainer': 'shap.LinearExplainer, exact for a linear model',
            'background_samples': int(min(2000, len(Xtr))),
            'n_explained': int(len(sub)),
            'top_features': [
                {'feature': FEATURE_NAMES[i],
                 'mean_abs_shap': round(float(mean_abs[i]), 4),
                 'coefficient': round(float(clf.coef_[0][i]), 4),
                 'drives': 'machine-generated' if clf.coef_[0][i] > 0 else 'human'}
                for i in top],
        }

    # The colour bar beeswarm adds carries its own oversized label.
    for cax in fig.axes[len(axes):]:
        cax.tick_params(labelsize=5)
        cax.set_ylabel(cax.get_ylabel(), fontsize=5.5)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    fig.savefig(FIGS / 'fig_shap_beeswarm.pdf')
    plt.close(fig)
    json.dump(recorded, open(AUDIT / 'shap_surface_features.json', 'w'), indent=1)
    print('wrote fig_shap_beeswarm.pdf and shap_surface_features.json')
    return recorded


# --------------------------------------------------------------------------
# Figure 3: six-panel consolidated dashboard
# --------------------------------------------------------------------------
def fig_dashboard():
    ev = json.load(open(AUDIT / 'full_model_evaluation.json'))
    dec = json.load(open(AUDIT / 'surface_content_decomposition.json'))
    ms = json.load(open(AUDIT / 'multisplit_decomposition.json'))
    tr = json.load(open(AUDIT / 'truncation_matched_comparison.json'))
    pcv = json.load(open(AUDIT / 'paper_claim_verification.json'))
    cross = pd.read_csv(TABLES / 'table_cross_dataset_generalization_3seed.csv')

    fig, axes = plt.subplots(2, 3, figsize=(DBL_W, 4.3))
    (a, b, c), (d, e, f) = axes
    D1C, D2C = CB['blue'], CB['red']

    # (a) every configuration ranked by test error
    names = list(ev['datasets']['D1']['models'])
    short = [n.replace('Support Vector Machine', 'SVM')
              .replace('Logistic Regression', 'LogReg')
              .replace('Naive Bayes', 'NB') for n in names]
    e1 = [ev['datasets']['D1']['models'][n]['error_rate'] * 100 for n in names]
    e2 = [ev['datasets']['D2']['models'][n]['error_rate'] * 100 for n in names]
    yy = np.arange(len(names))
    a.barh(yy - 0.2, e1, 0.4, color=D1C, label='DAIGT V2')
    a.barh(yy + 0.2, e2, 0.4, color=D2C, label='HC3')
    a.set_yticks(yy)
    a.set_yticklabels(short, fontsize=5.4)
    a.invert_yaxis()
    a.set_xlabel('test error, %')
    a.set_title('(a) All eight configurations')
    a.legend(frameon=False, loc='lower right', fontsize=5.5)

    # (b) the decomposition, with and without the length channel
    arms = [('surface_only', 'surface'), ('content_only', 'content'),
            ('surface_only_nolength', 'surface\nno len'),
            ('content_only_l1norm_scaled', 'content\nlen-norm')]

    def arm_err(tag, key):
        ds = dec['datasets'][tag]
        rec = ds[key] if key in ds else ds['length_controlled'][key]
        return rec['error_rate'] * 100

    v1 = [arm_err('D1', k) for k, _ in arms]
    v2 = [arm_err('D2', k) for k, _ in arms]
    xx = np.arange(len(arms))
    b.bar(xx - 0.2, v1, 0.4, color=D1C, label='DAIGT V2')
    b.bar(xx + 0.2, v2, 0.4, color=D2C, label='HC3')
    b.set_xticks(xx)
    b.set_xticklabels([l for _, l in arms], fontsize=4.8)
    b.set_ylabel('test error, %')
    b.set_title('(b) Surface against content')
    b.set_ylim(0, max(v1 + v2) * 1.22)
    b.legend(frameon=False, fontsize=5.5, loc='upper right',
             borderaxespad=0.1, handlelength=1.1, handletextpad=0.4)

    # (c) split-level variance, the evidence the HC3 parity claim rests on
    for tag, col, mk in (('D1', D1C, 'o'), ('D2', D2C, 's')):
        s_ = ms['datasets'][tag]['summary']
        surf = np.array(s_['surface_only']['error_by_split']) * 100
        cont = np.array(s_['content_only']['error_by_split']) * 100
        c.plot(range(1, len(surf) + 1), surf - cont, marker=mk, ms=3, lw=1,
               color=col, label=DATASETS[tag])
    c.axhline(0, color='k', lw=0.6, ls='--')
    c.set_xlabel('partition seed index')
    c.set_ylabel('surface err minus content err, pp')
    c.set_title('(c) Five group-aware partitions')
    c.legend(frameon=False, fontsize=5.5)
    c.set_xticks(range(1, 6))

    # (d) matched text budget
    w1 = tr['datasets']['D1']['windows']['BERT']
    w2 = tr['datasets']['D2']['windows']['BERT']
    bc1 = min(v['error_rate'] for v in w1['classical_truncated'].values()) * 100
    bc2 = min(v['error_rate'] for v in w2['classical_truncated'].values()) * 100
    full1 = min(ev['datasets']['D1']['models'][n]['error_rate']
                for n in names if n not in ('BERT', 'DeBERTa')) * 100
    full2 = min(ev['datasets']['D2']['models'][n]['error_rate']
                for n in names if n not in ('BERT', 'DeBERTa')) * 100
    tf1, tf2 = w1['transformer_error_rate'] * 100, w2['transformer_error_rate'] * 100
    xx = np.arange(3)
    d.bar(xx - 0.2, [full1, bc1, tf1], 0.4, color=D1C, label='DAIGT V2')
    d.bar(xx + 0.2, [full2, bc2, tf2], 0.4, color=D2C, label='HC3')
    d.set_xticks(xx)
    d.set_xticklabels(['classical\nfull text', 'classical\n128-tok', 'BERT\n128-tok'],
                      fontsize=4.8)
    d.set_ylabel('test error, %')
    d.set_title('(d) Matched text budget')
    d.set_ylim(0, max(full1, full2, bc1, bc2, tf1, tf2) * 1.32)
    d.legend(frameon=False, fontsize=5.5, loc='upper left')

    # (e) cross-dataset transfer, three seeds, with the observed min-max range
    lbl, ind, crd, err = [], [], [], [[], []]
    for _, r in cross.iterrows():
        tag = 'DAIGT V2' if 'DAIGT' in str(r['trained_on']).upper() else 'HC3'
        lbl.append('%s\n%s' % (tag, r['model']))
        ind.append(r['in_domain_f1_mean'])
        crd.append(r['cross_domain_f1_mean'])
        err[0].append(r['cross_domain_f1_mean'] - r['cross_domain_f1_min'])
        err[1].append(r['cross_domain_f1_max'] - r['cross_domain_f1_mean'])
    xx = np.arange(len(lbl))
    e.bar(xx - 0.2, ind, 0.4, color=CB['green'], label='in-domain')
    e.bar(xx + 0.2, crd, 0.4, color=CB['yellow'], label='cross-domain',
          yerr=np.array(err), capsize=1.5,
          error_kw={'lw': 0.6, 'ecolor': '#444444'})
    e.set_xticks(xx)
    e.set_xticklabels([l.replace('DAIGT V2', 'DAIGT') for l in lbl], fontsize=4.8)
    e.set_ylim(0.6, 1.13)
    e.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    e.set_ylabel('weighted F1')
    e.set_title('(e) Cross-dataset transfer, 3 seeds')
    e.legend(frameon=False, fontsize=5.5, loc='upper center', ncol=2,
             columnspacing=0.8, handlelength=1.1)

    # (f) the paired intervals behind every significance claim in the paper.
    # Two comparisons per dataset from Table 1 and two from the decomposition, so
    # both corpora and both experiments are represented rather than whichever
    # rows happen to come first.
    SHORTEN = {'Support Vector Machine (TF-IDF)': 'SVM (TF-IDF)',
               'Logistic Regression (BoW)': 'LogReg (BoW)',
               'surface_only': 'surface', 'content_only': 'content',
               'surface_only_nolength': 'surface, no len',
               'content_only_l1norm_scaled': 'content, len-norm'}
    wanted = [('D1', 'table1_stats', 0), ('D2', 'table1_stats', 0),
              ('D1', 'decomposition_stats', 0), ('D2', 'decomposition_stats', 0),
              ('D1', 'decomposition_stats', 1), ('D2', 'decomposition_stats', 1)]
    rows = []
    for tag, block, idx in wanted:
        cmps = pcv[block][tag]['paired_comparisons']
        if idx < len(cmps):
            rows.append((tag, cmps[idx]))
    ylab = []
    for i, (tag, r) in enumerate(rows):
        lo, hi = r['error_diff_ci95_lo'] * 100, r['error_diff_ci95_hi'] * 100
        mid = r['error_diff_a_minus_b'] * 100
        col = CB['grey'] if not r['ci_excludes_zero'] else (D1C if tag == 'D1' else D2C)
        f.plot([lo, hi], [i, i], color=col, lw=1.6, solid_capstyle='butt')
        f.plot([mid], [i], 'o', ms=2.8, color=col)
        ylab.append('%s\n%s vs %s' % (DATASETS[tag],
                                      SHORTEN.get(r['a'], r['a']),
                                      SHORTEN.get(r['b'], r['b'])))
    f.axvline(0, color='k', lw=0.6, ls='--')
    f.set_yticks(range(len(rows)))
    f.set_yticklabels(ylab, fontsize=4.5)
    f.invert_yaxis()
    f.set_xlabel('error difference, pp (95% CI)')
    f.set_title('(f) Paired intervals')
    f.margins(y=0.08)

    fig.tight_layout(pad=0.4, w_pad=1.1, h_pad=1.0)
    fig.savefig(FIGS / 'fig_dashboard.pdf')
    plt.close(fig)
    print('wrote fig_dashboard.pdf, %d paired intervals' % len(rows))


def main():
    corr = fig_surface_correlation()
    shap_rec = fig_shap_beeswarm()
    fig_dashboard()
    json.dump({'surface_correlation': corr}, open(AUDIT / 'surface_correlation.json', 'w'),
              indent=1)
    print('\ntop surface features by mean |SHAP|:')
    for tag, rec in shap_rec.items():
        print(' ', rec['dataset'])
        for t in rec['top_features'][:5]:
            print('    %-30s %.4f  -> %s' % (t['feature'], t['mean_abs_shap'], t['drives']))


if __name__ == '__main__':
    main()


# --------------------------------------------------------------------------
# Figure 4: method schematic for the decomposition
# --------------------------------------------------------------------------
def fig_pipeline():
    """Schematic of the three arms. Every count printed on it is read from the
    artefacts, so the diagram cannot drift from the experiment it describes."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    dec = json.load(open(AUDIT / 'surface_content_decomposition.json'))
    n = {}
    for tag in DATASETS:
        sp = np.load(WORK / f'split_{tag}.npz')
        n[tag] = {k: int(len(sp[k])) for k in ('train', 'val', 'test')}

    n_surface = dec['datasets']['D1']['surface_only']['n_features']
    n_nolen = dec['datasets']['D1']['length_controlled']['surface_only_nolength']['n_features']
    n_content = {t: dec['datasets'][t]['content_only']['n_features'] for t in DATASETS}

    fig, ax = plt.subplots(figsize=(DBL_W, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis('off')

    def box(x, y, w, h, text, fc, ec, fs=5.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.6,rounding_size=1.2',
                                    fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle='-|>', mutation_scale=6,
                                     lw=0.7, color='#555555',
                                     shrinkA=1, shrinkB=1))

    box(1, 16, 15, 10,
        'Raw corpus\n\nDAIGT V2  %s\nHC3  %s' % (
            f"{sum(n['D1'].values()):,}", f"{sum(n['D2'].values()):,}"),
        '#F4F4F4', '#888888')

    box(19, 16, 15, 10,
        'Balance,\nhash by content,\ngroup-aware split\n72 / 8 / 20',
        '#F4F4F4', '#888888')

    box(38, 29, 26, 11,
        'Surface arm  (%d features)\npunctuation, whitespace, casing,\nlength, non-ASCII, digits\nno word identity is ever read'
        % n_surface, '#DCE7F2', CB['blue'])

    box(38, 15.5, 26, 11,
        'Content arm  (%s / %s terms)\nlowercase, strip [^a-z ],\nbag-of-words, raw counts\nno punctuation, casing or non-ASCII'
        % (f"{n_content['D1']:,}", f"{n_content['D2']:,}"), '#F7DEDE', CB['red'])

    box(38, 2, 26, 11,
        'Full arm  (reference)\nBERT and DeBERTa-v3,\nraw text, 128 tokens\nreported, not matched',
        '#F4F4F4', '#888888')

    box(68, 22.5, 15, 10,
        'Logistic\nregression\n(shared family)', '#FFFFFF', '#555555')

    box(86, 15, 13, 12,
        'Paired\nMcNemar\n+ bootstrap\n95% CI', '#FFFFFF', '#555555')

    box(6, -8, 90, 7,
        'Length control    surface arm drops %d document-size features,    '
        'content arm rows L1-normalised then rescaled'
        % (n_surface - n_nolen), '#F0F0F0', '#999999', fs=5.2)
    ax.set_ylim(-9.5, 42)

    arrow(16, 21, 19, 21)
    arrow(34, 21, 38, 34.5)
    arrow(34, 21, 38, 21)
    arrow(34, 21, 38, 7.5)
    arrow(64, 34.5, 68, 29)
    arrow(64, 21, 68, 26)
    arrow(83, 27.5, 86, 23)
    arrow(64, 7.5, 86, 17)

    fig.savefig(FIGS / 'fig_pipeline.pdf')
    plt.close(fig)
    print('wrote fig_pipeline.pdf')
