"""Column-width versions of the four result figures, for the page-limited cut.

The double-column originals in make_paper_figures.py and make_visual_figures.py
are 7.16 in wide. Five full-width floats will not pack into six pages, so these
are redrawn at the 3.5 in IEEE column, laid out for that width rather than
scaled down to it: fewer panels side by side, short model names, and type sized
so nothing falls below 5 pt on the page.

Reads only committed artefacts, so every number traces to a file:
  experiments/audit/full_model_evaluation.json    confusion counts, AUC
  experiments/audit/full_model_scores.npz         ROC curves
  experiments/audit/shap_surface_features.json    recorded Shapley means
  notebooks/tables/notebook_all_models_comparison.csv   the family sweep
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "experiments" / "audit"
TABLES = ROOT / "notebooks" / "tables"
OUT = ROOT / "paper" / "iccit6_profstyle"

COL_W = 3.5
DATASETS = {"D1": "DAIGT V2", "D2": "HC3"}
SHORT = {
    "Naive Bayes (BoW)": "NB/BoW", "Naive Bayes (TF-IDF)": "NB/TF-IDF",
    "Logistic Regression (BoW)": "LR/BoW",
    "Logistic Regression (TF-IDF)": "LR/TF-IDF",
    "Support Vector Machine (BoW)": "SVM/BoW",
    "Support Vector Machine (TF-IDF)": "SVM/TF-IDF",
    "BERT": "BERT", "DeBERTa": "DeBERTa",
}
STYLE = {
    "Naive Bayes (BoW)": ("#E69F00", (0, (1, 1))),
    "Naive Bayes (TF-IDF)": ("#E69F00", (0, (3, 1, 1, 1))),
    "Logistic Regression (BoW)": ("#56B4E9", (0, (1, 1))),
    "Logistic Regression (TF-IDF)": ("#56B4E9", (0, (3, 1, 1, 1))),
    "Support Vector Machine (BoW)": ("#009E73", (0, (1, 1))),
    "Support Vector Machine (TF-IDF)": ("#009E73", (0, (3, 1, 1, 1))),
    "BERT": ("#D55E00", (0, ())),
    "DeBERTa": ("#0072B2", (0, ())),
}

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "axes.titlesize": 6.5,
    "xtick.labelsize": 5.4, "ytick.labelsize": 5.4, "legend.fontsize": 4.9,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.35,
    "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.012, "pdf.fonttype": 42,
})


def fig_sweep():
    """Twenty-one families, both corpora, as one dot plot instead of two bar panels."""
    best = defaultdict(dict)
    for r in csv.DictReader(open(TABLES / "notebook_all_models_comparison.csv")):
        d, m, f = r["dataset"], r["model"], float(r["f1_weighted"])
        if f > best[d].get(m, (0.0,))[0]:
            best[d][m] = (f, r["kind"])
    order = sorted(best["DAIGT V2"], key=lambda m: -best["DAIGT V2"][m][0])
    y = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(COL_W, 2.95))
    for d, mk, fc in (("DAIGT V2", "o", "#0072B2"), ("HC3", "s", "#D55E00")):
        xs = [best[d][m][0] for m in order]
        ax.scatter(xs, y, s=6.5, marker=mk, facecolor=fc, edgecolor="none",
                   label=d, zorder=3)
    for i, m in zip(y, order):
        a, b = best["DAIGT V2"][m][0], best["HC3"][m][0]
        ax.plot([min(a, b), max(a, b)], [i, i], color="0.72", lw=0.5, zorder=2)
    trans = [m for m in order if best["DAIGT V2"][m][1] == "transformer"]
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"$\\bf{{{m}}}$" if m in trans else m for m in order], fontsize=4.9)
    ax.set_xlim(0.84, 1.005)
    ax.set_xlabel("best weighted F1 over the four representations", labelpad=1.5)
    # the lower-left corner holds data rows, so the key goes outside the axes
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
              frameon=False, handletextpad=0.3, borderpad=0.0, columnspacing=1.2)
    ax.grid(axis="y", alpha=0.18, linewidth=0.3)
    fig.savefig(OUT / "classical_vs_transformer.pdf")
    plt.close(fig)
    print("wrote classical_vs_transformer.pdf")


def fig_roc(ev, sc):
    """The two zoomed panels stacked, so each keeps the full column width."""
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 3.05), sharex=True)
    for ax, (tag, name) in zip(axes, DATASETS.items()):
        y = sc[f"{tag}|y_true"]
        for mname, m in ev["datasets"][tag]["models"].items():
            key = f"{tag}|{mname}"
            if key not in sc:
                continue
            fpr, tpr, _ = roc_curve(y, sc[key])
            c, ls = STYLE[mname]
            ax.plot(fpr, tpr, color=c, linestyle=ls,
                    linewidth=1.15 if mname in ("BERT", "DeBERTa") else 0.75,
                    label=f'{SHORT[mname]} ({m["roc_auc"]:.4f})')
        ax.set_xlim(0, 0.15)
        ax.set_ylim(0.85, 1.003)
        ax.set_ylabel("true positive rate")
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}, detail',
                     loc="left")
        ax.legend(loc="lower right", frameon=False, ncol=2, handlelength=1.9,
                  columnspacing=0.8, handletextpad=0.35, borderpad=0.15)
    axes[1].set_xlabel("false positive rate")
    fig.tight_layout(pad=0.25, h_pad=0.6)
    fig.savefig(OUT / "fig_roc_zoom.pdf")
    plt.close(fig)
    print("wrote fig_roc_zoom.pdf")


def fig_confusion(ev):
    """Three representative configurations rather than five, so the cells stay readable."""
    order = ["Naive Bayes (BoW)", "Support Vector Machine (TF-IDF)", "DeBERTa"]
    fig, axes = plt.subplots(2, 3, figsize=(COL_W, 2.25))
    for r, (tag, name) in enumerate(DATASETS.items()):
        for c, mname in enumerate(order):
            ax = axes[r, c]
            tn, fp, fn, tp = ev["datasets"][tag]["models"][mname]["confusion_tn_fp_fn_tp"]
            cm = np.array([[tn, fp], [fn, tp]], float)
            cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cmn[i, j] * 100:.1f}%", ha="center",
                            va="center", fontsize=4.8,
                            color="white" if cmn[i, j] > 0.5 else "black")
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["hum", "mach"], fontsize=4.6)
            ax.set_yticklabels(["hum", "mach"], fontsize=4.6)
            ax.tick_params(length=1.2, pad=1.0)
            ax.grid(False)
            if r == 0:
                ax.set_title(SHORT[mname], fontsize=5.6, pad=2.0)
            if c == 0:
                ax.set_ylabel(f"{name}\ntrue", fontsize=5.4)
            if r == 1:
                ax.set_xlabel("predicted", fontsize=5.2, labelpad=1.0)
    fig.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.5)
    fig.savefig(OUT / "fig_confusion.pdf")
    plt.close(fig)
    print("wrote fig_confusion.pdf")


def fig_shap():
    """Mean absolute Shapley value per feature. A beeswarm is unreadable at 3.5 in,
    and the mean is the quantity the text actually quotes."""
    rec = json.load(open(AUDIT / "shap_surface_features.json"))
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 2.05))
    for ax, (tag, name) in zip(axes, DATASETS.items()):
        feats = rec[tag]["top_features"][:10][::-1]
        y = np.arange(len(feats))
        vals = [f["mean_abs_shap"] for f in feats]
        cols = ["#D55E00" if f["drives"] == "machine-generated" else "#0072B2"
                for f in feats]
        ax.barh(y, vals, color=cols, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels([f["feature"] for f in feats], fontsize=4.6)
        ax.set_title(f'({"ab"[list(DATASETS).index(tag)]}) {name}', loc="left")
        ax.set_xlabel("mean $|$SHAP$|$", fontsize=5.4)
        ax.tick_params(axis="x", labelsize=4.8)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(pad=0.2, w_pad=1.0)
    fig.savefig(OUT / "fig_shap_beeswarm.pdf")
    plt.close(fig)
    print("wrote fig_shap_beeswarm.pdf")


def main():
    ev = json.load(open(AUDIT / "full_model_evaluation.json"))
    sc = np.load(AUDIT / "full_model_scores.npz")
    fig_sweep()
    fig_roc(ev, sc)
    fig_confusion(ev)
    fig_shap()


if __name__ == "__main__":
    main()
