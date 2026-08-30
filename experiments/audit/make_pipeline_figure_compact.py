"""Compact study pipeline for the six-page cut, fig_pipeline_compact.pdf.

The long report's diagram (experiments/audit/make_pipeline_figure.py) draws the
whole sweep, one chip per fitted configuration, over five numbered stages. At
0.6 textwidth in a six-page paper its type falls below legibility and it carries
detail the short paper never discusses. This version keeps only what the cut
argues about, namely that one corpus is split once and then read through three
disjoint views whose held-out errors are compared in pairs.

Every number is a literal from the paper's own text, which in turn traces to:
  paper/iccit6_profstyle/sections/02_methods.tex   balanced rows, split, features
  experiments/audit/paper_claim_verification.json  the deployed checkpoints
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "paper" / "iccit6_profstyle"

CB = {"data": "#4477AA", "surf": "#EE6677", "cont": "#4477AA", "ref": "#AA3377",
      "out": "#228833", "grey": "#555555"}
FILL = {"data": "#E4EDF6", "surf": "#FBE7E9", "cont": "#E4EDF6", "ref": "#F2E9F5",
        "out": "#E6F2E8", "grey": "#EFEFEF"}

plt.rcParams.update({
    "font.size": 7.4, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
})


def box(ax, x, y, w, h, title, sub, kind, title_size=7.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=0.9, edgecolor=CB[kind], facecolor=FILL[kind], zorder=2))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#111111", zorder=3)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
            fontsize=6.1, color="#333333", zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7,
        linewidth=0.8, color=CB["grey"], shrinkA=0, shrinkB=0, zorder=1))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.00, 0.38, 0.155, 0.26, "Corpora",
        "34,994 and\n53,806 rows", "data")
    box(ax, 0.185, 0.38, 0.155, 0.26, "Split",
        "72 / 8 / 20\ngroup-aware", "data")

    box(ax, 0.375, 0.72, 0.225, 0.24, "Surface view",
        "47 orthographic features", "surf")
    box(ax, 0.375, 0.38, 0.225, 0.24, "Content view",
        "bag of words, no marks", "cont")
    box(ax, 0.375, 0.04, 0.225, 0.24, "Raw view",
        "128 subword tokens", "ref")

    box(ax, 0.635, 0.55, 0.175, 0.24, "Logistic reg.",
        "one per arm", "grey")
    box(ax, 0.635, 0.11, 0.175, 0.24, "Fine-tuning",
        "BERT, DeBERTa", "grey")

    box(ax, 0.845, 0.38, 0.155, 0.26, "Comparison",
        "$\\Delta = \\varepsilon_S - \\varepsilon_C$\nMcNemar, CI", "out")

    arrow(ax, 0.155, 0.51, 0.185, 0.51)
    for y in (0.84, 0.50, 0.16):
        arrow(ax, 0.340, 0.51, 0.375, y)
    arrow(ax, 0.600, 0.84, 0.635, 0.72)
    arrow(ax, 0.600, 0.50, 0.635, 0.64)
    arrow(ax, 0.600, 0.16, 0.635, 0.20)
    arrow(ax, 0.810, 0.67, 0.845, 0.56)
    arrow(ax, 0.810, 0.23, 0.845, 0.46)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline_compact.pdf")
    print("wrote", OUT / "fig_pipeline_compact.pdf")


if __name__ == "__main__":
    main()
