"""Compact study pipeline for the six-page cut, fig_pipeline_compact.pdf.

Matches the numbered-stage, card-shadow visual style of the long report's
diagram (experiments/audit/make_pipeline_figure.py), but scoped to only what
this six-page cut actually runs and reports: one corpus split once, read
through three disjoint views, each arm fit as one logistic regression (or one
fine-tuned transformer for the raw view), compared pairwise. It does not draw
the 8-configuration classical-vs-transformer sweep (Table I) or the 38-
configuration/4-representation sweep from the long report, since this cut
does not discuss either as a pipeline stage.

Every number is a literal from the paper's own text, which in turn traces to:
  paper/iccit6_profstyle/main.tex                  balanced rows, split, features
  experiments/audit/paper_claim_verification.json  the deployed checkpoints
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "paper" / "iccit6_profstyle"

CB = {"data": "#4477AA", "surf": "#EE6677", "cont": "#4477AA", "ref": "#AA3377",
      "out": "#228833", "grey": "#8A8A8A"}
FILL = {"data": "#E4EDF6", "surf": "#FBE7E9", "cont": "#E4EDF6", "ref": "#F2E9F5",
        "out": "#E6F2E8", "grey": "#F4F4F4"}
STAGE_COLOR = {1: "#4477AA", 2: "#AA3377", 3: "#8A8A8A", 4: "#228833"}
ARROW_COLOR = "#5B6B7A"
SHADOW = "#000000"

plt.rcParams.update({
    "font.size": 8.0, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})


def box(ax, x, y, w, h, title, sub, kind, title_size=8.2):
    ax.add_patch(FancyBboxPatch(
        (x + 0.0028, y - 0.0034), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=0, facecolor=SHADOW, alpha=0.07, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.028",
        linewidth=0.8, edgecolor=CB[kind], facecolor=FILL[kind], zorder=2))
    ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
            fontsize=title_size, fontweight="bold", color="#1A1A1A", zorder=3)
    ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
            fontsize=7.1, color="#4A4A4A", zorder=3, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8,
        linewidth=1.1, color=ARROW_COLOR, shrinkA=0, shrinkB=0,
        capstyle="round", joinstyle="round", zorder=1))


def stage(ax, x, y, n):
    ax.add_patch(FancyBboxPatch((x - 0.017, y - 0.017), 0.034, 0.034,
                                boxstyle="circle,pad=0", fc=SHADOW, ec="none",
                                alpha=0.10, zorder=2))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=7.0,
            color="#FFFFFF", fontweight="bold", zorder=4,
            bbox=dict(boxstyle="circle,pad=0.32", fc=STAGE_COLOR[n], ec="none"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 1.95))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.00, 0.38, 0.155, 0.26, "Corpora",
        "34,994 and\n53,806 rows", "data")
    box(ax, 0.185, 0.38, 0.155, 0.26, "Split",
        "72 / 8 / 20\ngroup-aware", "data")
    stage(ax, 0.014, 0.615, 1)

    box(ax, 0.370, 0.72, 0.250, 0.24, "Surface view",
        "47 orthographic features", "surf")
    box(ax, 0.370, 0.38, 0.250, 0.24, "Content view",
        "bag of words, no marks", "cont")
    box(ax, 0.370, 0.04, 0.250, 0.24, "Raw view",
        "128 subword tokens", "ref")
    stage(ax, 0.384, 0.935, 2)

    box(ax, 0.650, 0.55, 0.185, 0.24, "Logistic reg.",
        "one per arm", "grey")
    box(ax, 0.650, 0.11, 0.185, 0.24, "Fine-tuning",
        "BERT, DeBERTa", "grey")
    stage(ax, 0.664, 0.765, 3)

    box(ax, 0.845, 0.38, 0.155, 0.26, "Comparison",
        "$\\Delta = \\varepsilon_S - \\varepsilon_C$\nMcNemar, CI", "out")
    stage(ax, 0.859, 0.615, 4)

    arrow(ax, 0.155, 0.51, 0.185, 0.51)
    for y in (0.84, 0.50, 0.16):
        arrow(ax, 0.340, 0.51, 0.370, y)
    arrow(ax, 0.620, 0.84, 0.650, 0.72)
    arrow(ax, 0.620, 0.50, 0.650, 0.64)
    arrow(ax, 0.620, 0.16, 0.650, 0.20)
    arrow(ax, 0.835, 0.67, 0.845, 0.56)
    arrow(ax, 0.835, 0.23, 0.845, 0.46)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline_compact.pdf")
    print("wrote", OUT / "fig_pipeline_compact.pdf")


if __name__ == "__main__":
    main()
