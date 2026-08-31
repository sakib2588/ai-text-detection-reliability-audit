"""Measurement pipeline for the six-page cut, fig_pipeline_compact.pdf.

Built as a publication figure rather than a flowchart of labelled boxes. Three
choices carry it.

The three views are the paper's contribution, so they get the space and are the
only elements carrying colour. Each shows a worked specimen of the same input
sentence rather than naming what it does, so a reader sees that the surface view
keeps the marks and drops the words while the content view does the reverse,
which is the argument in one glance.

The partition is drawn to scale as a segmented bar rather than written as
"72 / 8 / 20", and the corpus sizes as proportional rules, so the quantities a
reader might compare are comparable by eye.

Everything else is neutral: hairline rules, no shadows, no fills except the view
accents, one stroke weight for flow and a lighter one for structure.

Geometry note. The axis is 100 units over 2.45 in, so one unit is about 1.75 pt.
A bold title over a muted note needs at least six units between baselines, and
the column widths below are what the longest string in each column fits into at
6.5 pt. Both are why the strings here are short.

Numbers are literals from the paper's own text:
  paper/iccit6_profstyle/main.tex   balanced rows, split, feature counts
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[2] / "paper" / "iccit6_profstyle"

INK = "#1A1A1A"
MUTED = "#6B7280"
RULE = "#D5D9DE"
FLOW = "#98A2AC"
VIEW = {"surface": "#C2185B", "content": "#1565C0", "raw": "#6A1B9A"}

C1, W1 = 0.0, 16.0
C2, W2 = 21.0, 16.0
C3, W3 = 42.0, 31.0
C4, W4 = 78.0, 22.0

plt.rcParams.update({
    "font.size": 7.4, "figure.dpi": 400, "savefig.dpi": 400,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015, "pdf.fonttype": 42,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
})

MONO = {"family": "monospace", "fontsize": 6.3}


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 2.18))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def header(x, w, n, label):
        ax.text(x + 1.0, 95.5, str(n), ha="center", va="center", fontsize=6.2,
                color="#FFFFFF", fontweight="bold", zorder=4,
                bbox=dict(boxstyle="circle,pad=0.30", fc=INK, ec="none"))
        ax.text(x + 4.4, 95.5, " ".join(label.upper()), ha="left", va="center",
                fontsize=6.3, color=MUTED, fontweight="bold")
        ax.plot([x, x + w], [90.0, 90.0], color=RULE, lw=0.7, zorder=1)

    def flow(x1, y1, x2, y2, mid):
        ax.plot([x1, mid, mid], [y1, y1, y2], color=FLOW, lw=0.85,
                solid_capstyle="round", solid_joinstyle="round", zorder=1)
        ax.add_patch(FancyArrowPatch((mid, y2), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=6, lw=0.85, color=FLOW,
                                     shrinkA=0, shrinkB=0, zorder=1))

    # ------------------------------------------------------------ 1  corpus
    header(C1, W1, 1, "Corpus")
    for name, n, y in (("DAIGT V2", 34994, 66.0), ("HC3", 53806, 40.0)):
        ax.text(C1, y + 12.0, name, ha="left", va="center", fontsize=7.6,
                color=INK, fontweight="bold")
        ax.add_patch(Rectangle((C1, y + 5.5), W1 * n / 53806, 2.4, fc=INK,
                               ec="none", alpha=0.32, zorder=2))
        ax.text(C1, y, f"{n:,} rows", ha="left", va="center", fontsize=6.5,
                color=MUTED)
        ax.text(C1, y - 6.0, "balanced", ha="left", va="center", fontsize=6.5,
                color=MUTED)

    # ------------------------------------------------------------ 2 partition
    header(C2, W2, 2, "Partition")
    ax.text(C2, 78.0, "Grouped split", ha="left", va="center", fontsize=7.6,
            color=INK, fontweight="bold")
    x0, seg_y, seg_h = C2, 68.0, 4.4
    for frac, lbl in ((0.72, "72"), (0.08, "8"), (0.20, "20")):
        ax.add_patch(Rectangle((x0, seg_y), W2 * frac, seg_h, fc=INK,
                               ec="#FFFFFF", lw=0.7,
                               alpha=0.14 + 0.50 * frac, zorder=2))
        ax.text(x0 + W2 * frac / 2, seg_y - 4.6, lbl, ha="center", va="center",
                fontsize=6.3, color=MUTED, zorder=3)
        x0 += W2 * frac
    ax.text(C2, 56.0, "train / val / test", ha="left", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(C2, 46.0, "hashed by content,", ha="left", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(C2, 40.0, "no group crosses", ha="left", va="center", fontsize=6.5,
            color=MUTED)

    # ------------------------------------------------------------ 3  views
    header(C3, W3, 3, "Disjoint views")
    ax.text(C3, 84.0, 'in   He said, "yes" — 42 times.', ha="left",
            va="center", color=MUTED, **MONO)

    views = [("surface", "Surface", ',   "   —   .', "47 marks, no words", 58.0),
             ("content", "Content", "he said yes times", "bag of words", 34.0),
             ("raw", "Raw", 'He said , " yes "', "128 subword tokens", 10.0)]
    for key, title, specimen, note, y in views:
        ax.add_patch(Rectangle((C3, y), 0.9, 15.5, fc=VIEW[key], ec="none",
                               zorder=3))
        ax.text(C3 + 2.4, y + 12.4, title, ha="left", va="center", fontsize=7.6,
                color=VIEW[key], fontweight="bold")
        ax.text(C3 + 2.4, y + 6.6, specimen, ha="left", va="center", color=INK,
                **MONO)
        ax.text(C3 + 2.4, y + 1.4, note, ha="left", va="center", fontsize=6.5,
                color=MUTED)
        if y > 10.0:
            ax.plot([C3, C3 + W3], [y - 4.6, y - 4.6], color=RULE, lw=0.6,
                    zorder=1)

    for y in (65.0, 41.0, 17.0):
        flow(C2 + W2, 70.0, C3 - 0.8, y, mid=C3 - 3.4)

    # ------------------------------------------------------------ 4 arm, test
    header(C4, W4, 4, "Arm and test")
    for title, note, y in (("Logistic regression", "one per matched arm", 72.0),
                           ("Fine-tuned transformer", "BERT, DeBERTa", 52.0)):
        ax.text(C4, y + 6.0, title, ha="left", va="center", fontsize=7.5,
                color=INK, fontweight="bold")
        ax.text(C4, y, note, ha="left", va="center", fontsize=6.5, color=MUTED)
        ax.plot([C4, C4 + W4], [y - 6.5, y - 6.5], color=RULE, lw=0.6, zorder=1)

    flow(C3 + W3, 65.0, C4 - 0.8, 74.0, mid=C4 - 3.2)
    flow(C3 + W3, 41.0, C4 - 0.8, 74.0, mid=C4 - 3.2)
    flow(C3 + W3, 17.0, C4 - 0.8, 54.0, mid=C4 - 3.2)

    ax.text(C4, 32.0, r"$\Delta=\varepsilon_{\mathrm{S}}-\varepsilon_{\mathrm{C}}$",
            ha="left", va="center", fontsize=9.6, color=INK)
    ax.text(C4, 23.0, "exact McNemar, 95% CI,", ha="left", va="center",
            fontsize=6.5, color=MUTED)
    ax.text(C4, 17.0, "five partitions", ha="left", va="center", fontsize=6.5,
            color=MUTED)
    ax.add_patch(FancyArrowPatch((C4 + 5.0, 43.0), (C4 + 5.0, 37.0),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.85,
                                 color=FLOW, shrinkA=0, shrinkB=0, zorder=1))

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_pipeline_compact.pdf")
    print("wrote", OUT / "fig_pipeline_compact.pdf")


if __name__ == "__main__":
    main()
