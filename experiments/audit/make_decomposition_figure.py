"""Two-panel decomposition figure for the six-page cut, fig_decomp_2panel.pdf.

Replaces the six-panel dashboard, which at the width a six-page paper can spare
was too small to read. Two panels at column width carry the paper's claim, the
arm errors per corpus and the surface-minus-content difference across the five
group-aware partitions, where DAIGT V2 stays far from zero and HC3 crosses it.

Every value is read from experiments/audit/multisplit_decomposition.json,
key datasets.<D>.summary.<arm>.error_by_split, and the p-values from
datasets.<D>.per_split.<seed>.tests.surface_vs_content.exact_p.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FINAL = Path(__file__).resolve().parents[2]
SRC = FINAL / "experiments" / "audit" / "multisplit_decomposition.json"
OUT = FINAL / "paper" / "iccit6_profstyle"

SURF, CONT = "#EE6677", "#4477AA"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False,
})


def main() -> None:
    d = json.loads(SRC.read_text())["datasets"]
    seeds = ["42", "123", "456", "789", "1337"]
    err = {k: {arm: [100 * x for x in d[k]["summary"][arm]["error_by_split"]]
               for arm in ("surface_only", "content_only")} for k in ("D1", "D2")}
    pval = {k: [d[k]["per_split"][s]["tests"]["surface_vs_content"]["exact_p"] for s in seeds]
            for k in ("D1", "D2")}
    names = {k: d[k]["name"] for k in ("D1", "D2")}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.45, 1.65))

    x = np.arange(2)
    w = 0.34
    s_mean = [np.mean(err[k]["surface_only"]) for k in ("D1", "D2")]
    c_mean = [np.mean(err[k]["content_only"]) for k in ("D1", "D2")]
    s_err = [[m - min(err[k]["surface_only"]) for m, k in zip(s_mean, ("D1", "D2"))],
             [max(err[k]["surface_only"]) - m for m, k in zip(s_mean, ("D1", "D2"))]]
    c_err = [[m - min(err[k]["content_only"]) for m, k in zip(c_mean, ("D1", "D2"))],
             [max(err[k]["content_only"]) - m for m, k in zip(c_mean, ("D1", "D2"))]]

    ax1.bar(x - w / 2, s_mean, w, yerr=s_err, color=SURF, label="surface", capsize=2,
            error_kw={"lw": 0.7})
    ax1.bar(x + w / 2, c_mean, w, yerr=c_err, color=CONT, label="content", capsize=2,
            error_kw={"lw": 0.7})
    ax1.set_xticks(x)
    ax1.set_xticklabels([names["D1"], names["D2"]])
    ax1.set_ylabel("test error, %")
    ax1.set_title("(a) arm error, 5 splits")
    ax1.legend(frameon=False, loc="upper right", handlelength=1.1)
    for xi, k in enumerate(("D1", "D2")):
        top_s = max(err[k]["surface_only"])
        top_c = max(err[k]["content_only"])
        ax1.text(xi - w / 2, top_s + 0.3, f"{s_mean[xi]:.1f}", ha="center", fontsize=7)
        ax1.text(xi + w / 2, top_c + 0.3, f"{c_mean[xi]:.1f}", ha="center", fontsize=7)
    ax1.set_ylim(0, max(s_mean) * 1.32)

    pos = np.arange(1, 6)
    for k, colour, marker in (("D1", "#228833", "o"), ("D2", "#AA3377", "s")):
        diff = [s - c for s, c in zip(err[k]["surface_only"], err[k]["content_only"])]
        sig = [p < 0.05 for p in pval[k]]
        ax2.plot(pos, diff, marker=marker, ms=3.4, lw=0.9, color=colour,
                 label=names[k], mfc=[colour if s else "white" for s in sig][0])
        ax2.scatter(pos, diff, s=14, marker=marker, color=colour,
                    facecolors=[colour if s else "white" for s in sig], zorder=3, linewidths=0.8)
    ax2.axhline(0, color="#888888", lw=0.8, ls="--")
    ax2.set_xticks(pos)
    ax2.set_xticklabels([1, 2, 3, 4, 5])
    ax2.set_xlabel("partition")
    ax2.set_ylabel("surface $-$ content, pts")
    ax2.set_title("(b) difference per split")
    ax2.legend(frameon=False, loc="center right", handlelength=1.1)

    fig.tight_layout(pad=0.35, w_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_decomp_2panel.pdf")
    print("wrote", OUT / "fig_decomp_2panel.pdf")


if __name__ == "__main__":
    main()
