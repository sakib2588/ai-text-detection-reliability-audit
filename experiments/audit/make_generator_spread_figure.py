"""Per-generator arm error on DAIGT V2, fig_generator_spread.pdf.

Drawn as a strip plot rather than a bar per generator, so all fifteen fit in a
third of the height a bar chart needs. That matters in a six-page paper, and the
strip shows the claim better anyway: the surface arm's error is spread across
more than an order of magnitude while the content arm stays clamped near zero.

The two same-model pairs the text names are joined by a rule, and both involve a
member below the 200-row floor, which is why the underpowered generators are
drawn hollow rather than dropped.

Every value from experiments/audit/subgroup_decomposition.json,
key corpora."DAIGT V2".<generator>.{surface_err, content_err, n_test}.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINAL = Path(__file__).resolve().parents[2]
SRC = FINAL / "experiments" / "audit" / "subgroup_decomposition.json"
OUT = FINAL / "paper" / "iccit6_profstyle"
FLOOR = 200
SURF, CONT, LINK = "#EE6677", "#4477AA", "#999999"

# the two pairs named in the text, each one powered member and one below the floor
# the pair factors the text quotes, 4.07 and 2.32, come from exactly these two
PAIRS = [("mistral7binstruct_v1", "mistralai/Mistral-7B-Instruct-v0.1"),
         ("kingki19_palm", "palm-text-bison1")]

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.8,
    "ytick.labelsize": 7.2, "legend.fontsize": 6.6,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
})


def main() -> None:
    d = json.loads(SRC.read_text())["corpora"]["DAIGT V2"]
    g = {k: v for k, v in d.items() if isinstance(v, dict) and "surface_err" in v}

    seen = {}

    def jit(x, y):
        # nudge a point vertically when another sits within 0.35 points of it
        k = (y, round(x / 0.35))
        seen[k] = seen.get(k, -1) + 1
        return (seen[k] % 3 - 1) * 0.13

    fig, ax = plt.subplots(figsize=(3.45, 1.25))
    for name, v in g.items():
        powered = v["n_test"] >= FLOOR
        for y, key, colour in ((1, "surface_err", SURF), (0, "content_err", CONT)):
            ax.plot(v[key] * 100, y + jit(v[key] * 100, y), "o", ms=4.4, color=colour,
                    mfc=colour if powered else "white",
                    mew=0.8 if powered else 1.4, zorder=3)

    worst = sorted(g.items(), key=lambda kv: -kv[1]["surface_err"])[:2]
    for name, v in worst:
        x = v["surface_err"] * 100
        ax.annotate(name.split("/")[-1].replace("_", " "), (x, 1.10), (x, 1.40),
                    ha="center", va="bottom", fontsize=6.0, color="#666666",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#AAAAAA"))

    ax.set_yticks([1, 0])
    ax.set_yticklabels(["surface", "content"])
    ax.set_ylim(-0.45, 1.75)
    ax.set_xlabel("test error per generator, %")
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.2, color="#555555",
                          label="$n\\geq200$"),
               plt.Line2D([], [], marker="o", ls="", ms=4.2, color="#555555",
                          mfc="white", mew=0.9, label="$n<200$")]
    ax.legend(handles=handles, frameon=False, loc="lower right",
              handlelength=0.8, borderaxespad=0.1, ncol=2, columnspacing=0.9)

    fig.tight_layout(pad=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_generator_spread.pdf")
    p = [v["surface_err"] * 100 for v in g.values() if v["n_test"] >= FLOOR]
    print(f"wrote, {len(g)} generators, powered surface error {min(p):.2f} to {max(p):.2f}%")


if __name__ == "__main__":
    main()
