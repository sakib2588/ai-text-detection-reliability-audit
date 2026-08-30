"""Per-generator arm error on DAIGT V2, fig_generator_spread.pdf.

One labelled bar per generator, because the point of the figure is that the
surface arm's error depends on *which* generator, and a reader cannot see that
from unlabelled points. An earlier strip-plot version was more compact and was
abandoned after it proved unreadable: fifteen anonymous dots show a spread but
never say whose.

Restricted to the ten generators at or above the 200-row floor, which is exactly
the set the sixteenfold claim in the text is about.

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
SURF, CONT = "#EE6677", "#4477AA"

plt.rcParams.update({
    "font.size": 8.0, "axes.labelsize": 8.0, "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0, "legend.fontsize": 7.0,
    "figure.dpi": 400, "savefig.dpi": 400, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
})

SHORT = {"mistral7binstruct_v1": "mistral-7b v1", "mistral7binstruct_v2": "mistral-7b v2",
         "darragh_claude_v6": "claude v6", "darragh_claude_v7": "claude v7",
         "kingki19_palm": "palm", "chat_gpt_moth": "chatgpt", "llama2_chat": "llama-2 chat",
         "falcon_180b_v1": "falcon 180b", "llama_70b_v1": "llama 70b", "radek_500": "radek 500"}


def main() -> None:
    d = json.loads(SRC.read_text())["corpora"]["DAIGT V2"]
    rows = [(SHORT.get(k, k.split("/")[-1]), v["surface_err"] * 100, v["content_err"] * 100)
            for k, v in d.items()
            if isinstance(v, dict) and "surface_err" in v and v["n_test"] >= FLOOR]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(3.45, 1.58))
    y = list(range(len(rows)))
    ax.barh(y, [r[1] for r in rows], height=0.66, color=SURF,
            label="surface-only", zorder=2)
    ax.plot([r[2] for r in rows], y, "o", ms=3.4, color=CONT,
            label="content-only", zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test error, %")
    ax.grid(axis="x", lw=0.4, color="#DDDDDD", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", handlelength=1.1, borderaxespad=0.3)

    lo, hi = rows[0][1], rows[-1][1]

    fig.tight_layout(pad=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_generator_spread.pdf")
    print(f"wrote, {len(rows)} generators, surface {lo:.2f} to {hi:.2f}%, {hi/lo:.1f}x")


if __name__ == "__main__":
    main()
