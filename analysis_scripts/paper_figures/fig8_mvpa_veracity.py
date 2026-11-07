"""Figure 8 — MVPA decoding of belief veracity from filtered head sets (Sept 8).

Camera-ready planning figure for the Sept 3 decision "decoding is saturated in
LLMs; report it on filtered head sets only, RSA is the primary instrument".

(a) leave-one-template-out (LOTO) balanced accuracy of an L2 logistic decoder
    of belief veracity (false vs true) from the concatenated final-token
    outputs of a head set, against the size-matched random-head band
    (mean and 5th-95th percentile over draws);
(b) the same sets under pooled within-template 6-fold CV — every set is at
    ceiling, which is the saturation point.

Data: results/rsa_head_outputs/mvpa_veracity/{summary_belief_veracity,
random_band_belief_veracity}.csv (analysis_scripts/rsa_head_outputs/mvpa_veracity.py;
Qwen2.5-14B-Instruct, 144-stimulus capture, 3 templates).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, ORANGE, AQUA, VIOLET, GRAY, INK, INK2, apply_style,
                   despine, panel_label, save)

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results/rsa_head_outputs/mvpa_veracity"

# display order, label, colour
SETS = [
    ("div_unique", "belief-unique\n(C1 top-32 $\\setminus$ C5, 22)", BLUE),
    ("top10_C1", "belief top-10\n(C1, 10)", BLUE),
    ("div_shared", "belief $\\cap$ retrieval\n(10)", VIOLET),
    ("dual_sig", "belief $\\cap$ code\n(10)", VIOLET),
    ("photo_C2", "photo\n(C2 $\\setminus$ C6, 21)", AQUA),
    ("c5_retrieval", "answer retrieval\n(C5 top-22)", ORANGE),
    ("random32", "random\n(32)", GRAY),
]


def main():
    apply_style()
    s = pd.read_csv(SRC / "summary_belief_veracity.csv").set_index("head_set")
    band = pd.read_csv(SRC / "random_band_belief_veracity.csv").set_index("k")

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True,
                               gridspec_kw={"width_ratios": [1.6, 1]})
    xs = np.arange(len(SETS))

    # ---- (a) LOTO ------------------------------------------------------------
    for x, (key, label, col) in zip(xs, SETS):
        r = s.loc[key]
        k = int(r["n_heads"])
        lo, hi, mu = (band.loc[k, "rand_loto_acc_p5"],
                      band.loc[k, "rand_loto_acc_p95"],
                      band.loc[k, "rand_loto_acc_mean"])
        a.bar(x, mu, width=0.62, facecolor="white", edgecolor=GRAY,
              hatch="////", linewidth=0.9, zorder=2)
        a.plot([x, x], [lo, hi], color=INK2, lw=1.2, zorder=3)
        a.plot([x], [r["loto_acc"]], "o", color=col, ms=8, zorder=4,
               mec="white", mew=0.8)
        a.text(x + 0.16, r["loto_acc"], f"{r['loto_acc']:.2f}", ha="left",
               va="center", fontsize=7.5, color=col)
    a.axhline(0.5, color=INK2, lw=0.8, ls=(0, (1, 2)))
    a.text(len(SETS) - 0.55, 0.51, "chance", fontsize=7, color=INK2,
           ha="right", va="bottom")
    a.set_xticks(xs)
    a.set_xticklabels([l for _, l, _ in SETS], fontsize=6.8)
    a.set_ylabel("balanced accuracy, false vs true belief")
    a.set_ylim(0.4, 1.12)
    a.set_title("Leave-one-template-out: filtering buys cross-template\n"
                "generalisation", loc="left")
    handles = [
        plt.Line2D([], [], marker="o", color=INK, ls="none", ms=7,
                   label="CMA-selected head set"),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRAY,
                      hatch="////", label="size-matched random heads, mean"),
        plt.Line2D([], [], color=INK2, lw=1.2, label="random band, 5th–95th pct"),
    ]
    a.legend(handles=handles, loc="upper right", fontsize=7, frameon=False)
    panel_label(a, "a", x=-0.13)

    # ---- (b) within-template ---------------------------------------------
    for x, (key, label, col) in zip(xs, SETS):
        r = s.loc[key]
        b.bar(x, r["within_acc"], width=0.62, color=col, zorder=2)
    b.axhline(0.5, color=INK2, lw=0.8, ls=(0, (1, 2)))
    b.set_xticks(xs)
    b.set_xticklabels([l.split("\n")[0] for _, l, _ in SETS], fontsize=6.8,
                      rotation=35, ha="right")
    b.set_title("Within-template 6-fold: saturated\n(every set, random included)",
                loc="left")
    b.text(len(SETS) / 2 - 0.5, 1.03, "all 1.00", ha="center", fontsize=8,
           color=INK2)
    panel_label(b, "b", x=-0.08)

    for ax in (a, b):
        despine(ax)
        ax.grid(axis="x", visible=False)

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig8_mvpa_veracity")

    for key, _, _ in SETS:
        r = s.loc[key]
        k = int(r["n_heads"])
        print(f"{key:13s} k={k:2d} LOTO acc {r['loto_acc']:.3f} "
              f"(AUC {r['loto_auc']:.3f}) | within acc {r['within_acc']:.2f} | "
              f"random band mean {band.loc[k, 'rand_loto_acc_mean']:.3f} "
              f"p95 {band.loc[k, 'rand_loto_acc_p95']:.3f}")


if __name__ == "__main__":
    main()
