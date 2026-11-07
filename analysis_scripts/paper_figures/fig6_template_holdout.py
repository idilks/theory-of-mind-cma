"""Rebuttal figure 6 — no template overfitting (R1 pt 2, R2 Quality 3).

(a) leave-one-template-out: heads selected on two scenario templates land at
    the 96th-99.7th percentile of the held-out template's causal-score
    distribution (median across the selected set); a random head sits at 50
(b) per-template fingerprints agree pairwise at r = 0.70-0.94

Data: writeups/results_for_pi/template_holdout/stats_{loto,crosstemplate}.csv
(produced by scripts/template_holdout_check.py).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, AQUA, VIOLET, ORANGE, YELLOW, INK2, apply_style,
                   despine, panel_label, save)

REPO = Path(__file__).resolve().parents[2]
STATS = REPO / "writeups/results_for_pi/template_holdout"

CONDS = ["C1", "C2", "C3", "C5", "C6"]
COND_COLOR = {"C1": BLUE, "C2": AQUA, "C3": VIOLET, "C5": ORANGE, "C6": YELLOW}
COND_NAME = {"C1": "belief", "C2": "photo", "C3": "belief+ans",
             "C5": "answer", "C6": "answer (photo)"}
TMPL_MARK = {"food_truck": "o", "hair_styling": "^", "library_book": "s"}
TMPL_NAME = {"food_truck": "food truck", "hair_styling": "hair styling",
             "library_book": "library book"}


def main():
    apply_style()
    loto = pd.read_csv(STATS / "stats_loto.csv")
    cross = pd.read_csv(STATS / "stats_crosstemplate.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(8.0, 3.2))

    # ---- (a) LOTO percentiles -------------------------------------------
    for i, cond in enumerate(CONDS):
        sub = loto[loto.condition == cond]
        for j, (_, row) in enumerate(sub.iterrows()):
            a.plot([i + (j - 1) * 0.16], [row.median_pctile_heldout_fp],
                   TMPL_MARK[row.held_out], color=COND_COLOR[cond], ms=6,
                   alpha=0.9, mec="white", mew=0.5)
    a.axhline(50, color=INK2, lw=0.9, ls="--")
    a.text(4.42, 51.5, "chance\n(random head)", fontsize=7.5, color=INK2,
           ha="right", va="bottom")
    lo = loto.median_pctile_heldout_fp.min()
    hi = loto.median_pctile_heldout_fp.max()
    a.text(2.0, 88, f"all folds: {lo:.0f}th–{hi:.1f}th\npercentile",
           fontsize=8, ha="center", color=INK2)
    a.set_xticks(range(len(CONDS)))
    a.set_xticklabels([f"{c}\n{COND_NAME[c]}" for c in CONDS], fontsize=7.5)
    a.set_ylim(0, 104)
    a.set_ylabel("percentile of selected heads in\nheld-out template's scores")
    a.set_title("Heads selected on 2 templates\nscore high on the 3rd", loc="left")
    handles = [plt.Line2D([], [], marker=m, color=INK2, ls="none", ms=5.5,
                          label=f"held out: {TMPL_NAME[t]}")
               for t, m in TMPL_MARK.items()]
    a.legend(handles=handles, loc="center left", fontsize=6.8, frameon=False,
             bbox_to_anchor=(0.02, 0.32))
    panel_label(a, "a", x=-0.24)

    # ---- (b) cross-template fingerprint agreement -----------------------
    pair_marks = {("food_truck", "hair_styling"): "o",
                  ("food_truck", "library_book"): "^",
                  ("hair_styling", "library_book"): "s"}
    for i, cond in enumerate(CONDS):
        sub = cross[cross.condition == cond]
        for j, (_, row) in enumerate(sub.iterrows()):
            mk = pair_marks[(row.template_a, row.template_b)]
            b.plot([i + (j - 1) * 0.16], [row.pearson_r], mk,
                   color=COND_COLOR[cond], ms=6, alpha=0.9, mec="white",
                   mew=0.5)
    b.axhline(0, color=INK2, lw=0.9, ls="--")
    b.text(4.42, 0.03, "unrelated\nfingerprints", fontsize=7.5, color=INK2,
           ha="right", va="bottom")
    b.set_xticks(range(len(CONDS)))
    b.set_xticklabels([f"{c}\n{COND_NAME[c]}" for c in CONDS], fontsize=7.5)
    b.set_ylim(-0.08, 1.02)
    b.set_ylabel("Pearson r between per-template\nfingerprints (1,920 heads)")
    b.set_title("Per-template fingerprints agree\n(r = 0.70–0.94, all 15 pairs)",
                loc="left")
    handles = [plt.Line2D([], [], marker=m, color=INK2, ls="none", ms=5.5,
                          label=f"{TMPL_NAME[ta]} $\\times$ {TMPL_NAME[tb]}")
               for (ta, tb), m in pair_marks.items()]
    b.legend(handles=handles, loc="center left", fontsize=6.8, frameon=False,
             bbox_to_anchor=(0.02, 0.30))
    panel_label(b, "b", x=-0.24)

    for ax in (a, b):
        despine(ax)
        ax.grid(axis="x", visible=False)

    fig.tight_layout(w_pad=3.0)
    save(fig, "fig6_template_holdout")

    print(f"LOTO percentiles: {lo:.2f}-{hi:.2f}; "
          f"cross-template r: {cross.pearson_r.min():.3f}-"
          f"{cross.pearson_r.max():.3f}")


if __name__ == "__main__":
    main()
