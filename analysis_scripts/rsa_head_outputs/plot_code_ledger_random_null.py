#!/usr/bin/env python
"""Plot the random-head nulls from code_ledger_random_null.py (diagnostic figure).

One panel per divergence set: histograms of obj|slot over random head sets
(unmatched, layer-matched, causally inert layer-matched) with the observed value.

Output: results/rsa_head_outputs/aggregation_code_ledger/random_null_partials.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "paper_figures"))
from style import BLUE, GRAY, INK, YELLOW, apply_style, despine  # noqa: E402

AGG = ROOT / "results/rsa_head_outputs/aggregation_code_ledger"
NULLS = [("unmatched", "random heads", "#c3c2b7"),
         ("layer_matched", "layer-matched random", GRAY),
         ("inert_matched", "layer-matched, causally inert", YELLOW)]
TITLES = {"code_abstract": "code divergence set (12 heads)",
          "tom_c1": "story divergence set on code (32 heads)"}


def main():
    apply_style()
    draws = pd.read_csv(AGG / "random_null_draws.csv")
    summ = pd.read_csv(AGG / "random_null_partials.csv").set_index(["head_set", "null"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    bins = np.linspace(-0.1, 0.45, 56)
    for ax, s in zip(axes, TITLES):
        for null, label, c in NULLS:
            v = draws.query("head_set == @s and null == @null").obj_given_slot
            p = summ.loc[(s, null), "p_one_sided"]
            ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.hist(v, bins=bins, color=c, alpha=0.75, label=f"{label} ({ptxt})",
                    histtype="stepfilled", edgecolor="none")
        obs = summ.loc[(s, "unmatched"), "obs_obj_given_slot"]
        ax.axvline(obs, color=BLUE, lw=2)
        ax.text(obs + 0.008, ax.get_ylim()[1] * 0.5, f"observed\n{obs:.3f}", color=BLUE,
                fontsize=8, va="center", ha="left")
        ax.set_title(TITLES[s], loc="left", fontsize=9, color=INK)
        ax.set_xlabel("object divergence | slot divergence\n(partial Spearman r of set RDM)")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=1)
        ax.grid(axis="x", visible=False)
        despine(ax)
    axes[0].set_ylabel("random head sets\n(2,000 per null)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(AGG / f"random_null_partials.{ext}")
    print(f"saved {AGG / 'random_null_partials.png'}")


if __name__ == "__main__":
    main()
