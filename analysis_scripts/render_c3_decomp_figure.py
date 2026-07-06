"""Render the C3 ≈ C1 + C5 decomposition figure for the NeurIPS paper.

Two panels only (no OLS, no betas, no R^2):
  - bar chart of r(C3, C1), r(C3, C5), r(C3, C1+C5)
  - per-head scatter of (C1+C5) vs C3 with y=x reference

Reads pre-computed clamped fingerprints from results/aggregated/fingerprints/
and writes directly into theory_of_mind_neurips/figures/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
FP_DIR = ROOT / "results/aggregated/fingerprints"
OUT = ROOT / "theory_of_mind_neurips/figures/h2_C3_eq_C1plusC5_clamped.png"


def main():
    c1 = np.load(FP_DIR / "C1_abstract_belief_clamped.npy")
    c3 = np.load(FP_DIR / "C3_answer_changes_belief_clamped.npy")
    c5 = np.load(FP_DIR / "C5_answer_changes_location_only_clamped.npy")
    s = c1 + c5

    r_c1 = pearsonr(c3, c1).statistic
    r_c5 = pearsonr(c3, c5).statistic
    r_sum = pearsonr(c3, s).statistic
    print(f"r(C3, C1)     = {r_c1:.3f}")
    print(f"r(C3, C5)     = {r_c5:.3f}")
    print(f"r(C3, C1+C5)  = {r_sum:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    labels = [r"$r(C_3, C_1)$", r"$r(C_3, C_5)$", r"$r(C_3, C_1{+}C_5)$"]
    vals = [r_c1, r_c5, r_sum]
    colors = ["steelblue", "steelblue", "crimson"]
    bars = ax.bar(range(3), vals, color=colors, edgecolor="black", linewidth=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Pearson $r$", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(r"Marginal vs joint correlation with $C_3$", fontsize=12)
    ax.axhline(0, color="grey", lw=0.5)
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=10)

    ax = axes[1]
    ax.scatter(s, c3, s=10, alpha=0.45, color="steelblue", edgecolor="none")
    lo = float(min(s.min(), c3.min()))
    hi = float(max(s.max(), c3.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label=r"$y=x$")
    ax.set_xlabel(r"$C_1 + C_5$ (per head)", fontsize=11)
    ax.set_ylabel(r"$C_3$ (per head)", fontsize=11)
    ax.set_title(rf"Per-head scatter ($n={len(c3)}$ heads), "
                 rf"$r={r_sum:.3f}$", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()
