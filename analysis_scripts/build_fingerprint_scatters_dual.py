"""Two-panel fingerprint scatters: clamped (left) vs raw (right).

Same data, significance sets and colouring as build_fingerprint_scatters.py;
only the axis values differ between panels. Pearson r, Spearman rho and a
two-sided permutation p (|r|, N_PERM shuffles) are computed per panel.

Outputs -> results/summary/fingerprint_scatters/figures/dual_*.png
"""

import sys
from pathlib import Path

import numpy as np
from scipy import stats as sps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis_scripts"))
from build_fingerprint_scatters import (  # noqa: E402
    TOM_CONDS, LABELS, FIG, tom_sig_set, published_set, flat_idx)
from h2_additivity_robustness import (  # noqa: E402
    load_tom_fp, build_code_fp, N_PERM, BLUE, ORANGE, GREEN, GRAY,
    CROSS_RULES)

RNG = np.random.default_rng(0)

PAIRS = [("C1", "C2"), ("C1", "code_abs"), ("SUM_C1_C5", "C3")]
LABELS = {**LABELS, "SUM_C1_C5": "C1 + C5 (unweighted sum)"}
# pairs where y = x is the hypothesis (H2: C3 = C1 + C5, unweighted)
IDENTITY_PAIRS = {("SUM_C1_C5", "C3")}


def perm_p(x, y):
    obs = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, RNG.permutation(y))[0, 1]
                     for _ in range(N_PERM)])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))


def panel(ax, x, y, mx, my, xk, yk, kind):
    both, xo, yo = mx & my, mx & ~my, ~mx & my
    neither = ~(mx | my)
    r, p = perm_p(x, y)
    rho = sps.spearmanr(x, y).statistic

    ax.scatter(x[neither], y[neither], s=6, color=GRAY, alpha=0.25,
               linewidths=0, label=f"neither (n={neither.sum()})")
    ax.scatter(x[xo], y[xo], s=22, color=BLUE, alpha=0.9, linewidths=0,
               label=f"{xk} only (n={xo.sum()})")
    ax.scatter(x[yo], y[yo], s=22, color=ORANGE, alpha=0.9, linewidths=0,
               label=f"{yk} only (n={yo.sum()})")
    ax.scatter(x[both], y[both], s=34, color=GREEN, alpha=0.95,
               edgecolors="white", linewidths=0.8,
               label=f"both (n={both.sum()})")
    if (xk, yk) in IDENTITY_PAIRS:
        lo_, hi_ = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo_, hi_], [lo_, hi_], ls="--", color="0.3", lw=0.9,
                zorder=0, label="y = x")
    if kind == "raw":
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.axvline(0, color="0.7", lw=0.6, zorder=0)
    ax.set_xlabel(f"{LABELS[xk]} ({kind})")
    ax.set_ylabel(f"{LABELS[yk]} ({kind})")
    ax.set_title("negatives clamped to 0" if kind == "clamped"
                 else "raw (signed)", fontsize=10)
    # headroom so extreme points don't hide behind the annotation/legend
    lo, hi = y.min(), y.max()
    ax.set_ylim(top=hi + 0.28 * (hi - min(lo, 0)))
    p_str = "p = 2×10⁻⁴" if p <= 2.5e-4 else f"p = {p:.4f}"
    ax.text(0.97, 0.97, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left",
              handletextpad=0.15, borderaxespad=0.2)
    return r, rho, p


def main():
    fps = {k: load_tom_fp(f"{k}_{TOM_CONDS[k][0]}")
           for k in ("C1", "C2", "C3", "C5")}
    fps["SUM_C1_C5"] = {k: fps["C1"][k] + fps["C5"][k]
                        for k in ("raw", "clamped")}
    fps["code_abs"], _ = build_code_fp("code_abstract", CROSS_RULES)

    sig = {k: flat_idx(tom_sig_set(*TOM_CONDS[k]))
           for k in ("C1", "C2", "C3", "C5")}
    sig["SUM_C1_C5"] = sig["C1"] | sig["C5"]
    sig["code_abs"] = flat_idx(published_set("code_abstract_head_list.csv"))
    for k, m in sig.items():
        print(f"sig[{k}]: {m.sum()} heads")

    for xk, yk in PAIRS:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
        for ax, kind in zip(axes, ("clamped", "raw")):
            r, rho, p = panel(ax, fps[xk][kind], fps[yk][kind],
                              sig[xk], sig[yk], xk, yk, kind)
            print(f"{xk}~{yk} {kind:8s} r={r:.3f} rho={rho:.3f} p={p:.4g}")
        fig.tight_layout()
        path = FIG / f"dual_{xk}_vs_{yk}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
