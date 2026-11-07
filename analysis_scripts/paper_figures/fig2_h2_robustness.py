"""Rebuttal figure 2 — H2 additive-decomposition robustness (R1 points 3-4).

(a) C3 vs C1+C5 over all 1,920 heads, raw (unclamped) scores, y=x additive line
(b) residual magnitude scales with component magnitude (raw OLS residual)
(c) within-layer permutation null vs observed r (clamped, matches rebuttal)
(d) OLS coefficients a, b with 95% bootstrap CIs across Qwen ToM / code / Gemma
(e) specificity: the C1+C5 sum vs each component alone and the axis-matched
    C2+C6 substitute (R1 point 4)
(f) Gemma replication scatter gC3 vs gC1+gC5

Coefficients/correlations are read from results/summary/h2_robustness/
stats_*.csv (the same files behind the rebuttal numbers); scatters and null
draws are recomputed from the fingerprints.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2_additivity_robustness import (  # noqa: E402
    CROSS_RULES, WITHIN_RULES, build_gemma_fp, load_tom_fp, perm_nulls,
    GEMMA_SHAPE)
from style import (BLUE, ORANGE, AQUA, GRAY, INK, INK2, apply_style, despine,
                   panel_label, save)

STATS = REPO / "results/summary/h2_robustness"

LAYER_CMAP = LinearSegmentedColormap.from_list(
    "layers_blue", ["#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])


def scatter_panel(ax, x, y, n_layers, n_heads, xlabel, ylabel, r_raw, r_clamped):
    layers = np.repeat(np.arange(n_layers), n_heads)
    order = np.argsort(np.abs(x + y))          # big scores drawn on top
    sc = ax.scatter(x[order], y[order], c=layers[order], cmap=LAYER_CMAP,
                    s=7, alpha=0.75, lw=0, rasterized=True)
    cax = ax.inset_axes([0.05, 0.60, 0.30, 0.035])
    cb = plt.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([0, n_layers - 1])
    cax.tick_params(labelsize=7, length=2)
    cax.set_title("layer", fontsize=7, pad=1.5)
    cb.outline.set_visible(False)
    lim_lo = min(x.min(), y.min()) - 0.3
    lim_hi = max(x.max(), y.max()) + 0.3
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color=INK2, lw=0.9,
            ls="--", zorder=1)
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.03, 0.97, f"r = {r_raw:.3f} raw\nr = {r_clamped:.3f} clamped",
            transform=ax.transAxes, va="top", fontsize=8.5, color=INK)
    ax.text(0.97, 0.03, "y = x\n(unweighted additive)", transform=ax.transAxes,
            va="bottom", ha="right", fontsize=7.5, color=INK2)
    return sc


def main():
    apply_style()

    fp = {c: load_tom_fp(f"{c}_{f}") for c, f in [
        ("C1", "abstract_belief"), ("C2", "abstract_photo"),
        ("C3", "answer_changes_belief"), ("C5", "answer_changes_location_only"),
        ("C6", "answer_changes_location_only_photo")]}
    corr = pd.read_csv(STATS / "stats_correlations.csv")
    ols = pd.read_csv(STATS / "stats_ols.csv")

    def r_of(domain, key, pair):
        m = corr[(corr.domain == domain) & (corr.key == key) & (corr.pair == pair)]
        return float(m.pearson_r.iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.4))
    (a, b_ax, c_ax), (d_ax, e_ax, f_ax) = axes

    # ---- (a) Qwen scatter, raw ----------------------------------------
    x = fp["C1"]["raw"] + fp["C5"]["raw"]
    y = fp["C3"]["raw"]
    scatter_panel(a, x, y, 48, 40, "C1 + C5 causal score (raw)",
                  "C3 causal score (raw)",
                  r_of("ToM", "raw", "C3~sum(C1+C5)"),
                  r_of("ToM", "clamped", "C3~sum(C1+C5)"))
    a.set_title("Qwen: C3 against C1 + C5, all 1,920 heads", loc="left")
    panel_label(a, "a")

    # ---- (b) residual scaling ------------------------------------------
    # residual of the *additive prediction* (C3 - (C1+C5)) — the quantity in
    # the rebuttal and stats_residual_significance.csv, not the OLS residual
    resid = y - x
    mag = np.abs(x)
    r_scaling = float(np.corrcoef(mag, np.abs(resid))[0, 1])
    slope = float(np.linalg.lstsq(
        np.column_stack([mag, np.ones_like(mag)]), np.abs(resid), rcond=None
    )[0][0])
    b_ax.scatter(mag, np.abs(resid), s=6, alpha=0.5, color=BLUE, lw=0,
                 rasterized=True)
    ms = np.linspace(0, mag.max(), 50)
    b_ax.plot(ms, slope * ms + np.abs(resid).mean() - slope * mag.mean(),
              color=INK, lw=1.2)
    b_ax.text(0.03, 0.97,
              f"|residual| $\\approx$ {slope:.2f} $\\times$ |C1+C5|\n"
              f"r = {r_scaling:.2f}",
              transform=b_ax.transAxes, va="top", fontsize=8.5)
    b_ax.set_xlabel("|C1 + C5| (raw)")
    b_ax.set_ylabel("|C3 $-$ (C1 + C5)|")
    b_ax.set_title("Residuals scale with component size", loc="left")
    panel_label(b_ax, "b")

    # ---- (c) permutation nulls (clamped, matches rebuttal wording) -----
    tgt = fp["C3"]["clamped"]
    summed = fp["C1"]["clamped"] + fp["C5"]["clamped"]
    obs, nulls = perm_nulls(tgt, summed)
    wl = nulls["within_layer_shuffle"]["draws"]
    fs = nulls["full_shuffle"]["draws"]
    bins = np.linspace(-0.15, 1.0, 90)
    c_ax.hist(fs, bins=bins, color="#d5d3cc", label="full head shuffle")
    c_ax.hist(wl, bins=bins, color=GRAY, alpha=0.75,
              label="within-layer shuffle")
    c_ax.axvline(obs, color=BLUE, lw=1.8)
    c_ax.text(obs - 0.02, c_ax.get_ylim()[1] * 0.97,
              f"observed\nr = {obs:.3f}", ha="right", va="top", fontsize=8.5,
              color=BLUE)
    c_ax.text(np.percentile(wl, 95) + 0.05, c_ax.get_ylim()[1] * 0.16,
              f"within-layer null\n95th pct = {np.percentile(wl, 95):.2f}",
              fontsize=7.5, color=INK2)
    c_ax.set_xlabel("r(C3, shuffled C1 + C5)")
    c_ax.set_ylabel("permutations (of 5,000)")
    c_ax.legend(loc="center left", fontsize=7.5)
    c_ax.set_title("Null baselines: p < 2$\\times$10$^{-4}$", loc="left")
    panel_label(c_ax, "c")

    # ---- (d) coefficient forest across domains -------------------------
    rows = [
        ("Qwen  ToM\n(C3 ~ C1 + C5)", "ToM", "beta[C1]", "beta[C5]"),
        ("Qwen  code\n(code_ac ~ abs + ctl)", "code", "beta[code_abs]",
         "beta[code_ctl]"),
        ("Gemma  ToM\n(gC3 ~ gC1 + gC5)", "gemma", "beta[gC1]", "beta[gC5]"),
    ]
    yticks, ylabels = [], []
    ypos = 0
    for label, domain, ca, cb in rows:
        for coef, color, tag in [(ca, BLUE, "a"), (cb, ORANGE, "b")]:
            for key, mfc, dy in [("clamped", None, 0.16), ("raw", "white", -0.16)]:
                m = ols[(ols.domain == domain) & (ols.key == key)
                        & (ols.coef == coef)]
                est, lo, hi = (float(m.estimate.iloc[0]),
                               float(m.ci_lo.iloc[0]), float(m.ci_hi.iloc[0]))
                d_ax.plot([lo, hi], [ypos + dy] * 2, color=color, lw=1.4,
                          solid_capstyle="round")
                d_ax.plot([est], [ypos + dy], "o", color=color, ms=5,
                          mfc=mfc or color, mew=1.2)
            d_ax.text(0.35, ypos, tag, fontsize=9, color=color,
                      va="center", ha="left", fontweight="bold")
            ypos -= 1
        yticks.append(ypos + 1.5)
        ylabels.append(label)
        ypos -= 0.6
    d_ax.axvline(1.0, color=INK2, lw=0.8, ls="--")
    d_ax.text(1.02, 0.62, "unweighted\nprediction (= 1)", fontsize=7.5,
              color=INK2, va="top")
    d_ax.set_yticks(yticks)
    d_ax.set_yticklabels(ylabels, fontsize=8)
    d_ax.set_xlabel("OLS coefficient (95% bootstrap CI)")
    d_ax.set_title("a $\\approx$ 1, mild compression of b", loc="left")
    d_ax.plot([], [], "o", color=INK, label="clamped")
    d_ax.plot([], [], "o", color=INK, mfc="white", label="raw")
    d_ax.legend(loc="lower left", fontsize=7.5, ncol=1)
    panel_label(d_ax, "d", x=-0.32)

    # ---- (e) specificity ------------------------------------------------
    combos = [
        ("C1 + C5", r_of("ToM", "clamped", "C3~sum(C1+C5)"),
         r_of("ToM", "raw", "C3~sum(C1+C5)"), BLUE),
        ("C1 alone", r_of("ToM", "clamped", "C3~C1"),
         r_of("ToM", "raw", "C3~C1"), GRAY),
        ("C2 + C6\n(axis-matched substitute)", r_of("ToM", "clamped", "C3~sum(C2+C6)"),
         r_of("ToM", "raw", "C3~sum(C2+C6)"), AQUA),
        ("C5 alone", r_of("ToM", "clamped", "C3~C5"),
         r_of("ToM", "raw", "C3~C5"), GRAY),
    ]
    ys = np.arange(len(combos))[::-1]
    for yv, (label, rc, rr, color) in zip(ys, combos):
        e_ax.barh(yv, rc, height=0.55, color=color, alpha=0.9)
        e_ax.plot([rr], [yv], "D", color=INK, ms=4, mfc="white", mew=1.1)
        e_ax.text(rc - 0.015, yv, f"{rc:.3f}", va="center", ha="right",
                  fontsize=8, color="white", fontweight="medium")
    e_ax.set_yticks(ys)
    e_ax.set_yticklabels([c[0] for c in combos], fontsize=8)
    e_ax.set_xlim(0, 1.0)
    e_ax.set_xlabel("r with C3 (bar clamped, $\\diamond$ raw)")
    e_ax.set_title("The fit needs C3's own components", loc="left")
    panel_label(e_ax, "e", x=-0.40)

    # ---- (f) Gemma scatter ----------------------------------------------
    g1, _ = build_gemma_fp("abstract_belief", CROSS_RULES)
    g3, _ = build_gemma_fp("answer_changes_belief", CROSS_RULES)
    g5, _ = build_gemma_fp("answer_changes_location_only", WITHIN_RULES)
    gx = g1["raw"] + g5["raw"]
    gy = g3["raw"]
    scatter_panel(f_ax, gx, gy, *GEMMA_SHAPE, "gC1 + gC5 causal score (raw)",
                  "gC3 causal score (raw)",
                  r_of("gemma", "raw", "gC3~sum(gC1+gC5)"),
                  r_of("gemma", "clamped", "gC3~sum(gC1+gC5)"))
    f_ax.set_title("Gemma-2-27B-it: replication, 1,472 heads", loc="left")
    panel_label(f_ax, "f")

    for ax in axes.ravel():
        despine(ax)
    fig.tight_layout(w_pad=2.0, h_pad=2.4)
    save(fig, "fig2_h2_robustness")

    print(f"residual scaling: slope={slope:.3f}, r={r_scaling:.3f}")
    print(f"within-layer null: obs={obs:.3f}, max={wl.max():.3f}, "
          f"p95={np.percentile(wl, 95):.3f}")


if __name__ == "__main__":
    main()
