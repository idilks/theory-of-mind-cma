"""H2 additivity robustness package (NeurIPS rebuttal, July 2026).

Reviewer 1: "Pearson correlation on clamped CMA fingerprints does not by
itself establish additive computation. The authors should report robustness
without clamping, magnitude/slope fits, residuals, and/or null baselines."

This script delivers all four, for both decomposition problems:
  ToM : C3 (answer_changes_belief)  ~  C1 (abstract_belief) + C5 (loc-only)
  code: code_answer_changes         ~  code_abstract + code_control

Analyses per problem x {clamped, raw}:
  1. correlations  — Pearson AND Spearman for target~comp1, ~comp2, ~sum,
                     and the ToM competitor sum C2+C6
  2. slope fits    — OLS with intercept: target = a*comp1 + b*comp2 + c;
                     unweighted additivity predicts a ~= b ~= 1, c ~= 0.
                     95% CIs from 10,000 head-resampling bootstraps.
  3. residuals     — target - (comp1+comp2), mapped over 48x40 heads;
                     structure (a layer band) would mean C3 has machinery
                     the sum does not capture. Top-|residual| heads tabled.
  4. null baselines— H0(full shuffle): r arises under random head-identity
                     alignment. H0(within-layer shuffle): r arises from the
                     shared layer-magnitude profile alone (late layers are
                     big in every condition — Taylor's dominance concern),
                     with no head-level correspondence. 5,000 perms each.

ToM fingerprints load from results/aggregated/fingerprints/*.npy (the exact
arrays behind the published r=0.971). Code fingerprints are built here with
the identical construction (per-cell scores; raw = mean; clamped = per-cell
clip-at-0 then mean) from best cells via condition_registry.

Outputs -> writeups/results_for_pi/h2_robustness/
  figures/*.png, stats_correlations.csv, stats_ols.csv, stats_nulls.csv,
  stats_residual_heads.csv, stats_fragment.md (tables for the .qmd)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from condition_registry import resolve_best_source  # noqa: E402

FP_DIR = REPO / "results/aggregated/fingerprints"
RESULTS_ROOT = REPO / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
OUT = REPO / "writeups/results_for_pi/h2_robustness"
FIG = OUT / "figures"

N_LAYERS, N_HEADS = 48, 40
N_PERM = 5000
N_BOOT = 10000
RNG = np.random.default_rng(0)

CROSS_RULES = {("ABA", "ABB"), ("ABB", "ABA")}
WITHIN_RULES = {("ABA", "ABA"), ("ABB", "ABB")}

# Okabe-Ito subset used across this repo's PI reports (CVD-safe)
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#8C8C8C"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 180, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})


# ---------------------------------------------------------------- loading

def load_tom_fp(name: str) -> dict:
    return {k: np.load(FP_DIR / f"{name}_{k}.npy") for k in ("raw", "clamped")}


def build_code_fp(folder: str, rules: set) -> tuple[dict, list]:
    """Same construction as aggregate_conditions: raw = mean of per-cell
    scores; clamped = mean of per-cell clip(scores, 0)."""
    best = resolve_best_source(RESULTS_ROOT, [folder], rules)
    cells, prov = [], []
    for (tmpl, rule), e in sorted(best.items()):
        if tmpl != "template_indexed_list":
            continue  # template_list_append = removed zero-signal design
        s = torch.load(e["scores_pt"], map_location="cpu",
                       weights_only=False).float().numpy().ravel()
        assert s.size == N_LAYERS * N_HEADS, f"{folder} {tmpl} {rule}: {s.shape}"
        cells.append(s)
        prov.append({"condition": folder, "template": tmpl, "rule": rule,
                     "n": e["n"], "seed": e["seed_tag"]})
    raw = np.stack(cells)
    return {"raw": raw.mean(axis=0),
            "clamped": np.clip(raw, 0, None).mean(axis=0)}, prov


# ---------------------------------------------------------------- stats

def ols_with_bootstrap(target, comps):
    """target ~ a*comp1 + b*comp2 + c. Percentile bootstrap over heads."""
    X = np.column_stack(comps + [np.ones_like(target)])
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    r2 = 1 - np.sum((target - fitted) ** 2) / np.sum((target - target.mean()) ** 2)
    # unique contribution of each comp: R2_full - R2_without_comp
    unique = []
    for drop in range(len(comps)):
        Xd = np.column_stack([c for j, c in enumerate(comps) if j != drop]
                             + [np.ones_like(target)])
        bd, *_ = np.linalg.lstsq(Xd, target, rcond=None)
        fd = Xd @ bd
        r2d = 1 - np.sum((target - fd) ** 2) / np.sum((target - target.mean()) ** 2)
        unique.append(r2 - r2d)
    n = target.size
    boots = np.empty((N_BOOT, len(beta)))
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, n)
        bb, *_ = np.linalg.lstsq(X[idx], target[idx], rcond=None)
        boots[i] = bb
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {"beta": beta, "ci_lo": lo, "ci_hi": hi, "r2": float(r2),
            "unique_r2": unique, "fitted": fitted, "resid_ols": target - fitted}


def perm_nulls(target, summed):
    """Observed r(target, sum) against two nulls."""
    obs = float(np.corrcoef(target, summed)[0, 1])
    grid = summed.reshape(N_LAYERS, N_HEADS)
    null_full = np.empty(N_PERM)
    null_layer = np.empty(N_PERM)
    for i in range(N_PERM):
        null_full[i] = np.corrcoef(target, RNG.permutation(summed))[0, 1]
        shuf = np.take_along_axis(
            grid, RNG.permuted(np.tile(np.arange(N_HEADS), (N_LAYERS, 1)),
                               axis=1), axis=1).ravel()
        null_layer[i] = np.corrcoef(target, shuf)[0, 1]
    def summ(null):
        return {"null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
                "null_max": float(null.max()),
                "p": float((np.sum(null >= obs) + 1) / (N_PERM + 1))}
    return obs, {"full_shuffle": {**summ(null_full), "draws": null_full},
                 "within_layer_shuffle": {**summ(null_layer), "draws": null_layer}}


# ---------------------------------------------------------------- figures

LAYER_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "blues_vis", plt.cm.Blues(np.linspace(0.30, 1.0, 256)))


def scatter_panel(ax, summed, target, sum_label, tgt_label, ols):
    layer = np.repeat(np.arange(N_LAYERS), N_HEADS)
    # single-hue sequential, truncated to the visible band of Blues
    sc = ax.scatter(summed, target, c=layer, cmap=LAYER_CMAP, s=9,
                    alpha=0.75, vmin=0, vmax=N_LAYERS - 1, linewidths=0)
    lims = [min(summed.min(), target.min()), max(summed.max(), target.max())]
    ax.plot(lims, lims, ls="--", lw=1.2, color="#333333", zorder=3)
    xs = np.array(lims)
    a, b, c = ols["beta"]
    # OLS line drawn in comp-sum space is only exact when a==b; plot the
    # fitted-vs-target relation instead as a light orange loess-free guide
    fit_line = np.polyfit(summed, target, 1)
    ax.plot(xs, np.polyval(fit_line, xs), lw=1.6, color=ORANGE, zorder=3)
    r_p = np.corrcoef(summed, target)[0, 1]
    r_s = sps.spearmanr(summed, target).statistic
    ax.set_xlabel(sum_label)
    ax.set_ylabel(tgt_label)
    ax.text(0.03, 0.97, f"Pearson r = {r_p:.3f}\nSpearman ρ = {r_s:.3f}\n"
                        f"slope (sum) = {fit_line[0]:.2f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.75, pad=2))
    return sc


def fig_scatters(name, fps, tgt, comps, labels, ols_by_key, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    for ax, key in zip(axes, ("clamped", "raw")):
        summed = fps[comps[0]][key] + fps[comps[1]][key]
        sc = scatter_panel(ax, summed, fps[tgt][key],
                           f"{labels[comps[0]]} + {labels[comps[1]]}  ({key})",
                           f"{labels[tgt]}  ({key})", ols_by_key[key])
        ax.set_title(f"{key} fingerprints")
    handles = [Line2D([], [], ls="--", lw=1.2, color="#333333", label="y = x (additive prediction)"),
               Line2D([], [], lw=1.6, color=ORANGE, label="best-fit line")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("layer", fontsize=9)
    fig.suptitle(f"{name}: one dot per head (n=1920)", y=1.02, fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def fig_coefficients(rows, path):
    """Dot-whisker: rows = (label, a, lo, hi, color)."""
    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(rows) + 1.4))
    ys = np.arange(len(rows))[::-1]
    for y, (label, b, lo, hi, color) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], lw=2.2, color=color, solid_capstyle="round")
        ax.plot(b, y, "o", ms=7, color=color, mec="white", mew=1.2, zorder=3)
        ax.text(hi + 0.03, y, f"{b:.2f} [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=8.5, color="#333333")
    ax.axvline(1.0, color="#333333", lw=1.1, ls="--")
    ax.axvline(0.0, color=GRAY, lw=0.9, ls=":")
    ax.text(1.0, len(rows) - 0.25, " a = b = 1: unweighted additivity",
            fontsize=8.5, color="#333333")
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("OLS coefficient (95% bootstrap CI, 10,000 resamples)")
    ax.set_xlim(left=min(0, min(r[2] for r in rows)) - 0.1,
                right=max(r[3] for r in rows) + 0.75)
    ax.grid(axis="y", alpha=0)
    fig.savefig(path)
    plt.close(fig)


def fig_residual_maps(panels, path):
    """panels: list of (title, residual_1920). Diverging map, shared scale
    per panel, symmetric around 0 (residuals are signed by construction —
    the PI clamp-negatives rule applies to CMA score maps, not residuals)."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4))
    axes = np.atleast_1d(axes)
    for ax, (title, resid) in zip(axes, panels):
        grid = resid.reshape(N_LAYERS, N_HEADS)
        v = np.abs(grid).max()
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v,
                       origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    fig.savefig(path)
    plt.close(fig)


def fig_null_hists(items, path):
    """items: list of (title, obs, nulls_dict)."""
    fig, axes = plt.subplots(1, len(items), figsize=(5.4 * len(items), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (title, obs, nulls) in zip(axes, items):
        bins = np.linspace(-0.4, 1.0, 120)
        ax.hist(nulls["full_shuffle"]["draws"], bins=bins, color=GRAY,
                alpha=0.65, label="null: full head shuffle")
        ax.hist(nulls["within_layer_shuffle"]["draws"], bins=bins, color=BLUE,
                alpha=0.65, label="null: within-layer shuffle")
        ax.axvline(obs, color=ORANGE, lw=2.0)
        ax.text(obs - 0.02, ax.get_ylim()[1] * 0.96, f"observed r = {obs:.3f} ",
                ha="right", va="top", fontsize=9, color=ORANGE)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Pearson r under H0")
        ax.set_ylabel("permutations (of 5,000)")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(path)
    plt.close(fig)


def fig_competitors(groups, path):
    """groups: list of (panel_title, [(label, r, highlight)])."""
    fig, axes = plt.subplots(1, len(groups), figsize=(6.2 * len(groups), 3.4))
    fig.subplots_adjust(wspace=0.62)
    axes = np.atleast_1d(axes)
    for ax, (title, rows) in zip(axes, groups):
        ys = np.arange(len(rows))[::-1]
        for y, (label, r, hl) in zip(ys, rows):
            ax.barh(y, r, height=0.62, color=BLUE if hl else GRAY, alpha=0.95)
            ax.text(r + 0.012, y, f"{r:.3f}", va="center", fontsize=9,
                    fontweight="bold" if hl else "normal", color="#333333")
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("Pearson r with target (clamped)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    fps = {
        "C1": load_tom_fp("C1_abstract_belief"),
        "C2": load_tom_fp("C2_abstract_photo"),
        "C3": load_tom_fp("C3_answer_changes_belief"),
        "C5": load_tom_fp("C5_answer_changes_location_only"),
        "C6": load_tom_fp("C6_answer_changes_location_only_photo"),
    }
    prov_rows = []
    fps["code_abs"], p1 = build_code_fp("code_abstract", CROSS_RULES)
    fps["code_ac"], p2 = build_code_fp("code_answer_changes", CROSS_RULES)
    fps["code_ctl"], p3 = build_code_fp("code_control", WITHIN_RULES)
    prov_rows = p1 + p2 + p3

    LABELS = {"C1": "C1 (belief)", "C3": "C3 (belief+answer)", "C5": "C5 (answer)",
              "code_abs": "code abstract", "code_ac": "code answer-changes",
              "code_ctl": "code control"}

    problems = [
        ("ToM", "C3", ["C1", "C5"], ("C2", "C6")),      # competitor = C2+C6
        ("code", "code_ac", ["code_abs", "code_ctl"], None),
    ]

    corr_rows, ols_rows, null_rows = [], [], []
    ols_store, null_store, resid_store = {}, {}, {}

    for domain, tgt, comps, competitor in problems:
        for key in ("clamped", "raw"):
            t = fps[tgt][key]
            c1, c2 = fps[comps[0]][key], fps[comps[1]][key]
            summed = c1 + c2

            pairs = {f"{tgt}~{comps[0]}": c1, f"{tgt}~{comps[1]}": c2,
                     f"{tgt}~sum({comps[0]}+{comps[1]})": summed}
            if competitor:
                pairs[f"{tgt}~sum({competitor[0]}+{competitor[1]})"] = (
                    fps[competitor[0]][key] + fps[competitor[1]][key])
            for pair, v in pairs.items():
                corr_rows.append({
                    "domain": domain, "key": key, "pair": pair,
                    "pearson_r": float(np.corrcoef(t, v)[0, 1]),
                    "spearman_rho": float(sps.spearmanr(t, v).statistic)})

            ols = ols_with_bootstrap(t, [c1, c2])
            ols_store[(domain, key)] = ols
            for i, comp in enumerate(comps):
                ols_rows.append({
                    "domain": domain, "key": key, "coef": f"beta[{comp}]",
                    "estimate": float(ols["beta"][i]),
                    "ci_lo": float(ols["ci_lo"][i]), "ci_hi": float(ols["ci_hi"][i]),
                    "unique_R2": float(ols["unique_r2"][i]), "R2_full": ols["r2"]})
            ols_rows.append({
                "domain": domain, "key": key, "coef": "intercept",
                "estimate": float(ols["beta"][2]),
                "ci_lo": float(ols["ci_lo"][2]), "ci_hi": float(ols["ci_hi"][2]),
                "unique_R2": np.nan, "R2_full": ols["r2"]})

            obs, nulls = perm_nulls(t, summed)
            null_store[(domain, key)] = (obs, nulls)
            for null_name, d in nulls.items():
                null_rows.append({
                    "domain": domain, "key": key, "null": null_name,
                    "observed_r": obs, "null_mean": d["null_mean"],
                    "null_p95": d["null_p95"], "null_max": d["null_max"],
                    "p_value": d["p"]})

            resid_store[(domain, key, "sum")] = t - summed
            resid_store[(domain, key, "ols")] = ols["resid_ols"]

    # ------------------------------------------------------------ figures
    fig_scatters("ToM decomposition", fps, "C3", ["C1", "C5"], LABELS,
                 {k: ols_store[("ToM", k)] for k in ("clamped", "raw")},
                 FIG / "fig1_scatter_tom.png")
    fig_scatters("Code decomposition", fps, "code_ac", ["code_abs", "code_ctl"],
                 LABELS, {k: ols_store[("code", k)] for k in ("clamped", "raw")},
                 FIG / "fig2_scatter_code.png")

    rows = []
    for domain, tgt, comps, _ in problems:
        for key in ("clamped", "raw"):
            ols = ols_store[(domain, key)]
            for i, comp in enumerate(comps):
                rows.append((f"{domain} · {key} · a[{LABELS[comp]}]",
                             ols["beta"][i], ols["ci_lo"][i], ols["ci_hi"][i],
                             BLUE if domain == "ToM" else GREEN))
    fig_coefficients(rows, FIG / "fig3_coefficients.png")

    fig_residual_maps(
        [("ToM: C3 − (C1+C5), clamped", resid_store[("ToM", "clamped", "sum")]),
         ("ToM: OLS residual, clamped", resid_store[("ToM", "clamped", "ols")])],
        FIG / "fig4_residuals_tom.png")
    fig_residual_maps(
        [("code: ac − (abs+ctl), clamped", resid_store[("code", "clamped", "sum")]),
         ("code: OLS residual, clamped", resid_store[("code", "clamped", "ols")])],
        FIG / "fig5_residuals_code.png")

    fig_null_hists(
        [("ToM: r(C3, C1+C5) clamped", *null_store[("ToM", "clamped")]),
         ("code: r(ac, abs+ctl) clamped", *null_store[("code", "clamped")])],
        FIG / "fig6_nulls.png")

    corr_c = {(r["domain"], r["pair"]): r["pearson_r"]
              for r in corr_rows if r["key"] == "clamped"}
    fig_competitors(
        [("ToM: what best predicts C3?",
          [("C1 alone", corr_c[("ToM", "C3~C1")], False),
           ("C5 alone", corr_c[("ToM", "C3~C5")], False),
           ("C2+C6 (photo-frame sum)", corr_c[("ToM", "C3~sum(C2+C6)")], False),
           ("C1+C5 (claimed sum)", corr_c[("ToM", "C3~sum(C1+C5)")], True)]),
         ("code: what best predicts answer-changes?",
          [("code_abstract alone", corr_c[("code", "code_ac~code_abs")], False),
           ("code_control alone", corr_c[("code", "code_ac~code_ctl")], False),
           ("abs+ctl (claimed sum)", corr_c[("code", "code_ac~sum(code_abs+code_ctl)")], True)])],
        FIG / "fig7_competitors.png")

    # ------------------------------------------------------------ tables
    df_corr = pd.DataFrame(corr_rows)
    df_ols = pd.DataFrame(ols_rows)
    df_null = pd.DataFrame(null_rows)
    df_corr.to_csv(OUT / "stats_correlations.csv", index=False)
    df_ols.to_csv(OUT / "stats_ols.csv", index=False)
    df_null.to_csv(OUT / "stats_nulls.csv", index=False)
    pd.DataFrame(prov_rows).to_csv(OUT / "stats_code_cells.csv", index=False)

    resid = resid_store[("ToM", "clamped", "sum")]
    order = np.argsort(-np.abs(resid))[:12]
    head_rows = [{"layer": int(i // N_HEADS), "head": int(i % N_HEADS),
                  "residual": float(resid[i]),
                  "C3": float(fps["C3"]["clamped"][i]),
                  "C1+C5": float(fps["C1"]["clamped"][i] + fps["C5"]["clamped"][i])}
                 for i in order]
    df_heads = pd.DataFrame(head_rows)
    df_heads.to_csv(OUT / "stats_residual_heads.csv", index=False)

    # markdown fragment for the qmd
    def md(df, floatfmt="%.3f"):
        return df.to_markdown(index=False, floatfmt=floatfmt.replace("%", "").replace("f", "f"))

    frag = ["<!-- AUTO-GENERATED by scripts/h2_additivity_robustness.py — do not edit -->\n"]
    frag.append("### Correlations (all 1920 heads)\n")
    frag.append(df_corr.round(3).to_markdown(index=False) + "\n")
    frag.append("\n### OLS fits: target = a·comp1 + b·comp2 + c\n")
    frag.append(df_ols.round(3).to_markdown(index=False) + "\n")
    frag.append("\n### Permutation nulls for r(target, unweighted sum)\n")
    frag.append(df_null.round(4).to_markdown(index=False) + "\n")
    frag.append("\n### Largest ToM additivity residuals (clamped), top 12 heads\n")
    frag.append(df_heads.round(3).to_markdown(index=False) + "\n")
    frag.append("\n### Code-arm source cells\n")
    frag.append(pd.DataFrame(prov_rows).to_markdown(index=False) + "\n")
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    # ------------------------------------------------------------ sanity
    print("== sanity vs published numbers ==")
    print(f"clamped r(C3, C1+C5)      = {corr_c[('ToM','C3~sum(C1+C5)')]:.3f}  (published 0.971)")
    print(f"clamped r(C3, C1)         = {corr_c[('ToM','C3~C1')]:.3f}  (published 0.834)")
    print(f"clamped r(C3, C5)         = {corr_c[('ToM','C3~C5')]:.3f}  (published 0.714)")
    print(f"clamped r(ac, abs+ctl)    = {corr_c[('code','code_ac~sum(code_abs+code_ctl)')]:.3f}  (n=50-era 0.992)")
    print("\n== headline robustness numbers ==")
    for domain in ("ToM", "code"):
        for key in ("clamped", "raw"):
            obs, nulls = null_store[(domain, key)]
            o = ols_store[(domain, key)]
            print(f"{domain:4s} {key:7s}: r_sum={obs:.3f} | a={o['beta'][0]:.2f} "
                  f"b={o['beta'][1]:.2f} c={o['beta'][2]:.3f} R2={o['r2']:.3f} | "
                  f"p_full={nulls['full_shuffle']['p']:.2e} "
                  f"p_layer={nulls['within_layer_shuffle']['p']:.2e} "
                  f"(layer-null p95={nulls['within_layer_shuffle']['null_p95']:.3f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
