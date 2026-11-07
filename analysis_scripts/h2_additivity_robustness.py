"""H2 additivity robustness package.

Pearson correlation on clamped CMA fingerprints does not by itself establish
additive computation. This script adds the four checks that do: robustness
without clamping, magnitude/slope fits, residuals, and null baselines — for
all three decomposition problems:
  ToM  : C3 (answer_changes_belief)  ~  C1 (abstract_belief) + C5 (loc-only)
  code : code_answer_changes         ~  code_abstract + code_control
  gemma: same as ToM on Gemma-2-27B-it (46x32 = 1472 heads) — added July 2026
         because Reviewers 2 and 3 both flag the partial Gemma replication,
         so the robustness machinery runs on the replication arm too.

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

Outputs -> results/summary/h2_robustness/
  figures/*.png, stats_correlations.csv, stats_ols.csv, stats_nulls.csv,
  stats_residual_heads.csv, stats_fragment.md (tables for the .qmd)
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy import stats as sps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis_scripts"))
from condition_registry import (  # noqa: E402
    resolve_best_source, lookup, FLIP_RULES, KEEP_RULES,
)

FP_DIR = REPO / "results/aggregated/fingerprints"
RESULTS_ROOT = REPO / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
GEMMA_ROOT = REPO / "results/causal_analysis/cma/google/gemma-2-27b-it_ungroup_gqa"
OUT = REPO / "results/summary/h2_robustness"
FIG = OUT / "figures"

N_LAYERS, N_HEADS = 48, 40          # Qwen2.5-14B-Instruct
QWEN_SHAPE = (N_LAYERS, N_HEADS)
GEMMA_SHAPE = (46, 32)              # gemma-2-27b-it, ungrouped GQA
N_PERM = 5000
N_BOOT = 10000
RNG = np.random.default_rng(0)

CROSS_RULES = FLIP_RULES
WITHIN_RULES = KEEP_RULES

# Okabe-Ito subset used across this repo's summary reports (CVD-safe)
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


def load_cells(root, folder: str, rules: set, shape: tuple,
               keep_template=None) -> tuple[dict, list]:
    """Per-cell causal-score vectors, keyed by (template, direction).

    direction = the *base* rule: ABA = primary (false->true), ABB = reverse.
    Keying on direction rather than the full rule pair is what lets the
    residual significance test pair a cross-context condition (ABA->ABB)
    with a within-context one (ABA->ABA) cell for cell.
    """
    n_expect = shape[0] * shape[1]
    best = resolve_best_source(root, [folder], rules)
    cells, prov = {}, []
    for (tmpl, rule), e in sorted(best.items()):
        if keep_template is not None and not keep_template(tmpl):
            continue
        s = torch.load(e["scores_pt"], map_location="cpu",
                       weights_only=False).float().numpy().ravel()
        assert s.size == n_expect, f"{folder} {tmpl} {rule}: {s.shape}"
        cells[(tmpl, rule.split("_")[2])] = s
        prov.append({"condition": folder, "template": tmpl, "rule": rule,
                     "n": e["n"], "seed": e["seed_tag"]})
    assert cells, f"no cells resolved for {folder} under {root}"
    return cells, prov


def build_fp(root, folder: str, rules: set, shape: tuple,
             keep_template=None) -> tuple[dict, list]:
    """Same construction as aggregate_conditions: raw = mean of per-cell
    scores; clamped = mean of per-cell clip(scores, 0).

    keep_template: optional predicate on the template folder name, used by
    the code arm to drop the removed zero-signal `template_list_append` runs.
    """
    cells, prov = load_cells(root, folder, rules, shape, keep_template)
    raw = np.stack([cells[k] for k in sorted(cells)])
    return {"raw": raw.mean(axis=0),
            "clamped": np.clip(raw, 0, None).mean(axis=0)}, prov


def build_code_fp(folder: str, rules: set) -> tuple[dict, list]:
    templates = lookup(folder).templates
    return build_fp(RESULTS_ROOT, folder, rules, QWEN_SHAPE,
                    keep_template=lambda t: t in templates)


def build_gemma_fp(folder: str, rules: set) -> tuple[dict, list]:
    """Gemma replication arm. Seed 0 is all Gemma has."""
    return build_fp(GEMMA_ROOT, folder, rules, GEMMA_SHAPE)


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


def layer_localization(v, shape, strata_by=None, n_perm=N_PERM):
    """Is |v| concentrated in a layer band?

    Statistic: max over layers of the mean |value| in that layer. Two nulls,
    both shuffling head identity across the whole grid:
      plain      — any layer concentration at all;
      stratified — heads are permuted only within quantile bins of
                   `strata_by` (here |comp1 + comp2|), so ANY monotone
                   dependence of |v| on component magnitude is held fixed and
                   only layer identity moves. This is what separates "the
                   residual is big where the components are big" (scaling)
                   from "the residual is big at a particular depth"
                   (machinery the sum does not capture).

    NOTE — do not replace this with the signed per-layer mean. Positive and
    negative heads inside a layer cancel, and the resulting test cannot even
    detect the known band in the C1 fingerprint (p = 0.62 on Gemma). The
    positive controls in main() exist to keep this honest.
    """
    a = np.abs(v)
    stat = lambda x: x.reshape(shape).mean(1).max()
    obs = float(stat(a))
    if strata_by is None:
        null = np.array([stat(RNG.permutation(a)) for _ in range(n_perm)])
    else:
        q = np.quantile(np.abs(strata_by), np.linspace(0, 1, 21)[1:-1])
        bins = np.searchsorted(q, np.abs(strata_by))
        groups = [np.where(bins == k)[0] for k in range(bins.max() + 1)]
        null = np.empty(n_perm)
        for i in range(n_perm):
            x = a.copy()
            for g in groups:
                x[g] = a[RNG.permutation(g)]
            null[i] = stat(x)
    return {"max_layer_mean_abs": obs,
            "null_p95": float(np.percentile(null, 95)),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "peak_layer": int(np.argmax(a.reshape(shape).mean(1)))}


def residual_significance(target_cells, comp_cells, shape, clamp):
    """Do the additivity residuals carry signal, or are they noise?

    The residual map alone cannot answer this — a signed diverging map always
    *looks* structured. Four tests, none of them thresholded on a top-k list:

    (A) per-head: is any single head's residual reliably nonzero? Per-cell
        residual r_c = target_c - (comp1_c + comp2_c) over the matched cells
        (paired by template AND direction), then t = mean/SE across cells,
        FWER via the project's max-statistic convention with an EXACT
        sign-flip null (2^n_cells). Low power by construction: with 6 cells
        the smallest achievable p is 1/64 = 0.016. Report as "no head
        survives correction", never as "the residuals are zero".

    (A2) whole-population reproducibility: for every head, how many of the
        cells agree on the sign of its mean residual, against the coin-flip
        expectation. Uses all heads, so it says whether reproducible
        residuals exist anywhere without picking any out.

    (B) layer localization, plain and magnitude-stratified — see
        layer_localization(). The stratified p is the one that answers "does
        the additive model fail at a particular depth".

    (C) spatial clustering: Moran's I on the 4-neighbour head lattice, plus
        the largest 4-connected positive patch vs its null. Diverging
        colormaps invite the eye to read patches into near-zero noise.
    """
    f = (lambda x: np.clip(x, 0, None)) if clamp else (lambda x: x)
    keys = sorted(set(target_cells) & set(comp_cells[0]) & set(comp_cells[1]))
    R = np.stack([f(target_cells[k]) - (f(comp_cells[0][k]) + f(comp_cells[1][k]))
                  for k in keys])                      # n_cells x n_heads
    n = R.shape[0]
    m = R.mean(0)
    summed = (np.stack([f(comp_cells[0][k]) for k in keys]).mean(0)
              + np.stack([f(comp_cells[1][k]) for k in keys]).mean(0))
    out = {"n_cells": n, "n_heads": int(m.size),
           "mean_residual": float(m.mean()),
           "frac_negative": float((m < 0).mean())}

    # ---- (A) per-head sign-flip max-|t|, exact enumeration
    def tstat(A):
        return A.mean(0) / (A.std(0, ddof=1) / np.sqrt(n) + 1e-12)
    t = tstat(R)
    null_t = np.array([np.abs(tstat(R * np.array(s)[:, None])).max()
                       for s in itertools.product([1, -1], repeat=n)])
    thr = float(np.percentile(null_t, 95))
    out.update({
        "max_abs_t": float(np.abs(t).max()), "fwer_t_threshold": thr,
        "n_sig_fwer": int((np.abs(t) > thr).sum()),
        "p_min_achievable": 1.0 / null_t.size,
        "n_sig_uncorrected": int((np.abs(t) > sps.t.ppf(0.975, n - 1)).sum()),
        "n_expected_uncorrected": 0.05 * m.size})

    # ---- (A2) sign agreement across cells, every head, vs coin flips
    agree = (np.sign(R) == np.sign(m)).sum(0)
    for k in range(n // 2 + 1, n + 1):
        # P(k of n cells agree with the majority sign), fair-coin baseline
        exp = 2 * sps.binom.pmf(k, n, 0.5)
        out[f"frac_heads_{k}of{n}_agree"] = float((agree == k).mean())
        out[f"expected_{k}of{n}_agree"] = float(exp)

    # ---- (B) layer localization
    plain = layer_localization(m, shape)
    strat = layer_localization(m, shape, strata_by=summed)
    out.update({"layer_max_mean_abs": plain["max_layer_mean_abs"],
                "layer_peak": plain["peak_layer"],
                "layer_p_plain": plain["p"],
                "layer_p_magnitude_matched": strat["p"],
                "sum_peak_layer": int(np.abs(summed).reshape(shape).mean(1).argmax())})

    # ---- (C) spatial clustering
    def moran(v):
        z = v.reshape(shape) - v.mean()
        num = (z[:-1, :] * z[1:, :]).sum() + (z[:, :-1] * z[:, 1:]).sum()
        w = z[:-1, :].size + z[:, :-1].size
        return float((z.size / w) * num / (z ** 2).sum())

    def biggest_patch(v):
        lab, k = ndimage.label(v.reshape(shape) > 0)
        return 0 if k == 0 else int(ndimage.sum(np.ones(shape), lab,
                                                range(1, k + 1)).max())

    obs_i = moran(m)
    null_i = np.array([moran(RNG.permutation(m)) for _ in range(N_PERM)])
    obs_patch = biggest_patch(m)
    null_patch = np.array([biggest_patch(RNG.permutation(m)) for _ in range(1000)])
    out.update({
        "frac_positive": float((m > 0).mean()),
        "moran_I": obs_i,
        "moran_p": float((np.sum(null_i >= obs_i) + 1) / (N_PERM + 1)),
        "largest_positive_patch": obs_patch,
        "patch_null_p95": float(np.percentile(null_patch, 95)),
        "patch_p": float((np.sum(null_patch >= obs_patch) + 1) / 1001)})

    # ---- magnitude dependence: |resid| ~ alpha*|sum|, no intercept
    a, b = np.abs(m), np.abs(summed)
    out.update({"corr_resid_sum": float(np.corrcoef(m, summed)[0, 1]),
                "corr_absresid_abssum": float(np.corrcoef(a, b)[0, 1]),
                "alpha_absresid_per_abssum": float(np.sum(a * b) / np.sum(b * b))})
    return out


def perm_nulls(target, summed, shape=QWEN_SHAPE):
    """Observed r(target, sum) against two nulls."""
    n_layers, n_heads = shape
    obs = float(np.corrcoef(target, summed)[0, 1])
    grid = summed.reshape(n_layers, n_heads)
    null_full = np.empty(N_PERM)
    null_layer = np.empty(N_PERM)
    for i in range(N_PERM):
        null_full[i] = np.corrcoef(target, RNG.permutation(summed))[0, 1]
        shuf = np.take_along_axis(
            grid, RNG.permuted(np.tile(np.arange(n_heads), (n_layers, 1)),
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


def scatter_panel(ax, summed, target, sum_label, tgt_label, ols,
                  shape=QWEN_SHAPE):
    n_layers, n_heads = shape
    layer = np.repeat(np.arange(n_layers), n_heads)
    # single-hue sequential, truncated to the visible band of Blues
    sc = ax.scatter(summed, target, c=layer, cmap=LAYER_CMAP, s=9,
                    alpha=0.75, vmin=0, vmax=n_layers - 1, linewidths=0)
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


def fig_scatters(name, fps, tgt, comps, labels, ols_by_key, path,
                 shape=QWEN_SHAPE):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    fig.subplots_adjust(wspace=0.30)   # long axis labels must not collide
    for ax, key in zip(axes, ("clamped", "raw")):
        summed = fps[comps[0]][key] + fps[comps[1]][key]
        sc = scatter_panel(ax, summed, fps[tgt][key],
                           f"{labels[comps[0]]} + {labels[comps[1]]}  ({key})",
                           f"{labels[tgt]}  ({key})", ols_by_key[key], shape)
        ax.set_title(f"{key} fingerprints")
    handles = [Line2D([], [], ls="--", lw=1.2, color="#333333", label="y = x (additive prediction)"),
               Line2D([], [], lw=1.6, color=ORANGE, label="best-fit line")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    cb = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.015)
    cb.set_label("layer", fontsize=9)
    fig.suptitle(f"{name}: one dot per head (n={shape[0] * shape[1]})",
                 y=1.02, fontsize=12)
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


def fig_residual_maps(panels, path, shape=QWEN_SHAPE, ncols=2):
    """panels: list of (title, residual vector). Diverging map, symmetric
    around 0 (residuals are signed by construction — the clamp-negatives
    rule applies to CMA score maps, not residuals).

    Layer axis runs top-down (layer 0 at top), matching the sns.heatmap
    convention used by build_summary_heatmaps / build_avg_heatmaps — imshow's
    default origin. Do not set origin="lower" here: it mirrors these maps
    relative to every other head grid in this repo.

    Colour scale is per panel (each auto-scales to its own max), so panels
    are NOT visually comparable across arms — the range is printed in each
    title so the reader can see it.
    """
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.9 * ncols, 4.6 * nrows),
                             layout="constrained")   # titles must not collide
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, resid) in zip(axes, panels):
        grid = resid.reshape(shape)
        v = np.abs(grid).max()
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-v, vmax=v,
                       aspect="auto", interpolation="nearest")
        ax.set_title(f"{title}\n(range ±{v:.2f})", fontsize=9.5)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    for ax in axes[n:]:
        ax.axis("off")
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

    # Gemma replication arm (46x32 = 1472 heads). Within-model only: there is
    # no head correspondence across architectures, so no cross-model panel.
    gemma_prov = []
    for key, (folder, rules) in {
            "gC1": ("abstract_belief", CROSS_RULES),
            "gC3": ("answer_changes_belief", CROSS_RULES),
            "gC5": ("answer_changes_location_only", WITHIN_RULES)}.items():
        fps[key], pg = build_gemma_fp(folder, rules)
        gemma_prov.extend([{**r, "arm": key} for r in pg])

    LABELS = {"C1": "C1 (belief)", "C3": "C3 (belief+answer)", "C5": "C5 (answer)",
              "code_abs": "code abstract", "code_ac": "code answer-changes",
              "code_ctl": "code control",
              "gC1": "gC1 (belief)", "gC3": "gC3 (belief+answer)",
              "gC5": "gC5 (answer)"}

    problems = [
        ("ToM", "C3", ["C1", "C5"], ("C2", "C6"), QWEN_SHAPE),  # competitor = C2+C6
        ("code", "code_ac", ["code_abs", "code_ctl"], None, QWEN_SHAPE),
        ("gemma", "gC3", ["gC1", "gC5"], None, GEMMA_SHAPE),
    ]

    corr_rows, ols_rows, null_rows = [], [], []
    ols_store, null_store, resid_store = {}, {}, {}

    for domain, tgt, comps, competitor, shape in problems:
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

            obs, nulls = perm_nulls(t, summed, shape)
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
    fig_scatters("Gemma-2-27B decomposition", fps, "gC3", ["gC1", "gC5"],
                 LABELS, {k: ols_store[("gemma", k)] for k in ("clamped", "raw")},
                 FIG / "fig8_scatter_gemma.png", GEMMA_SHAPE)

    domain_color = {"ToM": BLUE, "code": GREEN, "gemma": ORANGE}
    rows = []
    for domain, tgt, comps, _, _ in problems:
        for key in ("clamped", "raw"):
            ols = ols_store[(domain, key)]
            for i, comp in enumerate(comps):
                rows.append((f"{domain} · {key} · a[{LABELS[comp]}]",
                             ols["beta"][i], ols["ci_lo"][i], ols["ci_hi"][i],
                             domain_color[domain]))
    fig_coefficients(rows, FIG / "fig3_coefficients.png")

    # Both keys are plotted: the clamped additivity residual carries a
    # mechanical negative offset (the summed components get clipped at 0
    # twice, the target once), so the raw row is the honest reference.
    fig_residual_maps(
        [("ToM: C3 − (C1+C5), clamped", resid_store[("ToM", "clamped", "sum")]),
         ("ToM: OLS residual, clamped", resid_store[("ToM", "clamped", "ols")]),
         ("ToM: C3 − (C1+C5), raw", resid_store[("ToM", "raw", "sum")]),
         ("ToM: OLS residual, raw", resid_store[("ToM", "raw", "ols")])],
        FIG / "fig4_residuals_tom.png")
    fig_residual_maps(
        [("code: ac − (abs+ctl), clamped", resid_store[("code", "clamped", "sum")]),
         ("code: OLS residual, clamped", resid_store[("code", "clamped", "ols")]),
         ("code: ac − (abs+ctl), raw", resid_store[("code", "raw", "sum")]),
         ("code: OLS residual, raw", resid_store[("code", "raw", "ols")])],
        FIG / "fig5_residuals_code.png")

    fig_residual_maps(
        [("Gemma: gC3 − (gC1+gC5), clamped", resid_store[("gemma", "clamped", "sum")]),
         ("Gemma: OLS residual, clamped", resid_store[("gemma", "clamped", "ols")]),
         ("Gemma: gC3 − (gC1+gC5), raw", resid_store[("gemma", "raw", "sum")]),
         ("Gemma: OLS residual, raw", resid_store[("gemma", "raw", "ols")])],
        FIG / "fig9_residuals_gemma.png", GEMMA_SHAPE)

    fig_null_hists(
        [("ToM: r(C3, C1+C5) clamped", *null_store[("ToM", "clamped")]),
         ("code: r(ac, abs+ctl) clamped", *null_store[("code", "clamped")]),
         ("Gemma: r(gC3, gC1+gC5) clamped", *null_store[("gemma", "clamped")])],
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
           ("abs+ctl (claimed sum)", corr_c[("code", "code_ac~sum(code_abs+code_ctl)")], True)]),
         ("Gemma: what best predicts gC3?",
          [("gC1 alone", corr_c[("gemma", "gC3~gC1")], False),
           ("gC5 alone", corr_c[("gemma", "gC3~gC5")], False),
           ("gC1+gC5 (claimed sum)", corr_c[("gemma", "gC3~sum(gC1+gC5)")], True)])],
        FIG / "fig7_competitors.png")

    # ------------------------------------------------------------ tables
    df_corr = pd.DataFrame(corr_rows)
    df_ols = pd.DataFrame(ols_rows)
    df_null = pd.DataFrame(null_rows)
    df_corr.to_csv(OUT / "stats_correlations.csv", index=False)
    df_ols.to_csv(OUT / "stats_ols.csv", index=False)
    df_null.to_csv(OUT / "stats_nulls.csv", index=False)
    pd.DataFrame(prov_rows).to_csv(OUT / "stats_code_cells.csv", index=False)
    pd.DataFrame(gemma_prov).to_csv(OUT / "stats_gemma_cells.csv", index=False)

    # NB: an earlier version tabled the top-12 |residual| heads here. Removed —
    # a top-k cut has no principled k, and every claim it supported is now made
    # over all heads by residual_significance().

    # ------------------------------------- residual significance (A)/(B)/(C)
    # The code arm has only 2 cells (1 template x 2 directions): a sign-flip
    # null over 4 flips cannot reach p < 0.25, so it is excluded rather than
    # reported as a non-result.
    sig_specs = [
        ("ToM", RESULTS_ROOT, QWEN_SHAPE, "C3", "C1", "C5"),
        ("gemma", GEMMA_ROOT, GEMMA_SHAPE, "gC3", "gC1", "gC5"),
    ]
    FOLDER = {"C1": "abstract_belief", "C3": "answer_changes_belief",
              "C5": "answer_changes_location_only"}
    sig_rows, ctrl_rows = [], []
    for domain, root, shape, tgt, c1k, c5k in sig_specs:
        cells = {}
        for k, rules in ((c1k, CROSS_RULES), (tgt, CROSS_RULES),
                         (c5k, WITHIN_RULES)):
            cells[k], _ = load_cells(root, FOLDER[k.lstrip("g")], rules, shape)
        # the per-cell stack must reproduce the fingerprint used above, or the
        # residual stats would describe different data than the correlations
        for k in (tgt, c1k, c5k):
            built = np.stack([cells[k][j] for j in sorted(cells[k])]).mean(0)
            assert np.allclose(built, fps[k]["raw"], atol=1e-3), \
                f"{domain} {k}: per-cell stack != fingerprint"
        for key in ("clamped", "raw"):
            sig_rows.append({"domain": domain, "key": key,
                             **residual_significance(cells[tgt],
                                                     [cells[c1k], cells[c5k]],
                                                     shape, key == "clamped")})
        # POSITIVE CONTROL: the same layer-localization test on the component
        # fingerprints, which are known to have a band. If these do not come
        # out significant the test has no power and the residual's p-value
        # means nothing — which is exactly how the first version of this
        # analysis went wrong (signed per-layer mean, p = 0.62 on gC1).
        summed = fps[c1k]["raw"] + fps[c5k]["raw"]
        for k in (c1k, c5k, tgt):
            r = layer_localization(fps[k]["raw"], shape)
            rs = layer_localization(fps[k]["raw"], shape, strata_by=summed)
            ctrl_rows.append({"domain": domain, "map": f"{k} fingerprint (raw)",
                              "max_layer_mean_abs": r["max_layer_mean_abs"],
                              "null_p95": r["null_p95"], "p_plain": r["p"],
                              "p_magnitude_matched": rs["p"],
                              "peak_layer": r["peak_layer"]})
    df_sig = pd.DataFrame(sig_rows)
    df_ctrl = pd.DataFrame(ctrl_rows)
    df_sig.to_csv(OUT / "stats_residual_significance.csv", index=False)
    df_ctrl.to_csv(OUT / "stats_layer_localization_control.csv", index=False)
    (OUT / "stats_residual_heads.csv").unlink(missing_ok=True)      # top-12, removed
    (OUT / "stats_residual_top_heads.csv").unlink(missing_ok=True)  # top-12, removed

    # markdown fragment for the qmd
    def md(df, floatfmt="%.3f"):
        return df.to_markdown(index=False, floatfmt=floatfmt.replace("%", "").replace("f", "f"))

    frag = ["<!-- AUTO-GENERATED by analysis_scripts/h2_additivity_robustness.py — do not edit -->\n"]
    frag.append("### Correlations (all 1920 heads)\n")
    frag.append(df_corr.round(3).to_markdown(index=False) + "\n")
    frag.append("\n### OLS fits: target = a·comp1 + b·comp2 + c\n")
    frag.append(df_ols.round(3).to_markdown(index=False) + "\n")
    frag.append("\n### Permutation nulls for r(target, unweighted sum)\n")
    frag.append(df_null.round(4).to_markdown(index=False) + "\n")
    frag.append("\n### Do the residuals carry structure?\n")
    frag.append(
        "Per-cell residuals over the 6 matched cells (3 templates x 2 "
        "directions, paired by template and direction). No top-k anywhere: "
        "every statistic below is over all heads.\n\n"
        "- `n_sig_fwer` — heads whose |t| beats the 95th percentile of an "
        "exact sign-flip max-|t| null (2^6 = 64 flips). With 6 cells the "
        "smallest achievable p is 0.016, so 0 means *no head survives "
        "correction*, not *the residuals are zero*.\n"
        "- `frac_heads_6of6_agree` vs `expected_6of6_agree` — reproducibility "
        "across cells for the whole population: how often all 6 cells agree "
        "on a head's residual sign, against the fair-coin baseline.\n"
        "- `layer_p_plain` / `layer_p_magnitude_matched` — max per-layer mean "
        "|residual| against a whole-grid head shuffle, then against a shuffle "
        "restricted to quantile bins of |comp1+comp2|. The stratified one is "
        "the claim-relevant test: it holds any monotone "
        "bigger-components-bigger-residual dependence fixed and asks whether "
        "a particular *depth* is special. See the positive-control table "
        "below for its power.\n"
        "- `moran_p`, `patch_p` — spatial clustering of same-signed residuals "
        "on the head lattice. Read `largest_positive_patch` only where the "
        "minority sign is sparse: ToM raw is 61.5% positive, so patches "
        "percolate (1020 heads, null p95 1032) and the statistic is empty "
        "there; it is informative on Gemma raw (8.7% positive) and on both "
        "clamped keys.\n\n"
        "The code arm is omitted: 2 cells cannot reach p < 0.25.\n")
    sig_cols = ["domain", "key", "n_cells", "mean_residual", "frac_negative",
                "max_abs_t", "fwer_t_threshold", "n_sig_fwer",
                "n_sig_uncorrected", "n_expected_uncorrected",
                "frac_heads_6of6_agree", "expected_6of6_agree",
                "layer_max_mean_abs", "layer_peak", "sum_peak_layer",
                "layer_p_plain", "layer_p_magnitude_matched",
                "moran_I", "moran_p", "largest_positive_patch",
                "patch_null_p95", "patch_p",
                "corr_resid_sum", "corr_absresid_abssum",
                "alpha_absresid_per_abssum"]
    frag.append(df_sig[sig_cols].round(4).to_markdown(index=False) + "\n")
    frag.append("\n### Positive control for the layer-localization test\n")
    frag.append(
        "The same test applied to the component fingerprints, which are known "
        "to have a late-layer band. Small `p_plain` there = the test can see "
        "a band when one exists, so the residual's p-value is interpretable. "
        "(The first version of this analysis used the *signed* per-layer mean "
        "and returned p = 0.62 on the gC1 fingerprint itself — positives and "
        "negatives cancel within a layer. That test was discarded.)\n\n"
        "The control validates `p_plain` only. `p_magnitude_matched` is "
        "circular here — the strata come from |C1+C5|, so stratifying a "
        "component fingerprint removes its own band by construction (hence "
        "C1 0.14, C3 0.54). It is non-circular for the residual, which is a "
        "different quantity from the sum used to build the strata.\n")
    frag.append(df_ctrl.round(4).to_markdown(index=False) + "\n")
    frag.append("\n### Code-arm source cells\n")
    frag.append(pd.DataFrame(prov_rows).to_markdown(index=False) + "\n")
    frag.append("\n### Gemma-arm source cells\n")
    frag.append(pd.DataFrame(gemma_prov).to_markdown(index=False) + "\n")
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    # ------------------------------------------------------------ sanity
    print("== sanity vs published numbers ==")
    print(f"clamped r(C3, C1+C5)      = {corr_c[('ToM','C3~sum(C1+C5)')]:.3f}  (published 0.971)")
    print(f"clamped r(C3, C1)         = {corr_c[('ToM','C3~C1')]:.3f}  (published 0.834)")
    print(f"clamped r(C3, C5)         = {corr_c[('ToM','C3~C5')]:.3f}  (published 0.714)")
    print(f"clamped r(ac, abs+ctl)    = {corr_c[('code','code_ac~sum(code_abs+code_ctl)')]:.3f}  (n=50-era 0.992)")
    print(f"clamped r(gC3, gC1+gC5)   = {corr_c[('gemma','gC3~sum(gC1+gC5)')]:.3f}  (scatter pack 0.985)")
    print("\n== headline robustness numbers ==")
    for domain in ("ToM", "code", "gemma"):
        for key in ("clamped", "raw"):
            obs, nulls = null_store[(domain, key)]
            o = ols_store[(domain, key)]
            print(f"{domain:4s} {key:7s}: r_sum={obs:.3f} | a={o['beta'][0]:.2f} "
                  f"b={o['beta'][1]:.2f} c={o['beta'][2]:.3f} R2={o['r2']:.3f} | "
                  f"p_full={nulls['full_shuffle']['p']:.2e} "
                  f"p_layer={nulls['within_layer_shuffle']['p']:.2e} "
                  f"(layer-null p95={nulls['within_layer_shuffle']['null_p95']:.3f})")
    print("\n== layer-localization positive control (raw fingerprints) ==")
    for r in ctrl_rows:
        print(f"{r['domain']:5s} {r['map']:22s}: p_plain={r['p_plain']:.4f} "
              f"p_magmatched={r['p_magnitude_matched']:.4f} (peak L{r['peak_layer']})")
    print("\n== residual structure ==")
    for r in sig_rows:
        print(f"{r['domain']:5s} {r['key']:7s}: {100*r['frac_negative']:.1f}% negative | "
              f"max|t|={r['max_abs_t']:.2f} vs FWER thr {r['fwer_t_threshold']:.2f} "
              f"-> {r['n_sig_fwer']}/{r['n_heads']} sig "
              f"({r['n_sig_uncorrected']} uncorrected, "
              f"{r['n_expected_uncorrected']:.0f} expected) | "
              f"6/6 sign agree {100*r['frac_heads_6of6_agree']:.1f}% "
              f"(chance {100*r['expected_6of6_agree']:.1f}%) | "
              f"layer L{r['layer_peak']} p={r['layer_p_plain']:.4f}, "
              f"magnitude-matched p={r['layer_p_magnitude_matched']:.4f} "
              f"(|sum| peaks L{r['sum_peak_layer']}) | "
              f"Moran p={r['moran_p']:.3f} | "
              f"|resid|={r['alpha_absresid_per_abssum']:.3f}*|sum| "
              f"(r={r['corr_absresid_abssum']:+.2f})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
