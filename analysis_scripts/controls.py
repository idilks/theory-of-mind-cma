#!/usr/bin/env python
"""Robustness controls for the cross-condition aggregation.

1. Partial correlation matrix controlling for the across-context indicator.
   The cell-level Mantel test showed the across/within-context axis
   explains cell similarity better than condition identity. For each
   pair (C_i, C_j), Pearson r(C_i, C_j | across-context indicator) on
   per-cell vectors, with cells as observations, quantifies how much of
   the condition-condition similarity remains once that axis is partialled out.
   The across-context indicator = 1 for C1/C2/C3/C4 (false_template <->
   true_template patches) and 0 for C5/C6 (within-context, no template flip).

2. Bootstrap null for the H2 OLS R^2.
   Regressing a target onto two predictors that each correlate ~0.7-0.8 with
   it and are partly orthogonal to each other constrains R^2 to be high on
   arithmetic alone. The relevant null is therefore: what R^2 do random
   component vectors with matched marginal correlation to the target produce?

Generates a "matched-r" null:
   * For each pair of components, sample 1000 pairs (X1, X2) of random
     1920-vecs such that r(target, X1) ~= r(target, C1) +/- 0.02 and
     r(target, X2) ~= r(target, C5) +/- 0.02 (same for the C4 fits).
   * Fit OLS, record R^2.
   * Report observed R^2 vs null distribution -> percentile p.

Outputs:
- results/aggregated/controls/partial_corr_pearson.csv
- results/aggregated/controls/h2_bootstrap_null.csv
- results/summary/aggregation/controls_partial_corr.png
- results/summary/aggregation/controls_h2_null.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/summary/heatmaps/manifest.csv"
FP_DIR = ROOT / "results/aggregated/fingerprints"
OUT_AGG = ROOT / "results/aggregated/controls"
OUT_FIG = ROOT / "results/summary/aggregation"

CONDITIONS = [
    "C1_abstract_belief", "C2_abstract_photo", "C3_answer_changes_belief",
    "C4_answer_changes_photo", "C5_answer_changes_location_only",
    "C6_answer_changes_location_only_photo", "C7_state_only_no_belief",
]
SHORT = {c: c.split("_", 1)[0] for c in CONDITIONS}
# Context-axis indicator: across-context (false_template <-> true_template
# patch) vs within-context (no template flip, pure location retrieval).
# This is the substantive cognitive axis. See aggregate_conditions.py header
# and tom_templates.py:580-611 for why the legacy folder labels do not
# correspond to surface-token swaps for C3/C4.
ACROSS_CONTEXT = {"C1_abstract_belief", "C2_abstract_photo",
                  "C3_answer_changes_belief", "C4_answer_changes_photo"}
N_HEADS = 1920
RNG = np.random.default_rng(20260501)


# ---------- (1) Partial correlation against across-context indicator ----------

def load_cell_vectors(manifest: pd.DataFrame) -> dict:
    out = {}
    for _, r in manifest.iterrows():
        src = ROOT / r["source"]
        v = torch.load(src / "causal_scores.pt", map_location="cpu", weights_only=False)
        v = v.float().cpu().numpy().reshape(-1)
        v = np.clip(v, 0, None)  # match clamped fingerprint convention
        out[(r["condition"], r["template"], r["rule_pair"])] = v
    return out


def partial_corr_against_context_axis(cells: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-head partial correlation: r(C_i, C_j | across-context indicator).

    Right framing: partial correlation across CELLS. Stack the 36 cells, treat
    the across-context binary as a covariate per cell, residualize each head's
    cell-vector against it, then build per-condition fingerprints from the
    residualized scores -> 6x6 partial Pearson matrix.

    The covariate = 1 for cells in C1/C2/C3/C4 (false<->true template patch),
    0 for C5/C6 (within-context patch). This isolates how much condition-vs-
    condition similarity remains after removing the across/within axis.
    """
    # Order cells consistently
    keys = sorted(cells.keys())
    n_cells = len(keys)
    Y = np.stack([cells[k] for k in keys])  # (36, 1920)
    ctx = np.array([1.0 if k[0] in ACROSS_CONTEXT else 0.0 for k in keys])  # (36,)

    # Per-head: residualize against context-axis indicator (intercept + ctx).
    X = np.stack([np.ones(n_cells), ctx], axis=1)  # (36, 2)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)    # (2, 1920)
    Y_resid = Y - X @ beta                          # (36, 1920)

    # Recompute per-condition fingerprints on residualized scores
    fp_resid = {}
    for c in CONDITIONS:
        idxs = [i for i, k in enumerate(keys) if k[0] == c]
        fp_resid[c] = Y_resid[idxs].mean(axis=0)

    R = pd.DataFrame(np.eye(len(CONDITIONS)), index=CONDITIONS, columns=CONDITIONS)
    for i, a in enumerate(CONDITIONS):
        for j, b in enumerate(CONDITIONS):
            if i == j:
                continue
            r = np.corrcoef(fp_resid[a], fp_resid[b])[0, 1]
            R.loc[a, b] = r

    # Also return the original fingerprint Pearson for side-by-side comparison
    fp_orig = {}
    for c in CONDITIONS:
        idxs = [i for i, k in enumerate(keys) if k[0] == c]
        fp_orig[c] = Y[idxs].mean(axis=0)
    R_orig = pd.DataFrame(np.eye(len(CONDITIONS)), index=CONDITIONS, columns=CONDITIONS)
    for i, a in enumerate(CONDITIONS):
        for j, b in enumerate(CONDITIONS):
            if i == j:
                continue
            R_orig.loc[a, b] = np.corrcoef(fp_orig[a], fp_orig[b])[0, 1]

    return R_orig, R


def plot_partial_panel(R_orig: pd.DataFrame, R_partial: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    delta = R_orig - R_partial
    for ax, M, title, vmin, vmax, cmap in [
        (axes[0], R_orig, "Pearson r (clamped, original)", -1, 1, "RdBu_r"),
        (axes[1], R_partial, "Pearson r | context-axis partialled out", -1, 1, "RdBu_r"),
        (axes[2], delta, "Δr (original − partial)", -0.5, 0.5, "PiYG_r"),
    ]:
        im = ax.imshow(M.values.astype(float), vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xticks(range(len(M.columns)))
        ax.set_xticklabels([SHORT[c] for c in M.columns])
        ax.set_yticks(range(len(M.index)))
        ax.set_yticklabels([SHORT[c] for c in M.index])
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > (vmax - vmin) / 4 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


# ---------- (2) Bootstrap null for H2 R^2 ----------

def make_random_vec_with_target_r(target: np.ndarray, target_r: float,
                                  tol: float = 0.02, max_tries: int = 5000) -> np.ndarray:
    """Generate a random vector v such that r(target, v) ~ target_r within tol.

    Uses the closed-form construction:
        v = a * target + b * orthogonal_noise
    Choose a, b so that r(target, v) = target_r exactly, then return v with
    matched scale (var ~= var(target)). Sign of target_r is preserved.
    """
    # Standardize working copies
    t = (target - target.mean()) / (target.std() + 1e-12)
    # Sample noise orthogonal to t (Gram-Schmidt)
    for _ in range(max_tries):
        noise = RNG.standard_normal(len(t))
        noise = noise - (noise @ t) * t / (t @ t)
        noise = noise / (noise.std() + 1e-12)
        # Construct v = rho * t + sqrt(1 - rho^2) * noise
        rho = float(target_r)
        v = rho * t + np.sqrt(max(0.0, 1 - rho ** 2)) * noise
        # Check
        actual_r = np.corrcoef(t, v)[0, 1]
        if abs(actual_r - target_r) <= tol:
            # rescale to original target's std for OLS realism
            return v * target.std() + target.mean()
    raise RuntimeError("could not match target r within tolerance")


def ols_r2(target: np.ndarray, comps: list[np.ndarray]) -> tuple[float, np.ndarray]:
    X = np.stack(comps, axis=1)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    ss_res = np.sum((target - fitted) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1 - ss_res / ss_tot), beta


def bootstrap_h2_null(target: np.ndarray, comps: list[np.ndarray],
                      n_boot: int = 1000) -> dict:
    """For each bootstrap iter, sample random vectors with matched marginal r
    to target, fit OLS, record R^2. Return observed R^2 + null distribution."""
    target_rs = [float(np.corrcoef(target, c)[0, 1]) for c in comps]
    obs_r2, obs_beta = ols_r2(target, comps)
    null = np.empty(n_boot)
    for b in range(n_boot):
        rand_comps = [make_random_vec_with_target_r(target, r) for r in target_rs]
        null[b], _ = ols_r2(target, rand_comps)
    p = float((null >= obs_r2).mean())
    return {"observed_r2": obs_r2, "obs_beta": obs_beta, "null": null,
            "target_rs": target_rs, "p": p,
            "null_mean": float(null.mean()), "null_std": float(null.std()),
            "null_q05": float(np.quantile(null, 0.05)),
            "null_q95": float(np.quantile(null, 0.95))}


def load_fp(key: str = "clamped") -> dict:
    return {c: np.load(FP_DIR / f"{c}_{key}.npy") for c in CONDITIONS}


def plot_h2_null_panel(results: list[dict], out: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        ax.hist(res["null"], bins=40, color="lightgrey", edgecolor="grey",
                label=f"matched-r null (n={len(res['null'])})")
        ax.axvline(res["observed_r2"], color="crimson", lw=2,
                   label=f"observed R²={res['observed_r2']:.3f}")
        ax.axvline(res["null_q95"], color="grey", ls="--", lw=1, label="null 95%")
        ax.set_title(f"{res['label']}\np = {res['p']:.4f} | null mean = {res['null_mean']:.3f}",
                     fontsize=10)
        ax.set_xlabel("OLS R²")
        ax.set_ylabel("count")
        ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    OUT_AGG.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    # ---- (1) partial correlation
    print("(1) partial correlation against across-context indicator...")
    manifest = pd.read_csv(MANIFEST)
    cells = load_cell_vectors(manifest)
    R_orig, R_partial = partial_corr_against_context_axis(cells)
    R_orig.to_csv(OUT_AGG / "partial_corr_original.csv")
    R_partial.to_csv(OUT_AGG / "partial_corr_pearson.csv")
    (R_orig - R_partial).to_csv(OUT_AGG / "partial_corr_delta.csv")
    print("\nOriginal:")
    print(R_orig.round(2).rename(columns=SHORT, index=SHORT).to_string())
    print("\nPartialled (across-context axis removed):")
    print(R_partial.round(2).rename(columns=SHORT, index=SHORT).to_string())
    print("\nΔ (original − partial), >0 means correlation shrinks once context-axis is removed:")
    print((R_orig - R_partial).round(2).rename(columns=SHORT, index=SHORT).to_string())
    plot_partial_panel(R_orig, R_partial, OUT_FIG / "controls_partial_corr.png")

    # ---- (2) H2 bootstrap null
    print("\n(2) H2 bootstrap null (matched marginal r) ...")
    fp = load_fp("clamped")
    decomps = [
        ("C3_answer_changes_belief", ["C1_abstract_belief", "C5_answer_changes_location_only"], "C3 ≈ C1 + C5"),
        ("C4_answer_changes_photo", ["C2_abstract_photo", "C5_answer_changes_location_only"], "C4 ≈ C2 + C5"),
        ("C4_answer_changes_photo", ["C2_abstract_photo", "C6_answer_changes_location_only_photo"], "C4 ≈ C2 + C6"),
        ("C4_answer_changes_photo", ["C2_abstract_photo", "C7_state_only_no_belief"], "C4 ≈ C2 + C7"),
    ]
    boot_results = []
    rows = []
    for target, comps, label in decomps:
        target_v = fp[target]
        comp_vs = [fp[n] for n in comps]
        res = bootstrap_h2_null(target_v, comp_vs, n_boot=args.n_boot)
        res["label"] = label
        boot_results.append(res)
        print(f"\n  {label}")
        print(f"    target marginal r: {[f'{r:.3f}' for r in res['target_rs']]}")
        print(f"    observed R²={res['observed_r2']:.4f} | obs β={res['obs_beta']}")
        print(f"    null R² mean={res['null_mean']:.4f} std={res['null_std']:.4f}")
        print(f"    null 5/95%: {res['null_q05']:.4f} / {res['null_q95']:.4f}")
        print(f"    p (observed >= null) = {res['p']:.4f}")
        rows.append({
            "decomposition": label,
            "observed_r2": res["observed_r2"],
            "null_mean": res["null_mean"], "null_std": res["null_std"],
            "null_q05": res["null_q05"], "null_q95": res["null_q95"],
            "p": res["p"],
            "target_r_comp1": res["target_rs"][0],
            "target_r_comp2": res["target_rs"][1],
        })
    pd.DataFrame(rows).to_csv(OUT_AGG / "h2_bootstrap_null.csv", index=False)
    plot_h2_null_panel(boot_results, OUT_FIG / "controls_h2_null.png")

    print(f"\noutputs:\n  tables -> {OUT_AGG}\n  figures -> {OUT_FIG}")


if __name__ == "__main__":
    main()
