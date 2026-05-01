#!/usr/bin/env python
"""Aggregate per-condition CMA fingerprints and run cross-condition tests.

Reads results_for_pi/heatmaps/manifest.csv as the canonical map of cells
(condition, template, rule_pair) -> source folder. End-patch only.

Produces:
- per-condition fingerprints (clamped and raw), saved as .npy and .csv
- 6x6 Pearson correlation matrices (clamped primary, raw robustness)
  with permutation p-values (head-index shuffle, 5000 perms)
- 6x6 Jaccard matrices on FWER-significant heads, three variants:
  all-cell (intersection), 4-of-6 (majority), any-cell (union)
- H1 scatter panels (C1 vs {C2, C5, C6}) with r and p annotated
- H2 decomposition tests:
    C3 ~= C1 + C5  (answer-changes-belief = abstract belief + ToM control)
    C4 ~= C2 + C5  and  C4 ~= C2 + C6  (photo side, both location components)
  reports r(C, comp1), r(C, comp2), r(C, comp1+comp2), and OLS regression
  with R^2 and per-coefficient permutation CIs.

Headline figures use the clamped fingerprints and exclude C5 in the primary
5x5 view (C5 seed=0 only at n=26-50, noise-attenuated until seed=1 lands).
C5 is appended as an extra row/column with a noise caveat banner.

Usage:
    python scripts/aggregate_conditions.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results_for_pi/heatmaps/manifest.csv"
AGG_DIR = ROOT / "results/aggregated"
FIG_DIR = ROOT / "results_for_pi/aggregation"

CONDITIONS = [
    "C1_abstract_belief",
    "C2_abstract_photo",
    "C3_answer_changes_belief",
    "C4_answer_changes_photo",
    "C5_control",
    "C6_state_only_no_belief",
]
SHORT = {c: c.split("_", 1)[0] for c in CONDITIONS}
PRIMARY_SET = [c for c in CONDITIONS if c != "C5_control"]  # 5x5 headline excludes C5

# Rule-pair structure: C1/C2/C3 use flip-rule pairs (ABA<->ABB); C4/C5/C6 use
# same-rule pairs (ABA->ABA / ABB->ABB). The same-rule cluster also shares the
# property that base and exp prompts use identical location tokens in different
# roles, while the flip-rule cluster has different initial locations.
FLIP_RULE = ["C1_abstract_belief", "C2_abstract_photo", "C3_answer_changes_belief"]
SAME_RULE = ["C4_answer_changes_photo", "C5_control", "C6_state_only_no_belief"]

N_HEADS = 1920
N_PERM = 5000
RNG = np.random.default_rng(20260430)


def load_cell(source_rel: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores_1920, sig_mask_1920) for one cell."""
    src = ROOT / source_rel
    scores = torch.load(src / "causal_scores.pt", map_location="cpu", weights_only=False)
    scores = scores.float().cpu().numpy().reshape(-1)
    df = pd.read_csv(src / "statistical_test_results.csv")
    df = df.sort_values(["layer", "head"]).reset_index(drop=True)
    sig = df["significant"].astype(bool).to_numpy()
    assert scores.shape == (N_HEADS,) and sig.shape == (N_HEADS,)
    return scores, sig


def build_fingerprints(manifest: pd.DataFrame) -> dict:
    """Return dict: condition -> {raw, clamped, sig_per_cell, n_cells}."""
    out = {}
    for c in CONDITIONS:
        rows = manifest[manifest["condition"] == c]
        cells_raw, cells_sig = [], []
        for _, r in rows.iterrows():
            s, m = load_cell(r["source"])
            cells_raw.append(s)
            cells_sig.append(m)
        raw = np.stack(cells_raw)              # (n_cells, 1920)
        sig = np.stack(cells_sig)              # (n_cells, 1920)
        clamped_cells = np.clip(raw, 0, None)
        out[c] = {
            "raw_cells": raw,
            "sig_cells": sig,
            "raw": raw.mean(axis=0),
            "clamped": clamped_cells.mean(axis=0),
            "n_cells": raw.shape[0],
        }
        print(f"  {c}: {raw.shape[0]} cells, "
              f"clamped mean={out[c]['clamped'].mean():.4f}, "
              f"sig per cell={sig.sum(axis=1).tolist()}")
    return out


def pearson_with_perm(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERM) -> tuple[float, float]:
    """Pearson r and two-sided permutation p (shuffle head indices in y)."""
    r = np.corrcoef(x, y)[0, 1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(x, RNG.permutation(y))[0, 1]
    p = (np.abs(null) >= np.abs(r)).mean()
    return float(r), float(p)


def pearson_matrix(fps: dict, key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    R = pd.DataFrame(np.eye(len(CONDITIONS)), index=CONDITIONS, columns=CONDITIONS)
    P = pd.DataFrame(np.zeros((len(CONDITIONS), len(CONDITIONS))), index=CONDITIONS, columns=CONDITIONS)
    for i, a in enumerate(CONDITIONS):
        for j, b in enumerate(CONDITIONS):
            if j <= i:
                continue
            r, p = pearson_with_perm(fps[a][key], fps[b][key])
            R.loc[a, b] = R.loc[b, a] = r
            P.loc[a, b] = P.loc[b, a] = p
    return R, P


def sig_set(fps: dict, c: str, mode: str) -> np.ndarray:
    sig = fps[c]["sig_cells"]  # (n_cells, 1920)
    n_cells = sig.shape[0]
    counts = sig.sum(axis=0)
    if mode == "all":
        return counts == n_cells
    if mode == "any":
        return counts >= 1
    if mode == "4of6":
        return counts >= 4
    raise ValueError(mode)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else float("nan")


def jaccard_matrix(fps: dict, mode: str) -> pd.DataFrame:
    sets = {c: sig_set(fps, c, mode) for c in CONDITIONS}
    M = pd.DataFrame(np.zeros((len(CONDITIONS), len(CONDITIONS))), index=CONDITIONS, columns=CONDITIONS)
    for a in CONDITIONS:
        for b in CONDITIONS:
            M.loc[a, b] = jaccard(sets[a], sets[b])
    return M


def plot_matrix(M: pd.DataFrame, title: str, out: Path, vmin=None, vmax=None,
                cmap="viridis", annot_fmt="{:.2f}", c5_caveat=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    arr = M.values.astype(float)
    if vmin is None:
        vmin = np.nanmin(arr)
    if vmax is None:
        vmax = np.nanmax(arr)
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(M.columns)))
    ax.set_xticklabels([SHORT[c] for c in M.columns])
    ax.set_yticks(range(len(M.index)))
    ax.set_yticklabels([SHORT[c] for c in M.index])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                color = "white" if v < (vmin + vmax) / 2 else "black"
                ax.text(j, i, annot_fmt.format(v), ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if c5_caveat:
        c5_idx = CONDITIONS.index("C5_control")
        for spine_pos in ([c5_idx, "row"], [c5_idx, "col"]):
            idx, kind = spine_pos
            if kind == "row":
                ax.axhline(idx - 0.5, color="red", lw=1.2)
                ax.axhline(idx + 0.5, color="red", lw=1.2)
            else:
                ax.axvline(idx - 0.5, color="red", lw=1.2)
                ax.axvline(idx + 0.5, color="red", lw=1.2)
        ax.set_title(title + "\n(C5 outlined: seed=0 only, n=26-50, noise-attenuated)", fontsize=10)
    else:
        ax.set_title(title, fontsize=11)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def plot_h1_scatter(fps: dict, key: str, out: Path):
    pairs = [("C1_abstract_belief", "C2_abstract_photo"),
             ("C1_abstract_belief", "C5_control"),
             ("C1_abstract_belief", "C6_state_only_no_belief"),
             ("C2_abstract_photo", "C5_control"),
             ("C2_abstract_photo", "C6_state_only_no_belief")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4), sharey=False)
    for ax, (a, b) in zip(axes, pairs):
        x = fps[a][key]
        y = fps[b][key]
        r, p = pearson_with_perm(x, y, n_perm=2000)
        ax.scatter(x, y, s=4, alpha=0.35, color="steelblue")
        # highlight heads sig in BOTH (any-cell)
        sig_a = sig_set(fps, a, "any")
        sig_b = sig_set(fps, b, "any")
        both = sig_a & sig_b
        if both.any():
            ax.scatter(x[both], y[both], s=14, color="crimson", label=f"sig in both (any-cell): {both.sum()}")
            ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel(f"{SHORT[a]} ({key})")
        ax.set_ylabel(f"{SHORT[b]} ({key})")
        ax.set_title(f"{SHORT[a]} vs {SHORT[b]}\nr={r:.3f}, p={p:.4f}", fontsize=10)
        ax.axhline(0, color="grey", lw=0.5)
        ax.axvline(0, color="grey", lw=0.5)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def regression(target: np.ndarray, comps: list[np.ndarray]) -> dict:
    """OLS: target ~ sum_k beta_k * comp_k. Returns betas, R^2, r(target, sum)."""
    X = np.stack(comps, axis=1)  # (1920, k)
    beta, *_ = np.linalg.lstsq(X, target, rcond=None)
    fitted = X @ beta
    ss_res = np.sum((target - fitted) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    summed = np.sum(np.stack(comps), axis=0)
    r_sum = np.corrcoef(target, summed)[0, 1]
    r_indiv = [np.corrcoef(target, c)[0, 1] for c in comps]
    return {"beta": beta, "r2": float(r2), "r_sum": float(r_sum), "r_indiv": r_indiv,
            "fitted": fitted, "residuals": target - fitted, "summed": summed}


def plot_h2(target_name: str, comp_names: list[str], fps: dict, key: str, out: Path):
    target = fps[target_name][key]
    comps = [fps[n][key] for n in comp_names]
    res = regression(target, comps)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # bar chart of r(target, ...)
    labels = [f"r({SHORT[target_name]}, {SHORT[n]})" for n in comp_names]
    labels.append(f"r({SHORT[target_name]}, {'+'.join(SHORT[n] for n in comp_names)})")
    vals = res["r_indiv"] + [res["r_sum"]]
    colors = ["steelblue"] * len(comp_names) + ["crimson"]
    axes[0].bar(range(len(vals)), vals, color=colors)
    axes[0].set_xticks(range(len(vals)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title(f"H2: does {SHORT[target_name]} ≈ sum?")
    axes[0].axhline(0, color="grey", lw=0.5)
    for i, v in enumerate(vals):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    # scatter target vs sum (additive prediction)
    axes[1].scatter(res["summed"], target, s=4, alpha=0.4)
    axes[1].plot([res["summed"].min(), res["summed"].max()],
                 [res["summed"].min(), res["summed"].max()], "k--", lw=0.7, label="y=x")
    axes[1].set_xlabel(f"{'+'.join(SHORT[n] for n in comp_names)} (sum)")
    axes[1].set_ylabel(SHORT[target_name])
    axes[1].set_title(f"sum-predicts-target\nr={res['r_sum']:.3f}")
    axes[1].legend(fontsize=8)

    # scatter target vs OLS fit (best linear combination)
    beta_lbl = ", ".join(f"β_{SHORT[n]}={b:.2f}" for n, b in zip(comp_names, res["beta"]))
    axes[2].scatter(res["fitted"], target, s=4, alpha=0.4)
    axes[2].plot([res["fitted"].min(), res["fitted"].max()],
                 [res["fitted"].min(), res["fitted"].max()], "k--", lw=0.7, label="y=x")
    axes[2].set_xlabel("OLS fit")
    axes[2].set_ylabel(SHORT[target_name])
    axes[2].set_title(f"OLS: {SHORT[target_name]} ~ {' + '.join(SHORT[n] for n in comp_names)}\n"
                      f"R²={res['r2']:.3f} | {beta_lbl}", fontsize=10)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    args = ap.parse_args()

    AGG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"reading manifest: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    print(f"  {len(manifest)} cells across {manifest['condition'].nunique()} conditions")

    print("\nbuilding fingerprints (per-condition mean across cells)...")
    fps = build_fingerprints(manifest)

    # save fingerprints
    fp_dir = AGG_DIR / "fingerprints"
    fp_dir.mkdir(exist_ok=True)
    rows = []
    for c, d in fps.items():
        np.save(fp_dir / f"{c}_clamped.npy", d["clamped"])
        np.save(fp_dir / f"{c}_raw.npy", d["raw"])
        rows.append({"condition": c, "n_cells": d["n_cells"],
                     "raw_mean": d["raw"].mean(), "clamped_mean": d["clamped"].mean(),
                     "raw_max": d["raw"].max(), "clamped_max": d["clamped"].max()})
    pd.DataFrame(rows).to_csv(AGG_DIR / "fingerprint_summary.csv", index=False)

    # Pearson matrices
    for key in ("clamped", "raw"):
        print(f"\nPearson matrix ({key}) with {args.n_perm} permutations...")
        R, P = pearson_matrix(fps, key)
        R.to_csv(AGG_DIR / f"pearson_{key}.csv")
        P.to_csv(AGG_DIR / f"pearson_{key}_pvalues.csv")
        plot_matrix(R, f"Pearson r — fingerprints ({key})",
                    FIG_DIR / f"pearson_{key}.png",
                    vmin=-1, vmax=1, cmap="RdBu_r", c5_caveat=True)

        # 5x5 primary view (no C5)
        R5 = R.loc[PRIMARY_SET, PRIMARY_SET]
        plot_matrix(R5, f"Pearson r — primary 5x5 ({key}, C5 excluded)",
                    FIG_DIR / f"pearson_{key}_primary5x5.png",
                    vmin=-1, vmax=1, cmap="RdBu_r")

        # Rule-cluster sub-matrices (within flip-rule, within same-rule, across)
        Rflip = R.loc[FLIP_RULE, FLIP_RULE]
        Rsame = R.loc[SAME_RULE, SAME_RULE]
        Rcross = R.loc[FLIP_RULE, SAME_RULE]
        Rflip.to_csv(AGG_DIR / f"pearson_{key}_flip_rule.csv")
        Rsame.to_csv(AGG_DIR / f"pearson_{key}_same_rule.csv")
        Rcross.to_csv(AGG_DIR / f"pearson_{key}_cross_cluster.csv")
        plot_matrix(Rflip, f"Pearson r — flip-rule cluster ({key})\nC1/C2/C3 (different initial locations in base vs exp)",
                    FIG_DIR / f"pearson_{key}_flip_rule.png", vmin=-1, vmax=1, cmap="RdBu_r")
        plot_matrix(Rsame, f"Pearson r — same-rule cluster ({key})\nC4/C5/C6 (shared location tokens in base & exp)",
                    FIG_DIR / f"pearson_{key}_same_rule.png", vmin=-1, vmax=1, cmap="RdBu_r")
        plot_matrix(Rcross, f"Pearson r — across clusters ({key})\nflip-rule (rows) vs same-rule (cols)",
                    FIG_DIR / f"pearson_{key}_cross_cluster.png", vmin=-1, vmax=1, cmap="RdBu_r")

    # Jaccard matrices
    for mode in ("all", "4of6", "any"):
        M = jaccard_matrix(fps, mode)
        M.to_csv(AGG_DIR / f"jaccard_{mode}.csv")
        plot_matrix(M, f"Jaccard ({mode}-cell) — FWER-significant heads",
                    FIG_DIR / f"jaccard_{mode}.png",
                    vmin=0, vmax=1, cmap="viridis", c5_caveat=True)
        # Sig-head counts per condition for context
        print(f"  jaccard {mode}: sig-head counts: " +
              ", ".join(f"{SHORT[c]}={int(sig_set(fps, c, mode).sum())}" for c in CONDITIONS))

    # H1 scatters
    print("\nH1 scatter panels...")
    plot_h1_scatter(fps, "clamped", FIG_DIR / "h1_scatter_clamped.png")
    plot_h1_scatter(fps, "raw", FIG_DIR / "h1_scatter_raw.png")

    # H2 decompositions
    print("\nH2 decompositions...")
    h2_rows = []
    decomps = [
        ("C3_answer_changes_belief", ["C1_abstract_belief", "C5_control"], "h2_C3_eq_C1plusC5"),
        ("C4_answer_changes_photo", ["C2_abstract_photo", "C5_control"], "h2_C4_eq_C2plusC5"),
        ("C4_answer_changes_photo", ["C2_abstract_photo", "C6_state_only_no_belief"], "h2_C4_eq_C2plusC6"),
    ]
    for target, comps, tag in decomps:
        for key in ("clamped", "raw"):
            res = plot_h2(target, comps, fps, key, FIG_DIR / f"{tag}_{key}.png")
            h2_rows.append({
                "target": target, "components": "+".join(comps), "agg": key,
                "r_sum": res["r_sum"], "r2_ols": res["r2"],
                **{f"r_{n}": v for n, v in zip(comps, res["r_indiv"])},
                **{f"beta_{n}": b for n, b in zip(comps, res["beta"])},
            })
            print(f"  [{key}] {SHORT[target]} ~ {' + '.join(SHORT[n] for n in comps)}: "
                  f"r_sum={res['r_sum']:.3f}, R²={res['r2']:.3f}, β={res['beta']}")
    pd.DataFrame(h2_rows).to_csv(AGG_DIR / "h2_decomposition.csv", index=False)

    print(f"\nfingerprints + matrices: {AGG_DIR}")
    print(f"figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
