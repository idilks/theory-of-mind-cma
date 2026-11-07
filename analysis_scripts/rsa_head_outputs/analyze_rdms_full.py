"""All-heads CMA-weighted Mantel analysis (Taylor's variant).

Complement to analyze_rdms_7cond.py. That script restricts the head pool to
the union of top-20 per condition (~43 heads), so the per-condition RDMs are
weighted averages over the same fixed 43-head set. The shared stimulus
baseline carried by those 43 heads inflates Mantel correlations between
condition RDMs (e.g. r(C1, C2) = 0.97).

This script uses ALL 1920 heads. Per condition, the weight on each head's
RDM is the head's clamped CMA fingerprint score (max(0, score)),
normalized to sum to 1. Heads with near-zero CMA in a given condition
contribute near-zero to that condition's RDM, so the effective head pool is
condition-specific and shared baseline is dampened.

Inputs:
  - results/rsa_head_outputs/activations_full/{template}.pt
      from capture_head_outputs_full.py: z is (n_stim, 48, 40, 128) fp16
  - results/aggregated/fingerprints/{cond}_clamped.npy
      1920-vector of clamped CMA scores (already exists, indexed
      layer*40+head)

Outputs:
  results/rsa_head_outputs/aggregation_full/mantel_7x7_full.csv
  results/summary/aggregation/mantel_7x7_full.png
  results/summary/aggregation/mantel_7x7_full_vs_top20.png  (side-by-side)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations_full"
FP_DIR = ROOT / "results/aggregated/fingerprints"
TOP20_MANTEL_CSV = ROOT / "results/rsa_head_outputs/aggregation_7cond/mantel_7x7.csv"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_full"
FIG_DIR = ROOT / "results/summary/aggregation"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
N_LAYERS, N_HEADS = 48, 40

CONDITIONS = [
    ("C1", "C1_abstract_belief"),
    ("C2", "C2_abstract_photo"),
    ("C3", "C3_answer_changes_belief"),
    ("C4", "C4_answer_changes_photo"),
    ("C5", "C5_answer_changes_location_only"),
    ("C6", "C6_answer_changes_location_only_photo"),
    ("C7", "C7_state_only_no_belief"),
]


from rsa_utils import cosine_rdm


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


def mantel(d1: np.ndarray, d2: np.ndarray, n_perm: int,
           rng: np.random.Generator) -> tuple[float, float]:
    obs = float(spearmanr(upper_tri(d1), upper_tri(d2)).correlation)
    n = d1.shape[0]
    flat1 = upper_tri(d1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = float(spearmanr(flat1, upper_tri(d2[perm][:, perm])).correlation)
    p = float((np.abs(null) >= np.abs(obs)).mean())
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="Mantel permutations (1000 fast, 5000 final)")
    ap.add_argument("--min_weight_frac", type=float, default=0.0,
                    help="Optional: zero out head weights below this fraction "
                         "of the per-condition max (0 = use all positive scores).")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}

    z_all = np.concatenate(
        [payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0,
    )  # (n_stim_total, 48, 40, 128)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)

    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)
    n_stim = len(stim)
    print(f"loaded {n_stim} stimuli x {N_LAYERS} layers x {N_HEADS} heads = "
          f"{N_LAYERS * N_HEADS} heads")

    # Flatten head axis: (n_stim, 1920, head_dim)
    z_heads = z_all.reshape(n_stim, N_LAYERS * N_HEADS, -1)

    # Build all 1920 head RDMs once (~158 MB float32, fine in RAM).
    print("building 1920 head RDMs...")
    head_rdms = np.empty((N_LAYERS * N_HEADS, n_stim, n_stim), dtype=np.float32)
    for h in range(N_LAYERS * N_HEADS):
        head_rdms[h] = cosine_rdm(z_heads[:, h, :])
        if (h + 1) % 200 == 0:
            print(f"  {h + 1}/{N_LAYERS * N_HEADS}")

    # Per-condition CMA-weighted average over all 1920 heads.
    cond_rdms: dict[str, np.ndarray] = {}
    weight_stats = []
    for short, full in CONDITIONS:
        fp = np.load(FP_DIR / f"{full}_clamped.npy").astype(np.float32)
        assert fp.shape == (N_LAYERS * N_HEADS,), fp.shape
        w = np.clip(fp, 0, None)
        if args.min_weight_frac > 0:
            thresh = args.min_weight_frac * w.max()
            w = np.where(w >= thresh, w, 0.0)
        wsum = w.sum()
        assert wsum > 0, f"all weights zero for {short}"
        w = w / wsum
        n_active = int((w > 0).sum())
        weight_stats.append({
            "condition": short, "n_active_heads": n_active,
            "max_weight": float(w.max()),
            "top20_mass": float(np.sort(w)[-20:].sum()),
        })
        # weighted sum of RDMs: tensordot over the head axis
        cond_rdms[short] = np.tensordot(w, head_rdms, axes=1)

    pd.DataFrame(weight_stats).to_csv(OUT_DIR / "weight_stats.csv", index=False)
    print("\nper-condition weight stats:")
    print(pd.DataFrame(weight_stats).to_string(index=False))

    # Model RDMs (Taylor's dissociation test).
    belief = stim["belief_veracity"].to_numpy()
    answer = stim["answer_state"].to_numpy()
    model_rdms = {
        "belief": (belief[:, None] != belief[None, :]).astype(np.float32),
        "answer": (answer[:, None] != answer[None, :]).astype(np.float32),
    }

    rng = np.random.default_rng(0)
    keys = [s for s, _ in CONDITIONS]
    K = len(keys)

    # 7x2 condition-vs-model Mantel (all heads, CMA-weighted).
    print(f"\n7x2 condition-vs-model Mantel ({args.n_perm} perms):")
    model_rows = []
    model_mat = np.zeros((K, 2))
    for i, ki in enumerate(keys):
        for j, mname in enumerate(["belief", "answer"]):
            r, p = mantel(cond_rdms[ki], model_rdms[mname], args.n_perm, rng)
            model_mat[i, j] = r
            model_rows.append({"condition": ki, "model": mname,
                               "spearman_r": r, "p_perm": p})
            print(f"  {ki} vs {mname:6s} model: r={r:+.3f}, p={p:.4f}")
    pd.DataFrame(model_rows).to_csv(OUT_DIR / "mantel_vs_models_full.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(model_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["belief-veracity\nmodel", "answer-state\nmodel"])
    ax.set_yticks(range(K)); ax.set_yticklabels(keys)
    for i in range(K):
        for j in range(2):
            color = "white" if abs(model_mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{model_mat[i, j]:+.2f}", ha="center", va="center",
                    color=color, fontsize=11)
    ax.set_title(f"All-heads CMA-weighted RDM vs model RDMs\n({args.n_perm} perms)")
    plt.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mantel_vs_models_full.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'mantel_vs_models_full.png'}")

    # 7x7 between-condition Mantel.
    rmat = np.zeros((K, K))
    pmat = np.zeros((K, K))
    rows = []
    print(f"\n7x7 between-condition Mantel ({args.n_perm} perms each):")
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i > j:
                rmat[i, j] = rmat[j, i]; pmat[i, j] = pmat[j, i]; continue
            if i == j:
                rmat[i, j] = 1.0; pmat[i, j] = 0.0; continue
            r, p = mantel(cond_rdms[ki], cond_rdms[kj], args.n_perm, rng)
            rmat[i, j] = r; pmat[i, j] = p
            rows.append({"a": ki, "b": kj, "spearman_r": r, "p_perm": p})
            print(f"  {ki} <-> {kj}: r={r:+.3f}, p={p:.4f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "mantel_7x7_full.csv", index=False)
    print(f"\nmantel csv -> {OUT_DIR / 'mantel_7x7_full.csv'}")

    # Heatmap of full-heads Mantel.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(keys)
    ax.set_yticks(range(K)); ax.set_yticklabels(keys)
    for i in range(K):
        for j in range(K):
            color = "white" if abs(rmat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{rmat[i, j]:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_title(f"7x7 Mantel matrix, ALL 1920 heads, CMA-weighted\n"
                 f"({args.n_perm} stimulus perms)")
    plt.colorbar(im, ax=ax, label="Spearman r (RDM upper triangles)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mantel_7x7_full.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'mantel_7x7_full.png'}")

    # Side-by-side with the top-20 union version, if it exists.
    if TOP20_MANTEL_CSV.exists():
        df20 = pd.read_csv(TOP20_MANTEL_CSV)
        rmat20 = np.eye(K)
        for r in df20.itertuples():
            i = keys.index(r.a); j = keys.index(r.b)
            rmat20[i, j] = rmat20[j, i] = r.spearman_r

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for ax, M, title in zip(
            axes,
            [rmat20, rmat, rmat - rmat20],
            ["top-20 union (43 heads)", "all 1920 heads", "diff (full - top20)"],
        ):
            vmax = 1 if "diff" not in title else float(np.abs(M).max())
            vmin = -vmax
            im = ax.imshow(M, cmap="RdBu_r", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(K)); ax.set_xticklabels(keys)
            ax.set_yticks(range(K)); ax.set_yticklabels(keys)
            for i in range(K):
                for j in range(K):
                    color = "white" if abs(M[i, j]) > 0.5 * vmax else "black"
                    ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                            color=color, fontsize=8)
            ax.set_title(title)
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("Mantel between condition RDMs: top-20 pool vs all heads")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "mantel_7x7_full_vs_top20.png", dpi=150)
        print(f"figure -> {FIG_DIR / 'mantel_7x7_full_vs_top20.png'}")
    else:
        print(f"(skipping side-by-side: {TOP20_MANTEL_CSV} not found)")


if __name__ == "__main__":
    main()
