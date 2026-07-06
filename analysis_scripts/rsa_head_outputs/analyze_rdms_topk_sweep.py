"""Top-k sweep + noise-floor null for the per-condition Mantel-vs-model test.

Two diagnostic extensions of analyze_rdms_per_condition.py, both using the
full 1920-head capture (so we can sample any subset, not just from a fixed
43-head union):

  1. Top-k sweep: for each condition and k in {5, 10, 20, 50, 100, 200},
     build the CMA-weighted RDM from the condition's top-k heads and Mantel
     against the belief-veracity and answer-state model RDMs. Tests whether
     the C1 dissociation is robust across k or peaks at a specific size.

  2. Noise-floor null: for each condition, sample N random sets of 20 heads
     uniformly from all 1920, weight uniformly (or by their actual CMA score
     in that condition — both reported), build the weighted RDM, Mantel
     against both models. Gives a null distribution: what dissociation do
     20 randomly chosen heads produce? The observed top-20 numbers should
     sit far above the null.

Inputs:
  results/rsa_head_outputs/activations_full/{template}.pt
  results/aggregated/fingerprints/{cond}_clamped.npy

Outputs:
  results/rsa_head_outputs/aggregation_topk_sweep/topk_sweep.csv
  results/rsa_head_outputs/aggregation_topk_sweep/noise_floor.csv
  results_for_pi/aggregation/topk_sweep.png
  results_for_pi/aggregation/noise_floor_dissociation.png
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
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_topk_sweep"
FIG_DIR = ROOT / "results_for_pi/aggregation"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
N_LAYERS, N_HEADS = 48, 40
N_TOTAL = N_LAYERS * N_HEADS

CONDITIONS = [
    ("C1", "C1_abstract_belief"),
    ("C2", "C2_abstract_photo"),
    ("C3", "C3_answer_changes_belief"),
    ("C4", "C4_answer_changes_photo"),
    ("C5", "C5_answer_changes_location_only"),
    ("C6", "C6_answer_changes_location_only_photo"),
    ("C7", "C7_state_only_no_belief"),
]

K_VALUES = [5, 10, 20, 50, 100, 200]


from rsa_utils import cosine_rdm


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices_from(M, k=1)]


def fast_mantel(rdm: np.ndarray, model_flat: np.ndarray) -> float:
    """Spearman r between upper-triangles. No permutation null here — we use
    head-resampling as the null instead, which is more diagnostic."""
    return float(spearmanr(upper_tri(rdm), model_flat).correlation)


def weighted_rdm(head_rdms: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 0, None)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return np.tensordot(w, head_rdms[idx], axes=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_null", type=int, default=200,
                    help="Number of random head-sample null draws per condition.")
    ap.add_argument("--null_size", type=int, default=20,
                    help="Heads per null draw (matched to top-20).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load all-heads activations.
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    z_all = np.concatenate(
        [payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0,
    )  # (n_stim, 48, 40, 128)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)
    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)
    n_stim = len(stim)
    z_heads = z_all.reshape(n_stim, N_TOTAL, -1)
    print(f"loaded {n_stim} stimuli x {N_TOTAL} heads")

    # Build all 1920 head RDMs.
    print("building 1920 head RDMs...")
    head_rdms = np.empty((N_TOTAL, n_stim, n_stim), dtype=np.float32)
    for h in range(N_TOTAL):
        head_rdms[h] = cosine_rdm(z_heads[:, h, :])
        if (h + 1) % 400 == 0:
            print(f"  {h + 1}/{N_TOTAL}")

    # Model RDMs.
    belief = stim["belief_veracity"].to_numpy()
    answer = stim["answer_state"].to_numpy()
    model_belief = (belief[:, None] != belief[None, :]).astype(np.float32)
    model_answer = (answer[:, None] != answer[None, :]).astype(np.float32)
    flat_belief = upper_tri(model_belief)
    flat_answer = upper_tri(model_answer)

    # 1. Top-k sweep.
    print("\nTop-k sweep:")
    sweep_rows = []
    for short, full in CONDITIONS:
        fp = np.load(FP_DIR / f"{full}_clamped.npy").astype(np.float32)
        order = np.argsort(-fp)  # descending by score
        for k in K_VALUES:
            idx = order[:k]
            weights = fp[idx]
            rdm = weighted_rdm(head_rdms, idx, weights)
            r_b = fast_mantel(rdm, flat_belief)
            r_a = fast_mantel(rdm, flat_answer)
            sweep_rows.append({
                "condition": short, "k": k,
                "r_belief": r_b, "r_answer": r_a,
                "gap": r_b - r_a,
                "ratio": r_b / r_a if r_a > 0 else float("inf"),
            })
            print(f"  {short} k={k:3d}: r_belief={r_b:+.3f} r_answer={r_a:+.3f} gap={r_b - r_a:+.3f}")
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(OUT_DIR / "topk_sweep.csv", index=False)

    # 2. Noise-floor null.
    print(f"\nNoise floor: {args.n_null} random head-samples (size={args.null_size}) per condition...")
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for short, full in CONDITIONS:
        fp = np.load(FP_DIR / f"{full}_clamped.npy").astype(np.float32)
        for draw in range(args.n_null):
            idx = rng.choice(N_TOTAL, size=args.null_size, replace=False)
            # Two weighting flavors: uniform (pure random pool) and the heads'
            # own CMA scores in this condition (most are near-zero — emulates
            # what you'd get if you just picked random heads but kept CMA weighting).
            for flavor, w in [("uniform", np.ones(args.null_size)),
                              ("cma_at_random", fp[idx])]:
                rdm = weighted_rdm(head_rdms, idx, w)
                r_b = fast_mantel(rdm, flat_belief)
                r_a = fast_mantel(rdm, flat_answer)
                null_rows.append({
                    "condition": short, "flavor": flavor, "draw": draw,
                    "r_belief": r_b, "r_answer": r_a, "gap": r_b - r_a,
                })
    null_df = pd.DataFrame(null_rows)
    null_df.to_csv(OUT_DIR / "noise_floor.csv", index=False)

    # Summary table: observed top-20 vs null distribution.
    print("\nObserved top-20 vs null distribution (uniform-weighted random 20):")
    summary_rows = []
    for short, _ in CONDITIONS:
        obs = sweep_df[(sweep_df.condition == short) & (sweep_df.k == 20)].iloc[0]
        nu = null_df[(null_df.condition == short) & (null_df.flavor == "uniform")]
        summary_rows.append({
            "condition": short,
            "obs_r_belief": obs.r_belief,
            "obs_r_answer": obs.r_answer,
            "obs_gap": obs.gap,
            "null_r_belief_mean": nu.r_belief.mean(),
            "null_r_belief_p95": nu.r_belief.quantile(0.95),
            "null_r_answer_mean": nu.r_answer.mean(),
            "null_r_answer_p95": nu.r_answer.quantile(0.95),
            "null_gap_mean": nu.gap.mean(),
            "null_gap_p95": nu.gap.quantile(0.95),
            "p_belief_gt_null": float((nu.r_belief >= obs.r_belief).mean()),
            "p_gap_gt_null":    float((nu.gap >= obs.gap).mean()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "summary_top20_vs_null.csv", index=False)
    print(summary_df.to_string(index=False))

    # Plot 1: top-k sweep.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model in zip(axes, ["r_belief", "r_answer"]):
        for short, _ in CONDITIONS:
            sub = sweep_df[sweep_df.condition == short].sort_values("k")
            ax.plot(sub.k, sub[model], "-o", label=short)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES); ax.set_xticklabels([str(k) for k in K_VALUES])
        ax.set_xlabel("top-k heads")
        ax.set_ylabel("Mantel r")
        ax.set_title(f"vs {model.split('_')[1]}-model RDM")
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=9, loc="best")
    fig.suptitle("Top-k sweep: per-condition CMA-weighted RDM vs model templates")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "topk_sweep.png", dpi=150)
    print(f"\nfigure -> {FIG_DIR / 'topk_sweep.png'}")

    # Plot 2: noise-floor scatter (belief vs answer).
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)
    keys = [s for s, _ in CONDITIONS]
    for ax, key in zip(axes.flat[:7], keys):
        nu = null_df[(null_df.condition == key) & (null_df.flavor == "uniform")]
        ax.scatter(nu.r_answer, nu.r_belief, s=8, alpha=0.4, color="grey",
                   label=f"null (n={len(nu)})")
        obs = sweep_df[(sweep_df.condition == key) & (sweep_df.k == 20)].iloc[0]
        ax.scatter([obs.r_answer], [obs.r_belief], s=120, color="red",
                   edgecolor="black", zorder=5, label="observed top-20")
        lim = max(0.5, sweep_df.r_belief.max() * 1.1)
        ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.5)
        ax.set_xlim(-0.05, lim); ax.set_ylim(-0.05, lim)
        ax.set_title(key)
        ax.set_xlabel("Mantel r vs answer-model")
        ax.set_ylabel("Mantel r vs belief-model")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Observed top-20 vs noise floor (n={args.n_null} random 20-head samples, uniform weights)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "noise_floor_dissociation.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'noise_floor_dissociation.png'}")


if __name__ == "__main__":
    main()
