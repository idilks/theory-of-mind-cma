#!/usr/bin/env python
"""Head-level RSA over the 6 condition fingerprints.

For each of the 1920 heads, take its 6-vec [C1, C2, C3, C4, C5, C6] of clamped
fingerprint scores. That vector is the head's "condition profile". Build a
1920x1920 RDM over heads (1 - Pearson r between profiles), cluster, and inspect.

Answers:
- Which heads form coherent functional groups across conditions?
- Do C3 universal heads cluster with the union of C1 and C5 universal heads
  (the H2 additive-decomposition signature at the head-cluster level)?
- Which heads load on the abstract block (C1, C2) vs the location-retrieval
  block (C5, C6) vs both?

Inputs: per-condition fingerprints (clamped) saved by aggregate_conditions.py.
Outputs:
- results/aggregated/head_profile_rsa/cluster_assignments.csv (1920 rows)
- results/aggregated/head_profile_rsa/cluster_profiles.csv
- results_for_pi/aggregation/rsa_head_rdm.png  (sorted by cluster)
- results_for_pi/aggregation/rsa_head_cluster_profiles.png
- results_for_pi/aggregation/rsa_head_clusters_on_layer_grid.png

Filtering: heads with max profile < ACTIVE_THRESHOLD are dropped from the
clustering step (their profiles are noise-around-zero so Pearson is ill-
defined). They are kept in the per-(layer, head) grid as "inactive".

Usage:
    python scripts/head_profile_rsa.py [--k 8] [--threshold 1.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[1]
FP_DIR = ROOT / "results/aggregated/fingerprints"
OUT_AGG = ROOT / "results/aggregated/head_profile_rsa"
OUT_FIG = ROOT / "results_for_pi/aggregation"

CONDITIONS = [
    "C1_abstract_belief",
    "C2_abstract_photo",
    "C3_answer_changes_belief",
    "C4_answer_changes_photo",
    "C5_answer_changes_location_only",
    "C6_answer_changes_location_only_photo",
    "C7_state_only_no_belief",
]
SHORT = [c.split("_", 1)[0] for c in CONDITIONS]
N_LAYERS, N_HEADS_PER_LAYER, N_HEADS = 48, 40, 1920


def load_profiles(key: str = "clamped") -> np.ndarray:
    """Return (1920, 6) head profile matrix."""
    cols = [np.load(FP_DIR / f"{c}_{key}.npy") for c in CONDITIONS]
    P = np.stack(cols, axis=1)
    assert P.shape == (N_HEADS, len(CONDITIONS))
    return P


def pearson_rdm(active_profiles: np.ndarray) -> np.ndarray:
    """1 - Pearson r between rows. Profiles are first standardized per row."""
    centered = active_profiles - active_profiles.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    z = centered / norms
    R = z @ z.T
    R = np.clip(R, -1, 1)
    return 1 - R


def hierarchical_cluster(rdm: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ward linkage, return (labels_1..k, linkage_matrix)."""
    cond = squareform(rdm, checks=False)
    Z = linkage(cond, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, Z


def plot_sorted_rdm(rdm: np.ndarray, labels: np.ndarray, head_idx: np.ndarray, out: Path):
    order = np.argsort(labels, kind="stable")
    rdm_s = rdm[np.ix_(order, order)]
    labels_s = labels[order]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rdm_s, cmap="viridis", vmin=0, vmax=2, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="1 - Pearson r")
    boundaries = np.where(np.diff(labels_s) != 0)[0]
    for b in boundaries:
        ax.axhline(b + 0.5, color="white", lw=0.6)
        ax.axvline(b + 0.5, color="white", lw=0.6)
    ax.set_title(f"Head-profile RDM (sorted by cluster)\n"
                 f"{len(head_idx)} active heads, k={labels.max()}")
    ax.set_xlabel("head")
    ax.set_ylabel("head")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=140)
    plt.close()


def plot_cluster_profiles(profiles: np.ndarray, labels: np.ndarray, out: Path):
    k = labels.max()
    fig, axes = plt.subplots(1, k, figsize=(2.5 * k, 4), sharey=True)
    if k == 1:
        axes = [axes]
    for ci, ax in enumerate(axes, start=1):
        members = profiles[labels == ci]
        mean_p = members.mean(axis=0)
        sem_p = members.std(axis=0) / np.sqrt(max(1, len(members)))
        bars = ax.bar(range(len(SHORT)), mean_p, yerr=sem_p, color="steelblue", capsize=2)
        ax.set_xticks(range(len(SHORT)))
        ax.set_xticklabels(SHORT, rotation=45, fontsize=8)
        ax.set_title(f"cluster {ci}\nn={len(members)}", fontsize=10)
        ax.axhline(0, color="grey", lw=0.5)
    axes[0].set_ylabel("mean clamped CMA score")
    fig.suptitle("Per-cluster condition profile (mean ± SEM across cluster heads)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def plot_layer_grid(labels_full: np.ndarray, out: Path):
    """Render cluster ID per (layer, head) cell. Inactive heads = 0."""
    grid = labels_full.reshape(N_LAYERS, N_HEADS_PER_LAYER)
    fig, ax = plt.subplots(figsize=(10, 12))
    cmap = plt.get_cmap("tab20", labels_full.max() + 1)
    im = ax.imshow(grid, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title("Cluster assignment per (layer, head)\n0 = inactive (filtered out before clustering)")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="cluster id")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()


def membership_summary(profiles: np.ndarray, labels_full: np.ndarray, threshold: float) -> pd.DataFrame:
    """Per-head cluster + per-condition profile values; for downstream H2 inspection."""
    rows = []
    for h in range(N_HEADS):
        layer = h // N_HEADS_PER_LAYER
        head = h % N_HEADS_PER_LAYER
        prof = profiles[h]
        rows.append({
            "head_idx": h, "layer": layer, "head": head,
            "cluster": int(labels_full[h]),
            "active": bool(prof.max() >= threshold),
            **{f"prof_{c}": float(profiles[h, i]) for i, c in enumerate(SHORT)},
        })
    return pd.DataFrame(rows)


def cluster_profile_table(profiles: np.ndarray, labels_full: np.ndarray) -> pd.DataFrame:
    rows = []
    for ci in sorted(set(labels_full)):
        if ci == 0:
            continue
        members = profiles[labels_full == ci]
        rows.append({
            "cluster": int(ci), "n_heads": int(len(members)),
            **{f"mean_{c}": float(members[:, i].mean()) for i, c in enumerate(SHORT)},
            **{f"max_{c}": float(members[:, i].max()) for i, c in enumerate(SHORT)},
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="number of clusters")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="head is 'active' if max condition score >= threshold")
    args = ap.parse_args()

    OUT_AGG.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print(f"loading clamped fingerprints from {FP_DIR}")
    P = load_profiles("clamped")
    active_mask = P.max(axis=1) >= args.threshold
    n_active = int(active_mask.sum())
    print(f"  {n_active}/{N_HEADS} active heads (max profile >= {args.threshold})")

    if n_active < 4:
        raise SystemExit("too few active heads to cluster; lower --threshold")

    active_idx = np.where(active_mask)[0]
    P_active = P[active_idx]

    rdm = pearson_rdm(P_active)
    np.save(OUT_AGG / "head_rdm.npy", rdm)
    np.save(OUT_AGG / "active_head_idx.npy", active_idx)

    labels_active, Z = hierarchical_cluster(rdm, k=args.k)
    print(f"  cluster sizes: {[int((labels_active == c).sum()) for c in range(1, args.k + 1)]}")

    # Map back to full 1920-length label vector (0 = inactive)
    labels_full = np.zeros(N_HEADS, dtype=int)
    labels_full[active_idx] = labels_active

    # Save tables
    member_df = membership_summary(P, labels_full, args.threshold)
    member_df.to_csv(OUT_AGG / "cluster_assignments.csv", index=False)
    cprof_df = cluster_profile_table(P, labels_full)
    cprof_df.to_csv(OUT_AGG / "cluster_profiles.csv", index=False)
    print("\ncluster profile means (clamped CMA):")
    print(cprof_df.round(3).to_string(index=False))

    # H2-relevant inspection: where do C3-loading heads sit relative to C1- and C5-loading heads?
    print("\nH2 head-cluster check:")
    for cond_short, cond_idx in zip(SHORT, range(len(SHORT))):
        top_heads = np.argsort(P[:, cond_idx])[::-1][:20]  # top-20 by raw clamped score
        clusters_hit = labels_full[top_heads]
        print(f"  top-20 heads by {cond_short}: clusters = "
              f"{dict(pd.Series(clusters_hit).value_counts().sort_index())}")

    # Figures
    plot_sorted_rdm(rdm, labels_active, active_idx, OUT_FIG / "rsa_head_rdm.png")
    plot_cluster_profiles(P[active_idx], labels_active, OUT_FIG / "rsa_head_cluster_profiles.png")
    plot_layer_grid(labels_full, OUT_FIG / "rsa_head_clusters_on_layer_grid.png")

    print(f"\noutputs:\n  tables -> {OUT_AGG}\n  figures -> {OUT_FIG}")


if __name__ == "__main__":
    main()
