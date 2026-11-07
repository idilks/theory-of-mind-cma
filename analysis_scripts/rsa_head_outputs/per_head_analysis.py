"""Per-head dissociation profile + PCA over per-head RDMs.

For each of the 34 captured heads:
  1. Build its (144, 144) cosine-distance RDM.
  2. Compute belief_effect and answer_effect (mean diff-pair distance minus
     mean same-pair distance, for each factor).
  3. Flatten the upper triangle into a 10,296-dim vector.

Outputs:
  - per_head_profiles.csv  (34 rows: layer, head, belief_effect, answer_effect, set)
  - per_head_dissociation.png  (scatter of belief vs answer effect, colored by set)
  - per_head_rdm_pca.png  (PCA of per-head RDMs in 2D; one dot per head)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations"
TOPK_DIR = ROOT / "results/rsa_head_outputs/topk_heads"
FIG_DIR = ROOT / "results/summary/aggregation"
CSV_OUT = ROOT / "results/rsa_head_outputs/per_head_profiles.csv"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]


from rsa_utils import cosine_rdm as cosine_dist


def main():
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    heads = [(h["layer"], h["head"]) for h in payloads[TEMPLATES[0]]["heads"]]
    z_all = np.concatenate([payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)
    n = len(stim)

    iu = np.triu_indices(n, k=1)
    same_belief = stim["belief_veracity"].to_numpy()[iu[0]] == stim["belief_veracity"].to_numpy()[iu[1]]
    same_answer = stim["answer_state"].to_numpy()[iu[0]] == stim["answer_state"].to_numpy()[iu[1]]

    # Tag heads by which top-20 set(s) they belong to.
    sets_csv = {
        "C1": pd.read_csv(TOPK_DIR / "C1_abstract_belief_top20.csv"),
        "C2": pd.read_csv(TOPK_DIR / "C2_abstract_photo_top16.csv"),
        "C3": pd.read_csv(TOPK_DIR / "C3_answer_changes_belief_top20.csv"),
        "C5": pd.read_csv(TOPK_DIR / "C5_answer_changes_location_only_top20.csv"),
    }
    set_membership = {h: [] for h in heads}
    for name, df in sets_csv.items():
        for r in df.itertuples():
            key = (int(r.layer), int(r.head))
            if key in set_membership:
                set_membership[key].append(name)

    # Per-head: RDM, belief_effect, answer_effect, flat upper triangle.
    rdm_flats = np.empty((len(heads), len(iu[0])), dtype=np.float32)
    rows = []
    for hi, (L, H) in enumerate(heads):
        D = cosine_dist(z_all[:, hi, :])
        d = D[iu]
        rdm_flats[hi] = d
        belief_effect = d[~same_belief].mean() - d[same_belief].mean()
        answer_effect = d[~same_answer].mean() - d[same_answer].mean()
        membership = "+".join(set_membership[(L, H)]) or "none"
        rows.append({
            "layer": L, "head": H,
            "belief_effect": float(belief_effect),
            "answer_effect": float(answer_effect),
            "ratio_b_over_a": float(belief_effect / answer_effect) if answer_effect != 0 else np.nan,
            "membership": membership,
            "in_C1": "C1" in set_membership[(L, H)],
            "in_C2": "C2" in set_membership[(L, H)],
            "in_C3": "C3" in set_membership[(L, H)],
            "in_C5": "C5" in set_membership[(L, H)],
        })

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"profiles -> {CSV_OUT}")
    print("\ntop 5 belief-pure (high belief_effect, low answer_effect):")
    print(df.assign(diff=df.belief_effect - df.answer_effect).sort_values("diff", ascending=False).head().to_string(index=False))
    print("\ntop 5 answer-pure:")
    print(df.assign(diff=df.answer_effect - df.belief_effect).sort_values("diff", ascending=False).head().to_string(index=False))

    # ---- Plot 1: dissociation scatter ----
    fig, ax = plt.subplots(figsize=(8, 7))
    def color_for(membership: str) -> str:
        # priority: pure C1 > pure C5 > pure C2 > overlap
        sets = set(membership.split("+")) if membership != "none" else set()
        if sets == {"C1"}: return "#4c72b0"
        if sets == {"C5"}: return "#dd8452"
        if sets == {"C2"}: return "#55a868"
        if sets == {"C3"}: return "#c44e52"
        if "C1" in sets and "C5" in sets: return "#888888"
        if "C1" in sets: return "#4c72b0"
        if "C5" in sets: return "#dd8452"
        return "#cccccc"

    for r in df.itertuples():
        c = color_for(r.membership)
        ax.scatter(r.answer_effect, r.belief_effect, c=c, s=80, edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(f"L{r.layer}H{r.head}", (r.answer_effect, r.belief_effect),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    lim = max(df.belief_effect.max(), df.answer_effect.max()) * 1.05
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.7, label="belief = answer")
    ax.set_xlim(-0.02, lim); ax.set_ylim(-0.02, lim)
    ax.set_xlabel("answer effect (diff − same answer pair distance)")
    ax.set_ylabel("belief effect (diff − same belief pair distance)")
    ax.set_title("Per-head dissociation profile (n=34 captured heads)\n"
                 "above diagonal = belief-dominant, below = answer-dominant")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0", markersize=10, label="C1 only", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dd8452", markersize=10, label="C5 only", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#55a868", markersize=10, label="C2 only", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c44e52", markersize=10, label="C3 only", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markersize=10, label="C1+C5 overlap", markeredgecolor="black"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_head_dissociation.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'per_head_dissociation.png'}")

    # ---- Plot 2: PCA on per-head RDMs (each head = 10,296-dim vector) ----
    pca = PCA(n_components=2)
    coords = pca.fit_transform(rdm_flats)
    var_explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    for r, (x, y) in zip(df.itertuples(), coords):
        c = color_for(r.membership)
        ax.scatter(x, y, c=c, s=80, edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(f"L{r.layer}H{r.head}", (x, y), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)")
    ax.set_title("PCA of per-head RDMs (each dot = one head's full similarity pattern)\n"
                 "Heads near each other have similar pairwise similarity structure")
    ax.legend(handles=handles, loc="best", fontsize=8)
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_head_rdm_pca.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'per_head_rdm_pca.png'}")
    print(f"\nPCA variance explained: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, total={sum(var_explained):.3f}")


if __name__ == "__main__":
    main()
