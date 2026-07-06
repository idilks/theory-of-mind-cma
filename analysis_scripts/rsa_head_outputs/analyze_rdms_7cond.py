"""7-condition RDM analysis on the union-of-top-20 head capture.

Reads activations from results/rsa_head_outputs/activations_7cond/ (43 heads
across 144 stimuli, captured from union of top-20 heads in C1..C7), then:

  1. Builds a (n_stim, n_stim) cosine-distance RDM per head.
  2. For each of the 7 conditions, builds a CMA-score-weighted average RDM
     using that condition's top-20 (with weights = clipped CMA scores,
     normalized to sum to 1).
  3. Computes a 7x7 Mantel (Spearman) matrix between condition-weighted RDMs,
     with a permutation null (stimulus relabeling) for each off-diagonal.
  4. Computes per-head dissociation (belief_effect, answer_effect) and tags
     each head with its membership across all 7 conditions.
  5. Saves a heatmap of the 7x7 Mantel matrix and an updated dissociation
     scatter.

Outputs:
  results/rsa_head_outputs/aggregation_7cond/mantel_7x7.csv
  results/rsa_head_outputs/aggregation_7cond/per_head_profiles_7cond.csv
  results_for_pi/aggregation/mantel_7x7_7cond.png
  results_for_pi/aggregation/per_head_dissociation_7cond.png
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
ACT_DIR = ROOT / "results/rsa_head_outputs/activations_7cond"
TOPK_DIR = ROOT / "results/rsa_head_outputs/topk_heads_7cond"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_7cond"
FIG_DIR = ROOT / "results_for_pi/aggregation"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]

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
    null = np.empty(n_perm)
    flat1 = upper_tri(d1)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = float(spearmanr(flat1, upper_tri(d2[perm][:, perm])).correlation)
    p = float((np.abs(null) >= np.abs(obs)).mean())
    return obs, p


def weighted_rdm(head_rdms: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 0, None)
    w = w / w.sum() if w.sum() > 0 else w
    return np.tensordot(w, head_rdms[idx], axes=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="permutations per Mantel test (1000 is fast, bump to 5000 for final)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    head_lists = [[(h["layer"], h["head"]) for h in payloads[t]["heads"]] for t in TEMPLATES]
    assert all(hl == head_lists[0] for hl in head_lists), "head order mismatch across templates"
    heads = head_lists[0]
    n_heads = len(heads)

    z_all = np.concatenate([payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)

    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)
    n_stim = len(stim)
    print(f"loaded {n_stim} stimuli x {n_heads} heads")

    head_rdms = np.stack([cosine_rdm(z_all[:, h, :]) for h in range(n_heads)], axis=0)

    head_to_idx = {(L, H): i for i, (L, H) in enumerate(heads)}
    cond_rdms: dict[str, np.ndarray] = {}
    cond_top: dict[str, pd.DataFrame] = {}
    for short, full in CONDITIONS:
        df = pd.read_csv(TOPK_DIR / f"{full}_top20.csv")
        cond_top[short] = df
        idx = np.array([head_to_idx[(int(r.layer), int(r.head))] for r in df.itertuples()
                        if (int(r.layer), int(r.head)) in head_to_idx])
        if len(idx) < len(df):
            print(f"warning: {short} top-20 missing {len(df) - len(idx)} heads in capture")
        weights = df["score"].to_numpy()[:len(idx)]
        cond_rdms[short] = weighted_rdm(head_rdms, idx, weights)

    rng = np.random.default_rng(0)
    keys = [s for s, _ in CONDITIONS]
    K = len(keys)
    rmat = np.zeros((K, K))
    pmat = np.zeros((K, K))
    rows = []
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            if i > j:
                rmat[i, j] = rmat[j, i]
                pmat[i, j] = pmat[j, i]
                continue
            if i == j:
                rmat[i, j] = 1.0
                pmat[i, j] = 0.0
                continue
            r, p = mantel(cond_rdms[ki], cond_rdms[kj], args.n_perm, rng)
            rmat[i, j] = r
            pmat[i, j] = p
            rows.append({"a": ki, "b": kj, "spearman_r": r, "p_perm": p})
            print(f"  {ki} <-> {kj}: r={r:+.3f}, p={p:.4f}")

    pd.DataFrame(rows).to_csv(OUT_DIR / "mantel_7x7.csv", index=False)
    print(f"mantel csv -> {OUT_DIR / 'mantel_7x7.csv'}")

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(keys)
    ax.set_yticks(range(K)); ax.set_yticklabels(keys)
    for i in range(K):
        for j in range(K):
            color = "white" if abs(rmat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{rmat[i, j]:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_title(f"7x7 Mantel matrix between condition-weighted RDMs\n"
                 f"(43 heads, top-20 per condition, {args.n_perm} perms)")
    plt.colorbar(im, ax=ax, label="Spearman r (RDM upper triangles)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mantel_7x7_7cond.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'mantel_7x7_7cond.png'}")

    iu = np.triu_indices(n_stim, k=1)
    same_belief = stim["belief_veracity"].to_numpy()[iu[0]] == stim["belief_veracity"].to_numpy()[iu[1]]
    same_answer = stim["answer_state"].to_numpy()[iu[0]] == stim["answer_state"].to_numpy()[iu[1]]

    membership_set = {h: set() for h in heads}
    for short, df in cond_top.items():
        for r in df.itertuples():
            key = (int(r.layer), int(r.head))
            if key in membership_set:
                membership_set[key].add(short)

    prof_rows = []
    for hi, (L, H) in enumerate(heads):
        D = head_rdms[hi]
        d = D[iu]
        be = float(d[~same_belief].mean() - d[same_belief].mean())
        ae = float(d[~same_answer].mean() - d[same_answer].mean())
        members = sorted(membership_set[(L, H)], key=lambda x: int(x[1:]))
        prof_rows.append({
            "layer": L, "head": H,
            "belief_effect": be, "answer_effect": ae,
            "membership": "+".join(members) if members else "none",
            **{f"in_{k}": (k in membership_set[(L, H)]) for k in keys},
        })
    prof = pd.DataFrame(prof_rows)
    prof.to_csv(OUT_DIR / "per_head_profiles_7cond.csv", index=False)
    print(f"profiles -> {OUT_DIR / 'per_head_profiles_7cond.csv'}")

    print("\ntop-5 belief-dominant:")
    print(prof.assign(d=prof.belief_effect - prof.answer_effect)
              .sort_values("d", ascending=False).head().to_string(index=False))
    print("\ntop-5 answer-dominant:")
    print(prof.assign(d=prof.answer_effect - prof.belief_effect)
              .sort_values("d", ascending=False).head().to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 8))
    def color_for(members: set[str]) -> str:
        across = members & {"C1", "C2", "C3", "C4"}
        within = members & {"C5", "C6", "C7"}
        if across and not within: return "#4c72b0"
        if within and not across: return "#dd8452"
        if across and within: return "#888888"
        return "#cccccc"
    for r in prof.itertuples():
        members = set(r.membership.split("+")) if r.membership != "none" else set()
        ax.scatter(r.answer_effect, r.belief_effect,
                   c=color_for(members), s=80,
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.annotate(f"L{r.layer}H{r.head}", (r.answer_effect, r.belief_effect),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    lim = max(prof.belief_effect.max(), prof.answer_effect.max()) * 1.1
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.7)
    ax.set_xlim(-0.05, lim); ax.set_ylim(-0.05, lim)
    ax.set_xlabel("answer effect")
    ax.set_ylabel("belief effect")
    ax.set_title(f"Per-head dissociation, 7-condition union (n={n_heads} heads)\n"
                 "above diagonal = belief-dominant, below = answer-dominant")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0",
                   markersize=10, label="across-context only (C1-C4)", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dd8452",
                   markersize=10, label="within-context only (C5-C7)", markeredgecolor="black"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888",
                   markersize=10, label="both axes", markeredgecolor="black"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_head_dissociation_7cond.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'per_head_dissociation_7cond.png'}")


if __name__ == "__main__":
    main()
