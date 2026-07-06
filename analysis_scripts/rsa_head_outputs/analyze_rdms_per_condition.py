"""Per-condition top-20 RSA (Taylor's dissociation test).

Two analyses on the 43-head union capture, both using each condition's OWN
top-20 (no shared pool):

  1. 7x7 between-condition Mantel — per-condition CMA-weighted RDM built
     from ONLY that condition's top-20 heads. Removes the shared-pool
     inflation that pushed r(C1,C2) to 0.97 in analyze_rdms_7cond.py.

  2. 7x2 condition-vs-model Mantel (Taylor's original RSA proposal) —
     each condition's weighted RDM Mantel'd against:
       - belief-veracity model RDM (0 if same belief, 1 if different)
       - answer-state model RDM (0 if same answer location, 1 if different)
     Prediction: abstract-belief (C1) and abstract-photo (C2) load on the
     belief model; control conditions (C5, C6, C7) load on the answer model.

Inputs:
  results/rsa_head_outputs/activations_7cond/{template}.pt
  results/rsa_head_outputs/topk_heads_7cond/{cond}_top20.csv

Outputs:
  results/rsa_head_outputs/aggregation_per_condition/mantel_7x7_per_cond.csv
  results/rsa_head_outputs/aggregation_per_condition/mantel_vs_models.csv
  results_for_pi/aggregation/mantel_7x7_per_cond.png
  results_for_pi/aggregation/mantel_vs_models.png
  results_for_pi/aggregation/per_cond_rdms.png   (grid of 7 condition RDMs + 2 model RDMs)
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
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_per_condition"
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
    flat1 = upper_tri(d1)
    null = np.empty(n_perm)
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
                    help="Mantel permutations (1000 fast, 5000 final)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    head_lists = [[(h["layer"], h["head"]) for h in payloads[t]["heads"]] for t in TEMPLATES]
    assert all(hl == head_lists[0] for hl in head_lists), "head order mismatch across templates"
    heads = head_lists[0]
    n_heads = len(heads)

    z_all = np.concatenate(
        [payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0,
    )
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)

    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)
    n_stim = len(stim)
    print(f"loaded {n_stim} stimuli x {n_heads} captured heads")

    head_rdms = np.stack([cosine_rdm(z_all[:, h, :]) for h in range(n_heads)], axis=0)

    # Per-condition top-20: separate pool per condition.
    head_to_idx = {(L, H): i for i, (L, H) in enumerate(heads)}
    cond_rdms: dict[str, np.ndarray] = {}
    cond_pool_size: dict[str, int] = {}
    for short, full in CONDITIONS:
        df = pd.read_csv(TOPK_DIR / f"{full}_top20.csv")
        idx = np.array([head_to_idx[(int(r.layer), int(r.head))] for r in df.itertuples()
                        if (int(r.layer), int(r.head)) in head_to_idx])
        if len(idx) < len(df):
            print(f"warning: {short} top-20 missing {len(df) - len(idx)} heads in 43-head capture")
        weights = df["score"].to_numpy()[:len(idx)]
        cond_rdms[short] = weighted_rdm(head_rdms, idx, weights)
        cond_pool_size[short] = len(idx)
        print(f"  {short}: pool size = {len(idx)}")

    # Model RDMs.
    belief = stim["belief_veracity"].to_numpy()
    answer = stim["answer_state"].to_numpy()
    model_belief = (belief[:, None] != belief[None, :]).astype(np.float32)
    model_answer = (answer[:, None] != answer[None, :]).astype(np.float32)
    model_rdms = {"belief": model_belief, "answer": model_answer}

    rng = np.random.default_rng(0)

    # 7x7 between-condition Mantel.
    keys = [s for s, _ in CONDITIONS]
    K = len(keys)
    rmat = np.zeros((K, K))
    pmat = np.zeros((K, K))
    rows = []
    print(f"\n7x7 between-condition Mantel ({args.n_perm} perms):")
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
    pd.DataFrame(rows).to_csv(OUT_DIR / "mantel_7x7_per_cond.csv", index=False)

    # 7x2 condition vs model Mantel.
    print(f"\n7x2 condition-vs-model Mantel:")
    model_rows = []
    model_mat = np.zeros((K, 2))
    model_pmat = np.zeros((K, 2))
    for i, ki in enumerate(keys):
        for j, mname in enumerate(["belief", "answer"]):
            r, p = mantel(cond_rdms[ki], model_rdms[mname], args.n_perm, rng)
            model_mat[i, j] = r
            model_pmat[i, j] = p
            model_rows.append({
                "condition": ki, "model": mname,
                "spearman_r": r, "p_perm": p,
                "pool_size": cond_pool_size[ki],
            })
            print(f"  {ki} vs {mname:6s} model: r={r:+.3f}, p={p:.4f}")
    pd.DataFrame(model_rows).to_csv(OUT_DIR / "mantel_vs_models.csv", index=False)

    # Plot 1: 7x7 between-condition.
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_xticklabels(keys)
    ax.set_yticks(range(K)); ax.set_yticklabels(keys)
    for i in range(K):
        for j in range(K):
            color = "white" if abs(rmat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{rmat[i, j]:+.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    ax.set_title(f"7x7 Mantel — per-condition top-20 (separate pool)\n"
                 f"({args.n_perm} stimulus perms)")
    plt.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mantel_7x7_per_cond.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'mantel_7x7_per_cond.png'}")

    # Plot 2: condition-vs-model bar / heatmap.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(model_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["belief-veracity\nmodel", "answer-state\nmodel"])
    ax.set_yticks(range(K)); ax.set_yticklabels(keys)
    for i in range(K):
        for j in range(2):
            color = "white" if abs(model_mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{model_mat[i, j]:+.2f}", ha="center", va="center",
                    color=color, fontsize=11)
    ax.set_title(f"Per-condition top-20 RDM vs model RDMs\n"
                 f"prediction: belief-conds load on belief model;\n"
                 f"control-conds load on answer model")
    plt.colorbar(im, ax=ax, label="Spearman r")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mantel_vs_models.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'mantel_vs_models.png'}")

    # Plot 3: grid of all 7 condition RDMs + 2 model RDMs.
    fig, axes = plt.subplots(3, 3, figsize=(13, 13))
    ax_iter = iter(axes.flat)
    for short, full in CONDITIONS:
        ax = next(ax_iter)
        rdm = cond_rdms[short]
        im = ax.imshow(rdm, cmap="viridis", vmin=0, vmax=float(rdm.max()))
        ax.set_title(f"{short} (top-{cond_pool_size[short]} weighted)")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    for mname, mrdm in model_rdms.items():
        ax = next(ax_iter)
        im = ax.imshow(mrdm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"model: {mname}-veracity" if mname == "belief"
                     else f"model: {mname}-state")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Per-condition CMA-weighted RDMs and model templates "
                 "(rows/cols sorted by belief_veracity, then answer_state)",
                 y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_cond_rdms.png", dpi=150)
    print(f"figure -> {FIG_DIR / 'per_cond_rdms.png'}")


if __name__ == "__main__":
    main()
