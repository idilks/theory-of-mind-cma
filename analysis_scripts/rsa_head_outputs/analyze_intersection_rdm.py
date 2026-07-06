"""C1 ∩ C3 intersection RSA: do shared-belief heads correlate with the
belief-veracity model RDM beyond what each condition does alone?

Builds three CMA-weighted RDMs on the 144-stimulus capture and Mantels each
against the belief-veracity and answer-state model templates:

  - C1 ∩ C3 (12 heads): weight = mean of C1 and C3 clamped CMA scores
  - C1 top-12 alone:    weight = C1 clamped CMA score
  - C3 top-12 alone:    weight = C3 clamped CMA score

The two top-12-alone RDMs are apples-to-apples baselines that match the
intersection pool size, so any gain from the intersection is attributable
to the cross-condition filter rather than just narrowing k.

Noise-floor null: 200 random 12-head pools (uniform weight) from the 43-head
capture, Mantel'd against both model RDMs.

Inputs:
  results/rsa_head_outputs/activations_7cond/{template}.pt
  results/rsa_head_outputs/topk_heads_7cond/C1_abstract_belief_top20.csv
  results/rsa_head_outputs/topk_heads_7cond/C3_answer_changes_belief_top20.csv

Outputs:
  results/rsa_head_outputs/aggregation_per_condition/intersection_C1_C3.csv
  results_for_pi/aggregation/intersection_C1_C3.png
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


def weighted_rdm(head_rdms: np.ndarray, idx: np.ndarray,
                 weights: np.ndarray) -> np.ndarray:
    w = np.clip(weights, 0, None)
    w = w / w.sum() if w.sum() > 0 else w
    return np.tensordot(w, head_rdms[idx], axes=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=5000,
                    help="Mantel permutations (1000 fast, 5000 final)")
    ap.add_argument("--n_null", type=int, default=200,
                    help="random 12-head null draws")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- load activations and stimuli (same recipe as analyze_rdms_per_condition).
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False)
                for t in TEMPLATES}
    head_lists = [[(h["layer"], h["head"]) for h in payloads[t]["heads"]]
                  for t in TEMPLATES]
    assert all(hl == head_lists[0] for hl in head_lists), \
        "head order mismatch across templates"
    heads = head_lists[0]
    n_heads = len(heads)

    z_all = np.concatenate(
        [payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0,
    )
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES],
                     ignore_index=True)
    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)
    print(f"loaded {len(stim)} stimuli x {n_heads} captured heads")

    head_rdms = np.stack([cosine_rdm(z_all[:, h, :]) for h in range(n_heads)],
                         axis=0)
    head_to_idx = {(L, H): i for i, (L, H) in enumerate(heads)}

    # --- load C1 and C3 top-20 lists.
    c1 = pd.read_csv(TOPK_DIR / "C1_abstract_belief_top20.csv")
    c3 = pd.read_csv(TOPK_DIR / "C3_answer_changes_belief_top20.csv")
    c1_set = {(int(r.layer), int(r.head)): float(r.score) for r in c1.itertuples()}
    c3_set = {(int(r.layer), int(r.head)): float(r.score) for r in c3.itertuples()}

    inter = sorted(set(c1_set) & set(c3_set))
    inter = [h for h in inter if h in head_to_idx]
    n_inter = len(inter)
    print(f"|C1 ∩ C3| = {n_inter} heads (in capture)")

    # --- build the three RDMs.
    inter_idx = np.array([head_to_idx[h] for h in inter])
    inter_w = np.array([(c1_set[h] + c3_set[h]) / 2 for h in inter])
    rdm_inter = weighted_rdm(head_rdms, inter_idx, inter_w)

    # C1 top-n_inter alone (apples-to-apples size match).
    c1_sorted = sorted(c1_set.items(), key=lambda kv: -kv[1])[:n_inter]
    c1_top_idx = np.array([head_to_idx[h] for h, _ in c1_sorted
                           if h in head_to_idx])
    c1_top_w = np.array([s for h, s in c1_sorted if h in head_to_idx])
    rdm_c1_topN = weighted_rdm(head_rdms, c1_top_idx, c1_top_w)

    # C3 top-n_inter alone.
    c3_sorted = sorted(c3_set.items(), key=lambda kv: -kv[1])[:n_inter]
    c3_top_idx = np.array([head_to_idx[h] for h, _ in c3_sorted
                           if h in head_to_idx])
    c3_top_w = np.array([s for h, s in c3_sorted if h in head_to_idx])
    rdm_c3_topN = weighted_rdm(head_rdms, c3_top_idx, c3_top_w)

    # --- model RDMs.
    belief = stim["belief_veracity"].to_numpy()
    answer = stim["answer_state"].to_numpy()
    model_belief = (belief[:, None] != belief[None, :]).astype(np.float32)
    model_answer = (answer[:, None] != answer[None, :]).astype(np.float32)
    model_rdms = {"belief": model_belief, "answer": model_answer}

    # --- Mantel: each pool vs each model.
    rng = np.random.default_rng(0)
    pools = [
        ("C1 ∩ C3 (n=%d, mean weights)" % n_inter, rdm_inter),
        ("C1 top-%d alone" % n_inter, rdm_c1_topN),
        ("C3 top-%d alone" % n_inter, rdm_c3_topN),
    ]
    rows = []
    print(f"\nMantel ({args.n_perm} stimulus perms):")
    for name, rdm in pools:
        for mname, mrdm in model_rdms.items():
            r, p = mantel(rdm, mrdm, args.n_perm, rng)
            rows.append({"pool": name, "model": mname,
                         "spearman_r": r, "p_perm": p,
                         "pool_size": n_inter})
            print(f"  {name} vs {mname:6s}: r={r:+.3f}, p={p:.4f}")

    # --- noise-floor null: 200 random 12-head uniform-weighted pools.
    print(f"\nnoise-floor null: {args.n_null} random {n_inter}-head pools")
    null_rb = np.empty(args.n_null)
    null_ra = np.empty(args.n_null)
    for i in range(args.n_null):
        idx = rng.choice(n_heads, size=n_inter, replace=False)
        rdm_rand = head_rdms[idx].mean(axis=0)
        null_rb[i] = float(spearmanr(upper_tri(rdm_rand),
                                     upper_tri(model_belief)).correlation)
        null_ra[i] = float(spearmanr(upper_tri(rdm_rand),
                                     upper_tri(model_answer)).correlation)
    obs_inter_rb = next(r["spearman_r"] for r in rows
                        if r["pool"].startswith("C1 ∩ C3") and r["model"] == "belief")
    obs_inter_ra = next(r["spearman_r"] for r in rows
                        if r["pool"].startswith("C1 ∩ C3") and r["model"] == "answer")
    p_null_rb = float((null_rb >= obs_inter_rb).mean())
    p_null_ra = float((null_ra >= obs_inter_ra).mean())
    print(f"  null r_b mean={null_rb.mean():+.3f}, p(intersection ≥ null)={p_null_rb:.3f}")
    print(f"  null r_a mean={null_ra.mean():+.3f}, p(intersection ≥ null)={p_null_ra:.3f}")

    # write csv with summary + null stats.
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"pool": "null (200 random %d-head pools)" % n_inter,
                       "model": "belief",
                       "spearman_r": float(null_rb.mean()),
                       "p_perm": p_null_rb,
                       "pool_size": n_inter}
    df.loc[len(df)] = {"pool": "null (200 random %d-head pools)" % n_inter,
                       "model": "answer",
                       "spearman_r": float(null_ra.mean()),
                       "p_perm": p_null_ra,
                       "pool_size": n_inter}
    csv_path = OUT_DIR / "intersection_C1_C3.csv"
    df.to_csv(csv_path, index=False)
    print(f"csv -> {csv_path}")

    # --- figure: 2-panel. left = bar of Mantel r vs belief+answer for the 3 pools
    # plus null. right = scatter of (r_a, r_b) with null cloud and the three pools.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    pool_names_short = ["C1 ∩ C3", f"C1 top-{n_inter}", f"C3 top-{n_inter}"]
    rb_by_pool = []
    ra_by_pool = []
    for name, _ in pools:
        rb_by_pool.append(next(r["spearman_r"] for r in rows
                               if r["pool"] == name and r["model"] == "belief"))
        ra_by_pool.append(next(r["spearman_r"] for r in rows
                               if r["pool"] == name and r["model"] == "answer"))
    x = np.arange(3)
    ax.bar(x - 0.18, rb_by_pool, 0.36, label="vs belief-veracity model",
           color="#3a7", edgecolor="black")
    ax.bar(x + 0.18, ra_by_pool, 0.36, label="vs answer-state model",
           color="#a73", edgecolor="black")
    ax.axhline(null_rb.mean(), color="#3a7", linestyle="--", alpha=0.5,
               label=f"null mean r_b = {null_rb.mean():+.3f}")
    ax.axhline(null_ra.mean(), color="#a73", linestyle="--", alpha=0.5,
               label=f"null mean r_a = {null_ra.mean():+.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(pool_names_short)
    ax.set_ylabel("Spearman r (Mantel)")
    ax.set_title(f"C1 ∩ C3 vs apples-to-apples baselines\n"
                 f"(pool size = {n_inter} heads each)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.scatter(null_ra, null_rb, c="grey", alpha=0.4, s=18,
               label=f"null ({args.n_null} random {n_inter}-head pools)")
    colors = ["#d33", "#36a", "#aa3"]
    for (name, _), rb, ra, c in zip(pools, rb_by_pool, ra_by_pool, colors):
        ax.scatter(ra, rb, c=c, s=120, edgecolor="black", zorder=10,
                   label=name)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.plot([-0.2, 0.6], [-0.2, 0.6], color="black", linestyle=":", alpha=0.4)
    ax.set_xlabel("Mantel r vs answer-state model")
    ax.set_ylabel("Mantel r vs belief-veracity model")
    ax.set_title("Belief vs answer dissociation\n(above diagonal = belief-leaning)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"C1 ∩ C3 intersection RSA — does cross-condition filtering "
                 f"sharpen belief signal?\n"
                 f"({args.n_perm} stimulus perms, {args.n_null} null draws)")
    fig.tight_layout()
    fig_path = FIG_DIR / "intersection_C1_C3.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()
