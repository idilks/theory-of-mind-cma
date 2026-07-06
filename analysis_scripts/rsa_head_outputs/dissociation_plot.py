"""Direct test of hypothesis: do C1 heads represent belief and C5 heads represent answer?

For each head set, partition the n*(n-1)/2 stimulus pairs into 4 buckets by
(same/diff belief) x (same/diff answer), and compute mean cosine distance per bucket.

Belief-coding heads should show:  SS ~ SD  <<  DS ~ DD   (belief matters, answer doesn't)
Answer-coding heads should show:  SS ~ DS  <<  SD ~ DD   (answer matters, belief doesn't)

Outputs:
  results_for_pi/aggregation/rsa_head_dissociation.png
  results/rsa_head_outputs/dissociation_buckets.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations"
TOPK_DIR = ROOT / "results/rsa_head_outputs/topk_heads"
FIG_OUT = ROOT / "results_for_pi/aggregation/rsa_head_dissociation.png"
CSV_OUT = ROOT / "results/rsa_head_outputs/dissociation_buckets.csv"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
K = 20


from rsa_utils import cosine_rdm as cosine_dist_matrix


def main():
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    heads = [(h["layer"], h["head"]) for h in payloads[TEMPLATES[0]]["heads"]]
    head_to_idx = {h: i for i, h in enumerate(heads)}

    z_all = np.concatenate([payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)
    n = len(stim)
    print(f"{n} stimuli x {len(heads)} heads")

    # Pair-mask: same/diff belief, same/diff answer (upper triangle only).
    iu = np.triu_indices(n, k=1)
    belief = stim["belief_veracity"].to_numpy()
    answer = stim["answer_state"].to_numpy()
    same_belief = belief[iu[0]] == belief[iu[1]]
    same_answer = answer[iu[0]] == answer[iu[1]]

    # Top-k head sets + CMA-score weights.
    c1 = pd.read_csv(TOPK_DIR / f"C1_abstract_belief_top{K}.csv")
    c2 = pd.read_csv(TOPK_DIR / "C2_abstract_photo_top16.csv")
    c3 = pd.read_csv(TOPK_DIR / f"C3_answer_changes_belief_top{K}.csv")
    c5 = pd.read_csv(TOPK_DIR / f"C5_answer_changes_location_only_top{K}.csv")
    sets = {
        "C1 belief heads (top-20)": (
            np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c1.itertuples()]),
            np.clip(c1["score"].to_numpy(), 0, None),
        ),
        "C2 photo (state) heads (top-16)": (
            np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c2.itertuples()]),
            np.clip(c2["score"].to_numpy(), 0, None),
        ),
        "C3 belief+answer heads (top-20)": (
            np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c3.itertuples()]),
            np.clip(c3["score"].to_numpy(), 0, None),
        ),
        "C5 answer heads (top-20)": (
            np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c5.itertuples()]),
            np.clip(c5["score"].to_numpy(), 0, None),
        ),
    }

    rows = []
    bucket_means = {}
    for name, (idx, w) in sets.items():
        w = w / w.sum()
        # Weighted average distance matrix across heads in the set.
        head_dists = np.stack([cosine_dist_matrix(z_all[:, h, :]) for h in idx], axis=0)  # (k, n, n)
        D = np.tensordot(w, head_dists, axes=1)  # (n, n)
        d = D[iu]

        buckets = {
            "same belief, same answer": d[same_belief & same_answer],
            "same belief, diff answer": d[same_belief & ~same_answer],
            "diff belief, same answer": d[~same_belief & same_answer],
            "diff belief, diff answer": d[~same_belief & ~same_answer],
        }
        bucket_means[name] = {k: (v.mean(), v.std() / np.sqrt(len(v)), len(v)) for k, v in buckets.items()}
        for bname, vals in buckets.items():
            rows.append({"head_set": name, "bucket": bname,
                         "mean_dist": float(vals.mean()),
                         "sem": float(vals.std() / np.sqrt(len(vals))),
                         "n_pairs": int(len(vals))})

        # Hypothesis-aligned contrasts.
        b_effect = d[~same_belief].mean() - d[same_belief].mean()
        a_effect = d[~same_answer].mean() - d[same_answer].mean()
        print(f"\n{name}")
        print(f"  belief effect (diff - same belief, regardless of answer): {b_effect:+.4f}")
        print(f"  answer effect (diff - same answer, regardless of belief): {a_effect:+.4f}")
        print(f"  belief/answer ratio: {b_effect/a_effect if a_effect != 0 else float('inf'):+.2f}")

    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"\nbucket means -> {CSV_OUT}")

    # Plot.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    bucket_order = ["same belief,\nsame answer", "same belief,\ndiff answer",
                    "diff belief,\nsame answer", "diff belief,\ndiff answer"]
    bucket_keys = ["same belief, same answer", "same belief, diff answer",
                   "diff belief, same answer", "diff belief, diff answer"]
    colors = ["#4c72b0", "#7aa6d2", "#dd8452", "#e8b285"]

    for ax, (name, _) in zip(axes, sets.items()):
        means = [bucket_means[name][k][0] for k in bucket_keys]
        sems = [bucket_means[name][k][1] for k in bucket_keys]
        bars = ax.bar(bucket_order, means, yerr=sems, color=colors, capsize=4)
        ax.set_title(name, fontsize=11)
        ax.set_ylabel("mean cosine distance (pairs)")
        ax.tick_params(axis="x", labelsize=9)
        # Annotate bars with values.
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Hypothesis test: which factor drives head-output similarity?\n"
                 "Belief-coding -> bars 1,2 low; bars 3,4 high.   "
                 "Answer-coding -> bars 1,3 low; bars 2,4 high.",
                 fontsize=10)
    fig.tight_layout()
    FIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT, dpi=150)
    print(f"figure -> {FIG_OUT}")


if __name__ == "__main__":
    main()
