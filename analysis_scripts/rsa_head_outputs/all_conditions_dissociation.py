"""Dissociation profile across all 7 conditions (C1-C7), using captured head subsets.

For each condition, take its top-20 heads (or as many as are in the captured 34),
weight by clipped CMA score, build the weighted-average 144x144 RDM, bucket pairs
by (same/diff belief) x (same/diff answer), and report the four bucket means
plus belief_effect, answer_effect, and ratio.

Outputs:
  - results/rsa_head_outputs/all_conditions_buckets.csv
  - results_for_pi/aggregation/all_conditions_dissociation.png
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
FIG_OUT = ROOT / "results_for_pi/aggregation/all_conditions_dissociation.png"
CSV_OUT = ROOT / "results/rsa_head_outputs/all_conditions_buckets.csv"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]

# (label, csv_filename) pairs.
CONDITIONS = [
    ("C1 belief", "C1_abstract_belief_top20.csv"),
    ("C2 photo (state)", "C2_abstract_photo_top16.csv"),
    ("C3 belief+answer", "C3_answer_changes_belief_top20.csv"),
    ("C4 photo+answer", "C4_answer_changes_photo_top18.csv"),
    ("C5 answer", "C5_answer_changes_location_only_top20.csv"),
    ("C6 photo+answer (alt)", "C6_answer_changes_location_only_photo_top15.csv"),
    ("C7 state-only", "C7_state_only_no_belief_top15.csv"),
]


from rsa_utils import cosine_rdm as cosine_dist


def main():
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}
    heads = [(h["layer"], h["head"]) for h in payloads[TEMPLATES[0]]["heads"]]
    head_to_idx = {h: i for i, h in enumerate(heads)}
    z_all = np.concatenate([payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)
    n = len(stim)

    iu = np.triu_indices(n, k=1)
    same_belief = stim["belief_veracity"].to_numpy()[iu[0]] == stim["belief_veracity"].to_numpy()[iu[1]]
    same_answer = stim["answer_state"].to_numpy()[iu[0]] == stim["answer_state"].to_numpy()[iu[1]]

    # Pre-compute per-head RDM stack so we don't recompute for each condition.
    head_rdms = np.stack([cosine_dist(z_all[:, h, :]) for h in range(len(heads))], axis=0)  # (34, 144, 144)

    rows = []
    bar_records = []
    for label, csv_name in CONDITIONS:
        df = pd.read_csv(TOPK_DIR / csv_name)
        idx = np.array([head_to_idx[(int(r.layer), int(r.head))] for r in df.itertuples()])
        w = np.clip(df["score"].to_numpy(), 0, None)
        w = w / w.sum()
        D = np.tensordot(w, head_rdms[idx], axes=1)
        d = D[iu]

        ss = d[same_belief & same_answer].mean()
        sd = d[same_belief & ~same_answer].mean()
        ds = d[~same_belief & same_answer].mean()
        dd = d[~same_belief & ~same_answer].mean()
        b_eff = d[~same_belief].mean() - d[same_belief].mean()
        a_eff = d[~same_answer].mean() - d[same_answer].mean()
        ratio = b_eff / a_eff if a_eff != 0 else np.nan

        rows.append({
            "condition": label, "n_heads": len(idx),
            "same_belief_same_answer": ss,
            "same_belief_diff_answer": sd,
            "diff_belief_same_answer": ds,
            "diff_belief_diff_answer": dd,
            "belief_effect": b_eff,
            "answer_effect": a_eff,
            "ratio_b_over_a": ratio,
        })
        bar_records.append((label, [ss, sd, ds, dd]))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(CSV_OUT, index=False)
    print(out_df.round(3).to_string(index=False))
    print(f"\nbuckets -> {CSV_OUT}")

    # Bar plot: 7 panels in a row.
    n_cond = len(CONDITIONS)
    fig, axes = plt.subplots(1, n_cond, figsize=(3.2 * n_cond, 5), sharey=False)
    bucket_labels = ["same belief\nsame answer", "same belief\ndiff answer",
                     "diff belief\nsame answer", "diff belief\ndiff answer"]
    colors = ["#4c72b0", "#7aa6d2", "#dd8452", "#e8b285"]
    for ax, (label, vals) in zip(axes, bar_records):
        bars = ax.bar(bucket_labels, vals, color=colors)
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", labelsize=7)
        for b, m in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("mean cosine distance")
    fig.suptitle("Dissociation profile across all 7 CMA conditions (using captured head subsets)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"figure -> {FIG_OUT}")


if __name__ == "__main__":
    main()
