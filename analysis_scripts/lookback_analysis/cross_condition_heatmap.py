"""Aggregate lookback->state attention across all conditions and render a heads x conditions heatmap.

Reads every lookback_scores.csv under --root, picks the union of top-N heads
(ranked by mean lookback->state across all runs), and plots one cell per
(head, condition), value = mean lookback->state attention averaged over
templates and directions for that condition.

Usage:
  python scripts/lookback_analysis/cross_condition_heatmap.py \
      --root other_results/lookback_analysis \
      --top_n 20 \
      --out_dir other_results/lookback_analysis/_summary
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATTERN = re.compile(
    r"lookback_analysis/([^/]+)/template_([^/]+)/base_rule_([^_]+)_exp_rule_([^/]+)/"
)


def load_all(root: Path) -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(root / "*/template_*/base_rule_*/patch_before_movement/lookback_scores.csv")):
        m = PATTERN.search(f)
        if not m:
            continue
        cond, tmpl, br, er = m.groups()
        df = pd.read_csv(f)
        df["condition"] = cond
        df["template"] = tmpl
        df["direction"] = f"{br}->{er}"
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="other_results/lookback_analysis")
    ap.add_argument("--top_n", type=int, default=20,
                    help="number of heads to include (ranked by mean lookback->state across all runs)")
    ap.add_argument("--out_dir", default="other_results/lookback_analysis/_summary")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_df = load_all(root)
    print(f"loaded {len(all_df)} head-rows from {all_df.condition.nunique()} conditions, "
          f"{all_df.template.nunique()} templates, {all_df.direction.nunique()} directions")

    # rank heads by mean lookback->state across everything
    head_rank = (all_df.groupby(["layer", "head"])
                 .agg(mean_l2s=("lookback_attn_to_state", "mean"),
                      n_runs=("lookback_attn_to_state", "count"))
                 .reset_index())
    head_rank = head_rank.sort_values("mean_l2s", ascending=False)
    top_heads = head_rank.head(args.top_n).reset_index(drop=True)
    head_labels = [f"L{int(r.layer)}H{int(r.head)}" for r in top_heads.itertuples()]
    print(f"top {len(top_heads)} heads selected")

    # build heads x conditions matrix
    conditions = sorted(all_df.condition.unique())
    matrix = np.full((len(top_heads), len(conditions)), np.nan)
    n_runs_matrix = np.zeros_like(matrix)
    for i, hr in top_heads.iterrows():
        L, H = int(hr["layer"]), int(hr["head"])
        for j, cond in enumerate(conditions):
            sub = all_df[(all_df.layer == L) & (all_df["head"] == H) & (all_df.condition == cond)]
            if len(sub) > 0:
                matrix[i, j] = sub["lookback_attn_to_state"].mean()
                n_runs_matrix[i, j] = len(sub)

    # plot
    fig, ax = plt.subplots(figsize=(1.6 * len(conditions) + 2, 0.4 * len(top_heads) + 2))
    vmax = np.nanmax(matrix)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(top_heads)))
    ax.set_yticklabels(head_labels)
    ax.set_xlabel("condition (averaged over templates × directions)")
    ax.set_ylabel(f"top-{args.top_n} heads (ranked by mean lookback→state)")
    ax.set_title("Lookback→state attention across conditions\n"
                 "(end-position CMA top-50 heads, mean over prompts)")

    # annotate cells with the value (and mark missing with 'x')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="gray", fontsize=8)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < vmax * 0.55 else "black", fontsize=7)

    fig.colorbar(im, ax=ax, label="mean attention from lookback token to state token")
    fig.tight_layout()
    out_path = out_dir / f"heatmap_top{args.top_n}_lookback_to_state.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")

    # also save the matrix as CSV for the slides
    mat_df = pd.DataFrame(matrix, index=head_labels, columns=conditions)
    mat_df.to_csv(out_dir / f"heatmap_top{args.top_n}_matrix.csv")
    print(f"wrote {out_dir / f'heatmap_top{args.top_n}_matrix.csv'}")

    # second figure: specificity (state - max(other)) for the same heads
    spec_matrix = np.full_like(matrix, np.nan)
    for i, hr in top_heads.iterrows():
        L, H = int(hr["layer"]), int(hr["head"])
        for j, cond in enumerate(conditions):
            sub = all_df[(all_df.layer == L) & (all_df["head"] == H) & (all_df.condition == cond)]
            if len(sub) > 0:
                state = sub["lookback_attn_to_state"].mean()
                others = sub[["lookback_attn_to_char_q", "lookback_attn_to_obj_q",
                              "lookback_attn_to_belief_form"]].mean(axis=0).max()
                spec_matrix[i, j] = state - others

    fig, ax = plt.subplots(figsize=(1.6 * len(conditions) + 2, 0.4 * len(top_heads) + 2))
    vmax2 = np.nanmax(spec_matrix)
    im = ax.imshow(spec_matrix, aspect="auto", cmap="magma", vmin=0, vmax=vmax2)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_yticks(range(len(top_heads)))
    ax.set_yticklabels(head_labels)
    ax.set_xlabel("condition")
    ax.set_ylabel("head")
    ax.set_title("Lookback specificity: attn_to_state − max(attn_to_char_q, obj_q, belief_form)")
    for i in range(spec_matrix.shape[0]):
        for j in range(spec_matrix.shape[1]):
            v = spec_matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "·", ha="center", va="center", color="gray", fontsize=8)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < vmax2 * 0.55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="state − max(other) attention")
    fig.tight_layout()
    out_path2 = out_dir / f"heatmap_top{args.top_n}_specificity.png"
    fig.savefig(out_path2, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path2}")


if __name__ == "__main__":
    main()


