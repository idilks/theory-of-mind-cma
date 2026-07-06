#!/usr/bin/env python3
"""Build a synthetic head CSV row block for the abstract_belief x abstract_photo
intersection so the existing ablation pipeline can rank-and-ablate it.

The intersection heads are read from
  results/lookback_analysis/_summary/condition_overlap_heads_end.csv
which lists (layer, head) only -- no scores. We pull the per-head
max_observed_score from each parent condition in
  results/lookback_analysis/_summary/top20_by_condition_end.csv
and assign the geometric mean as the intersection's score (so heads weak in
either parent rank low). Output is appended to top20 in the same schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

INTERSECTION_LABEL = "intersection_belief_photo"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary_dir", type=Path,
                   default=Path("results/lookback_analysis/_summary"))
    p.add_argument("--cond_a", default="abstract_belief")
    p.add_argument("--cond_b", default="abstract_photo")
    p.add_argument("--label", default=INTERSECTION_LABEL)
    p.add_argument("--out_csv", type=Path,
                   default=Path("results/lookback_analysis/_summary/top20_with_intersection_end.csv"))
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    summary_dir = args.summary_dir if args.summary_dir.is_absolute() else project_root / args.summary_dir
    out_csv = args.out_csv if args.out_csv.is_absolute() else project_root / args.out_csv

    overlap = pd.read_csv(summary_dir / "condition_overlap_heads_end.csv")
    top20 = pd.read_csv(summary_dir / "top20_by_condition_end.csv")

    intersect = overlap[
        (overlap.cond_a == args.cond_a)
        & (overlap.cond_b == args.cond_b)
        & (overlap["set"] == "intersection")
    ][["layer", "head"]].drop_duplicates()
    print(f"intersection {args.cond_a} ∩ {args.cond_b}: {len(intersect)} heads")

    a_scores = top20[top20.condition == args.cond_a].set_index(["layer", "head"])
    b_scores = top20[top20.condition == args.cond_b].set_index(["layer", "head"])

    rows = []
    for _, row in intersect.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        key = (layer, head)
        if key not in a_scores.index or key not in b_scores.index:
            continue
        a = a_scores.loc[key]
        b = b_scores.loc[key]
        rows.append({
            "condition": args.label,
            "layer": layer,
            "head": head,
            "n_templates": min(a.n_templates, b.n_templates),
            "templates_present": a.templates_present,
            "n_rules": min(a.n_rules, b.n_rules),
            "rules_present": a.rules_present,
            "max_observed_score": float(np.sqrt(max(a.max_observed_score, 0) * max(b.max_observed_score, 0))),
            "mean_observed_score": float(np.sqrt(max(a.mean_observed_score, 0) * max(b.mean_observed_score, 0))),
            "min_p_value": max(a.min_p_value, b.min_p_value),
        })

    inter_df = pd.DataFrame(rows).sort_values("max_observed_score", ascending=False)
    print(f"matched scores for {len(inter_df)}/{len(intersect)} heads")
    print(f"score range: [{inter_df.max_observed_score.min():.2f}, {inter_df.max_observed_score.max():.2f}]")

    out = pd.concat([top20, inter_df], ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(out)} rows, +{len(inter_df)} for label '{args.label}')")


if __name__ == "__main__":
    main()
