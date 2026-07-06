#!/usr/bin/env python3
"""Build a per_condition-format head CSV that decomposes the published
cross-frame ablation head sets (belief K=32, photo K=26) into:

  intersection_photo_rank  : 19 heads in B ∩ P, ranked by their photo score
  photo_only               :  7 heads in P \\ B, ranked by their photo score
  intersection_belief_rank : 19 heads in B ∩ P, ranked by their belief score
  belief_only              : 13 heads in B \\ P, ranked by their belief score

The decomposition answers the reviewer question "isn't the cross-frame transfer
just the intersection?" by isolating the heads each frame selects but the
other frame's selection missed.

Output CSV slots into the existing ablation_experiment.py --head_source
per_condition path with --rank_by mean_score.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BELIEF_HEAD_CSV = PROJECT_ROOT / "other_results/ablation/belief_only_sysprompt_prefill/head_list.csv"
PHOTO_HEAD_CSV  = PROJECT_ROOT / "other_results/ablation/photo_heads_sysprompt_prefill/head_list.csv"

OUT_CSV = PROJECT_ROOT / "results/lookback_analysis/_summary/decomposition_heads_end.csv"


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)
    df["mean_score"] = df["mean_score"].astype(float)
    return df[["layer", "head", "mean_score"]]


def _subset_rows(source_df: pd.DataFrame, keep: set, condition: str) -> pd.DataFrame:
    sub = source_df[source_df.apply(lambda r: (r["layer"], r["head"]) in keep, axis=1)].copy()
    sub = sub.sort_values("mean_score", ascending=False).reset_index(drop=True)
    sub.insert(0, "condition", condition)
    return sub


def main():
    belief = _load(BELIEF_HEAD_CSV)
    photo  = _load(PHOTO_HEAD_CSV)

    B = set(zip(belief["layer"], belief["head"]))
    P = set(zip(photo["layer"], photo["head"]))
    inter = B & P
    photo_only  = P - B
    belief_only = B - P

    print(f"|B|={len(B)}  |P|={len(P)}  |B∩P|={len(inter)}  "
          f"|P\\B|={len(photo_only)}  |B\\P|={len(belief_only)}")

    pieces = [
        _subset_rows(photo,  inter,        "intersection_photo_rank"),
        _subset_rows(photo,  photo_only,   "photo_only"),
        _subset_rows(belief, inter,        "intersection_belief_rank"),
        _subset_rows(belief, belief_only,  "belief_only"),
    ]
    out = pd.concat(pieces, ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}  ({len(out)} rows)")
    print(out.groupby("condition").size())


if __name__ == "__main__":
    main()
