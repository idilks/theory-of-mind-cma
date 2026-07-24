#!/usr/bin/env python3
"""Build head-set CSVs for the code-domain ablation arm (July 2026 redesign).

Sources (seed 0, end-patch, template_indexed_list, max sample_num):
  code_abstract  : union of POSITIVE significant heads over ABA->ABB and
                   ABB->ABA directions (divergence-tracking analogs)
  code_control   : union of POSITIVE significant heads over ABA->ABA and
                   ABB->ABB directions (answer-writer heads)
  tom_intersection : the ToM C1 ∩ C2 19-head set, copied verbatim from
                   other_results/ablation/decomp_intersection_belief_rank/head_list.csv

Ranking: mean observed_score across the directions where the head is
significant, sorted by |mean| descending. Negative-score heads are dropped
(the ablation sweeps positive carriers; L34H12 / L46H10 / L44H18 excluded).

Outputs (results/aggregated/code_head_sets/):
  code_heads_end.csv            per_condition format (condition,layer,head,mean_score)
                                -> ablation_experiment.py --head_source per_condition
                                   --rank_by mean_score
  {condition}_head_list.csv     head_list.csv format (layer,head,rank,mean_score,
                                abs_mean_score) -> random_baseline_ablation.py
                                --target_csv

NOTE: results/ is excluded from sync_to_hpc.sh — scp the output folder to the
HPC before submitting the code ablation jobs (the SLURM scripts check for it).
"""

from pathlib import Path
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# code conditions are not in the C1-C7 condition_registry (ToM-only); import
# the enum directly for canonical folder names.
sys.path.insert(0, str(PROJECT_ROOT / "codebase" / "tasks" / "causal_analysis"))
from cma_config import CMACondition  # noqa: E402

CMA_ROOT = PROJECT_ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
TOM_INTERSECTION_CSV = (
    PROJECT_ROOT / "other_results/ablation/decomp_intersection_belief_rank/head_list.csv"
)
OUT_DIR = PROJECT_ROOT / "results/aggregated/code_head_sets"

# (condition folder, [(base_rule, exp_rule), ...]) — code_abstract flips the
# rule across directions; code_control keeps it (within-context pairs).
CODE_SOURCES = {
    CMACondition.CODE_ABSTRACT.value: [("ABA", "ABB"), ("ABB", "ABA")],
    CMACondition.CODE_CONTROL.value: [("ABA", "ABA"), ("ABB", "ABB")],
}


def resolve_significant_heads(condition: str, base_rule: str, exp_rule: str) -> Path:
    """Find significant_heads.csv at max sample_num (seed 0, end patch)."""
    cell = (CMA_ROOT / condition / "template_indexed_list"
            / f"base_rule_{base_rule}_exp_rule_{exp_rule}"
            / "patch_before_movement" / "z_seed_0_shuffle_False" / "logit")
    if not cell.is_dir():
        raise FileNotFoundError(f"missing CMA cell: {cell}")
    candidates = []
    for d in cell.iterdir():
        m = re.match(r"sample_num_(\d+)_gen_acc_", d.name)
        csv = d / "significant_heads.csv"
        if m and csv.is_file():
            candidates.append((int(m.group(1)), csv))
    if not candidates:
        raise FileNotFoundError(f"no significant_heads.csv under {cell}")
    n, path = max(candidates)
    print(f"  {condition} {base_rule}->{exp_rule}: sample_num={n}  {path.relative_to(PROJECT_ROOT)}")
    return path


def union_positive_heads(paths) -> pd.DataFrame:
    """Union positive significant heads across direction CSVs; mean score
    over the directions where each head appears; rank by |mean| desc."""
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df = df[df["significant"].astype(str).str.lower() == "true"]
        frames.append(df[["layer", "head", "observed_score"]])
    merged = pd.concat(frames, ignore_index=True)
    agg = (merged.groupby(["layer", "head"], as_index=False)["observed_score"]
                 .mean().rename(columns={"observed_score": "mean_score"}))
    n_neg = int((agg["mean_score"] <= 0).sum())
    if n_neg:
        dropped = agg[agg["mean_score"] <= 0]
        print(f"  dropping {n_neg} negative-score head(s): "
              f"{[(int(r.layer), int(r.head)) for r in dropped.itertuples()]}")
    agg = agg[agg["mean_score"] > 0].copy()
    agg["abs_mean_score"] = agg["mean_score"].abs()
    agg = agg.sort_values("abs_mean_score", ascending=False).reset_index(drop=True)
    agg["layer"] = agg["layer"].astype(int)
    agg["head"] = agg["head"].astype(int)
    return agg


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = {}

    for condition, rule_pairs in CODE_SOURCES.items():
        print(f"== {condition} ==")
        paths = [resolve_significant_heads(condition, b, e) for b, e in rule_pairs]
        sets[condition] = union_positive_heads(paths)

    print("== tom_intersection ==")
    inter = pd.read_csv(TOM_INTERSECTION_CSV)
    inter = inter[["layer", "head", "mean_score", "abs_mean_score"]].copy()
    inter["layer"] = inter["layer"].astype(int)
    inter["head"] = inter["head"].astype(int)
    inter = inter.sort_values("abs_mean_score", ascending=False).reset_index(drop=True)
    print(f"  copied {len(inter)} heads from {TOM_INTERSECTION_CSV.relative_to(PROJECT_ROOT)}")
    sets["tom_intersection"] = inter

    # per_condition combined CSV
    pieces = []
    for cond, df in sets.items():
        piece = df[["layer", "head", "mean_score"]].copy()
        piece.insert(0, "condition", cond)
        pieces.append(piece)
    combined = pd.concat(pieces, ignore_index=True)
    combined_path = OUT_DIR / "code_heads_end.csv"
    combined.to_csv(combined_path, index=False)
    print(f"wrote {combined_path.relative_to(PROJECT_ROOT)} ({len(combined)} rows)")

    # per-set head_list.csv files (random_baseline_ablation --target_csv)
    for cond, df in sets.items():
        out = df[["layer", "head"]].copy()
        out["rank"] = range(1, len(out) + 1)
        out["mean_score"] = df["mean_score"].values
        out["abs_mean_score"] = df["abs_mean_score"].values
        path = OUT_DIR / f"{cond}_head_list.csv"
        out.to_csv(path, index=False)
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(out)} heads)")

    print("\nhead counts:", {c: len(df) for c, df in sets.items()})


if __name__ == "__main__":
    main()
