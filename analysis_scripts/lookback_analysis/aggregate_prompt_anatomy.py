"""
Aggregate prompt-anatomy attention from lookback + belief_formation query
positions across all 42 cells × top heads, for the analysis described in
neurips/neurips_2026.tex sec. 'Attention-pattern characterization'.

Reads:
  other_results/lookback_analysis/{condition}/template_*/base_rule_*/patch_before_movement/lookback_scores.csv
  other_results/lookback_analysis/_summary/top20_by_condition_end.csv

Writes (to other_results/lookback_analysis/_summary/):
  prompt_anatomy_long.csv             one row per (condition, layer, head, template, base_rule, query_pos, source_cat)
  prompt_anatomy_per_head_profile.csv head x (query, source) averaged over templates and rules
  prompt_anatomy_template_consistency.csv  per-template std for each (head, query, source)
  prompt_anatomy_belief_state_diff.csv     ABA - ABB diff per head for each (query, source)

Scope:
  - Top heads per condition with n_templates == 3 in top20_by_condition_end.csv
  - Query positions: lookback, belief_formation (agent_first / agent_last to be added after HPC re-run)
  - Source categories: state_first, state_second, state_correct, info_loss, char_q, obj_q, belief_form
    (omitting bare 'state' since it equals state_first + state_second + ... in some configs and is redundant)
  - Mask logic:
      * photo conditions (abstract_photo, answer_changes_photo, answer_changes_location_only_photo): char_q is undefined
      * state_only_no_belief: info_loss, belief_form are undefined
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prompt_anatomy")

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = ROOT / "other_results" / "lookback_analysis"
SUMMARY = DATA_ROOT / "_summary"
PATCH = "patch_before_movement"

QUERIES = ["lookback", "belief_formation", "agent_first", "agent_last"]
SOURCES = ["state_first", "state_second", "state_correct",
           "info_loss", "char_q", "obj_q", "belief_form"]

PHOTO_CONDS = {"abstract_photo", "answer_changes_photo",
               "answer_changes_location_only_photo"}
NO_BELIEF_CONDS = {"state_only_no_belief"}

# Conditions to include in the prompt-anatomy analysis. C3 (answer_changes_belief)
# and C4 (answer_changes_photo) are excluded: C3 is confounded (belief + answer)
# and C4 cells were captured with an older version of the position detector and
# have empty position columns. Five-condition focus per user request.
INCLUDE_CONDS = {"abstract_belief", "abstract_photo",
                 "answer_changes_location_only",
                 "answer_changes_location_only_photo",
                 "state_only_no_belief"}


def cell_dirs(condition: str):
    cond_root = DATA_ROOT / condition
    if not cond_root.is_dir():
        return
    for tpl_dir in sorted(cond_root.glob("template_*")):
        template = tpl_dir.name.replace("template_", "")
        for rule_dir in sorted(tpl_dir.glob("base_rule_*_exp_rule_*")):
            base_rule = rule_dir.name.split("_")[2]  # base_rule_ABA_exp_rule_ABB -> ABA
            cell = rule_dir / PATCH
            csv = cell / "lookback_scores.csv"
            if csv.is_file():
                yield template, base_rule, csv


def load_top_heads():
    df = pd.read_csv(SUMMARY / "top20_by_condition_end.csv")
    df = df[df["n_templates"] == 3].copy()
    df = df[df["condition"].isin(INCLUDE_CONDS)].copy()
    return df


def is_masked(condition: str, source: str) -> bool:
    if condition in PHOTO_CONDS and source == "char_q":
        return True
    if condition in NO_BELIEF_CONDS and source in {"info_loss", "belief_form"}:
        return True
    return False


def main():
    top_heads = load_top_heads()
    log.info(f"loaded {len(top_heads)} (condition, layer, head) triples with n_templates==3")

    long_rows = []
    for cond, sub in top_heads.groupby("condition"):
        head_set = {(int(r.layer), int(r.head)) for r in sub.itertuples()}
        n_cells = 0
        for template, base_rule, csv_path in cell_dirs(cond):
            n_cells += 1
            df = pd.read_csv(csv_path)
            mask = [(int(L), int(H)) in head_set for L, H in zip(df["layer"], df["head"])]
            df_match = df[mask]
            for _, r in df_match.iterrows():
                for q in QUERIES:
                    for s in SOURCES:
                        if is_masked(cond, s):
                            continue
                        col = f"{q}_attn_to_{s}"
                        if col not in df.columns:
                            continue
                        val = r.get(col, np.nan)
                        long_rows.append({
                            "condition": cond,
                            "layer": int(r["layer"]),
                            "head": int(r["head"]),
                            "template": template,
                            "base_rule": base_rule,
                            "query_pos": q,
                            "source_cat": s,
                            "attention": float(val) if pd.notna(val) else np.nan,
                        })
        log.info(f"  {cond}: {n_cells} cells")

    long = pd.DataFrame(long_rows)
    long_path = SUMMARY / "prompt_anatomy_long.csv"
    long.to_csv(long_path, index=False)
    log.info(f"wrote {long_path}  rows={len(long)}")

    if len(long) == 0:
        log.warning("no data — exiting")
        return

    # per-head profile: average across templates and base_rules
    profile = (long.groupby(["condition", "layer", "head", "query_pos", "source_cat"])
                    ["attention"].mean().reset_index())
    profile_wide = profile.pivot_table(
        index=["condition", "layer", "head"],
        columns=["query_pos", "source_cat"],
        values="attention",
    )
    profile_wide.columns = [f"{q}__{s}" for q, s in profile_wide.columns]
    profile_wide = profile_wide.reset_index()
    profile_path = SUMMARY / "prompt_anatomy_per_head_profile.csv"
    profile_wide.to_csv(profile_path, index=False)
    log.info(f"wrote {profile_path}  rows={len(profile_wide)}")

    # template consistency: std across templates (averaging over base_rules first)
    by_tpl = (long.groupby(["condition", "layer", "head", "template", "query_pos", "source_cat"])
                   ["attention"].mean().reset_index())
    consistency = (by_tpl.groupby(["condition", "layer", "head", "query_pos", "source_cat"])
                         ["attention"]
                         .agg(mean="mean", std="std", n_templates="count")
                         .reset_index())
    consistency["cv"] = consistency["std"] / consistency["mean"].where(consistency["mean"] > 0, np.nan)
    consistency_path = SUMMARY / "prompt_anatomy_template_consistency.csv"
    consistency.to_csv(consistency_path, index=False)
    log.info(f"wrote {consistency_path}  rows={len(consistency)}")

    # belief-state diff: ABA - ABB per (head, query, source); skip C7 (only one rule)
    by_rule = (long.groupby(["condition", "layer", "head", "base_rule", "query_pos", "source_cat"])
                    ["attention"].mean().reset_index())
    pivot = by_rule.pivot_table(
        index=["condition", "layer", "head", "query_pos", "source_cat"],
        columns="base_rule", values="attention",
    ).reset_index()
    if "ABA" in pivot.columns and "ABB" in pivot.columns:
        pivot["diff_ABA_minus_ABB"] = pivot["ABA"] - pivot["ABB"]
        pivot["abs_diff"] = pivot["diff_ABA_minus_ABB"].abs()
    diff_path = SUMMARY / "prompt_anatomy_belief_state_diff.csv"
    pivot.to_csv(diff_path, index=False)
    log.info(f"wrote {diff_path}  rows={len(pivot)}")


if __name__ == "__main__":
    main()
