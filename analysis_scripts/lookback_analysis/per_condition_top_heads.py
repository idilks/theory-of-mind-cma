"""
Per-condition top-K head selection from permutation results.

For each of the 7 CMA conditions, finds the top-K heads (default 20) that are
both `significant=True` (FWER-corrected via the permutation test) and ranked by
signed positive `observed_score`. Aggregates as union across the 3 templates
within a condition (so a head in any template's top-20 is in the per-condition
top-K). Then computes overlaps between every condition pair so we can ask
questions like:

  - what's in abstract_belief AND abstract_photo?  (shared state-tracking)
  - what's in abstract_belief but NOT state_only_no_belief?
                                                   (belief-specific)
  - what's in abstract_photo but NOT abstract_belief?
                                                   (photo-specific)

Outputs:
  results/lookback_analysis/_summary/
    top20_by_condition.csv          long form: condition, layer, head, n_templates,
                                    templates_present, max_observed_score, ...
    condition_overlap_pairs.csv     pairwise: cond_a, cond_b, |A|, |B|, |A∩B|,
                                    |A−B|, |B−A|, jaccard
    condition_overlap_heads.csv     long form of the actual head sets per pair
                                    (so you can dig into shared vs unique heads)
"""
from __future__ import annotations

import argparse
import logging
import re
from itertools import combinations
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("per_condition_top_heads")


PATCH_FOLDER = {"end": "patch_before_movement", "belief": "patch_after_movement"}

DEFAULT_CONDITIONS = [
    "abstract_belief",
    "abstract_photo",
    "answer_changes_belief",
    "answer_changes_photo",
    "answer_changes_location_only",
    "answer_changes_location_only_photo",
    "state_only_no_belief",
]

DEFAULT_TEMPLATES = ["food_truck", "hair_styling", "library_book"]

DEFAULT_RULES = [
    ("ABA", "ABB"),  # cross-rule (used by abstract_*, answer_changes_belief/photo)
    ("ABB", "ABA"),
    ("ABA", "ABA"),  # same-rule (used by state_only_no_belief and
    ("ABB", "ABB"),  # answer_changes_location_only*); ignored by other conditions
]


def find_csv(results_root: Path, condition: str, template: str,
             base_rule: str, exp_rule: str, patch_folder: str) -> Path | None:
    base = (
        results_root
        / "causal_analysis" / "cma" / "Qwen" / "Qwen2.5-14B-Instruct"
        / condition / f"template_{template}"
        / f"base_rule_{base_rule}_exp_rule_{exp_rule}"
        / patch_folder
    )
    if not base.exists():
        return None
    candidates = list(base.glob("z_seed_*_shuffle_False/logit/sample_num_*_gen_acc_*/statistical_test_results.csv"))
    if not candidates:
        return None

    def sample_num(p: Path) -> int:
        m = re.search(r"sample_num_(\d+)_gen_acc", str(p))
        return int(m.group(1)) if m else 0

    return max(candidates, key=sample_num)


def top_k_for_cell(csv_path: Path, k: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["significant"] == True].copy()  # noqa: E712 — pandas idiom
    df = df[df["observed_score"] > 0]
    df = df.sort_values("observed_score", ascending=False).head(k)
    return df[["layer", "head", "observed_score", "p_value"]].reset_index(drop=True)


def collect_per_condition(results_root: Path, condition: str, templates: list[str],
                           rules: list[tuple[str, str]], patch_folder: str,
                           k: int) -> pd.DataFrame:
    """Union over templates × directions for one condition. One row per head."""
    by_head: dict[tuple[int, int], dict] = {}
    for tmpl in templates:
        for base_rule, exp_rule in rules:
            csv_path = find_csv(results_root, condition, tmpl,
                                base_rule, exp_rule, patch_folder)
            if csv_path is None:
                log.warning(f"missing CSV: {condition}/{tmpl}/{base_rule}->{exp_rule}/{patch_folder}")
                continue
            top = top_k_for_cell(csv_path, k)
            for _, row in top.iterrows():
                key = (int(row["layer"]), int(row["head"]))
                slot = by_head.setdefault(key, {
                    "layer": key[0], "head": key[1],
                    "templates_present": set(),
                    "rules_present": set(),
                    "scores": [],
                    "p_values": [],
                })
                slot["templates_present"].add(tmpl)
                slot["rules_present"].add(f"{base_rule}->{exp_rule}")
                slot["scores"].append(float(row["observed_score"]))
                slot["p_values"].append(float(row["p_value"]))

    rows = []
    for (L, H), s in by_head.items():
        rows.append({
            "condition": condition,
            "layer": L,
            "head": H,
            "n_templates": len(s["templates_present"]),
            "templates_present": "|".join(sorted(s["templates_present"])),
            "n_rules": len(s["rules_present"]),
            "rules_present": "|".join(sorted(s["rules_present"])),
            "max_observed_score": max(s["scores"]),
            "mean_observed_score": sum(s["scores"]) / len(s["scores"]),
            "min_p_value": min(s["p_values"]),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "condition", "layer", "head", "n_templates", "templates_present",
            "n_rules", "rules_present", "max_observed_score",
            "mean_observed_score", "min_p_value",
        ])
    df = pd.DataFrame(rows).sort_values(
        ["n_templates", "max_observed_score"], ascending=[False, False]
    )
    return df


def build_overlap_tables(per_cond: dict[str, set[tuple[int, int]]]):
    pair_rows = []
    head_rows = []
    for a, b in combinations(per_cond.keys(), 2):
        A, B = per_cond[a], per_cond[b]
        inter = A & B
        a_only = A - B
        b_only = B - A
        union = A | B
        pair_rows.append({
            "cond_a": a, "cond_b": b,
            "n_a": len(A), "n_b": len(B),
            "n_intersection": len(inter),
            "n_a_only": len(a_only),
            "n_b_only": len(b_only),
            "jaccard": (len(inter) / len(union)) if union else 0.0,
        })
        for L, H in sorted(inter):
            head_rows.append({"cond_a": a, "cond_b": b, "set": "intersection",
                              "layer": L, "head": H})
        for L, H in sorted(a_only):
            head_rows.append({"cond_a": a, "cond_b": b, "set": "a_only",
                              "layer": L, "head": H})
        for L, H in sorted(b_only):
            head_rows.append({"cond_a": a, "cond_b": b, "set": "b_only",
                              "layer": L, "head": H})
    return pd.DataFrame(pair_rows), pd.DataFrame(head_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=Path, default=Path("results"),
                    help="repo results/ root containing causal_analysis/")
    ap.add_argument("--out_root", type=Path, default=Path("results/lookback_analysis"),
                    help="where to write _summary/ outputs")
    ap.add_argument("--patch_position", default="end", choices=["end", "belief"],
                    help="end=patch_before_movement (default, lookback-relevant); "
                         "belief=patch_after_movement")
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    args = ap.parse_args()

    patch_folder = PATCH_FOLDER[args.patch_position]
    log.info(f"patch_position={args.patch_position} -> folder={patch_folder}")

    summary_dir = args.out_root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    per_cond_df_parts = []
    per_cond_sets: dict[str, set[tuple[int, int]]] = {}
    for cond in args.conditions:
        df = collect_per_condition(
            args.results_root, cond, args.templates, DEFAULT_RULES,
            patch_folder, args.top_k,
        )
        if df.empty:
            log.warning(f"no heads found for {cond}")
            continue
        per_cond_df_parts.append(df)
        per_cond_sets[cond] = set(zip(df["layer"].astype(int), df["head"].astype(int)))
        log.info(f"{cond}: {len(df)} unique heads (union across "
                 f"{len(args.templates)} templates × {len(DEFAULT_RULES)} rules)")

    if not per_cond_df_parts:
        log.error("no per-condition results; aborting")
        return

    long_df = pd.concat(per_cond_df_parts, ignore_index=True)
    out1 = summary_dir / f"top{args.top_k}_by_condition_{args.patch_position}.csv"
    long_df.to_csv(out1, index=False)
    log.info(f"wrote {out1}  ({len(long_df)} rows)")

    pair_df, head_df = build_overlap_tables(per_cond_sets)
    out2 = summary_dir / f"condition_overlap_pairs_{args.patch_position}.csv"
    out3 = summary_dir / f"condition_overlap_heads_{args.patch_position}.csv"
    pair_df.to_csv(out2, index=False)
    head_df.to_csv(out3, index=False)
    log.info(f"wrote {out2}  ({len(pair_df)} pairs)")
    log.info(f"wrote {out3}  ({len(head_df)} rows)")

    # Stdout: a quick-glance overlap table
    print()
    print(pair_df.sort_values("jaccard", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
