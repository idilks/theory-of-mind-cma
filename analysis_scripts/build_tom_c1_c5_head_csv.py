"""Build the per_condition head CSV for the follow-up C1/C5-on-code ablation.

Combines the published ToM head lists into one CSV consumable by
ablation_experiment.py --head_source per_condition --rank_by mean_score:
  tom_c1 = T_C1 belief heads   (results/ablation/belief_heads/head_list.csv, k=32)
  tom_c5 = T_C5 location heads (results/ablation/loc_belief_heads/head_list.csv, k=22)

Output: results/aggregated/code_head_sets/tom_c1_c5_heads.csv
"""

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCES = {
    "tom_c1": REPO / "results/ablation/belief_heads/head_list.csv",
    "tom_c5": REPO / "results/ablation/loc_belief_heads/head_list.csv",
}
OUT = REPO / "results/aggregated/code_head_sets/tom_c1_c5_heads.csv"


def main():
    rows = []
    for cond, src in SOURCES.items():
        with open(src) as f:
            heads = list(csv.DictReader(f))
        for r in heads:
            rows.append({"condition": cond, "layer": int(r["layer"]),
                         "head": int(r["head"]),
                         "mean_score": float(r["mean_score"])})
        print(f"{cond}: {len(heads)} heads from {src.relative_to(REPO)} "
              f"(top: L{heads[0]['layer']}H{heads[0]['head']})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "layer", "head", "mean_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
