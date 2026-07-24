"""Build the full-1920-head ranking CSV for the Yang-style cumulative ablation.

PI-requested (July 2026, superseding the T_C1/T_C5-only version): cumulative
ablation over ALL heads in the model ordered by the C1 / C5 CMA fingerprints,
per Yang et al. 2025 §3.4 / B.5 ("the full range of h = 1...H heads in the
entire model"). Ranks by the RAW 6-cell mean fingerprint (seed=1, end
position, same averaging as aggregate_conditions.py) — raw rather than
clamped so the tail has a full ordering and the same-layer lowest-score
control is well-defined.

Output: results/aggregated/code_head_sets/tom_c1_c5_all_heads_ranked.csv
(consumed by ablation_experiment.py --head_source per_condition
 --rank_by mean_score --control_source ranking; results/aggregated syncs
 to the HPC via sync_to_hpc.sh.)
"""

import csv
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FP_DIR = REPO / "results/aggregated/fingerprints"
SOURCES = {
    "tom_c1": FP_DIR / "C1_abstract_belief_raw.npy",
    "tom_c5": FP_DIR / "C5_answer_changes_location_only_raw.npy",
}
OUT = REPO / "results/aggregated/code_head_sets/tom_c1_c5_all_heads_ranked.csv"
N_LAYERS, N_HEADS = 48, 40


def main():
    rows = []
    for cond, src in SOURCES.items():
        fp = np.load(src)
        assert fp.shape == (N_LAYERS * N_HEADS,), f"{src}: {fp.shape}"
        order = np.argsort(fp)[::-1]  # descending; fingerprint is layer-major
        for i in order:
            rows.append({"condition": cond, "layer": int(i // N_HEADS),
                         "head": int(i % N_HEADS),
                         "mean_score": float(fp[i])})
        top = order[0]
        print(f"{cond}: {len(order)} heads from {src.relative_to(REPO)} "
              f"(top: L{top // N_HEADS}H{top % N_HEADS} = {fp[top]:.4f}, "
              f"min = {fp[order[-1]]:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "layer", "head", "mean_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT.relative_to(REPO)} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
