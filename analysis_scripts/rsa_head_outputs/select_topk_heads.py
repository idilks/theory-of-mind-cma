#!/usr/bin/env python
"""Select top-k heads per condition by signed positive CMA fingerprint.

Reads results/aggregated/fingerprints/{cond}_clamped.npy (1920-vec),
ranks descending by signed score, writes the top-k as a CSV with
layer/head/score and the union across conditions as a single CSV
(used by capture_head_outputs.py to know which heads to hook).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FP_DIR = ROOT / "results/aggregated/fingerprints"
OUT_DIR = ROOT / "results/rsa_head_outputs/topk_heads"

N_LAYERS, N_HEADS = 48, 40

DEFAULT_CONDITIONS = [
    "C1_abstract_belief",
    "C5_answer_changes_location_only",
]


def topk(fp: np.ndarray, k: int) -> pd.DataFrame:
    assert fp.shape == (N_LAYERS * N_HEADS,), fp.shape
    order = np.argsort(-fp)[:k]
    layers = order // N_HEADS
    heads = order % N_HEADS
    return pd.DataFrame({"layer": layers, "head": heads, "score": fp[order]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for cond in args.conditions:
        path = FP_DIR / f"{cond}_clamped.npy"
        fp = np.load(path)
        df = topk(fp, args.k)
        df["condition"] = cond
        out = args.out_dir / f"{cond}_top{args.k}.csv"
        df.to_csv(out, index=False)
        print(f"{cond}: top-{args.k}, score range [{df.score.min():.3f}, {df.score.max():.3f}] -> {out}")
        all_rows.append(df)

    union = (pd.concat(all_rows, ignore_index=True)
               .sort_values(["layer", "head"])
               .drop_duplicates(["layer", "head"], keep="first")
               .reset_index(drop=True))
    union_out = args.out_dir / f"union_top{args.k}.csv"
    union.to_csv(union_out, index=False)
    print(f"union: {len(union)} unique heads -> {union_out}")


if __name__ == "__main__":
    main()
