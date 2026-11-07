#!/usr/bin/env python
"""Significance-count head selection, as an alternative to fingerprint top-k.

The Qwen head-output steering (paper Fig. 11) and the appendix story RDMs
(Fig. 12) select heads by ranking the clamped fingerprint, which is the mean
over a condition's six end-position cells (3 templates x 2 directions) of
max(0, cell mean CMA score). That ranking admits heads that are significant in
no cell at all: 7 of the C1 top-32 are FWER-significant in 0/6 cells.

This script selects instead by *how many cells* a head passes the per-cell
FWER permutation test in (`statistical_test_results.csv`, `significant`
column), and writes one CSV per condition and rule so that every consumer
reads the same list:

    results/aggregated/head_selection/{rule}/{condition}.csv
        layer, head, flat, n_sig_cells, n_cells, mean_clamped, fp_rank
    results/aggregated/head_selection/{rule}/sets_qwen.json
        div_unique = C1 \\ C5, div_shared = C1 & C5, photo_C2 = C2 \\ C6,
        plus the raw per-condition lists (flat indices, layer*40+head)
    results/aggregated/head_selection/comparison.csv
        set sizes and overlap with the fingerprint top-32/top-22 sets

Rules: sig4of6 (significant in >= 4 of 6 cells, the strict majority already
used by aggregate_conditions.py's "4of6" Jaccard mode) and sig3of6
(>= 3 of 6, sensitivity). Cells come from results/summary/heatmaps/manifest.csv
(end patch, canonical seeds), exactly as aggregate_conditions.py reads them.

Usage:
    python analysis_scripts/build_significance_head_sets.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results/summary/heatmaps/manifest.csv"
FP_DIR = ROOT / "results/aggregated/fingerprints"
OUT_DIR = ROOT / "results/aggregated/head_selection"

N_LAYERS, N_HEADS = 48, 40
NH = N_LAYERS * N_HEADS
CONDITIONS = ["C1_abstract_belief", "C2_abstract_photo",
              "C5_answer_changes_location_only",
              "C6_answer_changes_location_only_photo"]
RULES = {"sig4of6": 4, "sig3of6": 3}
# fingerprint top-k sizes used by the current paper pipeline
TOPK = {"C1_abstract_belief": 32, "C2_abstract_photo": 26,
        "C5_answer_changes_location_only": 22,
        "C6_answer_changes_location_only_photo": 13}
# Fig. 12 uses the belief sizes (32 / 22) for both frames
TOPK_FIG12 = {"C2_abstract_photo": 32, "C6_answer_changes_location_only_photo": 22}


def load_cells(manifest: pd.DataFrame, cond: str) -> tuple[np.ndarray, np.ndarray]:
    """(clamped_mean_1920, sig_counts_1920) over the condition's manifest cells."""
    rows = manifest[manifest["condition"] == cond]
    assert len(rows) == 6, f"{cond}: expected 6 cells, manifest has {len(rows)}"
    raw, sig = [], []
    for _, r in rows.iterrows():
        src = ROOT / r["source"]
        s = torch.load(src / "causal_scores.pt", map_location="cpu",
                       weights_only=False).float().cpu().numpy().reshape(-1)
        df = pd.read_csv(src / "statistical_test_results.csv")
        df = df.sort_values(["layer", "head"]).reset_index(drop=True)
        m = df["significant"].astype(bool).to_numpy()
        assert s.shape == (NH,) and m.shape == (NH,), (cond, s.shape, m.shape)
        raw.append(s)
        sig.append(m)
    raw, sig = np.stack(raw), np.stack(sig)
    return np.clip(raw, 0, None).mean(0), sig.sum(0)


def select(sig_counts: np.ndarray, min_cells: int) -> np.ndarray:
    """Flat indices of heads significant in at least `min_cells` cells, sorted."""
    return np.flatnonzero(sig_counts >= min_cells)


def derived_sets(by_cond: dict[str, set[int]]) -> dict[str, list[int]]:
    c1, c2 = by_cond["C1_abstract_belief"], by_cond["C2_abstract_photo"]
    c5 = by_cond["C5_answer_changes_location_only"]
    c6 = by_cond["C6_answer_changes_location_only_photo"]
    return {"div_unique": sorted(c1 - c5), "div_shared": sorted(c1 & c5),
            "photo_C2": sorted(c2 - c6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    manifest = pd.read_csv(MANIFEST)

    fp, cnt = {}, {}
    for cond in CONDITIONS:
        fp[cond], cnt[cond] = load_cells(manifest, cond)
        saved = np.load(FP_DIR / f"{cond}_clamped.npy")
        assert np.allclose(fp[cond], saved), f"{cond}: recomputed fingerprint != saved"
    print("per-cell FWER test counts loaded; fingerprints match saved .npy")

    comparison = []
    for rule, min_cells in RULES.items():
        rdir = args.out_dir / rule
        rdir.mkdir(parents=True, exist_ok=True)
        by_cond = {}
        for cond in CONDITIONS:
            flat = select(cnt[cond], min_cells)
            rank = np.empty(NH, dtype=int)
            rank[np.argsort(-fp[cond])] = np.arange(1, NH + 1)
            df = pd.DataFrame({"layer": flat // N_HEADS, "head": flat % N_HEADS,
                               "flat": flat, "n_sig_cells": cnt[cond][flat],
                               "n_cells": 6, "mean_clamped": fp[cond][flat],
                               "fp_rank": rank[flat]})
            df = df.sort_values(["n_sig_cells", "mean_clamped"],
                                ascending=[False, False]).reset_index(drop=True)
            df.to_csv(rdir / f"{cond}.csv", index=False)
            by_cond[cond] = set(flat.tolist())
            top = set(np.argsort(fp[cond])[-TOPK[cond]:].tolist())
            comparison.append({"rule": rule, "set": cond, "n": len(flat),
                               "n_topk": TOPK[cond], "overlap_topk": len(by_cond[cond] & top)})
            print(f"  {rule} {cond}: {len(flat)} heads "
                  f"(top-{TOPK[cond]} overlap {len(by_cond[cond] & top)})")
        sets = derived_sets(by_cond)
        fp1 = np.argsort(fp["C1_abstract_belief"])
        fp5 = np.argsort(fp["C5_answer_changes_location_only"])
        old_unique = set(fp1[-32:].tolist()) - set(fp5[-22:].tolist())
        comparison.append({"rule": rule, "set": "div_unique", "n": len(sets["div_unique"]),
                           "n_topk": len(old_unique),
                           "overlap_topk": len(set(sets["div_unique"]) & old_unique)})
        comparison.append({"rule": rule, "set": "div_shared", "n": len(sets["div_shared"]),
                           "n_topk": 10, "overlap_topk": np.nan})
        out = {"rule": rule, "min_sig_cells": min_cells, "n_cells": 6,
               "model": "Qwen2.5-14B-Instruct", "flat_index": "layer*40+head",
               "conditions": {c: sorted(v) for c, v in by_cond.items()}, **sets,
               "layer_head": {k: [[h // N_HEADS, h % N_HEADS] for h in v]
                              for k, v in sets.items()}}
        (rdir / "sets_qwen.json").write_text(json.dumps(out, indent=1))
        print(f"  {rule}: div_unique={len(sets['div_unique'])} "
              f"div_shared={len(sets['div_shared'])} photo_C2={len(sets['photo_C2'])}")

    comp = pd.DataFrame(comparison)
    comp.to_csv(args.out_dir / "comparison.csv", index=False)
    print(comp.to_string(index=False))
    print(f"wrote -> {args.out_dir}")


if __name__ == "__main__":
    main()
