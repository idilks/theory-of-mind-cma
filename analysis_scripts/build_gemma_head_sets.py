#!/usr/bin/env python
"""Build Gemma-2-27B head sets from its CMA statistical tests.

Scans the results/ tree, taking the max-sample_num stat CSV per
(condition, template, rule pair), seed 0 (the only Gemma seed).

Output: results/aggregated/code_head_sets/gemma_{c1,c2,c3,c5}_head_list.csv
        columns: layer, head, n_sig_cells, n_cells, mean_score
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEMMA_SUBPATH = "causal_analysis/cma/google/gemma-2-27b-it_ungroup_gqa"
ROOTS = [REPO / "results" / GEMMA_SUBPATH]
OUT_DIR = REPO / "results/aggregated/code_head_sets"

CONDITIONS = {
    "c1": "abstract_belief",
    "c2": "abstract_photo",
    "c3": "answer_changes_belief",
    "c5": "answer_changes_location_only",
    # code-domain arm: 2 cells per
    # condition (indexed_list x ABA/ABB), so the >=half rule becomes
    # "significant in >=1 of 2". Prints 0 cells harmlessly until the CMA
    # results land.
    "code_abstract": "code_abstract",
    "code_control": "code_control",
    "code_answer_changes": "code_answer_changes",
}

SAMPLE_RE = re.compile(r"sample_num_(\d+)_")


def best_cells(folder_name: str) -> dict[tuple, Path]:
    """(template, base_rule, exp_rule) -> stat CSV with the largest sample_num."""
    cells: dict[tuple, tuple[int, Path]] = {}
    for root in ROOTS:
        cond_dir = root / folder_name
        if not cond_dir.exists():
            continue
        for p in cond_dir.rglob("statistical_test_results.csv"):
            m = SAMPLE_RE.search(str(p))
            n = int(m.group(1)) if m else 0
            parts = p.parts
            template = next(x for x in parts if x.startswith("template_"))
            rules = next(x for x in parts if x.startswith("base_rule_"))
            key = (template, rules)
            if key not in cells or n > cells[key][0]:
                cells[key] = (n, p)
    return {k: v[1] for k, v in cells.items()}


def build_set(folder_name: str):
    cells = best_cells(folder_name)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    scores: dict[tuple[int, int], list] = defaultdict(list)
    for key, path in sorted(cells.items()):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["significant"].strip().lower() == "true":
                    hk = (int(row["layer"]), int(row["head"]))
                    counts[hk] += 1
                    scores[hk].append(float(row["observed_score"]))
    n_cells = len(cells)
    threshold = max(1, (n_cells + 1) // 2)  # >= half, and >=1 for tiny n
    heads = [{"layer": L, "head": H, "n_sig_cells": c, "n_cells": n_cells,
              "mean_score": sum(scores[(L, H)]) / len(scores[(L, H)])}
             for (L, H), c in counts.items() if c >= threshold]
    heads.sort(key=lambda r: -r["mean_score"])
    return heads, n_cells, threshold, cells


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tag, folder in CONDITIONS.items():
        heads, n_cells, thr, cells = build_set(folder)
        out = OUT_DIR / f"gemma_{tag}_head_list.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer", "head", "n_sig_cells",
                                              "n_cells", "mean_score"])
            w.writeheader()
            w.writerows(heads)
        top = f"L{heads[0]['layer']}H{heads[0]['head']}" if heads else "—"
        flag = "  [PARTIAL: <6 cells]" if n_cells < 6 else ""
        print(f"gemma_{tag} ({folder}): {n_cells} cells, sig in >={thr} -> "
              f"{len(heads)} heads (top {top}) -> {out.relative_to(REPO)}{flag}")
        for (tmpl, rules) in sorted(cells):
            print(f"    {tmpl:24s} {rules}")


if __name__ == "__main__":
    main()
