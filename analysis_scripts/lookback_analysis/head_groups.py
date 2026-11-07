"""
Fixed head groups for the attention-region figures, shared by the capture
(capture_attention_patterns.py --head_conditions) and the plot
(region_bubble_matrix.py), so the false-belief and true-belief figures use
identical heads.

A story head belongs to a condition's group if it is CMA-significant in at least
`min_frac` of that condition's end-position cells (all templates, both rule
pairs). Seed 1 is preferred (CLAUDE.md: story end-position heads use seed 1);
a local `combined` C1 tree exists that the HPC lacks, and preferring it gave a
27-head C1 group locally vs 22 on the HPC.

Use a head's mean_score in THIS group as its weight. A head can be in two groups
with very different scores (L35H11: C5 +2.8, C1 -1.1).
Code groups are the published head lists in results/aggregated/code_head_sets/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))
from condition_registry import lookup, resolve_condition  # noqa: E402

STORY_SEED_PRIORITY = {"1": 0, "0": 1}  # lower = preferred

CMA_SUBDIR = Path("causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct")


def story_head_group(results_root: Path, condition: str, min_frac: float = 0.5,
                     seed_priority: dict = STORY_SEED_PRIORITY) -> pd.DataFrame:
    """Heads significant in >= min_frac of the condition's cells.

    results_root: the CMA results root ("results" locally, "codebase/results" on the HPC).
    Returns layer, head, n_sig, n_cells, mean_score (mean observed_score over all cells).
    """
    cells = resolve_condition(Path(results_root) / CMA_SUBDIR, lookup(condition),
                              seed_priority=seed_priority)
    if not cells:
        raise FileNotFoundError(f"no CMA cells for {condition} under {results_root}")
    frames = []
    for (template_dir, rule_dir), entry in sorted(cells.items()):
        df = pd.read_csv(entry["csv"])[["layer", "head", "observed_score", "significant"]]
        df["cell"] = f"{template_dir}/{rule_dir}"
        frames.append(df)
    d = pd.concat(frames)
    d["significant"] = d["significant"].astype(bool)
    g = d.groupby(["layer", "head"]).agg(n_sig=("significant", "sum"),
                                         mean_score=("observed_score", "mean")).reset_index()
    g["n_cells"] = len(cells)
    return (g[g.n_sig >= min_frac * len(cells)]
            .sort_values("mean_score", ascending=False).reset_index(drop=True))


def code_head_group(name: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / f"results/aggregated/code_head_sets/{name}_head_list.csv")
    return df[["layer", "head", "mean_score"]]


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results"
    for c in ("abstract_belief", "answer_changes_location_only",
              "abstract_photo", "answer_changes_location_only_photo"):
        g = story_head_group(root, c)
        print(f"{c:36s} {len(g):3d} heads (of {int(g.n_cells.iloc[0]) if len(g) else '?'} cells): "
              + ", ".join(f"L{a}H{b}" for a, b in zip(g.layer, g['head'])))
