"""
Render end-patch heatmaps + binary significance maps for the summary report.

For every (condition, template, base_rule->exp_rule) at patch_before_movement,
pick the source folder with the most samples in the order: combined > seed_1 > seed_0.
Render two PNGs side-by-side input for the qmd:
  - <stem>_viridis.png        : neg-clamped, vmin=0, vmax=pos.max(), cmap=viridis
  - <stem>_significance.png   : binary FWER mask (alpha=0.05) from statistical_test_results.csv

Outputs go to results/summary/heatmaps/<C#>_<condition>/<template>/<rule_pair>/

Run from repo root:
    python analysis_scripts/build_summary_heatmaps.py
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from condition_registry import heatmap_condition_table, resolve_best_source

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
OUT = ROOT / "results/summary/heatmaps"

# canonical condition -> (label, alias folder names, allowed rule pairs).
# C1-C4 are across-context -> rule flips (ABA->ABB / ABB->ABA); C5-C7 are
# within-context -> rule kept (ABA->ABA / ABB->ABB). See condition_registry.
# Legacy aliases (abstract_context, photo_context, control_context) are
# duplicates / content-invalid per the project conventions in README.md and intentionally excluded.
CONDITIONS = heatmap_condition_table()


def render_viridis(scores_pt: Path, out_png: Path, title: str):
    x = torch.load(scores_pt, map_location="cpu", weights_only=False).numpy()
    if x.ndim == 1:
        x = x[:, None]
    x = np.maximum(x, 0.0)
    pos = x[x > 0]
    vmax = float(pos.max()) if pos.size else 1.0
    if vmax <= 0:
        vmax = 1e-6
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(x, cmap="viridis", vmin=0, vmax=vmax, cbar=True, ax=ax, square=False)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"{title}\n(neg→0, vmax={vmax:.3f})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return vmax


def render_significance(csv_path: Path, out_png: Path, title: str):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    n_layers = int(df["layer"].max()) + 1
    n_heads = int(df["head"].max()) + 1
    mask = np.zeros((n_layers, n_heads), dtype=int)
    for _, r in df.iterrows():
        if bool(r["significant"]):
            mask[int(r["layer"]), int(r["head"])] = 1
    n_sig = int(mask.sum())
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(mask, cmap=["#f5f5f5", "#1f77b4"], vmin=0, vmax=1, cbar=False, ax=ax, square=False)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"{title}\nFWER-significant heads (α=0.05): {n_sig}/{mask.size}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return n_sig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for c_key, (cond_label, aliases, allowed_rules) in CONDITIONS.items():
        best = resolve_best_source(RESULTS, aliases, allowed_rules)
        if not best:
            print(f"[{c_key}] no end-patch sources found")
            continue
        for (template, rule_pair), entry in sorted(best.items()):
            out_dir = OUT / c_key / template / rule_pair
            out_dir.mkdir(parents=True, exist_ok=True)
            base, exp = re.match(r"base_rule_(\w+)_exp_rule_(\w+)", rule_pair).groups()
            title = f"{c_key} · {template} · {base}→{exp}\nn={entry['n']} (seed={entry['seed_tag']})"
            v_png = out_dir / "viridis.png"
            s_png = out_dir / "significance.png"
            render_viridis(entry["scores_pt"], v_png, title)
            n_sig = render_significance(entry["csv"], s_png, title)
            manifest_rows.append({
                "condition": c_key,
                "template": template,
                "rule_pair": rule_pair,
                "n": entry["n"],
                "seed": entry["seed_tag"],
                "n_sig": n_sig if n_sig is not None else "",
                "source": str(entry["source_dir"].relative_to(ROOT)),
                "viridis": str(v_png.relative_to(ROOT)),
                "significance": str(s_png.relative_to(ROOT)),
            })
            print(f"  {c_key}/{template}/{rule_pair} n={entry['n']} seed={entry['seed_tag']} sig={n_sig}")

    manifest_path = OUT / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\nwrote {len(manifest_rows)} cells, manifest at {manifest_path}")


if __name__ == "__main__":
    main()
