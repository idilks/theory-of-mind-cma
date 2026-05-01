"""
Render end-patch heatmaps (viridis) + binary significance maps for the PI report.

For every (condition, template, base_rule->exp_rule) at patch_before_movement,
pick the source folder with the most samples in the order: combined > seed_1 > seed_0.
Render two PNGs side-by-side input for the qmd:
  - <stem>_viridis.png        : neg-clamped, vmin=0, vmax=pos.max(), cmap=viridis
  - <stem>_significance.png   : binary FWER mask (alpha=0.05) from statistical_test_results.csv

Outputs go to results_for_pi/heatmaps/<C#>_<condition>/<template>/<rule_pair>/

Run from repo root:
    python scripts/build_pi_heatmaps.py
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
OUT = ROOT / "results_for_pi/heatmaps"

# canonical condition -> (label, alias folder names from legacy seed=0 runs, allowed rule pairs)
# C1, C2 are "same-answer" -> rule must flip (ABA->ABB / ABB->ABA).
# C3, C4, C5, C6 are "different-answer" -> rule kept (ABA->ABA / ABB->ABB).
# legacy photo_context (seed=0) used keep-rule by mistake -> filter those out for C2.
FLIP_RULES = {("ABA", "ABB"), ("ABB", "ABA")}
KEEP_RULES = {("ABA", "ABA"), ("ABB", "ABB")}
CONDITIONS = {
    "C1_abstract_belief": ("abstract_belief", ["abstract_belief", "abstract_context"], FLIP_RULES),
    "C2_abstract_photo": ("abstract_photo", ["abstract_photo", "photo_context"], FLIP_RULES),
    "C3_answer_changes_belief": ("answer_changes_belief", ["answer_changes_belief"], FLIP_RULES),
    "C4_answer_changes_photo": ("answer_changes_photo", ["answer_changes_photo"], KEEP_RULES),
    "C5_control": ("control", ["control", "control_context"], KEEP_RULES),
    "C6_state_only_no_belief": ("state_only_no_belief", ["state_only_no_belief"], KEEP_RULES),
}

SEED_PRIORITY = {"combined": 0, "1": 1, "0": 2}  # lower = preferred
SAMPLE_RE = re.compile(r"sample_num_(\d+)_(.+)")


RULE_PAIR_RE = re.compile(r"base_rule_(\w+)_exp_rule_(\w+)")


def find_best_source(condition_label: str, alias_folders: list[str], allowed_rule_pairs: set):
    """Return dict: (template, rule_pair) -> {scores_pt, csv, n, seed_tag, source_dir}"""
    best: dict[tuple[str, str], dict] = {}
    for alias in alias_folders:
        cond_root = RESULTS / alias
        if not cond_root.exists():
            continue
        for scores_pt in cond_root.rglob("causal_scores.pt"):
            parts = scores_pt.parts
            try:
                sample_dir = scores_pt.parent.name
                logit_dir = scores_pt.parent.parent.name
                seed_dir = scores_pt.parent.parent.parent.name
                patch_dir = scores_pt.parent.parent.parent.parent.name
                rule_dir = scores_pt.parent.parent.parent.parent.parent.name
                template_dir = scores_pt.parent.parent.parent.parent.parent.parent.name
            except IndexError:
                continue
            if logit_dir != "logit" or patch_dir != "patch_before_movement":
                continue
            if not seed_dir.startswith("z_seed_") or not seed_dir.endswith("_shuffle_False"):
                continue
            seed_tag = seed_dir[len("z_seed_"):-len("_shuffle_False")]
            if seed_tag not in SEED_PRIORITY:
                continue
            m = SAMPLE_RE.match(sample_dir)
            if not m:
                continue
            rp = RULE_PAIR_RE.match(rule_dir)
            if not rp or (rp.group(1), rp.group(2)) not in allowed_rule_pairs:
                continue
            n = int(m.group(1))
            key = (template_dir, rule_dir)
            entry = {
                "scores_pt": scores_pt,
                "per_sample_pt": scores_pt.parent / "causal_scores_per_sample.pt",
                "csv": scores_pt.parent / "statistical_test_results.csv",
                "n": n,
                "seed_tag": seed_tag,
                "priority": SEED_PRIORITY[seed_tag],
                "source_dir": scores_pt.parent,
            }
            cur = best.get(key)
            if cur is None or entry["priority"] < cur["priority"] or (
                entry["priority"] == cur["priority"] and entry["n"] > cur["n"]
            ):
                best[key] = entry
    return best


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
        best = find_best_source(cond_label, aliases, allowed_rules)
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
