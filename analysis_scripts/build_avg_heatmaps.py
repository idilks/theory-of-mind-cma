"""
Aggregated end-patch heatmaps per condition.

For each condition:
  1. Load `causal_scores_per_sample.pt` from each of the 6 cells
     (3 templates x 2 directions) at patch_before_movement, picking the same
     source (combined > seed=1 > seed=0) as build_pi_heatmaps.py.
  2. Per-cell mean -> average those means -> 1 vector of length n_heads.
     Negatives clamped to 0 (PI heatmap rule). Render with the paper's
     white-to-dark map (rebuttal_figures/style.py).
  3. Re-run the permutation test on the averaged statistic:
     for each permutation, draw one sign-flip per cell, compute that cell's
     permuted mean, then average the cell-means. Apply max-statistic FWER
     at alpha=0.05 to the resulting null distribution. Render binary mask.

Outputs:
    <out-dir>/<C#>_avg_scores.pdf
    <out-dir>/<C#>_avg_significance.pdf
    <out-dir>/manifest.csv   (n per cell, vmax, FWER threshold, significant heads)

Panels follow the main-text heatmap style (fig7_fingerprint_heatmaps.py) and are
untitled: draft2.tex names each panel in a subfigure caption. The manifest records
the paper's condition name per row; the appendix table reports its values.

Qwen (default, no args) reproduces the original build_avg_heatmaps behavior:
    python scripts/build_avg_heatmaps.py

Gemma-2-27B replication arm (subsumes the former build_avg_heatmaps_gemma.py;
tolerant of incomplete cell coverage -- Gemma C2 has 4/6 cells, C3 partial):
    python scripts/build_avg_heatmaps.py \
        --results-root results/causal_analysis/cma/google/gemma-2-27b-it_ungroup_gqa \
        --out-dir results_for_pi/heatmaps_gemma/aggregated \
        --conditions C1 C2 C3 \
        --tolerate-missing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import torch

from condition_registry import BY_CNUM, BY_KEY, REGISTRY, resolve_best_source

sys.path.insert(0, str(Path(__file__).resolve().parent / "rebuttal_figures"))
from style import BLUE, CMAP_SEQ, apply_style

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
DEFAULT_OUT = ROOT / "results_for_pi/heatmaps/aggregated"

N_PERMUTATIONS = 5000
ALPHA = 0.05
RNG_SEED = 0
EXPECT_CELLS = 6  # 3 templates x 2 directions
# Condition names used in draft2.tex, by registry key.
PAPER_NAMES = {
    "C1_abstract_belief": "divergence-change belief",
    "C2_abstract_photo": "divergence-change photo",
    "C3_answer_changes_belief": "dual-change belief",
    "C4_answer_changes_photo": "dual-change photo",
    "C5_answer_changes_location_only": "answer-change belief",
    "C6_answer_changes_location_only_photo": "answer-change photo",
    "C7_state_only_no_belief": "answer-change state-only",
}


def aggregate_condition(cells):
    """Returns (avg_scores [L,H], significance_mask [L,H], threshold, n_sig)."""
    per_cell_tensors = []
    for entry in cells:
        t = torch.load(entry["per_sample_pt"], map_location="cpu", weights_only=False).float()
        per_cell_tensors.append(t)  # [n_i, L, H]
    n_layers, n_heads = per_cell_tensors[0].shape[1], per_cell_tensors[0].shape[2]
    cell_means = torch.stack([t.mean(dim=0) for t in per_cell_tensors], dim=0)  # [n_cells, L, H]
    avg_scores = cell_means.mean(dim=0).numpy()  # [L, H]

    rng = np.random.default_rng(RNG_SEED)
    null_avg = np.zeros((N_PERMUTATIONS, n_layers, n_heads), dtype=np.float32)
    for t in per_cell_tensors:
        n_i = t.shape[0]
        signs = rng.integers(0, 2, size=(N_PERMUTATIONS, n_i)).astype(np.float32) * 2 - 1
        t_np = t.numpy()
        permuted_mean = np.einsum("pn,nlh->plh", signs, t_np) / n_i
        null_avg += permuted_mean
    null_avg /= len(per_cell_tensors)

    max_null = np.abs(null_avg).reshape(N_PERMUTATIONS, -1).max(axis=1)
    threshold = float(np.percentile(max_null, 100 * (1 - ALPHA)))
    sig_mask = (np.abs(avg_scores) >= threshold).astype(int)
    return avg_scores, sig_mask, threshold, int(sig_mask.sum())


def new_panel():
    """Same axes box for scores and masks, so the two columns line up on the page."""
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    fig.subplots_adjust(left=0.11, right=0.80, top=0.97, bottom=0.15)
    return fig, ax


def finish_panel(fig, ax, out_path):
    # No in-figure title: draft2.tex names each panel in its subfigure caption.
    grid = ax.images[0].get_array()
    ax.set_xlabel("head", fontsize=8.5)
    ax.set_ylabel("layer", fontsize=8.5)
    ax.set_xticks(np.arange(0, grid.shape[1], 10))
    ax.set_yticks(np.arange(0, grid.shape[0], 10))
    ax.tick_params(length=2, labelsize=7)
    fig.savefig(out_path)
    plt.close(fig)


def render_scores(scores, out_path):
    x = np.maximum(scores, 0.0)
    pos = x[x > 0]
    vmax = float(pos.max()) if pos.size else 1e-6
    fig, ax = new_panel()
    im = ax.imshow(x, cmap=CMAP_SEQ, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    cax = fig.add_axes([0.815, 0.15, 0.025, 0.82])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    cb.set_label("Causal mediation score", fontsize=7.5)
    finish_panel(fig, ax, out_path)
    return vmax


def render_significance(mask, out_path):
    fig, ax = new_panel()
    ax.imshow(mask, cmap=ListedColormap(["#ffffff", BLUE]), vmin=0, vmax=1, aspect="auto",
              interpolation="nearest")
    finish_panel(fig, ax, out_path)


def select_conditions(names):
    """Resolve a list of cnums/keys to Condition objects, preserving registry order."""
    if not names:
        return list(REGISTRY)
    chosen = []
    for n in names:
        if n in BY_CNUM:
            chosen.append(BY_CNUM[n])
        elif n in BY_KEY:
            chosen.append(BY_KEY[n])
        else:
            raise SystemExit(f"unknown condition '{n}' (expected a cnum like C1 or a key like C1_abstract_belief)")
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                    help="CMA results root (model dir). Default: Qwen tree.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="output directory for aggregated PNGs + manifest.csv")
    ap.add_argument("--conditions", nargs="*", default=None,
                    help="subset of conditions (cnums like C1 C2 C3, or full keys). Default: all 7.")
    ap.add_argument("--tolerate-missing", action="store_true",
                    help="aggregate whatever cells exist (>=1) instead of requiring all 6. "
                         "Also requires per-sample tensors to exist. Use for partial Gemma coverage.")
    args = ap.parse_args()

    apply_style()
    plt.rcParams["axes.grid"] = False
    plt.rcParams["savefig.bbox"] = "standard"  # keep the shared box; tight would crop each panel differently
    results_root = args.results_root if args.results_root.is_absolute() else ROOT / args.results_root
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    conditions = select_conditions(args.conditions)

    rows = []
    for cond in conditions:
        c_key = cond.key
        best = resolve_best_source(
            results_root, [cond.folder], cond.rule_pairs,
            require_per_sample=args.tolerate_missing,
        )
        if args.tolerate_missing:
            if not best:
                print(f"[{c_key}] no cells with per-sample data; skipping")
                continue
        else:
            if len(best) != EXPECT_CELLS:
                print(f"[{c_key}] expected {EXPECT_CELLS} cells, got {len(best)}: {sorted(best.keys())}")
                continue

        cells = [best[k] for k in sorted(best.keys())]
        cell_keys = [f"{t}/{r}" for (t, r) in sorted(best.keys())]
        n_per_cell = [c["n"] for c in cells]
        seeds = sorted({c["seed_tag"] for c in cells})
        if args.tolerate_missing:
            print(f"[{c_key}] aggregating {len(cells)}/{EXPECT_CELLS} cells, n={n_per_cell}, seeds={seeds}")
            for ck in cell_keys:
                print(f"    cell: {ck}")
        else:
            print(f"[{c_key}] aggregating {EXPECT_CELLS} cells, n={n_per_cell}, seeds={seeds}")

        avg_scores, mask, threshold, n_sig = aggregate_condition(cells)
        v_png = out_dir / f"{c_key}_avg_scores.pdf"
        s_png = out_dir / f"{c_key}_avg_significance.pdf"
        vmax = render_scores(avg_scores, v_png)
        render_significance(mask, s_png)
        row = {
            "condition": c_key,
            "paper_name": PAPER_NAMES[c_key],
            "n_cells": len(cells),
            "n_per_cell": n_per_cell,
            "seeds": seeds,
            "vmax": vmax,
            "fwer_threshold": threshold,
            "n_sig_avg": n_sig,
            "scores": str(v_png.relative_to(ROOT)),
            "significance": str(s_png.relative_to(ROOT)),
        }
        if args.tolerate_missing:
            row["cells"] = cell_keys
        rows.append(row)
        print(f"  -> n_sig={n_sig}, threshold={threshold:.4f}")
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"\nwrote {len(rows)} aggregated heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
