#!/usr/bin/env python3
"""Cell-level RSA + MDS over the 6 conditions (end-patch only).

Each cell = (condition, template, direction). 6 conds x 3 templates x 2 dirs
= 36 cells. Each cell's 1920-vec is its "stimulus representation".

Two views:
- 18-point MDS: directions averaged within (condition, template) -> 6x3 = 18.
  Cleaner for visual inspection of condition vs template structure.
- 6x6 condition-level RDM: directions and templates averaged -> 6 vectors.
  Mantel-tested against three models:
    * same-condition (categorical condition identity)
    * across-context vs within-context (C1/C2/C3/C4 = false_template <->
      true_template patch; C5/C6/C7 = within-context, no template flip)
    * task type (abstract / answer-changes / location)

Inputs: results_for_pi/heatmaps/manifest.csv (canonical end-patch cell map).
Outputs:
  results_for_pi/aggregation/rsa_cell_mds.png
  results_for_pi/aggregation/rsa_cell_rdm.png  (6x6 condition RDM)
  results/aggregated/cell_rsa/cell_rdm_18x18.csv
  results/aggregated/cell_rsa/condition_rdm_6x6.csv
  results/aggregated/cell_rsa/mantel_results.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.manifold import MDS

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results_for_pi/heatmaps/manifest.csv"
OUT_AGG = ROOT / "results/aggregated/cell_rsa"
OUT_FIG = ROOT / "results_for_pi/aggregation"

CONDITIONS = [
    "C1_abstract_belief", "C2_abstract_photo", "C3_answer_changes_belief",
    "C4_answer_changes_photo", "C5_answer_changes_location_only",
    "C6_answer_changes_location_only_photo", "C7_state_only_no_belief",
]
SHORT = {c: c.split("_", 1)[0] for c in CONDITIONS}
TEMPLATES = ["template_food_truck", "template_hair_styling", "template_library_book"]
TMPL_SHORT = {t: t.replace("template_", "") for t in TEMPLATES}

COND_COLORS = {
    "C1_abstract_belief": "#2166ac",
    "C2_abstract_photo": "#67a9cf",
    "C3_answer_changes_belief": "#d6604d",
    "C4_answer_changes_photo": "#f4a582",
    "C5_answer_changes_location_only": "#4daf4a",
    "C6_answer_changes_location_only_photo": "#a6d854",
    "C7_state_only_no_belief": "#984ea3",
}
TMPL_MARKERS = {"template_food_truck": "o", "template_hair_styling": "s", "template_library_book": "^"}


def load_cell_vectors(manifest: pd.DataFrame) -> dict:
    """Return dict[(cond, template, rule_pair)] -> 1920-vec."""
    out = {}
    for _, r in manifest.iterrows():
        src = ROOT / r["source"]
        v = torch.load(src / "causal_scores.pt", map_location="cpu", weights_only=False)
        v = v.float().cpu().numpy().reshape(-1)
        out[(r["condition"], r["template"], r["rule_pair"])] = v
    return out


def average_over_directions(cells: dict) -> dict:
    """(cond, tmpl, rule_pair) -> (cond, tmpl) by mean over rule_pair."""
    out = {}
    for c in CONDITIONS:
        for t in TEMPLATES:
            keys = [k for k in cells if k[0] == c and k[1] == t]
            if not keys:
                continue
            out[(c, t)] = np.mean([cells[k] for k in keys], axis=0)
    return out


def average_over_template_and_dir(cells: dict) -> dict:
    """{(cond, tmpl, rule_pair) -> vec} -> {cond -> vec}."""
    out = {}
    for c in CONDITIONS:
        keys = [k for k in cells if k[0] == c]
        out[c] = np.mean([cells[k] for k in keys], axis=0)
    return out


def pairwise_pearson_rdm(vectors: list[np.ndarray]) -> np.ndarray:
    n = len(vectors)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            r, _ = pearsonr(vectors[i], vectors[j])
            M[i, j] = M[j, i] = 1 - r
    return M


def mantel_test(diss: np.ndarray, model_groups: list[int], n_perm: int = 5000,
                seed: int = 42) -> tuple[float, float]:
    """Mantel: is diss correlated with model RDM (0 if same group, 1 else)?"""
    n = len(model_groups)
    g = np.asarray(model_groups)
    model = (g[:, None] != g[None, :]).astype(float)
    upper = np.triu_indices(n, k=1)
    obs_r, _ = pearsonr(diss[upper], model[upper])
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(n)
        null[k] = pearsonr(diss[upper], model[np.ix_(perm, perm)][upper])[0]
    p = float(np.mean(null >= obs_r))
    return float(obs_r), p


def plot_cell_mds(coords: np.ndarray, labels: list[tuple[str, str]], stress: float, out: Path):
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (c, t) in enumerate(labels):
        ax.scatter(coords[i, 0], coords[i, 1], c=COND_COLORS[c], marker=TMPL_MARKERS[t],
                   s=170, edgecolors="white", linewidth=1, zorder=3)
        ax.annotate(f"{SHORT[c]}/{TMPL_SHORT[t][:3]}",
                    (coords[i, 0], coords[i, 1]), fontsize=7, ha="center", va="center")
    color_legend = [mpatches.Patch(color=COND_COLORS[c], label=SHORT[c], alpha=0.85) for c in CONDITIONS]
    shape_legend = [Line2D([0], [0], marker=TMPL_MARKERS[t], color="gray", linestyle="None",
                           markersize=9, label=TMPL_SHORT[t]) for t in TEMPLATES]
    leg1 = ax.legend(handles=color_legend, title="Condition", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=shape_legend, title="Template", loc="lower left", fontsize=8)
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.set_title(f"Cell-level RSA — 18 points (6 conditions × 3 templates), end-patch\nstress = {stress:.4f}")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    plt.close()


def plot_condition_rdm(rdm: np.ndarray, mantel: dict, out: Path):
    cond_short = [SHORT[c] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rdm, cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(cond_short)))
    ax.set_xticklabels(cond_short)
    ax.set_yticks(range(len(cond_short)))
    ax.set_yticklabels(cond_short)
    for i in range(rdm.shape[0]):
        for j in range(rdm.shape[1]):
            ax.text(j, i, f"{rdm[i, j]:.2f}", ha="center", va="center",
                    color="white" if rdm[i, j] > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="1 - Pearson r")
    mantel_line = " | ".join(f"{name}: r={r:+.2f} p={p:.3f}" for name, (r, p) in mantel.items())
    ax.set_title(f"6×6 condition RDM (template+direction averaged)\nMantel: {mantel_line}", fontsize=10)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=160)
    plt.close()


def main():
    OUT_AGG.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print(f"reading manifest: {MANIFEST}")
    manifest = pd.read_csv(MANIFEST)
    cells = load_cell_vectors(manifest)
    print(f"  {len(cells)} cells loaded")

    cells_18 = average_over_directions(cells)
    print(f"  18 (cond, template) cells after direction averaging: {len(cells_18)}")

    labels_18 = [(c, t) for c in CONDITIONS for t in TEMPLATES if (c, t) in cells_18]
    vectors_18 = [cells_18[k] for k in labels_18]
    rdm_18 = pairwise_pearson_rdm(vectors_18)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto")
    coords_18 = mds.fit_transform(rdm_18)
    print(f"  MDS stress: {mds.stress_:.4f}")

    pd.DataFrame(rdm_18, index=[f"{SHORT[c]}_{TMPL_SHORT[t]}" for c, t in labels_18],
                 columns=[f"{SHORT[c]}_{TMPL_SHORT[t]}" for c, t in labels_18]).to_csv(
        OUT_AGG / "cell_rdm_18x18.csv")
    plot_cell_mds(coords_18, labels_18, mds.stress_, OUT_FIG / "rsa_cell_mds.png")

    cond_vecs = average_over_template_and_dir(cells)
    rdm_6 = pairwise_pearson_rdm([cond_vecs[c] for c in CONDITIONS])
    pd.DataFrame(rdm_6, index=[SHORT[c] for c in CONDITIONS],
                 columns=[SHORT[c] for c in CONDITIONS]).to_csv(OUT_AGG / "condition_rdm_6x6.csv")

    # Mantel models on the 18x18 (full structure where labels exist)
    cond_groups = [CONDITIONS.index(c) for c, _ in labels_18]
    context_groups = [0 if c in ("C1_abstract_belief", "C2_abstract_photo",
                                 "C3_answer_changes_belief", "C4_answer_changes_photo")
                      else 1 for c, _ in labels_18]
    task_groups = []
    for c, _ in labels_18:
        if c in ("C1_abstract_belief", "C2_abstract_photo"):
            task_groups.append(0)        # abstract
        elif c in ("C3_answer_changes_belief", "C4_answer_changes_photo"):
            task_groups.append(1)        # answer-changes
        else:
            task_groups.append(2)        # location
    template_groups = [TEMPLATES.index(t) for _, t in labels_18]

    mantel = {}
    mantel["same-cond"] = mantel_test(rdm_18, cond_groups)
    mantel["across-context"] = mantel_test(rdm_18, context_groups)
    mantel["task-type"] = mantel_test(rdm_18, task_groups)
    mantel["template"] = mantel_test(rdm_18, template_groups)

    print("\nMantel tests on 18x18 cell RDM:")
    rows = []
    for name, (r, p) in mantel.items():
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:12s} r={r:+.4f}  p={p:.4f}  {stars}")
        rows.append({"model": name, "r": r, "p": p})
    pd.DataFrame(rows).to_csv(OUT_AGG / "mantel_results.csv", index=False)

    plot_condition_rdm(rdm_6, mantel, OUT_FIG / "rsa_cell_rdm.png")
    print(f"\noutputs:\n  tables -> {OUT_AGG}\n  figures -> {OUT_FIG}")


if __name__ == "__main__":
    main()
