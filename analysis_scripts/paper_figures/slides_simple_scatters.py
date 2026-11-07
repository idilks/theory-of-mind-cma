"""Presentation scatters: raw (unclamped) fingerprints, dots + OLS line only.

  1) C3 against C1 + C5
  2) C2 photo against C1 belief
  3) code_abstract against C1 belief

One dot per head (1,920). No annotations; r and the fit are printed to
stdout for the speaker notes. code_abstract is pinned to seed 0 so it
matches the published fingerprint (r = 0.867 with C1); the seed-1 rerun
from Sept 2026 would otherwise win the seed priority.

Outputs -> paper_figures/slides/scatter_*.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import BLUE, INK, apply_style, despine
from h2_additivity_robustness import RESULTS_ROOT, CROSS_RULES
from condition_registry import resolve_best_source

REPO = Path(__file__).resolve().parents[2]
FP = REPO / "results/aggregated/fingerprints"
OUT = REPO / "paper_figures/slides"
NAMES = {"C1": "C1_abstract_belief", "C2": "C2_abstract_photo",
         "C3": "C3_answer_changes_belief",
         "C5": "C5_answer_changes_location_only"}


def load(cond):
    return np.load(FP / f"{NAMES[cond]}_raw.npy").ravel()


def load_code_seed0():
    """Raw fingerprint = mean of per-cell scores (as aggregate_conditions)."""
    best = resolve_best_source(RESULTS_ROOT, ["code_abstract"], CROSS_RULES,
                               seed_priority={"0": 0})
    cells, prov = [], []
    for (tmpl, rule), e in sorted(best.items()):
        if tmpl != "template_indexed_list":
            continue
        cells.append(torch.load(e["scores_pt"], map_location="cpu",
                                weights_only=False).float().numpy().ravel())
        prov.append((rule, e["seed_tag"], e["n"]))
    assert len(cells) == 2, prov
    return np.mean(cells, axis=0), prov


def scatter(x, y, xlabel, ylabel, name):
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.scatter(x, y, s=10, color=BLUE, alpha=0.6, lw=0, zorder=3)
    slope, icpt = np.polyfit(x, y, 1)
    xs = np.array([x.min(), x.max()])
    ax.plot(xs, slope * xs + icpt, color=INK, lw=1.2, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    despine(ax)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    r = float(np.corrcoef(x, y)[0, 1])
    print(f"{name:28s} r = {r:.3f}   fit: y = {slope:.3f} x + {icpt:+.4f}")


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fp = {c: load(c) for c in NAMES}
    code_raw, prov = load_code_seed0()  # seed 0 pinned, see docstring
    print("code_abstract cells:", prov)

    scatter(fp["C1"] + fp["C5"], fp["C3"],
            "C1 + C5 causal score", "C3 causal score", "scatter_C1plusC5_vs_C3")
    scatter(fp["C1"], fp["C2"],
            "belief (C1) causal score", "photo (C2) causal score",
            "scatter_belief_vs_photo")
    scatter(fp["C1"], code_raw,
            "belief (C1) causal score", "code (copy vs alias) causal score",
            "scatter_belief_vs_code")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
