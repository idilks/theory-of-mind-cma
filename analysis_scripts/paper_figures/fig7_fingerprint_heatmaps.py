"""Rebuttal figure 7 — CMA fingerprints, white-to-dark sequential map
(R3 points 6-7: "an uninvolved head is white and an implicated head dark";
prints legibly in greyscale).

Condition-averaged clamped fingerprints (negatives clamped to 0, vmax =
panel max, per the repo heatmap rule), Qwen top row and Gemma bottom row.
Three column layouts (critique.md pt 2, Sept 10):

  --variant c125    C1 belief | C2 photo | C5 answer retrieval   (original)
  --variant c135    C1 belief | C3 belief + answer | C5 answer retrieval
  --variant frames  belief (C1) | photo (C2) | Python code (code_abstract)
  --variant belief  belief divergence (C1) | belief answer (C5) | belief joint (C3)
  --variant all     build all four

Output: fig7_fingerprint_heatmaps{,_c135,_frames,_belief}.{png,pdf}.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2_additivity_robustness import (  # noqa: E402
    CROSS_RULES, WITHIN_RULES, GEMMA_SHAPE, QWEN_SHAPE, build_code_fp,
    build_gemma_fp, load_tom_fp)
from style import CMAP_SEQ, INK2, apply_style, save

CACHE = REPO / "results" / "figures" / "_cache_gemma_fps.npz"

# (panel title, qwen source, gemma folder, gemma rule set)
# qwen source: "fp:<name>" = results/aggregated/fingerprints npy,
#              "code:<folder>" = built from the indexed_list CMA cells
LAYOUTS = {
    "c125": ("", [
        ("C1 belief", "fp:C1_abstract_belief", "abstract_belief", CROSS_RULES),
        ("C2 photo", "fp:C2_abstract_photo", "abstract_photo", CROSS_RULES),
        ("C5 answer retrieval", "fp:C5_answer_changes_location_only",
         "answer_changes_location_only", WITHIN_RULES),
    ], "divergence band in the mid-upper layers (C1, C2); "
       "answer-retrieval band near the output (C5) — both models"),
    "c135": ("_c135", [
        ("C1 belief", "fp:C1_abstract_belief", "abstract_belief", CROSS_RULES),
        ("C3 belief + answer", "fp:C3_answer_changes_belief",
         "answer_changes_belief", CROSS_RULES),
        ("C5 answer retrieval", "fp:C5_answer_changes_location_only",
         "answer_changes_location_only", WITHIN_RULES),
    ], "C3 (belief and answer both vary) contains both the C1 divergence "
       "band and the C5 retrieval band — both models"),
    "frames": ("_frames", [
        ("divergence-change belief", "fp:C1_abstract_belief", "abstract_belief", CROSS_RULES),
        ("divergence-change photo", "fp:C2_abstract_photo", "abstract_photo", CROSS_RULES),
        ("divergence-change code", "code:code_abstract", "code_abstract",
         CROSS_RULES),
    ], "the same divergence band whether the initial representation is a "
       "belief, a photograph, or a `.copy()` — both models"),
    "belief": ("_belief", [
        ("divergence-change belief", "fp:C1_abstract_belief", "abstract_belief", CROSS_RULES),
        ("answer-change belief", "fp:C5_answer_changes_location_only",
         "answer_changes_location_only", WITHIN_RULES),
        ("dual-change belief", "fp:C3_answer_changes_belief",
         "answer_changes_belief", CROSS_RULES),
    ], "the joint condition carries the divergence band and the answer band"),
}


def gemma_fp(folder, rules):
    """Clamped condition-averaged Gemma fingerprint, cached per folder
    (cell loads take ~30 s)."""
    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    if folder not in cache:
        fp, _ = build_gemma_fp(folder, rules)
        cache[folder] = fp["clamped"]
        np.savez(CACHE, **cache)
    return cache[folder]


def qwen_fp(source):
    kind, name = source.split(":", 1)
    if kind == "fp":
        return load_tom_fp(name)["clamped"]
    fp, _ = build_code_fp(name, CROSS_RULES)
    return fp["clamped"]


def panel(ax, grid, title, vmax):
    im = ax.imshow(grid, cmap=CMAP_SEQ, vmin=0, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title(title, loc="left", fontsize=9)
    ax.tick_params(length=2, labelsize=7)
    return im


def build(variant, qwen_only=False):
    """Two rows (Qwen over Gemma) with a figure title, or --qwen-only: a
    single Qwen row, no figure title and no subtitle, for a paper slot where
    the caption carries the description."""
    suffix, cols, subtitle = LAYOUTS[variant]
    rows = (("Qwen", QWEN_SHAPE),) if qwen_only else (("Qwen", QWEN_SHAPE),
                                                      ("Gemma", GEMMA_SHAPE))
    # qwen-only is Fig. 3 of the paper; 2.6in tall keeps the heading and
    # wrapped scatter of the next subsection on the same page (draft2.tex guard).
    figsize = (9.4, 2.6) if qwen_only else (9.4, 2.8 * len(rows))
    fig, axes = plt.subplots(len(rows), 3, figsize=figsize, squeeze=False,
                             gridspec_kw={"hspace": 0.45,
                                          "wspace": 0.55 if qwen_only else 0.25})
    maxima = {}
    for j, (label, qsrc, gfolder, grules) in enumerate(cols):
        grids = {"Qwen": lambda: qwen_fp(qsrc),
                 "Gemma": lambda: gemma_fp(gfolder, grules)}
        for i, (model, shape) in enumerate(rows):
            grid = grids[model]().reshape(shape)
            vmax = grid.max()
            maxima[(model, label)] = float(vmax)
            # qwen-only: the model name is redundant, so the panel title is
            # just the condition.
            im = panel(axes[i, j], grid,
                       f"({chr(97 + j)}) {label}" if qwen_only
                       else f"{model}  {label}", vmax)
            cb = fig.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.03)
            cb.ax.tick_params(labelsize=6.5)
            cb.outline.set_visible(False)
            if qwen_only:
                cb.set_label("Causal mediation score", fontsize=7.5)

    for ax in axes[:, 0]:
        ax.set_ylabel("layer", fontsize=8.5)
    for ax in axes[-1, :]:
        ax.set_xlabel("head", fontsize=8.5)
    if qwen_only:
        suffix += "_qwen"
        fig.subplots_adjust(top=0.92)
    else:
        fig.text(0.5, 0.965,
                 "Condition-averaged causal fingerprints "
                 "(negatives clamped to 0, white = uninvolved, dark = implicated)",
                 ha="center", fontsize=10)
        fig.text(0.5, 0.935, subtitle, ha="center", fontsize=8.5, color=INK2)
        fig.subplots_adjust(top=0.87)
    save(fig, f"fig7_fingerprint_heatmaps{suffix}")
    for (model, label), v in maxima.items():
        print(f"  {variant:6s} {model:5s} {label:28s} vmax = {v:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*LAYOUTS, "all"], default="all")
    ap.add_argument("--qwen-only", action="store_true",
                    help="single Qwen row, no figure title or subtitle")
    args = ap.parse_args()
    apply_style()
    plt.rcParams["axes.grid"] = False
    for v in (LAYOUTS if args.variant == "all" else [args.variant]):
        build(v, qwen_only=args.qwen_only)


if __name__ == "__main__":
    main()
