"""Plot DLA results from compute_dla.py.

Supports two framings via --context_type:

  abstract (default) — belief-frame DLA. Reads from results/rsa_head_outputs/dla/
                       and uses C1 (abstract_belief) end-position CMA on the
                       scatter x-axis.
  photo              — photo-frame DLA. Reads from results/rsa_head_outputs/dla_photo/
                       and uses C2 (abstract_photo) end-position CMA on the
                       scatter x-axis. Outputs land alongside the inputs.

Produces three figures per template + one cross-template summary:

  {dla_dir}/{template}_dla_heatmap.png
    (layer x head) heatmap of mean logit_diff, split FB / TB / overall.

  {dla_dir}/{template}_dla_vs_cma.png
    Scatter of CMA observed_score vs DLA mean |logit_diff|. Tests "do high-CMA
    heads have high DLA?" High-CMA + high-|DLA| = direct OV effect;
    high-CMA + low-|DLA| = downstream-mediated (routing).

  {dla_dir}/{template}_dla_top20.csv
    Top-20 heads by |mean logit_diff|.

  {dla_dir}/dla_summary_grid.png
    3-template heatmap grid for fast comparison.

Usage (local):
  python scripts/rsa_head_outputs/plot_dla.py                       # belief
  python scripts/rsa_head_outputs/plot_dla.py --context_type photo  # photo
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DLA_DIR_ABSTRACT = ROOT / "results/rsa_head_outputs/dla"
DEFAULT_DLA_DIR_PHOTO = ROOT / "results/rsa_head_outputs/dla_photo"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]

# CMA conditions to pair with each DLA framing on the scatter x-axis.
# Both use the canonical end-position seed=1 run (per CLAUDE.md memory).
CMA_CONDITION_FOR = {
    "abstract": "abstract_belief",
    "photo": "abstract_photo",
}


def load_cma_scores(template: str, context_type: str,
                    seed: int | None = None) -> pd.DataFrame | None:
    """Load end-position CMA observed_scores for the condition matching context_type.

    abstract -> abstract_belief (C1).  photo -> abstract_photo (C2).
    If seed is given, only that seed dir is searched; otherwise picks the
    statistical_test_results.csv with the largest sample_num.
    """
    condition = CMA_CONDITION_FOR[context_type]
    base = (ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
            / condition / f"template_{template}"
            / "base_rule_ABA_exp_rule_ABB" / "patch_before_movement")
    if not base.exists():
        return None

    if seed is not None:
        seed_glob = f"z_seed_{seed}_shuffle_False"
    else:
        seed_glob = "z_seed_*_shuffle_False"
    candidates = list(base.glob(f"{seed_glob}/logit/sample_num_*_gen_acc_*/statistical_test_results.csv"))
    if not candidates:
        return None

    def sample_num(p: Path) -> int:
        m = re.search(r"sample_num_(\d+)_gen_acc", str(p))
        return int(m.group(1)) if m else 0

    csv_path = max(candidates, key=sample_num)
    df = pd.read_csv(csv_path)
    return df[["layer", "head", "observed_score", "significant"]].copy()


def per_head_summary(dla_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse stim dim: per (layer, head), mean logit_diff overall + FB / TB."""
    g_all = dla_df.groupby(["layer", "head"], as_index=False)["logit_diff"].mean()
    g_all = g_all.rename(columns={"logit_diff": "mean_logit_diff"})

    fb = dla_df[dla_df["belief_veracity"].astype(str).str.lower().isin(["false", "fb"])]
    tb = dla_df[dla_df["belief_veracity"].astype(str).str.lower().isin(["true", "tb"])]

    g_fb = fb.groupby(["layer", "head"], as_index=False)["logit_diff"].mean().rename(
        columns={"logit_diff": "mean_logit_diff_fb"}
    )
    g_tb = tb.groupby(["layer", "head"], as_index=False)["logit_diff"].mean().rename(
        columns={"logit_diff": "mean_logit_diff_tb"}
    )

    out = g_all.merge(g_fb, on=["layer", "head"], how="left").merge(g_tb, on=["layer", "head"], how="left")
    return out


def heatmap_matrix(summary: pd.DataFrame, value_col: str) -> np.ndarray:
    n_layers = int(summary["layer"].max()) + 1
    n_heads = int(summary["head"].max()) + 1
    mat = np.full((n_layers, n_heads), np.nan)
    for r in summary.itertuples():
        mat[int(r.layer), int(r.head)] = getattr(r, value_col)
    return mat


def plot_template_heatmap(summary: pd.DataFrame, template: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    cols = [("mean_logit_diff", "overall"),
            ("mean_logit_diff_fb", "FB only"),
            ("mean_logit_diff_tb", "TB only")]
    all_mats = [heatmap_matrix(summary, c) for c, _ in cols]
    vmax = float(max(np.nanmax(np.abs(m)) for m in all_mats))
    vmin = -vmax
    for ax, (col, label), mat in zip(axes, cols, all_mats):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head")
        ax.set_title(f"{label}\n(mean over stim)")
        ax.set_ylabel("layer")
    fig.suptitle(f"DLA logit-diff (answer - distractor)  |  template = {template}", fontsize=12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="mean logit_diff")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_summary_grid(summaries: dict[str, pd.DataFrame], out_path: Path):
    fig, axes = plt.subplots(1, len(TEMPLATES), figsize=(18, 7), sharey=True)
    mats = [heatmap_matrix(summaries[t], "mean_logit_diff") for t in TEMPLATES]
    vmax = float(max(np.nanmax(np.abs(m)) for m in mats))
    vmin = -vmax
    last_im = None
    for ax, tmpl, mat in zip(axes, TEMPLATES, mats):
        last_im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_xlabel("head")
        ax.set_title(tmpl)
        ax.set_ylabel("layer")
    fig.suptitle("DLA logit-diff (overall, mean over stim) — per template", fontsize=12)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.7, label="mean logit_diff")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_dla_vs_cma(summary: pd.DataFrame, cma: pd.DataFrame, template: str,
                    context_type: str, out_path: Path):
    merged = summary.merge(cma, on=["layer", "head"], how="inner")
    if merged.empty:
        print(f"skip {template}: no CMA-DLA overlap")
        return
    merged["abs_logit_diff"] = merged["mean_logit_diff"].abs()
    fig, ax = plt.subplots(figsize=(8, 7))
    sig = merged[merged["significant"] == True]  # noqa: E712
    nonsig = merged[merged["significant"] != True]
    ax.scatter(nonsig["observed_score"], nonsig["abs_logit_diff"], s=10, c="lightgray",
               label=f"non-sig (n={len(nonsig)})")
    ax.scatter(sig["observed_score"], sig["abs_logit_diff"], s=20, c="crimson",
               label=f"CMA-sig (n={len(sig)})")
    # Label the top-10 most-extreme heads
    top = merged.nlargest(10, "abs_logit_diff")
    for r in top.itertuples():
        ax.annotate(f"L{int(r.layer)}H{int(r.head)}",
                    (r.observed_score, r.abs_logit_diff),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    cma_cond = CMA_CONDITION_FOR[context_type]
    frame_label = "belief-frame stim" if context_type == "abstract" else "photo-frame stim"
    ax.set_xlabel(f"CMA observed_score ({cma_cond}, end-position)")
    ax.set_ylabel(f"|mean DLA logit_diff|  ({frame_label}, this template)")
    ax.set_title(f"CMA vs DLA  |  template = {template}  |  frame = {context_type}\n"
                 "high-CMA + high-|DLA| = direct OV effect; high-CMA + low-|DLA| = downstream-mediated")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context_type", choices=["abstract", "photo"], default="abstract",
                    help="abstract = belief-frame DLA paired with C1 CMA;"
                         " photo = photo-frame DLA paired with C2 CMA")
    ap.add_argument("--dla_dir", type=Path, default=None,
                    help="overrides the context_type default location")
    ap.add_argument("--cma_seed", type=int, default=None,
                    help="if set, only load CMA results from z_seed_<N>_shuffle_False/."
                         " photo defaults to seed=1 (canonical end-position run).")
    args = ap.parse_args()

    if args.dla_dir is None:
        args.dla_dir = (DEFAULT_DLA_DIR_ABSTRACT if args.context_type == "abstract"
                        else DEFAULT_DLA_DIR_PHOTO)
    # photo: lock to seed=1 unless explicitly overridden, since that's the only
    # seed with full coverage for the 7-condition end-position runs.
    cma_seed = args.cma_seed
    if cma_seed is None and args.context_type == "photo":
        cma_seed = 1

    summaries: dict[str, pd.DataFrame] = {}
    for tmpl in TEMPLATES:
        csv = args.dla_dir / f"{tmpl}_dla.csv"
        if not csv.exists():
            print(f"skip {tmpl}: {csv} not found")
            continue
        print(f"loading {csv}")
        dla_df = pd.read_csv(csv)
        summary = per_head_summary(dla_df)
        summaries[tmpl] = summary

        # Per-template artifacts
        plot_template_heatmap(summary, tmpl, args.dla_dir / f"{tmpl}_dla_heatmap.png")

        top = summary.assign(abs_ld=summary["mean_logit_diff"].abs()) \
                     .nlargest(20, "abs_ld").drop(columns="abs_ld")
        top_csv = args.dla_dir / f"{tmpl}_dla_top20.csv"
        top.to_csv(top_csv, index=False)
        print(f"wrote {top_csv}")

        cma = load_cma_scores(tmpl, args.context_type, seed=cma_seed)
        if cma is not None:
            plot_dla_vs_cma(summary, cma, tmpl, args.context_type,
                            args.dla_dir / f"{tmpl}_dla_vs_cma.png")
        else:
            print(f"  (no CMA scores found for {tmpl}, skipping scatter)")

    if summaries:
        plot_summary_grid(summaries, args.dla_dir / "dla_summary_grid.png")


if __name__ == "__main__":
    main()
