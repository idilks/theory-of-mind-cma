"""Render the intersection-ablation figure for the thesis/neurips update.

Layout (matches the visual grammar of figures/ablation_cross_frame_3col.png):
  Main left  : ablate B∩P intersection heads (19, B-rank). Eval on false_belief.
               Overlay (orange dashed): full T_C1 belief heads (32). The two
               curves should be nearly indistinguishable -- the H2 punch.
  Main right : same intersection ablation, eval on false_photo. Overlay
               (orange dashed): full T_C2 photo heads (26).
  Side col   : stacked basic_ab (top) and basic_ba (bottom). Intersection only.
               Should stay near baseline => belief/photo heads are not generic
               retrieval heads (specificity check).
  All panels : matched_sys + prefill null overlaid as grey rectangles at k=11
               and k=21 (null_mean +- 1.96*null_std).

Reads:
  other_results/ablation/decomp_intersection_belief_rank/   (intersection, 19)
  other_results/ablation/belief_only_sysprompt_prefill/     (T_C1, 32 heads)
  other_results/ablation/photo_heads_sysprompt_prefill/     (T_C2, 26 heads)
  other_results/ablation/random_baselines/belief_only_matched_prefill/null_summary.csv

Writes:
  neurips/figures/ablation_intersection_3col.{png,pdf}  (or path via --out)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

# ---------- style ----------
INTERSECTION_COLOR = "#c74e3b"    # red, primary
FULL_COLOR         = "#e58a1f"    # orange, comparator
NULL_COLOR         = "#7a7a7a"
NULL_FILL_ALPHA    = 0.22
RIBBON_ALPHA       = 0.10
BASELINE_GRID      = "#dadada"
N_PER_TEMPLATE     = 50
N_TEMPLATES        = 3

VARIANT_LABELS = {
    "false_belief": "false belief",
    "false_photo":  "false photo",
    "basic_ab":     "basic A→B",
    "basic_ba":     "basic B→A",
}

ABL = Path("other_results/ablation")
DEFAULT_NULL_PATH = ABL / "random_baselines" / "belief_only_matched_prefill" / "null_summary.csv"

INTERSECTION_CSV = ABL / "decomp_intersection_belief_rank" / "cumulative_ablation_results.csv"
BELIEF_CSV       = ABL / "belief_only_sysprompt_prefill"   / "cumulative_ablation_results.csv"
PHOTO_CSV        = ABL / "photo_heads_sysprompt_prefill"   / "cumulative_ablation_results.csv"


def load_curve(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["curve"] == "target"].copy()
    n = N_PER_TEMPLATE * N_TEMPLATES
    df["sem_p_correct"] = df["p_correct_std"] / np.sqrt(n)
    df["sem_p_alt"]     = df["p_alt_std"]     / np.sqrt(n)
    eps = 1e-12
    df["flip_lo"] = (df["mean_p_alt"] - df["sem_p_alt"]) / (
        (df["mean_p_alt"] - df["sem_p_alt"])
        + (df["mean_p_correct"] + df["sem_p_correct"]) + eps
    )
    df["flip_hi"] = (df["mean_p_alt"] + df["sem_p_alt"]) / (
        (df["mean_p_alt"] + df["sem_p_alt"])
        + (df["mean_p_correct"] - df["sem_p_correct"]) + eps
    )
    df["flip_lo"] = df["flip_lo"].clip(0, 1)
    df["flip_hi"] = df["flip_hi"].clip(0, 1)
    return df


def load_null(null_path: Path) -> pd.DataFrame:
    return pd.read_csv(null_path)


def style_panel(ax, *, ylabel=False, fontsize=10, baseline=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#444", which="both", labelsize=fontsize - 1)
    ax.grid(axis="y", color=BASELINE_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if baseline is not None:
        ax.axhline(baseline, color="#9b9b9b", linewidth=0.7,
                   linestyle=(0, (3, 3)), zorder=1)
    if ylabel:
        ax.set_ylabel(
            r"flip score $\phi = p_{\mathrm{alt}} / (p_{\mathrm{alt}} + p_{\mathrm{correct}})$",
            fontsize=fontsize, color="#222")


def plot_curve(ax, df, *, color, dashed=False, label, big=False):
    sub = df.sort_values("k")
    ax.fill_between(sub["k"], sub["flip_lo"], sub["flip_hi"],
                    color=color, alpha=RIBBON_ALPHA, linewidth=0)
    ax.plot(sub["k"], sub["flip_score"],
            color=color,
            linewidth=2.0 if big else 1.4,
            linestyle="--" if dashed else "-",
            marker="o" if not dashed else "s",
            markersize=4.0 if big else 3.0,
            markerfacecolor="white" if dashed else color,
            markeredgecolor=color,
            markeredgewidth=0.9,
            label=label, zorder=3)


def overlay_null(ax, null_df, k_max, *, eval_variant="false_belief",
                 metric="flip_score", small=False):
    nb = null_df[(null_df["eval_variant"] == eval_variant) & (null_df["metric"] == metric)]
    label_used = False
    for _, r in nb.iterrows():
        k = int(r["k"])
        if k > k_max:
            continue
        mu, sd = r["null_mean"], r["null_std"]
        ax.fill_between([k - 0.6, k + 0.6],
                        max(0, mu - 1.96 * sd), mu + 1.96 * sd,
                        color=NULL_COLOR, alpha=NULL_FILL_ALPHA, zorder=2,
                        label=(None if label_used or small else
                               "random null (matched+prefill, ±1.96σ)"))
        ax.hlines(mu, k - 0.6, k + 0.6,
                  colors=NULL_COLOR, linestyles=(0, (4, 2)),
                  linewidth=1.2, zorder=2)
        label_used = True


def render(out_path: Path, *, eval_belief: str = "false_belief",
           eval_photo: str = "false_photo",
           null_path: Path = DEFAULT_NULL_PATH) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })

    df_int    = load_curve(INTERSECTION_CSV)
    df_belief = load_curve(BELIEF_CSV)
    df_photo  = load_curve(PHOTO_CSV)
    null_df   = load_null(null_path)

    belief_label_pretty = eval_belief.replace("_", " ")
    photo_label_pretty  = eval_photo.replace("_", " ")

    fig = plt.figure(figsize=(12.5, 4.0), facecolor="white")
    gs  = GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.55,
                   left=0.06, right=0.99, top=0.82, bottom=0.18)

    # ---- main panels ----
    ax_belief = fig.add_subplot(gs[:, 0:2])
    ax_photo  = fig.add_subplot(gs[:, 2:4])

    # --- belief panel ---
    sub_int = df_int[df_int["eval_variant"] == eval_belief]
    sub_bel = df_belief[df_belief["eval_variant"] == eval_belief]
    baseline = float(sub_int[sub_int["k"] == 0]["flip_score"].mean())

    plot_curve(ax_belief, sub_int, color=INTERSECTION_COLOR,
               label="B∩P intersection (K=19)", big=True)
    plot_curve(ax_belief, sub_bel, color=FULL_COLOR, dashed=True,
               label="all belief heads $\\mathcal{T}_{C_1}$ (K=32)", big=True)
    k_max_left = int(max(sub_int["k"].max(), sub_bel["k"].max()))
    overlay_null(ax_belief, null_df, k_max=k_max_left, eval_variant=eval_belief)

    style_panel(ax_belief, ylabel=True, fontsize=10, baseline=baseline)
    ax_belief.set_xlabel("# heads ablated", fontsize=10, color="#222")
    ax_belief.set_ylim(0, 1.0)
    ax_belief.set_xlim(-0.5, k_max_left + 0.5)
    ax_belief.text(
        0.985, 0.97,
        r"$\bf{" + belief_label_pretty.replace(" ", r"\ ") + r"}$"
        + f"\nbaseline $\\phi$ = {baseline:.2f}",
        transform=ax_belief.transAxes, fontsize=9, va="top", ha="right",
        color="#222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.6))
    leg = ax_belief.legend(loc="upper left", frameon=True, fontsize=8.5,
                           edgecolor="#cccccc", facecolor="white",
                           bbox_to_anchor=(0.015, 0.985))
    leg.get_frame().set_linewidth(0.6)

    # --- photo panel ---
    sub_int_p = df_int[df_int["eval_variant"] == eval_photo]
    sub_pho   = df_photo[df_photo["eval_variant"] == eval_photo]
    baseline_p = float(sub_int_p[sub_int_p["k"] == 0]["flip_score"].mean())

    plot_curve(ax_photo, sub_int_p, color=INTERSECTION_COLOR,
               label="B∩P intersection (K=19)", big=True)
    plot_curve(ax_photo, sub_pho, color=FULL_COLOR, dashed=True,
               label="all photo heads $\\mathcal{T}_{C_2}$ (K=26)", big=True)
    k_max_right = int(max(sub_int_p["k"].max(), sub_pho["k"].max()))
    overlay_null(ax_photo, null_df, k_max=k_max_right, eval_variant=eval_photo)

    style_panel(ax_photo, ylabel=False, fontsize=10, baseline=baseline_p)
    ax_photo.set_xlabel("# heads ablated", fontsize=10, color="#222")
    ax_photo.set_ylim(0, 1.0)
    ax_photo.set_xlim(-0.5, k_max_right + 0.5)
    ax_photo.text(
        0.985, 0.97,
        r"$\bf{" + photo_label_pretty.replace(" ", r"\ ") + r"}$"
        + f"\nbaseline $\\phi$ = {baseline_p:.2f}",
        transform=ax_photo.transAxes, fontsize=9, va="top", ha="right",
        color="#222",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.6))
    leg_p = ax_photo.legend(loc="upper left", frameon=True, fontsize=8.5,
                            edgecolor="#cccccc", facecolor="white",
                            bbox_to_anchor=(0.015, 0.985))
    leg_p.get_frame().set_linewidth(0.6)

    # ---- side column: basic_ab and basic_ba specificity (intersection only) ----
    for i, v in enumerate(["basic_ab", "basic_ba"]):
        ax = fig.add_subplot(gs[i, 4])
        sub = df_int[df_int["eval_variant"] == v]
        ibase = float(sub[sub["k"] == 0]["flip_score"].mean())
        plot_curve(ax, sub, color=INTERSECTION_COLOR, label=None, big=False)
        overlay_null(ax, null_df, k_max=int(sub["k"].max()),
                     eval_variant=v, small=True)
        style_panel(ax, fontsize=8, baseline=ibase)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, sub["k"].max() + 0.5)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.04, 0.92, VARIANT_LABELS[v],
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                color="#333", weight="bold")
        if i == 1:
            ax.set_xlabel("# heads ablated", fontsize=8, color="#444")
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        f"B∩P intersection ablation matches the full belief/photo set on "
        f"{belief_label_pretty} and {photo_label_pretty}, "
        "with no effect on agent-free retrieval.",
        fontsize=11.5, color="#1a1a1a", y=0.965, x=0.06,
        ha="left", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    print(f"wrote {out_path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("neurips/figures/ablation_intersection_3col.png"))
    p.add_argument("--eval_belief", default="false_belief",
                   choices=["false_belief", "true_belief"],
                   help="Eval variant for the left main panel.")
    p.add_argument("--eval_photo", default="false_photo",
                   choices=["false_photo", "true_photo"],
                   help="Eval variant for the right main panel.")
    p.add_argument("--null_csv", type=Path, default=None,
                   help="Path to null_summary.csv. Default: "
                        "other_results/ablation/random_baselines/"
                        "belief_only_matched_prefill/null_summary.csv. "
                        "Use belief_only_matched_prefill_true/null_summary.csv "
                        "for the true-variant figure.")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out_path = args.out if args.out.is_absolute() else root / args.out
    null_path = args.null_csv if args.null_csv is not None else DEFAULT_NULL_PATH
    if not null_path.is_absolute():
        null_path = root / null_path
    render(out_path, eval_belief=args.eval_belief,
           eval_photo=args.eval_photo, null_path=null_path)


if __name__ == "__main__":
    main()
