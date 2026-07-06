#!/usr/bin/env python3
"""Render the Option-A ablation figure: one big false_belief panel + three small
inset panels (true_belief, basic_ab, basic_ba) showing selectivity."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

TARGET_COLOR = "#c74e3b"
CONTROL_COLOR = "#5c6b73"
RIBBON_ALPHA = 0.08
BASELINE_GRID = "#dadada"

VARIANT_LABELS = {
    "false_belief": "false belief",
    "true_belief": "true belief",
    "basic_ab": "basic A→B",
    "basic_ba": "basic B→A",
}

INSETS = ["true_belief", "basic_ab", "basic_ba"]


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["sem_p_correct"] = df["p_correct_std"] / np.sqrt(150)
    df["sem_p_alt"] = df["p_alt_std"] / np.sqrt(150)
    eps = 1e-12
    df["flip_lo"] = (df["mean_p_alt"] - df["sem_p_alt"]) / (
        (df["mean_p_alt"] - df["sem_p_alt"]) + (df["mean_p_correct"] + df["sem_p_correct"]) + eps
    )
    df["flip_hi"] = (df["mean_p_alt"] + df["sem_p_alt"]) / (
        (df["mean_p_alt"] + df["sem_p_alt"]) + (df["mean_p_correct"] - df["sem_p_correct"]) + eps
    )
    df["flip_lo"] = df["flip_lo"].clip(0, 1)
    df["flip_hi"] = df["flip_hi"].clip(0, 1)
    return df


def style_panel(ax, *, title=None, ylabel=False, fontsize=10, baseline=None):
    ax.set_facecolor("white")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#444444")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#444444", which="both", labelsize=fontsize - 1)
    ax.grid(axis="y", color=BASELINE_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if baseline is not None:
        ax.axhline(baseline, color="#9b9b9b", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
    if title:
        ax.set_title(title, fontsize=fontsize, pad=4, color="#222222")
    if ylabel:
        ax.set_ylabel(r"flip score $\phi = p_{\mathrm{alt}} / (p_{\mathrm{alt}} + p_{\mathrm{correct}})$",
                      fontsize=fontsize, color="#222222")


def plot_curves(ax, df_v, *, big=False):
    for curve_name, color in [("target", TARGET_COLOR), ("control", CONTROL_COLOR)]:
        sub = df_v[df_v["curve"] == curve_name].sort_values("k")
        ax.fill_between(sub["k"], sub["flip_lo"], sub["flip_hi"],
                        color=color, alpha=RIBBON_ALPHA, linewidth=0)
        ax.plot(sub["k"], sub["flip_score"],
                color=color, linewidth=2.0 if big else 1.4,
                marker="o", markersize=3.5 if big else 2.5,
                markerfacecolor=color, markeredgecolor="white",
                markeredgewidth=0.6, label=curve_name, zorder=3)


def render(df: pd.DataFrame, out_path: Path, *, condition_label: str):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(11.0, 5.8), facecolor="white")
    gs = GridSpec(3, 5, figure=fig, hspace=0.55, wspace=0.45,
                  left=0.07, right=0.985, top=0.86, bottom=0.10)

    # left big panel: false_belief, spans all 3 rows × 3 cols
    ax_main = fig.add_subplot(gs[:, :3])
    fb = df[df["eval_variant"] == "false_belief"]
    baseline_fb = fb[fb["k"] == 0]["flip_score"].mean()
    plot_curves(ax_main, fb, big=True)
    style_panel(ax_main, ylabel=True, fontsize=12, baseline=baseline_fb)
    ax_main.set_xlabel("# heads ablated (cumulative)", fontsize=11, color="#222222")
    ax_main.set_ylim(0, 1.0)
    ax_main.set_xlim(-0.5, fb["k"].max() + 0.5)
    ax_main.text(0.985, 0.97,
                 r"$\bf{false\ belief}$" + f"\nbaseline $\\phi$ = {baseline_fb:.2f}",
                 transform=ax_main.transAxes, fontsize=11, va="top", ha="right",
                 color="#222222",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                           edgecolor="#cccccc", linewidth=0.6))
    leg = ax_main.legend(loc="upper left", frameon=True, fontsize=10,
                          edgecolor="#cccccc", facecolor="white",
                          bbox_to_anchor=(0.015, 0.985))
    leg.get_frame().set_linewidth(0.6)

    # right column: 3 inset panels
    for i, var in enumerate(INSETS):
        ax = fig.add_subplot(gs[i, 3:])
        sub = df[df["eval_variant"] == var]
        baseline = sub[sub["k"] == 0]["flip_score"].mean()
        plot_curves(ax, sub, big=False)
        style_panel(ax, fontsize=9, baseline=baseline)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, sub["k"].max() + 0.5)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.04, 0.92, VARIANT_LABELS[var],
                transform=ax.transAxes, fontsize=9, va="top", ha="left",
                color="#333333", weight="bold")
        if i == 2:
            ax.set_xlabel("# heads ablated", fontsize=9, color="#444444")
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        f"Targeted ablation flips first-token mass on false belief; "
        f"control ablation does not.",
        fontsize=13.5, color="#1a1a1a", y=0.965, x=0.07, ha="left", weight="bold")
    fig.text(0.07, 0.915,
             f"head set: {condition_label}    —    target = top-22 by max-observed CMA    —    "
             f"control = matched-layer, lowest-|CMA|",
             fontsize=9, color="#666666", ha="left", style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {out_path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("other_results/ablation/loc_belief_heads/cumulative_ablation_results.csv"))
    p.add_argument("--label", type=str, default="loc_belief (C7-derived)")
    p.add_argument("--out", type=Path,
                   default=Path("neurips/figures/ablation_loc_belief.png"))
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = args.csv if args.csv.is_absolute() else project_root / args.csv
    out_path = args.out if args.out.is_absolute() else project_root / args.out

    df = load(csv_path)
    render(df, out_path, condition_label=args.label)


if __name__ == "__main__":
    main()
