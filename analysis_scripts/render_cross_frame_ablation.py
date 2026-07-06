#!/usr/bin/env python3
"""Cross-frame ablation figure: two stacked rows in the style of
render_ablation_figure.py.

Top row    : ablate photo  heads (C2) -> evaluate on false_belief  (+ insets)
Bottom row : ablate belief heads (C1) -> evaluate on false_photo   (+ insets)

Each row reproduces the original layout: one large main panel showing the
target/control flip-score curves on the cross-frame variant, plus three small
inset panels on the right showing non-target variants stay near baseline.
"""

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
N_PER_TEMPLATE = 150
N_TEMPLATES = 3

VARIANT_LABELS = {
    "false_belief": "false belief",
    "true_belief": "true belief",
    "false_photo": "false photo",
    "true_photo": "true photo",
    "basic_ab": "basic A→B",
    "basic_ba": "basic B→A",
}

ROWS = [
    {
        "csv": "other_results/ablation/photo_heads/cumulative_ablation_results.csv",
        "head_label": "photo heads (C2-derived, K=26)",
        "main_variant": "false_belief",
        "insets": ["true_belief", "basic_ab", "basic_ba"],
        "title": "ablate photo heads → evaluate on false belief (C1 contrast)",
    },
    {
        "csv": "other_results/ablation/belief_heads/cumulative_ablation_results.csv",
        "head_label": "belief heads (C1-derived, K=32)",
        "main_variant": "false_photo",
        "insets": ["true_photo", "basic_ab", "basic_ba"],
        "title": "ablate belief heads → evaluate on false photo (C2 contrast)",
    },
]

# Corrected protocol: matched sysprompt + per_variant prefill.
# K labels are derived dynamically from each run's head_list.csv at render time.
ROWS_CORRECTED = [
    {
        "csv": "other_results/ablation/photo_heads_sysprompt_prefill/cumulative_ablation_results.csv",
        "head_label_tmpl": "photo heads (C2-derived, K={K})",
        "main_variant": "false_belief",
        "insets": ["true_belief", "basic_ab", "basic_ba"],
        "title": "ablate photo heads → evaluate on false belief (C1 contrast)",
    },
    {
        "csv": "other_results/ablation/belief_only_sysprompt_prefill/cumulative_ablation_results.csv",
        "head_label_tmpl": "belief heads (C1-derived, K={K})",
        "main_variant": "false_photo",
        "insets": ["true_photo", "basic_ab", "basic_ba"],
        "title": "ablate belief heads → evaluate on false photo (C2 contrast)",
    },
]


def resolve_rows(project_root: Path, corrected: bool):
    """Returns ROWS with head_label filled in (dynamic K for corrected variant)."""
    if not corrected:
        return ROWS
    out = []
    for spec in ROWS_CORRECTED:
        head_list_path = (project_root / spec["csv"]).parent / "head_list.csv"
        try:
            k = len(pd.read_csv(head_list_path))
        except FileNotFoundError:
            k = "?"
        new_spec = dict(spec)
        new_spec["head_label"] = spec["head_label_tmpl"].format(K=k)
        out.append(new_spec)
    return out


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    n = N_PER_TEMPLATE * N_TEMPLATES
    df["sem_p_correct"] = df["p_correct_std"] / np.sqrt(n)
    df["sem_p_alt"] = df["p_alt_std"] / np.sqrt(n)
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


def style_panel(ax, *, ylabel=False, fontsize=10, baseline=None):
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
        ax.axhline(baseline, color="#9b9b9b", linewidth=0.7,
                   linestyle=(0, (3, 3)), zorder=1)
    if ylabel:
        ax.set_ylabel(
            r"flip score $\phi = p_{\mathrm{alt}} / (p_{\mathrm{alt}} + p_{\mathrm{correct}})$",
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


def render_three_col(out_path: Path, project_root: Path,
                     side_variant: str = "basic_ab",
                     corrected: bool = False):
    """Three-column layout: two main panels + stacked specificity insets."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    rows = resolve_rows(project_root, corrected)
    fig = plt.figure(figsize=(11.0, 3.5), facecolor="white")
    gs = GridSpec(2, 5, figure=fig, hspace=0.55, wspace=0.45,
                  left=0.06, right=0.99, top=0.83, bottom=0.16)

    main_axes = []
    for col_block, spec in zip([(0, 2), (2, 4)], rows):
        ax = fig.add_subplot(gs[:, col_block[0]:col_block[1]])
        main_axes.append((ax, spec))

    for ax, spec in main_axes:
        df = load(project_root / spec["csv"])
        v = spec["main_variant"]
        sub = df[df["eval_variant"] == v]
        baseline = sub[sub["k"] == 0]["flip_score"].mean()
        plot_curves(ax, sub, big=True)
        style_panel(ax, ylabel=(ax is main_axes[0][0]),
                    fontsize=10, baseline=baseline)
        ax.set_xlabel("# heads ablated", fontsize=10, color="#222222")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, sub["k"].max() + 0.5)
        ax.text(
            0.985, 0.97,
            r"$\bf{" + VARIANT_LABELS[v].replace(" ", "\\ ") + r"}$"
            + f"\n{spec['head_label']}\nbaseline $\\phi$ = {baseline:.2f}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6))

    # legend on the first main panel only
    leg = main_axes[0][0].legend(loc="upper left", frameon=True, fontsize=9,
                                 edgecolor="#cccccc", facecolor="white",
                                 bbox_to_anchor=(0.015, 0.985))
    leg.get_frame().set_linewidth(0.6)

    # ---- side column: two stacked specificity panels ----
    for row_idx, (_, spec) in enumerate(main_axes):
        ax = fig.add_subplot(gs[row_idx, 4])
        df = load(project_root / spec["csv"])
        sub = df[df["eval_variant"] == side_variant]
        ibase = sub[sub["k"] == 0]["flip_score"].mean()
        plot_curves(ax, sub, big=False)
        style_panel(ax, fontsize=8, baseline=ibase)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, sub["k"].max() + 0.5)
        ax.set_yticks([0, 0.5, 1.0])
        ax.text(0.04, 0.92, VARIANT_LABELS[side_variant],
                transform=ax.transAxes, fontsize=8.5, va="top", ha="left",
                color="#333333", weight="bold")
        # tag which head set this corresponds to
        head_tag = "C2 heads" if row_idx == 0 else "C1 heads"
        ax.text(0.96, 0.08, head_tag,
                transform=ax.transAxes, fontsize=7.5, va="bottom", ha="right",
                color="#777777", style="italic")
        if row_idx == 1:
            ax.set_xlabel("# heads ablated", fontsize=8, color="#444444")
        else:
            ax.set_xticklabels([])

    fig.suptitle(
        "Cross-frame transfer (left, center) vs. agent-free retrieval (right).",
        fontsize=11.5, color="#1a1a1a", y=0.965, x=0.06,
        ha="left", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {out_path.with_suffix('.pdf')}")


def render_compact(out_path: Path, project_root: Path, corrected: bool = False):
    """Two main panels side-by-side, no insets. Main-body version."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    rows = resolve_rows(project_root, corrected)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.17, wspace=0.22)

    for ax, spec in zip(axes, rows):
        df = load(project_root / spec["csv"])
        v = spec["main_variant"]
        sub = df[df["eval_variant"] == v]
        baseline = sub[sub["k"] == 0]["flip_score"].mean()
        plot_curves(ax, sub, big=True)
        style_panel(ax, ylabel=(ax is axes[0]), fontsize=10, baseline=baseline)
        ax.set_xlabel("# heads ablated", fontsize=10, color="#222222")
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, sub["k"].max() + 0.5)
        # combined corner annotation
        ax.text(
            0.985, 0.97,
            r"$\bf{" + VARIANT_LABELS[v].replace(" ", "\\ ") + r"}$"
            + f"\n{spec['head_label']}\nbaseline $\\phi$ = {baseline:.2f}",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6))

    # single shared legend in the first panel
    leg = axes[0].legend(loc="upper left", frameon=True, fontsize=9,
                         edgecolor="#cccccc", facecolor="white",
                         bbox_to_anchor=(0.015, 0.985))
    leg.get_frame().set_linewidth(0.6)

    fig.suptitle(
        "Cross-frame transfer: ablating heads selected on one frame "
        "flips the other frame.",
        fontsize=11.5, color="#1a1a1a", y=0.965, x=0.07,
        ha="left", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {out_path.with_suffix('.pdf')}")


def render(out_path: Path, project_root: Path, corrected: bool = False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
    })
    rows = resolve_rows(project_root, corrected)
    fig = plt.figure(figsize=(11.0, 10.5), facecolor="white")
    # 2 outer rows; each outer row is a (3 inner rows × 5 cols) GridSpec block.
    outer = GridSpec(2, 1, figure=fig, hspace=0.45,
                     left=0.07, right=0.985, top=0.93, bottom=0.06)

    for row_idx, spec in enumerate(rows):
        df = load(project_root / spec["csv"])
        inner = outer[row_idx].subgridspec(3, 5, hspace=0.55, wspace=0.45)

        # ---- main panel ----
        ax_main = fig.add_subplot(inner[:, :3])
        v = spec["main_variant"]
        sub_main = df[df["eval_variant"] == v]
        baseline = sub_main[sub_main["k"] == 0]["flip_score"].mean()
        plot_curves(ax_main, sub_main, big=True)
        style_panel(ax_main, ylabel=True, fontsize=12, baseline=baseline)
        ax_main.set_xlabel("# heads ablated (cumulative)",
                           fontsize=11, color="#222222")
        ax_main.set_ylim(0, 1.0)
        ax_main.set_xlim(-0.5, sub_main["k"].max() + 0.5)
        ax_main.text(
            0.985, 0.97,
            r"$\bf{" + VARIANT_LABELS[v].replace(" ", "\\ ") + r"}$"
            + f"\nbaseline $\\phi$ = {baseline:.2f}"
            + f"\n{spec['head_label']}",
            transform=ax_main.transAxes, fontsize=10, va="top", ha="right",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", linewidth=0.6))
        leg = ax_main.legend(loc="upper left", frameon=True, fontsize=10,
                             edgecolor="#cccccc", facecolor="white",
                             bbox_to_anchor=(0.015, 0.985))
        leg.get_frame().set_linewidth(0.6)

        # row title (single line, well-padded so it doesn't crash into axes)
        ax_main.set_title(spec["title"], fontsize=12.5, color="#1a1a1a",
                          loc="left", pad=14, weight="bold")

        # ---- 3 inset panels ----
        for i, var in enumerate(spec["insets"]):
            ax = fig.add_subplot(inner[i, 3:])
            sub = df[df["eval_variant"] == var]
            ibase = sub[sub["k"] == 0]["flip_score"].mean()
            plot_curves(ax, sub, big=False)
            style_panel(ax, fontsize=9, baseline=ibase)
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
        "Cross-frame transfer: ablating veracity heads selected on one frame "
        "flips the other frame.",
        fontsize=13.5, color="#1a1a1a", y=0.975, x=0.07,
        ha="left", weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    print(f"saved {out_path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path("neurips/figures/ablation_cross_frame.png"))
    p.add_argument("--compact", action="store_true",
                   help="Render compact two-panel main-body version (no insets).")
    p.add_argument("--three_col", action="store_true",
                   help="Render three-column layout: two main panels + stacked side specificity panels.")
    p.add_argument("--side_variant", type=str, default="basic_ab",
                   help="Eval variant to show in the side column (default: basic_ab).")
    p.add_argument("--corrected", action="store_true",
                   help="Use corrected-protocol runs (matched sysprompt + per_variant prefill).")
    args = p.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_path = args.out if args.out.is_absolute() else project_root / args.out
    if args.three_col:
        render_three_col(out_path, project_root,
                         side_variant=args.side_variant,
                         corrected=args.corrected)
    elif args.compact:
        render_compact(out_path, project_root, corrected=args.corrected)
    else:
        render(out_path, project_root, corrected=args.corrected)


if __name__ == "__main__":
    main()
