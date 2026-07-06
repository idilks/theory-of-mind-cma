"""Plot per-k paired sign-flip permutation results for FB-TB and FP-TP across ablation runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RUNS = {
    "belief_only_sysprompt_matched": {
        "label": "matched, no prefill",
        "color": "#888888",
        "linestyle": "--",
    },
    "belief_only_sysprompt_prefill": {
        "label": "matched + prefill",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "belief_only_neutral_prefill": {
        "label": "neutral + prefill",
        "color": "#ff7f0e",
        "linestyle": "-",
    },
    "decomp_intersection_belief_rank": {
        "label": "intersection, matched + prefill",
        "color": "#d62728",
        "linestyle": "-",
    },
}

PAIRS = [
    ("false_belief_vs_true_belief", "False belief vs True belief"),
    ("false_photo_vs_true_photo", "False photo vs True photo"),
]

RIBBON_ALPHA = 0.12
BASELINE_GRID = "#dcdcdc"


def style_panel(ax, *, fontsize=10):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors="#444", which="both", labelsize=fontsize - 1)
    ax.grid(axis="y", color=BASELINE_GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def load_all() -> pd.DataFrame:
    frames = []
    for run in RUNS:
        csv = ROOT / "other_results" / "ablation" / run / "permutation_fb_tb" / "permutation_per_k.csv"
        df = pd.read_csv(csv)
        if "run_label" not in df.columns or df["run_label"].isna().all():
            df["run_label"] = run
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def neglog10p(p):
    p = np.clip(p.astype(float), 1e-12, 1.0)
    return -np.log10(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "neurips" / "figures" / "ablation_permutation_per_k.png"))
    args = ap.parse_args()

    out_png = Path(args.out)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")

    df = load_all()

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor="white")

    for row, (pair_key, pair_title) in enumerate(PAIRS):
        ax_d = axes[row, 0]
        ax_p = axes[row, 1]
        sub_pair = df[df["pair"] == pair_key]

        for run, meta in RUNS.items():
            sub = sub_pair[sub_pair["run_label"] == run].sort_values("k")
            if sub.empty:
                continue
            color = meta["color"]
            ls = meta["linestyle"]
            label = meta["label"]

            ax_d.fill_between(sub["k"], sub["null_ci_lo"], sub["null_ci_hi"],
                              color=color, alpha=RIBBON_ALPHA, linewidth=0)
            ax_d.plot(sub["k"], sub["D_obs"], color=color, linestyle=ls,
                      linewidth=1.6, label=label, zorder=3)

            ax_p.plot(sub["k"], neglog10p(sub["p_two_sided"]),
                      color=color, linestyle=ls, linewidth=1.6, label=label, zorder=3)

        style_panel(ax_d)
        style_panel(ax_p)

        ax_d.axhline(0.0, color="#9b9b9b", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        ax_d.set_title(f"{pair_title}: Δ = φ_A − φ_B", fontsize=10, color="#222")
        ax_d.set_xlabel("k (heads ablated)", fontsize=9, color="#222")
        ax_d.set_ylabel(r"$\Delta = \varphi_A - \varphi_B$ (paired)", fontsize=9, color="#222")

        ax_p.axhline(-np.log10(0.05), color="#9b9b9b", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        ax_p.axhline(-np.log10(0.001), color="#9b9b9b", linewidth=0.7, linestyle=(0, (3, 3)), zorder=1)
        ax_p.text(ax_p.get_xlim()[1], -np.log10(0.05), " p=0.05",
                  fontsize=7, color="#666", va="center", ha="left")
        ax_p.text(ax_p.get_xlim()[1], -np.log10(0.001), " p=0.001",
                  fontsize=7, color="#666", va="center", ha="left")
        ax_p.set_title(f"{pair_title}: significance", fontsize=10, color="#222")
        ax_p.set_xlabel("k (heads ablated)", fontsize=9, color="#222")
        ax_p.set_ylabel(r"$-\log_{10}(p)$, two-sided", fontsize=9, color="#222")

        if row == 0:
            ax_d.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, facecolor="white")
    fig.savefig(out_pdf, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
