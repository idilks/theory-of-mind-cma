"""Rebuttal figure 4 — cross-frame / cross-domain ablation transfer (R2).

Figure version of the transfer table: heads selected on one task frame,
ablated while the model answers a different frame's prompts. All rows use
the matched-prompt/prefill protocol with a layer-matched control set.
Companion no-divergence variants (true photo / true belief / code_alias)
stay largely intact at the same k.

Values are read from the run CSVs, not hardcoded.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, AQUA, GRAY, INK, INK2, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
ABL = REPO / "results/ablation"
ABL_OTHER = REPO / "other_results/ablation"

# (row label, run dir, k, divergence eval, no-divergence eval)
ROWS = [
    ("belief heads (C1, 32)\n$\\rightarrow$ false-photo prompts",
     ABL / "belief_only_sysprompt_prefill", 32, "false_photo", "true_photo"),
    ("photo heads (C2, 26)\n$\\rightarrow$ false-belief prompts",
     ABL / "photo_heads_sysprompt_prefill", 26, "false_belief", "true_belief"),
    ("belief $\\cap$ photo (19)\n$\\rightarrow$ false-belief prompts",
     ABL / "decomp_intersection_belief_rank", 19, "false_belief", "true_belief"),
    ("belief $\\cap$ photo (19)\n$\\rightarrow$ false-photo prompts",
     ABL / "decomp_intersection_belief_rank", 19, "false_photo", "true_photo"),
    ("code copy-vs-alias heads (12)\n$\\rightarrow$ false-belief prompts",
     ABL / "code_cross_code_abstract_on_tom_matched_prefill", 12,
     "false_belief", "true_belief"),
    ("C1 full-1920 ranking (top 16)\n$\\rightarrow$ Python code_copy",
     ABL_OTHER / "tom_c1_full1920_on_code_tom_prefill_per_variant", 16,
     "code_copy", "code_alias"),
]


def cell(df, curve, variant, k, col):
    m = df[(df.curve == curve) & (df.eval_variant == variant) & (df.k == k)]
    assert len(m) == 1, (curve, variant, k, len(m))
    return float(m[col].iloc[0])


def main():
    apply_style()
    vals = []
    for label, run, k, ev, ev0 in ROWS:
        df = pd.read_csv(run / "cumulative_ablation_results.csv")
        vals.append({
            "label": label,
            "acc_clean": cell(df, "target", ev, 0, "accuracy"),
            "acc_abl": cell(df, "target", ev, k, "accuracy"),
            "acc_ctl": cell(df, "control", ev, k, "accuracy"),
            "acc_nodiv": cell(df, "target", ev0, k, "accuracy"),
            "flip_clean": cell(df, "target", ev, 0, "flip_score"),
            "flip_abl": cell(df, "target", ev, k, "flip_score"),
            "flip_ctl": cell(df, "control", ev, k, "flip_score"),
        })
        print(f"{label.splitlines()[0]:38s} acc {vals[-1]['acc_clean']:.2f}->"
              f"{vals[-1]['acc_abl']:.2f} flip {vals[-1]['flip_clean']:.3f}->"
              f"{vals[-1]['flip_abl']:.3f} ctl acc {vals[-1]['acc_ctl']:.2f}"
              f" nodiv {vals[-1]['acc_nodiv']:.2f}")

    fig, (ax_a, ax_f) = plt.subplots(
        1, 2, figsize=(9.2, 4.2), sharey=True,
        gridspec_kw={"width_ratios": [1, 0.85], "wspace": 0.10})
    ys = np.arange(len(vals))[::-1]

    # ---- accuracy panel -------------------------------------------------
    for yv, v in zip(ys, vals):
        ax_a.plot([v["acc_abl"], v["acc_clean"]], [yv, yv], color=BLUE,
                  lw=1.4, alpha=0.5, zorder=2)
        ax_a.plot([v["acc_clean"]], [yv], "o", color=INK2, mfc="white",
                  ms=6, mew=1.3, zorder=3)
        ax_a.plot([v["acc_abl"]], [yv], "o", color=BLUE, ms=7, zorder=4)
        ax_a.plot([v["acc_ctl"]], [yv + 0.28], "|", color=GRAY, ms=8, mew=1.8,
                  zorder=3)
        ax_a.plot([v["acc_nodiv"]], [yv - 0.28], "D", color=AQUA, ms=4.5,
                  zorder=3)
        ax_a.text(v["acc_abl"] - 0.03, yv, f"{v['acc_abl'] + 1e-9:.2f}",
                  ha="right", va="center", fontsize=7.5, color=BLUE,
                  fontweight="medium")
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels([v["label"] for v in vals], fontsize=8)
    ax_a.set_xlim(0, 1.06)
    ax_a.set_xlabel("accuracy on held-out prompts")
    ax_a.set_title("Accuracy: clean $\\circ$ $\\rightarrow$ ablated $\\bullet$",
                   loc="left")
    handles = [
        plt.Line2D([], [], marker="o", color=INK2, mfc="white", ls="none",
                   label="clean"),
        plt.Line2D([], [], marker="o", color=BLUE, ls="none", label="ablated"),
        plt.Line2D([], [], marker="|", color=GRAY, ls="none", mew=1.8,
                   label="layer-matched control, ablated"),
        plt.Line2D([], [], marker="D", color=AQUA, ls="none", ms=4.5,
                   label="no-divergence variant, ablated"),
    ]
    ax_a.legend(handles=handles, loc="upper left", fontsize=7,
                bbox_to_anchor=(0.0, -0.14), ncol=2, frameon=False)
    panel_label(ax_a, "a", x=-0.60)

    # ---- flip-score panel -----------------------------------------------
    for yv, v in zip(ys, vals):
        ax_f.barh(yv, v["flip_abl"], height=0.5, color=BLUE, alpha=0.9)
        ax_f.barh(yv, v["flip_ctl"], height=0.5, color=GRAY, alpha=0.9,
                  left=0, zorder=3)
        ax_f.plot([v["flip_clean"]], [yv], "|", color=INK, ms=10, mew=1.4,
                  zorder=4)
        ax_f.text(v["flip_abl"] + 0.012, yv, f"{v['flip_abl'] + 1e-9:.2f}",
                  va="center", fontsize=7.5, color=BLUE)
    ax_f.set_xlim(0, 0.55)
    ax_f.set_xlabel(r"flip score $\phi_p$ after ablation")
    ax_f.set_title("Redirection toward the alternate answer", loc="left")
    ax_f.text(0.0, -0.145,
              "bar = target heads;  grey = layer-matched control;  "
              "tick = clean baseline ($\\phi_p \\approx$ 0.01)",
              fontsize=7, color=INK2, va="top", ha="left",
              transform=ax_f.transAxes)
    panel_label(ax_f, "b", x=-0.06)

    for ax in (ax_a, ax_f):
        despine(ax)
        ax.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "fig4_cross_frame_transfer")


if __name__ == "__main__":
    main()
