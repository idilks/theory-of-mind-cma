"""Appendix figure 14 — why the steering coefficient stops at alpha = 4.

Extends the residual-steering coefficient sweep (fig13, alpha <= 4) to
alpha in {6, 8, 12, 16} and adds a belief-free collateral readout. Rank-1
mean-difference edit at the layer-41 residual, final token, Qwen2.5-14B.

(a) add direction on true-belief prompts: first-token accuracy, target vs
    three norm-matched controls, over the full grid
(b) what the target edit does to the argmax token on the two main-text
    legs: alternative-location rate (an on-manifold answer change, solid)
    and off-manifold rate (argmax is neither location, dashed)
(c) generic belief-free prompts: KL(steered || clean), nats
(d) generic prompts: fraction whose top-1 token is unchanged; the same
    edit on belief-free story-frame prompts (basic_ab) for contrast

Data: results/aggregated/circuit/dose_extension/{belief_outcomes,
merged_summary,generic_summary}.csv, written by
scripts/circuit/analyze_dose_extension.py from L41_main + L41_dose_ext.
"""

import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (AQUA, BLUE, GRAY, INK2, ORANGE, VIOLET, apply_style,
                   despine, panel_label, save)

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/aggregated/circuit/dose_extension"
PAPER_FIG_DIR = REPO / "writeups/neurips-overleaf/figures"
NAME = "fig14_dose_extension"

DOSES = [0.5, 1, 2, 4, 6, 8, 12, 16]
CTL = {  # same encoding as fig13
    "random": ("-", "Random direction"),
    "shuffled": ("--", "Shuffled labels"),
    "heads_ctl": (":", "Non-selected heads"),
}


def series(df, prompt_set, mode, direction, col, sem=None):
    s = df[(df.prompt_set == prompt_set) & (df["mode"] == mode)
           & (df.direction == direction)].sort_values("dose")
    assert s.dose.tolist() == DOSES, (prompt_set, mode, direction, s.dose.tolist())
    y = s[col].to_numpy(dtype=float)
    e = s[sem].to_numpy(dtype=float) if sem else None
    return y, e


def clean_value(df, prompt_set, col):
    r = df[(df.prompt_set == prompt_set) & (df.direction == "clean")]
    assert len(r) == 1
    return float(r[col].iloc[0])


def log_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(DOSES)
    ax.set_xticklabels([f"{d:g}" for d in DOSES])
    ax.minorticks_off()
    ax.axvspan(4 / 1.12, 4 * 1.12, color="#f0efe9", zorder=0, lw=0)
    ax.set_xlabel(r"Steering coefficient $\alpha$")
    despine(ax)


def main():
    apply_style()
    out = pd.read_csv(DATA / "belief_outcomes.csv")
    merged = pd.read_csv(DATA / "merged_summary.csv")
    gen = pd.read_csv(DATA / "generic_summary.csv")
    assert (out.n == 225).all() and (gen.n == 225).all()

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) true-belief inject accuracy, target vs controls
    y, e = series(out, "true_belief", "inject", "target_rank1", "accuracy", "acc_sem")
    ax_a.errorbar(DOSES, 100 * y, yerr=100 * e, color=VIOLET, lw=1.8, marker="o",
                  ms=3, elinewidth=0.7, capsize=2, label="Mean difference")
    for d, (ls, lab) in CTL.items():
        y, e = series(out, "true_belief", "inject", d, "accuracy", "acc_sem")
        ax_a.errorbar(DOSES, 100 * y, yerr=100 * e, color=GRAY, lw=1, ls=ls,
                      marker="o", ms=2.5, elinewidth=0.6, capsize=1.5, label=lab)
    ax_a.axhline(100 * clean_value(out, "true_belief", "accuracy"), color=INK2,
                 lw=0.8, ls=(0, (1, 2)), label="Unedited")
    ax_a.set_ylim(0, 102)
    ax_a.set_yticks([0, 25, 50, 75, 100])
    ax_a.set_ylabel("First-token accuracy (%)")
    ax_a.set_title("Add on true belief", loc="left")
    ax_a.legend(fontsize=7, loc="lower left", handlelength=2.0)

    # (b) argmax outcome for the target edit on the two main-text legs
    for ps, mode, color, lab in [("true_belief", "inject", AQUA, "add on true belief"),
                                 ("false_belief", "delete", BLUE, "subtract on false belief")]:
        y, e = series(out, ps, mode, "target_rank1", "alt_rate", "alt_sem")
        ax_b.errorbar(DOSES, 100 * y, yerr=100 * e, color=color, lw=1.8, marker="o",
                      ms=3, elinewidth=0.7, capsize=2, label=f"alternative location, {lab}")
        y, e = series(out, ps, mode, "target_rank1", "off_manifold", "off_sem")
        ax_b.errorbar(DOSES, 100 * y, yerr=100 * e, color=color, lw=1.2, ls="--",
                      marker="s", ms=2.5, elinewidth=0.6, capsize=1.5,
                      label=f"neither location, {lab}")
    ax_b.set_ylim(0, 102)
    ax_b.set_yticks([0, 25, 50, 75, 100])
    ax_b.set_ylabel("Prompts (%)")
    ax_b.set_title("Argmax token under the target edit", loc="left")
    ax_b.legend(fontsize=6.5, loc="upper left", handlelength=2.0)

    # (c) generic KL
    y, e = series(gen, "generic", "inject", "target_rank1", "kl", "kl_sem")
    ax_c.errorbar(DOSES, y, yerr=e, color=VIOLET, lw=1.8, marker="o", ms=3,
                  elinewidth=0.7, capsize=2, label="Mean difference")
    for d, (ls, lab) in CTL.items():
        y, e = series(gen, "generic", "inject", d, "kl", "kl_sem")
        ax_c.errorbar(DOSES, y, yerr=e, color=GRAY, lw=1, ls=ls, marker="o", ms=2.5,
                      elinewidth=0.6, capsize=1.5, label=lab)
    ax_c.set_yscale("log")
    ax_c.set_ylim(5e-4, 80)
    ax_c.set_ylabel("KL(edited || clean), nats")
    ax_c.set_title("Unrelated prompts: distribution shift", loc="left")

    # (d) generic top-1 unchanged, plus basic_ab for contrast
    y, e = series(gen, "generic", "inject", "target_rank1", "top1_agree", "top1_sem")
    ax_d.errorbar(DOSES, 100 * y, yerr=100 * e, color=VIOLET, lw=1.8, marker="o",
                  ms=3, elinewidth=0.7, capsize=2, label="Mean difference")
    for d, (ls, lab) in CTL.items():
        y, e = series(gen, "generic", "inject", d, "top1_agree", "top1_sem")
        ax_d.errorbar(DOSES, 100 * y, yerr=100 * e, color=GRAY, lw=1, ls=ls,
                      marker="o", ms=2.5, elinewidth=0.6, capsize=1.5, label=lab)
    b = merged[(merged.prompt_set == "basic_ab") & (merged.direction == "target_rank1")
               & merged.top1_agree.notna()].sort_values("dose")
    ax_d.plot(b.dose, 100 * b.top1_agree, color=ORANGE, lw=1.2, ls="--", marker="s",
              ms=2.5, label="Mean difference, belief-free story prompts")
    ax_d.set_ylim(0, 102)
    ax_d.set_yticks([0, 25, 50, 75, 100])
    ax_d.set_ylabel("Top-1 token unchanged (%)")
    ax_d.set_title("Unrelated prompts: argmax preserved", loc="left")
    # control encoding is labelled in (a); only the contrast curve needs a key
    h, l = ax_d.get_legend_handles_labels()
    ax_d.legend([h[-1]], ["Same edit, belief-free story prompts"], fontsize=7,
                loc="lower left", handlelength=2.0)

    for ax, letter in zip(axes.flat, "abcd"):
        log_axis(ax)
        panel_label(ax, letter, x=-0.17)

    fig.tight_layout(h_pad=1.6, w_pad=1.6)
    save(fig, NAME)
    PAPER_FIG_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        shutil.copy(REPO / "rebuttal_figures" / f"{NAME}.{ext}", PAPER_FIG_DIR / f"{NAME}.{ext}")
    print(f"copied to {PAPER_FIG_DIR / NAME}.pdf/.png")

    # numbers for the caption / index
    for ps, mode in [("true_belief", "inject"), ("false_belief", "delete")]:
        acc, _ = series(out, ps, mode, "target_rank1", "accuracy")
        alt, _ = series(out, ps, mode, "target_rank1", "alt_rate")
        off, _ = series(out, ps, mode, "target_rank1", "off_manifold")
        print(f"{ps} {mode} target: acc% {np.round(100 * acc, 1).tolist()}")
        print(f"{' ' * len(ps)} alt% {np.round(100 * alt, 1).tolist()}  off% {np.round(100 * off, 1).tolist()}")
    for d in ["target_rank1", "random", "shuffled", "heads_ctl"]:
        kl, _ = series(gen, "generic", "inject", d, "kl")
        t1, _ = series(gen, "generic", "inject", d, "top1_agree")
        print(f"generic {d:12s} KL {np.round(kl, 3).tolist()}  top1% {np.round(100 * t1, 1).tolist()}")


if __name__ == "__main__":
    main()
