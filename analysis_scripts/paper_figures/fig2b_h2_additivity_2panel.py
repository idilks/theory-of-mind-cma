"""Figure 2b — additive decomposition, split into two figures (Oct 2026).

`fig2b_h2_scatter` (main text, beside the C3 patch frame drawn in TikZ):
    the scatter below, with the C1 + C5 / C1 / C5 correlations printed in
    the corner so the H2 comparison stays visible without the bars.
`fig2b_h2_specificity` (appendix): the bar panel below.

Original two-panel design:

Camera-ready redesign of fig2 (Sept 9): keep only what the claim needs.
(a) C3 against C1 + C5 over all 1,920 heads, raw scores, y = x line, with
    r raw / clamped inline;
(b) specificity: r with C3 for C1 + C5, C1 alone, the axis-matched
    substitute C2 + C6, and C5 alone (clamped bar, raw diamond) — the
    answer to the design-axis confound (R1 pt 4).

Everything else in fig2 (residual scaling, permutation nulls, OLS
coefficients, Gemma scatter) moves to the appendix.

Data: results/aggregated/fingerprints/*.npy; correlations computed here.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, ORANGE, GRAY, INK, INK2, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
DARK_BLUE = "#184f95"  # dark end of style.CMAP_SEQ
FP = REPO / "results/aggregated/fingerprints"
NAMES = {"C1": "C1_abstract_belief", "C2": "C2_abstract_photo",
         "C3": "C3_answer_changes_belief",
         "C5": "C5_answer_changes_location_only",
         "C6": "C6_answer_changes_location_only_photo"}


def load(cond, kind):
    return np.load(FP / f"{NAMES[cond]}_{kind}.npy").ravel()


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def main():
    apply_style()
    raw = {c: load(c, "raw") for c in NAMES}
    cl = {c: load(c, "clamped") for c in NAMES}
    n = len(raw["C3"])
    assert n == 1920

    fig, a = plt.subplots(figsize=(3.7, 3.3))

    # ---- scatter, main text ---------------------------------------------------
    x, y = raw["C1"] + raw["C5"], raw["C3"]
    r_raw, r_cl = r(x, y), r(cl["C1"] + cl["C5"], cl["C3"])
    a.scatter(x, y, color=BLUE, s=9, alpha=0.7, lw=0, zorder=3)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = 0.04 * (hi - lo)
    a.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.9,
           ls="--", zorder=2)
    a.set_xlim(lo - pad, hi + pad)
    a.set_ylim(lo - pad, hi + pad)
    a.set_xlabel("divergence- + answer-change score (raw)")
    a.set_ylabel("dual-change score (raw)")
    # Nov 6: the r box (clamped/raw for sum and parts) was dropped; the main
    # text reports those values. Only the identity line is labelled.
    a.text(0.97, 0.04, "dashed line: $x = y$", transform=a.transAxes,
           va="bottom", ha="right", fontsize=7.5, color=INK2)
    despine(a)
    fig.tight_layout()
    save(fig, "fig2b_h2_scatter")

    # ---- (b) specificity, appendix ---------------------------------------------
    # Oct 22 restyle to match fig11 / steering_attention: plain left title,
    # colours by role (belief part BLUE, answer part ORANGE, their sum in the
    # dark end of the same blue ramp, the axis-matched substitute GRAY as a
    # control), raw r as a hollow diamond on the bar's own row, and the gain of
    # the sum over its better part marked between the two top bars.
    fig, b = plt.subplots(figsize=(3.7, 2.45))
    rows = [("divergence- + answer-", "change, belief",
             cl["C1"] + cl["C5"], raw["C1"] + raw["C5"], DARK_BLUE),
            ("divergence-change", "belief", cl["C1"], raw["C1"], BLUE),
            ("divergence- + answer-", "change, photo",
             cl["C2"] + cl["C6"], raw["C2"] + raw["C6"], GRAY),
            ("answer-change", "belief", cl["C5"], raw["C5"], ORANGE)]
    ys = np.arange(len(rows))[::-1].astype(float)
    vals = []
    for yv, (cond, desc, c_, r_, color) in zip(ys, rows):
        rc, rr = r(c_, cl["C3"]), r(r_, raw["C3"])
        vals.append((cond, rc, rr))
        b.barh(yv, rc, height=0.62, color=color, zorder=3)
        b.plot([rr], [yv], "D", color=INK, mfc="white", ms=4.2, mew=0.9,
               zorder=4)
        b.text(0.02, yv, f"{rc:.3f}", va="center", ha="left", fontsize=7.5,
               color="white", zorder=5)
    b.set_yticks(ys)
    b.set_yticklabels([f"{cond}\n{desc}" for cond, desc, *_ in rows],
                      fontsize=7.5, linespacing=1.15)
    for tick in b.get_yticklabels():
        tick.set_color(INK)
    # gain of the sum over its better single part (the H2 prediction)
    top, part = vals[0][1], vals[1][1]
    xg = 1.035
    b.annotate("", xy=(xg, ys[0]), xytext=(xg, ys[1]),
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                               shrinkA=0, shrinkB=0), annotation_clip=False)
    for yv, xv in ((ys[0], top), (ys[1], part)):
        b.plot([xv + 0.012, xg], [yv, yv], color=INK2, lw=0.6, ls=":",
               clip_on=False)
    b.text(xg + 0.015, (ys[0] + ys[1]) / 2, f"+{top - part:.3f}", va="center",
           ha="left", fontsize=7.5, color=INK2, clip_on=False)
    b.set_xlim(0, 1.0)
    b.set_ylim(ys[-1] - 0.5, ys[0] + 0.5)
    b.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    b.set_xlabel("Pearson $r$ with dual-change belief (1,920 heads)")
    b.legend(handles=[Patch(color="#c3c2b7", label="clamped scores"),
                      Line2D([], [], ls="", marker="D", color=INK, mfc="white",
                             ms=4.2, mew=0.9, label="raw scores")],
             loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
             fontsize=7.5, handlelength=1.2, columnspacing=1.4,
             borderaxespad=0.2)
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", length=0)
    despine(b)
    fig.tight_layout()
    save(fig, "fig2b_h2_specificity")

    print(f"scatter: r raw {r_raw:.3f}, clamped {r_cl:.3f}")
    for label, rc, rr in vals:
        print(f"{label.splitlines()[0]:10s} clamped {rc:.3f} raw {rr:.3f}")


if __name__ == "__main__":
    main()
