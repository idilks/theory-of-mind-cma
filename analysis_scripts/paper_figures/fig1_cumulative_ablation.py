"""Rebuttal figure 1 — cumulative ablation over all 1,920 heads (R3 point 5).

Answers "does the effect depend on a specific number of heads?":
(a) ToM false belief under the belief-fingerprint (C1) ordering — graded,
    saturates by k~8, layer-matched control flat through k=18;
(b) program-state domain (code_copy vs code_alias) under the same ordering;
(c) ordering matters — belief (C1) vs retrieval (C5) ranking on false belief.

Data: other_results/ablation/tom_c{1,5}_full1920_on_code_tom_prefill_per_variant/
(matched system prompt + answer prefill, n=75 prompts/variant, seed 1).
SEM ribbons from per_prompt_data.pt.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, ORANGE, AQUA, GRAY, INK2, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
RUN_C1 = REPO / "other_results/ablation/tom_c1_full1920_on_code_tom_prefill_per_variant"
RUN_C5 = REPO / "other_results/ablation/tom_c5_full1920_on_code_tom_prefill_per_variant"

EPS = 1e-12


def load_run(run_dir):
    df = pd.read_csv(run_dir / "cumulative_ablation_results.csv")
    per = torch.load(run_dir / "per_prompt_data.pt", map_location="cpu",
                     weights_only=False)
    return df, per


def curve(df, per, curve_name, variant):
    """(k, mean flip score, sem) with SEM over the 75 per-prompt ratios."""
    sub = df[(df.curve == curve_name) & (df.eval_variant == variant)]
    sub = sub.sort_values("k")
    k = sub.k.to_numpy()
    mean = sub.flip_score.to_numpy()
    d = per[curve_name][variant]
    pc, pa = d["p_correct"].numpy(), d["p_alt"].numpy()   # (n_prompts, n_k)
    ratio = pa / (pa + pc + EPS)
    k_order = np.asarray(per[f"{curve_name}_k_values"])
    assert np.array_equal(np.sort(k_order), k), "k grids disagree"
    idx = np.argsort(k_order)
    sem = (ratio.std(axis=0, ddof=1) / np.sqrt(ratio.shape[0]))[idx]
    mean_pp = ratio.mean(axis=0)[idx]
    assert np.allclose(mean_pp, mean, atol=1e-6), "per-prompt mean != CSV"
    return k, mean, sem


def draw(ax, k, m, s, color, ls="-", lw=1.8, ribbon=True, zorder=3):
    ax.plot(k, m, color=color, ls=ls, lw=lw, zorder=zorder,
            solid_capstyle="round")
    if ribbon:
        ax.fill_between(k, m - s, m + s, color=color, alpha=0.15, lw=0,
                        zorder=zorder - 1)


def label_end(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points",
                color=color, fontsize=8.5, va="center", ha="left",
                fontweight="medium")


def tied_line(ax):
    ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (1, 2)), alpha=0.6, zorder=1)


def main():
    apply_style()
    df1, per1 = load_run(RUN_C1)
    df5, per5 = load_run(RUN_C5)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), sharey=True)
    a, b, c = axes

    # ---- (a) ToM, belief ordering -------------------------------------
    xmax_a = 64
    for name, variant, color, ls in [
        ("target", "false_belief", BLUE, "-"),
        ("control", "false_belief", GRAY, "--"),
        ("target", "true_belief", AQUA, "-"),
    ]:
        k, m, s = curve(df1, per1, name, variant)
        sel = k <= xmax_a
        draw(a, k[sel], m[sel], s[sel], color, ls=ls,
             lw=1.8 if name == "target" else 1.4)
    a.axvspan(19, xmax_a, color="#f4f3f0", zorder=0)
    k, m, _ = curve(df1, per1, "target", "false_belief")
    label_end(a, xmax_a, m[k <= xmax_a][-1], "false belief", BLUE, dy=6)
    k, m, _ = curve(df1, per1, "target", "true_belief")
    label_end(a, xmax_a, m[k <= xmax_a][-1], "true belief", AQUA, dy=-6)
    k, m, _ = curve(df1, per1, "control", "false_belief")
    label_end(a, xmax_a, m[k <= xmax_a][-1], "layer-matched\ncontrol", GRAY)

    # annotate the early rise
    k, m, _ = curve(df1, per1, "target", "false_belief")
    for kk in (3, 8):
        yy = m[np.where(k == kk)[0][0]]
        a.plot([kk], [yy], "o", color=BLUE, ms=4.5, zorder=5)
        a.annotate(f"k={kk}: {yy:.2f}", (kk, yy), xytext=(6, -11 if kk == 3 else 6),
                   textcoords="offset points", fontsize=8, color=INK2)
    a.axvline(19, color=INK2, lw=0.7, ls=":", alpha=0.7)
    a.text(20.5, 0.97, "control leaves\nbaseline (k=19)", fontsize=7.5,
           color=INK2, va="top")
    tied_line(a)
    a.set_xlim(0, xmax_a)
    a.set_xticks([0, 16, 32, 48, 64])
    a.set_title("Theory of mind — belief-fingerprint order", loc="left")
    a.set_xlabel("heads ablated, k (of 1,920)")
    a.set_ylabel(r"flip score $\phi_p$")
    panel_label(a, "a", x=-0.18)

    # ---- (b) code domain, belief ordering ------------------------------
    xmax_b = 128
    for name, variant, color, ls in [
        ("target", "code_copy", BLUE, "-"),
        ("control", "code_copy", GRAY, "--"),
        ("target", "code_alias", AQUA, "-"),
    ]:
        k, m, s = curve(df1, per1, name, variant)
        sel = k <= xmax_b
        draw(b, k[sel], m[sel], s[sel], color, ls=ls,
             lw=1.8 if name == "target" else 1.4)
    b.axvspan(51, xmax_b, color="#f4f3f0", zorder=0)
    k, m, _ = curve(df1, per1, "target", "code_copy")
    label_end(b, xmax_b, m[k <= xmax_b][-1], "code_copy", BLUE)
    yy32 = m[np.where(k == 32)[0][0]]
    b.plot([32], [yy32], "o", color=BLUE, ms=4.5, zorder=5)
    b.annotate(f"k=32: {yy32:.2f}", (32, yy32), xytext=(6, 5),
               textcoords="offset points", fontsize=8, color=INK2)
    k, m, _ = curve(df1, per1, "target", "code_alias")
    label_end(b, xmax_b, m[k <= xmax_b][-1], "code_alias", AQUA, dy=-6)
    k, m, _ = curve(df1, per1, "control", "code_copy")
    label_end(b, xmax_b, m[k <= xmax_b][-1], "layer-matched\ncontrol", GRAY, dy=10)
    b.axvline(51, color=INK2, lw=0.7, ls=":", alpha=0.7)
    b.text(53, 0.97, "control leaves\nbaseline (k=51)", fontsize=7.5,
           color=INK2, va="top")
    tied_line(b)
    b.set_xlim(0, xmax_b)
    b.set_xticks([0, 32, 64, 96, 128])
    b.set_title("Program state — same head order", loc="left")
    b.set_xlabel("heads ablated, k (of 1,920)")
    panel_label(b, "b", x=-0.10)

    # ---- (c) ordering comparison ---------------------------------------
    xmax_c = 32
    k1, m1, s1 = curve(df1, per1, "target", "false_belief")
    k5, m5, s5 = curve(df5, per5, "target", "false_belief")
    sel1, sel5 = k1 <= xmax_c, k5 <= xmax_c
    draw(c, k1[sel1], m1[sel1], s1[sel1], BLUE)
    draw(c, k5[sel5], m5[sel5], s5[sel5], ORANGE)
    label_end(c, xmax_c, m1[sel1][-1], "belief\nranking", BLUE, dy=9)
    label_end(c, xmax_c, m5[sel5][-1], "retrieval\nranking", ORANGE, dy=-9)
    c.axhline(0.2, color=INK2, lw=0.7, ls=":", alpha=0.7)
    c.text(13.0, 0.215, r"$\phi_p$ = 0.2", fontsize=7.5, color=INK2)
    c.plot([3], [m1[np.where(k1 == 3)[0][0]]], "o", color=BLUE, ms=4.5, zorder=5)
    c.annotate("k=3", (3, m1[np.where(k1 == 3)[0][0]]), xytext=(6, 2),
               fontsize=8, textcoords="offset points", color=BLUE)
    c.plot([23], [m5[np.where(k5 == 23)[0][0]]], "o", color=ORANGE, ms=4.5,
           zorder=5)
    c.annotate("k=23", (23, m5[np.where(k5 == 23)[0][0]]), xytext=(4, -12),
               fontsize=8, textcoords="offset points", color=ORANGE)
    tied_line(c)
    c.set_xlim(0, xmax_c)
    c.set_xticks([0, 8, 16, 24, 32])
    c.set_title("False belief — head ordering matters", loc="left")
    c.set_xlabel("heads ablated, k (of 1,920)")
    panel_label(c, "c", x=-0.10)

    for ax in axes:
        despine(ax)
        ax.set_ylim(0, 1.0)

    fig.tight_layout(w_pad=2.2)
    save(fig, "fig1_cumulative_ablation")

    # console check against rebuttal-r3 paragraph 5
    k, m, _ = curve(df1, per1, "target", "false_belief")
    for kk in (0, 2, 3, 8):
        print(f"fb belief-rank k={kk}: {m[np.where(k == kk)[0][0]]:.3f}")
    k5v, m5v, _ = curve(df5, per5, "target", "false_belief")
    first = k5v[np.where(m5v >= 0.2)[0][0]]
    print(f"retrieval ranking first k with flip>=0.2: {first}")


if __name__ == "__main__":
    main()
