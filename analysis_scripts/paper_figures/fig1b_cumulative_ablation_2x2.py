"""Figure 1b — cumulative ablation, 2 x 2: redirection AND correctness.

Camera-ready redesign of fig1 (Sept 9). Rows: flip score phi_p (does the
answer reorient?) and a correctness metric (does the model still get it
right?). Columns: theory of mind (false / true belief) and Python program
state (code_copy / code_alias), both under the C1 belief-fingerprint head
order, both against the rank-aligned layer-matched control. Built twice:

  --metric accuracy   second row = held-out accuracy (argmax == correct)
  --metric pcorrect   second row = mean next-token probability of the correct
                      first token (not renormalised over the two candidates)

The ranking comparison (belief vs retrieval order) that was panel (c) of
fig1 moves to the appendix.

Data: results/ablation/tom_c1_full1920_on_code_tom_prefill_per_variant/
(matched system prompt + answer prefill, n = 75 prompts/variant, seed 1).
SEM over prompts from per_prompt_data.pt.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, AQUA, GRAY, INK2, apply_style, despine, panel_label,
                   save)

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/ablation/tom_c1_full1920_on_code_tom_prefill_per_variant"
EPS = 1e-12
XMAX = 64
DEPART = {"tom": 19, "code": 51}   # k at which the control leaves baseline
COLS = {
    "tom": ("Theory of mind", [
        ("target", "false_belief", BLUE, "-", "false belief (diverged)"),
        ("target", "true_belief", AQUA, "-", "true belief"),
        ("control", "false_belief", GRAY, "--", "layer-matched control"),
    ]),
    "code": ("Python program state, same head order", [
        ("target", "code_copy", BLUE, "-", "code_copy (diverged)"),
        ("target", "code_alias", AQUA, "-", "code_alias"),
        ("control", "code_copy", GRAY, "--", "layer-matched control"),
    ]),
}
METRIC = {
    "flip": (r"flip score $\phi_p$", "flip_score"),
    "accuracy": ("accuracy (held-out prompts)", "accuracy"),
    "pcorrect": ("p(correct first token)", "mean_p_correct"),
}


def per_prompt(per, curve, variant, metric):
    d = per[curve][variant]
    if metric == "flip":
        pc, pa = d["p_correct"].numpy(), d["p_alt"].numpy()
        x = pa / (pa + pc + EPS)
    elif metric == "accuracy":
        x = d["acc"].numpy().astype(float)
    else:
        x = d["p_correct"].numpy()
    k = np.asarray(per[f"{curve}_k_values"])
    idx = np.argsort(k)
    return k[idx], x[:, idx]


def curve(df, per, curve_name, variant, metric):
    k, x = per_prompt(per, curve_name, variant, metric)
    mean = x.mean(axis=0)
    sem = x.std(axis=0, ddof=1) / np.sqrt(x.shape[0])
    col = METRIC[metric][1]
    ref = (df[(df.curve == curve_name) & (df.eval_variant == variant)]
           .sort_values("k")[col].to_numpy())
    assert np.allclose(mean, ref, atol=1e-6), (curve_name, variant, metric)
    return k, mean, sem


def draw(ax, k, m, s, color, ls, lw):
    sel = k <= XMAX
    ax.plot(k[sel], m[sel], color=color, ls=ls, lw=lw, zorder=3,
            solid_capstyle="round")
    ax.fill_between(k[sel], (m - s)[sel], (m + s)[sel], color=color,
                    alpha=0.15, lw=0, zorder=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=["accuracy", "pcorrect"],
                    default="accuracy")
    args = ap.parse_args()
    second = args.metric

    apply_style()
    df = pd.read_csv(RUN / "cumulative_ablation_results.csv")
    per = torch.load(RUN / "per_prompt_data.pt", map_location="cpu",
                     weights_only=False)

    fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.0), sharex=True,
                             sharey="row",
                             gridspec_kw={"hspace": 0.12, "wspace": 0.08})
    for j, dom in enumerate(("tom", "code")):
        title, series = COLS[dom]
        for i, metric in enumerate(("flip", second)):
            ax = axes[i, j]
            for curve_name, variant, color, ls, label in series:
                k, m, s = curve(df, per, curve_name, variant, metric)
                draw(ax, k, m, s, color, ls, 1.8 if curve_name == "target" else 1.4)
            kd = DEPART[dom]
            ax.axvspan(kd, XMAX, color="#f4f3f0", zorder=0)
            ax.axvline(kd, color=INK2, lw=0.7, ls=":", alpha=0.7)
            despine(ax)
            ax.set_xlim(0, XMAX)
            ax.set_xticks([0, 16, 32, 48, 64])
            if metric == "flip":
                ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (1, 2)), alpha=0.6,
                           zorder=1)
                ax.set_ylim(0, 1.0)
                ax.set_title(title, loc="left")
                ax.text(kd, 1.02, f"control leaves baseline (k={kd}) $\\rightarrow$",
                        fontsize=7, color=INK2, va="bottom", ha="left",
                        clip_on=False)
            else:
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("heads ablated, k (of 1,920), belief-fingerprint order")
            if j == 0:
                ax.set_ylabel(METRIC[metric][0])
        panel_label(axes[0, j], "ab"[j], x=-0.17 if j == 0 else -0.06)
        panel_label(axes[1, j], "cd"[j], x=-0.17 if j == 0 else -0.06)
        # legend inside the flip panel (upper left is empty in both domains)
        handles = [plt.Line2D([], [], color=c, ls=ls, lw=1.8 if cn == "target" else 1.4,
                              label=lab)
                   for cn, _, c, ls, lab in series]
        axes[0, j].legend(handles=handles, loc="upper left", fontsize=7,
                          handlelength=2.0, labelspacing=0.3)

    # anchor numbers the text quotes
    k, m, _ = curve(df, per, "target", "false_belief", "flip")
    for kk in (3, 8):
        yy = m[k == kk][0]
        axes[0, 0].plot([kk], [yy], "o", color=BLUE, ms=4, zorder=5)
        axes[0, 0].annotate(f"k={kk}: {yy:.2f}", (kk, yy),
                            xytext=(6, -10 if kk == 3 else 5),
                            textcoords="offset points", fontsize=7.5,
                            color=INK2)
    k, m, _ = curve(df, per, "target", "code_copy", second)
    k0, m0, _ = curve(df, per, "target", "code_copy", "flip")
    y16, f16 = m[k == 16][0], m0[k0 == 16][0]
    axes[1, 1].plot([16], [y16], "o", color=BLUE, ms=4, zorder=5)
    axes[1, 1].annotate(f"k=16: {y16:.2f}", (16, y16), xytext=(8, 6),
                        textcoords="offset points", fontsize=7.5, color=INK2)
    axes[0, 1].plot([16], [f16], "o", color=BLUE, ms=4, zorder=5)
    axes[0, 1].annotate(f"k=16: {f16:.2f}", (16, f16), xytext=(6, 5),
                        textcoords="offset points", fontsize=7.5, color=INK2)

    save(fig, f"fig1b_ablation_2x2_{second}")

    for dom in ("tom", "code"):
        for cn, var, *_ in COLS[dom][1]:
            for metric in ("flip", second):
                k, m, _ = curve(df, per, cn, var, metric)
                pts = {kk: round(float(m[k == kk][0]), 3) for kk in (0, 3, 8, 16, 32, 64)}
                print(f"{dom:4s} {cn:7s} {var:12s} {metric:8s} {pts}")


if __name__ == "__main__":
    main()
