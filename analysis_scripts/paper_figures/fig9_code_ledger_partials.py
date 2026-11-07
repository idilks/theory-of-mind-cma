"""Figure 9 — code-ledger RSA: object-divergence vs answer structure by head set.

Camera-ready planning figure (Sept 8). On the code_ledger stimulus grid the
two model RDMs — "has the queried object's state diverged from its initial
binding" (object divergence) and "which answer token is correct" (answer) —
are orthogonal (r = 0.004), so each head set's neural RDM can be scored on
both without one masquerading as the other. Partial Spearman correlations
(rank-residualised) are plotted: object | answer and answer | object.

Prediction from the two-population account: divergence-tracking sets
(code_abstract, ToM C1, Gemma C1/C2) load on object divergence; answer-
retrieval sets (code_control, ToM C5, Gemma C5) load on answer identity.

Data: results/rsa_head_outputs/aggregation_code_ledger/partials_object_vs_answer.csv
(Qwen2.5-14B-Instruct and Gemma-2-27B-it captures; permutation p with 500
draws, floor 0.002; maxnull = largest |partial| over the null draws).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, ORANGE, GRAY, INK2, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
SRC = (REPO / "results/rsa_head_outputs/aggregation_code_ledger/"
       "partials_object_vs_answer.csv")

ROWS = [  # (model, head_set, label, population)
    ("qwen", "code_abstract", "code divergence heads (12)", "div"),
    ("qwen", "tom_c1", "ToM belief heads, C1 (32)", "div"),
    ("qwen", "code_control", "code retrieval heads (14)", "ret"),
    ("qwen", "tom_c5", "ToM retrieval heads, C5 (22)", "ret"),
    ("gemma", "gemma_c1", "belief heads, gC1 (21)", "div"),
    ("gemma", "gemma_c2", "photo heads, gC2 (32)", "div"),
    ("gemma", "gemma_c3", "belief+answer heads, gC3 (50)", "mix"),
    ("gemma", "gemma_c5", "retrieval heads, gC5 (37)", "ret"),
]


def main():
    apply_style()
    df = pd.read_csv(SRC).set_index(["model", "head_set"])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1]})
    h = 0.36
    for ax, model, letter, title in zip(
            axes, ("qwen", "gemma"), "ab",
            ("Qwen2.5-14B", "Gemma-2-27B-it")):
        rows = [r for r in ROWS if r[0] == model]
        ys = np.arange(len(rows))[::-1]
        for y, (_, key, label, pop) in zip(ys, rows):
            r = df.loc[(model, key)]
            ax.barh(y + h / 2, r["obj|ans"], height=h, color=BLUE, zorder=3)
            ax.barh(y - h / 2, r["ans|obj"], height=h, color=ORANGE, zorder=3)
            # null ceiling ticks (largest |partial| across permutations)
            ax.plot([r["maxnull_obj|ans"]] * 2, [y + h * 0.1, y + h * 0.9],
                    color=INK2, lw=1.0, zorder=4)
            ax.plot([r["maxnull_ans|obj"]] * 2, [y - h * 0.9, y - h * 0.1],
                    color=INK2, lw=1.0, zorder=4)
            ax.text(r["obj|ans"] + 0.008, y + h / 2, f"{r['obj|ans']:.2f}",
                    va="center", fontsize=7.2, color=BLUE)
            ax.text(r["ans|obj"] + 0.008, y - h / 2, f"{r['ans|obj']:.2f}",
                    va="center", fontsize=7.2, color=ORANGE)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[2] for r in rows], fontsize=8)
        ax.set_title(title, loc="left")
        ax.set_xlabel("partial Spearman r, neural RDM vs model RDM")
        ax.set_xlim(0, 0.55)
        ax.grid(axis="y", visible=False)
        despine(ax)
        panel_label(ax, letter, x=-0.62 if letter == "a" else -0.60)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE,
                      label="object divergence | answer"),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE,
                      label="answer identity | object divergence"),
        plt.Line2D([], [], color=INK2, lw=1.0,
                   label="largest |r| over 500 label permutations"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1), w_pad=4.0)
    save(fig, "fig9_code_ledger_partials")

    for model, key, label, _ in ROWS:
        r = df.loc[(model, key)]
        print(f"{model:5s} {key:14s} obj|ans {r['obj|ans']:.3f} "
              f"(p {r['p_obj|ans']:.3f}, null max {r['maxnull_obj|ans']:.3f})  "
              f"ans|obj {r['ans|obj']:.3f} (p {r['p_ans|obj']:.3f}, "
              f"null max {r['maxnull_ans|obj']:.3f})")


if __name__ == "__main__":
    main()
