"""Rebuttal figure 3 — Gemma-2-27B-it replication (R1 pt 1, R2 Quality 2, R3 pt 2).

(a) two-population architecture in both models: every FWER-significant head
    (>= half of a condition's cells) at its fractional depth; divergence
    band (C1/C2) below the answer-retrieval band (C5) in the same order in
    both models, nothing in the lower network
(b) the three hypothesis-level correlations, Qwen vs Gemma
(c) the code-domain split, Qwen vs Gemma — the belief/retrieval structure
    survives a change of model family and task domain at once

Selection rule identical across models: significant in >= half of the
condition's cells (Qwen 27/11/38/16 heads for C1/C2/C3/C5 of 1,920;
Gemma 21/32/50/37 of 1,472).
"""

import csv as csvmod
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from h2_additivity_robustness import (  # noqa: E402
    CROSS_RULES, WITHIN_RULES, GEMMA_ROOT, GEMMA_SHAPE, build_fp,
    load_tom_fp)
from build_fingerprint_scatters import tom_sig_set, TOM_CONDS  # noqa: E402
from style import (BLUE, ORANGE, AQUA, VIOLET, GRAY, INK, INK2, apply_style,
                   despine, panel_label, save)

HEADSETS = REPO / "results/aggregated/code_head_sets"
SCATTER_STATS = REPO / "results/summary/fingerprint_scatters/stats_scatter_pairs.csv"
H2_STATS = REPO / "results/summary/h2_robustness/stats_correlations.csv"
CACHE = REPO / "results" / "figures" / "_cache_gemma_code_corrs.csv"

QWEN_LAYERS, GEMMA_LAYERS = 48, GEMMA_SHAPE[0]
COND_COLOR = {"C1": BLUE, "C2": AQUA, "C3": VIOLET, "C5": ORANGE}
COND_NAME = {"C1": "belief", "C2": "photo", "C3": "belief\n+answer",
             "C5": "answer\nretrieval"}


def qwen_heads():
    """(layer, head, score) per condition under the >=half-of-cells rule."""
    fp = {c: load_tom_fp(f"{c}_{TOM_CONDS[c][0]}")["clamped"]
          for c in ("C1", "C2", "C3", "C5")}
    out = {}
    for c in ("C1", "C2", "C3", "C5"):
        heads = tom_sig_set(*TOM_CONDS[c])
        out[c] = [(l, h, fp[c][l * 40 + h]) for l, h in sorted(heads)]
    return out


def gemma_heads():
    out = {}
    for c in ("C1", "C2", "C3", "C5"):
        rows = list(csvmod.DictReader(open(HEADSETS / f"gemma_c{c[1]}_head_list.csv")))
        out[c] = [(int(r["layer"]), int(r["head"]), float(r["mean_score"]))
                  for r in rows]
    return out


def gemma_code_corrs():
    """r(gcode_abs, gC1) etc — cached because the cell loads take ~30 s."""
    if CACHE.exists():
        return pd.read_csv(CACHE).set_index("pair")
    from h2_additivity_robustness import build_gemma_fp
    gabs, _ = build_fp(GEMMA_ROOT, "code_abstract", CROSS_RULES, GEMMA_SHAPE)
    gctl, _ = build_fp(GEMMA_ROOT, "code_control", WITHIN_RULES, GEMMA_SHAPE)
    g1, _ = build_gemma_fp("abstract_belief", CROSS_RULES)
    g5, _ = build_gemma_fp("answer_changes_location_only", WITHIN_RULES)
    def r(a, b): return float(np.corrcoef(a, b)[0, 1])
    rows = []
    for pair, x, y in [("gcode_abs~gC1", gabs, g1), ("gcode_ctl~gC5", gctl, g5),
                       ("gcode_abs~gcode_ctl", gabs, gctl),
                       ("gcode_abs~gC5", gabs, g5), ("gcode_ctl~gC1", gctl, g1)]:
        rows.append({"pair": pair,
                     "clamped": r(x["clamped"], y["clamped"]),
                     "raw": r(x["raw"], y["raw"])})
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df.set_index("pair")


def strip(ax, heads_by_cond, n_layers, model_label, floor_layer, rng):
    for i, c in enumerate(("C1", "C2", "C3", "C5")):
        rows = heads_by_cond[c]
        depth = np.array([l for l, _, _ in rows]) / n_layers
        score = np.abs([s for _, _, s in rows])   # |score|: sig heads can be negative
        size = 8 + 60 * (score / score.max())
        xj = i + rng.uniform(-0.17, 0.17, len(rows))
        ax.scatter(xj, depth, s=size, color=COND_COLOR[c], alpha=0.55, lw=0)
        med = np.median(depth)
        ax.plot([i - 0.28, i + 0.28], [med, med], color=INK, lw=1.6, zorder=5)
    ax.axhline(floor_layer / n_layers, color=INK2, lw=0.8, ls=":")
    ax.text(3.45, floor_layer / n_layers - 0.015,
            f"no significant head below L{floor_layer}/{n_layers}",
            fontsize=7, color=INK2, ha="right", va="top")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [f"{c}\n{COND_NAME[c]}\nn={len(heads_by_cond[c])}"
         for c in ("C1", "C2", "C3", "C5")], fontsize=7.5)
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(model_label, loc="left")


def dumbbell(ax, rows, title):
    """rows: (label, qwen_r, gemma_r, color)."""
    ys = np.arange(len(rows))[::-1]
    for yv, (label, rq, rg, color) in zip(ys, rows):
        ax.plot([rq, rg], [yv, yv], color=color, lw=1.2, alpha=0.6, zorder=2)
        ax.plot([rq], [yv], "o", color=color, ms=7, zorder=3)
        ax.plot([rg], [yv], "s", color=color, ms=6.5, mfc="white", mew=1.6,
                zorder=3)
        ax.text(max(rq, rg) + 0.045, yv, f"{rq:.2f} / {rg:.2f}", va="center",
                fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 1.28)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(title, loc="left")


def main():
    apply_style()
    rng = np.random.default_rng(7)

    qh, gh = qwen_heads(), gemma_heads()
    sp = pd.read_csv(SCATTER_STATS).set_index("pair")
    h2 = pd.read_csv(H2_STATS)
    gcode = gemma_code_corrs()

    qwen_h2 = float(h2[(h2.domain == "ToM") & (h2.key == "clamped")
                       & (h2.pair == "C3~sum(C1+C5)")].pearson_r.iloc[0])

    fig = plt.figure(figsize=(11.8, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.5], wspace=0.55,
                          hspace=0.9, left=0.06, right=0.985, top=0.90,
                          bottom=0.15)
    ax_q = fig.add_subplot(gs[:, 0])
    ax_g = fig.add_subplot(gs[:, 1], sharey=ax_q)
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 2])

    # ---- (a) two-population depth structure ----------------------------
    strip(ax_q, qh, QWEN_LAYERS, "Qwen2.5-14B (48 layers)", 29, rng)
    strip(ax_g, gh, GEMMA_LAYERS, "Gemma-2-27B-it (46 layers)", 22, rng)
    ax_q.set_ylabel("fractional depth  (layer / n layers)")
    plt.setp(ax_g.get_yticklabels(), visible=False)
    panel_label(ax_q, "a", x=-0.22)

    # ---- (b) hypothesis correlations ------------------------------------
    dumbbell(ax_b, [
        ("Dissociation\nr(C1, C5) — low", sp.loc["C1~C5", "pearson_r_clamped"],
         sp.loc["gC1~gC5", "pearson_r_clamped"], BLUE),
        ("Domain generality\nr(C1, C2) — high", sp.loc["C1~C2", "pearson_r_clamped"],
         sp.loc["gC1~gC2", "pearson_r_clamped"], AQUA),
        ("Additivity\nr(C3, C1+C5) — high", qwen_h2,
         sp.loc["gC3~gSUM_C1_C5", "pearson_r_clamped"], VIOLET),
    ], "Fingerprint relationships replicate")
    ax_b.plot([], [], "o", color=INK, label="Qwen", ls="none")
    ax_b.plot([], [], "s", color=INK, mfc="white", label="Gemma", ls="none")
    ax_b.legend(loc="lower left", fontsize=7.5, handletextpad=0.3,
                borderaxespad=0.2)
    panel_label(ax_b, "b", x=-0.30)

    # ---- (c) code domain in both models ---------------------------------
    dumbbell(ax_c, [
        ("code divergence\n~ C1 belief", sp.loc["code_abs~C1", "pearson_r_clamped"],
         gcode.loc["gcode_abs~gC1", "clamped"], BLUE),
        ("code control\n~ C5 retrieval", sp.loc["code_ctl~C5", "pearson_r_clamped"],
         gcode.loc["gcode_ctl~gC5", "clamped"], ORANGE),
        ("code divergence\n~ code control", sp.loc["code_abs~code_ctl", "pearson_r_clamped"],
         gcode.loc["gcode_abs~gcode_ctl", "clamped"], GRAY),
    ], "Code-domain split re-forms in both models")
    ax_c.set_xlabel("Pearson r over all heads (clamped)")
    panel_label(ax_c, "c", x=-0.30)

    for ax in (ax_q, ax_g, ax_b, ax_c):
        despine(ax)
    ax_q.grid(axis="x", visible=False)
    ax_g.grid(axis="x", visible=False)

    fig.savefig  # layout fixed via gridspec margins; no tight_layout
    save(fig, "fig3_gemma_replication")

    for c in ("C1", "C2", "C3", "C5"):
        mq = np.median([l for l, _, _ in qh[c]]) / QWEN_LAYERS
        mg = np.median([l for l, _, _ in gh[c]]) / GEMMA_LAYERS
        print(f"{c}: n_qwen={len(qh[c])} depth_qwen={mq:.2f} | "
              f"n_gemma={len(gh[c])} depth_gemma={mg:.2f}")


if __name__ == "__main__":
    main()
