"""Figure 0 — paradigm figure: task structure, procedure, hypothesised mechanism.

Reviewer 3 (pts 6-7): "a paradigm-setting figure showcasing the trial and
information flow would have been nice." Camera-ready Figure 1 candidate.

(a) one structure, three frames — belief, photograph, Python: a representation
    is formed, the world changes, a query asks about the representation; the
    only thing that varies is whether the representation diverged from the
    world.
(b) procedure — matched prompt pairs, single-head activation patching at the
    final token over all 48 x 40 heads, FWER permutation test, head sets,
    then held-out interventions (ablation, steering) and RSA. The inset is
    the real C1 belief fingerprint (clamped, condition-averaged).
(c) hypothesised mechanism — a mid-layer divergence-tracking population
    writes a low-rank divergence direction into the residual stream (~L41);
    late answer-retrieval heads read it through their query weights and fetch
    the believed or the current location; the two contributions add (H2).

Half-text-width-ish, ~3 in tall. Output: paper_figures/fig0_design_schematic.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, ORANGE, AQUA, GRAY, INK, INK2, GRID, CMAP_SEQ,
                   apply_style, panel_label, save)

REPO = Path(__file__).resolve().parents[2]
FP_C1 = REPO / "results/aggregated/fingerprints/C1_abstract_belief_clamped.npy"

LIGHT_BLUE, LIGHT_ORANGE, LIGHT_AQUA = "#dbe8fa", "#fbe3d8", "#d3f1e6"
PAPER = "#f7f6f2"


def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=6.8, color=INK,
        lw=0.7, weight="normal", style="round,pad=0.008,rounding_size=0.015",
        ha="center", zorder=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                       zorder=zorder, mutation_aspect=1)
    ax.add_patch(p)
    tx = x + w / 2 if ha == "center" else x + 0.012
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=color,
            fontweight=weight, zorder=zorder + 1, linespacing=1.15)
    return p


def arrow(ax, p0, p1, color=INK2, lw=0.9, style="-|>", ms=7, ls="-",
          connectionstyle="arc3,rad=0", zorder=4):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, ls=ls, zorder=zorder,
                        connectionstyle=connectionstyle, shrinkA=1, shrinkB=1)
    ax.add_patch(a)


def setup(ax, title, letter):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("     " + title, loc="left", fontsize=9.5, pad=4)
    panel_label(ax, letter, x=-0.01, y=1.02)


# --------------------------------------------------------------------------
def panel_a(ax):
    setup(ax, "One structure, three frames", "a")
    cols = [("representation formed", 0.16, 0.35),
            ("world changes", 0.53, 0.19),
            ("query", 0.74, 0.25)]
    for name, x, w in cols:
        ax.text(x + w / 2, 0.955, name, ha="center", va="center", fontsize=6.8,
                color=INK2, style="italic")
    rows = [
        ("belief", 0.72,
         ("Alice sees the truck on A,\nthen",
          "looks away", "keeps watching"),
         "truck moved\nA $\\rightarrow$ B", "Where will Alice\nlook for it?"),
        ("photograph", 0.45,
         ("a photo of the truck on A\nis taken",
          "before the move", "after the move"),
         "truck moved\nA $\\rightarrow$ B", "Where is the truck\nin the photo?"),
        ("Python", 0.18,
         ("a = [.., 'R', ..]\nthen",
          "b = a.copy()", "b = a"),
         "a[1] = 'I'", "What is b[1]?"),
    ]
    h = 0.20
    for label, y, (lead, div, nodiv), world, query in rows:
        ax.text(0.005, y + h / 2, label, ha="left", va="center", fontsize=7.2,
                color=INK, fontweight="bold", rotation=0)
        # representation box with the two variants
        x, w = cols[0][1], cols[0][2]
        box(ax, x, y, w, h, "", fc="white", ec=INK2)
        mono = label == "Python"
        ax.text(x + 0.012, y + h - 0.045, lead, ha="left", va="top",
                fontsize=6.3, color=INK, family="monospace" if mono else None,
                linespacing=1.1)
        ax.text(x + w - 0.012, y + 0.058, div, ha="right", va="center",
                fontsize=6.6, color=BLUE, fontweight="bold",
                family="monospace" if mono else None)
        ax.text(x + w - 0.012, y + 0.018, nodiv, ha="right", va="center",
                fontsize=6.6, color=AQUA, fontweight="bold",
                family="monospace" if mono else None)
        arrow(ax, (x + w + 0.003, y + h / 2), (cols[1][1] - 0.003, y + h / 2))
        x, w = cols[1][1], cols[1][2]
        box(ax, x, y, w, h, world, fc=PAPER, ec=INK2, fs=6.5,
            color=INK)
        arrow(ax, (x + w + 0.003, y + h / 2), (cols[2][1] - 0.003, y + h / 2))
        x, w = cols[2][1], cols[2][2]
        box(ax, x, y, w, h, query, fc="white", ec=INK2, fs=6.5)
    # legend for the two variants
    ax.text(0.16, 0.09, "diverged $\\rightarrow$ the representation is stale: answer A / 'R'",
            fontsize=6.4, color=BLUE, ha="left", va="center")
    ax.text(0.16, 0.035, "not diverged $\\rightarrow$ it tracks the world: answer B / 'I'",
            fontsize=6.4, color=AQUA, ha="left", va="center")


# --------------------------------------------------------------------------
def panel_b(ax):
    setup(ax, "Procedure", "b")
    # step 1: prompt pair
    y0, h = 0.56, 0.36
    box(ax, 0.0, y0, 0.24, h, "", fc="white", ec=INK2)
    ax.text(0.12, y0 + h - 0.05, "matched prompt pair", ha="center", va="top",
            fontsize=6.8, fontweight="bold")
    ax.text(0.12, y0 + h - 0.13, "base: diverged\nexp: not diverged",
            ha="center", va="top", fontsize=6.3, color=INK, linespacing=1.15)
    ax.text(0.12, y0 + 0.04, "answer same (C1, C2)\nor flips (C3–C7)",
            ha="center", va="bottom", fontsize=5.9, color=INK2, linespacing=1.1)
    arrow(ax, (0.245, y0 + h / 2), (0.285, y0 + h / 2))
    # step 2: patch
    box(ax, 0.29, y0, 0.30, h, "", fc="white", ec=INK2)
    ax.text(0.44, y0 + h - 0.05, "patch one head", ha="center", va="top",
            fontsize=6.8, fontweight="bold")
    ax.text(0.44, y0 + h - 0.13,
            "copy $z_{\\ell,h}$ at the final token\nfrom base into exp's run",
            ha="center", va="top", fontsize=6.3, linespacing=1.15)
    ax.text(0.44, y0 + 0.04,
            "score = Δ logit difference\nall 1,920 heads\nFWER permutation test",
            ha="center", va="bottom", fontsize=5.7, color=INK2, linespacing=1.1)
    arrow(ax, (0.595, y0 + h / 2), (0.635, y0 + h / 2))
    # step 3: fingerprint inset (real data)
    box(ax, 0.64, y0, 0.36, h, "", fc="white", ec=INK2)
    ax.text(0.82, y0 + h - 0.05, "causal fingerprint", ha="center", va="top",
            fontsize=6.8, fontweight="bold")
    ins = ax.inset_axes([0.665, y0 + 0.045, 0.20, 0.20])
    if FP_C1.exists():
        fp = np.load(FP_C1).reshape(48, 40)
        ins.imshow(fp, cmap=CMAP_SEQ, aspect="auto", vmin=0, vmax=fp.max())
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor(GRID)
    ins.set_xlabel("head", fontsize=5.5, labelpad=1)
    ins.set_ylabel("layer", fontsize=5.5, labelpad=1)
    ax.text(0.885, y0 + 0.15, "C1 belief\n(1,920 heads)", ha="left",
            va="center", fontsize=5.9, color=INK2, linespacing=1.1)
    ax.text(0.885, y0 + 0.06, "$\\rightarrow$ head sets\nper condition", ha="left",
            va="center", fontsize=5.9, color=INK2, linespacing=1.1)
    # step 4: interventions on held-out prompts
    arrow(ax, (0.82, y0 - 0.01), (0.82, 0.44), connectionstyle="arc3,rad=0")
    y1, h1 = 0.04, 0.38
    box(ax, 0.0, y1, 1.0, h1, "", fc=PAPER, ec=INK2)
    ax.text(0.02, y1 + h1 - 0.045, "then, on held-out prompts", ha="left",
            va="top", fontsize=6.8, fontweight="bold")
    items = [
        (0.02, "cumulative ablation",
         "head sets vs layer-matched\ncontrols; flip score, accuracy"),
        (0.36, "steering",
         "add / remove the divergence\ndirection at L41, fit on other half"),
        (0.70, "RSA / decoding",
         "head-output RDMs vs model RDMs;\nbelief, photo, code, Gemma"),
    ]
    for x, head, body in items:
        ax.text(x, y1 + 0.20, head, ha="left", va="center", fontsize=6.5,
                color=BLUE, fontweight="bold")
        ax.text(x, y1 + 0.09, body, ha="left", va="center", fontsize=5.8,
                color=INK, linespacing=1.1)


# --------------------------------------------------------------------------
def panel_c(ax):
    setup(ax, "Hypothesised mechanism", "c")
    # layer column
    cx, cw = 0.10, 0.10
    ax.add_patch(Rectangle((cx, 0.06), cw, 0.86, fc="white", ec=INK2, lw=0.7,
                           zorder=2))
    def ly(layer):  # layer -> y
        return 0.06 + 0.86 * layer / 48
    for l in (0, 12, 24, 36, 47):
        ax.text(cx - 0.015, ly(l), f"L{l}", ha="right", va="center",
                fontsize=5.6, color=INK2)
    # bands
    ax.add_patch(Rectangle((cx, ly(28)), cw, ly(40) - ly(28), fc=LIGHT_BLUE,
                           ec="none", zorder=2.5))
    ax.add_patch(Rectangle((cx, ly(42)), cw, ly(47) - ly(42), fc=LIGHT_ORANGE,
                           ec="none", zorder=2.5))
    # residual stream arrow through the column
    arrow(ax, (cx + cw / 2, 0.065), (cx + cw / 2, 0.915), color=INK2, lw=0.8,
          style="-|>", ms=6, zorder=3)
    ax.text(cx + cw / 2, 0.02, "residual stream", ha="center", va="center",
            fontsize=5.8, color=INK2)
    # divergence population description
    bx = 0.30
    box(ax, bx, ly(26), 0.68, ly(39.5) - ly(26), "", fc=LIGHT_BLUE, ec=BLUE,
        lw=0.8)
    ax.text(bx + 0.02, ly(38.0), "divergence-tracking heads  L28–L40",
            ha="left", va="center", fontsize=6.8, color=BLUE, fontweight="bold")
    ax.text(bx + 0.02, ly(32.3),
            "compare the stored representation with\nthe current world; same heads for belief,\nphoto and Python (C1 ≈ C2 ≈ code)",
            ha="left", va="center", fontsize=6.0, color=INK, linespacing=1.15)
    # divergence direction marker at L41
    ax.plot([cx, cx + cw], [ly(41)] * 2, color=BLUE, lw=1.6, zorder=4)
    ax.text(bx + 0.02, ly(41.0),
            "writes a low-rank divergence direction (L41)",
            ha="left", va="center", fontsize=6.0, color=BLUE, style="italic")
    arrow(ax, (bx - 0.005, ly(33)), (cx + cw + 0.005, ly(34)), color=BLUE,
          lw=0.9, style="-|>", ms=6, connectionstyle="arc3,rad=0.15")
    # retrieval population description
    box(ax, bx, ly(42.4), 0.68, ly(48.4) - ly(42.4), "", fc=LIGHT_ORANGE,
        ec=ORANGE, lw=0.8)
    ax.text(bx + 0.02, ly(47.3), "answer-retrieval heads  L42–L46",
            ha="left", va="center", fontsize=6.8, color=ORANGE, fontweight="bold")
    ax.text(bx + 0.02, ly(44.6),
            "read it through $W_Q$ (2.1× chance alignment);\nfetch the believed or the current location",
            ha="left", va="center", fontsize=6.0, color=INK, linespacing=1.15)
    arrow(ax, (cx + cw + 0.005, ly(41.2)), (bx - 0.005, ly(44)), color=ORANGE,
          lw=0.9, style="-|>", ms=6, connectionstyle="arc3,rad=-0.15")
    # composition note
    ax.text(bx, ly(23.6),
            "additive: fingerprint(belief + answer) ≈ C1 + C5,  r = 0.97",
            ha="left", va="center", fontsize=6.0, color=INK2, style="italic")
    # tokens at the bottom
    toks = [("A\n(believed)", BLUE), ("looks\naway", GRAY), ("B\n(current)", ORANGE),
            ("query\n(final token)", INK2)]
    tx0 = 0.34
    for i, (t, c) in enumerate(toks):
        x = tx0 + i * 0.165
        box(ax, x, 0.06, 0.14, 0.13, t, fc="white", ec=c, fs=5.8, color=c,
            lw=0.7)
    ax.text(bx, 0.005, "tokens read by the two populations", ha="left",
            va="center", fontsize=5.6, color=INK2)


def main():
    apply_style()
    plt.rcParams["axes.grid"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.35),
                             gridspec_kw={"width_ratios": [1.15, 1.25, 1.0],
                                          "wspace": 0.10})
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.06)
    save(fig, "fig0_design_schematic")


if __name__ == "__main__":
    main()
