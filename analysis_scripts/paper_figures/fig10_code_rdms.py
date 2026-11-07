"""Figure 10 — code-domain RDMs (appendix B, Experiments 1 and 2), Qwen2.5-14B-Instruct.

Observed RDMs of the code divergence set (blue) and the code answer set
(orange) next to every model RDM they are tested against, all under one
stimulus sort (block, then answer value). Each model panel prints its Mantel Spearman r with the two
observed RDMs, in the matching colors, so the reader compares numbers in place
instead of looking up the table. A model's frame is colored when it is the
role model of that set (semantics / object divergence for the divergence set,
answer identity for the answer set), the same rule that bolds the appendix
tables.

Data: hook_z captures in results/rsa_head_outputs/activations_code_{position,
ledger}_full/Qwen2.5-14B-Instruct/; head sets from topk_heads/
code_position_union.csv; r values from aggregation_code_{position,ledger}/
mantel_code_{position,ledger}.csv. Model RDMs come from the analysis analysis_scripts'
own build_model_rdms, so each panel is the matrix that was tested.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsa_head_outputs"))
from style import BLUE, CMAP_SEQ, INK, INK2, ORANGE, apply_style, save
from rsa_utils import cosine_rdm
import analyze_code_position_rdms as exp1_models
import analyze_code_ledger_rdms as exp2_models

REPO = Path(__file__).resolve().parents[2]
RSA = REPO / "results/rsa_head_outputs"
ACTS = {
    1: RSA / "activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt",
    2: RSA / "activations_code_ledger_full/Qwen2.5-14B-Instruct/code_ledger.pt",
}
MANTEL = {
    1: RSA / "aggregation_code_position/mantel_code_position.csv",
    2: RSA / "aggregation_code_ledger/mantel_code_ledger.csv",
}
HEADS = RSA / "topk_heads/code_position_union.csv"
SETS = {"code_abstract": ("divergence set", 12, BLUE),
        "code_control": ("answer set", 14, ORANGE)}

CMAP_MODEL = LinearSegmentedColormap.from_list("white_ink", ["#ffffff", "#9a998f", "#2b2a27"])
# Nov 6: observed and model RDMs share one colormap and one colorbar label.
# Values are not rescaled: observed = mean cosine distance (2nd-98th pct.
# color limits), model = its own 0-1 values, each with a numbered colorbar.
CMAP_RDM = CMAP_SEQ
CBAR_LABEL = "dissimilarity"

# (Mantel model key, panel title, role-model-of set or None)
PANELS = {
    1: [("semantics", "semantics\ncopy vs. alias", "code_abstract"),
        ("answer_identity", "answer identity", "code_control"),
        ("queried_slot_rel", "queried slot\nlist index", None),
        ("queried_slot_abs", "queried slot\ntoken position", None),
        ("slot_dist_to_end", "queried slot\ntokens to end", None)],
    2: [("object_divergence", "object divergence\nqueried list is the copy", "code_abstract"),
        ("slot_divergence", "slot divergence\nqueried value changed", None),
        ("answer_identity", "answer identity", "code_control"),
        ("queried_name", "queried name\nb vs. c", None),
        ("copy_syntax", "copy syntax\n.copy() / [:] / list()", None),
        ("alias_syntax", "alias syntax", None),
        ("queried_slot_abs", "queried slot\ntoken position", None)],
}
GRID = {1: (2, 4), 2: (2, 5)}


def observed_rdm(z, heads):
    return np.mean([cosine_rdm(z[:, int(r.layer), int(r.head), :]) for r in heads.itertuples()], axis=0)


def sort_and_blocks(exp, stim):
    if exp == 1:
        sem = np.array([s["semantics"] for s in stim])
        pos = np.array([float(s["slot_char_pos"]) for s in stim])
        ans = np.array([s["answer"] for s in stim])
        order = np.lexsort((pos, ans, sem))
        return order, [int((sem == "alias").sum())], ["alias", "copy"]
    obj = np.array([s["queried_object"] for s in stim])
    slot = np.array([s["queried_slot_type"] for s in stim])
    ans = np.array([s["answer"] for s in stim])
    order = np.lexsort((ans, slot, obj))
    n_am = int(((obj == "alias") & (slot == "mutated")).sum())
    n_a = int((obj == "alias").sum())
    n_cm = int(((obj == "copy") & (slot == "mutated")).sum())
    return order, [n_am, n_a, n_a + n_cm], ["alias\nmut.", "alias\nunt.", "copy\nmut.", "copy\nunt."]


def color_limits(R, robust):
    off = ~np.eye(len(R), dtype=bool)
    return np.percentile(R[off], [2, 98]) if robust else (R.min(), R.max())


def draw(ax, M, order, bounds, labels, show_labels, robust):
    R = M[np.ix_(order, order)]
    vmin, vmax = color_limits(R, robust)
    im = ax.imshow(R, cmap=CMAP_RDM, vmin=vmin, vmax=vmax, interpolation="nearest")
    for b in bounds:
        ax.axhline(b - 0.5, color="white", lw=1.2)
        ax.axvline(b - 0.5, color="white", lw=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if show_labels:
        edges = [0] + bounds + [len(order)]
        ax.set_yticks([(s + e) / 2 - 0.5 for s, e in zip(edges[:-1], edges[1:])])
        ax.set_yticklabels(labels, fontsize=7.5, linespacing=0.9)
        ax.tick_params(axis="y", length=0)
    return im


def colorbar_right(ax, im, label=CBAR_LABEL):
    """Numbered vertical colorbar beside one panel, as in Yang et al. (2025) Fig. 4."""
    cax = ax.inset_axes([1.05, 0.0, 0.06, 1.0])
    cb = ax.figure.colorbar(im, cax=cax)
    cb.locator = MaxNLocator(nbins=3)
    cb.update_ticks()
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6.5, length=1.5, pad=1.2)
    cb.set_label(label, fontsize=6.5, labelpad=1)
    return cb


def frame(ax, color, lw):
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(color)
        sp.set_linewidth(lw)


def title(ax, k, text, **kw):
    ax.set_title(rf"$\bf{{{chr(97 + k)}}}$   {text}", loc="left", **kw)


KEY = ("Dark = dissimilar. Heads: cosine\ndistance; models: own units.\n\n"
       "Under each model: Mantel r\nwith the {blue} (left) and\n"
       "{orange} (right) heads.\n\n"
       "Colored frame: the model that\nset is predicted to fit.")


def build(exp):
    d = torch.load(ACTS[exp], weights_only=False)
    z, stim = d["z"].float().numpy(), d["stim"]
    heads = pd.read_csv(HEADS)
    mantel = pd.read_csv(MANTEL[exp])
    models = (exp1_models if exp == 1 else exp2_models).build_model_rdms(stim)
    order, bounds, labels = sort_and_blocks(exp, stim)

    nr, nc = GRID[exp]
    fig, axes = plt.subplots(nr, nc, figsize=(1.45 * nc + 0.4, 3.55))
    axes = axes.ravel()
    col0 = set(range(0, nr * nc, nc))
    fig.subplots_adjust(left=0.07, right=0.94, top=0.86, bottom=0.07, wspace=0.62, hspace=0.62)
    k = 0
    for name, (role, n, color) in SETS.items():
        ax = axes[k]
        im = draw(ax, observed_rdm(z, heads[heads["sources"].str.contains(name)]),
                  order, bounds, labels, k in col0, robust=True)
        colorbar_right(ax, im)
        frame(ax, color, 2.0)
        title(ax, k, f"{role.split()[0]} heads\n({n})", fontsize=8, color=color)
        k += 1
    for key, ttl, role_of in PANELS[exp]:
        ax = axes[k]
        im = draw(ax, models[key], order, bounds, labels, k in col0, robust=False)
        colorbar_right(ax, im)
        if role_of:
            frame(ax, SETS[role_of][2], 2.0)
        else:
            frame(ax, "#c3c2b7", 0.6)
        title(ax, k, ttl, fontsize=7.5, color=INK)
        r = {s: mantel.loc[(mantel.head_set == s) & (mantel.model == key), "spearman_r"].item()
             for s in SETS}
        ax.text(0.0, -0.05, f"r = {r['code_abstract']:.2f}", transform=ax.transAxes,
                fontsize=8, color=BLUE, ha="left", va="top", fontweight="bold")
        ax.text(1.0, -0.05, f"{r['code_control']:.2f}", transform=ax.transAxes,
                fontsize=8, color=ORANGE, ha="right", va="top", fontweight="bold")
        k += 1
    for ax in axes[k:]:
        ax.axis("off")
    axes[-1].text(0.0, 1.0, KEY.format(blue="blue", orange="orange"), transform=axes[-1].transAxes,
                  fontsize=7, color=INK2, va="top", ha="left", linespacing=1.15)
    return fig


def build_compact():
    """Main-text cut of Experiment 1: the two observed RDMs and their two role models.

    Same data, sort, colors and r labels as fig10_code_rdms_exp1 panels a-d; the
    position models and the reading key are left to the appendix figure/caption.
    """
    d = torch.load(ACTS[1], weights_only=False)
    z, stim = d["z"].float().numpy(), d["stim"]
    heads = pd.read_csv(HEADS)
    mantel = pd.read_csv(MANTEL[1])
    models = exp1_models.build_model_rdms(stim)
    order, bounds, labels = sort_and_blocks(1, stim)

    fig, axes = plt.subplots(1, 4, figsize=(6.2, 1.95))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.8, bottom=0.12, wspace=0.62)
    main_titles = {"code_abstract": "divergence heads", "code_control": "answer heads"}
    for k, (name, (role, n, color)) in enumerate(SETS.items()):
        ax = axes[k]
        im = draw(ax, observed_rdm(z, heads[heads["sources"].str.contains(name)]),
                  order, bounds, labels, k == 0, robust=True)
        colorbar_right(ax, im)
        frame(ax, color, 2.0)
        title(ax, k, f"{main_titles[name]} ({n})", fontsize=8, color=color)
    model_titles = {"semantics": "model: copy vs. alias", "answer_identity": "model: answer"}
    for k, (key, ttl, role_of) in enumerate(PANELS[1][:2], start=2):
        ax = axes[k]
        im = draw(ax, models[key], order, bounds, labels, False, robust=False)
        colorbar_right(ax, im)
        frame(ax, SETS[role_of][2], 2.0)
        title(ax, k, model_titles[key], fontsize=8, color=INK)
        r = {s: mantel.loc[(mantel.head_set == s) & (mantel.model == key), "spearman_r"].item()
             for s in SETS}
        ax.text(0.0, -0.05, f"r = {r['code_abstract']:.2f}", transform=ax.transAxes,
                fontsize=8, color=BLUE, ha="left", va="top", fontweight="bold")
        ax.text(1.0, -0.05, f"{r['code_control']:.2f}", transform=ax.transAxes,
                fontsize=8, color=ORANGE, ha="right", va="top", fontweight="bold")
    return fig


def main():
    apply_style()
    for exp in (1, 2):
        save(build(exp), f"fig10_code_rdms_exp{exp}")
    save(build_compact(), "fig10_code_rdms_main")


if __name__ == "__main__":
    main()
