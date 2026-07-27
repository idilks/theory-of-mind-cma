#!/usr/bin/env python
"""Figures for docs/design_code_pointer_rdm.md (Exp 1 + Exp 2 RDM results).

Encoding is constant across all figures — color = signal CLASS, never rank:
  blue   #2a78d6  divergence / veracity-analog models
  orange #eb6834  position / geometry models
  aqua   #1baf7a  identity models (what the answer is)
  yellow #eda100  surface-syntax models
(4-slot palette validated with the dataviz six-checks script, light surface.)

Outputs -> docs/figures/code_pointer_rdm/*.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_padtwins_rdms import load as load_padtwins  # noqa: E402
from padtwins_projection_test import (  # noqa: E402
    displacement_pairs, pad_subspace, project_out, twin_groups)

AGG1 = ROOT / "results/rsa_head_outputs/aggregation_code_position/mantel_code_position.csv"
AGG2 = ROOT / "results/rsa_head_outputs/aggregation_code_ledger/mantel_code_ledger.csv"
ACT1 = ROOT / "results/rsa_head_outputs/activations_code_position/code_position.pt"
ACT2 = (ROOT / "results/rsa_head_outputs/activations_code_ledger_full/"
        "Qwen2.5-14B-Instruct/code_ledger.pt")
HEADS = ROOT / "results/rsa_head_outputs/topk_heads/code_position_union.csv"
FIG_DIR = ROOT / "docs/figures/code_pointer_rdm"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"

SET_ORDER = ["code_abstract", "tom_c1", "code_control", "tom_c5"]
SET_LABEL = {"code_abstract": "code_abstract (12 heads)",
             "tom_c1": "tom_c1 (32 heads)",
             "code_control": "code_control (14 heads)",
             "tom_c5": "tom_c5 (22 heads)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
})

SEQ_BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def barh_panel(ax, labels, values, colors, xmax):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors)
    for yi, v in zip(y, values):
        ax.text(max(v, 0) + xmax * 0.015, yi,
                f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "-."),
                va="center", ha="left", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0 if min(values) >= -0.02 else min(values) * 1.3, xmax)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_mantel(agg_csv, models, colors, title, out_name, xmax):
    df = pd.read_csv(agg_csv)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex=True)
    for ax, hs in zip(axes.flat, SET_ORDER):
        sub = df[df.head_set == hs].set_index("model")
        vals = [sub.loc[m, "spearman_r"] for m in models]
        barh_panel(ax, models, vals, colors, xmax)
        ax.set_title(SET_LABEL[hs], fontsize=9.5, color=INK, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("Mantel Spearman r (neural RDM vs model RDM)")
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG_DIR / out_name, dpi=180)
    plt.close(fig)
    print(f"-> {FIG_DIR / out_name}")


def fig_dissociation():
    df = pd.read_csv(AGG2)
    signals = [("object_divergence", "object-divergence\n(which object diverged)", BLUE),
               ("answer_identity", "answer identity\n(what the answer is)", AQUA),
               ("queried_slot_abs", "queried-slot position\n(where the value sits)", ORANGE)]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    x = np.arange(len(SET_ORDER))
    w = 0.26
    for i, (m, label, c) in enumerate(signals):
        vals = [df[(df.head_set == hs) & (df.model == m)].spearman_r.iloc[0]
                for hs in SET_ORDER]
        bars = ax.bar(x + (i - 1) * (w + 0.015), vals, width=w, color=c, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}".lstrip("0"),
                    ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [SET_LABEL[s].split(" (")[0] for s in SET_ORDER])
    ax.tick_params(length=0)
    ax.set_ylabel("Mantel Spearman r")
    ax.set_ylim(-0.05, 0.58)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="lower left",
              bbox_to_anchor=(0, 1.01), ncols=3)
    fig.suptitle("Exp 2 triple dissociation: divergence sets carry object-divergence, "
                 "retrieval sets carry identity,\nposition is shared and graded "
                 "(Qwen, ledger grid, 1000-perm Mantel)",
                 fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(FIG_DIR / "fig_dissociation.png", dpi=180)
    plt.close(fig)
    print(f"-> {FIG_DIR / 'fig_dissociation.png'}")


def set_rdm(act_path, sources_pat):
    d = torch.load(act_path, weights_only=False)
    heads = pd.read_csv(HEADS)
    sel = heads[heads["sources"].str.contains(sources_pat)]
    z = d["z"].float().numpy()
    if z.ndim == 4:  # full capture (n_stim, L, H, dim)
        zs = np.stack([z[:, int(r.layer), int(r.head), :] for r in sel.itertuples()], 1)
    else:            # union capture (n_stim, n_heads, dim) — rows align with HEADS
        zs = z[:, sel.index.to_numpy(), :]
    rdm = np.mean([cosine_rdm(zs[:, h, :]) for h in range(zs.shape[1])], axis=0)
    return rdm, d["stim"]


def heatmap(ax, rdm, order, boundaries, block_labels, title):
    R = rdm[np.ix_(order, order)]
    off = ~np.eye(len(order), dtype=bool)
    vmin, vmax = np.percentile(R[off], [2, 98])
    ax.imshow(R, cmap=SEQ_BLUES, vmin=vmin, vmax=vmax, interpolation="nearest")
    for b in boundaries:
        ax.axhline(b - 0.5, color=SURFACE, linewidth=1.6)
        ax.axvline(b - 0.5, color=SURFACE, linewidth=1.6)
    edges = [0] + boundaries + [len(order)]
    for lab, s, e in zip(block_labels, edges[:-1], edges[1:]):
        ax.text(-4, (s + e) / 2, lab, ha="right", va="center", fontsize=8, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")


def fig_rdms():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6))

    rdm1, stim1 = set_rdm(ACT1, "code_abstract")
    sem = np.array([s["semantics"] for s in stim1])
    pos = np.array([s["slot_char_pos"] for s in stim1])
    order1 = np.lexsort((pos, sem))  # alias block first, copy second
    n_alias = int((sem == "alias").sum())
    heatmap(axes[0], rdm1, order1, [n_alias], ["alias", "copy"],
            "Exp 1 — code_abstract heads, position grid\n"
            "stimuli sorted by semantics, then slot position")

    rdm2, stim2 = set_rdm(ACT2, "tom_c1")
    qo = np.array([s["queried_object"] for s in stim2])
    qs = np.array([s["queried_slot_type"] for s in stim2])
    order2 = np.lexsort((qs, qo))
    b1 = int((qo == "alias").sum())
    q_m = int(((qo == "alias") & (qs == "mutated")).sum())
    c_m = int(((qo == "copy") & (qs == "mutated")).sum())
    heatmap(axes[1], rdm2, order2, [q_m, b1, b1 + c_m],
            ["alias·mut", "alias·unt", "copy·mut", "copy·unt"],
            "Exp 2 — tom_c1 heads, ledger grid\n"
            "stimuli sorted by queried object, then slot type")

    fig.suptitle("Neural RDMs (cosine distance, darker = more dissimilar; "
                 "final-token z, mean over set heads)", fontsize=10.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "fig_rdm_heatmaps.png", dpi=180)
    plt.close(fig)
    print(f"-> {FIG_DIR / 'fig_rdm_heatmaps.png'}")


def model_rdm_panel(ax, m_rdm, order, boundaries, block_labels, title):
    R = m_rdm[np.ix_(order, order)]
    if R.max() > 0:
        R = R / R.max()
    ax.imshow(R, cmap=SEQ_BLUES, vmin=0, vmax=1, interpolation="nearest")
    for b in boundaries:
        ax.axhline(b - 0.5, color=SURFACE, linewidth=1.6)
        ax.axvline(b - 0.5, color=SURFACE, linewidth=1.6)
    edges = [0] + boundaries + [len(order)]
    for lab, s, e in zip(block_labels, edges[:-1], edges[1:]):
        ax.text(-4, (s + e) / 2, lab, ha="right", va="center", fontsize=7, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")


def fig_model_rdms():
    """Model (predicted) RDMs under the same stimulus sort as fig_rdms, so the
    predicted charts can be compared to the neural charts panel-for-panel.
    Model RDMs are pure functions of the design factors — identical for Qwen
    and Gemma (position models here use the char-offset proxy)."""
    # --- Exp 1: sort by semantics, then slot position (matches fig_rdms left)
    stim1 = torch.load(ACT1, weights_only=False)["stim"]
    sem = np.array([s["semantics"] for s in stim1])
    pos = np.array([float(s["slot_char_pos"]) for s in stim1])
    idx = np.array([float(s["idx"]) for s in stim1])
    ans1 = np.array([s["answer"] for s in stim1])
    order1 = np.lexsort((pos, sem))
    n_alias = int((sem == "alias").sum())

    models1 = [
        ("semantics (copy/alias)", (sem[:, None] != sem[None, :]).astype(float)),
        ("queried slot (rel index)", np.abs(idx[:, None] - idx[None, :])),
        ("queried slot (char pos)", np.abs(pos[:, None] - pos[None, :])),
        ("answer identity", (ans1[:, None] != ans1[None, :]).astype(float)),
    ]

    # --- Exp 2: sort by queried object, then slot type (matches fig_rdms right)
    stim2 = torch.load(ACT2, weights_only=False)["stim"]
    qo = np.array([s["queried_object"] for s in stim2])
    qs = np.array([s["queried_slot_type"] for s in stim2])
    ans2 = np.array([s["answer"] for s in stim2])
    spos2 = np.array([float(s["slot_char_pos"]) for s in stim2])
    csyn = np.array([s["copy_syntax"] for s in stim2])
    order2 = np.lexsort((qs, qo))
    b1 = int((qo == "alias").sum())
    q_m = int(((qo == "alias") & (qs == "mutated")).sum())
    c_m = int(((qo == "copy") & (qs == "mutated")).sum())
    bounds2 = [q_m, b1, b1 + c_m]
    labels2 = ["alias·mut", "alias·unt", "copy·mut", "copy·unt"]

    slot_div = ((qo == "copy") & (qs == "mutated")).astype(float)
    models2 = [
        ("object divergence\n(queried object diverged?)",
         (qo[:, None] != qo[None, :]).astype(float)),
        ("slot divergence\n(queried value diverged?)",
         (slot_div[:, None] != slot_div[None, :]).astype(float)),
        ("answer identity", (ans2[:, None] != ans2[None, :]).astype(float)),
        ("queried slot (char pos)", np.abs(spos2[:, None] - spos2[None, :])),
        ("copy syntax\n(.copy() | [:] | list())",
         (csyn[:, None] != csyn[None, :]).astype(float)),
    ]

    fig, axes = plt.subplots(2, 5, figsize=(13.2, 6.0))
    models1[0] = ("Exp 1 sort — " + models1[0][0], models1[0][1])
    models2[0] = ("Exp 2 sort — " + models2[0][0], models2[0][1])
    for ax, (title, m) in zip(axes[0], models1):
        model_rdm_panel(ax, m, order1, [n_alias], ["alias", "copy"], title)
    axes[0, len(models1)].axis("off")
    for ax, (title, m) in zip(axes[1], models2):
        model_rdm_panel(ax, m, order2, bounds2, labels2, title)
    fig.suptitle("Model (predicted) RDMs under the fig_rdm_heatmaps stimulus sort "
                 "(darker = predicted more dissimilar; scalar models scaled to max)",
                 fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "fig_model_rdms.png", dpi=180)
    plt.close(fig)
    print(f"-> {FIG_DIR / 'fig_model_rdms.png'}")


# ---------------------------------------------------------------------------
# Exp 1 / Exp 2 per-experiment map grids (observed + model RDMs, one sort)
# ---------------------------------------------------------------------------

from analyze_code_ledger_rdms import GEMMA_HEADSET_DIR  # noqa: E402

EXP12_ACTS = {
    ("exp1", "qwen"): (ROOT / "results/rsa_head_outputs/"
                       "activations_code_position_full/Qwen2.5-14B-Instruct/"
                       "code_position.pt"),
    ("exp1", "gemma"): (ROOT / "results/rsa_head_outputs/"
                        "activations_code_position_full/gemma-2-27b-it/"
                        "code_position.pt"),
    ("exp2", "qwen"): ACT2,
    ("exp2", "gemma"): (ROOT / "results/rsa_head_outputs/"
                        "activations_code_ledger_full/gemma-2-27b-it/"
                        "code_ledger.pt"),
}
EXP12_SETS = {"qwen": ("code_abstract", "code_control"),
              "gemma": ("gemma_c1", "gemma_c5")}


def load_exp12(exp, model):
    d = torch.load(EXP12_ACTS[(exp, model)], weights_only=False)
    if model == "gemma":
        sets = {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in EXP12_SETS["gemma"]}
    else:
        heads = pd.read_csv(HEADS)
        sets = {s: heads[heads["sources"].str.contains(s)]
                for s in EXP12_SETS["qwen"]}
    return d, sets


def neural_set_rdm(z, hdf):
    zs = np.stack([z[:, int(r.layer), int(r.head), :]
                   for r in hdf.itertuples()], 1)
    return np.mean([cosine_rdm(zs[:, h, :]) for h in range(zs.shape[1])],
                   axis=0)


def fig_exp1_maps(model):
    """Exp 1 (position grid): observed RDMs for the divergence + retrieval
    set next to every model RDM, all under one sort (semantics › slot pos)."""
    d, sets = load_exp12("exp1", model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    sem = np.array([s["semantics"] for s in stim])
    pos = np.array([float(s["slot_char_pos"]) for s in stim])
    idx = np.array([float(s["idx"]) for s in stim])
    ans = np.array([s["answer"] for s in stim])
    order = np.lexsort((pos, sem))
    bounds = [int((sem == "alias").sum())]
    labels = ["alias", "copy"]
    div_set, ret_set = EXP12_SETS[model]

    panels = [
        (f"OBSERVED — {div_set}\n(divergence set)",
         neural_set_rdm(z, sets[div_set]), "neural"),
        (f"OBSERVED — {ret_set}\n(retrieval set)",
         neural_set_rdm(z, sets[ret_set]), "neural"),
        ("veracity analog — semantics\n(copy vs alias)",
         (sem[:, None] != sem[None, :]).astype(float), "model"),
        ("position — queried slot\n(rel index |Δidx|)",
         np.abs(idx[:, None] - idx[None, :]), "model"),
        ("position — queried slot\n(char pos |Δ|)",
         np.abs(pos[:, None] - pos[None, :]), "model"),
        ("answer identity", (ans[:, None] != ans[None, :]).astype(float),
         "model"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 7.2))
    for ax, (title, M, kind) in zip(axes.flat, panels):
        if kind == "neural":
            heatmap(ax, M, order, bounds, labels, title)
        else:
            model_rdm_panel(ax, M, order, bounds, labels, title)
    fig.suptitle(f"Exp 1 (position grid, {model}): observed vs model RDMs, "
                 "one shared sort — semantics › queried-slot position "
                 "(darker = more dissimilar)", fontsize=10.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"fig_exp1_maps_{model}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"-> {out}")


def fig_exp2_maps(model):
    """Exp 2 (ledger grid): observed RDMs for the divergence + retrieval set
    next to every model RDM, one sort (queried object › slot type)."""
    d, sets = load_exp12("exp2", model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    qo = np.array([s["queried_object"] for s in stim])
    qs = np.array([s["queried_slot_type"] for s in stim])
    qn = np.array([s["queried_name"] for s in stim])
    ans = np.array([s["answer"] for s in stim])
    spos = np.array([float(s["slot_char_pos"]) for s in stim])
    csyn = np.array([s["copy_syntax"] for s in stim])
    slot_div = ((qo == "copy") & (qs == "mutated")).astype(float)
    order = np.lexsort((qs, qo))
    b1 = int((qo == "alias").sum())
    q_m = int(((qo == "alias") & (qs == "mutated")).sum())
    c_m = int(((qo == "copy") & (qs == "mutated")).sum())
    bounds = [q_m, b1, b1 + c_m]
    labels = ["alias·mut", "alias·unt", "copy·mut", "copy·unt"]
    div_set, ret_set = EXP12_SETS[model]

    panels = [
        (f"OBSERVED — {div_set}\n(divergence set)",
         neural_set_rdm(z, sets[div_set]), "neural"),
        (f"OBSERVED — {ret_set}\n(retrieval set)",
         neural_set_rdm(z, sets[ret_set]), "neural"),
        ("object divergence\n(queried object diverged?)",
         (qo[:, None] != qo[None, :]).astype(float), "model"),
        ("slot divergence\n(queried value diverged?)",
         (slot_div[:, None] != slot_div[None, :]).astype(float), "model"),
        ("answer identity", (ans[:, None] != ans[None, :]).astype(float),
         "model"),
        ("position — queried slot\n(char pos |Δ|)",
         np.abs(spos[:, None] - spos[None, :]), "model"),
        ("queried name\n(b vs c)", (qn[:, None] != qn[None, :]).astype(float),
         "model"),
        ("copy syntax\n(.copy() | [:] | list())",
         (csyn[:, None] != csyn[None, :]).astype(float), "model"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.2))
    for ax, (title, M, kind) in zip(axes.flat, panels):
        if kind == "neural":
            heatmap(ax, M, order, bounds, labels, title)
        else:
            model_rdm_panel(ax, M, order, bounds, labels, title)
    fig.suptitle(f"Exp 2 (ledger grid, {model}): observed vs model RDMs, one "
                 "shared sort — queried object › slot type "
                 "(darker = more dissimilar)", fontsize=10.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"fig_exp2_maps_{model}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"-> {out}")


# ---------------------------------------------------------------------------
# Exp 3 (pad twins)
# ---------------------------------------------------------------------------

EXP3_SETS = {"qwen": ("code_abstract", "code_control"),
             "gemma": ("gemma_c1", "gemma_c5")}


def fig_exp3_rdms(model):
    """Observed RDMs (divergence + retrieval set) alongside every model RDM,
    all under one sort: queried object > slot type > twin group > pads."""
    d, sets = load_padtwins(model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    n = len(stim)

    qo = np.array([s["queried_object"] for s in stim])
    qs = np.array([s["queried_slot_type"] for s in stim])
    twin = np.array([s["twin_id"] for s in stim])
    pre = np.array([float(s["pad_len"]) for s in stim])
    suf = np.array([float(s["suffix_pad_len"]) for s in stim])
    ans = np.array([s["answer"] for s in stim])
    csyn = np.array([s["copy_syntax"] for s in stim])
    slot_div = ((qo == "copy") & (qs == "mutated")).astype(float)

    order = np.lexsort((suf, pre, twin, qs, qo))
    nb = n // 4
    bounds = [nb, 2 * nb, 3 * nb]
    labels = ["alias·mut", "alias·unt", "copy·mut", "copy·unt"]

    div_set, ret_set = EXP3_SETS[model]

    def neural(set_name):
        hdf = sets[set_name]
        zs = np.stack([z[:, int(r.layer), int(r.head), :]
                       for r in hdf.itertuples()], 1)
        return np.mean([cosine_rdm(zs[:, h, :]) for h in range(zs.shape[1])],
                       axis=0)

    models = [
        (f"OBSERVED — {div_set}\n(divergence set)", neural(div_set), "neural"),
        (f"OBSERVED — {ret_set}\n(retrieval set)", neural(ret_set), "neural"),
        ("veracity — object divergence\n(queried object diverged?)",
         (qo[:, None] != qo[None, :]).astype(float), "model"),
        ("lookback — source position\n(|Δ prefix pad|)",
         np.abs(pre[:, None] - pre[None, :]), "model"),
        ("lookback — distance to end\n(|Δ suffix pad|)",
         np.abs(suf[:, None] - suf[None, :]), "model"),
        ("slot divergence\n(queried value diverged?)",
         (slot_div[:, None] != slot_div[None, :]).astype(float), "model"),
        ("answer identity",
         (ans[:, None] != ans[None, :]).astype(float), "model"),
        ("copy syntax\n(.copy() | [:] | list())",
         (csyn[:, None] != csyn[None, :]).astype(float), "model"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.2))
    for ax, (title, M, kind) in zip(axes.flat, models):
        if kind == "neural":
            heatmap(ax, M, order, bounds, labels, title)
        else:
            model_rdm_panel(ax, M, order, bounds, labels, title)
    fig.suptitle(f"Exp 3 (pad twins, {model}): observed vs model RDMs, one "
                 "shared sort — queried object › slot type › twin group › pads "
                 "(darker = more dissimilar)", fontsize=10.5, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIG_DIR / f"fig_exp3_rdms_{model}.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"-> {out}")


def fig_exp3_projected_maps(k=2):
    """Divergence-set observed maps, raw vs pad-projected: a denoised VIEW of
    the same data. The projection is estimated from pad displacements only
    (object-blind, and pads are orthogonal to the query factors by design),
    so it can remove position striping but cannot manufacture object
    structure. All statistics in the report are computed on the raw maps."""
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.0))
    for row, model in enumerate(("qwen", "gemma")):
        d, sets = load_padtwins(model)
        z = d["z"].float().numpy()
        stim = d["stim"]
        n = len(stim)
        qo = np.array([s["queried_object"] for s in stim])
        qs = np.array([s["queried_slot_type"] for s in stim])
        twin = np.array([s["twin_id"] for s in stim])
        pre = np.array([float(s["pad_len"]) for s in stim])
        suf = np.array([float(s["suffix_pad_len"]) for s in stim])
        order = np.lexsort((suf, pre, twin, qs, qo))
        nb = n // 4
        bounds, labels = [nb, 2 * nb, 3 * nb], ["alias·mut", "alias·unt",
                                                "copy·mut", "copy·unt"]
        pairs = displacement_pairs(stim, twin_groups(stim))
        div_set = EXP3_SETS[model][0]
        hdf = sets[div_set]
        raw, proj = [], []
        for r in hdf.itertuples():
            zh = z[:, int(r.layer), int(r.head), :]
            U, _, _ = pad_subspace(zh, pairs, k)
            raw.append(cosine_rdm(zh))
            proj.append(cosine_rdm(project_out(zh, U)))
        heatmap(axes[row, 0], np.mean(raw, axis=0), order, bounds, labels,
                f"{div_set} — raw")
        heatmap(axes[row, 1], np.mean(proj, axis=0), order, bounds, labels,
                f"{div_set} — pad directions removed (k={k})")
    fig.suptitle("Divergence-set observed RDMs, raw vs pad-projected — same "
                 "data, position striping removed\n(projection estimated from "
                 "pad displacements only; object-blind; statistics computed "
                 "on raw)", fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIG_DIR / "fig_exp3_projected_maps.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"-> {out}")


def fig_exp3_padtwin_curves():
    """The tag-vs-lookback money plot: mean within-twin distance vs pad shift,
    per head set, with the content-twin floor as reference."""
    AGG3 = ROOT / "results/rsa_head_outputs/aggregation_code_padtwins"
    fig, axes = plt.subplots(2, 4, figsize=(14.5, 6.6))
    for row, model in enumerate(("qwen", "gemma")):
        sfx = "_gemma" if model == "gemma" else ""
        buckets = pd.read_csv(AGG3 / f"padtwin_distance_buckets{sfx}.csv")
        tests = pd.read_csv(AGG3 / f"padtwin_test{sfx}.csv").set_index("head_set")
        set_names = tests.index.tolist()
        for col, hs in enumerate(set_names):
            ax = axes[row, col]
            sub = buckets[buckets.head_set == hs]
            pre_marg = sub.groupby("d_prefix").mean_dist.mean()
            suf_marg = sub.groupby("d_suffix").mean_dist.mean()
            ax.plot(pre_marg.index.to_numpy(), pre_marg.to_numpy(), "o-",
                    color=ORANGE, label="vs |Δ prefix pad| (source moves)")
            ax.plot(suf_marg.index.to_numpy(), suf_marg.to_numpy(), "s--",
                    color="#8858c8", label="vs |Δ suffix pad| (end moves)")
            floor = tests.loc[hs, "content_twin_dist"]
            ax.axhline(floor, color=MUTED, linewidth=1.0, linestyle=":",
                       label="content-twin floor")
            r = tests.loc[hs, "rho_prefix|suffix"]
            p = tests.loc[hs, "p_rho_prefix|suffix"]
            ax.set_title(f"{hs} ({int(tests.loc[hs, 'n_heads'])} heads)\n"
                         f"pre|suf ρ = {r:+.3f} (p = {p:.3g})",
                         fontsize=8.5, color=INK, loc="left")
            ax.set_xlim(-3, 35)
            ax.set_xticks([0, 16, 32])
            ax.tick_params(length=0)
            ax.grid(axis="y", color=MUTED, alpha=0.25, linewidth=0.6)
            ax.set_axisbelow(True)
        axes[row, 0].set_ylabel(f"{model}\nmean within-twin cosine distance",
                                fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("pad shift (tokens)")
    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, frameon=False, fontsize=8.5,
               loc="upper right", ncols=3, bbox_to_anchor=(0.99, 0.985))
    fig.suptitle("Exp 3 pad-twin test: within-twin distance vs pad shift\n"
                 "(TAG predicts flat at floor; POINTER predicts growth with "
                 "prefix shift)", fontsize=10.5, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.01, 1, 0.92))
    out = FIG_DIR / "fig_exp3_padtwin_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"-> {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_mantel(
        AGG1,
        models=["semantics", "queried_slot_rel", "queried_slot_abs",
                "slot_dist_to_end", "answer_identity"],
        colors=[BLUE, ORANGE, ORANGE, ORANGE, AQUA],
        title="Exp 1 (position grid): veracity-analog dominates; position real but secondary",
        out_name="fig_exp1_mantel.png", xmax=0.98)
    fig_mantel(
        AGG2,
        models=["object_divergence", "slot_divergence", "copy_syntax",
                "alias_syntax", "answer_identity", "queried_slot_abs",
                "prompt_length"],
        colors=[BLUE, BLUE, YELLOW, YELLOW, AQUA, ORANGE, ORANGE],
        title="Exp 2 (ledger grid): object ledger wins; syntax dead; position prominent",
        out_name="fig_exp2_mantel.png", xmax=0.62)
    fig_dissociation()
    fig_rdms()
    fig_model_rdms()
    for m in ("qwen", "gemma"):
        fig_exp1_maps(m)
        fig_exp2_maps(m)
    # Exp 3 figures need the pad-twins captures + analysis CSVs on disk
    try:
        fig_exp3_rdms("qwen")
        fig_exp3_rdms("gemma")
        fig_exp3_padtwin_curves()
        fig_exp3_projected_maps()
    except FileNotFoundError as e:
        print(f"[skip] Exp 3 figures ({e})")


if __name__ == "__main__":
    main()
