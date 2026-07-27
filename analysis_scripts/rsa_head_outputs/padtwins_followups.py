#!/usr/bin/env python
"""CPU follow-ups to the pad-twins analysis (code_pointer_rdm_idil.qmd, July 27).

Three independent modes (run any subset):

  --baseline       Random-head pad-twin baseline (inheritance control): draw
                   size-matched head sets from heads NEVER selected by any CMA
                   set, run the observed pad-twin stats + divergence-model fits
                   per draw. Inheritance predicts random sets match the CMA
                   sets' pad response; a task-head-localized pointer substrate
                   predicts they don't. No new forward passes — uses the full
                   864-stimulus captures.

  --ceilings       Replicate-reliability noise ceilings for the Exp 1 (72-cell,
                   2 reps) and Exp 3 (432-cell, 2 content draws) grids, per
                   model x head set — the same leave-one-replicate-out +
                   Spearman-Brown + sqrt method diagnose_code_ledger_rdm.py
                   uses for the Exp 2 grid.

  --orthogonality  Tag-vs-carrier orthogonality figure (3 panels per model):
                   (A) |cos(tag, pad PC)| against the random-direction null;
                   (B) double-dissociation bars — mean |displacement| along
                       each axis for pad-only vs object-flip stimulus pairs;
                   (C) the (pad PC1, tag) scatter annotated with held-out AUCs.
                   Axes for the AUCs/displacements are estimated on one content
                   half and evaluated on the other (both folds, averaged), so
                   the numbers are out-of-sample; the scatter itself stays
                   full-data for display like fig_orthogonality_scatter.

Outputs:
  aggregation_code_padtwins/random_head_baseline{_gemma}.csv
  aggregation_code_padtwins/rdm_ceilings.csv
  aggregation_code_padtwins/orthogonality_fits.csv
  docs/figures/code_pointer_rdm/fig_orthogonality_panels.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import (  # noqa: E402
    HEADS_CSV, GEMMA_HEADSET_DIR, HEAD_SETS, GEMMA_HEAD_SETS,
    mantel, partial_spearman, upper_tri)
from analyze_code_padtwins_rdms import OUT_DIR, load  # noqa: E402
from padtwins_projection_test import twin_groups, displacement_pairs  # noqa: E402

ACT = {
    ("exp1", "qwen"): ROOT / "results/rsa_head_outputs/activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt",
    ("exp1", "gemma"): ROOT / "results/rsa_head_outputs/activations_code_position_full/gemma-2-27b-it/code_position.pt",
    ("exp3", "qwen"): ROOT / "results/rsa_head_outputs/activations_code_padtwins_full/Qwen2.5-14B-Instruct/code_padtwins.pt",
    ("exp3", "gemma"): ROOT / "results/rsa_head_outputs/activations_code_padtwins_full/gemma-2-27b-it/code_padtwins.pt",
}
CELL_KEYS = {
    "exp1": ("semantics", "idx", "pad_len", "suffix_pad_len"),
    "exp3": ("queried_object", "queried_slot_type", "role_b",
             "copy_syntax", "alias_syntax", "pad_len", "suffix_pad_len"),
}
REP_KEY = {"exp1": "replicate", "exp3": "content_id"}
FIG_DIR = ROOT / "docs/figures/code_pointer_rdm"
DOC_FIG_DIR = ROOT / "writeups/results_for_pi/code_pointer_rdm/figures"

# Okabe-Ito (colorblind-safe): blue / vermillion / gray
C_BLUE, C_ORANGE, C_GRAY = "#0072B2", "#D55E00", "#8a8a8a"


def head_sets_for(model):
    if model == "gemma":
        return {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in GEMMA_HEAD_SETS}
    heads = pd.read_csv(HEADS_CSV)
    return {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}


def set_neural_rdm(z, members):
    return np.mean([cosine_rdm(z[:, l, h, :]) for l, h in members], axis=0)


def members_of(hdf):
    return [(int(r.layer), int(r.head)) for r in hdf.itertuples()]


# ---------------------------------------------------------------- baseline

def observed_padtwin_stats(neural, stim):
    """Observed-only version of analyze_code_padtwins_rdms.padtwin_test."""
    groups = twin_groups(stim)
    ut9 = np.triu_indices(9, 1)
    acc = np.zeros(4)
    for idx in groups:
        D = neural[np.ix_(idx, idx)][ut9]
        pre = np.array([stim[i]["pad_len"] for i in idx], float)
        suf = np.array([stim[i]["suffix_pad_len"] for i in idx], float)
        dpre = np.abs(pre[ut9[0]] - pre[ut9[1]])
        dsuf = np.abs(suf[ut9[0]] - suf[ut9[1]])
        acc[0] += spearmanr(D, dpre).statistic
        acc[1] += spearmanr(D, dsuf).statistic
        acc[2] += partial_spearman(D, dpre, dsuf)
        acc[3] += partial_spearman(D, dsuf, dpre)
    return acc / len(groups)


def categorical_rdm(stim, key):
    v = np.array([hash(r[key]) for r in stim], float)
    return (v[:, None] != v[None, :]).astype(float)


def run_baseline(model, n_draws, seed):
    d, sets = load(model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    n_layers, n_heads = z.shape[1], z.shape[2]
    cma_union = {m for hdf in sets.values() for m in members_of(hdf)}
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
            if (l, h) not in cma_union]
    sizes = sorted({len(hdf) for hdf in sets.values()})[::len(sets) - 1] \
        if len(sets) > 1 else [len(next(iter(sets.values())))]
    # smallest and largest CMA-set size, e.g. {12, 32} qwen / {21, 50} gemma
    print(f"[{model}] pool = {len(pool)} never-selected heads "
          f"(of {n_layers * n_heads}); draw sizes {sizes}")

    obj_vec = upper_tri(categorical_rdm(stim, "queried_object"))
    ans_vec = upper_tri(categorical_rdm(stim, "answer"))
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        for draw in range(n_draws):
            idx = rng.choice(len(pool), size=size, replace=False)
            members = [pool[i] for i in idx]
            R = set_neural_rdm(z, members)
            s = observed_padtwin_stats(R, stim)
            nvec = upper_tri(R)
            rows.append({
                "model": model, "size": size, "draw": draw,
                "rho_prefix": s[0], "rho_suffix": s[1],
                "rho_prefix|suffix": s[2], "rho_suffix|prefix": s[3],
                "r_object_divergence": float(spearmanr(nvec, obj_vec).statistic),
                "r_answer_identity": float(spearmanr(nvec, ans_vec).statistic),
            })
        sub = pd.DataFrame([r for r in rows if r["size"] == size])
        print(f"[{model}] size {size}: pre|suf mean {sub['rho_prefix|suffix'].mean():+.3f} "
              f"[p5 {sub['rho_prefix|suffix'].quantile(.05):+.3f}, "
              f"p95 {sub['rho_prefix|suffix'].quantile(.95):+.3f}]  "
              f"obj_div mean {sub['r_object_divergence'].mean():+.3f} "
              f"[p95 {sub['r_object_divergence'].quantile(.95):+.3f}, "
              f"max {sub['r_object_divergence'].max():+.3f}]")
    suffix = "_gemma" if model == "gemma" else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"random_head_baseline{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


# ---------------------------------------------------------------- ceilings

def spearman_brown(r, k):
    return k * r / (1 + (k - 1) * r)


def run_ceilings():
    rows = []
    for grid in ("exp1", "exp3"):
        for model in ("qwen", "gemma"):
            d = torch.load(ACT[(grid, model)], weights_only=False)
            z = d["z"].float().numpy()
            stim = d["stim"]
            keys, rep_key = CELL_KEYS[grid], REP_KEY[grid]
            cells = [tuple(r[k] for k in keys) for r in stim]
            ucells = sorted(set(cells))
            reps = sorted({r[rep_key] for r in stim})
            sel = {}
            for rep in reps:
                by_cell = {}
                for i, (c, r) in enumerate(zip(cells, stim)):
                    if r[rep_key] == rep:
                        by_cell[c] = i
                assert len(by_cell) == len(ucells), (grid, model, rep)
                sel[rep] = [by_cell[c] for c in ucells]
            for set_name, hdf in head_sets_for(model).items():
                members = members_of(hdf)
                rep_rdms = [set_neural_rdm(z[sel[rep]], members) for rep in reps]
                rels = [float(spearmanr(upper_tri(rep_rdms[i]),
                                        upper_tri(rep_rdms[j])).statistic)
                        for i in range(len(reps)) for j in range(i + 1, len(reps))]
                rel1 = float(np.mean(rels))
                ceiling = float(np.sqrt(max(spearman_brown(rel1, len(reps)), 0.0)))
                rows.append({"grid": grid, "model": model, "head_set": set_name,
                             "n_heads": len(members), "n_cells": len(ucells),
                             "n_reps": len(reps), "rep_reliability_1": rel1,
                             "model_fit_ceiling": ceiling})
                print(f"[{grid}/{model}/{set_name}] rep reliability {rel1:.3f} "
                      f"-> ceiling ~{ceiling:.3f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "rdm_ceilings.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


# ------------------------------------------------------------ orthogonality

def concat_set(z, members):
    return np.concatenate([z[:, l, h, :] for l, h in members], axis=1)


def unit(v):
    return v / np.linalg.norm(v)


def tag_direction(X, stim, idx):
    is_copy = np.array([stim[i]["queried_object"] == "copy" for i in idx])
    return unit(X[idx][is_copy].mean(0) - X[idx][~is_copy].mean(0))


def pad_pc(X, stim, idx):
    idx_set = set(idx)
    pairs = [(i, j) for i, j in displacement_pairs(stim, twin_groups(stim))
             if i in idx_set and j in idx_set]
    disp = np.stack([X[j] - X[i] for i, j in pairs])
    _, _, vt = np.linalg.svd(disp, full_matrices=False)
    return vt[0], vt[1], pairs


def auc(scores, labels):
    u = mannwhitneyu(scores[labels], scores[~labels]).statistic
    a = u / (labels.sum() * (~labels).sum())
    return float(max(a, 1 - a))


def object_flip_pairs(stim, idx, rng, cap=4000):
    """Same pads + slot type + rep half, different queried object."""
    by = {}
    for i in idx:
        r = stim[i]
        by.setdefault((r["pad_len"], r["suffix_pad_len"],
                       r["queried_slot_type"]), []).append(i)
    pairs = [(i, j) for grp in by.values()
             for i in grp for j in grp
             if i < j and stim[i]["queried_object"] != stim[j]["queried_object"]]
    if len(pairs) > cap:
        pairs = [pairs[k] for k in rng.choice(len(pairs), cap, replace=False)]
    return pairs


def orthogonality_for_model(model, set_name, rng, rows):
    d, sets = load(model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    X = concat_set(z, members_of(sets[set_name]))

    # full-data axes for display + cosines
    all_idx = list(range(len(stim)))
    u_tag = tag_direction(X, stim, all_idx)
    pc1, pc2, _ = pad_pc(X, stim, all_idx)
    cos1, cos2 = float(abs(u_tag @ pc1)), float(abs(u_tag @ pc2))
    D = X.shape[1]
    null = np.abs([unit(rng.normal(size=D)) @ unit(rng.normal(size=D))
                   for _ in range(2000)])
    null_p95 = float(np.quantile(null, 0.95))

    # held-out AUCs + displacement dissociation, averaged over both folds
    halves = [[i for i in all_idx if stim[i]["content_id"] == c] for c in (0, 1)]
    aucs = np.zeros(4)
    disp22 = np.zeros(4)  # pad-pairs x (pad axis, tag axis), obj-pairs x same
    x_cf, y_cf = np.zeros(len(stim)), np.zeros(len(stim))  # cross-fitted coords
    for fit, ev in ((0, 1), (1, 0)):
        ut_f = tag_direction(X, stim, halves[fit])
        pc1_f, _, _ = pad_pc(X, stim, halves[fit])
        # SVD sign is arbitrary: orient on the FIT half so more pad = larger x
        fi = np.array(halves[fit])
        pf = np.array([stim[i]["pad_len"] for i in fi])
        xf = X[fi] @ pc1_f
        if xf[pf == 32].mean() < xf[pf == 0].mean():
            pc1_f = -pc1_f
        xi = np.array(halves[ev])
        x_pad, y_tag = X[xi] @ pc1_f, X[xi] @ ut_f
        # each stimulus is scored by the axes fit on the OTHER content half;
        # per-fold centring removes the arbitrary fold-specific offset
        x_cf[xi], y_cf[xi] = x_pad - x_pad.mean(), y_tag - y_tag.mean()
        pad_on = np.array([stim[i]["pad_len"] > 0 for i in xi])
        is_copy = np.array([stim[i]["queried_object"] == "copy" for i in xi])
        aucs += [auc(x_pad, pad_on), auc(y_tag, pad_on),
                 auc(y_tag, is_copy), auc(x_pad, is_copy)]
        _, _, pad_pairs = pad_pc(X, stim, halves[ev])
        obj_pairs = object_flip_pairs(stim, halves[ev], rng)
        dp = np.stack([X[j] - X[i] for i, j in pad_pairs])
        do = np.stack([X[j] - X[i] for i, j in obj_pairs])
        disp22 += [np.abs(dp @ pc1_f).mean(), np.abs(dp @ ut_f).mean(),
                   np.abs(do @ pc1_f).mean(), np.abs(do @ ut_f).mean()]
    aucs /= 2
    disp22 /= 2

    rows.append({
        "model": model, "head_set": set_name, "dim": D,
        "cos_tag_padPC1": cos1, "cos_tag_padPC2": cos2,
        "null_cos_p95": null_p95,
        "auc_pad_from_pad_axis": aucs[0], "auc_pad_from_tag_axis": aucs[1],
        "auc_object_from_tag_axis": aucs[2], "auc_object_from_pad_axis": aucs[3],
        "disp_padpairs_along_pad": disp22[0], "disp_padpairs_along_tag": disp22[1],
        "disp_objpairs_along_pad": disp22[2], "disp_objpairs_along_tag": disp22[3],
    })
    # display coordinates: full-data axes (x, y) and cross-fitted ones
    # (x_cf, y_cf) — the latter are what fig_orthogonality_scatter plots.
    return {"x": X @ pc1, "y": X @ u_tag, "x_cf": x_cf, "y_cf": y_cf,
            "pad": np.array([stim[i]["pad_len"] for i in all_idx]),
            "copy": np.array([stim[i]["queried_object"] == "copy"
                              for i in all_idx]),
            "cos": (cos1, cos2), "null_p95": null_p95,
            "aucs": aucs, "disp22": disp22}


def run_orthogonality(seed):
    rng = np.random.default_rng(seed)
    rows, panels = [], {}
    for model, set_name in (("qwen", "code_abstract"), ("gemma", "gemma_c1")):
        panels[model] = orthogonality_for_model(model, set_name, rng, rows)
        r = rows[-1]
        print(f"[{model}/{set_name}] |cos(tag,padPC1)| = {r['cos_tag_padPC1']:.3f} "
              f"(null p95 {r['null_cos_p95']:.3f})  AUCs: pad-from-pad "
              f"{r['auc_pad_from_pad_axis']:.3f}, pad-from-tag "
              f"{r['auc_pad_from_tag_axis']:.3f}, obj-from-tag "
              f"{r['auc_object_from_tag_axis']:.3f}, obj-from-pad "
              f"{r['auc_object_from_pad_axis']:.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for row_i, (model, p) in enumerate(panels.items()):
        set_label = "code_abstract" if model == "qwen" else "gemma_c1"
        axA, axB, axC = axes[row_i]

        # A: cosine bars vs null band
        axA.bar([0, 1], p["cos"], width=0.55, color=C_BLUE)
        axA.axhspan(0, p["null_p95"], color=C_GRAY, alpha=0.35,
                    label=f"random-direction null (p95 = {p['null_p95']:.2f})")
        axA.axhline(1.0, color=C_GRAY, lw=0.8, ls=":")
        for xpos, v in zip([0, 1], p["cos"]):
            axA.text(xpos, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        axA.set_xticks([0, 1], ["pad PC1", "pad PC2"])
        axA.set_ylim(0, 1.05)
        axA.set_ylabel("|cos(tag direction, pad direction)|")
        axA.set_title(f"{model} ({set_label}): alignment vs chance")
        axA.legend(fontsize=7, loc="upper right", frameon=False)

        # B: double dissociation
        d22 = p["disp22"]
        xs = np.arange(2)
        axB.bar(xs - 0.18, [d22[0], d22[2]], width=0.32, color=C_ORANGE,
                label="along pad axis")
        axB.bar(xs + 0.18, [d22[1], d22[3]], width=0.32, color=C_BLUE,
                label="along tag axis")
        for xpos, v in zip(np.concatenate([xs - 0.18, xs + 0.18]),
                           [d22[0], d22[2], d22[1], d22[3]]):
            axB.text(xpos, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        axB.set_xticks(xs, ["pad-only pairs", "object-flip pairs"])
        axB.set_ylabel("mean |displacement| along axis (held-out)")
        axB.set_title("who moves along which axis")
        axB.legend(fontsize=8, frameon=False)

        # C: scatter with AUC annotation
        for pad, marker_scale in ((0, 1.0), (16, 1.0), (32, 1.0)):
            m = p["pad"] == pad
            for is_copy, mk in ((True, "o"), (False, "^")):
                mm = m & (p["copy"] == is_copy)
                axC.scatter(p["x"][mm], p["y"][mm], s=10 * marker_scale,
                            marker=mk, alpha=0.55,
                            color={0: C_GRAY, 16: C_ORANGE, 32: C_BLUE}[pad])
        a = p["aucs"]
        axC.set_xlabel(f"pad-carrier coord (pad AUC {a[0]:.2f} | object AUC {a[3]:.2f})")
        axC.set_ylabel(f"tag coord (object AUC {a[2]:.2f} | pad AUC {a[1]:.2f})")
        axC.set_title("each axis reads only its own factor\n"
                      "(color = prefix pad; ○ copy / △ alias)")
    fig.tight_layout()
    for out_dir in (FIG_DIR, DOC_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig_orthogonality_panels.png", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "orthogonality_fits.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}\n-> {FIG_DIR / 'fig_orthogonality_panels.png'} "
          f"(+ copy in {DOC_FIG_DIR})")


# ------------------------------------------------------------ scatter (2-panel)

def run_scatter(seed):
    """Standalone (pad-carrier, tag) scatter, CROSS-FITTED: every stimulus is
    projected onto axes estimated from the other content half, so the visible
    copy/alias separation is out-of-sample, not a consequence of fitting the
    tag axis on the points being plotted. Both coordinates are then expressed
    in units of the panel's SD along the pad axis (equal aspect + matched
    limits) so the two models are directly comparable."""
    rng = np.random.default_rng(seed)
    rows, panels = [], {}
    for model, set_name in (("qwen", "code_abstract"), ("gemma", "gemma_c1")):
        p = orthogonality_for_model(model, set_name, rng, rows)
        x, y = p["x_cf"].astype(float), p["y_cf"].astype(float)
        sd = x.std()
        p["xn"], p["yn"], p["set"] = x / sd, y / sd, set_name
        panels[model] = p

    lim = 1.05 * max(np.abs(np.concatenate([p[k] for p in panels.values()
                                            for k in ("xn", "yn")])).max(), 1.0)
    pad_colors = {0: "#BFD9EE", 16: "#3B8FD0", 32: "#12395C"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.9), sharex=True, sharey=True)
    for ax, (model, p) in zip(axes, panels.items()):
        for pad in (0, 16, 32):
            for is_copy, mk in ((False, "o"), (True, "^")):
                m = (p["pad"] == pad) & (p["copy"] == is_copy)
                ax.scatter(p["xn"][m], p["yn"][m], s=11, marker=mk, alpha=0.6,
                           color=pad_colors[pad], linewidths=0)
        a = p["aucs"]
        ax.axhline(0, color=C_GRAY, lw=0.7)
        ax.axvline(0, color=C_GRAY, lw=0.7)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"{model} — {p['set']} heads   "
                     f"|cos(tag, pad PC1)| = {p['cos'][0]:.2f}", fontsize=11)
        ax.set_xlabel("pad-carrier coordinate (pad PC1, object-blind)\n"
                      f"held-out AUC — x: pad {a[0]:.2f}, object {a[3]:.2f}   "
                      f"y: object {a[2]:.2f}, pad {a[1]:.2f}", fontsize=9)
    axes[0].set_ylabel("tag coordinate (copy − alias mean direction)", fontsize=9)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, color=pad_colors[p],
                          label=f"prefix pad {p}") for p in (0, 16, 32)]
    handles += [plt.Line2D([], [], ls="", marker=m, ms=7, color=C_GRAY,
                           label=lab) for m, lab in (("o", "alias queried"),
                                                     ("^", "copy queried"))]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Pads move stimuli mostly horizontally, copy/alias mostly "
                 "vertically\n(cross-fitted: each point is scored by axes "
                 "estimated on the other content half; axes in SD units of "
                 "that panel's pad axis, equal aspect, matched limits)",
                 fontsize=11, y=1.10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for out_dir in (FIG_DIR, DOC_FIG_DIR):
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig_orthogonality_scatter.png", dpi=180,
                    bbox_inches="tight")
    plt.close(fig)
    for model, p in panels.items():
        print(f"[{model}] pad-axis SD normalisation: tag spread / pad spread = "
              f"{p['yn'].std():.2f}")
    print(f"-> {FIG_DIR / 'fig_orthogonality_scatter.png'} (+ copy in {DOC_FIG_DIR})")


# -------------------------------------------------------- address-matched

def run_address_matched(n_perm, seed):
    """Veracity fit on untouched-slot queries only: copy and alias answers
    both come from the line-1 init literal at the same token address, so any
    copy/alias separation here is veracity content with the lookback target
    held constant (idil's same-address question, qmd July 27)."""
    rng = np.random.default_rng(seed)
    rows = []
    for model in ("qwen", "gemma"):
        d, sets = load(model)
        z = d["z"].float().numpy()
        stim = d["stim"]
        keep = [i for i, r in enumerate(stim)
                if r["queried_slot_type"] == "untouched"]
        obj = np.array([1.0 if stim[i]["queried_object"] == "copy" else 0.0
                        for i in keep])
        obj_rdm = (obj[:, None] != obj[None, :]).astype(float)
        for set_name, hdf in sets.items():
            R = set_neural_rdm(z[keep], members_of(hdf))
            r, p = mantel(R, obj_rdm, n_perm, rng)
            rows.append({"model": model, "head_set": set_name,
                         "n_stim": len(keep), "r_object_addr_matched": r,
                         "p_perm": p})
            print(f"[{model}/{set_name}] address-matched veracity fit "
                  f"r = {r:+.3f}  p = {p:.4f}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "address_matched_veracity.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--ceilings", action="store_true")
    ap.add_argument("--orthogonality", action="store_true")
    ap.add_argument("--address_matched", action="store_true")
    ap.add_argument("--scatter", action="store_true")
    ap.add_argument("--n_draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.baseline:
        for model in ("qwen", "gemma"):
            run_baseline(model, args.n_draws, args.seed)
    if args.ceilings:
        run_ceilings()
    if args.orthogonality:
        run_orthogonality(args.seed)
    if args.address_matched:
        run_address_matched(1000, args.seed)
    if args.scatter:
        run_scatter(args.seed)
    if not (args.baseline or args.ceilings or args.orthogonality
            or args.address_matched or args.scatter):
        print("nothing to do: pass --baseline / --ceilings / --orthogonality "
              "/ --address_matched / --scatter")


if __name__ == "__main__":
    main()
