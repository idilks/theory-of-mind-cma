#!/usr/bin/env python
"""Exp 2 (code_ledger) RDM sanity diagnostics: is the map interpretable, or
nuisance mush?

Per divergence head set:
  1. within- vs between-design-cell mean cosine distance (how much of the map
     is filler/pad/answer-char nuisance);
  2. per-factor distance footprints (mean extra distance for pairs differing
     on one factor) — model-free version of the Mantel ranking;
  3. leave-one-replicate-out reliability of the 48-cell RDM + Spearman-Brown
     ceiling for the mean-of-3 RDM (upper bound on any model fit);
  4. object-divergence fit at the 48-cell-mean level (nuisance averaged out);
  5. figure: stimulus RDM sorted by query cell | 48-cell mean RDM.

Outputs: results/rsa_head_outputs/aggregation_code_ledger/rdm_diagnostic_{model}.csv
         results/summary/code_pointer_rdm/figures/fig_exp2_diagnostic_{model}.png
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
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402

from analyze_code_ledger_rdms import (  # noqa: E402
    ACT_PATH, HEADS_CSV, OUT_DIR, GEMMA_ACT_PATH, GEMMA_HEADSET_DIR)

FIG_DIR = ROOT / "results/summary/code_pointer_rdm/figures"
CELL_KEYS = ("queried_object", "queried_slot_type", "role_b",
             "copy_syntax", "alias_syntax")
FOOT_KEYS = ("queried_object", "queried_slot_type", "role_b",
             "copy_syntax", "alias_syntax", "answer")
N_REP = 3


def load_sets(model: str):
    if model == "gemma":
        d = torch.load(GEMMA_ACT_PATH, weights_only=False)
        sets = {}
        for s in ("gemma_c1", "gemma_c5"):
            hdf = pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
            sets[s] = [(int(r.layer), int(r.head)) for r in hdf.itertuples()]
    else:
        d = torch.load(ACT_PATH, weights_only=False)
        heads = pd.read_csv(HEADS_CSV)
        sets = {}
        for s in ("code_abstract", "tom_c1"):
            hdf = heads[heads["sources"].str.contains(s)]
            sets[s] = [(int(r.layer), int(r.head)) for r in hdf.itertuples()]
    return d["z"].float().numpy(), d["stim"], sets


def spearman_brown(r: float, k: int) -> float:
    return k * r / (1 + (k - 1) * r)


def diagnose(z, stim, members, set_name, model):
    zs = np.stack([z[:, l, h, :] for l, h in members], axis=1)
    R = np.mean([cosine_rdm(zs[:, i, :]) for i in range(zs.shape[1])], axis=0)
    n = len(stim)
    iu = np.triu_indices(n, 1)

    cells = [tuple(r[k] for k in CELL_KEYS) for r in stim]
    ucells = sorted(set(cells))
    same = np.array([cells[i] == cells[j] for i, j in zip(*iu)])
    within, between = R[iu][same].mean(), R[iu][~same].mean()

    feet = {}
    for k in FOOT_KEYS:
        v = np.array([r[k] for r in stim])
        s = v[iu[0]] == v[iu[1]]
        feet[k] = float(R[iu][~s].mean() - R[iu][s].mean())

    idx_by_cell = {c: [i for i, cc in enumerate(cells) if cc == c] for c in ucells}
    ut = lambda M: M[np.triu_indices(M.shape[0], 1)]  # noqa: E731
    rep_rdms = []
    for rep in range(N_REP):
        sel = [idx_by_cell[c][rep] for c in ucells]
        rep_rdms.append(np.mean([cosine_rdm(zs[sel, i, :])
                                 for i in range(zs.shape[1])], axis=0))
    rels = [float(spearmanr(ut(rep_rdms[i]), ut(rep_rdms[j])).statistic)
            for i in range(N_REP) for j in range(i + 1, N_REP)]
    rel1 = float(np.mean(rels))
    ceiling = float(np.sqrt(max(spearman_brown(rel1, N_REP), 0.0)))

    # cell-mean RDM (average z within cell, then cosine) + object fit
    cz = np.stack([zs[idx_by_cell[c]].mean(axis=0) for c in ucells])
    Rc = np.mean([cosine_rdm(cz[:, i, :]) for i in range(cz.shape[1])], axis=0)
    obj = np.array([1.0 if c[0] == "copy" else 0.0 for c in ucells])
    obj_rdm = (obj[:, None] != obj[None, :]).astype(float)
    r_obj_cell = float(spearmanr(ut(Rc), ut(obj_rdm)).statistic)

    row = {"model": model, "head_set": set_name, "n_heads": len(members),
           "within_cell_dist": float(within), "between_cell_dist": float(between),
           "within_between_ratio": float(within / between),
           "rep_reliability_1": rel1, "rep_reliability_mean3": spearman_brown(rel1, N_REP),
           "model_fit_ceiling": ceiling, "obj_div_fit_cellmean": r_obj_cell,
           **{f"footprint_{k}": v for k, v in feet.items()}}

    print(f"[{model}/{set_name}] within {within:.4f} vs between {between:.4f} "
          f"(ratio {within/between:.3f})")
    print(f"[{model}/{set_name}] rep reliability {rel1:.3f} -> mean-of-3 "
          f"{spearman_brown(rel1, N_REP):.3f} -> fit ceiling ~{ceiling:.3f}")
    print(f"[{model}/{set_name}] obj-div fit at cell level: {r_obj_cell:.3f}")
    for k, v in feet.items():
        print(f"    footprint {k:18s} {v:+.4f}")

    # figure (divergence set only gets the paired-panel figure)
    order = sorted(range(n), key=lambda i: tuple(stim[i][k] for k in CELL_KEYS)
                   + (stim[i]["replicate"],))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    im0 = axes[0].imshow(R[np.ix_(order, order)], cmap="viridis")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    for cut in (n // 4, n // 2, 3 * n // 4):
        axes[0].axhline(cut - 0.5, c="w", lw=0.8)
        axes[0].axvline(cut - 0.5, c="w", lw=0.8)
    axes[0].set_title("stimulus RDM, sorted by query cell\n"
                      "(blocks: alias·m | alias·u | copy·m | copy·u)")
    im1 = axes[1].imshow(Rc, cmap="viridis")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
    nc = len(ucells)
    for cut in (nc // 4, nc // 2, 3 * nc // 4):
        axes[1].axhline(cut - 0.5, c="w", lw=0.8)
        axes[1].axvline(cut - 0.5, c="w", lw=0.8)
    axes[1].set_title(f"{nc} design-cell mean RDM\n(replicate nuisance averaged out)")
    fig.suptitle(f"{set_name} set, Exp 2 ({model})")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"fig_exp2_diagnostic_{model}_{set_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {out}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen")
    args = ap.parse_args()

    z, stim, sets = load_sets(args.model)
    rows = [diagnose(z, stim, m, s, args.model) for s, m in sets.items()]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"rdm_diagnostic_{args.model}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
