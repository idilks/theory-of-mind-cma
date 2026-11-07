#!/usr/bin/env python
"""Permutation significance for the Exp 2 (code_ledger) quantities that the
main Mantel run leaves untested:

  1. the partial Spearman fits (obj|slot, slot|obj) — analyze_code_ledger_rdms
     reports them with p_perm = NaN;
  2. the model-free factor footprints from diagnose_code_ledger_rdm.

Null for both: permute stimulus labels of the tested structure (model RDM
rows+cols for partials; factor labels for footprints) with the neural RDM —
and, for partials, the control model — held fixed. Two-sided p with the +1
correction, matching statistical convention in analyze_code_ledger_rdms.

Output: results/rsa_head_outputs/aggregation_code_ledger/partial_significance.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import (  # noqa: E402
    ACT_PATH, HEADS_CSV, OUT_DIR, GEMMA_ACT_PATH, GEMMA_HEADSET_DIR,
    HEAD_SETS, GEMMA_HEAD_SETS, upper_tri)

FOOT_KEYS = ("queried_object", "queried_slot_type", "role_b",
             "copy_syntax", "alias_syntax", "answer")


def resid(a, b):
    b1 = np.column_stack([np.ones_like(b), b])
    coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
    return a - b1 @ coef


def partial_with_null(neural, x_rdm, c_rdm, n_perm, rng):
    """Spearman(neural, x | c); null permutes x's stimulus labels."""
    n = neural.shape[0]
    ry = rankdata(upper_tri(neural))
    rc = rankdata(upper_tri(c_rdm))
    ry_res = resid(ry, rc)
    obs = float(np.corrcoef(ry_res, resid(rankdata(upper_tri(x_rdm)), rc))[0, 1])
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        rx = rankdata(upper_tri(x_rdm[p][:, p]))
        null[i] = float(np.corrcoef(ry_res, resid(rx, rc))[0, 1])
    pval = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, pval, float(np.abs(null).max())


def footprint_with_null(R, values, n_perm, rng):
    """Mean excess distance for factor-different pairs; null permutes labels."""
    iu = np.triu_indices(R.shape[0], 1)
    flat = R[iu]

    def foot(v):
        same = v[iu[0]] == v[iu[1]]
        return float(flat[~same].mean() - flat[same].mean())

    obs = foot(values)
    null = np.array([foot(values[rng.permutation(len(values))])
                     for _ in range(n_perm)])
    pval = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, pval, float(np.abs(null).max())


def load(model):
    if model == "gemma":
        d = torch.load(GEMMA_ACT_PATH, weights_only=False)
        sets = {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in GEMMA_HEAD_SETS}
    else:
        d = torch.load(ACT_PATH, weights_only=False)
        heads = pd.read_csv(HEADS_CSV)
        sets = {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}
    return d["z"].float().numpy(), d["stim"], sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = []
    for model in ("qwen", "gemma"):
        z, stim, sets = load(model)
        slot_div = np.array([1.0 if (r["queried_object"] == "copy"
                                     and r["queried_slot_type"] == "mutated")
                             else 0.0 for r in stim])
        obj = np.array([hash(r["queried_object"]) for r in stim], float)
        sd_rdm = (slot_div[:, None] != slot_div[None, :]).astype(float)
        od_rdm = (obj[:, None] != obj[None, :]).astype(float)

        for set_name, hdf in sets.items():
            zs = np.stack([z[:, int(r.layer), int(r.head), :]
                           for r in hdf.itertuples()], axis=1)
            neural = np.mean([cosine_rdm(zs[:, h, :])
                              for h in range(zs.shape[1])], axis=0)

            for label, x, c in (("obj|slot", od_rdm, sd_rdm),
                                ("slot|obj", sd_rdm, od_rdm)):
                r, pv, nmax = partial_with_null(neural, x, c, args.n_perm, rng)
                rows.append({"model": model, "head_set": set_name,
                             "quantity": f"partial_{label}", "observed": r,
                             "p_perm": pv, "null_abs_max": nmax})
                print(f"{model:5s} {set_name:14s} {label:9s} "
                      f"r = {r:+.3f}  p = {pv:.4f}  max|null| = {nmax:.3f}")

            for k in FOOT_KEYS:
                v = np.array([r[k] for r in stim])
                f, pv, nmax = footprint_with_null(neural, v, args.n_perm, rng)
                rows.append({"model": model, "head_set": set_name,
                             "quantity": f"footprint_{k}", "observed": f,
                             "p_perm": pv, "null_abs_max": nmax})
                print(f"{model:5s} {set_name:14s} foot {k:18s} "
                      f"{f:+.4f}  p = {pv:.4f}  max|null| = {nmax:.4f}")

    out = OUT_DIR / "partial_significance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
