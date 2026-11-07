#!/usr/bin/env python
"""Deconfound step 1: is the pad response POSITIONAL INHERITANCE or a pointer?

The caveat this closes: a
head's output at the final token is an attention-weighted sum of value
vectors, and value vectors carry positional information through RoPE. So any
head that attends into the code block inherits some position signal
mechanically — no pointer computation required. Inserting pad tokens moves
the code block, so the head's z moves, so within-twin neural distance grows
with |d pad|. That is the pad response, and inheritance predicts it for free.

The discriminating quantity is the head's own ATTENTION CENTRE OF MASS. Under
pure inheritance the pad response is a deterministic function of how far the
head's attention-weighted mean source position moved: once that is
partialled out, no pad response should remain. Under a pointer that is
recomputed per stimulus, a residual survives.

Per head, within each twin group (9 stimuli, identical except pads), over
the 36 pairs:
    D      pairwise cosine distance between the head's z vectors
    dpre   |d prefix pad|        (published pad-response predictor)
    dsuf   |d suffix pad|
    dsrc   |d attention-weighted mean source position|   (the carrier)

and three numbers, each averaged over twin groups:
    rho_pad_raw   rho(D, dpre | dsuf)              — the published pad response
    rho_src       rho(D, dsrc)                     — carrier tracking
    rho_pad_resid rho(D, dpre | dsuf, dsrc)        — what survives the control
    r2_src        rank-R^2 of D from dsrc alone

Reading:
  rho_pad_resid ~ 0 and r2_src ~ 1   pure inheritance. The pad response is the
                                     mechanical consequence of attention
                                     moving with the text; nothing extra.
  rho_pad_resid stays elevated       the head's z tracks pad length beyond
                                     what its own attention movement explains.

Run on the CMA head sets AND on random never-selected heads: the control is
only informative relative to what an arbitrary head does, since the
random-head baseline already showed the pad response itself is generic
(padtwins_followups.py --baseline).

REQUIRES a capture made with capture_head_outputs_full.py
--capture_attn_summary (slurm_scripts/rsa_code_padtwins_attn.sh). The
original pad-twin captures store hook_z only, which is why this control was
blocked.

Outputs (results/rsa_head_outputs/aggregation_code_padtwins/):
  position_prediction{_gemma}.csv         per head
  position_prediction_summary{_gemma}.csv per head set + random baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from analyze_code_ledger_rdms import (  # noqa: E402
    HEADS_CSV, GEMMA_HEADSET_DIR, HEAD_SETS, GEMMA_HEAD_SETS)
from analyze_code_padtwins_rdms import OUT_DIR  # noqa: E402
from padtwins_projection_test import twin_groups  # noqa: E402

RSA = ROOT / "results/rsa_head_outputs"
MODEL_DIRS = {"qwen": "Qwen2.5-14B-Instruct", "gemma": "gemma-2-27b-it"}
# the attention-summary rerun writes to its own tree so the published Exp 3
# capture is never overwritten; fall back to the original tree so the script
# reports the real reason (no attn_mean_src) rather than a missing file
ACT_PATHS = {m: [RSA / f"activations_code_padtwins_attn/{d}/code_padtwins.pt",
                 RSA / f"activations_code_padtwins_full/{d}/code_padtwins.pt"]
             for m, d in MODEL_DIRS.items()}
UT9 = np.triu_indices(9, 1)


def head_sets_for(model):
    if model == "gemma":
        return {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in GEMMA_HEAD_SETS}
    heads = pd.read_csv(HEADS_CSV)
    return {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}


def cosine_dist_block(Z):
    """Pairwise cosine distance among the rows of Z (small: 9 x dim)."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Zn @ Zn.T


def _resid(a, C):
    coef, *_ = np.linalg.lstsq(C, a, rcond=None)
    return a - C @ coef


def partial_spearman_multi(y, x, controls):
    """Spearman(y, x | controls) by rank residualisation on >=1 covariate."""
    ry, rx = rankdata(y), rankdata(x)
    C = np.column_stack([np.ones(len(y))] + [rankdata(c) for c in controls])
    ey, ex = _resid(ry, C), _resid(rx, C)
    if ey.std() < 1e-12 or ex.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def rank_r2(y, x):
    """Fraction of rank-variance in y explained by a linear fit on rank(x)."""
    ry = rankdata(y)
    C = np.column_stack([np.ones(len(y)), rankdata(x)])
    e = _resid(ry, C)
    v = ry.var()
    return float(1.0 - e.var() / v) if v > 1e-12 else np.nan


def head_stats(z, mean_src, stim, groups, layer, head):
    """Four numbers per head, averaged over twin groups."""
    zh = z[:, layer, head, :]
    sh = mean_src[:, layer, head]
    acc, n = np.zeros(4), 0
    for idx in groups:
        D = cosine_dist_block(zh[idx])[UT9]
        pre = np.array([stim[i]["pad_len"] for i in idx], float)
        suf = np.array([stim[i]["suffix_pad_len"] for i in idx], float)
        src = sh[idx].astype(float)
        dpre = np.abs(pre[UT9[0]] - pre[UT9[1]])
        dsuf = np.abs(suf[UT9[0]] - suf[UT9[1]])
        dsrc = np.abs(src[UT9[0]] - src[UT9[1]])
        if np.ptp(dsrc) < 1e-9:  # head's attention never moves: no carrier
            acc += [partial_spearman_multi(D, dpre, [dsuf]),
                    0.0, partial_spearman_multi(D, dpre, [dsuf]), 0.0]
        else:
            acc += [partial_spearman_multi(D, dpre, [dsuf]),
                    float(spearmanr(D, dsrc).statistic),
                    partial_spearman_multi(D, dpre, [dsuf, dsrc]),
                    rank_r2(D, dsrc)]
        n += 1
    return acc / n


def summarise(rows, label, model, n_heads):
    df = pd.DataFrame(rows)
    return {"model": model, "head_set": label, "n_heads": n_heads,
            "rho_pad_raw": df["rho_pad_raw"].mean(),
            "rho_src": df["rho_src"].mean(),
            "rho_pad_resid": df["rho_pad_resid"].mean(),
            "r2_src": df["r2_src"].mean(),
            "attenuation": 1.0 - (df["rho_pad_resid"].mean()
                                  / df["rho_pad_raw"].mean())
            if abs(df["rho_pad_raw"].mean()) > 1e-9 else np.nan}


def run(model, n_random, seed):
    path = next((p for p in ACT_PATHS[model] if p.exists()), None)
    if path is None:
        print(f"[skip] no pad-twin capture for {model}")
        return None, None
    d = torch.load(path, weights_only=False)
    if "attn_mean_src" not in d:
        print(f"[skip] {path.parent.parent.name}/{path.name} has no "
              "attn_mean_src — captured without --capture_attn_summary. "
              "Run slurm_scripts/padtwins_deconfound.sh tasks 0-1.")
        return None, None
    print(f"[{model}] using {path.relative_to(ROOT)}")
    z = d["z"].float().numpy()
    mean_src = d["attn_mean_src"].float().numpy()
    stim = d["stim"]
    groups = twin_groups(stim)
    sets = head_sets_for(model)
    n_layers, n_heads = z.shape[1], z.shape[2]

    per_head, summary = [], []
    for set_name, hdf in sets.items():
        rows = []
        for r in hdf.itertuples():
            s = head_stats(z, mean_src, stim, groups, int(r.layer), int(r.head))
            rows.append(dict(model=model, head_set=set_name,
                             layer=int(r.layer), head=int(r.head),
                             rho_pad_raw=s[0], rho_src=s[1],
                             rho_pad_resid=s[2], r2_src=s[3]))
        per_head += rows
        summary.append(summarise(rows, set_name, model, len(rows)))
        s = summary[-1]
        print(f"[{model}/{set_name:14s}] pad raw {s['rho_pad_raw']:+.3f} -> "
              f"resid {s['rho_pad_resid']:+.3f} "
              f"({100 * s['attenuation']:.0f}% attenuated)   "
              f"rho(D, d mean-src) {s['rho_src']:+.3f}   "
              f"R2 from position {s['r2_src']:.3f}")

    # random never-selected heads, same statistics
    cma = {(int(r.layer), int(r.head)) for hdf in sets.values()
           for r in hdf.itertuples()}
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
            if (l, h) not in cma]
    rng = np.random.default_rng(seed)
    pick = [pool[i] for i in rng.choice(len(pool), min(n_random, len(pool)),
                                        replace=False)]
    rows = []
    for l, h in pick:
        s = head_stats(z, mean_src, stim, groups, l, h)
        rows.append(dict(model=model, head_set="random", layer=l, head=h,
                         rho_pad_raw=s[0], rho_src=s[1],
                         rho_pad_resid=s[2], r2_src=s[3]))
    per_head += rows
    summary.append(summarise(rows, "random", model, len(rows)))
    s = summary[-1]
    print(f"[{model}/{'random':14s}] pad raw {s['rho_pad_raw']:+.3f} -> "
          f"resid {s['rho_pad_resid']:+.3f} "
          f"({100 * s['attenuation']:.0f}% attenuated)   "
          f"rho(D, d mean-src) {s['rho_src']:+.3f}   "
          f"R2 from position {s['r2_src']:.3f}")
    return pd.DataFrame(per_head), pd.DataFrame(summary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen", "gemma"])
    ap.add_argument("--n_random", type=int, default=60,
                    help="never-selected heads to profile for comparison")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    any_run = False
    for model in args.models:
        ph, sm = run(model, args.n_random, args.seed)
        if ph is None:
            continue
        any_run = True
        suffix = "_gemma" if model == "gemma" else ""
        ph.to_csv(OUT_DIR / f"position_prediction{suffix}.csv", index=False)
        sm.to_csv(OUT_DIR / f"position_prediction_summary{suffix}.csv",
                  index=False)
        print(f"-> {OUT_DIR}/position_prediction{suffix}.csv\n")
    if not any_run:
        sys.exit("no capture with --capture_attn_summary found; nothing run")


if __name__ == "__main__":
    main()
