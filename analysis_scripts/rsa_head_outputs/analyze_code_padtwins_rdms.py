#!/usr/bin/env python
"""Pad-twins RDM analysis (Exp 3 of the code RDM arm): veracity TAG vs
LOOKBACK/pointer, adjudicated within-cell.

Two analyses on the code_padtwins capture:

  A. Exp-2 replication — the full Mantel battery + divergence partials from
     analyze_code_ledger_rdms, re-fit on the balanced grid. Because the query
     cells are identical to Exp 2, these numbers are directly comparable to
     mantel_code_ledger.csv (same models, same head sets, ~6x more pairs).

  B. The pad-twin test (new). Within each twin group (fixed query cell +
     fixed content, 9 stimuli differing ONLY in prefix/suffix pad), correlate
     pairwise neural distance with |d prefix pad| (source tokens move) and
     |d suffix pad| (distance-to-end moves). Null: permute pad labels within
     each twin group (exchangeable under the TAG hypothesis), 1000x.
       TAG predicts     mean within-group rho ~ 0, distances at the twin floor;
       POINTER predicts rho > 0 for the prefix contrast specifically.
     Also reported: mean within-twin distance by (dpre, dsuf) bucket, the
     content-twin floor (same cell+pads, different content), and the
     between-cell mean — the three reference levels.

Outputs (results/rsa_head_outputs/aggregation_code_padtwins/):
  mantel_code_padtwins{_gemma}.csv     — analysis A
  padtwin_test{_gemma}.csv             — analysis B, per head set
  padtwin_distance_buckets{_gemma}.csv — mean distance per (dpre, dsuf)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import (  # noqa: E402
    HEADS_CSV, GEMMA_HEADSET_DIR, HEAD_SETS, GEMMA_HEAD_SETS, GEMMA_TOKENIZER,
    DIVERGENCE_MODELS, build_model_rdms, mantel, partial_spearman, upper_tri)

# Within a 3x3 pad grid the 36 pair contrasts |dpre| and |dsuf| are negatively
# dependent (the identical-pads pair is excluded), so a pure-prefix signal
# induces a spurious negative raw rho_suffix. The partials (prefix|suffix,
# suffix|prefix) are the interpretable quantities; raw rhos are kept for
# reference.

ACT_DIR = ROOT / "results/rsa_head_outputs/activations_code_padtwins_full"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_code_padtwins"
ACT_PATHS = {"qwen": ACT_DIR / "Qwen2.5-14B-Instruct/code_padtwins.pt",
             "gemma": ACT_DIR / "gemma-2-27b-it/code_padtwins.pt"}


def load(model):
    d = torch.load(ACT_PATHS[model], weights_only=False)
    if model == "gemma":
        sets = {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in GEMMA_HEAD_SETS}
    else:
        heads = pd.read_csv(HEADS_CSV)
        sets = {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}
    return d, sets


def set_neural_rdm(z, hdf):
    zs = np.stack([z[:, int(r.layer), int(r.head), :] for r in hdf.itertuples()],
                  axis=1)
    return np.mean([cosine_rdm(zs[:, h, :]) for h in range(zs.shape[1])], axis=0)


def mantel_battery(neural, stim, set_name, n_heads, n_perm, rng, rows,
                   tokenizer):
    """Analysis A: identical structure to analyze_code_ledger_rdms.main."""
    model_rdms = build_model_rdms(stim, tokenizer)
    nvec = upper_tri(neural)
    for model_name, m_rdm in model_rdms.items():
        r, p = mantel(neural, m_rdm, n_perm, rng)
        rows.append({"head_set": set_name, "n_heads": n_heads,
                     "model": model_name, "spearman_r": r, "p_perm": p})
        print(f"{set_name:14s} ({n_heads:2d} heads) vs {model_name:17s} "
              f"r = {r:+.3f}  p = {p:.4f}")
    sd, od = (upper_tri(model_rdms[m]) for m in DIVERGENCE_MODELS)
    for label, y, c in (("slot_div|object_div", sd, od),
                        ("object_div|slot_div", od, sd)):
        pr = partial_spearman(nvec, y, c)
        rows.append({"head_set": set_name, "n_heads": n_heads,
                     "model": label, "spearman_r": pr, "p_perm": np.nan})
        print(f"{set_name:14s} partial {label} = {pr:+.3f}")


def padtwin_test(neural, stim, n_perm, rng):
    """Analysis B: within-twin-group distance vs pad-shift correlations."""
    groups = {}
    for i, r in enumerate(stim):
        groups.setdefault(r["twin_id"], []).append(i)
    for tid, idx in groups.items():
        assert len(idx) == 9, f"{tid}: {len(idx)} stimuli (expected 9)"

    ut9 = np.triu_indices(9, 1)

    def group_arrays(idx):
        D = neural[np.ix_(idx, idx)][ut9]
        pre = np.array([stim[i]["pad_len"] for i in idx], float)
        suf = np.array([stim[i]["suffix_pad_len"] for i in idx], float)
        return D, pre, suf

    def mean_stats(perms=None):
        """(raw rho_pre, raw rho_suf, partial pre|suf, partial suf|pre),
        each averaged over twin groups."""
        acc = np.zeros(4)
        for g in range(len(cached)):
            D, pre, suf = cached[g]
            if perms is not None:
                p = perms[g]
                pre, suf = pre[p], suf[p]
            dpre = np.abs(pre[ut9[0]] - pre[ut9[1]])
            dsuf = np.abs(suf[ut9[0]] - suf[ut9[1]])
            acc[0] += spearmanr(D, dpre).statistic
            acc[1] += spearmanr(D, dsuf).statistic
            acc[2] += partial_spearman(D, dpre, dsuf)
            acc[3] += partial_spearman(D, dsuf, dpre)
        return acc / len(cached)

    cached = [group_arrays(idx) for idx in groups.values()]
    obs = mean_stats()

    null = np.empty((n_perm, 4))
    for k in range(n_perm):
        perms = [rng.permutation(9) for _ in groups]
        null[k] = mean_stats(perms)
    pvals = [(float(np.sum(np.abs(null[:, j]) >= abs(obs[j])) + 1) / (n_perm + 1))
             for j in range(4)]
    null_max = [float(np.abs(null[:, j]).max()) for j in range(4)]

    # distance buckets within twin groups
    bucket = {}
    for g, idx in enumerate(groups.values()):
        D, pre, suf = cached[g]
        dpre = np.abs(pre[ut9[0]] - pre[ut9[1]])
        dsuf = np.abs(suf[ut9[0]] - suf[ut9[1]])
        for d, a, b in zip(D, dpre, dsuf):
            bucket.setdefault((int(a), int(b)), []).append(d)
    buckets = {k: float(np.mean(v)) for k, v in sorted(bucket.items())}

    # reference floors
    n = len(stim)
    iu = np.triu_indices(n, 1)
    cell = [tuple(stim[i][k] for k in ("queried_object", "queried_slot_type",
                                       "role_b", "copy_syntax", "alias_syntax"))
            for i in range(n)]
    twin = [stim[i]["twin_id"] for i in range(n)]
    pads = [(stim[i]["pad_len"], stim[i]["suffix_pad_len"]) for i in range(n)]
    flat = neural[iu]
    same_twin = np.array([twin[i] == twin[j] for i, j in zip(*iu)])
    same_cell = np.array([cell[i] == cell[j] for i, j in zip(*iu)])
    same_pads = np.array([pads[i] == pads[j] for i, j in zip(*iu)])
    content_floor = float(flat[same_cell & ~same_twin & same_pads].mean())
    within_twin = float(flat[same_twin].mean())
    between_cell = float(flat[~same_cell].mean())

    keys = ("rho_prefix", "rho_suffix", "rho_prefix|suffix", "rho_suffix|prefix")
    res = {}
    for j, k in enumerate(keys):
        res[k] = float(obs[j])
        res[f"p_{k}"] = pvals[j]
        res[f"max_null_{k}"] = null_max[j]
    res.update({"within_twin_dist": within_twin,
                "content_twin_dist": content_floor,
                "between_cell_dist": between_cell})
    return res, buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen")
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_mantel", action="store_true",
                    help="run only the pad-twin test (analysis B)")
    args = ap.parse_args()

    d, sets = load(args.model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    if "correct" in d:
        print(f"baseline accuracy: {np.mean(d['correct']):.3f}")
    rng = np.random.default_rng(args.seed)
    suffix = "_gemma" if args.model == "gemma" else ""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = (GEMMA_TOKENIZER if args.model == "gemma"
                 else "Qwen/Qwen2.5-14B-Instruct")
    mantel_rows, test_rows, bucket_rows = [], [], []
    for set_name, hdf in sets.items():
        neural = set_neural_rdm(z, hdf)
        res, buckets = padtwin_test(neural, stim, args.n_perm, rng)
        test_rows.append({"head_set": set_name, "n_heads": len(hdf), **res})
        print(f"\n[{set_name}] pad-twin test (partials are primary): "
              f"pre|suf = {res['rho_prefix|suffix']:+.3f} "
              f"(p = {res['p_rho_prefix|suffix']:.4f}, "
              f"max|null| = {res['max_null_rho_prefix|suffix']:.3f})  "
              f"suf|pre = {res['rho_suffix|prefix']:+.3f} "
              f"(p = {res['p_rho_suffix|prefix']:.4f})  "
              f"[raw: pre {res['rho_prefix']:+.3f}, suf {res['rho_suffix']:+.3f}]")
        print(f"[{set_name}] floors: within-twin {res['within_twin_dist']:.4f}  "
              f"content-twin {res['content_twin_dist']:.4f}  "
              f"between-cell {res['between_cell_dist']:.4f}")
        for (dpre, dsuf), md in buckets.items():
            bucket_rows.append({"head_set": set_name, "d_prefix": dpre,
                                "d_suffix": dsuf, "mean_dist": md})
        if not args.skip_mantel:
            mantel_battery(neural, stim, set_name, len(hdf),
                           args.n_perm, rng, mantel_rows, tokenizer)

    pd.DataFrame(test_rows).to_csv(OUT_DIR / f"padtwin_test{suffix}.csv",
                                   index=False)
    pd.DataFrame(bucket_rows).to_csv(
        OUT_DIR / f"padtwin_distance_buckets{suffix}.csv", index=False)
    if not args.skip_mantel:
        pd.DataFrame(mantel_rows).to_csv(
            OUT_DIR / f"mantel_code_padtwins{suffix}.csv", index=False)
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    main()
