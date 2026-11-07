#!/usr/bin/env python
"""Projection test on the pad-twins capture: is the divergence code a tag
riding on a shared positional carrier, or entangled with position?

Per head:
  1. Estimate the PAD SUBSPACE: within-twin displacement vectors where
     exactly one pad changes (9 prefix-only + 9 suffix-only pairs per twin
     group = 1728 vectors per head), SVD, top-k right singular vectors.
     Diagnostics: explained-variance ratio of the top-k (carrier prediction:
     low-rank) and split-half stability of PC1 (|cos| between PC1 fit on two
     disjoint halves of the twin groups; carrier prediction: high, i.e. the
     pad response direction is content-independent).
  2. Project the pad subspace out of every stimulus's z, rebuild the cosine
     RDM; also project out k RANDOM orthonormal directions as a matched-rank
     control.

Per head set (mean of per-head RDMs), re-run:
  - the pad-twin test (padtwin_test from analyze_code_padtwins_rdms);
  - Mantel fits for object_divergence, answer_identity, queried_slot_abs
    (r for all variants; permutation p for object_divergence after
    projection — the headline number).

Predictions:
  tag + carrier      pad-twin rho collapses under pad-projection (not under
                     random projection); object-divergence fit survives.
  entangled pointer  object-divergence fit collapses with the pad subspace.

Output: results/rsa_head_outputs/aggregation_code_padtwins/
  projection_test{_gemma}.csv       (set-level results, per variant)
  pad_direction_stats{_gemma}.csv   (per-head subspace diagnostics)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import (  # noqa: E402
    GEMMA_TOKENIZER, build_model_rdms, mantel, upper_tri)
from analyze_code_padtwins_rdms import OUT_DIR, load, padtwin_test  # noqa: E402

MANTEL_MODELS = ("object_divergence", "answer_identity", "queried_slot_abs")


def twin_groups(stim):
    groups = {}
    for i, r in enumerate(stim):
        groups.setdefault(r["twin_id"], []).append(i)
    return list(groups.values())


def displacement_pairs(stim, groups):
    """(i, j) with same twin group and exactly one pad differing."""
    pairs = []
    for idx in groups:
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                dpre = stim[i]["pad_len"] != stim[j]["pad_len"]
                dsuf = stim[i]["suffix_pad_len"] != stim[j]["suffix_pad_len"]
                if dpre != dsuf:  # exactly one pad changes
                    pairs.append((i, j))
    return pairs


def pad_subspace(zh, pairs, k):
    """Top-k right singular vectors of the displacement matrix, with
    explained-variance ratio and split-half PC1 stability."""
    disp = np.stack([zh[j] - zh[i] for i, j in pairs])
    _, s, vt = np.linalg.svd(disp, full_matrices=False)
    evr = float((s[:k] ** 2).sum() / (s ** 2).sum())
    half = len(pairs) // 2
    _, _, vt_a = np.linalg.svd(disp[:half], full_matrices=False)
    _, _, vt_b = np.linalg.svd(disp[half:], full_matrices=False)
    stability = float(abs(vt_a[0] @ vt_b[0]))
    return vt[:k].T, evr, stability  # U: (dim, k)


def project_out(zh, U):
    return zh - (zh @ U) @ U.T


def random_dirs(dim, k, rng):
    q, _ = np.linalg.qr(rng.normal(size=(dim, k)))
    return q[:, :k]


def set_rdms(z, hdf, pairs, k, rng, head_rows, set_name, model):
    """Per-head: raw / pad-projected / random-projected RDMs; mean per set."""
    raw, proj, ctrl = [], [], []
    for r in hdf.itertuples():
        zh = z[:, int(r.layer), int(r.head), :]
        U, evr, stab = pad_subspace(zh, pairs, k)
        head_rows.append({"model": model, "head_set": set_name,
                          "layer": int(r.layer), "head": int(r.head),
                          "evr_topk": evr, "pc1_split_half_cos": stab})
        raw.append(cosine_rdm(zh))
        proj.append(cosine_rdm(project_out(zh, U)))
        ctrl.append(cosine_rdm(project_out(zh, random_dirs(zh.shape[1], k, rng))))
    return (np.mean(raw, axis=0), np.mean(proj, axis=0), np.mean(ctrl, axis=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen")
    ap.add_argument("--k", type=int, default=2, help="pad-subspace rank")
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="pad-twin test permutations")
    ap.add_argument("--n_perm_mantel", type=int, default=500,
                    help="Mantel permutations for object_div after projection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, sets = load(args.model)
    z = d["z"].float().numpy()
    stim = d["stim"]
    groups = twin_groups(stim)
    pairs = displacement_pairs(stim, groups)
    print(f"{len(groups)} twin groups, {len(pairs)} single-pad displacement "
          f"pairs, k = {args.k}")

    tokenizer = (GEMMA_TOKENIZER if args.model == "gemma"
                 else "Qwen/Qwen2.5-14B-Instruct")
    model_rdms = {m: r for m, r in build_model_rdms(stim, tokenizer).items()
                  if m in MANTEL_MODELS}
    mvecs = {m: upper_tri(r) for m, r in model_rdms.items()}

    rng = np.random.default_rng(args.seed)
    rows, head_rows = [], []
    for set_name, hdf in sets.items():
        rdm_raw, rdm_proj, rdm_ctrl = set_rdms(
            z, hdf, pairs, args.k, rng, head_rows, set_name, args.model)
        for variant, R in (("raw", rdm_raw), ("pad_projected", rdm_proj),
                           ("random_projected", rdm_ctrl)):
            res, _ = padtwin_test(R, stim, args.n_perm, rng)
            row = {"model": args.model, "head_set": set_name,
                   "n_heads": len(hdf), "k": args.k, "variant": variant,
                   "rho_prefix|suffix": res["rho_prefix|suffix"],
                   "p_rho_prefix|suffix": res["p_rho_prefix|suffix"],
                   "rho_suffix|prefix": res["rho_suffix|prefix"],
                   "within_twin_dist": res["within_twin_dist"],
                   "content_twin_dist": res["content_twin_dist"]}
            nvec = upper_tri(R)
            for m in MANTEL_MODELS:
                row[f"r_{m}"] = float(spearmanr(nvec, mvecs[m]).statistic)
            if variant == "pad_projected":
                _, p = mantel(R, model_rdms["object_divergence"],
                              args.n_perm_mantel, rng)
                row["p_object_div"] = p
            rows.append(row)
            print(f"[{set_name}/{variant}] pre|suf = "
                  f"{row['rho_prefix|suffix']:+.3f}  obj_div r = "
                  f"{row['r_object_divergence']:+.3f}  slot_abs r = "
                  f"{row['r_queried_slot_abs']:+.3f}  answer r = "
                  f"{row['r_answer_identity']:+.3f}")
        hsub = [h for h in head_rows if h["head_set"] == set_name]
        print(f"[{set_name}] pad subspace: mean top-{args.k} EVR = "
              f"{np.mean([h['evr_topk'] for h in hsub]):.3f}, mean PC1 "
              f"split-half |cos| = "
              f"{np.mean([h['pc1_split_half_cos'] for h in hsub]):.3f}\n")

    suffix = ("_gemma" if args.model == "gemma" else "") + \
        (f"_k{args.k}" if args.k != 2 else "")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"projection_test{suffix}.csv",
                              index=False)
    pd.DataFrame(head_rows).to_csv(
        OUT_DIR / f"pad_direction_stats{suffix}.csv", index=False)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
