#!/usr/bin/env python
"""Divergence-ledger RDM analysis (Exp 2 of the code RDM arm).

Loads the full-head code_ledger capture, builds set-level neural RDMs (mean of
per-head cosine RDMs over the union-CSV head sets), and Mantel-tests them
against the ledger model RDMs. Adjudicates the nested slot- vs object-
divergence models via partial Spearman (rank residualization) since they
correlate ~0.25 by construction.

Robustness: --correct_only drops stimuli the model answered wrongly in the
capture's greedy decode (the (alias, mutated) cell sits at 0.778 accuracy);
run both and compare before interpreting.

Output: results/rsa_head_outputs/aggregation_code_ledger/mantel_code_ledger{,_correct_only}.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402

ACT_PATH = (ROOT / "results/rsa_head_outputs/activations_code_ledger_full/"
            "Qwen2.5-14B-Instruct/code_ledger.pt")
HEADS_CSV = ROOT / "results/rsa_head_outputs/topk_heads/code_position_union.csv"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_code_ledger"

HEAD_SETS = ["code_abstract", "code_control", "tom_c1", "tom_c5"]

# --model gemma: full-head capture, story-CMA-derived head sets, Gemma tokenizer
GEMMA_ACT_PATH = (ROOT / "results/rsa_head_outputs/activations_code_ledger_full/"
                  "gemma-2-27b-it/code_ledger.pt")
GEMMA_HEADSET_DIR = ROOT / "results/aggregated/code_head_sets"
GEMMA_HEAD_SETS = ["gemma_c1", "gemma_c2", "gemma_c3", "gemma_c5"]
GEMMA_TOKENIZER = "google/gemma-2-27b-it"
DIVERGENCE_MODELS = ("slot_divergence", "object_divergence")


def upper_tri(M):
    return M[np.triu_indices(M.shape[0], 1)]


def mantel(d1, d2, n_perm, rng):
    obs = float(spearmanr(upper_tri(d1), upper_tri(d2)).correlation)
    flat1 = upper_tri(d1)
    n = d1.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = float(spearmanr(flat1, upper_tri(d2[p][:, p])).correlation)
    return obs, float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def partial_spearman(y, x, control):
    """Spearman(y, x | control) via rank residualization on RDM vectors."""
    ry, rx, rc = rankdata(y), rankdata(x), rankdata(control)
    def resid(a, b):
        b1 = np.column_stack([np.ones_like(b), b])
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef
    return float(np.corrcoef(resid(ry, rc), resid(rx, rc))[0, 1])


def build_model_rdms(stim, tokenizer_name="Qwen/Qwen2.5-14B-Instruct"):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        slot_pos, length = [], []
        for r in stim:
            enc = tok(r["scenario"], add_special_tokens=False,
                      return_offsets_mapping=True)
            slot_pos.append(next((i for i, (s, e) in enumerate(enc["offset_mapping"])
                                  if s <= r["slot_char_pos"] < e),
                                 len(enc["offset_mapping"]) - 1))
            length.append(len(enc["offset_mapping"]))
        slot_pos, length = np.array(slot_pos, float), np.array(length, float)
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({type(e).__name__}); char proxy")
        slot_pos = np.array([r["slot_char_pos"] for r in stim], float)
        length = np.array([r["scenario_len"] for r in stim], float)

    cat = lambda key: np.array([hash(r[key]) for r in stim], float)  # noqa: E731
    slot_div = np.array([1.0 if (r["queried_object"] == "copy"
                                 and r["queried_slot_type"] == "mutated") else 0.0
                         for r in stim])
    scalars = {"queried_slot_abs": slot_pos, "prompt_length": length}
    cats = {
        "slot_divergence": slot_div,
        "object_divergence": cat("queried_object"),
        "queried_slot_type": cat("queried_slot_type"),
        "queried_name": cat("queried_name"),
        "copy_syntax": cat("copy_syntax"),
        "alias_syntax": cat("alias_syntax"),
        "answer_identity": cat("answer"),
    }
    rdms = {k: (v[:, None] != v[None, :]).astype(float) for k, v in cats.items()}
    rdms.update({k: np.abs(v[:, None] - v[None, :]) for k, v in scalars.items()})
    return rdms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--correct_only", action="store_true",
                    help="drop stimuli the model answered wrongly")
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen",
                    help="gemma = full-head capture + story-CMA head sets")
    args = ap.parse_args()

    act_path = GEMMA_ACT_PATH if args.model == "gemma" else ACT_PATH
    d = torch.load(act_path, weights_only=False)
    z = d["z"].float().numpy()  # (n_stim, n_layers, n_heads, head_dim)
    stim = d["stim"]
    if "correct" in d:
        print(f"baseline accuracy: {np.mean(d['correct']):.3f}")

    keep = np.arange(len(stim))
    if args.correct_only:
        keep = np.array([i for i, c in enumerate(d["correct"]) if c])
        print(f"[correct-only] keeping {len(keep)}/{len(stim)} stimuli")
        z, stim = z[keep], [stim[i] for i in keep]

    if args.model == "gemma":
        model_rdms = build_model_rdms(stim, GEMMA_TOKENIZER)
        set_rows = {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                    for s in GEMMA_HEAD_SETS}
    else:
        heads = pd.read_csv(HEADS_CSV)
        model_rdms = build_model_rdms(stim)
        set_rows = {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}
    rng = np.random.default_rng(args.seed)
    rows = []
    for set_name, hdf in set_rows.items():
        zs = np.stack([z[:, int(r.layer), int(r.head), :] for r in hdf.itertuples()],
                      axis=1)
        neural = np.mean([cosine_rdm(zs[:, h, :]) for h in range(zs.shape[1])], axis=0)
        nvec = upper_tri(neural)
        fits = {}
        for model_name, m_rdm in model_rdms.items():
            r, p = mantel(neural, m_rdm, args.n_perm, rng)
            fits[model_name] = r
            rows.append({"head_set": set_name, "n_heads": len(hdf),
                         "model": model_name, "spearman_r": r, "p_perm": p})
            print(f"{set_name:14s} ({len(hdf):2d} heads) vs {model_name:17s} "
                  f"r = {r:+.3f}  p = {p:.4f}")
        # nested-model adjudication
        sd, od = (upper_tri(model_rdms[m]) for m in DIVERGENCE_MODELS)
        p_sd = partial_spearman(nvec, sd, od)
        p_od = partial_spearman(nvec, od, sd)
        rows.append({"head_set": set_name, "n_heads": len(hdf),
                     "model": "slot_div|object_div", "spearman_r": p_sd, "p_perm": np.nan})
        rows.append({"head_set": set_name, "n_heads": len(hdf),
                     "model": "object_div|slot_div", "spearman_r": p_od, "p_perm": np.nan})
        print(f"{set_name:14s} partials: slot|object = {p_sd:+.3f}   "
              f"object|slot = {p_od:+.3f}\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_gemma" if args.model == "gemma" else "") + \
             ("_correct_only" if args.correct_only else "")
    out = OUT_DIR / f"mantel_code_ledger{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
