#!/usr/bin/env python
"""Pointer RDM test on the code-position capture (week plan Steps 5-6).

For each head set (code_abstract, code_control, tom_c1, tom_c5, all-union),
build the neural RDM as the unweighted mean of per-head cosine RDMs
(rsa_utils.cosine_rdm; same protocol as analyze_rdms_per_condition.py) over
the 144 code-position stimuli, then Mantel-test it against the design's model
RDMs:

  semantics        copy/alias (the veracity analog)
  queried_slot_rel |idx_i - idx_j| (relational pointer)
  queried_slot_abs |token pos of slot idx| (absolute pointer; prefix-pad-driven)
  slot_dist_to_end distance from slot to final token (suffix-pad-driven)
  answer_value_pos position of the answer value token
  answer_identity  same/different answer char
  prompt_length    total token count

The grid decorrelates semantics from both queried-slot RDMs (design r ~ -0.01)
so a significant fit to those cannot be explained by the veracity analog.
Token positions are recomputed with the Qwen tokenizer (falls back to the
stored char offsets if unavailable).

Output: results/rsa_head_outputs/aggregation_code_position/mantel_code_position.csv
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

ACT_PATH = ROOT / "results/rsa_head_outputs/activations_code_position/code_position.pt"
HEADS_CSV = ROOT / "results/rsa_head_outputs/topk_heads/code_position_union.csv"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_code_position"

HEAD_SETS = ["code_abstract", "code_control", "tom_c1", "tom_c5"]

# --model gemma: full-head capture, story-CMA-derived head sets (no Gemma code
# CMA exists), Gemma tokenizer for token-position model RDMs.
GEMMA_ACT_PATH = (ROOT / "results/rsa_head_outputs/activations_code_position_full/"
                  "gemma-2-27b-it/code_position.pt")
GEMMA_HEADSET_DIR = ROOT / "results/aggregated/code_head_sets"
GEMMA_HEAD_SETS = ["gemma_c1", "gemma_c2", "gemma_c3", "gemma_c5"]
GEMMA_TOKENIZER = "google/gemma-2-27b-it"


def upper_tri(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices(M.shape[0], 1)]


def mantel(d1: np.ndarray, d2: np.ndarray, n_perm: int, rng) -> tuple[float, float]:
    obs = float(spearmanr(upper_tri(d1), upper_tri(d2)).correlation)
    n = d1.shape[0]
    flat1 = upper_tri(d1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        null[i] = float(spearmanr(flat1, upper_tri(d2[perm][:, perm])).correlation)
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p


def token_positions(stim: list[dict], tokenizer_name="Qwen/Qwen2.5-14B-Instruct"):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"[warn] tokenizer unavailable ({type(e).__name__}); using char offsets")
        return (np.array([r["slot_char_pos"] for r in stim], float),
                np.array([r["ans_char_pos"] for r in stim], float),
                np.array([r["scenario_len"] for r in stim], float))
    slot, ans, length = [], [], []
    for r in stim:
        enc = tok(r["scenario"], add_special_tokens=False, return_offsets_mapping=True)
        def tok_at(cp):
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                if s <= cp < e:
                    return i
            return len(enc["offset_mapping"]) - 1
        slot.append(tok_at(r["slot_char_pos"]))
        ans.append(tok_at(r["ans_char_pos"]))
        length.append(len(enc["offset_mapping"]))
    return np.array(slot, float), np.array(ans, float), np.array(length, float)


def build_model_rdms(stim: list[dict],
                     tokenizer_name="Qwen/Qwen2.5-14B-Instruct") -> dict[str, np.ndarray]:
    slot, ans, length = token_positions(stim, tokenizer_name)
    sem = np.array([1.0 if r["semantics"] == "copy" else 0.0 for r in stim])
    idx = np.array([float(r["idx"]) for r in stim])
    ans_id = np.array([float(ord(r["answer"][1])) for r in stim])
    dist_end = length - slot
    return {
        "semantics": (sem[:, None] != sem[None, :]).astype(float),
        "queried_slot_rel": np.abs(idx[:, None] - idx[None, :]),
        "queried_slot_abs": np.abs(slot[:, None] - slot[None, :]),
        "slot_dist_to_end": np.abs(dist_end[:, None] - dist_end[None, :]),
        "answer_value_pos": np.abs(ans[:, None] - ans[None, :]),
        "answer_identity": (ans_id[:, None] != ans_id[None, :]).astype(float),
        "prompt_length": np.abs(length[:, None] - length[None, :]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=1000,
                    help="Mantel permutations (1000 fast, 5000 final)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen",
                    help="gemma = full-head capture + story-CMA head sets")
    args = ap.parse_args()

    if args.model == "gemma":
        d = torch.load(GEMMA_ACT_PATH, weights_only=False)
        z = d["z"].float().numpy()      # (n_stim, n_layers, n_heads, head_dim)
        stim = d["stim"]
        if "correct" in d:
            print(f"baseline accuracy: {np.mean(d['correct']):.3f}")
        model_rdms = build_model_rdms(stim, GEMMA_TOKENIZER)
        set_rdms = {}
        for s in GEMMA_HEAD_SETS:
            hdf = pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
            set_rdms[s] = (len(hdf), np.mean(
                [cosine_rdm(z[:, int(r.layer), int(r.head), :])
                 for r in hdf.itertuples()], axis=0))
    else:
        d = torch.load(ACT_PATH, weights_only=False)
        z = d["z"].float().numpy()      # (n_stim, n_heads, head_dim)
        stim = d["stim"]
        heads = pd.read_csv(HEADS_CSV)
        assert len(heads) == z.shape[1], "heads CSV / capture mismatch"
        model_rdms = build_model_rdms(stim)
        head_rdms = np.stack([cosine_rdm(z[:, h, :]) for h in range(z.shape[1])])
        set_members = {s: heads.index[heads["sources"].str.contains(s)].to_numpy()
                       for s in HEAD_SETS}
        set_members["all_union"] = np.arange(len(heads))
        set_rdms = {s: (len(m), head_rdms[m].mean(axis=0))
                    for s, m in set_members.items()}

    rng = np.random.default_rng(args.seed)
    rows = []
    for set_name, (n_heads, neural) in set_rdms.items():
        for model_name, m_rdm in model_rdms.items():
            r, p = mantel(neural, m_rdm, args.n_perm, rng)
            rows.append({"head_set": set_name, "n_heads": n_heads,
                         "model": model_name, "spearman_r": r, "p_perm": p})
            print(f"{set_name:14s} ({n_heads:2d} heads) vs {model_name:17s} "
                  f"r = {r:+.3f}  p = {p:.4f}")
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_gemma" if args.model == "gemma" else ""
    out = OUT_DIR / f"mantel_code_position{suffix}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
