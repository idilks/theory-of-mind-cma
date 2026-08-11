"""Cross-patching double dissociation: address heads vs divergence heads.

The RSA result (perhead_geometry.csv) says the relational-address code and
the copy/alias divergence code occupy the same band (L25-33) but disjoint
head populations (top-32 overlap = 1 head). This experiment is the causal
test of that operand separation. Two patch families:

ADDRESS AXIS (idx pairs, code_idx_pairs.build_idx_pairs):
  base and exp identical except the mutated+queried index. Patch a head
  SET's hook_z at the final position from base into exp. If the set carries
  the address operand, exp's answer moves toward flip_target = exp-context
  readout of idx_base (works in both frames: copy = answer-changes,
  alias = same-answer).
  Prediction: address set >> divergence set ~ random ~ layer_ctrl.

MECHANISM AXIS (semantics twins, same list/idx/mut, copy vs alias):
  patch base(copy) -> exp(alias) and the reverse. Flip target is the base
  prompt's answer in the exp context.
  Prediction: divergence set >> address set ~ random ~ layer_ctrl.
  (This is the set-level causal analog of the code_abstract CMA; agreement
  with the existing 12-head ablation result is the sanity anchor.)

Metrics per (pair family x head set): mean patch effect on the logit
difference (flip_target - correct), restricted flip preference
P(logit_flip > logit_correct) before vs after, and a per-pair effect vector
saved to .npy for stats. Paired design: every pair is evaluated under every
head set, so set contrasts are within-pair.

Cost: one clean base pass + one clean exp pass + one patched pass per head
set, per pair. Defaults (n_per_cell=1 -> 96 idx pairs + 48 twins, 4 sets)
~ 750 forward passes: an evening on MPS. --limit for a smoke run.

Outputs (results/rsa_head_outputs/cross_patch/):
  pairs_idx.csv, pairs_sem.csv      the generated pairs (reproducible; seed)
  effects_{family}.npy              (n_pairs, n_sets) patch effects
  cross_patch_summary.csv           the headline table
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from code_idx_pairs import build_idx_pairs, pairs_to_records, IdxPair  # noqa: E402
import patch_common as pc  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cross-patch")

OUT_DIR = pc.ROOT / "results/rsa_head_outputs/cross_patch"


def build_semantics_twins(n: int, seed: int) -> list[IdxPair]:
    """Copy/alias twins: same list, same idx, same mut; only semantics flips.
    Reuses IdxPair with idx_base == idx_exp and semantics encoding the patch
    direction: base=copy -> exp=alias and the reverse, alternating."""
    import random
    rng = random.Random(seed + 7)
    from code_idx_pairs import (QWEN_ANSWER_POOL, FILLER_POOL, _scenario)
    twins = []
    for i in range(n):
        idx = rng.choice((1, 2, 3, 4))
        pre = rng.choice((0, 16)); suf = rng.choice((0, 16))
        slot_chars = rng.sample(QWEN_ANSWER_POOL, 4)
        mut = rng.choice([c for c in QWEN_ANSWER_POOL if c not in slot_chars])
        used = set(slot_chars) | {mut}
        edge = [rng.choice([c for c in FILLER_POOL if c not in used])
                for _ in range(2)]
        items = [edge[0]] + slot_chars + [edge[1]]
        pre_v = [rng.choice([c for c in FILLER_POOL if c not in used])
                 for _ in range(pre)]
        suf_v = [rng.choice([c for c in FILLER_POOL if c not in used])
                 for _ in range(suf)]
        base_sem = "copy" if i % 2 == 0 else "alias"
        exp_sem = "alias" if base_sem == "copy" else "copy"
        base_sc = _scenario(items, idx, mut, base_sem, pre_v, suf_v)
        exp_sc = _scenario(items, idx, mut, exp_sem, pre_v, suf_v)
        base_ans = items[idx] if base_sem == "copy" else mut
        exp_ans = mut if base_sem == "copy" else items[idx]
        p = IdxPair(pair_id=i, semantics=f"{base_sem}->{exp_sem}",
                    idx_base=idx, idx_exp=idx, pad_len=pre,
                    suffix_pad_len=suf, list_items=items, mut_char=mut,
                    base_scenario=base_sc, exp_scenario=exp_sc,
                    base_question=f"What is b[{idx}]?",
                    exp_question=f"What is b[{idx}]?",
                    base_answer=base_ans, exp_answer=exp_ans,
                    flip_target=base_ans)
        assert p.flip_target != p.exp_answer
        twins.append(p)
    return twins


def run_family(model, tokenizer, pairs, head_sets, device, family):
    set_names = list(head_sets)
    effects = np.zeros((len(pairs), len(set_names)), dtype=np.float32)
    pref_clean = np.zeros(len(pairs), dtype=np.float32)
    pref_patch = np.zeros((len(pairs), len(set_names)), dtype=np.float32)
    acc_clean = np.zeros(len(pairs), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(pairs):
        bp = pc.chat_prompt(p.base_scenario, p.base_question, tokenizer)
        ep = pc.chat_prompt(p.exp_scenario, p.exp_question, tokenizer)
        bt = pc.encode(tokenizer, bp, device)
        et = pc.encode(tokenizer, ep, device)
        if family == "idx":
            assert bt.shape == et.shape, "idx pair lost token alignment"

        base_z, _ = pc.run_clean(model, bt)
        _, clean_logits = pc.run_clean(model, et)

        fid = pc.probe_id(tokenizer, p.flip_target)
        cid = pc.probe_id(tokenizer, p.exp_answer)
        clean_ld = clean_logits[fid] - clean_logits[cid]
        pref_clean[i] = float(clean_ld > 0)
        ok_ids = pc.answer_variant_ids(tokenizer, p.exp_answer)
        acc_clean[i] = float(clean_logits.argmax().item() in ok_ids)

        for j, sname in enumerate(set_names):
            subs = {lh: base_z[lh] for lh in head_sets[sname]}
            pl = pc.patched_logits(model, et, subs)
            effects[i, j] = float((pl[fid] - pl[cid]) - clean_ld)
            pref_patch[i, j] = float(pl[fid] > pl[cid])
        if device == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i == len(pairs) - 1:
            el = time.time() - t0
            log.info(f"[{family}] {i+1}/{len(pairs)} "
                     f"({el/(i+1):.1f}s/pair, "
                     f"~{el/(i+1)*(len(pairs)-i-1)/60:.0f} min left)")
    return effects, pref_clean, pref_patch, acc_clean, set_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--n_per_cell", type=int, default=1,
                    help="idx pairs per (semantics x 12 idx combos x 2 pads "
                         "x 2 suffixes) cell; 1 -> 96 pairs")
    ap.add_argument("--n_twins", type=int, default=48)
    ap.add_argument("--k", type=int, default=32, help="heads per set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perhead_csv", type=Path, default=pc.PERHEAD_CSV)
    ap.add_argument("--divergence_by", default="fp_C1",
                    choices=["fp_C1", "semantics"])
    ap.add_argument("--head_sets_json", type=Path, default=None,
                    help="JSON {set_name: [[layer, head], ...]} overriding "
                         "the derived sets entirely — e.g. "
                         "subset_split/subset_membership.json to patch "
                         "div_shared / div_unique separately. Controls are "
                         "NOT auto-added; include them in the JSON.")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: run only the first N pairs of each family "
                         "(writes to a _smoke/ output dir, never the "
                         "canonical one)")
    args = ap.parse_args()

    global OUT_DIR
    if args.limit:
        OUT_DIR = OUT_DIR.with_name(OUT_DIR.name + "_smoke")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.head_sets_json:
        raw = json.loads(Path(args.head_sets_json).read_text())
        head_sets = {name: [tuple(lh) for lh in lhs]
                     for name, lhs in raw.items()}
    else:
        head_sets = pc.derive_head_sets(args.perhead_csv, k=args.k,
                                        seed=args.seed,
                                        divergence_by=args.divergence_by)
    for n, s in head_sets.items():
        log.info(f"head set {n}: {len(s)} heads, e.g. {s[:3]}")
    (OUT_DIR / "head_sets.json").write_text(
        json.dumps({k: v for k, v in head_sets.items()}, indent=2))

    idx_pairs = build_idx_pairs(args.n_per_cell, args.seed)
    twins = build_semantics_twins(args.n_twins, args.seed)
    if args.limit:
        idx_pairs, twins = idx_pairs[:args.limit], twins[:args.limit]
    pd.DataFrame(pairs_to_records(idx_pairs)).to_csv(
        OUT_DIR / "pairs_idx.csv", index=False)
    pd.DataFrame(pairs_to_records(twins)).to_csv(
        OUT_DIR / "pairs_sem.csv", index=False)
    log.info(f"{len(idx_pairs)} idx pairs, {len(twins)} semantics twins")

    log.info("loading model (bf16; transient ~2x memory peak)")
    model, tokenizer = pc.load_model(args.device)

    rows = []
    for family, pairs in [("idx", idx_pairs), ("sem", twins)]:
        eff, pcln, ppat, acc, names = run_family(
            model, tokenizer, pairs, head_sets, args.device, family)
        np.save(OUT_DIR / f"effects_{family}.npy", eff)
        for j, sname in enumerate(names):
            rows.append({
                "family": family, "head_set": sname,
                "n_pairs": len(pairs),
                "mean_effect": float(eff[:, j].mean()),
                "sem_effect": float(eff[:, j].std(ddof=1)
                                    / np.sqrt(len(pairs))),
                "flip_pref_clean": float(pcln.mean()),
                "flip_pref_patched": float(ppat[:, j].mean()),
                "clean_acc": float(acc.mean()),
            })
        log.info(f"[{family}] clean accuracy {acc.mean():.2f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "cross_patch_summary.csv", index=False)
    print("\n=== cross-patch summary (effect = Δ logit-diff toward flip) ===")
    print(summary.round(3).to_string(index=False))
    print("\npredictions: family=idx -> address >> others; "
          "family=sem -> divergence >> others.")
    print(f"per-pair effect matrices in {OUT_DIR} for paired stats.")


if __name__ == "__main__":
    main()
