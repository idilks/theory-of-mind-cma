#!/usr/bin/env python
"""Random-head nulls for the code Exp 2 (divergence ledger) partial correlation.

The appendix claim (draft2 Table rdm-exp2): divergence-set RDMs fit object
divergence after partialling out slot divergence (obj|slot = +0.282 code
divergence set, +0.213 story divergence set), while the reverse partial is ~0.
Its nulls are stimulus-label permutations, which test "is there structure", not
"is the structure specific to these heads". On the content-separable story set,
layer-matched random heads separate veracity as well as the divergence set
(fig15b follow-up, Nov 6), so this asks the same question of the code result.

Statistic per head set: partial Spearman obj|slot of the set RDM (mean of
per-head cosine RDMs), plus raw object-divergence r and slot|obj.
Nulls (each draw has the target set's size; all draws exclude every head in the
union head-set CSV, i.e. never-selected heads):
  unmatched      uniform over never-selected heads
  layer_matched  one never-selected head from each target head's layer
  inert_matched  layer-matched, restricted to heads at or below the median of the
                 layer band on BOTH the story C1 fingerprint and the code K1
                 (code_abstract) causal score -- heads with no causal role in
                 either divergence condition
K1 score: seed-1 end-position causal_scores.pt, mean of the ABA->ABB and
ABB->ABA directions, negatives clamped to 0.

Output: results/rsa_head_outputs/aggregation_code_ledger/random_null_partials.csv
(one row per target set x null) and random_null_draws.csv (all draws).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rsa_head_outputs"))
from analyze_code_ledger_rdms import (ACT_PATH, DIVERGENCE_MODELS, HEADS_CSV, OUT_DIR,  # noqa: E402
                                      build_model_rdms, partial_spearman, upper_tri)
from rsa_utils import cosine_rdm  # noqa: E402

N_LAYERS, N_HEADS = 48, 40
FP_C1 = ROOT / "results/aggregated/fingerprints/C1_abstract_belief_clamped.npy"
K1_DIR = ROOT / ("results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct/code_abstract/"
                 "template_indexed_list")
K1_FILES = ["base_rule_ABA_exp_rule_ABB/patch_before_movement/z_seed_1_shuffle_False/logit/"
            "sample_num_149_gen_acc_0.9/causal_scores.pt",
            "base_rule_ABB_exp_rule_ABA/patch_before_movement/z_seed_1_shuffle_False/logit/"
            "sample_num_149_gen_acc_0.9/causal_scores.pt"]
TARGETS = ["code_abstract", "tom_c1"]          # divergence sets (nulls drawn for these)
REFERENCE = ["code_control", "tom_c5"]         # answer sets (observed values only)
PAPER = {"code_abstract": 0.282, "tom_c1": 0.213}


def k1_score() -> np.ndarray:
    s = [torch.load(K1_DIR / f, map_location="cpu", weights_only=False).float().numpy()
         for f in K1_FILES]
    return np.clip(np.mean(s, axis=0), 0, None).reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    d = torch.load(ACT_PATH, map_location="cpu", weights_only=False)
    z = d["z"].float().numpy()
    stim = d["stim"]
    n = len(stim)
    models = build_model_rdms(stim)
    sd, od = (upper_tri(models[m]) for m in DIVERGENCE_MODELS)
    print(f"{n} stimuli; computing {N_LAYERS * N_HEADS} per-head RDMs")
    per = np.stack([upper_tri(cosine_rdm(z[:, l, h, :]))
                    for l in range(N_LAYERS) for h in range(N_HEADS)]).astype(np.float64)

    heads = pd.read_csv(HEADS_CSV)
    flat = lambda df: (df["layer"] * N_HEADS + df["head"]).to_numpy()  # noqa: E731
    sets = {s: flat(heads[heads["sources"].str.contains(s)]) for s in TARGETS + REFERENCE}
    claimed = set(flat(heads).tolist())
    never = np.array([h for h in range(N_LAYERS * N_HEADS) if h not in claimed])
    layer = np.arange(N_LAYERS * N_HEADS) // N_HEADS
    c1, k1 = np.load(FP_C1).reshape(-1), k1_score()

    def stats(idx):
        v = per[idx].mean(0)
        return (partial_spearman(v, od, sd), float(np.corrcoef(
            np.argsort(np.argsort(v)), np.argsort(np.argsort(od)))[0, 1]),
            partial_spearman(v, sd, od))

    rows, draws_out = [], []
    for s in TARGETS + REFERENCE:
        obs = stats(sets[s])
        print(f"{s:14s} ({len(sets[s]):2d} heads, layers {layer[sets[s]].min()}-{layer[sets[s]].max()})"
              f"  obj|slot {obs[0]:+.3f} (paper {PAPER.get(s, float('nan')):+.3f})"
              f"  obj raw {obs[1]:+.3f}  slot|obj {obs[2]:+.3f}")
        if s not in TARGETS:
            rows.append(dict(head_set=s, null="observed_only", n_heads=len(sets[s]),
                             obs_obj_given_slot=obs[0], obs_obj_raw=obs[1], obs_slot_given_obj=obs[2]))
            continue
        tl = layer[sets[s]]
        band = np.isin(never, np.unique(tl))
        c1_med, k1_med = np.median(c1[never[band]]), np.median(k1[never[band]])
        pools = {
            "layer_matched": {l: never[layer[never] == l] for l in np.unique(tl)},
            "inert_matched": {l: never[(layer[never] == l) & (c1[never] <= c1_med) & (k1[never] <= k1_med)]
                              for l in np.unique(tl)},
        }
        counts = pd.Series(tl).value_counts()
        inert_mask = (c1[never] <= c1_med) & (k1[never] <= k1_med)
        for l in counts.index:  # widen a short inert pool to layers l-1..l+1
            if len(pools["inert_matched"][l]) < counts[l]:
                pools["inert_matched"][l] = never[inert_mask & (np.abs(layer[never] - l) <= 1)]
                print(f"   inert pool for L{l} widened to L{l - 1}-L{l + 1}")
        print(f"   inert pool sizes per layer: { {int(l): len(p) for l, p in pools['inert_matched'].items()} }")
        for null in ("unmatched", "layer_matched", "inert_matched"):
            if null != "unmatched":
                short = {int(l): len(pools[null][l]) for l in counts.index if len(pools[null][l]) < counts[l]}
                assert not short, f"{s}/{null}: pool smaller than target count in layers {short}"
            R = []
            for _ in range(args.n_draws):
                if null == "unmatched":
                    idx = rng.choice(never, len(tl), replace=False)
                else:  # sequential draw so widened (overlapping) pools never repeat a head
                    used: set[int] = set()
                    for l in counts.index:
                        avail = np.array([h for h in pools[null][l] if h not in used])
                        used.update(rng.choice(avail, counts[l], replace=False).tolist())
                    idx = np.array(sorted(used))
                    assert len(idx) == len(tl)
                R.append(stats(idx))
            R = np.array(R)
            draws_out.append(pd.DataFrame(R, columns=["obj_given_slot", "obj_raw", "slot_given_obj"])
                             .assign(head_set=s, null=null))
            frac = float(np.mean(R[:, 0] >= obs[0]))
            rows.append(dict(head_set=s, null=null, n_heads=len(tl), n_draws=args.n_draws,
                             obs_obj_given_slot=obs[0], obs_obj_raw=obs[1], obs_slot_given_obj=obs[2],
                             null_median=np.median(R[:, 0]), null_p95=np.percentile(R[:, 0], 95),
                             null_max=R[:, 0].max(), frac_null_ge_obs=frac,
                             p_one_sided=(np.sum(R[:, 0] >= obs[0]) + 1) / (args.n_draws + 1),
                             null_obj_raw_median=np.median(R[:, 1]),
                             null_slot_given_obj_median=np.median(R[:, 2])))
            print(f"   {null:14s} obj|slot null median {np.median(R[:, 0]):+.3f}  95th {np.percentile(R[:, 0], 95):+.3f}"
                  f"  max {R[:, 0].max():+.3f}  frac>=obs {frac:.3f}"
                  f"  | raw obj median {np.median(R[:, 1]):+.3f}  slot|obj median {np.median(R[:, 2]):+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "random_null_partials.csv", index=False)
    pd.concat(draws_out).to_csv(OUT_DIR / "random_null_draws.csv", index=False)
    print(f"-> {OUT_DIR / 'random_null_partials.csv'}")


if __name__ == "__main__":
    main()
