#!/usr/bin/env python
"""Init-vs-mutation targeting test on the mid-pad grid.

The Exp 3 pad-twins grid moves both candidate lookback targets together (the
prefix pad shifts the line-1 `a = [...]` init literal AND the mutation
literal), so it can only show that something positional moves. The mid-pad
grid (build_code_padtwins_stimuli.py --pad_axis mid) puts the second pad
BETWEEN the definitions and the mutation line, so it moves the mutation
literal and leaves the init literal fixed.

Which query reads which literal:
    (copy,  mutated  )  answer = orig_m   -> INIT literal
    (copy,  untouched)  answer = orig_u   -> INIT literal
    (alias, untouched)  answer = orig_u   -> INIT literal
    (alias, mutated  )  answer = mut_char -> MUTATION line   <- the odd one out

So the test is a 1-vs-3 interaction, not a main effect:

  target-specific pointer   the MID-pad response is elevated for
                            (alias, mutated) relative to the other three
                            cells, while the PREFIX-pad response is similar
                            across all four (the prefix moves every target).
  single shared carrier     mid response equal across all four cells.
  pure veracity tag         no pad response anywhere (already refuted).

Statistic: within each twin group, partial Spearman of pairwise neural
distance on |d prefix| controlling |d mid|, and on |d mid| controlling
|d prefix| — the same partials the Exp 3 pad-twin test uses, but computed
separately per query cell. Permutation null: pad labels shuffled within twin
group (exchangeable under any no-pad-effect hypothesis), which gives a null
for the interaction contrast directly rather than for each cell separately.

NOTE the mid-pad length is stored in `suffix_pad_len` (see the builder
docstring) so the shared pad-twin machinery runs unmodified; `mid_pad_len`
and `pad_axis` carry the truth and are asserted here.

Output: results/rsa_head_outputs/aggregation_code_midpad/
  midpad_targeting{_gemma}.csv        per head set x query cell
  midpad_interaction{_gemma}.csv      the 1-vs-3 contrast + permutation p
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
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import (  # noqa: E402
    HEADS_CSV, GEMMA_HEADSET_DIR, HEAD_SETS, GEMMA_HEAD_SETS,
    partial_spearman)

ACT_DIR = ROOT / "results/rsa_head_outputs/activations_code_midpad_full"
ACT_PATHS = {"qwen": ACT_DIR / "Qwen2.5-14B-Instruct/code_midpad.pt",
             "gemma": ACT_DIR / "gemma-2-27b-it/code_midpad.pt"}
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_code_midpad"
UT9 = np.triu_indices(9, 1)
# the one cell whose answer lives in the mutation line
MUT_CELL = ("alias", "mutated")


def head_sets_for(model):
    if model == "gemma":
        return {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                for s in GEMMA_HEAD_SETS}
    heads = pd.read_csv(HEADS_CSV)
    return {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}


def set_neural_rdm(z, hdf):
    return np.mean([cosine_rdm(z[:, int(r.layer), int(r.head), :])
                    for r in hdf.itertuples()], axis=0)


def group_cache(neural, stim):
    """Per twin group: (distances, prefix pads, mid pads, query cell)."""
    groups = {}
    for i, r in enumerate(stim):
        groups.setdefault(r["twin_id"], []).append(i)
    out = []
    for idx in groups.values():
        assert len(idx) == 9, f"expected 9 stimuli per twin, got {len(idx)}"
        D = neural[np.ix_(idx, idx)][UT9]
        pre = np.array([stim[i]["pad_len"] for i in idx], float)
        mid = np.array([stim[i]["suffix_pad_len"] for i in idx], float)
        cell = (stim[idx[0]]["queried_object"], stim[idx[0]]["queried_slot_type"])
        out.append((D, pre, mid, cell))
    return out


def cell_stats(cache, perms=None):
    """Mean partial rhos per query cell: {cell: (rho_prefix|mid, rho_mid|prefix)}."""
    acc, n = {}, {}
    for g, (D, pre, mid, cell) in enumerate(cache):
        if perms is not None:
            p = perms[g]
            pre, mid = pre[p], mid[p]
        dpre = np.abs(pre[UT9[0]] - pre[UT9[1]])
        dmid = np.abs(mid[UT9[0]] - mid[UT9[1]])
        v = np.array([partial_spearman(D, dpre, dmid),
                      partial_spearman(D, dmid, dpre)])
        acc[cell] = acc.get(cell, 0.0) + v
        n[cell] = n.get(cell, 0) + 1
    return {c: acc[c] / n[c] for c in acc}


def interaction(stats):
    """(mid response for the mutation-reading cell) minus (mean mid response
    of the three init-reading cells); same contrast for the prefix pad."""
    others = [c for c in stats if c != MUT_CELL]
    d_mid = stats[MUT_CELL][1] - np.mean([stats[c][1] for c in others])
    d_pre = stats[MUT_CELL][0] - np.mean([stats[c][0] for c in others])
    return float(d_mid), float(d_pre)


def run(model, n_perm, seed):
    path = ACT_PATHS[model]
    if not path.exists():
        print(f"[skip] {path} missing")
        return
    d = torch.load(path, weights_only=False)
    z = d["z"].float().numpy()
    stim = d["stim"]
    assert all(r.get("pad_axis") == "mid" for r in stim), \
        "this capture is not the mid-pad grid"
    if "correct" in d:
        acc = pd.DataFrame({"c": d["correct"],
                            "obj": [r["queried_object"] for r in stim],
                            "slot": [r["queried_slot_type"] for r in stim]})
        print(f"[{model}] baseline accuracy {np.mean(d['correct']):.3f}  "
              f"by cell: "
              f"{acc.groupby(['obj', 'slot']).c.mean().round(3).to_dict()}")

    rng = np.random.default_rng(seed)
    rows, inter_rows = [], []
    for set_name, hdf in head_sets_for(model).items():
        cache = group_cache(set_neural_rdm(z, hdf), stim)
        stats = cell_stats(cache)
        for cell, v in sorted(stats.items()):
            rows.append({"model": model, "head_set": set_name,
                         "queried_object": cell[0], "queried_slot_type": cell[1],
                         "reads": "mutation" if cell == MUT_CELL else "init",
                         "rho_prefix|mid": v[0], "rho_mid|prefix": v[1]})
        obs_mid, obs_pre = interaction(stats)
        null = np.empty((n_perm, 2))
        for k in range(n_perm):
            perms = [rng.permutation(9) for _ in cache]
            null[k] = interaction(cell_stats(cache, perms))
        p_mid = float((np.sum(np.abs(null[:, 0]) >= abs(obs_mid)) + 1)
                      / (n_perm + 1))
        inter_rows.append({"model": model, "head_set": set_name,
                           "n_heads": len(hdf),
                           "interaction_mid": obs_mid, "p_perm": p_mid,
                           "interaction_prefix": obs_pre,
                           "null_sd_mid": float(null[:, 0].std())})
        print(f"[{model}/{set_name:14s}] mid-pad interaction "
              f"(alias-mutated minus the 3 init-reading cells) = "
              f"{obs_mid:+.4f}  p = {p_mid:.4f}   "
              f"[prefix-pad interaction {obs_pre:+.4f}, "
              f"null sd {null[:, 0].std():.4f}]")
        for cell, v in sorted(stats.items()):
            tag = "MUT" if cell == MUT_CELL else "init"
            print(f"      {cell[0]:6s}/{cell[1]:10s} [{tag}]  "
                  f"prefix|mid {v[0]:+.3f}   mid|prefix {v[1]:+.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_gemma" if model == "gemma" else ""
    pd.DataFrame(rows).to_csv(OUT_DIR / f"midpad_targeting{suffix}.csv",
                              index=False)
    pd.DataFrame(inter_rows).to_csv(OUT_DIR / f"midpad_interaction{suffix}.csv",
                                    index=False)
    print(f"-> {OUT_DIR}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen", "gemma"])
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for m in args.models:
        run(m, args.n_perm, args.seed)


if __name__ == "__main__":
    main()
