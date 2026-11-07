#!/usr/bin/env python3
"""Held-out test: do the copy/alias ("veracity") heads and the index heads
mediate different contrasts? (Sept 10 2026 meeting, item 2-3.)

Design
------
Head sets are FROZEN before evaluation:
  tom_c1         ToM T_C1 set (primary; selected on ToM prompts only)
  code_abstract  code_abstract seed-0 significant heads (secondary)
  code_index     code_index seed-0 positive significant heads, union over
                 copy->copy (ABA) and alias->alias (ABB) — same rule as
                 build_code_head_csvs.py uses for the other code sets
All effects are measured on SEED-1 runs only (independent prompts), so no
cell of the table is estimated on the prompts that selected its heads.

Metric
------
Per-head CMA score = change in the (causal - original answer) logit
difference at the final token when that one head's z is patched from base
into exp; logits, unbounded, 0 = no effect, >0 = pushed toward the causal
answer. Per head set we report the mean score over the set's heads, averaged
over prompts, with a percentile-bootstrap 95% CI over prompts. Scores are
NOT normalized by a full-residual patch: at late layers that patch carries
the donor's already-computed answer (= exp's answer), so it is not an upper
bound on the index effect.

Tests
-----
1. Per cell, H0: set-mean effect = 0 (CI covers 0).
2. Smallness bound for the veracity sets: ratio = (effect on index contrast)
   / (effect on copy/alias contrast), bootstrapped with independent prompt
   resampling in numerator and denominator. "Small" = upper 95% limit below
   --bound (choose BEFORE looking; default 0.25 is a placeholder, not an
   established threshold).
3. Double dissociation as an interaction contrast,
   (V on abstract - V on index) - (I on abstract - I on index),
   V = veracity set, I = index set. H0: interaction = 0.
4. Whole-map: Pearson r over all 1920 heads between held-out mean
   fingerprints (descriptive; heads are not independent, so no p-value).
5. Mutate-b decomposition (H2-style): over 1920 heads,
   r(abstract, veracity_only + site_only) vs r with each part alone.
   H0: the unweighted sum correlates no better than the better single part;
   bootstrap CI (prompts resampled independently per contrast) on
   r(sum) - max(r(part)).

Not available from current CMA outputs: per-head frequency of flips to the
causal answer (cma.py saves logit differences only).

Usage:
    python analysis_scripts/analyze_code_index_dissociation.py
    python analysis_scripts/analyze_code_index_dissociation.py --bound 0.25 --n_boot 10000
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_code_head_csvs import resolve_significant_heads, union_positive_heads
from condition_registry import pick_sample_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "cma_scripts" / "tasks" / "causal_analysis"))
from cma_config import CMACondition  # noqa: E402

CMA_ROOT = PROJECT_ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
HEAD_SET_DIR = PROJECT_ROOT / "results/aggregated/code_head_sets"
OUT_DIR = PROJECT_ROOT / "results/aggregated/code_index_dissociation"
EVAL_SEED = 1

# contrast label -> (condition folder, base_rule, exp_rule)
CONTRASTS = {
    "abstract copy->alias":         (CMACondition.CODE_ABSTRACT.value, "ABA", "ABB"),
    "abstract alias->copy":         (CMACondition.CODE_ABSTRACT.value, "ABB", "ABA"),
    "abstract_two_mut copy->alias": (CMACondition.CODE_ABSTRACT_TWO_MUT.value, "ABA", "ABB"),
    "abstract_two_mut alias->copy": (CMACondition.CODE_ABSTRACT_TWO_MUT.value, "ABB", "ABA"),
    "index copy->copy":             (CMACondition.CODE_INDEX.value, "ABA", "ABA"),
    "index alias->alias":           (CMACondition.CODE_INDEX.value, "ABB", "ABB"),
    "index_flip copy->copy":        (CMACondition.CODE_INDEX_FLIP.value, "ABA", "ABA"),
    "index_flip alias->alias":      (CMACondition.CODE_INDEX_FLIP.value, "ABB", "ABB"),
    # mutate-b pairs (copy base only)
    "veracity_only copy->alias":    (CMACondition.CODE_VERACITY_ONLY.value, "ABA", "ABB"),
    "site_only list->mut_line":     (CMACondition.CODE_SITE_ONLY.value, "ABA", "ABA"),
    "site_only mut_line->list":     (CMACondition.CODE_SITE_ONLY_REV.value, "ABA", "ABA"),
    # one-element position pilot (n=40): fillers present in both
    "pos_unmatched copy->alias":    (CMACondition.CODE_POS_UNMATCHED.value, "ABA", "ABB"),
    "pos_unmatched alias->copy":    (CMACondition.CODE_POS_UNMATCHED.value, "ABB", "ABA"),
    "pos_matched copy->alias":      (CMACondition.CODE_POS_MATCHED.value, "ABA", "ABB"),
    "pos_matched alias->copy":      (CMACondition.CODE_POS_MATCHED.value, "ABB", "ABA"),
}
# pilot read-out: (matched cell, unmatched baseline) per direction
POS_PILOT = [("pos_matched copy->alias", "pos_unmatched copy->alias"),
             ("pos_matched alias->copy", "pos_unmatched alias->copy")]
# individual heads reported alongside the sets (fixed before results):
# L29H31 = top abstract copy->alias head, ~0 in site_only forward (joint-only);
# L35H30 = top head of every site_only / abstract cell (source selection)
PILOT_HEADS = {"L29H31": (29, 31), "L35H30": (35, 30)}
# contrasts pooled for the ratio / interaction tests (format-matched pair)
VERACITY_CONTRASTS = ["abstract_two_mut copy->alias", "abstract_two_mut alias->copy"]
INDEX_CONTRASTS = ["index copy->copy", "index alias->alias"]
# code_abstract (copy->alias) factors through the (copy, mutate b) cell:
# abstract base --site_only--> (copy, mutate b) --veracity_only--> abstract exp
DECOMP_WHOLE = "abstract copy->alias"
DECOMP_PARTS = ["veracity_only copy->alias", "site_only list->mut_line"]


def load_per_sample(condition: str, base_rule: str, exp_rule: str):
    """[n_prompts, 48, 40] per-prompt scores from the held-out seed, or None."""
    logit_dir = (CMA_ROOT / condition / "template_indexed_list"
                 / f"base_rule_{base_rule}_exp_rule_{exp_rule}"
                 / "patch_before_movement" / f"z_seed_{EVAL_SEED}_shuffle_False" / "logit")
    picked = pick_sample_dir(logit_dir, require_per_sample=True)
    if picked is None:
        return None
    sample_dir, n, _ = picked
    scores = torch.load(sample_dir / "causal_scores_per_sample.pt",
                        map_location="cpu", weights_only=True).numpy().astype(np.float64)
    assert scores.shape == (n, 48, 40), (sample_dir, scores.shape)
    return scores


def load_head_sets() -> dict:
    tom = pd.read_csv(HEAD_SET_DIR / "tom_c1_c5_heads.csv")
    tom_c1 = tom[tom["condition"] == "tom_c1"]
    code_abs = pd.read_csv(HEAD_SET_DIR / "code_abstract_head_list.csv")
    index = union_positive_heads([
        resolve_significant_heads(CMACondition.CODE_INDEX.value, "ABA", "ABA"),
        resolve_significant_heads(CMACondition.CODE_INDEX.value, "ABB", "ABB"),
    ])
    index.to_csv(HEAD_SET_DIR / "code_index_head_list.csv", index=False)
    sets = {"tom_c1": tom_c1, "code_abstract": code_abs, "code_index": index}
    out = {k: list(zip(df["layer"].astype(int), df["head"].astype(int)))
           for k, df in sets.items()}
    # T_C1 split by index-set membership (memberships fixed by the seed-0
    # discovery runs, i.e. before any pilot result is examined).
    shared = [h for h in out["tom_c1"] if h in set(out["code_index"])]
    out["tom_c1_index_shared"] = shared
    out["tom_c1_remaining"] = [h for h in out["tom_c1"] if h not in set(shared)]
    for name, h in PILOT_HEADS.items():
        out[name] = [h]
    return out


def set_effect_per_prompt(scores: np.ndarray, heads) -> np.ndarray:
    """Mean score over the set's heads, per prompt -> [n_prompts]."""
    layers, hs = zip(*heads)
    return scores[:, list(layers), list(hs)].mean(axis=1)


def ci(b: np.ndarray):
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bound", type=float, default=0.25,
                        help="pre-specified upper limit on index/veracity effect ratio")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    head_sets = load_head_sets()
    base_sets = ["tom_c1", "code_abstract", "code_index"]
    overlap = {f"{a}&{b}": len(set(head_sets[a]) & set(head_sets[b]))
               for a in base_sets for b in base_sets if a < b}
    print("head set sizes:", {k: len(v) for k, v in head_sets.items()}, "| overlaps:", overlap)

    scores = {label: load_per_sample(*cell) for label, cell in CONTRASTS.items()}
    missing = [k for k, v in scores.items() if v is None]
    if missing:
        print(f"[missing held-out runs, skipped] {missing}")
    scores = {k: v for k, v in scores.items() if v is not None}

    # --- table: head set x contrast, mean effect + bootstrap CI over prompts
    boots, rows = {}, []
    for label, s in scores.items():
        # one set of resampled prompts per contrast, shared by all head sets
        # (sets are measured on the same prompts -> paired resampling)
        idx = rng.integers(0, len(s), size=(args.n_boot, len(s)))
        for set_name, heads in head_sets.items():
            x = set_effect_per_prompt(s, heads)
            print(f"  [{label} | {set_name}] n_prompts={len(x)} per-prompt mean/min/max = "
                  f"{x.mean():.4f}/{x.min():.4f}/{x.max():.4f}")
            b = x[idx].mean(axis=1)
            boots[(set_name, label)] = b
            lo, hi = ci(b)
            rows.append({"contrast": label, "head_set": set_name, "n_heads": len(heads),
                         "n_prompts": len(x), "mean_logit_diff_change": x.mean(),
                         "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "set_by_contrast_effects.csv", index=False)
    print("\nMean per-head change in (causal - original) logit difference, 95% bootstrap CI over prompts")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- ratio bound + interaction (pooled over the two directions/kinds)
    have = all(c in scores for c in VERACITY_CONTRASTS + INDEX_CONTRASTS)
    if have:
        def pooled(set_name, contrasts):
            return np.mean([boots[(set_name, c)] for c in contrasts], axis=0)

        tests = []
        for v_set in ("tom_c1", "code_abstract"):
            v_abs, v_idx = pooled(v_set, VERACITY_CONTRASTS), pooled(v_set, INDEX_CONTRASTS)
            i_abs, i_idx = pooled("code_index", VERACITY_CONTRASTS), pooled("code_index", INDEX_CONTRASTS)
            ratio = v_idx / v_abs
            inter = (v_abs - v_idx) - (i_abs - i_idx)
            r_lo, r_hi = ci(ratio)
            d_lo, d_hi = ci(v_abs)
            n_lo, n_hi = ci(inter)
            tests.append({"veracity_set": v_set,
                          "denominator_ci_lo": d_lo, "denominator_ci_hi": d_hi,
                          "ratio_median": np.median(ratio), "ratio_ci_lo": r_lo, "ratio_ci_hi": r_hi,
                          f"ratio_upper_below_{args.bound}": bool(r_hi < args.bound and d_lo > 0),
                          "interaction_median": np.median(inter),
                          "interaction_ci_lo": n_lo, "interaction_ci_hi": n_hi})
        tests = pd.DataFrame(tests)
        tests.to_csv(OUT_DIR / "ratio_and_interaction.csv", index=False)
        print("\nratio = veracity set's index effect / its copy-alias effect (two-mutation format); "
              "interaction = (V on abstract - V on index) - (I on abstract - I on index)")
        print(tests.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- position pilot: matched / unmatched ratio per set, independent bootstrap
    pilot_rows = []
    for m_lab, u_lab in POS_PILOT:
        if m_lab not in scores or u_lab not in scores:
            continue
        for set_name in head_sets:
            bm, bu = boots[(set_name, m_lab)], boots[(set_name, u_lab)]
            ratio = bm / bu
            u_lo, u_hi = ci(bu)
            r_lo, r_hi = ci(ratio)
            pilot_rows.append({"direction": m_lab.split(" ")[1], "head_set": set_name,
                               "unmatched_mean": bu.mean(), "unmatched_ci_lo": u_lo, "unmatched_ci_hi": u_hi,
                               "matched_mean": bm.mean(),
                               "ratio_median": np.median(ratio), "ratio_ci_lo": r_lo, "ratio_ci_hi": r_hi,
                               "baseline_ok": bool(u_lo > 0)})
        r_map = np.corrcoef(scores[m_lab].mean(axis=0).ravel(), scores[u_lab].mean(axis=0).ravel())[0, 1]
        ref = "abstract_two_mut " + m_lab.split(" ")[1]
        r_ref = (np.corrcoef(scores[u_lab].mean(axis=0).ravel(), scores[ref].mean(axis=0).ravel())[0, 1]
                 if ref in scores else np.nan)
        print(f"\n[pilot {m_lab.split(' ')[1]}] fingerprint r(matched, unmatched) = {r_map:.3f}; "
              f"r(unmatched, {ref}) = {r_ref:.3f}  (check 1 wants >~0.5)")
    if pilot_rows:
        pilot = pd.DataFrame(pilot_rows)
        pilot.to_csv(OUT_DIR / "pos_pilot_matched_vs_unmatched.csv", index=False)
        print("\nPosition pilot: matched / unmatched set effects (ratio CI from independent prompt "
              "resampling; baseline_ok = unmatched CI excludes 0)")
        print(pilot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # --- mutate-b decomposition of the abstract fingerprint
    if all(c in scores for c in [DECOMP_WHOLE] + DECOMP_PARTS):
        def decomp(whole, a, b):
            r = lambda u, v: np.corrcoef(u, v)[0, 1]
            return r(whole, a + b), r(whole, a), r(whole, b)

        def fp(label, resample=False):
            s = scores[label]
            if resample:
                s = s[rng.integers(0, len(s), size=len(s))]
            return s.mean(axis=0).ravel()

        r_sum, r_a, r_b = decomp(fp(DECOMP_WHOLE), *[fp(c) for c in DECOMP_PARTS])
        gains = []
        for _ in range(min(args.n_boot, 2000)):
            b_sum, b_a, b_b = decomp(fp(DECOMP_WHOLE, True), *[fp(c, True) for c in DECOMP_PARTS])
            gains.append(b_sum - max(b_a, b_b))
        g_lo, g_hi = ci(np.array(gains))
        print(f"\nDecomposition over 1920 heads (held-out): r({DECOMP_WHOLE}, sum of parts) = {r_sum:.3f}; "
              f"r with {DECOMP_PARTS[0]} alone = {r_a:.3f}; with {DECOMP_PARTS[1]} alone = {r_b:.3f}; "
              f"r(sum) - max(r(part)) 95% CI = [{g_lo:.3f}, {g_hi:.3f}]")
        pd.DataFrame([{"r_sum": r_sum, f"r_{DECOMP_PARTS[0]}": r_a, f"r_{DECOMP_PARTS[1]}": r_b,
                       "gain_ci_lo": g_lo, "gain_ci_hi": g_hi}]).to_csv(
            OUT_DIR / "mutate_b_decomposition.csv", index=False)

    # --- whole-map fingerprint correlations on held-out means
    labels = list(scores)
    fps = np.stack([scores[k].mean(axis=0).ravel() for k in labels])
    r = pd.DataFrame(np.corrcoef(fps), index=labels, columns=labels)
    r.to_csv(OUT_DIR / "fingerprint_pearson_r.csv")
    print("\nPearson r over 1920 heads between held-out mean fingerprints (descriptive)")
    print(r.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
