"""Paired sign-flip permutation test for FB-vs-TB (and FP-vs-TP) ablation gaps.

For each scenario i (matched FB/TB or FP/TP pair) and each ablation step k,
compute the flip score
    phi_v,i(k) = p_alt_v[i,k] / (p_alt_v[i,k] + p_correct_v[i,k] + eps)
where v in {variant_A, variant_B}. The paired difference is
    Delta_i(k) = phi_A,i(k) - phi_B,i(k).

H0 (label exchangeability): for each scenario i, the labels A and B are
interchangeable, so the sign of Delta_i is symmetric around 0.
Sign-flip permutation: independently flip the sign of each Delta_i with
prob 0.5; recompute the mean; repeat N times to build a null distribution.

Three test statistics per pair:
    1. per_k         : D_obs(k) = mean_i Delta_i(k), one p-value per k.
    2. integrated    : mean_i mean_{k in [1, n_k]} Delta_i(k). Excludes k=0.
    3. baseline_ctrl : mean_i mean_{k in [1, n_k]} (Delta_i(k) - Delta_i(0)).
                       Isolates "does ablation widen the gap" from
                       "does baseline already have a gap".

Reads:
    --per_prompt_pt path/to/per_prompt_data.pt
Writes:
    {out_dir}/permutation_per_k.csv     (long: pair, k, D_obs, p_two_sided, ...)
    {out_dir}/permutation_summary.csv   (one row per pair, integrated stats)
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EPS = 1e-12
DEFAULT_PAIRS = [("false_belief", "true_belief"),
                 ("false_photo", "true_photo")]


def load_per_prompt(pt_path: Path) -> dict:
    """Load per_prompt_data.pt and return the 'target' sub-dict.

    Schema: {variant -> {p_correct, p_alt, acc, flip} each (n_prompts, n_k+1)}.
    """
    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    if "target" not in blob:
        raise KeyError(f"{pt_path} missing 'target' key; got {list(blob)}")
    target = blob["target"]
    # Older runs saved torch tensors; newer ones save numpy arrays.
    # Normalize to numpy so .size, broadcasting, etc. behave consistently.
    for variant, fields in target.items():
        for key, arr in fields.items():
            if isinstance(arr, torch.Tensor):
                fields[key] = arr.detach().cpu().numpy()
            else:
                fields[key] = np.asarray(arr)
    return target


def flip_score(p_correct: np.ndarray, p_alt: np.ndarray) -> np.ndarray:
    return p_alt / (p_alt + p_correct + EPS)


def sign_flip_perm(deltas: np.ndarray, n_perm: int, rng: np.random.Generator
                   ) -> np.ndarray:
    """Sign-flip permutation on a 1-D array of paired diffs.

    Returns the array of N permuted means.
    """
    n = deltas.size
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    return (signs * deltas).mean(axis=1)


def two_sided_p(obs: float, null_means: np.ndarray) -> float:
    n = null_means.size
    # +1 in numerator and denominator: standard correction so p>0 even when
    # no permutation exceeds |obs|.
    return (1 + np.sum(np.abs(null_means) >= np.abs(obs))) / (1 + n)


def test_pair(target: dict, var_a: str, var_b: str, n_perm: int,
              rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Run all three tests for one variant pair."""
    if var_a not in target or var_b not in target:
        missing = [v for v in (var_a, var_b) if v not in target]
        raise KeyError(f"variants {missing} not in per_prompt_data; have "
                       f"{list(target)}")

    phi_a = flip_score(target[var_a]["p_correct"], target[var_a]["p_alt"])
    phi_b = flip_score(target[var_b]["p_correct"], target[var_b]["p_alt"])
    if phi_a.shape != phi_b.shape:
        raise ValueError(f"shape mismatch {var_a}={phi_a.shape} vs "
                         f"{var_b}={phi_b.shape}")

    n_prompts, n_steps = phi_a.shape
    deltas = phi_a - phi_b  # (n_prompts, n_steps)

    # ---- per-k tests ----
    rows = []
    for k in range(n_steps):
        d_k = deltas[:, k]
        d_obs = float(d_k.mean())
        null = sign_flip_perm(d_k, n_perm, rng)
        p = two_sided_p(d_obs, null)
        rows.append({
            "pair": f"{var_a}_vs_{var_b}",
            "k": k,
            "n_scenarios": n_prompts,
            "D_obs": d_obs,
            "null_mean": float(null.mean()),
            "null_std": float(null.std(ddof=1)),
            "null_ci_lo": float(np.percentile(null, 2.5)),
            "null_ci_hi": float(np.percentile(null, 97.5)),
            "p_two_sided": p,
            "phi_A_mean": float(phi_a[:, k].mean()),
            "phi_B_mean": float(phi_b[:, k].mean()),
        })
    per_k_df = pd.DataFrame(rows)

    # ---- integrated tests (k>=1 only; k=0 is baseline) ----
    # statistic 2: mean Delta over k in [1, n_k]
    delta_int_per_scenario = deltas[:, 1:].mean(axis=1)
    d_int_obs = float(delta_int_per_scenario.mean())
    null_int = sign_flip_perm(delta_int_per_scenario, n_perm, rng)
    p_int = two_sided_p(d_int_obs, null_int)

    # statistic 3: baseline-controlled
    delta_ctrl_per_scenario = (deltas[:, 1:] - deltas[:, [0]]).mean(axis=1)
    d_ctrl_obs = float(delta_ctrl_per_scenario.mean())
    null_ctrl = sign_flip_perm(delta_ctrl_per_scenario, n_perm, rng)
    p_ctrl = two_sided_p(d_ctrl_obs, null_ctrl)

    summary = {
        "pair": f"{var_a}_vs_{var_b}",
        "n_scenarios": n_prompts,
        "n_k_steps": n_steps,
        "n_perm": n_perm,
        "D_obs_k0_baseline": float(deltas[:, 0].mean()),
        "D_obs_integrated": d_int_obs,
        "p_integrated_two_sided": p_int,
        "null_integrated_ci_lo": float(np.percentile(null_int, 2.5)),
        "null_integrated_ci_hi": float(np.percentile(null_int, 97.5)),
        "D_obs_baseline_controlled": d_ctrl_obs,
        "p_baseline_controlled_two_sided": p_ctrl,
        "null_ctrl_ci_lo": float(np.percentile(null_ctrl, 2.5)),
        "null_ctrl_ci_hi": float(np.percentile(null_ctrl, 97.5)),
    }
    return per_k_df, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_prompt_pt", type=Path, required=True,
                    help="Path to per_prompt_data.pt from an ablation run.")
    ap.add_argument("--out_dir", type=Path, required=True,
                    help="Directory to write CSVs into (created if missing).")
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", nargs="+", default=None,
                    help='Pairs as "A,B" strings. Default: '
                         'false_belief,true_belief false_photo,true_photo.')
    ap.add_argument("--run_label", default=None,
                    help="Tag added to every row, e.g. "
                         "'intersection_matched_prefill'. Defaults to parent "
                         "directory of --per_prompt_pt.")
    args = ap.parse_args()

    pairs = (DEFAULT_PAIRS if args.pairs is None
             else [tuple(p.split(",")) for p in args.pairs])
    run_label = args.run_label or args.per_prompt_pt.parent.name

    print(f"loading {args.per_prompt_pt}")
    target = load_per_prompt(args.per_prompt_pt)
    print(f"variants in target: {list(target)}")
    sample_var = next(iter(target))
    print(f"shape of {sample_var}.p_correct: "
          f"{target[sample_var]['p_correct'].shape}")

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_per_k = []
    all_summary = []
    for var_a, var_b in pairs:
        print(f"\n=== {var_a} vs {var_b} ===")
        per_k_df, summary = test_pair(target, var_a, var_b, args.n_perm, rng)
        per_k_df.insert(0, "run_label", run_label)
        summary["run_label"] = run_label
        all_per_k.append(per_k_df)
        all_summary.append(summary)

        # quick stdout report
        print(f"  k=0 baseline gap: D_obs = {summary['D_obs_k0_baseline']:+.4f}")
        print(f"  integrated (k>=1): D_obs = {summary['D_obs_integrated']:+.4f}, "
              f"p = {summary['p_integrated_two_sided']:.4f}")
        print(f"  baseline-ctrl:    D_obs = {summary['D_obs_baseline_controlled']:+.4f}, "
              f"p = {summary['p_baseline_controlled_two_sided']:.4f}")
        # show a few per-k rows
        sig_rows = per_k_df[per_k_df["p_two_sided"] < 0.05]
        print(f"  per-k: {len(sig_rows)}/{len(per_k_df)} k-steps with p<0.05")

    out_per_k = args.out_dir / "permutation_per_k.csv"
    out_summary = args.out_dir / "permutation_summary.csv"
    pd.concat(all_per_k, ignore_index=True).to_csv(out_per_k, index=False)
    pd.DataFrame(all_summary).to_csv(out_summary, index=False)
    print(f"\nwrote {out_per_k}")
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
