"""
re-run permutation tests with FWER (max-statistic) on all existing results.

loads causal_scores_per_sample.pt files and overwrites old FDR-based
statistical_test_results.csv / significant_heads.csv / statistical_summary.txt.

usage:
    python scripts/rerun_statistical_tests.py
    python scripts/rerun_statistical_tests.py --dry_run
    python scripts/rerun_statistical_tests.py --n_permutations 10000
"""

import sys
import os
import argparse
from pathlib import Path

# add cma module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codebase', 'tasks', 'causal_analysis'))

import torch
from statistical_testing import permutation_test_heads, save_statistical_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/causal_analysis")
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true", help="list files without running tests")
    args = parser.parse_args()

    # find all per_sample score files
    pt_files = sorted(Path(args.results_dir).rglob("causal_scores_per_sample.pt"))

    print(f"found {len(pt_files)} per-sample score files")
    if not pt_files:
        return

    for i, pt_file in enumerate(pt_files):
        folder = pt_file.parent
        # extract readable condition from path
        rel = str(pt_file.relative_to(args.results_dir))
        print(f"\n[{i+1}/{len(pt_files)}] {rel}")

        if args.dry_run:
            scores = torch.load(pt_file, map_location='cpu')
            print(f"  shape: {scores.shape}")
            continue

        # load and run
        scores = torch.load(pt_file, map_location='cpu')
        print(f"  shape: {scores.shape} ({scores.shape[0]} samples, {scores.shape[1]} layers, {scores.shape[2]} heads)")

        df, threshold = permutation_test_heads(
            scores,
            n_permutations=args.n_permutations,
            alpha=args.alpha,
        )

        n_sig = df['significant'].sum()
        print(f"  FWER threshold: {threshold:.6f}")
        print(f"  significant heads: {n_sig}/{len(df)} ({n_sig/len(df)*100:.1f}%)")

        save_statistical_results(df, str(folder), threshold=threshold)
        print(f"  saved to {folder}")

    print(f"\ndone. re-ran FWER tests on {len(pt_files)} files.")


if __name__ == "__main__":
    main()
