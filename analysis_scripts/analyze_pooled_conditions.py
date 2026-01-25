#!/usr/bin/env python3
"""
pooled statistical analysis across experimental conditions and templates.

this script performs post-hoc analysis after running multiple CMA experiments.
it pools per-sample causal scores across templates, runs statistical tests,
and performs differential testing to identify condition-specific effects.

usage:
    python scripts/analyze_pooled_conditions.py \
        --results_dir results/causal_analysis/cma \
        --conditions cross_belief within_belief control \
        --output_dir results/pooled_analysis

the script will:
1. discover all causal_scores_per_sample.pt files
2. group by condition type (based on context_type in path)
3. pool across templates/vignettes
4. run permutation tests on pooled data
5. perform differential testing (e.g., cross_belief vs control)
6. generate summary tables and visualizations
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# add codebase to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'codebase', 'tasks', 'causal_analysis'))
from statistical_testing import permutation_test_heads, save_statistical_results

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def discover_result_files(results_dir: str, pattern: str = "causal_scores_per_sample.pt") -> List[Path]:
    """
    recursively find all per-sample score files in results directory.

    args:
        results_dir: root directory to search
        pattern: filename pattern to match

    returns:
        list of Path objects to matching files
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise ValueError(f"results directory does not exist: {results_dir}")

    files = list(results_path.rglob(pattern))
    logger.info(f"discovered {len(files)} result files matching '{pattern}'")

    return files


def group_files_by_condition(files: List[Path]) -> Dict[str, List[Path]]:
    """
    group result files by experimental condition based on directory structure.

    conditions are identified by context_type in the path:
    - abstract_context → cross-belief condition
    - token_context → within-belief (location swap)
    - control_context → control

    args:
        files: list of paths to result files

    returns:
        dict mapping condition name to list of file paths
    """
    grouped = defaultdict(list)

    for file in files:
        path_str = str(file)

        # identify condition from path
        if "abstract_context" in path_str:
            condition = "cross_belief"
        elif "token_context" in path_str:
            condition = "within_belief"
        elif "control_context" in path_str:
            condition = "control"
        elif "basic_context" in path_str:
            condition = "basic"
        elif "photo" in path_str:
            condition = "photo"
        else:
            logger.warning(f"could not identify condition for: {file}")
            condition = "unknown"

        grouped[condition].append(file)

    for condition, file_list in grouped.items():
        logger.info(f"condition '{condition}': {len(file_list)} files")

    return grouped


def load_and_pool_scores(files: List[Path]) -> torch.Tensor:
    """
    load multiple per-sample score files and concatenate along sample dimension.

    args:
        files: list of paths to .pt files containing per-sample scores

    returns:
        tensor of shape (total_samples, n_layers, n_heads)
    """
    scores_list = []

    for file in files:
        try:
            scores = torch.load(file, map_location='cpu')
            logger.info(f"loaded {file.name}: shape {scores.shape}")
            scores_list.append(scores)
        except Exception as e:
            logger.error(f"failed to load {file}: {e}")
            continue

    if not scores_list:
        raise ValueError("no valid score files loaded")

    # check all have same layer/head dimensions
    first_shape = scores_list[0].shape
    for i, scores in enumerate(scores_list):
        if scores.shape[1:] != first_shape[1:]:
            logger.warning(f"shape mismatch: {scores.shape} vs {first_shape}")
            # could handle this by padding, but for now just warn

    # concatenate along sample dimension (dim=0)
    pooled = torch.cat(scores_list, dim=0)
    logger.info(f"pooled {len(scores_list)} files → total shape: {pooled.shape}")

    return pooled


def differential_test(
    condition_a: torch.Tensor,
    condition_b: torch.Tensor,
    n_permutations: int = 1000,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    test for differential effects between two conditions.

    H0: effect_a - effect_b = 0 (no difference between conditions)
    HA: effect_a - effect_b ≠ 0 (condition-specific effect)

    args:
        condition_a: per-sample scores for condition A (n_samples_a, n_layers, n_heads)
        condition_b: per-sample scores for condition B (n_samples_b, n_layers, n_heads)
        n_permutations: number of permutations
        alpha: significance level

    returns:
        dataframe with differential test results
    """
    logger.info("running differential test between conditions...")

    # compute observed difference in means
    mean_a = condition_a.mean(dim=0).cpu().numpy()
    mean_b = condition_b.mean(dim=0).cpu().numpy()
    observed_diff = mean_a - mean_b

    n_layers, n_heads = observed_diff.shape

    # permutation test: randomly assign samples to condition A or B
    n_a, n_b = condition_a.shape[0], condition_b.shape[0]
    combined = torch.cat([condition_a, condition_b], dim=0)  # (n_a + n_b, layers, heads)
    n_total = combined.shape[0]

    logger.info(f"combined {n_a} + {n_b} = {n_total} samples")
    logger.info(f"generating {n_permutations} permutations...")

    null_diffs = []
    for _ in range(n_permutations):
        # randomly permute sample assignments
        perm_idx = torch.randperm(n_total)
        perm_a = combined[perm_idx[:n_a]]
        perm_b = combined[perm_idx[n_a:]]

        null_diff = perm_a.mean(dim=0) - perm_b.mean(dim=0)
        null_diffs.append(null_diff.cpu().numpy())

    null_diffs = np.stack(null_diffs, axis=0)  # (n_perm, n_layers, n_heads)

    # compute two-tailed p-values
    p_values = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            obs = observed_diff[layer, head]
            null = null_diffs[:, layer, head]
            p_values[layer, head] = (np.abs(null) >= np.abs(obs)).mean()

    # FDR correction
    from statsmodels.stats.multitest import multipletests
    p_flat = p_values.flatten()
    reject, q_values, _, _ = multipletests(p_flat, alpha=alpha, method='fdr_bh')
    q_values = q_values.reshape(n_layers, n_heads)
    significant = reject.reshape(n_layers, n_heads)

    # build results dataframe
    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            results.append({
                'layer': layer,
                'head': head,
                'mean_condition_a': mean_a[layer, head],
                'mean_condition_b': mean_b[layer, head],
                'differential_effect': observed_diff[layer, head],
                'p_value': p_values[layer, head],
                'q_value': q_values[layer, head],
                'significant': significant[layer, head]
            })

    df = pd.DataFrame(results)
    logger.info(f"found {df['significant'].sum()} heads with significant differential effects")

    return df


def save_pooled_results(
    pooled_stats: Dict[str, pd.DataFrame],
    differential_stats: Dict[str, pd.DataFrame],
    output_dir: str
):
    """
    save all statistical results to organized directory structure.

    args:
        pooled_stats: dict mapping condition name to pooled test results
        differential_stats: dict mapping comparison name to differential test results
        output_dir: directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # save pooled condition results
    pooled_dir = output_path / "pooled_by_condition"
    pooled_dir.mkdir(exist_ok=True)

    for condition, df in pooled_stats.items():
        csv_path = pooled_dir / f"{condition}_pooled_stats.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"saved pooled stats: {csv_path}")

        # save significant heads only
        sig_df = df[df['significant']].sort_values('observed_score', ascending=False)
        sig_path = pooled_dir / f"{condition}_significant_heads.csv"
        sig_df.to_csv(sig_path, index=False)

    # save differential test results
    diff_dir = output_path / "differential_tests"
    diff_dir.mkdir(exist_ok=True)

    for comparison, df in differential_stats.items():
        csv_path = diff_dir / f"{comparison}_differential.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"saved differential test: {csv_path}")

        # save significant heads only
        sig_df = df[df['significant']].sort_values('differential_effect',
                                                    key=abs, ascending=False)
        sig_path = diff_dir / f"{comparison}_significant_differential.csv"
        sig_df.to_csv(sig_path, index=False)

    # generate summary report
    summary_path = output_path / "pooled_analysis_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("# pooled causal mediation analysis - statistical summary\n\n")

        f.write("## pooled condition tests\n\n")
        for condition, df in pooled_stats.items():
            n_sig = df['significant'].sum()
            total = len(df)
            f.write(f"{condition}:\n")
            f.write(f"  significant heads: {n_sig}/{total} ({n_sig/total:.1%})\n")

            if n_sig > 0:
                sig_df = df[df['significant']]
                f.write(f"  layers with significant heads: {sorted(sig_df['layer'].unique())}\n")
                top5 = df.nlargest(5, 'observed_score')[['layer', 'head', 'observed_score', 'q_value']]
                f.write(f"  top 5 heads:\n")
                for _, row in top5.iterrows():
                    f.write(f"    L{int(row['layer']):2d}H{int(row['head']):2d}: "
                           f"score={row['observed_score']:6.3f}, q={row['q_value']:.4f}\n")
            f.write("\n")

        f.write("\n## differential tests\n\n")
        for comparison, df in differential_stats.items():
            n_sig = df['significant'].sum()
            total = len(df)
            f.write(f"{comparison}:\n")
            f.write(f"  significant differential effects: {n_sig}/{total} ({n_sig/total:.1%})\n")

            if n_sig > 0:
                sig_df = df[df['significant']]
                f.write(f"  layers with differential effects: {sorted(sig_df['layer'].unique())}\n")
                top5 = df.nlargest(5, 'differential_effect', key=abs)[
                    ['layer', 'head', 'differential_effect', 'mean_condition_a', 'mean_condition_b', 'q_value']
                ]
                f.write(f"  top 5 differential effects:\n")
                for _, row in top5.iterrows():
                    f.write(f"    L{int(row['layer']):2d}H{int(row['head']):2d}: "
                           f"diff={row['differential_effect']:6.3f}, "
                           f"A={row['mean_condition_a']:6.3f}, "
                           f"B={row['mean_condition_b']:6.3f}, "
                           f"q={row['q_value']:.4f}\n")
            f.write("\n")

    logger.info(f"saved summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="pooled statistical analysis across experimental conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="root directory containing CMA results"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pooled_analysis",
        help="directory to save pooled analysis results"
    )

    parser.add_argument(
        "--conditions",
        nargs='+',
        default=["cross_belief", "within_belief", "control"],
        help="conditions to analyze (will auto-discover from results)"
    )

    parser.add_argument(
        "--n_permutations",
        type=int,
        default=1000,
        help="number of permutations for statistical tests"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="significance level for FDR correction"
    )

    parser.add_argument(
        "--differential_tests",
        nargs='+',
        default=["cross_belief:control", "cross_belief:within_belief"],
        help="differential tests to run (format: condition_a:condition_b)"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("pooled causal mediation analysis")
    logger.info("=" * 80)

    # discover result files
    files = discover_result_files(args.results_dir)
    if not files:
        logger.error("no result files found!")
        return

    # group by condition
    grouped = group_files_by_condition(files)

    # pool and test each condition
    pooled_stats = {}
    pooled_scores = {}

    for condition in args.conditions:
        if condition not in grouped:
            logger.warning(f"no files found for condition '{condition}'")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"analyzing condition: {condition}")
        logger.info(f"{'='*80}")

        # pool across templates
        pooled = load_and_pool_scores(grouped[condition])
        pooled_scores[condition] = pooled

        # run permutation test on pooled data
        results_df = permutation_test_heads(
            pooled,
            n_permutations=args.n_permutations,
            alpha=args.alpha
        )
        pooled_stats[condition] = results_df

    # run differential tests
    differential_stats = {}

    for test_spec in args.differential_tests:
        try:
            cond_a, cond_b = test_spec.split(':')

            if cond_a not in pooled_scores or cond_b not in pooled_scores:
                logger.warning(f"skipping differential test {test_spec}: missing condition")
                continue

            logger.info(f"\n{'='*80}")
            logger.info(f"differential test: {cond_a} vs {cond_b}")
            logger.info(f"{'='*80}")

            diff_df = differential_test(
                pooled_scores[cond_a],
                pooled_scores[cond_b],
                n_permutations=args.n_permutations,
                alpha=args.alpha
            )

            differential_stats[f"{cond_a}_vs_{cond_b}"] = diff_df

        except ValueError as e:
            logger.error(f"invalid test specification '{test_spec}': {e}")
            continue

    # save all results
    logger.info(f"\n{'='*80}")
    logger.info("saving results...")
    logger.info(f"{'='*80}")

    save_pooled_results(pooled_stats, differential_stats, args.output_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"analysis complete! results saved to: {args.output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()
