"""
statistical significance testing for causal mediation analysis.

implements permutation testing with FWER correction (max-statistic method)
to identify attention heads with statistically significant causal effects.
follows the methodology from LLMSymbMech (Section B.2).
"""

import logging
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def permutation_test_heads(
    per_sample_scores: torch.Tensor,
    n_permutations: int = 5000,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, float]:
    """
    permutation test with FWER correction via max-statistic method.

    for each permutation, randomly flip signs of per-sample scores
    (equivalent to swapping base/exp labels), compute mean per head,
    then take the max absolute score across ALL heads. the (1-alpha)
    percentile of these max scores is the global significance threshold.

    any head with |observed_mean| > threshold is significant, with
    FWER < alpha (i.e., <alpha chance that even one head is a false positive).

    args:
        per_sample_scores: tensor of shape (n_samples, n_layers, n_heads)
        n_permutations: number of permutations (default 5000, per LLMSymbMech)
        alpha: FWER significance level (default 0.05)

    returns:
        (dataframe, threshold) where dataframe has columns:
        - layer, head, observed_score, p_value, significant
        and threshold is the global FWER cutoff value.
    """
    logger.info(f"running permutation test with {n_permutations} permutations, FWER alpha={alpha}")

    # accept 2-D (n_samples, n_layers) for MLP/resid-stream patching, or 3-D
    # (n_samples, n_layers, n_heads) for attention-head patching. in the 2-D case
    # we add a singleton head dim so the rest of the function is dimension-agnostic.
    if per_sample_scores.dim() == 2:
        n_samples, n_layers = per_sample_scores.shape
        n_heads = 1
        per_sample_scores = per_sample_scores.unsqueeze(-1)
    elif per_sample_scores.dim() == 3:
        n_samples, n_layers, n_heads = per_sample_scores.shape
    else:
        raise ValueError(
            f"per_sample_scores must be 2-D or 3-D, got {per_sample_scores.dim()}-D "
            f"with shape {tuple(per_sample_scores.shape)}"
        )
    unit_label = "layers" if n_heads == 1 else "heads"
    logger.info(f"testing {n_layers * n_heads} {unit_label} across {n_samples} samples")

    # observed mean scores
    observed_mean = per_sample_scores.mean(dim=0).cpu().numpy()  # (n_layers, n_heads)

    # generate null distribution by sign-flipping (vectorized)
    logger.info("generating null distribution...")

    # random signs: (n_perm, n_samples, 1, 1) — same sign for all heads within a sample
    all_signs = torch.randint(0, 2, (n_permutations, n_samples, 1, 1),
                             device=per_sample_scores.device).float() * 2 - 1

    # broadcast multiply: (n_perm, n_samples, n_layers, n_heads)
    permuted_all = per_sample_scores.unsqueeze(0) * all_signs

    # permuted means: (n_perm, n_layers, n_heads)
    null_distributions = permuted_all.mean(dim=1).cpu().numpy()

    logger.info(f"generated {n_permutations} permutations")

    # FWER max-statistic: for each permutation, take the max |score| across all heads
    max_null_scores = np.abs(null_distributions).reshape(n_permutations, -1).max(axis=1)  # (n_perm,)

    # global threshold: (1-alpha) percentile of max null scores
    threshold = np.percentile(max_null_scores, 100 * (1 - alpha))
    logger.info(f"FWER threshold (alpha={alpha}): {threshold:.6f}")

    # significance: |observed| > threshold
    significant = np.abs(observed_mean) > threshold

    # also compute per-head p-values for reference (not used for significance)
    p_values = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            obs = observed_mean[layer, head]
            null = null_distributions[:, layer, head]
            p_values[layer, head] = (np.abs(null) >= np.abs(obs)).mean()

    # build results dataframe
    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            results.append({
                'layer': layer,
                'head': head,
                'observed_score': observed_mean[layer, head],
                'p_value': p_values[layer, head],
                'significant': significant[layer, head]
            })

    df = pd.DataFrame(results)
    n_sig = df['significant'].sum()
    logger.info(f"found {n_sig} significant heads (|score| > {threshold:.6f})")

    return df, threshold


def save_statistical_results(
    df: pd.DataFrame,
    save_folder: str,
    threshold: float = None
) -> Tuple[str, str, str]:
    """
    save statistical test results to files.

    args:
        df: dataframe from permutation_test_heads()
        save_folder: directory to save results
        threshold: FWER significance threshold (from max-statistic method)

    returns:
        tuple of (csv_path, sig_path, summary_path)
    """
    os.makedirs(save_folder, exist_ok=True)

    # save full table
    csv_path = os.path.join(save_folder, "statistical_test_results.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"saved full results table: {csv_path}")

    # save significant heads only
    sig_df = df[df['significant']].copy()
    sig_df = sig_df.sort_values('observed_score', ascending=False)
    sig_path = os.path.join(save_folder, "significant_heads.csv")
    sig_df.to_csv(sig_path, index=False)
    logger.info(f"saved significant heads: {sig_path}")

    # save human-readable summary
    summary_path = os.path.join(save_folder, "statistical_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("# causal mediation analysis - statistical testing results\n")
        f.write("# permutation test with FWER correction (max-statistic method)\n\n")

        if threshold is not None:
            f.write(f"FWER threshold: {threshold:.6f}\n")
        f.write(f"total heads tested: {len(df)}\n")
        f.write(f"significant heads (|score| > threshold): {sig_df.shape[0]}\n")
        f.write(f"proportion significant: {sig_df.shape[0] / len(df):.1%}\n\n")

        if len(sig_df) > 0:
            f.write("significant heads by layer:\n")
            for layer in sorted(sig_df['layer'].unique()):
                layer_heads = sig_df[sig_df['layer'] == layer]['head'].values
                f.write(f"  layer {layer}: heads {sorted(layer_heads)}\n")

            f.write(f"\ntop 10 heads by effect size:\n")
            top10 = df.nlargest(10, 'observed_score')[['layer', 'head', 'observed_score', 'p_value', 'significant']]
            f.write(top10.to_string(index=False))
            f.write("\n")
        else:
            f.write("no significant heads found\n")

    logger.info(f"saved summary: {summary_path}")

    return csv_path, sig_path, summary_path
