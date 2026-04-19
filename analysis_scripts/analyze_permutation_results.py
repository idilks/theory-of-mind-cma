#!/usr/bin/env python3
"""
analyze permutation test results from wandb exports.

produces:
1. individual template summaries
2. aggregate analyses across templates
3. comparison: abstract vs control conditions
"""

import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# setup
WANDB_DIR = "C:/Users/idilk/Desktop/tom_dev/wandb_permtest"
RESULTS_DIR = "C:/Users/idilk/Desktop/tom_dev/results/permutation_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_template_data(csv_path: str) -> pd.DataFrame:
    """load and clean a single template's permutation results."""
    # read without auto-converting to avoid boolean parsing issues
    df = pd.read_csv(csv_path, dtype={'significant': str})

    # strip quotes from all object columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.strip().str.strip('"')

    # convert significant column after cleaning
    df['significant'] = df['significant'].str.lower().map({'true': True, 'false': False})

    # convert numeric columns explicitly
    df['layer'] = pd.to_numeric(df['layer'], errors='coerce').astype(int)
    df['head'] = pd.to_numeric(df['head'], errors='coerce').astype(int)
    df['observed_score'] = pd.to_numeric(df['observed_score'], errors='coerce')
    df['p_value'] = pd.to_numeric(df['p_value'], errors='coerce')
    if 'q_value' in df.columns:
        df['q_value'] = pd.to_numeric(df['q_value'], errors='coerce')

    return df


def summarize_template(df: pd.DataFrame, template_name: str) -> Dict:
    """generate summary statistics for a single template."""
    sig_heads = df[df['significant'] == True]

    summary = {
        'template': template_name,
        'total_heads': len(df),
        'significant_heads': len(sig_heads),
        'proportion_significant': len(sig_heads) / len(df),
        'mean_effect_size': df['observed_score'].mean(),
        'mean_effect_size_sig': sig_heads['observed_score'].mean() if len(sig_heads) > 0 else 0,
        'max_effect_size': df['observed_score'].max(),
        'min_effect_size': df['observed_score'].min(),
        'top_layer': sig_heads['layer'].mode()[0] if len(sig_heads) > 0 else None,
        'layer_distribution': sig_heads.groupby('layer').size().to_dict() if len(sig_heads) > 0 else {}
    }

    return summary


def aggregate_templates(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """aggregate results across multiple templates."""
    # merge all dataframes on layer/head
    merged = None

    for template_name, df in dfs.items():
        df_copy = df[['layer', 'head', 'observed_score', 'significant']].copy()
        df_copy = df_copy.rename(columns={
            'observed_score': f'score_{template_name}',
            'significant': f'sig_{template_name}'
        })

        if merged is None:
            merged = df_copy
        else:
            merged = merged.merge(df_copy, on=['layer', 'head'], how='outer')

    # compute aggregate statistics
    score_cols = [c for c in merged.columns if c.startswith('score_')]
    sig_cols = [c for c in merged.columns if c.startswith('sig_')]

    merged['mean_score'] = merged[score_cols].mean(axis=1)
    merged['std_score'] = merged[score_cols].std(axis=1)
    merged['n_significant'] = merged[sig_cols].sum(axis=1)
    merged['consistency'] = merged['n_significant'] / len(sig_cols)
    merged['any_significant'] = merged['n_significant'] > 0
    merged['all_significant'] = merged['n_significant'] == len(sig_cols)

    return merged


def plot_layer_distribution(summary_stats: List[Dict], condition: str, output_path: str):
    """plot distribution of significant heads across layers."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for stat in summary_stats:
        template = stat['template']
        layer_dist = stat['layer_distribution']
        if layer_dist:
            layers = list(layer_dist.keys())
            counts = list(layer_dist.values())
            ax.plot(layers, counts, marker='o', label=template, alpha=0.7)

    ax.set_xlabel('layer')
    ax.set_ylabel('number of significant heads')
    ax.set_title(f'significant heads by layer - {condition}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_aggregate_heatmap(agg_df: pd.DataFrame, condition: str, output_path: str):
    """plot heatmap of mean causal scores."""
    # reshape for heatmap
    n_layers = agg_df['layer'].max() + 1
    n_heads = agg_df['head'].max() + 1

    heatmap_data = np.zeros((n_layers, n_heads))

    for _, row in agg_df.iterrows():
        heatmap_data[int(row['layer']), int(row['head'])] = row['mean_score']

    fig, ax = plt.subplots(figsize=(16, 10))

    sns.heatmap(heatmap_data,
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'mean causal score'},
                ax=ax)

    ax.set_xlabel('head')
    ax.set_ylabel('layer')
    ax.set_title(f'aggregate causal mediation scores - {condition}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def analyze_condition(condition_dir: str, condition_name: str):
    """analyze all templates in a condition directory."""
    print(f"\n{'='*60}")
    print(f"analyzing: {condition_name}")
    print(f"{'='*60}\n")

    # load all healthy templates (use forward slashes for WSL)
    condition_dir = condition_dir.replace('\\', '/')
    if "abstract" in condition_name:
        csv_files = glob.glob(f"{condition_dir}/*healthy*.csv")
    else:
        csv_files = glob.glob(f"{condition_dir}/*.csv")

    if not csv_files:
        print(f"no csv files found in {condition_dir}")
        return None, None

    print(f"found {len(csv_files)} templates:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # load data
    template_dfs = {}
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        template_name = basename.split('_z_')[0].split('_')[-1]  # extract template name
        df = load_template_data(csv_file)
        template_dfs[template_name] = df

    # individual template analysis
    summary_stats = []
    print("individual template results:")
    print("-" * 60)

    for template_name, df in template_dfs.items():
        summary = summarize_template(df, template_name)
        summary_stats.append(summary)

        print(f"\n{template_name}:")
        print(f"  significant heads: {summary['significant_heads']}/{summary['total_heads']} ({summary['proportion_significant']*100:.1f}%)")
        print(f"  mean effect size (all): {summary['mean_effect_size']:.4f}")
        print(f"  mean effect size (sig): {summary['mean_effect_size_sig']:.4f}")
        print(f"  max effect size: {summary['max_effect_size']:.4f}")

        if summary['layer_distribution']:
            top_layers = sorted(summary['layer_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"  top layers: {', '.join([f'L{l}({c})' for l, c in top_layers])}")

    # aggregate analysis
    print(f"\n\naggregate analysis ({len(template_dfs)} templates):")
    print("-" * 60)

    agg_df = aggregate_templates(template_dfs)

    # consistency analysis
    consistent_sig = agg_df[agg_df['all_significant'] == True]
    any_sig = agg_df[agg_df['any_significant'] == True]

    print(f"heads significant in ALL templates: {len(consistent_sig)}")
    print(f"heads significant in ANY template: {len(any_sig)}")
    print(f"mean consistency score: {agg_df['consistency'].mean():.3f}")

    # top consistent heads
    if len(consistent_sig) > 0:
        print("\ntop 10 most consistent heads (significant in all templates):")
        top_consistent = consistent_sig.nlargest(10, 'mean_score')[['layer', 'head', 'mean_score', 'std_score']]
        print(top_consistent.to_string(index=False))

    # save results
    condition_result_dir = os.path.join(RESULTS_DIR, condition_name)
    os.makedirs(condition_result_dir, exist_ok=True)

    # save summary stats
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(condition_result_dir, "template_summaries.csv"), index=False)

    # save aggregate data
    agg_df.to_csv(os.path.join(condition_result_dir, "aggregate_results.csv"), index=False)

    # save consistent significant heads
    if len(consistent_sig) > 0:
        consistent_sig.to_csv(os.path.join(condition_result_dir, "consistent_significant_heads.csv"), index=False)

    # plots
    plot_layer_distribution(summary_stats, condition_name, os.path.join(condition_result_dir, "layer_distribution.png"))
    plot_aggregate_heatmap(agg_df, condition_name, os.path.join(condition_result_dir, "aggregate_heatmap.png"))

    print(f"\nresults saved to: {condition_result_dir}")

    return summary_df, agg_df


def compare_conditions(abstract_agg: pd.DataFrame, control_agg: pd.DataFrame):
    """compare abstract vs control conditions."""
    print(f"\n{'='*60}")
    print("condition comparison: abstract vs control")
    print(f"{'='*60}\n")

    # merge on layer/head
    comparison = abstract_agg[['layer', 'head', 'mean_score', 'consistency', 'any_significant']].merge(
        control_agg[['layer', 'head', 'mean_score', 'consistency', 'any_significant']],
        on=['layer', 'head'],
        suffixes=('_abstract', '_control')
    )

    comparison['score_diff'] = comparison['mean_score_abstract'] - comparison['mean_score_control']
    comparison['abstract_specific'] = (comparison['any_significant_abstract'] == True) & (comparison['any_significant_control'] == False)
    comparison['control_specific'] = (comparison['any_significant_control'] == True) & (comparison['any_significant_abstract'] == False)
    comparison['both_significant'] = (comparison['any_significant_abstract'] == True) & (comparison['any_significant_control'] == True)

    print(f"abstract-specific heads: {comparison['abstract_specific'].sum()}")
    print(f"control-specific heads: {comparison['control_specific'].sum()}")
    print(f"heads significant in both: {comparison['both_significant'].sum()}")

    print("\ntop 10 abstract-specific heads:")
    abstract_specific = comparison[comparison['abstract_specific'] == True].nlargest(10, 'mean_score_abstract')
    print(abstract_specific[['layer', 'head', 'mean_score_abstract', 'consistency_abstract']].to_string(index=False))

    # save comparison
    comparison.to_csv(os.path.join(RESULTS_DIR, "abstract_vs_control_comparison.csv"), index=False)

    # plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # scatter plot
    axes[0].scatter(comparison['mean_score_control'],
                   comparison['mean_score_abstract'],
                   alpha=0.3, s=10)
    axes[0].axline((0, 0), slope=1, color='red', linestyle='--', alpha=0.5, label='y=x')
    axes[0].set_xlabel('control mean score')
    axes[0].set_ylabel('abstract mean score')
    axes[0].set_title('abstract vs control causal scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # difference histogram
    axes[1].hist(comparison['score_diff'], bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('score difference (abstract - control)')
    axes[1].set_ylabel('frequency')
    axes[1].set_title('distribution of score differences')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "abstract_vs_control_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\ncomparison results saved to: {RESULTS_DIR}")


def main():
    """run complete analysis pipeline."""
    print("permutation test analysis pipeline")
    print("="*60)

    # analyze abstract condition
    abstract_summary, abstract_agg = analyze_condition(
        f"{WANDB_DIR}/abstract",
        "abstract"
    )

    # analyze control conditions
    control_belief_summary, control_belief_agg = analyze_condition(
        f"{WANDB_DIR}/control/belief_patch",
        "control_belief_patch"
    )

    control_end_summary, control_end_agg = analyze_condition(
        f"{WANDB_DIR}/control/end_patch",
        "control_end_patch"
    )

    # compare abstract vs control (use belief_patch as main control)
    if abstract_agg is not None and control_belief_agg is not None:
        compare_conditions(abstract_agg, control_belief_agg)

    print("\n" + "="*60)
    print("analysis complete!")
    print(f"all results saved to: {RESULTS_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
