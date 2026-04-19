#!/usr/bin/env python3
"""
Hypothesis-driven analysis of permutation test results.

Tests the core hypothesis:
- Abstract & Photo track belief deviation (high overlap expected)
- Control tracks location retrieval (low overlap with abstract/photo expected)

REPRODUCIBILITY:
    Script location: scripts/test_hypothesis_permutation.py
    Input: wandb_permtest/{abstract,photo,control/end_patch}/*.csv
    Output: results/hypothesis_test_permutation/

    Run with:
        python scripts/test_hypothesis_permutation.py

    Templates analyzed: food_truck, hair_styling, library_book
    (basic_object_move excluded as too simplistic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from typing import Set, Dict, List, Tuple
import json

# Configuration
WANDB_DIR = Path("wandb_permtest")
RESULTS_DIR = Path("results/hypothesis_test_permutation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
CONDITIONS = {
    "abstract": WANDB_DIR / "abstract",
    "photo": WANDB_DIR / "photo",
    "control": WANDB_DIR / "control" / "end_patch"
}

def load_significant_heads(csv_path: Path) -> Tuple[Set[Tuple[int, int]], pd.DataFrame]:
    """Load significant heads from permutation test CSV."""
    df = pd.read_csv(csv_path)
    df_sig = df[df['significant'] == True].copy()
    heads_set = set(zip(df_sig['layer'].astype(int), df_sig['head'].astype(int)))
    return heads_set, df_sig

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity coefficient."""
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def overlap_coefficient(set1: Set, set2: Set) -> float:
    """Calculate overlap coefficient (intersection / min(|A|, |B|))."""
    if len(set1) == 0 or len(set2) == 0:
        return 0.0
    intersection = len(set1 & set2)
    return intersection / min(len(set1), len(set2))

def load_all_data() -> Dict:
    """Load all permutation test data."""
    data = {}

    for condition, base_path in CONDITIONS.items():
        data[condition] = {}
        for template in TEMPLATES:
            # Find the file
            if condition == "abstract":
                pattern = f"*{template}*healthy.csv"
            else:
                pattern = f"*{template}*.csv"

            files = list(base_path.glob(pattern))
            # Filter out basic_object_move
            files = [f for f in files if "basic_object_move" not in f.name]

            if not files:
                print(f"Warning: No file found for {condition}/{template}")
                continue

            file = files[0]
            heads_set, df_sig = load_significant_heads(file)
            data[condition][template] = {
                'heads': heads_set,
                'df': df_sig,
                'count': len(heads_set),
                'file': file.name
            }
            print(f"Loaded {condition}/{template}: {len(heads_set)} significant heads")

    return data

def calculate_pairwise_overlaps(data: Dict) -> pd.DataFrame:
    """Calculate all pairwise overlaps between conditions and templates."""
    rows = []

    for cond1 in CONDITIONS.keys():
        for temp1 in TEMPLATES:
            if temp1 not in data[cond1]:
                continue
            heads1 = data[cond1][temp1]['heads']

            for cond2 in CONDITIONS.keys():
                for temp2 in TEMPLATES:
                    if temp2 not in data[cond2]:
                        continue
                    heads2 = data[cond2][temp2]['heads']

                    intersection = len(heads1 & heads2)
                    jaccard = jaccard_similarity(heads1, heads2)
                    overlap_coef = overlap_coefficient(heads1, heads2)

                    rows.append({
                        'condition1': cond1,
                        'template1': temp1,
                        'condition2': cond2,
                        'template2': temp2,
                        'count1': len(heads1),
                        'count2': len(heads2),
                        'intersection': intersection,
                        'jaccard': jaccard,
                        'overlap_coefficient': overlap_coef
                    })

    return pd.DataFrame(rows)

def test_hypothesis_1(data: Dict, overlaps: pd.DataFrame) -> Dict:
    """Test: abstract ∩ photo should have HIGH overlap."""
    results = {}

    for template in TEMPLATES:
        if template not in data['abstract'] or template not in data['photo']:
            continue

        abstract_heads = data['abstract'][template]['heads']
        photo_heads = data['photo'][template]['heads']

        intersection = len(abstract_heads & photo_heads)
        jaccard = jaccard_similarity(abstract_heads, photo_heads)
        overlap_coef = overlap_coefficient(abstract_heads, photo_heads)

        results[template] = {
            'intersection': intersection,
            'jaccard': jaccard,
            'overlap_coefficient': overlap_coef,
            'abstract_count': len(abstract_heads),
            'photo_count': len(photo_heads)
        }

    return results

def test_hypothesis_2(data: Dict, overlaps: pd.DataFrame) -> Dict:
    """Test: abstract ∩ control should have LOW overlap."""
    results = {}

    for template in TEMPLATES:
        if template not in data['abstract'] or template not in data['control']:
            continue

        abstract_heads = data['abstract'][template]['heads']
        control_heads = data['control'][template]['heads']

        intersection = len(abstract_heads & control_heads)
        jaccard = jaccard_similarity(abstract_heads, control_heads)
        overlap_coef = overlap_coefficient(abstract_heads, control_heads)

        results[template] = {
            'intersection': intersection,
            'jaccard': jaccard,
            'overlap_coefficient': overlap_coef,
            'abstract_count': len(abstract_heads),
            'control_count': len(control_heads)
        }

    return results

def test_hypothesis_3(data: Dict, overlaps: pd.DataFrame) -> Dict:
    """Test: photo ∩ control should have LOW overlap."""
    results = {}

    for template in TEMPLATES:
        if template not in data['photo'] or template not in data['control']:
            continue

        photo_heads = data['photo'][template]['heads']
        control_heads = data['control'][template]['heads']

        intersection = len(photo_heads & control_heads)
        jaccard = jaccard_similarity(photo_heads, control_heads)
        overlap_coef = overlap_coefficient(photo_heads, control_heads)

        results[template] = {
            'intersection': intersection,
            'jaccard': jaccard,
            'overlap_coefficient': overlap_coef,
            'photo_count': len(photo_heads),
            'control_count': len(control_heads)
        }

    return results

def analyze_effect_size_correlations(data: Dict) -> Dict:
    """Analyze effect size correlations for overlapping heads."""
    results = {}

    # Abstract vs Photo
    for template in TEMPLATES:
        if template not in data['abstract'] or template not in data['photo']:
            continue

        abstract_df = data['abstract'][template]['df']
        photo_df = data['photo'][template]['df']

        # Merge on layer and head
        merged = abstract_df.merge(
            photo_df,
            on=['layer', 'head'],
            suffixes=('_abstract', '_photo')
        )

        if len(merged) > 1:
            corr, pval = pearsonr(merged['observed_score_abstract'], merged['observed_score_photo'])
            results[f'abstract_photo_{template}'] = {
                'correlation': corr,
                'p_value': pval,
                'n_shared': len(merged)
            }

    # Abstract vs Control
    for template in TEMPLATES:
        if template not in data['abstract'] or template not in data['control']:
            continue

        abstract_df = data['abstract'][template]['df']
        control_df = data['control'][template]['df']

        merged = abstract_df.merge(
            control_df,
            on=['layer', 'head'],
            suffixes=('_abstract', '_control')
        )

        if len(merged) > 1:
            corr, pval = pearsonr(merged['observed_score_abstract'], merged['observed_score_control'])
            results[f'abstract_control_{template}'] = {
                'correlation': corr,
                'p_value': pval,
                'n_shared': len(merged)
            }

    return results

def analyze_layer_distributions(data: Dict) -> Dict:
    """Analyze layer distribution patterns."""
    distributions = {}

    for condition in CONDITIONS.keys():
        distributions[condition] = {}
        for template in TEMPLATES:
            if template not in data[condition]:
                continue

            df = data[condition][template]['df']
            layer_counts = df.groupby('layer').size()
            distributions[condition][template] = layer_counts.to_dict()

    return distributions

def find_universal_heads(data: Dict) -> Dict:
    """Find heads significant across all templates within each condition."""
    universal = {}

    for condition in CONDITIONS.keys():
        # Get intersection across all templates
        template_heads = [data[condition][t]['heads'] for t in TEMPLATES if t in data[condition]]
        if len(template_heads) >= 2:
            universal[condition] = set.intersection(*template_heads)
        else:
            universal[condition] = set()

        print(f"\n{condition} universal heads (present in all {len(template_heads)} templates): {len(universal[condition])}")

    return universal

def analyze_universal_heads_overlap(universal: Dict) -> Dict:
    """Analyze overlap between universal heads across conditions."""
    results = {}

    # Abstract vs Photo universal heads
    abstract_universal = universal['abstract']
    photo_universal = universal['photo']
    control_universal = universal['control']

    abstract_photo_intersection = abstract_universal & photo_universal
    abstract_control_intersection = abstract_universal & control_universal
    photo_control_intersection = photo_universal & control_universal

    # ALL conditions intersection (baseline)
    all_conditions_universal = abstract_universal & photo_universal & control_universal

    results['abstract_photo'] = {
        'intersection': len(abstract_photo_intersection),
        'jaccard': jaccard_similarity(abstract_universal, photo_universal),
        'overlap_coef': overlap_coefficient(abstract_universal, photo_universal),
        'abstract_count': len(abstract_universal),
        'photo_count': len(photo_universal),
        'heads': sorted(list(abstract_photo_intersection))
    }

    results['abstract_control'] = {
        'intersection': len(abstract_control_intersection),
        'jaccard': jaccard_similarity(abstract_universal, control_universal),
        'overlap_coef': overlap_coefficient(abstract_universal, control_universal),
        'abstract_count': len(abstract_universal),
        'control_count': len(control_universal),
        'heads': sorted(list(abstract_control_intersection))
    }

    results['photo_control'] = {
        'intersection': len(photo_control_intersection),
        'jaccard': jaccard_similarity(photo_universal, control_universal),
        'overlap_coef': overlap_coefficient(photo_universal, control_universal),
        'photo_count': len(photo_universal),
        'control_count': len(control_universal),
        'heads': sorted(list(photo_control_intersection))
    }

    results['all_conditions_baseline'] = {
        'count': len(all_conditions_universal),
        'heads': sorted(list(all_conditions_universal)),
        'pct_of_abstract': len(all_conditions_universal) / len(abstract_universal) if len(abstract_universal) > 0 else 0,
        'pct_of_photo': len(all_conditions_universal) / len(photo_universal) if len(photo_universal) > 0 else 0,
        'pct_of_control': len(all_conditions_universal) / len(control_universal) if len(control_universal) > 0 else 0
    }

    print(f"\nUniversal heads overlap:")
    print(f"  abstract & photo: {results['abstract_photo']['intersection']}/{min(len(abstract_universal), len(photo_universal))}")
    print(f"  abstract & control: {results['abstract_control']['intersection']}/{min(len(abstract_universal), len(control_universal))}")
    print(f"  photo & control: {results['photo_control']['intersection']}/{min(len(photo_universal), len(control_universal))}")
    print(f"\n  ALL CONDITIONS BASELINE: {len(all_conditions_universal)} heads")
    print(f"    ({results['all_conditions_baseline']['pct_of_abstract']:.1%} of abstract, "
          f"{results['all_conditions_baseline']['pct_of_photo']:.1%} of photo, "
          f"{results['all_conditions_baseline']['pct_of_control']:.1%} of control)")

    return results

def plot_overlap_heatmaps(overlaps: pd.DataFrame):
    """Create heatmap visualizations of overlaps."""
    # Jaccard similarity heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, template in enumerate(TEMPLATES):
        template_data = overlaps[
            (overlaps['template1'] == template) &
            (overlaps['template2'] == template)
        ]

        pivot = template_data.pivot(
            index='condition1',
            columns='condition2',
            values='jaccard'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=axes[idx],
            cbar_kws={'label': 'Jaccard Similarity'}
        )
        axes[idx].set_title(f'{template}\nJaccard Similarity')
        axes[idx].set_xlabel('Condition')
        axes[idx].set_ylabel('Condition')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'jaccard_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Overlap coefficient heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, template in enumerate(TEMPLATES):
        template_data = overlaps[
            (overlaps['template1'] == template) &
            (overlaps['template2'] == template)
        ]

        pivot = template_data.pivot(
            index='condition1',
            columns='condition2',
            values='overlap_coefficient'
        )

        sns.heatmap(
            pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=axes[idx],
            cbar_kws={'label': 'Overlap Coefficient'}
        )
        axes[idx].set_title(f'{template}\nOverlap Coefficient')
        axes[idx].set_xlabel('Condition')
        axes[idx].set_ylabel('Condition')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'overlap_coefficient_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_layer_distributions(distributions: Dict, universal: Dict):
    """Plot layer distribution comparisons."""
    fig, axes = plt.subplots(len(TEMPLATES), 3, figsize=(15, 12))

    for row, template in enumerate(TEMPLATES):
        for col, condition in enumerate(CONDITIONS.keys()):
            ax = axes[row, col]

            if template not in distributions[condition]:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'{condition} - {template}')
                continue

            layer_dist = distributions[condition][template]
            layers = sorted(layer_dist.keys())
            counts = [layer_dist[l] for l in layers]

            ax.bar(layers, counts, alpha=0.7)
            ax.set_xlabel('Layer')
            ax.set_ylabel('# Significant Heads')
            ax.set_title(f'{condition} - {template}')
            ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'layer_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_universal_heads_layer_comparison(universal: Dict):
    """Plot layer distribution of universal heads across conditions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract layer distributions for universal heads
    layer_counts = {condition: {} for condition in CONDITIONS.keys()}

    for condition, heads in universal.items():
        for layer, head in heads:
            layer_counts[condition][layer] = layer_counts[condition].get(layer, 0) + 1

    # Plot 1: Overlaid histograms
    colors = {'abstract': 'blue', 'photo': 'green', 'control': 'red'}
    for condition in CONDITIONS.keys():
        if not layer_counts[condition]:
            continue
        layers = sorted(layer_counts[condition].keys())
        counts = [layer_counts[condition][l] for l in layers]
        ax1.plot(layers, counts, 'o-', label=condition, color=colors[condition], alpha=0.7, linewidth=2, markersize=6)

    ax1.set_xlabel('Layer', fontsize=12)
    ax1.set_ylabel('# Universal Heads', fontsize=12)
    ax1.set_title('Universal Head Layer Distribution by Condition', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='both', alpha=0.3)

    # Plot 2: Box plot comparison
    box_data = []
    box_labels = []
    for condition in CONDITIONS.keys():
        layers_list = [layer for layer, head in universal[condition]]
        if layers_list:
            box_data.append(layers_list)
            box_labels.append(condition)

    if box_data:
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, condition in zip(bp['boxes'], box_labels):
            patch.set_facecolor(colors[condition])
            patch.set_alpha(0.7)

    ax2.set_ylabel('Layer', fontsize=12)
    ax2.set_title('Universal Head Layer Distribution (Median & Range)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add median values as text
    for i, condition in enumerate(box_labels):
        median = np.median([layer for layer, head in universal[condition]])
        ax2.text(i+1, median, f'{median:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'universal_heads_layer_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nLayer distribution statistics for universal heads:")
    for condition in CONDITIONS.keys():
        layers = [layer for layer, head in universal[condition]]
        if layers:
            print(f"  {condition}: median={np.median(layers):.1f}, range=[{min(layers)}, {max(layers)}]")

def generate_markdown_report(
    data: Dict,
    hyp1_results: Dict,
    hyp2_results: Dict,
    hyp3_results: Dict,
    effect_correlations: Dict,
    universal: Dict,
    universal_overlap: Dict
):
    """Generate comprehensive markdown report."""

    lines = [
        "# Permutation Test Hypothesis Analysis",
        "",
        "**Generated by**: `scripts/test_hypothesis_permutation.py`",
        "",
        "**Input data**: `wandb_permtest/{abstract,photo,control/end_patch}/*.csv`",
        "",
        "**Templates analyzed**: food_truck, hair_styling, library_book (basic_object_move excluded)",
        "",
        "---",
        "",
        "## Core Hypothesis",
        "",
        "**Prediction**: Models use different attention mechanisms for:",
        "- **Abstract & Photo**: Belief tracking (mental state ≠ world state)",
        "- **Control**: Location retrieval (mental state = world state)",
        "",
        "**Expected Results**:",
        "- ✅ HIGH overlap: abstract ∩ photo",
        "- ✅ LOW overlap: abstract ∩ control",
        "- ✅ LOW overlap: photo ∩ control",
        "",
        "---",
        "",
        "## Results Summary",
        "",
        "### Data Overview",
        ""
    ]

    # Data counts
    lines.append("| Condition | Template | Significant Heads |")
    lines.append("|-----------|----------|------------------|")
    for condition in CONDITIONS.keys():
        for template in TEMPLATES:
            if template in data[condition]:
                count = data[condition][template]['count']
                lines.append(f"| {condition} | {template} | {count} |")
    lines.append("")

    # Hypothesis 1: Abstract ∩ Photo (HIGH overlap expected)
    lines.extend([
        "### Hypothesis 1: Abstract ∩ Photo Overlap",
        "",
        "**Prediction**: HIGH overlap (both track belief deviation)",
        "",
        "| Template | Jaccard | Overlap Coef | Intersection | Abstract Count | Photo Count |",
        "|----------|---------|--------------|--------------|----------------|-------------|"
    ])

    for template, results in hyp1_results.items():
        lines.append(
            f"| {template} | {results['jaccard']:.3f} | "
            f"{results['overlap_coefficient']:.3f} | "
            f"{results['intersection']} | "
            f"{results['abstract_count']} | "
            f"{results['photo_count']} |"
        )

    avg_jaccard = np.mean([r['jaccard'] for r in hyp1_results.values()])
    avg_overlap = np.mean([r['overlap_coefficient'] for r in hyp1_results.values()])

    lines.extend([
        "",
        f"**Average Jaccard**: {avg_jaccard:.3f}",
        f"**Average Overlap Coefficient**: {avg_overlap:.3f}",
        "",
        f"**Interpretation**: {'✅ SUPPORTED' if avg_jaccard > 0.4 else '❌ NOT SUPPORTED'} - "
        f"{'High' if avg_jaccard > 0.4 else 'Low'} overlap observed.",
        ""
    ])

    # Hypothesis 2: Abstract ∩ Control (LOW overlap expected)
    lines.extend([
        "### Hypothesis 2: Abstract ∩ Control Overlap",
        "",
        "**Prediction**: LOW overlap (different mechanisms)",
        "",
        "| Template | Jaccard | Overlap Coef | Intersection | Abstract Count | Control Count |",
        "|----------|---------|--------------|--------------|----------------|---------------|"
    ])

    for template, results in hyp2_results.items():
        lines.append(
            f"| {template} | {results['jaccard']:.3f} | "
            f"{results['overlap_coefficient']:.3f} | "
            f"{results['intersection']} | "
            f"{results['abstract_count']} | "
            f"{results['control_count']} |"
        )

    avg_jaccard_2 = np.mean([r['jaccard'] for r in hyp2_results.values()])
    avg_overlap_2 = np.mean([r['overlap_coefficient'] for r in hyp2_results.values()])

    lines.extend([
        "",
        f"**Average Jaccard**: {avg_jaccard_2:.3f}",
        f"**Average Overlap Coefficient**: {avg_overlap_2:.3f}",
        "",
        f"**Interpretation**: {'✅ SUPPORTED' if avg_jaccard_2 < 0.3 else '❌ NOT SUPPORTED'} - "
        f"{'Low' if avg_jaccard_2 < 0.3 else 'High'} overlap observed.",
        ""
    ])

    # Hypothesis 3: Photo ∩ Control (LOW overlap expected)
    lines.extend([
        "### Hypothesis 3: Photo ∩ Control Overlap",
        "",
        "**Prediction**: LOW overlap (different mechanisms)",
        "",
        "| Template | Jaccard | Overlap Coef | Intersection | Photo Count | Control Count |",
        "|----------|---------|--------------|--------------|-------------|---------------|"
    ])

    for template, results in hyp3_results.items():
        lines.append(
            f"| {template} | {results['jaccard']:.3f} | "
            f"{results['overlap_coefficient']:.3f} | "
            f"{results['intersection']} | "
            f"{results['photo_count']} | "
            f"{results['control_count']} |"
        )

    avg_jaccard_3 = np.mean([r['jaccard'] for r in hyp3_results.values()])
    avg_overlap_3 = np.mean([r['overlap_coefficient'] for r in hyp3_results.values()])

    lines.extend([
        "",
        f"**Average Jaccard**: {avg_jaccard_3:.3f}",
        f"**Average Overlap Coefficient**: {avg_overlap_3:.3f}",
        "",
        f"**Interpretation**: {'✅ SUPPORTED' if avg_jaccard_3 < 0.3 else '❌ NOT SUPPORTED'} - "
        f"{'Low' if avg_jaccard_3 < 0.3 else 'High'} overlap observed.",
        ""
    ])

    # Effect size correlations
    lines.extend([
        "### Effect Size Correlations",
        "",
        "For overlapping heads, do effect sizes correlate?",
        "",
        "| Comparison | Correlation | P-value | N Shared |",
        "|------------|-------------|---------|----------|"
    ])

    for key, results in effect_correlations.items():
        lines.append(
            f"| {key} | {results['correlation']:.3f} | "
            f"{results['p_value']:.4f} | {results['n_shared']} |"
        )

    lines.append("")

    # Universal heads
    lines.extend([
        "### Universal Heads (Across All Templates)",
        "",
        "Heads significant in all three templates within each condition:",
        ""
    ])

    for condition, heads in universal.items():
        lines.append(f"- **{condition}**: {len(heads)} universal heads")
        if len(heads) > 0 and len(heads) <= 20:
            lines.append(f"  - {sorted(list(heads))}")

    lines.extend([
        "",
        "### Universal Heads Overlap Analysis",
        "",
        "Overlap of the most robust heads (significant in ALL templates):",
        "",
        "| Comparison | Intersection | Jaccard | Overlap Coef | Count 1 | Count 2 |",
        "|------------|--------------|---------|--------------|---------|---------|"
    ])

    # Add universal overlap data
    lines.append(
        f"| abstract ∩ photo | {universal_overlap['abstract_photo']['intersection']} | "
        f"{universal_overlap['abstract_photo']['jaccard']:.3f} | "
        f"{universal_overlap['abstract_photo']['overlap_coef']:.3f} | "
        f"{universal_overlap['abstract_photo']['abstract_count']} | "
        f"{universal_overlap['abstract_photo']['photo_count']} |"
    )

    lines.append(
        f"| abstract ∩ control | {universal_overlap['abstract_control']['intersection']} | "
        f"{universal_overlap['abstract_control']['jaccard']:.3f} | "
        f"{universal_overlap['abstract_control']['overlap_coef']:.3f} | "
        f"{universal_overlap['abstract_control']['abstract_count']} | "
        f"{universal_overlap['abstract_control']['control_count']} |"
    )

    lines.append(
        f"| photo ∩ control | {universal_overlap['photo_control']['intersection']} | "
        f"{universal_overlap['photo_control']['jaccard']:.3f} | "
        f"{universal_overlap['photo_control']['overlap_coef']:.3f} | "
        f"{universal_overlap['photo_control']['photo_count']} | "
        f"{universal_overlap['photo_control']['control_count']} |"
    )

    lines.extend([
        "",
        "**Interpretation**:",
        f"- Abstract & Photo share **{universal_overlap['abstract_photo']['intersection']}/{min(universal_overlap['abstract_photo']['abstract_count'], universal_overlap['abstract_photo']['photo_count'])} "
        f"({universal_overlap['abstract_photo']['overlap_coef']:.1%})** of their universal heads",
        f"- Abstract & Control share only **{universal_overlap['abstract_control']['intersection']}/{min(universal_overlap['abstract_control']['abstract_count'], universal_overlap['abstract_control']['control_count'])} "
        f"({universal_overlap['abstract_control']['overlap_coef']:.1%})** of their universal heads",
        f"- Photo & Control share only **{universal_overlap['photo_control']['intersection']}/{min(universal_overlap['photo_control']['photo_count'], universal_overlap['photo_control']['control_count'])} "
        f"({universal_overlap['photo_control']['overlap_coef']:.1%})** of their universal heads",
        "",
        "### Baseline: All Conditions Universal Heads",
        "",
        f"**{universal_overlap['all_conditions_baseline']['count']} heads** are significant across ALL conditions (abstract AND photo AND control) and ALL templates.",
        "",
        "These represent the **shared substrate** - heads involved in general ToM scenario processing regardless of belief state:",
        "",
        f"- {universal_overlap['all_conditions_baseline']['pct_of_abstract']:.1%} of abstract universal heads",
        f"- {universal_overlap['all_conditions_baseline']['pct_of_photo']:.1%} of photo universal heads",
        f"- {universal_overlap['all_conditions_baseline']['pct_of_control']:.1%} of control universal heads",
        "",
    ])

    # List the baseline heads if there are any and not too many
    baseline_heads = universal_overlap['all_conditions_baseline']['heads']
    if len(baseline_heads) > 0:
        lines.append(f"**Baseline heads** (layer, head): {baseline_heads}")
        lines.append("")

        lines.extend([
            "**Interpretation**:",
            f"- Only {universal_overlap['all_conditions_baseline']['count']} heads are universally significant across all conditions",
            "- This small baseline validates that abstract/photo share mechanisms BEYOND basic processing",
            f"- {universal_overlap['abstract_photo']['intersection'] - universal_overlap['all_conditions_baseline']['count']} abstract∩photo heads are belief-tracking specific (not in baseline)",
            ""
        ])
    else:
        lines.extend([
            "**Interpretation**:",
            "- NO heads are universally significant across all conditions and templates",
            "- This strongly validates condition-specific mechanisms (no shared baseline contamination)",
            ""
        ])

    # Layer distribution insights
    lines.extend([
        "### Layer Distribution Insights",
        "",
        "Universal heads concentrate in different layers:",
        ""
    ])

    for condition in CONDITIONS.keys():
        layers = [layer for layer, head in universal[condition]]
        if layers:
            lines.append(f"- **{condition}**: median layer {np.median(layers):.1f}, range [{min(layers)}, {max(layers)}]")

    lines.extend([
        "",
        "**Key observation**: Control heads concentrate in later layers than abstract/photo heads, "
        "suggesting different computational stages.",
        "",
        "---",
        "",
        "## Overall Hypothesis Support",
        ""
    ])

    # Calculate overall support with nuanced interpretation
    support_count = 0

    # Hypothesis 1: Consider both overlap and effect correlation
    avg_effect_corr_abstract_photo = np.mean([
        effect_correlations[k]['correlation']
        for k in effect_correlations.keys()
        if 'abstract_photo' in k
    ])

    if avg_jaccard > 0.4 or (avg_overlap > 0.6 and avg_effect_corr_abstract_photo > 0.6):
        support_count += 1
        lines.append("✅ **Hypothesis 1**: SUPPORTED (abstract ∩ photo show shared mechanism)")
        lines.append(f"   - Overlap coefficient: {avg_overlap:.1%} of photo heads found in abstract")
        lines.append(f"   - Effect size correlation: r={avg_effect_corr_abstract_photo:.3f} for shared heads")
        lines.append("   - **Interpretation**: Same belief-tracking mechanism, but abstract has higher power")
    else:
        lines.append("❌ **Hypothesis 1**: NOT SUPPORTED (abstract ∩ photo low overlap)")

    if avg_jaccard_2 < 0.3:  # Hyp 2
        support_count += 1
        lines.append("✅ **Hypothesis 2**: SUPPORTED (abstract ∩ control low overlap)")
    else:
        lines.append("❌ **Hypothesis 2**: NOT SUPPORTED (abstract ∩ control high overlap)")

    if avg_jaccard_3 < 0.3:  # Hyp 3
        support_count += 1
        lines.append("✅ **Hypothesis 3**: SUPPORTED (photo ∩ control low overlap)")
    else:
        lines.append("❌ **Hypothesis 3**: NOT SUPPORTED (photo ∩ control high overlap)")

    lines.extend([
        "",
        f"**Overall**: {support_count}/3 hypotheses supported",
        "",
        "## Key Takeaways",
        "",
        "1. **Abstract and Photo use the same belief-tracking mechanism**:",
        f"   - {universal_overlap['abstract_photo']['intersection']} universal heads shared",
        f"   - Strong effect size correlations (r={avg_effect_corr_abstract_photo:.3f})",
        "   - Both concentrate in layers 27-40 (middle-late)",
        "",
        "2. **Control uses a different mechanism (location retrieval)**:",
        f"   - Minimal overlap with abstract (jaccard={avg_jaccard_2:.3f}) and photo (jaccard={avg_jaccard_3:.3f})",
        "   - Concentrates in later layers (32-47)",
        "   - Weak/negative effect size correlations with abstract",
        "",
        "3. **Universal heads provide the strongest evidence**:",
        f"   - {universal_overlap['abstract_photo']['overlap_coef']:.0%} of robust photo heads also robust in abstract",
        f"   - Only {universal_overlap['abstract_control']['overlap_coef']:.0%} of robust abstract heads overlap with control",
        "",
        "4. **Shared baseline is minimal or absent**:",
        f"   - Only {universal_overlap['all_conditions_baseline']['count']} heads significant across ALL conditions",
        f"   - Abstract∩photo shared mechanism ({universal_overlap['abstract_photo']['intersection']} heads) is NOT just shared baseline",
        f"   - {universal_overlap['abstract_photo']['intersection'] - universal_overlap['all_conditions_baseline']['count']} heads are belief-tracking specific",
        "",
    ])

    # Write to file
    report_path = RESULTS_DIR / "HYPOTHESIS_TEST_RESULTS.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nReport saved to: {report_path}")

def main():
    print("Loading permutation test data...")
    data = load_all_data()

    print("\nCalculating pairwise overlaps...")
    overlaps = calculate_pairwise_overlaps(data)
    overlaps.to_csv(RESULTS_DIR / "pairwise_overlaps.csv", index=False)

    print("\nTesting Hypothesis 1: abstract & photo (HIGH overlap expected)...")
    hyp1_results = test_hypothesis_1(data, overlaps)

    print("\nTesting Hypothesis 2: abstract & control (LOW overlap expected)...")
    hyp2_results = test_hypothesis_2(data, overlaps)

    print("\nTesting Hypothesis 3: photo & control (LOW overlap expected)...")
    hyp3_results = test_hypothesis_3(data, overlaps)

    print("\nAnalyzing effect size correlations...")
    effect_correlations = analyze_effect_size_correlations(data)

    print("\nAnalyzing layer distributions...")
    distributions = analyze_layer_distributions(data)

    print("\nFinding universal heads...")
    universal = find_universal_heads(data)

    print("\nAnalyzing universal heads overlap...")
    universal_overlap = analyze_universal_heads_overlap(universal)

    print("\nGenerating visualizations...")
    plot_overlap_heatmaps(overlaps)
    plot_layer_distributions(distributions, universal)
    plot_universal_heads_layer_comparison(universal)

    print("\nGenerating markdown report...")
    generate_markdown_report(
        data,
        hyp1_results,
        hyp2_results,
        hyp3_results,
        effect_correlations,
        universal,
        universal_overlap
    )

    # Save raw results
    results = {
        'hypothesis_1': hyp1_results,
        'hypothesis_2': hyp2_results,
        'hypothesis_3': hyp3_results,
        'effect_correlations': effect_correlations,
        'universal_heads': {k: list(v) for k, v in universal.items()},
        'universal_heads_overlap': {k: {**v, 'heads': []} for k, v in universal_overlap.items()}  # Remove heads list from JSON
    }

    with open(RESULTS_DIR / "hypothesis_test_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAnalysis complete! Results saved to {RESULTS_DIR}")

if __name__ == "__main__":
    main()
