#!/usr/bin/env python3
"""
Check if shared heads between abstract and control are in the same layers
or if they're actually different heads that happen to overlap by layer,head coordinates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
wandb_dir = Path("wandb_permtest")

# Food truck
abstract_ft = pd.read_csv(wandb_dir / "abstract" / "Qwen2.5-14B-Instruct_food_truck_z_ABA_abstract_healthy.csv")
control_ft = pd.read_csv(wandb_dir / "control" / "end_patch" / "Qwen2.5-14B-Instruct_food_truck_z_ABA_control_patch.csv")

# Get significant heads
abstract_sig = abstract_ft[abstract_ft['significant'] == True].copy()
control_sig = control_ft[control_ft['significant'] == True].copy()

# Merge on layer and head to get shared heads
shared = abstract_sig.merge(
    control_sig,
    on=['layer', 'head'],
    suffixes=('_abstract', '_control')
)

print(f"\n=== FOOD TRUCK SHARED HEADS ===")
print(f"Abstract significant: {len(abstract_sig)}")
print(f"Control significant: {len(control_sig)}")
print(f"Shared (both significant): {len(shared)}")
print(f"\nShared heads layer distribution:")
print(f"  Layers: {sorted(shared['layer'].unique())}")
print(f"  Layer range: [{shared['layer'].min()}, {shared['layer'].max()}]")
print(f"  Median layer: {shared['layer'].median()}")

print(f"\nEffect size analysis for shared heads:")
print(f"  Abstract mean effect: {shared['observed_score_abstract'].mean():.4f}")
print(f"  Control mean effect: {shared['observed_score_control'].mean():.4f}")
print(f"  Correlation: {np.corrcoef(shared['observed_score_abstract'], shared['observed_score_control'])[0,1]:.4f}")

print(f"\nEffect size by direction:")
positive_abstract = shared[shared['observed_score_abstract'] > 0]
print(f"  Heads with positive effect in abstract: {len(positive_abstract)}")
print(f"    Mean control effect for these: {positive_abstract['observed_score_control'].mean():.4f}")
print(f"    Proportion also positive in control: {(positive_abstract['observed_score_control'] > 0).mean():.1%}")

negative_abstract = shared[shared['observed_score_abstract'] < 0]
print(f"  Heads with negative effect in abstract: {len(negative_abstract)}")
print(f"    Mean control effect for these: {negative_abstract['observed_score_control'].mean():.4f}")

print(f"\nTop 5 shared heads by |abstract effect|:")
top_abstract = shared.nlargest(5, 'observed_score_abstract')[['layer', 'head', 'observed_score_abstract', 'observed_score_control']]
print(top_abstract.to_string(index=False))

print(f"\nTop 5 shared heads by |control effect|:")
top_control = shared.nlargest(5, 'observed_score_control')[['layer', 'head', 'observed_score_abstract', 'observed_score_control']]
print(top_control.to_string(index=False))

print(f"\n5 most discordant heads (opposite effects):")
shared['effect_diff'] = shared['observed_score_abstract'] - shared['observed_score_control']
discordant = shared.nlargest(5, 'effect_diff')[['layer', 'head', 'observed_score_abstract', 'observed_score_control', 'effect_diff']]
print(discordant.to_string(index=False))
