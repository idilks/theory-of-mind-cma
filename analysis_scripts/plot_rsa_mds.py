#!/usr/bin/env python3
"""
MDS scatter plot of CMA score profiles across conditions.
Shows representational similarity structure: which conditions
recruit similar attention heads?

Input: causal_scores.pt files (end-patch, all 5 conditions × 3 templates × 2 directions)
Output: results/rsa_mds_conditions.png, results/rsa_rdm_heatmap.png
"""

import torch
import os
import re
import numpy as np
from scipy.stats import pearsonr
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns

# --- Load data ---
BASE = os.path.join('results', 'causal_analysis', 'cma', 'Qwen', 'Qwen2.5-14B-Instruct')
OUT_DIR = 'results'

CONDS = ['abstract_belief', 'abstract_photo', 'answer_changes_belief',
         'answer_changes_photo', 'control']
TMPLS = ['food_truck', 'hair_styling', 'library_book']

# Map legacy folder names to canonical condition names
FOLDER_TO_COND = {
    'abstract_belief': 'abstract_belief',
    'abstract_photo': 'abstract_photo',
    'answer_changes_belief': 'answer_changes_belief',
    'answer_changes_photo': 'answer_changes_photo',
    'control': 'control',
    'photo_context': 'abstract_photo',
    'control_context': 'control',
}

scores = {}
for root, dirs, files in os.walk(BASE):
    if 'causal_scores.pt' not in files:
        continue
    if 'patch_before_movement' not in root or '_ohwell' in root:
        continue
    rel = os.path.relpath(root, BASE).replace('\\', '/')
    parts = rel.split('/')
    folder = parts[0]
    condition = FOLDER_TO_COND.get(folder)
    template = parts[1].replace('template_', '')
    if condition is None or condition not in CONDS or template not in TMPLS:
        continue
    m = re.match(r'base_rule_(\w+)_exp_rule_(\w+)', parts[2])
    if not m:
        continue
    t = torch.load(
        os.path.join(root, 'causal_scores.pt'),
        map_location='cpu', weights_only=True
    ).flatten().numpy()
    key = (condition, template)
    if key not in scores:
        scores[key] = []
    scores[key].append(t)

print(f"Loaded {len(scores)} condition×template pairs")

# Average over both directions
pooled = {k: np.mean(v, axis=0) for k, v in scores.items()}

labels = []
vectors = []
for c in CONDS:
    for t in TMPLS:
        labels.append((c, t))
        vectors.append(pooled[(c, t)])
vectors = np.array(vectors)  # (15, 1920)
print(f"Matrix shape: {vectors.shape}")

# --- Dissimilarity matrix (1 - Pearson r) ---
n = len(vectors)
diss = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        r, _ = pearsonr(vectors[i], vectors[j])
        diss[i, j] = 1 - r

# --- MDS ---
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
          normalized_stress='auto')
coords = mds.fit_transform(diss)
stress = mds.stress_
print(f"MDS stress: {stress:.4f}")

# --- Visual config ---
COND_SHORT = {
    'abstract_belief': 'Abstract\nBelief',
    'abstract_photo': 'Abstract\nPhoto',
    'answer_changes_belief': 'Ans. Change\nBelief',
    'answer_changes_photo': 'Ans. Change\nPhoto',
    'control': 'Control'
}
COND_COLORS = {
    'abstract_belief': '#2166ac',
    'abstract_photo': '#67a9cf',
    'answer_changes_belief': '#d6604d',
    'answer_changes_photo': '#f4a582',
    'control': '#4daf4a'
}
TMPL_MARKERS = {'food_truck': 'o', 'hair_styling': 's', 'library_book': '^'}
TMPL_LABELS = {'food_truck': 'Food Truck', 'hair_styling': 'Hair Styling',
               'library_book': 'Library Book'}

# ===== PLOT 1: MDS Scatter =====
fig, ax = plt.subplots(1, 1, figsize=(9, 7))

# Convex hulls
for c in CONDS:
    idxs = [i for i, (cond, _) in enumerate(labels) if cond == c]
    pts = coords[idxs]
    if len(pts) >= 3:
        hull = ConvexHull(pts)
        hull_pts = np.append(hull.vertices, hull.vertices[0])
        ax.fill(pts[hull_pts, 0], pts[hull_pts, 1],
                alpha=0.12, color=COND_COLORS[c], zorder=1)
        ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                color=COND_COLORS[c], alpha=0.4, linewidth=1.5,
                linestyle='--', zorder=2)

# Points
for i, (c, t) in enumerate(labels):
    ax.scatter(coords[i, 0], coords[i, 1],
               c=COND_COLORS[c], marker=TMPL_MARKERS[t],
               s=140, edgecolors='white', linewidth=0.8, zorder=3)

# Centroid labels with smart offsets to avoid overlap
centroids = {}
for c in CONDS:
    idxs = [i for i, (cond, _) in enumerate(labels) if cond == c]
    centroids[c] = coords[idxs].mean(axis=0)

# Compute offsets: push labels away from the global centroid
global_centroid = np.mean(list(centroids.values()), axis=0)
for c in CONDS:
    centroid = centroids[c]
    direction = centroid - global_centroid
    norm = np.linalg.norm(direction)
    if norm > 0:
        offset = direction / norm * 0.02  # small nudge outward
    else:
        offset = np.array([0, 0.02])

    ax.annotate(
        COND_SHORT[c], centroid + offset,
        fontsize=9, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=COND_COLORS[c],
                  alpha=0.3, edgecolor='none'),
        zorder=4
    )

# Legends
color_handles = [
    mpatches.Patch(color=COND_COLORS[c],
                   label=COND_SHORT[c].replace('\n', ' '), alpha=0.7)
    for c in CONDS
]
shape_handles = [
    Line2D([0], [0], marker=TMPL_MARKERS[t], color='gray',
           linestyle='None', markersize=8, label=TMPL_LABELS[t])
    for t in TMPLS
]

leg1 = ax.legend(handles=color_handles, title='Condition', loc='upper left',
                 fontsize=8, title_fontsize=9, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(handles=shape_handles, title='Template', loc='lower left',
          fontsize=8, title_fontsize=9, framealpha=0.9)

ax.set_xlabel('MDS Dimension 1', fontsize=11)
ax.set_ylabel('MDS Dimension 2', fontsize=11)
ax.set_title(
    'Representational Similarity of CMA Profiles Across Conditions\n'
    '(End-Patch Position, Qwen2.5-14B-Instruct)',
    fontsize=12, fontweight='bold'
)
ax.text(0.98, 0.02, f'Stress = {stress:.4f}',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=9, fontstyle='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax.grid(True, alpha=0.2)
ax.set_aspect('equal')
plt.tight_layout()

mds_path = os.path.join(OUT_DIR, 'rsa_mds_conditions.png')
plt.savefig(mds_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved MDS plot: {mds_path}")

# ===== PLOT 2: RDM Heatmap (5x5, averaged across templates) =====
# Build condition-level dissimilarity (average across templates)
cond_vectors = {}
for c in CONDS:
    vecs = [pooled[(c, t)] for t in TMPLS]
    cond_vectors[c] = np.mean(vecs, axis=0)

cond_labels_short = ['Abstract\nBelief', 'Abstract\nPhoto',
                     'Ans.Change\nBelief', 'Ans.Change\nPhoto', 'Control']

rdm = np.zeros((5, 5))
for i, c1 in enumerate(CONDS):
    for j, c2 in enumerate(CONDS):
        r, _ = pearsonr(cond_vectors[c1], cond_vectors[c2])
        rdm[i, j] = 1 - r

# ===== PERMUTATION TESTS =====
N_PERM = 10000
rng = np.random.default_rng(42)
n_heads = len(cond_vectors[CONDS[0]])

# --- Cell-wise permutation test ---
# For each pair: permute head indices of one vector, recompute 1-r
# H0: the head-level correspondence between conditions is no better than chance
# p-value: fraction of permuted dissimilarities <= observed (i.e., as similar or more)
print(f"\n--- Cell-wise permutation test ({N_PERM} permutations) ---")
pval_matrix = np.ones((5, 5))
for i, c1 in enumerate(CONDS):
    for j, c2 in enumerate(CONDS):
        if i == j:
            pval_matrix[i, j] = np.nan
            continue
        if i > j:
            pval_matrix[i, j] = pval_matrix[j, i]
            continue
        v1 = cond_vectors[c1]
        v2 = cond_vectors[c2]
        observed = rdm[i, j]
        # Count how many permuted dissimilarities are <= observed
        # (i.e., produce as much or more similarity by chance)
        count = 0
        for _ in range(N_PERM):
            v2_perm = rng.permutation(v2)
            r_perm, _ = pearsonr(v1, v2_perm)
            if (1 - r_perm) <= observed:
                count += 1
        pval = count / N_PERM
        pval_matrix[i, j] = pval
        pval_matrix[j, i] = pval

print("\nCell-wise p-values (H0: head correspondence is random):")
print(f"{'':>25s}", end='')
for c in CONDS:
    print(f"{c:>20s}", end='')
print()
for i, c1 in enumerate(CONDS):
    print(f"{c1:>25s}", end='')
    for j, c2 in enumerate(CONDS):
        if i == j:
            print(f"{'---':>20s}", end='')
        else:
            p = pval_matrix[i, j]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"{p:>14.4f} {stars:>4s}", end='')
    print()

# --- Mantel tests on 15x15 RDM (condition × template level) ---
# With 15 items we have 105 upper-triangle values and enormous permutation space,
# giving the test real statistical power.
print(f"\n--- Mantel tests on 15x15 RDM ({N_PERM} permutations) ---")

upper_idx_15 = np.triu_indices(15, k=1)
obs_upper_15 = diss[upper_idx_15]  # diss is the 15x15 matrix computed earlier

# Condition label for each of the 15 items (order: 5 conds × 3 templates)
cond_per_item = [c for c, t in labels]

# --- Model 1: Belief vs non-belief ---
# belief group: abstract_belief, answer_changes_belief = 0; rest = 1
belief_map = {'abstract_belief': 0, 'answer_changes_belief': 0,
              'abstract_photo': 1, 'answer_changes_photo': 1, 'control': 1}
belief_labels = [belief_map[c] for c in cond_per_item]
model_belief_15 = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        model_belief_15[i, j] = 0.0 if belief_labels[i] == belief_labels[j] else 1.0

obs_r_belief, _ = pearsonr(obs_upper_15, model_belief_15[upper_idx_15])

belief_null = np.zeros(N_PERM)
for p_idx in range(N_PERM):
    perm = rng.permutation(15)
    perm_model = model_belief_15[np.ix_(perm, perm)]
    r_perm, _ = pearsonr(obs_upper_15, perm_model[upper_idx_15])
    belief_null[p_idx] = r_perm

belief_p = np.mean(belief_null >= obs_r_belief)
print(f"\nModel 1 — Belief vs non-belief:")
print(f"  Observed r = {obs_r_belief:.4f}")
print(f"  Null: mean={np.mean(belief_null):.4f}, std={np.std(belief_null):.4f}")
print(f"  p = {belief_p:.4f} {'***' if belief_p < 0.001 else '**' if belief_p < 0.01 else '*' if belief_p < 0.05 else 'ns'}")

# --- Model 2: Task type (abstract vs answer_changes vs control) ---
task_map = {'abstract_belief': 0, 'abstract_photo': 0,
            'answer_changes_belief': 1, 'answer_changes_photo': 1, 'control': 2}
task_labels = [task_map[c] for c in cond_per_item]
model_task_15 = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        model_task_15[i, j] = 0.0 if task_labels[i] == task_labels[j] else 1.0

obs_r_task, _ = pearsonr(obs_upper_15, model_task_15[upper_idx_15])

task_null = np.zeros(N_PERM)
for p_idx in range(N_PERM):
    perm = rng.permutation(15)
    perm_model = model_task_15[np.ix_(perm, perm)]
    r_perm, _ = pearsonr(obs_upper_15, perm_model[upper_idx_15])
    task_null[p_idx] = r_perm

task_p = np.mean(task_null >= obs_r_task)
print(f"\nModel 2 — Abstract vs answer_changes vs control:")
print(f"  Observed r = {obs_r_task:.4f}")
print(f"  Null: mean={np.mean(task_null):.4f}, std={np.std(task_null):.4f}")
print(f"  p = {task_p:.4f} {'***' if task_p < 0.001 else '**' if task_p < 0.01 else '*' if task_p < 0.05 else 'ns'}")

# --- Model 3: Same condition (within-condition similarity) ---
# Tests whether items from the same condition are more similar than across conditions
model_same_15 = np.zeros((15, 15))
for i in range(15):
    for j in range(15):
        model_same_15[i, j] = 0.0 if cond_per_item[i] == cond_per_item[j] else 1.0

obs_r_same, _ = pearsonr(obs_upper_15, model_same_15[upper_idx_15])

same_null = np.zeros(N_PERM)
for p_idx in range(N_PERM):
    perm = rng.permutation(15)
    perm_model = model_same_15[np.ix_(perm, perm)]
    r_perm, _ = pearsonr(obs_upper_15, perm_model[upper_idx_15])
    same_null[p_idx] = r_perm

same_p = np.mean(same_null >= obs_r_same)
print(f"\nModel 3 — Same condition clusters together:")
print(f"  Observed r = {obs_r_same:.4f}")
print(f"  Null: mean={np.mean(same_null):.4f}, std={np.std(same_null):.4f}")
print(f"  p = {same_p:.4f} {'***' if same_p < 0.001 else '**' if same_p < 0.01 else '*' if same_p < 0.05 else 'ns'}")

# Store for plot titles
obs_mantel_belief = obs_r_belief
mantel_belief_p = belief_p
obs_mantel_task = obs_r_task
mantel_task_p = task_p
obs_mantel_same = obs_r_same
mantel_same_p = same_p

# ===== PLOT 2: RDM Heatmap with significance =====

# Build annotation strings: dissimilarity + stars
annot_strings = np.empty((5, 5), dtype=object)
for i in range(5):
    for j in range(5):
        if i == j:
            annot_strings[i, j] = f"{rdm[i, j]:.3f}"
        else:
            p = pval_matrix[i, j]
            stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            annot_strings[i, j] = f"{rdm[i, j]:.3f}{stars}"

fig, ax = plt.subplots(1, 1, figsize=(7, 6))
sns.heatmap(
    rdm, annot=annot_strings, fmt='', cmap='RdYlBu_r',
    xticklabels=cond_labels_short, yticklabels=cond_labels_short,
    vmin=0, vmax=1, ax=ax, linewidths=0.5,
    cbar_kws={'label': '1 − Pearson r (dissimilarity)'}
)

# Add Mantel test results to title
def fmt_p(p):
    return '<.001' if p < 0.001 else f'{p:.3f}'

mantel_lines = [
    f"15×15 Mantel: same-cond r={obs_mantel_same:.2f} p={fmt_p(mantel_same_p)}, "
    f"belief r={obs_mantel_belief:.2f} p={fmt_p(mantel_belief_p)}, "
    f"task r={obs_mantel_task:.2f} p={fmt_p(mantel_task_p)}"
]
ax.set_title(
    'Representational Dissimilarity Matrix (CMA Scores)\n'
    f'End-Patch, Template-Averaged\n'
    f'{mantel_lines[0]}\n'
    '* p<.05, ** p<.01, *** p<.001 (cell-wise permutation, 10k perms)',
    fontsize=9, fontweight='bold'
)
plt.tight_layout()

rdm_path = os.path.join(OUT_DIR, 'rsa_rdm_heatmap.png')
plt.savefig(rdm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"\nSaved RDM heatmap: {rdm_path}")

print("\nDone!")
