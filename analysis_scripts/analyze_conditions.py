#!/usr/bin/env python3
"""
cross-condition CMA analysis.

compares significant heads and causal score distributions between the 10 conditions
(5 prompt-pair types × 2 patch positions) to test hypotheses H1-H5.

no CLI args — paths are fixed. output to results/condition_analysis/.
"""

import re
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, fisher_exact

# ── paths ──
ROOT = Path("results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct")
OUT = Path("results/condition_analysis")
OUT.mkdir(parents=True, exist_ok=True)

TEMPLATES = ["food_truck", "hair_styling", "library_book"]

# condition label → (folder_name, base_rule, exp_rule, patch_position)
# patch_after_movement = belief_formation (bf), patch_before_movement = end
CONDITIONS = {
    "abstract_belief_bf":        ("abstract_belief",       "ABA", "ABB", "patch_after_movement"),
    "abstract_belief_end":       ("abstract_belief",       "ABA", "ABB", "patch_before_movement"),
    "abstract_photo_bf":         ("abstract_photo",        "ABA", "ABA", "patch_after_movement"),
    "abstract_photo_end":        ("abstract_photo",        "ABA", "ABA", "patch_before_movement"),
    "answer_changes_belief_bf":  ("answer_changes_belief", "ABA", "ABB", "patch_after_movement"),
    "answer_changes_belief_end": ("answer_changes_belief", "ABA", "ABB", "patch_before_movement"),
    "answer_changes_photo_bf":   ("answer_changes_photo",  "ABA", "ABA", "patch_after_movement"),
    "answer_changes_photo_end":  ("answer_changes_photo",  "ABA", "ABA", "patch_before_movement"),
    "control_bf":                ("control",               "ABA", "ABA", "patch_after_movement"),
    "control_end":               ("control",               "ABA", "ABA", "patch_before_movement"),
}

# reverse directions for replication check
REVERSE_CONDITIONS = {
    "abstract_belief_bf":        ("abstract_belief",       "ABB", "ABA", "patch_after_movement"),
    "abstract_belief_end":       ("abstract_belief",       "ABB", "ABA", "patch_before_movement"),
    "abstract_photo_bf":         ("abstract_photo",        "ABB", "ABB", "patch_after_movement"),
    "abstract_photo_end":        ("abstract_photo",        "ABB", "ABB", "patch_before_movement"),
    "answer_changes_belief_bf":  ("answer_changes_belief", "ABB", "ABA", "patch_after_movement"),
    "answer_changes_belief_end": ("answer_changes_belief", "ABB", "ABA", "patch_before_movement"),
    "answer_changes_photo_bf":   ("answer_changes_photo",  "ABB", "ABB", "patch_after_movement"),
    "answer_changes_photo_end":  ("answer_changes_photo",  "ABB", "ABB", "patch_before_movement"),
    "control_bf":                ("control",               "ABB", "ABB", "patch_after_movement"),
    "control_end":               ("control",               "ABB", "ABB", "patch_before_movement"),
}

COND_LABELS = list(CONDITIONS.keys())


# ── data loading ──

def load_csv(path: Path) -> pd.DataFrame:
    """load and clean a statistical_test_results.csv."""
    df = pd.read_csv(path, dtype={"significant": str})
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip().str.strip('"')
    df["significant"] = df["significant"].str.lower().map({"true": True, "false": False})
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce").astype(int)
    df["head"] = pd.to_numeric(df["head"], errors="coerce").astype(int)
    df["observed_score"] = pd.to_numeric(df["observed_score"], errors="coerce")
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df


def find_best_csv(context, template, base_rule, exp_rule, patch_pos) -> Path | None:
    """find the CSV with the highest sample_num for a given condition + template."""
    base_dir = ROOT / context / f"template_{template}" / f"base_rule_{base_rule}_exp_rule_{exp_rule}" / patch_pos / "z_seed_0_shuffle_False" / "logit"
    if not base_dir.exists():
        return None
    # find all sample_num_* folders
    candidates = []
    for d in base_dir.iterdir():
        m = re.match(r"sample_num_(\d+)", d.name)
        if m and (d / "statistical_test_results.csv").exists():
            candidates.append((int(m.group(1)), d / "statistical_test_results.csv"))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_all_data(condition_map):
    """returns {cond_label: {template: df}} for all conditions and templates."""
    data = {}
    for cond, (context, base, exp, patch) in condition_map.items():
        data[cond] = {}
        for tmpl in TEMPLATES:
            path = find_best_csv(context, tmpl, base, exp, patch)
            if path is not None:
                data[cond][tmpl] = load_csv(path)
    return data


# ── analysis functions ──

def get_score_vector(df: pd.DataFrame) -> np.ndarray:
    """return observed_score as flat array ordered by (layer, head)."""
    df_sorted = df.sort_values(["layer", "head"])
    return df_sorted["observed_score"].values


def get_sig_set(df: pd.DataFrame) -> set:
    """return set of (layer, head) tuples that are significant."""
    sig = df[df["significant"] == True]
    return set(zip(sig["layer"].astype(int), sig["head"].astype(int)))


def jaccard(a: set, b: set) -> float:
    if len(a | b) == 0:
        return 0.0
    return len(a & b) / len(a | b)


def fisher_overlap(sig_a: set, sig_b: set, total: int):
    """2×2 fisher exact test for above-chance overlap."""
    both = len(sig_a & sig_b)
    a_only = len(sig_a - sig_b)
    b_only = len(sig_b - sig_a)
    neither = total - both - a_only - b_only
    table = [[both, a_only], [b_only, neither]]
    odds, pval = fisher_exact(table, alternative="greater")
    return odds, pval


def compute_spearman_matrix(data, template):
    """6×6 spearman correlation matrix for one template."""
    n = len(COND_LABELS)
    mat = np.full((n, n), np.nan)
    for i, ci in enumerate(COND_LABELS):
        for j, cj in enumerate(COND_LABELS):
            if template in data[ci] and template in data[cj]:
                vi = get_score_vector(data[ci][template])
                vj = get_score_vector(data[cj][template])
                if len(vi) == len(vj):
                    rho, _ = spearmanr(vi, vj)
                    mat[i, j] = rho
    return mat


def compute_jaccard_matrix(data, template):
    """6×6 jaccard similarity matrix for one template."""
    n = len(COND_LABELS)
    mat = np.full((n, n), np.nan)
    for i, ci in enumerate(COND_LABELS):
        for j, cj in enumerate(COND_LABELS):
            if template in data[ci] and template in data[cj]:
                si = get_sig_set(data[ci][template])
                sj = get_sig_set(data[cj][template])
                mat[i, j] = jaccard(si, sj)
    return mat


def compute_fisher_results(data, template):
    """fisher exact test for all pairs, one template."""
    rows = []
    # need total heads count from any df
    any_df = None
    for cond in COND_LABELS:
        if template in data[cond]:
            any_df = data[cond][template]
            break
    if any_df is None:
        return pd.DataFrame()
    total = len(any_df)

    for ci, cj in combinations(COND_LABELS, 2):
        if template in data[ci] and template in data[cj]:
            si = get_sig_set(data[ci][template])
            sj = get_sig_set(data[cj][template])
            odds, pval = fisher_overlap(si, sj, total)
            rows.append({
                "template": template,
                "cond_a": ci, "cond_b": cj,
                "sig_a": len(si), "sig_b": len(sj),
                "overlap": len(si & sj),
                "jaccard": jaccard(si, sj),
                "odds_ratio": odds, "p_value": pval,
            })
    return pd.DataFrame(rows)


def cross_template_consistency(data):
    """for each condition, find heads significant in ALL 3 templates."""
    rows = []
    for cond in COND_LABELS:
        sig_sets = []
        for tmpl in TEMPLATES:
            if tmpl in data[cond]:
                sig_sets.append(get_sig_set(data[cond][tmpl]))
        if len(sig_sets) == len(TEMPLATES):
            consistent = sig_sets[0]
            for s in sig_sets[1:]:
                consistent = consistent & s
            for layer, head in sorted(consistent):
                rows.append({"condition": cond, "layer": layer, "head": head})
    return pd.DataFrame(rows)


def direction_consistency(primary_data, reverse_data):
    """correlate primary vs reverse direction scores per condition per template."""
    rows = []
    for cond in COND_LABELS:
        for tmpl in TEMPLATES:
            if tmpl in primary_data[cond] and tmpl in reverse_data[cond]:
                v1 = get_score_vector(primary_data[cond][tmpl])
                v2 = get_score_vector(reverse_data[cond][tmpl])
                if len(v1) == len(v2):
                    rho, pval = spearmanr(v1, v2)
                    rows.append({
                        "condition": cond, "template": tmpl,
                        "spearman_rho": rho, "p_value": pval,
                    })
    return pd.DataFrame(rows)


# ── visualizations ──

def plot_spearman_heatmaps(matrices, avg_matrix):
    """4-panel spearman heatmap (3 templates + average)."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    panels = [(t, matrices[t]) for t in TEMPLATES] + [("average", avg_matrix)]
    for ax, (title, mat) in zip(axes, panels):
        sns.heatmap(mat, xticklabels=COND_LABELS, yticklabels=COND_LABELS,
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", ax=ax, square=True,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle("spearman correlation of causal scores between conditions", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "spearman_correlations.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_jaccard_heatmaps(matrices, avg_matrix):
    """4-panel jaccard heatmap."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    panels = [(t, matrices[t]) for t in TEMPLATES] + [("average", avg_matrix)]
    for ax, (title, mat) in zip(axes, panels):
        sns.heatmap(mat, xticklabels=COND_LABELS, yticklabels=COND_LABELS,
                    cmap="YlOrRd", vmin=0, vmax=1,
                    annot=True, fmt=".2f", ax=ax, square=True,
                    annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)

    plt.suptitle("jaccard similarity of significant head sets", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "jaccard_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_sig_head_counts(data):
    """grouped bar chart: conditions × templates."""
    counts = {}
    for cond in COND_LABELS:
        counts[cond] = {}
        for tmpl in TEMPLATES:
            if tmpl in data[cond]:
                counts[cond][tmpl] = len(get_sig_set(data[cond][tmpl]))
            else:
                counts[cond][tmpl] = 0

    df = pd.DataFrame(counts).T
    df.index.name = "condition"

    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("number of significant heads")
    ax.set_title("significant heads per condition × template")
    ax.legend(title="template")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / "significant_head_counts.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_layer_distributions(data):
    """subplots (one per condition), overlaid histograms per template."""
    n_conds = len(COND_LABELS)
    ncols = 5
    nrows = (n_conds + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(24, 5 * nrows))
    axes = axes.flatten()

    for ax, cond in zip(axes, COND_LABELS):
        for tmpl in TEMPLATES:
            if tmpl in data[cond]:
                sig = data[cond][tmpl][data[cond][tmpl]["significant"] == True]
                if len(sig) > 0:
                    layers = sig["layer"].values
                    n_layers = data[cond][tmpl]["layer"].max() + 1
                    ax.hist(layers, bins=range(n_layers + 1), alpha=0.5,
                            label=tmpl, edgecolor="black", linewidth=0.3)
        ax.set_title(cond)
        ax.set_xlabel("layer")
        ax.set_ylabel("count")
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    # hide unused axes
    for ax in axes[n_conds:]:
        ax.set_visible(False)

    plt.suptitle("layer distribution of significant heads", y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / "layer_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── main ──

def main():
    print("loading primary direction data...")
    data = load_all_data(CONDITIONS)

    # verify
    found = 0
    for cond in COND_LABELS:
        for tmpl in TEMPLATES:
            if tmpl in data[cond]:
                n = len(data[cond][tmpl])
                n_sig = len(get_sig_set(data[cond][tmpl]))
                print(f"  {cond:15s} | {tmpl:15s} | {n:4d} heads | {n_sig:3d} significant")
                found += 1
            else:
                print(f"  {cond:15s} | {tmpl:15s} | MISSING")
    total_expected = len(COND_LABELS) * len(TEMPLATES)
    print(f"\nfound {found}/{total_expected} condition×template combinations\n")

    # 1. spearman correlations
    print("computing spearman correlations...")
    spearman_matrices = {}
    for tmpl in TEMPLATES:
        mat = compute_spearman_matrix(data, tmpl)
        spearman_matrices[tmpl] = mat
        pd.DataFrame(mat, index=COND_LABELS, columns=COND_LABELS).to_csv(
            OUT / f"spearman_{tmpl}.csv")

    # average across templates (nanmean)
    all_mats = np.stack([spearman_matrices[t] for t in TEMPLATES])
    avg_spearman = np.nanmean(all_mats, axis=0)
    pd.DataFrame(avg_spearman, index=COND_LABELS, columns=COND_LABELS).to_csv(
        OUT / "spearman_average.csv")

    plot_spearman_heatmaps(spearman_matrices, avg_spearman)

    # 2. jaccard similarity
    print("computing jaccard similarity...")
    jaccard_matrices = {}
    for tmpl in TEMPLATES:
        mat = compute_jaccard_matrix(data, tmpl)
        jaccard_matrices[tmpl] = mat
        pd.DataFrame(mat, index=COND_LABELS, columns=COND_LABELS).to_csv(
            OUT / f"jaccard_{tmpl}.csv")

    all_jac = np.stack([jaccard_matrices[t] for t in TEMPLATES])
    avg_jaccard = np.nanmean(all_jac, axis=0)
    pd.DataFrame(avg_jaccard, index=COND_LABELS, columns=COND_LABELS).to_csv(
        OUT / "jaccard_average.csv")

    plot_jaccard_heatmaps(jaccard_matrices, avg_jaccard)

    # 3. fisher exact tests
    print("computing fisher exact tests...")
    fisher_dfs = []
    for tmpl in TEMPLATES:
        fisher_dfs.append(compute_fisher_results(data, tmpl))
    fisher_all = pd.concat(fisher_dfs, ignore_index=True)
    fisher_all.to_csv(OUT / "fisher_exact_results.csv", index=False)

    # 4. cross-template consistency (H5)
    print("computing cross-template consistent heads...")
    consistent = cross_template_consistency(data)
    consistent.to_csv(OUT / "cross_template_consistent_heads.csv", index=False)

    print("\ncross-template consistent heads per condition:")
    for cond in COND_LABELS:
        subset = consistent[consistent["condition"] == cond]
        print(f"  {cond:15s}: {len(subset):3d} heads")
        if len(subset) > 0 and len(subset) <= 20:
            heads_str = ", ".join(f"L{r['layer']}H{r['head']}" for _, r in subset.iterrows())
            print(f"    {heads_str}")

    # 5. significant heads per template (for reference)
    for tmpl in TEMPLATES:
        rows = []
        for cond in COND_LABELS:
            if tmpl in data[cond]:
                for _, r in data[cond][tmpl].iterrows():
                    rows.append({
                        "condition": cond,
                        "layer": int(r["layer"]), "head": int(r["head"]),
                        "observed_score": r["observed_score"],
                        "p_value": r["p_value"],
                        "significant": r["significant"],
                    })
        pd.DataFrame(rows).to_csv(OUT / f"significant_heads_{tmpl}.csv", index=False)

    # 6. direction consistency
    print("\nloading reverse direction data...")
    reverse_data = load_all_data(REVERSE_CONDITIONS)
    dir_df = direction_consistency(data, reverse_data)
    dir_df.to_csv(OUT / "direction_consistency.csv", index=False)

    print("\ndirection consistency (primary vs reverse):")
    for _, r in dir_df.iterrows():
        print(f"  {r['condition']:15s} | {r['template']:15s} | rho={r['spearman_rho']:+.3f} | p={r['p_value']:.2e}")

    # remaining plots
    plot_sig_head_counts(data)
    plot_layer_distributions(data)

    # ── print hypothesis-relevant summaries ──
    print("\n" + "=" * 70)
    print("HYPOTHESIS TESTS SUMMARY")
    print("=" * 70)

    def _idx(name):
        return COND_LABELS.index(name)

    # H1: abstract_belief != control
    rho_abs_ctrl_bf = avg_spearman[_idx("abstract_belief_bf"), _idx("control_bf")]
    rho_abs_ctrl_end = avg_spearman[_idx("abstract_belief_end"), _idx("control_end")]
    print(f"\nH1 (abstract_belief != control):")
    print(f"  spearman abstract_belief_bf vs control_bf:   {rho_abs_ctrl_bf:+.3f}")
    print(f"  spearman abstract_belief_end vs control_end: {rho_abs_ctrl_end:+.3f}")
    jac_abs_ctrl_bf = avg_jaccard[_idx("abstract_belief_bf"), _idx("control_bf")]
    jac_abs_ctrl_end = avg_jaccard[_idx("abstract_belief_end"), _idx("control_end")]
    print(f"  jaccard abstract_belief_bf vs control_bf:    {jac_abs_ctrl_bf:.3f}")
    print(f"  jaccard abstract_belief_end vs control_end:  {jac_abs_ctrl_end:.3f}")

    # H2: abstract_belief ~ abstract_photo
    rho_abs_photo_bf = avg_spearman[_idx("abstract_belief_bf"), _idx("abstract_photo_bf")]
    rho_abs_photo_end = avg_spearman[_idx("abstract_belief_end"), _idx("abstract_photo_end")]
    print(f"\nH2 (abstract_belief ~ abstract_photo -> generic state tracking):")
    print(f"  spearman abstract_belief_bf vs abstract_photo_bf:   {rho_abs_photo_bf:+.3f}")
    print(f"  spearman abstract_belief_end vs abstract_photo_end: {rho_abs_photo_end:+.3f}")

    # H3: bf more specific than end
    n_abs_bf = sum(len(get_sig_set(data["abstract_belief_bf"][t])) for t in TEMPLATES if t in data["abstract_belief_bf"]) / max(1, sum(1 for t in TEMPLATES if t in data["abstract_belief_bf"]))
    n_abs_end = sum(len(get_sig_set(data["abstract_belief_end"][t])) for t in TEMPLATES if t in data["abstract_belief_end"]) / max(1, sum(1 for t in TEMPLATES if t in data["abstract_belief_end"]))
    print(f"\nH3 (bf more specific than end):")
    print(f"  avg significant heads abstract_belief_bf:  {n_abs_bf:.1f}")
    print(f"  avg significant heads abstract_belief_end: {n_abs_end:.1f}")

    # H4: control insensitive to patch position
    rho_ctrl_pos = avg_spearman[_idx("control_bf"), _idx("control_end")]
    jac_ctrl_pos = avg_jaccard[_idx("control_bf"), _idx("control_end")]
    print(f"\nH4 (control insensitive to patch position):")
    print(f"  spearman control_bf vs control_end:   {rho_ctrl_pos:+.3f}")
    print(f"  jaccard control_bf vs control_end:    {jac_ctrl_pos:.3f}")

    # H5: cross-template consistency
    print(f"\nH5 (cross-template consistency):")
    for cond in COND_LABELS:
        n = len(consistent[consistent["condition"] == cond])
        print(f"  {cond:15s}: {n:3d} heads consistent across all 3 templates")

    print(f"\nall outputs saved to: {OUT}/")
    print("done.")


if __name__ == "__main__":
    main()
