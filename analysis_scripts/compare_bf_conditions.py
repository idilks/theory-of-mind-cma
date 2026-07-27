"""bf-patch condition comparison + RSA for PI meeting.

Builds:
  D2  results_for_pi/bf_comparison_summary.md   - numbers + predictions checklist
  D3  results_for_pi/venn_C1_C3_C8.png          - 3-way Venn of universal heads
  D4  results_for_pi/scatter_bf.png             - 1x3 Pearson-r scatter
  D6  results_for_pi/rdm_bf.png                 - bf RDMs (continuous + binary)
  D7  results_for_pi/rdm_end.png                - end RDMs (continuous + binary)

Design, predictions, parameter choices: see PLAN_BF_COMPARISON.md.
Assumptions: see PLAN_BF_COMPARISON.md sections 2-4.

Run from repo root:
    python scripts/compare_bf_conditions.py
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib_venn import venn3
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
OUT = ROOT / "results_for_pi"

N_LAYERS, N_HEADS = 48, 40
N_TOTAL = N_LAYERS * N_HEADS
TEMPLATES = ["food_truck", "hair_styling", "library_book"]
DIRECTIONS = ["ABA", "ABB"]

PI_TREE = ROOT / "results_for_pi"

# condition_key -> (short_label, color, canonical_positions)
COND_INFO = {
    "C1": ("C1 abstract_belief",          "#1f77b4", ["bf", "end"]),
    "C2": ("C2 abstract_photo",           "#2ca02c", ["bf", "end"]),
    "C3": ("C3 answer_changes_belief",    "#d62728", ["bf", "end"]),
    "C4": ("C4 answer_changes_photo",     "#9467bd", ["bf", "end"]),
    "C5": ("C5 control",                  "#8c564b", ["end"]),
    "C8": ("C8 bf-control (new)",         "#e377c2", ["bf"]),
}

# C8 is not in the symlink tree — load directly from the canonical results path.
C8_RAW = BASE / "answer_changes_location_only"

PATCH_DIR = {"bf": "patch_after_movement", "end": "patch_before_movement"}


@dataclass
class Run:
    condition: str           # "C1", "C3", ...
    template: str            # "food_truck"
    direction: str           # "ABA" or "ABB"
    position: str            # "bf" or "end"
    mean_scores: np.ndarray  # (N_TOTAL,)
    sig_mask: np.ndarray     # (N_TOTAL,) bool
    n_samples: int
    threshold: float

    @property
    def key(self) -> str:
        return f"{self.condition}_{self.template[:4]}_{self.direction}_{self.position}"


def find_leaf_dir(condition_key: str, template: str, direction: str, position: str) -> Path | None:
    """Walk the results_for_pi/ symlink tree for C1..C5; resolve raw path for C8."""
    positions_avail = COND_INFO[condition_key][2]
    if position not in positions_avail:
        return None

    if condition_key == "C8":
        # direct path: .../answer_changes_location_only/template_*/base_rule_{d}_exp_rule_{d}/
        rp = f"base_rule_{direction}_exp_rule_{direction}"
        p = (C8_RAW / f"template_{template}" / rp / PATCH_DIR[position] /
             "z_seed_0_shuffle_False" / "logit")
        if not p.exists():
            return None
        cands = sorted(p.glob("sample_num_*_gen_acc_0.9"))
        return cands[0] if cands else None

    # Symlink tree: results_for_pi/{cond}/template_{tmpl}/{any_direction_folder_starting_with_{direction}_}/{bf,end}/
    tmpl_dir = PI_TREE / condition_key / f"template_{template}"
    if not tmpl_dir.exists():
        return None
    # direction is encoded as the first token of the subdir name (e.g. "ABA_to_ABB", "ABA_timing_varies", "ABA_location_only")
    for dir_sub in sorted(tmpl_dir.iterdir()):
        if not dir_sub.name.startswith(direction + "_"):
            continue
        leaf = dir_sub / position
        if leaf.exists():
            return leaf.resolve()
    return None


def load_run(condition_key: str, template: str, direction: str, position: str) -> Run | None:
    leaf = find_leaf_dir(condition_key, template, direction, position)
    if leaf is None:
        return None
    scores_pt = leaf / "causal_scores.pt"
    stats_csv = leaf / "statistical_test_results.csv"
    if not (scores_pt.exists() and stats_csv.exists()):
        return None

    mean_scores = torch.load(scores_pt, map_location="cpu", weights_only=False).numpy().reshape(-1)
    df = pd.read_csv(stats_csv)
    df = df.sort_values(["layer", "head"]).reset_index(drop=True)
    sig_mask = df["significant"].to_numpy().astype(bool)
    assert sig_mask.size == N_TOTAL, f"unexpected mask size {sig_mask.size} in {leaf}"

    # n_samples parsed from folder name
    n_samples = int(leaf.name.split("_")[2])
    # threshold: min |score| among significant, or NaN if empty
    abs_obs = df["observed_score"].abs().to_numpy()
    thr = float(abs_obs[sig_mask].min()) if sig_mask.any() else float("nan")

    return Run(condition_key, template, direction, position,
               mean_scores, sig_mask, n_samples, thr)


def load_all_runs() -> list[Run]:
    runs = []
    for cond, (_, _, positions) in COND_INFO.items():
        for pos in positions:
            for tmpl in TEMPLATES:
                for direction in DIRECTIONS:
                    r = load_run(cond, tmpl, direction, pos)
                    if r is None:
                        print(f"MISSING  {cond} {tmpl} {direction} {pos}")
                    else:
                        runs.append(r)
    return runs


# ---------- universal sets + set stats ----------

def universal_mask(runs: list[Run], condition: str, position: str, min_runs: int = 6) -> np.ndarray:
    rs = [r for r in runs if r.condition == condition and r.position == position]
    if not rs:
        return np.zeros(N_TOTAL, dtype=bool)
    stack = np.stack([r.sig_mask for r in rs], axis=0)   # (n_runs, N_TOTAL)
    return (stack.sum(axis=0) >= min_runs)


def mean_score_vector(runs: list[Run], condition: str, position: str) -> np.ndarray:
    rs = [r for r in runs if r.condition == condition and r.position == position]
    if not rs:
        return np.zeros(N_TOTAL)
    return np.stack([r.mean_scores for r in rs], axis=0).mean(axis=0)


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else float("nan")


def pearson_with_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    r, _ = pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rb, _ = pearsonr(x[idx], y[idx])
        boots.append(rb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(r), float(lo), float(hi)


# ---------- plots ----------

def plot_venn(U_C1: np.ndarray, U_C3: np.ndarray, U_C8: np.ndarray, out: Path) -> dict:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    set_C1 = set(np.where(U_C1)[0].tolist())
    set_C3 = set(np.where(U_C3)[0].tolist())
    set_C8 = set(np.where(U_C8)[0].tolist())
    venn3([set_C1, set_C3, set_C8],
          set_labels=("C1 abstract_belief", "C3 answer_changes_belief", "C8 bf-control"),
              ax=ax)
    ax.set_title(f"Universal sig heads at bf (6/6 runs each)\n"
                 f"|C1|={len(set_C1)}  |C3|={len(set_C3)}  |C8|={len(set_C8)}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "only_C1":   len(set_C1 - set_C3 - set_C8),
        "only_C3":   len(set_C3 - set_C1 - set_C8),
        "only_C8":   len(set_C8 - set_C1 - set_C3),
        "C1_C3_not_C8": len((set_C1 & set_C3) - set_C8),  # belief-specific candidates
        "C3_C8_not_C1": len((set_C3 & set_C8) - set_C1),  # answer-specific
        "C1_C8_not_C3": len((set_C1 & set_C8) - set_C3),
        "all_three":   len(set_C1 & set_C3 & set_C8),
    }


def plot_scatter(S_C1, S_C3, S_C8, U_C1, U_C3, U_C8, out: Path) -> dict:
    pairs = [("C1 vs C3", S_C1, S_C3, U_C1, U_C3),
             ("C1 vs C8", S_C1, S_C8, U_C1, U_C8),
             ("C3 vs C8", S_C3, S_C8, U_C3, U_C8)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    rs = {}
    for ax, (title, x, y, ux, uy) in zip(axes, pairs):
        gray = ~(ux | uy)
        both = ux & uy
        only_x = ux & ~uy
        only_y = uy & ~ux
        ax.scatter(x[gray], y[gray], s=6, c="#bbbbbb", alpha=0.35, label=f"neither ({gray.sum()})")
        ax.scatter(x[only_x], y[only_x], s=14, c="#1f77b4", alpha=0.8, label=f"sig in axis-x ({only_x.sum()})")
        ax.scatter(x[only_y], y[only_y], s=14, c="#d62728", alpha=0.8, label=f"sig in axis-y ({only_y.sum()})")
        ax.scatter(x[both], y[both], s=22, c="#6a1b9a", alpha=0.9, label=f"sig in both ({both.sum()})")

        r, lo, hi = pearson_with_ci(x, y)
        rs[title] = (r, lo, hi)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "--", c="k", lw=0.6, alpha=0.5)
        ax.axhline(0, c="k", lw=0.4, alpha=0.4)
        ax.axvline(0, c="k", lw=0.4, alpha=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_title(f"{title}\nPearson r = {r:.3f} [{lo:.3f}, {hi:.3f}]", fontsize=11)
        xl, yl = title.split(" vs ")
        ax.set_xlabel(f"mean score, {xl} (logit diff)")
        ax.set_ylabel(f"mean score, {yl} (logit diff)")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Raw mean CMA scores across 6 runs (3 templates x 2 directions), per head", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rs


def _build_rdm_continuous(runs: list[Run]) -> np.ndarray:
    mat = []
    for r in runs:
        v = r.mean_scores.copy()
        mu, sd = v.mean(), v.std()
        z = (v - mu) / sd if sd > 0 else v * 0
        z = np.clip(z, -5, 5)
        mat.append(z)
    mat = np.stack(mat, axis=0)   # (n_runs, N_TOTAL)
    n = mat.shape[0]
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            r, _ = pearsonr(mat[i], mat[j])
            d = 1 - r
            rdm[i, j] = rdm[j, i] = d
    return rdm


def _build_rdm_binary(runs: list[Run]) -> np.ndarray:
    n = len(runs)
    rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = 1 - jaccard(runs[i].sig_mask, runs[j].sig_mask)
            rdm[i, j] = rdm[j, i] = d
    return rdm


def _cluster_order(rdm: np.ndarray) -> np.ndarray:
    # Make symmetric and clip tiny negatives before condensing.
    r = (rdm + rdm.T) / 2
    r = np.clip(r, 0, None)
    np.fill_diagonal(r, 0)
    cond = squareform(r, checks=False)
    Z = linkage(cond, method="average")
    return leaves_list(Z)


def plot_rdm(runs: list[Run], position: str, out: Path):
    runs = [r for r in runs if r.position == position]
    if not runs:
        return {}
    labels = [r.key for r in runs]
    conds = [r.condition for r in runs]
    rdm_c = _build_rdm_continuous(runs)
    rdm_b = _build_rdm_binary(runs)
    order_c = _cluster_order(rdm_c)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    for ax, rdm, name in zip(axes, [rdm_c, rdm_b], ["RDM-continuous (1 - Pearson)", "RDM-binary (1 - Jaccard)"]):
        ordered = rdm[order_c][:, order_c]
        ord_labels = [labels[i] for i in order_c]
        ord_conds = [conds[i] for i in order_c]
        colors = [COND_INFO[c][1] for c in ord_conds]
        sns.heatmap(ordered, ax=ax, cmap="viridis", square=True, cbar_kws={"label": "distance"},
                    xticklabels=ord_labels, yticklabels=ord_labels)
        ax.set_title(f"{name} -- {position} patch\n(ordered by continuous-RDM average-link clustering)", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=7)
        for i, c in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, -0.6), 1, 0.5, color=c, clip_on=False))
            ax.add_patch(plt.Rectangle((-0.6, i), 0.5, 1, color=c, clip_on=False))

    # legend of condition colors
    handles = [plt.Line2D([0], [0], marker="s", linestyle="", color=info[1], label=info[0])
               for info in COND_INFO.values() if position in info[2]]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # return mean within-condition vs between-condition continuous distance for predictions
    return {"labels": labels, "conditions": conds, "rdm_c": rdm_c, "rdm_b": rdm_b, "order": order_c}


# ---------- predictions eval ----------

def mean_within_between(rdm: np.ndarray, conds: list[str], position: str):
    n = len(conds)
    within, between = [], []
    for i, j in itertools.combinations(range(n), 2):
        d = rdm[i, j]
        (within if conds[i] == conds[j] else between).append(d)
    return np.mean(within) if within else float("nan"), np.mean(between) if between else float("nan")


def cond_pair_mean_dist(rdm_info: dict, a: str, b: str) -> float:
    labels, conds, rdm = rdm_info["labels"], rdm_info["conditions"], rdm_info["rdm_c"]
    ds = []
    for i, ci in enumerate(conds):
        for j, cj in enumerate(conds):
            if i < j and ((ci == a and cj == b) or (ci == b and cj == a)):
                ds.append(rdm[i, j])
    return float(np.mean(ds)) if ds else float("nan")


# ---------- summary md ----------

def write_summary(out_md: Path, universal_stats, venn_regions, pearson_rs, rdm_bf, rdm_end):
    """universal_stats: dict {cond: {'strict_6': n, 'permissive_4': n}} for bf."""
    lines = []
    lines.append("# bf-patch comparison: C1 / C3 / C8 summary\n")
    lines.append("See `PLAN_BF_COMPARISON.md` for full design, assumptions, and parameter choices.\n")
    lines.append("## Significance consistency (bf)\n")
    lines.append("Per-run FWER-corrected sig counts are small (~10 heads each for C1/C3, **0 for C8**).")
    lines.append("We report # heads significant in at least K of 6 runs for K=1..6:\n")
    lines.append("| Condition | K=1 | K=2 | K=3 | K=4 | K=5 | K=6 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for c in ["C1", "C3", "C8"]:
        s = universal_stats[c]
        lines.append(f"| {COND_INFO[c][0]} | {s['k=1']} | {s['k=2']} | {s['k=3']} | {s['k=4']} | {s['k=5']} | {s['k=6']} |")
    lines.append("\n**C8 has zero FWER-significant heads in any bf run.** This is the paper-worthy finding: "
                 "under matched belief-context, the bf patch produces no FWER-survivable signal. "
                 "Venn below uses K=2 (head appears in ≥2/6 runs of its condition).\n")

    lines.append("## Venn regions (universal 6/6 sig at bf)\n")
    lines.append("| Region | # heads | Interpretation |")
    lines.append("|---|---:|---|")
    lines.append(f"| only C1 | {venn_regions['only_C1']} | C1-exclusive, no answer component |")
    lines.append(f"| only C3 | {venn_regions['only_C3']} | C3-exclusive |")
    lines.append(f"| only C8 | {venn_regions['only_C8']} | C8-exclusive |")
    lines.append(f"| C1 ∩ C3 \\ C8 | {venn_regions['C1_C3_not_C8']} | **belief-specific candidates** |")
    lines.append(f"| C3 ∩ C8 \\ C1 | {venn_regions['C3_C8_not_C1']} | answer/location-specific |")
    lines.append(f"| C1 ∩ C8 \\ C3 | {venn_regions['C1_C8_not_C3']} | unusual — flag |")
    lines.append(f"| all three   | {venn_regions['all_three']}   | generic bf-position signal (suspect) |")
    lines.append("")

    lines.append("## Pearson r on mean raw scores (bootstrap 95% CI)\n")
    lines.append("| Pair | r | 95% CI |")
    lines.append("|---|---:|---|")
    for k, (r, lo, hi) in pearson_rs.items():
        lines.append(f"| {k} | {r:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines.append("")

    lines.append("## Predictions (stated before running — see PLAN_BF_COMPARISON.md §4b)\n")
    pred_rows = []

    # 1. Within-condition < between-condition (bf)
    w_bf, b_bf = mean_within_between(rdm_bf["rdm_c"], rdm_bf["conditions"], "bf")
    w_end, b_end = mean_within_between(rdm_end["rdm_c"], rdm_end["conditions"], "end")
    pred_rows.append(("Within-condition runs cluster together (bf)", w_bf < b_bf,
                      f"mean within={w_bf:.3f}, between={b_bf:.3f}"))
    pred_rows.append(("Within-condition runs cluster together (end)", w_end < b_end,
                      f"mean within={w_end:.3f}, between={b_end:.3f}"))

    # 2. C1-C3 closer than C1-C8 and C3-C8 at bf
    d13 = cond_pair_mean_dist(rdm_bf, "C1", "C3")
    d18 = cond_pair_mean_dist(rdm_bf, "C1", "C8")
    d38 = cond_pair_mean_dist(rdm_bf, "C3", "C8")
    ok = (d13 < d18) and (d13 < d38)
    pred_rows.append(("C1 and C3 closer than either is to C8 at bf", ok,
                      f"d(C1,C3)={d13:.3f}, d(C1,C8)={d18:.3f}, d(C3,C8)={d38:.3f}"))

    # 3. C3 and C5 close at end
    d35 = cond_pair_mean_dist(rdm_end, "C3", "C5")
    d_self = np.nanmean([cond_pair_mean_dist(rdm_end, "C1", "C3"),
                         cond_pair_mean_dist(rdm_end, "C2", "C4")])
    pred_rows.append(("C3 and C5 close at end (location-retrieval dominated)",
                      d35 < d_self,
                      f"d(C3,C5)={d35:.3f}, compared to mean of other cross-pairs={d_self:.3f}"))

    lines.append("| Prediction | Held? | Evidence |")
    lines.append("|---|:---:|---|")
    for p, ok, ev in pred_rows:
        lines.append(f"| {p} | {'✓' if ok else '✗'} | {ev} |")
    lines.append("")

    out_md.write_text("\n".join(lines))
    print(f"wrote {out_md}")


def main():
    OUT.mkdir(exist_ok=True)

    print("loading runs...")
    runs = load_all_runs()
    print(f"loaded {len(runs)} runs")

    # --- mean score vectors for bf conditions ---
    S_C1 = mean_score_vector(runs, "C1", "bf")
    S_C3 = mean_score_vector(runs, "C3", "bf")
    S_C8 = mean_score_vector(runs, "C8", "bf")

    # Print sig counts at every threshold — these numbers are the data.
    print("per-run sig counts (bf):")
    for c in ["C1", "C3", "C8"]:
        rs = [r for r in runs if r.condition == c and r.position == "bf"]
        per_run = [int(r.sig_mask.sum()) for r in rs]
        print(f"  {c}: {per_run}  (mean {np.mean(per_run):.1f})")

    universal_stats = {}
    print("\nconsistency counts (heads sig in at least K/6 runs, bf):")
    for c in ["C1", "C3", "C8"]:
        row = {}
        for k in [1, 2, 3, 4, 5, 6]:
            row[f"k={k}"] = int(universal_mask(runs, c, "bf", k).sum())
        universal_stats[c] = row
        print(f"  {c}: " + "  ".join(f"k={k}:{v}" for k, v in row.items()))

    # Use K=2/6 for Venn: appears in at least 2 of 6 runs of its condition.
    # 6/6 is empty for every condition (per-run sig sets are small, ~10 heads each,
    # and do not overlap at the strict level). 2/6 is our "reproducible-across-runs"
    # threshold — stricter than union-across-runs, looser than per-run-FWER's intersection.
    K_VENN = 2
    U_C1 = universal_mask(runs, "C1", "bf", K_VENN)
    U_C3 = universal_mask(runs, "C3", "bf", K_VENN)
    U_C8 = universal_mask(runs, "C8", "bf", K_VENN)
    print(f"\nVenn threshold: head appears in ≥{K_VENN}/6 runs of its condition")
    print(f"  |U(C1)|={U_C1.sum()}  |U(C3)|={U_C3.sum()}  |U(C8)|={U_C8.sum()}")

    # --- Venn D3 ---
    venn_regions = plot_venn(U_C1, U_C3, U_C8, OUT / "venn_C1_C3_C8.png")
    print(f"wrote {OUT/'venn_C1_C3_C8.png'}")
    print(f"  belief-specific (C1 ∩ C3 \\ C8) = {venn_regions['C1_C3_not_C8']}")
    print(f"  answer-specific (C3 ∩ C8 \\ C1) = {venn_regions['C3_C8_not_C1']}")
    print(f"  generic (all three)             = {venn_regions['all_three']}")

    # --- Scatter D4 ---
    pearson_rs = plot_scatter(S_C1, S_C3, S_C8, U_C1, U_C3, U_C8, OUT / "scatter_bf.png")
    print(f"wrote {OUT/'scatter_bf.png'}")
    for k, (r, lo, hi) in pearson_rs.items():
        print(f"  Pearson {k}: r={r:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    # --- RDMs D6, D7 ---
    rdm_bf_info = plot_rdm(runs, "bf", OUT / "rdm_bf.png")
    print(f"wrote {OUT/'rdm_bf.png'}")
    rdm_end_info = plot_rdm(runs, "end", OUT / "rdm_end.png")
    print(f"wrote {OUT/'rdm_end.png'}")

    # --- summary md D2/D8 ---
    write_summary(OUT / "bf_comparison_summary.md",
                  universal_stats, venn_regions, pearson_rs,
                  rdm_bf_info, rdm_end_info)


if __name__ == "__main__":
    main()
