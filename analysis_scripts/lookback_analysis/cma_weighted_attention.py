"""CMA-weighted lookback attention, aggregated per condition.

For each condition c, pool every (head, template, direction) row from
lookback_scores.csv where the head passed the FWER-corrected CMA
permutation test (significant == True). For each query position q in
{lookback, belief_formation, agent_last, agent_first} and each named
source category s in {state_first, state_second, state_correct,
info_loss, char_q, obj_q, belief_form}, compute the CMA-weighted mean
attention:

    WAttn(c, q, s) = sum_r  max(w_r, 0) * a_r(q -> s)
                   / sum_r  max(w_r, 0)

where r ranges over the pooled significant rows of condition c, w_r is
the head's CMA observed_score in that (template, direction) cell, and
a_r is the per-prompt mean attention from query q to source s.

Outputs (under --out_dir, default results/lookback_analysis/_summary):
  cma_weighted_attention.csv  -- long format: condition, query, source,
                                 wattn, n_rows, sum_w
  cma_weighted_attention_<query>.png  -- one heatmap per query position,
                                         rows = conditions, cols = sources
  cma_weighted_attention_grid.png     -- 2x2 grid of all four queries

Usage:
  python analysis_scripts/lookback_analysis/cma_weighted_attention.py
  python analysis_scripts/lookback_analysis/cma_weighted_attention.py \
      --root results/lookback_analysis \
      --out_dir results/lookback_analysis/_summary
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

QUERIES = ["lookback", "belief_formation", "agent_last", "agent_first"]
SOURCES = ["state_first", "state_second", "state_correct",
           "info_loss", "char_q", "obj_q", "belief_form"]

PATH_RE = re.compile(
    r"lookback_analysis/(?P<cond>[^/]+)/template_(?P<tmpl>[^/]+)/"
    r"base_rule_(?P<br>[^_]+)_exp_rule_(?P<er>[^/]+)/patch_before_movement/"
    r"lookback_scores\.csv$"
)


def load_pool(root: Path) -> pd.DataFrame:
    rows = []
    pattern = str(root / "*/template_*/base_rule_*/patch_before_movement/lookback_scores.csv")
    for f in sorted(glob.glob(pattern)):
        m = PATH_RE.search(f)
        if not m:
            continue
        df = pd.read_csv(f)
        df["condition"] = m.group("cond")
        df["template"] = m.group("tmpl")
        df["base_rule"] = m.group("br")
        df["direction"] = f"{m.group('br')}->{m.group('er')}"
        rows.append(df)
    if not rows:
        raise SystemExit(f"no lookback_scores.csv found under {root}")
    return pd.concat(rows, ignore_index=True)


def weighted_attention(pool: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """CMA-weighted mean attention, grouped by `group_cols` (e.g. ['condition'] or
    ['condition', 'base_rule'])."""
    sig = pool[pool["significant"] == True].copy()  # noqa: E712
    sig["w"] = sig["observed_score"].clip(lower=0)

    out = []
    for key, sub in sig.groupby(group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        for q in QUERIES:
            for s in SOURCES:
                col = f"{q}_attn_to_{s}"
                row = {gc: kv for gc, kv in zip(group_cols, key)}
                row.update({"query": q, "source": s})
                if col not in sub.columns:
                    row.update({"wattn": np.nan, "n_rows": 0, "sum_w": 0.0})
                    out.append(row); continue
                vals = sub[col]
                mask = vals.notna()
                if not mask.any():
                    row.update({"wattn": np.nan, "n_rows": 0, "sum_w": 0.0})
                    out.append(row); continue
                num = float((sub.loc[mask, "w"] * vals[mask]).sum())
                den = float(sub.loc[mask, "w"].sum())
                wattn = num / den if den > 0 else np.nan
                row.update({"wattn": wattn, "n_rows": int(mask.sum()), "sum_w": den})
                out.append(row)
    return pd.DataFrame(out)


def to_matrix(long_df: pd.DataFrame, query: str,
              conditions: list[str],
              base_rule: str | None = None) -> np.ndarray:
    sub = long_df[long_df["query"] == query]
    if base_rule is not None:
        sub = sub[sub["base_rule"] == base_rule]
    mat = np.full((len(conditions), len(SOURCES)), np.nan)
    for i, c in enumerate(conditions):
        for j, s in enumerate(SOURCES):
            row = sub[(sub.condition == c) & (sub.source == s)]
            if len(row):
                mat[i, j] = row["wattn"].iloc[0]
    return mat


def plot_panel(ax, mat, conditions, title, vmax,
               show_yticks: bool = True, show_xticks: bool = True):
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(SOURCES)))
    if show_xticks:
        ax.set_xticklabels(SOURCES, rotation=30, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(len(conditions)))
    if show_yticks:
        ax.set_yticklabels(conditions, fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "·", ha="center", va="center", color="gray", fontsize=8)
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < vmax * 0.55 else "black", fontsize=8)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/lookback_analysis", type=Path)
    ap.add_argument("--out_dir", default="results/lookback_analysis/_summary", type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = load_pool(args.root)
    n_sig = int((pool["significant"] == True).sum())  # noqa: E712
    print(f"pooled {len(pool)} rows from {pool.condition.nunique()} conditions; "
          f"{n_sig} significant.")

    # Pooled (across directions) -- kept for completeness.
    pooled_df = weighted_attention(pool, ["condition"])
    pooled_df.to_csv(args.out_dir / "cma_weighted_attention_pooled.csv", index=False)

    # Per-direction (the main output now).
    perdir_df = weighted_attention(pool, ["condition", "base_rule"])
    out_csv = args.out_dir / "cma_weighted_attention_by_direction.csv"
    perdir_df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    conditions = sorted(pool.condition.unique())
    base_rules = ["ABA", "ABB"]  # ABA-base = state_first is the answer; ABB-base = state_second is.
    dir_titles = {
        "ABA": "base = ABA  (FB-frame / photo-before-move: state_first = answer)",
        "ABB": "base = ABB  (TB-frame / photo-after-move: state_second = answer)",
    }

    # Compute matrices and global vmax across queries x directions for fair comparison.
    matrices = {(q, br): to_matrix(perdir_df, q, conditions, base_rule=br)
                for q in QUERIES for br in base_rules}
    vmax = float(np.nanmax(np.concatenate([m.ravel() for m in matrices.values()])))
    print(f"global vmax across all 4 queries x 2 directions: {vmax:.3f}")

    # 4-rows x 2-cols mega-grid: rows = queries, cols = directions.
    # y-tick labels (condition names) only on leftmost column.
    # x-tick labels (source names) only on bottom row.
    n_rows = len(QUERIES)
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 3.4 * n_rows))
    last_im = None
    for r, q in enumerate(QUERIES):
        for c, br in enumerate(base_rules):
            ax = axes[r, c]
            last_im = plot_panel(
                ax, matrices[(q, br)], conditions,
                f"query = {q}  |  base = {br}", vmax,
                show_yticks=(c == 0),
                show_xticks=(r == n_rows - 1),
            )
    fig.suptitle(
        "CMA-weighted attention by query x direction\n"
        "(rows = query position; cols = base_rule; weights = max(CMA "
        f"observed_score, 0); pooled over templates; shared vmax={vmax:.2f})",
        fontsize=11,
    )
    fig.subplots_adjust(hspace=0.30, wspace=0.05, top=0.93, bottom=0.08,
                        left=0.18, right=0.88)
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.6,
                 label="CMA-weighted mean attention")
    out_grid = args.out_dir / "cma_weighted_attention_by_direction_grid.png"
    fig.savefig(out_grid, dpi=150)
    plt.close(fig)
    print(f"wrote {out_grid}")

    # Per-query 2-panel figures (ABA | ABB). y-ticks only on left panel;
    # x-ticks on both since both are "bottom row" in a 1-row layout.
    for q in QUERIES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for c, br in enumerate(base_rules):
            im = plot_panel(
                axes[c], matrices[(q, br)], conditions,
                dir_titles[br], vmax,
                show_yticks=(c == 0),
                show_xticks=True,
            )
        fig.suptitle(f"CMA-weighted attention, query = {q}", fontsize=11)
        fig.subplots_adjust(wspace=0.05, top=0.86, bottom=0.20,
                            left=0.20, right=0.88)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                     label="CMA-weighted mean attention")
        out_png = args.out_dir / f"cma_weighted_attention_by_direction_{q}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
