"""
Decide whether Yang-style Q×K attention figures are worth making.

For each top-K CMA head in each cell, partition lookback (and other query)
attention mass into:
  named_total      — sum over orthogonal named keys
                     (state_first + state_second + info_loss + char_q
                      + obj_q + belief_form)
  top_unnamed_mass — single largest unnamed peak (from unannotated_peaks*.csv)
  diffuse          — residual = max(0, 1 - named_total - top_unnamed_mass)

Verdict per (condition, query):
  diffuse-dominant  → heatmap will be smear; not worth making
  unnamed-dominant  → heads attend somewhere we didn't tag; figure (or just
                      the unnamed-peak token examples) is the right tool
  named-adequate    → null on named scalars is the real story

Run:
  python analysis_scripts/lookback_analysis/diagnose_named_coverage.py \
      --root results/lookback_analysis --top_k 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NAMED_ORTHOGONAL = [
    "state_first", "state_second", "info_loss",
    "char_q", "obj_q", "belief_form",
]
QUERIES = ["lookback", "belief_formation", "agent_last", "agent_first"]
UNNAMED_FILES = {
    "lookback":          "unannotated_peaks.csv",
    "belief_formation":  "unannotated_peaks_belief_formation.csv",
    "agent_last":        "unannotated_peaks_agent_last.csv",
    "agent_first":       "unannotated_peaks_agent_first.csv",
}


def named_total(row: pd.Series, query: str) -> float:
    cols = [f"{query}_attn_to_{k}" for k in NAMED_ORTHOGONAL]
    vals = [row.get(c, 0.0) or 0.0 for c in cols]
    return float(sum(v for v in vals if pd.notna(v)))


def cell_dirs(root: Path):
    """Yield every (condition, template, rule_dir, patch_dir) cell that has
    a lookback_scores.csv."""
    for cond_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for tmpl_dir in sorted(cond_dir.glob("template_*")):
            for rule_dir in sorted(tmpl_dir.glob("base_rule_*")):
                for patch_dir in sorted(rule_dir.iterdir()):
                    if (patch_dir / "lookback_scores.csv").exists():
                        yield cond_dir.name, tmpl_dir.name.replace("template_", ""), \
                              rule_dir.name, patch_dir.name, patch_dir


def load_unnamed(patch_dir: Path, query: str) -> pd.DataFrame:
    fp = patch_dir / UNNAMED_FILES[query]
    empty = pd.DataFrame(columns=["layer", "head", "offset_from_end",
                                  "mean_attn", "examples"])
    if not fp.exists() or fp.stat().st_size == 0:
        return empty
    try:
        df = pd.read_csv(fp)
    except pd.errors.EmptyDataError:
        return empty
    if df.empty or "mean_attn" not in df.columns:
        return empty
    # Keep only the row with highest mean_attn per (layer, head)
    df = df.sort_values("mean_attn", ascending=False) \
           .drop_duplicates(["layer", "head"], keep="first")
    return df[["layer", "head", "offset_from_end", "mean_attn", "examples"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/lookback_analysis")
    ap.add_argument("--top_k", type=int, default=10,
                    help="top-K CMA heads per cell to diagnose (Yang uses 10)")
    ap.add_argument("--out_dir", default="results/lookback_analysis/_diagnostic")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_head_rows = []

    for cond, tmpl, rule, patch, patch_dir in cell_dirs(root):
        scores = pd.read_csv(patch_dir / "lookback_scores.csv")
        scores = scores[scores["observed_score"] > 0]
        scores = scores.sort_values("observed_score", ascending=False).head(args.top_k)
        if scores.empty:
            continue

        unnamed = {q: load_unnamed(patch_dir, q) for q in QUERIES}

        for _, h in scores.iterrows():
            for q in QUERIES:
                # named-mass column may be missing for some old captures
                anchor_col = f"{q}_attn_to_state"
                if anchor_col not in h.index:
                    continue
                ntot = named_total(h, q)
                un = unnamed[q]
                m = un[(un["layer"] == h["layer"]) & (un["head"] == h["head"])]
                if len(m):
                    top_un = float(m.iloc[0]["mean_attn"])
                    off = m.iloc[0]["offset_from_end"]
                    ex = m.iloc[0]["examples"]
                else:
                    top_un, off, ex = 0.0, None, ""
                diffuse = max(0.0, 1.0 - ntot - top_un)
                per_head_rows.append({
                    "condition": cond, "template": tmpl, "rule": rule,
                    "patch_position": patch, "query": q,
                    "layer": int(h["layer"]), "head": int(h["head"]),
                    "observed_score": float(h["observed_score"]),
                    "named_total": ntot,
                    "top_unnamed_mass": top_un,
                    "top_unnamed_offset_from_end": off,
                    "top_unnamed_examples": ex,
                    "diffuse": diffuse,
                })

    df = pd.DataFrame(per_head_rows)
    if df.empty:
        print("no data found under", root)
        return

    per_head_csv = out_dir / "named_coverage_per_head.csv"
    df.to_csv(per_head_csv, index=False)
    print(f"wrote {per_head_csv}  rows={len(df)}")

    # Aggregate per (condition, query)
    THRESH = 0.30
    agg_rows = []
    for (cond, q), g in df.groupby(["condition", "query"]):
        n = len(g)
        named_dom = ((g["named_total"] > g["top_unnamed_mass"]) & (g["named_total"] > THRESH)).sum()
        unnam_dom = ((g["top_unnamed_mass"] > g["named_total"]) & (g["top_unnamed_mass"] > THRESH)).sum()
        diffuse_dom = (g["diffuse"] > (1 - 2 * THRESH)).sum()  # < ~40% accounted for
        if diffuse_dom / n > 0.5:
            verdict = "diffuse: figure unlikely to help"
        elif unnam_dom / n > 0.3:
            verdict = "unnamed-dominant: investigate peak offsets/examples, figure useful"
        else:
            verdict = "named-adequate: null on named scalars is real"
        agg_rows.append({
            "condition": cond, "query": q, "n_heads": n,
            "mean_named_total": g["named_total"].mean(),
            "mean_top_unnamed": g["top_unnamed_mass"].mean(),
            "mean_diffuse": g["diffuse"].mean(),
            "frac_named_dominant": named_dom / n,
            "frac_unnamed_dominant": unnam_dom / n,
            "frac_diffuse": diffuse_dom / n,
            "verdict": verdict,
        })

    agg = pd.DataFrame(agg_rows).sort_values(["query", "condition"])
    agg_csv = out_dir / "named_coverage_summary.csv"
    agg.to_csv(agg_csv, index=False)
    print(f"wrote {agg_csv}")

    # Print summary table
    print("\n=== verdict per (condition, query) ===")
    for q in QUERIES:
        sub = agg[agg["query"] == q]
        if sub.empty:
            continue
        print(f"\n[{q}]")
        for _, r in sub.iterrows():
            print(f"  {r['condition']:32s}  "
                  f"named={r['mean_named_total']:.2f}  "
                  f"top_unnamed={r['mean_top_unnamed']:.2f}  "
                  f"diffuse={r['mean_diffuse']:.2f}  "
                  f"→ {r['verdict']}")

    # Top unnamed-peak token examples per condition (lookback only) — fastest
    # way to see *what* the heads actually look at when not on named positions.
    print("\n=== top unnamed-peak examples (lookback) ===")
    look = df[(df["query"] == "lookback") & (df["top_unnamed_mass"] > THRESH)]
    if look.empty:
        print("  (none above threshold)")
    else:
        for cond, g in look.groupby("condition"):
            g = g.sort_values("top_unnamed_mass", ascending=False).head(5)
            print(f"\n  {cond}")
            for _, r in g.iterrows():
                ex = (r["top_unnamed_examples"] or "")[:80]
                print(f"    L{r['layer']}H{r['head']}  "
                      f"mass={r['top_unnamed_mass']:.2f}  "
                      f"offset_from_end={r['top_unnamed_offset_from_end']}  "
                      f"ex: {ex}")


if __name__ == "__main__":
    main()
