"""
Cross-run ablation comparison vs random null.

Pulls cumulative_ablation_results.csv from each of the 8 ablation runs
(sysprompt x prefill 2x2 -- minus the original neutral+no_prefill which is the
top-level CSV with old schema -- plus cross-frame photo and 4 decomp variants)
and the random-baseline null distribution (belief_minus_loc), then produces:

- table_ablation_summary.csv : long-form (run, k, eval_variant, metric, value)
  with one row per (run, eval_variant, k in {0, 11, 21, k_max}, metric in
  {flip_rate, mean_p_correct}).
- table_ablation_summary.md  : human-readable pivot, flip_rate only.
- fig_false_belief_curves.png : 2-panel figure. (a) flip_rate vs k for the 8
  runs on false_belief, with null mean +- 1.96*std band at k=11 and k=21.
  (b) mean_p_correct vs k, same overlay.
- fig_k0_flip_rates.png : at k=0 for every run, bar chart of flip_rate by
  eval_variant. Answers "is the false-belief flip_rate-at-baseline really 0.37
  while everything else is ~0".
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ABL = ROOT / "results" / "ablation"
OUT = ROOT / "results" / "ablation" / "_baseline_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# (label, subdir) in the order we want them displayed
RUNS = [
    ("matched_sys / no_prefill",      "belief_only_sysprompt_matched"),
    ("matched_sys / prefill",         "belief_only_sysprompt_prefill"),
    ("neutral_sys / prefill",         "belief_only_neutral_prefill"),
    ("photo_heads / sys+prefill",     "photo_heads_sysprompt_prefill"),
    ("decomp belief_only",            "decomp_belief_only"),
    ("decomp photo_only",             "decomp_photo_only"),
    ("decomp intersection (B rank)",  "decomp_intersection_belief_rank"),
    ("decomp intersection (P rank)",  "decomp_intersection_photo_rank"),
]
EVAL_VARIANTS = ["false_belief", "true_belief", "false_photo", "true_photo", "basic_ab", "basic_ba"]
METRICS = ["flip_rate", "mean_p_correct", "flip_score"]
K_PROBES = [0, 11, 21]  # k_max appended per-run


def load_run(label: str, subdir: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_csv(ABL / subdir / "cumulative_ablation_results.csv")
    df = df[df["curve"] == "target"].copy()
    df["run"] = label
    n_heads = len(pd.read_csv(ABL / subdir / "head_list.csv"))
    return df, n_heads


def build_summary_table(runs: list[tuple[pd.DataFrame, int, str]]) -> pd.DataFrame:
    rows = []
    for df, n_heads, label in runs:
        probes = sorted(set(K_PROBES + [n_heads]))
        for k in probes:
            if k > n_heads:
                continue
            sub = df[df["k"] == k]
            for v in EVAL_VARIANTS:
                cell = sub[sub["eval_variant"] == v]
                if cell.empty:
                    continue
                for m in METRICS:
                    rows.append(dict(run=label, k=k, k_is_max=(k == n_heads),
                                     eval_variant=v, metric=m,
                                     value=float(cell[m].iloc[0])))
    return pd.DataFrame(rows)


def load_nulls() -> dict[str, pd.DataFrame]:
    """Return {label: null_summary_df} for each random-null run that exists.

    label encodes the eval protocol so the figure can show both bands.
    """
    paths = {
        "null: neutral_sys / no_prefill (21-head target)":
            ABL / "random_baselines" / "belief_minus_loc" / "null_summary.csv",
        "null: matched_sys / prefill (32-head target)":
            ABL / "random_baselines" / "belief_only_matched_prefill" / "null_summary.csv",
    }
    return {label: pd.read_csv(p) for label, p in paths.items() if p.exists()}


def plot_curves(runs: list[tuple[pd.DataFrame, int, str]],
                nulls: dict[str, pd.DataFrame]) -> None:
    plot_metrics = ["flip_score", "flip_rate", "mean_p_correct"]
    null_colors = {label: c for label, c in zip(nulls.keys(), ["grey", "tab:cyan"])}
    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharex=True)
    for metric, ax in zip(plot_metrics, axes):
        for df, n_heads, label in runs:
            sub = df[df["eval_variant"] == "false_belief"].sort_values("k")
            ax.plot(sub["k"], sub[metric], marker=".", linewidth=1.4, label=label)
        for null_label, null_df in nulls.items():
            nb = null_df[(null_df["eval_variant"] == "false_belief") & (null_df["metric"] == metric)]
            for _, r in nb.iterrows():
                k = int(r["k"])
                mu, sd = r["null_mean"], r["null_std"]
                ax.fill_between([k - 0.5, k + 0.5], mu - 1.96 * sd, mu + 1.96 * sd,
                                color=null_colors[null_label], alpha=0.25, zorder=0,
                                label=(null_label if k == 11 and metric == plot_metrics[0] else None))
                ax.hlines(mu, k - 0.5, k + 0.5,
                          colors=null_colors[null_label], linestyles="--", linewidth=1.2, zorder=1)
        ax.set_xlabel("k (cumulative heads ablated)")
        ax.set_ylabel(metric)
        ax.set_title(f"false_belief: {metric} vs k")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=7, ncol=1)
    fig.tight_layout()
    out = OUT / "fig_false_belief_curves.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")
    plt.close(fig)


def plot_k0_bars(runs: list[tuple[pd.DataFrame, int, str]]) -> None:
    # at k=0 every run is identical in *principle* (same prompts, no ablation).
    # we plot one bar per (run, eval_variant) to see if the prompts differ across
    # runs (they shouldn't for belief_only_* but might for photo/decomp depending
    # on how the eval set was built).
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(len(RUNS))
    width = 0.14
    colors = plt.cm.tab10(np.linspace(0, 1, len(EVAL_VARIANTS)))
    for i, v in enumerate(EVAL_VARIANTS):
        vals = []
        for df, n_heads, label in runs:
            cell = df[(df["k"] == 0) & (df["eval_variant"] == v)]
            vals.append(float(cell["flip_rate"].iloc[0]) if not cell.empty else np.nan)
        ax.bar(x + (i - len(EVAL_VARIANTS) / 2) * width + width / 2,
               vals, width, label=v, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in RUNS], rotation=30, ha="right")
    ax.set_ylabel("flip_rate at k=0 (no ablation)")
    ax.set_title("Baseline flip_rate by run x eval_variant\n(the 0.37 false-belief baseline question)")
    ax.axhline(0.05, color="red", linestyle=":", linewidth=1, label="0.05 chance reference")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_k0_flip_rates.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def _df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v):
        if isinstance(v, float):
            return format(v, floatfmt) if not np.isnan(v) else ""
        return "" if v is None else str(v)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def emit_markdown(tbl: pd.DataFrame, null_df: dict[str, pd.DataFrame] | pd.DataFrame) -> None:
    # accept either a single DataFrame (legacy) or dict of {label: df}
    if isinstance(null_df, pd.DataFrame):
        null_df = {"random null": null_df}
    lines = ["# Ablation baseline comparison\n"]
    for metric in ["flip_score", "flip_rate", "mean_p_correct"]:
        fr = tbl[tbl["metric"] == metric].copy()
        fr["k_label"] = fr.apply(lambda r: f"k={r['k']}" + (" (max)" if r["k_is_max"] else ""), axis=1)
        pivot = (fr[fr["eval_variant"] == "false_belief"]
                 .pivot_table(index="run", columns="k_label", values="value", aggfunc="first")
                 .reset_index())
        lines += [f"\n## false_belief {metric} by run and k\n", _df_to_md(pivot), ""]

        if metric == "flip_score":
            k0 = (fr[fr["k"] == 0]
                  .pivot_table(index="run", columns="eval_variant", values="value", aggfunc="first")
                  .reset_index())
            lines += [f"\n## k=0 baseline {metric} (φ) across eval_variants — the 0.37 mystery\n",
                      _df_to_md(k0), ""]

    lines += ["\n## Random-null references (false_belief)\n"]
    for null_label, null_df in null_df.items():  # null_df is now a dict
        lines.append(f"### {null_label}\n")
        nb = null_df[(null_df["eval_variant"] == "false_belief") &
                     (null_df["metric"].isin(["flip_rate", "flip_score", "mean_p_correct"]))][
            ["k", "metric", "target", "null_mean", "null_std",
             "null_p05", "null_p95", "p_target_geq_null"]
        ].reset_index(drop=True)
        lines.append(_df_to_md(nb))
        lines.append("")
    out = OUT / "table_ablation_summary.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    runs = []
    for label, subdir in RUNS:
        df, n_heads = load_run(label, subdir)
        runs.append((df, n_heads, label))
        print(f"  {label:<35s} k_max={n_heads:>3d} rows={len(df):>4d}")

    tbl = build_summary_table(runs)
    tbl.to_csv(OUT / "table_ablation_summary.csv", index=False)
    print(f"wrote {OUT / 'table_ablation_summary.csv'}")

    nulls = load_nulls()
    print(f"  loaded {len(nulls)} null distribution(s):")
    for label in nulls:
        print(f"    - {label}")

    plot_curves(runs, nulls)
    plot_k0_bars(runs)
    emit_markdown(tbl, nulls)


if __name__ == "__main__":
    main()
