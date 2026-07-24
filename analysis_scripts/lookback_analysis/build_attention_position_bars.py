"""Re-cut of the lookback attention captures as position bar plots
(July 2026 plan, step 4 — "the figure Taylor described", meeting 00:44).

Data: other_results/lookback_analysis/{cond}/template_*/base_rule_*/
patch_before_movement/lookback_scores.csv — per-head (top-50 by CMA score
per cell) attention mass from 4 query positions (lookback = final token,
belief_formation, agent_last, agent_first) to 7 labeled source positions
(state_first, state_second, state_correct, info_loss, char_q, obj_q,
belief_form). End position, seed=1 heads, Qwen2.5-14B-Instruct.

Aggregation (matches _summary convention): rows with significant == True,
weights = max(observed_score, 0), weighted mean per (query, source); pooled
over a condition's cells, or split by template.

Caveat stated in the report: each condition's profile uses its own
significant heads on its own prompts (captures were per-condition top-50;
C5 heads' attention on C1 prompts was never captured — no new captures
launched per plan).

Outputs -> writeups/results_for_pi/attention_bars/
  figures/*.png, stats_attention_position.csv, stats_fragment.md
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from h2_additivity_robustness import BLUE, ORANGE, GREEN, GRAY  # noqa: E402

LB_ROOT = REPO / "other_results/lookback_analysis"
OUT = REPO / "writeups/results_for_pi/attention_bars"
FIG = OUT / "figures"

CONDS = ["abstract_belief", "abstract_photo", "answer_changes_belief",
         "answer_changes_location_only", "answer_changes_location_only_photo",
         "answer_changes_photo", "state_only_no_belief"]
CNUM = {"abstract_belief": "C1", "abstract_photo": "C2",
        "answer_changes_belief": "C3", "answer_changes_location_only": "C5",
        "answer_changes_location_only_photo": "C6",
        "answer_changes_photo": "C4", "state_only_no_belief": "C7"}
QUERIES = ["lookback", "belief_formation", "agent_last", "agent_first"]
SOURCES = ["state_first", "state_second", "state_correct", "info_loss",
           "char_q", "obj_q", "belief_form"]
SRC_LABEL = {"state_first": "1st location", "state_second": "2nd location",
             "state_correct": "correct answer", "info_loss": "info-loss event",
             "char_q": "agent (question)", "obj_q": "object (question)",
             "belief_form": "belief formation"}
PURPLE = "#CC79A7"  # Okabe-Ito, for the 3rd template


def load_rows():
    rows = []
    for cond in CONDS:
        for csv in sorted(LB_ROOT.glob(
                f"{cond}/template_*/base_rule_*/patch_before_movement/lookback_scores.csv")):
            df = pd.read_csv(csv)
            df = df[df["significant"].astype(str).str.lower() == "true"].copy()
            if df.empty:
                continue
            df["w"] = df["observed_score"].clip(lower=0)
            df["condition"] = cond
            df["template"] = csv.parts[-4].replace("template_", "")
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def wmean(df, query, source):
    """Weighted mean over head-rows; rows from cells lacking this labeled
    position (photo prompts have no info_loss/char_q) enter as NaN and are
    masked out rather than poisoning the pooled average."""
    col = f"{query}_attn_to_{source}"
    if col not in df:
        return np.nan
    m = df[col].notna() & (df["w"] > 0)
    if not m.any():
        return np.nan
    return float(np.average(df.loc[m, col], weights=df.loc[m, "w"]))


def profile(df, query):
    return [wmean(df, query, s) for s in SOURCES]


def bars_two_sets(ax, prof_a, prof_b, label_a, label_b, title):
    x = np.arange(len(SOURCES))
    w = 0.38
    ax.bar(x - w / 2, prof_a, w, color=BLUE, label=label_a)
    ax.bar(x + w / 2, prof_b, w, color=ORANGE, label=label_b)
    for xi, v in zip(x - w / 2, prof_a):
        if np.isfinite(v) and v > 0.015:
            ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, prof_b):
        if np.isfinite(v) and v > 0.015:
            ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SRC_LABEL[s] for s in SOURCES], rotation=28,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("CMA-weighted attention mass")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    all_rows = load_rows()

    # tidy long table: condition x template(+pooled) x query x source
    recs = []
    for cond, g in all_rows.groupby("condition"):
        groups = [("pooled", g)] + [(t, gt) for t, gt in g.groupby("template")]
        for tmpl, gg in groups:
            for q in QUERIES:
                for s, v in zip(SOURCES, profile(gg, q)):
                    recs.append({"condition": cond, "template": tmpl,
                                 "query": q, "source": s, "wattn": v,
                                 "n_head_rows": len(gg)})
    tidy = pd.DataFrame(recs)
    tidy.to_csv(OUT / "stats_attention_position.csv", index=False)

    g_c1 = all_rows[all_rows.condition == "abstract_belief"]
    g_c5 = all_rows[all_rows.condition == "answer_changes_location_only"]

    # F1 — the Taylor figure: C1 vs C5 at the two informative queries
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    bars_two_sets(axes[0], profile(g_c1, "lookback"), profile(g_c5, "lookback"),
                  "C1 heads (belief)", "C5 heads (answer)",
                  "query = final token (lookback)")
    bars_two_sets(axes[1], profile(g_c1, "belief_formation"),
                  profile(g_c5, "belief_formation"),
                  "C1 heads (belief)", "C5 heads (answer)",
                  "query = belief-formation token")
    ymax = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, ymax)
    fig.suptitle("Where do C1 vs C5 significant heads attend? "
                 "(each set on its own condition's prompts, pooled over templates)",
                 fontsize=11, y=1.02)
    fig.savefig(FIG / "fig1_c1_vs_c5_bars.png")
    plt.close(fig)

    # F2 — one panel per condition, lookback query
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 6.6), sharey=True)
    for ax, cond in zip(axes.ravel(), CONDS):
        g = all_rows[all_rows.condition == cond]
        prof = profile(g, "lookback")
        x = np.arange(len(SOURCES))
        colors = [GREEN if s == "state_correct" else BLUE for s in SOURCES]
        ax.bar(x, prof, 0.62, color=colors)
        for xi, v in zip(x, prof):
            if np.isfinite(v) and v > 0.02:
                ax.text(xi, v + 0.004, f"{v:.2f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([SRC_LABEL[s] for s in SOURCES], rotation=45,
                           ha="right", fontsize=7)
        ax.set_title(f"{CNUM[cond]} · {cond}", fontsize=9)
    axes[1, 3].axis("off")
    axes[0, 0].set_ylabel("weighted attention")
    axes[1, 0].set_ylabel("weighted attention")
    fig.suptitle("Final-token attention by condition (green = correct-answer position)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_all_conditions_lookback.png")
    plt.close(fig)

    # F3 — template split for C1 and C5 (pointer check)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2), sharey=True)
    tmpl_colors = {"food_truck": BLUE, "hair_styling": ORANGE,
                   "library_book": PURPLE}
    for ax, (cond, qlabel, q) in zip(
            axes, [("abstract_belief", "C1 heads, final-token query", "lookback"),
                   ("answer_changes_location_only", "C5 heads, final-token query", "lookback")]):
        g = all_rows[all_rows.condition == cond]
        x = np.arange(len(SOURCES))
        w = 0.26
        for j, (tmpl, color) in enumerate(tmpl_colors.items()):
            prof = profile(g[g.template == tmpl], q)
            ax.bar(x + (j - 1) * w, prof, w * 0.92, color=color, label=tmpl)
        ax.set_xticks(x)
        ax.set_xticklabels([SRC_LABEL[s] for s in SOURCES], rotation=28,
                           ha="right", fontsize=8)
        ax.set_title(qlabel, fontsize=10)
    axes[0].set_ylabel("CMA-weighted attention mass")
    axes[0].legend(fontsize=8.5, frameon=False, title="template", title_fontsize=8.5)
    fig.suptitle("Template split — does the attention target move with the story?",
                 fontsize=11, y=1.02)
    fig.savefig(FIG / "fig3_template_split.png")
    plt.close(fig)

    # selectivity stat: correct vs incorrect state token, lookback query
    sel_rows = []
    for cond, g in all_rows.groupby("condition"):
        corr = wmean(g, "lookback", "state_correct")
        total = wmean(g, "lookback", "state_first") + wmean(g, "lookback", "state_second")
        incorrect = max(total - corr, 1e-6)
        sel_rows.append({"condition": cond, "C": CNUM[cond],
                         "attn_correct": corr, "attn_incorrect": incorrect,
                         "selectivity": corr / incorrect})
    df_sel = pd.DataFrame(sel_rows).sort_values("C")
    df_sel.to_csv(OUT / "stats_selectivity.csv", index=False)

    frag = ["<!-- AUTO-GENERATED by scripts/lookback_analysis/build_attention_position_bars.py -->\n",
            "### Correct-vs-incorrect state-token selectivity (final-token query)\n",
            df_sel.round(3).to_markdown(index=False) + "\n",
            "\n### Pooled weighted attention (all conditions, queries, sources)\n",
            tidy[tidy.template == "pooled"].round(3).to_markdown(index=False) + "\n"]
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    print("== selectivity (lookback query) ==")
    print(df_sel.round(3).to_string(index=False))
    print("\n== C1 profiles ==")
    for q in ("lookback", "belief_formation"):
        print(f"{q}: " + ", ".join(f"{s}={v:.3f}" for s, v in
                                   zip(SOURCES, profile(g_c1, q))))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
