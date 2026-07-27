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
CONDS_BY_NUM = sorted(CONDS, key=lambda c: int(CNUM[c][1:]))  # panel order C1..C7
QUERIES = ["lookback", "belief_formation", "agent_last", "agent_first"]
SOURCES = ["state_first", "state_second", "state_correct", "info_loss",
           "char_q", "obj_q", "belief_form"]
SRC_LABEL = {"state_first": "1st location", "state_second": "2nd location",
             "state_correct": "correct answer", "info_loss": "info-loss event",
             "char_q": "agent (question)", "obj_q": "object (question)",
             "belief_form": "belief formation"}
PURPLE = "#CC79A7"  # Okabe-Ito, for the 3rd template


# Prompt type of each cell's *base* prompts. The captures are forward passes on
# base prompts only (capture_attention_patterns.py:319), so a cell is identified
# by the epistemic state of the prompt the model read, not by a patch direction.
# C5-C7 hold prompt type fixed by design: their two cells differ only in the
# location substitutions and the head set, which makes them a built-in control.
PROMPT_TYPE = {
    ("abstract_belief", "ABA"): "false belief",
    ("abstract_belief", "ABB"): "true belief",
    ("answer_changes_belief", "ABA"): "false belief",
    ("answer_changes_belief", "ABB"): "true belief",
    ("abstract_photo", "ABA"): "outdated photo",
    ("abstract_photo", "ABB"): "current photo",
    ("answer_changes_photo", "ABA"): "outdated photo",
    ("answer_changes_photo", "ABB"): "current photo",
    ("answer_changes_location_only", "ABA"): "false belief (a)",
    ("answer_changes_location_only", "ABB"): "false belief (b)",
    ("answer_changes_location_only_photo", "ABA"): "outdated photo (a)",
    ("answer_changes_location_only_photo", "ABB"): "outdated photo (b)",
    ("state_only_no_belief", "ABA"): "no belief (a)",
    ("state_only_no_belief", "ABB"): "no belief (b)",
}
# Which location slot holds the correct answer, per prompt type. Asserted against
# the data in check_answer_slots() rather than assumed.
EXPECTED_SLOT = {"false belief": "state_first", "outdated photo": "state_first",
                 "true belief": "state_second", "current photo": "state_second",
                 "no belief": "state_second"}


def load_rows(significant_only=True):
    """All captured head rows. The capture kept each cell's top-50 CMA heads
    regardless of significance; the non-significant ones are retained here to
    serve as the null pool in perm_test_vs_nonsignificant()."""
    rows = []
    for cond in CONDS:
        for csv in sorted(LB_ROOT.glob(
                f"{cond}/template_*/base_rule_*/patch_before_movement/lookback_scores.csv")):
            df = pd.read_csv(csv)
            df["sig"] = df["significant"].astype(str).str.lower() == "true"
            if significant_only:
                df = df[df["sig"]].copy()
            if df.empty:
                continue
            df["w"] = df["observed_score"].clip(lower=0)
            df["condition"] = cond
            df["template"] = csv.parts[-4].replace("template_", "")
            df["base_rule"] = csv.parts[-3].replace("base_rule_", "").split("_exp")[0]
            df["cell"] = df["template"] + "/" + csv.parts[-3]
            df["prompt_type"] = PROMPT_TYPE[(cond, df["base_rule"].iloc[0])]
            rows.append(df)
    return pd.concat(rows, ignore_index=True)


def check_answer_slots(all_rows):
    """state_correct is an alias of state_first or state_second within a cell.
    Verify that the alias matches the prompt type declared in PROMPT_TYPE, so the
    prompt-type labels on every figure are checked against the data."""
    bad = []
    for (cond, cell, ptype), g in all_rows.groupby(
            ["condition", "cell", "prompt_type"]):
        c = g["lookback_attn_to_state_correct"]
        if c.isna().all():
            continue
        want = EXPECTED_SLOT[ptype.split(" (")[0]]
        hit = np.isclose(c, g[f"lookback_attn_to_{want}"], equal_nan=True).mean()
        if hit < 1.0:
            bad.append(f"{cond} {cell} ({ptype}): only {hit:.0%} of rows have "
                       f"state_correct == {want}")
    if bad:
        raise AssertionError("prompt-type / answer-slot mismatch:\n  " +
                             "\n  ".join(bad))
    print("answer-slot check: state_correct matches the prompt type's expected "
          "slot in 100% of rows, all cells")


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


# ---------- inferential statistics ----------
#
# Two questions, two nulls. Neither was addressed by the CMA permutation test,
# which tested head *selection*, not the attention values of the selected heads.
#
# (1) How precise is a plotted bar / a selectivity ratio?  Cluster bootstrap over
#     heads: resample distinct (layer, head) identities with replacement within a
#     condition, carrying all of that head's rows (a head recurs across the
#     condition's six cells, so rows are not independent), recompute the weighted
#     statistic, and take percentile intervals. A selectivity CI excluding 1
#     means the head set's preference for the correct location is resolvable at
#     this sample size.
#
# (2) Do CMA-significant heads attend differently from other heads?  Permutation
#     against the captured-but-not-significant heads of the same cells. H0: the
#     attention profile of a significant head is exchangeable with that of any
#     other captured head in its cell. Each draw replaces the n_k significant
#     rows of cell k with n_k rows sampled from that cell's non-significant rows
#     and reuses the observed weights, so the weighting scheme is held fixed and
#     only head identity is permuted. One-sided (upper) p, add-one corrected.
#     Note this is conservative: the top-50 capture pool is itself CMA-enriched,
#     so the null pool is not a neutral sample of the 1920 heads.

BOOT_B = 10000
PERM_B = 5000


def _selectivity(g):
    corr = wmean(g, "lookback", "state_correct")
    tot = (wmean(g, "lookback", "state_first") +
           wmean(g, "lookback", "state_second"))
    inc = max(tot - corr, 1e-6)
    return corr, inc, corr / inc


def bootstrap_condition(g, rng):
    """Percentile CIs for the final-token correct/incorrect values, the
    selectivity ratio, and the two bf-query quantities."""
    heads = g[["layer", "head"]].drop_duplicates().to_numpy()
    # NB: g["head"], not g.head — the attribute is DataFrame.head, the method.
    by_head = {(l, h): g[(g["layer"] == l) & (g["head"] == h)] for l, h in heads}
    keys = list(by_head)
    stats = {k: [] for k in ("correct", "incorrect", "selectivity",
                             "bf_to_belief_form", "bf_to_info_loss")}
    for _ in range(BOOT_B):
        draw = [by_head[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        gb = pd.concat(draw, ignore_index=True)
        corr, inc, sel = _selectivity(gb)
        stats["correct"].append(corr)
        stats["incorrect"].append(inc)
        stats["selectivity"].append(sel)
        stats["bf_to_belief_form"].append(wmean(gb, "belief_formation", "belief_form"))
        stats["bf_to_info_loss"].append(wmean(gb, "belief_formation", "info_loss"))
    out = {}
    for k, v in stats.items():
        v = np.asarray(v, dtype=float)
        v = v[np.isfinite(v)]
        out[k] = (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v.size else (np.nan, np.nan)
    return out


def perm_test_vs_nonsignificant(raw, cond, query, source, rng):
    """H0: significant heads are exchangeable with other captured heads of the
    same cell. Returns (observed, null median, p, n_cells_resampled_with_repl)."""
    col = f"{query}_attn_to_{source}"
    g = raw[raw.condition == cond]
    if col not in g:
        return np.nan, np.nan, np.nan, 0
    cells, short, onum, oden = [], 0, 0.0, 0.0
    for cell, gc in g.groupby("cell"):
        sig = gc[gc.sig & gc[col].notna() & (gc["w"] > 0)]
        nul = gc[(~gc.sig) & gc[col].notna()]
        if sig.empty or nul.empty:
            continue
        w, vals = sig["w"].to_numpy(), sig[col].to_numpy()
        onum += float(np.sum(w * vals))
        oden += float(np.sum(w))
        short += int(len(nul) < len(sig))
        cells.append((w, nul[col].to_numpy(), len(nul) < len(sig)))
    if not cells or oden == 0:
        return np.nan, np.nan, np.nan, 0
    obs = onum / oden  # same support as the null draws, so the two are comparable
    null = []
    for _ in range(PERM_B):
        num = den = 0.0
        for w, pool, with_repl in cells:
            idx = (rng.integers(0, len(pool), len(w)) if with_repl
                   else rng.choice(len(pool), len(w), replace=False))
            vals = pool[idx]
            m = w > 0
            num += float(np.sum(w[m] * vals[m]))
            den += float(np.sum(w[m]))
        null.append(num / den if den > 0 else np.nan)
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    p = (1 + int((null >= obs).sum())) / (len(null) + 1)
    return obs, float(np.median(null)), p, short


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
    raw_rows = load_rows(significant_only=False)   # includes the null pool
    all_rows = raw_rows[raw_rows.sig].reset_index(drop=True)
    check_answer_slots(all_rows)

    # tidy long table: condition x template(+pooled) x query x source
    recs = []
    for cond, g in all_rows.groupby("condition"):
        groups = ([("pooled", g)] + [(t, gt) for t, gt in g.groupby("template")]
                  + [(f"ptype:{p}", gp) for p, gp in g.groupby("prompt_type")])
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
    for ax, cond in zip(axes.ravel(), CONDS_BY_NUM):
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

    # F2b — location slots split by prompt type. Within a prompt type the
    # correct answer sits on a fixed slot, so the pooling artifact that makes
    # F2's green bar differ from both location bars cannot arise here, and the
    # answer-tracking vs serial-position question is readable directly.
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.0))
    for ax, cond in zip(axes.ravel(), CONDS_BY_NUM):
        g = all_rows[all_rows.condition == cond]
        ptypes = sorted(g.prompt_type.unique(),
                        key=lambda t: 0 if "(a)" in t or t in
                        ("false belief", "outdated photo") else 1)
        xs, heights, colors, edges, labels = [], [], [], [], []
        for j, pt in enumerate(ptypes):
            gp = g[g.prompt_type == pt]
            want = EXPECTED_SLOT[pt.split(" (")[0]]
            for k, slot in enumerate(["state_first", "state_second"]):
                xs.append(j * 2.6 + k)
                heights.append(wmean(gp, "lookback", slot))
                is_ans = slot == want
                colors.append(GREEN if is_ans else GRAY)
                edges.append("black" if is_ans else "none")
                labels.append("original" if slot == "state_first" else "current")
        ax.bar(xs, heights, 0.9, color=colors, edgecolor=edges, linewidth=1.4)
        for xi, v in zip(xs, heights):
            if np.isfinite(v) and v > 0.004:
                ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
        for j, pt in enumerate(ptypes):
            ax.text(j * 2.6 + 0.5, -0.30, pt, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
        ax.set_ylim(0, max(0.05, np.nanmax(heights) * 1.30))
        ax.set_title(f"{CNUM[cond]} · {cond}", fontsize=9)
        ax.margins(x=0.06)
    axes[1, 3].axis("off")
    axes[1, 3].legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=GREEN, edgecolor="black",
                      label="location holding the correct answer"),
        plt.Rectangle((0, 0), 1, 1, facecolor=GRAY,
                      label="the other location")],
        loc="center", frameon=False, fontsize=9)
    for r in (0, 1):
        axes[r, 0].set_ylabel("weighted attention")
    fig.suptitle("Final-token attention to the two location tokens, split by prompt type "
                 "(note the free y-axis per panel)", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(FIG / "fig2b_prompt_type_split.png", dpi=150)
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

    # --- (1) cluster bootstrap over heads: CIs on the plotted quantities ---
    rng = np.random.default_rng(0)
    ci_rows = []
    for cond, g in all_rows.groupby("condition"):
        ci = bootstrap_condition(g, rng)
        ci_rows.append({"condition": cond, "C": CNUM[cond],
                        "n_heads": g[["layer", "head"]].drop_duplicates().shape[0],
                        **{f"{k}_lo": v[0] for k, v in ci.items()},
                        **{f"{k}_hi": v[1] for k, v in ci.items()}})
    df_ci = pd.DataFrame(ci_rows).sort_values("C")
    df_sel = df_sel.merge(df_ci[["condition", "n_heads", "selectivity_lo",
                                 "selectivity_hi", "correct_lo", "correct_hi"]],
                          on="condition")
    df_sel["sel_ci_excludes_1"] = ~((df_sel.selectivity_lo <= 1) &
                                    (df_sel.selectivity_hi >= 1))
    df_sel.to_csv(OUT / "stats_selectivity.csv", index=False)
    df_ci.to_csv(OUT / "stats_bootstrap_ci.csv", index=False)

    # --- (2) permutation vs captured-but-not-significant heads ---
    perm_rows = []
    for cond in CONDS:
        for q, s in [("lookback", "state_correct"), ("lookback", "state_second"),
                     ("belief_formation", "belief_form"),
                     ("belief_formation", "info_loss")]:
            obs, nullmed, p, short = perm_test_vs_nonsignificant(
                raw_rows, cond, q, s, rng)
            perm_rows.append({"condition": cond, "C": CNUM[cond], "query": q,
                              "source": s, "observed": obs,
                              "null_median": nullmed, "p_one_sided": p,
                              "cells_pool_smaller_than_n_sig": short})
    df_perm = pd.DataFrame(perm_rows).sort_values(["query", "source", "C"])
    df_perm.to_csv(OUT / "stats_permutation_vs_nonsig.csv", index=False)

    sel_cols = ["C", "attn_correct", "attn_incorrect", "selectivity",
                "selectivity_lo", "selectivity_hi", "sel_ci_excludes_1", "n_heads"]
    frag = ["<!-- AUTO-GENERATED by scripts/lookback_analysis/build_attention_position_bars.py -->\n",
            "### Correct-vs-incorrect state-token selectivity (final-token query)\n",
            "95% percentile CIs from a cluster bootstrap over heads "
            f"(B = {BOOT_B}, resampling distinct (layer, head) identities within a "
            "condition and carrying all of that head's cell rows).\n",
            df_sel[sel_cols].round(3).to_markdown(index=False) + "\n",
            "\n### Permutation test against captured non-significant heads\n",
            "H0: a CMA-significant head's attention is exchangeable with that of "
            "any other captured head in the same cell. Each of the "
            f"{PERM_B} draws swaps the significant rows of every cell for "
            "non-significant rows of that cell and reuses the observed weights; "
            "p is one-sided (upper), add-one corrected. Conservative: the top-50 "
            "capture pool is itself CMA-enriched.\n",
            df_perm.round(4).to_markdown(index=False) + "\n",
            "\n### Weighted attention by prompt type (final-token query)\n",
            tidy[tidy.template.str.startswith("ptype:") &
                 tidy["query"].eq("lookback") &
                 tidy.source.isin(["state_first", "state_second", "state_correct"])]
            .round(3).to_markdown(index=False) + "\n",
            "\n### Pooled weighted attention (all conditions, queries, sources)\n",
            tidy[tidy.template == "pooled"].round(3).to_markdown(index=False) + "\n"]
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    print("\n== selectivity with bootstrap CI ==")
    print(df_sel[sel_cols].round(3).to_string(index=False))
    print("\n== permutation vs non-significant heads ==")
    print(df_perm.round(4).to_string(index=False))

    print("== selectivity (lookback query) ==")
    print(df_sel.round(3).to_string(index=False))
    print("\n== C1 profiles ==")
    for q in ("lookback", "belief_formation"):
        print(f"{q}: " + ", ".join(f"{s}={v:.3f}" for s, v in
                                   zip(SOURCES, profile(g_c1, q))))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
