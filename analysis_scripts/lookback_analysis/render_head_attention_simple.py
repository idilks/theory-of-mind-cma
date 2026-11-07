"""Where divergence and retrieval heads attend: the slide-ready summary of the region analysis.

2 x 3 grid: prompt type (false belief / true belief) x domain (belief / photo / code).
Each panel: mean final-token attention of the divergence and retrieval head groups on
the correct-answer location, the other location, and the question. A dashed tick per
region marks what uniform attention would give (the region's share of the prompt's
tokens). Error bars: 95% bootstrap intervals over heads (heads resampled within a
group, CMA-weighted mean recomputed; 5,000 resamples). The region CSVs store per-head
means over prompts, so prompts are not resampled.

Sources (both head groups measured on the same prompts):
  results/lookback_analysis/_regions/region_shares_matched.csv  belief / photo
      (lookback_capture_matched.sh -> region_attention_shares.py); head groups from its
      `groups` column, i.e. exactly the heads the capture used
  results/lookback_analysis/_regions/code_region_shares.csv     code
      (capture_code_attention_rows.sh -> code_region_shares.py); K1 / K5 head lists

The full per-head view is region_bubble_matrix.py.
Run: python analysis_scripts/lookback_analysis/render_head_attention_simple.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_groups import story_head_group, code_head_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "results/lookback_analysis/_regions"
OUT = ROOT / "results/summary/aggregation"
STEM = OUT / "head_attention_simple"

DIV = "#E07B39"
RET = "#2368A2"
INK = "#172D40"
MUTED = "#526577"
B = 5000
SEED = 0

# (panel title, source, prompt condition, divergence group, retrieval group)
DOMAINS = [("Belief", "story", "abstract_belief", "abstract_belief", "answer_changes_location_only"),
           ("Photo", "story", "abstract_photo", "abstract_photo", "answer_changes_location_only_photo"),
           ("Code", "code", None, "code_abstract", "code_control")]
# (key, title, story base rule, code query, x-index of the correct-answer location)
ROWS = [("FB", "False belief", "ABA", "copy", 0),
        ("TB", "True belief", "ABB", "alias", 1)]
# (key, source region, story tick label, code tick label)
REGIONS = [("loc1", "orig_loc", "Location 1\n(original)", "Value 1\n(original)"),
           ("loc2", "current_loc", "Location 2\n(updated)", "Value 2\n(updated)"),
           ("change", "change_context", "State-change\nsentence", "a[m] = ...\nline"),
           ("question", "question", "Question", "Question"),
           ("end", "end", "End of\nprompt", "End of\nprompt")]
ANSWER_TINT = "#EAF3DE"
# stacked segments of the end-of-prompt bar, bottom to top: (column, label, colour lightening)
END_PARTS = [("end_self", "final token", 0.0), ("end_chat", "chat-template tokens", 0.45),
             ("end_instruction", "answer instruction", 0.75)]


def lighten(hex_color: str, amount: float) -> tuple:
    rgb = np.array(matplotlib.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * amount)


def head_table(domain, source, pcond, group, base_rule, queried, answer):
    """Per-head shares and token fractions for one group on one prompt type."""
    if source == "story":
        d = pd.read_csv(REG / "region_shares_matched.csv")
        d = d[(d.condition == pcond) & (d.base_rule == base_rule)
              & d.groups.str.split("+").apply(lambda g: group in g)]
    else:
        d = pd.read_csv(REG / "code_region_shares.csv")
        d = d[(d.condition == group) & (d.queried == queried)]
    d = d.copy()
    # a region with no tokens in a cell (belief hair_styling has no separate state-change
    # sentence) is left out of that region's average instead of counting as zero
    for c in [c for c in d.columns if c.startswith("frac_")]:
        d.loc[d[c] == 0, [c, "share_" + c[5:]]] = np.nan
    cols = [c for c in d.columns if c.startswith(("share_", "frac_"))]
    h = d.groupby(["layer", "head"]).agg(**{c: (c, "mean") for c in cols},
                                         n_cells=("layer", "size")).reset_index()
    # weight = the head's CMA score in THIS group (the capture stored the max over groups)
    g = story_head_group(ROOT / "results", group) if source == "story" else code_head_group(group)
    assert set(zip(g.layer, g["head"])) == set(zip(h.layer, h["head"])), \
        f"{group}: local head group differs from the captured one"
    h = h.merge(g[["layer", "head", "mean_score"]], on=["layer", "head"]).rename(columns={"mean_score": "score"})
    for key, reg, *_ in REGIONS:
        h[key] = h[f"share_{reg}"]
        h[f"uniform_{key}"] = h[f"frac_{reg}"]
    for part, *_ in END_PARTS:  # absent in region CSVs written before the end split
        h[part] = h[f"share_{part}"] if f"share_{part}" in h else np.nan
    return h


def weighted_mean_ci(h: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    h = h[h[col].notna()]
    w = h.score.clip(lower=0).to_numpy()
    v = h[col].to_numpy()
    mean = float((w * v).sum() / w.sum())
    idx = rng.integers(0, len(v), size=(B, len(v)))
    ww, vv = w[idx], v[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (ww * vv).sum(1) / ww.sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return mean, float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for pt, _, base_rule, queried, answer in ROWS:
        for domain, source, pcond, div, ret in DOMAINS:
            for gname, group in (("divergence", div), ("retrieval", ret)):
                h = head_table(domain, source, pcond, group, base_rule, queried, answer)
                w = h.score.clip(lower=0)
                parts = {p: float((w * h[p]).sum() / w.sum()) if h[p].notna().all() else np.nan
                         for p, *_ in END_PARTS}
                for key, *_ in REGIONS:
                    m, lo, hi = weighted_mean_ci(h, key, rng)
                    uni = float(h[f"uniform_{key}"].mean())
                    rows.append(dict(prompt_type=pt, domain=domain, group=gname, region=key,
                                     mean=m, ci_lo=lo, ci_hi=hi, uniform=uni, ratio_to_uniform=m / uni,
                                     **({p: v for p, v in parts.items()} if key == "end" else {}),
                                     n_heads=len(h), heads=" ".join(f"L{a}H{b}" for a, b in zip(h.layer, h["head"]))))
    tab = pd.DataFrame(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12.0), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.15, top=0.78, wspace=0.10, hspace=0.45)
    fig.suptitle("Where divergence and retrieval heads attend", x=0.07, y=0.985, ha="left",
                 fontsize=18, fontweight="bold")
    n = {(r.domain, r.group): r.n_heads for r in tab.itertuples()}
    fig.text(0.07, 0.915,
             "Qwen2.5-14B-Instruct  ·  attention from the final prompt token  ·  both head groups measured on the same prompts\n"
             f"heads per group (divergence / retrieval): belief {n[('Belief', 'divergence')]} / {n[('Belief', 'retrieval')]}, "
             f"photo {n[('Photo', 'divergence')]} / {n[('Photo', 'retrieval')]}, code {n[('Code', 'divergence')]} / {n[('Code', 'retrieval')]}",
             fontsize=10, color=MUTED)
    fig.legend(handles=[Patch(facecolor=DIV, label="Divergence heads (track whether the state changed)"),
                        Patch(facecolor=RET, label="Retrieval heads (carry the answer)"),
                        Line2D([0], [0], color=MUTED, ls=(0, (3, 2)), lw=1.4, label="Uniform attention over the prompt")]
               + [Patch(facecolor=lighten("#6B7C8C", light), edgecolor="white", label=f"End of prompt: {lab}")
                  for _, lab, light in END_PARTS],
               loc="upper left", bbox_to_anchor=(0.064, 0.885), ncol=3, frameon=False,
               handlelength=1.4, columnspacing=2.2)

    domain_sub = {"Belief": "", "Photo": "", "Code": ""}
    row_sub = {"FB": {"Belief": "the agent misses the move: location 1 is the answer",
                      "Photo": "photo taken before the move: location 1 is the answer",
                      "Code": "copy queries (b = a.copy()): value 1 is the answer"},
               "TB": {"Belief": "the agent sees the move: location 2 is the answer",
                      "Photo": "photo taken after the move: location 2 is the answer",
                      "Code": "alias queries (c = a): value 2 is the answer"}}
    for i, (pt, pt_title, *_rest) in enumerate(ROWS):
        for j, (domain, *_d) in enumerate(DOMAINS):
            ax = axes[i, j]
            sub = tab[(tab.prompt_type == pt) & (tab.domain == domain)]
            ax.set_title(f"{domain}  ·  {pt_title.lower()}", loc="left", fontsize=13, fontweight="bold", pad=22)
            ax.text(0, 1.035, row_sub[pt][domain], transform=ax.transAxes, fontsize=9.5, color=MUTED)
            ax.axvspan(ROWS[i][4] - 0.47, ROWS[i][4] + 0.47, color=ANSWER_TINT, zorder=0)
            ax.text(ROWS[i][4], 0.945, "correct answer", ha="center", va="top", fontsize=9,
                    color="#27500A", fontweight="bold")
            for x, (key, *_) in enumerate(REGIONS):
                for gname, off, color in (("divergence", -0.19, DIV), ("retrieval", 0.19, RET)):
                    r = sub[(sub.group == gname) & (sub.region == key)].iloc[0]
                    if key == "end" and not np.isnan(r.get("end_self", np.nan)):
                        bottom = 0.0
                        for part, _, light in END_PARTS:
                            ax.bar(x + off, r[part], bottom=bottom, width=0.34, color=lighten(color, light),
                                   edgecolor="white", linewidth=0.6, zorder=3)
                            bottom += r[part]
                    else:
                        ax.bar(x + off, r["mean"], width=0.34, color=color, zorder=3,
                               hatch="//" if key == "end" else None, edgecolor="white" if key == "end" else None)
                    ax.errorbar(x + off, r["mean"], yerr=[[r["mean"] - r.ci_lo], [r.ci_hi - r["mean"]]],
                                fmt="none", color=INK, capsize=3, linewidth=1.0, zorder=4)
                    ax.text(x + off, r.ci_hi + 0.012, f"{r['mean']:.0%}" if r["mean"] >= 0.095 else f"{r['mean']:.1%}",
                            ha="center", va="bottom", fontsize=9.5, fontweight="bold")
                uni = sub[sub.region == key].uniform.iloc[0]
                ax.plot([x - 0.42, x + 0.42], [uni, uni], color=MUTED, ls=(0, (3, 2)), lw=1.4, zorder=5)
            ax.set_xticks(range(len(REGIONS)), [r[3] if domain == "Code" else r[2] for r in REGIONS])
            ax.tick_params(axis="x", length=0, pad=8)
            ax.tick_params(axis="y", length=0, pad=6)
            ax.set_ylim(0, 0.95)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="y", color="#E2E7EB", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#B4BFC8")
            ax.set_xlim(-0.6, len(REGIONS) - 0.4)
        axes[i, 0].set_ylabel("Share of the final token's attention", labelpad=12)

    notes = [
        "Bars: CMA-weighted mean over the heads of each group, averaged over 3 story templates × 150 prompts (code: 24 prompts). "
        "Error bars: 95% bootstrap intervals over heads (5,000 resamples); wide bars mean the heads in a group differ, not that prompts are noisy.",
        "Locations count every token of every mention of the location name (code: the value token). State-change sentence = the sentence "
        "that moves the object, minus the new location itself (code: the a[m] = ... line minus the new value); belief averages "
        "food_truck and library_book only, since in hair_styling that sentence is also the access event. "
        "Belief and photo heads: CMA-significant in at least half of the condition's cells (C1/C5, C2/C6; seed 1); code: K1/K5 head lists.",
        "Each head is weighted by its CMA score in its own group; negative-score heads get weight 0. End of prompt = the answer instruction after "
        "the question (none in code prompts), the chat-template tokens <|im_end|> <|im_start|>assistant, and the final token itself. "
        "The rest of the attention goes to the rest of the story or code and the system turn, each at or below uniform. Per-head view: bubble_matrix_FB / _TB.",
    ]
    if tab[tab.region == "end"].end_self.isna().any():
        missing = sorted(tab[(tab.region == "end") & tab.end_self.isna()].domain.unique())
        notes.append(f"Hatched end-of-prompt bars ({', '.join(missing)}): the region CSV predates the end split; "
                     "rerun slurm_scripts/region_attention_shares.sh to split them.")
    import textwrap
    y = 0.088
    for line in notes:
        for part in textwrap.wrap(line, 175):
            fig.text(0.07, y, part, fontsize=9.5, color=MUTED)
            y -= 0.019

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(STEM.with_suffix(f".{ext}"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    tab.to_csv(STEM.with_suffix(".csv"), index=False)
    pd.set_option("display.width", 200)
    print(tab.drop(columns="heads").round(3).to_string(index=False))
    print(f"Saved {STEM}.{{png,pdf,svg,csv}}")


if __name__ == "__main__":
    main()
