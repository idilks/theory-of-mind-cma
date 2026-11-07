"""Which words of the state change do divergence and retrieval heads read?

Token roles inside the state-change sentence (story: the sentence that moves the object;
code: the a[m] = 'X' line). 2 x 3 grid: prompt type x domain; bars = CMA-weighted mean
share of final-token attention per role for the divergence and retrieval groups; dashed
tick = uniform attention for that role's tokens; error bars = 95% bootstrap over heads.

Roles come from region_attention_shares.py (story, share_role_*) and
code_region_shares.py (code). A role absent from a template (no preposition in
hair_styling: "dyes her hair maroon") is left out of that template's average.
Run: python analysis_scripts/lookback_analysis/render_move_roles.py
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_groups import story_head_group, code_head_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REG = ROOT / "results/lookback_analysis/_regions"
OUT = ROOT / "results/summary/aggregation"
STEM = OUT / "move_sentence_roles"
DIV, RET, INK, MUTED = "#E07B39", "#2368A2", "#172D40", "#526577"
B, SEED = 5000, 0

STORY_ROLES = [("agent", "Agent\n(Bob)"), ("verb", "Verb\n(drives)"), ("object", "Object\n(the truck)"),
               ("prep", "Preposition\n(to)"), ("newloc", "New location\n(Garden St.)"), ("other", "Other\nwords")]
CODE_ROLES = [("var", "List\n(a)"), ("index", "Index\n(3)"), ("assign", "Assign\n(=)"),
              ("quote", "Opening\nquote (')"), ("newloc", "New value\n(M)"), ("other", "Brackets,\nclosing quote")]
DOMAINS = [("Belief", "story", "abstract_belief", "abstract_belief", "answer_changes_location_only"),
           ("Photo", "story", "abstract_photo", "abstract_photo", "answer_changes_location_only_photo"),
           ("Code", "code", None, "code_abstract", "code_control")]
ROWS = [("FB", "False belief", "ABA", "copy"), ("TB", "True belief", "ABB", "alias")]


def heads(source, pcond, group, base_rule, queried, roles):
    if source == "story":
        d = pd.read_csv(REG / "region_shares_matched.csv")
        if "share_role_newloc" not in d:
            return None
        d = d[(d.condition == pcond) & (d.base_rule == base_rule)
              & d.groups.str.split("+").apply(lambda g: group in g)].copy()
        g = story_head_group(ROOT / "results", group)
    else:
        d = pd.read_csv(REG / "code_region_shares.csv")
        d = d[(d.condition == group) & (d.queried == queried)].copy()
        g = code_head_group(group)
    for r, _ in roles:
        d.loc[d[f"frac_role_{r}"] == 0, [f"frac_role_{r}", f"share_role_{r}"]] = np.nan
    cols = [f"{a}_role_{r}" for r, _ in roles for a in ("share", "frac")]
    h = d.groupby(["layer", "head"])[cols].mean().reset_index()
    return h.merge(g[["layer", "head", "mean_score"]], on=["layer", "head"])


def mean_ci(h, col, rng):
    h = h[h[col].notna()]
    w, v = h.mean_score.clip(lower=0).to_numpy(), h[col].to_numpy()
    idx = rng.integers(0, len(v), size=(B, len(v)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (w[idx] * v[idx]).sum(1) / w[idx].sum(1)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return float((w * v).sum() / w.sum()), float(lo), float(hi)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for pt, _, br, q in ROWS:
        for dom, src, pc, div, ret in DOMAINS:
            roles = CODE_ROLES if src == "code" else STORY_ROLES
            for gname, grp in (("divergence", div), ("retrieval", ret)):
                h = heads(src, pc, grp, br, q, roles)
                if h is None:
                    continue
                for r, _ in roles:
                    m, lo, hi = mean_ci(h, f"share_role_{r}", rng)
                    rows.append(dict(prompt_type=pt, domain=dom, group=gname, role=r, mean=m, ci_lo=lo, ci_hi=hi,
                                     uniform=float(h[f"frac_role_{r}"].mean()), n_heads=len(h)))
    tab = pd.DataFrame(rows)
    doms = [d for d in DOMAINS if d[0] in set(tab.domain)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, len(doms), figsize=(6.6 * len(doms), 10.5), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.17, top=0.80, wspace=0.12, hspace=0.5)
    fig.suptitle("Which words of the state change do the heads read?", x=0.08, y=0.985, ha="left",
                 fontsize=18, fontweight="bold")
    fig.text(0.08, 0.925, "Qwen2.5-14B-Instruct  ·  attention from the final prompt token to each word of the "
             "state-change sentence (code: the a[m] = 'X' line)  ·  both head groups on the same prompts",
             fontsize=10, color=MUTED)
    fig.legend(handles=[Patch(facecolor=DIV, label="Divergence heads"), Patch(facecolor=RET, label="Retrieval heads"),
                        Line2D([0], [0], color=MUTED, ls=(0, (3, 2)), lw=1.4, label="Uniform attention")],
               loc="upper left", bbox_to_anchor=(0.074, 0.905), ncol=3, frameon=False)
    for i, (pt, title, *_r) in enumerate(ROWS):
        for j, (dom, src, *_d) in enumerate(doms):
            ax = axes[i, j]
            roles = CODE_ROLES if src == "code" else STORY_ROLES
            sub = tab[(tab.prompt_type == pt) & (tab.domain == dom)]
            ax.set_title(f"{dom}  ·  {title.lower()}", loc="left", fontsize=13, fontweight="bold", pad=10)
            for x, (r, _) in enumerate(roles):
                for gname, off, color in (("divergence", -0.19, DIV), ("retrieval", 0.19, RET)):
                    row = sub[(sub.group == gname) & (sub.role == r)].iloc[0]
                    ax.bar(x + off, row["mean"], width=0.34, color=color, zorder=3)
                    ax.errorbar(x + off, row["mean"], yerr=[[row["mean"] - row.ci_lo], [row.ci_hi - row["mean"]]],
                                fmt="none", color=INK, capsize=3, linewidth=1.0, zorder=4)
                    ax.text(x + off, row.ci_hi + 0.008, f"{row['mean']:.0%}" if row["mean"] >= 0.095 else f"{row['mean']:.1%}",
                            ha="center", va="bottom", fontsize=8.5, fontweight="bold")
                uni = sub[sub.role == r].uniform.iloc[0]
                ax.plot([x - 0.42, x + 0.42], [uni, uni], color=MUTED, ls=(0, (3, 2)), lw=1.4, zorder=5)
            ax.set_xticks(range(len(roles)), [lab for _, lab in roles], fontsize=9)
            ax.tick_params(length=0)
            ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.grid(axis="y", color="#E2E7EB", linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.set_xlim(-0.6, len(roles) - 0.4)
        axes[i, 0].set_ylabel("Share of the final token's attention", labelpad=12)
    notes = ("Bars: CMA-weighted mean over each head group (belief/photo: 3 templates × 150 prompts; code: 24 prompts); "
             "error bars: 95% bootstrap over heads. Story examples in the labels are from food_truck; the verbs are "
             "drives / dyes / returns, moves; objects truck / hair / book, it; hair_styling has no preposition and is left "
             "out of that bar. The new location / value is the same quantity as 'location 2' in head_attention_simple.")
    for k, part in enumerate(textwrap.wrap(notes, 190)):
        fig.text(0.08, 0.09 - 0.022 * k, part, fontsize=9.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(STEM.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    tab.to_csv(STEM.with_suffix(".csv"), index=False)
    print(tab.round(3).to_string(index=False))
    print(f"Saved {STEM}.{{png,pdf,csv}}  domains: {[d[0] for d in doms]}")


if __name__ == "__main__":
    main()
