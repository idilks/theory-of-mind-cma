"""
Bubble matrix: where do divergence heads and retrieval heads attend, in three domains,
for one prompt type.

  --prompt_type FB   false-belief / outdated photo / copy queries: the ORIGINAL
                     location (value) is the correct answer
  --prompt_type TB   true-belief / current photo / alias queries: the UPDATED
                     location (value) is the correct answer

The column holding the correct answer is tinted. Rows are heads (plus a
CMA-weighted set mean on top of each block); columns are prompt regions aligned
across domains. Bubble AREA = attention relative to uniform: the head's share of
final-token attention in the region divided by the region's share of the prompt's
tokens (1x = no preference), area proportional to ratio**0.6. Filled = above 1x,
hollow = below. The set-mean row prints the ratio where it is >= 2.

Both head groups are measured on the SAME prompts (matched captures):
  belief  C1 (divergence) and C5 (retrieval) heads on C1 prompts
  photo   C2 (divergence) and C6 (retrieval) heads on C2 prompts
  code    K1 (divergence) and K5 (retrieval) heads on the unpadded padtwins stimuli,
          queries of the mutated slot
Head groups are fixed across prompt types (head_groups.py): story heads significant
in >= half of the condition's CMA cells; code heads from the published lists.
* = head in both groups of its domain.

Inputs:
  results/lookback_analysis/_regions/region_shares_matched.csv  region_attention_shares.py (HPC)
  results/lookback_analysis/_regions/code_region_shares.csv     code_region_shares.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from head_groups import story_head_group, code_head_group  # noqa: E402

REGIONS = ["orig_loc", "current_loc", "orig_context", "access_cue", "change_context",
           "other_story", "question", "system", "end"]
COLS = {
    "Belief": ["original\nlocation", "updated\nlocation", "setup\nsentence",
               "access\nevent", "move\nsentence",
               "other\nstory", "question", "system\nturn", "end"],
    "Photo": ["original\nlocation", "updated\nlocation", "setup\nsentence",
              "photo\ntaken", "move\nsentence", "other\nstory", "question", "system\nturn", "end"],
    "Code": ["original\nvalue", "updated\nvalue", "a = [...]\nline",
             "copy/alias\nlines", "a[m]=...\nline", "pad\nlines", "question", "system\nturn", "end"],
}
GROUP_COLOR = {"divergence": "#eb6834", "retrieval": "#2a78d6"}
ANSWER_TINT = "#eaf3de"
# (title, prompt condition, divergence group, retrieval group)
DOMAINS = [("Belief", "abstract_belief", "abstract_belief", "answer_changes_location_only"),
           ("Photo", "abstract_photo", "abstract_photo", "answer_changes_location_only_photo"),
           ("Code", None, "code_abstract", "code_control")]
PROMPT_TYPE = {"FB": {"base_rule": "ABA", "queried": "copy", "answer_col": 0,
                      "title": "false-belief prompts (outdated photo, copy queries): the original location is the answer"},
               "TB": {"base_rule": "ABB", "queried": "alias", "answer_col": 1,
                      "title": "true-belief prompts (current photo, alias queries): the updated location is the answer"}}
CAP = 40.0
S_BASE = 34.0
GAMMA = 0.6


def marker_area(v: float) -> float:
    return S_BASE * min(v, CAP) ** GAMMA


def heads_on_prompts(rows: pd.DataFrame, group: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Mean share/frac over cells for the heads in `group`, with group weights and labels."""
    keep = set(zip(group.layer, group["head"]))
    both = keep & set(zip(other.layer, other["head"]))
    d = rows[[k in keep for k in zip(rows.layer, rows["head"])]]
    cols = [f"share_{r}" for r in REGIONS] + [f"frac_{r}" for r in REGIONS]
    out = d.groupby(["layer", "head"])[cols].mean().reset_index()
    out = out.merge(group[["layer", "head", "mean_score"]], on=["layer", "head"]).rename(columns={"mean_score": "score"})
    out["label"] = [f"L{a}H{b}" + ("*" if (a, b) in both else "") for a, b in zip(out.layer, out["head"])]
    missing = keep - set(zip(out.layer, out["head"]))
    if missing:
        print(f"  warning: {len(missing)} group heads not in the capture: {sorted(missing)}")
    return out


def block(h: pd.DataFrame):
    share = h[[f"share_{r}" for r in REGIONS]].to_numpy(float)
    frac = h[[f"frac_{r}" for r in REGIONS]].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        enr = np.where(frac > 0, share / frac, np.nan)
    order = np.lexsort((-h.score.to_numpy(), np.nan_to_num(enr, nan=-1).argmax(1)))
    w = h.score.clip(lower=0).to_numpy()
    m_share = (w[:, None] * np.nan_to_num(share)).sum(0) / w.sum()
    m_frac = (w[:, None] * np.nan_to_num(frac)).sum(0) / w.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        m_enr = np.where(m_frac > 0, m_share / m_frac, np.nan)
    return ["set mean"] + list(h.label.to_numpy()[order]), np.vstack([m_enr, enr[order]]), m_share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_type", choices=["FB", "TB"], required=True)
    ap.add_argument("--story_csv", type=Path, default=Path("results/lookback_analysis/_regions/region_shares_matched.csv"))
    ap.add_argument("--code_csv", type=Path, default=Path("results/lookback_analysis/_regions/code_region_shares.csv"))
    ap.add_argument("--cma_root", type=Path, default=Path("results"),
                    help="CMA results root for the story head groups")
    ap.add_argument("--out_stem", type=Path, default=None)
    ap.add_argument("--unmatched", action="store_true",
                    help="story CSV from the per-cell captures (each group measured on its own "
                         "condition's prompts: C5/C6 heads on C5/C6 prompts) instead of the matched "
                         "captures; false-belief only, since C5/C6 have no true-belief prompts")
    args = ap.parse_args()
    if args.unmatched and args.prompt_type == "TB":
        raise SystemExit("--unmatched has no true-belief prompts for the retrieval groups (C5/C6)")
    pt = PROMPT_TYPE[args.prompt_type]
    out_stem = args.out_stem or Path(f"results/lookback_analysis/_regions/bubble_matrix_{args.prompt_type}")

    story = pd.read_csv(args.story_csv) if args.story_csv.exists() else None
    code = pd.read_csv(args.code_csv) if args.code_csv.exists() else None
    if code is not None and "share_orig_loc" not in code.columns:
        print("code CSV is in the old summed-mask format; rerun code_region_shares.py")
        code = None

    panels = []
    for title, pcond, div, ret in DOMAINS:
        if title == "Code":
            if code is None:
                continue
            gd, gr = code_head_group(div), code_head_group(ret)
            rows_d = code[(code.condition == div) & (code.queried == pt["queried"])]
            rows_r = code[(code.condition == ret) & (code.queried == pt["queried"])]
        else:
            if story is None:
                continue
            gd, gr = story_head_group(args.cma_root, div), story_head_group(args.cma_root, ret)
            rows_d = story[(story.condition == pcond) & (story.base_rule == pt["base_rule"])]
            # C5 / C6 cells are false-belief in both rule pairs, so all of them count
            rows_r = story[story.condition == ret] if args.unmatched else \
                story[(story.condition == pcond) & (story.base_rule == pt["base_rule"])]
        print(f"{title}: divergence {len(gd)} heads, retrieval {len(gr)} heads")
        panels.append((title, block(heads_on_prompts(rows_d, gd, gr)), block(heads_on_prompts(rows_r, gr, gd))))
    if not panels:
        raise SystemExit("no input CSVs found")

    n_rows = max(len(d[0]) + len(r[0]) for _, d, r in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(8.0 * len(panels), 0.25 * n_rows + 2.9),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    y_extent = 0.0
    answer_label_y = []
    for ax, (title, (ld, ed, _), (lr, er, _)) in zip(axes, panels):
        ax.axvspan(pt["answer_col"] - 0.45, pt["answer_col"] + 0.45, color=ANSWER_TINT, zorder=0)
        y, yt, yl, ycol = 0.0, [], [], {}
        for grp, labels, enr in (("divergence", ld, ed), ("retrieval", lr, er)):
            yt.append(y); yl.append(f"{grp} heads ({len(labels) - 1})"); ycol[len(yl) - 1] = grp
            y += 1.3
            for i, (lab, row) in enumerate(zip(labels, enr)):
                for j, v in enumerate(row):
                    if np.isnan(v):
                        continue
                    above = v >= 1
                    ax.scatter(j, y, s=marker_area(v), linewidths=1.0 if above else 0.9,
                               facecolors=GROUP_COLOR[grp] if above else "none",
                               edgecolors="white" if above else GROUP_COLOR[grp],
                               alpha=1.0 if i == 0 else 0.6, zorder=3)
                    if i == 0 and v >= 2:
                        ax.text(j + 0.2, y - 0.38, f"{v:.0f}×", ha="left", fontsize=7, color="#0b0b0b", zorder=4)
                yt.append(y); yl.append(lab)
                y += 1
            y += 1.0
        ax.set_yticks(yt)
        ax.set_yticklabels(yl, fontsize=7)
        for k, t in enumerate(ax.get_yticklabels()):
            if k in ycol:
                t.set_color(GROUP_COLOR[ycol[k]]); t.set_fontweight("bold"); t.set_fontsize(8.5)
            elif t.get_text() == "set mean":
                t.set_fontweight("bold")
        answer_label_y.append((ax, y))
        ax.set_xticks(range(len(REGIONS)))
        ax.set_xticklabels(COLS[title], fontsize=7.5)
        ax.xaxis.tick_top()
        ax.set_xlim(-0.7, len(REGIONS) - 0.3)
        ax.set_title(title, fontsize=12, pad=10, loc="left")
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        y_extent = max(y_extent, y)
    for ax in axes:
        ax.set_ylim(y_extent + 0.6, -1.0)
        ax.text(pt["answer_col"], y_extent + 0.4, "correct\nanswer", ha="center", va="bottom",
                fontsize=7.5, color="#27500a", fontweight="bold", zorder=4)
    fig.get_layout_engine().set(w_pad=0.25, wspace=0.12)

    fig.suptitle(pt["title"] + ("\nbelief/photo: each head group measured on its own condition's prompts "
                                "(C1/C2 heads on C1/C2 prompts, C5/C6 heads on C5/C6 prompts)" if args.unmatched else ""),
                 fontsize=11, x=0.01, ha="left")
    handles = [axes[-1].scatter([], [], s=marker_area(0.5), facecolors="none", edgecolors="#8a8983")] + \
              [axes[-1].scatter([], [], s=marker_area(v), color="#8a8983", alpha=0.6) for v in (1, 5, 20, CAP)]
    fig.legend(handles, ["0.5× (hollow: below uniform)", "1×", "5×", "20×", f"≥{CAP:.0f}×"],
               title="final-token attention relative to uniform (region's attention share ÷ region's token share); "
                     "area ∝ ratio^0.6.  * = head in both groups.  Belief access event: the sentence where the agent "
                     "loses sight (false belief) or sees the change (true belief).",
               loc="outside lower center", ncol=5, fontsize=8, title_fontsize=8, frameon=False,
               columnspacing=2.5, handletextpad=1.0, handleheight=3.0, borderpad=1.0)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=220, bbox_inches="tight")
    print(f"-> {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
