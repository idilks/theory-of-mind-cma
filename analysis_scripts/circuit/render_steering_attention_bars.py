#!/usr/bin/env python
"""Reader-head attention by target position under steering — bar figure.

Sept 3 2026 action item: attention results go in as bars with error bars
over defined positions, not heatmaps (tokenisation varies across prompts).

What is plotted
---------------
For the L42–46 reader/retrieval heads (26 heads, the `primary` writer set of
the Stage-2 steering run), the mean attention mass from the final token onto
three positions:

  believed   the location the agent believes the object is in
  current    the location the object is actually in (world state)
  other      everything else (1 − believed − current)

under the clean run, the wire deleted, the wire injected, and the three
magnitude-matched control directions (shuffled-label, random, non-writer
heads).  False-belief and true-belief prompts are separate panels.  Every
bar is a mean over prompts with a 95% bootstrap CI over prompts (the head
average is taken inside each prompt first, so n = number of prompts).

Where the numbers come from
---------------------------
results/circuit/stage2_steering/{model}/L41_main/attention.csv       clean +
    target directions (rank-1 wire, top-k wire), both operators, both
    prompt sets
results/circuit/stage2_steering/{model}/L41_attn_null/attention.csv  the
    three control directions, natural operator only (delete on false
    belief, inject on true belief)

These are the same per-prompt tables `analyze_steering_attention_null.py`
reduces to the believed-minus-current readout; this script keeps the two
masses separate so the reader can see WHERE attention went, not just the
signed difference.

Outputs (results/summary/aggregation/)
  steering_attention_bars.png / .pdf
  steering_attention_bars.csv   one row per plotted bar: mean, CI, paired
                                delta vs clean with its own bootstrap CI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "circuit"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "paper_figures"))

from circuit_common import STAGE2_OUT_ROOT  # noqa: E402
from style import (BLUE, ORANGE, GRAY, INK, INK2, GRID,  # noqa: E402
                   apply_style, despine, panel_label)

OUT_DIR = ROOT / "results/summary" / "aggregation"
KEY = ["prompt_set", "mode", "direction", "dose", "template", "pair_idx"]
POSITIONS = ("believed", "current", "other")
POS_COLOR = {"believed": BLUE, "current": ORANGE, "other": GRAY}
POS_LABEL = {"believed": "believed location",
             "current": "current (world) location",
             "other": "other tokens"}
TARGETS = ("target_rank1", "target_topk")
CONTROLS = ("shuffled", "random", "heads_ctl")
DIR_SHORT = {"target_rank1": "rank-1", "target_topk": "top-k",
             "shuffled": "shuffled\nlabels", "random": "random",
             "heads_ctl": "non-writer\nheads"}
# the operator that has something to act on, per prompt set
NATURAL_OP = {"false_belief": "delete", "true_belief": "inject"}
OTHER_SET = {"false_belief": "true_belief", "true_belief": "false_belief"}
PANEL_TITLE = {"false_belief": "False-belief prompts",
               "true_belief": "True-belief prompts"}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_per_prompt(run_dir: Path) -> pd.DataFrame | None:
    """Per-prompt mass on each position, averaged over the reader heads."""
    p = run_dir / "attention.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p).drop_duplicates(subset=KEY + ["layer", "head"],
                                       keep="first")
    a["believed"] = a["mass_believed"]
    a["current"] = a["mass_current"]
    a["other"] = 1.0 - a["mass_believed"] - a["mass_current"]
    n_heads = a[["layer", "head"]].drop_duplicates().shape[0]
    per = a.groupby(KEY)[list(POSITIONS)].mean().reset_index()
    per.attrs["n_heads"] = n_heads
    per.attrs["layers"] = (int(a["layer"].min()), int(a["layer"].max()))
    return per


def boot_ci(x: np.ndarray, rng: np.random.Generator, n_boot: int):
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_specs(pset: str):
    """Ordered list of (label, mode, direction, is_control, precondition_ok)."""
    nat = NATURAL_OP[pset]
    rev = "inject" if nat == "delete" else "delete"
    specs = [("clean", "clean", "clean", False, True)]
    for mode in (nat, rev):
        for d in TARGETS:
            specs.append((f"{mode}\n{DIR_SHORT[d]}", mode, d, False,
                          mode == nat))
    for d in CONTROLS:
        specs.append((DIR_SHORT[d], nat, d, True, True))
    return specs


def summarise(per: pd.DataFrame, dose: float, rng, n_boot) -> pd.DataFrame:
    """One row per (prompt_set, group, position): mean, CI, paired delta."""
    rows = []
    for pset in ("false_belief", "true_belief"):
        sub = per[per["prompt_set"] == pset]
        clean = (sub[sub["direction"] == "clean"]
                 .set_index(["template", "pair_idx"]))
        for label, mode, direction, is_ctl, pre_ok in group_specs(pset):
            if direction == "clean":
                g = clean
            else:
                g = sub[(sub["mode"] == mode) & (sub["direction"] == direction)
                        & (np.isclose(sub["dose"], dose))]
                g = g.set_index(["template", "pair_idx"])
            if g.empty:
                print(f"[warn] {pset} {mode}/{direction} @ alpha={dose}: "
                      "no rows on disk, skipped")
                continue
            for pos in POSITIONS:
                x = g[pos].to_numpy()
                lo, hi = boot_ci(x, rng, n_boot)
                row = dict(prompt_set=pset, group=label.replace("\n", " "),
                           mode=mode, direction=direction,
                           dose=0.0 if direction == "clean" else dose,
                           is_control=is_ctl, precondition_present=pre_ok,
                           position=pos, n=int(len(x)),
                           mean=float(x.mean()), ci_lo=lo, ci_hi=hi)
                if direction != "clean":
                    d = (g[pos] - clean[pos].reindex(g.index)).dropna()
                    dlo, dhi = boot_ci(d.to_numpy(), rng, n_boot)
                    row.update(delta_vs_clean=float(d.mean()),
                               delta_lo=dlo, delta_hi=dhi,
                               sign_consistency=float(
                                   (np.sign(d) == np.sign(d.mean())).mean()))
                else:
                    row.update(delta_vs_clean=0.0, delta_lo=0.0, delta_hi=0.0,
                               sign_consistency=np.nan)
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
def _bar(ax, x, m, lo, hi, color, is_ctl, width):
    if is_ctl:
        ax.bar(x, m, width=width, facecolor="white", edgecolor=color,
               linewidth=1.0, hatch="////", zorder=3)
    else:
        ax.bar(x, m, width=width, color=color, linewidth=0, zorder=3)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=4)


def draw_panel(ax_main, ax_other, tab: pd.DataFrame, pset: str, dose: float,
               letter: str):
    specs = group_specs(pset)
    t = tab[tab["prompt_set"] == pset]
    present = [s for s in specs
               if not t[(t["mode"] == s[1]) & (t["direction"] == s[2])].empty]
    xs = np.arange(len(present))
    w = 0.34
    n_target = sum(1 for s in present if not s[3])

    # shaded background for the precondition-absent operator
    absent = [i for i, s in enumerate(present) if not s[4]]
    if absent:
        for ax in (ax_main, ax_other):
            ax.axvspan(min(absent) - 0.5, max(absent) + 0.5, color=GRID,
                       alpha=0.55, lw=0, zorder=0)
        ax_main.text(np.mean(absent), 0.985, "precondition absent\n(sign-reversal control)",
                     transform=ax_main.get_xaxis_transform(), ha="center",
                     va="top", fontsize=7, color=INK2, linespacing=1.1)

    # divider between target and control directions
    if n_target < len(present):
        for ax in (ax_main, ax_other):
            ax.axvline(n_target - 0.5, color=INK2, lw=0.8, ls=(0, (3, 2)),
                       zorder=1)
        ax_main.text(n_target - 0.5 + (len(present) - n_target) / 2, 0.985,
                     f"matched-norm controls\n({NATURAL_OP[pset]}, α = {dose:g})",
                     transform=ax_main.get_xaxis_transform(), ha="center",
                     va="top", fontsize=7, color=INK2, linespacing=1.1)

    for i, (label, mode, direction, is_ctl, _) in zip(xs, present):
        for pos, off in (("believed", -w / 2), ("current", w / 2)):
            r = t[(t["mode"] == mode) & (t["direction"] == direction)
                  & (t["position"] == pos)].iloc[0]
            _bar(ax_main, i + off, r["mean"], r["ci_lo"], r["ci_hi"],
                 POS_COLOR[pos], is_ctl, w)
        r = t[(t["mode"] == mode) & (t["direction"] == direction)
              & (t["position"] == "other")].iloc[0]
        ax_other.errorbar(i, r["mean"],
                          yerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                          fmt="o", ms=4.5, color=GRAY, mfc="white" if is_ctl else GRAY,
                          mew=1.2, ecolor=INK, elinewidth=0.9, capsize=2.2,
                          zorder=3)

    # where the OTHER belief condition sits without any intervention
    other = tab[(tab["prompt_set"] == OTHER_SET[pset])
                & (tab["direction"] == "clean")]
    for pos in ("believed", "current"):
        lvl = float(other[other["position"] == pos]["mean"].iloc[0])
        ax_main.axhline(lvl, xmin=0.5 / len(present), xmax=1.0,
                        color=POS_COLOR[pos], lw=1.0, ls=(0, (4, 2.5)),
                        alpha=0.9, zorder=2)
    lvl = float(other[other["position"] == "other"]["mean"].iloc[0])
    ax_other.axhline(lvl, xmin=0.5 / len(present), xmax=1.0, color=GRAY,
                     lw=1.0, ls=(0, (4, 2.5)), alpha=0.9, zorder=2)

    n = int(t["n"].iloc[0])
    ax_main.set_title(f"{PANEL_TITLE[pset]}  (n = {n} held-out prompts)",
                      loc="left")
    panel_label(ax_main, letter, x=-0.06 if letter == "b" else -0.13)
    ax_other.set_xticks(xs)
    ax_other.set_xticklabels([s[0] for s in present], fontsize=8)
    ax_main.set_xlim(-0.6, len(present) - 0.4)
    ax_main.grid(axis="x", visible=False)
    ax_other.grid(axis="x", visible=False)
    despine(ax_main)
    despine(ax_other)
    ax_main.spines["bottom"].set_visible(False)
    # sharex propagates the tick labels upward; hide them on the main panel
    ax_main.tick_params(axis="x", length=0, labelbottom=False)


def render(tab: pd.DataFrame, dose: float, n_heads: int, layers, out_stem: Path):
    apply_style()
    fig = plt.figure(figsize=(12.6, 5.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3.4, 1.0], hspace=0.06,
                  wspace=0.10)
    axes = {}
    for col, pset in enumerate(("false_belief", "true_belief")):
        ax_main = fig.add_subplot(gs[0, col])
        ax_other = fig.add_subplot(gs[1, col], sharex=ax_main)
        axes[pset] = (ax_main, ax_other)
        draw_panel(ax_main, ax_other, tab, pset, dose, "ab"[col])

    # shared y ranges
    main_max = tab[tab["position"] != "other"]["ci_hi"].max()
    for ax_main, _ in axes.values():
        ax_main.set_ylim(0, main_max * 1.30)
    oth = tab[tab["position"] == "other"]
    pad = (oth["ci_hi"].max() - oth["ci_lo"].min()) * 0.25
    for _, ax_other in axes.values():
        ax_other.set_ylim(oth["ci_lo"].min() - pad, oth["ci_hi"].max() + pad)

    axes["false_belief"][0].set_ylabel("attention mass from final token")
    axes["false_belief"][1].set_ylabel("other tokens", fontsize=8.5)
    for _, ax_other in axes.values():
        ax_other.set_xlabel("")
    axes["true_belief"][0].tick_params(axis="y", labelleft=False)
    axes["true_belief"][1].tick_params(axis="y", labelleft=False)

    handles = [
        Patch(color=BLUE, label=POS_LABEL["believed"]),
        Patch(color=ORANGE, label=POS_LABEL["current"]),
        Line2D([], [], marker="o", color=GRAY, ls="none", ms=4.5,
               label=POS_LABEL["other"] + " (lower strip)"),
        Patch(facecolor="white", edgecolor=INK2, hatch="////",
              label="control direction (hatched)"),
        Line2D([], [], color=INK2, ls=(0, (4, 2.5)), lw=1.0,
               label="clean level of the opposite belief condition"),
        Line2D([], [], color=INK, lw=0.9, label="95% bootstrap CI over prompts"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False,
               handlelength=1.6, columnspacing=1.8)
    fig.suptitle(
        f"Reader-head attention (L{layers[0]}–L{layers[1]}, {n_heads} heads) "
        f"re-aims under steering at α = {dose:g}",
        fontsize=11.5, x=0.5, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_stem.with_suffix('.png')}  (+ .pdf)")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--main_label", default="L41_main")
    ap.add_argument("--null_label", default="L41_attn_null")
    ap.add_argument("--dose", type=float, default=4.0,
                    help="steering coefficient alpha to plot (default 4)")
    ap.add_argument("--n_boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_stem", type=Path,
                    default=OUT_DIR / "steering_attention_bars")
    args = ap.parse_args()

    root = STAGE2_OUT_ROOT / args.model_type
    frames, n_heads, layers = [], None, None
    for label in (args.main_label, args.null_label):
        per = load_per_prompt(root / label)
        if per is None:
            print(f"[warn] no attention.csv under {label}")
            continue
        n_heads = per.attrs["n_heads"]
        layers = per.attrs["layers"]
        print(f"loaded {label}: {len(per)} prompt-level rows, directions "
              f"{sorted(per['direction'].unique())}")
        frames.append(per)
    if not frames:
        sys.exit(f"no attention data under {root}")
    per = pd.concat(frames, ignore_index=True).drop_duplicates(subset=KEY,
                                                               keep="first")

    rng = np.random.default_rng(args.seed)
    tab = summarise(per, args.dose, rng, args.n_boot)
    out_csv = args.out_stem.with_suffix(".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_csv, index=False)
    print(f"-> {out_csv}")

    show = tab[["prompt_set", "group", "position", "n", "mean", "ci_lo",
                "ci_hi", "delta_vs_clean", "delta_lo", "delta_hi"]]
    print(f"\nattention mass, mean over {n_heads} reader heads then over "
          f"prompts; 95% bootstrap CI ({args.n_boot} draws); alpha = {args.dose:g}")
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    render(tab, args.dose, n_heads, layers, args.out_stem)


if __name__ == "__main__":
    main()
