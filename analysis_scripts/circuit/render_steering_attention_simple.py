"""Two-panel, answer-aligned view of the existing steering attention bars.

Reuses the saved means and prompt-bootstrap intervals without recomputing
statistics. The full control figure remains steering_attention_bars.*.
Run from the project environment: python analysis_scripts/circuit/render_steering_attention_simple.py
--paper writes a compact copy for draft2.tex (no title, subtitle or footnotes; the
caption carries them) to results/figures/steering_attention_paper.pdf.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/summary/aggregation"
SOURCE = OUT / "steering_attention_bars.csv"
STEM = OUT / "steering_attention_simple"
PAPER_STEM = ROOT / "results/figures/steering_attention_paper"
PANELS = (
    ("false_belief", "False belief", "delete", "Deletion", "believed", "Old", "Current"),
    ("true_belief", "True belief", "inject", "Injection", "current", "Current", "Old"),
)
BEFORE = "#B7C4D1"
AFTER = "#2368A2"
INK = "#172D40"


def select_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="") as handle:
        source = list(csv.DictReader(handle))
    selected: list[dict[str, str]] = []
    for pset, _, mode, _, correct, _, _ in PANELS:
        for position in (correct, "current" if correct == "believed" else "believed"):
            for state in ("before", "after"):
                matches = [row for row in source
                           if row["prompt_set"] == pset and row["position"] == position
                           and row["mode"] == ("clean" if state == "before" else mode)
                           and row["direction"] == ("clean" if state == "before" else "target_rank1")
                           and float(row["dose"]) == (0 if state == "before" else 4)]
                if len(matches) != 1:
                    raise ValueError(f"Expected one row for {pset}/{position}/{state}, got {len(matches)}")
                row = dict(matches[0])
                mean, lo, hi = (float(row[key]) for key in ("mean", "ci_lo", "ci_hi"))
                if not 0 <= lo <= mean <= hi <= 1 or int(row["n"]) != 225:
                    raise ValueError(f"Unexpected interval or sample count: {row}")
                row.update(state=state, answer_role="correct" if position == correct else "alternative")
                selected.append(row)
    return selected


def render_paper(rows: list[dict[str, str]]) -> None:
    """Same bars and intervals, sized for the paper; text lives in the LaTeX caption."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.22, top=0.80, wspace=0.16)
    fig.legend(handles=[Patch(facecolor=BEFORE, label="before steering"),
                        Patch(facecolor=AFTER, label="after steering")],
               loc="upper right", bbox_to_anchor=(0.99, 1.0), ncol=2,
               frameon=False, handlelength=1.1, columnspacing=1.4, fontsize=8)
    # Bars are ordered by location (old, then current) in both panels, so each
    # location sits at the same x position; the answer role goes in parentheses.
    for letter, ax, (pset, title, _, operation, _, correct_name, alternative_name) in zip(
            "ab", axes, PANELS):
        ax.set_title(f"({letter}) {title.lower()} prompts, {operation.lower()}", loc="left",
                     fontsize=9, pad=6)
        roles = {correct_name: "correct", alternative_name: "alternative"}
        for x, location in enumerate(("Old", "Current")):
            role = roles[location]
            for state, offset, color in (("before", -0.17, BEFORE), ("after", 0.17, AFTER)):
                row = next(r for r in rows if r["prompt_set"] == pset
                           and r["answer_role"] == role and r["state"] == state)
                mean, lo, hi = (float(row[key]) for key in ("mean", "ci_lo", "ci_hi"))
                ax.bar(x + offset, mean, width=0.30, color=color, zorder=3)
                ax.errorbar(x + offset, mean, yerr=[[mean - lo], [hi - mean]],
                            fmt="none", color=INK, capsize=2, linewidth=0.8, zorder=4)
                ax.text(x + offset, hi + 0.004, f"{mean:.1%}", ha="center",
                        va="bottom", fontsize=7.5)
        answer = {"correct": "(correct answer)", "alternative": "(incorrect answer)"}
        ax.set_xticks([0, 1], [f"{loc.lower()} location\n{answer[roles[loc]]}"
                               for loc in ("Old", "Current")],
                      fontsize=8)
        ax.tick_params(axis="x", length=0, pad=4)
        ax.tick_params(axis="y", length=0, pad=3, labelsize=8)
        ax.set_ylim(0, 0.15)
        ax.set_yticks([0, 0.05, 0.10, 0.15])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E2E7EB", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B4BFC8")
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylabel("attention from final token", fontsize=8.5, labelpad=6)
    fig.savefig(PAPER_STEM.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(PAPER_STEM.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {PAPER_STEM}.{{pdf,png}}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true",
                        help="also write the compact copy used in draft2.tex")
    args = parser.parse_args()
    rows = select_rows()
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 5.8), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.26, top=0.72, wspace=0.20)
    fig.suptitle("Where answer-related heads attend before and after steering",
                 x=0.09, y=0.965, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.09, 0.90, "Qwen2.5-14B-Instruct  ·  26 heads in layers 42–46  ·  225 prompts per condition",
             fontsize=10.5, color="#526577")
    fig.legend(handles=[Patch(facecolor=BEFORE, label="Before steering"),
                        Patch(facecolor=AFTER, label="After steering")],
               loc="upper left", bbox_to_anchor=(0.084, 0.864), ncol=2,
               frameon=False, handlelength=1.3, columnspacing=2)

    for ax, (pset, title, _, operation, _, correct_name, alternative_name) in zip(axes, PANELS):
        ax.set_title(f"{title}  ·  {operation.lower()}", loc="left",
                     fontsize=13, fontweight="bold", pad=16)
        for x, role in enumerate(("correct", "alternative")):
            for state, offset, color in (("before", -0.18, BEFORE), ("after", 0.18, AFTER)):
                row = next(r for r in rows if r["prompt_set"] == pset
                           and r["answer_role"] == role and r["state"] == state)
                mean, lo, hi = (float(row[key]) for key in ("mean", "ci_lo", "ci_hi"))
                ax.bar(x + offset, mean, width=0.30, color=color, zorder=3)
                ax.errorbar(x + offset, mean, yerr=[[mean - lo], [hi - mean]],
                            fmt="none", color=INK, capsize=3, linewidth=1.1, zorder=4)
                ax.text(x + offset, hi + 0.005, f"{mean:.1%}", ha="center",
                        va="bottom", fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1], [f"Correct-answer location\n({correct_name.lower()} location)",
                              f"Alternative location\n({alternative_name.lower()} location)"])
        ax.tick_params(axis="x", length=0, pad=10)
        ax.tick_params(axis="y", length=0, pad=6)
        ax.set_ylim(0, 0.15)
        ax.set_yticks([0, 0.05, 0.10, 0.15])
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", color="#E2E7EB", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#B4BFC8")
        ax.set_xlim(-0.6, 1.6)
    axes[0].set_ylabel("Attention from the final prompt token", labelpad=13)
    fig.text(0.09, 0.115, "Bars: mean attention; error bars: 95% bootstrap confidence intervals over prompts.",
             fontsize=10, color="#526577")
    fig.text(0.09, 0.078, "Single-direction steering (rank-1), α = 4. Correct answer is defined by the original story.",
             fontsize=10, color="#526577")
    fig.text(0.09, 0.041, "Only the two answer locations are shown; attention to all other tokens is omitted, not renormalized.",
             fontsize=10, color="#526577")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(STEM.with_suffix(f".{ext}"), dpi=240, bbox_inches="tight")
    plt.close(fig)
    with STEM.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    STEM.with_suffix(".md").write_text(
        "# Attention before and after steering\n\n"
        "Mean attention from the final prompt token to the correct-answer and alternative "
        "locations, averaged over 26 answer-related heads in layers 42–46 of Qwen2.5-14B-Instruct. "
        "The correct answer is fixed by the original story: the old location for false belief "
        "and the current location for true belief. Location mass sums all token positions "
        "overlapping any occurrence of that location's name. Heads are averaged within "
        "each prompt before averaging across the 225 held-out prompts per condition. "
        "Error bars are descriptive 95% bootstrap confidence intervals over prompts "
        "(5,000 resamples), copied unchanged from the full figure's saved table. "
        "No significance stars or new hypothesis tests are added.\n\n"
        "Steering adds (injection) or subtracts (deletion) the rank-1 direction at the "
        "final token immediately before layer 42, with coefficient α = 4. "
        "The heads themselves are not directly edited. Both panels share an axis. "
        "Percentages retain the total attention denominator; omitted tokens carry "
        "the remaining mass.\n\n"
        "The clean mean is higher for the correct than the alternative location in both "
        "conditions. Deletion on false belief lowers correct-location attention and raises "
        "alternative-location attention. Injection on true belief lowers attention to "
        "both locations, with a larger decline at the correct location. In both cases the "
        "ordering reverses. This plot alone does not establish that attention changes "
        "mediate the answer change or that every head behaves this way.\n\n"
        "## Limit of the available evidence\n\n"
        "The steering attention.csv files save only mass_believed and mass_current "
        "per head and prompt. The runner discards full attention patterns after summing "
        "these masks. No token-level steering attention capture was found in the local "
        "results/circuit or results/circuit trees. The older lookback captures use "
        "different prompt/head selections and cannot establish a ranking for this experiment. "
        "Consequently we cannot claim that the correct-answer location is attended more "
        "than every other token or location. That claim needs matching full token-level "
        "attention measurements and a defined comparison between multi-token location "
        "spans and individual tokens.\n\n"
        "Source: steering_attention_bars.csv in this directory. Full appendix figure: "
        "steering_attention_bars.png / .pdf. Regenerate this figure with "
        "`python analysis_scripts/circuit/render_steering_attention_simple.py`.\n"
    )
    print("Selected means (attention fraction, range 0–1):")
    for row in rows:
        print(f"{row['prompt_set']:13} {row['answer_role']:11} {row['state']:6}: "
              f"{float(row['mean']):.6f} [{float(row['ci_lo']):.6f}, {float(row['ci_hi']):.6f}]")
    print(f"Saved {STEM}.{{png,pdf,svg,csv,md}}")
    if args.paper:
        render_paper(rows)


if __name__ == "__main__":
    main()
