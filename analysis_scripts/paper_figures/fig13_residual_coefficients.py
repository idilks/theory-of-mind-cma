"""Plot saved single-direction accuracy across steering coefficients.

Uses the existing held-out accuracy table, without rerunning inference or
mixing the additive intervention with the eight-dimensional replacement.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from style import VIOLET, apply_style, despine


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_main/accuracy_by_config.csv"
OUTPUT = ROOT / "writeups/neurips-overleaf/figures/fig13_residual_coefficients"
CURVES = [
    ("target_rank1", "Mean difference", VIOLET, "-"),
    ("random", "Random direction", "#777777", "-"),
    ("shuffled", "Shuffled labels", "#777777", "--"),
    ("heads_ctl", "Non-selected heads", "#777777", ":"),
]


def main():
    with SOURCE.open() as handle:
        rows = list(csv.DictReader(handle))
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, condition, operation, title in zip(
        axes, ("false_belief", "true_belief"), ("delete", "inject"),
        ("a  Subtract on false belief", "b  Add on true belief")
    ):
        clean, = [row for row in rows if row["prompt_set"] == condition and row["mode"] == "clean"]
        for direction, label, color, linestyle in CURVES:
            selected = sorted(
                [row for row in rows if row["prompt_set"] == condition
                 and row["mode"] == operation and row["direction"] == direction],
                key=lambda row: float(row["dose"]),
            )
            points = [clean] + selected
            coefficients = np.array([float(row["dose"]) for row in points])
            np.testing.assert_array_equal(coefficients, [0, 0.5, 1, 2, 4])
            assert all(int(row["n"]) == 225 for row in points)
            accuracy = 100 * np.array([float(row["accuracy"]) for row in points])
            error = 100 * np.array([float(row["acc_sem"]) for row in points])
            assert np.isfinite(accuracy).all() and ((accuracy >= 0) & (accuracy <= 100)).all()
            print(f"{condition} {direction}: accuracy (%)={accuracy.round(2).tolist()}")
            ax.errorbar(coefficients, accuracy, yerr=error, label=label, color=color,
                        linestyle=linestyle, linewidth=1.8 if direction == "target_rank1" else 1,
                        marker="o", markersize=3, elinewidth=0.7, capsize=2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlabel(r"Steering coefficient $\alpha$")
        ax.set_xticks([0, 0.5, 1, 2, 4], ["0", "0.5", "1", "2", "4"])
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        despine(ax)
    axes[0].set_ylabel("First-token accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.51, -0.01), handlelength=1.8, columnspacing=1.4)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.29, wspace=0.20)
    for suffix in (".pdf", ".png"):
        fig.savefig(OUTPUT.with_suffix(suffix), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"Saved {OUTPUT}.pdf and .png")


if __name__ == "__main__":
    main()
