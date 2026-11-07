"""Shared style for rebuttal figures.

One palette, one set of rcParams, so the whole figure set reads as a system.
Colors are a CVD-validated categorical set; sequential maps are single-hue
white-to-dark per Reviewer 3's request.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results" / "figures"

# categorical slots (fixed assignment across all rebuttal figures)
BLUE = "#2a78d6"     # divergence-tracking / belief / target curves
ORANGE = "#eb6834"   # answer-retrieval / C5
AQUA = "#1baf7a"     # no-divergence variants (true belief, code_alias)
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"
RED = "#e34948"

GRAY = "#898781"       # layer-matched controls / nulls
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"

# white-to-dark single-hue sequential map (Reviewer 3, point 6-7)
CMAP_SEQ = LinearSegmentedColormap.from_list(
    "white_blue", ["#ffffff", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]
)

RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style():
    mpl.rcParams.update(RC)


def despine(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panel_label(ax, letter, x=-0.14, y=1.02):
    """Bold letter in the left margin; pair with left-aligned titles."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name):
    OUT_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{name}.{ext}")
    print(f"saved {OUT_DIR / name}.png/.pdf")
    plt.close(fig)
