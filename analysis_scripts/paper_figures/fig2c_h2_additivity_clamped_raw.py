"""Figure 2c — C3 against C1 + C5, clamped (a) and raw (b), coloured by layer.

Companion to fig2b panel (a): same scatter style, one panel per score
convention. Clamped panels sit below y = x wherever a head's C1 and C5
scores have opposite signs (clamping drops the cancellation); raw is the
fair test of unweighted additivity.

Data: results/aggregated/fingerprints/*.npy; correlations computed here.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import INK2, CMAP_SEQ, apply_style, despine, panel_label, save

REPO = Path(__file__).resolve().parents[2]
FP = REPO / "results/aggregated/fingerprints"
NAMES = {"C1": "C1_abstract_belief", "C3": "C3_answer_changes_belief",
         "C5": "C5_answer_changes_location_only"}


def load(cond, kind):
    return np.load(FP / f"{NAMES[cond]}_{kind}.npy").ravel()


def main():
    apply_style()
    layers = np.repeat(np.arange(48), 40)

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, kind, letter in zip(axes, ("clamped", "raw"), "ab"):
        fp = {c: load(c, kind) for c in NAMES}
        assert len(fp["C3"]) == 1920
        x, y = fp["C1"] + fp["C5"], fp["C3"]
        r = float(np.corrcoef(x, y)[0, 1])

        sc = ax.scatter(x, y, c=layers, cmap=CMAP_SEQ, s=9, alpha=0.85, lw=0,
                        vmin=0, vmax=47, zorder=3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2,
                lw=0.9, ls="--", zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"C1 + C5 causal score ({kind})")
        ax.set_ylabel(f"C3 causal score ({kind})")
        ax.text(0.03, 0.97, f"r = {r:.3f}", transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5)
        ax.text(0.97, 0.04, "y = x\n(unweighted additive)",
                transform=ax.transAxes, va="bottom", ha="right",
                fontsize=7.5, color=INK2)
        cax = ax.inset_axes([0.62, 0.32, 0.30, 0.04])
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 47]); cb.ax.tick_params(labelsize=6.5, length=2)
        cax.set_title("layer", fontsize=7, pad=2)
        ax.set_title("negatives clamped to 0" if kind == "clamped"
                     else "raw (signed)", loc="left")
        despine(ax)
        panel_label(ax, letter, x=-0.16)
        print(f"{kind:8s} r = {r:.3f}")

    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2c_h2_additivity_clamped_raw")


if __name__ == "__main__":
    main()
