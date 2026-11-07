"""Figure 11 — per-head axis steering across frames (div_unique heads, Qwen2.5-14B-Instruct).

Each axis is a per-head state direction (non-diverged minus diverged mean output
at the final token), estimated on one frame only and added to the 22 div_unique
heads' outputs on another frame's prompts, scaled to that head's state-signal
norm in the target frame. Effect = change in logit difference (alternate minus
correct answer) relative to clean, minus the same quantity for a matched-norm
random direction at the same alpha. signed alpha = +1 pushes toward the
opposite divergence status, -1 toward the item's own status. axis_shuffled =
the real axes permuted across the same heads (same norms).

The belief/code and belief/photo harnesses agree on shared cells to three decimals.
All panels use one x order; the shaded column is that panel's native axis.

Data (--selection top32, the original figure):
results/rsa_head_outputs/steering_qwen/steer_summary.csv (72 ToM + 96 code
items), results/rsa_head_outputs/steering_photo_qwen/photo_steer_summary.csv
(72 belief + 72 photo items) and, for the photo axis on code,
results/rsa_head_outputs/steering_code_photo_qwen/steer_summary.csv (96 code
items; that column stays empty until the file exists).

--selection sig4of6 | sig3of6 reads the single rerun under the significance-
count head selection, results/rsa_head_outputs/
steering_qwen_{selection}_tom-code-photo/steer_summary.csv (steer_state_axes.py
--selection ... --domains tom code photo), whose axis names are u_tom, u_photo,
u_cm, axis_shuffled in every domain. The figure is saved with the selection as
a suffix, and the transfer percentages (cross-axis / native-axis effect at
alpha=+1) that the results paragraph quotes are printed and written to
paper_figures/fig11_transfer_{selection}.csv.
"""

import argparse

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, GRAY, GRID, INK2, MAGENTA, YELLOW, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
RSA = REPO / "results/rsa_head_outputs"
SRC_CODE = RSA / "steering_qwen/steer_summary.csv"
SRC_PHOTO = RSA / "steering_photo_qwen/photo_steer_summary.csv"
SRC_CODE_PHOTO = RSA / "steering_code_photo_qwen/steer_summary.csv"

COLORS = {"belief": BLUE, "photo": MAGENTA, "code": YELLOW, "shuffled": GRAY}

# (panel title, source csv, domain key, native axis label,
#  [(axis key, axis source label, csv override)])
PANELS = [
    ("belief prompts", SRC_PHOTO, "belief", "belief",
     [("u_belief", "belief", None), ("u_photo", "photo", None), ("u_cm", "code", None),
      ("axis_shuffled", "shuffled", None)]),
    ("photo prompts", SRC_PHOTO, "photo", "photo",
     [("u_belief", "belief", None), ("u_photo", "photo", None), ("u_cm", "code", None),
      ("axis_shuffled", "shuffled", None)]),
    ("code prompts", SRC_CODE, "code", "code",
     [("u_tom", "belief", None), ("u_photo", "photo", SRC_CODE_PHOTO), ("u_cm", "code", None),
      ("axis_shuffled", "shuffled", None)]),
]


def panels_for(selection):
    """Panel spec for the significance-selected rerun: one file, one axis naming."""
    src = RSA / f"steering_qwen_{selection}_tom-code-photo/steer_summary.csv"
    rows = [("u_tom", "belief", None), ("u_photo", "photo", None), ("u_cm", "code", None),
            ("axis_shuffled", "shuffled", None)]
    return [("belief prompts", src, "tom", "belief", rows),
            ("photo prompts", src, "photo", "photo", rows),
            ("code prompts", src, "code", "code", rows)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="top32", choices=["top32", "sig4of6", "sig3of6"])
    args = ap.parse_args()
    panels = PANELS if args.selection == "top32" else panels_for(args.selection)
    suffix = "" if args.selection == "top32" else f"_{args.selection}"
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.9), sharey=True)
    w = 0.38
    transfer = []
    for ax, letter, (title, src, domain, native, rows) in zip(axes, "abc", panels):
        default = pd.read_csv(src)
        native_key = next(k for k, lab, _ in rows if lab == native)
        native_plus = (default[default.domain == domain]
                       .set_index(["axis", "signed_alpha"]).loc[(native_key, 1.0), "mean_diff"])
        xs = np.arange(len(rows))
        for x, (key, label, override) in zip(xs, rows):
            if label == native:
                ax.axvspan(x - 0.5, x + 0.5, color=GRID, alpha=0.55, lw=0, zorder=0)
            if override is not None and not override.exists():
                print(f"{domain:6s} {key:14s} missing {override}, column left empty")
                continue
            df = default if override is None else pd.read_csv(override)
            df = df[df.domain == domain].set_index(["axis", "signed_alpha"])
            c = COLORS[label]
            for sa, dx, alpha in ((1.0, -w / 2, 1.0), (-1.0, w / 2, 0.45)):
                r = df.loc[(key, sa)]
                ax.bar(x + dx, r["mean_diff"], width=w, color=c, alpha=alpha,
                       yerr=r["sem"], error_kw={"lw": 0.8, "ecolor": INK2},
                       zorder=3)
            plus = df.loc[(key, 1.0), "mean_diff"]
            transfer.append({"target": domain, "axis": label, "plus1": plus,
                             "minus1": df.loc[(key, -1.0), "mean_diff"],
                             "sem_plus1": df.loc[(key, 1.0), "sem"],
                             "n": int(df.loc[(key, 1.0), "n"]),
                             "pct_of_native": 100 * plus / native_plus})
            print(f"{domain:6s} {key:14s} +1 {plus:+7.2f}"
                  f"  -1 {df.loc[(key, -1.0), 'mean_diff']:+7.2f}"
                  f"  n={int(df.loc[(key, 1.0), 'n'])}  {100 * plus / native_plus:6.1f}% of native")
        ax.axhline(0, color=INK2, lw=0.8, zorder=2)
        ax.set_xlim(-0.5, len(rows) - 0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{lab}\naxis" if lab != "shuffled" else "shuffled\ncontrol"
                            for _, lab, _ in rows], fontsize=8)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        despine(ax)
        panel_label(ax, letter, x=-0.18 if letter == "a" else -0.10)
    axes[0].set_ylabel("Δ logit diff. vs matched random")

    handles = [plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=a, label=l)
               for a, l in ((1.0, "α = +1 (toward the opposite status)"),
                            (0.45, "α = −1 (toward the item's own status)"))]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.55,
                                 label="axis from the prompts' own domain"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.5)
    pd.DataFrame(transfer).to_csv(REPO / "results" / "figures" / f"fig11_transfer{suffix}.csv",
                                  index=False)
    save(fig, f"fig11_head_steering{suffix}")


if __name__ == "__main__":
    main()
