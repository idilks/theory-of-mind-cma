#!/usr/bin/env python
"""Reader-head attention readout under steering: effect sizes and nulls.

The Stage-2 steering run records TWO readouts with DIFFERENT coverage:

  behavior (flip score)  every config — target directions AND the
                         random / heads_ctl / shuffled controls;
  reader attention       clean + target directions only, unless the run was
                         launched with --attn_all_directions.

So "steering shifts reader attention" is well supported by L41_main alone,
but "more than a magnitude-matched non-wire edit would" is not — the
control-direction attention numbers do not exist in the original run. This
script computes everything the first claim needs from L41_main, and folds in
the orthogonal-direction null as soon as the L41_attn_null run (see
slurm_scripts/circuit_stage2_steering_controls.sh) has been pulled.

Three nulls, in increasing strength, all reported here:

  1. paired-vs-clean          each steered prompt against its own clean run.
                              Available now. Answers "is the shift real".
  2. sign-reversal control    the SAME direction at the SAME norm applied
                              where its precondition fails (delete on
                              true_belief, inject on false_belief). Available
                              now for TB-delete. This is a tighter control
                              than a random direction — it holds the edit
                              direction fixed and flips only the sign — and
                              it predicts the attention shift should reverse,
                              not vanish.
  3. orthogonal-direction     random / heads_ctl / shuffled at matched norm.
                              Needs L41_attn_null. Answers "is the shift
                              specific to the wire".

Readout definition: per prompt, mean over the L42-46 reader heads of
(attention mass on the BELIEVED location) - (mass on the CURRENT location)
at query position -1. Which of the two is *correct* differs between FB and
TB, so the believed-minus-current convention is what makes the sign
comparable across legs. The natural gap is the clean-FB minus clean-TB value
of this readout, i.e. how far apart the two conditions sit without any
intervention — the yardstick for "how much of the effect did we move".

Outputs (results/circuit/stage2_steering/{model}/_attention_null/):
  attention_effects.csv   per config: paired shift, SEM, t, sign consistency,
                          fraction of the natural gap, and (when available)
                          the target/control ratio at matched alpha
  attention_null.csv      target-vs-control comparison, one row per
                          (prompt_set, operator, alpha)
  figures/attention_with_controls.png
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "circuit"))

from circuit_common import STAGE2_OUT_ROOT  # noqa: E402

KEY = ["prompt_set", "mode", "direction", "dose", "template", "pair_idx"]
TARGETS = ("target_topk", "target_rank1")
# Okabe-Ito
C_BLUE, C_ORANGE, C_GREEN, C_GRAY = "#0072B2", "#D55E00", "#009E73", "#8a8a8a"


def load_attention(run_dir: Path) -> pd.DataFrame | None:
    """Per-prompt believed-minus-current shift, averaged over reader heads."""
    p = run_dir / "attention.csv"
    if not p.exists():
        return None
    a = pd.read_csv(p).drop_duplicates(subset=KEY + ["layer", "head"],
                                       keep="first")
    a["shift"] = a["mass_believed"] - a["mass_current"]
    return (a.groupby(KEY)["shift"].mean().reset_index()
            .assign(run=run_dir.name))


def paired_effects(per: pd.DataFrame, gap: float) -> pd.DataFrame:
    """Steered minus clean, paired within (prompt_set, template, pair_idx)."""
    clean = (per[per["direction"] == "clean"]
             .set_index(["prompt_set", "template", "pair_idx"])["shift"])
    rows = []
    for (pset, mode, direction, dose), g in (
            per[per["direction"] != "clean"]
            .groupby(["prompt_set", "mode", "direction", "dose"])):
        gi = g.set_index(["prompt_set", "template", "pair_idx"])
        d = (gi["shift"] - clean.reindex(gi.index)).dropna()
        if d.empty:
            continue
        m, sem = float(d.mean()), float(d.sem())
        rows.append(dict(
            prompt_set=pset, mode=mode, direction=direction, dose=dose,
            n=int(len(d)), mean_shift=m, sem=sem,
            t=m / sem if sem > 0 else np.nan,
            sign_consistency=float((np.sign(d) == np.sign(m)).mean()),
            frac_natural_gap=abs(m) / gap if gap else np.nan,
            steered_level=float(gi["shift"].mean()),
        ))
    return pd.DataFrame(rows)


def target_vs_control(eff: pd.DataFrame) -> pd.DataFrame:
    """|target shift| / |control shift| at matched (prompt_set, mode, alpha).

    Controls are magnitude-matched by construction, so the ratio is a like-
    for-like specificity number: how much more the wire direction moves
    reader attention than a non-wire edit of the same size.
    """
    rows = []
    ctl = eff[~eff["direction"].isin(TARGETS)]
    for (pset, mode, dose), g in eff[eff["direction"].isin(TARGETS)].groupby(
            ["prompt_set", "mode", "dose"]):
        c = ctl[(ctl["prompt_set"] == pset) & (ctl["mode"] == mode)
                & (ctl["dose"] == dose)]
        if c.empty:
            continue
        for r in g.itertuples():
            worst = c.loc[c["mean_shift"].abs().idxmax()]
            rows.append(dict(
                prompt_set=pset, mode=mode, dose=dose,
                target=r.direction, target_shift=r.mean_shift,
                max_control=worst["direction"],
                max_control_shift=float(worst["mean_shift"]),
                ratio=abs(r.mean_shift) / abs(worst["mean_shift"])
                if worst["mean_shift"] else np.inf,
                # Welch on the two paired-difference means
                z=(abs(r.mean_shift) - abs(float(worst["mean_shift"])))
                / np.sqrt(r.sem ** 2 + float(worst["sem"]) ** 2),
                mean_control_shift=float(c["mean_shift"].mean()),
                n_controls=int(len(c)),
            ))
    return pd.DataFrame(rows)


def render(eff: pd.DataFrame, clean_levels: dict, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    legs = sorted(set(zip(eff["prompt_set"], eff["mode"])))
    fig, axes = plt.subplots(1, len(legs), figsize=(4.6 * len(legs), 4.2),
                             squeeze=False, sharey=True)
    colors = {"target_topk": C_BLUE, "target_rank1": C_GREEN,
              "random": C_ORANGE, "shuffled": C_GRAY, "heads_ctl": "#CC79A7"}
    for ax, (pset, mode) in zip(axes.flat, legs):
        sub = eff[(eff["prompt_set"] == pset) & (eff["mode"] == mode)]
        for d, g in sub.groupby("direction"):
            g = g.sort_values("dose")
            ax.errorbar(g["dose"], g["steered_level"], yerr=g["sem"],
                        marker="o", ms=4, lw=1.6 if d in TARGETS else 1.0,
                        ls="-" if d in TARGETS else "--",
                        color=colors.get(d, "0.5"), label=d)
        if pset in clean_levels:
            ax.axhline(clean_levels[pset], color="0.3", ls=":", lw=1,
                       label="clean")
        ax.axhline(0, color="0.85", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"steering coefficient $\alpha$")
        ax.set_title(f"{pset} / {mode}", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
    axes.flat[0].set_ylabel("reader attn mass: believed − current")
    fig.suptitle("Reader-head attention under steering "
                 "(dashed = magnitude-matched control directions)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = fig_dir / "attention_with_controls.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--main_label", default="L41_main")
    ap.add_argument("--null_label", default="L41_attn_null")
    args = ap.parse_args()

    root = STAGE2_OUT_ROOT / args.model_type
    frames = []
    for label in (args.main_label, args.null_label):
        a = load_attention(root / label)
        if a is None:
            print(f"[warn] no attention.csv in {label} — "
                  "the orthogonal-direction null is NOT available; run "
                  "slurm_scripts/circuit_stage2_steering_controls.sh task 0")
            continue
        print(f"loaded {label}: {len(a)} prompt-level rows, "
              f"{a['direction'].nunique()} directions")
        frames.append(a)
    if not frames:
        sys.exit(f"no attention data under {root}")
    per = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=KEY, keep="first")

    clean_levels = (per[per["direction"] == "clean"]
                    .groupby("prompt_set")["shift"].mean().to_dict())
    fb = clean_levels.get("false_belief", np.nan)
    tb = clean_levels.get("true_belief", np.nan)
    gap = float(fb - tb)
    print(f"\nclean levels: false_belief {fb:+.4f}, true_belief {tb:+.4f}  "
          f"-> natural gap {gap:.4f}")

    eff = paired_effects(per, gap).sort_values(
        ["prompt_set", "mode", "direction", "dose"])
    print("\n=== paired shift vs each prompt's own clean run ===")
    print(eff[["prompt_set", "mode", "direction", "dose", "n", "mean_shift",
               "t", "sign_consistency", "frac_natural_gap"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    null = target_vs_control(eff)
    if not null.empty:
        print("\n=== target vs magnitude-matched control directions ===")
        print(null.to_string(index=False,
                             float_format=lambda v: f"{v:+.4f}"))
    else:
        print("\n[no control-direction attention on disk — null not computed]")

    # sign-reversal control, computable from L41_main alone
    print("\n=== sign-reversal control (same direction, opposite sign) ===")
    for pset, on, off in (("true_belief", "inject", "delete"),
                          ("false_belief", "delete", "inject")):
        a = eff[(eff["prompt_set"] == pset) & (eff["mode"] == on)]
        b = eff[(eff["prompt_set"] == pset) & (eff["mode"] == off)]
        if a.empty or b.empty:
            print(f"  {pset}: {off} leg not on disk yet (queued)")
            continue
        for d in TARGETS:
            aa = a[a["direction"] == d].set_index("dose")["mean_shift"]
            bb = b[b["direction"] == d].set_index("dose")["mean_shift"]
            common = aa.index.intersection(bb.index)
            print(f"  {pset}/{d}: {on} {aa[common].round(4).tolist()} vs "
                  f"{off} {bb[common].round(4).tolist()}  (alphas "
                  f"{list(common)})")

    out_dir = root / "_attention_null"
    out_dir.mkdir(parents=True, exist_ok=True)
    eff.to_csv(out_dir / "attention_effects.csv", index=False)
    if not null.empty:
        null.to_csv(out_dir / "attention_null.csv", index=False)
    render(eff, clean_levels, out_dir)
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
