"""Rebuttal figure 5 — residual-stream steering dose-response (R3 point 3).

Sept 10 revision (critique.md pt 1): panel (c) is now ACCURACY under the
same two operators, replacing the "precondition absent" panel. The
precondition-absent legs (delete on true belief, inject into false belief)
are printed to stdout and reported in the compendium as one sentence.

Operators (see analysis_scripts/circuit/stage2_steering.py):
  v          = unit false-minus-true-belief mean-difference direction of the
               L41 residual stream at the final token (fit half only)
  s          = its natural scale ||mean_FB - mean_TB||
  rank-1     : x -> x + alpha * s * v   (inject)      x -> x - alpha * s * v   (delete)
  top-k      : x -> x + alpha * P_k(mu_FB - x) (inject) / P_k(mu_TB - x) (delete),
               P_k = projector onto the k = 8 writer-head subspace, mu = fit-half means
  controls   : additive, same alpha * s magnitude: random unit direction,
               label-permuted difference, sum of layer-matched non-writer heads

(a) delete on false-belief prompts, flip score
(b) inject into true-belief prompts, flip score
(c) accuracy on the same held-out prompts under both operators

Data: results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_main/
{summary,per_prompt}.csv; answer strings from
results/circuit/stage1_capture/Qwen2.5-14B-Instruct/*_behavior.csv.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import (BLUE, VIOLET, GRAY, INK2, apply_style, despine,
                   panel_label, save)

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_main"
CAPTURE = REPO / "results/circuit/stage1_capture/Qwen2.5-14B-Instruct"
TEMPLATES = ("food_truck", "hair_styling", "library_book")

DOSES = [0.0, 0.5, 1.0, 2.0, 4.0]
CTL_STYLE = {  # matched-norm controls, all in grey
    "random": ("-", "random direction"),
    "shuffled": ("--", "label-permuted"),
    "heads_ctl": (":", "non-significant heads"),
}
OPS = {  # natural operator per prompt set, and the reversed one
    "false_belief": ("delete", "inject"),
    "true_belief": ("inject", "delete"),
}


def leg(df, prompt_set, mode, direction, col="flip_score", sem="flip_sem"):
    """Dose-response including the clean point at alpha=0."""
    clean = df[(df.prompt_set == prompt_set) & (df["mode"] == "clean")]
    sub = df[(df.prompt_set == prompt_set) & (df["mode"] == mode)
             & (df.direction == direction)].sort_values("dose")
    x = [0.0] + sub.dose.tolist()
    y = [float(clean[col].iloc[0])] + sub[col].tolist()
    s = [float(clean[sem].iloc[0])] + sub[sem].tolist()
    return np.array(x), np.array(y), np.array(s)


def draw(ax, df, prompt_set, mode, direction, color, ls="-", lw=1.8,
         label=None, **kw):
    x, y, s = leg(df, prompt_set, mode, direction, **kw)
    ax.errorbar(x, y, yerr=s, color=color, ls=ls, lw=lw, marker="o", ms=3.5,
                capsize=0, elinewidth=0.9, label=label)
    return y


def accuracy_table():
    """Per-(prompt_set, mode, direction, dose) accuracy: the argmax
    next token is the first token of the correct location and not of the
    alternate (no first-token collisions in this stimulus set)."""
    pp = pd.read_csv(RUN / "per_prompt.csv")
    beh = pd.concat(
        pd.read_csv(CAPTURE / f"{t}_behavior.csv").assign(template=t)
        for t in TEMPLATES)
    assert not beh.first_token_collision.any()
    m = pp.merge(beh[["template", "idx", "variant", "correct", "alt"]],
                 left_on=["template", "pair_idx", "variant"],
                 right_on=["template", "idx", "variant"], how="inner")
    t1 = m.top1_str.astype(str).str.strip()
    m["acc"] = [len(t) > 0 and c.startswith(t) and not a.startswith(t)
                for t, c, a in zip(t1, m.correct, m.alt)]
    m["acc"] = m["acc"].astype(float)
    g = (m.groupby(["prompt_set", "mode", "direction", "dose"])["acc"]
          .agg(["mean", "sem", "count"]).reset_index())
    return g.rename(columns={"mean": "accuracy", "sem": "acc_sem", "count": "n"})


def main():
    apply_style()
    df = pd.read_csv(RUN / "summary.csv")
    acc = accuracy_table()

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.6, 3.0), sharex=True)

    # ---- (a) delete on false belief -------------------------------------
    y_topk = draw(a, df, "false_belief", "delete", "target_topk", BLUE,
                  label="divergence direction, top-k")
    draw(a, df, "false_belief", "delete", "target_rank1", VIOLET,
         label="divergence direction, rank-1")
    for ctl, (ls, name) in CTL_STYLE.items():
        draw(a, df, "false_belief", "delete", ctl, GRAY, ls=ls, lw=1.1,
             label=name)
    a.annotate(f"{y_topk[-1]:.2f}", (4, y_topk[-1]), xytext=(-2, 8),
               textcoords="offset points", fontsize=8, color=BLUE, ha="right")
    a.set_title("Delete on false belief", loc="left")
    a.set_ylabel(r"flip score $\phi_p$")
    a.legend(loc="upper left", fontsize=6.8, handlelength=1.7,
             labelspacing=0.35)
    panel_label(a, "a", x=-0.20)

    # ---- (b) inject into true belief ------------------------------------
    draw(b, df, "true_belief", "inject", "target_topk", BLUE)
    y_r1 = draw(b, df, "true_belief", "inject", "target_rank1", VIOLET)
    ctl_end = {}
    for ctl, (ls, name) in CTL_STYLE.items():
        yc = draw(b, df, "true_belief", "inject", ctl, GRAY, ls=ls, lw=1.1)
        ctl_end[ctl] = yc[-1]
    b.annotate(f"rank-1: {y_r1[-1]:.2f}", (4, y_r1[-1]), xytext=(-4, 9),
               textcoords="offset points", fontsize=8, color=VIOLET,
               ha="right")
    b.annotate("controls\n$\\leq$ " + f"{max(ctl_end.values()):.2f}",
               (4, max(ctl_end.values())), xytext=(-4, 10),
               textcoords="offset points", fontsize=7.5, color=INK2,
               ha="right", va="bottom")
    b.set_title("Inject into true belief", loc="left")
    b.set_ylabel(r"flip score $\phi_p$")
    panel_label(b, "b", x=-0.20)

    for ax in (a, b):
        ax.axhline(0.5, color=INK2, lw=0.7, ls=(0, (1, 2)), alpha=0.6, zorder=1)
        ax.set_ylim(0, 0.58)
    a.text(3.95, 0.512, "candidates tied", fontsize=6.8, color=INK2,
           ha="right", va="bottom")

    # ---- (c) accuracy under both operators ------------------------------
    kw = dict(col="accuracy", sem="acc_sem")
    series = [
        ("false_belief", "delete", "target_topk", BLUE, "-", "delete on false belief, top-k"),
        ("false_belief", "delete", "target_rank1", VIOLET, "-", "delete on false belief, rank-1"),
        ("true_belief", "inject", "target_topk", BLUE, "--", "inject into true belief, top-k"),
        ("true_belief", "inject", "target_rank1", VIOLET, "--", "inject into true belief, rank-1"),
    ]
    ends = {}
    for ps, mode, d, color, ls, label in series:
        ends[(ps, d)] = draw(c, acc, ps, mode, d, color, ls=ls, label=label,
                             **kw)[-1]
    # strongest control per operator (label-permuted, the "any two halves
    # differ" null) as the reference for how much a matched-norm edit costs
    for ps, ls in (("false_belief", "-"), ("true_belief", "--")):
        draw(c, acc, ps, OPS[ps][0], "shuffled", GRAY, ls=ls, lw=1.1,
             label=f"label-permuted control ({ps.replace('_', ' ')})", **kw)
    c.set_title("Accuracy under the same operators", loc="left")
    c.set_ylabel("accuracy (held-out prompts)")
    c.set_ylim(0, 1.04)
    c.legend(loc="lower left", fontsize=6.2, handlelength=2.0,
             labelspacing=0.3)
    panel_label(c, "c", x=-0.20)

    for ax in (a, b, c):
        despine(ax)
        ax.set_xticks(DOSES)
        ax.set_xlabel(r"steering coefficient $\alpha$")

    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_steering")

    # ---- numbers quoted in the compendium --------------------------------
    print("flip score  (alpha 0 -> 2 -> 4)")
    for ps, mode, d in [("false_belief", "delete", "target_rank1"),
                        ("false_belief", "delete", "target_topk"),
                        ("true_belief", "inject", "target_rank1"),
                        ("true_belief", "inject", "target_topk"),
                        ("true_belief", "delete", "target_rank1"),
                        ("true_belief", "delete", "target_topk"),
                        ("false_belief", "inject", "target_rank1"),
                        ("false_belief", "inject", "target_topk")]:
        x, y, _ = leg(df, ps, mode, d)
        tag = "" if mode == OPS[ps][0] else "  [precondition absent]"
        print(f"  {ps:13s} {mode:6s} {d:12s} "
              + " -> ".join(f"{v:.3f}" for v in (y[0], y[3], y[4])) + tag)
    print("accuracy (alpha 0 -> 2 -> 4)")
    for ps, mode, d in [("false_belief", "delete", "target_topk"),
                        ("false_belief", "delete", "target_rank1"),
                        ("false_belief", "delete", "shuffled"),
                        ("true_belief", "inject", "target_topk"),
                        ("true_belief", "inject", "target_rank1"),
                        ("true_belief", "inject", "shuffled"),
                        ("true_belief", "delete", "target_rank1"),
                        ("true_belief", "delete", "target_topk"),
                        ("false_belief", "inject", "target_rank1"),
                        ("false_belief", "inject", "target_topk")]:
        x, y, _ = leg(acc, ps, mode, d, **kw)
        tag = "" if mode == OPS[ps][0] else "  [precondition absent]"
        print(f"  {ps:13s} {mode:6s} {d:12s} "
              + " -> ".join(f"{v:.3f}" for v in (y[0], y[3], y[4])) + tag)
    acc.to_csv(RUN / "accuracy_by_config.csv", index=False)
    print(f"wrote {RUN / 'accuracy_by_config.csv'}")


if __name__ == "__main__":
    main()
