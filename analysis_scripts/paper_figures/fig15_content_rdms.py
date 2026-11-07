"""Figure 15 — content-separable belief RDMs (appendix), Qwen2.5-14B-Instruct.

Story counterpart of fig10's code dissociation. The content stimulus set crosses
6 locations per template in all 30 ordered pairs with belief veracity and two
replicates (120 prompts per template, 3 templates). Pooled across templates every
head set, random included, fits template identity (r ~ 0.81), so the analysis is
run inside each template (analyze_content_rdms.py --within_template).

(a-d) One template's RDMs: the belief divergence set, the answer-retrieval set,
and the veracity and believed-location models. In this set the believed location
is the answer location (false belief: the first-mentioned location; true belief:
the current one). (e) Within-template Mantel r, mean over templates (bars),
per template (dots), and the size-matched random-head band (k = 22).

Head sets (as in analyze_content_rdms.py / head_sets_source.py):
  divergence = top-32 C1 heads minus top-22 C5 heads (22 heads)
  retrieval  = top-22 C5 heads
Mantel values are read from the saved analysis, not recomputed; the map panels
recompute only the RDMs being drawn.

Data: results/rsa_head_outputs/activations_content/{template}.pt,
results/rsa_head_outputs/aggregation_content/{mantel_content_within,random_band_within}.csv
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsa_head_outputs"))
from fig10_code_rdms import CMAP_MODEL, frame
from fig12_story_rdms import draw
from mvpa_veracity import valid_rows
from rsa_utils import cosine_rdm
from style import BLUE, CMAP_SEQ, GRAY, INK, INK2, ORANGE, apply_style, despine, save

REPO = Path(__file__).resolve().parents[2]
RSA = REPO / "results/rsa_head_outputs"
FP = REPO / "results/aggregated/fingerprints"
AGG = RSA / "aggregation_content"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]
MAP_TEMPLATE = "food_truck"
N_HEADS = 1920
CEILING = 0.866  # max Spearman r of a balanced two-class model RDM (60 vs 60)

SETS = [("div_unique", "divergence set", BLUE), ("c5_retrieval", "retrieval set", ORANGE)]
MODELS = [("veracity", "veracity\nfalse vs. true belief"),
          ("believed", "believed location\n(= answer)"),
          ("world", "current location")]


def top(name, k):
    return set(np.argsort(np.load(FP / f"{name}_clamped.npy"))[-k:].tolist())


def head_sets():
    c1, c5 = top("C1_abstract_belief", 32), top("C5_answer_changes_location_only", 22)
    return {"div_unique": sorted(c1 - c5), "c5_retrieval": sorted(c5)}


def load_template(t):
    """One template's capture, minus malformed rows (as analyze_content_rdms.py drops them:
    library_book.pt carries one all-NaN fragment row from a cluster write race)."""
    d = torch.load(RSA / "activations_content" / f"{t}.pt", weights_only=False)
    st = pd.DataFrame(d["stim"])
    z = d["z"].float().numpy()
    assert len(z) == len(st), (t, len(z), len(st))
    ok = valid_rows(st)
    if not ok.all():
        print(f"{t}: dropping {int((~ok).sum())} malformed row(s)")
    st = st[ok].reset_index(drop=True)
    assert len(st) == 120, (t, len(st))
    return z[ok].reshape(len(st), N_HEADS, -1), st


def main():
    apply_style()
    sets = head_sets()
    assert len(sets["div_unique"]) == 22 and len(sets["c5_retrieval"]) == 22, \
        {k: len(v) for k, v in sets.items()}

    z, st = load_template(MAP_TEMPLATE)
    assert (st.believed == st.answer_state).all(), "believed location should equal the answer"
    order = st.sort_values(["belief_veracity", "believed_idx", "world_idx", "replicate"]).index.to_numpy()
    bound = int((st.belief_veracity.astype(str) == "False").sum())
    print(f"{MAP_TEMPLATE}: {len(st)} stimuli, {bound} false belief")

    obs = {s: np.mean([cosine_rdm(z[:, h, :]) for h in sets[s]], axis=0) for s, _, _ in SETS}
    lab = {"veracity": st.belief_veracity.astype(str).to_numpy(), "believed": st.believed.to_numpy()}
    models = {m: (v[:, None] != v[None, :]).astype(float) for m, v in lab.items()}

    within = pd.read_csv(AGG / "mantel_content_within.csv")
    band = pd.read_csv(AGG / "random_band_within.csv").set_index("k").loc[22]
    r_map = within.query("template == @MAP_TEMPLATE").set_index("head_set")

    fig = plt.figure(figsize=(6.6, 4.9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.42, wspace=0.18)
    for j, (s, name, c) in enumerate(SETS):
        ax = fig.add_subplot(gs[0, j])
        draw(ax, obs[s], order, bound, CMAP_SEQ, True, ["false\nbelief", "true\nbelief"] if j == 0 else None)
        frame(ax, c, 2.0)
        ax.set_title(rf"$\bf{{{'ab'[j]}}}$  {name}" + f"\n({len(sets[s])} heads)", loc="left", fontsize=8, color=c)
    for j, (m, c) in enumerate((("veracity", BLUE), ("believed", ORANGE)), start=2):
        ax = fig.add_subplot(gs[0, j])
        draw(ax, models[m], order, bound, CMAP_MODEL, False)
        frame(ax, c, 2.0)
        ax.set_title(rf"$\bf{{{'abcd'[j]}}}$  model" + "\n" + dict(MODELS)[m].split("\n")[0],
                     loc="left", fontsize=8, color=INK)
        ax.text(0.0, -0.05, f"r = {r_map.loc['div_unique', f'r_{m}']:.2f}", transform=ax.transAxes,
                fontsize=8, color=BLUE, ha="left", va="top", fontweight="bold")
        ax.text(1.0, -0.05, f"{r_map.loc['c5_retrieval', f'r_{m}']:.2f}", transform=ax.transAxes,
                fontsize=8, color=ORANGE, ha="right", va="top", fontweight="bold")

    ax = fig.add_subplot(gs[1, :])
    w, x0 = 0.34, np.arange(len(MODELS))
    for i, (s, name, c) in enumerate(SETS):
        sub = within.query("head_set == @s").set_index("template")
        xs = x0 + (i - 0.5) * w
        means = [sub[f"r_{m}"].mean() for m, _ in MODELS]
        ax.bar(xs, means, w * 0.9, color=c, label=name, zorder=2)
        for k, (m, _) in enumerate(MODELS):
            ax.scatter(np.full(3, xs[k]) + np.array([-0.07, 0, 0.07]), sub.loc[TEMPLATES, f"r_{m}"],
                       s=9, color="white", edgecolor=INK, linewidth=0.6, zorder=3)
    for k, (m, _) in enumerate(MODELS):
        lo, hi = x0[k] - 0.42, x0[k] + 0.42
        ax.hlines(band[f"rand_mean_{m}"], lo, hi, color=GRAY, lw=1.2, ls="--", zorder=4,
                  label="random 22 heads: mean" if k == 0 else None)
        ax.hlines(band[f"rand_p95_{m}"], lo, hi, color=GRAY, lw=1.2, zorder=4,
                  label="random 22 heads: 95th pct." if k == 0 else None)
    ax.axhline(CEILING, color=INK2, lw=0.6, ls=":", zorder=1)
    ax.text(0.58, CEILING + 0.015, "two-class maximum (0.866)", fontsize=7, color=INK2,
            ha="left", va="bottom")
    ax.set_xticks(x0)
    ax.set_xticklabels([t for _, t in MODELS])
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("within-template Mantel r\n(Spearman)")
    ax.set_title(r"$\bf{e}$  model fits inside each template (bars: mean of 3; dots: per template)",
                 loc="left", fontsize=8.5)
    ax.legend(loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.2),
              handlelength=1.6, columnspacing=1.2)
    ax.grid(axis="x", visible=False)
    despine(ax)

    for s, _, _ in SETS:
        sub = within.query("head_set == @s")
        print(f"{s}: " + ", ".join(f"{m} {sub[f'r_{m}'].min():.3f}-{sub[f'r_{m}'].max():.3f}"
                                    for m, _ in MODELS))
    print("random k=22:", {m: (round(band[f'rand_mean_{m}'], 3), round(band[f'rand_p95_{m}'], 3))
                           for m, _ in MODELS})
    save(fig, "fig15_content_rdms")


if __name__ == "__main__":
    main()
