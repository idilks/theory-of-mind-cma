"""Figure 15b — content-separable belief RDMs, every template (appendix).

3 x 3 grid: rows are the three templates, columns the belief divergence set, the
answer-retrieval set, and one size-matched random set of never-selected heads (the
same 22 heads in every row). Each template's 120 prompts are sorted as in fig15
(veracity, believed location, current location, replicate), so the veracity model
is the 2 x 2 block split and the believed-location model the within-block diagonal
stripes (fig15 c-d). Under each map: Spearman r with the veracity and
believed-location model RDMs inside that template.

Divergence/retrieval r values are read from mantel_content_within.csv (and checked
against a recomputation); the random set's are computed here (no permutation p).
The random column is one draw; the 100-draw band is in fig15e.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsa_head_outputs"))
from fig10_code_rdms import frame
from fig12_story_rdms import draw
from fig15_content_rdms import AGG, N_HEADS, TEMPLATES, head_sets, load_template, top
from rsa_utils import cosine_rdm
from style import BLUE, CMAP_SEQ, GRAY, INK, ORANGE, apply_style, save

SEED = 0
TITLES = {"food_truck": "food truck", "hair_styling": "hair styling", "library_book": "library book"}


def random_set(k):
    claimed = set().union(*head_sets().values(), top("C1_abstract_belief", 32))
    pool = np.array([h for h in range(N_HEADS) if h not in claimed])
    return sorted(np.random.default_rng(SEED).choice(pool, k, replace=False).tolist())


def r_models(R, st):
    iu = np.triu_indices(len(st), 1)
    out = {}
    for m, col in (("veracity", "belief_veracity"), ("believed", "believed")):
        v = st[col].astype(str).to_numpy()
        out[m] = float(spearmanr(R[iu], (v[:, None] != v[None, :])[iu]).statistic)
    return out


def main():
    apply_style()
    sets = head_sets()
    sets["random"] = random_set(22)
    cols = [("div_unique", "divergence set", BLUE), ("c5_retrieval", "retrieval set", ORANGE),
            ("random", "random set", GRAY)]
    within = pd.read_csv(AGG / "mantel_content_within.csv").set_index(["head_set", "template"])

    fig, axes = plt.subplots(3, 3, figsize=(5.4, 6.0))
    for i, t in enumerate(TEMPLATES):
        z, st = load_template(t)
        order = st.sort_values(["belief_veracity", "believed_idx", "world_idx", "replicate"]).index.to_numpy()
        bound = int((st.belief_veracity.astype(str) == "False").sum())
        for j, (s, name, c) in enumerate(cols):
            R = np.mean([cosine_rdm(z[:, h, :]) for h in sets[s]], axis=0)
            r = r_models(R, st)
            if s != "random":
                for m in r:  # recomputation must match the saved analysis
                    assert abs(r[m] - within.loc[(s, t), f"r_{m}"]) < 1e-4, (s, t, m, r[m])
            ax = axes[i, j]
            draw(ax, R, order, bound, CMAP_SEQ, True,
                 ["false\nbelief", "true\nbelief"] if j == 0 else None)
            frame(ax, c, 2.0)
            if i == 0:
                ax.set_title(f"{name}\n(22 heads)", loc="left", fontsize=8.5, color=c)
            if j == 0:
                ax.text(-0.42, 0.5, TITLES[t], transform=ax.transAxes, rotation=90,
                        fontsize=8.5, ha="center", va="center", color=INK, fontweight="bold")
            ax.text(0.0, -0.04, f"veracity {r['veracity']:.2f}", transform=ax.transAxes,
                    fontsize=7.5, color=INK, ha="left", va="top")
            ax.text(1.0, -0.04, f"believed {r['believed']:.2f}", transform=ax.transAxes,
                    fontsize=7.5, color=INK, ha="right", va="top")
            print(f"{t:13s} {s:13s} veracity {r['veracity']:.3f}  believed {r['believed']:.3f}")
    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.04, wspace=0.12, hspace=0.3)
    save(fig, "fig15b_content_rdms_templates")


if __name__ == "__main__":
    main()
