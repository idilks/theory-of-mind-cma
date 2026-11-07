"""Figure 12 — story RDMs (appendix), Qwen2.5-14B-Instruct.

Belief and photo counterparts of fig10: the frame's divergence set and answer
set beside the veracity and answer-identity models, over 144 story prompts (48
per template). Stimuli are ordered by veracity, then template, then answer.

The veracity model cannot be separated from answer position on these stimuli
(draft2 appendix, "Why code stimuli"), so this is a descriptive companion to the
code experiments, not a test of what the heads encode.

Head sets (--selection): top32 = top 32 by the frame's divergence fingerprint
and top 22 by its answer fingerprint, the belief sizes (T_C1, T_C5) applied to
both frames (the original figure); sig4of6 / sig3of6 = heads FWER-significant
in >= 4 (>= 3) of the condition's six cells, read from
results/aggregated/head_selection/ (analysis_scripts/build_significance_head_sets.py),
so set sizes are data-determined and differ by frame. A head-set RDM is the
unweighted mean of its heads' cosine RDMs, as in fig10. The Mantel values
printed here are written to aggregation_story_rdms/mantel_story_rdms[_{sel}].csv
and the figure to fig12_story_rdms[_{sel}].

Data: hook_z captures in results/rsa_head_outputs/activations_full{,_photo}/;
fingerprints in results/aggregated/fingerprints/.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rsa_head_outputs"))
from fig10_code_rdms import CMAP_RDM, color_limits, colorbar_right, frame
from rsa_utils import cosine_rdm
from analyze_code_position_rdms import mantel
import dims
import head_sets_source as hss
from style import BLUE, INK, ORANGE, apply_style, save

REPO = Path(__file__).resolve().parents[2]
RSA = REPO / "results/rsa_head_outputs"
FP = REPO / "results/aggregated/fingerprints"
OUT = RSA / "aggregation_story_rdms"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]
N_PERM, SEED = 1000, 0

# frame label, activations, divergence fingerprint, answer fingerprint
FRAMES = [("belief", "activations_full", "C1_abstract_belief", "C5_answer_changes_location_only"),
          ("photo", "activations_full_photo", "C2_abstract_photo",
           "C6_answer_changes_location_only_photo")]
K_DIV, K_ANS = 32, 22


def top_heads(name, k):
    return np.argsort(np.load(FP / f"{name}_clamped.npy"))[-k:]


def select_heads(name, k, selection):
    if selection == "top32":
        return top_heads(name, k)
    return np.array(sorted(hss.significance_set(name, selection, dims.QWEN)))


def set_rdm(z, heads):
    return np.mean([cosine_rdm(z[:, h, :]) for h in heads], axis=0)


def load(folder):
    ds = [torch.load(RSA / folder / f"{t}.pt", weights_only=False) for t in TEMPLATES]
    z = np.concatenate([d["z"].float().numpy() for d in ds])
    st = pd.concat([pd.DataFrame(d["stim"]) for d in ds], ignore_index=True)
    return z.reshape(len(st), 1920, -1), st


def draw(ax, M, order, bound, robust, labels=None):
    """One colormap for observed and model RDMs, numbered colorbars (as fig10)."""
    R = M[np.ix_(order, order)]
    vmin, vmax = color_limits(R, robust)
    im = ax.imshow(R, cmap=CMAP_RDM, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axhline(bound - 0.5, color="white", lw=1.2)
    ax.axvline(bound - 0.5, color="white", lw=1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if labels:
        ax.set_yticks([bound / 2 - 0.5, (bound + len(order)) / 2 - 0.5])
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.tick_params(axis="y", length=0)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="top32", choices=hss.SELECTIONS)
    args = ap.parse_args()
    suffix = "" if args.selection == "top32" else f"_{args.selection}"
    apply_style()
    OUT.mkdir(exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(6.2, 3.7))
    for i, (name, folder, fp_div, fp_ans) in enumerate(FRAMES):
        z, st = load(folder)
        ver = st.belief_veracity.astype(str).to_numpy()
        ans = st.answer_state.to_numpy()
        order = st.sort_values(["belief_veracity", "template", "answer_state"]).index.to_numpy()
        bound = int((ver == "False").sum())
        print(f"{name}: {len(st)} stimuli, {bound} diverged")
        heads = {"divergence": select_heads(fp_div, K_DIV, args.selection),
                 "answer": select_heads(fp_ans, K_ANS, args.selection)}
        sizes = {k: len(v) for k, v in heads.items()}
        print(f"  head sets ({args.selection}): {sizes}")
        obs = {k: set_rdm(z, v) for k, v in heads.items()}
        models = {"veracity": (ver[:, None] != ver[None, :]).astype(float),
                  "answer identity": (ans[:, None] != ans[None, :]).astype(float)}
        rng = np.random.default_rng(SEED)
        for s, m in ((s, m) for s in obs for m in models):
            r, p = mantel(obs[s], models[m], N_PERM, rng)
            rows.append({"frame": name, "head_set": s, "n_heads": sizes[s], "model": m,
                         "spearman_r": r, "p_perm": p, "selection": args.selection})
            print(f"  {s:10s} x {m:15s} r = {r:+.3f}  p = {p:.4f}")
        res = pd.DataFrame(rows).query("frame == @name").set_index(["head_set", "model"]).spearman_r

        for j, (s, c) in enumerate((("divergence", BLUE), ("answer", ORANGE))):
            n = sizes[s]
            ax = axes[i, j]
            im = draw(ax, obs[s], order, bound, True, ["diverged", "not\ndiverged"] if j == 0 else None)
            colorbar_right(ax, im)
            frame(ax, c, 2.0)
            ax.set_title(f"{name} prompts\n{s} heads ({n})", loc="left", fontsize=8, color=c)
        for j, (m, c) in enumerate((("veracity", BLUE), ("answer identity", ORANGE)), start=2):
            ax = axes[i, j]
            im = draw(ax, models[m], order, bound, False)
            colorbar_right(ax, im)
            frame(ax, c, 2.0)
            ax.set_title(f"model\n{m}", loc="left", fontsize=8, color=INK)
            ax.text(0.0, -0.05, f"r = {res[('divergence', m)]:.2f}", transform=ax.transAxes,
                    fontsize=8, color=BLUE, ha="left", va="top", fontweight="bold")
            ax.text(1.0, -0.05, f"{res[('answer', m)]:.2f}", transform=ax.transAxes,
                    fontsize=8, color=ORANGE, ha="right", va="top", fontweight="bold")
    fig.subplots_adjust(left=0.09, right=0.93, top=0.9, bottom=0.05, wspace=0.55, hspace=0.45)
    pd.DataFrame(rows).to_csv(OUT / f"mantel_story_rdms{suffix}.csv", index=False)
    save(fig, f"fig12_story_rdms{suffix}")


if __name__ == "__main__":
    main()
