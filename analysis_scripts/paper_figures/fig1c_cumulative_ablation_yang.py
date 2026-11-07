"""Figure 1c — cumulative ablation in the Yang et al. (2025) Fig. 5 format.

critique.md pt 3 (Sept 10): the flip-score curves are hard to read; show
ACCURACY, in the format of Yang et al. 2025 §3.4 — one red "Ablation" curve
and one black "Control" curve per panel, y = accuracy, x = number of heads
ablated over the full sweep of all 1,920 heads.

Layout: 2 x 2. Rows = head ordering (C1 belief fingerprint / C5 retrieval
fingerprint). Columns = evaluation task (theory-of-mind false belief /
Python code_copy, the two diverged variants). Control = rank-aligned
layer-matched lowest-fingerprint heads (Yang's control construction).

  --xmax 1920   full sweep (default)          -> fig1c_ablation_yang_accuracy
  --xmax 128    zoom to the first 128 heads   -> fig1c_ablation_yang_accuracy_zoom128

Data: results/ablation/tom_c{1,5}_full1920_on_code_tom_prefill_per_variant/
(matched system prompt + answer prefill; n = 75 prompts per variant; k grid
dense to 128 then stride 16, 241 sizes). Accuracy = argmax first token is
the correct answer (per_prompt_data.pt `acc`).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import INK, INK2, RED, apply_style, despine, panel_label, save

REPO = Path(__file__).resolve().parents[2]
RUNS = {
    "c1": REPO / "results/ablation/tom_c1_full1920_on_code_tom_prefill_per_variant",
    "c5": REPO / "results/ablation/tom_c5_full1920_on_code_tom_prefill_per_variant",
}
ROWS = [("c1", "Belief heads (C1 fingerprint order)"),
        ("c5", "Retrieval heads (C5 fingerprint order)")]
COLS = [("false_belief", "Theory of mind: false belief"),
        ("code_copy", "Python: code_copy")]
CURVES = [("control", INK, "Control"), ("target", RED, "Ablation")]


def load(run):
    df = pd.read_csv(run / "cumulative_ablation_results.csv")
    per = torch.load(run / "per_prompt_data.pt", map_location="cpu",
                     weights_only=False)
    return df, per


def curve(df, per, curve_name, variant):
    acc = per[curve_name][variant]["acc"].numpy().astype(float)
    k = np.asarray(per[f"{curve_name}_k_values"])
    idx = np.argsort(k)
    k, acc = k[idx], acc[:, idx]
    mean = acc.mean(axis=0)
    ref = (df[(df.curve == curve_name) & (df.eval_variant == variant)]
           .sort_values("k").accuracy.to_numpy())
    assert np.allclose(mean, ref, atol=1e-6), (curve_name, variant)
    return k, mean, acc.std(axis=0, ddof=1) / np.sqrt(acc.shape[0])


def first_k_below(k, m, thr=0.5):
    hit = np.nonzero(m < thr)[0]
    return int(k[hit[0]]) if hit.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=int, default=1920)
    ap.add_argument("--ribbon", action="store_true",
                    help="add ±1 SEM ribbons (off by default, as in Yang)")
    args = ap.parse_args()
    apply_style()
    plt.rcParams["axes.grid"] = False

    data = {key: load(run) for key, run in RUNS.items()}
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.4), sharex=True,
                             sharey=True,
                             gridspec_kw={"hspace": 0.32, "wspace": 0.10})
    stats = []
    for i, (okey, olabel) in enumerate(ROWS):
        df, per = data[okey]
        for j, (variant, vlabel) in enumerate(COLS):
            ax = axes[i, j]
            for cname, color, clabel in CURVES:
                k, m, s = curve(df, per, cname, variant)
                sel = k <= args.xmax
                ax.plot(k[sel], m[sel], color=color, lw=1.4, label=clabel,
                        zorder=3 if cname == "target" else 2)
                if args.ribbon:
                    ax.fill_between(k[sel], (m - s)[sel], (m + s)[sel],
                                    color=color, alpha=0.12, lw=0)
                stats.append((okey, variant, cname, float(m[0]),
                              first_k_below(k, m),
                              {kk: round(float(m[k == kk][0]), 3)
                               for kk in (8, 16, 32, 64, 128) if kk in k}))
            despine(ax)
            ax.set_ylim(-0.02, 1.04)
            ax.set_xlim(-args.xmax * 0.02, args.xmax * 1.02)
            ax.set_xticks([0, 32, 64, 96, 128] if args.xmax <= 128
                          else [0, 500, 1000, 1500, 1920])
            ax.set_title(vlabel if i == 0 else "", loc="left")
            if j == 0:
                ax.set_ylabel(f"{olabel}\naccuracy", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("number of heads ablated (of 1,920)")
            panel_label(ax, "abcd"[2 * i + j], x=-0.20 if j == 0 else -0.06)
    axes[0, 0].legend(loc="upper right", fontsize=8, handlelength=1.8)
    fig.text(0.5, 0.975,
             "Cumulative ablation, matched-prompt + prefill protocol, "
             "n = 75 held-out prompts per task (SEM ribbons "
             + ("on" if args.ribbon else "omitted") + ")",
             ha="center", fontsize=8.5, color=INK2)
    fig.subplots_adjust(top=0.90)
    suffix = "" if args.xmax >= 1920 else f"_zoom{args.xmax}"
    save(fig, f"fig1c_ablation_yang_accuracy{suffix}")

    print("ordering variant      curve    acc(k=0)  first k acc<0.5  acc at k=8/16/32/64/128")
    for okey, variant, cname, a0, kh, pts in stats:
        print(f"{okey:8s} {variant:12s} {cname:8s} {a0:6.3f}    {str(kh):>6s}      {pts}")


if __name__ == "__main__":
    main()
