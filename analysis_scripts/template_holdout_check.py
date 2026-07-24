"""Template hold-out check (July 2026 plan, step 3).

Reviewer 2: "The selection and ablation of heads are from the same
distribution. There is no description of a test split, so this might
introduce template overfitting."

Two analyses over the three scenario templates (food_truck, hair_styling,
library_book), per condition C1/C2/C3/C5/C6:

1. Cross-template fingerprint agreement — per-template fingerprints
   (mean of that template's direction cells, clamped per-cell then averaged)
   correlated pairwise. High r = the head ranking is not a property of any
   one story.

2. Leave-one-template-out (LOTO) selection — significant heads chosen from
   the two held-in templates' cells (significant in >= half of those cells),
   then evaluated on the held-out template:
     - replication rate: fraction of selected heads significant in >= 1 of
       the held-out template's 2 direction cells
     - chance level: n_heldout_sig / 1920 (random selection)
     - enrichment: replication / chance
     - median percentile of selected heads in the held-out clamped
       fingerprint (random selection -> ~50)

Outputs -> writeups/results_for_pi/template_holdout/
  figures/*.png, stats_crosstemplate.csv, stats_loto.csv, stats_fragment.md
"""

import csv as csvmod
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from condition_registry import resolve_best_source  # noqa: E402
from h2_additivity_robustness import (  # noqa: E402
    N_LAYERS, N_HEADS, BLUE, ORANGE, GREEN, GRAY,
    RESULTS_ROOT, CROSS_RULES, WITHIN_RULES)

OUT = REPO / "writeups/results_for_pi/template_holdout"
FIG = OUT / "figures"

TEMPLATES = ["template_food_truck", "template_hair_styling", "template_library_book"]
TSHORT = {t: t.replace("template_", "") for t in TEMPLATES}

CONDS = {
    "C1": ("abstract_belief", CROSS_RULES),
    "C2": ("abstract_photo", CROSS_RULES),
    "C3": ("answer_changes_belief", CROSS_RULES),
    "C5": ("answer_changes_location_only", WITHIN_RULES),
    "C6": ("answer_changes_location_only_photo", WITHIN_RULES),
}


def load_condition_cells(folder, rules):
    """dict[(template, rule)] -> {'scores': clamped 1920, 'sig': set}."""
    best = resolve_best_source(RESULTS_ROOT, [folder], rules)
    cells = {}
    for (tmpl, rule), e in sorted(best.items()):
        s = torch.load(e["scores_pt"], map_location="cpu",
                       weights_only=False).float().numpy().ravel()
        sig = set()
        p = Path(e["csv"])
        if p.exists():
            with open(p) as f:
                for row in csvmod.DictReader(f):
                    if row["significant"].strip().lower() == "true":
                        sig.add(int(row["layer"]) * N_HEADS + int(row["head"]))
        cells[(tmpl, rule)] = {"scores": np.clip(s, 0, None), "sig": sig,
                               "n": e["n"], "seed": e["seed_tag"]}
    return cells


def template_fp(cells, tmpl):
    arrs = [c["scores"] for (t, _), c in cells.items() if t == tmpl]
    return np.mean(arrs, axis=0) if arrs else None


def sig_by_count(cells, tmpls, min_count):
    counts = {}
    for (t, _), c in cells.items():
        if t in tmpls:
            for h in c["sig"]:
                counts[h] = counts.get(h, 0) + 1
    return {h for h, v in counts.items() if v >= min_count}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    all_cells = {k: load_condition_cells(f, r) for k, (f, r) in CONDS.items()}

    # ---------------- 1. cross-template agreement ----------------
    ct_rows = []
    for cond, cells in all_cells.items():
        fps = {t: template_fp(cells, t) for t in TEMPLATES}
        for ta, tb in combinations(TEMPLATES, 2):
            if fps[ta] is None or fps[tb] is None:
                continue
            r = float(np.corrcoef(fps[ta], fps[tb])[0, 1])
            ct_rows.append({"condition": cond, "template_a": TSHORT[ta],
                            "template_b": TSHORT[tb], "pearson_r": r})
    df_ct = pd.DataFrame(ct_rows)

    # ---------------- 2. leave-one-template-out ----------------
    loto_rows = []
    for cond, cells in all_cells.items():
        n_cells_per_tmpl = {t: sum(1 for (tt, _) in cells if tt == t)
                            for t in TEMPLATES}
        for held_out in TEMPLATES:
            train = [t for t in TEMPLATES if t != held_out]
            n_train_cells = sum(n_cells_per_tmpl[t] for t in train)
            if n_train_cells == 0 or n_cells_per_tmpl[held_out] == 0:
                continue
            sel = sig_by_count(cells, train, int(np.ceil(n_train_cells / 2)))
            held_sig = sig_by_count(cells, [held_out], 1)  # sig in >=1 of 2 cells
            if not sel:
                continue
            rep = len(sel & held_sig) / len(sel)
            chance = len(held_sig) / (N_LAYERS * N_HEADS)
            fp_out = template_fp(cells, held_out)
            pct = [sps_percentile(fp_out, h) for h in sel]
            loto_rows.append({
                "condition": cond, "held_out": TSHORT[held_out],
                "n_selected": len(sel), "n_heldout_sig": len(held_sig),
                "n_replicated": len(sel & held_sig),
                "replication_rate": rep, "chance_rate": chance,
                "enrichment": rep / chance if chance > 0 else np.inf,
                "median_pctile_heldout_fp": float(np.median(pct))})
    df_loto = pd.DataFrame(loto_rows)

    # ---------------- figures ----------------
    # F1: pairwise cross-template r, dots per pair + mean bar
    conds = list(CONDS)
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for i, cond in enumerate(conds):
        vals = df_ct[df_ct.condition == cond].pearson_r.values
        if len(vals) == 0:
            continue
        ax.bar(i, vals.mean(), width=0.6, color=BLUE, alpha=0.35, zorder=2)
        ax.scatter([i] * len(vals), vals, s=42, color=BLUE, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.text(i, vals.mean() - 0.06, f"{vals.mean():.2f}", ha="center",
                fontsize=9, color="#1a1a1a", zorder=4)
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("cross-template Pearson r")
    ax.set_ylim(0, 1.0)
    ax.set_title("Fingerprint agreement across scenario templates\n"
                 "(dots = the 3 template pairs, bar = mean)", fontsize=10)
    fig.savefig(FIG / "fig1_cross_template_r.png")
    plt.close(fig)

    # F2: LOTO replication vs chance
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    width = 0.26
    xbase = np.arange(len(conds)) * 1.4
    for j, held in enumerate([TSHORT[t] for t in TEMPLATES]):
        sub = df_loto[df_loto.held_out == held].set_index("condition")
        xs = [xbase[i] + (j - 1) * width for i, c in enumerate(conds) if c in sub.index]
        reps = [sub.loc[c, "replication_rate"] for c in conds if c in sub.index]
        chances = [sub.loc[c, "chance_rate"] for c in conds if c in sub.index]
        color = [BLUE, ORANGE, GREEN][j]
        ax.bar(xs, reps, width=width * 0.92, color=color, label=f"held out: {held}")
        for x, ch in zip(xs, chances):
            ax.plot([x - width / 2, x + width / 2], [ch, ch], color="#1a1a1a",
                    lw=1.6, zorder=4)
    ax.plot([], [], color="#1a1a1a", lw=1.6, label="chance (random heads)")
    ax.set_xticks(xbase)
    ax.set_xticklabels(conds)
    ax.set_ylabel("fraction of selected heads significant\nin held-out template")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.set_title("Leave-one-template-out: heads picked on 2 templates replicate on the 3rd",
                 fontsize=10)
    fig.savefig(FIG / "fig2_loto_replication.png")
    plt.close(fig)

    # F3: train vs held-out fingerprint scatters for C1 (3 folds)
    cells = all_cells["C1"]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0))
    for ax, held_out in zip(axes, TEMPLATES):
        train = [t for t in TEMPLATES if t != held_out]
        fp_train = np.mean([template_fp(cells, t) for t in train], axis=0)
        fp_out = template_fp(cells, held_out)
        n_train_cells = sum(1 for (tt, _) in cells if tt in train)
        sel = sig_by_count(cells, train, int(np.ceil(n_train_cells / 2)))
        mask = np.zeros(N_LAYERS * N_HEADS, dtype=bool)
        for h in sel:
            mask[h] = True
        r = np.corrcoef(fp_train, fp_out)[0, 1]
        ax.scatter(fp_train[~mask], fp_out[~mask], s=6, color=GRAY, alpha=0.3,
                   linewidths=0)
        ax.scatter(fp_train[mask], fp_out[mask], s=26, color=BLUE, alpha=0.9,
                   linewidths=0, label=f"selected on train (n={mask.sum()})")
        ax.set_xlabel(f"train fingerprint ({' + '.join(TSHORT[t] for t in train)})",
                      fontsize=9)
        ax.set_ylabel(f"held-out fingerprint ({TSHORT[held_out]})", fontsize=9)
        ax.set_title(f"C1, hold out {TSHORT[held_out]}  ·  r = {r:.3f}", fontsize=10)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(FIG / "fig3_c1_folds.png")
    plt.close(fig)

    # ---------------- tables ----------------
    df_ct.to_csv(OUT / "stats_crosstemplate.csv", index=False)
    df_loto.to_csv(OUT / "stats_loto.csv", index=False)
    frag = ["<!-- AUTO-GENERATED by scripts/template_holdout_check.py -->\n",
            "### Cross-template fingerprint correlations (clamped)\n",
            df_ct.round(3).to_markdown(index=False) + "\n",
            "\n### Leave-one-template-out replication\n",
            df_loto.round(3).to_markdown(index=False) + "\n"]
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    print("== cross-template mean r per condition ==")
    print(df_ct.groupby("condition").pearson_r.mean().round(3).to_string())
    print("\n== LOTO summary ==")
    print(df_loto.round(3).to_string(index=False))
    print(f"\nwrote {OUT}")


def sps_percentile(arr, idx):
    return 100.0 * (arr < arr[idx]).mean()


if __name__ == "__main__":
    main()
