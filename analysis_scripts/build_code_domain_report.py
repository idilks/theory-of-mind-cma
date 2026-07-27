"""Build figures + stats for the code-domain CMA report.

Reads best-available cells via condition_registry (ToM: seed-priority
combined>1>0; code: seed 0 indexed_list) and writes PNGs, CSVs and a
markdown stats fragment to writeups/results_for_pi/code_domain/.
Rerun after new cells land (e.g. code_answer_changes), then re-render
writeups/results_for_pi/code_domain_results.qmd.

Heatmap rendering follows the PI rule: clamp negatives to 0, no
percentile/floor thresholding, vmax = positive max.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from condition_registry import resolve_best_source  # noqa: E402

RESULTS_ROOT = REPO / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
OUT = REPO / "writeups/results_for_pi/code_domain"
INTERSECTION_CSV = REPO / "other_results/ablation/decomp_intersection_belief_rank/head_list.csv"

CROSS_RULES = {("ABA", "ABB"), ("ABB", "ABA")}
WITHIN_RULES = {("ABA", "ABA"), ("ABB", "ABB")}

ABLATION_ROOT = REPO / "other_results/ablation"
CUMAB_ORDERINGS = [("tom_c1", "C1 belief-fingerprint ordering"),
                   ("tom_c5", "C5 answer-fingerprint ordering")]
CUMAB_PROTOCOLS = [("per_variant", "quote-prefill (primary)"),
                   ("none", "free generation")]
CUMAB_VARIANTS = [("code_copy", "code copy\n(false-belief equiv)"),
                  ("code_alias", "code alias\n(true-belief equiv)"),
                  ("false_belief", "ToM false belief"),
                  ("true_belief", "ToM true belief"),
                  ("basic_ab", "ToM basic retrieval")]
# Yang et al. 2025 figure convention: red ablation curve, black control
CURVE_COLORS = {"target": "#E8000B", "control": "black"}
CURVE_LABELS = {"target": "Ablation", "control": "Control"}

# (key, results folder, allowed rule pairs, template filter, display label)
# Code conditions restricted to template_indexed_list: template_list_append
# holds pre-July-2026 runs of the removed zero-signal design.
CONDITIONS = [
    ("code_abs", "code_abstract", CROSS_RULES, "template_indexed_list", "code abstract"),
    ("code_ac", "code_answer_changes", CROSS_RULES, "template_indexed_list", "code answer-changes"),
    ("code_ctl", "code_control", WITHIN_RULES, "template_indexed_list", "code control"),
    ("C1", "abstract_belief", CROSS_RULES, None, "C1 belief"),
    ("C2", "abstract_photo", CROSS_RULES, None, "C2 photo"),
    ("C3", "answer_changes_belief", CROSS_RULES, None, "C3 ans-changes belief"),
    ("C4", "answer_changes_photo", CROSS_RULES, None, "C4 ans-changes photo"),
    ("C5", "answer_changes_location_only", WITHIN_RULES, None, "C5 loc-only"),
    ("C6", "answer_changes_location_only_photo", WITHIN_RULES, None, "C6 loc-only photo"),
    ("C7", "state_only_no_belief", WITHIN_RULES, None, "C7 state-only"),
]
LABELS = {k: lab for k, _, _, _, lab in CONDITIONS}

# Okabe-Ito (CVD-safe), fixed assignment per head set
SET_COLORS = {"intersection": "#0072B2", "code_abs": "#D55E00", "code_ctl": "#009E73"}


def load_cells(folder, rules, template=None):
    best = resolve_best_source(RESULTS_ROOT, [folder], rules)
    cells = {}
    for (tmpl, rule), e in best.items():
        if template is not None and tmpl != template:
            continue
        scores = torch.load(e["scores_pt"], map_location="cpu", weights_only=False).float()
        cells[(tmpl, rule)] = {"scores": scores.numpy(), "n": e["n"],
                               "seed": e["seed_tag"], "csv": e["csv"]}
    return cells


def sig_heads_from_csv(path):
    heads = []
    if not Path(path).exists():
        return heads
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get("significant", "").strip() in ("True", "true", "1"):
                heads.append((int(row["layer"]), int(row["head"]),
                              float(row["observed_score"])))
    return heads


def perm_p(a, b, n_perm=10000, seed=0):
    """Permutation p for Pearson r: shuffle b's head labels."""
    rng = np.random.default_rng(seed)
    obs = np.corrcoef(a, b)[0, 1]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = np.corrcoef(a, rng.permutation(b))[0, 1]
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1))


def clamped(fp):
    return np.clip(fp, 0, None)


def build_cumulative_section():
    """Figures + stats fragment for the full-1920 Yang-style cumulative
    ablation (C1/C5 orderings on code_copy/code_alias + co-run ToM evals)."""
    curves = {}  # (ordering, protocol) -> DataFrame
    for okey, _ in CUMAB_ORDERINGS:
        for pkey, _ in CUMAB_PROTOCOLS:
            path = (ABLATION_ROOT / f"{okey}_full1920_on_code_tom_prefill_{pkey}"
                    / "cumulative_ablation_results.csv")
            if path.exists():
                curves[(okey, pkey)] = pd.read_csv(path)
            else:
                print(f"[skip] cumulative ablation {okey}/{pkey}: {path} missing")
    if not curves:
        (OUT / "cumab_stats.md").write_text(
            "_Cumulative-ablation results not pulled yet — rerun after "
            "`pull_results.sh` and merge._\n")
        return

    legend_handles = [
        Line2D([0], [0], color=CURVE_COLORS["target"], lw=1.5,
               label="Ablation"),
        Line2D([0], [0], color=CURVE_COLORS["control"], lw=1.5,
               label="Control"),
    ]

    def _style_axis(ax, xmax):
        pad = xmax * 0.02
        ax.set_xlim(-pad, xmax + pad)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([0, 500, 1000, 1500] if xmax > 200
                      else [0, 32, 64, 96, 128])
        ax.spines[["top", "right"]].set_visible(False)

    views = [(1920, "", "full sweep"), (128, "_zoom", "zoom: first 128 heads")]
    for okey, olabel in CUMAB_ORDERINGS:
        if not any(k[0] == okey for k in curves):
            continue
        for xmax, suffix, vlab in views:
            fig, axes = plt.subplots(2, len(CUMAB_VARIANTS), figsize=(19, 7.5),
                                     sharex=True, sharey=True,
                                     constrained_layout=True)
            for r, (pkey, plabel) in enumerate(CUMAB_PROTOCOLS):
                df = curves.get((okey, pkey))
                for c, (vkey, vlabel) in enumerate(CUMAB_VARIANTS):
                    ax = axes[r, c]
                    if df is None:
                        ax.axis("off")
                        continue
                    for curve in ("control", "target"):
                        sub = df[(df.eval_variant == vkey)
                                 & (df.curve == curve) & (df.k <= xmax)]
                        sub = sub.sort_values("k")
                        ax.plot(sub.k, sub.accuracy, color=CURVE_COLORS[curve],
                                linewidth=1.3)
                    _style_axis(ax, xmax)
                    if r == 0:
                        ax.set_title(vlabel, fontsize=12)
                    if c == 0:
                        ax.set_ylabel(f"{plabel}\naccuracy", fontsize=11)
                    if r == 1:
                        ax.set_xlabel("Number of heads ablated")
            axes[0, 0].legend(handles=legend_handles, loc="upper right",
                              fontsize=11, frameon=False)
            fig.suptitle(f"Cumulative ablation, {olabel} — accuracy "
                         f"({vlab}, n=75/variant)", fontsize=14,
                         ha="left", x=0.02)
            fig.savefig(OUT / f"cumab_{okey}{suffix}.png", dpi=150)
            plt.close(fig)

    # ---- flip figure: does copy turn INTO alias? -------------------------
    flip_rows = [("per_variant", "flip_score",
                  "quote-prefill\nflip score  P(alt)/(P(alt)+P(correct))"),
                 ("none", "flip_rate",
                  "free generation\nflip rate  (answers = counterfactual)")]
    flip_cols = [(o, v) for o, _ in CUMAB_ORDERINGS
                 for v in ("code_copy", "code_alias")]
    if any((o, p) in curves for o, _ in CUMAB_ORDERINGS
           for p, _, _ in flip_rows):
        for xmax, suffix, vlab in views:
            fig, axes = plt.subplots(2, len(flip_cols), figsize=(16.5, 7),
                                     sharex=True, sharey=True,
                                     constrained_layout=True)
            for r, (pkey, metric, rlabel) in enumerate(flip_rows):
                for c, (okey, vkey) in enumerate(flip_cols):
                    ax = axes[r, c]
                    df = curves.get((okey, pkey))
                    if df is None:
                        ax.axis("off")
                        continue
                    for curve in ("control", "target"):
                        sub = df[(df.eval_variant == vkey)
                                 & (df.curve == curve)
                                 & (df.k <= xmax)].sort_values("k")
                        ax.plot(sub.k, sub[metric], color=CURVE_COLORS[curve],
                                linewidth=1.3)
                    if metric == "flip_score":
                        ax.axhline(0.5, color="#888", linewidth=0.8,
                                   linestyle=":", zorder=0)
                    _style_axis(ax, xmax)
                    if r == 0:
                        order = ("C1 (belief)" if okey == "tom_c1"
                                 else "C5 (answer)")
                        variant = ("code copy (FB equiv)"
                                   if vkey == "code_copy"
                                   else "code alias (TB equiv)")
                        ax.set_title(f"{order} ordering — {variant}",
                                     fontsize=12)
                    if c == 0:
                        ax.set_ylabel(rlabel, fontsize=10)
                    if r == 1:
                        ax.set_xlabel("Number of heads ablated")
            axes[0, 1].legend(handles=legend_handles, loc="upper left",
                              fontsize=11, frameon=False)
            fig.suptitle("Flip toward the counterfactual (alias) answer, "
                         f"copy vs alias variants ({vlab})", fontsize=14,
                         ha="left", x=0.02)
            fig.savefig(OUT / f"cumab_flip{suffix}.png", dpi=150)
            plt.close(fig)

    # ---- stats fragment: k½ (first k with accuracy < 0.5) ----------------
    def k_half(df, vkey, curve):
        sub = df[(df.eval_variant == vkey)
                 & (df.curve == curve)].sort_values("k")
        below = sub[sub.accuracy < 0.5]
        return str(int(below.k.iloc[0])) if len(below) else "never"

    lines = ["### How many heads does it take? (k½)\n",
             "First k at which accuracy drops below 50% — ablation curve vs "
             "the same-layer control curve. A small ablation k½ next to a "
             "large control k½ = the damage is specific to those heads, not "
             "to removing heads per se.\n"]
    for pkey, plabel in CUMAB_PROTOCOLS:
        avail = [(o, ol) for o, ol in CUMAB_ORDERINGS if (o, pkey) in curves]
        if not avail:
            continue
        header = "| eval | " + " | ".join(
            f"{ol.split()[0]} ablation | {ol.split()[0]} control"
            for _, ol in avail) + " |"
        lines += [f"\n**{plabel}**\n", header,
                  "|" + "---|" * (1 + 2 * len(avail))]
        for vkey, vlabel in CUMAB_VARIANTS:
            cells = []
            for okey, _ in avail:
                df = curves[(okey, pkey)]
                cells += [k_half(df, vkey, "target"),
                          k_half(df, vkey, "control")]
            lines.append(f"| {vlabel.replace(chr(10), ' ')} | "
                         + " | ".join(cells) + " |")
    (OUT / "cumab_stats.md").write_text("\n".join(lines) + "\n")
    print(f"wrote cumulative-ablation figures + cumab_stats.md "
          f"({len(curves)}/4 configs)")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    build_cumulative_section()

    data, fps = {}, {}
    for key, folder, rules, template, _ in CONDITIONS:
        cells = load_cells(folder, rules, template)
        if not cells:
            print(f"[skip] {key}: no cells found")
            continue
        data[key] = cells
        fps[key] = np.mean([c["scores"] for c in cells.values()], axis=0)
        seeds = {c["seed"] for c in cells.values()}
        print(f"[ok] {key}: {len(cells)} cells, seeds={sorted(seeds)}, "
              f"n={[c['n'] for c in cells.values()]}")

    present = list(fps)
    n_layers, n_heads = next(iter(fps.values())).shape

    # ---- fingerprint heatmap grid --------------------------------------
    # row 1: abstract/divergence; row 2: control/answer; row 3: answer-changes
    grid_rows = [["C1", "C2", "code_abs"],
                 ["C5", "C6", "code_ctl"],
                 ["C3", "C4", "code_ac"]]
    fig, axes = plt.subplots(3, 3, figsize=(17, 16), constrained_layout=True)
    for r, row_keys in enumerate(grid_rows):
        for c, key in enumerate(row_keys):
            ax = axes[r, c]
            if key not in fps:
                ax.axis("off")
                continue
            pos = clamped(fps[key])
            im = ax.imshow(pos, aspect="auto", cmap="viridis", vmin=0,
                           vmax=pos.max(), interpolation="nearest")
            ax.set_title(f"{LABELS[key]}  (max {pos.max():.2f})", fontsize=12)
            ax.set_xlabel("head")
            ax.set_ylabel("layer")
            fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("Mean causal-score fingerprints — abstract / control / "
                 "answer-changes rows (negatives clamped to 0, per-panel vmax)",
                 fontsize=13)
    fig.savefig(OUT / "fp_grid.png", dpi=150)
    plt.close(fig)

    # ---- correlation matrix --------------------------------------------
    M = np.stack([fps[k].flatten() for k in present])
    C = np.corrcoef(M)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(C, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(present)), [LABELS[k] for k in present],
                  rotation=35, ha="right")
    ax.set_yticks(range(len(present)), [LABELS[k] for k in present])
    for i in range(len(present)):
        for j in range(len(present)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.6 else "black", fontsize=10)
    ax.set_title("Pearson r between head fingerprints (1920 heads)")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.savefig(OUT / "corr_matrix.png", dpi=150)
    plt.close(fig)

    # permutation p for headline correlations
    headline = []
    for a, b in [("code_abs", "C1"), ("code_abs", "C2"),
                 ("code_ctl", "C5"), ("code_ctl", "C7"),
                 ("code_abs", "code_ctl")]:
        if a in fps and b in fps:
            r, p = perm_p(fps[a].flatten(), fps[b].flatten())
            headline.append((a, b, r, r * r, p))

    # ---- head sets -------------------------------------------------------
    inter = set()
    if INTERSECTION_CSV.exists():
        with open(INTERSECTION_CSV) as f:
            inter = {(int(r["layer"]), int(r["head"])) for r in csv.DictReader(f)}

    cond_sig = {}  # key -> {(l,h): best |score|}
    for key in present:
        agg = {}
        for c in data[key].values():
            for l, h, s in sig_heads_from_csv(c["csv"]):
                agg[(l, h)] = max(agg.get((l, h), 0.0), abs(s))
        cond_sig[key] = agg

    # C1 fingerprint rank lookup (1 = strongest)
    c1_rank = {}
    if "C1" in fps:
        flat = fps["C1"].flatten()
        order = np.argsort(-flat)
        for rank, idx in enumerate(order, start=1):
            c1_rank[(idx // n_heads, idx % n_heads)] = rank

    # heads table: any head significant in a code cell
    rows = []
    code_keys = [k for k in ("code_abs", "code_ac", "code_ctl") if k in cond_sig]
    all_code = sorted(set().union(*(cond_sig[k] for k in code_keys)))
    for (l, h) in all_code:
        rows.append({
            "layer": l, "head": h,
            **{f"{k}_score": round(cond_sig[k].get((l, h), 0.0), 3) for k in code_keys},
            "in_C1xC2_intersection": (l, h) in inter,
            "C1_fingerprint_rank": c1_rank.get((l, h), ""),
        })
    with open(OUT / "code_significant_heads.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- paired scatters: overlap + R^2 in one figure ---------------------
    pairs = [(a, b) for a, b in [("code_abs", "C1"), ("code_ctl", "C5")]
             if a in fps and b in fps]
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6.5),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        for ax, (a, b) in zip(axes, pairs):
            xa, xb = fps[b].flatten(), fps[a].flatten()  # x: ToM, y: code
            sig_a, sig_b = set(cond_sig.get(a, {})), set(cond_sig.get(b, {}))
            idx = lambda s: [l * n_heads + h for l, h in s]
            both = sig_a & sig_b
            only_a = sig_a - sig_b
            only_b = sig_b - sig_a
            rest = np.ones(xa.size, bool)
            for s in (both, only_a, only_b):
                rest[idx(s)] = False
            ax.scatter(xa[rest], xb[rest], s=8, color="#B0B0B0", alpha=0.4,
                       edgecolors="none", label="other heads")
            ax.scatter(xa[idx(only_b)], xb[idx(only_b)], s=55, marker="o",
                       facecolors="none", edgecolors="#0072B2", linewidths=1.8,
                       label=f"sig in {LABELS[b]} only [{len(only_b)}]")
            ax.scatter(xa[idx(only_a)], xb[idx(only_a)], s=55, marker="s",
                       facecolors="none", edgecolors="#009E73", linewidths=1.8,
                       label=f"sig in {LABELS[a]} only [{len(only_a)}]")
            ax.scatter(xa[idx(both)], xb[idx(both)], s=70, marker="X",
                       color="#D55E00", label=f"sig in both [{len(both)}]")
            r = np.corrcoef(xa, xb)[0, 1]
            ax.set_xlabel(f"{LABELS[b]} causal score")
            ax.set_ylabel(f"{LABELS[a]} causal score")
            ax.set_title(f"{LABELS[a]} vs {LABELS[b]}:  r = {r:.3f},  "
                         f"R² = {r * r:.3f}")
            ax.grid(alpha=0.25, linewidth=0.5)
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
        fig.savefig(OUT / "scatter_pairs.png", dpi=150)
        plt.close(fig)

    # ---- head location map ----------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    sets = [("intersection", inter, "o", "ToM C1∩C2 intersection"),
            ("code_abs", set(cond_sig.get("code_abs", {})), "X", "code abstract sig"),
            ("code_ctl", set(cond_sig.get("code_ctl", {})), "s", "code control sig")]
    for name, hs, marker, label in sets:
        if not hs:
            continue
        xs = [l for l, _ in hs]
        ys = [h for _, h in hs]
        ax.scatter(xs, ys, s=110, marker=marker, facecolors="none",
                   edgecolors=SET_COLORS[name], linewidths=2.2,
                   label=f"{label} [{len(hs)}]")
    ax.set_xlim(-1, n_layers)
    ax.set_ylim(-1, n_heads)
    ax.set_xlabel("layer")
    ax.set_ylabel("head index")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_title("Where the heads are: divergence band (L28–36) vs writer band (L42–46)")
    fig.savefig(OUT / "head_map.png", dpi=150)
    plt.close(fig)

    # ---- H2-analog decomposition (auto-skips until code_ac lands) --------
    decomp = None
    if all(k in fps for k in ("code_ac", "code_abs", "code_ctl")):
        target = fps["code_ac"].flatten()
        comp = (fps["code_abs"] + fps["code_ctl"]).flatten()
        r_sum = np.corrcoef(target, comp)[0, 1]
        r_abs = np.corrcoef(target, fps["code_abs"].flatten())[0, 1]
        r_ctl = np.corrcoef(target, fps["code_ctl"].flatten())[0, 1]
        decomp = {"r_sum": r_sum, "r_abs_only": r_abs, "r_ctl_only": r_ctl}
        fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
        ax.scatter(clamped(comp), clamped(target), s=12, alpha=0.5,
                   color="#0072B2", edgecolors="none")
        lim = max(clamped(comp).max(), clamped(target).max()) * 1.05
        ax.plot([0, lim], [0, lim], color="#888", linewidth=1, linestyle="--")
        ax.set_xlabel("code_abstract + code_control (clamped)")
        ax.set_ylabel("code_answer_changes (clamped)")
        ax.set_title(f"H2 analog: r = {r_sum:.3f} "
                     f"(abs only {r_abs:.3f}, ctl only {r_ctl:.3f})")
        fig.savefig(OUT / "h2_decomp_code.png", dpi=150)
        plt.close(fig)

    # ---- stats fragment for the qmd ---------------------------------------
    lines = ["### Cell inventory\n",
             "| condition | cells | seeds | n per cell | sig heads (per cell) |",
             "|---|---|---|---|---|"]
    for key in present:
        cells = data[key]
        sig_counts = [len(sig_heads_from_csv(c["csv"])) for c in cells.values()]
        lines.append(
            f"| {LABELS[key]} | {len(cells)} | "
            f"{','.join(sorted({c['seed'] for c in cells.values()}))} | "
            f"{','.join(str(c['n']) for c in cells.values())} | "
            f"{','.join(map(str, sig_counts))} |")

    lines += ["\n### Headline correlations (fingerprints over 1920 heads)\n",
              "| pair | r | R² | permutation p (10k label shuffles) |",
              "|---|---|---|---|"]
    for a, b, r, r2, p in headline:
        lines.append(f"| {LABELS[a]} ↔ {LABELS[b]} | {r:.3f} | {r2:.3f} | "
                     f"{'<1e-4' if p < 1e-4 + 1e-9 else f'{p:.4f}'} |")

    lines += ["\n### Direction consistency within code conditions\n",
              "| condition | r(direction 1, direction 2) |", "|---|---|"]
    for key in code_keys:
        cells = list(data[key].values())
        if len(cells) == 2:
            r = np.corrcoef(cells[0]["scores"].flatten(),
                            cells[1]["scores"].flatten())[0, 1]
            lines.append(f"| {LABELS[key]} | {r:.3f} |")
        else:
            lines.append(f"| {LABELS[key]} | only {len(cells)} cell(s) available |")

    if decomp:
        lines += ["\n### H2-analog decomposition\n",
                  f"r(code_ac, code_abs + code_ctl) = **{decomp['r_sum']:.3f}** "
                  f"vs abs-only {decomp['r_abs_only']:.3f}, "
                  f"ctl-only {decomp['r_ctl_only']:.3f}"]
        if all(k in fps for k in ("C3", "C1", "C5")):
            r_tom = np.corrcoef(fps["C3"].flatten(),
                                (fps["C1"] + fps["C5"]).flatten())[0, 1]
            lines += [f"— ToM benchmark on the same sources: "
                      f"r(C3, C1 + C5) = {r_tom:.3f}."]
        lines += ["\n![](code_domain/h2_decomp_code.png){width=65%}"]
    else:
        lines += ["\n### H2-analog decomposition\n",
                  "_code_answer_changes cells not yet available — rerun this "
                  "script after tonight's jobs land._"]

    (OUT / "generated_stats.md").write_text("\n".join(lines) + "\n")
    (OUT / "stats.json").write_text(json.dumps(
        {"corr_matrix": {f"{a}|{b}": float(C[i, j])
                         for i, a in enumerate(present) for j, b in enumerate(present)},
         "headline": [{"a": a, "b": b, "r": float(r), "p": p}
                      for a, b, r, _, p in headline],
         "decomp": decomp}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
