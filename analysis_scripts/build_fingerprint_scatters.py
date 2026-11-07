"""Scatterplot pack: pairwise fingerprint correlations (July 2026 plan, step 1).

One dot per head (n=1920), clamped fingerprints on both axes, colored by
permutation-test significance membership:
  gray = significant in neither condition, blue = x-condition only,
  orange = y-condition only, green = both.

Pairs (meeting to-do, July 23):
  C1 x C5            — H1 disjointness (the Taylor ask: orthogonal cloud,
                        a few dual-significant heads)
  C1 x C2            — belief/photo frame-generality (published r = 0.760)
  C5 x C6            — retrieval across frames
  C3 x (C2+C6)       — cross-frame competitor sum
  code_abs x C1      — code/belief divergence machinery (published 0.867)
  code_ctl x C5      — code/ToM retrieval (published 0.706)
  code_abs x code_ctl— within-code dissociation (published 0.067)
  
  
Outputs -> results/summary/fingerprint_scatters/
  figures/scatter_*.png, stats_scatter_pairs.csv, stats_fragment.md
"""

import csv as csvmod
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as sps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis_scripts"))
from condition_registry import resolve_best_source  # noqa: E402
from h2_additivity_robustness import (  # noqa: E402  (shares style + loaders)
    load_tom_fp, build_code_fp, build_gemma_fp, N_LAYERS, N_HEADS, N_PERM,
    GEMMA_SHAPE, GEMMA_ROOT, BLUE, ORANGE, GREEN, GRAY, RESULTS_ROOT,
    CROSS_RULES, WITHIN_RULES)

OUT = REPO / "results/summary/fingerprint_scatters"
FIG = OUT / "figures"
HEADSETS = REPO / "results/aggregated/code_head_sets"
RNG = np.random.default_rng(0)

GEMMA_N_LAYERS, GEMMA_N_HEADS = GEMMA_SHAPE

TOM_CONDS = {
    "C1": ("abstract_belief", CROSS_RULES),
    "C2": ("abstract_photo", CROSS_RULES),
    "C3": ("answer_changes_belief", CROSS_RULES),
    "C5": ("answer_changes_location_only", WITHIN_RULES),
    "C6": ("answer_changes_location_only_photo", WITHIN_RULES),
}
LABELS = {
    "C1": "C1 belief (abstract)", "C2": "C2 photo (abstract)",
    "C3": "C3 belief+answer", "C5": "C5 answer (ToM frame)",
    "C6": "C6 answer (photo frame)", "SUM_C2_C6": "C2 + C6 (photo-frame sum)",
    "code_abs": "code abstract", "code_ctl": "code control",
    "gC1": "Gemma C1 belief (abstract)", "gC2": "Gemma C2 photo (abstract)",
    "gC3": "Gemma C3 belief+answer", "gC5": "Gemma C5 answer (ToM frame)",
    "gSUM_C1_C5": "Gemma C1 + C5 (own-frame sum)",
}

GEMMA_CONDS = {
    "gC1": ("abstract_belief", CROSS_RULES, "gemma_c1_head_list.csv"),
    "gC2": ("abstract_photo", CROSS_RULES, "gemma_c2_head_list.csv"),
    "gC3": ("answer_changes_belief", CROSS_RULES, "gemma_c3_head_list.csv"),
    "gC5": ("answer_changes_location_only", WITHIN_RULES, "gemma_c5_head_list.csv"),
}


def tom_sig_set(folder: str, rules: set) -> set:
    """Heads significant in >= half of best-source cells."""
    best = resolve_best_source(RESULTS_ROOT, [folder], rules)
    counts, n_cells = {}, 0
    for (tmpl, rule), e in sorted(best.items()):
        p = Path(e["csv"])
        if not p.exists():
            continue
        n_cells += 1
        with open(p) as f:
            for row in csvmod.DictReader(f):
                if row["significant"].strip().lower() == "true":
                    key = (int(row["layer"]), int(row["head"]))
                    counts[key] = counts.get(key, 0) + 1
    need = int(np.ceil(n_cells / 2))
    return {k for k, v in counts.items() if v >= need}


def published_set(csv_name: str) -> set:
    heads = set()
    with open(HEADSETS / csv_name) as f:
        for row in csvmod.DictReader(f):
            heads.add((int(row["layer"]), int(row["head"])))
    return heads


def flat_idx(heads: set, n_layers: int = N_LAYERS,
             n_heads: int = N_HEADS) -> np.ndarray:
    mask = np.zeros(n_layers * n_heads, dtype=bool)
    for l, h in heads:
        mask[l * n_heads + h] = True
    return mask


def perm_p(x, y):
    obs = np.corrcoef(x, y)[0, 1]
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = np.corrcoef(x, RNG.permutation(y))[0, 1]
    # two-sided on |r|: pairs here include near-zero expected correlations
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1))


def scatter_pair(xk, yk, fps, sig, path):
    x, y = fps[xk]["clamped"], fps[yk]["clamped"]
    mx, my = sig[xk], sig[yk]
    both, xo, yo = mx & my, mx & ~my, ~mx & my
    neither = ~(mx | my)

    r, p = perm_p(x, y)
    rho = sps.spearmanr(x, y).statistic

    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.scatter(x[neither], y[neither], s=6, color=GRAY, alpha=0.25,
               linewidths=0, label=f"neither (n={neither.sum()})")
    ax.scatter(x[xo], y[xo], s=22, color=BLUE, alpha=0.9, linewidths=0,
               label=f"{xk} only (n={xo.sum()})")
    ax.scatter(x[yo], y[yo], s=22, color=ORANGE, alpha=0.9, linewidths=0,
               label=f"{yk} only (n={yo.sum()})")
    ax.scatter(x[both], y[both], s=34, color=GREEN, alpha=0.95,
               edgecolors="white", linewidths=0.8,
               label=f"both (n={both.sum()})")
    ax.set_xlabel(f"{LABELS[xk]} (clamped)")
    ax.set_ylabel(f"{LABELS[yk]} (clamped)")
    # headroom so extreme points don't hide behind the corner annotation/legend
    ax.set_ylim(top=y.max() * 1.22)
    ax.set_xlim(right=x.max() * 1.08)
    p_str = "p = 2×10⁻⁴" if p <= 2.5e-4 else f"p = {p:.4f}"
    ax.text(0.97, 0.97, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\n{p_str}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    ax.legend(fontsize=7.5, frameon=False, loc="upper left",
              handletextpad=0.15, borderaxespad=0.2)
    fig.savefig(path)
    plt.close(fig)

    inter, union = int(both.sum()), int((mx | my).sum())
    return {
        "pair": f"{xk}~{yk}",
        "pearson_r_clamped": r,
        "spearman_rho_clamped": float(rho),
        "pearson_r_raw": float(np.corrcoef(fps[xk]["raw"], fps[yk]["raw"])[0, 1]),
        "spearman_rho_raw": float(sps.spearmanr(fps[xk]["raw"], fps[yk]["raw"]).statistic),
        "perm_p_two_sided": p,
        "n_sig_x": int(mx.sum()), "n_sig_y": int(my.sum()),
        "n_both": inter, "jaccard": inter / union if union else np.nan,
    }


def noise_scale_diagnostic(fps, sig, cell_sources):
    """Why is Spearman lower than Pearson, and lower still under raw?

    Clamping is per-cell clip(scores, 0) then mean, so a head's across-cell
    noise *scale* leaks into its clamped score: a head bouncing +-2 averages
    to ~+1 clamped but ~0 raw. Noise scale is a property of the head and is
    shared across conditions, so clamped fingerprints agree with each other
    even among heads carrying no signal. Pearson barely notices (that
    agreement lives at tiny magnitudes); Spearman does, because ranks ignore
    magnitude.

    Returns rho(fingerprint, across-cell SD) over non-significant heads.
    """
    rows = []
    for key, (root, folder, rules, shape) in cell_sources.items():
        best = resolve_best_source(root, [folder], rules)
        cells = np.stack([
            torch.load(e["scores_pt"], map_location="cpu",
                       weights_only=False).float().numpy().ravel()
            for _, e in sorted(best.items())])
        sd = cells.std(axis=0)
        null = ~sig[key]
        rows.append({
            "condition": key, "n_null_heads": int(null.sum()),
            "rho_clamped_vs_cellSD": float(
                sps.spearmanr(fps[key]["clamped"][null], sd[null]).statistic),
            "rho_raw_vs_cellSD": float(
                sps.spearmanr(fps[key]["raw"][null], sd[null]).statistic)})
    return rows


def restricted_spearman(xk, yk, fps, sig):
    """Spearman with the noise sea removed, two ways.

    sig-union: heads significant in either condition. Valid as a diagnostic
      for pairs that SHARE machinery; for dissociated pairs the union is two
      near-disjoint clusters (high-in-x / high-in-y), so selection alone
      manufactures a negative correlation (Berkson). Flagged, not hidden.
    null-only: heads significant in neither — isolates how much agreement
      clamping manufactures where there is no signal to find.
    """
    keep, null = sig[xk] | sig[yk], ~(sig[xk] | sig[yk])
    out = {"pair": f"{xk}~{yk}", "n_sig_union": int(keep.sum()),
           "n_null": int(null.sum())}
    for key in ("clamped", "raw"):
        x, y = fps[xk][key], fps[yk][key]
        out[f"rho_full_{key}"] = float(sps.spearmanr(x, y).statistic)
        out[f"rho_sigunion_{key}"] = float(
            sps.spearmanr(x[keep], y[keep]).statistic)
        out[f"rho_nullonly_{key}"] = float(
            sps.spearmanr(x[null], y[null]).statistic)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    fps = {k: load_tom_fp(f"{k}_{folder}")
           for k, (folder, _) in TOM_CONDS.items()}
    fps["code_abs"], _ = build_code_fp("code_abstract", CROSS_RULES)
    fps["code_ctl"], _ = build_code_fp("code_control", WITHIN_RULES)
    fps["SUM_C2_C6"] = {k: fps["C2"][k] + fps["C6"][k] for k in ("raw", "clamped")}

    sig = {}
    for k, (folder, rules) in TOM_CONDS.items():
        sig[k] = flat_idx(tom_sig_set(folder, rules))
        print(f"sig[{k}]: {sig[k].sum()} heads (>=half of cells)")
    sig["code_abs"] = flat_idx(published_set("code_abstract_head_list.csv"))
    sig["code_ctl"] = flat_idx(published_set("code_control_head_list.csv"))
    sig["SUM_C2_C6"] = sig["C2"] | sig["C6"]
    print(f"sig[code_abs]: {sig['code_abs'].sum()}  sig[code_ctl]: {sig['code_ctl'].sum()}"
          f"  sig[C2 or C6]: {sig['SUM_C2_C6'].sum()}")

    # -- Gemma replication (46x32 = 1472 heads; within-Gemma pairs only,
    #    no head correspondence across architectures)
    gemma_prov = []
    for k, (folder, rules, csv_name) in GEMMA_CONDS.items():
        fps[k], prov = build_gemma_fp(folder, rules)
        gemma_prov.extend(prov)
        sig[k] = flat_idx(published_set(csv_name),
                          GEMMA_N_LAYERS, GEMMA_N_HEADS)
        print(f"sig[{k}]: {sig[k].sum()} heads ({len(prov)} cells)")
    fps["gSUM_C1_C5"] = {k: fps["gC1"][k] + fps["gC5"][k]
                         for k in ("raw", "clamped")}
    sig["gSUM_C1_C5"] = sig["gC1"] | sig["gC5"]

    PAIRS = [("C1", "C5"), ("C1", "C2"), ("C5", "C6"), ("C3", "SUM_C2_C6"),
             ("code_abs", "C1"), ("code_ctl", "C5"), ("code_abs", "code_ctl"),
             ("gC1", "gC5"), ("gC1", "gC2"), ("gC3", "gSUM_C1_C5"),
             ("gC3", "gC1"), ("gC3", "gC5")]

    rows = []
    for xk, yk in PAIRS:
        rows.append(scatter_pair(xk, yk, fps, sig,
                                 FIG / f"scatter_{xk}_vs_{yk}.png"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stats_scatter_pairs.csv", index=False)

    # -- why Spearman is low, and what it looks like without the noise sea
    cell_sources = {k: (RESULTS_ROOT, folder, rules, (N_LAYERS, N_HEADS))
                    for k, (folder, rules) in TOM_CONDS.items()}
    cell_sources.update({
        k: (GEMMA_ROOT, folder, rules, GEMMA_SHAPE)
        for k, (folder, rules, _) in GEMMA_CONDS.items()})
    df_sd = pd.DataFrame(noise_scale_diagnostic(fps, sig, cell_sources))
    df_sd.to_csv(OUT / "stats_noise_scale.csv", index=False)

    df_restr = pd.DataFrame([restricted_spearman(xk, yk, fps, sig)
                             for xk, yk in PAIRS])
    df_restr.to_csv(OUT / "stats_spearman_restricted.csv", index=False)

    frag = ["<!-- AUTO-GENERATED by analysis_scripts/build_fingerprint_scatters.py -->\n",
            "### Pairwise fingerprint statistics (all heads per pair)\n",
            df.round(3).to_markdown(index=False) + "\n",
            "\n### Clamping leaks per-head noise scale into the score\n",
            "Spearman of the fingerprint against the head's across-cell SD,"
            " over non-significant heads only.\n",
            df_sd.round(3).to_markdown(index=False) + "\n",
            "\n### Spearman with the noise sea removed\n",
            "`sigunion` = heads significant in either condition"
            " (**Berkson-biased for dissociated pairs — see the doc**);"
            " `nullonly` = heads significant in neither.\n",
            df_restr.round(3).to_markdown(index=False) + "\n"]
    (OUT / "stats_fragment.md").write_text("\n".join(frag))

    print("\n== sanity vs published ==")
    pub = {"C1~C2": 0.760, "code_abs~C1": 0.867, "code_ctl~C5": 0.706,
           "code_abs~code_ctl": 0.067}
    for _, row in df.iterrows():
        tag = f"  (published {pub[row['pair']]})" if row["pair"] in pub else ""
        print(f"{row['pair']:22s} r={row['pearson_r_clamped']:.3f} "
              f"ρ={row['spearman_rho_clamped']:.3f} "
              f"sig: {row['n_sig_x']}/{row['n_sig_y']} shared {row['n_both']}{tag}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
