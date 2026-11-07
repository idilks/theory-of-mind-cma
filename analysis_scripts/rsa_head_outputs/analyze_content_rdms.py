"""RSA on the content-separable ToM capture: veracity vs content model RDMs.

Sept 2026, Sept 3 meeting (Taylor): the neuroscience-proxy result is "one
model RDM for true-vs-false belief correlates with the belief heads, another
for the actual content of the belief does not" -- the RSA analog of Bio et
al.'s TPJ decoding profile. The code-domain version of this table exists
(rdm-only.qmd Exp 2: divergence sets load on object divergence, retrieval
sets on answer identity); this is the same table in the belief frame.

Neural RDM per head set: unweighted mean over member heads of the cosine RDM
of that head's final-token output (the rdm-only.qmd convention).

Model RDMs (0 = same, 1 = different):
  veracity   false vs true belief
  believed   which location the agent believes  (Bio et al. "content")
  world      where the object actually is
  scenario   same template + pair + replicate draw (same non-contrastive
             substitutions) -- the confound the old capture could not break
  template   food_truck / hair_styling / library_book

Statistics: Mantel Spearman r with a stimulus-permutation p; rank-residualized
partials content|veracity, veracity|content, believed|scenario; a size-matched
random-head band (n_random never-selected draws); and, when the capture has
`replicate`, a split-half noise ceiling (Spearman-Brown, as in
padtwins_followups.run_ceilings).

Runs unchanged on activations_full (48/template) as a smoke test -- there the
`believed` model is degenerate (each location in one scenario), which is the
reason the content capture exists.

Outputs: results/rsa_head_outputs/aggregation_content/{mantel_content.csv,
random_band.csv, ceilings.csv, baseline_accuracy.csv} and
results_for_pi/aggregation/content_rdm_bars.png

--within_template (Nov 2026): the pooled Mantel is dominated by template
identity (r ~ 0.81 for every head set, random included; 67% of stimulus pairs
are cross-template), and the believed/world models inherit template structure
(r = 0.34) because the location vocabulary is per template. This mode repeats
the Mantel inside each template's 120 x 120 sub-RDM (7,140 same-template
pairs, label permutations within the template), averages r over templates,
and rebuilds the random-head band the same way. As a check it also reports
the pooled RDM restricted to same-template pairs (`*_samepairs`). Extra
outputs: mantel_content_within.csv (long: head_set x template),
mantel_content_within_summary.csv, random_band_within.csv, and a second
panel in the bar figure. It does not remove the answer-position confound
(within a template the false-belief answer is still the first mention).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dims as _dims  # noqa: E402
from analyze_code_ledger_rdms import mantel, partial_spearman, upper_tri  # noqa: E402
from mvpa_veracity import build_sets, load_capture, valid_rows  # noqa: E402
from padtwins_followups import spearman_brown  # noqa: E402
from rsa_utils import cosine_rdm  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations_content"
OUT_DIR = ROOT / "results/rsa_head_outputs/aggregation_content"
FIG_PATH = ROOT / "results_for_pi/aggregation/content_rdm_bars.png"
TEMPLATES = ("food_truck", "hair_styling", "library_book")
MODEL_ORDER = ("veracity", "believed", "world", "scenario", "template")


def categorical_rdm(labels) -> np.ndarray:
    v = np.asarray(labels)
    return (v[:, None] != v[None, :]).astype(np.float32)


def model_rdms(stim: pd.DataFrame) -> dict[str, np.ndarray]:
    ver = stim["belief_veracity"].astype(str).str.lower()
    rep = stim["replicate"] if "replicate" in stim else 0
    scen = (stim["template"].astype(str) + "|" + stim["pair_idx"].astype(str)
            + "|" + pd.Series(rep, index=stim.index).astype(str))
    return {
        "veracity": categorical_rdm(ver),
        "believed": categorical_rdm(stim["believed"]),
        "world": categorical_rdm(stim["loc_b"]),
        "scenario": categorical_rdm(scen),
        "template": categorical_rdm(stim["template"]),
    }


def set_rdm(z: np.ndarray, heads, d) -> np.ndarray:
    acc = np.zeros((z.shape[0], z.shape[0]), dtype=np.float64)
    for h in heads:
        l, hh = d.unflat(h)
        acc += cosine_rdm(z[:, l, hh, :])
    return acc / len(heads)


def ceiling_for(z, stim, heads, d) -> float | None:
    """Split-half over replicate draws at the (template, pair, veracity) cell."""
    if "replicate" not in stim or stim["replicate"].nunique() < 2:
        return None
    reps = sorted(stim["replicate"].unique())
    cell = list(zip(stim["template"], stim["pair_idx"],
                    stim["belief_veracity"].astype(str)))
    ucells = sorted(set(cell))
    sel = {}
    for r in reps:
        by = {c: i for i, (c, rr) in enumerate(zip(cell, stim["replicate"])) if rr == r}
        assert len(by) == len(ucells), f"replicate {r} missing cells"
        sel[r] = [by[c] for c in ucells]
    rdms = [set_rdm(z[sel[r]], heads, d) for r in reps]
    rels = [float(spearmanr(upper_tri(rdms[i]), upper_tri(rdms[j])).statistic)
            for i in range(len(reps)) for j in range(i + 1, len(reps))]
    rel1 = float(np.mean(rels))
    return float(np.sqrt(max(spearman_brown(rel1, len(reps)), 0.0)))


WITHIN_MODELS = ("veracity", "believed", "world", "scenario")


def template_masks(stim: pd.DataFrame) -> dict[str, np.ndarray]:
    return {t: (stim["template"].astype(str) == t).to_numpy() for t in TEMPLATES
            if (stim["template"].astype(str) == t).any()}


def same_template_pair_mask(stim: pd.DataFrame) -> np.ndarray:
    """Upper-triangle mask of stimulus pairs drawn from the same template."""
    t = stim["template"].astype(str).to_numpy()
    return upper_tri((t[:, None] == t[None, :]).astype(np.float32)) > 0


def within_template_rows(nr: np.ndarray, stim: pd.DataFrame, masks: dict,
                         n_perm: int, rng, name: str, k: int) -> list[dict]:
    """Mantel of one head-set RDM against the models inside each template.

    `nr` is the full set RDM; each template's block is indexed out of it so
    the head-set RDM is built once. Model RDMs are rebuilt on the template's
    own stimuli, so `believed` / `world` carry no template information here.
    """
    rows = []
    for t, m in masks.items():
        idx = np.flatnonzero(m)
        sub = nr[np.ix_(idx, idx)]
        sub_stim = stim.loc[m].reset_index(drop=True)
        models = model_rdms(sub_stim)
        flat = upper_tri(sub)
        fm = {mm: upper_tri(models[mm]) for mm in WITHIN_MODELS}
        row = dict(head_set=name, template=t, n_heads=k, n_stim=len(idx))
        for mm in WITHIN_MODELS:
            r, p = mantel(sub, models[mm], n_perm, rng)
            row[f"r_{mm}"], row[f"p_{mm}"] = r, p
        row["believed|veracity"] = partial_spearman(flat, fm["believed"], fm["veracity"])
        row["believed|world"] = partial_spearman(flat, fm["believed"], fm["world"])
        row["world|believed"] = partial_spearman(flat, fm["world"], fm["believed"])
        rows.append(row)
    return rows


def within_template_random_band(z, stim, masks: dict, sets: dict, d, n_random: int,
                                rng) -> pd.DataFrame:
    """Size-matched random-head band for the template-averaged Mantel r."""
    claimed = set().union(*[set(v) for v in sets.values()])
    pool = np.array([h for h in range(d.nh) if h not in claimed])
    flat_models = {}
    for t, m in masks.items():
        sub_stim = stim.loc[m].reset_index(drop=True)
        models = model_rdms(sub_stim)
        flat_models[t] = {mm: upper_tri(models[mm]) for mm in WITHIN_MODELS}
    rows = []
    for k in sorted({len(v) for v in sets.values()}):
        draws = []
        for _ in range(n_random):
            heads = rng.choice(pool, k, replace=False).tolist()
            nr = set_rdm(z, heads, d)
            per_t = []
            for t, m in masks.items():
                idx = np.flatnonzero(m)
                flat = upper_tri(nr[np.ix_(idx, idx)])
                per_t.append([float(spearmanr(flat, flat_models[t][mm]).statistic)
                              for mm in WITHIN_MODELS])
            draws.append(np.mean(per_t, axis=0))
        draws = np.array(draws)
        rows.append(dict(k=k, n_draws=n_random,
                         **{f"rand_mean_{mm}": draws[:, i].mean() for i, mm in enumerate(WITHIN_MODELS)},
                         **{f"rand_p95_{mm}": np.percentile(draws[:, i], 95)
                            for i, mm in enumerate(WITHIN_MODELS)}))
        print(f"  within-template random k={k:3d}: veracity {draws[:, 0].mean():+.3f} "
              f"(p95 {np.percentile(draws[:, 0], 95):+.3f})  believed {draws[:, 1].mean():+.3f} "
              f"(p95 {np.percentile(draws[:, 1], 95):+.3f})")
    return pd.DataFrame(rows)


def bar_panel(ax, summ: pd.DataFrame, order: list[str], title: str, ceilings: bool):
    s = summ.set_index("head_set")
    x = np.arange(len(order)); w = 0.26
    for i, (m, col) in enumerate((("veracity", "#c0392b"), ("believed", "#2980b9"), ("world", "#7f8c8d"))):
        ax.bar(x + (i - 1) * w, [s.loc[o, f"r_{m}"] for o in order], w, color=col, label=f"{m} model")
        ax.scatter(x + (i - 1) * w, [s.loc[o, f"rand_p95_{m}"] for o in order], marker="_",
                   color="k", s=120, zorder=3, label="random-head p95" if i == 0 else None)
    if ceilings and "ceiling" in s:
        ceil = [s.loc[o, "ceiling"] for o in order]
        if all(c is not None and not np.isnan(c) for c in ceil):
            ax.scatter(x, ceil, marker="v", color="k", s=30, zorder=3, label="noise ceiling")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("Mantel Spearman r"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title(title); ax.legend(fontsize=8, frameon=False)


def baseline_accuracy(act_dir: Path) -> pd.DataFrame | None:
    rows = []
    for t in TEMPLATES:
        p = torch.load(act_dir / f"{t}.pt", map_location="cpu", weights_only=False)
        if "generations" not in p:
            return None
        ok = valid_rows(pd.DataFrame(p["stim"]))
        for s, g, keep in zip(p["stim"], p["generations"], ok):
            if not keep:
                continue
            rows.append(dict(template=t, belief_veracity=str(s["belief_veracity"]).lower(),
                             believed=s["answer_state"],
                             correct=s["answer_state"].lower() in g.lower()))
    df = pd.DataFrame(rows)
    return df.groupby(["template", "belief_veracity"])["correct"].mean().reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act_dir", type=Path, default=ACT_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--fig", type=Path, default=FIG_PATH)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--top_c1", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--within_template", action="store_true",
                    help="also run the Mantel inside each template (see module doc)")
    args = ap.parse_args()

    d = _dims.for_model_type("Qwen2.5-14B-Instruct")
    z, stim = load_capture(args.act_dir, TEMPLATES)
    print(f"capture {args.act_dir.name}: {z.shape[0]} stimuli; believed vocabulary "
          f"{stim['believed'].nunique()} names; "
          f"{'replicates present' if 'replicate' in stim else 'no replicate column'}")

    acc = baseline_accuracy(args.act_dir)
    if acc is not None:
        print("baseline accuracy (believed location appears in generation):")
        print(acc.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    models = model_rdms(stim)
    flat_models = {k: upper_tri(v) for k, v in models.items()}
    print("model-RDM inter-correlations (Spearman, upper triangles):")
    keys = list(MODEL_ORDER)
    inter = pd.DataFrame(
        [[float(spearmanr(flat_models[a], flat_models[b]).statistic) for b in keys] for a in keys],
        index=keys, columns=keys)
    print(inter.to_string(float_format=lambda v: f"{v:+.2f}"))

    sets = build_sets(d, args.top_c1)
    rng = np.random.default_rng(args.seed)
    # Separate generator so the within-template mode leaves the pooled
    # permutations and random band bit-identical to a run without the flag.
    rng_within = np.random.default_rng(args.seed + 1)
    masks = template_masks(stim) if args.within_template else {}
    same_pairs = same_template_pair_mask(stim) if args.within_template else None
    rows, within_rows = [], []
    for name, heads in sets.items():
        nr = set_rdm(z, heads, d)
        flat = upper_tri(nr)
        row = dict(head_set=name, n_heads=len(heads))
        if args.within_template:
            within_rows.extend(within_template_rows(nr, stim, masks, args.n_perm,
                                                    rng_within, name, len(heads)))
            for mm in WITHIN_MODELS:
                row[f"r_{mm}_samepairs"] = float(
                    spearmanr(flat[same_pairs], flat_models[mm][same_pairs]).statistic)
        for m in MODEL_ORDER:
            r, p = mantel(nr, models[m], args.n_perm, rng)
            row[f"r_{m}"], row[f"p_{m}"] = r, p
        row["believed|veracity"] = partial_spearman(flat, flat_models["believed"], flat_models["veracity"])
        row["veracity|believed"] = partial_spearman(flat, flat_models["veracity"], flat_models["believed"])
        row["believed|scenario"] = partial_spearman(flat, flat_models["believed"], flat_models["scenario"])
        row["ceiling"] = ceiling_for(z, stim, heads, d)
        rows.append(row)
        print(f"  {name:14s} k={len(heads):3d}  veracity {row['r_veracity']:+.3f} (p={row['p_veracity']:.3f})"
              f"  believed {row['r_believed']:+.3f} (p={row['p_believed']:.3f})"
              f"  world {row['r_world']:+.3f}  scenario {row['r_scenario']:+.3f}"
              f"  believed|scen {row['believed|scenario']:+.3f}"
              f"  ceiling {row['ceiling'] if row['ceiling'] is None else round(row['ceiling'], 3)}")
    summ = pd.DataFrame(rows)

    claimed = set().union(*[set(v) for v in sets.values()])
    pool = np.array([h for h in range(d.nh) if h not in claimed])
    band_rows = []
    for k in sorted({len(v) for v in sets.values()}):
        draws = []
        for _ in range(args.n_random):
            heads = rng.choice(pool, k, replace=False).tolist()
            flat = upper_tri(set_rdm(z, heads, d))
            draws.append([float(spearmanr(flat, flat_models[m]).statistic) for m in MODEL_ORDER])
        draws = np.array(draws)
        band_rows.append(dict(k=k, n_draws=args.n_random,
                              **{f"rand_mean_{m}": draws[:, i].mean() for i, m in enumerate(MODEL_ORDER)},
                              **{f"rand_p95_{m}": np.percentile(draws[:, i], 95) for i, m in enumerate(MODEL_ORDER)}))
        print(f"  random k={k:3d}: veracity {draws[:, 0].mean():+.3f} (p95 {np.percentile(draws[:, 0], 95):+.3f})"
              f"  believed {draws[:, 1].mean():+.3f} (p95 {np.percentile(draws[:, 1], 95):+.3f})")
    band = pd.DataFrame(band_rows)
    summ = summ.merge(band, left_on="n_heads", right_on="k", how="left").drop(columns="k")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out_dir / "mantel_content.csv", index=False)
    band.to_csv(args.out_dir / "random_band.csv", index=False)
    inter.to_csv(args.out_dir / "model_rdm_intercorrelations.csv")
    if acc is not None:
        acc.to_csv(args.out_dir / "baseline_accuracy.csv", index=False)

    within_summ = None
    if args.within_template:
        within = pd.DataFrame(within_rows)
        within.to_csv(args.out_dir / "mantel_content_within.csv", index=False)
        num = [c for c in within.columns if c.startswith(("r_", "p_")) or "|" in c]
        within_summ = (within.groupby("head_set", sort=False)[num].mean().reset_index()
                       .merge(within[["head_set", "n_heads"]].drop_duplicates(), on="head_set"))
        band_w = within_template_random_band(z, stim, masks, sets, d, args.n_random, rng_within)
        band_w.to_csv(args.out_dir / "random_band_within.csv", index=False)
        within_summ = within_summ.merge(band_w, left_on="n_heads", right_on="k", how="left").drop(columns="k")
        same_cols = [c for c in summ.columns if c.endswith("_samepairs")]
        within_summ = within_summ.merge(summ[["head_set", *same_cols]], on="head_set")
        within_summ.to_csv(args.out_dir / "mantel_content_within_summary.csv", index=False)

    # figure: per set, r for veracity / believed / world with the random band
    order = [s for s in ("div_unique", "top10_C1", "dual_sig", "div_shared", "photo_C2",
                         "c5_retrieval", "random32") if s in set(summ.head_set)]
    n_panels = 2 if within_summ is not None else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 4 * n_panels), squeeze=False)
    bar_panel(axes[0, 0], summ, order,
              f"ToM content RDMs, pooled ({args.act_dir.name}, n={z.shape[0]})", ceilings=True)
    if within_summ is not None:
        bar_panel(axes[1, 0], within_summ, order,
                  f"within template, mean over {len(masks)} templates "
                  f"({len(masks)} x {int(np.mean([m.sum() for m in masks.values()]))} stimuli)",
                  ceilings=False)
    args.fig.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.fig, dpi=150)

    print("\n=== summary (pooled) ===")
    print(summ[["head_set", "n_heads", "r_veracity", "r_believed", "r_world", "r_scenario",
                "believed|scenario", "veracity|believed", "ceiling",
                "rand_p95_veracity", "rand_p95_believed"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    if within_summ is not None:
        print("\n=== summary (within template, mean r over templates) ===")
        print(within_summ[["head_set", "n_heads", "r_veracity", "r_believed", "r_world",
                           "believed|veracity", "believed|world", "world|believed",
                           "rand_p95_veracity", "rand_p95_believed", "rand_p95_world",
                           "r_veracity_samepairs", "r_believed_samepairs"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        print("\nper-template veracity / believed r:")
        print(within.pivot(index="head_set", columns="template", values=["r_veracity", "r_believed"])
              .to_string(float_format=lambda v: f"{v:+.3f}"))
    print(f"\nwrote {args.out_dir} and {args.fig}")


if __name__ == "__main__":
    main()
