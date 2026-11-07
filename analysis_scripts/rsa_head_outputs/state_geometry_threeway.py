"""Three-way state geometry: belief / photo / code.

Adds the false-photograph frame (C2-style stimuli: no agent, photo timing
flips the divergence) as a third state measurement per head, between the
belief frame and the code domain. Three payoffs:

  1. ABSTRACTION LADDER — pairwise corrected alignments ordered by
     representational distance: within-code benchmark >= belief<->photo
     (same narrative, different frame) >= belief<->code (different domain).
     The graded curve is H3 (frame generality) at the geometry level.
  2. ONE CORE OR PAIRWISE ACCIDENTS — partial-projection test: if a single
     shared core carries all pairwise sharing, then projecting the photo
     direction out of both belief and code directions should kill the
     belief<->code alignment (cos(u_b [perp] p, u_c [perp] p) ~ 0). If
     belief<->code sharing survives photo removal, the overlaps are
     pairwise-idiosyncratic.
  3. HIERARCHY / LOCALITY, finally evaluable — the two-vector bisector
     construction forced residues to be one axis (the SS/SG2 artifact). A
     third measurement fixes this WITHOUT any constructed core: project the
     CODE direction out of belief and photo; if the residuals still align
     (cos(u_b [perp] c, u_p [perp] c) > 0), there is a NARRATIVE layer shared by
     the two frames beyond the cross-domain core. The residual-decoding
     matrix then tests locality: u_b [perp] c should decode belief and photo
     (narrative content) but not code.

  DESIGN NOTE (why no "core axis" is ever constructed): a core defined as
  the normalized signed mean of the three directions forces the three
  residues to sum to zero, which constrains their pairwise cosines toward
  -0.5 — the same class of estimator artifact as the bisector. All tests
  here use only (a) pairwise cosines, (b) single-vector projections of one
  direction out of another, and (c) data-based decoding — none of which
  carry that constraint.

Inputs: belief capture (activations_full/), photo capture
(activations_photo/, from capture_head_outputs_local.py on the photo
stimulus set), code_position capture, optional code_divergence capture
(internal benchmark). Head sets as before, plus a photo-defined set
(fp_C2 top-26 minus fp_C6 top-13) when those fingerprints exist.

Outputs (results/rsa_head_outputs/state_geometry_threeway/):
  pairwise_alignment.csv, partial_projections.csv, decoding_matrix.csv,
  perhead_threeway.csv, figure state_geometry_threeway.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dims as _dims  # noqa: E402
import state_geometry_alignment as _sga  # noqa: E402
from state_geometry_alignment import (  # noqa: E402
    unit, directions, load_tom, load_code, head_sets,
    N_LAYERS, N_HEADS, D, NH, TOM_ACT, CODE_POS, CODE_DIV, FP_DIR, ROOT)
# NOTE: the dimension names above are bound at IMPORT time. configure() in the
# alignment module cannot reach them, so main() rebinds them explicitly after
# calling it. Without that rebind a non-Qwen run would silently use 48x40.

PHOTO_ACT = ROOT / "results/rsa_head_outputs/activations_full_photo"
OUT_DIR = ROOT / "results/rsa_head_outputs/state_geometry_threeway"
FIG_DIR = ROOT / "results/summary/aggregation"
FIG_TAG = _dims.tag("Qwen2.5-14B-Instruct")


def auc_of(scores, lab):
    from scipy.stats import rankdata
    r = rankdata(scores)
    n1 = lab.sum()
    return float((r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * (~lab).sum()))


def perp(u, m):
    """Project direction m out of direction u, per head; renormalize."""
    return unit(u - np.einsum("hd,hd->h", u, m)[:, None] * m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo_dir", type=Path, default=PHOTO_ACT)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--min_rel", type=float, default=0.1)
    ap.add_argument("--tom_dir", type=Path, default=None)
    ap.add_argument("--code_pos", type=Path, default=None)
    ap.add_argument("--code_div", type=Path, default=None)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--gemma_dir", type=Path, default=None)
    args = ap.parse_args()

    # Model-suffixed outputs (see dims.tag): the un-suffixed dir let a Gemma
    # run overwrite the Qwen one.
    global OUT_DIR, FIG_TAG
    FIG_TAG = _dims.tag(args.model_type)
    OUT_DIR = OUT_DIR.with_name(f"{OUT_DIR.name}_{FIG_TAG}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    global N_LAYERS, N_HEADS, D, NH, TOM_ACT, CODE_POS, CODE_DIV
    if args.tom_dir:  TOM_ACT  = args.tom_dir
    if args.code_pos: CODE_POS = args.code_pos
    if args.code_div: CODE_DIV = args.code_div
    _d = _sga.configure(args.model_type, args.gemma_dir)
    N_LAYERS, N_HEADS, D, NH = _sga.N_LAYERS, _sga.N_HEADS, _sga.D, _sga.NH
    print(f"model={args.model_type}  dims={_d}")
    print("loading captures...")
    z_b, lab_b = load_tom(TOM_ACT)
    z_p, lab_p = load_tom(args.photo_dir)   # photo stim also carry belief_veracity
    z_c, lab_c, _ = load_code(CODE_POS, "semantics", "alias")
    print(f"  belief {z_b.shape}, photo {z_p.shape}, code {z_c.shape}")

    print("directions + reliabilities...")
    d_b, r_b = directions(z_b, lab_b, True, args.n_splits, rng)
    d_p, r_p = directions(z_p, lab_p, True, args.n_splits, rng)
    d_c, r_c = directions(z_c, lab_c, True, args.n_splits, rng)
    u_b, u_p, u_c = unit(d_b), unit(d_p), unit(d_c)

    bench = None
    if CODE_DIV.exists():
        z_d, lab_d, _ = load_code(
            CODE_DIV, "order", "mutate_first",
            ROOT / "results/rsa_head_outputs/stimuli_code_divergence/"
                   "code_divergence.csv")
        d_d, r_d = directions(z_d, lab_d, True, args.n_splits, rng)
        bench = (np.einsum("hd,hd->h", u_c, unit(d_d)), r_d)

    sets = head_sets()
    # photo-defined set, when C2/C6 fingerprints exist
    c2 = FP_DIR / "C2_abstract_photo_clamped.npy"
    c6 = FP_DIR / "C6_answer_changes_location_only_photo_clamped.npy"
    # Only valid when the fingerprint arrays match THIS model's head grid.
    # FP_DIR holds the Qwen fingerprints (len 1920); applying them to Gemma
    # (NH=1472) injects out-of-range head indices. head_sets() already supplies
    # a model-appropriate photo_C2 via head_sets_source, so skip the override.
    if c2.exists() and c6.exists() and len(np.load(c2)) == NH:
        f2, f6 = np.load(c2), np.load(c6)
        sets["photo_C2"] = sorted(
            set(np.argsort(f2)[-26:].tolist())
            - set(np.argsort(f6)[-13:].tolist()))
    elif c2.exists():
        print(f"  (photo_C2: fingerprint grid {len(np.load(c2))} != NH {NH}; "
              "using the model-appropriate set from head_sets_source)")

    pairs = {"belief_photo": (u_b, u_p, r_b, r_p),
             "belief_code": (u_b, u_c, r_b, r_c),
             "photo_code": (u_p, u_c, r_p, r_c)}
    cos = {k: np.einsum("hd,hd->h", a, b) for k, (a, b, _, _) in pairs.items()}

    # ---- 1. abstraction ladder -------------------------------------------
    print("\n=== pairwise alignment (raw | corrected) per head set ===")
    rows = []
    for name, hs in sets.items():
        hs = np.array(hs)
        row = {"set": name, "n": len(hs)}
        for k, (a, b, ra, rb) in pairs.items():
            ok = (ra[hs] > args.min_rel) & (rb[hs] > args.min_rel)
            raw = float(cos[k][hs].mean())
            corr = float(np.nanmean(
                cos[k][hs][ok] / np.sqrt(ra[hs][ok] * rb[hs][ok]))) \
                if ok.any() else float("nan")
            null = np.empty(args.n_perm)
            ua = pairs[k][0]
            for i in range(args.n_perm):
                null[i] = np.einsum(
                    "hd,hd->h", ua[hs],
                    pairs[k][1][rng.permutation(NH)[:len(hs)]]).mean()
            row[f"{k}_raw"] = raw
            row[f"{k}_corr"] = corr
            row[f"{k}_p"] = float((null >= raw).mean())
        if bench is not None:
            row["internal_bench_raw"] = float(bench[0][hs].mean())
        rows.append(row)
        print(f"  {name:11s} b<->p {row['belief_photo_raw']:+.3f}|"
              f"{row['belief_photo_corr']:+.3f}  "
              f"b<->c {row['belief_code_raw']:+.3f}|"
              f"{row['belief_code_corr']:+.3f}  "
              f"p<->c {row['photo_code_raw']:+.3f}|"
              f"{row['photo_code_corr']:+.3f}"
              + (f"  bench {row.get('internal_bench_raw', float('nan')):+.3f}"
                 if bench is not None else ""))
    pd.DataFrame(rows).to_csv(OUT_DIR / "pairwise_alignment.csv", index=False)

    # ---- 2 & 3. partial projections --------------------------------------
    # one-core test: does b<->c survive removing p from both?
    # narrative-layer test: does b<->p survive removing c from both?
    print("\n=== partial projections (per set, raw cosines) ===")
    tests = {
        "b_c_given_p": (perp(u_b, u_p), perp(u_c, u_p), "belief_code"),
        "b_p_given_c": (perp(u_b, u_c), perp(u_p, u_c), "belief_photo"),
        "p_c_given_b": (perp(u_p, u_b), perp(u_c, u_b), "photo_code"),
    }
    prows = []
    for name, hs in sets.items():
        hs = np.array(hs)
        row = {"set": name}
        for tname, (a, b, full_key) in tests.items():
            part = float(np.einsum("hd,hd->h", a, b)[hs].mean())
            row[tname] = part
            row[f"{tname}_vs_full"] = part - float(cos[full_key][hs].mean())
        prows.append(row)
        print(f"  {name:11s} b<->c|p {row['b_c_given_p']:+.3f}  "
              f"b<->p|c {row['b_p_given_c']:+.3f}  "
              f"p<->c|b {row['p_c_given_b']:+.3f}")
    pd.DataFrame(prows).to_csv(OUT_DIR / "partial_projections.csv",
                               index=False)
    print("  one core -> b<->c|p ~ 0; narrative layer -> b<->p|c > 0.")

    # ---- decoding matrix (split-honest on the decoded domain) ------------
    print("\n=== zero-shot decoding matrix, div_unique (rows: axis source; "
          "cols: decoded data) ===")
    data = {"belief": (z_b, lab_b), "photo": (z_p, lab_p),
            "code": (z_c, lab_c)}
    axes = {"belief": u_b, "photo": u_p, "code": u_c,
            "belief_perp_code": perp(u_b, u_c),
            "photo_perp_code": perp(u_p, u_c)}
    hs = np.array(sets["div_unique"])
    mrows = []
    for aname, ax in axes.items():
        row = {"axis": aname}
        for dname, (z, lab) in data.items():
            # split-honest when axis source == decoded domain
            src = aname.split("_")[0]
            if src == dname:
                aucs = []
                pos_i, neg_i = np.flatnonzero(lab), np.flatnonzero(~lab)
                for si in range(args.n_splits):
                    r2 = np.random.default_rng(200 + si)
                    pa, na = r2.permutation(pos_i), r2.permutation(neg_i)
                    tr = np.concatenate([pa[:len(pa)//2], na[:len(na)//2]])
                    te = np.concatenate([pa[len(pa)//2:], na[len(na)//2:]])
                    ua = unit(z[tr][lab[tr]].mean(0)
                              - z[tr][~lab[tr]].mean(0))
                    if aname.endswith("perp_code"):
                        ua = perp(ua, u_c)
                    aucs.append(np.mean(
                        [auc_of(z[te][:, h] @ ua[h], lab[te]) for h in hs]))
                row[dname] = float(np.mean(aucs))
            else:
                row[dname] = float(np.mean(
                    [auc_of(z[:, h] @ ax[h], lab) for h in hs]))
        mrows.append(row)
        print(f"  {row['axis']:18s} belief={row['belief']:.3f} "
              f"photo={row['photo']:.3f} code={row['code']:.3f}")
    pd.DataFrame(mrows).to_csv(OUT_DIR / "decoding_matrix.csv", index=False)
    print("  locality: belief_perp_code should decode belief & photo "
          "(narrative layer) but not code.")

    # ---- per-head + figure -----------------------------------------------
    ph = pd.DataFrame({
        "layer": np.repeat(np.arange(N_LAYERS), N_HEADS),
        "head": np.tile(np.arange(N_HEADS), N_LAYERS),
        "cos_bp": cos["belief_photo"], "cos_bc": cos["belief_code"],
        "cos_pc": cos["photo_code"],
        "rel_belief": r_b, "rel_photo": r_p, "rel_code": r_c,
    })
    ph["set"] = "other"
    for name, hsx in sets.items():
        ph.loc[hsx, "set"] = name
    ph.to_csv(OUT_DIR / "perhead_threeway.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY = "#0072B2", "#E69F00", "#B0B0B0"
    hs = np.array(sets["div_unique"])
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["within-code\n(benchmark)", "belief<->photo\n(cross-frame)",
              "belief<->code\n(cross-domain)", "photo<->code\n(cross-domain)"]
    vals = [float(bench[0][hs].mean()) if bench is not None else np.nan,
            float(cos["belief_photo"][hs].mean()),
            float(cos["belief_code"][hs].mean()),
            float(cos["photo_code"][hs].mean())]
    ax.bar(range(len(vals)), vals,
           color=[GRAY, ORANGE, BLUE, BLUE], width=0.62)
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("mean per-head cosine (div_unique, raw)")
    ax.set_title("State-code sharing vs representational distance")
    ax.axhline(0, color="#DDDDDD", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"state_geometry_threeway_{FIG_TAG}.png", dpi=160)
    print(f"\nwrote -> {OUT_DIR}, figure -> {FIG_DIR}")


if __name__ == "__main__":
    main()
