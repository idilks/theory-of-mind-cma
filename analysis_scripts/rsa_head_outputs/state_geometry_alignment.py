"""Cross-domain state-geometry alignment: one variable or two cohabiting codes?

The program has established that div_unique heads carry a transferable state
in BOTH domains (copy/alias mechanism in code, belief state in ToM), each
certified within-domain. What it has NOT established is whether that is one
shared representational variable or two domain-specific codes cohabiting the
same heads. This analysis is the cheapest direct test: no patching, pure
geometry on captured activations.

Per head, compute the state DIRECTION in each domain as a mean difference
over that head's z outputs at the final token, with a single sign convention
(non-divergent minus divergent — the conditions where the default source
governs minus those where the preserved representation does):

  tom       mean(z | true_belief)  - mean(z | false_belief)
  code_mech mean(z | alias)        - mean(z | copy)         [code_position]
  code_div  mean(z | mutate_first) - mean(z | copy_first)   [code_divergence]

Then:
  1. Per-head cosine alignment between tom and each code direction.
     Null: shuffled head pairing (head i's tom direction vs head j's code
     direction), 1000 permutations of the pairing within the 1920 grid.
  2. Split-half reliability of each direction (20 random splits) — noisy
     72-vs-72 mean differences in 128-d attenuate cosines, so raw alignment
     is reported alongside attenuation-corrected alignment
     raw / sqrt(rel_a * rel_b), computed only where both reliabilities > 0.1.
  3. Zero-shot cross-domain decoding: project each ToM stimulus's z onto the
     head's code_mech direction; AUC for true vs false. A shared variable
     predicts AUC > 0.5 using the OTHER domain's axis with no training.
  4. Internal check: cosine(code_mech, code_div) within code — the mechanism
     and token-controlled divergence directions should align with each other
     if both index the same state (and this calibrates what "aligned" looks
     like for a genuinely shared variable measured two ways).

Interpretation (preregister against these):
  shared variable   div_unique alignment ≫ shuffled null, above address/
                    random sets; decoding AUC > 0.5; corrected alignment in
                    the same range as the within-code check (4).
  cohabitation      reliabilities high but cross-domain cosine ≈ null.
  underpowered      reliabilities low; report as inconclusive, do not
                    interpret raw cosines.

Inputs:
  results/rsa_head_outputs/activations_full/{template}.pt   (ToM capture,
      capture_head_outputs_local.py --domain tom; stim records must carry
      belief_veracity)
  results/rsa_head_outputs/activations_code_position/code_position.pt
  results/rsa_head_outputs/activations_code_divergence/code_divergence.pt
  fingerprints + perhead_geometry.csv (head sets, as in the subset split)

Outputs (results/rsa_head_outputs/state_geometry/):
  perhead_geometry_alignment.csv, summary printout,
  figure results_for_pi/aggregation/state_geometry_alignment.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TOM_ACT = ROOT / "results/rsa_head_outputs/activations_full"
CODE_POS = ROOT / "results/rsa_head_outputs/activations_code_position/code_position.pt"
CODE_SIG_CSV = ROOT / "results/aggregated/code_head_sets/code_abstract_head_list.csv"
CODE_DIV = ROOT / "results/rsa_head_outputs/activations_code_divergence/code_divergence.pt"
FP_DIR = ROOT / "results/aggregated/fingerprints"
GEOM_CSV = ROOT / "results/rsa_head_outputs/code_position_analysis/perhead_geometry.csv"
OUT_DIR = ROOT / "results/rsa_head_outputs/state_geometry"
FIG_DIR = ROOT / "results_for_pi/aggregation"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
# Qwen defaults. configure() overwrites these for other models; every prior
# section (G/G2/G3/SG4) was produced with these values, so leaving them alone
# reproduces those results byte-identically.
N_LAYERS, N_HEADS, D = 48, 40, 128
NH = N_LAYERS * N_HEADS
MODEL_TYPE = "Qwen2.5-14B-Instruct"
GEMMA_DIR = None


def configure(model_type: str, gemma_dir=None):
    """Set the head-grid dimensions for `model_type` (see dims.py)."""
    global N_LAYERS, N_HEADS, D, NH, MODEL_TYPE, GEMMA_DIR
    import dims as _dims
    d = _dims.for_model_type(model_type)
    N_LAYERS, N_HEADS, D, NH = d.n_layers, d.n_heads, d.head_dim, d.nh
    MODEL_TYPE, GEMMA_DIR = model_type, gemma_dir
    return d


def unit(v, axis=-1):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n[n == 0] = 1
    return v / n


def directions(z, labels, pos_label, n_splits, rng):
    """z: (n_stim, NH, D); labels: bool array (True = pos_label condition).
    Returns (dirs (NH, D), reliability (NH,))."""
    pos, neg = z[labels], z[~labels]
    dirs = pos.mean(0) - neg.mean(0)
    rels = np.zeros((n_splits, NH), dtype=np.float32)
    for s in range(n_splits):
        pi, ni = rng.permutation(len(pos)), rng.permutation(len(neg))
        ph, nh = len(pos) // 2, len(neg) // 2
        d1 = pos[pi[:ph]].mean(0) - neg[ni[:nh]].mean(0)
        d2 = pos[pi[ph:]].mean(0) - neg[ni[nh:]].mean(0)
        rels[s] = np.einsum("hd,hd->h", unit(d1), unit(d2))
    return dirs.astype(np.float32), rels.mean(0)


def load_tom(tom_dir: Path):
    import torch
    zs, labels = [], []
    for t in TEMPLATES:
        p = torch.load(tom_dir / f"{t}.pt", weights_only=False)
        z = p["z"].numpy().astype(np.float32).reshape(-1, NH, D)
        stim = pd.DataFrame(p["stim"])
        assert "belief_veracity" in stim, "ToM capture lacks belief_veracity"
        zs.append(z)
        labels.append(stim["belief_veracity"].astype(str)
                      .str.lower().eq("true").to_numpy())
    return np.concatenate(zs), np.concatenate(labels)


def load_code(path: Path, col: str, pos_val: str, stim_csv: Path = None):
    import torch
    p = torch.load(path, weights_only=False)
    z = p["z"].numpy().astype(np.float32).reshape(-1, NH, D)
    stim = pd.DataFrame(p["stim"])
    if col not in stim and stim_csv is not None:
        full = pd.read_csv(stim_csv)
        stim = stim.merge(full, on="prompt_id", suffixes=("", "_f"),
                          validate="one_to_one")
    return z, stim[col].astype(str).eq(pos_val).to_numpy(), stim


def head_sets():
    """Delegates to head_sets_source so Qwen and Gemma share one definition.

    Verified to reproduce the previous inline implementation exactly for Qwen:
    div_unique/div_shared/dual_sig/address/random32 are bit-identical, and
    photo_C2 (added by the threeway script) is unchanged at 21 heads.
    """
    import dims as _dims
    import head_sets_source as _hss
    d = _dims.for_model_type(MODEL_TYPE)
    return _hss.build(MODEL_TYPE, d, FP_DIR, gemma_dir=GEMMA_DIR,
                      geom_csv=GEOM_CSV,
                      code_sig_csv=CODE_SIG_CSV if "gemma" not in MODEL_TYPE.lower() else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tom_dir", type=Path, default=TOM_ACT)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--min_rel", type=float, default=0.1)
    ap.add_argument("--code_pos", type=Path, default=None,
                    help="code_position capture (.pt); defaults to the Qwen one")
    ap.add_argument("--code_div", type=Path, default=None,
                    help="code_divergence capture (.pt); optional internal check")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--gemma_dir", type=Path, default=None,
                    help="dir holding gemma_c{1,2,5}_head_list.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    # Capture paths must match the model, or Gemma head indices get applied to
    # Qwen activations. Defaults stay Qwen so existing invocations are unchanged.
    global CODE_POS, CODE_DIV
    if args.code_pos: CODE_POS = args.code_pos
    if args.code_div: CODE_DIV = args.code_div
    _d = configure(args.model_type, args.gemma_dir)
    print(f"model={args.model_type}  dims={_d}")
    print("loading captures...")
    z_tom, lab_tom = load_tom(args.tom_dir)
    z_cm, lab_cm, stim_cm = load_code(CODE_POS, "semantics", "alias")
    # The code_divergence capture supplies the INTERNAL BENCHMARK (the
    # "same variable measured twice" calibration that G/G2 read corrected
    # alignment against). It is optional: without it the benchmark column is
    # unavailable and the ladder loses its top rung, but every cross-domain
    # number is still computable. Degrade loudly, not silently.
    if CODE_DIV.exists():
        z_cd, lab_cd, _ = load_code(
            CODE_DIV, "order", "mutate_first",
            ROOT / "results/rsa_head_outputs/stimuli_code_divergence/"
                   "code_divergence.csv")
    else:
        raise SystemExit(
            f"code_divergence capture not found at {CODE_DIV}.\n"
            "It supplies the internal benchmark that corrected alignments are "
            "read against (G/G2). Capture it first:\n"
            "  capture_head_outputs_local.py --device mps --domain code "
            "--templates code_divergence \\\n"
            "    --stim_dir results/rsa_head_outputs/stimuli_code_divergence "
            "\\\n    --out_dir <act_dir> --model_type <model>\n"
            "or pass --code_div explicitly.")
    print(f"  tom {z_tom.shape} ({lab_tom.sum()} true / {(~lab_tom).sum()} false)")
    print(f"  code_mech {z_cm.shape}, code_div {z_cd.shape}")

    print("computing directions + split-half reliabilities...")
    d_tom, r_tom = directions(z_tom, lab_tom, True, args.n_splits, rng)
    d_cm, r_cm = directions(z_cm, lab_cm, True, args.n_splits, rng)
    d_cd, r_cd = directions(z_cd, lab_cd, True, args.n_splits, rng)

    u_tom, u_cm, u_cd = unit(d_tom), unit(d_cm), unit(d_cd)
    cos_tm = np.einsum("hd,hd->h", u_tom, u_cm)      # tom vs code_mech
    cos_td = np.einsum("hd,hd->h", u_tom, u_cd)      # tom vs code_div
    cos_md = np.einsum("hd,hd->h", u_cm, u_cd)       # internal code check

    def corrected(raw, ra, rb):
        ok = (ra > args.min_rel) & (rb > args.min_rel)
        out = np.full_like(raw, np.nan)
        out[ok] = raw[ok] / np.sqrt(ra[ok] * rb[ok])
        return out

    ccos_tm = corrected(cos_tm, r_tom, r_cm)
    ccos_td = corrected(cos_td, r_tom, r_cd)
    ccos_md = corrected(cos_md, r_cm, r_cd)

    # zero-shot cross-domain decoding: project tom z on code_mech direction
    proj = np.einsum("shd,hd->sh", z_tom, u_cm)      # (n_stim, NH)
    from scipy.stats import rankdata
    n_pos = lab_tom.sum()
    ranks = rankdata(proj, axis=0)
    auc = (ranks[lab_tom].sum(0) - n_pos * (n_pos + 1) / 2) / \
          (n_pos * (~lab_tom).sum())

    sets = head_sets()
    df = pd.DataFrame({
        "layer": np.repeat(np.arange(N_LAYERS), N_HEADS),
        "head": np.tile(np.arange(N_HEADS), N_LAYERS),
        "cos_tom_codemech": cos_tm, "cos_tom_codediv": cos_td,
        "cos_codemech_codediv": cos_md,
        "ccos_tom_codemech": ccos_tm, "ccos_tom_codediv": ccos_td,
        "ccos_codemech_codediv": ccos_md,
        "rel_tom": r_tom, "rel_code_mech": r_cm, "rel_code_div": r_cd,
        "auc_tom_on_codemech": auc,
    })
    df["set"] = "other"
    for name, hs in sets.items():
        df.loc[hs, "set"] = name
    df.to_csv(OUT_DIR / "perhead_geometry_alignment.csv", index=False)

    # shuffled-pairing null per set, on the raw cosine
    print(f"\n=== alignment by head set (null: {args.n_perm} shuffled pairings) ===")
    rows = []
    for name, hs in sets.items():
        hs = np.array(hs)
        obs = float(cos_tm[hs].mean())
        null = np.empty(args.n_perm)
        for i in range(args.n_perm):
            null[i] = np.einsum("hd,hd->h", u_tom[hs],
                                u_cm[rng.permutation(NH)[:len(hs)]]).mean()
        p = float((null >= obs).mean())
        row = {
            "set": name, "n": len(hs),
            "cos_tom_codemech": obs, "null95": float(np.quantile(null, 0.95)),
            "p_perm": p,
            "ccos_tom_codemech": float(np.nanmean(ccos_tm[hs])),
            "cos_tom_codediv": float(cos_td[hs].mean()),
            "internal_code_cos": float(cos_md[hs].mean()),
            "auc": float(auc[hs].mean()),
            "rel_tom": float(r_tom[hs].mean()),
            "rel_code_mech": float(r_cm[hs].mean()),
        }
        rows.append(row)
        print(f"  {name:11s} cos(tom,code_mech)={obs:+.3f} "
              f"(null95={row['null95']:+.3f}, p={p:.4f})  "
              f"corrected={row['ccos_tom_codemech']:+.3f}  "
              f"AUC={row['auc']:.3f}  "
              f"[rel: tom={row['rel_tom']:.2f} code={row['rel_code_mech']:.2f}]  "
              f"internal code check={row['internal_code_cos']:+.3f}")
    pd.DataFrame(rows).to_csv(OUT_DIR / "alignment_summary.csv", index=False)

    print("\nreading guide: shared variable -> div_unique cos & AUC above "
          "null, corrected alignment comparable to the internal code check; "
          "cohabitation -> reliabilities fine but cross-domain cos ~ null; "
          "low reliabilities -> inconclusive, do not interpret.")

    # ------------------------------------------------------------------
    # v2 §1: contrast specificity — anisotropy control.
    # A model-wide baseline fraction could be activation anisotropy (any two
    # mean-difference directions share the dominant components). Anisotropy
    # inflates state<->ADDRESS alignment just as much as state<->state; a
    # genuine global state axis inflates only state<->state.
    # ------------------------------------------------------------------
    if "idx" in stim_cm:
        lab_addr = stim_cm["idx"].astype(int).to_numpy() >= 3
        d_ad, r_ad = directions(z_cm, lab_addr, True, args.n_splits, rng)
        u_ad = unit(d_ad)
        cos_ta = np.einsum("hd,hd->h", u_tom, u_ad)
        print("\n=== v2 specificity: state<->state vs state<->address "
              "fraction (corrected/internal-style, per set) ===")
        spec_rows = []
        for name, hs in sets.items():
            hs = np.array(hs)
            ok = (r_tom[hs] > args.min_rel) & (r_cm[hs] > args.min_rel)
            oka = (r_tom[hs] > args.min_rel) & (r_ad[hs] > args.min_rel)
            f_ss = float(np.nanmean(
                cos_tm[hs][ok] / np.sqrt(r_tom[hs][ok] * r_cm[hs][ok])))
            f_sa = float(np.nanmean(
                cos_ta[hs][oka] / np.sqrt(r_tom[hs][oka] * r_ad[hs][oka]))) \
                if oka.any() else float("nan")
            spec_rows.append({"set": name, "frac_state_state": f_ss,
                              "frac_state_address": f_sa,
                              "rel_addr": float(r_ad[hs].mean())})
            print(f"  {name:11s} state<->state={f_ss:+.3f}  "
                  f"state<->address={f_sa:+.3f}  (rel_addr={r_ad[hs].mean():.2f})")
        pd.DataFrame(spec_rows).to_csv(OUT_DIR / "specificity.csv", index=False)
        print("  anisotropy predicts the two columns match; a genuine "
              "state axis predicts state<->state >> state<->address.")
    else:
        print("\n(v2 specificity skipped: code_position stim lacks idx)")

    # ------------------------------------------------------------------
    # v2 §2: concat-level alignment — do the per-head shared components
    # cohere into ONE model-level axis? If concat alignment >> per-head
    # mean, sharing is a single broadcast axis (with per-head gain).
    # ------------------------------------------------------------------
    print("\n=== v2 concat-level alignment (per set) ===")
    cc_rows = []
    for name, hs in sets.items():
        hs = np.array(hs)
        zt = z_tom[:, hs, :].reshape(len(z_tom), -1)
        zc = z_cm[:, hs, :].reshape(len(z_cm), -1)

        def cdir(z, lab):
            return z[lab].mean(0) - z[~lab].mean(0)

        def crel(z, lab, n=args.n_splits):
            out = []
            for _ in range(n):
                pi = rng.permutation(lab.sum()); ni = rng.permutation((~lab).sum())
                p, q = z[lab], z[~lab]
                d1 = p[pi[:len(pi)//2]].mean(0) - q[ni[:len(ni)//2]].mean(0)
                d2 = p[pi[len(pi)//2:]].mean(0) - q[ni[len(ni)//2:]].mean(0)
                out.append(float(np.dot(unit(d1, 0), unit(d2, 0))))
            return float(np.mean(out))

        dt, dc = cdir(zt, lab_tom), cdir(zc, lab_cm)
        raw = float(np.dot(unit(dt, 0), unit(dc, 0)))
        rt, rc = crel(zt, lab_tom), crel(zc, lab_cm)
        corr = raw / np.sqrt(rt * rc) if min(rt, rc) > args.min_rel else float("nan")
        cc_rows.append({"set": name, "concat_cos": raw,
                        "concat_corrected": corr, "rel_tom": rt, "rel_code": rc,
                        "perhead_mean_cos": float(cos_tm[hs].mean())})
        print(f"  {name:11s} concat cos={raw:+.3f} corrected={corr:+.3f} "
              f"(per-head mean {cos_tm[hs].mean():+.3f}; rel {rt:.2f}/{rc:.2f})")
    pd.DataFrame(cc_rows).to_csv(OUT_DIR / "concat_alignment.csv", index=False)

    # ------------------------------------------------------------------
    # v2 §3: shared-axis decomposition, split-honest.
    # Estimate the per-head shared axis s from TRAIN ToM data + all code
    # data; on held-out ToM data test: (a) cross-domain decoding via the
    # code direction's residue after removing s (1-D sharing -> ~0.5);
    # (b) within-domain decoding after projecting s out of the data and the
    # axis (domain-specific residue is real signal if this stays high).
    # ------------------------------------------------------------------
    print("\n=== v2 shared-axis decomposition (20 stratified splits) ===")
    dec = {name: {"cross_raw": [], "cross_resid": [],
                  "within_raw": [], "within_proj": []} for name in sets}

    def auc_of(scores, lab):
        from scipy.stats import rankdata as rd
        r = rd(scores)
        n1 = lab.sum()
        return (r[lab].sum() - n1 * (n1 + 1) / 2) / (n1 * (~lab).sum())

    pos_i = np.flatnonzero(lab_tom); neg_i = np.flatnonzero(~lab_tom)
    for s_i in range(20):
        rng2 = np.random.default_rng(100 + s_i)
        pa = rng2.permutation(pos_i); na = rng2.permutation(neg_i)
        tr = np.concatenate([pa[:len(pa)//2], na[:len(na)//2]])
        te = np.concatenate([pa[len(pa)//2:], na[len(na)//2:]])
        lab_te = lab_tom[te]
        u_tomA = unit(z_tom[tr][lab_tom[tr]].mean(0)
                      - z_tom[tr][~lab_tom[tr]].mean(0))
        agree = np.einsum("hd,hd->h", u_tomA, u_cm) > 0
        s_ax = unit(u_tomA + u_cm)                       # per-head shared axis
        r_cm_res = unit(u_cm - np.einsum("hd,hd->h", u_cm, s_ax)[:, None] * s_ax)
        u_tom_res = unit(u_tomA - np.einsum("hd,hd->h", u_tomA, s_ax)[:, None] * s_ax)
        zte = z_tom[te]
        zte_proj = zte - np.einsum("shd,hd->sh", zte, s_ax)[..., None] * s_ax
        for name, hs in sets.items():
            hs = np.array(hs)[agree[np.array(hs)]]
            if len(hs) == 0:
                continue
            dec[name]["cross_raw"].append(np.mean(
                [auc_of(np.einsum("sd,d->s", zte[:, h], u_cm[h]), lab_te)
                 for h in hs]))
            dec[name]["cross_resid"].append(np.mean(
                [auc_of(np.einsum("sd,d->s", zte[:, h], r_cm_res[h]), lab_te)
                 for h in hs]))
            dec[name]["within_raw"].append(np.mean(
                [auc_of(np.einsum("sd,d->s", zte[:, h], u_tomA[h]), lab_te)
                 for h in hs]))
            dec[name]["within_proj"].append(np.mean(
                [auc_of(np.einsum("sd,d->s", zte_proj[:, h], u_tom_res[h]),
                        lab_te) for h in hs]))
    dec_rows = []
    for name in sets:
        row = {"set": name}
        for k, v in dec[name].items():
            row[k] = float(np.mean(v)) if v else float("nan")
        dec_rows.append(row)
        print(f"  {name:11s} cross raw={row['cross_raw']:.3f} -> "
              f"resid={row['cross_resid']:.3f}   within raw="
              f"{row['within_raw']:.3f} -> proj={row['within_proj']:.3f}")
    pd.DataFrame(dec_rows).to_csv(OUT_DIR / "decomposition.csv", index=False)
    print("  1-D sharing -> cross resid ~0.5; real domain-specific residue "
          "-> within proj stays high; both together = shared core + "
          "domain dressing.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GRAY, VERM = "#0072B2", "#E69F00", "#B0B0B0", "#D55E00"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"div_unique": ORANGE, "div_shared": VERM,
              "address": BLUE, "random32": GRAY}
    ax = axes[0]
    for name, hs in sets.items():
        ax.hist(cos_tm[np.array(hs)], bins=15, alpha=0.55,
                label=name, color=colors.get(name, GRAY))
    ax.axvline(0, color="#DDDDDD")
    ax.set_xlabel("cos(tom_state, code_mech_state) per head")
    ax.set_title("cross-domain state alignment"); ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.scatter(np.sqrt(np.clip(r_tom * r_cm, 0, None)), cos_tm, s=6,
               color=GRAY, alpha=0.4)
    hs = np.array(sets["div_unique"])
    ax.scatter(np.sqrt(np.clip(r_tom[hs] * r_cm[hs], 0, None)), cos_tm[hs],
               s=26, color=ORANGE, label="div_unique")
    ax.set_xlabel("sqrt(rel_tom x rel_code)"); ax.set_ylabel("raw cosine")
    ax.set_title("alignment vs reliability ceiling"); ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    for name, hs in sets.items():
        ax.hist(auc[np.array(hs)], bins=15, alpha=0.55, label=name,
                color=colors.get(name, GRAY))
    ax.axvline(0.5, color="#DDDDDD")
    ax.set_xlabel("AUC: ToM true/false via code direction (zero-shot)")
    ax.set_title("cross-domain decoding"); ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "state_geometry_alignment.png", dpi=160)
    print(f"\nwrote -> {OUT_DIR}, figure -> {FIG_DIR}")


if __name__ == "__main__":
    main()
