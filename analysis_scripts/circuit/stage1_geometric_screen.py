"""Stage 1 of the layered-lookback circuit: the geometric screen.

Laptop-side linear algebra over the capture artifacts produced by
capture_circuit_activations.py (no model load). Implements S1a/S1b/S1c
from docs/design_layered_lookback_circuit.md plus the direction ladder
rungs 1-2, and prints the go/no-go verdict that gates Stage 2.

Split discipline: all directions (Delta r_h, Delta resid, SVD subspace)
are fit on a random half of matched pairs per template; every reported
statistic projects the HELD-OUT half's bus difference. Stage 2 must steer
held-out prompts only — the split indices are saved into the directions
file so it can't drift.

  S1a  writer projection   frac = ||P_span{Dr_h}(Dresid_hold)||^2 / ||Dresid_hold||^2
       null: 1000 random same-size head sets from L28-40 (matches layer
       range + subspace dim). MLP variant decides head- vs MLP-written
       wire; its null is random Gaussian subspaces of matched dim (there
       is no natural "random MLP set" of matched granularity).
  S1b  per-head authorship + cumulative-by-layer curve
  S1c  reader survival: ||LN(v) W_Q^(g)|| (and W_V) vs 1000 random unit
       vectors. Model was loaded fold_ln=False, so LN(v) here is
       RMSNorm-without-bias with the reader layer's own ln1.w applied.
  Ladder: rank-1 (Dresid vs sum of writer writes — their cosine is a
       sanity stat) and top-k SVD of per-pair diffs (k: smallest
       capturing 75% of total sigma^2, capped at 8).

GO to Stage 2 iff (pooled, both sites): S1a frac > 95th pct of the
random-head-set null AND at least one reader family (Q or V) has median
read ratio > 2. S1a-fail + MLP-pass => proceed with Dresid-derived
directions only (wire real, writers not C1). Both fail => stop, rethink.

Outputs -> results/aggregated/circuit/stage1[_sensitivity]/
  s1a_projection.csv, s1b_writer_contributions.csv, s1b_cumulative.csv,
  s1c_reader_read.csv, svd_spectrum.csv, directions_{site}.npz,
  verdict.txt, figures/*.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_common import (  # noqa: E402
    BAND_LAYERS, CAPTURE_ROOT, DOMAINS, SITE_KEYS, STAGE1_OUT,
    WRITER_CSV, load_head_csv, orthonormal_basis,
    projection_fraction, rms_normalize, split_pair_indices,
)


# ---------------------------------------------------------------------------
# loading + means
# ---------------------------------------------------------------------------

def load_captures(capture_dir: Path, templates: list, weights_path: Path):
    caps = {}
    for t in templates:
        p = capture_dir / f"{t}.pt"
        if not p.exists():
            print(f"WARNING: {p} missing — screening without template {t}")
            continue
        caps[t] = torch.load(p, map_location="cpu", weights_only=False)
    if not caps:
        sys.exit(f"no capture files under {capture_dir}")
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    assert weights["band_layers"] == list(BAND_LAYERS)
    return caps, weights


def split_means(caps: dict, split_seed: int, variants: list):
    """Per (template, split-half): mean z/mlp/resid per variant, float64.

    Returns dict[template] = {"fit_idx", "hold_idx",
        "z": {half: {variant: (n_band, n_heads, d_head)}}, "mlp": ...,
        "resid": {half: {variant: (n_sites, d_model)}}, "n_pairs"}.
    """
    out = {}
    for t, cap in caps.items():
        n = cap["meta"]["n_pairs"]
        fit_idx, hold_idx = split_pair_indices(n, t, split_seed)
        entry = {"fit_idx": fit_idx, "hold_idx": hold_idx, "n_pairs": n,
                 "z": {}, "mlp": {}, "resid": {}, "resid_raw": cap["resid"]}
        for half, idx in (("fit", fit_idx), ("hold", hold_idx)):
            entry["z"][half] = {v: cap["z"][v][idx].to(torch.float64).mean(0).numpy()
                                for v in variants}
            entry["mlp"][half] = {v: cap["mlp"][v][idx].to(torch.float64).mean(0).numpy()
                                  for v in variants}
            entry["resid"][half] = {v: cap["resid"][v][idx].to(torch.float64).mean(0).numpy()
                                    for v in variants}
        out[t] = entry
    return out


def pooled_delta(per_template: dict, kind: str, half: str, fb: str, tb: str) -> np.ndarray:
    """divergent-minus-nondivergent mean difference pooled over templates."""
    num, count = None, 0
    for t, e in per_template.items():
        n = len(e[f"{half}_idx"])
        d = e[kind][half][fb] - e[kind][half][tb]
        num = d * n if num is None else num + d * n
        count += n
    return num / count


# ---------------------------------------------------------------------------
# S1a / S1b
# ---------------------------------------------------------------------------

def writer_delta_writes(delta_z_fit: np.ndarray, w_o_band: np.ndarray) -> np.ndarray:
    """Per-head divergence write Dr_h = Dz_h W_O^(h) for ALL band heads.

    delta_z_fit: (n_band, n_heads, d_head); w_o_band: same leading dims +
    (d_head, d_model). Returns (n_band, n_heads, d_model).
    """
    return np.einsum("lhd,lhdm->lhm", delta_z_fit, w_o_band)


def s1a_row(basis_vectors, delta_hold, delta_fit, site, scope, subspace,
            null_fracs=None, null_type=""):
    basis = orthonormal_basis(np.asarray(basis_vectors))
    row = dict(site=site, scope=scope, subspace=subspace,
               n_vectors=len(basis_vectors), rank=basis.shape[0],
               frac_fit=projection_fraction(basis, delta_fit),
               frac_hold=projection_fraction(basis, delta_hold),
               null_type=null_type, null_median=np.nan, null_p95=np.nan,
               p_perm=np.nan)
    if null_fracs is not None:
        null_fracs = np.asarray(null_fracs)
        row["null_median"] = float(np.median(null_fracs))
        row["null_p95"] = float(np.quantile(null_fracs, 0.95))
        row["p_perm"] = float((1 + np.sum(null_fracs >= row["frac_hold"]))
                              / (len(null_fracs) + 1))
    return row


def random_headset_null(dr_flat, k, delta_hold, n_null, rng):
    fracs = np.empty(n_null)
    n_pool = dr_flat.shape[0]
    for i in range(n_null):
        sel = rng.choice(n_pool, size=k, replace=False)
        fracs[i] = projection_fraction(orthonormal_basis(dr_flat[sel]), delta_hold)
    return fracs


def gaussian_subspace_null(dim, d_model, delta_hold, n_null, rng):
    fracs = np.empty(n_null)
    for i in range(n_null):
        basis = orthonormal_basis(rng.standard_normal((dim, d_model)))
        fracs[i] = projection_fraction(basis, delta_hold)
    return fracs


# ---------------------------------------------------------------------------
# S1c
# ---------------------------------------------------------------------------

def reader_read_norms(directions: np.ndarray, weights: dict) -> dict:
    """||LN(v) W|| per reader head for each direction (rows of `directions`).

    fold_ln=False at load time => LN(v) = ln1.w * (v / rms(v)), using the
    reader layer's own ln1.w. Returns {"Q": (n_dirs, n_readers), "V": ...}.
    """
    v = rms_normalize(directions)  # (n_dirs, d_model)
    ln_w = weights["ln1_w_readers"].numpy().astype(np.float64)  # (n_readers, d)
    n_readers = ln_w.shape[0]
    out = {}
    for family, key in (("Q", "W_Q_readers"), ("V", "W_V_readers")):
        w = weights[key].numpy().astype(np.float64)  # (n_readers, d, d_head)
        norms = np.empty((v.shape[0], n_readers))
        for ri in range(n_readers):  # loop keeps peak memory at O(n_dirs * d)
            norms[:, ri] = np.linalg.norm((v * ln_w[ri]) @ w[ri], axis=-1)
        out[family] = norms
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="tom")
    ap.add_argument("--capture_dir", type=Path, default=None,
                    help="default: stage1_capture/{model}[/code_domain]")
    ap.add_argument("--weights_path", type=Path, default=None,
                    help="default: the ToM capture dir's weights.pt")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--writer_set",
                    choices=["primary", "sensitivity", "code_abstract"],
                    default=None, help="default: domain's canonical set")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--null_seed", type=int, default=1234)
    ap.add_argument("--n_null", type=int, default=1000)
    ap.add_argument("--var_target", type=float, default=0.75)
    ap.add_argument("--k_cap", type=int, default=8)
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    FB, TB = domain["variants"]  # divergent, non-divergent
    if args.writer_set is None:
        args.writer_set = domain["default_writer_set"]
    tom_capture = CAPTURE_ROOT / args.model_type
    capture_dir = args.capture_dir or (tom_capture / domain["capture_subdir"])
    weights_path = args.weights_path or (tom_capture / "weights.pt")

    if args.out_dir:
        out_dir = args.out_dir
    elif args.domain == "code":
        out_dir = STAGE1_OUT.parent / "stage1_code"
    elif args.writer_set == "sensitivity":
        out_dir = STAGE1_OUT.parent / "stage1_sensitivity"
    else:
        out_dir = STAGE1_OUT
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    caps, weights = load_captures(capture_dir, domain["templates"], weights_path)
    templates = list(caps.keys())
    per_t = split_means(caps, args.split_seed, domain["variants"])

    writers = load_head_csv(WRITER_CSV, head_set=args.writer_set)
    writer_pairs = list(zip(writers["layer"], writers["head"]))
    band_index = {L: i for i, L in enumerate(BAND_LAYERS)}
    w_o = weights["W_O_band"].to(torch.float64).numpy()
    n_heads, d_model = weights["n_heads"], weights["d_model"]
    n_band = len(BAND_LAYERS)
    print(f"writer set '{args.writer_set}': {len(writer_pairs)} heads; "
          f"readers: {len(weights['readers'])}; templates: {templates}")

    rng = np.random.default_rng(args.null_seed)
    writer_flat_idx = np.array([band_index[L] * n_heads + H
                                for L, H in writer_pairs])

    s1a_rows, s1b_rows, cum_rows, s1c_rows, spec_rows = [], [], [], [], []
    verdict_lines = []
    scopes = templates + ["pooled"]

    for site_i, site in enumerate(SITE_KEYS):
        # -- deltas per scope ------------------------------------------------
        for scope in scopes:
            if scope == "pooled":
                dz_fit = pooled_delta(per_t, "z", "fit", FB, TB)
                dmlp_fit = pooled_delta(per_t, "mlp", "fit", FB, TB)
                dres_fit = pooled_delta(per_t, "resid", "fit", FB, TB)[site_i]
                dres_hold = pooled_delta(per_t, "resid", "hold", FB, TB)[site_i]
            else:
                e = per_t[scope]
                dz_fit = e["z"]["fit"][FB] - e["z"]["fit"][TB]
                dmlp_fit = e["mlp"]["fit"][FB] - e["mlp"]["fit"][TB]
                dres_fit = (e["resid"]["fit"][FB] - e["resid"]["fit"][TB])[site_i]
                dres_hold = (e["resid"]["hold"][FB] - e["resid"]["hold"][TB])[site_i]

            dr_all = writer_delta_writes(dz_fit, w_o)          # (band, heads, d)
            dr_flat = dr_all.reshape(n_band * n_heads, d_model)
            dr_writers = dr_flat[writer_flat_idx]

            # S1a: writers + random-head-set null
            null_heads = random_headset_null(dr_flat, len(writer_pairs),
                                             dres_hold, args.n_null, rng)
            row_writers = s1a_row(dr_writers, dres_hold, dres_fit, site, scope,
                                  "writers", null_heads, "random_head_sets")
            s1a_rows.append(row_writers)
            # S1a MLP variant + gaussian null of matched dim
            null_mlp = gaussian_subspace_null(n_band, d_model, dres_hold,
                                              args.n_null, rng)
            row_mlp = s1a_row(dmlp_fit, dres_hold, dres_fit, site, scope,
                              "mlp", null_mlp, "gaussian_subspaces")
            s1a_rows.append(row_mlp)

            if scope == "pooled":
                pooled_writers_row, pooled_mlp_row = row_writers, row_mlp
                pooled_dr_writers, pooled_dres_fit, pooled_dres_hold = \
                    dr_writers, dres_fit, dres_hold
                pooled_null_heads = null_heads

                # S1b per-head + cumulative-by-layer (fit spans, hold bus)
                v_hold = dres_hold / np.linalg.norm(dres_hold)
                for (L, H), dr in zip(writer_pairs, dr_writers):
                    s1b_rows.append(dict(
                        site=site, layer=L, head=H,
                        dr_norm=float(np.linalg.norm(dr)),
                        proj_on_bus_hold=float(dr @ v_hold),
                        cos_bus_hold=float(dr @ v_hold / max(np.linalg.norm(dr), 1e-12)),
                    ))
                for L in sorted({L for L, _ in writer_pairs}):
                    sel = [i for i, (l, _) in enumerate(writer_pairs) if l <= L]
                    cum_rows.append(dict(
                        site=site, layer=L, n_heads=len(sel),
                        frac_hold_cum=projection_fraction(
                            orthonormal_basis(dr_writers[sel]), dres_hold),
                    ))

        # -- direction ladder (pooled fit half only) -------------------------
        v_rank1 = pooled_dres_fit / np.linalg.norm(pooled_dres_fit)
        sum_writes = pooled_dr_writers.sum(0)
        cos_sanity = float(v_rank1 @ (sum_writes / np.linalg.norm(sum_writes)))

        diffs, fb_fit, tb_fit = [], [], []
        for t in templates:
            e = per_t[t]
            r = e["resid_raw"]
            fit = e["fit_idx"]
            diffs.append((r[FB][fit, site_i].to(torch.float64)
                          - r[TB][fit, site_i].to(torch.float64)).numpy())
            fb_fit.append(r[FB][fit, site_i].to(torch.float64).numpy())
            tb_fit.append(r[TB][fit, site_i].to(torch.float64).numpy())
        D = np.concatenate(diffs)                       # (n_fit_pairs, d)
        _, svals, vt = np.linalg.svd(D, full_matrices=False)
        var_frac = svals ** 2 / np.sum(svals ** 2)
        cum = np.cumsum(var_frac)
        k = int(min(np.searchsorted(cum, args.var_target) + 1, args.k_cap))
        for ci in range(min(32, len(svals))):
            spec_rows.append(dict(site=site, component=ci + 1,
                                  sigma2_frac=float(var_frac[ci]),
                                  cum_frac=float(cum[ci]),
                                  within_chosen_k=bool(ci < k)))
        print(f"[{site}] ladder: cos(Dresid, sum writer writes)={cos_sanity:.3f}; "
              f"top-k SVD k={k} (cum var {cum[k - 1]:.3f})")

        np.savez(out_dir / f"directions_{site}.npz",
                 v_rank1=v_rank1, sum_writes_dir=sum_writes / np.linalg.norm(sum_writes),
                 cos_rank1_sum_writes=cos_sanity,
                 U_topk=vt[:k], sigma=svals, k=k,
                 mean_FB_fit=np.concatenate(fb_fit).mean(0),
                 mean_TB_fit=np.concatenate(tb_fit).mean(0),
                 natural_scale=np.linalg.norm(pooled_dres_fit),
                 split_seed=args.split_seed, writer_set=args.writer_set,
                 domain=args.domain, variants=np.array(domain["variants"]),
                 templates=np.array(templates),
                 **{f"fit_idx_{t}": per_t[t]["fit_idx"] for t in templates},
                 **{f"hold_idx_{t}": per_t[t]["hold_idx"] for t in templates})

        # -- S1c: reader survival (rank-1 + top-k dirs vs random unit v) -----
        dirs = np.vstack([v_rank1, vt[:k]])
        reads = reader_read_norms(dirs, weights)
        rand_dirs = rng.standard_normal((args.n_null, d_model))
        rand_dirs /= np.linalg.norm(rand_dirs, axis=1, keepdims=True)
        rand_reads = reader_read_norms(rand_dirs, weights)
        family_medians = {}
        for family in ("Q", "V"):
            rand_median = np.median(rand_reads[family], axis=0)  # (n_readers,)
            ratio_rank1 = reads[family][0] / rand_median
            ratio_topk = (reads[family][1:] / rand_median).max(axis=0)
            for ri, r in enumerate(weights["readers"]):
                s1c_rows.append(dict(
                    site=site, layer=r["layer"], head=r["head"],
                    source=r["source"], family=family,
                    read_rank1=float(reads[family][0, ri]),
                    rand_median=float(rand_median[ri]),
                    ratio_rank1=float(ratio_rank1[ri]),
                    ratio_topk_max=float(ratio_topk[ri]),
                ))
            family_medians[family] = float(np.median(ratio_rank1))

        # -- go/no-go per site (pooled) ---------------------------------------
        writers_pass = pooled_writers_row["frac_hold"] > pooled_writers_row["null_p95"]
        mlp_pass = pooled_mlp_row["frac_hold"] > pooled_mlp_row["null_p95"]
        reader_pass = max(family_medians.values()) > 2.0
        verdict = ("GO" if writers_pass and reader_pass else
                   "GO_MLP_WIRE" if (not writers_pass and mlp_pass and reader_pass) else
                   "NO_GO")
        verdict_lines.append(
            f"[{site}] writers frac_hold={pooled_writers_row['frac_hold']:.4f} "
            f"(null p95={pooled_writers_row['null_p95']:.4f}, "
            f"p={pooled_writers_row['p_perm']:.4f}) -> "
            f"{'pass' if writers_pass else 'FAIL'} | "
            f"mlp frac_hold={pooled_mlp_row['frac_hold']:.4f} "
            f"(null p95={pooled_mlp_row['null_p95']:.4f}) -> "
            f"{'pass' if mlp_pass else 'FAIL'} | "
            f"reader median ratio Q={family_medians['Q']:.2f} "
            f"V={family_medians['V']:.2f} -> "
            f"{'pass' if reader_pass else 'FAIL'} | "
            f"cos(rank1, sum-writes)={cos_sanity:.3f} | verdict: {verdict}")

        # -- figures ----------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(pooled_null_heads, bins=40, color="0.7",
                label=f"null: random {len(writer_pairs)}-head sets (L28-40)")
        ax.axvline(pooled_writers_row["frac_hold"], color="tab:blue", lw=2,
                   label=f"writers frac={pooled_writers_row['frac_hold']:.3f}")
        ax.axvline(pooled_mlp_row["frac_hold"], color="tab:orange", lw=2, ls="--",
                   label=f"MLP frac={pooled_mlp_row['frac_hold']:.3f}")
        ax.set_xlabel("projection fraction of held-out Δresid")
        ax.set_ylabel("count")
        ax.set_title(f"S1a writer projection — {site} (pooled)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"s1a_null_hist_{site}.png", dpi=150)
        plt.close(fig)

        cdf = pd.DataFrame([r for r in cum_rows if r["site"] == site])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(cdf["layer"], cdf["frac_hold_cum"], marker="o")
        ax.set_xlabel("max writer layer included")
        ax.set_ylabel("cumulative projection fraction (hold)")
        ax.set_title(f"S1b cumulative-by-layer — {site}")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(fig_dir / f"s1b_cumulative_{site}.png", dpi=150)
        plt.close(fig)

        sdf = pd.DataFrame([r for r in s1c_rows if r["site"] == site])
        fig, ax = plt.subplots(figsize=(9, 4))
        labels = [f"L{r.layer}H{r.head}" for r in
                  sdf[sdf["family"] == "Q"].itertuples()]
        x = np.arange(len(labels))
        ax.bar(x - 0.2, sdf[sdf["family"] == "Q"]["ratio_rank1"], 0.4, label="Q")
        ax.bar(x + 0.2, sdf[sdf["family"] == "V"]["ratio_rank1"], 0.4, label="V")
        ax.axhline(2.0, color="k", ls="--", lw=1, label="ratio=2 (go threshold)")
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_ylabel("read ratio vs random-v median")
        ax.set_title(f"S1c reader survival (rank-1 direction) — {site}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"s1c_read_ratios_{site}.png", dpi=150)
        plt.close(fig)

        spdf = pd.DataFrame([r for r in spec_rows if r["site"] == site])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(spdf["component"], spdf["sigma2_frac"], marker="o",
                label="per-component σ² frac")
        ax.plot(spdf["component"], spdf["cum_frac"], marker=".",
                label="cumulative")
        ax.axvline(k, color="k", ls="--", lw=1, label=f"chosen k={k}")
        ax.axhline(args.var_target, color="0.5", ls=":", lw=1)
        ax.set_xlabel("SVD component")
        ax.set_title(f"pair-difference spectrum — {site}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"svd_spectrum_{site}.png", dpi=150)
        plt.close(fig)

    # -- write tables + verdict ------------------------------------------------
    pd.DataFrame(s1a_rows).to_csv(out_dir / "s1a_projection.csv", index=False)
    pd.DataFrame(s1b_rows).to_csv(out_dir / "s1b_writer_contributions.csv", index=False)
    pd.DataFrame(cum_rows).to_csv(out_dir / "s1b_cumulative.csv", index=False)
    pd.DataFrame(s1c_rows).to_csv(out_dir / "s1c_reader_read.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(out_dir / "svd_spectrum.csv", index=False)

    header = (f"stage-1 geometric screen — writer_set={args.writer_set}, "
              f"split_seed={args.split_seed}, n_null={args.n_null}, "
              f"templates={templates}")
    print("\n" + header)
    for line in verdict_lines:
        print(line)
    (out_dir / "verdict.txt").write_text(header + "\n" + "\n".join(verdict_lines) + "\n")
    print(f"\noutputs -> {out_dir}")


if __name__ == "__main__":
    main()
