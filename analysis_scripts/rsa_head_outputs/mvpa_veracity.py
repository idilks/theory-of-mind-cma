"""MVPA (fMRI-style linear decoding) of belief veracity from filtered head sets.

Sept 2026 (Sept 3 meeting action item: "decoding on filtered heads"). Mirrors
the §H1 protocol in docs/divergence_content_results.md, whose script and
"compiled 2x2" capture live on Steven's machine, on the capture that IS here:
results/rsa_head_outputs/activations_full/{template}.pt (144 ToM stimuli,
z at the final token, [stim, layer, head, d_head]).

Decoder: L2 logistic regression on the concatenated head-output vectors of a
head set (n_heads x 128 features), features z-scored on the training fold.
No hidden layers, nothing learned beyond one hyperplane.

Two cross-validation schemes:
  LOTO      leave-one-template-out (train on 2 templates, test on the 3rd);
            the strict scheme, no scenario shared between train and test.
  within    stratified 6-fold pooled over templates (the "compiled" analog).

Null: labels permuted WITHIN template (n_perm draws), same pipeline.
Size-matched random head sets (n_random draws) give the "is this set special"
band that §H1 found the CMA sets do NOT clear.

CONTENT decoding (which location is believed) is NOT runnable on this capture:
36 distinct location pairs across 144 stimuli, so each believed location occurs
in only two false-belief prompts and never in a held-out fold's training set.
It needs a capture over a small repeated location vocabulary (see the Sept 8
plan). This script exposes --target so it can be pointed at that capture
unchanged once it exists.

Outputs: results/rsa_head_outputs/mvpa_veracity/{summary.csv, random_band.csv}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dims as _dims  # noqa: E402
import head_sets_source as _hss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations_full"
FP_DIR = ROOT / "results/aggregated/fingerprints"
CODE_SIG_CSV = ROOT / "results/aggregated/code_head_sets/code_abstract_head_list.csv"
OUT_DIR = ROOT / "results/rsa_head_outputs/mvpa_veracity"
TEMPLATES = ("food_truck", "hair_styling", "library_book")


def valid_rows(stim: pd.DataFrame) -> np.ndarray:
    """Mask of well-formed stimulus rows.

    A concurrent-writer race on the cluster (job 9383566) appended a fragment
    row to library_book.csv; the capture ran it anyway. Such rows have NaN /
    non-string scenario fields and are dropped here, loudly."""
    ok = np.ones(len(stim), dtype=bool)
    for col in ("scenario", "question", "loc_a", "loc_b", "answer_state"):
        ok &= stim[col].map(lambda v: isinstance(v, str) and len(v) > 0).to_numpy()
    ok &= stim["belief_veracity"].astype(str).str.lower().isin(["true", "false"]).to_numpy()
    return ok


def load_capture(act_dir: Path, templates):
    zs, rows = [], []
    for t in templates:
        p = torch.load(act_dir / f"{t}.pt", map_location="cpu", weights_only=False)
        zs.append(p["z"].float().numpy())
        rows.extend(p["stim"])
    z = np.concatenate(zs, 0)  # [n_stim, L, H, D]
    stim = pd.DataFrame(rows)
    ok = valid_rows(stim)
    if not ok.all():
        bad = stim.index[~ok].tolist()
        print(f"WARNING: dropping {len(bad)} malformed stimulus row(s) at index {bad} "
              f"(templates {stim.loc[~ok, 'template'].astype(str).tolist()})")
        z, stim = z[ok], stim[ok].reset_index(drop=True)
    stim["belief_veracity"] = stim["belief_veracity"].astype(str).str.lower() == "true"
    stim["believed"] = np.where(stim["belief_veracity"], stim["loc_b"], stim["loc_a"])
    return z, stim


def build_sets(d, top_c1: int) -> dict:
    sets = _hss.build("Qwen2.5-14B-Instruct", d, FP_DIR, code_sig_csv=CODE_SIG_CSV)
    fp1 = np.load(FP_DIR / "C1_abstract_belief_clamped.npy")
    fp5 = np.load(FP_DIR / "C5_answer_changes_location_only_clamped.npy")
    sets["c5_retrieval"] = sorted(np.argsort(fp5)[-22:].tolist())
    sets[f"top{top_c1}_C1"] = sorted(np.argsort(fp1)[-top_c1:].tolist())
    sets.pop("address", None)  # code-position set, not a ToM set
    return sets


def features(z, heads, d):
    idx = np.array([d.unflat(h) for h in heads])  # (k, 2) layer, head
    return z[:, idx[:, 0], idx[:, 1], :].reshape(z.shape[0], -1)


def fit_eval(Xtr, ytr, Xte, yte, C: float):
    """Binary: (accuracy, AUC). Multi-class: (balanced accuracy, nan)."""
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xtr), ytr)
    if len(clf.classes_) > 2:
        pred = clf.predict(sc.transform(Xte))
        return float(balanced_accuracy_score(yte, pred)), np.nan
    s = clf.decision_function(sc.transform(Xte))
    acc = float(((s > 0).astype(int) == yte).mean())
    auc = float(roc_auc_score(yte, s)) if len(np.unique(yte)) > 1 else np.nan
    return acc, auc


def cv_loto(X, y, template, C):
    accs, aucs = [], []
    for t in np.unique(template):
        te = template == t
        a, u = fit_eval(X[~te], y[~te], X[te], y[te], C)
        accs.append(a); aucs.append(u)
    return float(np.mean(accs)), float(np.nanmean(aucs))


def cv_within(X, y, C, n_splits=6, seed=0):
    accs, aucs = [], []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        a, u = fit_eval(X[tr], y[tr], X[te], y[te], C)
        accs.append(a); aucs.append(u)
    return float(np.mean(accs)), float(np.nanmean(aucs))


def permute_within_template(y, template, rng):
    yp = y.copy()
    for t in np.unique(template):
        m = template == t
        yp[m] = rng.permutation(y[m])
    return yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act_dir", type=Path, default=ACT_DIR)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--target", default="belief_veracity",
                    help="stim column to decode (binary); 'believed' once a "
                         "repeated-vocabulary capture exists")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--n_random", type=int, default=50)
    ap.add_argument("--top_c1", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = _dims.for_model_type("Qwen2.5-14B-Instruct")
    z, stim = load_capture(args.act_dir, TEMPLATES)
    assert z.shape[1:] == (d.n_layers, d.n_heads, d.head_dim), z.shape
    print(f"capture: {z.shape[0]} stimuli, grid {z.shape[1:]}")

    y_raw = stim[args.target].astype(str).to_numpy()
    template = stim["template"].to_numpy()
    if args.target == "believed":
        # content is a per-template vocabulary; decode within template so
        # LOTO stays meaningful: relabel as vocabulary INDEX, which the
        # content capture keeps aligned across templates (believed_idx)
        assert "believed_idx" in stim, ("content decoding needs the content "
                                        "capture (believed_idx column)")
        y_raw = stim["believed_idx"].astype(int).to_numpy()
    classes = np.unique(y_raw)
    assert len(classes) >= 2, f"--target {args.target} has one class"
    y = ((y_raw == classes[-1]).astype(int) if len(classes) == 2
         else np.searchsorted(classes, y_raw))
    chance = 1.0 / len(classes)
    print(f"target {args.target}: {len(classes)} classes {classes.tolist()[:8]}"
          f"{'...' if len(classes) > 8 else ''}; chance {chance:.3f}; "
          f"{'accuracy' if len(classes) == 2 else 'balanced accuracy'} reported")

    sets = build_sets(d, args.top_c1)
    rng = np.random.default_rng(args.seed)
    rows = []
    for name, heads in sets.items():
        X = features(z, heads, d)
        loto_acc, loto_auc = cv_loto(X, y, template, args.C)
        w_acc, w_auc = cv_within(X, y, args.C)
        null = np.array([cv_loto(X, permute_within_template(y, template, rng),
                                 template, args.C)[0] for _ in range(args.n_perm)])
        p = float((np.sum(null >= loto_acc) + 1) / (args.n_perm + 1))
        rows.append(dict(head_set=name, n_heads=len(heads), n_features=X.shape[1],
                         loto_acc=loto_acc, loto_auc=loto_auc,
                         within_acc=w_acc, within_auc=w_auc,
                         null_acc_mean=float(null.mean()),
                         null_acc_p95=float(np.percentile(null, 95)), p_perm=p))
        print(f"  {name:14s} k={len(heads):3d}  LOTO acc {loto_acc:.3f} auc {loto_auc:.3f}"
              f"  within acc {w_acc:.3f} auc {w_auc:.3f}"
              f"  null p95 {np.percentile(null, 95):.3f}  p={p:.3f}")
    summ = pd.DataFrame(rows)

    # size-matched random-head bands for each CMA set size
    claimed = set().union(*[set(v) for v in sets.values()])
    pool = np.array([h for h in range(d.nh) if h not in claimed])
    band_rows = []
    for k in sorted({len(v) for v in sets.values()}):
        accs = []
        for i in range(args.n_random):
            heads = rng.choice(pool, k, replace=False).tolist()
            X = features(z, heads, d)
            a, u = cv_loto(X, y, template, args.C)
            accs.append((a, u))
        accs = np.array(accs)
        band_rows.append(dict(k=k, n_draws=args.n_random,
                              rand_loto_acc_mean=accs[:, 0].mean(),
                              rand_loto_acc_p5=np.percentile(accs[:, 0], 5),
                              rand_loto_acc_p95=np.percentile(accs[:, 0], 95),
                              rand_loto_auc_mean=accs[:, 1].mean()))
        print(f"  random k={k:3d}: LOTO acc mean {accs[:, 0].mean():.3f} "
              f"[p5 {np.percentile(accs[:, 0], 5):.3f}, p95 {np.percentile(accs[:, 0], 95):.3f}]"
              f"  auc mean {accs[:, 1].mean():.3f}")
    band = pd.DataFrame(band_rows)
    summ = summ.merge(band, left_on="n_heads", right_on="k", how="left").drop(columns="k")
    summ["pct_of_random"] = [
        float((rng_acc := summ.loc[i, "rand_loto_acc_mean"]) and summ.loc[i, "loto_acc"] / rng_acc)
        for i in summ.index]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(args.out_dir / f"summary_{args.target}.csv", index=False)
    band.to_csv(args.out_dir / f"random_band_{args.target}.csv", index=False)
    print("\n=== summary ===")
    print(summ[["head_set", "n_heads", "loto_acc", "loto_auc", "within_auc",
                "null_acc_p95", "p_perm", "rand_loto_acc_mean", "rand_loto_acc_p95"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
