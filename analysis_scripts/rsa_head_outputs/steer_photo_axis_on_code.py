"""Photo-axis steering on code prompts: the one missing cell of paper Fig. 5c.

Same intervention as steer_state_axes.py (additive at the final position, on the
22 div_unique heads, scaled by each head's state-signal norm in the code frame).
Adds u_photo, the per-head unit direction photo true - false, and reruns u_cm,
u_tom, axis_shuffled and rand_matched on the same 96 code items with the same
seeds. The reruns must reproduce steering_qwen/steer_summary.csv, which is what
licenses putting the new cell beside the old ones.

CODE_POS is redirected to the all-heads capture: the path steer_state_axes.py
names now holds a 50-head subset, which it cannot index by flat head id.

Outputs (results/rsa_head_outputs/steering_code_photo_qwen/):
  steer_effects.csv    one row per (item, axis, alpha)
  steer_summary.csv    differentials vs matched random, same columns as §S
  selfcheck.csv        alpha=0 exact-reproduction record
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_common as pc  # noqa: E402
import steer_state_axes as sa  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("steer_code_photo")

RSA = pc.ROOT / "results/rsa_head_outputs"
OUT_DIR = RSA / "steering_code_photo_qwen"
PHOTO_ACT = RSA / "activations_full_photo"
CODE_FULL = RSA / "activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt"
MODEL_TYPE = "Qwen2.5-14B-Instruct"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]
AXES = ["u_photo", "u_cm", "u_tom"]

# steering_qwen/steer_summary.csv, code domain: (axis, signed_alpha) -> mean_diff
EXPECTED = {("u_cm", 1.0): 42.895, ("u_cm", -1.0): -2.040,
            ("u_tom", 1.0): 14.995, ("u_tom", -1.0): -0.279,
            ("axis_shuffled", 1.0): -0.010, ("axis_shuffled", -1.0): -0.190}
TOLERANCE = 0.05


def photo_axis():
    """Per-head unit direction photo true - false, shape (n_heads_total, D)."""
    zs, labs = [], []
    for t in TEMPLATES:
        p = torch.load(PHOTO_ACT / f"{t}.pt", weights_only=False)
        st = pd.DataFrame(p["stim"])
        zs.append(p["z"].float().numpy().reshape(len(st), -1, 128))
        labs.append(st["belief_veracity"].astype(str).eq("True").to_numpy())
    z, lab = np.concatenate(zs), np.concatenate(labs)
    assert lab.sum() == (~lab).sum(), f"unbalanced photo labels: {lab.sum()}/{(~lab).sum()}"
    return sa.unit(z[lab].mean(0) - z[~lab].mean(0))


def shuffled_axis(s, heads, seed):
    """The derangement of s across heads, drawn as in steer_state_axes.main."""
    perm = np.random.default_rng(seed).permutation(len(heads))
    for i in range(len(heads)):
        if perm[i] == i:
            j = (i + 1) % len(heads)
            perm[i], perm[j] = perm[j], perm[i]
    assert not any(perm[i] == i for i in range(len(heads))), "derangement failed"
    return s[np.array(heads)][perm]


def load_inputs(seed):
    for path in (CODE_FULL, sa.TOM_ACT, PHOTO_ACT, sa.STIM_DIR, sa.FP_DIR):
        assert path.exists(), f"missing input: {path}"
    sa.CODE_POS = CODE_FULL
    sa.MODEL_TYPE = MODEL_TYPE
    axes, statenorm, heads = sa.build_axes()
    axes["u_photo"] = photo_axis()
    assert axes["u_photo"].shape == axes["u_cm"].shape, axes["u_photo"].shape
    assert len(heads) == 22, f"expected 22 div_unique heads, got {len(heads)}"
    return axes, statenorm["code"], heads, shuffled_axis(axes["s"], heads, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N code items")
    ap.add_argument("--check_inputs", action="store_true",
                    help="build the axes, print their geometry and exit, without the model")
    args = ap.parse_args()

    axes, statenorm, heads, shuffled = load_inputs(args.seed)
    h = np.array(heads)
    log.info(f"div_unique heads: {len(heads)}")
    for name in ("u_tom", "u_cm"):
        cos = np.einsum("hd,hd->h", axes["u_photo"][h], axes[name][h])
        log.info(f"cos(u_photo, {name}) over heads: mean {cos.mean():+.3f}, "
                 f"range [{cos.min():+.3f}, {cos.max():+.3f}]")
    if args.check_inputs:
        return

    out_dir = OUT_DIR.with_name(OUT_DIR.name + "_smoke") if args.limit else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = pc.load_model(args.device, MODEL_TYPE)
    # Keep the ToM items in the list: rand_matched is seeded on the item's index
    # in the full tom + code list, so dropping them would change every baseline.
    items = [(ii, it) for ii, it in enumerate(sa.tom_items(tokenizer) + sa.code_items(tokenizer))
             if it["domain"] == "code"]
    if args.limit:
        items = items[:args.limit]
    log.info(f"{len(items)} code items")

    hid = [(x // sa.N_HEADS, x % sa.N_HEADS) for x in heads]
    rows, checks = [], []
    t0 = time.time()
    for done, (ii, it) in enumerate(items):
        tok = pc.encode(tokenizer, it["prompt"], args.device)
        clean = pc.run_clean(model, tok)[1]
        clean_ld = float(clean[it["id_flip"]] - clean[it["id_ans"]])

        if done < 8:
            zero = sa.steered_logits(model, tok, {lh: torch.zeros(128) for lh in hid})
            checks.append({"item_id": it["item_id"],
                           "max_abs_diff": float((zero - clean).abs().max()),
                           "exact": bool(torch.equal(zero, clean))})

        for alpha in (-1.0, 1.0):
            for name in AXES + sa.CONTROLS:
                deltas = {}
                for k, lh in enumerate(hid):
                    if name == "rand_matched":
                        r = torch.randn(128, generator=torch.Generator()
                                        .manual_seed(args.seed * 7919 + ii * 131 + k))
                        vec = (r / r.norm()).numpy()
                    elif name == "axis_shuffled":
                        vec = shuffled[k]
                    else:
                        vec = axes[name][heads[k]]
                    deltas[lh] = torch.tensor(np.asarray(vec) * alpha * float(statenorm[heads[k]]),
                                              dtype=torch.float32)
                pl = sa.steered_logits(model, tok, deltas)
                rows.append({"domain": it["domain"], "item_id": it["item_id"],
                             "cls": it["cls"], "divergent": it["divergent"],
                             "axis": name, "alpha": alpha,
                             "signed_alpha": alpha * (1 if it["divergent"] else -1),
                             "effect": float((pl[it["id_flip"]] - pl[it["id_ans"]]) - clean_ld)})
        if (done + 1) % 5 == 0 or done == len(items) - 1:
            el = time.time() - t0
            log.info(f"{done+1}/{len(items)} ({el/(done+1):.1f}s/item, "
                     f"~{el/(done+1)*(len(items)-done-1)/60:.0f} min left)")

    df = pd.DataFrame(rows)
    ck = pd.DataFrame(checks)
    df.to_csv(out_dir / "steer_effects.csv", index=False)
    ck.to_csv(out_dir / "selfcheck.csv", index=False)

    base = (df[df.axis == "rand_matched"][["item_id", "alpha", "effect"]]
            .rename(columns={"effect": "rand_effect"}))
    df = df.merge(base, on=["item_id", "alpha"], how="left")
    assert df.rand_effect.notna().all(), "missing matched-random baseline"
    df["diff_vs_rand"] = df["effect"] - df["rand_effect"]
    summ = (df[df.axis != "rand_matched"]
            .groupby(["domain", "axis", "signed_alpha"])
            .agg(mean_diff=("diff_vs_rand", "mean"),
                 sem=("diff_vs_rand", "sem"),
                 n=("diff_vs_rand", "count")).reset_index())
    summ.to_csv(out_dir / "steer_summary.csv", index=False)

    print(f"\nalpha=0 self-check: all exact = {bool(ck.exact.all())}, "
          f"max |diff| = {ck.max_abs_diff.max():.3e}")
    print("\n=== differential vs matched random ===")
    print(summ.round(3).to_string(index=False))
    if args.limit:
        print("\nsmoke run: reproduction check skipped (needs all 96 items)")
    else:
        got = summ.set_index(["axis", "signed_alpha"]).mean_diff
        worst = max(abs(got[k] - v) for k, v in EXPECTED.items())
        print(f"\nreproduction of steering_qwen, worst |diff| = {worst:.3f} "
              f"-> {'OK' if worst <= TOLERANCE else '*** MISMATCH: do not use u_photo ***'}")
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
