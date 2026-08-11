"""Shared-axis steering: is the cross-domain state axis causally potent?

Preregistered in docs/STEERING_PREREG.md. Read that first.

Axes per div_unique head (sign convention: non-divergent minus divergent):
    u_tom         ToM true - false
    u_cm          code alias - copy      <- ZERO ToM contamination (provenance)
    s             unit(u_tom + u_cm)     <- shared axis
    tom_residue   u_tom orthogonalised against s
    code_residue  u_cm  orthogonalised against s
Controls:
    rand_matched  fresh random unit per (item, head), same magnitude
    axis_shuffled s permuted across heads (derangement), same magnitude

Intervention (additive, final position, div_unique heads):
    z_h <- z_h + alpha * axis_h * ||state_signal||_{h, target domain}

Every axis is tested in BOTH domains. Effects are reported ONLY as
effect(axis) - effect(rand_matched) at the same alpha.

Outputs (results/rsa_head_outputs/steering/):
  steer_effects.csv    one row per (domain, item, axis, alpha)
  steer_summary.csv    aggregated differentials vs matched random
  selfcheck.csv        alpha=0 exact-reproduction record
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patch_common as pc  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("steer")

OUT_DIR = pc.ROOT / "results/rsa_head_outputs/steering"
TOM_ACT = pc.ROOT / "results/rsa_head_outputs/activations_full"
CODE_POS = pc.ROOT / "results/rsa_head_outputs/activations_code_position/code_position.pt"
FP_DIR = pc.ROOT / "results/aggregated/fingerprints"
STIM_DIR = pc.ROOT / "results/rsa_head_outputs/stimuli"

TOM_SYSTEM_PROMPT = "You are a helpful assistant reasoning about agents' beliefs."
CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that traces through Python code and "
    "determines variable values. Answer with only the Python value, nothing else.")
AXES = ["s", "u_cm", "u_tom", "tom_residue", "code_residue"]
CONTROLS = ["rand_matched", "axis_shuffled"]
# Qwen default; resolved from the loaded model in main(). See dims.py.
N_HEADS = 40
MODEL_TYPE = "Qwen2.5-14B-Instruct"
GEMMA_DIR = None


def unit(v, axis=-1):
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def chat(scenario, question, tokenizer, system):
    import chat_compat
    return chat_compat.chat_prompt(tokenizer, system,
                                   f"{scenario}\n{question}")


def steered_logits(model, tokens, deltas):
    """Forward pass with hook_z[(L,H)] at the final position INCREMENTED."""
    by_layer = {}
    for (L, H), v in deltas.items():
        by_layer.setdefault(L, {})[H] = v

    def make_hook(subs):
        def hook(value, hook=None):
            for H, v in subs.items():
                value[:, -1, H, :] = value[:, -1, H, :] + v.to(value.dtype).to(value.device)
            return value
        return hook

    fwd = [(f"blocks.{L}.attn.hook_z", make_hook(sub))
           for L, sub in by_layer.items()]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd)
    return logits[0, -1].detach().float().cpu()


def build_axes():
    """Returns axes dict {name: (NH,D)}, statenorms {domain: (NH,)}, head ids."""
    zs, labs = [], []
    for t in ["food_truck", "hair_styling", "library_book"]:
        p = torch.load(TOM_ACT / f"{t}.pt", weights_only=False)
        st = pd.DataFrame(p["stim"])
        zs.append(p["z"].float().numpy().reshape(len(st), -1, 128))
        labs.append(st["belief_veracity"].astype(str).eq("True").to_numpy())
    z_tom = np.concatenate(zs); lab_tom = np.concatenate(labs)

    p = torch.load(CODE_POS, weights_only=False)
    st_c = pd.DataFrame(p["stim"])
    z_c = p["z"].float().numpy().reshape(len(st_c), -1, 128)
    lab_c = st_c["semantics"].astype(str).eq("alias").to_numpy()

    d_tom = z_tom[lab_tom].mean(0) - z_tom[~lab_tom].mean(0)     # (NH, D)
    d_cm = z_c[lab_c].mean(0) - z_c[~lab_c].mean(0)
    u_tom, u_cm = unit(d_tom), unit(d_cm)
    s = unit(u_tom + u_cm)
    tom_res = unit(u_tom - np.einsum("hd,hd->h", u_tom, s)[:, None] * s)
    code_res = unit(u_cm - np.einsum("hd,hd->h", u_cm, s)[:, None] * s)
    axes = {"s": s, "u_cm": u_cm, "u_tom": u_tom,
            "tom_residue": tom_res, "code_residue": code_res}
    for k, v in axes.items():
        n = np.linalg.norm(v, axis=-1)
        assert np.allclose(n, 1.0, atol=1e-4), f"{k} not unit: {n.min()}-{n.max()}"
    statenorm = {"tom": np.linalg.norm(d_tom, axis=-1),
                 "code": np.linalg.norm(d_cm, axis=-1)}

    import dims as _dims
    import head_sets_source as _hss
    _d = _dims.for_model_type(MODEL_TYPE)
    heads = _hss.build(MODEL_TYPE, _d, FP_DIR,
                       gemma_dir=GEMMA_DIR)["div_unique"]
    return axes, statenorm, heads


def tom_items(tokenizer, per_class_per_template=12):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(STIM_DIR / "*.csv")))]
    d = pd.concat(frames, ignore_index=True)
    out = []
    for (t, ver), g in d.groupby(["template", d.belief_veracity.astype(str)]):
        g = g.sort_values("prompt_id").head(per_class_per_template)
        for _, r in g.iterrows():
            ans = r["answer_state"]
            other = r["loc_b"] if ans == r["loc_a"] else r["loc_a"]
            fa = tokenizer.encode(" " + ans, add_special_tokens=False)[0]
            fo = tokenizer.encode(" " + other, add_special_tokens=False)[0]
            if fa == fo:
                continue
            out.append({"domain": "tom", "item_id": f"{t}:{r['prompt_id']}",
                        "cls": "true" if ver == "True" else "false",
                        "divergent": ver != "True",
                        "prompt": chat(r["scenario"], r["question"], tokenizer,
                                       TOM_SYSTEM_PROMPT),
                        "id_ans": int(fa), "id_flip": int(fo)})
    return out


def code_items(tokenizer, per_class=48):
    p = torch.load(CODE_POS, weights_only=False)
    st = pd.DataFrame(p["stim"])
    out = []
    for sem, g in st.groupby("semantics"):
        g = g.sort_values("prompt_id").head(per_class)
        for _, r in g.iterrows():
            ans_ch = str(r["answer"]).strip("'")
            flip_ch = r["mut_char"] if sem == "copy" else r["orig_char"]
            ia, ifl = pc.probe_id(tokenizer, ans_ch), pc.probe_id(tokenizer, flip_ch)
            if ia == ifl:
                continue
            out.append({"domain": "code", "item_id": f"code:{r['prompt_id']}",
                        "cls": sem, "divergent": sem == "copy",
                        "prompt": chat(r["scenario"], r["question"], tokenizer,
                                       CODE_SYSTEM_PROMPT),
                        "id_ans": int(ia), "id_flip": int(ifl)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--gemma_dir", type=Path, default=None)
    args = ap.parse_args()
    # Must precede build_axes()/head-set construction: those read MODEL_TYPE.
    global N_HEADS, MODEL_TYPE, GEMMA_DIR
    MODEL_TYPE, GEMMA_DIR = args.model_type, args.gemma_dir
    import dims as _dims
    N_HEADS = _dims.for_model_type(args.model_type).n_heads
    assert max(abs(a) for a in args.alphas) <= 2.0, \
        "escalation prohibition: |alpha| > 2 is not permitted (see prereg)"

    global OUT_DIR
    if args.limit:
        OUT_DIR = OUT_DIR.with_name(OUT_DIR.name + "_smoke")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    axes, statenorm, heads = build_axes()
    log.info(f"div_unique heads: {len(heads)}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(heads))
    for i in range(len(heads)):                 # force derangement
        if perm[i] == i:
            j = (i + 1) % len(heads)
            perm[i], perm[j] = perm[j], perm[i]
    assert not any(perm[i] == i for i in range(len(heads))), "derangement failed"
    axes_shuf = {"axis_shuffled": axes["s"][np.array(heads)][perm]}

    model, tokenizer = pc.load_model(args.device, args.model_type)
    items = tom_items(tokenizer) + code_items(tokenizer)
    if args.limit:
        t = [x for x in items if x["domain"] == "tom"][:args.limit]
        c = [x for x in items if x["domain"] == "code"][:args.limit]
        items = t + c
    log.info(f"{len(items)} items "
             f"({sum(1 for x in items if x['domain']=='tom')} tom / "
             f"{sum(1 for x in items if x['domain']=='code')} code)")

    hid = [(h // N_HEADS, h % N_HEADS) for h in heads]
    rows, checks = [], []
    t0 = time.time()
    for ii, it in enumerate(items):
        tok = pc.encode(tokenizer, it["prompt"], args.device)
        clean = pc.run_clean(model, tok)[1]
        clean_ld = float(clean[it["id_flip"]] - clean[it["id_ans"]])
        sn = statenorm[it["domain"]]

        if ii < 8 or (it["domain"] == "code" and ii < 8 + 8):
            zero = steered_logits(model, tok, {lh: torch.zeros(128)
                                               for lh in hid})
            checks.append({"item_id": it["item_id"], "domain": it["domain"],
                           "max_abs_diff": float((zero - clean).abs().max()),
                           "exact": bool(torch.equal(zero, clean))})

        for alpha in args.alphas:
            for name in AXES + CONTROLS:
                deltas = {}
                for k, (L, H) in enumerate(hid):
                    h = heads[k]
                    if name == "rand_matched":
                        r = torch.randn(128, generator=torch.Generator()
                                        .manual_seed(args.seed * 7919 + ii * 131 + k))
                        vec = (r / r.norm()).numpy()
                    elif name == "axis_shuffled":
                        vec = axes_shuf["axis_shuffled"][k]
                    else:
                        vec = axes[name][h]
                    deltas[(L, H)] = torch.tensor(
                        np.asarray(vec) * alpha * float(sn[h]),
                        dtype=torch.float32)
                pl = steered_logits(model, tok, deltas)
                eff = float((pl[it["id_flip"]] - pl[it["id_ans"]]) - clean_ld)
                rows.append({"domain": it["domain"], "item_id": it["item_id"],
                             "cls": it["cls"], "divergent": it["divergent"],
                             "axis": name, "alpha": alpha,
                             "signed_alpha": alpha * (1 if it["divergent"] else -1),
                             "effect": eff})
        if args.device == "mps":
            torch.mps.empty_cache()
        if (ii + 1) % 10 == 0 or ii == len(items) - 1:
            el = time.time() - t0
            log.info(f"{ii+1}/{len(items)} ({el/(ii+1):.1f}s/item, "
                     f"~{el/(ii+1)*(len(items)-ii-1)/60:.0f} min left)")

    df = pd.DataFrame(rows)
    ck = pd.DataFrame(checks)
    df.to_csv(OUT_DIR / "steer_effects.csv", index=False)
    ck.to_csv(OUT_DIR / "selfcheck.csv", index=False)

    print("\n=== alpha=0 self-check (must be exact) ===")
    print(f"  items checked: {len(ck)}   all exact: {bool(ck.exact.all())}   "
          f"max |diff|: {ck.max_abs_diff.max():.3e}")
    if not ck.exact.all():
        print("  *** SELF-CHECK FAILED — steering hook is not identity at alpha=0")

    base = (df[df.axis == "rand_matched"]
            [["domain", "item_id", "alpha", "effect"]]
            .rename(columns={"effect": "rand_effect"}))
    df = df.merge(base, on=["domain", "item_id", "alpha"], how="left")
    assert df.rand_effect.notna().all(), "missing matched-random baseline"
    df["diff_vs_rand"] = df["effect"] - df["rand_effect"]
    summ = (df[df.axis != "rand_matched"]
            .groupby(["domain", "axis", "signed_alpha"])
            .agg(mean_diff=("diff_vs_rand", "mean"),
                 sem=("diff_vs_rand", "sem"),
                 n=("diff_vs_rand", "count")).reset_index())
    summ.to_csv(OUT_DIR / "steer_summary.csv", index=False)
    print("\n=== differential vs matched random, by domain x axis x signed_alpha ===")
    print(summ.round(3).to_string(index=False))
    print(f"\nwrote -> {OUT_DIR}")


if __name__ == "__main__":
    main()
