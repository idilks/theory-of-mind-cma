"""Shared-axis steering: is the cross-domain state axis causally potent?

Preregistered in docs/STEERING_PREREG.md, which is NOT in this repo (never committed);
the design and results are recorded in docs/divergence_content_results.md §S
(belief/code) and §SG5 (belief/photo; that harness, steer_photo_axes.py, was
never committed either, so the photo domain is re-implemented here).

Axes per div_unique head (sign convention: non-divergent minus divergent):
    u_tom         ToM true - false
    u_photo       photo true - false      <- agentless frame
    u_cm          code alias - copy      <- ZERO ToM contamination (provenance)
    s             unit(u_tom + u_cm)     <- shared axis
    tom_residue   u_tom orthogonalised against s
    code_residue  u_cm  orthogonalised against s
Controls:
    rand_matched  fresh random unit per (item, head), same magnitude
    axis_shuffled s permuted across heads (derangement), same magnitude

Intervention (additive, final position, div_unique heads):
    z_h <- z_h + alpha * axis_h * ||state_signal||_{h, target domain}

Every axis is tested in every requested domain. Effects are reported ONLY as
effect(axis) - effect(rand_matched) at the same alpha.

Head selection (--selection, see head_sets_source.py):
    top32    fingerprint top-32 C1 minus top-22 C5 (22 heads; the paper's original)
    sig4of6  FWER-significant in >= 4/6 cells, C1 minus C5 (16 heads)
    sig3of6  >= 3/6 cells (22 heads; sensitivity)

Item order is tom, code, photo. rand_matched is seeded on the item's index in
that list, so the tom and code baselines are identical to the original
steering_qwen run and a --selection top32 rerun must reproduce §S exactly.

Outputs (results/rsa_head_outputs/steering_{qwen,gemma}[_{selection}_{domains}]/):
  steer_effects.csv    one row per (domain, item, axis, alpha)
  steer_summary.csv    aggregated differentials vs matched random
  selfcheck.csv        alpha=0 exact-reproduction record
  heads.csv            the steered heads (layer, head, flat) and selection rule
  run_meta.json        arguments, item counts, axis cosines
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
import dims as _dims  # noqa: E402
import head_sets_source as _hss  # noqa: E402
import patch_common as pc  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("steer")

OUT_DIR = pc.ROOT / "results/rsa_head_outputs/steering"
TOM_ACT = pc.ROOT / "results/rsa_head_outputs/activations_full"
PHOTO_ACT = pc.ROOT / "results/rsa_head_outputs/activations_full_photo"
# All-heads code capture. The original path (activations_code_position/
# code_position.pt) now holds a 50-head subset that cannot be indexed by flat
# head id, so steer_photo_axis_on_code.py redirected here; build_axes asserts.
CODE_POS = (pc.ROOT / "results/rsa_head_outputs/activations_code_position_full"
            / "Qwen2.5-14B-Instruct/code_position.pt")
FP_DIR = pc.ROOT / "results/aggregated/fingerprints"
STIM_DIR = pc.ROOT / "results/rsa_head_outputs/stimuli"
STIM_PHOTO_DIR = pc.ROOT / "results/rsa_head_outputs/stimuli_photo"

TOM_SYSTEM_PROMPT = "You are a helpful assistant reasoning about agents' beliefs."
CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that traces through Python code and "
    "determines variable values. Answer with only the Python value, nothing else.")
TEMPLATES = ["food_truck", "hair_styling", "library_book"]
AXES = ["s", "u_cm", "u_tom", "u_photo", "tom_residue", "code_residue"]
CONTROLS = ["rand_matched", "axis_shuffled"]
DOMAINS = ["tom", "code", "photo"]
# Qwen default; resolved from the loaded model in main(). See dims.py.
N_HEADS = 40
MODEL_TYPE = "Qwen2.5-14B-Instruct"
GEMMA_DIR = None
SELECTION = "top32"


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


def _story_capture(act_dir: Path):
    """(z (n, NH, D), label true-veracity (n,)) over the three templates."""
    zs, labs = [], []
    for t in TEMPLATES:
        p = torch.load(act_dir / f"{t}.pt", weights_only=False)
        st = pd.DataFrame(p["stim"])
        zs.append(p["z"].float().numpy().reshape(len(st), -1, 128))
        labs.append(st["belief_veracity"].astype(str).eq("True").to_numpy())
    z, lab = np.concatenate(zs), np.concatenate(labs)
    assert lab.sum() == (~lab).sum(), f"{act_dir}: unbalanced labels {lab.sum()}/{(~lab).sum()}"
    return z, lab


def build_axes():
    """Returns axes dict {name: (NH,D)}, statenorms {domain: (NH,)}, head ids."""
    z_tom, lab_tom = _story_capture(TOM_ACT)
    z_ph, lab_ph = _story_capture(PHOTO_ACT)

    p = torch.load(CODE_POS, weights_only=False)
    st_c = pd.DataFrame(p["stim"])
    z_c = p["z"].float().numpy().reshape(len(st_c), -1, 128)
    lab_c = st_c["semantics"].astype(str).eq("alias").to_numpy()

    _d = _dims.for_model_type(MODEL_TYPE)
    for name, z in (("tom", z_tom), ("photo", z_ph), ("code", z_c)):
        assert z.shape[1] == _d.nh, (
            f"{name} capture has {z.shape[1]} heads, need all {_d.nh} to index by flat id")

    d_tom = z_tom[lab_tom].mean(0) - z_tom[~lab_tom].mean(0)     # (NH, D)
    d_ph = z_ph[lab_ph].mean(0) - z_ph[~lab_ph].mean(0)
    d_cm = z_c[lab_c].mean(0) - z_c[~lab_c].mean(0)
    u_tom, u_ph, u_cm = unit(d_tom), unit(d_ph), unit(d_cm)
    s = unit(u_tom + u_cm)
    tom_res = unit(u_tom - np.einsum("hd,hd->h", u_tom, s)[:, None] * s)
    code_res = unit(u_cm - np.einsum("hd,hd->h", u_cm, s)[:, None] * s)
    axes = {"s": s, "u_cm": u_cm, "u_tom": u_tom, "u_photo": u_ph,
            "tom_residue": tom_res, "code_residue": code_res}
    for k, v in axes.items():
        n = np.linalg.norm(v, axis=-1)
        assert np.allclose(n, 1.0, atol=1e-4), f"{k} not unit: {n.min()}-{n.max()}"
    statenorm = {"tom": np.linalg.norm(d_tom, axis=-1),
                 "photo": np.linalg.norm(d_ph, axis=-1),
                 "code": np.linalg.norm(d_cm, axis=-1)}

    heads = _hss.build(MODEL_TYPE, _d, FP_DIR, gemma_dir=GEMMA_DIR,
                       selection=SELECTION)["div_unique"]
    return axes, statenorm, heads


def _story_items(tokenizer, stim_dir: Path, domain: str, per_class_per_template=12):
    frames = [pd.read_csv(f) for f in sorted(glob.glob(str(stim_dir / "*.csv")))]
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
            out.append({"domain": domain, "item_id": f"{t}:{r['prompt_id']}",
                        "cls": "true" if ver == "True" else "false",
                        "divergent": ver != "True",
                        "prompt": chat(r["scenario"], r["question"], tokenizer,
                                       TOM_SYSTEM_PROMPT),
                        "id_ans": int(fa), "id_flip": int(fo)})
    return out


def tom_items(tokenizer, per_class_per_template=12):
    return _story_items(tokenizer, STIM_DIR, "tom", per_class_per_template)


def photo_items(tokenizer, per_class_per_template=12):
    """Photo-frame counterpart: same template/prompt_id sampling on stimuli_photo.

    The photo capture's prompts (activations_full_photo) were run with the same
    belief system prompt as the belief capture, so the steering prompts keep it.
    """
    items = _story_items(tokenizer, STIM_PHOTO_DIR, "photo", per_class_per_template)
    return [{**it, "item_id": f"photo:{it['item_id']}"} for it in items]


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


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for i in range(n):                 # force derangement
        if perm[i] == i:
            j = (i + 1) % n
            perm[i], perm[j] = perm[j], perm[i]
    assert not any(perm[i] == i for i in range(n)), "derangement failed"
    return perm


def out_dir_for(model_type: str, selection: str, domains: list[str], limit: int) -> Path:
    """Legacy name for the legacy invocation; otherwise suffix rule and domains."""
    name = f"{OUT_DIR.name}_{_dims.tag(model_type)}"
    if selection != "top32" or domains != ["tom", "code"]:
        name += f"_{selection}_{'-'.join(domains)}"
    if limit:
        name += "_smoke"
    return OUT_DIR.with_name(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1.0, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--gemma_dir", type=Path, default=None)
    ap.add_argument("--selection", default="top32", choices=_hss.SELECTIONS,
                    help="head-selection rule (head_sets_source.py)")
    ap.add_argument("--domains", nargs="+", default=["tom", "code"], choices=DOMAINS,
                    help="target domains to steer; item order is always tom, code, photo")
    ap.add_argument("--check_inputs", action="store_true",
                    help="build axes and head set, print geometry and exit, without the model")
    args = ap.parse_args()
    # Must precede build_axes()/head-set construction: those read MODEL_TYPE.
    global N_HEADS, MODEL_TYPE, GEMMA_DIR, SELECTION
    MODEL_TYPE, GEMMA_DIR, SELECTION = args.model_type, args.gemma_dir, args.selection
    N_HEADS = _dims.for_model_type(args.model_type).n_heads
    assert max(abs(a) for a in args.alphas) <= 2.0, \
        "escalation prohibition: |alpha| > 2 is not permitted (see prereg)"
    domains = [d for d in DOMAINS if d in args.domains]

    out_dir = out_dir_for(args.model_type, args.selection, domains, args.limit)

    axes, statenorm, heads = build_axes()
    log.info(f"selection {args.selection}: {len(heads)} div_unique heads -> {out_dir.name}")
    h = np.array(heads)
    cosines = {}
    for a, b in (("u_tom", "u_cm"), ("u_tom", "u_photo"), ("u_photo", "u_cm")):
        c = np.einsum("hd,hd->h", axes[a][h], axes[b][h])
        cosines[f"cos({a},{b})"] = {"mean": float(c.mean()), "min": float(c.min()),
                                    "max": float(c.max())}
        log.info(f"cos({a}, {b}) over heads: mean {c.mean():+.3f}, "
                 f"range [{c.min():+.3f}, {c.max():+.3f}]")
    if args.check_inputs:
        print(pd.DataFrame({"layer": h // N_HEADS, "head": h % N_HEADS, "flat": h}).to_string(index=False))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layer": h // N_HEADS, "head": h % N_HEADS, "flat": h,
                  "selection": args.selection}).to_csv(out_dir / "heads.csv", index=False)

    perm = derangement(len(heads), args.seed)
    axes_shuf = {"axis_shuffled": axes["s"][h][perm]}

    model, tokenizer = pc.load_model(args.device, args.model_type)
    # Fixed order so rand_matched seeds (indexed by position) match the
    # original steering_qwen run for tom and code items.
    all_items = tom_items(tokenizer) + code_items(tokenizer) + photo_items(tokenizer)
    items = [(ii, it) for ii, it in enumerate(all_items) if it["domain"] in domains]
    if args.limit:
        kept = []
        for dom in domains:
            kept += [x for x in items if x[1]["domain"] == dom][:args.limit]
        items = kept
    counts = {dom: sum(1 for _, x in items if x["domain"] == dom) for dom in domains}
    log.info(f"{len(items)} items {counts}")

    hid = [(x // N_HEADS, x % N_HEADS) for x in heads]
    rows, checks = [], []
    seen = {dom: 0 for dom in domains}
    t0 = time.time()
    for done, (ii, it) in enumerate(items):
        tok = pc.encode(tokenizer, it["prompt"], args.device)
        clean = pc.run_clean(model, tok)[1]
        clean_ld = float(clean[it["id_flip"]] - clean[it["id_ans"]])
        sn = statenorm[it["domain"]]

        if seen[it["domain"]] < 8:
            seen[it["domain"]] += 1
            zero = steered_logits(model, tok, {lh: torch.zeros(128) for lh in hid})
            checks.append({"item_id": it["item_id"], "domain": it["domain"],
                           "max_abs_diff": float((zero - clean).abs().max()),
                           "exact": bool(torch.equal(zero, clean))})

        for alpha in args.alphas:
            for name in AXES + CONTROLS:
                deltas = {}
                for k, (L, H) in enumerate(hid):
                    hh = heads[k]
                    if name == "rand_matched":
                        r = torch.randn(128, generator=torch.Generator()
                                        .manual_seed(args.seed * 7919 + ii * 131 + k))
                        vec = (r / r.norm()).numpy()
                    elif name == "axis_shuffled":
                        vec = axes_shuf["axis_shuffled"][k]
                    else:
                        vec = axes[name][hh]
                    deltas[(L, H)] = torch.tensor(
                        np.asarray(vec) * alpha * float(sn[hh]),
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
        if (done + 1) % 10 == 0 or done == len(items) - 1:
            el = time.time() - t0
            log.info(f"{done+1}/{len(items)} ({el/(done+1):.1f}s/item, "
                     f"~{el/(done+1)*(len(items)-done-1)/60:.0f} min left)")

    df = pd.DataFrame(rows)
    ck = pd.DataFrame(checks)
    df.to_csv(out_dir / "steer_effects.csv", index=False)
    ck.to_csv(out_dir / "selfcheck.csv", index=False)
    (out_dir / "run_meta.json").write_text(json.dumps(
        {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
         "domains": domains, "n_items": counts, "n_heads": len(heads),
         "heads_flat": [int(x) for x in heads], "axis_cosines": cosines,
         "seconds": time.time() - t0}, indent=1))

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
    summ.to_csv(out_dir / "steer_summary.csv", index=False)
    print("\n=== differential vs matched random, by domain x axis x signed_alpha ===")
    print(summ.round(3).to_string(index=False))
    print(f"\nwrote -> {out_dir}")


if __name__ == "__main__":
    main()
