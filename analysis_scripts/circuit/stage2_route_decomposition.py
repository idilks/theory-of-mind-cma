"""Which route carries the residual-steering effect? Exact three-route
accounting at the final token.

After the edit at resid_pre L42 (= resid_post L41), the final residual is

    resid_final = resid_pre42 + sum_{l=42..47} attn_out_l + sum_{l=42..47} mlp_out_l

at the final token. Three "routes" therefore carry the edit to the
logits: DIRECT (the edited resid_pre42 itself), ATTN (the six
attention-layer outputs) and MLP (the six MLP outputs). Pinning each
route to its recorded clean or steered final-token value gives 2^3 = 8
exact configurations (the sum above is an identity, so a pinned
configuration's final residual is determined without any recomputation
downstream). From the 8 margins the Shapley share of each route is exact
and needs no additivity assumption.

Runs per prompt (margin = logit(alt) - logit(correct)):
  fac_{d}{a}{m}      d,a,m in {c,s}: direct/attn/mlp pinned to clean or
                     steered. fac_ccc == clean, fac_sss == steered.
  pattern_all        steer on, ALL heads' final-token attention rows in
                     L42-47 <- clean (the attention-WEIGHT route alone)
  pattern_all_tr     steer off, those rows <- steered
  attn_L{l}          steer on, attn_out_l <- clean (per-layer necessity;
                     downstream recomputes, so not pinned-exact)
  mlp_L{l}           steer on, mlp_out_l <- clean (same)

Readouts: margin, flip score, KL from clean, reader attention mass
(believed/current, 26 readers) in attention.csv. The final residual is
recorded for fac_ccc, fac_sss and fac_scc and the exactness
resid_final(fac_scc) == resid_final(clean) + (resid_pre42(steered) -
resid_pre42(clean)) is checked per prompt (exact.csv). Same prompts,
direction, dose and readers as stage2_reader_restoration.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "circuit"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from circuit_common import (  # noqa: E402
    CAPTURE_ROOT, DOMAINS, INJECTION_HOOKS, READER_DLA_LAYERS, SITE_KEYS,
    STAGE1_OUT, STAGE2_OUT_ROOT, TEMPLATES,
)
from stage2_steering import (  # noqa: E402
    heldout_capture_prompts, location_token_mask, make_steer_fn,
)
from stage2_reader_restoration import (  # noqa: E402
    attention_mass, make_restore_hooks, readouts,
)
from tasks.causal_analysis.ablation_experiment import compute_answer_token_id  # noqa: E402
from tasks.causal_analysis.cma_config import (  # noqa: E402
    EvaluationConfig, ExperimentConfig, GenerationConfig, ModelConfig,
    PatchingConfig, PromptConfig, WandbConfig)
from tasks.causal_analysis.models import ModelLoader  # noqa: E402

ROUTES = ["direct", "attn", "mlp"]
LAYER_HOOK = {"attn": "hook_attn_out", "mlp": "hook_mlp_out"}

log = logging.getLogger("route-decomp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# hooks: layer-level final-token record / pin
# ---------------------------------------------------------------------------

def record_layer_hooks(layers: list[int], store: dict, extra: list[str] = ()):
    """Record final-token attn_out/mlp_out of `layers` (and any extra full
    hook names, e.g. resid_pre / resid_post) into store[name]."""
    names = [f"blocks.{L}.{LAYER_HOOK[r]}" for L in layers for r in LAYER_HOOK]
    names += list(extra)
    hooks = []
    for name in names:
        def rec(value, hook, name=name):
            store[name] = value[0, -1].detach().clone()
            return value
        hooks.append((name, rec))
    return hooks


def pin_hooks(names: list[str], source: dict, errors: list[float]):
    """Set the final-token activation at each hook name to source[name]."""
    hooks = []
    for name in names:
        def pin(value, hook, name=name):
            target = source[name].to(value.device, value.dtype)
            value = value.clone()
            value[0, -1] = target
            errors.append(float((value[0, -1].float() - target.float()).abs().max()))
            return value
        hooks.append((name, pin))
    return hooks


def route_names(route: str, layers: list[int]) -> list[str]:
    return [f"blocks.{L}.{LAYER_HOOK[route]}" for L in layers]


def forward(model, tokens, hooks, record_layers, reader_layers, extra=()):
    store: dict = {}
    pat: dict = {}
    all_hooks = list(hooks) + record_layer_hooks(record_layers, store, extra)
    for L in reader_layers:
        def rec_pat(value, hook, L=L):
            pat[(L, "pattern")] = value[0, :, -1, :].detach().clone()
            return value
        all_hooks.append((f"blocks.{L}.attn.hook_pattern", rec_pat))
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
    return logits[0, -1].detach().float().cpu(), store, pat


# ---------------------------------------------------------------------------
# per-prompt driver
# ---------------------------------------------------------------------------

def run_prompt(model, tokens, steer_hook, layers, readers, n_heads, cid, aid,
               m_bel, m_cur, site_hook: str, final_hook: str):
    reader_layers = sorted({L for L, _ in readers})
    rows, attn_rows = [], []
    extra = [site_hook, final_hook]

    def add(run, logits, pat, clean_logits, errors):
        rows.append(dict(run=run, pin_max_err=(max(errors) if errors else float("nan")),
                         **readouts(logits, clean_logits, cid, aid)))
        for r in attention_mass(pat, readers, m_bel, m_cur):
            attn_rows.append(dict(run=run, **r))

    clean_lg, clean_st, clean_pat = forward(model, tokens, [], layers, reader_layers, extra)
    steer_lg, steer_st, steer_pat = forward(model, tokens, [steer_hook], layers, reader_layers, extra)
    add("fac_ccc", clean_lg, clean_pat, None, [])
    add("fac_sss", steer_lg, steer_pat, clean_lg, [])
    exact: dict = {}
    delta = steer_st[site_hook].double() - clean_st[site_hook].double()
    exact["delta_norm"] = float(delta.norm())
    exact["resid_final_shift_norm"] = float((steer_st[final_hook].double()
                                             - clean_st[final_hook].double()).norm())

    # factorial: pin each route to clean (c) or steered (s)
    for d, a, m in itertools.product("cs", repeat=3):
        name = f"fac_{d}{a}{m}"
        if name in ("fac_ccc", "fac_sss"):
            continue
        err: list[float] = []
        hooks = [steer_hook] if d == "s" else []
        src_a = steer_st if a == "s" else clean_st
        src_m = steer_st if m == "s" else clean_st
        hooks += pin_hooks(route_names("attn", layers), src_a, err)
        hooks += pin_hooks(route_names("mlp", layers), src_m, err)
        lg, st, pat = forward(model, tokens, hooks, layers, reader_layers, extra)
        add(name, lg, pat, clean_lg, err)
        if name == "fac_scc":
            expected = clean_st[final_hook].double() + delta
            got = st[final_hook].double()
            exact["scc_max_abs_err"] = float((got - expected).abs().max())
            exact["scc_rel_err"] = float((got - expected).norm() / expected.norm())

    # attention-weight route: all heads' final-token rows in these layers
    all_heads = [(L, h) for L in layers for h in range(n_heads)]
    # need all layers' patterns recorded, not only reader layers
    _, _, clean_pat_all = forward(model, tokens, [], layers, layers, extra)
    _, _, steer_pat_all = forward(model, tokens, [steer_hook], layers, layers, extra)
    err = []
    lg, _, pat = forward(model, tokens,
                         [steer_hook] + make_restore_hooks("pattern", all_heads, clean_pat_all,
                                                           n_heads, n_heads, err),
                         layers, reader_layers, extra)
    add("pattern_all", lg, pat, clean_lg, err)
    err = []
    lg, _, pat = forward(model, tokens,
                         make_restore_hooks("pattern", all_heads, steer_pat_all,
                                            n_heads, n_heads, err),
                         layers, reader_layers, extra)
    add("pattern_all_tr", lg, pat, clean_lg, err)

    # per-layer necessity (not pinned-exact: downstream recomputes)
    for L in layers:
        for route in ("attn", "mlp"):
            err = []
            lg, _, pat = forward(model, tokens,
                                 [steer_hook] + pin_hooks(route_names(route, [L]), clean_st, err),
                                 layers, reader_layers, extra)
            add(f"{route}_L{L}", lg, pat, clean_lg, err)
    return rows, attn_rows, exact


def expected_runs(layers: list[int]) -> set[str]:
    runs = {f"fac_{d}{a}{m}" for d, a, m in itertools.product("cs", repeat=3)}
    runs |= {"pattern_all", "pattern_all_tr"}
    runs |= {f"{r}_L{L}" for L in layers for r in ("attn", "mlp")}
    return runs


def load_done(per_path: Path, runs: set[str]) -> set[tuple]:
    if not per_path.exists():
        return set()
    df = pd.read_csv(per_path)
    key = ["prompt_set", "template", "pair_idx"]
    have = df.groupby(key)["run"].apply(set)
    complete = {k for k, s in have.items() if s >= runs}
    if len(complete) < len(have):
        keep = df[[tuple(r) in complete for r in df[key].itertuples(index=False)]]
        keep.to_csv(per_path, index=False)
        log.info(f"dropped {len(have) - len(complete)} incomplete prompts")
    return complete


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=list(DOMAINS), default="tom")
    ap.add_argument("--site", choices=SITE_KEYS, default="L41")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--direction", choices=["target_rank1", "target_topk"],
                    default="target_rank1")
    ap.add_argument("--dose", type=float, default=4.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run_label", default="route_decomp")
    ap.add_argument("--capture_dir", type=Path, default=None)
    ap.add_argument("--directions", type=Path, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="cuda")
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    FBD, TBD = domain["variants"]
    tom_root = CAPTURE_ROOT / args.model_type
    capture_dir = args.capture_dir or (tom_root / domain["capture_subdir"])
    weights_path = tom_root / "weights.pt"
    stage1_dir = (STAGE1_OUT if args.domain == "tom"
                  else STAGE1_OUT.parent / "stage1_code")
    dirs_path = args.directions or (stage1_dir / f"directions_{args.site}.npz")
    out_dir = STAGE2_OUT_ROOT / args.model_type / f"{args.site}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path, attn_path, exact_path = (out_dir / "per_prompt.csv",
                                       out_dir / "attention.csv",
                                       out_dir / "exact.csv")

    dz = np.load(dirs_path)
    steer_hook_name = INJECTION_HOOKS[args.site]
    first_layer = int(steer_hook_name.split(".")[1])
    scale = float(dz["natural_scale"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    readers = sorted((int(r["layer"]), int(r["head"])) for r in weights["readers"]
                     if READER_DLA_LAYERS[0] <= r["layer"] <= READER_DLA_LAYERS[1])
    n_heads = int(weights["n_heads"])

    config = dict(vars(args), capture_dir=str(capture_dir), directions=str(dirs_path),
                  out_dir=str(out_dir), natural_scale=scale, hook=steer_hook_name,
                  first_layer=first_layer, n_heads=n_heads, readers=readers)
    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type, n_devices=args.n_devices,
                          device_map=args.device_map),
        generation=GenerationConfig(), evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=TEMPLATES),
        wandb=WandbConfig(enabled=False))
    model, tokenizer, *_ = ModelLoader(cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()
    device = next(model.parameters()).device
    if model.cfg.n_heads != n_heads:
        sys.exit(f"weights.pt n_heads={n_heads} but the model has {model.cfg.n_heads}")
    layers = list(range(first_layer, model.cfg.n_layers))
    final_hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    runs = expected_runs(layers)
    config.update(layers=layers, final_hook=final_hook, runs=sorted(runs))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    log.info(f"layers {layers[0]}-{layers[-1]}; {len(runs)} runs/prompt; "
             f"direction={args.direction} dose={args.dose} hook={steer_hook_name}")

    tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    applied_norms: list[float] = []

    def steer_fn(mode):
        if args.direction == "target_topk":
            mu = dz["mean_FB_fit"] if mode == "inject" else dz["mean_TB_fit"]
            return make_steer_fn("swap", args.dose, applied_norms, U=tt(dz["U_topk"]), mu=tt(mu))
        return make_steer_fn("add", args.dose, applied_norms, vec=tt(dz["v_rank1"]),
                             scale=scale, sign=(+1.0 if mode == "inject" else -1.0))

    done = load_done(per_path, runs)
    for variant, mode in ((FBD, "delete"), (TBD, "inject")):
        prompts = heldout_capture_prompts(capture_dir, dz, variant, domain["templates"],
                                          divergent=FBD)
        if args.limit:
            prompts = prompts[:args.limit]
        log.info(f"{variant} ({mode}): {len(prompts)} held-out prompts")
        hook = (steer_hook_name, steer_fn(mode))
        for i, rec in enumerate(prompts):
            key = (variant, rec["template"], rec["pair_idx"])
            if key in done:
                continue
            tokens = tokenizer(rec["prompt"], return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(device)
            cid = compute_answer_token_id(tokenizer, variant, rec["correct"], rec["prompt"])
            aid = compute_answer_token_id(tokenizer, variant, rec["alt"], rec["prompt"])
            n_tok = tokens.shape[1]
            m_bel = location_token_mask(tokenizer, rec["prompt"], rec["believed"], n_tok)
            m_cur = location_token_mask(tokenizer, rec["prompt"], rec["current"], n_tok)
            rows, attn_rows, exact = run_prompt(
                model, tokens, hook, layers, readers, n_heads, cid, aid, m_bel, m_cur,
                steer_hook_name, final_hook)
            meta = dict(prompt_set=variant, mode=mode, direction=args.direction,
                        dose=args.dose, template=rec["template"], pair_idx=rec["pair_idx"])
            for path, rs in ((per_path, rows), (attn_path, attn_rows), (exact_path, [exact])):
                pd.DataFrame([{**meta, **r} for r in rs]).to_csv(
                    path, mode="a", header=not path.exists(), index=False)
            done.add(key)
            if (i + 1) % 10 == 0:
                log.info(f"    {variant}: {i + 1}/{len(prompts)}")
    log.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
