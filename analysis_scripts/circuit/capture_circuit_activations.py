"""Stage-1 capture for the layered-lookback circuit (GPU job, one template).

Per prompt (matched FB/TB vignette pairs, n per side per template, matched
belief system prompt + per-variant answer prefill — the primary eval
protocol), a single forward pass captures at position -1:
  - hook_z for ALL heads in layers 28..40 (writer candidates + the pool
    for the random-head-set null in S1a)
  - hook_mlp_out for layers 28..40 (the MLP-written-wire alternative)
  - resid_post at L40 and L41 (the bus; resid_post L41 == resid_pre L42,
    the Stage-2 injection site)
  - first-token answer probabilities (behavioral sanity: TB p_correct
    should be ~0.99 under this protocol)

FB/TB prompts are index-aligned matched pairs: generate_unpaired_prompts
renders both variants from the same sampled substitutions, so pair i
differs only in belief veracity.

Also saves (once per model, --save_weights) the weight slices the laptop
screen needs, so no model load is required downstream:
  - W_O for layers 28..40 (all heads)         -> writer projections
  - W_Q / W_V slices + ln1.w for reader heads -> reader survival (S1c)
Model is loaded fold_ln=False (repo default): ln1.w is a real scale and
S1c must apply it; resid hooks are pre-LN either way.

Output: results/circuit/stage1_capture/{model}/{template}.pt (+ behavior
CSV) and .../weights.pt.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "circuit"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from circuit_common import (  # noqa: E402
    BAND_LAYERS, CAPTURE_ROOT, DOMAINS, READER_CSV, SITES,
    SYSTEM_PROMPT_MATCHED, load_head_csv,
)

# NOTE: importing ablation_experiment os.chdir()s to cma_scripts/ — every path
# in this script is absolute (via ROOT), so that side effect is harmless.
from tasks.causal_analysis.ablation_experiment import (  # noqa: E402
    compute_answer_token_id, generate_eval_prompts,
)
from tasks.causal_analysis.cma_config import (  # noqa: E402
    EvaluationConfig, ExperimentConfig, GenerationConfig, ModelConfig,
    PatchingConfig, PromptConfig, WandbConfig,
)
from tasks.causal_analysis.models import ModelLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("circuit-capture")


def build_names_filter():
    names = {f"blocks.{L}.attn.hook_z" for L in BAND_LAYERS}
    names |= {f"blocks.{L}.hook_mlp_out" for L in BAND_LAYERS}
    names |= set(SITES.values())
    return names, (lambda name: name in names)


def save_weight_slices(model, out_path: Path):
    """Slice out everything the laptop-side screen needs from the weights."""
    readers = load_head_csv(READER_CSV)
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    # (n_band_layers, n_heads, d_head, d_model)
    w_o = torch.stack([model.W_O[L] for L in BAND_LAYERS]).detach().cpu().to(torch.float16)

    # W_Q is per query head; W_K/W_V may be grouped (Qwen2.5-14B: 8 KV heads).
    w_v_full = model.W_V
    n_kv = w_v_full.shape[1]
    group = n_heads // n_kv
    if n_kv != n_heads:
        log.info(f"GQA: {n_heads} Q heads over {n_kv} KV heads (group={group}); "
                 "reader W_V taken from the head's KV group")

    w_q_readers, w_v_readers, ln1_w_readers = [], [], []
    for r in readers.itertuples():
        L, H = int(r.layer), int(r.head)
        w_q_readers.append(model.W_Q[L, H].detach().cpu().float())
        w_v_readers.append(w_v_full[L, H // group if n_kv != n_heads else H]
                           .detach().cpu().float())
        ln1_w_readers.append(model.blocks[L].ln1.w.detach().cpu().float())

    torch.save({
        "W_O_band": w_o,
        "band_layers": list(BAND_LAYERS),
        "readers": readers.to_dict("records"),
        "W_Q_readers": torch.stack(w_q_readers),      # (n_readers, d_model, d_head)
        "W_V_readers": torch.stack(w_v_readers),      # (n_readers, d_model, d_head)
        "ln1_w_readers": torch.stack(ln1_w_readers),  # (n_readers, d_model)
        "n_heads": n_heads, "n_kv_heads": int(n_kv),
        "d_model": d_model, "d_head": model.cfg.d_head,
        "fold_ln": False,
        "model_type": model.cfg.model_name,
    }, out_path)
    log.info(f"saved weight slices -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="tom")
    ap.add_argument("--template", default=None,
                    help="required for tom; defaults to indexed_list for code")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--prompt_num", type=int, default=150,
                    help="matched FB/TB pairs per template")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="default: results/circuit/stage1_capture/{model_type}")
    ap.add_argument("--save_weights", action="store_true",
                    help="also save weights.pt (one array task only)")
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="cuda")
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    variants = domain["variants"]
    if args.template is None:
        if len(domain["templates"]) != 1:
            sys.exit("--template is required for the tom domain")
        args.template = domain["templates"][0]
    out_dir = args.out_dir or (CAPTURE_ROOT / args.model_type
                               / domain["capture_subdir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not READER_CSV.exists():
        sys.exit(f"{READER_CSV} missing — run analysis_scripts/circuit/"
                 "build_circuit_head_sets.py locally and re-sync")

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type, n_devices=args.n_devices,
                          device_map=args.device_map),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=[args.template]),
        wandb=WandbConfig(enabled=False),
    )
    log.info("loading model")
    model, tokenizer, *_ = ModelLoader(cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()

    if args.save_weights:
        save_weight_slices(model, out_dir / "weights.pt")

    log.info(f"generating {args.prompt_num} matched {variants[0]}/{variants[1]} "
             f"pairs for {args.template}")
    # code variants ignore the system prompt arg and use CODE_SYSTEM_PROMPT
    prompt_sets = generate_eval_prompts(
        tokenizer, [args.template], args.prompt_num,
        SYSTEM_PROMPT_MATCHED, variants, prefill_mode="per_variant",
    )
    n = len(prompt_sets[variants[0]]["prompts"])
    assert all(len(prompt_sets[v]["prompts"]) == n for v in variants)

    names, names_filter = build_names_filter()
    n_band = len(BAND_LAYERS)
    d_model = model.cfg.d_model
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    device = next(model.parameters()).device

    z = {v: torch.full((n, n_band, n_heads, d_head), float("nan"), dtype=torch.float16)
         for v in variants}
    mlp = {v: torch.full((n, n_band, d_model), float("nan"), dtype=torch.float16)
           for v in variants}
    resid = {v: torch.full((n, len(SITES), d_model), float("nan"), dtype=torch.float32)
             for v in variants}
    behavior = {v: [] for v in variants}

    site_names = list(SITES.values())
    for variant in variants:
        ps = prompt_sets[variant]
        for i, prompt in enumerate(ps["prompts"]):
            tokens = tokenizer(prompt, return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(device)
            with torch.no_grad():
                logits, cache = model.run_with_cache(tokens, names_filter=names_filter)

            for li, L in enumerate(BAND_LAYERS):
                z[variant][i, li] = cache[f"blocks.{L}.attn.hook_z"][0, -1].float().cpu().to(torch.float16)
                mlp[variant][i, li] = cache[f"blocks.{L}.hook_mlp_out"][0, -1].float().cpu().to(torch.float16)
            for si, name in enumerate(site_names):
                resid[variant][i, si] = cache[name][0, -1].float().cpu()

            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            correct, alt = ps["correct"][i], ps["alt"][i]
            cid = compute_answer_token_id(tokenizer, variant, correct, prompt)
            aid = compute_answer_token_id(tokenizer, variant, alt, prompt)
            top1 = int(probs.argmax())
            behavior[variant].append(dict(
                idx=i, variant=variant, correct=correct, alt=alt,
                p_correct=float(probs[cid]), p_alt=float(probs[aid]),
                top1_str=tokenizer.decode([top1]),
                token_len=int(tokens.shape[1]),
                first_token_collision=bool(cid == aid),
            ))
            del cache
            if (i + 1) % 25 == 0:
                log.info(f"  {variant}: {i + 1}/{n}")

    beh_df = pd.concat([pd.DataFrame(behavior[v]) for v in variants],
                       ignore_index=True)
    for v in variants:
        sub = beh_df[beh_df["variant"] == v]
        log.info(f"{v}: mean p_correct={sub['p_correct'].mean():.3f} "
                 f"mean p_alt={sub['p_alt'].mean():.3f} "
                 f"collisions={int(sub['first_token_collision'].sum())}")
    # non-divergent variant (true_belief / code_alias) is the working-model check
    nondiv = variants[1]
    tb = beh_df[beh_df["variant"] == nondiv]["p_correct"].mean()
    if tb < 0.9:
        log.warning(f"{nondiv} mean p_correct={tb:.3f} < 0.9 — protocol "
                    "expects a working model; check prompts/prefill")

    out_path = out_dir / f"{args.template}.pt"
    torch.save({
        "z": z, "mlp": mlp, "resid": resid,
        "prompts": {v: prompt_sets[v]["prompts"] for v in variants},
        "behavior": {v: behavior[v] for v in variants},
        "meta": {
            "template": args.template, "model_type": args.model_type,
            "domain": args.domain, "n_pairs": n, "seed": args.seed,
            "system_prompt": ("CODE_SYSTEM_PROMPT" if args.domain == "code"
                              else SYSTEM_PROMPT_MATCHED),
            "prefill_mode": "per_variant",
            "eval_variants": variants,
            "band_layers": list(BAND_LAYERS),
            "sites": dict(SITES), "position": -1,
        },
    }, out_path)
    beh_df.to_csv(out_dir / f"{args.template}_behavior.csv", index=False)
    log.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
