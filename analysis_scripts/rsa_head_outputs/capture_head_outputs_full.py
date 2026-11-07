"""Capture per-head output activations (hook_z) at the final token position
for ALL 1920 heads (48 layers x 40 heads), across the balanced RSA stimulus
set.

This is the "all heads" complement to capture_head_outputs.py. The latter
hooks only a union-top-k subset; this one captures the full grid so
downstream analyses can CMA-score-weight every head, not just selected ones.

Output (per template): results/rsa_head_outputs/activations_full/{template}.pt
  - z: (n_stim, n_layers, n_heads, head_dim) float16
  - stim: list of dicts mirroring the stimulus CSV
  - meta: {template, model, n_stim, n_layers, n_heads, head_dim}
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from cma_config import (  # noqa: E402
    EvaluationConfig, GenerationConfig, ModelConfig, PatchingConfig,
    PromptConfig, WandbConfig, ExperimentConfig,
)
from models import ModelLoader  # noqa: E402
from prompt_generators.code_templates import (  # noqa: E402
    CODE_SYSTEM_PROMPT, extract_value_from_code_response, code_values_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rsa-capture-full")

DEFAULT_STIM_DIR = ROOT / "results/rsa_head_outputs/stimuli"
DEFAULT_OUT_DIR = ROOT / "results/rsa_head_outputs/activations_full"

BELIEF_SYSTEM_PROMPT = "You are a helpful assistant reasoning about agents' beliefs."


def supports_system_role(tokenizer) -> bool:
    """Same probe as unified_tom.py — Gemma templates reject system messages."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "check"},
             {"role": "user", "content": "check"}],
            tokenize=False, add_generation_prompt=True)
        return True
    except Exception as exc:
        if "System role not supported" in str(exc):
            return False
        raise


def build_chat_prompt(scenario: str, question: str, tokenizer,
                      system_prompt: str = BELIEF_SYSTEM_PROMPT,
                      system_role_ok: bool = True) -> str:
    full = f"{scenario}\n{question}"
    if system_role_ok:
        msgs = [{"role": "system", "content": system_prompt},
                {"role": "user",   "content": full}]
    else:
        # fold, matching unified_tom.py's convention for Gemma
        msgs = [{"role": "user", "content": f"{system_prompt}\n\n{full}"}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--stim_dir", type=Path, default=DEFAULT_STIM_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--ungroup_grouped_query_attention", action="store_true",
                    help="required for Gemma-2 (GQA) so hook_z is per-head")
    ap.add_argument("--system_prompt", choices=["belief", "code"], default="belief",
                    help="'code' uses CODE_SYSTEM_PROMPT (for code_* stimuli)")
    ap.add_argument("--decode_answers", action="store_true",
                    help="also greedy-decode a few tokens per prompt and store "
                         "them (baseline accuracy check, ~2x runtime)")
    args = ap.parse_args()

    if args.system_prompt == "code":
        system_prompt = CODE_SYSTEM_PROMPT
    else:
        system_prompt = BELIEF_SYSTEM_PROMPT

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stim_csv = args.stim_dir / f"{args.template}.csv"
    stim_df = pd.read_csv(stim_csv)
    log.info(f"loaded {len(stim_df)} stimuli from {stim_csv}")

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type,
                          ungroup_grouped_query_attention=args.ungroup_grouped_query_attention),
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

    def names_filter(name: str) -> bool:
        return name.endswith(".attn.hook_z")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    head_dim = model.cfg.d_head
    log.info(f"model dims: {n_layers} layers x {n_heads} heads x {head_dim}")

    n_stim = len(stim_df)
    z_out = torch.full((n_stim, n_layers, n_heads, head_dim),
                       float("nan"), dtype=torch.float16)
    generations: list[str] = []

    system_role_ok = supports_system_role(tokenizer)
    if not system_role_ok:
        log.info("chat template lacks system role; folding into user message")

    for si, row in stim_df.iterrows():
        prompt = build_chat_prompt(row["scenario"], row["question"], tokenizer,
                                   system_prompt=system_prompt,
                                   system_role_ok=system_role_ok)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        for L in range(n_layers):
            key = f"blocks.{L}.attn.hook_z"
            if key not in cache:
                continue
            # cache[key]: (batch, seq, n_heads, head_dim) -> last token, all heads
            z_out[si, L] = cache[key][0, -1].float().cpu().to(torch.float16)
        del cache

        if args.decode_answers:
            with torch.no_grad():
                # KV cache is incompatible with ungrouped GQA (16 KV heads vs
                # 32 ungrouped): past_key_value_caching.py append() crashes.
                gen = model.generate(tokens, max_new_tokens=8, do_sample=False,
                                     use_past_kv_cache=False, verbose=False)
            generations.append(tokenizer.decode(gen[0, tokens.shape[1]:]))

        if (si + 1) % 10 == 0:
            log.info(f"  {si + 1}/{n_stim}")

    payload = {
        "z": z_out,
        "stim": stim_df.to_dict("records"),
        "meta": {
            "template": args.template, "model": args.model_type,
            "n_stim": n_stim, "n_layers": n_layers,
            "n_heads": n_heads, "head_dim": head_dim,
            "system_prompt": args.system_prompt,
        },
    }
    if args.decode_answers:
        payload["generations"] = generations
        if "answer" in stim_df.columns:
            correct = [code_values_match(extract_value_from_code_response(g), a)
                       for g, a in zip(generations, stim_df["answer"])]
            payload["correct"] = correct
            log.info(f"baseline accuracy: {np.mean(correct):.3f} "
                     f"({int(np.sum(correct))}/{len(correct)})")

    out_path = args.out_dir / f"{args.template}.pt"
    torch.save(payload, out_path)
    log.info(f"saved {out_path}  (shape={tuple(z_out.shape)}, dtype=float16)")


if __name__ == "__main__":
    main()
