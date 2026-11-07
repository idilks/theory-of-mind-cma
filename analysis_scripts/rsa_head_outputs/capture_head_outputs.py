"""Capture per-head output activations (hook_z) at the final token position
across the balanced RSA stimulus set.

For each stimulus prompt, run a single forward pass with transformer_lens'
run_with_cache and grab `blocks.{L}.attn.hook_z` for the layers we need.
hook_z has shape (batch, seq, n_heads, head_dim=128) in TL convention.
We extract position -1 (the final token, where the answer is read out) and
slice out the heads in the union top-k CSV.

Output (per template): results/rsa_head_outputs/activations/{template}.pt
  - z: (n_stim, n_heads, head_dim) float16
  - heads: list of {layer, head}  (same order as columns of z)
  - stim: list of dicts mirroring the stimulus CSV
  - meta: {template, model, k_per_cell}
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

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
from prompt_generators.code_templates import CODE_SYSTEM_PROMPT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rsa-capture")

DEFAULT_STIM_DIR = ROOT / "results/rsa_head_outputs/stimuli"
DEFAULT_HEADS_CSV = ROOT / "results/rsa_head_outputs/topk_heads/union_top20.csv"
DEFAULT_OUT_DIR = ROOT / "results/rsa_head_outputs/activations"


BELIEF_SYSTEM_PROMPT = "You are a helpful assistant reasoning about agents' beliefs."


def build_chat_prompt(scenario: str, question: str, tokenizer,
                      system_prompt: str = BELIEF_SYSTEM_PROMPT) -> str:
    full = f"{scenario}\n{question}"
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": full}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        if "System role not supported" not in str(exc):
            raise
        # fold, matching unified_tom.py's convention for Gemma
        msgs = [{"role": "user", "content": f"{system_prompt}\n\n{full}"}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--stim_dir", type=Path, default=DEFAULT_STIM_DIR)
    ap.add_argument("--heads_csv", type=Path, default=DEFAULT_HEADS_CSV)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--system_prompt", choices=["belief", "code"], default="belief",
                    help="'code' uses CODE_SYSTEM_PROMPT (for code_position stimuli)")
    args = ap.parse_args()

    if args.system_prompt == "code":
        system_prompt = CODE_SYSTEM_PROMPT
    else:
        system_prompt = BELIEF_SYSTEM_PROMPT

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stim_csv = args.stim_dir / f"{args.template}.csv"
    stim_df = pd.read_csv(stim_csv)
    log.info(f"loaded {len(stim_df)} stimuli from {stim_csv}")

    heads_df = pd.read_csv(args.heads_csv)
    layers_needed = sorted(set(heads_df["layer"].astype(int).tolist()))
    log.info(f"{len(heads_df)} heads across {len(layers_needed)} layers")

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type),
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
        if "hook_z" not in name:
            return False
        m = re.search(r"blocks\.(\d+)\.attn\.hook_z", name)
        return bool(m) and int(m.group(1)) in set(layers_needed)

    n_stim = len(stim_df)
    n_heads = len(heads_df)
    head_dim = 128  # Qwen2.5-14B
    z_out = torch.full((n_stim, n_heads, head_dim), float("nan"), dtype=torch.float16)

    head_layers = heads_df["layer"].astype(int).tolist()
    head_idxs = heads_df["head"].astype(int).tolist()

    for si, row in stim_df.iterrows():
        prompt = build_chat_prompt(row["scenario"], row["question"], tokenizer,
                                   system_prompt=system_prompt)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        for hi, (L, H) in enumerate(zip(head_layers, head_idxs)):
            key = f"blocks.{L}.attn.hook_z"
            if key not in cache:
                continue
            # (batch, seq, n_heads, head_dim) — last token, head H
            z_out[si, hi] = cache[key][0, -1, H].float().cpu().to(torch.float16)
        del cache
        if (si + 1) % 10 == 0:
            log.info(f"  {si + 1}/{n_stim}")

    out_path = args.out_dir / f"{args.template}.pt"
    torch.save({
        "z": z_out,
        "heads": heads_df[["layer", "head"]].to_dict("records"),
        "stim": stim_df.to_dict("records"),
        "meta": {"template": args.template, "model": args.model_type, "n_stim": n_stim},
    }, out_path)
    log.info(f"saved {out_path}")


if __name__ == "__main__":
    main()
