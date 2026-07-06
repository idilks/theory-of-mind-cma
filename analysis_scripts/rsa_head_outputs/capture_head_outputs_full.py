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

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from cma_config import (  # noqa: E402
    EvaluationConfig, GenerationConfig, ModelConfig, PatchingConfig,
    PromptConfig, WandbConfig, ExperimentConfig,
)
from models import ModelLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("rsa-capture-full")

DEFAULT_STIM_DIR = ROOT / "results/rsa_head_outputs/stimuli"
DEFAULT_OUT_DIR = ROOT / "results/rsa_head_outputs/activations_full"

N_LAYERS = 48
N_HEADS = 40
HEAD_DIM = 128


def build_chat_prompt(scenario: str, question: str, tokenizer) -> str:
    system_prompt = "You are a helpful assistant reasoning about agents' beliefs."
    full = f"{scenario}\n{question}"
    msgs = [{"role": "system", "content": system_prompt},
            {"role": "user",   "content": full}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--stim_dir", type=Path, default=DEFAULT_STIM_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stim_csv = args.stim_dir / f"{args.template}.csv"
    stim_df = pd.read_csv(stim_csv)
    log.info(f"loaded {len(stim_df)} stimuli from {stim_csv}")

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
        return name.endswith(".attn.hook_z")

    n_stim = len(stim_df)
    z_out = torch.full((n_stim, N_LAYERS, N_HEADS, HEAD_DIM),
                       float("nan"), dtype=torch.float16)

    for si, row in stim_df.iterrows():
        prompt = build_chat_prompt(row["scenario"], row["question"], tokenizer)
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        for L in range(N_LAYERS):
            key = f"blocks.{L}.attn.hook_z"
            if key not in cache:
                continue
            # cache[key]: (batch, seq, n_heads, head_dim) -> last token, all heads
            z_out[si, L] = cache[key][0, -1].float().cpu().to(torch.float16)
        del cache
        if (si + 1) % 10 == 0:
            log.info(f"  {si + 1}/{n_stim}")

    out_path = args.out_dir / f"{args.template}.pt"
    torch.save({
        "z": z_out,
        "stim": stim_df.to_dict("records"),
        "meta": {
            "template": args.template, "model": args.model_type,
            "n_stim": n_stim, "n_layers": N_LAYERS,
            "n_heads": N_HEADS, "head_dim": HEAD_DIM,
        },
    }, out_path)
    log.info(f"saved {out_path}  (shape={tuple(z_out.shape)}, dtype=float16)")


if __name__ == "__main__":
    main()
