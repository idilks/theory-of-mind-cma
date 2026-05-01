"""
Capture attention patterns for top-K CMA-significant heads on existing ToM prompts.

Tests the lookback hypothesis from Prakash et al. 2025 (ICLR 2026): for each top
head flagged by end-position CMA, does its attention from the lookback token
(final ":") peak on a recalled token (state / character / object)?

Reads:
  - significant heads + observed_score from existing
    statistical_test_results.csv (default: patch_before_movement)
  - regenerates the same base prompts via UnifiedTomGenerator (deterministic,
    seeded the same way as the original CMA run)

Writes a single .pt per template at:
  results/lookback_analysis/{condition}/template_{T}/attention.pt

Containing:
  - attn: (n_prompts, n_heads_kept, n_query_pos, max_seq) float16
            attention weights from each query position to all key positions.
            Padding positions filled with NaN.
  - heads: list of dicts [{layer, head, observed_score, p_value, significant}]
  - prompts: list of base prompt strings
  - positions: list of dicts {state, char_q, obj_q, belief_form, lookback,
                              seq_len, base_ans} per prompt
  - query_pos_names: ["lookback", "belief_formation"]
  - meta: {template, condition, base_rule, seed, top_k, source_csv}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# project root on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from cma_config import (  # noqa: E402
    CMACondition,
    EvaluationConfig,
    GenerationConfig,
    ModelConfig,
    PatchingConfig,
    PromptConfig,
    WandbConfig,
    ExperimentConfig,
)
from models import ModelLoader  # noqa: E402
from prompt_generators.unified_tom import UnifiedTomGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lookback")


# ---------- Component 1: significant-head loader ----------

def find_csv(condition: str, template: str, base_rule: str, exp_rule: str,
             patch_position: str, results_root: Path) -> Path:
    """Pick the CSV with the largest sample_num under the given slice."""
    base_dir = (
        results_root
        / "causal_analysis" / "cma" / "Qwen" / "Qwen2.5-14B-Instruct"
        / condition / f"template_{template}"
        / f"base_rule_{base_rule}_exp_rule_{exp_rule}"
        / patch_position
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"slice dir not found: {base_dir}")

    candidates = list(base_dir.glob("z_seed_*_shuffle_False/logit/sample_num_*_gen_acc_*/statistical_test_results.csv"))
    if not candidates:
        raise FileNotFoundError(f"no statistical_test_results.csv under {base_dir}")

    def sample_num(p: Path) -> int:
        m = re.search(r"sample_num_(\d+)_gen_acc", str(p))
        return int(m.group(1)) if m else 0

    return max(candidates, key=sample_num)


def load_top_heads(csv_path: Path, top_k: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["abs_score"] = df["observed_score"].abs()
    top = df.sort_values("abs_score", ascending=False).head(top_k).reset_index(drop=True)
    return top[["layer", "head", "observed_score", "p_value", "significant"]]


# ---------- Component 2: prompt regen + token-position extractor ----------

def build_prompt_config(template: str, base_rule: str, condition: CMACondition,
                        prompt_num: int, seed: int,
                        device_map: str = "cuda", load_in_4bit: bool = False,
                        load_in_8bit: bool = False) -> ExperimentConfig:
    """Mirror the cma.py config used at training time so prompts are regenerated identically."""
    return ExperimentConfig(
        model=ModelConfig(model_type="Qwen2.5-14B-Instruct", device_map=device_map, n_devices=1,
                          load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit),
        generation=GenerationConfig(max_new_tokens=5),
        prompts=PromptConfig(
            base_rule=base_rule,
            context_type=condition.context_type,
            prompt_num=prompt_num,
            condition=condition,
            use_template_system=True,
            question_style="instruction",
            template_names=[template],
        ),
        patching=PatchingConfig(activation_name="z"),
        evaluation=EvaluationConfig(),
        wandb=WandbConfig(enabled=False),
        seed=seed,
    )


def char_to_token(offset_mapping, char_pos_start: int, char_pos_end: int) -> int:
    """Find token whose offsets cover the start of the char span. Returns -1 if not found."""
    for i, (start, end) in enumerate(offset_mapping):
        if start <= char_pos_start < end or (start == char_pos_start and end == char_pos_start):
            return i
        if start > char_pos_start:
            return i
    return -1


# Per-template metadata: which strings to find in the prompt for char_q / obj_q.
# state token is always the base_ans returned by the generator.
TEMPLATE_QUERY_KEYS = {
    "food_truck":     {"char_q": "Alice",    "obj_q": "food truck"},
    "hair_styling":   {"char_q": "Jennifer", "obj_q": "hair"},
    "library_book":   {"char_q": "Sarah",    "obj_q": "book"},
}


def extract_positions(prompt: str, base_ans: str, template: str, tokenizer,
                      generator: UnifiedTomGenerator, base_rule: str) -> dict:
    """Map source-token strings to token indices in the given prompt."""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    seq_len = len(enc["input_ids"])

    # state token: first occurrence of base_ans in story
    state_pos = -1
    if base_ans:
        idx = prompt.find(base_ans)
        if idx != -1:
            state_pos = char_to_token(offsets, idx, idx + len(base_ans))

    # char_q / obj_q: first occurrence of template-specific query strings
    keys = TEMPLATE_QUERY_KEYS.get(template, {})
    char_pos = -1
    if keys.get("char_q"):
        idx = prompt.find(keys["char_q"])
        if idx != -1:
            char_pos = char_to_token(offsets, idx, idx + len(keys["char_q"]))

    obj_pos = -1
    if keys.get("obj_q"):
        idx = prompt.find(keys["obj_q"])
        if idx != -1:
            obj_pos = char_to_token(offsets, idx, idx + len(keys["obj_q"]))

    # belief_formation: existing detector (handles ABA = false belief, ABB = true belief)
    is_false_belief = (base_rule == "ABA")
    bf_pos = generator.detect_belief_formation_position(prompt, tokenizer, is_false_belief=is_false_belief)
    # generator returns position based on tokenizer-with-special-tokens; rerun without special tokens for consistency
    # If detector returned -1 or out of range, leave -1.
    if bf_pos < 0 or bf_pos >= seq_len:
        # try the version with special tokens (tokenizer's default)
        enc_with = tokenizer(prompt, return_offsets_mapping=True)
        bf_pos2 = generator.detect_belief_formation_position(prompt, tokenizer, is_false_belief=is_false_belief)
        if 0 <= bf_pos2 < len(enc_with["input_ids"]):
            # reproject onto our no-special-tokens encoding by character offset
            if bf_pos2 < len(enc_with["offset_mapping"]):
                target_char = enc_with["offset_mapping"][bf_pos2 - 1][1] if bf_pos2 > 0 else 0
                bf_pos = char_to_token(offsets, target_char, target_char + 1)
            else:
                bf_pos = -1
        else:
            bf_pos = -1

    # lookback: last input token (right before model generates)
    lookback_pos = seq_len - 1

    return {
        "state": state_pos,
        "char_q": char_pos,
        "obj_q": obj_pos,
        "belief_form": bf_pos,
        "lookback": lookback_pos,
        "seq_len": seq_len,
        "base_ans": base_ans,
    }


# ---------- Component 3: forward pass + attention capture ----------

def capture_for_template(
    template: str,
    condition: CMACondition,
    base_rule: str,
    exp_rule: str,
    patch_position: str,
    top_k: int,
    prompt_num: int,
    seed: int,
    out_dir: Path,
    results_root: Path,
    device_map: str = "cuda",
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
):
    csv_path = find_csv(
        condition=condition.value, template=template,
        base_rule=base_rule, exp_rule=exp_rule,
        patch_position=patch_position, results_root=results_root,
    )
    log.info(f"loading significant heads from {csv_path}")
    heads_df = load_top_heads(csv_path, top_k)
    log.info(f"top {len(heads_df)} heads (by |observed_score|): "
             f"min={heads_df.abs_score.min() if 'abs_score' in heads_df else heads_df.observed_score.abs().min():.4g}")

    cfg = build_prompt_config(template, base_rule, condition, prompt_num, seed,
                              device_map=device_map,
                              load_in_4bit=load_in_4bit,
                              load_in_8bit=load_in_8bit)

    # Set seed before model load + prompt gen
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.info("loading model + tokenizer (Qwen2.5-14B-Instruct, bf16)")
    loader = ModelLoader(cfg.model, cfg.generation, cfg.prompts)
    model, tokenizer, *_ = loader.load_model_and_tokenizer()
    model.eval()

    # Reseed after model load (model load can perturb global RNG)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    log.info("regenerating prompts")
    generator = UnifiedTomGenerator(cfg, tokenizer)
    prompt_pairs, correct_ans_pairs, _ = generator.generate_prompts()
    base_prompts = [p[0] for p in prompt_pairs]
    base_answers = [a[0] for a in correct_ans_pairs]
    log.info(f"got {len(base_prompts)} base prompts")

    # Tokenize + extract positions
    positions = []
    for p, a in zip(base_prompts, base_answers):
        positions.append(extract_positions(p, a, template, tokenizer, generator, base_rule))

    max_seq = max(pos["seq_len"] for pos in positions)
    log.info(f"max seq len = {max_seq}")

    n_prompts = len(base_prompts)
    n_heads = len(heads_df)
    n_query_pos = 2  # lookback, belief_formation
    query_pos_names = ["lookback", "belief_formation"]

    # Pre-allocate output (NaN for padding)
    attn_out = torch.full((n_prompts, n_heads, n_query_pos, max_seq),
                          float("nan"), dtype=torch.float16)

    head_layers = heads_df["layer"].tolist()
    head_idxs = heads_df["head"].tolist()

    # Group heads by layer for cache filtering
    layers_needed = sorted(set(head_layers))
    log.info(f"layers needed for hooks: {len(layers_needed)} unique layers")

    def names_filter(name: str) -> bool:
        if "hook_pattern" not in name:
            return False
        m = re.search(r"blocks\.(\d+)\.attn\.hook_pattern", name)
        if not m:
            return False
        return int(m.group(1)) in set(layers_needed)

    log.info("running forward passes")
    for pi, prompt in enumerate(base_prompts):
        tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        seq_len = tokens.shape[1]
        lookback = positions[pi]["lookback"]
        bf = positions[pi]["belief_form"]

        for hi, (L, H) in enumerate(zip(head_layers, head_idxs)):
            key = f"blocks.{L}.attn.hook_pattern"
            if key not in cache:
                continue
            # shape (batch, n_heads, seq_q, seq_k)
            pattern = cache[key][0, H]  # (seq_q, seq_k)

            # Query 0: lookback
            if 0 <= lookback < seq_len:
                attn_out[pi, hi, 0, :seq_len] = pattern[lookback].float().cpu().to(torch.float16)
            # Query 1: belief_formation (may be -1)
            if 0 <= bf < seq_len:
                attn_out[pi, hi, 1, :seq_len] = pattern[bf].float().cpu().to(torch.float16)

        del cache
        if (pi + 1) % 10 == 0:
            log.info(f"  prompt {pi+1}/{n_prompts}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "attention.pt"
    payload = {
        "attn": attn_out,
        "heads": heads_df.to_dict(orient="records"),
        "prompts": base_prompts,
        "positions": positions,
        "query_pos_names": query_pos_names,
        "meta": {
            "template": template,
            "condition": condition.value,
            "base_rule": base_rule,
            "exp_rule": exp_rule,
            "patch_position": patch_position,
            "seed": seed,
            "top_k": top_k,
            "source_csv": str(csv_path),
            "max_seq": max_seq,
        },
    }
    torch.save(payload, out_path)
    log.info(f"saved {out_path}  shape={tuple(attn_out.shape)}")

    # Also drop a quick-look summary
    summary = {
        "n_prompts": n_prompts,
        "n_heads": n_heads,
        "max_seq": max_seq,
        "pct_state_found": sum(1 for p in positions if p["state"] >= 0) / n_prompts,
        "pct_char_found": sum(1 for p in positions if p["char_q"] >= 0) / n_prompts,
        "pct_obj_found": sum(1 for p in positions if p["obj_q"] >= 0) / n_prompts,
        "pct_bf_found": sum(1 for p in positions if p["belief_form"] >= 0) / n_prompts,
    }
    (out_dir / "capture_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"summary: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, choices=["food_truck", "hair_styling", "library_book"])
    ap.add_argument("--condition", default="abstract_belief",
                    help="CMACondition value (default: abstract_belief)")
    ap.add_argument("--base_rule", default="ABA", choices=["ABA", "ABB"])
    ap.add_argument("--exp_rule", default="ABB", choices=["ABA", "ABB"])
    ap.add_argument("--patch_position", default="patch_before_movement",
                    choices=["patch_before_movement", "patch_after_movement"],
                    help="which CMA result folder to read significant heads from "
                         "(default: end / patch_before_movement, the lookback-relevant slice)")
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--prompt_num", type=int, default=50,
                    help="must match the prompt_num used in the original CMA run for prompt regeneration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results_root", default="results")
    ap.add_argument("--out_root", default="results/lookback_analysis")
    ap.add_argument("--device_map", default="cuda",
                    help="cuda | cpu | auto. Use 'auto' for HF accelerate device_map auto.")
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="bitsandbytes 4-bit quantization (Qwen2.5-14B fits ~9GB).")
    ap.add_argument("--load_in_8bit", action="store_true",
                    help="bitsandbytes 8-bit quantization (~16GB).")
    args = ap.parse_args()

    condition = CMACondition(args.condition)
    out_dir = Path(args.out_root) / condition.value / f"template_{args.template}" \
              / f"base_rule_{args.base_rule}_exp_rule_{args.exp_rule}" / args.patch_position

    capture_for_template(
        template=args.template,
        condition=condition,
        base_rule=args.base_rule,
        exp_rule=args.exp_rule,
        patch_position=args.patch_position,
        top_k=args.top_k,
        prompt_num=args.prompt_num,
        seed=args.seed,
        out_dir=out_dir,
        results_root=Path(args.results_root),
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )


if __name__ == "__main__":
    main()
