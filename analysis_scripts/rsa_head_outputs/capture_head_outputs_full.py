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

With --capture_attn_summary, four REDUCED attention readouts are added, each
(n_stim, n_layers, n_heads) float32, computed from the final-token query row
of hook_pattern:
  - attn_mean_src   attention-weighted mean absolute source position
  - attn_entropy    entropy of that attention row (nats)
  - attn_mass_slot  mass on the tokens spanning the queried slot's character
  - attn_mass_mut   mass on the tokens spanning the mutated character
plus attn_mass_code (mass on the whole code block) and n_tokens.

Raw patterns are NOT stored: 864 stimuli x 48 layers x 40 heads x T^2 is
hundreds of GB, and everything the position-inheritance control needs is a
scalar per (stimulus, head). attn_mean_src is that scalar — a head that
merely inherits position through its value vectors has a pad response
predictable from how far its attention centre of mass moves, and nothing
else. See scripts/rsa_head_outputs/padtwins_position_prediction.py.
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
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

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


def char_span_mask(offsets, lo: int, hi: int, n_tokens: int) -> np.ndarray:
    """Tokens whose character span overlaps [lo, hi). Special tokens carry
    the (0, 0) span and never match for lo >= 0."""
    m = np.zeros(n_tokens, dtype=bool)
    for ti, (a, b) in enumerate(offsets[:n_tokens]):
        if a < hi and b > lo:
            m[ti] = True
    return m


def stimulus_masks(tokenizer, prompt: str, row, n_tokens: int):
    """Boolean key-position masks for the queried slot char, the mutated
    char, and the code block. Character offsets in the stimulus CSV are
    relative to `scenario`; shift them by where the scenario starts inside
    the chat-templated prompt."""
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    base = prompt.find(row["scenario"])
    if base < 0:  # chat template altered the text — masks are unusable
        z = np.zeros(n_tokens, dtype=bool)
        return z, z, z
    slot = int(row["slot_char_pos"]) + base
    mut = int(row["mut_char_pos"]) + base
    code0 = int(row["code_start_char"]) + base
    code1 = base + int(row["scenario_len"])
    return (char_span_mask(offsets, slot, slot + 1, n_tokens),
            char_span_mask(offsets, mut, mut + 1, n_tokens),
            char_span_mask(offsets, code0, code1, n_tokens))


def make_pattern_hooks(store, n_layers):
    """Reduce hook_pattern's final-token query row to per-head scalars.

    Registered as forward hooks rather than cached, so peak memory stays at
    one layer's (n_heads, T) row instead of every layer's (n_heads, T, T)."""
    def hook(pattern, hook):
        # pattern: (batch, n_heads, query, key); batch is always 1 here
        p = pattern[0, :, -1, :].float()
        pos = torch.arange(p.shape[-1], device=p.device, dtype=p.dtype)
        L = hook.layer()
        store["mean_src"][L] = (p * pos).sum(-1).cpu()
        store["entropy"][L] = (-(p * torch.log(p + 1e-12)).sum(-1)).cpu()
        for key, m in store["masks"].items():
            store[key][L] = p[:, m].sum(-1).cpu()
        return pattern

    names = {f"blocks.{L}.attn.hook_pattern" for L in range(n_layers)}
    return [(lambda n: n in names, hook)]


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
    ap.add_argument("--capture_attn_summary", action="store_true",
                    help="also store per-head reduced attention readouts at "
                         "the final token (mean source position, entropy, "
                         "mass on the queried-slot / mutated / code spans). "
                         "Needs the stimulus CSV to carry slot_char_pos, "
                         "mut_char_pos, code_start_char, scenario_len.")
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

    ATTN_KEYS = ("mean_src", "entropy", "mass_slot", "mass_mut", "mass_code")
    attn_out, attn_store, pattern_hooks, n_tokens_out = None, None, [], None
    if args.capture_attn_summary:
        missing = {"slot_char_pos", "mut_char_pos", "code_start_char",
                   "scenario_len"} - set(stim_df.columns)
        if missing:
            raise SystemExit(f"--capture_attn_summary needs columns {missing} "
                             f"in {stim_csv}")
        attn_out = {k: torch.full((n_stim, n_layers, n_heads), float("nan"),
                                  dtype=torch.float32) for k in ATTN_KEYS}
        n_tokens_out = torch.zeros(n_stim, dtype=torch.int32)
        attn_store = {k: {} for k in ATTN_KEYS}
        attn_store["masks"] = {}
        pattern_hooks = make_pattern_hooks(attn_store, n_layers)

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
        if args.capture_attn_summary:
            n_tok = tokens.shape[1]
            m_slot, m_mut, m_code = stimulus_masks(tokenizer, prompt, row, n_tok)
            attn_store["masks"] = {
                "mass_slot": torch.tensor(m_slot, device=tokens.device),
                "mass_mut": torch.tensor(m_mut, device=tokens.device),
                "mass_code": torch.tensor(m_code, device=tokens.device)}
            n_tokens_out[si] = n_tok

        with torch.no_grad(), model.hooks(fwd_hooks=pattern_hooks):
            _, cache = model.run_with_cache(tokens, names_filter=names_filter)

        if args.capture_attn_summary:
            for k in ATTN_KEYS:
                for L, v in attn_store[k].items():
                    attn_out[k][si, L] = v
                attn_store[k].clear()

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
    if args.capture_attn_summary:
        payload.update({f"attn_{k}": v for k, v in attn_out.items()})
        payload["n_tokens"] = n_tokens_out
        payload["meta"]["attn_summary"] = list(ATTN_KEYS)
        log.info("attention summary: mean source position ranges "
                 f"{float(attn_out['mean_src'].nanmean()):.1f} "
                 f"(mean over all heads/stimuli), mean code-span mass "
                 f"{float(attn_out['mass_code'].nanmean()):.3f}")
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
