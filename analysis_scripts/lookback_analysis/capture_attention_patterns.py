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
  - positions: list of dicts {state, state_first, state_second, state_correct,
                              info_loss, char_q, obj_q, belief_form, lookback,
                              seq_len, base_ans} per prompt
              (state == state_correct, kept for backward compat with older
              analyze scripts.)
  - query_pos_names: ["lookback", "belief_formation", "agent_last", "agent_first"]
                     (older payloads may have only the first two)
  - meta: {template, condition, base_rule, seed, top_k, source_csv}
"""

from __future__ import annotations

import argparse
import json
import logging
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
    top = df.sort_values("observed_score", ascending=False).head(top_k).reset_index(drop=True)
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


def _find_token_pos(prompt: str, needle: str, offsets) -> int:
    """First-occurrence char->token lookup. Returns -1 if needle not in prompt."""
    if not needle:
        return -1
    idx = prompt.find(needle)
    if idx == -1:
        return -1
    return char_to_token(offsets, idx, idx + len(needle))


def _find_all_token_pos(prompt: str, needle: str, offsets) -> list[int]:
    """All occurrences of `needle` mapped to token positions (first-token-of-span)."""
    if not needle:
        return []
    out: list[int] = []
    start = 0
    while True:
        idx = prompt.find(needle, start)
        if idx == -1:
            break
        tok = char_to_token(offsets, idx, idx + len(needle))
        if tok >= 0:
            out.append(tok)
        start = idx + max(1, len(needle))
    return out


def extract_positions(prompt: str, base_ans: str, template: str, tokenizer,
                      generator: UnifiedTomGenerator, base_rule: str,
                      pair_meta: dict | None = None,
                      context_type: str = "abstract") -> dict:
    """Map source-token strings to token indices in the given prompt.

    pair_meta (from UnifiedTomGenerator.last_metadata) supplies the exact
    location strings filling slots {var_1} and {var_2} of the base scenario,
    plus the info_loss_marker phrase. When supplied, location anchors come
    directly from generation metadata — no regex / substring guessing.
    Falls back to base_ans-only behavior if pair_meta is None.
    """
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    seq_len = len(enc["input_ids"])

    # Three location anchors
    if pair_meta is not None:
        loc_first  = pair_meta.get("loc_first_str")  or ""
        loc_second = pair_meta.get("loc_second_str") or ""
        loc_correct = pair_meta.get("loc_correct_str") or base_ans or ""
        info_loss_marker = pair_meta.get("info_loss_marker_str")
    else:
        loc_first = loc_second = ""
        loc_correct = base_ans or ""
        info_loss_marker = None

    state_first_pos  = _find_token_pos(prompt, loc_first, offsets)
    state_second_pos = _find_token_pos(prompt, loc_second, offsets)
    state_correct_pos = _find_token_pos(prompt, loc_correct, offsets)

    # `state` retained as alias of state_correct for back-compat with older
    # analyze scripts that read positions["state"].
    state_pos = state_correct_pos

    # Information-loss event (e.g. "is unable to see the truck"). Returns -1 in
    # ABB true-belief prompts where the marker phrase is absent — that absence
    # is informative for the universal-info-loss-head hypothesis.
    info_loss_pos = _find_token_pos(prompt, info_loss_marker or "", offsets)

    # char_q / obj_q: first occurrence of template-specific query strings.
    # Photo prompts replace the named character with an unnamed photographer,
    # so use "photographer" as the char_q lookup in that frame; the substring
    # matches "photographer" and "college photographer" alike.
    keys = TEMPLATE_QUERY_KEYS.get(template, {})
    char_str = "photographer" if context_type == "photo" else keys.get("char_q", "")
    char_pos = _find_token_pos(prompt, char_str, offsets)
    obj_pos = _find_token_pos(prompt, keys.get("obj_q", ""), offsets)

    # Agent first/last mention. For photo conditions the relevant "agent" is
    # the photographer (the source of epistemic access on the photo); for
    # belief / control conditions it's the named character. Last mention is
    # the most recent re-invocation of the agent before the question — the
    # natural query position for binding-update reads. First mention is the
    # introduction site, where binding is initially registered.
    agent_str = "photographer" if context_type == "photo" else char_str
    agent_mentions = _find_all_token_pos(prompt, agent_str, offsets)
    agent_first_pos = agent_mentions[0] if agent_mentions else -1
    agent_last_pos = agent_mentions[-1] if agent_mentions else -1

    # belief_formation / photo-event: probe the prompt directly for whichever
    # epistemic-access marker is present. No base_rule heuristic — the detector
    # finds the marker actually in the prompt, so cells where base/exp share a
    # template kind regardless of base_rule (e.g. C5 same_belief control) work
    # without special-casing.
    if context_type == "photo":
        bf_pos = generator.detect_photo_position(prompt, tokenizer)
    else:
        bf_pos = generator.detect_belief_formation_position(prompt, tokenizer)
    if bf_pos < 0 or bf_pos >= seq_len:
        bf_pos = -1

    # lookback: last input token (right before model generates)
    lookback_pos = seq_len - 1

    return {
        "state": state_pos,                  # alias of state_correct (back-compat)
        "state_first": state_first_pos,      # value at template slot {var_1} in base
        "state_second": state_second_pos,    # value at template slot {var_2} in base
        "state_correct": state_correct_pos,  # = base_ans (correct answer to base prompt)
        "info_loss": info_loss_pos,          # info-loss event marker; -1 if absent
        "char_q": char_pos,
        "obj_q": obj_pos,
        "belief_form": bf_pos,
        "lookback": lookback_pos,
        "agent_first": agent_first_pos,      # first mention of agent (intro / binding site)
        "agent_last": agent_last_pos,        # last mention of agent before question
        "agent_str": agent_str,
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
    log.info(f"top {len(heads_df)} heads (by signed observed_score, positive only): "
             f"min={heads_df.observed_score.min():.4g}")

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
    pair_metadata = getattr(generator, "last_metadata", None) or [None] * len(base_prompts)
    log.info(f"got {len(base_prompts)} base prompts")

    # Tokenize + extract positions
    positions = []
    for p, a, meta in zip(base_prompts, base_answers, pair_metadata):
        positions.append(extract_positions(p, a, template, tokenizer, generator,
                                          base_rule, pair_meta=meta,
                                          context_type=condition.context_type))

    max_seq = max(pos["seq_len"] for pos in positions)
    log.info(f"max seq len = {max_seq}")

    n_prompts = len(base_prompts)
    n_heads = len(heads_df)
    # Query rows we capture from. Order matters — downstream tools index by name
    # via query_pos_names but old payloads (n_query_pos=2) remain readable.
    query_pos_names = ["lookback", "belief_formation", "agent_last", "agent_first"]
    n_query_pos = len(query_pos_names)

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
        # Query positions in the order of query_pos_names. -1 / out-of-range
        # leaves NaN row (downstream filters those out).
        q_positions = [
            positions[pi]["lookback"],
            positions[pi]["belief_form"],
            positions[pi]["agent_last"],
            positions[pi]["agent_first"],
        ]

        for hi, (L, H) in enumerate(zip(head_layers, head_idxs)):
            key = f"blocks.{L}.attn.hook_pattern"
            if key not in cache:
                continue
            # shape (batch, n_heads, seq_q, seq_k)
            pattern = cache[key][0, H]  # (seq_q, seq_k)

            for qi, qpos in enumerate(q_positions):
                if 0 <= qpos < seq_len:
                    attn_out[pi, hi, qi, :seq_len] = pattern[qpos].float().cpu().to(torch.float16)

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
        "pct_state_first_found":   sum(1 for p in positions if p["state_first"] >= 0) / n_prompts,
        "pct_state_second_found":  sum(1 for p in positions if p["state_second"] >= 0) / n_prompts,
        "pct_state_correct_found": sum(1 for p in positions if p["state_correct"] >= 0) / n_prompts,
        "pct_info_loss_found":     sum(1 for p in positions if p["info_loss"] >= 0) / n_prompts,
        "pct_char_found": sum(1 for p in positions if p["char_q"] >= 0) / n_prompts,
        "pct_obj_found": sum(1 for p in positions if p["obj_q"] >= 0) / n_prompts,
        "pct_bf_found": sum(1 for p in positions if p["belief_form"] >= 0) / n_prompts,
        "pct_agent_first_found": sum(1 for p in positions if p["agent_first"] >= 0) / n_prompts,
        "pct_agent_last_found":  sum(1 for p in positions if p["agent_last"]  >= 0) / n_prompts,
        "agent_str": positions[0].get("agent_str", "") if positions else "",
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
