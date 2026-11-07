"""
Local validation: confirm the photo-frame bf detector lands on a real token in
photo prompts after the capture-script fix. No model load required; tokenizer only.

Run:
  python analysis_scripts/lookback_analysis/test_photo_detector.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from transformers import AutoTokenizer

from cma_config import (
    CMACondition, EvaluationConfig, ExperimentConfig, GenerationConfig,
    ModelConfig, PatchingConfig, PromptConfig, WandbConfig,
)
from prompt_generators.unified_tom import UnifiedTomGenerator

# import the capture script's extract_positions so we test the real code path
sys.path.insert(0, str(ROOT / "analysis_scripts" / "lookback_analysis"))
from capture_attention_patterns import extract_positions


def build_cfg(template: str, base_rule: str, condition: CMACondition) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(model_type="Qwen2.5-14B-Instruct", device_map="cpu", n_devices=1),
        generation=GenerationConfig(max_new_tokens=5),
        prompts=PromptConfig(
            base_rule=base_rule,
            context_type=condition.context_type,
            prompt_num=5,
            condition=condition,
            use_template_system=True,
            question_style="instruction",
            template_names=[template],
        ),
        patching=PatchingConfig(activation_name="z"),
        evaluation=EvaluationConfig(),
        wandb=WandbConfig(enabled=False),
        seed=1,
    )


def show_positions(label: str, condition: CMACondition, template: str, base_rule: str,
                   tokenizer, n: int = 3):
    print(f"\n========== {label} ==========")
    print(f"  condition={condition.value}  template={template}  base_rule={base_rule}")
    print(f"  context_type={condition.context_type}")
    cfg = build_cfg(template, base_rule, condition)
    gen = UnifiedTomGenerator(cfg, tokenizer)
    pairs, ans_pairs, _ = gen.generate_prompts()
    metas = getattr(gen, "last_metadata", None) or [None] * len(pairs)

    for i, (pair, ans_pair, meta) in enumerate(zip(pairs[:n], ans_pairs[:n], metas[:n])):
        prompt, _ = pair
        base_ans, _ = ans_pair
        positions = extract_positions(
            prompt, base_ans, template, tokenizer, gen,
            base_rule=base_rule, pair_meta=meta,
            context_type=condition.context_type,
        )
        bf = positions["belief_form"]
        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        bf_tok = tokenizer.decode([tokens[bf]]) if 0 <= bf < len(tokens) else "<not found>"
        state_tok = tokenizer.decode([tokens[positions["state"]]]) if 0 <= positions["state"] < len(tokens) else "<not found>"
        print(f"  --- prompt {i} ({len(tokens)} tokens) ---")
        print(f"    bf_pos={bf:4d}  -> token={bf_tok!r}")
        print(f"    state_pos={positions['state']:4d}  -> token={state_tok!r}  (base_ans={base_ans!r})")
        # show 4-token context around bf for sanity
        if 0 <= bf < len(tokens):
            lo, hi = max(0, bf - 3), min(len(tokens), bf + 2)
            ctx = tokenizer.decode(tokens[lo:hi])
            print(f"    bf context (tokens {lo}..{hi-1}): {ctx!r}")


def main():
    print("Loading tokenizer (Qwen/Qwen2.5-14B-Instruct) — no model load")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

    cases = [
        # (label, condition, template, base_rule)
        ("BELIEF baseline (sanity check, should already work)",
         CMACondition.ABSTRACT_BELIEF, "food_truck", "ABA"),
        ("PHOTO food_truck ABA (should now resolve photo-event token)",
         CMACondition.ABSTRACT_PHOTO, "food_truck", "ABA"),
        ("PHOTO food_truck ABB",
         CMACondition.ABSTRACT_PHOTO, "food_truck", "ABB"),
        ("PHOTO hair_styling ABA",
         CMACondition.ABSTRACT_PHOTO, "hair_styling", "ABA"),
        ("PHOTO library_book ABA",
         CMACondition.ABSTRACT_PHOTO, "library_book", "ABA"),
    ]
    for label, cond, tmpl, br in cases:
        try:
            show_positions(label, cond, tmpl, br, tokenizer, n=2)
        except Exception as e:
            print(f"\n!! {label} failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
