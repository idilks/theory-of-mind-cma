"""
Baseline generation accuracy for code-tracing CMA prompts (no patching).

Smoke-tests whether the model can answer copy-vs-reference Python tracing
questions before we commit to running CMA on them. If accuracy here is low,
CMA results on these prompts are uninterpretable.

Usage:
    python scripts/code_baseline_acc.py \
        --model_type Qwen2.5-14B-Instruct \
        --prompt_num 50 \
        --templates list_append dict_update list_extend \
        --output results/code_baseline/acc.json
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "codebase" / "tasks" / "causal_analysis"))

from prompt_generators.code_templates import (
    CODE_SYSTEM_PROMPT,
    CODE_QUESTIONS,
    CODE_CONTROL_QUESTIONS,
    CodeTemplateSystem,
    code_values_match,
    extract_value_from_code_response,
)


MODEL_IDS = {
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
}


def build_prompt(tokenizer, code: str, question: str) -> str:
    user = (
        "Consider the following Python code:\n\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        f"{question}"
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_id = MODEL_IDS.get(args.model_type, args.model_type)
    print(f"[load] model={model_id}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    ts = CodeTemplateSystem()

    records = []
    by_template = {}

    for template_name in args.templates:
        if template_name not in ts.templates:
            raise ValueError(f"unknown template: {template_name}")

        question = CODE_QUESTIONS[template_name]
        control_question = CODE_CONTROL_QUESTIONS[template_name]

        per_template_correct = {"copy": 0, "reference": 0, "control": 0}
        per_template_total = {"copy": 0, "reference": 0, "control": 0}

        for i in range(args.prompt_num):
            subs = ts._sample_substitutions(template_name)
            tmpl = ts.templates[template_name]

            # copy variant (false belief analog)
            code_copy = tmpl.false_context_template.format(**subs)
            ans_copy = ts._compute_answer(template_name, subs, is_copy=True)

            # reference variant (true belief analog)
            code_ref = tmpl.true_context_template.format(**subs)
            ans_ref = ts._compute_answer(template_name, subs, is_copy=False)

            # control variant (just the mutated source)
            ctrl_subs = ts._sample_substitutions(template_name, is_control=True)
            ctrl_tmpl = ts.control_templates[template_name]
            code_ctrl = ctrl_tmpl.template.format(**ctrl_subs)
            ans_ctrl = ts._compute_control_answer(template_name, ctrl_subs)

            for variant, code, expected, q in [
                ("copy", code_copy, ans_copy, question),
                ("reference", code_ref, ans_ref, question),
                ("control", code_ctrl, ans_ctrl, control_question),
            ]:
                prompt = build_prompt(tokenizer, code, q)
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                gen = tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                extracted = extract_value_from_code_response(gen)
                ok = code_values_match(extracted, expected)

                per_template_total[variant] += 1
                per_template_correct[variant] += int(ok)

                records.append({
                    "template": template_name,
                    "variant": variant,
                    "code": code,
                    "expected": expected,
                    "raw_response": gen,
                    "extracted": extracted,
                    "correct": ok,
                })

                if i < 2:
                    print(f"[{template_name}/{variant}] expected={expected!r} "
                          f"got={extracted!r} raw={gen!r} ok={ok}", flush=True)

        by_template[template_name] = {
            v: {
                "correct": per_template_correct[v],
                "total": per_template_total[v],
                "acc": per_template_correct[v] / max(1, per_template_total[v]),
            }
            for v in ("copy", "reference", "control")
        }
        print(f"[summary] {template_name}: {by_template[template_name]}", flush=True)

    overall = {}
    for variant in ("copy", "reference", "control"):
        c = sum(by_template[t][variant]["correct"] for t in args.templates)
        n = sum(by_template[t][variant]["total"] for t in args.templates)
        overall[variant] = {"correct": c, "total": n, "acc": c / max(1, n)}

    print(f"[OVERALL] {overall}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": model_id,
                "seed": args.seed,
                "prompt_num_per_template": args.prompt_num,
                "by_template": by_template,
                "overall": overall,
                "records": records,
            },
            f,
            indent=2,
        )
    print(f"[saved] {out_path}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, default="Qwen2.5-14B-Instruct")
    p.add_argument("--prompt_num", type=int, default=50,
                   help="Prompts per template per variant (copy/reference/control)")
    p.add_argument("--templates", nargs="+",
                   default=["list_append", "dict_update", "list_extend"])
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=str,
                   default="results/code_baseline/acc.json")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
