"""
Diff one rendered false_belief prompt from the CMA pipeline against the
matching one from the ablation pipeline.

Both pipelines render the same scenario template + question + chat-template
wrapper. They differ in the system-prompt string. This script prints both
prompts and a unified diff so the divergence is visible at a glance.

Run locally (CPU only — only the tokenizer is needed):
    python scripts/diff_cma_vs_ablation_prompts.py

Optional flags:
    --template     food_truck | hair_styling | library_book  (default: food_truck)
    --seed         int (default: 42)  shared vocab sampling seed
    --model_type   tokenizer to load (default: Qwen2.5-14B-Instruct)
"""

import argparse
import difflib
import os
import random
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
codebase_root = os.path.join(project_root, "codebase")
for p in (project_root, codebase_root):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(codebase_root)

from transformers import AutoTokenizer

from tasks.causal_analysis.prompt_generators.tom_templates import (
    UnifiedTemplateSystem, UnifiedQuestionFormatter,
)


CMA_SYSTEM_PROMPT = "You are a helpful assistant reasoning about agents' beliefs."
ABLATION_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."


def render(tokenizer, system_prompt, user_content):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def sample_subs(template_system, template_name):
    """Sample one set of substitutions, mirroring what ablation does.

    vocab_loader.sample_different uses python's global `random`, so the
    caller is expected to have already seeded it.
    """
    abstract = template_system.get_template(template_name, "abstract")
    var_1 = next(v for v in abstract.variables if v.endswith("_1"))
    var_2 = next(v for v in abstract.variables if v.endswith("_2"))

    subs = {}
    for var_name, vocab_type in abstract.variables.items():
        subs[var_name] = template_system.vocab_loader.sample_different(vocab_type, 1)[0]
    vt = abstract.variables[var_1]
    v1, v2 = template_system.vocab_loader.sample_different(vt, 2)
    subs[var_1] = v1
    subs[var_2] = v2
    return subs, var_1, var_2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="food_truck",
                    choices=["food_truck", "hair_styling", "library_book"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_type", default="Qwen/Qwen2.5-14B-Instruct",
                    help="HF tokenizer id. Local path also works.")
    args = ap.parse_args()

    random.seed(args.seed)
    # NB: vocab_loader uses python's `random`, so seeding above is enough
    # to make both renders draw the same subs.

    print(f"Loading tokenizer: {args.model_type}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    template_system = UnifiedTemplateSystem(vocab_dir="tom_datasets")
    question_formatter = UnifiedQuestionFormatter(tokenizer=tokenizer)

    subs, var_1, var_2 = sample_subs(template_system, args.template)

    # --- shared scenario + question (both pipelines build these identically
    # for the false_belief variant) -----------------------------------------
    abstract = template_system.get_template(args.template, "abstract")
    template_subs = {k: v for k, v in subs.items() if k in abstract.variables}
    scenario = abstract.false_context_template.format(**template_subs)

    question = question_formatter.get_cma_questions(
        args.template, context_type="abstract", belief_type="ABA",
        question_style="instruction",
    )
    user_content = f"{scenario}\n{question}"

    cma_prompt = render(tokenizer, CMA_SYSTEM_PROMPT, user_content)
    abl_prompt = render(tokenizer, ABLATION_SYSTEM_PROMPT, user_content)

    print("\n" + "=" * 72)
    print("CMA prompt  (system = {!r})".format(CMA_SYSTEM_PROMPT))
    print("=" * 72)
    print(cma_prompt)

    print("\n" + "=" * 72)
    print("Ablation prompt  (system = {!r})".format(ABLATION_SYSTEM_PROMPT))
    print("=" * 72)
    print(abl_prompt)

    print("\n" + "=" * 72)
    print("Unified diff  (cma -> ablation)")
    print("=" * 72)
    diff = difflib.unified_diff(
        cma_prompt.splitlines(keepends=True),
        abl_prompt.splitlines(keepends=True),
        fromfile="cma", tofile="ablation", n=2,
    )
    sys.stdout.writelines(diff)

    cma_ids = tokenizer(cma_prompt, return_tensors=None).input_ids
    abl_ids = tokenizer(abl_prompt, return_tensors=None).input_ids
    print(f"\ntoken counts: cma={len(cma_ids)}  ablation={len(abl_ids)}  "
          f"delta={len(cma_ids) - len(abl_ids)}")


if __name__ == "__main__":
    main()
