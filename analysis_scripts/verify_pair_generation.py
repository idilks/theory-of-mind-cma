"""Verify (base, exp, base_ans, exp_ans, causal_ans) prompt-pair generation.

Runs the template-system pair generator directly (no model, no tokenizer, no
chat template) for any condition. Prints a few examples per template +
direction and asserts the basic invariants:

  - same_belief conditions: base and exp scenarios must come from the SAME
    context template (both false_context_template) — only substitutions differ.
  - answer_differs conditions: base_ans != exp_ans for every pair.
  - !answer_differs conditions: base_ans == exp_ans for every pair.

Usage:
    python analysis_scripts/verify_pair_generation.py --condition answer_changes_location_only_photo
    python analysis_scripts/verify_pair_generation.py --condition answer_changes_location_only
    python analysis_scripts/verify_pair_generation.py --all
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from cma_config import CMACondition  # noqa: E402
from prompt_generators.tom_templates import UnifiedTemplateSystem  # noqa: E402

TEMPLATES = ["food_truck", "library_book", "hair_styling"]
DIRECTIONS = ["ABA", "ABB"]
N_EXAMPLES = 2


def verify_condition(condition: CMACondition, n_examples: int = N_EXAMPLES) -> int:
    print(f"\n{'=' * 72}")
    print(f"CONDITION: {condition.value}")
    print(f"  context_type    = {condition.context_type}")
    print(f"  answer_differs  = {condition.answer_differs}")
    print(f"  same_belief     = {condition.same_belief}")
    print(f"  within_context  = {condition.within_context}")
    print(f"{'=' * 72}")

    ctx = condition.context_type
    if ctx == "basic":
        print("  (basic context handled by generate_basic_pair, not exercised here)")
        return 0
    if ctx not in ("abstract", "photo"):
        print(f"  (verifier only covers abstract/photo, skipping ctx={ctx})")
        return 0

    ts = UnifiedTemplateSystem(vocab_dir=str(REPO_ROOT / "cma_scripts" / "tom_datasets"))
    failures = 0

    for template_name in TEMPLATES:
        template = ts.get_template(template_name, ctx)
        for base_rule in DIRECTIONS:
            for i in range(n_examples):
                base_scn, exp_scn, base_ans, exp_ans, causal_ans = ts.generate_cross_context_pair(
                    template_name=template_name,
                    base_rule=base_rule,
                    context_type=ctx,
                    answer_differs=condition.answer_differs,
                    same_belief=condition.same_belief,
                )

                tag = f"[{template_name} | {base_rule} | ex{i}]"
                print(f"\n{tag}")
                print(f"  BASE:        {base_scn}")
                print(f"  EXP:         {exp_scn}")
                print(f"  base_ans={base_ans!r}  exp_ans={exp_ans!r}  causal_ans={causal_ans!r}")

                if condition.answer_differs:
                    if base_ans == exp_ans:
                        print(f"  !! FAIL: answer_differs=True but base_ans == exp_ans")
                        failures += 1
                else:
                    if base_ans != exp_ans:
                        print(f"  !! FAIL: answer_differs=False but base_ans != exp_ans")
                        failures += 1

                if condition.same_belief:
                    base_from_false = base_scn.startswith(template.false_context_template.split("{")[0])
                    exp_from_false = exp_scn.startswith(template.false_context_template.split("{")[0])
                    if not (base_from_false and exp_from_false):
                        print(f"  !! FAIL: same_belief=True but scenarios don't both start from false_context_template prefix")
                        failures += 1

    print(f"\n{condition.value}: {failures} failure(s)")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", type=str, default=None,
                        help="condition value (e.g. answer_changes_location_only_photo)")
    parser.add_argument("--all", action="store_true",
                        help="run every CMACondition (where applicable)")
    parser.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    args = parser.parse_args()

    if not args.condition and not args.all:
        parser.error("specify --condition <name> or --all")

    if args.all:
        conds = list(CMACondition)
    else:
        conds = [CMACondition(args.condition)]

    total_failures = sum(verify_condition(c, args.n_examples) for c in conds)
    print(f"\n{'=' * 72}\nTOTAL FAILURES: {total_failures}")
    sys.exit(1 if total_failures else 0)


if __name__ == "__main__":
    main()
