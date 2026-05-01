"""Pre-flight verification for C8 (ANSWER_CHANGES_LOCATION_ONLY) prompts.

Builds base/exp prompts for each template × direction, asserts the expected
structural invariants (both prompts use false_context, swapped subs, different
answers, belief-formation marker resolvable in both), and prints one full
pair per template for eyeball inspection.

Run:
    python scripts/verify_c6_prompts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from prompt_generators.tom_templates import UnifiedTemplateSystem

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
RULES = ["ABA", "ABB"]


def main():
    ts = UnifiedTemplateSystem(vocab_dir=str(ROOT / "tom_datasets"))
    all_ok = True
    for tmpl in TEMPLATES:
        tpl = ts.get_template(tmpl, "abstract")
        marker = tpl.false_belief_marker
        for rule in RULES:
            base, exp, base_ans, exp_ans, causal = ts.generate_cross_context_pair(
                tmpl, base_rule=rule, context_type="abstract",
                answer_differs=True, same_belief=True,
            )
            checks = {
                "base has false marker": marker in base,
                "exp has false marker": marker in exp,
                "base lacks true marker": (not tpl.true_belief_marker) or (tpl.true_belief_marker not in base),
                "exp lacks true marker": (not tpl.true_belief_marker) or (tpl.true_belief_marker not in exp),
                "answers differ": base_ans != exp_ans,
                "causal_ans == base_ans": causal == base_ans,
            }
            ok = all(checks.values())
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {tmpl} {rule}: base_ans={base_ans!r} exp_ans={exp_ans!r} causal={causal!r}")
            for k, v in checks.items():
                if not v:
                    print(f"      !! {k}")
        # Eyeball dump, ABA direction
        base, exp, *_ = ts.generate_cross_context_pair(
            tmpl, base_rule="ABA", context_type="abstract",
            answer_differs=True, same_belief=True,
        )
        print(f"\n--- {tmpl} ABA base ---\n{base}\n--- {tmpl} ABA exp ---\n{exp}\n")
    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
