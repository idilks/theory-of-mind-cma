"""Quick test to verify photo prompt generation logic."""

from transformers import AutoTokenizer
from prompt_generators.tom_templates import UnifiedTemplateSystem

# Mock tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Create template system
template_system = UnifiedTemplateSystem("../../../tom_datasets")

# Test ABA (false photo -> true photo with swapped locations)
print("="*80)
print("Testing ABA (base=false photo, exp=true photo, swapped locations)")
print("="*80)

for i in range(2):
    print(f"\n--- Prompt Pair {i+1} ---")

    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans = template_system.generate_cross_context_pair(
        template_name="food_truck",
        base_rule="ABA",
        context_type="photo",
        patch_after_movement=False  # using abstract logic
    )

    print(f"\nBASE (false photo):")
    print(base_scenario)
    print(f"Base answer: {base_ans}")

    print(f"\nEXP (true photo):")
    print(exp_scenario)
    print(f"Exp answer: {exp_ans}")

    print(f"\nCAUSAL answer (after patching base->exp): {causal_ans}")

    # Verify expectations
    print("\nChecking logic:")
    print(f"  - Base scenario should be: truck at A, photo, moves to B")
    print(f"  - Exp scenario should be: truck at B, moves to A, photo")
    print(f"  - Base ans = Exp ans? {base_ans == exp_ans} (should be True)")
    print(f"  - Causal ans different from base? {causal_ans != base_ans} (should be True)")

print("\n" + "="*80)
print("Testing ABB (base=true photo, exp=false photo, swapped locations)")
print("="*80)

for i in range(2):
    print(f"\n--- Prompt Pair {i+1} ---")

    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans = template_system.generate_cross_context_pair(
        template_name="food_truck",
        base_rule="ABB",
        context_type="photo",
        patch_after_movement=False  # using abstract logic
    )

    print(f"\nBASE (true photo):")
    print(base_scenario)
    print(f"Base answer: {base_ans}")

    print(f"\nEXP (false photo):")
    print(exp_scenario)
    print(f"Exp answer: {exp_ans}")

    print(f"\nCAUSAL answer (after patching base->exp): {causal_ans}")

    # Verify expectations
    print("\nChecking logic:")
    print(f"  - Base scenario should be: truck at A, moves to B, photo")
    print(f"  - Exp scenario should be: truck at B, photo, moves to A")
    print(f"  - Base ans = Exp ans? {base_ans == exp_ans} (should be True)")
    print(f"  - Causal ans different from base? {causal_ans != base_ans} (should be True)")
