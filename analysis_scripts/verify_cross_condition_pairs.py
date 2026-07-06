"""
Verification script for cross-condition CMA prompt pairs.

Generates matched (false-photo, false-belief) prompts that share the same
variable assignments, so both prompts yield the same answer (e.g. both
answer "canal street"). Prints pairs side-by-side and checks:
  1. answer string is identical across the pair
  2. answer-token IDs are identical when tokenized with leading space
     (skipped if the Qwen tokenizer isn't loadable)

Run from project root:
    source .venv/bin/activate
    python scripts/verify_cross_condition_pairs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from codebase.tasks.causal_analysis.prompt_generators.tom_templates import (
    UnifiedTemplateSystem,
)

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
N_PAIRS_PER_TEMPLATE = 2
TOKENIZER_ID = "Qwen/Qwen2.5-14B-Instruct"


def sample_matched_subs(ts, template_name):
    """Sample one substitutions dict usable by both belief and photo templates.

    Belief and photo templates can have overlapping but non-identical variable
    sets (e.g. hair_styling belief has {gift_item}, photo doesn't). We sample
    the union, then each renderer takes the subset it needs.

    Returns (subs, var_1, var_2). var_1/var_2 are the contrastive *_1/*_2 slots
    shared by both templates (both answer keys point at *_1 in the false branch).
    """
    belief_tpl = ts.get_template(template_name, "abstract")
    photo_tpl = ts.get_template(template_name, "photo")

    var_1 = next(v for v in belief_tpl.variables if v.endswith("_1"))
    var_2 = next(v for v in belief_tpl.variables if v.endswith("_2"))

    # Sanity: photo template must use the same contrastive slot names and vocab
    assert var_1 in photo_tpl.variables, \
        f"{template_name}: belief var_1={var_1} missing from photo template"
    assert var_2 in photo_tpl.variables, \
        f"{template_name}: belief var_2={var_2} missing from photo template"
    assert belief_tpl.variables[var_1] == photo_tpl.variables[var_1], \
        f"{template_name}: vocab type mismatch on {var_1}"
    assert belief_tpl.variables[var_2] == photo_tpl.variables[var_2], \
        f"{template_name}: vocab type mismatch on {var_2}"
    assert belief_tpl.belief_answer_key == var_1, \
        f"{template_name}: belief_answer_key {belief_tpl.belief_answer_key} != var_1 {var_1}"
    assert photo_tpl.photo_answer_key == var_1, \
        f"{template_name}: photo_answer_key {photo_tpl.photo_answer_key} != var_1 {var_1}"

    # Sample the contrastive pair from the shared vocab
    vocab_type = belief_tpl.variables[var_1]
    v1, v2 = ts.vocab_loader.sample_different(vocab_type, 2)
    subs = {var_1: v1, var_2: v2}

    # Sample all remaining variables (union over both templates)
    all_vars = {**belief_tpl.variables, **photo_tpl.variables}
    for var, vt in all_vars.items():
        if var in subs:
            continue
        subs[var] = ts.vocab_loader.sample_different(vt, 1)[0]

    return subs, var_1, var_2


def render_pair(ts, template_name, subs, var_1):
    """Render the false-belief and false-photo prompts for a single subs set.

    Both render the `false_context_template` branch. In that branch the answer
    key resolves to subs[var_1] for both belief (belief_answer_key=var_1) and
    photo (photo_answer_key=var_1) — so the answer strings are identical by
    construction.
    """
    belief_tpl = ts.get_template(template_name, "abstract")
    photo_tpl = ts.get_template(template_name, "photo")

    belief_subs = {k: v for k, v in subs.items() if k in belief_tpl.variables}
    photo_subs = {k: v for k, v in subs.items() if k in photo_tpl.variables}

    belief_scenario = belief_tpl.false_context_template.format(**belief_subs)
    photo_scenario = photo_tpl.false_context_template.format(**photo_subs)

    belief_answer = belief_subs[var_1]
    photo_answer = photo_subs[var_1]

    return belief_scenario, photo_scenario, belief_answer, photo_answer


def try_load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
        return tok, None
    except Exception as e:
        return None, str(e)


def main():
    ts = UnifiedTemplateSystem(vocab_dir="tom_datasets")

    tokenizer, err = try_load_tokenizer()
    if tokenizer is None:
        print(f"[WARN] Could not load tokenizer ({err});")
        print("[WARN] skipping token-ID checks. Install/auth transformers + HF_TOKEN to enable.\n")
    else:
        print(f"[OK] Loaded tokenizer: {TOKENIZER_ID}\n")

    all_ok = True
    failures = []

    for template_name in TEMPLATES:
        print(f"\n{'='*80}\nTEMPLATE: {template_name}\n{'='*80}")
        for i in range(N_PAIRS_PER_TEMPLATE):
            subs, var_1, var_2 = sample_matched_subs(ts, template_name)
            belief, photo, b_ans, p_ans = render_pair(ts, template_name, subs, var_1)

            print(f"\n--- Pair {i+1} ---")
            print(f"Substitutions: {subs}")
            print(f"  var_1 ({var_1}) = {subs[var_1]!r}  <- shared answer")
            print(f"  var_2 ({var_2}) = {subs[var_2]!r}")
            print()
            print(f"FALSE-BELIEF scenario:\n{belief}")
            print()
            print(f"FALSE-PHOTO scenario:\n{photo}")
            print()
            print(f"Belief answer: {b_ans!r}")
            print(f"Photo  answer: {p_ans!r}")

            if b_ans != p_ans:
                print("  [FAIL] answer strings differ")
                all_ok = False
                failures.append(f"{template_name} pair {i+1}: string mismatch")
            else:
                print("  [OK]  answer strings identical")

            if tokenizer is not None:
                ans_with_space = " " + b_ans
                tok_ids = tokenizer.encode(ans_with_space, add_special_tokens=False)
                decoded = [tokenizer.decode([t]) for t in tok_ids]
                print(f"  answer tokenized (leading space): ids={tok_ids} decoded={decoded}")

                # also check that the answer tokenizes the same way when extracted
                # from inside each scenario (to catch context-fusion BPE quirks
                # that could break logit reads)
                belief_ids = tokenizer.encode(" " + b_ans, add_special_tokens=False)
                photo_ids = tokenizer.encode(" " + p_ans, add_special_tokens=False)
                if belief_ids != photo_ids:
                    print(f"  [FAIL] token ID mismatch: belief={belief_ids} photo={photo_ids}")
                    all_ok = False
                    failures.append(f"{template_name} pair {i+1}: token-id mismatch")
                else:
                    print("  [OK]  answer token IDs identical across pair")

    print(f"\n{'='*80}")
    if all_ok:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print(f"OVERALL: {len(failures)} failure(s)")
        for f in failures:
            print(f"  - {f}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
