"""
Code template system for CMA experiments.

Tests whether transformer attention heads track general divergent state
(not just beliefs) by using Python copy semantics as the divergence mechanism:
- Shallow reference (list_b = list_a) = true belief analog (state propagates)
- Deep copy (list_b = list_a[:]) = false belief analog (state diverges)

All code-specific helpers live here — position detection, answer extraction,
vocabulary, templates. Nothing touches the ToM files.
"""

import re
import ast
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Vocabulary (inline — no external txt files needed)
# ---------------------------------------------------------------------------

CODE_VOCAB = {
    "list_values": [1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 20, 25, 30, 42, 50, 99],
    "dict_keys": ["x", "y", "z", "a", "b", "m", "n", "p", "q", "k"],
    "dict_values": [1, 2, 3, 4, 5, 10, 20, 50, 100],
}


def sample_different(pool: list, count: int = 2) -> list:
    """Sample `count` distinct items from a pool."""
    if len(pool) < count:
        return random.choices(pool, k=count)
    return random.sample(pool, count)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that traces through Python code and "
    "determines variable values. Answer with only the Python value, nothing else."
)

CODE_QUESTIONS = {
    "list_append": "What is the value of list_b?",
    "dict_update": "What is the value of dict_b?",
    "list_extend": "What is the value of backup?",
}

CODE_CONTROL_QUESTIONS = {
    "list_append": "What is the value of list_a?",
    "dict_update": "What is the value of dict_a?",
    "list_extend": "What is the value of nums?",
}


# ---------------------------------------------------------------------------
# Template dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CodeScenarioTemplate:
    """Template for code-tracing CMA scenarios."""
    name: str
    false_context_template: str   # deep copy version (state diverges)
    true_context_template: str    # shallow reference version (state propagates)
    variables: Dict[str, str]     # variable_name -> vocab pool key in CODE_VOCAB
    copy_answer_key: str          # variable holding the "copy" (false) answer
    reference_answer_key: str     # variable holding the "reference" (true) answer
    copy_marker: str = ""         # regex pattern for deep-copy line
    reference_marker: str = ""    # regex pattern for shallow-reference line
    description: str = ""


@dataclass
class CodeControlTemplate:
    """Template for code control condition (no copy semantics)."""
    name: str
    template: str
    variables: Dict[str, str]
    answer_key: str
    description: str = ""


# ---------------------------------------------------------------------------
# CodeTemplateSystem
# ---------------------------------------------------------------------------

class CodeTemplateSystem:
    """Manages code scenario templates and generates CMA prompt pairs."""

    def __init__(self):
        self.templates = self._create_templates()
        self.control_templates = self._create_control_templates()

    # ---- template definitions ---------------------------------------------

    def _create_templates(self) -> Dict[str, CodeScenarioTemplate]:
        return {
            "list_append": CodeScenarioTemplate(
                name="list_append",
                false_context_template=(
                    "list_a = [{val_1}]\n"
                    "list_b = list_a[:]\n"
                    "list_a.append({val_2})"
                ),
                true_context_template=(
                    "list_a = [{val_1}]\n"
                    "list_b = list_a\n"
                    "list_a.append({val_2})"
                ),
                variables={"val_1": "list_values", "val_2": "list_values"},
                copy_answer_key="val_1",
                reference_answer_key="val_1",  # computed at runtime
                copy_marker=r"list_b = list_a\[:\]",
                reference_marker=r"list_b = list_a(?!\[)",
                description="List with append — copy vs reference",
            ),
            "dict_update": CodeScenarioTemplate(
                name="dict_update",
                false_context_template=(
                    'dict_a = {{"{key_1}": {val_1}}}\n'
                    "dict_b = dict(dict_a)\n"
                    'dict_a["{key_2}"] = {val_2}'
                ),
                true_context_template=(
                    'dict_a = {{"{key_1}": {val_1}}}\n'
                    "dict_b = dict_a\n"
                    'dict_a["{key_2}"] = {val_2}'
                ),
                variables={
                    "key_1": "dict_keys",
                    "val_1": "dict_values",
                    "key_2": "dict_keys",
                    "val_2": "dict_values",
                },
                copy_answer_key="key_1",
                reference_answer_key="key_1",
                copy_marker=r"dict_b = dict\(dict_a\)",
                reference_marker=r"dict_b = dict_a(?!\()",
                description="Dict with key addition — copy vs reference",
            ),
            "list_extend": CodeScenarioTemplate(
                name="list_extend",
                false_context_template=(
                    "nums = [{val_1}, {val_2}]\n"
                    "backup = nums[:]\n"
                    "nums.extend([{val_3}, {val_4}])"
                ),
                true_context_template=(
                    "nums = [{val_1}, {val_2}]\n"
                    "backup = nums\n"
                    "nums.extend([{val_3}, {val_4}])"
                ),
                variables={
                    "val_1": "list_values",
                    "val_2": "list_values",
                    "val_3": "list_values",
                    "val_4": "list_values",
                },
                copy_answer_key="val_1",
                reference_answer_key="val_1",
                copy_marker=r"backup = nums\[:\]",
                reference_marker=r"backup = nums(?!\[)",
                description="List with extend — copy vs reference",
            ),
        }

    def _create_control_templates(self) -> Dict[str, CodeControlTemplate]:
        """Control templates: no copy semantics, just direct mutation."""
        return {
            "list_append": CodeControlTemplate(
                name="list_append",
                template=(
                    "list_a = [{val_1}]\n"
                    "list_a.append({val_2})"
                ),
                variables={"val_1": "list_values", "val_2": "list_values"},
                answer_key="val_1",
                description="List append — direct mutation, no copy",
            ),
            "dict_update": CodeControlTemplate(
                name="dict_update",
                template=(
                    'dict_a = {{"{key_1}": {val_1}}}\n'
                    'dict_a["{key_2}"] = {val_2}'
                ),
                variables={
                    "key_1": "dict_keys",
                    "val_1": "dict_values",
                    "key_2": "dict_keys",
                    "val_2": "dict_values",
                },
                answer_key="key_1",
                description="Dict update — direct mutation, no copy",
            ),
            "list_extend": CodeControlTemplate(
                name="list_extend",
                template=(
                    "nums = [{val_1}, {val_2}]\n"
                    "nums.extend([{val_3}, {val_4}])"
                ),
                variables={
                    "val_1": "list_values",
                    "val_2": "list_values",
                    "val_3": "list_values",
                    "val_4": "list_values",
                },
                answer_key="val_1",
                description="List extend — direct mutation, no copy",
            ),
        }

    # ---- answer computation -----------------------------------------------

    def _compute_answer(self, template_name: str, substitutions: dict, is_copy: bool) -> str:
        """Compute the expected Python value as a string.

        For copy (false belief analog): the tracked variable keeps its original value.
        For reference (true belief analog): the tracked variable reflects the mutation.
        """
        if template_name == "list_append":
            if is_copy:
                return f"[{substitutions['val_1']}]"
            else:
                return f"[{substitutions['val_1']}, {substitutions['val_2']}]"

        elif template_name == "dict_update":
            k1, v1 = substitutions["key_1"], substitutions["val_1"]
            k2, v2 = substitutions["key_2"], substitutions["val_2"]
            if is_copy:
                return '{' + f'"{k1}": {v1}' + '}'
            else:
                return '{' + f'"{k1}": {v1}, "{k2}": {v2}' + '}'

        elif template_name == "list_extend":
            v1, v2 = substitutions["val_1"], substitutions["val_2"]
            v3, v4 = substitutions["val_3"], substitutions["val_4"]
            if is_copy:
                return f"[{v1}, {v2}]"
            else:
                return f"[{v1}, {v2}, {v3}, {v4}]"

        raise ValueError(f"Unknown template: {template_name}")

    def _compute_control_answer(self, template_name: str, substitutions: dict) -> str:
        """Compute the expected answer for control templates (always the mutated value)."""
        if template_name == "list_append":
            return f"[{substitutions['val_1']}, {substitutions['val_2']}]"
        elif template_name == "dict_update":
            k1, v1 = substitutions["key_1"], substitutions["val_1"]
            k2, v2 = substitutions["key_2"], substitutions["val_2"]
            return '{' + f'"{k1}": {v1}, "{k2}": {v2}' + '}'
        elif template_name == "list_extend":
            v1, v2 = substitutions["val_1"], substitutions["val_2"]
            v3, v4 = substitutions["val_3"], substitutions["val_4"]
            return f"[{v1}, {v2}, {v3}, {v4}]"
        raise ValueError(f"Unknown template: {template_name}")

    # ---- variable sampling ------------------------------------------------

    def _sample_substitutions(self, template_name: str, is_control: bool = False) -> dict:
        """Sample variable values for a template."""
        if is_control:
            template = self.control_templates[template_name]
        else:
            template = self.templates[template_name]

        subs = {}
        for var_name, pool_key in template.variables.items():
            pool = CODE_VOCAB[pool_key]
            subs[var_name] = random.choice(pool)

        # Ensure distinctness for key contrastive pairs
        if template_name == "list_append":
            vals = sample_different(CODE_VOCAB["list_values"], 2)
            subs["val_1"], subs["val_2"] = vals[0], vals[1]
        elif template_name == "dict_update":
            keys = sample_different(CODE_VOCAB["dict_keys"], 2)
            vals = sample_different(CODE_VOCAB["dict_values"], 2)
            subs["key_1"], subs["key_2"] = keys[0], keys[1]
            subs["val_1"], subs["val_2"] = vals[0], vals[1]
        elif template_name == "list_extend":
            vals = sample_different(CODE_VOCAB["list_values"], 4)
            subs["val_1"], subs["val_2"] = vals[0], vals[1]
            subs["val_3"], subs["val_4"] = vals[2], vals[3]

        return subs

    # ---- pair generation --------------------------------------------------

    def generate_cross_context_pair(
        self, template_name: str, base_rule: str, answer_differs: bool
    ) -> Tuple[str, str, str, str, str]:
        """Generate a base/exp code prompt pair for CMA.

        Returns:
            (base_scenario, exp_scenario, base_ans, exp_ans, causal_ans)

        For answer_differs=True (CODE_ANSWER_CHANGES):
            Same variable values, different copy semantics.
            ABA: base=copy(false), exp=reference(true)
            ABB: base=reference(true), exp=copy(false)

        For control (CODE_CONTROL):
            Different variable values, same direct-assignment structure.
        """
        if not answer_differs and template_name in self.control_templates:
            return self._generate_control_pair(template_name, base_rule)

        template = self.templates[template_name]
        subs = self._sample_substitutions(template_name)

        if base_rule == "ABA":
            # base = copy (false belief analog), exp = reference (true belief analog)
            base_scenario = template.false_context_template.format(**subs)
            exp_scenario = template.true_context_template.format(**subs)
            base_ans = self._compute_answer(template_name, subs, is_copy=True)
            exp_ans = self._compute_answer(template_name, subs, is_copy=False)
            causal_ans = base_ans  # after patching, expect base answer
        elif base_rule == "ABB":
            # base = reference (true), exp = copy (false)
            base_scenario = template.true_context_template.format(**subs)
            exp_scenario = template.false_context_template.format(**subs)
            base_ans = self._compute_answer(template_name, subs, is_copy=False)
            exp_ans = self._compute_answer(template_name, subs, is_copy=True)
            causal_ans = base_ans
        else:
            raise ValueError(f"base_rule must be ABA or ABB, got {base_rule}")

        return base_scenario, exp_scenario, base_ans, exp_ans, causal_ans

    def _generate_control_pair(
        self, template_name: str, base_rule: str
    ) -> Tuple[str, str, str, str, str]:
        """Generate a control pair: same structure, different values."""
        template = self.control_templates[template_name]

        subs_1 = self._sample_substitutions(template_name, is_control=True)
        subs_2 = self._sample_substitutions(template_name, is_control=True)

        # Ensure the two sets produce different answers
        while self._compute_control_answer(template_name, subs_1) == self._compute_control_answer(template_name, subs_2):
            subs_2 = self._sample_substitutions(template_name, is_control=True)

        base_scenario = template.template.format(**subs_1)
        exp_scenario = template.template.format(**subs_2)
        base_ans = self._compute_control_answer(template_name, subs_1)
        exp_ans = self._compute_control_answer(template_name, subs_2)
        causal_ans = base_ans

        return base_scenario, exp_scenario, base_ans, exp_ans, causal_ans


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_value_from_code_response(response: str) -> str:
    """Extract a Python value from the model's response.

    Looks for list literals [...], dict literals {...}, or bare numbers.
    """
    # Try list literal
    list_match = re.search(r'\[[\d\s,]*\]', response)
    if list_match:
        return list_match.group(0).strip()

    # Try dict literal
    dict_match = re.search(r'\{[^}]+\}', response)
    if dict_match:
        return dict_match.group(0).strip()

    # Bare number
    num_match = re.search(r'\b\d+\b', response)
    if num_match:
        return num_match.group(0)

    return ""


def code_values_match(predicted: str, expected: str) -> bool:
    """Compare Python value strings semantically.

    Uses ast.literal_eval so whitespace and dict ordering don't matter.
    """
    if not predicted or not expected:
        return False
    try:
        pred_val = ast.literal_eval(predicted.strip())
        exp_val = ast.literal_eval(expected.strip())
        return pred_val == exp_val
    except (ValueError, SyntaxError):
        # Fallback: normalize whitespace and compare
        return predicted.replace(" ", "") == expected.replace(" ", "")


# ---------------------------------------------------------------------------
# Position detection
# ---------------------------------------------------------------------------

def detect_copy_semantics_position(
    prompt: str, tokenizer, is_copy: bool, templates: Dict[str, CodeScenarioTemplate]
) -> int:
    """Detect token position where copy/reference semantics are established.

    Searches for the copy or reference marker in the prompt, maps the end
    of the match to a token position using offset_mapping.

    Args:
        prompt: Full formatted prompt string.
        tokenizer: HF tokenizer with offset_mapping support.
        is_copy: True for deep copy (false belief analog), False for reference.
        templates: Dict of CodeScenarioTemplate instances to get markers from.

    Returns:
        Token position (int) after the marker, or -1 if not found.
    """
    for template in templates.values():
        pattern = template.copy_marker if is_copy else template.reference_marker
        if not pattern:
            continue

        match = re.search(pattern, prompt)
        if match is None:
            continue

        end_char = match.end()
        encoding = tokenizer(prompt, return_offsets_mapping=True)

        for i, (start, end) in enumerate(encoding["offset_mapping"]):
            if start <= end_char <= end:
                return i + 1
            elif start > end_char:
                return i

        return len(encoding["offset_mapping"])

    return -1
