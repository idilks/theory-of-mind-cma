"""
Code template system for CMA experiments.

Single template: indexed-list mutation. A list `a` is initialized with
random fillers, the target index holds the "orig" value, then `b` is
either an alias (`b = a`, true-belief analog: state propagates) or a deep
copy (`b = a.copy()`, false-belief analog: state diverges), then `a[idx]`
is mutated. The question asks `b[idx]`.

Pair generators hold causal_ans constant across all three conditions
(abstract, answer_changes, control) for a given direction:
- ABA → causal = X (the "non-base" letter for that pair)
- ABB → causal = X
This double-dissociates rule effects from token-position effects.

Worked example (abstract condition, ABA, idx=2, Y='C', X='X'; answers SAME):

    base (copy / false-belief analog):     exp (alias / true-belief analog):
    a = ['A', 'B', 'C', 'D']               a = ['A', 'B', 'X', 'D']
    b = a.copy()                           b = a
    a[2] = 'X'                             a[2] = 'C'
    What is b[2]?  -> 'C'                  What is b[2]?  -> 'C'

Both answer 'C'; patching base's copy semantics into exp makes b[2] the
pre-mutation value 'X' — causal_ans = 'X', distinct from the surface answer.
For answer_changes the values are NOT swapped, so base/exp answers differ.

Answer tokenization constraint: answers are rendered as quoted chars ('C').
Most quote+char sequences tokenize with a bare leading ' token, making the
first-token logit diff between causal and original answers identically 0.
`filter_answer_pool` keeps only chars whose quoted form starts with a merged
'{c} token; CodeGenerator samples the answer pair (x, y) from that subset
(fillers still draw from the full pool).
"""

import re
import ast
import random
from dataclasses import dataclass
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

CODE_VALUE_POOL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def filter_answer_pool(tokenizer, pool=None):
    """Chars whose quoted form '{c}' tokenizes as ['{c}, '] — i.e. the first
    token is a merged quote+char token, distinct across chars. Answers (x, y)
    must come from this subset or the first-token logit diff degenerates to
    comparing ' against ' (identically 0).
    """
    pool = pool if pool is not None else CODE_VALUE_POOL
    kept = [c for c in pool
            if len(tokenizer(f"'{c}'", add_special_tokens=False)["input_ids"]) == 2]
    if len(kept) < 2:
        raise ValueError(
            f"Answer pool collapsed to {kept} under this tokenizer; "
            "need >= 2 chars with merged quote+char first tokens.")
    return kept


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that traces through Python code and "
    "determines variable values. Answer with only the Python value, nothing else."
)

# Patch-position markers.
COPY_MARKER = r"b = a\.copy\(\)"
ALIAS_MARKER = r"b = a(?!\.)"


# ---------------------------------------------------------------------------
# Template dataclass (one entry; consumed by detect_copy_semantics_position)
# ---------------------------------------------------------------------------

@dataclass
class CodeScenarioTemplate:
    name: str
    copy_marker: str
    reference_marker: str
    description: str = ""


# ---------------------------------------------------------------------------
# CodeTemplateSystem
# ---------------------------------------------------------------------------

class CodeTemplateSystem:
    """Generates indexed-list code-tracing CMA prompt pairs."""

    def __init__(self, length_range=(4, 8), pool=None, answer_pool=None):
        self.length_range = length_range
        self.pool = pool if pool is not None else CODE_VALUE_POOL
        self.answer_pool = answer_pool if answer_pool is not None else self.pool
        self.templates = {
            "indexed_list": CodeScenarioTemplate(
                name="indexed_list",
                copy_marker=COPY_MARKER,
                reference_marker=ALIAS_MARKER,
                description="Indexed-list mutation: alias vs deep-copy semantics",
            ),
        }

    # ---- prompt rendering -------------------------------------------------

    def _sample_pair_setup(self):
        """Shared per-pair setup: list length, target idx, distinct (X, Y),
        and filler values for non-target slots."""
        length = random.randint(*self.length_range)
        idx = random.randint(0, length - 1)
        x, y = random.sample(self.answer_pool, 2)
        fillers = [random.choice(self.pool) for _ in range(length)]
        return length, idx, x, y, fillers

    @staticmethod
    def _render_init(fillers, idx, orig_value):
        items = list(fillers)
        items[idx] = orig_value
        return "[" + ", ".join(f"'{c}'" for c in items) + "]"

    def _render_scenario(self, fillers, idx, orig, kind, mut):
        body = "b = a.copy()" if kind == "copy" else "b = a"
        init = self._render_init(fillers, idx, orig)
        return f"a = {init}\n{body}\na[{idx}] = '{mut}'"

    @staticmethod
    def _quoted(c):
        return f"'{c}'"

    @staticmethod
    def _question(idx):
        return f"What is b[{idx}]?"

    def _check_template(self, template_name):
        if template_name not in self.templates:
            raise ValueError(
                f"Unknown code template: {template_name}. "
                f"Available: {list(self.templates.keys())}"
            )

    # ---- pair generators --------------------------------------------------

    def generate_abstract_pair(
        self, template_name: str, base_rule: str
    ) -> Tuple[str, str, str, str, str, str]:
        """Abstract: base_ans == exp_ans (=Y); causal=X.

        ABA: base=(orig=Y, copy, mut=X); exp=(orig=X, alias, mut=Y)
        ABB: base=(orig=X, alias, mut=Y); exp=(orig=Y, copy, mut=X)
        """
        self._check_template(template_name)
        _, idx, x, y, fillers = self._sample_pair_setup()

        if base_rule == "ABA":
            base = self._render_scenario(fillers, idx, y, "copy", x)
            exp = self._render_scenario(fillers, idx, x, "alias", y)
        elif base_rule == "ABB":
            base = self._render_scenario(fillers, idx, x, "alias", y)
            exp = self._render_scenario(fillers, idx, y, "copy", x)
        else:
            raise ValueError(f"base_rule must be ABA or ABB, got {base_rule}")

        return (base, exp,
                self._quoted(y), self._quoted(y),
                self._quoted(x), self._question(idx))

    def generate_cross_context_pair(
        self, template_name: str, base_rule: str, answer_differs: bool
    ) -> Tuple[str, str, str, str, str, str]:
        """answer_differs=True → answer_changes; False → control."""
        self._check_template(template_name)
        if not answer_differs:
            return self._generate_control_pair(template_name, base_rule)

        _, idx, x, y, fillers = self._sample_pair_setup()
        if base_rule == "ABA":
            # base=(X, copy, Y); exp=(X, alias, Y). same subs, mechanism differs.
            base = self._render_scenario(fillers, idx, x, "copy", y)
            exp = self._render_scenario(fillers, idx, x, "alias", y)
        elif base_rule == "ABB":
            # base=(Y, alias, X); exp=(Y, copy, X).
            base = self._render_scenario(fillers, idx, y, "alias", x)
            exp = self._render_scenario(fillers, idx, y, "copy", x)
        else:
            raise ValueError(f"base_rule must be ABA or ABB, got {base_rule}")

        return (base, exp,
                self._quoted(x), self._quoted(y),
                self._quoted(x), self._question(idx))

    def _generate_control_pair(
        self, template_name: str, base_rule: str
    ) -> Tuple[str, str, str, str, str, str]:
        """Control: both prompts use the same copy/alias choice as base of
        the sibling conditions in this direction; only orig/mut letters swap.
        Patch-position aligned with abstract/answer_changes ABA/ABB.

        ABA: both copy.  base=(X, copy, Y); exp=(Y, copy, X).
        ABB: both alias. base=(Y, alias, X); exp=(X, alias, Y).
        """
        _, idx, x, y, fillers = self._sample_pair_setup()
        if base_rule == "ABA":
            base = self._render_scenario(fillers, idx, x, "copy", y)
            exp = self._render_scenario(fillers, idx, y, "copy", x)
        elif base_rule == "ABB":
            base = self._render_scenario(fillers, idx, y, "alias", x)
            exp = self._render_scenario(fillers, idx, x, "alias", y)
        else:
            raise ValueError(f"base_rule must be ABA or ABB, got {base_rule}")

        return (base, exp,
                self._quoted(x), self._quoted(y),
                self._quoted(x), self._question(idx))


# ---------------------------------------------------------------------------
# Answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_value_from_code_response(response: str) -> str:
    """Extract a Python value from the model's response.

    Order: quoted single char (primary case for indexed_list), list literal,
    dict literal, bare digit, bare letter. Bare digits/letters are
    re-quoted so ast.literal_eval comparison works (the pool stores values
    as single-character strings like '7', not ints).
    """
    quoted = re.search(r"['\"]([A-Za-z0-9])['\"]", response)
    if quoted:
        return f"'{quoted.group(1)}'"

    list_match = re.search(r'\[[^\]]*\]', response)
    if list_match:
        return list_match.group(0).strip()

    dict_match = re.search(r'\{[^}]+\}', response)
    if dict_match:
        return dict_match.group(0).strip()

    num_match = re.search(r'\b\d\b', response)
    if num_match:
        return f"'{num_match.group(0)}'"

    letter = re.search(r'\b[A-Za-z]\b', response)
    if letter:
        return f"'{letter.group(0)}'"

    return ""


def code_values_match(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    try:
        return ast.literal_eval(predicted.strip()) == ast.literal_eval(expected.strip())
    except (ValueError, SyntaxError):
        return predicted.replace(" ", "") == expected.replace(" ", "")


# ---------------------------------------------------------------------------
# Position detection
# ---------------------------------------------------------------------------

def detect_copy_semantics_position(
    prompt: str, tokenizer, is_copy: bool, templates: Dict[str, CodeScenarioTemplate]
) -> int:
    """Token position after the alias/copy marker in the prompt."""
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
