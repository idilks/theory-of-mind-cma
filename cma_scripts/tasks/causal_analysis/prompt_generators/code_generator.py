"""
Code prompt generator for CMA experiments.

Generates Python code-tracing prompts where copy semantics (shallow alias
vs deep copy) serve as analogs to true/false belief conditions.

The pair generator returns the per-pair question (which embeds the random
target index), so questions are not held in static dicts.
"""

import logging
from typing import List, Tuple

from .base import PromptGenerator
from .code_templates import (
    CodeTemplateSystem,
    CODE_SYSTEM_PROMPT,
    detect_copy_semantics_position,
    filter_answer_pool,
    filter_answer_pool_prefill,
)

logger = logging.getLogger(__name__)


class CodeGenerator(PromptGenerator):
    """Generates code-tracing prompts for CMA experiments."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        try:
            answer_pool = filter_answer_pool(tokenizer)
            self.quote_prefill = False
        except ValueError:
            # Tokenizer has no merged quote+char tokens (Gemma-2): fall back
            # to quote-prefill — prompts end with the opening quote and the
            # readout token is the bare char. The flag propagates to the
            # accuracy filter and the logit readout through the shared config.
            answer_pool = filter_answer_pool_prefill(tokenizer)
            self.quote_prefill = True
            config.prompts.code_quote_prefill = True
            logger.warning(
                "filter_answer_pool collapsed for this tokenizer — enabling "
                f"QUOTE-PREFILL readout (pool: {''.join(answer_pool)})")
        self.template_system = CodeTemplateSystem(answer_pool=answer_pool, tokenizer=tokenizer)
        self.system_role_ok = self._supports_system_role(tokenizer)
        if not self.system_role_ok:
            logger.warning("chat template lacks system role; folding "
                           "CODE_SYSTEM_PROMPT into the user message "
                           "(unified_tom.py convention)")
        self.template_names = config.prompts.template_names or ["indexed_list"]
        unknown = [n for n in self.template_names
                   if n not in self.template_system.templates]
        if unknown:
            raise ValueError(
                f"Unknown code template(s) {unknown}. "
                f"Available: {list(self.template_system.templates.keys())}. "
                "(list_append/dict_update/list_extend were removed with the "
                "July 2026 indexed_list redesign.)")

    def generate_prompts(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Generate base/exp prompt pairs for CMA with code templates.

        Returns:
            prompt_pairs: List of (base_prompt, exp_prompt) tuples
            correct_ans_pairs: List of (base_ans, exp_ans) tuples
            causal_ans_list: List of expected answers after patching
        """
        base_prompts = []
        exp_prompts = []
        base_answers = []
        exp_answers = []
        causal_answers = []

        num_prompts = self.config.prompts.prompt_num
        prompts_per_template = max(1, num_prompts // len(self.template_names))

        # Route on the condition itself, NOT condition.answer_differs:
        # code_control's answers also differ (base=X, exp=Y), so answer_differs
        # is True for it and cannot distinguish it from code_answer_changes.
        condition = self.config.prompts.condition
        cond_name = condition.value if condition is not None else "code_answer_changes"
        is_abstract = cond_name == "code_abstract"
        is_control = cond_name == "code_control"
        is_index = cond_name in ("code_index", "code_index_flip")

        for template_name in self.template_names:
            for _ in range(prompts_per_template):
                base_rule = self.config.prompts.base_rule

                exp_question = None  # only the index contrast asks a different question per side
                if is_index:
                    (base_scenario, exp_scenario, base_ans, exp_ans, causal_ans,
                     question, exp_question) = self.template_system.generate_index_pair(
                        template_name, base_rule=base_rule,
                        flip_order=cond_name == "code_index_flip",
                    )
                elif cond_name == "code_veracity_only":
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_veracity_only_pair(
                            template_name, base_rule=base_rule
                        )
                    )
                elif cond_name in ("code_site_only", "code_site_only_rev"):
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_site_only_pair(
                            template_name, base_rule=base_rule,
                            reverse=cond_name == "code_site_only_rev",
                        )
                    )
                elif cond_name in ("code_pos_matched", "code_pos_unmatched"):
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_pos_matched_pair(
                            template_name, base_rule=base_rule,
                            matched=cond_name == "code_pos_matched",
                        )
                    )
                elif cond_name == "code_abstract_two_mut":
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_abstract_two_mut_pair(
                            template_name, base_rule=base_rule
                        )
                    )
                elif is_abstract:
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_abstract_pair(
                            template_name, base_rule=base_rule
                        )
                    )
                else:
                    # answer_differs=False selects the control pair (same
                    # copy/alias kind on both sides, letters swapped)
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, question = (
                        self.template_system.generate_cross_context_pair(
                            template_name,
                            base_rule=base_rule,
                            answer_differs=not is_control,
                        )
                    )

                base_full = self._format_code_prompt(base_scenario, question)
                exp_full = self._format_code_prompt(
                    exp_scenario, exp_question if exp_question is not None else question)

                base_formatted = self._apply_chat_template(base_full)
                exp_formatted = self._apply_chat_template(exp_full)
                if self.quote_prefill:
                    base_formatted += "'"
                    exp_formatted += "'"

                base_prompts.append(base_formatted)
                exp_prompts.append(exp_formatted)
                base_answers.append(base_ans)
                exp_answers.append(exp_ans)
                causal_answers.append(causal_ans)

        prompt_pairs = list(zip(base_prompts[:num_prompts], exp_prompts[:num_prompts]))
        correct_ans_pairs = list(zip(base_answers[:num_prompts], exp_answers[:num_prompts]))

        return prompt_pairs, correct_ans_pairs, causal_answers[:num_prompts]

    def detect_copy_semantics_position(self, prompt: str, tokenizer, is_copy: bool) -> int:
        """Detect token position where copy/alias semantics are established."""
        return detect_copy_semantics_position(
            prompt, tokenizer, is_copy, self.template_system.templates
        )

    # ---- internal helpers -------------------------------------------------

    def _format_code_prompt(self, code_snippet: str, question: str) -> str:
        return (
            "Consider the following Python code:\n\n"
            "```python\n"
            f"{code_snippet}\n"
            "```\n\n"
            f"{question}"
        )

    @staticmethod
    def _supports_system_role(tokenizer) -> bool:
        """Same probe as unified_tom.py — Gemma templates reject system messages."""
        try:
            tokenizer.apply_chat_template(
                [{"role": "system", "content": "check"},
                 {"role": "user", "content": "check"}],
                tokenize=False, add_generation_prompt=True)
            return True
        except Exception as exc:
            if "System role not supported" in str(exc):
                return False
            raise

    def _apply_chat_template(self, user_content: str) -> str:
        if self.system_role_ok:
            messages = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        else:
            messages = [
                {"role": "user",
                 "content": f"{CODE_SYSTEM_PROMPT}\n\n{user_content}"},
            ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
