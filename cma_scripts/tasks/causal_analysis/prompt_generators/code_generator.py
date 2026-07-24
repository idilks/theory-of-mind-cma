"""
Code prompt generator for CMA experiments.

Generates Python code-tracing prompts where copy semantics (shallow alias
vs deep copy) serve as analogs to true/false belief conditions.

The pair generator returns the per-pair question (which embeds the random
target index), so questions are not held in static dicts.
"""

from typing import List, Tuple

from .base import PromptGenerator
from .code_templates import (
    CodeTemplateSystem,
    CODE_SYSTEM_PROMPT,
    detect_copy_semantics_position,
    filter_answer_pool,
)


class CodeGenerator(PromptGenerator):
    """Generates code-tracing prompts for CMA experiments."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.template_system = CodeTemplateSystem(
            answer_pool=filter_answer_pool(tokenizer)
        )
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

        for template_name in self.template_names:
            for _ in range(prompts_per_template):
                base_rule = self.config.prompts.base_rule

                if is_abstract:
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
                exp_full = self._format_code_prompt(exp_scenario, question)

                base_formatted = self._apply_chat_template(base_full)
                exp_formatted = self._apply_chat_template(exp_full)

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

    def _apply_chat_template(self, user_content: str) -> str:
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
