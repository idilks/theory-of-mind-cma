"""
Code prompt generator for CMA experiments.

Generates Python code-tracing prompts where copy semantics (shallow reference
vs deep copy) serve as analogs to true/false belief conditions.
"""

from typing import List, Tuple

from .base import PromptGenerator
from .code_templates import (
    CodeTemplateSystem,
    CODE_SYSTEM_PROMPT,
    CODE_QUESTIONS,
    CODE_CONTROL_QUESTIONS,
    detect_copy_semantics_position,
)


class CodeGenerator(PromptGenerator):
    """Generates code-tracing prompts for CMA experiments."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.template_system = CodeTemplateSystem()
        self.template_names = config.prompts.template_names or ["list_append"]

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

        condition = self.config.prompts.condition
        answer_differs = condition.answer_differs if condition is not None else True

        is_control = (condition is not None and condition.value == "code_control")

        for template_name in self.template_names:
            if template_name not in CODE_QUESTIONS:
                raise ValueError(
                    f"Unknown code template: {template_name}. "
                    f"Available: {list(CODE_QUESTIONS.keys())}"
                )

            question = CODE_CONTROL_QUESTIONS[template_name] if is_control else CODE_QUESTIONS[template_name]

            for _ in range(prompts_per_template):
                base_rule = self.config.prompts.base_rule

                base_scenario, exp_scenario, base_ans, exp_ans, causal_ans = (
                    self.template_system.generate_cross_context_pair(
                        template_name,
                        base_rule=base_rule,
                        answer_differs=answer_differs,
                    )
                )

                # Format prompts with code block and question
                base_full = self._format_code_prompt(base_scenario, question)
                exp_full = self._format_code_prompt(exp_scenario, question)

                # Apply chat template
                base_formatted = self._apply_chat_template(base_full)
                exp_formatted = self._apply_chat_template(exp_full)

                base_prompts.append(base_formatted)
                exp_prompts.append(exp_formatted)
                base_answers.append(base_ans)
                exp_answers.append(exp_ans)
                causal_answers.append(causal_ans)

        # Trim to requested count
        prompt_pairs = list(zip(base_prompts[:num_prompts], exp_prompts[:num_prompts]))
        correct_ans_pairs = list(zip(base_answers[:num_prompts], exp_answers[:num_prompts]))

        return prompt_pairs, correct_ans_pairs, causal_answers[:num_prompts]

    def detect_copy_semantics_position(self, prompt: str, tokenizer, is_copy: bool) -> int:
        """Detect token position where copy/reference semantics are established."""
        return detect_copy_semantics_position(
            prompt, tokenizer, is_copy, self.template_system.templates
        )

    # ---- internal helpers -------------------------------------------------

    def _format_code_prompt(self, code_snippet: str, question: str) -> str:
        """Wrap code snippet in markdown code block with question."""
        return (
            "Consider the following Python code:\n\n"
            "```python\n"
            f"{code_snippet}\n"
            "```\n\n"
            f"{question}"
        )

    def _apply_chat_template(self, user_content: str) -> str:
        """Apply the tokenizer's chat template with code system prompt."""
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
