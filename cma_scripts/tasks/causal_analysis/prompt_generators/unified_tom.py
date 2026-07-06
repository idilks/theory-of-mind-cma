"""
Unified ToM prompt generator for CMA integration.
Uses the unified template system for generating ToM prompts in CMA experiments.

Abstract context (different rules):
When # The `base_rule` variable in this code is determining the belief state used as the base scenario
# and the belief state used as the experimental scenario in generating prompts for the CMA
# experiments.
base_rule="ABA": Base=false belief, Exp=true belief
When base_rule="ABB": Base=true belief, Exp=false belief
Causal hypothesis: patching from base->exp should preserve base belief state

Token context: control and photo contexts.

"""

import sys
import os
import re

# Import classes from tom_templates.py
from .tom_templates import UnifiedTemplateSystem, UnifiedQuestionFormatter
from .base import PromptGenerator
from typing import List, Tuple

class UnifiedTomGenerator(PromptGenerator):
    """Generates ToM prompts using the unified template system for CMA experiments."""
    
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        
        # Determine vocab_dir from config
        vocab_dir = "tom_datasets"
        if hasattr(config.prompts, 'tom_locations_file'):
            vocab_dir = os.path.dirname(config.prompts.tom_locations_file) or "tom_datasets"
        
        # Create template system
        self.template_system = UnifiedTemplateSystem(vocab_dir)
        self.question_formatter = UnifiedQuestionFormatter(tokenizer)
        
        # Get template names from config
        self.template_names = config.prompts.template_names or ["food_truck"]
        print(f"DEBUG: Using template_names: {self.template_names}")
        self._supports_system_role = self._detect_system_role_support()
        if self._supports_system_role:
            print("DEBUG: Chat template supports system role; using system+user messages")
        else:
            print("DEBUG: Chat template does not support system role; folding system prompt into user message")

    def _detect_system_role_support(self) -> bool:
        """Return True when tokenizer chat template accepts a system role."""
        test_messages = [
            {"role": "system", "content": "System prompt compatibility check."},
            {"role": "user", "content": "User prompt compatibility check."},
        ]
        try:
            self.tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return True
        except Exception as exc:
            # Gemma templates can raise this specific message.
            if "System role not supported" in str(exc):
                return False
            raise

    def _build_messages(self, system_prompt: str, user_prompt: str):
        """Build chat messages while preserving behavior for templates that support system role."""
        if self._supports_system_role:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        combined_user_prompt = f"{system_prompt}\n\n{user_prompt}"
        return [{"role": "user", "content": combined_user_prompt}]
        
    def generate_prompts(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Generate base/exp prompt pairs for CMA.
        
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
        prompt_metadata = []  # per-prompt dicts from template_system *_with_meta
        
        num_prompts = self.config.prompts.prompt_num
        prompts_per_template = max(1, num_prompts // len(self.template_names))

        # UnifiedTomGenerator always uses the template system logic
        for template_name in self.template_names:
            for _ in range(prompts_per_template):
                base_rule = self.config.prompts.base_rule  # "ABA" or "ABB"
                context_type = self.config.prompts.context_type  # "abstract", "photo", or "control"

                if context_type == "abstract" or context_type == "photo":
                    # Generate CMA pair using template system
                    # answer_differs is derived from the condition enum when available
                    if self.config.prompts.condition is not None:
                        answer_differs = self.config.prompts.condition.answer_differs
                        same_belief = getattr(self.config.prompts.condition, "same_belief", False)
                    else:
                        raise ValueError("Condition must be specified in config for UnifiedTomGenerator to determine answer_differs. Please set config.prompts.condition with appropriate answer_differs value.")
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, pair_meta = self.template_system.generate_cross_context_pair_with_meta(template_name, base_rule=base_rule, context_type=context_type, answer_differs=answer_differs, same_belief=same_belief)
                elif context_type == "basic":
                    # C5 STATE_ONLY_NO_BELIEF: no observer/belief/photo. End-patch only.
                    base_scenario, exp_scenario, base_ans, exp_ans, causal_ans, pair_meta = self.template_system.generate_basic_pair_with_meta(template_name, base_rule=base_rule)
                else:
                    raise ValueError(f"UnifiedTomGenerator does not support context_type={context_type!r}. Supported: 'abstract', 'photo', 'basic'.")

                # Get question based on config
                question_style = self.config.prompts.question_style  # "completion" or "instruction"
                base_question = self.question_formatter.get_cma_questions(template_name, context_type=context_type, belief_type=base_rule, question_style=question_style)
                system_prompt = "You are a helpful assistant reasoning about agents' beliefs."

                # Build prompts with assigned scenarios
                base_full_prompt = f"{base_scenario}\n{base_question}"

                # now adding system/user prompt structure
                base_messages = self._build_messages(system_prompt, base_full_prompt)

                # Apply chat template - this adds the assistant header automatically
                formatted_prompt = self.tokenizer.apply_chat_template(
                    base_messages,
                    tokenize=False,  # return string, not token IDs
                    add_generation_prompt=True  # adds <|start_header_id|>assistant<|end_header_id|> at end
                )



                base_prompts.append(formatted_prompt)

                exp_question = base_question

                exp_full_prompt = f"{exp_scenario}\n{exp_question}"

                exp_messages = self._build_messages(system_prompt, exp_full_prompt)

                formatted_exp_prompt = self.tokenizer.apply_chat_template(
                    exp_messages,
                    tokenize=False,  # return string, not token IDs
                    add_generation_prompt=True  # adds <|start_header_id|>assistant<|end_header_id|> at end
                )

                exp_prompts.append(formatted_exp_prompt)

                # doesn't support dual question format yet
                base_answers.append(base_ans)
                exp_answers.append(exp_ans)
                # Causal answer: after patching base->exp, expect base belief state
                causal_answers.append(causal_ans)
                # Stash per-prompt metadata (loc_first/second/correct + info_loss marker)
                # for downstream attention-position extraction (lookback analysis).
                # Add the template name so downstream code can pick char_q/obj_q.
                pair_meta_with_template = {**pair_meta, "template_name": template_name,
                                           "base_rule": base_rule}
                prompt_metadata.append(pair_meta_with_template)

        # Prepare return format
        prompt_pairs = list(zip(base_prompts[:num_prompts], exp_prompts[:num_prompts]))
        correct_ans_pairs = list(zip(base_answers[:num_prompts], exp_answers[:num_prompts]))

        # Expose metadata on the generator for callers that need it (e.g. the
        # lookback capture script). Kept off the return tuple to preserve the
        # existing 3-tuple signature used by cma.py.
        self.last_metadata = prompt_metadata[:num_prompts]

        return prompt_pairs, correct_ans_pairs, causal_answers[:num_prompts]
    
    def detect_belief_formation_position(self, prompt: str, tokenizer,
                                          is_false_belief: bool | None = None) -> int:
        """Detect the token position where the agent's belief state is sealed.

        When `is_false_belief` is supplied (True/False), looks up that
        specific marker on each template — used by CMA, which knows in
        advance which template kind the base/exp prompt was rendered from.

        When `is_false_belief` is None (default), probes the prompt for
        whichever of the false- and true-belief markers actually appears.
        Used by lookback analysis, which only needs the marker that's in
        the prompt and shouldn't depend on CMA's base_rule structure.

        Returns the token position at the END of the matched marker, or -1.
        """
        if is_false_belief is True:
            marker_attrs = ("false_belief_marker",)
        elif is_false_belief is False:
            marker_attrs = ("true_belief_marker",)
        else:
            marker_attrs = ("false_belief_marker", "true_belief_marker")

        for template in self.template_system.templates.values():
            for attr in marker_attrs:
                marker = getattr(template, attr, "") or ""
                if not marker:
                    continue
                pos = prompt.find(marker)
                if pos == -1:
                    continue
                end_char = pos + len(marker)
                encoding = tokenizer(prompt, return_offsets_mapping=True)
                for i, (start, end) in enumerate(encoding['offset_mapping']):
                    if start <= end_char <= end:
                        return i + 1
                    elif start > end_char:
                        return i
                return len(encoding['offset_mapping'])
        return -1

    def detect_photo_position(self, prompt: str, tokenizer) -> int:
        """Detect the token position where the photo/picture is taken (epistemic access moment).

        For photo conditions, the relevant epistemic access moment is when
        the photograph captures the state — not when the object moves.

        Returns token position at the END of the photo-taking event, or -1 if not found.
        """
        photo_patterns = [
            r'photographer takes a photo of [^.]+\.',
            r'photographer takes a picture of [^.]+\.',
            r'prints out the reservation details from the website\.',
        ]

        for pattern in photo_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                end_char = match.end()
                encoding = tokenizer(prompt, return_offsets_mapping=True)
                for i, (start, end) in enumerate(encoding['offset_mapping']):
                    if start <= end_char <= end:
                        return i + 1
                    elif start > end_char:
                        return i
                return len(encoding['offset_mapping'])
        return -1

    def detect_post_movement_position(self, prompt: str, tokenizer) -> int:
        """Food-truck-only: token position right after 'drives the truck to <X>.'

        Used by the C3/C4 within-template residual-stream patch experiment, where
        we want a string-anchored landing site that is identical between base
        (false_context) and exp (true_context) and lies after the world-state
        change in both. Returns -1 if the anchor is not found, in which case
        callers must fall back or raise.
        """
        match = re.search(r'drives the truck to [^.]+\.', prompt, re.IGNORECASE)
        if match is None:
            return -1
        end_char = match.end()
        encoding = tokenizer(prompt, return_offsets_mapping=True)
        for i, (start, end) in enumerate(encoding['offset_mapping']):
            if start <= end_char <= end:
                return i + 1
            elif start > end_char:
                return i
        return len(encoding['offset_mapping'])

    def detect_movement_position(self, prompt: str, tokenizer) -> int:
        """Detect the token position after the movement event.
        
        Args:
            prompt: The prompt string to analyze
            tokenizer: Tokenizer to map character positions to token positions
            
        Returns:
            Token position after movement completion, or -1 if not found
        """
        # Look for movement patterns in ToM scenarios
        movement_patterns = [
            r'drives the truck to [^.]+\.',  # food truck movement
            r'moves? (?:the )?[^.]+? to [^.]+\.',  # general object movement
            r'relocates? [^.]+? to [^.]+\.',  # relocation
            r'transfers? [^.]+? to [^.]+\.',  # transfer, library book
            r'dyes? her hair [^.]+\.',  # hair color change
            r'changes? the reservation to [^.]+\.',  # reservation time change
        ]
        
        for pattern in movement_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                # Get the end character position of the movement event
                end_char = match.end()
                
                # Tokenize the prompt to get character-to-token mapping
                encoding = tokenizer(prompt, return_offsets_mapping=True)
                
                # Find the token that contains or follows the movement end
                for i, (start, end) in enumerate(encoding['offset_mapping']):
                    if start <= end_char <= end:
                        # Return position after this token
                        return i + 1
                    elif start > end_char:
                        # Movement ends before this token starts
                        return i
                
                # If we reach here, movement ends at or after the last token
                return len(encoding['offset_mapping'])
        
        # Fallback: if no movement pattern found, use final position
        return -1