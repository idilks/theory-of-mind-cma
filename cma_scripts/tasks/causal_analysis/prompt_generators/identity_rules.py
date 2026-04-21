"""
Identity rules prompt generation (original ABA/ABB logic).
"""

import os
import sys
import random
import numpy as np
import logging
from typing import List, Tuple

from .base import PromptGenerator

# Add codebase root to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from utils import vocab_dict, get_model_id_family

logger = logging.getLogger(__name__)

class IdentityRulesGenerator(PromptGenerator):
    """Generates identity rule prompts (ABA/ABB patterns)."""
    
    def generate_prompts(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[str]]:
        """Generate identity rule prompt pairs."""
        prompts = []
        correct_ans_list = []
        causal_ans_list = []
        
        # Get vocabulary file
        model_id, model_family = get_model_id_family(self.config.model.model_type)
        vocab_file = vocab_dict[model_family]
        
        assert os.path.exists(vocab_file)
        with open(vocab_file, "r") as f:
            vocab_list = [l.rstrip() for l in f.readlines()]

        if self.config.prompts.do_shuffle:
            np.random.shuffle(vocab_list)

        while len(prompts) < self.config.prompts.prompt_num:
            prompt_pair = self._generate_single_prompt_pair(vocab_list)
            if prompt_pair is not None:
                base_prompt, exp_prompt, base_ans, exp_ans, causal_ans = prompt_pair
                
                if (base_prompt, exp_prompt) not in prompts:
                    prompts.append((base_prompt, exp_prompt))
                    correct_ans_list.append((base_ans, exp_ans))
                    causal_ans_list.append(causal_ans)
                elif self.config.verbose:
                    logger.info(f"Repeated prompt: {base_prompt[:50]}...")

        return prompts, correct_ans_list, causal_ans_list
    
    def _generate_single_prompt_pair(self, vocab_list: List[str]) -> Tuple[str, str, str, str, str]:
        """Generate a single prompt pair."""
        # Get tokens for examples
        tokens = random.sample(vocab_list, k=(self.config.prompts.in_context_example_num + 1) * 2)
        
        base_example_list = []
        exp_example_list = []
        
        # Generate in-context examples
        for idx in range(self.config.prompts.in_context_example_num):
            base_example, exp_example = self._create_example_pair(tokens, idx)
            base_example_list.append(base_example)
            exp_example_list.append(exp_example)
        
        # Generate test question
        base_question_tokens = [tokens[-2], tokens[-1]]  # hi ha
        base_ans_index = 0 if self.config.prompts.base_rule == "ABA" else 1
        base_ans = base_question_tokens[base_ans_index]
        
        base_question = f"{self.config.prompts.sep_symbol}".join(base_question_tokens) + f"{self.config.prompts.sep_symbol}"
        base_example_list.append(base_question)
        base_prompt = "\n".join(base_example_list)
        
        # Exp question - swap 1st/2nd tokens
        exp_question_tokens = [tokens[-1], tokens[-2]]  # ha hi
        exp_question = f"{self.config.prompts.sep_symbol}".join(exp_question_tokens) + f"{self.config.prompts.sep_symbol}"
        exp_example_list.append(exp_question)
        exp_prompt = "\n".join(exp_example_list)
        
        # Determine answers based on context type
        causal_ans = self._determine_causal_answer(base_ans_index, exp_question_tokens, base_ans)
        
        if self.config.prompts.context_type == "abstract":
            exp_ans_index = 1 - base_ans_index  # different rule
            exp_ans = exp_question_tokens[exp_ans_index]
        else:  # token context
            exp_ans_index = base_ans_index  # same rule
            exp_ans = exp_question_tokens[exp_ans_index]
        
        # Validate prompt format
        if not self._validate_prompt_pair(base_prompt, exp_prompt):
            return None
            
        return base_prompt, exp_prompt, base_ans, exp_ans, causal_ans
    
    def _create_example_pair(self, tokens: List[str], idx: int) -> Tuple[str, str]:
        """Create a single in-context example pair."""
        if self.config.prompts.base_rule == "ABA":
            base_tokens = [tokens[idx*2], tokens[idx*2+1], tokens[idx*2]]  # la li la
            if self.config.prompts.context_type == "abstract":
                exp_tokens = [tokens[idx*2+1], tokens[idx*2], tokens[idx*2]]  # li la la
            else:
                exp_tokens = base_tokens  # la li la
        else:  # ABB
            base_tokens = [tokens[idx*2], tokens[idx*2+1], tokens[idx*2+1]]  # la li li
            if self.config.prompts.context_type == "abstract":
                exp_tokens = [tokens[idx*2+1], tokens[idx*2], tokens[idx*2+1]]  # li la li
            else:
                exp_tokens = base_tokens  # la li li
        
        base_example = f"{self.config.prompts.sep_symbol}".join(base_tokens)
        exp_example = f"{self.config.prompts.sep_symbol}".join(exp_tokens)
        
        return base_example, exp_example
    
    def _determine_causal_answer(self, base_ans_index: int, exp_question_tokens: List[str], base_ans: str) -> str:
        """Determine expected answer after causal patching."""
        if self.config.prompts.context_type == "abstract":
            causal_ans_index = base_ans_index  # expected answer after causal mediation
            return exp_question_tokens[causal_ans_index]
        else:  # token context
            return base_ans  # literal token retrieval should stay same
    
    def _validate_prompt_pair(self, base_prompt: str, exp_prompt: str) -> bool:
        """Validate prompt pair format."""
        for prompt in [base_prompt, exp_prompt]:
            proc_tokens = self.tokenizer.tokenize(prompt)
            token_num = len(proc_tokens)
            expected_tokens = 3 * 2 * self.config.prompts.in_context_example_num + 4
            
            if token_num != expected_tokens:
                if self.config.verbose:
                    logger.info("=" * 50)
                    logger.info(f"Invalid prompt: {prompt}, token_num: {token_num}, expected: {expected_tokens}")
                return False
            
            # Check separator tokens
            sep_tokens = [self.config.prompts.sep_symbol, self.tokenizer.tokenize("\n")[0]]
            if np.sum(~np.isin(proc_tokens[1::2], sep_tokens)):
                if self.config.verbose:
                    logger.info("=" * 50)
                    logger.info(f"Invalid prompt: {prompt}, tokens: {proc_tokens}")
                return False
                
        return True