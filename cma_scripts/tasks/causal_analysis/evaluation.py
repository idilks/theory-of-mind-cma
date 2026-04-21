"""
Evaluation utilities for CMA experiments.
Handles generation accuracy, logit differences, and dual-answer analysis.
"""

import logging
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import AutoTokenizer

from prompt_generators.base import extract_dual_answers, normalize_loc, classify_belief_prediction, classify_world_prediction
from models import CustomHookedTransformer
from prompt_generators.base import extract_location_from_response, locations_match




logger = logging.getLogger(__name__)

class GenerationEvaluator:
    """Handles model generation and evaluation."""
    
    def __init__(self, config):
        self.config = config


    def evaluate_generation_accuracy(
        self,
        model: CustomHookedTransformer,
        input_ids: torch.Tensor,
        eos_token_id: List[int],
        tokenizer: AutoTokenizer,
        correct_ans: str,
        prompt: str,
        fwd_hooks: List = None,
        return_texts: bool = False,
        **generation_kwargs
    ) -> Tuple[int, int, float, Optional[List[str]]]:
        """Generate responses and evaluate accuracy."""
        if fwd_hooks is None:
            fwd_hooks = []
            
        # Filter out forced-choice parameters that don't belong in generation
        gen_kwargs = {k: v for k, v in generation_kwargs.items() 
                      if k not in ['A_tok_id', 'B_tok_id', 'eval_metric', 'better_cma', 
                                   'exp_prompt', 'exp_input_ids', 'base_prompt', 'base_input_ids']}
        
        input_ids = input_ids.repeat(self.config.generation.samples_per_condition, 1)
        # logger.debug(f"DEBUG: About to generate with input_ids shape: {input_ids.shape}")
        generated_content = model.generate_with_hooks(
            input_ids, 
            max_new_tokens=self.config.generation.max_new_tokens, 
            eos_token_id=eos_token_id, 
            stop_at_eos=True,
            return_type="input",
            verbose=False,
            fwd_hooks=fwd_hooks,
            **gen_kwargs
        )
        
        generated_ids = generated_content        
        
        correct_num = 0
        total_num = 0
        gen_texts = []

        for i, full_ids in enumerate(generated_ids):
            # Get only the newly generated tokens
            prompt_length = input_ids[i].shape[0]
            new_token_ids = full_ids[prompt_length:]
            
            # Decode only the new tokens
            gen_ans = tokenizer.decode(new_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()    
                    
            # Determine belief type and format informative log
            belief_type = self._determine_belief_type(prompt)
            print(f"=== GENERATION DEBUG ===")
            print(f"Belief type: {belief_type}")
            print(f"Prompt (full): {prompt}")
            print(f"Generated: '{gen_ans}'")
            print(f"Expected: '{correct_ans}'")
            # print(f"Full response: '{full_text}'")
            print(f"=========================")

            for eos_token in [tokenizer.decode([eos_id]) for eos_id in eos_token_id]:
                if eos_token in gen_ans:
                    if self.config.prompts.ask_world and self.config.verbose:
                        print(f"DEBUG: found eos_token '{eos_token}' in gen_ans, truncating")
                    gen_ans = gen_ans.split(eos_token)[0].strip()
                    break
            if self.config.prompts.ask_world and self.config.verbose:
                print(f"DEBUG: gen_ans after eos cleanup: '{gen_ans}'")
            
            
            gen_texts.append(gen_ans)
            
            # Evaluate correctness based on task type
            is_correct = self._evaluate_answer_correctness(gen_ans, correct_ans)
            
            if is_correct:
                correct_num += 1
            
            total_num += 1
            
            # # Debug logging for long generated answers
            # if len(gen_ans) > 20:
            #     logger.warning("--------------------------------------------------")
            #     logger.warning(f"Question: {prompt}")
            #     logger.warning(f"Long generated answer: {gen_ans[:50]}...")
            # print(f"Gen answer: '{gen_ans}' | Correct: '{correct_ans}' | Is correct: {is_correct}")    
        acc = correct_num / self.config.generation.samples_per_condition
                
        if return_texts:
            return correct_num, total_num, acc, gen_texts
        return correct_num, total_num, acc
    
    def _determine_belief_type(self, prompt: str) -> str: # this function is a terrible hack as i forgot to pass the belief type from the prompt generator
        """Determine if prompt represents false belief or true belief scenario."""
        if not self.config.prompts.use_template_system:
            return "non-tom"
        
        # Key patterns that distinguish false belief vs true belief scenarios:
        # False belief: agent misses object movement (leaves before/during movement)
        # True belief: agent witnesses object movement (present during movement)
        
        # Look for patterns indicating agent is absent during movement
        false_belief_patterns = [
            "leaves",
            "goes away", 
            "exits",
            "steps out",
            "walks away"
        ]
        
        # Look for patterns indicating agent is present during movement  
        true_belief_patterns = [
            "sees",
            "watches", 
            "observes",
            "notices",
            "witnesses"
        ]
        
        prompt_lower = prompt.lower()
        
        # Check for false belief indicators
        for pattern in false_belief_patterns:
            if pattern in prompt_lower:
                return "false_belief"
        
        # Check for true belief indicators  
        for pattern in true_belief_patterns:
            if pattern in prompt_lower:
                return "true_belief"
        
        # Fallback: analyze sentence structure
        # False belief typically has: [agent action] ... [object moves] ... [agent returns]
        # True belief typically has: [object moves] ... [agent action] or agent present throughout
        
        if "returns" in prompt_lower or "comes back" in prompt_lower:
            return "false_belief"
        
        return "unknown_belief_type"
    
    def _evaluate_answer_correctness(self, gen_ans: str, correct_ans) -> bool:
        """Evaluate if generated answer is correct based on task type."""
        # Debug logging
        # logger.debug(f"Evaluating: gen_ans='{gen_ans}', correct_ans='{correct_ans}'")
        # logger.debug(f"Config: use_template_system={self.config.prompts.use_template_system}, ask_world={self.config.prompts.ask_world}")
        
        if self.config.prompts.use_template_system and self.config.prompts.ask_world:
            # Dual-answer mode: check belief answer only
            # logger.debug("Using dual-answer evaluation path")
            belief_pred, world_pred, malformed = extract_dual_answers(gen_ans)
            if isinstance(correct_ans, tuple):
                expected_belief, expected_world = correct_ans
                return (not malformed) and (normalize_loc(belief_pred) == normalize_loc(expected_belief))
            else:
                # Fallback for single answer case
                return (not malformed) and (normalize_loc(belief_pred) == normalize_loc(correct_ans))
        
        elif self.config.prompts.context_type == "code":
            from prompt_generators.code_templates import extract_value_from_code_response, code_values_match
            extracted = extract_value_from_code_response(gen_ans)
            matches = code_values_match(extracted, correct_ans)
            if not matches and extracted == "":
                matches = gen_ans.strip() == correct_ans.strip()
            logger.debug(f"Evaluating code answer: Generated='{gen_ans}', Extracted='{extracted}', Correct='{correct_ans}', Matches={matches}")
            return matches

        elif self.config.prompts.use_template_system:
            # TOM tasks: extract location from tags and compare properly
            # logger.debug("Using TOM location extraction evaluation path")
            extracted = extract_location_from_response(gen_ans)
            matches = locations_match(extracted, correct_ans)
            
            # Fallback: if extraction fails but answers match directly (case-insensitive)
            if not matches and extracted == "":
                matches = gen_ans.lower().strip() == correct_ans.lower().strip()
                if matches:
                    logger.debug(f"TOM extraction failed, using direct match: '{gen_ans}' == '{correct_ans}'")
            
            logger.debug(f"Evaluating TOM answer: Generated='{gen_ans}', Extracted='{extracted}', Correct='{correct_ans}', Matches={matches}")
            return matches
        
        else:
            # Identity rules: case-insensitive matching
            # logger.debug("Using identity rules case-insensitive matching evaluation path")
            return gen_ans.lower() == correct_ans.lower()

class DualAnswerStatsTracker:
    """Tracks statistics for dual-answer (--ask_world) mode."""
    
    def __init__(self):
        self.belief_counts_base = defaultdict(int)
        self.belief_counts_exp = defaultdict(int)
        self.world_counts_base = defaultdict(int)
        self.world_counts_exp = defaultdict(int)
        self.malformed_base = 0
        self.malformed_exp = 0
    
    def update_base_stats(self, gen_texts: List[str], gold_belief: str, gold_world: str, opposite_belief: str):
        """Update statistics for base context generations."""
        for text in gen_texts:
            bpred, wpred, malformed = extract_dual_answers(text)
            if malformed:
                self.malformed_base += 1
                continue
            
            self.belief_counts_base[classify_belief_prediction(bpred, gold_belief, opposite_belief)] += 1
            self.world_counts_base[classify_world_prediction(wpred, gold_world, gold_belief)] += 1
    
    def update_exp_stats(self, gen_texts: List[str], gold_belief: str, gold_world: str, opposite_belief: str):
        """Update statistics for exp context generations."""
        for text in gen_texts:
            bpred, wpred, malformed = extract_dual_answers(text)
            if malformed:
                self.malformed_exp += 1
                continue
                
            self.belief_counts_exp[classify_belief_prediction(bpred, gold_belief, opposite_belief)] += 1
            self.world_counts_exp[classify_world_prediction(wpred, gold_world, gold_belief)] += 1
    
    def save_stats(self, save_path: str):
        """Save dual-answer statistics to file with comprehensive documentation."""
        with open(save_path, "w") as f:
            # Write header explaining the analysis
            f.write("# Theory of Mind Dual-Answer Statistics\n")
            f.write("# Analyzes model responses when asked for both belief and world state\n")
            f.write("# Format: 'belief: <location>' and 'world: <location>' in same response\n")
            f.write("#\n")
            f.write("# CONTEXT TYPES:\n")
            f.write("# - base: False belief context (agent misses object movement)\n")
            f.write("# - exp: True belief context (agent witnesses movement) OR token context variant\n")
            f.write("#\n")
            f.write("# PREDICTION CATEGORIES:\n")
            f.write("# Belief predictions:\n")
            f.write("#   - correct: Matches expected belief state (false belief = original loc)\n")
            f.write("#   - opposite: Agent belief matches alternative location (systematic error)\n")
            f.write("#   - third: Neither expected nor alternative location (random error)\n")
            f.write("#\n")
            f.write("# World predictions:\n")
            f.write("#   - correct: Matches actual world state (where object really is)\n")
            f.write("#   - belief_confusion: Model reports belief state instead of world state\n")
            f.write("#   - third: Neither world nor belief state (random error)\n")
            f.write("#   - malformed: Response didn't follow required dual-answer format\n")
            f.write("#\n")
            f.write("# INTERPRETATION:\n")
            f.write("# High belief_confusion in world predictions = model conflates belief with reality\n")
            f.write("# Low belief accuracy = model struggles with false belief reasoning\n")
            f.write("# High malformed count = model has trouble following output schema\n")
            f.write("\n" + "="*80 + "\n\n")
            
            def dump_block(title, counts, malformed):
                total = sum(counts.values()) + malformed
                f.write(f"{title}\n")
                f.write(f"total: {total}\n")
                
                # Show counts and percentages
                for k in ["correct","opposite","third","belief_confusion"]:
                    if k in counts:
                        pct = (counts[k] / total * 100) if total > 0 else 0
                        f.write(f"{k}: {counts[k]} ({pct:.1f}%)\n")
                        
                if malformed:
                    pct = (malformed / total * 100) if total > 0 else 0
                    f.write(f"malformed: {malformed} ({pct:.1f}%)\n")
                f.write("\n")
            
            dump_block("base/belief", self.belief_counts_base, self.malformed_base)
            dump_block("base/world", self.world_counts_base, self.malformed_base)
            dump_block("exp/belief", self.belief_counts_exp, self.malformed_exp)
            dump_block("exp/world", self.world_counts_exp, self.malformed_exp)
            
            # Add summary analysis
            f.write("="*80 + "\n")
            f.write("SUMMARY ANALYSIS:\n\n")
            
            # Belief tracking performance
            base_belief_total = sum(self.belief_counts_base.values()) + self.malformed_base
            exp_belief_total = sum(self.belief_counts_exp.values()) + self.malformed_exp
            if base_belief_total > 0 and exp_belief_total > 0:
                base_belief_acc = self.belief_counts_base.get('correct', 0) / base_belief_total
                exp_belief_acc = self.belief_counts_exp.get('correct', 0) / exp_belief_total
                f.write(f"Belief tracking accuracy: base={base_belief_acc:.3f}, exp={exp_belief_acc:.3f}\n")
            
            # World state performance  
            base_world_total = sum(self.world_counts_base.values()) + self.malformed_base
            exp_world_total = sum(self.world_counts_exp.values()) + self.malformed_exp
            if base_world_total > 0 and exp_world_total > 0:
                base_world_acc = self.world_counts_base.get('correct', 0) / base_world_total
                exp_world_acc = self.world_counts_exp.get('correct', 0) / exp_world_total
                base_confusion = self.world_counts_base.get('belief_confusion', 0) / base_world_total
                exp_confusion = self.world_counts_exp.get('belief_confusion', 0) / exp_world_total
                f.write(f"World state accuracy: base={base_world_acc:.3f}, exp={exp_world_acc:.3f}\n")
                f.write(f"Belief confusion rate: base={base_confusion:.3f}, exp={exp_confusion:.3f}\n")


def get_first_token_id(model: CustomHookedTransformer, text: str) -> int:
    """Get first token ID for text, handling multi-token cases."""
    try:
        return model.to_single_token(text)
    except (AssertionError, ValueError):
        # Multi-token case - use first token
        tokens = model.to_tokens(text, prepend_bos=False)[0]
        return tokens[0].item()

def calculate_logit_probability_difference(
    logits: torch.Tensor,
    model: CustomHookedTransformer,
    causal_ans: str,
    original_ans: str
) -> float:
    """
    Calculate logit difference between causal answer and original answer.
    Handles multi-token locations by using first token.
    """
    # Get first token of each location
    causal_token_id = get_first_token_id(model, causal_ans)
    original_token_id = get_first_token_id(model, original_ans)
    
    logit_diff = logits[0, -1, causal_token_id] - logits[0, -1, original_token_id]
    return logit_diff

# this case is for multi-answer format (asking about the  world and the belief)
def calculate_dual_logit_differences(
    logits: torch.Tensor,
    model: CustomHookedTransformer,
    causal_ans_tuple,
    original_ans_tuple
) -> float:
    """Calculate logit differences for dual-answer format, focusing on belief component."""
    belief_causal, world_causal = causal_ans_tuple
    belief_orig, world_orig = original_ans_tuple
    
    # Focus on belief tracking mechanism - handle multi-token words
    belief_causal_id = get_first_token_id(model, belief_causal)
    belief_orig_id = get_first_token_id(model, belief_orig)
    
    belief_diff = logits[0, -1, belief_causal_id] - logits[0, -1, belief_orig_id]
    return belief_diff

def get_first_token_id(model: CustomHookedTransformer, text: str) -> int:
    """Get the first token ID for a text, handling multi-token words."""
    try:
        # Try single token first
        return model.to_single_token(text)
    except AssertionError:
        # Handle multi-token case by using first token
        tokens = model.to_tokens(text, prepend_bos=False)[0]  # Remove batch dimension
        return tokens[0].item()

def calculate_ab_margin(logits: torch.Tensor, A_tok_id: int, B_tok_id: int) -> float:
    """Compute next-token margin for forced-choice A vs B."""
    if isinstance(A_tok_id, torch.Tensor):
        A_tok_id = A_tok_id.item()
    if isinstance(B_tok_id, torch.Tensor):
        B_tok_id = B_tok_id.item()
    return (logits[0, -1, A_tok_id] - logits[0, -1, B_tok_id])

class PromptFilter:
    """Filters prompts based on model performance thresholds."""
    
    def __init__(self, config, evaluator: GenerationEvaluator):
        self.config = config
        self.evaluator = evaluator
    
    def evaluate_and_check_threshold(
        self,
        model: CustomHookedTransformer,
        input_ids: torch.Tensor,
        eos_token_ids: List[int],
        tokenizer: AutoTokenizer,
        correct_ans: str,
        alternative_ans: str,
        prompt: str,
        A_tok_id: Optional[int] = None,
        B_tok_id: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[bool, Optional[List[str]]]:
        """Evaluate prompt and check if it passes threshold, ALWAYS returning gen_texts for dual-answer mode."""
        gen_texts = None
        
        # Filter out non-generation parameters ONCE at the top level
        filtered_kwargs = {k: v for k, v in generation_kwargs.items() 
                          if k not in ['A_tok_id', 'B_tok_id', 'eval_metric', 'better_cma', 
                                       'exp_prompt', 'exp_input_ids', 'base_prompt', 'base_input_ids',
                                       'alternative_ans']}
        
        if self.config.evaluation.eval_metric == "gen_acc":
            # Always return texts for dual-answer mode, regardless of threshold
            if self.config.prompts.ask_world:
                _, _, acc, gen_texts = self.evaluator.evaluate_generation_accuracy(
                    model, input_ids, eos_token_ids, tokenizer, correct_ans, prompt,
                    return_texts=True, **filtered_kwargs
                )
                passes = acc >= self.config.evaluation.low_prob_threshold
                if not passes:
                    logger.warning(f"low acc {acc} < {self.config.evaluation.low_prob_threshold}")
                return passes, gen_texts
            else:
                # For behavioral_tom, always return texts for response logging
                if self.config.prompts.use_template_system:
                    _, _, acc, gen_texts = self.evaluator.evaluate_generation_accuracy(
                        model, input_ids, eos_token_ids, tokenizer, correct_ans, prompt,
                        return_texts=True, **filtered_kwargs
                    )
                else:
                    _, _, acc = self.evaluator.evaluate_generation_accuracy(
                        model, input_ids, eos_token_ids, tokenizer, correct_ans, prompt,
                        **filtered_kwargs
                    )
                passes = acc >= self.config.evaluation.low_prob_threshold
                if not passes:
                    logger.warning(f"low acc {acc} < {self.config.evaluation.low_prob_threshold}")
                return passes, gen_texts
                
        elif self.config.evaluation.eval_metric == "ab_forced":
            logits = model(input_ids)
            margin = calculate_ab_margin(logits, A_tok_id, B_tok_id)
            pred = "A" if margin.item() > 0 else "B"
            passes = pred == correct_ans
            if not passes:
                logger.warning(f"wrong A/B ({pred} != {correct_ans})")
            return passes, gen_texts
                
        elif self.config.evaluation.eval_metric == "ans_prob":
            logits = model(input_ids)
            
            top_tokens = torch.topk(logits[0, -1, :], k=5)
            logger.info(f"Top 5 predictions: {[tokenizer.decode([t]) for t in top_tokens.indices]}")
            logger.info(f"Checking: correct='{correct_ans}' vs alternative='{alternative_ans}'")
            logger.info(f"Token for 'First': {get_first_token_id(model, 'First Street')}")
            logger.info(f"Token for ' First': {get_first_token_id(model, ' First Street')}")  # with space
            logger.info(f"Rank of 'First' in predictions: {(logits[0, -1, :].argsort(descending=True) == get_first_token_id(model, 'First Street')).nonzero()}")
                
            
            # Get token IDs for both answers
            correct_token_id = get_first_token_id(model, correct_ans)
            
            if alternative_ans:
                # Compare two answers directly
                alt_token_id = get_first_token_id(model, alternative_ans)
                print(f"Comparing logits for {correct_ans} and {alternative_ans}")
                
                correct_logit = logits[0, -1, correct_token_id]
                alt_logit = logits[0, -1, alt_token_id]
                
                logit_margin = correct_logit - alt_logit
                passes = logit_margin > 0  # correct answer has higher logit
                
                if not passes:
                    logger.warning(f"wrong answer preferred: margin={logit_margin:.3f}")
            else:
                # Fallback to probability threshold
                real_prob = torch.softmax(logits, dim=-1)[0, -1, correct_token_id]
                passes = real_prob >= self.config.evaluation.low_prob_threshold
                
            return passes, gen_texts

    def passes_threshold(
        self,
        model: CustomHookedTransformer,
        input_ids: torch.Tensor,
        eos_token_ids: List[int],
        tokenizer: AutoTokenizer,
        correct_ans: str,
        prompt: str,
        A_tok_id: Optional[int] = None,
        B_tok_id: Optional[int] = None,
        **generation_kwargs
    ) -> Tuple[bool, Optional[List[str]]]:
        """Check if prompt passes performance threshold. Legacy method for backward compatibility."""
        return self.evaluate_and_check_threshold(
            model, input_ids, eos_token_ids, tokenizer, correct_ans, prompt,
            A_tok_id, B_tok_id, **generation_kwargs
        )