#####################################
from html import parser
import os 
import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

from utils import HF_TOKEN, HF_HOME, plot
os.environ["HF_TOKEN"] = HF_TOKEN # configure the User Access Token to authenticate to the Hub
os.environ["HF_HOME"] = HF_HOME  ## set the cache directory for Hugging Face 
#####################################
## implementation of CMA method (Section 3.1 and Alogorithm 1 in https://arxiv.org/pdf/2502.20332?)
## context_type: abstract (symbol abstraction heads/ symbolic induction heads) or token (retrieval heads)
## notations: (patching from base context to exp context (c2 -> c1))
# base context (c2): base_rule/base_prompt/base_ans (y_c2)
# exp context (c1): exp_rule/exp_prompt/exp_ans (y_c1)
# patched exp context (c1*): expected answer after patching activations from c2 to c1, causal_ans (y_c1*)
######################################

from typing import List, Tuple
import logging
import argparse
import os
import wandb

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

from utils import set_seed

# Import modular components
from cma_config import ExperimentConfig, CMACondition
from models import ModelLoader
from evaluation import GenerationEvaluator, DualAnswerStatsTracker, PromptFilter
from patching import ActivationPatcher
from prompt_generators import get_prompt_generator
import torch

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, nargs="?", help="random seed")

    ### load the model
    parser.add_argument("--model_type", type=str, default="Qwen2.5-14B-Instruct", help="model type")
    parser.add_argument("--device_map", type=str, default="cpu", help="device map (cpu or auto)")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    parser.add_argument("--n_devices", type=int, default=2, help="number of devices. please set to 2 for Llama-3.1-70B and Qwen2,5-72B while 1 for others")
    parser.add_argument("--fold_ln", action="store_true", help="fold layer normalization weights into the weights of the preceeding layer")

    #### build the prompt (context) pairs
    parser.add_argument("--condition", type=str, default=None,
                        choices=[c.value for c in CMACondition],
                        help="CMA condition (overrides --context_type). See CMACondition enum for all values.")
    parser.add_argument("--context_type", type=str, default="abstract", help="context type for prompt generation (legacy; prefer --condition)", choices=["abstract", "token", "control", "photo", "basic", "code"])
    parser.add_argument("--base_rule", type=str, default="ABA", help="rule for base prompt in prompt pairs", choices=["ABA", "ABB"])
    parser.add_argument("--prompt_num", type=int, default=None, help="the number of prompt pairs to be generated")
    parser.add_argument("--in_context_example_num", type=int, default=0, help="in-context example number")
    parser.add_argument("--sep_symbol", type=str, default="^", help="separator symbol")
    parser.add_argument("--do_shuffle", action="store_true", help="whether to shuffle the vocabulary list for generating prompts")
    
    #### theory of mind task
    parser.add_argument("--use_template_system", action="store_true", help="Use unified template system for ToM prompts")
    parser.add_argument("--tom_locations_file", type=str, default="tom_datasets/locations.txt", help="File containing location phrases for ToM prompts")
    parser.add_argument("--tom_format", type=str, default="direct", choices=["direct", "multiple_choice"], help="ToM prompt format")
    parser.add_argument("--ask_world", action="store_true", help="Ask for both belief and world state in dual-answer format")
    parser.add_argument("--question_style", type=str, default="instruction", choices=["completion", "instruction"], help="Question format style for ToM prompts")
    parser.add_argument("--template_names", nargs='+', type=str, default=["food_truck"], help="Template names for unified ToM system")
    parser.add_argument("--better_cma", action="store_true", help="Use improved CMA prompt generation logic for unified ToM")
    
    ### Wandb logging
    parser.add_argument("--wandb_project", type=str, default="cma-mechanistic-analysis", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity/team name")  
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")

    ### Causal mediation analysis on activations
    parser.add_argument("--activation_name", type=str, default="resid_post", help="activation name")
    parser.add_argument("--patch_mlp_out", action="store_true", help="whether to patch MLP output as well for resid stream patching")
    parser.add_argument("--token_pos_list", nargs='+', type=int, default=[-1], help="target token position for doing causal mediation analysis")
    parser.add_argument("--patch_after_movement", action="store_true", help="patch activations after the movement event in theory of mind prompts")
    parser.add_argument("--patch_at_world_change", action="store_true", help="food_truck-only: patch right after 'drives the truck to <X>.' regardless of context_type. Takes precedence over --patch_after_movement.")
    parser.add_argument("--group_heads", action="store_true", help="whether to group attention heads by their grouped query attention (used for LLaMA-2 and Code Llama)")

    ### generation and evaluation parameters
    parser.add_argument("--samples_per_condition", type=int, default=1, help="number of samples per condition for generation")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="max new tokens for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for generation") 
    parser.add_argument("--generate", action="store_true", help="whether to use generation for causal mediation analysis instead of logits")
    parser.add_argument("--eval_metric", type=str, default="gen_acc", choices=["gen_acc", "ans_prob", "ab_forced"], help="evaluation metric")
    parser.add_argument("--low_prob_threshold", type=float, default=0.9, help="lower bound threshold for answer probability")
    parser.add_argument("--min_valid_sample_num", type=int, default=2, help="minimum number of valid samples for causal mediation analysis")

    ### statistical testing
    parser.add_argument("--run_statistical_tests", action="store_true", default=True, help="run permutation testing for statistical significance")
    parser.add_argument("--no_statistical_tests", action="store_false", dest="run_statistical_tests", help="disable statistical testing")
    parser.add_argument("--n_permutations", type=int, default=5000, help="number of permutations for statistical testing")
    parser.add_argument("--alpha", type=float, default=0.05, help="FWER significance level")

    ### logging and caching
    parser.add_argument("--log_dir", type=str, default="results/causal_analysis/cma", help="log directory") 
    parser.add_argument("--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--ungroup_grouped_query_attention", action="store_true", help="ungroup the keys/values which were grouped in the grouped query attention") 
    parser.add_argument("--load_generation_config", action="store_true", help="Load model generation config")
    parser.add_argument("--generation_config_name", type=str, default=None, help="Generation config name")
    parser.add_argument("--eos_token", type=str, default=None, help="End of sequence token")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional tag appended to result directory name (e.g. 'special-run')")

    args = parser.parse_args()
    
    # Set defaults if not specified
    if args.prompt_num is None:
        args.prompt_num = 3 if args.use_template_system else 50  # Default prompt numbers based on prompt type

    # Note: use_template_system requires generation mode for response collection
    # Do not force generate=False here as it conflicts with response logging needs
    
    return args



class CMAOrchestrator:
    """Orchestrates the full CMA experiment pipeline."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        
        # Initialize components
        self.model_loader = ModelLoader(config.model, config.generation, config.prompts)
        self.evaluator = GenerationEvaluator(config)
        self.filter = PromptFilter(config, self.evaluator)
        self.patcher = ActivationPatcher(config, self.evaluator)
        self.dual_stats = DualAnswerStatsTracker() if config.prompts.ask_world else None
    
    def run_experiment(self):
        """Run the complete CMA experiment."""
        logger.info(f"DEBUG: Config use_template_system={self.config.prompts.use_template_system}")
        logger.info(f"DEBUG: Config ask_world={self.config.prompts.ask_world}")
        logger.info(f"DEBUG: Config generate={self.config.generate}")
        set_seed(self.config.seed)
        
        # Load model and tokenizer
        model, tokenizer, generation_kwargs, eos_token_ids, A_tok_id, B_tok_id, model_id = (
            self.model_loader.load_model_and_tokenizer()
        )
        
        # Initialize wandb if enabled
        condition_name = self.config.prompts.condition.value if self.config.prompts.condition else None
        config_dict = {
                "model_type": self.config.model.model_type,
                "activation_name": self.config.patching.activation_name,
                "base_rule": self.config.prompts.base_rule,
                "condition": condition_name,
                "context_type": self.config.prompts.context_type,
                "samples_per_condition": self.config.generation.samples_per_condition,
                "temperature": self.config.generation.temperature,
                "eval_metric": self.config.evaluation.eval_metric,
                "template_names": self.config.prompts.template_names,
                "patch_after_movement": self.config.patching.patch_after_movement,
                "patch_at_world_change": self.config.patching.patch_at_world_change
            }
        logger.info(f"Condition: {condition_name}, context_type: {self.config.prompts.context_type}, "
                     f"answer_differs: {self.config.prompts.condition.answer_differs if self.config.prompts.condition else 'N/A'}")
        template_str = "_".join(config_dict['template_names']) if isinstance(config_dict['template_names'], list) else str(config_dict['template_names'])
        experiment_name = f"{self.config.model.model_type}_{template_str}_{self.config.patching.activation_name}_{self.config.prompts.base_rule}_{self.config.prompts.context_type}"
        
        if self.config.wandb.enabled:
            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=experiment_name,
                config=config_dict
            )  
    
        # Handle ungrouped query attention
        model_remark = ""
        if self.config.model.ungroup_grouped_query_attention:
            logger.info("ungroup the keys/values which were grouped in the grouped query attention")
            model.set_ungroup_grouped_query_attention(True)
            generation_kwargs["use_past_kv_cache"] = False
            model_remark += "_ungroup_gqa"
        
        # Create save directory
        save_folder = self._create_save_directory(model_id, model_remark)
        
        # Generate prompts
        prompt_generator = get_prompt_generator(self.config, tokenizer)
        logger.info(f"DEBUG: Using prompt generator: {type(prompt_generator).__name__}")
        prompts, correct_ans_list, causal_ans_list = prompt_generator.generate_prompts()
        logger.info(f"DEBUG: Generated {len(prompts)} prompts")
        if prompts:
            logger.info(f"DEBUG: First prompt base: {prompts[0][0][:100]}...")
            logger.info(f"DEBUG: First prompt exp: {prompts[0][1][:100]}...")
        if correct_ans_list:
            logger.info(f"DEBUG: First correct answers: {correct_ans_list[0]}")
        
        # Derive base_rule and exp_rule
        base_rule = self.config.prompts.base_rule
        exp_rule = self.config.get_exp_rule()
        
        # Save prompts to files
        self._save_prompts(save_folder, prompts, correct_ans_list, causal_ans_list, base_rule, exp_rule)
        
        # # Run activation patching
        # self._run_activation_patching(
        #     model, tokenizer, prompts, correct_ans_list, causal_ans_list,
        #     save_folder, eos_token_ids, A_tok_id, B_tok_id, base_rule, exp_rule, **generation_kwargs
        # )
        base_records, exp_records, valid_prompts, correct_ans_list = self._run_activation_patching(
            model, tokenizer, prompts, correct_ans_list, causal_ans_list,
            save_folder, eos_token_ids, A_tok_id, B_tok_id, base_rule, exp_rule, prompt_generator, **generation_kwargs
        )
        # Log experiment to wandb
        self._create_cma_prompt_artifact(base_records, exp_records, valid_prompts, correct_ans_list, save_folder)
        
        # Finish wandb run
        if self.config.wandb.enabled:
            wandb.finish()
        
        return save_folder
    
    def _create_save_directory(self, model_id: str, model_remark: str) -> str:
        """Create and return save directory path."""
        save_folder = self.config.get_save_folder(model_id, model_remark)
        logger.info(f"save_folder: {save_folder}")
        os.makedirs(save_folder, exist_ok=True)
        return save_folder
    
    def _save_prompts(self, save_folder: str, prompts: List[Tuple[str, str]], 
                     correct_ans_list: List[Tuple[str, str]], causal_ans_list: List[str], base_rule, exp_rule):
        """Save prompts and answers to files."""
        with open(os.path.join(save_folder, f"base_input_prompts_{self.config.prompts.prompt_num}.txt"), "w") as f:
            for base_prompt, _ in prompts:
                f.write(base_prompt + "\n")
        
        with open(os.path.join(save_folder, f"exp_input_prompts_{self.config.prompts.prompt_num}.txt"), "w") as f:
            for _, exp_prompt in prompts:
                f.write(exp_prompt + "\n")
        
        with open(os.path.join(save_folder, f"ans_{self.config.prompts.prompt_num}.txt"), "w") as f:
            for i, ((base_ans, exp_ans), causal_ans) in enumerate(zip(correct_ans_list, causal_ans_list)):
                f.write(f"{i}: base_ans: {base_ans}, exp_ans: {exp_ans}, causal_ans: {causal_ans}\n")

    def _save_model_responses(self, save_folder: str, records: List[dict], name: str):
        path = os.path.join(save_folder, f"model_responses_{self.config.prompts.prompt_num}_{name}.txt")
        with open(path, "w") as f:
            for record in records:
                sample_info = f" (sample {record['sample_idx']})" if 'sample_idx' in record else ""
                f.write(f"prompt{sample_info}: {record['prompt']}\n")
                f.write(f"response{sample_info}: {record['response']}\n")
                f.write("---\n")

    def _run_activation_patching(self, model, tokenizer, prompts, correct_ans_list,
                            causal_ans_list, save_folder, eos_token_ids, A_tok_id, B_tok_id,
                            base_rule, exp_rule, prompt_generator, **generation_kwargs):
        """Run activation patching analysis with streaming processing to reduce memory."""
        valid_prompts = []
        results_list = []

        # Response logs for ALL prompts (not just valid ones)
        base_records, exp_records = [], []
        base_correct_num_all, exp_correct_num_all = 0, 0

        logger.info(f"Starting evaluation of {len(prompts)} prompt pairs...")

        # STEP 1: Evaluate all prompts and collect stats (needed for dual-answer tracking)
        prompt_validity = []  # Track which prompts pass threshold

        for i, ((base_prompt, exp_prompt), (base_ans, exp_ans), causal_ans) in enumerate(
            zip(prompts, correct_ans_list, causal_ans_list)
        ):
            logger.info(f"=== Evaluating prompt pair {i+1}/{len(prompts)} ===")

            # Tokenize
            base_input_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(model.cfg.device)
            exp_input_ids = tokenizer(exp_prompt, return_tensors="pt").input_ids.to(model.cfg.device)

            # Evaluate both prompts
            if self.config.prompts.better_cma:
                logger.info("Using better_cma evaluation logic.")
                logger.info(f"Base ans: '{base_ans}', Exp ans: '{exp_ans}', Causal ans: '{causal_ans}'")
                logger.info(f"Base alternative_ans: '{causal_ans}', Exp alternative_ans: '{causal_ans}'")
                base_passes, base_gen_texts = self.filter.evaluate_and_check_threshold(
                    model, base_input_ids, eos_token_ids, tokenizer, base_ans, alternative_ans=causal_ans, prompt=base_prompt,
                    A_tok_id=A_tok_id, B_tok_id=B_tok_id, better_cma=True, exp_prompt=exp_prompt,
                    exp_input_ids=exp_input_ids, **generation_kwargs
                )
                exp_passes, exp_gen_texts = self.filter.evaluate_and_check_threshold(
                    model, exp_input_ids, eos_token_ids, tokenizer, exp_ans, alternative_ans=causal_ans, prompt=exp_prompt,
                    A_tok_id=A_tok_id, B_tok_id=B_tok_id, better_cma=True, base_prompt=base_prompt,
                    base_input_ids=base_input_ids, **generation_kwargs
                )
            else:
                logger.info("Using original CMA evaluation logic.")
                base_passes, base_gen_texts = self.filter.evaluate_and_check_threshold(
                    model, base_input_ids, eos_token_ids, tokenizer, base_ans, alternative_ans=exp_ans, prompt=base_prompt,
                    A_tok_id=A_tok_id, B_tok_id=B_tok_id, **generation_kwargs
                )
                exp_passes, exp_gen_texts = self.filter.evaluate_and_check_threshold(
                    model, exp_input_ids, eos_token_ids, tokenizer, exp_ans, alternative_ans=base_ans, prompt=exp_prompt,
                    A_tok_id=A_tok_id, B_tok_id=B_tok_id, **generation_kwargs
                )

            logger.info(f"Prompt {i+1}: base_passes={base_passes}, exp_passes={exp_passes}")

            # Count passing prompts for accuracy tracking
            if base_passes:
                base_correct_num_all += 1
            if exp_passes:
                exp_correct_num_all += 1

            # Record ALL responses (even from failed prompts)
            if base_gen_texts:
                for j, response in enumerate(base_gen_texts):
                    base_records.append({
                        "prompt": base_prompt,
                        "response": response,
                        "sample_idx": j,
                        "prompt_idx": i
                    })
            if exp_gen_texts:
                for j, response in enumerate(exp_gen_texts):
                    exp_records.append({
                        "prompt": exp_prompt,
                        "response": response,
                        "sample_idx": j,
                        "prompt_idx": i
                    })

            # Update dual-answer stats if enabled (for ALL prompts)
            if self.dual_stats and self.config.prompts.context_type == "abstract":
                if isinstance(base_ans, tuple):
                    base_belief, base_world = base_ans
                    exp_belief, exp_world = exp_ans
                    opposite_belief = exp_belief if self.config.prompts.base_rule == "ABA" else base_belief

                    if base_gen_texts:
                        self.dual_stats.update_base_stats(base_gen_texts, base_belief, base_world, opposite_belief)
                    if exp_gen_texts:
                        self.dual_stats.update_exp_stats(exp_gen_texts, exp_belief, exp_world, base_belief)

            # Track validity for second pass
            is_valid = base_passes and exp_passes
            prompt_validity.append(is_valid)
            if is_valid:
                valid_prompts.append((base_prompt, exp_prompt, base_ans, exp_ans, causal_ans))

        # Log accuracy to wandb
        valid_num = len(valid_prompts)
        if self.config.wandb.enabled:
            wandb.log({
                "valid_prompt_num": valid_num,
                "total_prompt_num": len(prompts),
                "valid_prompt_ratio": valid_num / len(prompts) if len(prompts) > 0 else 0.0,
                "base_accuracy": base_correct_num_all / len(prompts) if len(prompts) > 0 else 0.0,
                "exp_accuracy": exp_correct_num_all / len(prompts) if len(prompts) > 0 else 0.0,
            })

        # STEP 2: Check if we have enough valid prompts
        logger.info(f"Valid prompts for CMA: {valid_num}/{len(prompts)}")

        if self.config.evaluation.min_valid_sample_num > 0 and valid_num < self.config.evaluation.min_valid_sample_num:
            logger.warning(f"Not enough valid samples: {valid_num} < {self.config.evaluation.min_valid_sample_num}")
            logger.warning(f"Returning records for debugging without running CMA patching")
            return base_records, exp_records, valid_prompts, correct_ans_list

        # STEP 3: Run activation patching with STREAMING (cache → patch → clear)
        logger.info(f"Running activation patching on {valid_num} valid prompts (streaming mode)...")

        for i, (base_prompt, exp_prompt, base_ans, exp_ans, causal_ans) in enumerate(valid_prompts):
            logger.info(f"Patching valid prompt {i+1}/{valid_num}")

            # Quote-prefill readout (code conditions on tokenizers without
            # merged quote+char tokens, e.g. Gemma-2): the prompt already ends
            # with the opening quote, so the logit readout must use the
            # bare-char continuation ("'R'" -> "R'", first token = R). Applies
            # to the unpatched diffs below AND the patched diffs inside
            # run_activation_patching, which receives these same variables.
            if getattr(self.config.prompts, "code_quote_prefill", False):
                causal_ans = causal_ans[1:]
                exp_ans = exp_ans[1:]

            # Tokenize fresh (small overhead, huge memory savings)
            base_input_ids = tokenizer(base_prompt, return_tensors="pt").input_ids.to(model.cfg.device)
            exp_input_ids = tokenizer(exp_prompt, return_tensors="pt").input_ids.to(model.cfg.device)

            # Cache base activations (this is the memory-heavy part)
            base_logits, base_cache = model.run_with_cache(base_input_ids, names_filter=lambda x: True)

            # Calculate exp logits difference if needed
            exp_logits_diff = None
            base_logits_diff = None
            if not self.config.generate:
                exp_logits = model(exp_input_ids)
                if self.config.evaluation.eval_metric == "ab_forced":
                    from evaluation import calculate_ab_margin
                    exp_logits_diff = calculate_ab_margin(exp_logits, A_tok_id, B_tok_id)
                    base_logits_diff = calculate_ab_margin(base_logits, A_tok_id, B_tok_id)
                else:
                    from evaluation import calculate_logit_probability_difference
                    exp_logits_diff = calculate_logit_probability_difference(
                        exp_logits, model, causal_ans=causal_ans, original_ans=exp_ans
                    )
                    base_logits_diff = calculate_logit_probability_difference(
                        base_logits, model, causal_ans=causal_ans, original_ans=exp_ans
                    )

            # Determine token position for patching
            if self.config.patching.patch_at_world_change:
                # food_truck-only: string-anchored patch at "drives the truck to <X>."
                # Used by C3/C4 within-template residual-stream experiment. Same anchor
                # in both base and exp; world state matches at the patch point.
                base_pos = prompt_generator.detect_post_movement_position(base_prompt, tokenizer)
                exp_pos = prompt_generator.detect_post_movement_position(exp_prompt, tokenizer)
                if base_pos == -1 or exp_pos == -1:
                    raise ValueError(
                        f"Post-movement anchor 'drives the truck to <X>.' not found in prompt {i+1} "
                        f"(base: {base_pos}, exp: {exp_pos}). --patch_at_world_change is food_truck-only.")
                token_pos = (base_pos, exp_pos)
                logger.info(f"Patching at post-movement: base pos {base_pos}, exp pos {exp_pos} for prompt {i+1}")
            elif self.config.patching.patch_after_movement:
                context_type = self.config.prompts.context_type
                # ABA -> base=false context, ABB -> base=true context. Always.
                base_is_aba = (self.config.prompts.base_rule == "ABA")

                if context_type in ("abstract", "control"):
                    condition = self.config.prompts.condition
                    same_belief = getattr(condition, "same_belief", False) if condition is not None else False
                    if same_belief:
                        # C8 bf-control: both prompts are false-belief, detector uses false marker for both.
                        base_pos = prompt_generator.detect_belief_formation_position(
                            base_prompt, tokenizer, is_false_belief=True)
                        exp_pos = prompt_generator.detect_belief_formation_position(
                            exp_prompt, tokenizer, is_false_belief=True)
                    else:
                        # Belief formation markers: ABA base is false context -> is_false_belief=True
                        base_pos = prompt_generator.detect_belief_formation_position(
                            base_prompt, tokenizer, is_false_belief=base_is_aba)
                        exp_pos = prompt_generator.detect_belief_formation_position(
                            exp_prompt, tokenizer, is_false_belief=not base_is_aba)

                    if base_pos == -1 or exp_pos == -1:
                        raise ValueError(
                            f"Belief formation position not found in prompt {i+1} "
                            f"(base: {base_pos}, exp: {exp_pos}). "
                            f"Check that template markers match the prompt text.")
                    token_pos = (base_pos, exp_pos)
                    logger.info(f"Patching at belief formation: base pos {base_pos}, exp pos {exp_pos} for prompt {i+1}")

                elif context_type == "photo":
                    condition = self.config.prompts.condition
                    if condition is not None and condition.patch_at_photo_event:
                        # Patch at photo-taking event (epistemic access moment)
                        base_pos = prompt_generator.detect_photo_position(base_prompt, tokenizer)
                        exp_pos = prompt_generator.detect_photo_position(exp_prompt, tokenizer)

                        if base_pos == -1 or exp_pos == -1:
                            raise ValueError(
                                f"Photo position not found in prompt {i+1} "
                                f"(base: {base_pos}, exp: {exp_pos}). "
                                f"Check that photo regex patterns match the prompt text.")
                        token_pos = (base_pos, exp_pos)
                        logger.info(f"Patching at PHOTO event: base pos {base_pos}, exp pos {exp_pos} for prompt {i+1}")
                    else:
                        # Movement position detection (regex-based, no agent markers)
                        base_pos = prompt_generator.detect_movement_position(base_prompt, tokenizer)
                        exp_pos = prompt_generator.detect_movement_position(exp_prompt, tokenizer)

                        if base_pos == -1 or exp_pos == -1:
                            raise ValueError(
                                f"Movement position not found in prompt {i+1} "
                                f"(base: {base_pos}, exp: {exp_pos}). "
                                f"Check that movement regex patterns match the prompt text.")
                        token_pos = (base_pos, exp_pos)
                        logger.info(f"Patching at movement position: base pos {base_pos}, exp pos {exp_pos} for prompt {i+1}")

                elif context_type == "code":
                    # base kind: ABA=copy, ABB=alias (all code conditions).
                    # exp flips the kind, except code_control (within_context)
                    # where both prompts share the base kind.
                    condition = self.config.prompts.condition
                    within = condition is not None and condition.within_context
                    exp_is_copy = base_is_aba if within else not base_is_aba
                    base_pos = prompt_generator.detect_copy_semantics_position(
                        base_prompt, tokenizer, is_copy=base_is_aba)
                    exp_pos = prompt_generator.detect_copy_semantics_position(
                        exp_prompt, tokenizer, is_copy=exp_is_copy)

                    if base_pos == -1 or exp_pos == -1:
                        raise ValueError(
                            f"Copy semantics position not found in prompt {i+1} "
                            f"(base: {base_pos}, exp: {exp_pos}). "
                            f"Check that code template markers match the prompt text.")
                    token_pos = (base_pos, exp_pos)
                    logger.info(f"Patching at copy semantics: base pos {base_pos}, exp pos {exp_pos} for prompt {i+1}")

                else:
                    raise ValueError(f"patch_after_movement not supported for context_type={context_type}")
            else:
                token_pos = self.config.patching.token_pos_list[0]

            # Run patching and get result (just scalars/small tensors)
            result = self.patcher.run_activation_patching(
                model=model,
                base_cache=base_cache,
                exp_input_ids=exp_input_ids,
                exp_logits_diff=exp_logits_diff,
                causal_ans=causal_ans,
                exp_ans=exp_ans,
                token_pos=token_pos,
                device=model.cfg.device,
                eos_token_id=eos_token_ids,
                tokenizer=tokenizer,
                correct_ans=exp_ans,
                prompt=exp_prompt,
                A_tok_id=A_tok_id,
                B_tok_id=B_tok_id,
                base_logits_diff=base_logits_diff,
                **generation_kwargs
            )
            results_list.append(result)

            # base_cache goes out of scope here and gets garbage collected
            # memory usage: only stores ~32 small result tensors instead of 32 huge caches
        
        # STEP 5: Save results
        if self.config.generate:
            # For generation mode, results are tuples: (correct_num_all, total_num_all, acc_all)
            acc_tensors = [r[2] for r in results_list]
            mean_acc = self._stack_and_average_tensors(acc_tensors)
            final_save_folder = os.path.join(
                save_folder, "generate",
                f"sample_num_{valid_num}_{self.config.evaluation.eval_metric}_{self.config.evaluation.low_prob_threshold}"
            )
            os.makedirs(final_save_folder, exist_ok=True)
            torch.save(mean_acc, os.path.join(final_save_folder, "acc_all.pt"))
        else:
            # For logit mode: residual-stream patching now returns (raw, normalized);
            # head-level patching returns a single tensor (no normalization). Detect
            # which case we're in and split accordingly.
            is_layer_tuple = isinstance(results_list[0], tuple) and len(results_list[0]) == 2
            if is_layer_tuple:
                raw_list = [r[0] for r in results_list]
                norm_list = [r[1] for r in results_list if r[1] is not None]
                if len(norm_list) != len(raw_list):
                    norm_list = None  # mixed / partial; skip normalized save
            else:
                raw_list = results_list
                norm_list = None

            mean_logits_diff = self._stack_and_average_tensors(raw_list)

            if mean_logits_diff.ndim == 1:
                mean_logits_diff = mean_logits_diff[:, None]  # (n_layers,) -> (n_layers, 1)


            final_save_folder = os.path.join(
                save_folder, "logit",
                f"sample_num_{valid_num}_{self.config.evaluation.eval_metric}_{self.config.evaluation.low_prob_threshold}"
            )
            os.makedirs(final_save_folder, exist_ok=True)
            torch.save(mean_logits_diff, os.path.join(final_save_folder, "causal_scores.pt"))

            # save per-sample scores for statistical testing
            per_sample_scores = torch.stack(raw_list, dim=0)
            torch.save(per_sample_scores, os.path.join(final_save_folder, "causal_scores_per_sample.pt"))
            logger.info(f"saved per-sample scores with shape {per_sample_scores.shape}")

            if norm_list is not None:
                mean_norm = self._stack_and_average_tensors(norm_list)
                if mean_norm.ndim == 1:
                    mean_norm = mean_norm[:, None]
                torch.save(mean_norm, os.path.join(final_save_folder, "causal_scores_normalized.pt"))
                per_sample_norm = torch.stack(norm_list, dim=0)
                torch.save(per_sample_norm, os.path.join(final_save_folder, "causal_scores_normalized_per_sample.pt"))
                logger.info(f"saved normalized scores; mean range "
                            f"[{float(mean_norm.min()):.3f}, {float(mean_norm.max()):.3f}]")

            patch_info = ""
            if self.config.patching.patch_at_world_change:
                patch_info = " | patch at post-movement"
            elif self.config.patching.patch_after_movement:
                ctx = self.config.prompts.context_type
                condition = self.config.prompts.condition
                if ctx in ('abstract', 'control'):
                    patch_label = 'belief formation'
                elif ctx == 'code':
                    patch_label = 'copy semantics'
                elif condition is not None and condition.patch_at_photo_event:
                    patch_label = 'photo event'
                else:
                    patch_label = 'movement position'
                patch_info = f" | patch at {patch_label}"
            metric_name = f"CMA: {base_rule}→{exp_rule} ({self.config.prompts.context_type}){patch_info}"
            
            plot(mean_logits_diff.cpu().numpy(), final_save_folder, metric_name=metric_name)
            
            if self.config.wandb.enabled:
                plot_filename = f"{metric_name}_heatmap.png"
                plot_path = os.path.join(final_save_folder, plot_filename)
                if os.path.exists(plot_path):
                    wandb.log({"causal_scores_heatmap": wandb.Image(plot_path)})
                else:
                    logger.warning(f"Plot file not found: {plot_path}")
        
        # Save model responses
        self._save_model_responses(final_save_folder, base_records, name="base")
        self._save_model_responses(final_save_folder, exp_records, name="exp")
        
        # Save dual-answer stats if enabled
        if self.dual_stats:
            self.dual_stats.save_stats(os.path.join(final_save_folder, "tom_dual_stats.txt"))

        # Run statistical testing if per-sample scores exist
        if not self.config.generate and self.config.evaluation.run_statistical_tests:
            per_sample_path = os.path.join(final_save_folder, "causal_scores_per_sample.pt")
            if os.path.exists(per_sample_path):
                try:
                    from statistical_testing import permutation_test_heads, save_statistical_results

                    logger.info("running permutation testing for statistical significance...")
                    per_sample_scores = torch.load(per_sample_path)

                    results_df, threshold = permutation_test_heads(
                        per_sample_scores,
                        n_permutations=self.config.evaluation.n_permutations,
                        alpha=self.config.evaluation.alpha,
                    )

                    csv_path, sig_path, summary_path = save_statistical_results(
                        results_df,
                        final_save_folder,
                        threshold=threshold
                    )

                    logger.info(f"statistical results saved:")
                    logger.info(f"  full table: {csv_path}")
                    logger.info(f"  significant heads: {sig_path}")
                    logger.info(f"  summary: {summary_path}")

                    # optionally log to wandb
                    if self.config.wandb.enabled:
                        wandb.log({
                            "n_significant_heads": int(results_df['significant'].sum()),
                            "statistical_summary": wandb.Table(dataframe=results_df)
                        })

                except Exception as e:
                    logger.error(f"statistical testing failed: {e}")
                    logger.info("continuing without statistical analysis...")

        logger.info(f"Results saved to {final_save_folder}")
        return base_records, exp_records, valid_prompts, correct_ans_list
    




    def _stack_and_average_tensors(self, tensor_list):
        """Handle stacking tensors with potentially different shapes."""
        if not tensor_list:
            raise ValueError("Cannot stack empty tensor list")
        
        # Debug tensor shapes
        logger.info(f"Stacking {len(tensor_list)} tensors with shapes: {[t.shape for t in tensor_list[:3]]}...")
        
        # Check if all tensors have the same shape
        first_shape = tensor_list[0].shape
        if all(t.shape == first_shape for t in tensor_list):
            # All same shape - use standard stacking
            result = torch.stack(tensor_list).mean(dim=0)
            logger.info(f"Standard stacking result shape: {result.shape}")
            return result
        
        # Handle variable shapes (common for layer activations with different sequence lengths)
        logger.info(f"Handling variable tensor shapes: {[t.shape for t in tensor_list[:3]]}...")
        
        # For layer activations, pad to max sequence length
        if len(first_shape) == 2:  # (n_layers, seq_len) format
            max_seq_len = max(t.shape[1] for t in tensor_list)
            n_layers = first_shape[0]
            device = tensor_list[0].device
            
            padded_tensors = []
            for tensor in tensor_list:
                if tensor.shape[1] < max_seq_len:
                    # Pad with zeros on the right
                    pad_size = max_seq_len - tensor.shape[1]
                    padded = torch.cat([tensor, torch.zeros(n_layers, pad_size, device=device)], dim=1)
                    padded_tensors.append(padded)
                else:
                    padded_tensors.append(tensor)
            
            result = torch.stack(padded_tensors).mean(dim=0)
            logger.info(f"Padded stacking result shape: {result.shape}")
            return result
        else:
            if len(first_shape) == 1 and all(t.shape == first_shape for t in tensor_list):
                result = torch.stack(tensor_list).mean(dim=0)
                logger.info(f"1D stacking result shape: {result.shape}")
                return result
            else:
                raise ValueError("Unsupported tensor shape for stacking with variable sizes")
                return tensor_list[0]

            
    def _create_cma_prompt_artifact(self, base_records, exp_records, valid_prompts, correct_ans_list, save_folder):
        """Create wandb artifact with base/exp prompt-response pairs following behavioral pattern."""
        if not self.config.wandb.enabled:
            logger.debug("Wandb not available, skipping artifact creation")
            return
        
        try:
            import tempfile
            import json
            
            # Create artifact
            artifact_name = f"cma_prompt_responses_{self.config.prompts.context_type}_{self.config.prompts.base_rule}"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="cma_prompt_response_data", 
                description=f"CMA base/exp prompt-response pairs for {self.config.model.model_type}"
            )
            
            # Extract metadata
            run_metadata = {
                'model_name': self.config.model.model_type,
                'activation_name': self.config.patching.activation_name,
                'context_type': self.config.prompts.context_type,
                'base_rule': self.config.prompts.base_rule,
                'use_template_system': self.config.prompts.use_template_system,
                'prompt_num': self.config.prompts.prompt_num,
                'valid_prompts': len(valid_prompts),
                'total_base_responses': len(base_records),
                'total_exp_responses': len(exp_records),
                'save_folder': save_folder
            }
            
            # Group records by prompt_idx for easier pairing
            base_by_prompt = {}
            for record in base_records:
                prompt_idx = record.get('prompt_idx', 0)
                if prompt_idx not in base_by_prompt:
                    base_by_prompt[prompt_idx] = []
                base_by_prompt[prompt_idx].append(record)
            
            exp_by_prompt = {}  
            for record in exp_records:
                prompt_idx = record.get('prompt_idx', 0)
                if prompt_idx not in exp_by_prompt:
                    exp_by_prompt[prompt_idx] = []
                exp_by_prompt[prompt_idx].append(record)
            
            # Create paired entries
            paired_responses = []
            all_prompt_indices = set(base_by_prompt.keys()) | set(exp_by_prompt.keys())
            
            for prompt_idx in sorted(all_prompt_indices):
                base_responses = base_by_prompt.get(prompt_idx, [])
                exp_responses = exp_by_prompt.get(prompt_idx, [])
                
                # Get expected answers for this prompt
                expected_answers = correct_ans_list[prompt_idx] if prompt_idx < len(correct_ans_list) else ("", "")
                
                paired_entry = {
                    'prompt_idx': prompt_idx,
                    'expected_base_answer': expected_answers[0] if isinstance(expected_answers, tuple) else str(expected_answers),
                    'expected_exp_answer': expected_answers[1] if isinstance(expected_answers, tuple) else str(expected_answers),
                    'base': {
                        'prompt': base_responses[0]['prompt'] if base_responses else '',
                        'responses': [r['response'] for r in base_responses]
                    },
                    'exp': {
                        'prompt': exp_responses[0]['prompt'] if exp_responses else '',
                        'responses': [r['response'] for r in exp_responses]
                    }
                }
                paired_responses.append(paired_entry)
            
            # Final structure matching behavioral pattern but adapted for base/exp
            structured_data = {
                'run_metadata': run_metadata,
                'paired_responses': paired_responses
            }
            
            # Save as JSON and add to artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(structured_data, f, indent=2)
                temp_path = f.name
            
            artifact.add_file(temp_path, name=f"cma_prompt_responses.json")
            
            # Log artifact to wandb
            wandb.log_artifact(artifact)
            logger.info(f"Created wandb artifact: {artifact_name}")
            
            # Clean up temp file    
            import os
            os.unlink(temp_path)
            
        except Exception as e:
            logger.warning(f"Failed to create wandb artifact: {e}")
        
def main(args):
    """Main entry point using refactored architecture."""
    config = ExperimentConfig.from_args(args)
    orchestrator = CMAOrchestrator(config)
    save_folder = orchestrator.run_experiment()
    logger.info(f"Experiment completed. Results saved to: {save_folder}")


if __name__ == "__main__":
    args = get_args()
    main(args)