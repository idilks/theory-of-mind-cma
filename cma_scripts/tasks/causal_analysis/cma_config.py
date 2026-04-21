"""
Configuration classes for CMA experiments.
Replaces parameter soup with structured configuration objects.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
import os


class CMACondition(Enum):
    """Maps 1:1 to the PI's 5 conditions in the 20-CMA design."""
    ABSTRACT_BELIEF = "abstract_belief"              # C1: belief false↔true, answer SAME
    ABSTRACT_PHOTO = "abstract_photo"                # C2: photo false↔true, answer SAME
    ANSWER_CHANGES_BELIEF = "answer_changes_belief"  # C3: belief false↔true, answer DIFFERENT
    ANSWER_CHANGES_PHOTO = "answer_changes_photo"    # C4: photo false↔true, answer DIFFERENT
    CONTROL = "control"                              # C5: location swap, answer DIFFERENT
    ABSTRACT_PHOTO_NEW = "abstract_photo_new"        # C2 variant: patch at photo event, answer SAME
    ANSWER_CHANGES_PHOTO_NEW = "answer_changes_photo_new"  # C4 variant: patch at photo event, answer DIFFERENT
    CODE_ANSWER_CHANGES = "code_answer_changes"      # C6: copy vs reference semantics, answer DIFFERENT
    CODE_CONTROL = "code_control"                    # C7: direct mutation only, answer DIFFERENT

    @property
    def context_type(self) -> str:
        """Which prompt template family to use."""
        if self in (CMACondition.ABSTRACT_BELIEF, CMACondition.ANSWER_CHANGES_BELIEF):
            return "abstract"
        elif self in (CMACondition.ABSTRACT_PHOTO, CMACondition.ANSWER_CHANGES_PHOTO,
                      CMACondition.ABSTRACT_PHOTO_NEW, CMACondition.ANSWER_CHANGES_PHOTO_NEW):
            return "photo"
        elif self in (CMACondition.CODE_ANSWER_CHANGES, CMACondition.CODE_CONTROL):
            return "code"
        return "control"

    @property
    def answer_differs(self) -> bool:
        """Whether base and exp prompts have different correct answers."""
        return self in (
            CMACondition.ANSWER_CHANGES_BELIEF,
            CMACondition.ANSWER_CHANGES_PHOTO,
            CMACondition.ANSWER_CHANGES_PHOTO_NEW,
            CMACondition.CONTROL,
            CMACondition.CODE_ANSWER_CHANGES,
            CMACondition.CODE_CONTROL,
        )

    @property
    def patch_at_photo_event(self) -> bool:
        """Whether to patch at the photo-taking event instead of movement."""
        return self in (CMACondition.ABSTRACT_PHOTO_NEW, CMACondition.ANSWER_CHANGES_PHOTO_NEW)

# Constants
LOC_L, LOC_R = '<loc>', '</loc>'
LINE_SEP = "=" * 50

@dataclass
class ModelConfig:
    """Model loading and device configuration."""
    model_type: str
    device_map: str = "cpu"
    device: str = "cuda" 
    n_devices: int = 2
    fold_ln: bool = False
    ungroup_grouped_query_attention: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    @property
    def needs_model_remark(self) -> bool:
        return self.ungroup_grouped_query_attention

@dataclass 
class GenerationConfig:
    """Generation parameters for model evaluation."""
    max_new_tokens: int = 30
    samples_per_condition: int = 1 # reduced for faster testing
    load_generation_config: bool = False
    generation_config_name: Optional[str] = None
    eos_token: Optional[str] = None
    
    # Default generation kwargs
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    temperature: float = 0.2

@dataclass
class PromptConfig:
    """Prompt generation configuration."""
    base_rule: str  # "ABA" or "ABB"
    context_type: str  # "abstract" or "token" — derived from condition when condition is set
    prompt_num: int
    condition: Optional[CMACondition] = None  # First-class condition enum (overrides context_type)
    in_context_example_num: int = 0
    sep_symbol: str = "^"
    do_shuffle: bool = False

    # Theory of mind specific
    use_template_system: bool = False  # Use unified template system for ToM prompts
    tom_locations_file: str = "tom_datasets/locations.txt"
    tom_format: str = "direct"  # "direct" or "multiple_choice"
    prompt_variant: str = "standard"  # "standard" or "detailed"
    ask_world: bool = False
    question_style: str = "instruction"  # "completion" or "instruction" for CMA questions
    template_names: List[str] = None  # Templates to use for unified ToM system
    better_cma: bool = False  # Use improved CMA prompt generation logic

    def __post_init__(self):
        if self.base_rule not in ["ABA", "ABB"]:
            raise ValueError(f"base_rule must be ABA or ABB, got {self.base_rule}")

        # If condition is set, derive context_type from it
        if self.condition is not None:
            self.context_type = self.condition.context_type

        # Set default template names for behavioral ToM integration
        if self.template_names is None:
            self.template_names = ["basic_object_move"]
        if self.context_type not in ["abstract", "token", "control", "photo", "basic", "code"]:
            raise ValueError(f"context_type must be abstract, token, control, photo, or basic, got {self.context_type}")

@dataclass
class PatchingConfig:
    """Activation patching configuration."""
    activation_name: str = "resid_post" # default to residual stream post-activation
    patch_mlp_out: bool = False
    token_pos_list: List[int] = None
    group_heads: bool = False
    patch_after_movement: bool = False  # Patch after movement event in ToM prompts
    
    def __post_init__(self):
        if self.token_pos_list is None:
            self.token_pos_list = [-1]

@dataclass
class EvaluationConfig:
    """Evaluation and filtering configuration."""
    eval_metric: str = "gen_acc"  # "gen_acc", "ans_prob", "ab_forced"
    low_prob_threshold: float = 0.9
    min_valid_sample_num: int = 10

    # statistical testing parameters
    run_statistical_tests: bool = True  # enable/disable permutation testing
    n_permutations: int = 5000  # number of permutations (per LLMSymbMech Section B.2)
    alpha: float = 0.05  # FWER significance level

    def __post_init__(self):
        if self.eval_metric not in ["gen_acc", "ans_prob", "ab_forced"]:
            raise ValueError(f"eval_metric must be one of gen_acc, ans_prob, ab_forced")

@dataclass
class WandbConfig:
    """Wandb logging configuration."""
    project: Optional[str] = "cma-mechanistic-analysis"
    entity: Optional[str] = None
    enabled: bool = True

@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all sub-configs."""
    model: ModelConfig
    generation: GenerationConfig
    prompts: PromptConfig  
    patching: Optional[PatchingConfig]
    evaluation: EvaluationConfig
    wandb: WandbConfig
    
    seed: int = 0
    log_dir: str = "results/causal_analysis/cma"
    generate: bool = False
    verbose: bool = False
    run_tag: Optional[str] = None
    
    @classmethod
    def from_args(cls, args):
        """Create ExperimentConfig from argparse namespace."""
        # Resolve condition: explicit --condition takes precedence, else infer from legacy flags
        condition = cls._resolve_condition(args)

        return cls(
            model=ModelConfig(
                model_type=args.model_type,
                device_map=args.device_map,
                device=args.device,
                n_devices=args.n_devices,
                fold_ln=args.fold_ln,
                ungroup_grouped_query_attention=args.ungroup_grouped_query_attention
            ),
            generation=GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                samples_per_condition=args.samples_per_condition,
                load_generation_config=args.load_generation_config,
                generation_config_name=args.generation_config_name,
                eos_token=args.eos_token
            ),
            prompts=PromptConfig(
                base_rule=args.base_rule,
                context_type=args.context_type,  # will be overridden by condition in __post_init__
                prompt_num=args.prompt_num,
                condition=condition,
                in_context_example_num=args.in_context_example_num,
                sep_symbol=args.sep_symbol,
                do_shuffle=args.do_shuffle,
                use_template_system=args.use_template_system,
                tom_locations_file=args.tom_locations_file,
                tom_format=args.tom_format,
                ask_world=args.ask_world,
                question_style=getattr(args, 'question_style', 'instruction'),
                template_names=getattr(args, 'template_names', ["food_truck"]),
                better_cma=getattr(args, 'better_cma', False)
            ),
            patching=PatchingConfig(
                activation_name=args.activation_name,
                patch_mlp_out=args.patch_mlp_out,
                token_pos_list=args.token_pos_list,
                group_heads=args.group_heads,
                patch_after_movement=getattr(args, 'patch_after_movement', False)
            ),
            evaluation=EvaluationConfig(
                eval_metric=args.eval_metric,
                low_prob_threshold=args.low_prob_threshold,
                min_valid_sample_num=args.min_valid_sample_num,
                run_statistical_tests=args.run_statistical_tests,
                n_permutations=args.n_permutations,
                alpha=args.alpha
            ),
            wandb=WandbConfig(
                project=getattr(args, 'wandb_project', 'cma-mechanistic-analysis'),
                entity=getattr(args, 'wandb_entity', None),
                enabled=not getattr(args, 'no_wandb', False)
            ),
            seed=args.seed,
            log_dir=args.log_dir,
            generate=args.generate,
            verbose=args.verbose,
            run_tag=getattr(args, 'run_tag', None)
        )

    @staticmethod
    def _resolve_condition(args) -> Optional[CMACondition]:
        """Resolve CMACondition from --condition or infer from legacy flags."""
        condition_str = getattr(args, 'condition', None)
        if condition_str is not None:
            return CMACondition(condition_str)

        # Backward compat: infer condition from context_type + patch_after_movement
        context_type = args.context_type
        patch_after = getattr(args, 'patch_after_movement', False)

        if context_type == "control":
            return CMACondition.CONTROL
        elif context_type == "photo":
            return CMACondition.ABSTRACT_PHOTO
        elif context_type == "abstract":
            if patch_after:
                return CMACondition.ANSWER_CHANGES_BELIEF
            else:
                return CMACondition.ABSTRACT_BELIEF

        elif context_type == "code":
            return CMACondition.CODE_ANSWER_CHANGES

        # Non-ToM context types (token, basic) — no condition mapping
        return None
    
    def get_exp_rule(self) -> str:
        """Derive the experimental rule label from base_rule and condition.

        Same-answer conditions (answer_differs=False): flip rule (ABA↔ABB).
        Different-answer conditions: keep rule (ABA→ABA).
        """
        rule_list = ["ABA", "ABB"]
        if self.prompts.condition is not None:
            # Same answers -> flip rule; different answers -> keep rule
            if not self.prompts.condition.answer_differs:
                exp_rule_idx = 1 - rule_list.index(self.prompts.base_rule)
            else:
                exp_rule_idx = rule_list.index(self.prompts.base_rule)
        elif self.prompts.context_type == "abstract":
            # Legacy fallback for when condition is None
            exp_rule_idx = 1 - rule_list.index(self.prompts.base_rule)
        else:
            exp_rule_idx = rule_list.index(self.prompts.base_rule)
        return rule_list[exp_rule_idx]

    def get_save_folder(self, model_id: str, model_remark: str = "") -> str:
        """Generate save folder path based on configuration."""
        exp_rule = self.get_exp_rule()

        # add template names to path
        template_str = "_".join(self.prompts.template_names) if self.prompts.template_names else "no_template"
        patch_str = "patch_after_movement" if self.patching.patch_after_movement else "patch_before_movement"

        # Use condition name if available, else fall back to context_type_context
        if self.prompts.condition is not None:
            causal_remark = self.prompts.condition.value
        else:
            causal_remark = f"{self.prompts.context_type}_context"
        tag_str = f"_{self.run_tag}" if self.run_tag else ""
        remark = f"{model_id}{model_remark}/{causal_remark}/template_{template_str}/base_rule_{self.prompts.base_rule}_exp_rule_{exp_rule}/{patch_str}/{self.patching.activation_name}_seed_{self.seed}_shuffle_{self.prompts.do_shuffle}{tag_str}"

        return os.path.join(self.log_dir, remark)
    
    
