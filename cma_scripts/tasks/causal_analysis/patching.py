"""
Activation patching core functionality for CMA experiments.
"""

import logging
import os
import sys
from functools import partial
from typing import List, Tuple, Union, Callable

import torch
import transformer_lens.utils as utils
from tqdm import tqdm

from models import CustomHookedTransformer
from evaluation import GenerationEvaluator, calculate_logit_probability_difference, calculate_ab_margin

# Add codebase root to path for utils import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils import HEAD_ACTIVATIONS

logger = logging.getLogger(__name__)

class ActivationPatcher:
    """Handles activation patching operations."""
    
    def __init__(self, config, evaluator: GenerationEvaluator):
        self.config = config
        self.evaluator = evaluator
    
    def patch_head_activations(
        self,
        model: CustomHookedTransformer,
        base_cache,
        input_ids: torch.Tensor,
        token_pos: Union[int, List[int], Tuple[int, int]],
        exp_logits_diff: torch.Tensor,
        patched_exp_ans: str,
        exp_ans: str,
        total_layers: int,
        total_heads: int,
        group_size: int = 1,
        activation_name: str = "z",
        device: str = "cuda",
        eos_token_id: List[int] = None,
        tokenizer = None,
        correct_ans: str = None,
        prompt: str = None,
        A_tok_id: int = None,
        B_tok_id: int = None,
        **generation_kwargs
    ):
        """Patch activations in different layers x heads at certain token positions."""
        
        # Handle different token_pos formats
        if isinstance(token_pos, tuple) and len(token_pos) == 2:
            base_pos, exp_pos = token_pos
        else:
            # Backward compatibility: same position for both
            base_pos = exp_pos = token_pos
        
        def replace_head_activation_hook(activation, hook, head_idx, token_len, group_size):
            activation_tag = hook.name
            if activation.shape[1] == token_len:
                # Extract from base_cache at base_pos (where movement happened in base)
                rule_1_act_cache = base_cache[activation_tag][:, base_pos, head_idx * group_size: (head_idx+1) * group_size, :]
                # Insert into current activation at exp_pos (where movement happens in exp)
                activation[:, exp_pos, head_idx * group_size: (head_idx+1) * group_size, :] = rule_1_act_cache
        
        assert input_ids is not None
        token_len = input_ids.shape[-1]
        assert token_len > 1, "token length should be larger than 1"
        
        if self.config.generate:
            correct_num_all = torch.zeros((total_layers, total_heads), device=device)
            total_num_all = torch.zeros((total_layers, total_heads), device=device)
            acc_all = torch.zeros((total_layers, total_heads), device=device)
        else:
            logits_diff_change = torch.zeros((total_layers, total_heads), device=device)
        
        for layer_idx in tqdm(range(total_layers)):
            act_tag = utils.get_act_name(activation_name, layer_idx)
            
            for head_idx in range(total_heads):
                if self.config.generate:
                    correct_num, total_num, acc = self.evaluator.evaluate_generation_accuracy(
                        model, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
                        fwd_hooks=[(
                            act_tag,
                            partial(replace_head_activation_hook, head_idx=head_idx, 
                                   token_len=token_len, group_size=group_size)
                        )], 
                        **generation_kwargs
                    )
                    correct_num_all[layer_idx, head_idx] = correct_num
                    total_num_all[layer_idx, head_idx] = total_num
                    acc_all[layer_idx, head_idx] = acc
                else:
                    patched_logits = model.run_with_hooks(
                        input_ids,
                        fwd_hooks=[(
                            act_tag,
                            partial(replace_head_activation_hook, head_idx=head_idx, 
                                   token_len=token_len, group_size=group_size)
                        )]
                    )
                    
                    if self.config.evaluation.eval_metric == "ab_forced":
                        patched_score = calculate_ab_margin(patched_logits, A_tok_id, B_tok_id)
                        logits_diff_change[layer_idx, head_idx] = patched_score - exp_logits_diff
                    else:
                        patched_exp_logits_diff = calculate_logit_probability_difference(
                            patched_logits, model, causal_ans=patched_exp_ans, original_ans=exp_ans
                        )
                        logits_diff_change[layer_idx, head_idx] = patched_exp_logits_diff - exp_logits_diff
        
        if self.config.generate:
            return correct_num_all, total_num_all, acc_all
        else:
            return logits_diff_change
    
    def patch_layer_activations(
        self,
        model: CustomHookedTransformer,
        base_cache,
        input_ids: torch.Tensor,
        exp_logits_diff: torch.Tensor,
        patched_exp_ans: str,
        exp_ans: str,
        total_layers: int,
        activation_name: str = "attn_out",
        patch_mlp_out: bool = True,
        device: str = "cuda",
        eos_token_id: List[int] = None,
        tokenizer = None,
        correct_ans: str = None,
        prompt: str = None,
        A_tok_id: int = None,
        B_tok_id: int = None,
        movement_positions: Tuple[int, int] = None,
        token_pos: int = -1,
        base_logits_diff: torch.Tensor = None,
        **generation_kwargs
    ):
        """Patch activations in different layers at certain token positions."""
        
        num_position = input_ids.shape[-1]
        assert num_position > 1, "token length should be larger than 1"
        
        if self.config.generate:
            correct_num_all = torch.zeros((total_layers, num_position), device=device)
            total_num_all = torch.zeros((total_layers, num_position), device=device)
            acc_all = torch.zeros((total_layers, num_position), device=device)
        else:
            logits_diff_change = torch.zeros((total_layers,), device=device)
            logits_diff_normalized = torch.zeros((total_layers,), device=device) if base_logits_diff is not None else None
        
        for layer_idx in tqdm(range(total_layers)):
            if self.config.generate:
                # Generation mode: loop over positions (needed for positional accuracy)
                def replace_layer_activation_hook(activation, hook, token_pos, token_len):
                    activation_tag = hook.name
                    if activation.shape[1] == token_len:
                        # Handle movement position mapping if provided
                        if movement_positions is not None:
                            base_pos, exp_pos = movement_positions
                            # Only patch if this token_pos is the exp movement position
                            if token_pos == exp_pos:
                                rule_1_act_cache = base_cache[activation_tag][:, base_pos, :]
                                activation[:, exp_pos, :] = rule_1_act_cache
                        else:
                            # Default behavior: patch same position
                            rule_1_act_cache = base_cache[activation_tag][:, token_pos, :]
                            activation[:, token_pos, :] = rule_1_act_cache
                
                for token_pos in range(num_position):
                    act_tag = utils.get_act_name(activation_name, layer_idx)
                    fwd_hooks = [(
                        act_tag,
                        partial(replace_layer_activation_hook, token_pos=token_pos, token_len=num_position)
                    )]
                    
                    if activation_name == "attn_out" and patch_mlp_out:
                        fwd_hooks.append((
                            utils.get_act_name("mlp_out", layer_idx),
                            partial(replace_layer_activation_hook, token_pos=token_pos, token_len=num_position)
                        ))
                    
                    correct_num, total_num, acc = self.evaluator.evaluate_generation_accuracy(
                        model, input_ids, eos_token_id, tokenizer, correct_ans, prompt,
                        fwd_hooks=fwd_hooks, **generation_kwargs
                    )
                    correct_num_all[layer_idx, token_pos] = correct_num
                    total_num_all[layer_idx, token_pos] = total_num
                    acc_all[layer_idx, token_pos] = acc
            else:
                # Logit-diff mode: patch a SINGLE token position. movement_positions
                # carries (base_pos, exp_pos) when the call site detected a structural
                # anchor (post-movement, belief formation, etc.); otherwise fall back
                # to token_pos for both (default -1 = end token).
                if movement_positions is not None:
                    base_pos, exp_pos = movement_positions
                else:
                    base_pos = exp_pos = token_pos

                def replace_position_hook(activation, hook):
                    activation_tag = hook.name
                    if activation.shape[1] == num_position:
                        activation[:, exp_pos, :] = base_cache[activation_tag][:, base_pos, :]

                act_tag = utils.get_act_name(activation_name, layer_idx)
                fwd_hooks = [(act_tag, replace_position_hook)]

                if activation_name == "attn_out" and patch_mlp_out:
                    fwd_hooks.append((
                        utils.get_act_name("mlp_out", layer_idx),
                        replace_position_hook
                    ))

                patched_logits = model.run_with_hooks(input_ids, fwd_hooks=fwd_hooks)
                patched_exp_logits_diff = calculate_logit_probability_difference(
                    patched_logits, model, causal_ans=patched_exp_ans, original_ans=exp_ans
                )
                raw_change = patched_exp_logits_diff - exp_logits_diff
                logits_diff_change[layer_idx] = raw_change

                if logits_diff_normalized is not None:
                    # Normalized indirect effect (Wang/Conmy/Prakash convention):
                    #   (patched - exp) / (base - exp)
                    # 0 = no patch effect; 1 = patch fully flipped exp toward base.
                    # When base and exp give nearly the same logit gap, the metric
                    # is undefined; fall back to 0 instead of dividing by ~zero.
                    denom = base_logits_diff - exp_logits_diff
                    if torch.is_tensor(denom):
                        denom_val = denom.detach().float()
                    else:
                        denom_val = torch.tensor(float(denom), device=device)
                    if torch.abs(denom_val) < 1e-6:
                        logits_diff_normalized[layer_idx] = 0.0
                    else:
                        logits_diff_normalized[layer_idx] = raw_change / denom_val
        
        if self.config.generate:
            return correct_num_all, total_num_all, acc_all
        else:
            # Return (raw, normalized) so the caller can save both. normalized is None
            # when base_logits_diff was not provided (older callers).
            return logits_diff_change, logits_diff_normalized

    def run_activation_patching(
        self,
        model: CustomHookedTransformer,
        base_cache,
        exp_input_ids: torch.Tensor,
        exp_logits_diff: torch.Tensor,
        causal_ans: str,
        exp_ans: str,
        token_pos: Union[int, List[int]],
        device: str,
        A_tok_id: int = None,
        B_tok_id: int = None,
        **kwargs
    ):
        """Run appropriate patching method based on activation type."""
        total_layers = model.cfg.n_layers
        total_heads = model.cfg.n_heads
        n_key_value_heads = model.cfg.n_key_value_heads or total_heads
        
        # Adjust for grouped query attention
        if ("k" in self.config.patching.activation_name or "v" in self.config.patching.activation_name) and \
           (not model.cfg.ungroup_grouped_query_attention):
            total_heads = n_key_value_heads
        
        group_size = 1
        if self.config.patching.group_heads:
            group_size = total_heads // n_key_value_heads
            total_heads = total_heads // group_size

        # Pop layer-only kwarg so it doesn't leak into the head/generation path.
        base_logits_diff = kwargs.pop("base_logits_diff", None)

        if self.config.patching.activation_name in HEAD_ACTIVATIONS:
            return self.patch_head_activations(
                model=model,
                base_cache=base_cache,
                input_ids=exp_input_ids,
                token_pos=token_pos,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans,
                exp_ans=exp_ans,
                total_layers=total_layers,
                total_heads=total_heads,
                group_size=group_size,
                activation_name=self.config.patching.activation_name,
                device=device,
                A_tok_id=A_tok_id,
                B_tok_id=B_tok_id,
                **kwargs
            )
        
        elif "resid" in self.config.patching.activation_name or "out" in self.config.patching.activation_name:
            # Pass movement positions for layer patching if available
            movement_positions = token_pos if isinstance(token_pos, tuple) else None
            token_pos_int = token_pos if isinstance(token_pos, int) else -1
            return self.patch_layer_activations(
                model=model,
                base_cache=base_cache,
                input_ids=exp_input_ids,
                exp_logits_diff=exp_logits_diff,
                patched_exp_ans=causal_ans,
                exp_ans=exp_ans,
                total_layers=total_layers,
                activation_name=self.config.patching.activation_name,
                patch_mlp_out=self.config.patching.patch_mlp_out,
                device=device,
                A_tok_id=A_tok_id,
                B_tok_id=B_tok_id,
                movement_positions=movement_positions,
                token_pos=token_pos_int,
                base_logits_diff=base_logits_diff,
                **kwargs
            )
            
        else:
            raise ValueError(f"Unknown activation name: {self.config.patching.activation_name}")