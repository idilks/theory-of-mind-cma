"""
Cumulative ablation experiment for belief-tracking heads.

Establishes necessity: do the heads identified by permutation-tested CMA
need to be active for the model to perform false-belief reasoning?

Usage:
    python codebase/tasks/causal_analysis/ablation_experiment.py \
        --model_type Qwen2.5-14B-Instruct \
        --prompt_num 50 \
        --seed 42
"""

import argparse
import os
import sys
import random
import logging
from functools import partial

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ensure project root on path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

codebase_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
if codebase_root not in sys.path:
    sys.path.insert(0, codebase_root)

# need to be in the right directory for relative imports in models.py
os.chdir(codebase_root)

from tasks.causal_analysis.models import ModelLoader, setup_hf_authentication
from tasks.causal_analysis.cma_config import ModelConfig, GenerationConfig, PromptConfig
from tasks.causal_analysis.prompt_generators.tom_templates import (
    UnifiedTemplateSystem, UnifiedQuestionFormatter
)
from tasks.causal_analysis.prompt_generators.base import locations_match

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# head list computation
# ---------------------------------------------------------------------------

def load_permutation_csv(path):
    """Load a wandb permutation test CSV. Returns DataFrame with layer, head, observed_score, significant."""
    df = pd.read_csv(path)
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)
    df["observed_score"] = df["observed_score"].astype(float)
    df["significant"] = df["significant"].astype(str).str.lower() == "true"
    return df


def load_abstract_only_heads(wandb_dir):
    """
    Compute abstract-only heads:
      significant in ALL abstract templates AND NOT significant in ANY control template.
    Returns (ranked_heads, all_scores_df).
      ranked_heads: list of (layer, head) sorted by |mean abstract score| descending
      all_scores_df: DataFrame with mean abstract score per head
    """
    abstract_dir = os.path.join(wandb_dir, "abstract")
    control_dir = os.path.join(wandb_dir, "control", "belief_patch")

    # load abstract CSVs (healthy only)
    abstract_files = sorted([
        os.path.join(abstract_dir, f)
        for f in os.listdir(abstract_dir)
        if f.endswith(".csv") and "healthy" in f
    ])
    assert len(abstract_files) >= 1, f"no healthy abstract CSVs found in {abstract_dir}"
    logger.info(f"loading {len(abstract_files)} abstract CSVs: {[os.path.basename(f) for f in abstract_files]}")

    abstract_dfs = [load_permutation_csv(f) for f in abstract_files]

    # load control CSVs
    control_files = sorted([
        os.path.join(control_dir, f)
        for f in os.listdir(control_dir)
        if f.endswith(".csv") and "corrupt" not in f
    ])
    assert len(control_files) >= 1, f"no control CSVs found in {control_dir}"
    logger.info(f"loading {len(control_files)} control CSVs: {[os.path.basename(f) for f in control_files]}")

    control_dfs = [load_permutation_csv(f) for f in control_files]

    # find heads significant in ALL abstract templates
    sig_sets = []
    for df in abstract_dfs:
        sig = set(zip(df.loc[df["significant"], "layer"], df.loc[df["significant"], "head"]))
        sig_sets.append(sig)
    abstract_sig = sig_sets[0]
    for s in sig_sets[1:]:
        abstract_sig = abstract_sig & s

    logger.info(f"heads significant in all {len(abstract_dfs)} abstract templates: {len(abstract_sig)}")

    # find heads significant in ANY control template
    control_sig = set()
    for df in control_dfs:
        sig = set(zip(df.loc[df["significant"], "layer"], df.loc[df["significant"], "head"]))
        control_sig = control_sig | sig

    logger.info(f"heads significant in any control template: {len(control_sig)}")

    # abstract-only = abstract ∩ ¬control
    abstract_only = abstract_sig - control_sig
    logger.info(f"abstract-only heads: {len(abstract_only)}")

    # compute mean abstract score for ranking
    score_records = []
    for layer, head in abstract_only:
        scores = []
        for df in abstract_dfs:
            row = df[(df["layer"] == layer) & (df["head"] == head)]
            if len(row) > 0:
                scores.append(row["observed_score"].values[0])
        score_records.append({
            "layer": layer, "head": head,
            "mean_score": np.mean(scores),
            "abs_mean_score": np.abs(np.mean(scores))
        })

    scores_df = pd.DataFrame(score_records)
    scores_df = scores_df.sort_values("abs_mean_score", ascending=False).reset_index(drop=True)

    ranked_heads = list(zip(scores_df["layer"], scores_df["head"]))
    return ranked_heads, scores_df


def _heads_from_overlap_csv(overlap_csv, cond_a, cond_b, set_name):
    """Mirror of cross_condition_per_token.heads_from_overlap_csv.
    Auto-flips a_only/b_only if the user wrote the pair backwards."""
    df = pd.read_csv(overlap_csv)
    m = ((df["cond_a"] == cond_a) & (df["cond_b"] == cond_b)
         & (df["set"] == set_name))
    sub = df[m]
    if sub.empty:
        flip = {"a_only": "b_only", "b_only": "a_only", "intersection": "intersection"}
        m2 = ((df["cond_a"] == cond_b) & (df["cond_b"] == cond_a)
              & (df["set"] == flip[set_name]))
        sub = df[m2]
    if sub.empty:
        return []
    return [(int(L), int(H)) for L, H in sub[["layer", "head"]].astype(int).values]


def load_head_set_from_csv(args):
    """Dispatch on args.head_source. Returns (ranked_heads, scores_df).

    ranked_heads: list[(layer, head)] in ablation order (most-important first).
    scores_df: DataFrame with at least layer, head, mean_score, abs_mean_score.
    """
    source = args.head_source
    if source == "wandb":
        return load_abstract_only_heads(args.wandb_dir)

    if source == "overlap_set":
        if not args.head_csv:
            raise ValueError("--head_csv required for --head_source=overlap_set")
        if not args.pair or not args.set:
            raise ValueError("--pair and --set required for --head_source=overlap_set")
        cond_a, cond_b = [s.strip() for s in args.pair.split(",")]
        heads = _heads_from_overlap_csv(args.head_csv, cond_a, cond_b, args.set)
        if not heads:
            raise ValueError(
                f"no heads found in {args.head_csv} for "
                f"({cond_a}, {cond_b}, {args.set})"
            )
        # No native ranking metric in overlap CSV — keep CSV order, NaN scores.
        scores_df = pd.DataFrame(heads, columns=["layer", "head"])
        scores_df["mean_score"] = np.nan
        scores_df["abs_mean_score"] = np.nan
        logger.info(f"loaded {len(heads)} heads from overlap_set "
                    f"({cond_a},{cond_b},{args.set})")
        return heads, scores_df

    if source == "per_condition":
        if not args.head_csv:
            raise ValueError("--head_csv required for --head_source=per_condition")
        if not args.condition:
            raise ValueError("--condition required for --head_source=per_condition")
        df = pd.read_csv(args.head_csv)
        sub = df[df["condition"] == args.condition].copy()
        if sub.empty:
            raise ValueError(
                f"no rows for condition={args.condition} in {args.head_csv}"
            )
        rank_col = args.rank_by  # "max_observed_score" or "mean_observed_score"
        if rank_col not in sub.columns:
            raise ValueError(f"{rank_col} not in {args.head_csv} columns: "
                             f"{list(sub.columns)}")
        sub = sub.sort_values(rank_col, ascending=False).reset_index(drop=True)
        heads = list(zip(sub["layer"].astype(int), sub["head"].astype(int)))
        scores_df = pd.DataFrame({
            "layer": sub["layer"].astype(int),
            "head": sub["head"].astype(int),
            "mean_score": sub[rank_col].astype(float),
            "abs_mean_score": sub[rank_col].astype(float).abs(),
        })
        logger.info(f"loaded {len(heads)} heads for condition={args.condition} "
                    f"ranked by {rank_col}")
        return heads, scores_df

    raise ValueError(f"unknown head_source: {source}")


def build_control_heads(target_heads, wandb_dir):
    """
    Build control set: for each layer that has target heads, pick the lowest-|score|
    heads from that layer (same layer distribution, different heads).
    """
    abstract_dir = os.path.join(wandb_dir, "abstract")
    abstract_files = sorted([
        os.path.join(abstract_dir, f)
        for f in os.listdir(abstract_dir)
        if f.endswith(".csv") and "healthy" in f
    ])
    abstract_dfs = [load_permutation_csv(f) for f in abstract_files]

    # compute mean |score| for ALL heads
    all_heads = set(zip(abstract_dfs[0]["layer"], abstract_dfs[0]["head"]))
    head_scores = {}
    for layer, head in all_heads:
        scores = []
        for df in abstract_dfs:
            row = df[(df["layer"] == layer) & (df["head"] == head)]
            if len(row) > 0:
                scores.append(row["observed_score"].values[0])
        head_scores[(layer, head)] = np.abs(np.mean(scores))

    target_set = set(target_heads)

    # count how many target heads per layer
    layer_counts = {}
    for layer, head in target_heads:
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    # for each layer, pick the lowest-|score| non-target heads
    control_heads = []
    for layer, count in sorted(layer_counts.items()):
        # all heads in this layer, excluding target heads
        candidates = [
            (layer, h) for (l, h) in all_heads
            if l == layer and (l, h) not in target_set
        ]
        candidates.sort(key=lambda x: head_scores[x])
        picked = candidates[:count]
        if len(picked) < count:
            logger.warning(f"layer {layer}: only {len(picked)} control heads available, needed {count}")
        control_heads.extend(picked)

    # sort control heads by the same ranking metric as target (|score| descending)
    # so cumulative ablation is comparable
    control_heads.sort(key=lambda x: head_scores[x], reverse=True)
    return control_heads


# ---------------------------------------------------------------------------
# prompt generation (unpaired)
# ---------------------------------------------------------------------------

# Each eval *variant* produces one prompt per sampled instance.
# Spec fields:
#   context:           "abstract" | "photo" | "basic" (which template family)
#   subtemplate_attr:  attribute on template holding the format string
#                      (e.g. "false_context_template", "true_context_template",
#                       or "template" for BasicTemplate)
#   correct_attr:      attribute on template holding the var-name of the
#                      correct answer key (e.g. "belief_answer_key" → "street_1")
#   ordering:          "natural"  -> use subs as sampled
#                      "swapped"  -> swap subs[var_1] and subs[var_2] before
#                                    formatting (gives the directional twin
#                                    for state_only / basic)
#
# alt_answer is always derived as the value at the *other* contrastive slot
# (var_1 vs var_2). It's the answer the model would give if the head we're
# ablating was driving an override — the CMA "causal answer" analogue.
EVAL_VARIANT_SPECS = {
    "false_belief": {
        "context": "abstract", "subtemplate_attr": "false_context_template",
        "correct_attr": "belief_answer_key", "ordering": "natural",
    },
    "true_belief": {
        "context": "abstract", "subtemplate_attr": "true_context_template",
        "correct_attr": "world_answer_key", "ordering": "natural",
    },
    "false_photo": {
        "context": "photo", "subtemplate_attr": "false_context_template",
        "correct_attr": "photo_answer_key", "ordering": "natural",
    },
    "true_photo": {
        "context": "photo", "subtemplate_attr": "true_context_template",
        "correct_attr": "world_answer_key", "ordering": "natural",
    },
    "basic_ab": {
        "context": "basic", "subtemplate_attr": "template",
        "correct_attr": "answer_key", "ordering": "natural",
    },
    "basic_ba": {
        "context": "basic", "subtemplate_attr": "template",
        "correct_attr": "answer_key", "ordering": "swapped",
    },
}

# Tolerated input aliases (resolved before lookup). "basic" used to be the
# only basic eval; now there are two directional variants — default to ab.
EVAL_VARIANT_ALIASES = {"basic": "basic_ab"}

DEFAULT_EVAL_VARIANTS = ["false_belief", "true_belief", "basic_ab"]


# Assistant prefill table keyed by (context, template_name). Appended *after*
# apply_chat_template(..., add_generation_prompt=True) so the model's first
# generated token IS the answer (e.g. " Monroe"), rather than a preamble like
# "Alice would go to...". Each string ends with a trailing space so the next
# token is leading-space-prefixed; run_cumulative_ablation handles the leading
# space when looking up first-answer token IDs.
#
# Keyed by context (abstract/photo/basic) because false_belief and true_belief
# share the same observer-centric question and answer-noun (street/color/
# section); they differ only in which key counts as "correct". Same for
# false_photo/true_photo and basic_ab/basic_ba.
PREFILL_TABLE = {
    "abstract": {
        "food_truck":   "Alice would go to ",
        "hair_styling": "Jennifer would say Alice's hair is ",
        "library_book": "Sarah would look for her book in the ",
    },
    "photo": {
        "food_truck":   "The photograph shows the food truck at ",
        "hair_styling": "The ID card photo shows Alice's hair color as ",
        "library_book": "The photograph shows the book in the ",
    },
    "basic": {
        "food_truck":   "The food truck is at ",
        "hair_styling": "Alice's hair is ",
        "library_book": "The book is in the ",
    },
}


def _resolve_variant(name):
    return EVAL_VARIANT_ALIASES.get(name, name)


def _render_one(variant, template_system, question_formatter, tokenizer,
                template_name, subs, var_1, var_2, system_prompt,
                prefill_mode="none"):
    """Render one prompt for the given variant.

    Returns (prompt, correct_answer, alt_answer). alt is the value at the
    contrastive slot opposite to the correct one.

    prefill_mode:
      "none"         -> prompt ends right after <|im_start|>assistant header
      "per_variant"  -> append PREFILL_TABLE[context][template_name] so the
                        model's first generated token IS the answer
    """
    variant = _resolve_variant(variant)
    if variant not in EVAL_VARIANT_SPECS:
        raise ValueError(
            f"unknown eval variant {variant!r}. Known: "
            f"{sorted(EVAL_VARIANT_SPECS)}"
        )
    spec = EVAL_VARIANT_SPECS[variant]
    template = template_system.get_template(template_name, spec["context"])

    use_subs = dict(subs)
    if spec["ordering"] == "swapped":
        use_subs[var_1], use_subs[var_2] = subs[var_2], subs[var_1]

    template_subs = {k: v for k, v in use_subs.items() if k in template.variables}
    scenario = getattr(template, spec["subtemplate_attr"]).format(**template_subs)

    correct_key = getattr(template, spec["correct_attr"])  # e.g. "street_1"
    correct = template_subs[correct_key]
    # alt = whichever of {var_1, var_2} is NOT the correct slot
    alt_key = var_2 if correct_key == var_1 else var_1
    alt = template_subs[alt_key]

    question = question_formatter.get_cma_questions(
        template_name, context_type=spec["context"], belief_type="ABA",
        question_style="instruction",
    )
    prompt = _format_chat(tokenizer, system_prompt, f"{scenario}\n{question}")

    if prefill_mode == "per_variant":
        try:
            prefill = PREFILL_TABLE[spec["context"]][template_name]
        except KeyError as e:
            raise KeyError(
                f"no prefill defined for (context={spec['context']!r}, "
                f"template={template_name!r}). Add an entry to PREFILL_TABLE."
            ) from e
        prompt = prompt + prefill
    elif prefill_mode != "none":
        raise ValueError(f"unknown prefill_mode {prefill_mode!r}")

    return prompt, correct, alt


def generate_unpaired_prompts(template_system, question_formatter, tokenizer,
                               template_names, num_prompts, system_prompt,
                               eval_variants=None, prefill_mode="none"):
    """Generate one prompt per (template × instance × variant).

    Returns dict[variant -> {"prompts": [...], "correct": [...], "alt": [...]}].
    All variants share the same sampled substitutions per instance, so the
    correct/alt pair lines up across true/false twins.
    """
    if eval_variants is None:
        eval_variants = DEFAULT_EVAL_VARIANTS
    eval_variants = [_resolve_variant(v) for v in eval_variants]
    for v in eval_variants:
        if v not in EVAL_VARIANT_SPECS:
            raise ValueError(
                f"unknown eval variant {v!r}. Known: "
                f"{sorted(EVAL_VARIANT_SPECS)}"
            )

    prompts_per_template = max(1, num_prompts // len(template_names))
    results = {v: {"prompts": [], "correct": [], "alt": []}
               for v in eval_variants}

    for template_name in template_names:
        # Abstract template gives the canonical variable schema (var_1, var_2);
        # basic/photo templates share these contrastive slots.
        abstract_template = template_system.get_template(template_name, "abstract")
        var_1 = [v for v in abstract_template.variables if v.endswith("_1")][0]
        var_2 = [v for v in abstract_template.variables if v.endswith("_2")][0]

        for _ in range(prompts_per_template):
            subs = {}
            for var_name, vocab_type in abstract_template.variables.items():
                subs[var_name] = template_system.vocab_loader.sample_different(vocab_type, 1)[0]
            vt = abstract_template.variables[var_1]
            v1, v2 = template_system.vocab_loader.sample_different(vt, 2)
            subs[var_1] = v1
            subs[var_2] = v2

            for variant in eval_variants:
                prompt, correct, alt = _render_one(
                    variant, template_system, question_formatter, tokenizer,
                    template_name, subs, var_1, var_2, system_prompt,
                    prefill_mode=prefill_mode,
                )
                results[variant]["prompts"].append(prompt)
                results[variant]["correct"].append(correct)
                results[variant]["alt"].append(alt)

    return results


def _format_chat(tokenizer, system_prompt, user_content):
    """Apply chat template to a single prompt."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# ablation hooks
# ---------------------------------------------------------------------------

def zero_heads_hook(activation, hook, head_dict):
    """Zero out specified heads' z activations at all positions."""
    layer_idx = hook.layer()
    if layer_idx in head_dict:
        for h in head_dict[layer_idx]:
            activation[:, :, h, :] = 0
    return activation


def build_ablation_hooks(head_list):
    """Build forward hooks list for zeroing the given heads."""
    head_dict = {}
    for layer, head in head_list:
        head_dict.setdefault(layer, []).append(head)

    act_filter = lambda name: name.endswith("hook_z")
    fwd_hooks = [(act_filter, partial(zero_heads_hook, head_dict=head_dict))]
    return fwd_hooks


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def get_first_answer_token_id(tokenizer, answer, prepend_space=False):
    """Get the first token ID of the expected answer.

    For multi-token answers like "Oak Street", returns the ID of "Oak".
    The absolute probability may be noisy for multi-token answers,
    but relative degradation across ablation steps is the signal.

    prepend_space=True encodes " <answer>" instead of "<answer>". Required
    when the prompt ends in whitespace (e.g. assistant prefill ending with
    a trailing space) — in that case the model's next BPE token will be
    leading-space-prefixed, so we need to score that token, not the
    no-leading-space variant.
    """
    text = (" " + answer) if prepend_space else answer
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return tokens[0]


def evaluate_prompt(model, tokenizer, prompt, correct_answer, alt_answer,
                     correct_tok, alt_tok,
                     eos_token_ids, fwd_hooks=None, max_new_tokens=5):
    """Evaluate a single prompt. Returns
    (correct_match, alt_match, p_correct, p_alt).

    correct_match / alt_match: did the generated text match the correct /
        alt answer (mutually exclusive in practice for short location-style
        answers, but not enforced — both could be False if the model bailed).
    p_correct / p_alt: probability of the first answer token under the
        next-token distribution at the prompt's final position.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    hooks = fwd_hooks or []

    logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    p_correct = probs[correct_tok].item()
    p_alt = probs[alt_tok].item()

    generated = model.generate_with_hooks(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_ids,
        fwd_hooks=hooks,
    )
    gen_text = tokenizer.decode(
        generated[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()
    correct_match = locations_match(gen_text, correct_answer)
    alt_match = locations_match(gen_text, alt_answer)

    return correct_match, alt_match, p_correct, p_alt


# ---------------------------------------------------------------------------
# cumulative ablation (per-prompt, following yukang_dev/ablation.py structure)
# ---------------------------------------------------------------------------

def run_cumulative_ablation(model, tokenizer, prompt_dict, ranked_heads,
                             eos_token_ids, max_new_tokens, label="target"):
    """Ablate heads cumulatively (top-1..top-N) and track correct/alt metrics.

    prompt_dict: {variant -> {"prompts", "correct", "alt"}}
    Returns (summary_rows, per_prompt_data).
      per_prompt_data[variant] = {
          "p_correct": np.array(n_prompts, n_steps),
          "p_alt":     np.array(n_prompts, n_steps),
          "acc":       np.array(n_prompts, n_steps),  # gen matches correct
          "flip":      np.array(n_prompts, n_steps),  # gen matches alt
      }
    """
    variants = list(prompt_dict.keys())
    n_steps = len(ranked_heads) + 1  # k=0..N

    # Precompute first-token IDs for both correct and alt per variant.
    # When the rendered prompt ends in whitespace (assistant prefill active),
    # the model's next token is leading-space-prefixed — encode the answer
    # with a leading space to match. We check per-prompt so a mixed run
    # (some variants prefilled, some not) would still score correctly,
    # though in practice prefill is applied uniformly.
    correct_toks = {}
    alt_toks = {}
    for v in variants:
        prompts_v = prompt_dict[v]["prompts"]
        correct_toks[v] = [
            get_first_answer_token_id(tokenizer, a,
                                      prepend_space=p.endswith(" "))
            for a, p in zip(prompt_dict[v]["correct"], prompts_v)
        ]
        alt_toks[v] = [
            get_first_answer_token_id(tokenizer, a,
                                      prepend_space=p.endswith(" "))
            for a, p in zip(prompt_dict[v]["alt"], prompts_v)
        ]

    per_prompt_data = {}
    for v in variants:
        n_prompts = len(prompt_dict[v]["prompts"])
        per_prompt_data[v] = {
            "p_correct": np.zeros((n_prompts, n_steps)),
            "p_alt":     np.zeros((n_prompts, n_steps)),
            "acc":       np.zeros((n_prompts, n_steps)),
            "flip":      np.zeros((n_prompts, n_steps)),
        }

    for k in range(n_steps):
        hooks = None if k == 0 else build_ablation_hooks(ranked_heads[:k])
        if k > 0:
            logger.info(f"[{label}] k={k}, ablating {ranked_heads[k-1]}")
        else:
            logger.info(f"[{label}] k=0 (baseline)")

        for v in variants:
            prompts = prompt_dict[v]["prompts"]
            corrects = prompt_dict[v]["correct"]
            alts = prompt_dict[v]["alt"]
            ctoks = correct_toks[v]
            atoks = alt_toks[v]

            for p_idx in range(len(prompts)):
                cm, am, pc, pa = evaluate_prompt(
                    model, tokenizer, prompts[p_idx],
                    corrects[p_idx], alts[p_idx],
                    ctoks[p_idx], atoks[p_idx],
                    eos_token_ids, fwd_hooks=hooks,
                    max_new_tokens=max_new_tokens,
                )
                per_prompt_data[v]["p_correct"][p_idx, k] = pc
                per_prompt_data[v]["p_alt"][p_idx, k] = pa
                per_prompt_data[v]["acc"][p_idx, k] = float(cm)
                per_prompt_data[v]["flip"][p_idx, k] = float(am)

            acc = per_prompt_data[v]["acc"][:, k].mean()
            flip = per_prompt_data[v]["flip"][:, k].mean()
            pc = per_prompt_data[v]["p_correct"][:, k].mean()
            pa = per_prompt_data[v]["p_alt"][:, k].mean()
            logger.info(f"  {v}: acc={acc:.3f} flip={flip:.3f} "
                        f"p_correct={pc:.4f} p_alt={pa:.4f}")

    # long-form summary: one row per (k, curve, eval_variant)
    EPS = 1e-12
    summary = []
    for k in range(n_steps):
        for v in variants:
            pc_arr = per_prompt_data[v]["p_correct"][:, k]
            pa_arr = per_prompt_data[v]["p_alt"][:, k]
            flip_score = pa_arr / (pa_arr + pc_arr + EPS)
            summary.append({
                "k": k,
                "curve": label,
                "eval_variant": v,
                "accuracy": float(per_prompt_data[v]["acc"][:, k].mean()),
                "flip_rate": float(per_prompt_data[v]["flip"][:, k].mean()),
                "mean_p_correct": float(pc_arr.mean()),
                "mean_p_alt": float(pa_arr.mean()),
                "p_correct_std": float(pc_arr.std()),
                "p_alt_std": float(pa_arr.std()),
                "flip_score": float(flip_score.mean()),
            })

    return summary, per_prompt_data


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_results(results_df, output_path, run_label="target"):
    """3-row × N-col grid (rows = accuracy/flip-rate, P(correct) vs P(alt),
    flip-score). One column per eval_variant; target and control overlaid."""
    variants = list(dict.fromkeys(results_df["eval_variant"].tolist()))
    n = len(variants)
    fig, axes = plt.subplots(3, n, figsize=(5 * n, 11), sharex=True, squeeze=False)

    target_color = "#d62728"
    control_color = "#1f77b4"

    for col, variant in enumerate(variants):
        ax_acc, ax_prob, ax_score = axes[0, col], axes[1, col], axes[2, col]

        for curve_type, base_color, ls, marker in [
            ("target", target_color, "-", "o"),
            ("control", control_color, "--", "s"),
        ]:
            sub = results_df[
                (results_df["eval_variant"] == variant)
                & (results_df["curve"] == curve_type)
            ].sort_values("k")
            if sub.empty:
                continue

            # row 0: accuracy (solid) + flip rate (dotted) on same axis
            ax_acc.plot(sub["k"], sub["accuracy"],
                        color=base_color, linestyle=ls, marker=marker,
                        markersize=3, linewidth=2, alpha=0.9,
                        label=f"{curve_type} acc")
            ax_acc.plot(sub["k"], sub["flip_rate"],
                        color=base_color, linestyle=":", marker=marker,
                        markersize=3, linewidth=2, alpha=0.7,
                        label=f"{curve_type} flip")

            # row 1: P(correct) and P(alt)
            ax_prob.plot(sub["k"], sub["mean_p_correct"],
                         color=base_color, linestyle=ls, marker=marker,
                         markersize=3, linewidth=2, alpha=0.9,
                         label=f"{curve_type} P(correct)")
            ax_prob.plot(sub["k"], sub["mean_p_alt"],
                         color=base_color, linestyle=":", marker=marker,
                         markersize=3, linewidth=2, alpha=0.7,
                         label=f"{curve_type} P(alt)")

            # row 2: flip score = P(alt) / (P(alt) + P(correct))
            ax_score.plot(sub["k"], sub["flip_score"],
                          color=base_color, linestyle=ls, marker=marker,
                          markersize=3, linewidth=2, alpha=0.9,
                          label=curve_type)

        ax_acc.set_title(variant, fontsize=11)
        ax_acc.set_ylim(-0.05, 1.05)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.legend(fontsize=7, loc="upper right", ncol=2)

        ax_prob.grid(True, alpha=0.3)
        ax_prob.legend(fontsize=7, loc="upper right", ncol=2)

        ax_score.set_xlabel("# heads ablated (cumulative)", fontsize=10)
        ax_score.set_ylim(-0.05, 1.05)
        ax_score.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
        ax_score.grid(True, alpha=0.3)
        ax_score.legend(fontsize=7, loc="lower right")

    axes[0, 0].set_ylabel("accuracy / flip rate", fontsize=11)
    axes[1, 0].set_ylabel("P(first token)", fontsize=11)
    axes[2, 0].set_ylabel("flip score = P(alt)/(P(alt)+P(corr))", fontsize=10)

    fig.suptitle(f"Ablation of {run_label}", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=150)
    logger.info(f"plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cumulative ablation of belief-tracking heads")
    parser.add_argument("--model_type", type=str, default="Qwen2.5-14B-Instruct")
    parser.add_argument("--prompt_num", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--templates", nargs="+", default=["food_truck", "hair_styling", "library_book"])
    parser.add_argument("--wandb_dir", type=str, default="wandb_permtest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/ablation")
    parser.add_argument("--n_devices", type=int, default=1)
    parser.add_argument("--device_map", type=str, default="cuda")
    # Pluggable head source (defaults reproduce existing wandb behavior)
    parser.add_argument("--head_source", choices=["wandb", "overlap_set", "per_condition"],
                        default="wandb")
    parser.add_argument("--head_csv", type=str, default=None,
                        help="overlap CSV (for overlap_set) or per-condition CSV (for per_condition)")
    parser.add_argument("--pair", type=str, default=None,
                        help="for overlap_set: 'cond_a,cond_b'")
    parser.add_argument("--set", choices=["intersection", "a_only", "b_only"], default=None,
                        help="for overlap_set: which subset within the pair")
    parser.add_argument("--condition", type=str, default=None,
                        help="for per_condition: which condition to pull from the CSV")
    parser.add_argument("--rank_by",
                        choices=["max_observed_score", "mean_observed_score", "mean_score"],
                        default="max_observed_score",
                        help="for per_condition: which column to rank by")
    # Eval prompt variants
    parser.add_argument("--eval_variants", nargs="+", default=DEFAULT_EVAL_VARIANTS,
                        help=f"any subset of {sorted(EVAL_VARIANT_SPECS)} "
                             f"(aliases: {EVAL_VARIANT_ALIASES})")
    # BC alias for old flag name
    parser.add_argument("--eval_conditions", nargs="+", default=None,
                        help="DEPRECATED alias for --eval_variants")
    parser.add_argument("--run_label", type=str, default="abstract_only_heads",
                        help="label for plot title and bookkeeping")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant. Answer concisely.",
                        help="system prompt used for all eval prompts. The "
                             "previous default ('reasoning about agents' beliefs') "
                             "biased photo evals toward mentalizing — kept neutral here.")
    parser.add_argument("--assistant_prefill", choices=["none", "per_variant"],
                        default="none",
                        help="If 'per_variant', append a scenario-neutral "
                             "preamble (e.g. 'Alice would go to ') after the "
                             "<|im_start|>assistant header so the model's "
                             "first generated token IS the answer. Per-(context, "
                             "template) strings are in PREFILL_TABLE. Eval-time "
                             "only — note that CMA selected heads without prefill.")
    args = parser.parse_args()

    # BC: --eval_conditions overrides --eval_variants if explicitly passed
    if args.eval_conditions is not None:
        logger.warning("--eval_conditions is deprecated; use --eval_variants")
        args.eval_variants = args.eval_conditions

    # resolve paths relative to project root
    if not os.path.isabs(args.wandb_dir):
        args.wandb_dir = os.path.join(project_root, args.wandb_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    if args.head_csv and not os.path.isabs(args.head_csv):
        args.head_csv = os.path.join(project_root, args.head_csv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- compute head lists ---
    logger.info(f"=== loading target heads (head_source={args.head_source}) ===")
    target_heads, scores_df = load_head_set_from_csv(args)
    control_heads = build_control_heads(target_heads, args.wandb_dir)

    logger.info(f"target heads ({len(target_heads)}): {target_heads}")
    logger.info(f"control heads ({len(control_heads)}): {control_heads}")

    # save head list
    head_list_path = os.path.join(args.output_dir, "head_list.csv")
    head_df = pd.DataFrame(target_heads, columns=["layer", "head"])
    head_df["rank"] = range(1, len(target_heads) + 1)
    head_df = head_df.merge(scores_df[["layer", "head", "mean_score", "abs_mean_score"]],
                             on=["layer", "head"], how="left")
    head_df.to_csv(head_list_path, index=False)
    logger.info(f"head list saved to {head_list_path}")

    control_list_path = os.path.join(args.output_dir, "control_head_list.csv")
    control_df = pd.DataFrame(control_heads, columns=["layer", "head"])
    control_df["rank"] = range(1, len(control_heads) + 1)
    control_df.to_csv(control_list_path, index=False)

    # --- load model ---
    logger.info("=== loading model ===")
    model_config = ModelConfig(
        model_type=args.model_type,
        device_map=args.device_map,
        device="cuda",
        n_devices=args.n_devices,
    )
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=0.0,
    )
    prompt_config = PromptConfig(
        use_template_system=True,
        base_rule="ABA",
        context_type="abstract",
        prompt_num=args.prompt_num,
    )

    loader = ModelLoader(model_config, gen_config, prompt_config)
    model, tokenizer, generation_kwargs, eos_token_ids, _, _, model_id = (
        loader.load_model_and_tokenizer()
    )

    # --- generate prompts ---
    logger.info("=== generating prompts ===")
    template_system = UnifiedTemplateSystem(vocab_dir="tom_datasets")
    question_formatter = UnifiedQuestionFormatter(tokenizer=tokenizer)

    prompt_dict = generate_unpaired_prompts(
        template_system, question_formatter, tokenizer,
        args.templates, args.prompt_num, args.system_prompt,
        eval_variants=args.eval_variants,
        prefill_mode=args.assistant_prefill,
    )
    # variant names may have been resolved via aliases inside the function
    resolved_variants = list(prompt_dict.keys())

    for v in resolved_variants:
        logger.info(f"  {v}: {len(prompt_dict[v]['prompts'])} prompts")

    # Sanity check: log the first prompt of the first variant plus the
    # first-answer token the eval will score against. Catches prefill /
    # leading-space tokenization mistakes before any GPU work happens.
    _v0 = resolved_variants[0]
    _p0 = prompt_dict[_v0]["prompts"][0]
    _c0 = prompt_dict[_v0]["correct"][0]
    _leading = _p0.endswith(" ")
    _tok0 = get_first_answer_token_id(tokenizer, _c0, prepend_space=_leading)
    logger.info(f"prefill_mode={args.assistant_prefill}  leading_space={_leading}")
    logger.info(f"first prompt tail: ...{_p0[-80:]!r}")
    logger.info(f"correct={_c0!r}  first_token_id={_tok0}  "
                f"decoded={tokenizer.decode([_tok0])!r}")

    # save prompts for reproducibility (correct + alt answers)
    for v in resolved_variants:
        d = prompt_dict[v]
        prompt_path = os.path.join(args.output_dir, f"{v}_prompts.txt")
        with open(prompt_path, "w") as f:
            for p, c, a in zip(d["prompts"], d["correct"], d["alt"]):
                f.write(f"CORRECT: {c}\nALT: {a}\n{p}\n{'='*60}\n")

    # --- run cumulative ablation ---
    logger.info("=== running target ablation ===")
    target_results, target_per_prompt = run_cumulative_ablation(
        model, tokenizer, prompt_dict, target_heads,
        eos_token_ids, args.max_new_tokens, label="target"
    )

    logger.info("=== running control ablation ===")
    control_results, control_per_prompt = run_cumulative_ablation(
        model, tokenizer, prompt_dict, control_heads,
        eos_token_ids, args.max_new_tokens, label="control"
    )

    # --- save results ---
    all_results = target_results + control_results
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, "cumulative_ablation_results.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"results saved to {csv_path}")

    # save per-prompt data as .pt for downstream analysis
    per_prompt_path = os.path.join(args.output_dir, "per_prompt_data.pt")
    torch.save({
        "target": {cond: {k: torch.tensor(v) for k, v in data.items()}
                   for cond, data in target_per_prompt.items()},
        "control": {cond: {k: torch.tensor(v) for k, v in data.items()}
                    for cond, data in control_per_prompt.items()},
        "target_heads": target_heads,
        "control_heads": control_heads,
    }, per_prompt_path)
    logger.info(f"per-prompt data saved to {per_prompt_path}")

    # --- plot ---
    plot_path = os.path.join(args.output_dir, "cumulative_ablation_plot.png")
    plot_results(results_df, plot_path, run_label=args.run_label)

    logger.info("=== done ===")


if __name__ == "__main__":
    main()
