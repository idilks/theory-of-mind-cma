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

def generate_unpaired_prompts(template_system, question_formatter, tokenizer,
                               template_names, num_prompts, system_prompt):
    """
    Generate 3 independent prompt sets: false-belief, true-belief, basic.
    Returns dict of {condition: (prompts_list, answers_list)}.
    """
    prompts_per_template = max(1, num_prompts // len(template_names))

    results = {
        "false_belief": ([], []),
        "true_belief": ([], []),
        "basic": ([], []),
    }

    for template_name in template_names:
        template = template_system.get_template(template_name, "abstract")
        basic_template = template_system.get_template(template_name, "basic")

        for _ in range(prompts_per_template):
            # sample variables
            subs = {}
            for var_name, vocab_type in template.variables.items():
                subs[var_name] = template_system.vocab_loader.sample_different(vocab_type, 1)[0]
            # ensure contrastive pair is different
            var_1 = [v for v in template.variables.keys() if v.endswith("_1")][0]
            var_2 = [v for v in template.variables.keys() if v.endswith("_2")][0]
            vocab_type = template.variables[var_1]
            values = template_system.vocab_loader.sample_different(vocab_type, 2)
            subs[var_1] = values[0]
            subs[var_2] = values[1]

            # false-belief prompt
            false_scenario = template.false_context_template.format(**subs)
            false_question = question_formatter.get_cma_questions(
                template_name, context_type="abstract", belief_type="ABA", question_style="instruction"
            )
            false_prompt = _format_chat(tokenizer, system_prompt, f"{false_scenario}\n{false_question}")
            false_answer = subs[template.belief_answer_key]
            results["false_belief"][0].append(false_prompt)
            results["false_belief"][1].append(false_answer)

            # true-belief prompt
            true_scenario = template.true_context_template.format(**subs)
            true_question = false_question  # same question
            true_prompt = _format_chat(tokenizer, system_prompt, f"{true_scenario}\n{true_question}")
            true_answer = subs[template.world_answer_key]
            results["true_belief"][0].append(true_prompt)
            results["true_belief"][1].append(true_answer)

            # basic prompt (no agent)
            basic_subs = {k: v for k, v in subs.items() if k in basic_template.variables}
            basic_scenario = basic_template.template.format(**basic_subs)
            basic_question = question_formatter.get_cma_questions(
                template_name, context_type="basic", belief_type="ABA", question_style="instruction"
            )
            basic_prompt = _format_chat(tokenizer, system_prompt, f"{basic_scenario}\n{basic_question}")
            basic_answer = subs[basic_template.answer_key]
            results["basic"][0].append(basic_prompt)
            results["basic"][1].append(basic_answer)

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

def get_first_answer_token_id(tokenizer, answer):
    """Get the first token ID of the expected answer.

    For multi-token answers like "Oak Street", returns the ID of "Oak".
    The absolute probability may be noisy for multi-token answers,
    but relative degradation across ablation steps is the signal.
    """
    tokens = tokenizer.encode(answer, add_special_tokens=False)
    return tokens[0]


def evaluate_prompt(model, tokenizer, prompt, answer, answer_token_id,
                     eos_token_ids, fwd_hooks=None, max_new_tokens=5):
    """Evaluate a single prompt. Returns (correct: bool, answer_prob: float)."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.cfg.device)
    hooks = fwd_hooks or []

    # probability via single forward pass
    logits = model.run_with_hooks(input_ids, fwd_hooks=hooks)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    answer_prob = probs[answer_token_id].item()

    # accuracy via generation
    generated = model.generate_with_hooks(
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_ids,
        fwd_hooks=hooks,
    )
    gen_text = tokenizer.decode(
        generated[0, input_ids.shape[1]:], skip_special_tokens=True
    )
    correct = locations_match(gen_text.strip(), answer)

    return correct, answer_prob


# ---------------------------------------------------------------------------
# cumulative ablation (per-prompt, following yukang_dev/ablation.py structure)
# ---------------------------------------------------------------------------

def run_cumulative_ablation(model, tokenizer, prompt_dict, ranked_heads,
                             eos_token_ids, max_new_tokens, label="target"):
    """
    Ablate heads cumulatively (top-1, top-2, ..., top-N) and measure both
    accuracy and P(first answer token) per prompt.

    Returns (summary_results: list[dict], per_prompt_data: dict).
      per_prompt_data[condition] = {"prob": np.array(n_prompts, n_steps),
                                     "acc":  np.array(n_prompts, n_steps)}
    """
    conditions = ["false_belief", "true_belief", "basic"]
    n_steps = len(ranked_heads) + 1  # k=0..N

    # precompute answer token IDs per condition
    answer_token_ids = {}
    for cond in conditions:
        _, answers = prompt_dict[cond]
        answer_token_ids[cond] = [
            get_first_answer_token_id(tokenizer, a) for a in answers
        ]

    # allocate per-prompt matrices
    per_prompt_data = {}
    for cond in conditions:
        n_prompts = len(prompt_dict[cond][0])
        per_prompt_data[cond] = {
            "prob": np.zeros((n_prompts, n_steps)),
            "acc": np.zeros((n_prompts, n_steps)),
        }

    # main loop: k on outside, prompts on inside (hooks built once per k)
    for k in range(n_steps):
        hooks = None if k == 0 else build_ablation_hooks(ranked_heads[:k])
        if k > 0:
            logger.info(f"[{label}] k={k}, ablating {ranked_heads[k-1]}")
        else:
            logger.info(f"[{label}] k=0 (baseline)")

        for cond in conditions:
            prompts, answers = prompt_dict[cond]
            ans_toks = answer_token_ids[cond]

            for p_idx, (prompt, answer, ans_tok) in enumerate(
                zip(prompts, answers, ans_toks)
            ):
                correct, prob = evaluate_prompt(
                    model, tokenizer, prompt, answer, ans_tok,
                    eos_token_ids, fwd_hooks=hooks,
                    max_new_tokens=max_new_tokens,
                )
                per_prompt_data[cond]["prob"][p_idx, k] = prob
                per_prompt_data[cond]["acc"][p_idx, k] = float(correct)

            mean_acc = per_prompt_data[cond]["acc"][:, k].mean()
            mean_prob = per_prompt_data[cond]["prob"][:, k].mean()
            logger.info(f"  {cond}: acc={mean_acc:.3f}  prob={mean_prob:.4f}")

    # build summary DataFrame
    summary = []
    for k in range(n_steps):
        row = {"k": k, "curve": label}
        for cond in conditions:
            row[f"{cond}_acc"] = per_prompt_data[cond]["acc"][:, k].mean()
            row[f"{cond}_prob"] = per_prompt_data[cond]["prob"][:, k].mean()
            row[f"{cond}_prob_std"] = per_prompt_data[cond]["prob"][:, k].std()
        summary.append(row)

    return summary, per_prompt_data


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_results(results_df, output_path):
    """Plot 2 panels: accuracy (top) and P(answer token) (bottom), each with 6 curves."""
    fig, (ax_acc, ax_prob) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    conditions = ["false_belief", "true_belief", "basic"]
    colors = {"false_belief": "#d62728", "true_belief": "#2ca02c", "basic": "#1f77b4"}
    cond_labels = {"false_belief": "false belief", "true_belief": "true belief", "basic": "basic (no agent)"}

    for curve_type in ["target", "control"]:
        subset = results_df[results_df["curve"] == curve_type]
        linestyle = "-" if curve_type == "target" else "--"
        marker = "o" if curve_type == "target" else "s"

        for cond in conditions:
            label = f"{cond_labels[cond]} ({curve_type})"

            ax_acc.plot(
                subset["k"], subset[f"{cond}_acc"],
                color=colors[cond], linestyle=linestyle,
                label=label, linewidth=2, marker=marker,
                markersize=3, alpha=0.85
            )

            ax_prob.plot(
                subset["k"], subset[f"{cond}_prob"],
                color=colors[cond], linestyle=linestyle,
                label=label, linewidth=2, marker=marker,
                markersize=3, alpha=0.85
            )
            # shade ±1 std for probability
            if f"{cond}_prob_std" in subset.columns:
                mean = subset[f"{cond}_prob"].values
                std = subset[f"{cond}_prob_std"].values
                ax_prob.fill_between(
                    subset["k"], mean - std, mean + std,
                    color=colors[cond], alpha=0.1
                )

    ax_acc.set_ylabel("accuracy", fontsize=12)
    ax_acc.set_title("cumulative ablation of belief-tracking heads", fontsize=14)
    ax_acc.legend(fontsize=8, loc="lower left")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.grid(True, alpha=0.3)

    ax_prob.set_xlabel("number of heads ablated (cumulative)", fontsize=12)
    ax_prob.set_ylabel("P(first answer token)", fontsize=12)
    ax_prob.legend(fontsize=8, loc="lower left")
    ax_prob.grid(True, alpha=0.3)

    plt.tight_layout()
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
    args = parser.parse_args()

    # resolve wandb_dir relative to project root
    if not os.path.isabs(args.wandb_dir):
        args.wandb_dir = os.path.join(project_root, args.wandb_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- compute head lists ---
    logger.info("=== computing head lists from wandb CSVs ===")
    target_heads, scores_df = load_abstract_only_heads(args.wandb_dir)
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
    system_prompt = "You are a helpful assistant reasoning about agents' beliefs."

    prompt_dict = generate_unpaired_prompts(
        template_system, question_formatter, tokenizer,
        args.templates, args.prompt_num, system_prompt
    )

    for cond in ["false_belief", "true_belief", "basic"]:
        logger.info(f"  {cond}: {len(prompt_dict[cond][0])} prompts")

    # save prompts for reproducibility
    for cond in ["false_belief", "true_belief", "basic"]:
        prompts, answers = prompt_dict[cond]
        prompt_path = os.path.join(args.output_dir, f"{cond}_prompts.txt")
        with open(prompt_path, "w") as f:
            for p, a in zip(prompts, answers):
                f.write(f"EXPECTED: {a}\n{p}\n{'='*60}\n")

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
    plot_results(results_df, plot_path)

    logger.info("=== done ===")


if __name__ == "__main__":
    main()
