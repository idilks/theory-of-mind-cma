"""
Random layer-matched baseline for ablation experiments.

For each k in --k_eval, draws M random head sets of size k whose per-layer
counts match the first-k heads of --target_csv, ablates each, and writes
a long-form CSV of metrics. Also writes the target's own metrics at the
same k values for comparison.

Pool for random draws (--sampling layer_matched, default): at each layer that
has target heads, all 40 heads EXCEPT the heads in the target set itself.
(Not excluding other condition sets is intentional — see agenda discussion.)
--sampling uniform instead draws k heads uniformly from all 48x40=1920 heads
excluding the target set (used by the code-arm nulls).

Usage (cheap C1\\C5 pilot):
    python scripts/random_baseline_ablation.py \\
        --target_csv other_results/ablation/belief_minus_loc/head_list.csv \\
        --n_random 100 \\
        --k_eval 11 21 \\
        --eval_variants false_belief \\
        --prompt_num 50 \\
        --output_dir results/ablation/random_baselines/belief_minus_loc
"""

import argparse
import logging
import os
import random
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

# project paths — match ablation_experiment.py's path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
codebase_root = os.path.join(project_root, "codebase")
if codebase_root not in sys.path:
    sys.path.insert(0, codebase_root)

# Importing ablation_experiment triggers os.chdir(codebase_root); fine.
from tasks.causal_analysis.ablation_experiment import (  # noqa: E402
    build_ablation_hooks,
    compute_answer_token_id,
    evaluate_prompt,
    generate_eval_prompts,
    get_match_fn,
)
from tasks.causal_analysis.cma_config import (  # noqa: E402
    GenerationConfig,
    ModelConfig,
    PromptConfig,
)
from tasks.causal_analysis.models import ModelLoader  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# layer-matched random sampling
# ---------------------------------------------------------------------------

def read_target_heads(csv_path):
    """Read a head_list.csv-style file and return [(layer, head), ...] in
    the CSV's existing row order (interpreted as ablation rank order)."""
    df = pd.read_csv(csv_path)
    if "layer" not in df.columns or "head" not in df.columns:
        raise ValueError(f"{csv_path} must have 'layer' and 'head' columns")
    return list(zip(df["layer"].astype(int), df["head"].astype(int)))


def per_layer_counts(heads):
    """Counter of how many heads in `heads` are at each layer."""
    return Counter(layer for layer, _ in heads)


def stratified_random_set(counts, target_set, n_heads_per_layer, rng):
    """Sample a head set with the given per-layer counts, excluding heads in
    target_set. Returns a list of (layer, head) tuples."""
    sampled = []
    for layer, n_to_draw in counts.items():
        all_heads_at_layer = set(range(n_heads_per_layer))
        target_at_layer = {h for (l, h) in target_set if l == layer}
        pool = sorted(all_heads_at_layer - target_at_layer)
        if len(pool) < n_to_draw:
            raise ValueError(
                f"layer {layer}: pool size {len(pool)} < requested {n_to_draw}. "
                f"Increase --n_heads_per_layer or shrink target."
            )
        picked = rng.sample(pool, n_to_draw)
        sampled.extend((layer, h) for h in picked)
    return sampled


def uniform_random_set(k, target_set, n_layers, n_heads_per_layer, rng):
    """Sample k heads uniformly from ALL layers x heads (e.g. 48x40 = 1920),
    excluding heads in target_set. Returns a list of (layer, head) tuples."""
    pool = [(l, h) for l in range(n_layers) for h in range(n_heads_per_layer)
            if (l, h) not in target_set]
    if len(pool) < k:
        raise ValueError(f"uniform pool size {len(pool)} < requested k={k}")
    return rng.sample(pool, k)


# ---------------------------------------------------------------------------
# one-shot ablation evaluator (no cumulative sweep)
# ---------------------------------------------------------------------------

def _precompute_answer_tokens(prompt_dict, tokenizer):
    """For each variant, encode correct/alt answers as first-token IDs.
    Same logic as run_cumulative_ablation, factored out. Dispatches per
    variant (ToM leading-space vs code quote-prefill handling)."""
    correct_toks, alt_toks = {}, {}
    for v, d in prompt_dict.items():
        prompts_v = d["prompts"]
        correct_toks[v] = [
            compute_answer_token_id(tokenizer, v, a, p)
            for a, p in zip(d["correct"], prompts_v)
        ]
        alt_toks[v] = [
            compute_answer_token_id(tokenizer, v, a, p)
            for a, p in zip(d["alt"], prompts_v)
        ]
    return correct_toks, alt_toks


def ablate_once(model, tokenizer, prompt_dict, head_list,
                correct_toks, alt_toks, eos_token_ids, max_new_tokens):
    """Ablate `head_list` (or no ablation if empty) and return one row per
    variant with accuracy, flip_rate, mean_p_correct, mean_p_alt, flip_score."""
    hooks = build_ablation_hooks(head_list) if head_list else None
    EPS = 1e-12
    rows = []
    for v, d in prompt_dict.items():
        prompts = d["prompts"]
        corrects = d["correct"]
        alts = d["alt"]
        ctoks = correct_toks[v]
        atoks = alt_toks[v]
        match_fn = get_match_fn(v, prompts[0]) if prompts else None

        n = len(prompts)
        pc_arr = np.zeros(n)
        pa_arr = np.zeros(n)
        acc_arr = np.zeros(n)
        flip_arr = np.zeros(n)

        for i in range(n):
            cm, am, pc, pa = evaluate_prompt(
                model, tokenizer, prompts[i],
                corrects[i], alts[i], ctoks[i], atoks[i],
                eos_token_ids, fwd_hooks=hooks,
                max_new_tokens=max_new_tokens,
                match_fn=match_fn,
            )
            pc_arr[i] = pc
            pa_arr[i] = pa
            acc_arr[i] = float(cm)
            flip_arr[i] = float(am)

        flip_score = pa_arr / (pa_arr + pc_arr + EPS)
        rows.append({
            "eval_variant": v,
            "accuracy": float(acc_arr.mean()),
            "flip_rate": float(flip_arr.mean()),
            "mean_p_correct": float(pc_arr.mean()),
            "mean_p_alt": float(pa_arr.mean()),
            "p_correct_std": float(pc_arr.std()),
            "p_alt_std": float(pa_arr.std()),
            "flip_score": float(flip_score.mean()),
        })
    return rows


# ---------------------------------------------------------------------------
# null summarization
# ---------------------------------------------------------------------------

def summarize_null(random_df, target_df):
    """For each (k, eval_variant), produce a null summary row with target value,
    null median, percentiles, and one-sided p-value (= rank of target in null)."""
    out = []
    for (k, v), null_sub in random_df.groupby(["k", "eval_variant"]):
        target_row = target_df[(target_df["k"] == k) & (target_df["eval_variant"] == v)]
        if target_row.empty:
            continue
        for metric in ["flip_rate", "flip_score", "mean_p_correct", "mean_p_alt"]:
            null_vals = null_sub[metric].values
            target_val = float(target_row[metric].values[0])
            # one-sided p-value: P(null >= target) for flip metrics; user can
            # flip sign for p_correct since for that one "below target" is the
            # interesting tail. We report both tails so user picks.
            p_right = float((null_vals >= target_val).mean())
            p_left = float((null_vals <= target_val).mean())
            out.append({
                "k": k,
                "eval_variant": v,
                "metric": metric,
                "target": target_val,
                "null_n": len(null_vals),
                "null_mean": float(null_vals.mean()),
                "null_std": float(null_vals.std()),
                "null_p01": float(np.percentile(null_vals, 1)),
                "null_p05": float(np.percentile(null_vals, 5)),
                "null_p50": float(np.percentile(null_vals, 50)),
                "null_p95": float(np.percentile(null_vals, 95)),
                "null_p99": float(np.percentile(null_vals, 99)),
                "null_max": float(null_vals.max()),
                "null_min": float(null_vals.min()),
                "p_target_geq_null": p_right,
                "p_target_leq_null": p_left,
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Random layer-matched ablation baseline")

    # what to compare against
    p.add_argument("--target_csv", required=True,
                   help="head_list.csv-style file with 'layer' and 'head' columns")
    p.add_argument("--n_random", type=int, default=100,
                   help="M = number of random head sets to draw per k value")
    p.add_argument("--k_eval", type=int, nargs="+", default=None,
                   help="k values to evaluate at. Default: just len(target)")
    p.add_argument("--n_heads_per_layer", type=int, default=40,
                   help="Qwen2.5-14B = 40")
    p.add_argument("--n_layers", type=int, default=48,
                   help="Qwen2.5-14B = 48 (used by --sampling uniform)")
    p.add_argument("--sampling", choices=["layer_matched", "uniform"],
                   default="layer_matched",
                   help="layer_matched: per-layer counts match the first-k "
                        "target heads (original convention). uniform: k heads "
                        "drawn uniformly from all n_layers x n_heads_per_layer "
                        "heads, excluding the target set (code-arm nulls).")

    # mirror ablation_experiment.py for reproducibility
    p.add_argument("--model_type", type=str, default="Qwen2.5-14B-Instruct")
    p.add_argument("--prompt_num", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=5)
    p.add_argument("--templates", nargs="+",
                   default=["food_truck", "hair_styling", "library_book"])
    p.add_argument("--eval_variants", nargs="+", default=["false_belief"],
                   help="Default: just false_belief. Add more for cross-frame.")
    p.add_argument("--system_prompt", type=str,
                   default="You are a helpful assistant. Answer concisely.")
    p.add_argument("--assistant_prefill", choices=["none", "per_variant"],
                   default="none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_devices", type=int, default=1)
    p.add_argument("--device_map", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, required=True)

    args = p.parse_args()

    # resolve paths
    if not os.path.isabs(args.target_csv):
        args.target_csv = os.path.join(project_root, args.target_csv)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # --- load target heads ---
    target_heads = read_target_heads(args.target_csv)
    n_target = len(target_heads)
    logger.info(f"loaded target ({n_target} heads) from {args.target_csv}")

    k_eval = args.k_eval if args.k_eval is not None else [n_target]
    for k in k_eval:
        if k > n_target:
            raise ValueError(f"--k_eval {k} > len(target) {n_target}")

    # per-layer counts for each k (based on the first-k heads of target)
    counts_by_k = {k: per_layer_counts(target_heads[:k]) for k in k_eval}
    for k, c in counts_by_k.items():
        logger.info(f"  k={k}: per-layer counts = {dict(sorted(c.items()))}")

    # --- load model ---
    logger.info("=== loading model ===")
    model_config = ModelConfig(
        model_type=args.model_type,
        device_map=args.device_map,
        device="cuda",
        n_devices=args.n_devices,
    )
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens, do_sample=False, temperature=0.0,
    )
    prompt_config = PromptConfig(
        use_template_system=True, base_rule="ABA",
        context_type="abstract", prompt_num=args.prompt_num,
    )
    loader = ModelLoader(model_config, gen_config, prompt_config)
    model, tokenizer, _, eos_token_ids, _, _, _ = loader.load_model_and_tokenizer()

    # --- generate prompts (same machinery as ablation_experiment) ---
    logger.info("=== generating prompts ===")
    prompt_dict = generate_eval_prompts(
        tokenizer, args.templates, args.prompt_num, args.system_prompt,
        eval_variants=args.eval_variants,
        prefill_mode=args.assistant_prefill,
    )
    for v in prompt_dict:
        logger.info(f"  {v}: {len(prompt_dict[v]['prompts'])} prompts")

    correct_toks, alt_toks = _precompute_answer_tokens(prompt_dict, tokenizer)

    # sanity print
    _v0 = list(prompt_dict.keys())[0]
    _p0 = prompt_dict[_v0]["prompts"][0]
    _c0 = prompt_dict[_v0]["correct"][0]
    _tok0 = correct_toks[_v0][0]
    logger.info(f"first prompt tail ({_v0}): ...{_p0[-80:]!r}")
    logger.info(f"correct={_c0!r}  first_token={tokenizer.decode([_tok0])!r}")

    # --- target eval (baseline + each k) ---
    logger.info("=== evaluating target ===")
    target_rows = []
    # k=0 baseline (no ablation)
    for row in ablate_once(model, tokenizer, prompt_dict, [],
                            correct_toks, alt_toks,
                            eos_token_ids, args.max_new_tokens):
        target_rows.append({"k": 0, "sample_idx": -1, **row})
    # each requested k
    for k in k_eval:
        logger.info(f"  target k={k}")
        for row in ablate_once(model, tokenizer, prompt_dict, target_heads[:k],
                                correct_toks, alt_toks,
                                eos_token_ids, args.max_new_tokens):
            target_rows.append({"k": k, "sample_idx": -1, **row})

    target_df = pd.DataFrame(target_rows)
    target_df.to_csv(os.path.join(args.output_dir, "target_results.csv"), index=False)
    logger.info("target results:")
    for _, r in target_df.iterrows():
        logger.info(f"  k={r['k']:3d} {r['eval_variant']:>20s} "
                    f"flip_rate={r['flip_rate']:.3f} "
                    f"flip_score={r['flip_score']:.3f} "
                    f"p_correct={r['mean_p_correct']:.3f} "
                    f"p_alt={r['mean_p_alt']:.3f}")

    # --- random null draws ---
    logger.info(f"=== drawing {args.n_random} random sets per k "
                f"({len(k_eval)} k values) ===")
    target_set = set(target_heads)
    random_rows = []
    for k in k_eval:
        counts = counts_by_k[k]
        for i in range(args.n_random):
            if args.sampling == "uniform":
                rand_set = uniform_random_set(
                    k, target_set, args.n_layers, args.n_heads_per_layer, rng
                )
            else:
                rand_set = stratified_random_set(
                    counts, target_set, args.n_heads_per_layer, rng
                )
            if (i % max(1, args.n_random // 10)) == 0:
                logger.info(f"  k={k} sample {i+1}/{args.n_random}")
            for row in ablate_once(model, tokenizer, prompt_dict, rand_set,
                                    correct_toks, alt_toks,
                                    eos_token_ids, args.max_new_tokens):
                random_rows.append({"k": k, "sample_idx": i, **row})

    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(
        os.path.join(args.output_dir, "random_baseline_results.csv"), index=False
    )

    # --- summary with p-values ---
    summary = summarize_null(random_df, target_df)
    summary.to_csv(os.path.join(args.output_dir, "null_summary.csv"), index=False)
    logger.info("null summary (flip_rate only):")
    for _, r in summary[summary["metric"] == "flip_rate"].iterrows():
        logger.info(
            f"  k={int(r['k']):3d} {r['eval_variant']:>20s} "
            f"target={r['target']:.3f} | null median={r['null_p50']:.3f} "
            f"[{r['null_p01']:.3f}, {r['null_p99']:.3f}] "
            f"max={r['null_max']:.3f} | p(target>=null)={r['p_target_geq_null']:.4f}"
        )

    logger.info("=== done ===")


if __name__ == "__main__":
    main()
