"""
combine per-sample CMA scores from seed_0 and seed_1 runs into a merged result.

usage:
    python scripts/combine_per_sample_runs.py
    python scripts/combine_per_sample_runs.py --dry_run
    python scripts/combine_per_sample_runs.py --patch_position patch_before_movement
    python scripts/combine_per_sample_runs.py --seeds 0 1 --output_seed_tag combined

what it does:
    for each (condition, template, base_rule->exp_rule, patch_position) under
    results/causal_analysis/cma/.../, find every z_seed_{S}_shuffle_False/logit/
    sample_num_*_gen_acc_*/ folder for the requested seeds. concatenate the
    per-sample tensors along dim 0, recompute the mean, save under
    z_seed_{output_seed_tag}_shuffle_False/logit/sample_num_{N_total}_gen_acc_{thr}/
    and re-run the permutation test on the combined tensor.

    seed-0 and seed-1 runs use disjoint RNG streams (set_seed() seeds python
    random + numpy), so concatenated samples are independent.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codebase', 'tasks', 'causal_analysis'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codebase'))

import torch
from statistical_testing import permutation_test_heads, save_statistical_results
from utils import plot


SAMPLE_DIR_RE = re.compile(r"sample_num_(\d+)_(.+)")
SEED_DIR_RE = re.compile(r"z_seed_(\w+)_shuffle_(True|False)")

# legacy folder names -> canonical condition names. results land under canonical.
CONDITION_ALIASES = {
    "abstract_context": "abstract_belief",
    "photo_context": "abstract_photo",
    "control_context": "control",
}


def find_runs(results_dir: Path, seeds, patch_position: str):
    """group per-sample tensors by (condition, template, rule_pair, patch_position, eval_suffix, shuffle)."""
    groups = defaultdict(dict)  # key -> {seed: (path, n_samples, eval_suffix)}

    for pt in results_dir.rglob("causal_scores_per_sample.pt"):
        # expect: .../<condition>/template_<name>/base_rule_<X>_exp_rule_<Y>/<patch_pos>/z_seed_<S>_shuffle_<B>/logit/sample_num_<N>_<suffix>/causal_scores_per_sample.pt
        try:
            sample_dir = pt.parent.name
            logit_dir = pt.parent.parent.name
            seed_dir = pt.parent.parent.parent.name
            patch_dir = pt.parent.parent.parent.parent.name
            rule_dir = pt.parent.parent.parent.parent.parent.name
            template_dir = pt.parent.parent.parent.parent.parent.parent.name
            condition_dir = pt.parent.parent.parent.parent.parent.parent.parent.name
        except IndexError:
            continue

        if logit_dir != "logit":
            continue
        if patch_position and patch_dir != patch_position:
            continue

        seed_match = SEED_DIR_RE.match(seed_dir)
        if not seed_match:
            continue
        seed_str, shuffle_str = seed_match.group(1), seed_match.group(2)
        try:
            seed_int = int(seed_str)
        except ValueError:
            continue  # skip already-combined folders
        if seed_int not in seeds:
            continue

        sample_match = SAMPLE_DIR_RE.match(sample_dir)
        if not sample_match:
            continue
        n_samples = int(sample_match.group(1))
        eval_suffix = sample_match.group(2)  # e.g. "gen_acc_0.9"

        canonical_condition = CONDITION_ALIASES.get(condition_dir, condition_dir)
        key = (canonical_condition, template_dir, rule_dir, patch_dir, eval_suffix, shuffle_str)
        if seed_int in groups[key]:
            existing_path = groups[key][seed_int][0]
            print(f"warn: duplicate seed={seed_int} for {key}: keeping {existing_path}, ignoring {pt}")
            continue
        groups[key][seed_int] = (pt, n_samples)

    return groups


def combine_group(key, seed_to_run, results_dir: Path, output_seed_tag: str,
                  n_permutations: int, alpha: float, dry_run: bool, skip_existing: bool):
    condition, template, rule_pair, patch_pos, eval_suffix, shuffle_str = key

    tensors = []
    n_total = 0
    seed_summary = []
    for seed in sorted(seed_to_run.keys()):
        pt_path, n_s = seed_to_run[seed]
        scores = torch.load(pt_path, map_location='cpu')
        tensors.append(scores)
        n_total += scores.shape[0]
        seed_summary.append(f"seed={seed} n={scores.shape[0]} ({pt_path.name})")

    rel = f"{condition}/{template}/{rule_pair}/{patch_pos}"
    print(f"\n[{rel}]")
    for s in seed_summary:
        print(f"    {s}")

    if len(tensors) < 2:
        print(f"    skip: only {len(tensors)} seed(s) found, need >=2 to combine")
        return

    combined = torch.cat(tensors, dim=0)
    print(f"    combined shape: {tuple(combined.shape)}")

    out_seed_dir = (results_dir / condition / template / rule_pair / patch_pos
                    / f"z_seed_{output_seed_tag}_shuffle_{shuffle_str}"
                    / "logit"
                    / f"sample_num_{n_total}_{eval_suffix}")

    if skip_existing and (out_seed_dir / "causal_scores_per_sample.pt").exists():
        print(f"    skip: already exists at {out_seed_dir}")
        return

    if dry_run:
        print(f"    [dry] would write to {out_seed_dir}")
        return

    out_seed_dir.mkdir(parents=True, exist_ok=True)

    torch.save(combined, out_seed_dir / "causal_scores_per_sample.pt")

    mean_scores = combined.mean(dim=0)
    if mean_scores.ndim == 1:
        mean_scores = mean_scores[:, None]
    torch.save(mean_scores, out_seed_dir / "causal_scores.pt")

    # heatmap (matches cma.py:534 call signature)
    base_rule = rule_pair.split("_exp_rule_")[0].replace("base_rule_", "")
    exp_rule = rule_pair.split("_exp_rule_")[1]
    metric_name = f"CMA: {base_rule}->{exp_rule} ({condition}) | combined seeds {sorted(seed_to_run.keys())} | n={n_total}"
    try:
        plot(mean_scores.cpu().numpy(), str(out_seed_dir), metric_name=metric_name)
    except Exception as e:
        print(f"    warn: heatmap failed ({e})")

    # permutation test on combined per-sample tensor
    print(f"    running permutation test (n={n_permutations}) on {n_total} samples...")
    df, threshold = permutation_test_heads(combined, n_permutations=n_permutations, alpha=alpha)
    n_sig = int(df['significant'].sum())
    print(f"    FWER threshold={threshold:.6f}  significant={n_sig}/{len(df)} ({n_sig/len(df)*100:.1f}%)")
    save_statistical_results(df, str(out_seed_dir), threshold=threshold)

    print(f"    wrote {out_seed_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str,
                        default="results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                        help="seeds to combine (must be int folder names z_seed_{S}_shuffle_*)")
    parser.add_argument("--output_seed_tag", type=str, default="combined",
                        help="tag for the output folder z_seed_<tag>_shuffle_*/. default: combined")
    parser.add_argument("--patch_position", type=str, default="patch_before_movement",
                        help="restrict to one patch position folder. set to '' to process both")
    parser.add_argument("--n_permutations", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_existing", action="store_true",
                        help="skip groups whose combined output already exists")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"results_dir does not exist: {results_dir}")
        sys.exit(1)

    groups = find_runs(results_dir, set(args.seeds), args.patch_position)
    print(f"found {len(groups)} groups (patch_position={args.patch_position or 'any'}, seeds={args.seeds})")

    n_combinable = sum(1 for v in groups.values() if len(v) >= 2)
    print(f"{n_combinable} groups have >=2 of the requested seeds and can be combined")

    for key, seed_to_run in sorted(groups.items()):
        combine_group(key, seed_to_run, results_dir,
                      output_seed_tag=args.output_seed_tag,
                      n_permutations=args.n_permutations,
                      alpha=args.alpha,
                      dry_run=args.dry_run,
                      skip_existing=args.skip_existing)

    print("\ndone.")


if __name__ == "__main__":
    main()
