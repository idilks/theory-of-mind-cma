"""
One-off combiner for C2 (abstract_photo) and C3 (answer_changes_belief).

The seed=0 and seed=1 runs used different rule_pair label conventions for these
two conditions but ran semantically equivalent experiments. The folder name
encodes within-prompt label pattern, not the across-prompt belief/photo flip.
Verified by reading the prompts: seed=0 ABB->ABA and seed=1 ABA->ABA are both
false-belief base -> true-belief exp (C3 primary direction); the other pairs
mirror this.

This script concatenates the per-sample tensors across the two seeds inside
each direction and writes a combined output under the canonical flip-rule
label (ABA->ABB for primary, ABB->ABA for reverse), then re-runs the
permutation test on the merged tensor and copies the seed=1 prompt files into
the combined directory so the qmd renderer picks up the correct base/exp text.

Aliases:
  C2 abstract_photo (legacy folder: photo_context for seed=0):
    primary  = photo_context/ABA->ABA  +  abstract_photo/ABA->ABB
    reverse  = photo_context/ABB->ABB  +  abstract_photo/ABB->ABA
  C3 answer_changes_belief (canonical folder for both seeds):
    primary  = answer_changes_belief/ABB->ABA (seed=0) + answer_changes_belief/ABA->ABA (seed=1)
    reverse  = answer_changes_belief/ABA->ABB (seed=0) + answer_changes_belief/ABB->ABB (seed=1)

Output canonical labels: ABA->ABB (primary), ABB->ABA (reverse).
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codebase" / "tasks" / "causal_analysis"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codebase"))

import torch
from statistical_testing import permutation_test_heads, save_statistical_results
from utils import plot

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
TEMPLATES = ["template_food_truck", "template_hair_styling", "template_library_book"]
PATCH = "patch_before_movement"

# (canonical_condition_dir, seed0_alias_dir, seed0_label, seed1_label, canonical_label)
ALIASES = [
    # C2 entries removed: seed=0 photo_context used same-rule pairs (ABA->ABA, ABB->ABB)
    # which is base==exp prompts, not a photo-timing flip. seed=1 abstract_photo is the
    # canonical run; do not merge legacy photo_context into it.
    # C3 primary (false->true)
    ("answer_changes_belief", "answer_changes_belief", "base_rule_ABB_exp_rule_ABA", "base_rule_ABA_exp_rule_ABA", "base_rule_ABA_exp_rule_ABB"),
    # C3 reverse (true->false)
    ("answer_changes_belief", "answer_changes_belief", "base_rule_ABA_exp_rule_ABB", "base_rule_ABB_exp_rule_ABB", "base_rule_ABB_exp_rule_ABA"),
]

SAMPLE_RE = re.compile(r"sample_num_(\d+)_(.+)")
N_PERMUTATIONS = 5000
ALPHA = 0.05


def find_seed_dir(cond_dir: str, template: str, rule_label: str, seed: str) -> Path | None:
    base = RESULTS / cond_dir / template / rule_label / PATCH / f"z_seed_{seed}_shuffle_False"
    return base if base.exists() else None


def find_sample_dir(seed_dir: Path) -> tuple[Path, int, str] | None:
    """return (sample_dir, n_samples, eval_suffix)"""
    logit = seed_dir / "logit"
    if not logit.exists():
        return None
    candidates = []
    for d in logit.iterdir():
        m = SAMPLE_RE.match(d.name)
        if m and (d / "causal_scores_per_sample.pt").exists():
            candidates.append((d, int(m.group(1)), m.group(2)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


def copy_prompts(seed1_dir: Path, out_seed_dir: Path, sample_dir_name: str):
    """copy base/exp/ans prompt files from seed=1 into the combined seed dir."""
    # prompt files live at seed_dir level (not inside sample_num_*)
    for pattern in ["base_input_prompts_*.txt", "exp_input_prompts_*.txt"]:
        for p in seed1_dir.glob(pattern):
            shutil.copy2(p, out_seed_dir / p.name)
    # ans_*.txt lives inside sample_num_*/
    seed1_logit = seed1_dir / "logit"
    if seed1_logit.exists():
        for sd in seed1_logit.iterdir():
            for ans in sd.glob("ans_*.txt"):
                # mirror into the combined sample_num_* dir
                target = out_seed_dir / "logit" / sample_dir_name
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ans, target / ans.name)


def combine_one(cond_dir: str, seed0_alias: str, seed0_label: str, seed1_label: str, canonical_label: str):
    print(f"\n[{cond_dir} :: canonical={canonical_label}]")
    print(f"  seed=0 source: {seed0_alias}/{seed0_label}")
    print(f"  seed=1 source: {cond_dir}/{seed1_label}")

    for template in TEMPLATES:
        s0_dir = find_seed_dir(seed0_alias, template, seed0_label, "0")
        s1_dir = find_seed_dir(cond_dir, template, seed1_label, "1")
        if s0_dir is None or s1_dir is None:
            print(f"    [{template}] missing seed (s0={s0_dir is not None}, s1={s1_dir is not None})")
            continue
        s0_sample = find_sample_dir(s0_dir)
        s1_sample = find_sample_dir(s1_dir)
        if s0_sample is None or s1_sample is None:
            print(f"    [{template}] missing sample tensor")
            continue

        s0_path, n0, suffix0 = s0_sample
        s1_path, n1, suffix1 = s1_sample
        if suffix0 != suffix1:
            print(f"    [{template}] eval suffix mismatch: {suffix0} vs {suffix1}, using seed=1's")
        eval_suffix = suffix1

        t0 = torch.load(s0_path / "causal_scores_per_sample.pt", map_location="cpu")
        t1 = torch.load(s1_path / "causal_scores_per_sample.pt", map_location="cpu")
        combined = torch.cat([t0, t1], dim=0)
        n_total = combined.shape[0]
        print(f"    [{template}] seed=0 n={t0.shape[0]} + seed=1 n={t1.shape[0]} -> combined n={n_total}")

        out_seed_dir = (RESULTS / cond_dir / template / canonical_label / PATCH
                        / "z_seed_combined_shuffle_False")
        sample_dir_name = f"sample_num_{n_total}_{eval_suffix}"
        out_sample = out_seed_dir / "logit" / sample_dir_name
        out_sample.mkdir(parents=True, exist_ok=True)

        torch.save(combined, out_sample / "causal_scores_per_sample.pt")
        mean_scores = combined.mean(dim=0)
        if mean_scores.ndim == 1:
            mean_scores = mean_scores[:, None]
        torch.save(mean_scores, out_sample / "causal_scores.pt")

        try:
            plot(mean_scores.cpu().numpy(), str(out_sample),
                 metric_name=f"CMA combined seeds 0+1 :: {cond_dir}/{template}/{canonical_label} (n={n_total})")
        except Exception as e:
            print(f"      warn: heatmap failed ({e})")

        df, threshold = permutation_test_heads(combined, n_permutations=N_PERMUTATIONS, alpha=ALPHA)
        n_sig = int(df['significant'].sum())
        save_statistical_results(df, str(out_sample), threshold=threshold)
        print(f"      perm: threshold={threshold:.6f}  significant={n_sig}/{len(df)}")

        # copy seed=1 prompts into combined seed_dir so qmd builder finds them
        for p in s1_dir.glob("base_input_prompts_*.txt"):
            shutil.copy2(p, out_seed_dir / p.name)
        for p in s1_dir.glob("exp_input_prompts_*.txt"):
            shutil.copy2(p, out_seed_dir / p.name)
        # ans_*.txt sits inside sample_num_*: copy seed=1's into the combined sample_num_* dir
        for ans in (s1_path).glob("ans_*.txt"):
            shutil.copy2(ans, out_sample / ans.name)

        print(f"      wrote {out_sample}")


def main():
    for entry in ALIASES:
        combine_one(*entry)
    print("\ndone.")


if __name__ == "__main__":
    main()
