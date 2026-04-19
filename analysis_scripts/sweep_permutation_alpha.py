"""
Sweep permutation-test alpha values over existing CMA per-sample score files.

This script loads each `causal_scores_per_sample.pt` file, builds the max-null
distribution once, then evaluates significance counts for multiple alpha values.

Usage:
    python scripts/sweep_permutation_alpha.py
    python scripts/sweep_permutation_alpha.py --alphas 0.01,0.05,0.1,0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct",
        help="Root directory that contains CMA condition folders.",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.01,0.05,0.1,0.2",
        help="Comma-separated alpha values.",
    )
    parser.add_argument(
        "--n_permutations",
        type=int,
        default=5000,
        help="Number of sign-flip permutations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Permutation batch size to control memory usage.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for permutation reproducibility.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/permutation_analysis",
        help="Output directory for sweep tables.",
    )
    return parser.parse_args()


def parse_alphas(raw: str) -> List[float]:
    alphas = [float(x.strip()) for x in raw.split(",") if x.strip()]
    for a in alphas:
        if not (0.0 < a < 1.0):
            raise ValueError(f"alpha must be in (0,1), got {a}")
    return sorted(set(alphas))


def parse_metadata(results_root: Path, pt_file: Path) -> Dict[str, str]:
    rel = pt_file.relative_to(results_root)
    parts = rel.parts

    condition = parts[0] if len(parts) > 0 else "unknown"
    template = "unknown"
    base_rule = "unknown"
    exp_rule = "unknown"
    patch_position = "unknown"
    sample_folder = "unknown"

    for p in parts:
        if p.startswith("template_"):
            template = p.replace("template_", "")
        elif p.startswith("base_rule_") and "_exp_rule_" in p:
            mid = p.replace("base_rule_", "")
            base_rule, exp_rule = mid.split("_exp_rule_", 1)
        elif p in {"patch_after_movement", "patch_before_movement"}:
            patch_position = p
        elif p.startswith("sample_num_"):
            sample_folder = p

    direction = f"{base_rule}->{exp_rule}" if base_rule != "unknown" and exp_rule != "unknown" else "unknown"

    return {
        "condition": condition,
        "template": template,
        "base_rule": base_rule,
        "exp_rule": exp_rule,
        "direction": direction,
        "patch_position": patch_position,
        "sample_folder": sample_folder,
        "relative_path": str(rel).replace("\\", "/"),
    }


def compute_max_null_scores(
    scores: torch.Tensor,
    n_permutations: int,
    batch_size: int,
    generator: torch.Generator,
) -> np.ndarray:
    n_samples = scores.shape[0]
    max_null = np.empty(n_permutations, dtype=np.float32)
    offset = 0

    while offset < n_permutations:
        cur = min(batch_size, n_permutations - offset)
        signs = torch.randint(
            0,
            2,
            (cur, n_samples, 1, 1),
            generator=generator,
            device=scores.device,
            dtype=torch.int8,
        ).float()
        signs = signs * 2 - 1
        perm_means = (scores.unsqueeze(0) * signs).mean(dim=1)
        batch_max = perm_means.abs().reshape(cur, -1).max(dim=1).values.cpu().numpy()
        max_null[offset : offset + cur] = batch_max
        offset += cur

    return max_null


def format_pct(x: float) -> str:
    return f"{x:.2f}%"


def main() -> None:
    args = parse_args()
    alphas = parse_alphas(args.alphas)

    results_root = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(results_root.rglob("causal_scores_per_sample.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No causal_scores_per_sample.pt files found under {results_root}")

    print(f"Found {len(pt_files)} per-sample score files")
    print(f"Alphas: {alphas}")
    print(f"Permutations per run: {args.n_permutations}")

    all_rows: List[Dict[str, object]] = []
    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.seed)

    for i, pt_file in enumerate(pt_files, start=1):
        scores = torch.load(pt_file, map_location="cpu")
        if scores.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got shape {tuple(scores.shape)} for {pt_file}")

        obs_abs = scores.mean(dim=0).abs().reshape(-1).cpu().numpy()
        max_null = compute_max_null_scores(
            scores=scores,
            n_permutations=args.n_permutations,
            batch_size=args.batch_size,
            generator=gen,
        )

        meta = parse_metadata(results_root, pt_file)
        total_heads = int(obs_abs.shape[0])
        n_samples = int(scores.shape[0])

        print(f"[{i}/{len(pt_files)}] {meta['relative_path']} (n={n_samples})")

        for alpha in alphas:
            threshold = float(np.percentile(max_null, 100 * (1 - alpha)))
            n_sig = int((obs_abs > threshold).sum())
            all_rows.append(
                {
                    **meta,
                    "alpha": alpha,
                    "n_samples": n_samples,
                    "n_heads": total_heads,
                    "threshold": threshold,
                    "n_significant": n_sig,
                    "pct_significant": (100.0 * n_sig / total_heads),
                }
            )

    df = pd.DataFrame(all_rows)
    run_csv = out_dir / "alpha_sweep_per_run.csv"
    df.to_csv(run_csv, index=False)

    cond = (
        df.groupby(["alpha", "condition"], as_index=False)
        .agg(
            runs=("n_significant", "count"),
            avg_sig_heads=("n_significant", "mean"),
            min_sig_heads=("n_significant", "min"),
            max_sig_heads=("n_significant", "max"),
            avg_pct=("pct_significant", "mean"),
        )
        .sort_values(["alpha", "condition"])
    )
    cond_csv = out_dir / "alpha_sweep_by_condition.csv"
    cond.to_csv(cond_csv, index=False)

    overall = (
        df.groupby("alpha", as_index=False)
        .agg(
            runs=("n_significant", "count"),
            avg_sig_heads=("n_significant", "mean"),
            min_sig_heads=("n_significant", "min"),
            max_sig_heads=("n_significant", "max"),
            avg_pct=("pct_significant", "mean"),
        )
        .sort_values("alpha")
    )
    overall_csv = out_dir / "alpha_sweep_overall.csv"
    overall.to_csv(overall_csv, index=False)

    md_path = out_dir / "alpha_sweep_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Alpha Sweep Summary (Permutation FWER)\n\n")
        f.write(f"- Results root: `{results_root.as_posix()}`\n")
        f.write(f"- Runs: {len(pt_files)}\n")
        f.write(f"- Permutations per run: {args.n_permutations}\n")
        f.write(f"- Alphas: {', '.join(str(a) for a in alphas)}\n\n")

        f.write("## Overall\n\n")
        f.write("| alpha | runs | avg sig. heads | avg % | range |\n")
        f.write("|---:|---:|---:|---:|---:|\n")
        for _, r in overall.iterrows():
            f.write(
                f"| {r['alpha']:.2f} | {int(r['runs'])} | {r['avg_sig_heads']:.2f} | "
                f"{format_pct(r['avg_pct'])} | {int(r['min_sig_heads'])}-{int(r['max_sig_heads'])} |\n"
            )

        f.write("\n## By Condition\n\n")
        f.write("| alpha | condition | runs | avg sig. heads | avg % | range |\n")
        f.write("|---:|---|---:|---:|---:|---:|\n")
        for _, r in cond.iterrows():
            f.write(
                f"| {r['alpha']:.2f} | {r['condition']} | {int(r['runs'])} | {r['avg_sig_heads']:.2f} | "
                f"{format_pct(r['avg_pct'])} | {int(r['min_sig_heads'])}-{int(r['max_sig_heads'])} |\n"
            )

    print("\nWrote:")
    print(f"- {run_csv.as_posix()}")
    print(f"- {cond_csv.as_posix()}")
    print(f"- {overall_csv.as_posix()}")
    print(f"- {md_path.as_posix()}")


if __name__ == "__main__":
    main()
