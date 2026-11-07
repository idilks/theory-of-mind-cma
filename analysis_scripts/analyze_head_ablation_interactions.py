#!/usr/bin/env python3
"""Analyze saved belief/photo group ablations; never loads a language model.

Null: mean D(full)-D(shared)-D(remainder) = 0 on first-answer-token
probability scale. Paired, template-stratified percentile bootstrap intervals
are exploratory and conditional on these saved runs, not reproducibility CIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

VARIANTS = ("false_belief", "true_belief", "false_photo", "true_photo", "basic_ab", "basic_ba")
TEMPLATES = ("food_truck", "hair_styling", "library_book")
GROUPS = {
    "belief_minus_photo": ("belief_only_sysprompt_prefill", "decomp_intersection_belief_rank", "decomp_belief_only"),
    "photo_minus_belief": ("photo_heads_sysprompt_prefill", "decomp_intersection_photo_rank", "decomp_photo_only"),
}
SOURCES = {
    "belief_only_sysprompt_prefill": ("slurm_scripts/ablation_runs/baseline_systemprompt_prefill.sh", "slurm_logs/abl_prefill_8684593.err"),
    "photo_heads_sysprompt_prefill": ("slurm_scripts/ablation_runs/cross_frame_photo_heads.sh", "slurm_logs/abl_xframe_photo_8686510.err"),
    "decomp_intersection_photo_rank": ("slurm_scripts/ablation_decomposition.sh", "slurm_logs/abl_decomp_8686580_0.err"),
    "decomp_photo_only": ("slurm_scripts/ablation_decomposition.sh", "slurm_logs/abl_decomp_8686580_1.err"),
    "decomp_intersection_belief_rank": ("slurm_scripts/ablation_decomposition.sh", "slurm_logs/abl_decomp_8686580_2.err"),
    "decomp_belief_only": ("slurm_scripts/ablation_decomposition.sh", "slurm_logs/abl_decomp_8686580_3.err"),
}
METRICS = ("loss_full", "loss_shared", "loss_remainder", "additional_loss", "interaction")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_blocks(path: Path) -> list[str]:
    # Keep answer labels and rendered system/user/assistant text in identity.
    return [b.strip() for b in path.read_text().split("=" * 60) if b.strip()]


def template_of(block: str) -> str:
    user = block.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    matches = [name for name, marker in zip(TEMPLATES, ("food truck", "hair", "book")) if marker in user]
    if len(matches) != 1:
        raise ValueError(f"Ambiguous template classification: {matches}")
    return matches[0]


def verify_pairing(blocks: list[list[str]]) -> None:
    if not blocks or not blocks[0] or any(b != blocks[0] for b in blocks[1:]):
        raise ValueError("Prompt/answer records must match in exact row order across runs")


def endpoint_values(payload: dict[str, Any], summary: pd.DataFrame, variant: str) -> NDArray[np.float64]:
    raw = payload["target"][variant]["p_correct"]
    values = np.asarray(raw.cpu() if isinstance(raw, torch.Tensor) else raw, dtype=np.float64)
    expected = len(payload["target_heads"]) + 1
    if values.ndim != 2 or values.shape[1] != expected or not np.isfinite(values).all():
        raise ValueError(f"Bad p_correct shape/values for {variant}: {values.shape}")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be in [0,1]")
    saved = summary[(summary["curve"] == "target") & (summary["eval_variant"] == variant)].sort_values("k")
    if saved["k"].tolist() != list(range(expected)):
        raise ValueError("Cumulative k grid must be contiguous from zero through all target heads")
    if not np.allclose(values.mean(axis=0), saved["mean_p_correct"].to_numpy(), atol=1e-12, rtol=1e-10):
        raise ValueError("Tensor row order/means do not reproduce saved summary")
    return values[:, [0, -1]]


def contrasts(full: NDArray[np.float64], shared: NDArray[np.float64], remainder: NDArray[np.float64]) -> NDArray[np.float64]:
    if full.shape != shared.shape or full.shape != remainder.shape or full.ndim != 2 or full.shape[1] != 2:
        raise ValueError("Matched arrays must all have shape (n_prompts, 2)")
    df, ds, du = (a[:, 0] - a[:, 1] for a in (full, shared, remainder))
    return np.column_stack((df, ds, du, df - ds, df - ds - du))


def bootstrap_means(values: NDArray[np.float64], labels: NDArray[np.str_], draws: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Resample prompt rows jointly for all metrics, independently per template.

    Templates are fixed strata, weighted equally. This does not estimate
    generalization to new templates. Batching avoids a draws x n x metrics copy.
    """
    names = np.unique(labels)
    result = np.zeros((draws, values.shape[1]), dtype=np.float64)
    for name in names:
        subset = values[labels == name]
        for start in range(0, draws, 2000):
            stop = min(draws, start + 2000)
            indices = rng.integers(0, len(subset), size=(stop - start, len(subset)))
            result[start:stop] += subset[indices].mean(axis=1) / len(names)
    return result


def analyze(root: Path, out: Path, draws: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    runs: dict[str, dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for run, (launcher, log) in SOURCES.items():
        directory = root / "results/ablation" / run
        payload = torch.load(directory / "per_prompt_data.pt", map_location="cpu", weights_only=False)
        summary = pd.read_csv(directory / "cumulative_ablation_results.csv")
        heads = [tuple(map(int, h)) for h in payload["target_heads"]]
        csv_heads = pd.read_csv(directory / "head_list.csv")
        if heads != list(zip(csv_heads["layer"].astype(int), csv_heads["head"].astype(int))) or len(set(heads)) != len(heads):
            raise ValueError(f"Head identity/order mismatch in {run}")
        runs[run] = {"payload": payload, "summary": summary, "heads": set(heads)}
        paths = [directory / f for f in ("per_prompt_data.pt", "head_list.csv", "cumulative_ablation_results.csv")]
        paths += [directory / f"{v}_prompts.txt" for v in VARIANTS]
        paths += [root / launcher, root / log]
        for path in paths:
            artifacts.append({"run": run, "path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size})
        log_text = (root / log).read_text()
        expected_log = ("model id: Qwen/Qwen2.5-14B-Instruct", "'torch_dtype': torch.bfloat16", "prefill_mode=per_variant", f"/{run}/per_prompt_data.pt")
        if not all(t in log_text for t in expected_log):
            raise ValueError(f"Protocol/output linkage missing in {log}")
        metadata.append({"run": run, "n_heads": len(heads), "launcher": launcher, "log": log,
                         "model_id": "Qwen/Qwen2.5-14B-Instruct", "dtype_logged": "torch.bfloat16",
                         "prefill_logged": "per_variant", "seed_in_current_launcher": 1,
                         "historical_git_commit": "", "model_weight_revision": "", "gpu_model": "",
                         "exact_saved_config": "", "protocol_assessment": "rendered prompts and logged settings agree; exact historical revisions unrecorded"})
    if runs["decomp_intersection_belief_rank"]["heads"] != runs["decomp_intersection_photo_rank"]["heads"]:
        raise ValueError("Shared runs are not the same set")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    family_size = len(GROUPS) * len(VARIANTS) * (len(TEMPLATES) + 1)
    for domain, names in GROUPS.items():
        full_run, shared_run, remainder_run = names
        full_heads, shared_heads, remainder_heads = (runs[n]["heads"] for n in names)
        if shared_heads & remainder_heads or full_heads != shared_heads | remainder_heads:
            raise ValueError(f"Invalid disjoint union for {domain}")
        alternate_shared = "decomp_intersection_photo_rank" if "belief_rank" in shared_run else "decomp_intersection_belief_rank"
        for variant in VARIANTS:
            blocks = [prompt_blocks(root / "results/ablation" / n / f"{variant}_prompts.txt") for n in (*names, alternate_shared)]
            verify_pairing(blocks)
            labels = np.array([template_of(b) for b in blocks[0]])
            if set(labels.tolist()) != set(TEMPLATES):
                raise ValueError("Each evaluation must contain all three expected templates")
            arrays = [endpoint_values(runs[n]["payload"], runs[n]["summary"], variant) for n in (*names, alternate_shared)]
            if any(len(a) != len(labels) for a in arrays):
                raise ValueError("Saved prompt and tensor row counts differ")
            full, shared, remainder, alternate = arrays
            values = contrasts(full, shared, remainder)
            alternate_i = contrasts(full, alternate, remainder)[:, -1]
            # If a common clean probability is used, I = -p0-pF+pS+pU.
            endpoints = -full[:, 1] + shared[:, 1] + remainder[:, 1]
            anchor_i = np.column_stack([endpoints - a[:, 0] for a in arrays[:3]])
            base_range = np.ptp(np.column_stack([a[:, 0] for a in arrays]), axis=1)
            for i, block in enumerate(blocks[0]):
                row: dict[str, Any] = {"remainder_group": domain, "eval_variant": variant, "prompt_index": i,
                                      "template": labels[i], "prompt_sha256": hashlib.sha256(block.encode()).hexdigest(),
                                      "full_run": full_run, "shared_run": shared_run, "remainder_run": remainder_run}
                for role, a in zip(("full", "shared", "remainder", "alternate_shared"), arrays):
                    row[f"{role}_baseline"] = a[i, 0]
                    row[f"{role}_endpoint"] = a[i, 1]
                row.update(zip(METRICS, values[i]))
                row.update({"interaction_alternate_shared": alternate_i[i], "baseline_range_across_runs": base_range[i]})
                for role, val in zip(("full", "shared", "remainder"), anchor_i[i]):
                    row[f"interaction_common_{role}_baseline"] = val
                all_rows.append(row)
            for template in ("equal_template", *TEMPLATES):
                mask = np.ones(len(labels), dtype=bool) if template == "equal_template" else labels == template
                local_labels, local_values = labels[mask], values[mask]
                means = np.stack([local_values[local_labels == t].mean(axis=0) for t in np.unique(local_labels)]).mean(axis=0)
                boot = bootstrap_means(local_values, local_labels, draws, rng)
                marginal = np.quantile(boot, [0.025, 0.975], axis=0)
                family = np.quantile(boot[:, -1], [0.025 / family_size, 1 - 0.025 / family_size])
                row = {"remainder_group": domain, "eval_variant": variant, "template": template,
                       "n_prompts": int(mask.sum()), "n_templates": len(np.unique(local_labels)),
                       "n_remainder_heads": len(remainder_heads), "n_shared_heads": len(shared_heads),
                       "bootstrap_draws": draws, "seed": seed, "interaction_family_size": family_size,
                       "interpretation": "exploratory; conditional on saved runs and fixed templates"}
                for j, metric in enumerate(METRICS):
                    row[metric] = means[j]
                    row[f"{metric}_ci95_low"], row[f"{metric}_ci95_high"] = marginal[:, j]
                row["interaction_family95_low"], row["interaction_family95_high"] = family
                row["baseline_range_mean"] = base_range[mask].mean()
                row["baseline_range_max"] = base_range[mask].max()
                row["interaction_alternate_shared"] = alternate_i[mask].mean()
                row["shared_endpoint_difference_mean"] = (alternate[:, 1] - shared[:, 1])[mask].mean()
                row["shared_endpoint_difference_max_abs"] = np.abs(alternate[:, 1] - shared[:, 1])[mask].max()
                row["common_baseline_interaction_min"] = anchor_i[mask].mean(axis=0).min()
                row["common_baseline_interaction_max"] = anchor_i[mask].mean(axis=0).max()
                summaries.append(row)
            counts = {t: int((labels == t).sum()) for t in TEMPLATES}
            print(f"{domain}/{variant}: n={len(labels)}, templates={counts}, shape={values.shape}, dtype={values.dtype}, I={values[:, -1].mean():.6f}")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out / "ablation_per_prompt.csv", index=False)
    pd.DataFrame(summaries).to_csv(out / "ablation_summary.csv", index=False)
    pd.DataFrame(artifacts).to_csv(out / "ablation_source_manifest.csv", index=False)
    pd.DataFrame(metadata).to_csv(out / "ablation_run_metadata.csv", index=False)
    (out / "ablation_analysis_config.json").write_text(json.dumps({
        "script": str(Path(__file__).resolve().relative_to(root)), "script_sha256": digest(Path(__file__)),
        "draws": draws, "seed": seed, "interaction_null": "mean D(full)-D(shared)-D(remainder)=0",
        "sampling": "paired prompts within fixed template; equal weight across available templates",
        "intervals": "percentile bootstrap marginal 95%; interaction Bonferroni 95% family over 48 summaries",
        "inference_scope": "exploratory; no run-to-run or new-template uncertainty represented",
        "historical_git_commit": "", "model_weight_revision": "",
    }, indent=2) + "\n")
    print(f"Saved {len(all_rows)} paired prompt rows, {len(summaries)} summaries, {len(artifacts)} source checksums to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.draws < 20000:
        parser.error("Use at least 20,000 draws for the family interval tails")
    root = args.root.resolve()
    analyze(root, args.output_dir or root / "results/aggregated/head_domain_comparison", args.draws, args.seed)


if __name__ == "__main__":
    main()
