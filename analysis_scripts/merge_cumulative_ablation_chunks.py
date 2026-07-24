"""Merge chunked cumulative-ablation outputs into the canonical run artifacts.

ablation_experiment.py can now run a subset of the work (--curves, and a
--k_index_start/--k_index_end slice of the k grid) so a full-1920 sweep fits
inside SLURM wall-time limits. Each chunk job writes a complete-looking
output dir (cumulative_ablation_results.csv + per_prompt_data.pt) covering
only its slice. This script stitches the chunks back into the single-run
format every downstream consumer expects, in the parent output dir:

    cumulative_ablation_results.csv   (all curves, all k)
    per_prompt_data.pt                (step axes concatenated in k order)
    cumulative_ablation_plot.png

Usage:
    python scripts/merge_cumulative_ablation_chunks.py \
        --chunks results/ablation/LABEL/chunk_* \
        --output_dir results/ablation/LABEL \
        --run_label LABEL
"""

import argparse
import filecmp
import glob
import os
import sys

import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "codebase", "tasks", "causal_analysis"))


def _load_plot_results():
    # deferred: importing ablation_experiment os.chdir()s to codebase/, which
    # would break relative --chunks/--output_dir paths resolved before it
    from ablation_experiment import plot_results
    return plot_results


CURVES = ("target", "control")


def _concat_step_axis(chunks_data):
    """Concatenate per-variant packed dicts along the k-step axis.

    chunks_data: list of {variant: {key: tensor/list}} in k order.
    """
    variants = list(chunks_data[0].keys())
    merged = {}
    for v in variants:
        keys = list(chunks_data[0][v].keys())
        merged[v] = {}
        for key in keys:
            vals = [c[v][key] for c in chunks_data]
            if key == "pool_chars":
                assert all(x == vals[0] for x in vals), \
                    f"{v}: pool_chars differ across chunks"
                merged[v][key] = vals[0]
            elif key == "gen_text":
                merged[v][key] = [sum((chunk[i] for chunk in vals), [])
                                  for i in range(len(vals[0]))]
            else:  # tensors shaped (n_prompts, n_steps[, n_pool])
                merged[v][key] = torch.cat(vals, dim=1)
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", nargs="+", required=True,
                    help="chunk output dirs (globs ok)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--run_label", default="merged")
    args = ap.parse_args()

    chunk_dirs = sorted(set(
        os.path.abspath(d) for pattern in args.chunks for d in glob.glob(pattern)))
    chunk_dirs = [d for d in chunk_dirs if os.path.isdir(d)]
    args.output_dir = os.path.abspath(args.output_dir)
    if not chunk_dirs:
        sys.exit("no chunk dirs matched")
    print(f"merging {len(chunk_dirs)} chunks:")
    for d in chunk_dirs:
        print(f"  {d}")

    # sanity: eval prompts must be identical across chunks (same seed/args)
    ref = chunk_dirs[0]
    for d in chunk_dirs[1:]:
        for f in glob.glob(os.path.join(ref, "*_prompts.txt")):
            other = os.path.join(d, os.path.basename(f))
            if os.path.exists(other) and not filecmp.cmp(f, other, shallow=False):
                sys.exit(f"prompt mismatch: {f} vs {other} — chunks were not "
                         "generated with identical seeds/args")

    payloads = []
    for d in chunk_dirs:
        p = os.path.join(d, "per_prompt_data.pt")
        if not os.path.exists(p):
            sys.exit(f"missing {p} — chunk incomplete")
        payloads.append(torch.load(p, weights_only=False))

    merged = {}
    for curve in CURVES:
        pieces = [(pl[f"{curve}_k_values"], pl[curve])
                  for pl in payloads if curve in pl]
        if not pieces:
            print(f"note: no chunks contain the {curve} curve")
            continue
        pieces.sort(key=lambda kv: kv[0][0])
        k_all = [k for ks, _ in pieces for k in ks]
        if any(b <= a for a, b in zip(k_all, k_all[1:])):
            sys.exit(f"{curve}: chunk k grids overlap or are out of order: "
                     f"{[ks for ks, _ in pieces]}")
        merged[curve] = _concat_step_axis([data for _, data in pieces])
        merged[f"{curve}_k_values"] = k_all
        print(f"{curve}: {len(pieces)} chunks -> {len(k_all)} k steps "
              f"(k {k_all[0]}..{k_all[-1]})")

    for key in ("target_heads", "control_heads"):
        vals = [pl[key] for pl in payloads if key in pl]
        assert all(v == vals[0] for v in vals), f"{key} differ across chunks"
        merged[key] = vals[0]

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = [pd.read_csv(os.path.join(d, "cumulative_ablation_results.csv"))
           for d in chunk_dirs]
    df = pd.concat(dfs, ignore_index=True)
    dup = df.duplicated(subset=["k", "curve", "eval_variant"])
    if dup.any():
        sys.exit(f"{dup.sum()} duplicate (k, curve, eval_variant) rows — "
                 "overlapping chunks?")
    df = df.sort_values(["curve", "k"], kind="stable").reset_index(drop=True)
    csv_path = os.path.join(args.output_dir, "cumulative_ablation_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")

    pt_path = os.path.join(args.output_dir, "per_prompt_data.pt")
    torch.save(merged, pt_path)
    print(f"wrote {pt_path}")

    plot_path = os.path.join(args.output_dir, "cumulative_ablation_plot.png")
    _load_plot_results()(df, plot_path, run_label=args.run_label)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
