"""
Per-token lookback-attention profile.

Beyond the named-anchor scalars in lookback_scores.csv (state_first /
state_second / state_correct / info_loss / char_q / obj_q / belief_form),
look at the full per-token attention distribution from the lookback
position. Detects peaks NOT on any named anchor -- positions the prior
analysis might have missed (e.g., a different epistemic-access point).

Uses the full attn tensor saved by capture_attention_patterns.py:
attn[prompt, head, query, key] where query=0 is lookback.

Usage:
  # single capture dir
  python per_token_profile.py --capture_dir results/lookback_analysis/.../patch_before_movement

  # cross-cell aggregation (after all per-cell runs have written
  # unannotated_peaks.csv files)
  python per_token_profile.py --root results/lookback_analysis
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("per_token_profile")

ANCHOR_KEYS = ["state_first", "state_second", "state_correct",
               "info_loss", "char_q", "obj_q", "belief_form"]
ANCHOR_COLORS = {
    "state_first":   "#1f77b4",
    "state_second":  "#ff7f0e",
    "state_correct": "#2ca02c",
    "info_loss":     "#d62728",
    "char_q":        "#9467bd",
    "obj_q":         "#8c564b",
    "belief_form":   "#e377c2",
}


def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")


def build_right_aligned_profile(attn: torch.Tensor, positions: list, hi: int,
                                 q_idx: int = 0):
    """Mean attention vs offset-from-end (0 = final token), from query row q_idx."""
    max_offset = max(p["seq_len"] for p in positions) - 1
    sums = np.zeros(max_offset + 1, dtype=np.float64)
    counts = np.zeros(max_offset + 1, dtype=np.int64)

    for pi, pos in enumerate(positions):
        seq_len = pos["seq_len"]
        row = attn[pi, hi, q_idx, :seq_len].float().numpy()
        if not np.all(np.isfinite(row)):
            continue
        for k in range(seq_len):
            offset = (seq_len - 1) - k
            sums[offset] += row[k]
            counts[offset] += 1

    mean_attn = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)

    anchor_offsets = {a: [] for a in ANCHOR_KEYS}
    for pos in positions:
        seq_len = pos["seq_len"]
        for a in ANCHOR_KEYS:
            v = pos.get(a, -1)
            if 0 <= v < seq_len:
                anchor_offsets[a].append((seq_len - 1) - v)

    return mean_attn, counts, anchor_offsets


def find_unannotated_peaks(mean_attn, counts, anchor_offsets,
                           threshold=0.02, window=2, min_support=3):
    """Offsets-from-end with mean attn >= threshold, not within +/- window of any anchor mean."""
    anchor_means = {a: int(round(float(np.mean(off))))
                    for a, off in anchor_offsets.items() if off}
    excluded = set()
    for off in anchor_means.values():
        for d in range(-window, window + 1):
            excluded.add(off + d)

    peaks = []
    for offset, val in enumerate(mean_attn):
        if not np.isfinite(val) or val < threshold:
            continue
        if counts[offset] < min_support:
            continue
        if offset in excluded:
            continue
        peaks.append((offset, float(val), int(counts[offset])))
    peaks.sort(key=lambda r: -r[1])
    return peaks, anchor_means


def token_examples_at_offset(prompt_token_ids, positions, offset, tokenizer, n=5):
    """Return up to n token strings from different prompts at this offset-from-end."""
    out = []
    for ids, pos in zip(prompt_token_ids, positions):
        seq_len = pos["seq_len"]
        idx = (seq_len - 1) - offset
        if 0 <= idx < len(ids):
            out.append(tokenizer.decode([ids[idx]]).replace("\n", "\\n"))
        if len(out) >= n:
            break
    return out


def analyze_capture_dir(capture_dir: Path, top_k: int = 5,
                        peak_threshold: float = 0.02, anchor_window: int = 2,
                        min_support: int = 3, max_offset_plot: int = 120,
                        load_tokenizer: bool = True,
                        query_row: str = "lookback"):
    payload = torch.load(capture_dir / "attention.pt", map_location="cpu", weights_only=False)
    attn = payload["attn"]
    heads = payload["heads"]
    positions = payload["positions"]
    prompts = payload["prompts"]
    meta = payload["meta"]
    query_pos_names = payload.get("query_pos_names", ["lookback", "belief_formation"])
    if query_row not in query_pos_names:
        raise ValueError(
            f"query_row={query_row!r} not in capture {capture_dir}: available={query_pos_names}"
        )
    q_idx = query_pos_names.index(query_row)
    # File suffix: empty for the historical default (lookback), explicit otherwise.
    suffix = "" if query_row == "lookback" else f"_{query_row}"

    score_csv = capture_dir / "lookback_scores.csv"
    if score_csv.exists():
        df = pd.read_csv(score_csv).sort_values("lookback_attn_to_state", ascending=False)
        ranked = df.head(top_k)[["layer", "head"]].astype(int).values.tolist()
    else:
        ranked = [[int(h["layer"]), int(h["head"])] for h in heads[:top_k]]

    tokenizer = _load_tokenizer() if load_tokenizer else None
    prompt_token_ids = (
        [tokenizer(p, add_special_tokens=False)["input_ids"] for p in prompts]
        if tokenizer is not None else [[] for _ in prompts]
    )

    fig, axes = plt.subplots(len(ranked), 1, figsize=(13, 2.4 * len(ranked)),
                             sharex=True, squeeze=False)
    peaks_rows = []

    for rank, (L, H) in enumerate(ranked):
        try:
            hi = next(i for i, h in enumerate(heads) if int(h["layer"]) == L and int(h["head"]) == H)
        except StopIteration:
            continue

        mean_attn, counts, anchor_offsets = build_right_aligned_profile(attn, positions, hi, q_idx=q_idx)
        peaks, anchor_means = find_unannotated_peaks(
            mean_attn, counts, anchor_offsets,
            threshold=peak_threshold, window=anchor_window, min_support=min_support,
        )

        ax = axes[rank, 0]
        xs = np.arange(min(len(mean_attn), max_offset_plot))
        ax.plot(xs, mean_attn[:len(xs)], lw=1.0, color="black")
        ax.fill_between(xs, 0, np.nan_to_num(mean_attn[:len(xs)]), alpha=0.15, color="black")
        ax.set_xlim(left=max_offset_plot, right=0)  # 0 (lookback) on the right
        ax.set_ylabel(f"L{L}H{H}\nmean attn", fontsize=9)

        for a, mean_off in anchor_means.items():
            if mean_off > max_offset_plot:
                continue
            ax.axvline(mean_off, color=ANCHOR_COLORS[a], lw=1.2, alpha=0.75)

        for off, val, support in peaks[:5]:
            if off > max_offset_plot:
                continue
            ax.scatter([off], [val], s=55, marker="*", c="gold",
                       edgecolors="black", lw=0.5, zorder=10)
            ex = (token_examples_at_offset(prompt_token_ids, positions, off, tokenizer)
                  if tokenizer is not None else [])
            peaks_rows.append({
                "layer": L, "head": H,
                "offset_from_end": off,
                "mean_attn": val,
                "support_n_prompts": support,
                "examples": " | ".join(repr(t) for t in ex),
            })

    legend_handles = [plt.Line2D([0], [0], color=ANCHOR_COLORS[a], lw=2, label=a)
                      for a in ANCHOR_KEYS]
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="gold",
                                     markeredgecolor="black", lw=0,
                                     markersize=10, label="unannotated peak"))
    axes[0, 0].legend(handles=legend_handles, loc="upper left", fontsize=7, ncol=4)
    axes[-1, 0].set_xlabel(f"tokens before query={query_row}  (0 = query token)")
    fig.suptitle(
        f"Per-token attention from query={query_row}  --  "
        f"{meta['condition']} / {meta['template']} / "
        f"base={meta['base_rule']} / {meta.get('patch_position','')}",
        fontsize=10,
    )
    fig.tight_layout()
    out_png = capture_dir / f"per_token_profile{suffix}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log.info(f"wrote {out_png}")

    peaks_csv = capture_dir / f"unannotated_peaks{suffix}.csv"
    pd.DataFrame(peaks_rows).to_csv(peaks_csv, index=False)
    log.info(f"wrote {peaks_csv}  ({len(peaks_rows)} rows)")


def aggregate_root(root: Path, query_row: str = "lookback"):
    rows = []
    suffix = "" if query_row == "lookback" else f"_{query_row}"
    pattern = f"unannotated_peaks{suffix}.csv"
    for csv in root.rglob(pattern):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        parts = csv.relative_to(root).parts
        if len(parts) < 4:
            continue
        df["condition"] = parts[0]
        df["template"] = parts[1].replace("template_", "")
        df["base_rule"] = parts[2].split("_")[2] if "_" in parts[2] else ""
        df["patch_position"] = parts[3]
        rows.append(df)

    summary_dir = root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        log.warning(f"no {pattern} under {root}")
        return
    full = pd.concat(rows, ignore_index=True)
    full.to_csv(summary_dir / f"unannotated_peaks_all{suffix}.csv", index=False)

    grp = (full.groupby(["layer", "head", "offset_from_end"])
           .agg(n_runs=("mean_attn", "size"),
                mean_attn=("mean_attn", "mean"),
                max_attn=("mean_attn", "max"))
           .reset_index()
           .sort_values(["n_runs", "mean_attn"], ascending=[False, False]))
    out = summary_dir / f"unannotated_peaks_by_head{suffix}.csv"
    grp.to_csv(out, index=False)
    log.info(f"wrote {out}  ({len(grp)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", type=Path,
                    help="single capture dir containing attention.pt")
    ap.add_argument("--root", type=Path,
                    help="aggregate mode: root containing many capture dirs")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--peak_threshold", type=float, default=0.02)
    ap.add_argument("--anchor_window", type=int, default=2)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--max_offset_plot", type=int, default=120)
    ap.add_argument("--no_tokenizer", action="store_true",
                    help="skip token-text lookup (no HF tokenizer load)")
    ap.add_argument("--query_row", default="lookback",
                    choices=["lookback", "belief_formation", "agent_last", "agent_first"],
                    help="which query row of attn[.,.,q,.] to profile (default: lookback)")
    ap.add_argument("--all_query_rows", action="store_true",
                    help="run for every query row available in the capture")
    args = ap.parse_args()

    rows_to_run = (
        ["lookback", "belief_formation", "agent_last", "agent_first"]
        if args.all_query_rows else [args.query_row]
    )

    # When --root is given, walk every cell with an attention.pt and run
    # analyze_capture_dir per query row before aggregating. This is what most
    # users want from --root: "do everything end-to-end across the tree."
    cell_dirs = (
        sorted({p.parent for p in args.root.rglob("attention.pt")})
        if args.root else []
    )
    if args.root:
        log.info(f"found {len(cell_dirs)} capture cells under {args.root}")

    for qrow in rows_to_run:
        if args.capture_dir:
            try:
                analyze_capture_dir(args.capture_dir, top_k=args.top_k,
                                    peak_threshold=args.peak_threshold,
                                    anchor_window=args.anchor_window,
                                    min_support=args.min_support,
                                    max_offset_plot=args.max_offset_plot,
                                    load_tokenizer=not args.no_tokenizer,
                                    query_row=qrow)
            except ValueError as e:
                log.warning(f"skipping query_row={qrow}: {e}")
        if args.root:
            for cd in cell_dirs:
                try:
                    analyze_capture_dir(cd, top_k=args.top_k,
                                        peak_threshold=args.peak_threshold,
                                        anchor_window=args.anchor_window,
                                        min_support=args.min_support,
                                        max_offset_plot=args.max_offset_plot,
                                        load_tokenizer=not args.no_tokenizer,
                                        query_row=qrow)
                except ValueError as e:
                    log.warning(f"skipping {cd} query_row={qrow}: {e}")
            aggregate_root(args.root, query_row=qrow)
    if not args.capture_dir and not args.root:
        ap.error("provide --capture_dir or --root")
    return


if __name__ == "__main__":
    main()
