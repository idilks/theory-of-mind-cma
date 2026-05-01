"""
Component 4 + 5: analyze the .pt files written by capture_attention_patterns.py.

For each top head, compute the average attention mass (across prompts) from
each query position to each named source-token category (state / char_q /
obj_q / belief_form / other). Output:

  - lookback_scores.csv: per (layer, head) summary stats + observed_score
  - top10_bar.png:       stacked bar of top-10 heads by attn_to_state
  - strip_<L>_<H>.png:   per-prompt heatmap of the top-3 heads' lookback attn

Usage:
  python analyze_lookback_attention.py \
      --capture_dir results/lookback_analysis/abstract_belief/template_food_truck/base_rule_ABA_exp_rule_ABB/patch_before_movement
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_lookback")


def attention_mass(attn_row: torch.Tensor, pos: int) -> float:
    """Sum attention over a single key position. Returns NaN if pos is invalid."""
    if pos < 0 or pos >= attn_row.shape[0]:
        return float("nan")
    val = attn_row[pos].item()
    if not np.isfinite(val):
        return float("nan")
    return val


def analyze(capture_dir: Path):
    payload = torch.load(capture_dir / "attention.pt", map_location="cpu", weights_only=False)
    attn = payload["attn"]  # (P, H, Q, S) float16, NaN-padded
    heads = payload["heads"]
    positions = payload["positions"]
    query_names = payload["query_pos_names"]
    meta = payload["meta"]
    n_prompts, n_heads, n_q, max_seq = attn.shape
    log.info(f"loaded {capture_dir / 'attention.pt'}  shape={tuple(attn.shape)}")
    log.info(f"meta: {meta}")

    # For each head, average attention to each named position across prompts.
    rows = []
    for hi, h in enumerate(heads):
        for qi, qname in enumerate(query_names):
            masses = {"state": [], "char_q": [], "obj_q": [], "belief_form": []}
            for pi, pos_dict in enumerate(positions):
                row = attn[pi, hi, qi].float()  # (max_seq,)
                # Replace NaN-padding with 0 for the "valid" range only when we use them as floats
                seq_len = pos_dict["seq_len"]
                valid = row[:seq_len]
                if torch.isnan(valid).any():
                    # head not captured for this layer (shouldn't happen)
                    continue
                for cat in masses:
                    p = pos_dict[cat]
                    if 0 <= p < seq_len:
                        masses[cat].append(valid[p].item())

                # Also compute "self" mass at the query token itself for sanity
                # (we report mean elsewhere)
            mean_masses = {f"{qname}_attn_to_{k}": (np.mean(v) if v else float("nan"))
                           for k, v in masses.items()}
            rows.append({
                "layer": h["layer"],
                "head": h["head"],
                "observed_score": h["observed_score"],
                "p_value": h["p_value"],
                "significant": h["significant"],
                **mean_masses,
            })

    # Collapse: one row per (layer, head) with both query-position columns flattened.
    per_head = {}
    for r in rows:
        key = (r["layer"], r["head"])
        per_head.setdefault(key, {"layer": r["layer"], "head": r["head"],
                                  "observed_score": r["observed_score"],
                                  "p_value": r["p_value"],
                                  "significant": r["significant"]})
        for k, v in r.items():
            if k.startswith("lookback_") or k.startswith("belief_formation_"):
                per_head[key][k] = v

    df = pd.DataFrame(list(per_head.values()))
    df = df.sort_values("lookback_attn_to_state", ascending=False).reset_index(drop=True)

    out_csv = capture_dir / "lookback_scores.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}")

    # ---- viz 1: top-10 bar chart ----
    top10 = df.head(10).copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(top10))
    width = 0.25
    cats = [("lookback_attn_to_state", "state"),
            ("lookback_attn_to_char_q", "char"),
            ("lookback_attn_to_obj_q", "obj")]
    for i, (col, label) in enumerate(cats):
        vals = top10[col].fillna(0).values
        ax.bar(x + (i - 1) * width, vals, width, label=label)
    labels = [f"L{int(r.layer)}H{int(r.head)}" for r in top10.itertuples()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean attention from lookback to source")
    ax.set_title(f"top-10 heads by lookback→state attention\n"
                 f"{meta['condition']} / template_{meta['template']} / "
                 f"{meta['patch_position']} / base_rule={meta['base_rule']}")
    ax.legend()
    fig.tight_layout()
    bar_path = capture_dir / "top10_bar.png"
    fig.savefig(bar_path, dpi=120)
    plt.close(fig)
    log.info(f"wrote {bar_path}")

    # ---- viz 2: per-prompt strip heatmap for top-3 heads ----
    for rank in range(min(3, len(df))):
        L = int(df.iloc[rank].layer)
        H = int(df.iloc[rank].head)
        # find head index in original heads list
        hi = next(i for i, h in enumerate(heads) if h["layer"] == L and h["head"] == H)

        # Build a (n_prompts, max_seq) matrix from query=lookback (qi=0)
        mat = attn[:, hi, 0, :].float().numpy()
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0,
                       vmax=np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0)
        # overlay state-token positions as red dots
        state_xs, state_ys = [], []
        for pi, pos_dict in enumerate(positions):
            if 0 <= pos_dict["state"] < pos_dict["seq_len"]:
                state_xs.append(pos_dict["state"])
                state_ys.append(pi)
        ax.scatter(state_xs, state_ys, s=8, c="red", marker="x", label="state token")
        ax.set_xlabel("key position")
        ax.set_ylabel("prompt index")
        ax.set_title(f"L{L}H{H}  lookback→key attention  "
                     f"(state tokens marked in red)\n"
                     f"observed_score={df.iloc[rank].observed_score:.3f}, "
                     f"mean attn_to_state={df.iloc[rank].lookback_attn_to_state:.3f}")
        ax.legend(loc="upper right")
        fig.colorbar(im, ax=ax, label="attention weight")
        fig.tight_layout()
        strip_path = capture_dir / f"strip_L{L}_H{H}.png"
        fig.savefig(strip_path, dpi=120)
        plt.close(fig)
        log.info(f"wrote {strip_path}")

    # Summary printout
    log.info("=== top-5 heads by lookback→state ===")
    print(df.head(5).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", required=True, type=Path,
                    help="directory containing attention.pt (output of capture script)")
    args = ap.parse_args()
    analyze(args.capture_dir)


if __name__ == "__main__":
    main()
