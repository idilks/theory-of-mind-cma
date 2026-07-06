"""
Cross-condition per-token attention comparison.

For a given (layer, head) and query row, overlays the right-aligned per-token
attention profile across all conditions on a single axis. The point is:
"this head spikes at offset 60 in abstract_belief but not in state_only_no_belief"
is more informative than "this head spikes at offset 60." The base
per_token_profile.py only plots one cell at a time.

Two modes:
  --layer L --head H        : single (L, H) cross-condition figure
  --top_n_universal N       : pulls top N heads from
                              _summary/universal_heads_new_schema.csv
                              and writes one figure per head.

Pools prompts across templates × directions within a condition (each cell's
attention.pt contributes its per-prompt rows). Uses the same right-aligned
offset-from-end coordinate as per_token_profile.py so anchors / peaks line up.

Reads:
  results/lookback_analysis/{condition}/template_*/base_rule_*/patch_*/attention.pt

Writes:
  results/lookback_analysis/_summary/
      cross_condition_L{L}H{H}_{query_row}.png
      cross_condition_L{L}H{H}_{query_row}.csv     (mean per offset per condition)
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cross_condition_per_token")


# Preset comparisons answering specific hypotheses. Each is
# (cond_a, cond_b, set_name) — set_name is one of "intersection",
# "a_only", "b_only" matching condition_overlap_heads_end.csv.
# Add or edit entries here to extend the default sweep.
PRESET_SETS: dict[str, tuple[str, str, str]] = {
    "belief_specific":     ("abstract_belief", "state_only_no_belief", "a_only"),
    "shared_belief_photo": ("abstract_belief", "abstract_photo",       "intersection"),
    "belief_not_photo":    ("abstract_belief", "abstract_photo",       "a_only"),
    "photo_specific":      ("abstract_photo",  "state_only_no_belief", "a_only"),
    "photo_not_belief":    ("abstract_photo",  "abstract_belief",      "a_only"),
}


CONDITION_ORDER = [
    "abstract_belief",
    "abstract_photo",
    "answer_changes_belief",
    "answer_changes_photo",
    "answer_changes_location_only",
    "answer_changes_location_only_photo",
    "state_only_no_belief",
]
CONDITION_COLORS = {
    "abstract_belief":                    "#1f77b4",
    "abstract_photo":                     "#ff7f0e",
    "answer_changes_belief":              "#2ca02c",
    "answer_changes_photo":               "#d62728",
    "answer_changes_location_only":       "#9467bd",
    "answer_changes_location_only_photo": "#8c564b",
    "state_only_no_belief":               "#7f7f7f",
}


def discover_cells(root: Path):
    """Yield (condition, capture_dir) for every attention.pt under root."""
    for pt in root.rglob("attention.pt"):
        rel = pt.relative_to(root).parts
        if not rel:
            continue
        condition = rel[0]
        yield condition, pt.parent


def accumulate_profile_for_head(capture_dirs, layer: int, head: int,
                                 query_row: str = "lookback"):
    """Pool right-aligned attention from (L, H, query_row) across capture cells.

    Returns (mean, count, max_offset). NaN where count == 0.
    Per-prompt rows shorter than max_offset are simply not counted at the
    missing offsets — counts[off] tells you the support.
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    max_offset = 0

    for cd in capture_dirs:
        try:
            payload = torch.load(cd / "attention.pt", map_location="cpu",
                                 weights_only=False)
        except Exception as e:
            log.warning(f"failed to load {cd}: {e}")
            continue
        attn = payload["attn"]
        heads = payload["heads"]
        positions = payload["positions"]
        query_pos_names = payload.get(
            "query_pos_names", ["lookback", "belief_formation"]
        )
        if query_row not in query_pos_names:
            log.warning(f"{cd}: query_row={query_row} not in {query_pos_names}; skipping")
            continue
        q_idx = query_pos_names.index(query_row)
        try:
            hi = next(i for i, h in enumerate(heads)
                      if int(h["layer"]) == layer and int(h["head"]) == head)
        except StopIteration:
            continue

        for pi, pos in enumerate(positions):
            seq_len = pos["seq_len"]
            row = attn[pi, hi, q_idx, :seq_len].float().numpy()
            if not np.all(np.isfinite(row)):
                continue
            for k in range(seq_len):
                offset = (seq_len - 1) - k
                sums[offset] += float(row[k])
                counts[offset] += 1
                if offset > max_offset:
                    max_offset = offset

    if max_offset == 0 and not sums:
        return None, None, 0
    arr_sum = np.zeros(max_offset + 1)
    arr_cnt = np.zeros(max_offset + 1, dtype=np.int64)
    for off, v in sums.items():
        arr_sum[off] = v
        arr_cnt[off] = counts[off]
    mean = np.where(arr_cnt > 0, arr_sum / np.maximum(arr_cnt, 1), np.nan)
    return mean, arr_cnt, max_offset


def cells_by_condition(root: Path, conditions=None):
    by_cond = defaultdict(list)
    for cond, cd in discover_cells(root):
        if conditions and cond not in conditions:
            continue
        by_cond[cond].append(cd)
    return by_cond


def plot_one_head(root: Path, layer: int, head: int, query_row: str,
                   max_offset_plot: int, conditions=None, min_support: int = 5):
    by_cond = cells_by_condition(root, conditions)
    if not by_cond:
        log.warning(f"no cells found under {root}")
        return

    summary_dir = root / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    rows_df = []
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ordered = [c for c in CONDITION_ORDER if c in by_cond]
    for cond in ordered:
        mean, counts, max_off = accumulate_profile_for_head(
            by_cond[cond], layer, head, query_row=query_row
        )
        if mean is None:
            continue
        # Mask offsets with too little support (otherwise the tail is all noise).
        plot_vals = np.where(counts >= min_support, mean, np.nan)
        xs = np.arange(min(len(plot_vals), max_offset_plot))
        ax.plot(xs, plot_vals[:len(xs)], lw=1.5,
                color=CONDITION_COLORS.get(cond, "black"),
                label=cond, alpha=0.9)
        for off, val, cnt in zip(range(len(mean)), mean, counts):
            if cnt >= min_support:
                rows_df.append({
                    "layer": layer, "head": head, "query_row": query_row,
                    "condition": cond, "offset_from_end": int(off),
                    "mean_attn": float(val), "n_prompts": int(cnt),
                })

    ax.set_xlim(left=max_offset_plot, right=0)
    ax.set_xlabel(f"tokens before query={query_row}  (0 = query token)")
    ax.set_ylabel("mean attention")
    ax.set_title(f"L{layer}H{head}  --  per-token attention by condition  (query={query_row})")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()

    out_png = summary_dir / f"cross_condition_L{layer}H{head}_{query_row}.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    log.info(f"wrote {out_png}")

    out_csv = summary_dir / f"cross_condition_L{layer}H{head}_{query_row}.csv"
    pd.DataFrame(rows_df).to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(rows_df)} rows)")


def heads_from_universal(root: Path, top_n: int) -> list[tuple[int, int]]:
    """Pull top heads from universal_heads_new_schema.csv if present.
    Falls back to top heads in unannotated_peaks_by_head.csv (excluding offset 0)."""
    candidates = [
        root / "_summary" / "universal_heads_new_schema.csv",
        root / "_summary" / "universal_heads_new_cols.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            l_col = cols.get("layer")
            h_col = cols.get("head")
            if l_col and h_col:
                heads = df[[l_col, h_col]].astype(int).values.tolist()
                if heads:
                    log.info(f"using {path.name} (top {top_n} of {len(heads)})")
                    return [(int(L), int(H)) for L, H in heads[:top_n]]
    fallback = root / "_summary" / "unannotated_peaks_by_head.csv"
    if fallback.exists():
        df = pd.read_csv(fallback)
        df = df[df["offset_from_end"] > 0]
        df = (df.groupby(["layer", "head"])["mean_attn"].max()
              .reset_index().sort_values("mean_attn", ascending=False))
        log.info(f"using fallback {fallback.name} (top {top_n} by max mean_attn)")
        return [(int(L), int(H)) for L, H in df.head(top_n)[["layer", "head"]].values.tolist()]
    raise FileNotFoundError(f"no universal-heads CSV under {root}/_summary")


def heads_from_overlap_csv(overlap_csv: Path, cond_a: str, cond_b: str,
                            set_name: str) -> list[tuple[int, int]]:
    """Read condition_overlap_heads_end.csv and pull every head in
    (cond_a, cond_b, set_name). set_name in {intersection, a_only, b_only}."""
    df = pd.read_csv(overlap_csv)
    m = ((df["cond_a"] == cond_a) & (df["cond_b"] == cond_b)
         & (df["set"] == set_name))
    sub = df[m]
    if sub.empty:
        # Try the swapped pair: (cond_b, cond_a) with flipped a_only/b_only.
        flip_map = {"a_only": "b_only", "b_only": "a_only", "intersection": "intersection"}
        m2 = ((df["cond_a"] == cond_b) & (df["cond_b"] == cond_a)
              & (df["set"] == flip_map[set_name]))
        sub = df[m2]
    if sub.empty:
        return []
    return [(int(L), int(H)) for L, H in sub[["layer", "head"]].astype(int).values]


def run_preset(root: Path, overlap_csv: Path, preset_name: str,
                query_rows: list[str], max_offset_plot: int, min_support: int,
                conditions=None):
    if preset_name not in PRESET_SETS:
        raise ValueError(f"unknown preset {preset_name!r}; "
                         f"available: {sorted(PRESET_SETS)}")
    a, b, s = PRESET_SETS[preset_name]
    heads = heads_from_overlap_csv(overlap_csv, a, b, s)
    log.info(f"preset {preset_name!r}: {len(heads)} heads "
             f"({a} vs {b}, set={s})")
    if not heads:
        log.warning(f"  no heads found for {preset_name}; skipping")
        return
    for qrow in query_rows:
        for L, H in heads:
            plot_one_head(root, L, H, qrow,
                          max_offset_plot=max_offset_plot,
                          conditions=conditions,
                          min_support=min_support)


def main():
    ap = argparse.ArgumentParser(
        description="Cross-condition per-token attention rendering. "
                    "Default behavior (no head-selection flags): run every "
                    "preset in PRESET_SETS across all 4 query rows."
    )
    ap.add_argument("--root", type=Path, required=True,
                    help="lookback_analysis root containing per-condition cells")
    # Head selection — pick exactly one of these modes (or none, to use the
    # default = all presets × all query rows).
    ap.add_argument("--layer", type=int, help="single-head mode (with --head)")
    ap.add_argument("--head", type=int)
    ap.add_argument("--top_n_universal", type=int, default=0,
                    help="if >0, plot top N heads from universal_heads CSV")
    ap.add_argument("--preset", default=None,
                    help=f"named preset from PRESET_SETS "
                         f"(available: {sorted(PRESET_SETS)})")
    ap.add_argument("--from_overlap_csv", type=Path, default=None,
                    help="custom set: read this CSV and select --pair / --set")
    ap.add_argument("--pair", default=None,
                    help="for --from_overlap_csv: 'cond_a,cond_b'")
    ap.add_argument("--set", dest="set_name", default=None,
                    choices=["intersection", "a_only", "b_only"],
                    help="for --from_overlap_csv: which set within the pair")
    # Rendering / filtering
    ap.add_argument("--query_row", default="lookback",
                    choices=["lookback", "belief_formation", "agent_last", "agent_first"])
    ap.add_argument("--all_query_rows", action="store_true")
    ap.add_argument("--max_offset_plot", type=int, default=120)
    ap.add_argument("--min_support", type=int, default=5,
                    help="hide offsets with fewer than this many prompts contributing")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the overlay to these conditions (default: all found)")
    args = ap.parse_args()

    # Default overlap CSV path (auto-discovered) for --preset or default mode.
    default_overlap = args.root / "_summary" / "condition_overlap_heads_end.csv"

    # Determine the mode based on what's set.
    single_head = args.layer is not None and args.head is not None
    custom_set = args.from_overlap_csv is not None or args.pair is not None
    selection_set = single_head or args.top_n_universal > 0 or args.preset or custom_set

    # Default behavior: no head-selection flags → run every preset, all query rows.
    if not selection_set:
        log.info("no head-selection flag given; default = all presets × all 4 query rows")
        if not default_overlap.exists():
            ap.error(f"default overlap CSV not found at {default_overlap}; "
                     f"run per_condition_top_heads.py first or pass "
                     f"--layer/--head/--top_n_universal/--preset/--from_overlap_csv")
        rows = ["lookback", "belief_formation", "agent_last", "agent_first"]
        for preset_name in PRESET_SETS:
            run_preset(args.root, default_overlap, preset_name, rows,
                       max_offset_plot=args.max_offset_plot,
                       min_support=args.min_support,
                       conditions=args.conditions)
        return

    # Otherwise: figure out which heads to render.
    rows = (
        ["lookback", "belief_formation", "agent_last", "agent_first"]
        if args.all_query_rows else [args.query_row]
    )

    if args.preset:
        run_preset(args.root, default_overlap, args.preset, rows,
                   max_offset_plot=args.max_offset_plot,
                   min_support=args.min_support,
                   conditions=args.conditions)
        return

    if custom_set:
        if args.pair is None or args.set_name is None:
            ap.error("--from_overlap_csv requires --pair and --set")
        try:
            cond_a, cond_b = [s.strip() for s in args.pair.split(",")]
        except ValueError:
            ap.error("--pair must be 'cond_a,cond_b'")
        overlap_csv = args.from_overlap_csv or default_overlap
        heads = heads_from_overlap_csv(overlap_csv, cond_a, cond_b, args.set_name)
        log.info(f"custom set ({cond_a} vs {cond_b}, {args.set_name}): {len(heads)} heads")
    elif args.top_n_universal > 0:
        heads = heads_from_universal(args.root, args.top_n_universal)
    else:
        heads = [(args.layer, args.head)]

    for qrow in rows:
        for L, H in heads:
            plot_one_head(args.root, L, H, qrow,
                          max_offset_plot=args.max_offset_plot,
                          conditions=args.conditions,
                          min_support=args.min_support)


if __name__ == "__main__":
    main()
