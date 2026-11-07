"""
Token-level attention bars over a real prompt, for CMA-significant heads only.

For each capture dir (one condition x template x rule cell written by
capture_attention_patterns.py), plots final-token ("lookback") attention over
every token of an example prompt:

  - top row: CMA-weighted mean over the cell's significant heads
            (weight = max(observed_score, 0)), one bar per token
  - below:   one heatmap row per significant head (sorted by CMA score), so
            per-head targets are visible instead of averaged away

Values are averaged over the prompts whose token layout matches the example
prompt exactly (same seq_len and same anchor indices), so every bar sits on the
same token slot in every averaged prompt. The number of matched prompts is
printed in the panel title. Tokens that differ across matched prompts (e.g.
location names of equal token length) are labelled with the example prompt's
text.

Also writes, over ALL prompts of each cell (no layout matching needed):
  region_summary.csv: per significant head, attention mass on
      first_token | state_correct | state_other | other_anchors | self | rest

No forward passes; CPU only. Reads:
  {capture_dir}/attention.pt   (attn, heads, prompts, positions, query_pos_names, meta)

Usage:
  python scripts/lookback_analysis/prompt_attention_bars.py \
      --capture_dir results/lookback_analysis/abstract_belief/template_food_truck/base_rule_ABA_exp_rule_ABB/patch_before_movement \
      --label "C1 divergence (false-belief prompt)" \
      --capture_dir results/lookback_analysis/answer_changes_location_only/template_food_truck/base_rule_ABA_exp_rule_ABA/patch_before_movement \
      --label "C5 retrieval (false-belief prompt)" \
      --out_stem results/lookback_analysis/_prompt_bars/C1_vs_C5_food_truck_ABA
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
log = logging.getLogger("prompt_attention_bars")

ANCHORS = ["state_first", "state_second", "state_correct",
           "info_loss", "char_q", "obj_q", "belief_form"]
OTHER_ANCHORS = ["info_loss", "char_q", "obj_q", "belief_form"]

C_CORRECT = "#2a78d6"   # correct-answer location token
C_OTHER_LOC = "#eb6834" # the other location token
C_ANCHOR = "#52514e"    # info_loss / char_q / obj_q / belief_form
C_REST = "#c9c8c2"


def load_tokenizer(name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # offline node / no cache: fall back to index labels
        log.warning(f"tokenizer unavailable ({e}); labelling tokens by index")
        return None


def significant_heads(heads: list[dict]) -> tuple[list[int], np.ndarray, list[str]]:
    """Indices into the payload's head axis, CMA weights, and labels, sorted by score."""
    df = pd.DataFrame(heads).reset_index().rename(columns={"index": "hi"})
    if "significant" not in df.columns:
        raise ValueError("payload heads lack a 'significant' field")
    score_col = "observed_score" if "observed_score" in df.columns else "causal_score"
    df = df[df["significant"].astype(bool)].sort_values(score_col, ascending=False)
    if df.empty:
        raise ValueError("no significant heads in this cell")
    w = np.clip(df[score_col].to_numpy(dtype=float), 0, None)
    labels = [f"L{int(r['layer'])}H{int(r['head'])}  ({r[score_col]:.2f})" for _, r in df.iterrows()]
    return df["hi"].astype(int).tolist(), w, labels


def layout_key(pos: dict) -> tuple:
    return (pos["seq_len"],) + tuple(pos.get(a, -1) for a in ANCHORS)


def analyze_cell(capture_dir: Path, prompt_idx: int, tokenizer, q_name: str = "lookback"):
    payload = torch.load(capture_dir / "attention.pt", map_location="cpu", weights_only=False)
    attn = payload["attn"].float()                       # (P, H, Q, S)
    positions = payload["positions"]
    prompts = payload["prompts"]
    q_idx = payload.get("query_pos_names", ["lookback"]).index(q_name)
    his, w, head_labels = significant_heads(payload["heads"])

    ex = positions[prompt_idx]
    key = layout_key(ex)
    matched = [i for i, p in enumerate(positions) if layout_key(p) == key]
    seq_len = ex["seq_len"]

    # (n_matched, n_sig, seq_len) -> mean over matched prompts
    block = attn[matched][:, his, q_idx, :seq_len].numpy()
    per_head = np.nanmean(block, axis=0)                 # (n_sig, seq_len)
    weighted = (w[:, None] * per_head).sum(0) / w.sum() if w.sum() > 0 else per_head.mean(0)

    if tokenizer is not None:
        ids = tokenizer(prompts[prompt_idx], add_special_tokens=False)["input_ids"]
        assert len(ids) == seq_len, f"token count {len(ids)} != stored seq_len {seq_len}"
        toks = [tokenizer.decode([t]).replace("\n", "\\n") for t in ids]
    else:
        toks = [str(i) for i in range(seq_len)]

    role = np.array(["rest"] * seq_len, dtype=object)
    for a in OTHER_ANCHORS:
        if 0 <= ex.get(a, -1) < seq_len:
            role[ex[a]] = "anchor"
    other_loc = ex["state_second"] if ex["state_correct"] == ex["state_first"] else ex["state_first"]
    role[other_loc] = "other_location"
    role[ex["state_correct"]] = "correct_location"

    # Region summary over ALL prompts (anchor indices are per prompt, no matching needed)
    rows = []
    for j, hi in enumerate(his):
        acc = {k: [] for k in ["first_token", "state_correct", "state_other",
                               "other_anchors", "self", "rest"]}
        for pi, p in enumerate(positions):
            row = attn[pi, hi, q_idx, :p["seq_len"]].numpy()
            if not np.all(np.isfinite(row)):
                continue
            sc = p["state_correct"]
            so = p["state_second"] if sc == p["state_first"] else p["state_first"]
            oa = sorted({p[a] for a in OTHER_ANCHORS if 0 <= p.get(a, -1) < p["seq_len"]} - {sc, so, 0})
            last = p["seq_len"] - 1
            named = {0, sc, so, last, *oa}
            acc["first_token"].append(row[0])
            acc["state_correct"].append(row[sc])
            acc["state_other"].append(row[so])
            acc["other_anchors"].append(row[oa].sum() if oa else 0.0)
            acc["self"].append(row[last])
            acc["rest"].append(row.sum() - sum(row[k] for k in named))
        rows.append({"head": head_labels[j].split()[0], "cma_weight": w[j],
                     **{k: float(np.mean(v)) for k, v in acc.items()}})
    region = pd.DataFrame(rows)

    tok_df = pd.DataFrame({"token_idx": range(seq_len), "token": toks, "role": role,
                           "weighted_mean": weighted})
    for j, lab in enumerate(head_labels):
        tok_df[lab.split()[0]] = per_head[j]

    return dict(toks=toks, role=role, weighted=weighted, per_head=per_head,
                head_labels=head_labels, n_matched=len(matched), n_prompts=len(positions),
                seq_len=seq_len, tok_df=tok_df, region=region, meta=payload.get("meta", {}))


def plot(cells: list[dict], labels: list[str], out_stem: Path, max_heads: int):
    n_tok = max(c["seq_len"] for c in cells)
    heights = []
    for c in cells:
        heights += [2.2, 0.28 * min(len(c["head_labels"]), max_heads) + 0.4]
    fig, axes = plt.subplots(len(heights), 1, figsize=(max(14, 0.16 * n_tok), sum(heights) + 0.6),
                             gridspec_kw={"height_ratios": heights}, constrained_layout=True)
    color = {"correct_location": C_CORRECT, "other_location": C_OTHER_LOC,
             "anchor": C_ANCHOR, "rest": C_REST}
    vmax = max(np.nanmax(c["per_head"][:max_heads]) for c in cells)

    for k, (c, lab) in enumerate(zip(cells, labels)):
        ax, axh = axes[2 * k], axes[2 * k + 1]
        x = np.arange(c["seq_len"])
        ax.bar(x, c["weighted"], width=0.8, color=[color[r] for r in c["role"]])
        ax.axhline(1 / c["seq_len"], color=C_ANCHOR, lw=0.8, ls=":")
        ax.set_xlim(-0.6, n_tok - 0.4)
        ax.set_ylabel("attention\n(CMA-weighted)", fontsize=8)
        ax.set_title(f"{lab}: {len(c['head_labels'])} significant heads, mean over "
                     f"{c['n_matched']}/{c['n_prompts']} prompts with this exact token layout",
                     fontsize=10, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(c["toks"], rotation=90, fontsize=6)
        for t, r in zip(ax.get_xticklabels(), c["role"]):
            if r != "rest":
                t.set_color(color[r]); t.set_fontweight("bold")
        ax.spines[["top", "right"]].set_visible(False)

        ph = c["per_head"][:max_heads]
        pad = np.full((ph.shape[0], n_tok), np.nan)
        pad[:, :c["seq_len"]] = ph
        im = axh.imshow(pad, aspect="auto", cmap="Blues", vmin=0, vmax=vmax, interpolation="nearest")
        axh.set_yticks(range(ph.shape[0]))
        axh.set_yticklabels(c["head_labels"][:max_heads], fontsize=7)
        axh.set_xticks([])
        axh.set_xlim(-0.6, n_tok - 0.4)
        fig.colorbar(im, ax=axh, fraction=0.015, pad=0.005, label="per head")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color[r]) for r in color] + \
              [plt.Line2D([0], [0], color=C_ANCHOR, ls=":")]
    axes[0].legend(handles, ["correct-answer location", "other location",
                             "info-loss / agent / object / belief-formation", "other tokens",
                             "uniform attention (1 / prompt length)"],
                   loc="upper right", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(1, 1.35))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    log.info(f"-> {out_stem.with_suffix('.png')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--out_stem", type=Path, required=True)
    ap.add_argument("--prompt_idx", type=int, default=0)
    ap.add_argument("--max_heads", type=int, default=20)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-14B-Instruct")
    args = ap.parse_args()
    assert len(args.capture_dir) == len(args.label), "one --label per --capture_dir"

    tok = load_tokenizer(args.tokenizer)
    cells = []
    for cd, lab in zip(args.capture_dir, args.label):
        c = analyze_cell(cd, args.prompt_idx, tok)
        log.info(f"{lab}: {len(c['head_labels'])} sig heads, layout-matched prompts "
                 f"{c['n_matched']}/{c['n_prompts']}, seq_len {c['seq_len']}")
        log.info("region summary (mean attention mass, all prompts):\n"
                 + c["region"].round(3).to_string(index=False))
        cells.append(c)

    args.out_stem.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([c["tok_df"].assign(cell=lab) for c, lab in zip(cells, args.label)]) \
        .to_csv(args.out_stem.with_name(args.out_stem.name + "_tokens.csv"), index=False)
    pd.concat([c["region"].assign(cell=lab) for c, lab in zip(cells, args.label)]) \
        .to_csv(args.out_stem.with_name(args.out_stem.name + "_region_summary.csv"), index=False)
    plot(cells, args.label, args.out_stem, args.max_heads)


if __name__ == "__main__":
    main()
