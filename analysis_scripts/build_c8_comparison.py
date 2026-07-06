"""C8 bf-control comparison: put C1 / C3 / C8 side-by-side at belief formation.

For each template x direction:
  C1 abstract_belief bf      (same belief varies false<->true, same answer)
  C3 answer_changes_belief bf (belief varies, answer differs)
  C8 answer_changes_location_only bf (belief CONSTANT false-false, answer differs)  <-- new

C8 isolates location/answer retrieval at bf under matched belief engagement.
Interpretation:
  - heads in C1 ∩ C3 but NOT C8  -> belief-specific signal
  - heads in C3 ∩ C8 but NOT C1  -> answer/location-specific signal
  - heads in all three            -> generic bf-position signal (suspect non-specific)

Writes c8_comparison.html. Renders *_singlecolor.png next to each source
heatmap using the same per-cell 99th-percentile YlOrRd scheme.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from condition_registry import BY_CNUM

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
# (direction_label, C1_rule_pair, C3_rule_pair, C8_rule_pair)
# C1/C3 use belief-varying rules (ABA<->ABB); C8 uses matching rule (ABA-ABA).
DIRECTIONS = [
    ("ABA → (false→true for C1/C3 | false→false for C8)", "ABA_exp_rule_ABB", "ABA_exp_rule_ABB", "ABA_exp_rule_ABA"),
    ("ABB → (true→false for C1/C3 | false→false for C8)", "ABB_exp_rule_ABA", "ABB_exp_rule_ABA", "ABB_exp_rule_ABB"),
]

# Folder names come from the canonical registry; the "C8" label here is this
# script's own bf-control nomenclature (== canonical C5 answer_changes_location_only).
COND_DIRS = {
    "C1 abstract_belief": BY_CNUM["C1"].folder,
    "C3 answer_changes_belief": BY_CNUM["C3"].folder,
    "C8 bf-control (new)": BY_CNUM["C5"].folder,
}


def find_sample_dir(cond_dir: str, template: str, rule_pair: str) -> Path | None:
    base = BASE / cond_dir / f"template_{template}" / f"base_rule_{rule_pair}" / "patch_after_movement" / "z_seed_0_shuffle_False" / "logit"
    if not base.exists():
        return None
    # one sample_num_*_gen_acc_0.9 child
    children = sorted(base.glob("sample_num_*_gen_acc_0.9"))
    return children[0] if children else None


def find_heatmap_png(d: Path) -> Path | None:
    # prefer the canonical (shortest) title
    cands = sorted(d.glob("*_heatmap.png"), key=lambda p: len(p.name))
    return cands[0] if cands else None


def render_singlecolor(scores_pt: Path, out_png: Path, title: str):
    x = torch.load(scores_pt, map_location="cpu", weights_only=False).numpy()
    x = np.maximum(x, 0.0)
    pos = x[x > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else 1.0
    if vmax <= 0:
        vmax = 1e-6
    n_clipped = int((x > vmax).sum())
    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(x, ax=ax, cmap="YlOrRd", vmin=0, vmax=vmax, cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_xlabel("Head index (0-39)", fontsize=11)
    ax.set_ylabel("Layer index (0-47)", fontsize=11)
    ax.set_title(f"{title}\n(neg->0, vmax={vmax:.3f} · {n_clipped}/1920 clipped)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return vmax, n_clipped


def process(scores_pt: Path, title_stem: str) -> tuple[Path | None, str]:
    out_png = scores_pt.parent / f"{title_stem}_singlecolor.png"
    try:
        vmax, n_clipped = render_singlecolor(scores_pt, out_png, title_stem)
    except Exception as e:
        return None, f"ERR: {e}"
    return out_png, f"vmax={vmax:.3f} clip={n_clipped}"


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def main():
    rows = []
    for template in TEMPLATES:
        for dir_label, c1_rp, c3_rp, c8_rp in DIRECTIONS:
            cells = []
            rule_pairs = {
                "C1 abstract_belief": c1_rp,
                "C3 answer_changes_belief": c3_rp,
                "C8 bf-control (new)": c8_rp,
            }
            for col_label, cond_dir in COND_DIRS.items():
                rp = rule_pairs[col_label]
                sd = find_sample_dir(cond_dir, template, rp)
                if sd is None:
                    cells.append({"label": col_label, "rule_pair": rp, "orig": None, "single": None, "info": "no sample dir"})
                    continue
                scores = sd / "causal_scores.pt"
                orig = find_heatmap_png(sd)
                single, info = (None, "no scores.pt")
                if scores.exists():
                    single, info = process(scores, f"{col_label.split()[0]}_{template}_{rp}")
                cells.append({
                    "label": col_label,
                    "rule_pair": rp,
                    "orig": rel(orig) if orig else None,
                    "single": rel(single) if single else None,
                    "info": info,
                })
                print(f"[{template}] {dir_label[:22]:22s} | {col_label:28s} | {info}")
            rows.append({"template": template, "dir_label": dir_label, "cells": cells})

    write_html(rows)


def write_html(rows):
    out = ROOT / "c8_comparison.html"
    chunks = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>C8 bf-control comparison</title>
<style>
body { font-family: -apple-system, sans-serif; margin: 20px; background: #fafafa; }
h1 { font-size: 20px; }
h2 { font-size: 14px; margin-top: 32px; color: #222; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
.note { font-size: 13px; color: #444; max-width: 1000px; line-height: 1.55; }
table { border-collapse: collapse; margin-top: 8px; }
th { font-size: 12px; padding: 6px 10px; background: #eee; text-align: center; }
td { padding: 4px; vertical-align: top; text-align: center; border: 1px solid #e5e5e5; }
.pair { display: flex; gap: 6px; align-items: flex-start; }
.pair img { max-width: 240px; height: auto; border: 1px solid #ccc; display: block; }
.pair > div { display: flex; flex-direction: column; align-items: center; }
.lbl { font-size: 10.5px; color: #666; margin-top: 2px; }
.info { font-size: 10.5px; color: #888; }
</style></head>
<body>
<h1>C8 bf-control: side-by-side with C1 & C3 (all at belief formation)</h1>
<p class="note">
Three conditions, all patched at the <b>belief-formation position</b>:
<b>C1 abstract_belief</b> (belief false↔true, answer same),
<b>C3 answer_changes_belief</b> (belief false↔true, answer differs),
<b>C8 answer_changes_location_only</b> (belief <u>constant</u> false-false, answer differs — new bf-control).
Each cell: left = canonical diverging heatmap, right = per-cell single-color (YlOrRd, negatives clamped to 0,
vmax = 99th pct of positive values). Single-color scales are <i>per cell</i>, so magnitudes are not comparable
across cells — use the diverging map for that; the single-color map surfaces within-cell peak structure.
</p>
<p class="note">
<b>What C8 isolates.</b> C8 holds belief type constant (both prompts false-belief) while varying the correct location answer.
So any bf-position heads that light up in C8 are doing <i>location/answer retrieval</i>, not belief tracking.
<br>
<b>Reading the three columns together:</b>
heads in C1 ∩ C3 but NOT in C8 are the cleanest belief-specific candidates.
Heads in all three are likely generic bf-position / retrieval signal and should be suspect.
</p>
"""]

    for r in rows:
        chunks.append(f'<h2>{r["template"]} &middot; {r["dir_label"]}</h2>')
        chunks.append("<table><tr>")
        for c in r["cells"]:
            chunks.append(f'<th>{c["label"]}<br><span class="info">{c["rule_pair"]}</span></th>')
        chunks.append("</tr><tr>")
        for c in r["cells"]:
            chunks.append("<td><div class='pair'>")
            if c["orig"]:
                chunks.append(f'<div><img src="{c["orig"]}"><div class="lbl">diverging</div></div>')
            else:
                chunks.append('<div class="lbl">(no heatmap)</div>')
            if c["single"]:
                chunks.append(f'<div><img src="{c["single"]}"><div class="lbl">single-color</div></div>')
            else:
                chunks.append(f'<div class="lbl">({c["info"]})</div>')
            chunks.append("</div>")
            chunks.append(f'<div class="info">{c["info"]}</div></td>')
        chunks.append("</tr></table>")
    chunks.append("</body></html>\n")
    out.write_text("\n".join(chunks))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
