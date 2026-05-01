"""Generate results_for_pi/report.qmd — PI-facing Quarto report.

Design decisions:
- Images are referenced via resolved paths (no symlink indirection) so quarto
  preview's dev server can find them. Requires `_quarto.yml` at the repo root
  so the project root includes both results_for_pi/ and results/.
- Every heatmap gets a schema block next to it containing the FULL base and
  exp prompts, with the belief-formation (or photo-event) marker highlighted
  in purple via a .marker CSS class.
- A summary navigation table at the top links into each condition × template
  block.

Run:
    python scripts/build_qmd.py
    quarto preview results_for_pi/report.qmd
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import os
import re
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results_for_pi" / "report.qmd"
TREE = ROOT / "results_for_pi"
C8_BASE = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct/answer_changes_location_only"

sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))
from prompt_generators.tom_templates import UnifiedTemplateSystem

TEMPLATES = ["food_truck", "hair_styling", "library_book"]
TS = UnifiedTemplateSystem(vocab_dir=str(ROOT / "tom_datasets"))

ABSTRACT_MARKERS = {t: (TS.get_template(t, "abstract").false_belief_marker,
                        TS.get_template(t, "abstract").true_belief_marker)
                    for t in TEMPLATES}

# Photo event is not stored as a single marker on the template; both prompts
# contain "photographer takes a photo of the {object}" — we highlight that
# phrase in both, since it's the event being re-ordered relative to the move.
PHOTO_MARKER_PATTERN = re.compile(r"(?i)(a photographer takes a photo of the [^.]+?)(?=[.])")


@dataclass
class Cell:
    condition_key: str
    condition_title: str
    template: str
    direction_dir: str          # e.g. ABA_to_ABB
    patch: str                  # bf | end
    heatmap_rel: str
    singlecolor_rel: str | None
    schema_md: str
    anchor: str


def rel_to_qmd(p: Path) -> str:
    rel = os.path.relpath(str(p.resolve()), str(OUT.parent))
    return quote(rel, safe="/")


def pick_heatmap(leaf: Path) -> tuple[Path | None, Path | None]:
    if not leaf.exists():
        return None, None
    divs = sorted(
        [p for p in leaf.glob("*_heatmap.png") if "singlecolor" not in p.name],
        key=lambda p: len(p.name),
    )
    sc = sorted(leaf.glob("*_heatmap_singlecolor_global.png"), key=lambda p: len(p.name))
    if not sc:
        sc = sorted(leaf.glob("*_singlecolor.png"), key=lambda p: len(p.name))
    return (divs[0] if divs else None), (sc[0] if sc else None)


def highlight(text: str, phrases: list[str]) -> str:
    """Wrap each phrase in text with a purple-marker span. Longer matches first."""
    out = text
    for phr in sorted(set(p for p in phrases if p), key=len, reverse=True):
        if phr in out:
            out = out.replace(phr, f'[[MARKER_OPEN]]{phr}[[MARKER_CLOSE]]')
    # Also try photo-event pattern (regex) if no literal matches yet applied
    # Guarded so we only tag if the literal marker phrases weren't present.
    if "[[MARKER_OPEN]]" not in out:
        out = PHOTO_MARKER_PATTERN.sub(r'[[MARKER_OPEN]]\1[[MARKER_CLOSE]]', out, count=1)
    out = out.replace("[[MARKER_OPEN]]", '<span class="marker">')
    out = out.replace("[[MARKER_CLOSE]]", "</span>")
    return out


def prompt_pair(template: str, context_type: str, base_rule: str,
                answer_differs: bool, same_belief: bool = False) -> tuple[str, str, str, str]:
    b, e, ba, ea, _ = TS.generate_cross_context_pair(
        template, base_rule=base_rule, context_type=context_type,
        answer_differs=answer_differs, same_belief=same_belief,
    )
    return b, e, str(ba), str(ea)


def schema_block(*, cond_key: str, cond_desc: str, template: str, direction_label: str,
                 patch: str, base: str, exp: str, base_ans: str, exp_ans: str,
                 what_varies: str, markers: list[str]) -> str:
    base_h = highlight(base, markers)
    exp_h = highlight(exp, markers)
    return (
        f"**Condition:** {cond_key} — {cond_desc}  \n"
        f"**Template:** `{template}`  \n"
        f"**Direction:** {direction_label}  \n"
        f"**Patch position:** `{patch}`  \n"
        f"**What varies:** {what_varies}  \n"
        f"**Answers:** base = `{base_ans}`, exp = `{exp_ans}`\n\n"
        f'<div class="prompt"><b>Base prompt:</b><br>{base_h}</div>\n\n'
        f'<div class="prompt"><b>Exp prompt:</b><br>{exp_h}</div>\n'
    )


def make_c1(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_to_ABB" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "abstract", base_rule, answer_differs=False)
    fb, tb = ABSTRACT_MARKERS[tmpl]
    belief_dir = "false → true belief" if direction == "ABA_to_ABB" else "true → false belief"
    return schema_block(
        cond_key="C1 abstract_belief",
        cond_desc="belief varies, answer same",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} ({belief_dir})",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="belief type only; answer held constant.",
        markers=[fb, tb],
    )


def make_c2(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_to_ABB" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "photo", base_rule, answer_differs=False)
    photo_dir = "photo-before-move → photo-after-move" if direction == "ABA_to_ABB" else "photo-after → photo-before"
    return schema_block(
        cond_key="C2 abstract_photo",
        cond_desc="photo timing varies, answer same",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} ({photo_dir})",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="photo-timing (before vs after object move); answer held constant. State-tracking analogue of C1.",
        markers=[],
    )


def make_c3(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_to_ABB" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "abstract", base_rule, answer_differs=True)
    fb, tb = ABSTRACT_MARKERS[tmpl]
    belief_dir = "false → true belief" if direction == "ABA_to_ABB" else "true → false belief"
    return schema_block(
        cond_key="C3 answer_changes_belief",
        cond_desc="belief AND answer vary (confounded)",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} ({belief_dir})",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="belief type + answer (confounded).",
        markers=[fb, tb],
    )


def make_c4(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_timing_varies" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "photo", base_rule, answer_differs=True)
    photo_dir = "photo-before → photo-after" if direction == "ABA_timing_varies" else "photo-after → photo-before"
    return schema_block(
        cond_key="C4 answer_changes_photo",
        cond_desc="photo timing AND answer vary",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} ({photo_dir})",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="photo timing + answer. State-tracking analogue of C3.",
        markers=[],
    )


def make_c5(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_location_only" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "control", base_rule, answer_differs=True)
    loc_dir = "ABA (base) vs ABA-swapped (exp)" if direction == "ABA_location_only" else "ABB (base) vs ABB-swapped (exp)"
    return schema_block(
        cond_key="C5 control",
        cond_desc="location tokens only — pure location-retrieval control",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} ({loc_dir})",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="substitution tokens only (no belief, no photo).",
        markers=[],
    )


def make_c8(tmpl: str, direction: str, patch: str) -> str:
    base_rule = "ABA" if direction == "ABA_to_ABA" else "ABB"
    b, e, ba, ea = prompt_pair(tmpl, "abstract", base_rule, answer_differs=True, same_belief=True)
    fb, _ = ABSTRACT_MARKERS[tmpl]
    return schema_block(
        cond_key="C8 answer_changes_location_only",
        cond_desc="BF-CONTROL — belief held constant (both false), answer varies",
        template=tmpl,
        direction_label=f"{direction.replace('_', ' ')} (both false-belief, substitutions swap)",
        patch=patch, base=b, exp=e, base_ans=ba, exp_ans=ea,
        what_varies="substitutions only (→ answer). Belief type constant. Isolates answer/location retrieval at bf under matched belief engagement.",
        markers=[fb],
    )


SCHEMA_FNS = {"C1": make_c1, "C2": make_c2, "C3": make_c3, "C4": make_c4, "C5": make_c5}
COND_TITLE = {
    "C1": "C1 Abstract belief (same answer, belief varies)",
    "C2": "C2 Abstract photo (same answer, photo timing varies)",
    "C3": "C3 Answer-changes belief (belief and answer vary)",
    "C4": "C4 Answer-changes photo (timing and answer vary)",
    "C5": "C5 Control (location tokens only, no belief/photo)",
    "C8": "C8 Answer-changes location-only (bf-control: belief constant, answer varies)",
}
DIR_ORDER = {
    "C1": ["ABA_to_ABB", "ABB_to_ABA"],
    "C2": ["ABA_to_ABB", "ABB_to_ABA"],
    "C3": ["ABA_to_ABB", "ABB_to_ABA"],
    "C4": ["ABA_timing_varies", "ABB_timing_varies"],
    "C5": ["ABA_location_only", "ABB_location_only"],
    "C8": ["ABA_to_ABA", "ABB_to_ABB"],
}
PATCHES = {"C1": ["bf", "end"], "C2": ["bf", "end"], "C3": ["bf", "end"],
           "C4": ["bf", "end"], "C5": ["end"], "C8": ["bf"]}


def anchor_for(cond: str, tmpl: str) -> str:
    return f"sec-{cond.lower()}-{tmpl}"


def collect(cond: str) -> list[Cell]:
    cells = []
    fn = SCHEMA_FNS[cond] if cond != "C8" else make_c8
    for tmpl in TEMPLATES:
        for direction in DIR_ORDER[cond]:
            for patch in PATCHES[cond]:
                leaf = TREE / cond / f"template_{tmpl}" / direction / patch
                div, sc = pick_heatmap(leaf)
                if div is None:
                    print(f"  MISSING: {cond} {tmpl} {direction} {patch}")
                    continue
                cells.append(Cell(
                    condition_key=cond,
                    condition_title=COND_TITLE[cond],
                    template=tmpl,
                    direction_dir=direction,
                    patch=patch,
                    heatmap_rel=rel_to_qmd(div),
                    singlecolor_rel=rel_to_qmd(sc) if sc else None,
                    schema_md=fn(tmpl, direction, patch),
                    anchor=anchor_for(cond, tmpl),
                ))
    return cells


def emit_cell(cell: Cell) -> str:
    imgs = f'![diverging]({cell.heatmap_rel}){{width=100%}}\n\n'
    if cell.singlecolor_rel:
        imgs += f'![single-color (global vmax)]({cell.singlecolor_rel}){{width=100%}}\n'
    head = f"#### {cell.template} · {cell.direction_dir.replace('_', ' ')} · patch at `{cell.patch}`\n\n"
    return (
        f"{head}"
        f"::: {{layout-ncol=2}}\n\n"
        f"{cell.schema_md}\n\n"
        f"{imgs}\n"
        f":::\n\n"
    )


def write_nav_table(all_cells: dict[str, list[Cell]]) -> str:
    out = ["\n## Quick-jump table\n\n"]
    out.append("Every cell links to that condition × template block (which contains all directions and patch positions for that pair).\n\n")
    headers = ["Template"] + list(all_cells.keys())
    out.append("| " + " | ".join(headers) + " |\n")
    out.append("|" + "|".join(["---"] * len(headers)) + "|\n")
    for tmpl in TEMPLATES:
        row = [f"`{tmpl}`"]
        for cond in all_cells:
            has = any(c.template == tmpl for c in all_cells[cond])
            if has:
                row.append(f"[{cond}](#{anchor_for(cond, tmpl)})")
            else:
                row.append("—")
        out.append("| " + " | ".join(row) + " |\n")
    out.append("\n")
    return "".join(out)


HEADER = """---
title: "CMA of theory of mind in Qwen2.5-14B-Instruct — PI report"
author: "Idil Sahin"
date: today
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    number-sections: true
    theme: cosmo
    fig-align: center
    embed-resources: false
    css: report.css
---

# Executive summary

Causal mediation analysis (CMA) patches the z-activation of each of the 1920 attention heads (48 layers × 40 heads) between pairs of prompts and measures the change in the final-token logit. We ran a **2×2 factorial** of (belief varies?) × (answer varies?) across two patch positions (belief-formation `bf` and end-of-prompt `end`), three templates, and two base rules. The factorial is completed by **C8** (`answer_changes_location_only`) which was run only at `bf` as the bf-control.

**Headline.** At `bf`, C8 produces **zero FWER-significant heads in every one of 6 runs**, while C1 and C3 produce ~10 sig heads each. Pearson r on mean raw causal scores: r(C1,C3)=**0.607**, r(C1,C8)=**0.033**, r(C3,C8)=**−0.065**. The bf-position signal is **belief-specific**, not generic location/answer retrieval.

Belief-formation markers are highlighted in <span class="marker">purple</span> inline in each prompt so the patched token is obvious.
"""

BF_SUMMARY = """
# Belief-formation comparison: C1 / C3 / C8 summary figures

## Universal heads Venn (≥2 of 6 runs)

![](venn_C1_C3_C8.png){width=70%}

|U(C1)|=2, |U(C3)|=2, |U(C8)|=0. Belief-specific region C1 ∩ C3 \\ C8 = **1 head**.

## Pearson r on mean raw scores (1920-d)

![](scatter_bf.png){width=95%}

r(C1,C3)=0.607 [0.493, 0.705]; r(C1,C8)=0.033 [−0.022, 0.086]; r(C3,C8)=−0.065 [−0.112, −0.014].

## RSA — condition-space clustering at bf (18 runs)

![](rdm_bf.png){width=95%}

Top: 1 − Pearson on z-scored raw scores. Bottom: 1 − Jaccard on sig masks.

## RSA — end runs (30 runs, full C1–C5)

![](rdm_end.png){width=95%}

End shows clean within-condition clustering (within=0.345, between=0.507). C3 and C5 collapse (d=0.391), confirming that at end the signal is dominated by location/answer retrieval.
"""

METHODS = """
# Methods {#sec-methods}

**Patch positions.** `bf` (belief-formation) patches at the token where the epistemic event occurs — for belief conditions this is the false/true-belief marker (see schema blocks), for photo conditions it is the photograph-event sentence. `end` patches at the final token (−1).

**Permutation test (per run).** For each of the 1920 heads we ran a 5000-permutation sign-flip test on the per-sample causal scores (LLMSymbMech §B.2). FWER is controlled across heads via the max-statistic method. α=0.05. Each condition × template × direction is an independent run (50 prompts; gen-acc filter ≥0.9).

**Universal set.** A head is in U_K(C) if it is FWER-significant in ≥K of the 6 runs (3 templates × 2 directions) for condition C. For the Venn we use K=2 because per-run sig sets are small (mean 4–5 heads for C1/C3) and C8 is empty at every K.

**Pearson r and bootstrap CI.** On mean raw causal scores averaged across the 6 bf runs of a condition. 95% CI from 5000 nonparametric bootstrap resamples over the 1920 heads.

**RDMs.** Continuous: within-run z-score clipped to |z|≤5, pairwise 1 − Pearson. Binary: pairwise 1 − Jaccard on FWER-sig masks. Average-linkage hierarchical clustering on the condensed distance matrix.

**Scale (single-color heatmaps).** 99th percentile of positive scores pooled across all cells at the same patch position. bf vmax = 0.1750; end vmax = 1.8593.

# Known caveats

1. **Control `bf` omitted** — the belief-formation detector collapsed to −1 for 5 of 6 control cells. Control shows end only.
2. **C3 hair_styling ABB→ABA at bf** has anomalously large negatives — possible residual partial-fallback.
3. **End-patch magnitudes are substantially larger** than bf-patch magnitudes.
4. **C8 per-run FWER-sig counts are all 0** — this is the finding, not a bug.

# Predictions stated before running

| Prediction | Held? | Evidence |
|---|:---:|---|
| Within-condition runs cluster together (bf) | ✗ | within=1.006, between=0.994 |
| Within-condition runs cluster together (end) | ✓ | within=0.345, between=0.507 |
| C1 and C3 closer than either to C8 at bf | ✓ | d(C1,C3)=0.934, d(C1,C8)=0.993, d(C3,C8)=1.004 |
| C3 and C5 close at end (location-retrieval) | ✓ | d(C3,C5)=0.391 vs mean other-cross=0.491 |
"""


def main():
    all_cells = {cond: collect(cond) for cond in ["C1", "C2", "C3", "C4", "C5", "C8"]}
    chunks: list[str] = [HEADER]
    chunks.append(write_nav_table(all_cells))
    chunks.append(BF_SUMMARY)

    chunks.append("\n# Full heatmap grid — every condition × template\n\n")
    chunks.append("Each block below is a condition × template pair. Within the block, all directions and patch positions for that pair are shown in order.\n\n")

    for cond in ["C1", "C3", "C8", "C2", "C4", "C5"]:
        chunks.append(f"## {COND_TITLE[cond]}\n\n")
        by_tmpl: dict[str, list[Cell]] = {}
        for c in all_cells[cond]:
            by_tmpl.setdefault(c.template, []).append(c)
        for tmpl in TEMPLATES:
            tcells = by_tmpl.get(tmpl, [])
            if not tcells:
                continue
            chunks.append(f"### {tmpl} {{#{anchor_for(cond, tmpl)}}}\n\n")
            for c in tcells:
                chunks.append(emit_cell(c))

    chunks.append(METHODS)

    OUT.write_text("".join(chunks))
    print(f"wrote {OUT} ({sum(len(c) for c in chunks)} chars)")


def write_css():
    css = """body { max-width: 1500px; }
img { border: 1px solid #ccc; }
h4 { color: #555; font-size: 13px; margin-top: 18px; }
.marker { color: #8e44ad; font-weight: 600; background: #f3e8ff; padding: 0 3px; border-radius: 2px; }
.prompt { background: #fafafa; border-left: 3px solid #bbb; padding: 8px 12px; margin: 6px 0; font-size: 13px; line-height: 1.45; }
.prompt b { color: #333; }
table { font-size: 13px; }
"""
    (OUT.parent / "report.css").write_text(css)


if __name__ == "__main__":
    write_css()
    main()
