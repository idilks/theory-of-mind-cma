"""
Generate results_for_pi/report-28-04.qmd from the heatmap manifest and the
canonical seed-source folders. End-patch only.

For each (condition, template, base->exp) cell:
  - read base_input_prompts_<N>.txt and exp_input_prompts_<N>.txt from the seed source
  - read ans_<N>.txt for base/exp answers (sample 0)
  - mark the final narrative token before the answer-instruction line in purple
  - emit two images: viridis CMA heatmap + binary FWER significance mask

Layout:
  preamble + condition table
  methods box
  full grid (one block per (condition, template, direction))
  predictions/hypothesis recap (placeholder; Step 5 will refresh numbers)

Run:
    python scripts/build_pi_qmd.py
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "results_for_pi/heatmaps/manifest.csv"
QMD_OUT = ROOT / "results_for_pi/report-28-04.qmd"

CONDITION_LABELS = {
    "C1_abstract_belief": ("C1 Abstract belief",
                           "belief type (false ↔ true), answer held constant",
                           "Does the model carry a belief representation independent of which answer it produces?"),
    "C2_abstract_photo": ("C2 Abstract photo",
                          "photo timing (snapshot before vs after change), answer held constant",
                          "Is the model doing generic state tracking rather than mentalising?"),
    "C3_answer_changes_belief": ("C3 Answer-changes belief",
                                 "belief type, answer changes",
                                 "Which heads carry belief information together with the answer?"),
    "C4_answer_changes_photo": ("C4 Answer-changes photo",
                                "photo timing, answer changes",
                                "Which heads carry state-tracking information together with the answer?"),
    "C5_control": ("C5 Control (location-only, ToM frame)",
                   "location tokens only; full ToM scaffold around them",
                   "Which heads do location retrieval inside an agent-with-belief frame?"),
    "C6_state_only_no_belief": ("C6 State-only (agent-free)",
                                "location tokens only; no observer, no belief frame",
                                "Which heads do location retrieval when no agent or belief is present at all?"),
}

PREAMBLE = """---
title: "CMA of theory of mind — end-patch report"
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
    embed-resources: true
    css: report.css
---

This report covers **end-patch only** (`patch_before_movement` in result paths).
Belief-formation patch results are not included here.

The patched token sits at position `-1` of the formatted prompt — i.e. at the
assistant-turn boundary, right after the answer-instruction line and just
before the model emits its first answer token. The last word of the
instruction is highlighted in <span class="marker">purple</span> as a visual
proxy for that position.
"""

METHODS = """
# Methods

**Causal mediation score (per head).** For each prompt pair we patch the `z`
output of one head from base into exp at the end-patch position and record the
change in the model's probability of the target answer. The score is the mean
shift across valid samples (gen_acc=0.9 filter).

**Permutation test.** 5000 permutations with max-statistic FWER correction at
α=0.05, per LLMSymbMech §B.2. The binary mask in the right-hand column shows
which heads survived this threshold.

**Sample sizes & seed sourcing.** seed=0 (50 prompts) and seed=1 (150 prompts)
runs use disjoint RNG streams (set_seed seeds python random + numpy), so when
they share a `(condition, template, rule_pair)` key they can be concatenated.
- `seed=combined` (n≈146-200): C1 abstract_belief, C4 answer_changes_photo —
  both seeds used matching rule pairs, so per-sample tensors are merged and
  the permutation test is re-run on the union.
- `seed=1` only (n≈129-150): C2 abstract_photo, C6 state_only_no_belief.
  C2's seed=0 photo_context runs predate the current condition system and used
  same-rule pairs (ABA→ABA / ABB→ABB), which doesn't test "same answer,
  different photo timing" — those legacy cells are dropped here. C6 had no
  seed=0 run.
- `seed=0` only (n=26-50): C5 control. No seed=1 control run.
- `seed=1` and `seed=0` mixed by cell: C3 answer_changes_belief. seed=1 used
  the canonical keep-rule convention (ABA→ABA / ABB→ABB) while seed=0 used
  flip-rule (ABA→ABB / ABB→ABA). Both rule-pair types are kept here as
  separate cells; no merging.

**Heatmap rendering.** Negatives clamped to 0; vmin=0; vmax = max of remaining
positive values; cmap = viridis. No upper thresholding.
"""


def find_prompts(source_dir: Path):
    """source_dir is a sample_num_*/ folder. Walk up to the seed_dir; if that
    seed_dir has no prompt files (e.g. z_seed_combined), fall back to sibling
    z_seed_0 then z_seed_1."""
    seed_dir = source_dir.parent.parent  # logit -> seed_dir
    candidates_dirs = [seed_dir]
    parent = seed_dir.parent
    for sibling_name in ("z_seed_0_shuffle_False", "z_seed_1_shuffle_False"):
        sib = parent / sibling_name
        if sib != seed_dir and sib.exists():
            candidates_dirs.append(sib)
    for d in candidates_dirs:
        files = sorted(d.glob("base_input_prompts_*.txt"), key=lambda p: -int(re.search(r"_(\d+)\.txt$", p.name).group(1)))
        if files:
            base_path = files[0]
            n_str = re.search(r"_(\d+)\.txt$", base_path.name).group(1)
            exp_path = d / f"exp_input_prompts_{n_str}.txt"
            ans_path = d / f"ans_{n_str}.txt"
            return base_path, exp_path, ans_path
    return None, None, None


def extract_first_user_prompt(path: Path) -> str:
    txt = path.read_text()
    parts = txt.split("<|im_end|>\n<|im_start|>user\n")
    if len(parts) < 2:
        return ""
    return parts[1].split("<|im_end|>")[0].strip()


def extract_first_answers(ans_path: Path):
    if not ans_path.exists():
        return "?", "?"
    line = ans_path.read_text().splitlines()[0]
    m = re.search(r"base_ans:\s*([^,]+),\s*exp_ans:\s*([^,]+)", line)
    if not m:
        return "?", "?"
    return m.group(1).strip(), m.group(2).strip()


ANSWER_INSTRUCTION_RE = re.compile(
    r"(?P<narrative>.+?)(?P<sep>\s*\n+|\s+)(?P<question>(?:Where|What|Which)[^\n]+\?(?:\s+Answer[^\n]*)?)\s*$",
    re.DOTALL,
)


def mark_end_patch(prompt: str) -> str:
    """Wrap the very last word of the prompt (end of instruction line) in purple.
    -1 sits at the assistant-turn boundary; the last word of 'be concise' is the
    closest visual proxy in the user-text view."""
    body = prompt.rstrip()
    last_space = body.rfind(" ")
    if last_space < 0:
        return prompt
    head, last = body[:last_space], body[last_space + 1:]
    return f'{head} <span class="marker">{last}</span>'


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def build_section(c_key: str, rows: pd.DataFrame, lines: list[str]):
    label, varies, asking = CONDITION_LABELS[c_key]
    lines.append(f"## {label}\n")
    lines.append(f"**What varies:** {varies}.  ")
    lines.append(f"**What we're asking:** {asking}\n")
    for template, t_rows in rows.groupby("template"):
        tname = template.replace("template_", "")
        anchor = f"sec-{c_key.lower()}-{tname}"
        lines.append(f"### {tname} {{#{anchor}}}\n")
        for _, r in t_rows.iterrows():
            base, exp = re.match(r"base_rule_(\w+)_exp_rule_(\w+)", r["rule_pair"]).groups()
            source_dir = ROOT / r["source"]
            base_path, exp_path, ans_path = find_prompts(source_dir)
            base_p = mark_end_patch(extract_first_user_prompt(base_path)) if base_path else "(prompt file missing)"
            exp_p = mark_end_patch(extract_first_user_prompt(exp_path)) if exp_path else "(prompt file missing)"
            base_a, exp_a = extract_first_answers(ans_path) if ans_path else ("?", "?")
            lines.append(f"#### {tname} · {base} → {exp}\n")
            lines.append(f"**Direction:** `{base}` → `{exp}`  ")
            lines.append(f"**n samples:** {r['n']} (seed={r['seed']})  ")
            lines.append(f"**Significant heads (FWER, α=0.05):** {r['n_sig']} / 1920  ")
            lines.append(f"**Answers (sample 0):** base = `{base_a}`, exp = `{exp_a}`\n")
            lines.append(f'<div class="prompt"><b>Base prompt:</b><br>{base_p}</div>\n')
            lines.append(f'<div class="prompt"><b>Exp prompt:</b><br>{exp_p}</div>\n')
            lines.append("::: {layout-ncol=2}")
            lines.append(f'![CMA score (viridis)](heatmaps/{c_key}/{template}/{r["rule_pair"]}/viridis.png){{width=100%}}\n')
            lines.append(f'![FWER mask (α=0.05)](heatmaps/{c_key}/{template}/{r["rule_pair"]}/significance.png){{width=100%}}\n')
            lines.append(":::\n")


def main():
    df = pd.read_csv(MANIFEST)
    lines: list[str] = [PREAMBLE, ""]

    # Condition table
    lines.append("# Conditions\n")
    lines.append("| # | Condition | What varies | Question |")
    lines.append("|---|---|---|---|")
    for c_key, (label, varies, asking) in CONDITION_LABELS.items():
        lines.append(f"| {label.split()[0]} | {label.split(' ', 1)[1]} | {varies} | {asking} |")
    lines.append("")

    # Quick-jump
    templates = sorted(df["template"].unique())
    cond_keys = list(CONDITION_LABELS.keys())
    lines.append("## Quick jump\n")
    lines.append("| Template | " + " | ".join(k.split("_")[0] for k in cond_keys) + " |")
    lines.append("|" + "|".join(["---"] * (len(cond_keys) + 1)) + "|")
    for t in templates:
        tname = t.replace("template_", "")
        cells = []
        for k in cond_keys:
            anchor = f"sec-{k.lower()}-{tname}"
            cells.append(f"[{k.split('_')[0]}](#{anchor})")
        lines.append(f"| `{tname}` | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(METHODS)
    lines.append("")
    lines.append("# Heatmap grid — every condition × template × direction\n")
    lines.append("Left column: continuous CMA score (viridis, neg→0). Right column: binary FWER-significant heads (α=0.05).\n")

    for c_key in cond_keys:
        rows = df[df["condition"] == c_key].copy()
        if rows.empty:
            continue
        build_section(c_key, rows, lines)

    lines.append("\n# Hypothesis recap\n")
    lines.append("(Numbers in this section will be refreshed once the RSA / clustering pass on combined-seed scores is run — Step 5 in the plan.)\n")
    lines.append("| Hypothesis | Evidence so far |")
    lines.append("|---|---|")
    lines.append("| H1: abstract belief ≠ control (location retrieval) | Spearman ρ ≈ 0, Jaccard ≈ 0 on seed=0 end-patch. Re-check on combined-seed CMA. |")
    lines.append("| H2: abstract belief ≈ photo (state-tracking, not mentalising) | r=0.69 on effect sizes; one shared cross-template head (L35H30) on seed=0. Re-check on combined-seed and on C6. |")
    lines.append("| H3: belief computed locally vs distributed | Mixed across templates — needs the bf vs end comparison; outside this report's scope (end-patch only). |")
    lines.append("| H4: control insensitive to patch position | Was supported on seed=0 (perfect overlap). Outside scope here (end-patch only). |")
    lines.append("| H5: cross-template robust heads | Abstract end-patch had L29H31, L29H34, L31H19, L34H12, L35H30 across all 3 templates on seed=0. Recompute from combined-seed FWER masks. |")

    QMD_OUT.write_text("\n".join(lines))
    print(f"wrote {QMD_OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
