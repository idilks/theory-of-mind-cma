#!/usr/bin/env python
"""Build a balanced 2x2 stimulus set for RSA on attention head outputs.

Design: per template, K=12 items per cell, 4 cells (belief x answer) = 48 stimuli.
Cells: {true_belief, false_belief} x {answer=v_a, answer=v_b}.

For each (loc_a, loc_b) pair we instantiate all 4 cells from the same non-
contrastive substitutions, so within a pair the only varying things are the
belief-template (false vs true) and which value fills var_1 vs var_2.

Belief is set by which context_template is used. Answer is determined by:
  false context -> answer = subs[belief_answer_key]   (= subs[var_1])
  true  context -> answer = subs[world_answer_key]    (= subs[var_2])

So with subs_orig = {var_1: v_a, var_2: v_b} and subs_swap swapping them,
we cover all 4 cells:
  (false, v_a): false_template.format(**subs_orig)
  (true,  v_b): true_template.format(**subs_orig)
  (false, v_b): false_template.format(**subs_swap)
  (true,  v_a): true_template.format(**subs_swap)

Output CSV (per template) at results/rsa_head_outputs/stimuli/{template}.csv.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cma_scripts"))

from tasks.causal_analysis.prompt_generators.tom_templates import UnifiedTemplateSystem, UnifiedQuestionFormatter

OUT_DIR = ROOT / "results/rsa_head_outputs/stimuli"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]


def build_for_template(ts: UnifiedTemplateSystem, qf: UnifiedQuestionFormatter,
                       template_name: str, k_per_cell: int, seed: int,
                       context_type: str = "abstract") -> list[dict]:
    rng = random.Random(seed)
    template = ts.get_template(template_name, context_type)
    var_1 = next(v for v in template.variables if v.endswith("_1"))
    var_2 = next(v for v in template.variables if v.endswith("_2"))
    contrastive_vt = template.variables[var_1]

    pool = list(ts.vocab_loader.vocabularies[contrastive_vt])
    rng.shuffle(pool)
    if len(pool) < 2 * k_per_cell:
        raise ValueError(f"vocab {contrastive_vt} has {len(pool)} items, need >= {2 * k_per_cell}")
    pairs = [(pool[2 * i], pool[2 * i + 1]) for i in range(k_per_cell)]

    question_true  = qf.get_cma_questions(template_name, context_type=context_type,
                                          belief_type="true_belief",  question_style="instruction")
    question_false = qf.get_cma_questions(template_name, context_type=context_type,
                                          belief_type="false_belief", question_style="instruction")

    rows = []
    pid = 0
    for pair_idx, (v_a, v_b) in enumerate(pairs):
        non_contrastive = {}
        for var, vt in template.variables.items():
            if var in (var_1, var_2):
                continue
            non_contrastive[var] = ts.vocab_loader.sample(vt)

        subs_orig = {**non_contrastive, var_1: v_a, var_2: v_b}
        subs_swap = {**non_contrastive, var_1: v_b, var_2: v_a}

        for belief, subs, answer in [
            ("false", subs_orig, v_a),
            ("true",  subs_orig, v_b),
            ("false", subs_swap, v_b),
            ("true",  subs_swap, v_a),
        ]:
            tpl = template.false_context_template if belief == "false" else template.true_context_template
            scenario = tpl.format(**subs)
            question = question_false if belief == "false" else question_true
            rows.append({
                "prompt_id": pid,
                "template": template_name,
                "pair_idx": pair_idx,
                "belief_veracity": belief,
                "answer_state": answer,
                "loc_a": v_a,
                "loc_b": v_b,
                "scenario": scenario,
                "question": question,
            })
            pid += 1
    return rows


def audit(rows: list[dict], k_per_cell: int) -> None:
    cells = Counter((r["belief_veracity"], r["answer_state"] == r["loc_a"]) for r in rows)
    # Cells: (true|false, ans_is_loc_a True|False). Each pair contributes 1 to each cell.
    assert len(rows) == 4 * k_per_cell, f"expected {4 * k_per_cell} rows, got {len(rows)}"
    for key, count in cells.items():
        assert count == k_per_cell, f"cell {key} has {count}, expected {k_per_cell}"

    # Decorrelation check: across the K_per_cell pairs, ans=loc_a co-occurs
    # equally with true and false belief (each k_per_cell/... actually each
    # cell has k_per_cell items by construction, so phi=0 exactly).
    n_true_a  = sum(1 for r in rows if r["belief_veracity"] == "true"  and r["answer_state"] == r["loc_a"])
    n_true_b  = sum(1 for r in rows if r["belief_veracity"] == "true"  and r["answer_state"] == r["loc_b"])
    n_false_a = sum(1 for r in rows if r["belief_veracity"] == "false" and r["answer_state"] == r["loc_a"])
    n_false_b = sum(1 for r in rows if r["belief_veracity"] == "false" and r["answer_state"] == r["loc_b"])
    assert n_true_a == n_true_b == n_false_a == n_false_b == k_per_cell, \
        f"unbalanced: {n_true_a, n_true_b, n_false_a, n_false_b}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k_per_cell", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context_type", choices=["abstract", "photo"], default="abstract",
                    help="abstract = belief framing; photo = false-photograph framing")
    ap.add_argument("--out_dir", type=Path, default=None,
                    help="defaults to results/rsa_head_outputs/stimuli{,_photo}/")
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = OUT_DIR if args.context_type == "abstract" else (OUT_DIR.parent / "stimuli_photo")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = UnifiedTemplateSystem(vocab_dir=str(ROOT / "cma_scripts/tom_datasets"))
    qf = UnifiedQuestionFormatter()

    for i, template_name in enumerate(TEMPLATES):
        rows = build_for_template(ts, qf, template_name, args.k_per_cell,
                                  args.seed + i, context_type=args.context_type)
        audit(rows, args.k_per_cell)
        out_path = args.out_dir / f"{template_name}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{template_name}: {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
