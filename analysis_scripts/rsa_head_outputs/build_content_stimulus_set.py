#!/usr/bin/env python
"""Build the CONTENT-separable ToM stimulus set for RSA / MVPA.

Sept 2026, Sept 3 meeting: the neuroscience-proxy test (Bio et al. / Taylor)
needs believed-location IDENTITY decorrelated from veracity and from scenario.
In `build_stimulus_set.py` every (loc_a, loc_b) pair is drawn fresh, so each
location name occurs in exactly one scenario pair and "same believed location"
equals "same scenario". Here a FIXED vocabulary of V locations is reused
across all scenarios:

  per template:  all V*(V-1) ordered pairs (loc_a, loc_b)
               x {false, true} belief
               x R replicate draws of the non-contrastive substitutions
  = V(V-1) * 2 * R stimuli   (V=6, R=2 -> 120 per template, 360 total)

Believed location = loc_a under false belief, loc_b under true belief, so
every location is believed in (V-1)*R false-belief and (V-1)*R true-belief
prompts, each time in a different scenario. Content is therefore a V-way
target that is balanced against veracity and repeats across scenarios.

Columns are a superset of the original stimulus CSV, so
`capture_head_outputs_full.py` runs unchanged:
  prompt_id, template, pair_idx, belief_veracity, answer_state, loc_a, loc_b,
  scenario, question                       (original)
  believed, world, replicate, believed_idx, world_idx   (new)

`answer_state` == `believed` (the question asks where the AGENT would go).
`world` is the truck's actual location (loc_b in both veracities), kept so a
world-content model RDM can be tested alongside the belief-content one.

Output: results/rsa_head_outputs/stimuli_content/{template}.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cma_scripts"))

from tasks.causal_analysis.prompt_generators.tom_templates import (  # noqa: E402
    UnifiedTemplateSystem, UnifiedQuestionFormatter)

OUT_DIR = ROOT / "results/rsa_head_outputs/stimuli_content"
TEMPLATES = ["food_truck", "hair_styling", "library_book"]


def build_for_template(ts: UnifiedTemplateSystem, qf: UnifiedQuestionFormatter,
                       template_name: str, n_vocab: int, n_rep: int, seed: int,
                       context_type: str = "abstract") -> list[dict]:
    rng = random.Random(seed)
    # vocab_loader.sample() draws from the GLOBAL random module, so seed it too
    # or the non-contrastive fillers change on every rebuild (found Sept 8:
    # the cluster and local rebuilds disagreed in 106/120 scenarios).
    random.seed(seed)
    template = ts.get_template(template_name, context_type)
    var_1 = next(v for v in template.variables if v.endswith("_1"))
    var_2 = next(v for v in template.variables if v.endswith("_2"))
    pool = list(ts.vocab_loader.vocabularies[template.variables[var_1]])
    rng.shuffle(pool)
    vocab = pool[:n_vocab]
    assert len(vocab) == n_vocab, f"{template_name}: vocab too small for V={n_vocab}"

    q_true = qf.get_cma_questions(template_name, context_type=context_type,
                                  belief_type="true_belief", question_style="instruction")
    q_false = qf.get_cma_questions(template_name, context_type=context_type,
                                   belief_type="false_belief", question_style="instruction")

    rows, pid = [], 0
    for pair_idx, (v_a, v_b) in enumerate(permutations(vocab, 2)):
        for rep in range(n_rep):
            non_contrastive = {var: ts.vocab_loader.sample(vt)
                               for var, vt in template.variables.items()
                               if var not in (var_1, var_2)}
            subs = {**non_contrastive, var_1: v_a, var_2: v_b}
            for belief in ("false", "true"):
                tpl = (template.false_context_template if belief == "false"
                       else template.true_context_template)
                believed = v_a if belief == "false" else v_b
                rows.append({
                    "prompt_id": pid,
                    "template": template_name,
                    "pair_idx": pair_idx,
                    "belief_veracity": belief,
                    "answer_state": believed,
                    "loc_a": v_a,
                    "loc_b": v_b,
                    "scenario": tpl.format(**subs),
                    "question": q_false if belief == "false" else q_true,
                    "believed": believed,
                    "world": v_b,
                    "replicate": rep,
                    "believed_idx": vocab.index(believed),
                    "world_idx": vocab.index(v_b),
                })
                pid += 1
    return rows


def audit(rows: list[dict], n_vocab: int, n_rep: int) -> None:
    n_expected = n_vocab * (n_vocab - 1) * 2 * n_rep
    assert len(rows) == n_expected, f"expected {n_expected} rows, got {len(rows)}"
    per = (n_vocab - 1) * n_rep
    cells = Counter((r["belief_veracity"], r["believed"]) for r in rows)
    assert len(cells) == 2 * n_vocab, cells
    assert all(c == per for c in cells.values()), cells
    # believed and world never coincide under false belief, always under true
    assert all((r["believed"] == r["world"]) == (r["belief_veracity"] == "true")
               for r in rows)
    # answer_state kept equal to believed for downstream compatibility
    assert all(r["answer_state"] == r["believed"] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_vocab", type=int, default=6)
    ap.add_argument("--n_rep", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context_type", choices=["abstract", "photo"], default="abstract")
    ap.add_argument("--out_dir", type=Path, default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = OUT_DIR if args.context_type == "abstract" else (
            OUT_DIR.parent / "stimuli_content_photo")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = UnifiedTemplateSystem(vocab_dir=str(ROOT / "cma_scripts/tom_datasets"))
    qf = UnifiedQuestionFormatter()

    for i, template_name in enumerate(TEMPLATES):
        rows = build_for_template(ts, qf, template_name, args.n_vocab, args.n_rep,
                                  args.seed + i, context_type=args.context_type)
        audit(rows, args.n_vocab, args.n_rep)
        out_path = args.out_dir / f"{template_name}.csv"
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        vocab = sorted({r["believed"] for r in rows})
        print(f"{template_name}: {len(rows)} rows, vocab {vocab} -> {out_path}")


if __name__ == "__main__":
    main()
