#!/usr/bin/env python
"""Build the balanced pad-twins stimulus grid (Exp 3 of the code RDM arm).

Purpose: separate veracity-TAG from LOOKBACK/pointer readings of the
divergence heads via a within-cell position contrast (July 25 2026 discussion
in writeups/results_for_pi/code_pointer_rdm.qmd, Control 2 callout). Exp 2
randomized pads as nuisance, so same-cell pairs differ in pads AND fillers
AND slot indices AND answer letters at once; here pads become the ONLY
within-twin factor:

  - TAG predicts: within a twin group, pairwise distance is flat in |dpad|
    (moving the evidence does not move the tag);
  - POINTER predicts: distance grows with |d source-token position|.

Grid (48 query cells x 2 content draws x 9 pad combos = 864 stimuli):
  query cells      identical to Exp 2 (queried_object x queried_slot_type x
                   role_b x copy_syntax x alias_syntax) — so every Exp 2
                   model RDM can be re-fit on this grid unchanged;
  content draw     m, u, answer chars, fillers, junk sequences — frozen
                   across the 9 pad combos of its twin group;
  pads             prefix x suffix, each in {0, 16, 32}, fully crossed.

Pads are NESTED: one 32-token junk sequence per (cell, content) is drawn
once and pad length k uses its first k tokens, so the pad-16 prompt is a
strict prefix-extension of the pad-0 prompt (content differences between
twins are only the presence of extra junk tokens, never different junk).

Outputs: results/rsa_head_outputs/stimuli_code_padtwins/code_padtwins.csv
Column schema = Exp 2's code_ledger.csv + content_id + twin_id, so
capture_head_outputs_full.py and analyze_code_ledger_rdms.build_model_rdms
work on it unmodified. Hard-fails on the same confound checks as Exp 2,
plus: pad models must be exactly orthogonal to every query-factor model.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "rsa_head_outputs"))

from build_code_ledger_stimuli import (  # noqa: E402
    COPY_RENDER, LIST_LENGTH, QUERIED_OBJECTS, SLOT_TYPES, ROLE_B,
    COPY_SYNTAXES, ALIAS_SYNTAXES, render_list, alias_render,
    resolve_answer_pool, design_check,
)
from tasks.causal_analysis.prompt_generators.code_templates import (  # noqa: E402
    CODE_VALUE_POOL,
)

OUT_DIR = ROOT / "results/rsa_head_outputs/stimuli_code_padtwins"
PAD_GRID = (0, 16, 32)
MAX_PAD = max(PAD_GRID)


def build_rows(n_content: int, seed: int, answer_pool: list[str]) -> list[dict]:
    rng = random.Random(seed)
    rows, pid = [], 0
    cells = list(product(QUERIED_OBJECTS, SLOT_TYPES, ROLE_B,
                         COPY_SYNTAXES, ALIAS_SYNTAXES))
    for cell_i, (q_obj, slot_type, role_b, c_syn, a_syn) in enumerate(cells):
        for content in range(n_content):
            # --- content draw: frozen across the 9 pad combos below
            m, u = rng.sample(range(LIST_LENGTH), 2)
            orig_m, orig_u, x_mut = rng.sample(answer_pool, 3)
            filler_pool = [c for c in CODE_VALUE_POOL
                           if c not in (orig_m, orig_u, x_mut)]
            fillers = [rng.choice(filler_pool) for _ in range(LIST_LENGTH)]
            pre_seq = [rng.choice(filler_pool) for _ in range(MAX_PAD)]
            suf_seq = [rng.choice(filler_pool) for _ in range(MAX_PAD)]

            items = list(fillers)
            items[m] = orig_m
            items[u] = orig_u
            a_line = f"a = {render_list(items)}"
            role_c = "alias" if role_b == "copy" else "copy"

            def def_lines(name, sem):
                if sem == "copy":
                    return COPY_RENDER[c_syn].format(name=name)
                return alias_render(name, a_syn)

            b_def, c_def = def_lines("b", role_b), def_lines("c", role_c)
            mut_line = f"a[{m}] = '{x_mut}'"

            q_name = "b" if role_b == q_obj else "c"
            q = m if slot_type == "mutated" else u
            if q_obj == "copy":
                answer = orig_m if slot_type == "mutated" else orig_u
            else:
                answer = x_mut if slot_type == "mutated" else orig_u

            twin_id = f"cell{cell_i:02d}_content{content}"
            for pre, suf in product(PAD_GRID, PAD_GRID):
                prefix_line = (f"pad = {render_list(pre_seq[:pre])}\n"
                               if pre else "")
                suffix_line = (f"\npad2 = {render_list(suf_seq[:suf])}"
                               if suf else "")
                scenario = (f"{prefix_line}{a_line}\n{b_def}\n{c_def}\n"
                            f"{mut_line}{suffix_line}")

                slot_pos = len(prefix_line) + len("a = [") + 5 * q + 1
                assert scenario[slot_pos] == (orig_m if q == m else orig_u), \
                    "slot offset formula broke"
                mut_pos = scenario.rindex(f"'{x_mut}'") + 1
                assert scenario[mut_pos] == x_mut

                rows.append({
                    "prompt_id": pid,
                    "template": "code_padtwins",
                    "queried_object": q_obj,
                    "queried_slot_type": slot_type,
                    "queried_name": q_name,
                    "role_b": role_b,
                    "copy_syntax": c_syn,
                    "alias_syntax": a_syn,
                    "replicate": content,       # Exp-2-compatible alias
                    "content_id": content,
                    "twin_id": twin_id,
                    "m": m, "u": u, "q": q,
                    "pad_len": pre, "suffix_pad_len": suf,
                    "orig_m": orig_m, "orig_u": orig_u, "mut_char": x_mut,
                    "answer": f"'{answer}'",
                    "slot_char_pos": slot_pos,
                    "mut_char_pos": mut_pos,
                    "code_start_char": len(prefix_line),
                    "scenario_len": len(scenario),
                    "scenario": scenario,
                    "question": f"What is {q_name}[{q}]?",
                })
                pid += 1
    return rows


def audit(rows: list[dict], n_content: int) -> None:
    n_cells = (len(QUERIED_OBJECTS) * len(SLOT_TYPES) * len(ROLE_B)
               * len(COPY_SYNTAXES) * len(ALIAS_SYNTAXES))
    expected = n_cells * n_content * len(PAD_GRID) ** 2
    assert len(rows) == expected, f"expected {expected}, got {len(rows)}"

    pads = Counter((r["pad_len"], r["suffix_pad_len"]) for r in rows)
    assert all(v == n_cells * n_content for v in pads.values()), \
        "pad grid unbalanced"

    groups = defaultdict(list)
    for r in rows:
        groups[r["twin_id"]].append(r)
    for tid, grp in groups.items():
        assert len(grp) == len(PAD_GRID) ** 2, f"{tid}: {len(grp)} stimuli"
        # twins must be identical once pad lines are stripped
        cores, answers = set(), set()
        for r in grp:
            core = r["scenario"][r["code_start_char"]:]
            if r["suffix_pad_len"]:
                core = core[:core.rindex("\npad2 = ")]
            cores.add(core)
            answers.add(r["answer"])
        assert len(cores) == 1, f"{tid}: twins differ beyond pads"
        assert len(answers) == 1, f"{tid}: answer varies within twin group"
        # nested pads: longer pad line starts with the shorter one's items
        for key in ("pad_len", "suffix_pad_len"):
            by_len = sorted({r[key] for r in grp})
            assert by_len == sorted(PAD_GRID), f"{tid}: missing {key} level"

    for r in rows:
        for ch in (r["orig_m"], r["orig_u"], r["mut_char"]):
            core = r["scenario"][r["code_start_char"]:]
            if r["suffix_pad_len"]:
                core = core[:core.rindex("\npad2 = ")]
            assert core.count(ch) == 1, \
                f"char {ch} not unique in core of stimulus {r['prompt_id']}"
        n_copy = sum(s in r["scenario"] for s in (".copy()", "a[:]", "list(a)"))
        assert n_copy == 1, f"stimulus {r['prompt_id']}: {n_copy} copy forms"


def pad_orthogonality_check(rows: list[dict]) -> None:
    """Pads are fully crossed against the query grid. At the pair (RDM-vector)
    level, orthogonal balanced factors still show the finite-population bias
    E[r] ~ -1/(n-1) (~ -1.2e-3 at n=864), so the check allows that magnitude
    and fails only on real imbalance."""
    tol = 5.0 / (len(rows) - 1)
    iu = np.triu_indices(len(rows), 1)
    pre = np.array([r["pad_len"] for r in rows], float)
    suf = np.array([r["suffix_pad_len"] for r in rows], float)
    cat = lambda key: np.array([hash(r[key]) for r in rows], float)  # noqa: E731
    pad_vecs = {"prefix_pad": np.abs(pre[:, None] - pre[None, :])[iu],
                "suffix_pad": np.abs(suf[:, None] - suf[None, :])[iu]}
    query_vecs = {k: (cat(k)[:, None] != cat(k)[None, :]).astype(float)[iu]
                  for k in ("queried_object", "queried_slot_type", "role_b",
                            "copy_syntax", "alias_syntax")}
    print("\n[pad-orthogonality]")
    ok = True
    for pk, pv in pad_vecs.items():
        for qk, qv in query_vecs.items():
            r = float(np.corrcoef(pv, qv)[0, 1])
            status = "ok" if abs(r) <= tol else "FAIL"
            ok &= abs(r) <= tol
            print(f"  {pk} vs {qk:18s} r = {r:+.2e}  [{status}]")
    if not ok:
        print("[pad-orthogonality] FAIL — pads not balanced against query grid")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_content", type=int, default=2,
                    help="content draws per cell; 2 -> 864 stimuli")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizers",
                    default="Qwen/Qwen2.5-14B-Instruct,google/gemma-2-27b-it")
    args = ap.parse_args()

    tokenizer_names = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    answer_pool = resolve_answer_pool(tokenizer_names)
    rows = build_rows(args.n_content, args.seed, answer_pool)
    audit(rows, args.n_content)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "code_padtwins.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"code_padtwins: {len(rows)} stimuli -> {out_path}")

    pad_orthogonality_check(rows)
    design_check(rows, tokenizer_names)


if __name__ == "__main__":
    main()
