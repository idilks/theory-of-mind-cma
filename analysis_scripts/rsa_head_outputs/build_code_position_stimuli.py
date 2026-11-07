#!/usr/bin/env python
"""Build the balanced code-position stimulus grid for the pointer RDM test.

Design (week plan Step 6): copy/alias semantics x queried index x prefix pad
x suffix pad, fully crossed, so answer position varies independently of the
veracity analog — the property the ToM stimuli lack (their veracity<->position
model-RDM correlation is r = 0.996-0.999 within template; see
archive/docs/week_plan_2026-07-23.md Step 5a).

Grid: semantics {copy, alias} x idx {1,2,3,4} x prefix pad {0,16,32} x suffix
pad {0,16,32} x n_rep (default 2) = 144 stimuli. List length fixed at 6 so
prompt geometry never covaries with idx. The pads are inert `pad = [...]` /
`pad2 = [...]` lines around the real code. The copy/alias twins of a setup
share fillers, pads, and (orig, mut) chars, so within a setup ONLY the
semantics line and the resulting answer differ.

The two pads decorrelate three otherwise-collinear geometry variables:
  - absolute slot position  = f(prefix pad) + ~5*idx chars
  - slot -> final-token distance = f(suffix pad, idx)
  - total prompt length     = f(prefix + suffix)
With prefix-only padding (the first version of this design) absolute position,
prompt length, and pad were near-collinear (RDM r ~ 0.99), and relational idx
doubled as distance-to-end; the crossed suffix breaks both degeneracies.

Known, deliberate limitation (reported by the design check): the *answer
value* position is tied to semantics (copy answers live in the `a = [...]`
line, alias answers in the mutation line) — a mini ToM confound. Unlike ToM,
there is large within-semantics position variance, and the queried-slot RDMs
are clean by construction; pointer conclusions should rest on those.

Outputs:
  results/rsa_head_outputs/stimuli_code_position/code_position.csv
  results/rsa_head_outputs/topk_heads/code_position_union.csv  (capture heads)
plus the printed design-check model-RDM correlation matrix. The check FAILS
(exit 1) if semantics correlates with either queried-slot RDM, or if the
position RDMs collapse into each other — either would mean the grid cannot
answer the question it exists for.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codebase"))

from tasks.causal_analysis.prompt_generators.code_templates import (  # noqa: E402
    CODE_VALUE_POOL,
)

OUT_DIR = ROOT / "results/rsa_head_outputs/stimuli_code_position"
HEADS_OUT = ROOT / "results/rsa_head_outputs/topk_heads/code_position_union.csv"
HEAD_SET_DIR = ROOT / "results/aggregated/code_head_sets"

# filter_answer_pool(Qwen2.5 tokenizer) result, hardcoded so this script runs
# without the tokenizer; re-derive with filter_answer_pool if the model changes.
QWEN_ANSWER_POOL = list("ABCDEHILMORST")

LIST_LENGTH = 6
IDX_LEVELS = (1, 2, 3, 4)
PREFIX_PAD_LEVELS = (0, 16, 32)
SUFFIX_PAD_LEVELS = (0, 16, 32)

# Every list element renders as `'C', ` — 5 chars — so slot k's value char sits
# at a fixed offset inside the `a = [...]` line. Asserted per stimulus below.
ELEM_STRIDE = 5


def render_list(values) -> str:
    return "[" + ", ".join(f"'{c}'" for c in values) + "]"


def build_rows(n_rep: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows, pid = [], 0
    for rep, idx, pre, suf in product(range(n_rep), IDX_LEVELS,
                                      PREFIX_PAD_LEVELS, SUFFIX_PAD_LEVELS):
        orig, mut = rng.sample(QWEN_ANSWER_POOL, 2)
        filler_pool = [c for c in CODE_VALUE_POOL if c not in (orig, mut)]
        fillers = [rng.choice(filler_pool) for _ in range(LIST_LENGTH)]
        pre_vals = [rng.choice(filler_pool) for _ in range(pre)]
        suf_vals = [rng.choice(filler_pool) for _ in range(suf)]

        prefix_line = f"pad = {render_list(pre_vals)}\n" if pre else ""
        suffix_line = f"\npad2 = {render_list(suf_vals)}" if suf else ""
        items = list(fillers)
        items[idx] = orig
        a_line = f"a = {render_list(items)}"
        mut_line = f"a[{idx}] = '{mut}'"

        for semantics in ("copy", "alias"):
            b_line = "b = a.copy()" if semantics == "copy" else "b = a"
            scenario = f"{prefix_line}{a_line}\n{b_line}\n{mut_line}{suffix_line}"
            answer = orig if semantics == "copy" else mut

            slot_pos = len(prefix_line) + len("a = [") + ELEM_STRIDE * idx + 1
            assert scenario[slot_pos] == orig, "slot offset formula broke"
            mut_pos = (len(prefix_line) + len(a_line) + 1 + len(b_line) + 1
                       + len(f"a[{idx}] = '"))
            assert scenario[mut_pos] == mut, "mutation offset formula broke"
            ans_pos = slot_pos if semantics == "copy" else mut_pos

            rows.append({
                "prompt_id": pid,
                "template": "code_position",
                "semantics": semantics,
                "idx": idx,
                "pad_len": pre,
                "suffix_pad_len": suf,
                "replicate": rep,
                "orig_char": orig,
                "mut_char": mut,
                "answer": f"'{answer}'",
                "slot_char_pos": slot_pos,
                "ans_char_pos": ans_pos,
                "scenario_len": len(scenario),
                "slot_dist_to_end": len(scenario) - slot_pos,
                "scenario": scenario,
                "question": f"What is b[{idx}]?",
            })
            pid += 1
    return rows


def audit(rows: list[dict], n_rep: int) -> None:
    n_cells = 2 * len(IDX_LEVELS) * len(PREFIX_PAD_LEVELS) * len(SUFFIX_PAD_LEVELS)
    assert len(rows) == n_cells * n_rep, f"expected {n_cells * n_rep}, got {len(rows)}"
    cells = Counter((r["semantics"], r["idx"], r["pad_len"], r["suffix_pad_len"])
                    for r in rows)
    assert all(v == n_rep for v in cells.values()), f"unbalanced cells: {cells}"
    for r in rows:
        # answer chars appear exactly where intended, nowhere else
        assert r["scenario"].count(r["orig_char"]) == 1
        assert r["scenario"].count(r["mut_char"]) == 1
        marker = "b = a.copy()" if r["semantics"] == "copy" else "b = a\n"
        assert marker in r["scenario"]


# ---------------------------------------------------------------------------
# Design check: model-RDM separability (the thing 5a failed for ToM stimuli)
# ---------------------------------------------------------------------------

def try_token_positions(rows: list[dict]):
    """Char->token positions via the Qwen tokenizer if available, else None."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
    except Exception as e:
        print(f"[design-check] tokenizer unavailable ({type(e).__name__}); "
              "using char positions as proxy (monotone, fine for this check)")
        return None
    slot_tok, ans_tok, n_tok = [], [], []
    for r in rows:
        enc = tok(r["scenario"], add_special_tokens=False, return_offsets_mapping=True)
        def tok_at(char_pos):
            for i, (s, e) in enumerate(enc["offset_mapping"]):
                if s <= char_pos < e:
                    return i
            return len(enc["offset_mapping"]) - 1
        slot_tok.append(tok_at(r["slot_char_pos"]))
        ans_tok.append(tok_at(r["ans_char_pos"]))
        n_tok.append(len(enc["offset_mapping"]))
    return (np.array(slot_tok, float), np.array(ans_tok, float),
            np.array(n_tok, float))


def design_check(rows: list[dict]) -> None:

    slot_pos = np.array([r["slot_char_pos"] for r in rows], float)
    ans_pos = np.array([r["ans_char_pos"] for r in rows], float)
    length = np.array([r["scenario_len"] for r in rows], float)
    tok = try_token_positions(rows)
    if tok is not None:
        slot_pos, ans_pos, length = tok
        unit = "tokens"
    else:
        unit = "chars"
    dist_end = length - slot_pos

    sem = np.array([1 if r["semantics"] == "copy" else 0 for r in rows], float)
    idx = np.array([r["idx"] for r in rows], float)
    ans_id = np.array([ord(r["answer"][1]) for r in rows], float)

    models = {
        "semantics": sem[:, None] != sem[None, :],
        "queried_slot_rel": np.abs(idx[:, None] - idx[None, :]),
        "queried_slot_abs": np.abs(slot_pos[:, None] - slot_pos[None, :]),
        "slot_dist_to_end": np.abs(dist_end[:, None] - dist_end[None, :]),
        "answer_value_pos": np.abs(ans_pos[:, None] - ans_pos[None, :]),
        "answer_identity": ans_id[:, None] != ans_id[None, :],
        "prompt_length": np.abs(length[:, None] - length[None, :]),
    }
    n = len(rows)
    iu = np.triu_indices(n, 1)
    vecs = {k: v[iu].astype(float) for k, v in models.items()}

    names = list(vecs)
    print(f"\n[design-check] model-RDM correlations over {n} stimuli "
          f"(positions in {unit}); pearson below diagonal, spearman above:")
    width = max(len(s) for s in names)
    print(" " * (width + 2) + "  ".join(f"{s[:12]:>12s}" for s in names))
    for i, a in enumerate(names):
        cells = []
        for j, b in enumerate(names):
            if i == j:
                cells.append(f"{'--':>12s}")
            elif i > j:
                cells.append(f"{np.corrcoef(vecs[a], vecs[b])[0, 1]:>12.3f}")
            else:
                cells.append(f"{spearmanr(vecs[a], vecs[b]).statistic:>12.3f}")
        print(f"{a:>{width}s}  " + "  ".join(cells))

    def r_of(a, b):
        return np.corrcoef(vecs[a], vecs[b])[0, 1]

    r_sem_rel = r_of("semantics", "queried_slot_rel")
    r_sem_abs = r_of("semantics", "queried_slot_abs")
    r_rel_abs = r_of("queried_slot_rel", "queried_slot_abs")
    r_abs_len = r_of("queried_slot_abs", "prompt_length")
    r_rel_dist = r_of("queried_slot_rel", "slot_dist_to_end")
    r_abs_dist = r_of("queried_slot_abs", "slot_dist_to_end")
    r_sem_anspos = r_of("semantics", "answer_value_pos")

    copy_std = ans_pos[sem == 1].std()
    alias_std = ans_pos[sem == 0].std()
    print(f"\n[design-check] verdicts:")
    print(f"  semantics vs queried_slot_rel  r = {r_sem_rel:+.3f}  (must be ~0)")
    print(f"  semantics vs queried_slot_abs  r = {r_sem_abs:+.3f}  (must be ~0)")
    print(f"  queried_slot_rel vs _abs       r = {r_rel_abs:+.3f}  (low = separable)")
    print(f"  queried_slot_abs vs length     r = {r_abs_len:+.3f}  "
          "(suffix pad exists to keep this off ~1)")
    print(f"  queried_slot_rel vs dist_to_end r = {r_rel_dist:+.3f}  "
          "(suffix pad exists to keep this off ~1)")
    print(f"  queried_slot_abs vs dist_to_end r = {r_abs_dist:+.3f}")
    print(f"  semantics vs answer_value_pos  r = {r_sem_anspos:+.3f}  "
          "(known limitation: alias answers sit in the mutation line; "
          f"within-semantics position std = {copy_std:.1f} copy / "
          f"{alias_std:.1f} alias {unit} — nonzero, unlike ToM)")

    fails = []
    if abs(r_sem_rel) > 0.05 or abs(r_sem_abs) > 0.05:
        fails.append("semantics is entangled with the queried-slot RDMs "
                     "(the 5a confound, rebuilt)")
    if r_rel_abs > 0.9:
        fails.append("relational and absolute position RDMs are near-identical")
    if r_abs_len > 0.95:
        fails.append("absolute position is indistinguishable from prompt length")
    if r_rel_dist > 0.95:
        fails.append("relational idx is indistinguishable from distance-to-end")
    if fails:
        for f in fails:
            print(f"[design-check] FAIL: {f}")
        sys.exit(1)
    print("[design-check] PASS: pointer RDMs are separable from the veracity "
          "analog and from each other.")


# ---------------------------------------------------------------------------
# Capture head set: union of code-arm + T_C1/T_C5 heads
# ---------------------------------------------------------------------------

def build_union_heads_csv(out_path: Path) -> None:
    sources = {
        "code_abstract": HEAD_SET_DIR / "code_abstract_head_list.csv",
        "code_control": HEAD_SET_DIR / "code_control_head_list.csv",
    }
    frames = []
    for name, p in sources.items():
        df = pd.read_csv(p)[["layer", "head"]]
        df["source"] = name
        frames.append(df)
    tom = pd.read_csv(HEAD_SET_DIR / "tom_c1_c5_heads.csv")
    tom = tom.rename(columns={"condition": "source"})[["layer", "head", "source"]]
    frames.append(tom)

    allh = pd.concat(frames, ignore_index=True)
    union = (allh.groupby(["layer", "head"])["source"]
             .apply(lambda s: "|".join(sorted(set(s)))).reset_index()
             .rename(columns={"source": "sources"})
             .sort_values(["layer", "head"]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    union.to_csv(out_path, index=False)
    print(f"[heads] {len(union)} unique heads "
          f"({', '.join(f'{n}={len(f)}' for n, f in zip(list(sources) + ['tom_c1+c5'], frames))}) "
          f"-> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=2,
                    help="replicates per (semantics, idx, prefix, suffix) cell; "
                         "2 -> 144 stimuli")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip_heads_csv", action="store_true",
                    help="skip building the union capture head set")
    args = ap.parse_args()

    rows = build_rows(args.n_rep, args.seed)
    audit(rows, args.n_rep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "code_position.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"code_position: {len(rows)} stimuli -> {out_path}")

    if not args.skip_heads_csv:
        build_union_heads_csv(HEADS_OUT)

    design_check(rows)


if __name__ == "__main__":
    main()
