#!/usr/bin/env python
"""Build the divergence-ledger stimulus grid (Exp 2 of the code RDM arm).

Every prompt contains
BOTH a copy and an alias of `a`, so a divergence always exists in context;
what varies is what the question asks about. This separates a global
divergence *alarm* ("something diverged") from a content-aware *ledger*
("the object/value you asked about diverged"), and the crossed syntax
variants test whether the Exp-1 semantics fit was surface wording.

Grid (2 x 2 x 2 x 3 x 2 x n_rep, default n_rep=3 -> 144 stimuli):
  queried object   copy | alias
  queried slot     mutated (m) | untouched (u)
  role assignment  b=copy,c=alias | b=alias,c=copy   (fixed line order: b then c)
  copy syntax      a.copy() | a[:] | list(a)
  alias syntax     plain (`c = a`) | chained (`t = a` then `c = t`)
Nuisance (randomized per setup, recorded): mutated slot m, untouched slot u,
prefix/suffix pad lengths in {0,16,32}, filler chars.

Answer chars come from the intersection of filter_answer_pool over the
requested tokenizers (Qwen + Gemma by default) so the same stimuli serve both
models; falls back to the documented Qwen pool if no tokenizer loads.

Outputs:
  results/rsa_head_outputs/stimuli_code_ledger/code_ledger.csv
plus a per-tokenizer design check. Hard fails (exit 1): either divergence RDM
entangled with either syntax RDM or with the queried variable name.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "cma_scripts"))

from tasks.causal_analysis.prompt_generators.code_templates import (  # noqa: E402
    CODE_VALUE_POOL, filter_answer_pool,
)

OUT_DIR = ROOT / "results/rsa_head_outputs/stimuli_code_ledger"

# Fallback = filter_answer_pool(Qwen2.5 tokenizer), documented in README.md.
QWEN_ANSWER_POOL = list("ABCDEHILMORST")

LIST_LENGTH = 6
PAD_CHOICES = (0, 16, 32)
QUERIED_OBJECTS = ("copy", "alias")
SLOT_TYPES = ("mutated", "untouched")
ROLE_B = ("copy", "alias")            # semantics of variable b (c gets the other)
COPY_SYNTAXES = ("method", "slice", "constructor")
ALIAS_SYNTAXES = ("plain", "chained")

COPY_RENDER = {
    "method": "{name} = a.copy()",
    "slice": "{name} = a[:]",
    "constructor": "{name} = list(a)",
}


def render_list(values) -> str:
    return "[" + ", ".join(f"'{c}'" for c in values) + "]"


def alias_render(name: str, syntax: str) -> str:
    if syntax == "plain":
        return f"{name} = a"
    return f"t = a\n{name} = t"


def resolve_answer_pool(tokenizer_names: list[str]) -> list[str]:
    pools = []
    for tn in tokenizer_names:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tn)
            pool = filter_answer_pool(tok)
            pools.append(set(pool))
            print(f"[answer-pool] {tn}: {len(pool)} chars ({''.join(sorted(pool))})")
        except Exception as e:
            print(f"[answer-pool] {tn} unavailable ({type(e).__name__}) — skipped")
    if not pools:
        print(f"[answer-pool] no tokenizer loaded; falling back to documented "
              f"Qwen pool ({''.join(QWEN_ANSWER_POOL)}) — verify Gemma overlap on cluster")
        return list(QWEN_ANSWER_POOL)
    inter = sorted(set.intersection(*pools))
    if len(inter) < 3:
        raise ValueError(f"answer-pool intersection too small: {inter}")
    print(f"[answer-pool] intersection: {len(inter)} chars ({''.join(inter)})")
    return inter


def build_rows(n_rep: int, seed: int, answer_pool: list[str]) -> list[dict]:
    rng = random.Random(seed)
    rows, pid = [], 0
    for rep, q_obj, slot_type, role_b, c_syn, a_syn in product(
            range(n_rep), QUERIED_OBJECTS, SLOT_TYPES, ROLE_B,
            COPY_SYNTAXES, ALIAS_SYNTAXES):
        m, u = rng.sample(range(LIST_LENGTH), 2)
        orig_m, orig_u, x_mut = rng.sample(answer_pool, 3)
        filler_pool = [c for c in CODE_VALUE_POOL
                       if c not in (orig_m, orig_u, x_mut)]
        fillers = [rng.choice(filler_pool) for _ in range(LIST_LENGTH)]
        pre = rng.choice(PAD_CHOICES)
        suf = rng.choice(PAD_CHOICES)
        pre_vals = [rng.choice(filler_pool) for _ in range(pre)]
        suf_vals = [rng.choice(filler_pool) for _ in range(suf)]

        items = list(fillers)
        items[m] = orig_m
        items[u] = orig_u
        prefix_line = f"pad = {render_list(pre_vals)}\n" if pre else ""
        suffix_line = f"\npad2 = {render_list(suf_vals)}" if suf else ""
        a_line = f"a = {render_list(items)}"

        role_c = "alias" if role_b == "copy" else "copy"
        def def_lines(name, sem):
            if sem == "copy":
                return COPY_RENDER[c_syn].format(name=name)
            return alias_render(name, a_syn)
        b_def, c_def = def_lines("b", role_b), def_lines("c", role_c)
        mut_line = f"a[{m}] = '{x_mut}'"
        scenario = (f"{prefix_line}{a_line}\n{b_def}\n{c_def}\n"
                    f"{mut_line}{suffix_line}")

        q_name = "b" if role_b == q_obj else "c"
        q = m if slot_type == "mutated" else u
        if q_obj == "copy":
            answer = orig_m if slot_type == "mutated" else orig_u
        else:
            answer = x_mut if slot_type == "mutated" else orig_u

        slot_pos = len(prefix_line) + len("a = [") + 5 * q + 1
        assert scenario[slot_pos] == (orig_m if q == m else orig_u), \
            "slot offset formula broke"
        mut_pos = scenario.rindex(f"'{x_mut}'") + 1
        assert scenario[mut_pos] == x_mut

        rows.append({
            "prompt_id": pid,
            "template": "code_ledger",
            "queried_object": q_obj,
            "queried_slot_type": slot_type,
            "queried_name": q_name,
            "role_b": role_b,
            "copy_syntax": c_syn,
            "alias_syntax": a_syn,
            "replicate": rep,
            "m": m, "u": u, "q": q,
            "pad_len": pre, "suffix_pad_len": suf,
            "orig_m": orig_m, "orig_u": orig_u, "mut_char": x_mut,
            "answer": f"'{answer}'",
            "slot_char_pos": slot_pos,
            "mut_char_pos": mut_pos,
            "scenario_len": len(scenario),
            "scenario": scenario,
            "question": f"What is {q_name}[{q}]?",
        })
        pid += 1
    return rows


def audit(rows: list[dict], n_rep: int) -> None:
    n_cells = (len(QUERIED_OBJECTS) * len(SLOT_TYPES) * len(ROLE_B)
               * len(COPY_SYNTAXES) * len(ALIAS_SYNTAXES))
    assert len(rows) == n_cells * n_rep, f"expected {n_cells * n_rep}, got {len(rows)}"
    cells = Counter((r["queried_object"], r["queried_slot_type"], r["role_b"],
                     r["copy_syntax"], r["alias_syntax"]) for r in rows)
    assert all(v == n_rep for v in cells.values()), "unbalanced cells"
    for r in rows:
        for ch in (r["orig_m"], r["orig_u"], r["mut_char"]):
            assert r["scenario"].count(ch) == 1, \
                f"char {ch} not unique in stimulus {r['prompt_id']}"
        # exactly one copy line and one alias mechanism present
        n_copy = sum(s in r["scenario"] for s in (".copy()", "a[:]", "list(a)"))
        assert n_copy == 1, f"stimulus {r['prompt_id']}: {n_copy} copy forms"


# ---------------------------------------------------------------------------
# Design check
# ---------------------------------------------------------------------------

def token_positions(rows, tokenizer_name):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"[design-check] {tokenizer_name} unavailable ({type(e).__name__})")
        return None
    slot, length = [], []
    for r in rows:
        enc = tok(r["scenario"], add_special_tokens=False, return_offsets_mapping=True)
        pos = next((i for i, (s, e) in enumerate(enc["offset_mapping"])
                    if s <= r["slot_char_pos"] < e), len(enc["offset_mapping"]) - 1)
        slot.append(pos)
        length.append(len(enc["offset_mapping"]))
    return np.array(slot, float), np.array(length, float)


def design_check(rows: list[dict], tokenizer_names: list[str]) -> None:

    cat = lambda key: np.array([hash(r[key]) for r in rows], float)  # noqa: E731
    slot_div = np.array([1.0 if (r["queried_object"] == "copy"
                                 and r["queried_slot_type"] == "mutated") else 0.0
                         for r in rows])
    base_models = {
        "slot_divergence": slot_div,
        "object_divergence": cat("queried_object"),
        "queried_slot_type": cat("queried_slot_type"),
        "queried_name": cat("queried_name"),
        "copy_syntax": cat("copy_syntax"),
        "alias_syntax": cat("alias_syntax"),
        "answer_identity": cat("answer"),
    }

    pos_sources = [(tn, token_positions(rows, tn)) for tn in tokenizer_names]
    pos_sources = [(tn, p) for tn, p in pos_sources if p is not None]
    if not pos_sources:
        print("[design-check] no tokenizer loaded; using char positions")
        pos_sources = [("chars", (np.array([r["slot_char_pos"] for r in rows], float),
                                  np.array([r["scenario_len"] for r in rows], float)))]

    any_fail = False
    for tn, (slot_pos, length) in pos_sources:
        models = dict(base_models)
        models["queried_slot_abs"] = slot_pos
        models["prompt_length"] = length
        rdms = {k: (np.abs(v[:, None] - v[None, :]) if k in
                    ("queried_slot_abs", "prompt_length")
                    else (v[:, None] != v[None, :]).astype(float))
                for k, v in models.items()}
        iu = np.triu_indices(len(rows), 1)
        vecs = {k: v[iu] for k, v in rdms.items()}
        names = list(vecs)

        print(f"\n[design-check] ({tn}) model-RDM pearson r:")
        width = max(len(s) for s in names)
        print(" " * (width + 2) + "  ".join(f"{s[:11]:>11s}" for s in names))
        for i, a in enumerate(names):
            cells = []
            for j, b in enumerate(names):
                cells.append(f"{'--':>11s}" if i == j else
                             f"{np.corrcoef(vecs[a], vecs[b])[0, 1]:>11.3f}")
            print(f"{a:>{width}s}  " + "  ".join(cells))

        r_of = lambda a, b: np.corrcoef(vecs[a], vecs[b])[0, 1]  # noqa: E731
        checks = [
            ("slot_divergence vs copy_syntax", r_of("slot_divergence", "copy_syntax")),
            ("slot_divergence vs alias_syntax", r_of("slot_divergence", "alias_syntax")),
            ("object_divergence vs copy_syntax", r_of("object_divergence", "copy_syntax")),
            ("object_divergence vs alias_syntax", r_of("object_divergence", "alias_syntax")),
            ("object_divergence vs queried_name", r_of("object_divergence", "queried_name")),
        ]
        print("\n[design-check] verdicts:")
        for label, r in checks:
            status = "ok" if abs(r) <= 0.05 else "FAIL"
            print(f"  {label:38s} r = {r:+.3f}  [{status}]")
            if abs(r) > 0.05:
                any_fail = True
        print(f"  slot_ vs object_divergence (nested)   r = "
              f"{r_of('slot_divergence', 'object_divergence'):+.3f}  "
              "[expected high; adjudicate via partials, see design doc]")

    if any_fail:
        print("[design-check] FAIL — grid is confounded, do not capture.")
        sys.exit(1)
    print("\n[design-check] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rep", type=int, default=3,
                    help="replicates per cell; 3 -> 144 stimuli")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--tokenizers", default="Qwen/Qwen2.5-14B-Instruct,google/gemma-2-27b-it",
                    help="comma-separated; answer pool = intersection of their "
                         "filter_answer_pool results")
    args = ap.parse_args()

    tokenizer_names = [t.strip() for t in args.tokenizers.split(",") if t.strip()]
    answer_pool = resolve_answer_pool(tokenizer_names)
    rows = build_rows(args.n_rep, args.seed, answer_pool)
    audit(rows, args.n_rep)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "code_ledger.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"code_ledger: {len(rows)} stimuli -> {out_path}")

    design_check(rows, tokenizer_names)


if __name__ == "__main__":
    main()
