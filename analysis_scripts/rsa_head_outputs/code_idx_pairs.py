"""Idx-contrast prompt pairs for the code-position grid.

Shared generator for cross_patch_experiment.py and scoped_idx_cma.py.

A pair is two prompts identical in list contents, pads, semantics, and
mutation character — differing ONLY in which index is mutated and queried
(idx_base vs idx_exp). This is the address-axis analog of the existing
code_abstract/code_answer_changes pairs, which contrast the mechanism axis.

Answer logic (list A, mutation char M at the queried index):
  copy : b = a.copy() -> answer is A[idx].     Base and exp answers DIFFER
         (answer-changes frame, like C3/C5).
  alias: b = a        -> answer is M for both. Base and exp answers are the
         SAME (same-answer frame, like C1) — but the causal flip target of an
         address patch is A[idx_base] read out of the EXP context, since the
         exp mutation sits at idx_exp.

In both frames the predicted readout of a successful address patch
(base -> exp) is exp's list value at idx_base:
  copy : model fetches slot idx_base from b (= a.copy()) -> A[idx_base]
  alias: model computes b[idx_base] = a[idx_base]; mutation was at idx_exp
         -> A[idx_base]
So flip_target = A[idx_base] always, and correct_answer is A[idx_exp] (copy)
or M (alias).

Token alignment: idx appears only as a single digit inside `a[i] = 'M'` and
`b[i]` in the question, and all list slots render as fixed-width `'C', `, so
base/exp prompts are token-aligned by construction (asserted downstream).

Positions 1..4 of the 6-element list hold DISTINCT answer-pool characters
(the 13 chars whose quoted form tokenizes to a merged 'X first token), so
correct and flip-target logits are always single-token probes. Positions 0
and 5, the pads, and M's exclusion set come from the wider filler pool.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from itertools import permutations

# Mirrors build_code_position_stimuli.QWEN_ANSWER_POOL (filter_answer_pool
# output for Qwen2.5, hardcoded there for tokenizer-free running).
QWEN_ANSWER_POOL = list("ABCDEHILMORST")
FILLER_POOL = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]

LIST_LENGTH = 6
IDX_LEVELS = (1, 2, 3, 4)

CODE_SYSTEM_PROMPT = (
    "You are a helpful assistant that traces through Python code and "
    "determines variable values. Answer with only the Python value, nothing else."
)


def render_list(values) -> str:
    return "[" + ", ".join(f"'{c}'" for c in values) + "]"


@dataclass
class IdxPair:
    pair_id: int
    semantics: str          # copy | alias  (same for base and exp)
    idx_base: int
    idx_exp: int
    pad_len: int
    suffix_pad_len: int
    list_items: list = field(default_factory=list)
    mut_char: str = ""
    base_scenario: str = ""
    exp_scenario: str = ""
    base_question: str = ""
    exp_question: str = ""
    base_answer: str = ""   # unquoted char
    exp_answer: str = ""
    flip_target: str = ""   # unquoted char: exp-context readout of idx_base


def _scenario(items, idx, mut, semantics, pre_vals, suf_vals) -> str:
    prefix = f"pad = {render_list(pre_vals)}\n" if pre_vals else ""
    suffix = f"\npad2 = {render_list(suf_vals)}" if suf_vals else ""
    b_line = "b = a.copy()" if semantics == "copy" else "b = a"
    return (f"{prefix}a = {render_list(items)}\n{b_line}\n"
            f"a[{idx}] = '{mut}'{suffix}")


def build_idx_pairs(n_per_cell: int, seed: int,
                    semantics_levels=("copy", "alias"),
                    pad_levels=(0, 16), suffix_levels=(0, 16)) -> list[IdxPair]:
    """n_per_cell pairs for each (semantics, ordered idx pair, pad, suffix)."""
    rng = random.Random(seed)
    pairs, pid = [], 0
    idx_combos = list(permutations(IDX_LEVELS, 2))  # 12 ordered pairs
    for sem in semantics_levels:
        for idx_b, idx_e in idx_combos:
            for pre in pad_levels:
                for suf in suffix_levels:
                    for _ in range(n_per_cell):
                        # slots 1..4: distinct answer-pool chars
                        slot_chars = rng.sample(QWEN_ANSWER_POOL, 4)
                        mut = rng.choice(
                            [c for c in QWEN_ANSWER_POOL
                             if c not in slot_chars])
                        used = set(slot_chars) | {mut}
                        edge = [rng.choice([c for c in FILLER_POOL
                                            if c not in used])
                                for _ in range(2)]
                        items = [edge[0]] + slot_chars + [edge[1]]
                        pre_vals = [rng.choice([c for c in FILLER_POOL
                                                if c not in used])
                                    for _ in range(pre)]
                        suf_vals = [rng.choice([c for c in FILLER_POOL
                                                if c not in used])
                                    for _ in range(suf)]

                        base_sc = _scenario(items, idx_b, mut, sem,
                                            pre_vals, suf_vals)
                        exp_sc = _scenario(items, idx_e, mut, sem,
                                           pre_vals, suf_vals)

                        if sem == "copy":
                            base_ans, exp_ans = items[idx_b], items[idx_e]
                        else:
                            base_ans = exp_ans = mut
                        flip = items[idx_b]  # exp-context readout of idx_base

                        p = IdxPair(
                            pair_id=pid, semantics=sem,
                            idx_base=idx_b, idx_exp=idx_e,
                            pad_len=pre, suffix_pad_len=suf,
                            list_items=items, mut_char=mut,
                            base_scenario=base_sc, exp_scenario=exp_sc,
                            base_question=f"What is b[{idx_b}]?",
                            exp_question=f"What is b[{idx_e}]?",
                            base_answer=base_ans, exp_answer=exp_ans,
                            flip_target=flip,
                        )
                        _audit(p)
                        pairs.append(p)
                        pid += 1
    return pairs


def _audit(p: IdxPair) -> None:
    # prompts identical except the idx digit (two sites in the scenario)
    diffs = [i for i, (a, b) in
             enumerate(zip(p.base_scenario, p.exp_scenario)) if a != b]
    assert len(p.base_scenario) == len(p.exp_scenario), "length drift"
    assert len(diffs) == 1, f"scenarios differ at {len(diffs)} sites, want 1"
    assert p.base_scenario[diffs[0]] == str(p.idx_base)
    # flip target must be probe-distinguishable from the correct answer
    assert p.flip_target != p.exp_answer, "degenerate pair"
    assert p.flip_target in QWEN_ANSWER_POOL
    assert p.exp_answer in QWEN_ANSWER_POOL
    # answer frame invariants
    if p.semantics == "alias":
        assert p.base_answer == p.exp_answer == p.mut_char
    else:
        assert p.base_answer != p.exp_answer


def pairs_to_records(pairs: list[IdxPair]) -> list[dict]:
    recs = []
    for p in pairs:
        d = asdict(p)
        d["list_items"] = "".join(d["list_items"])
        recs.append(d)
    return recs


if __name__ == "__main__":
    ps = build_idx_pairs(n_per_cell=1, seed=0)
    from collections import Counter
    print(f"{len(ps)} pairs;",
          Counter((p.semantics, p.pad_len, p.suffix_pad_len)
                  for p in ps).most_common(4))
    p = ps[0]
    print(f"\nexample ({p.semantics}, idx {p.idx_base}->{p.idx_exp}):")
    print("BASE:\n" + p.base_scenario + "\n# " + p.base_question
          + f"  -> '{p.base_answer}'")
    print("EXP:\n" + p.exp_scenario + "\n# " + p.exp_question
          + f"  -> '{p.exp_answer}'  (flip target: '{p.flip_target}')")
