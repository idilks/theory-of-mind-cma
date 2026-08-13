"""Content-addressable vs location-addressable code stimuli (dict frame).

Fills the second cell of the addressing 2x2 (the ToM vignette compiler fills
the first): in all data so far, DOMAIN and ADDRESSING-DEMAND are perfectly
confounded — every code task demanded positional indexing, every ToM task
demanded content matching. This grid unconfounds them inside the code domain:
the same scenario is queried either BY KEY (associative / content-addressed)
or BY POSITION (indexed / location-addressed), so retrieval mode is carried
entirely by the question, on identical scenario tokens.

Scenario frame (mirrors the idx grid's 6-slot list, as a 6-pair dict):

    pad = ['Q', ...]                # optional prefix pad, as in the idx grid
    a = {'f': 'Z', 'k': 'O', 'w': 'M', 'p': 'R', 'x': 'T', 'g': 'Y'}
    b = dict(a)     |  b = a        # copy | alias
    a['w'] = 'D'                    # mutation at the queried pair's key

Positions 0 and 5 are filler pairs; positions 1..4 hold the four queryable
slots. Slot KEYS are a FIXED alphabet ('k','w','p','x') whose dict ORDER is
permuted per item (balanced cycling over all 24 permutations), so key
identity is decorrelated from serial position by design — verified by
design_check(), which hard-fails on imbalance. Fixed keys (rather than
episodic per-item keys) are deliberate: they make the key a stable symbol,
and they guarantee a foreign (shuffled) donor's queried key exists in every
recipient's dict.

Slot VALUES are distinct answer-pool characters (merged 'X first-token
probes), so the four slots of any prompt form a complete within-prompt
competitor set for the transfer-index readout.

Two questions per (scenario, target):
    key arm:  What is b['w']?                    (content-addressed)
    pos arm:  What is list(b.values())[2]?       (location-addressed; digit
              indexes the 6-element insertion-order values list, 1..4)
Both query terms are single characters, so base/exp questions within an arm
differ at exactly one site, mirroring the idx-pair token-alignment audits.

Answer logic (values V by slot, mutation char M at the queried slot's key):
    copy : b unaffected by the mutation -> answer is V[target].
    alias: b is a                       -> answer is M.
A successful ADDRESS transplant (donor's target installed in the recipient)
reads the recipient's ORIGINAL value at the donor's target:
    flip_target = V[t_base]   (exp-context readout, exactly as in idx pairs)
and flip_target != exp_answer always (distinct slot values; M excluded).

The coordinate-diagnostic cell: because key and position are decorrelated, a
FOREIGN (shuffled) donor names a slot whose key and whose position pick out
DIFFERENT pairs of the recipient's dict. Where the patched mass lands —
the recipient's pair with the donor's KEY vs the recipient's pair at the
donor's POSITION — reveals which coordinate the heads actually transmit.
Pairing helpers emit both referents per shuffled row for exactly this test.

Pairing functions (both consume the same cores, so arms are matched):
    addressing_pairs(cores, sem, qmode)  base/exp differ ONLY in target
                                         (the TI / cross-patch family)
    semantics_twins(cores, qmode)        base/exp differ ONLY in copy|alias
                                         (the state-register family)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from itertools import permutations

from code_idx_pairs import QWEN_ANSWER_POOL, FILLER_POOL, render_list

SLOT_KEYS = ("k", "w", "p", "x")          # fixed symbolic key alphabet
KEY_FILLER_POOL = [c for c in "cdefghjmnqrstuvyz"]  # lowercase, no a/b/slot keys
N_PAIRS = 6                                # dict length; slots at positions 1..4
POS_LEVELS = (1, 2, 3, 4)

POS_QUESTION_STYLES = {
    "expr": "What is list(b.values())[{p}]?",
    # fallback if the expr style fails the clean-accuracy gate at smoke:
    "ordinal": "What is value number {p} of b, counting from 0 in order?",
}


def render_dict(keys, values) -> str:
    return "{" + ", ".join(f"'{k}': '{v}'" for k, v in zip(keys, values)) + "}"


@dataclass
class CACore:
    core_id: int
    t_base: int              # queried slot (0..3, abstract) for the base prompt
    t_exp: int               # queried slot for the exp prompt (t_exp != t_base)
    perm: tuple = ()         # perm[s] = dict position (1..4) of slot s
    pad_len: int = 0
    slot_values: list = field(default_factory=list)   # V[s], s = 0..3
    mut_char: str = ""
    keys_ordered: list = field(default_factory=list)   # 6 keys, dict order
    values_ordered: list = field(default_factory=list) # 6 values, dict order
    pad_vals: list = field(default_factory=list)

    @property
    def p_base(self) -> int:
        return self.perm[self.t_base]

    @property
    def p_exp(self) -> int:
        return self.perm[self.t_exp]

    def key_of(self, slot: int) -> str:
        return SLOT_KEYS[slot]

    def scenario(self, sem: str, target: int) -> str:
        prefix = f"pad = {render_list(self.pad_vals)}\n" if self.pad_vals else ""
        b_line = "b = dict(a)" if sem == "copy" else "b = a"
        return (f"{prefix}a = {render_dict(self.keys_ordered, self.values_ordered)}\n"
                f"{b_line}\n"
                f"a['{self.key_of(target)}'] = '{self.mut_char}'")

    def question(self, qmode: str, target: int, pos_style: str = "expr") -> str:
        if qmode == "key":
            return f"What is b['{self.key_of(target)}']?"
        return POS_QUESTION_STYLES[pos_style].format(p=self.perm[target])

    def answer(self, sem: str, target: int) -> str:
        return self.slot_values[target] if sem == "copy" else self.mut_char


def build_cores(n_per_cell: int, seed: int,
                pad_levels=(0, 16)) -> list[CACore]:
    """n_per_cell cores per (ordered target pair, pad) cell.

    Perms cycle through a seeded shuffle of all 24 permutations of the four
    positions; with total cores a multiple of 24 the key<->position
    assignment is exactly balanced (design_check verifies regardless).
    """
    rng = random.Random(seed)
    all_perms = list(permutations(POS_LEVELS))
    rng.shuffle(all_perms)
    cores, cid = [], 0
    target_combos = list(permutations(range(4), 2))   # 12 ordered pairs
    for t_b, t_e in target_combos:
        for pad in pad_levels:
            for _ in range(n_per_cell):
                perm = all_perms[cid % len(all_perms)]
                slot_values = rng.sample(QWEN_ANSWER_POOL, 4)
                mut = rng.choice([c for c in QWEN_ANSWER_POOL
                                  if c not in slot_values])
                used = set(slot_values) | {mut}
                filler_keys = rng.sample(KEY_FILLER_POOL, 2)
                filler_vals = [rng.choice([c for c in FILLER_POOL
                                           if c not in used])
                               for _ in range(2)]
                pad_vals = [rng.choice([c for c in FILLER_POOL
                                        if c not in used])
                            for _ in range(pad)]

                keys_ordered = [None] * N_PAIRS
                values_ordered = [None] * N_PAIRS
                keys_ordered[0], values_ordered[0] = filler_keys[0], filler_vals[0]
                keys_ordered[5], values_ordered[5] = filler_keys[1], filler_vals[1]
                for s in range(4):
                    keys_ordered[perm[s]] = SLOT_KEYS[s]
                    values_ordered[perm[s]] = slot_values[s]

                core = CACore(core_id=cid, t_base=t_b, t_exp=t_e, perm=perm,
                              pad_len=pad, slot_values=slot_values,
                              mut_char=mut, keys_ordered=keys_ordered,
                              values_ordered=values_ordered, pad_vals=pad_vals)
                _audit_core(core)
                cores.append(core)
                cid += 1
    return cores


def _audit_core(c: CACore) -> None:
    assert sorted(c.perm) == list(POS_LEVELS)
    assert len(set(c.keys_ordered)) == N_PAIRS
    assert all(k.islower() for k in c.keys_ordered)
    assert len(set(c.slot_values)) == 4
    assert all(v in QWEN_ANSWER_POOL for v in c.slot_values)
    assert c.mut_char in QWEN_ANSWER_POOL and c.mut_char not in c.slot_values
    for s in range(4):
        assert c.values_ordered[c.perm[s]] == c.slot_values[s]
        assert c.keys_ordered[c.perm[s]] == SLOT_KEYS[s]
    assert c.t_base != c.t_exp


# ---------------------------------------------------------------------------
# pairing functions
# ---------------------------------------------------------------------------

def addressing_pairs(cores: list[CACore], sem: str, qmode: str,
                     pos_style: str = "expr") -> list[dict]:
    """Base/exp differ ONLY in the queried target (the address axis)."""
    recs = []
    for c in cores:
        base_sc = c.scenario(sem, c.t_base)
        exp_sc = c.scenario(sem, c.t_exp)
        _audit_addressing(c, base_sc, exp_sc)
        base_q = c.question(qmode, c.t_base, pos_style)
        exp_q = c.question(qmode, c.t_exp, pos_style)
        assert len(base_q) == len(exp_q)
        assert sum(a != b for a, b in zip(base_q, exp_q)) == 1
        flip = c.slot_values[c.t_base]          # exp-context readout of t_base
        exp_ans = c.answer(sem, c.t_exp)
        assert flip != exp_ans, "degenerate pair"
        recs.append(dict(
            family="addr", qmode=qmode, semantics=sem, core_id=c.core_id,
            t_base=c.t_base, t_exp=c.t_exp, p_base=c.p_base, p_exp=c.p_exp,
            key_base=c.key_of(c.t_base), key_exp=c.key_of(c.t_exp),
            pad_len=c.pad_len, perm="".join(map(str, c.perm)),
            slot_values="".join(c.slot_values), mut_char=c.mut_char,
            keys_ordered="".join(c.keys_ordered),
            base_scenario=base_sc, exp_scenario=exp_sc,
            base_question=base_q, exp_question=exp_q,
            base_answer=c.answer(sem, c.t_base), exp_answer=exp_ans,
            flip_target=flip,
        ))
    return recs


def _audit_addressing(c: CACore, base_sc: str, exp_sc: str) -> None:
    assert len(base_sc) == len(exp_sc), "length drift"
    diffs = [i for i, (a, b) in enumerate(zip(base_sc, exp_sc)) if a != b]
    assert len(diffs) == 1, f"scenarios differ at {len(diffs)} sites, want 1"
    assert base_sc[diffs[0]] == c.key_of(c.t_base)
    assert exp_sc[diffs[0]] == c.key_of(c.t_exp)


def semantics_twins(cores: list[CACore], qmode: str,
                    pos_style: str = "expr") -> list[dict]:
    """Base/exp differ ONLY in copy|alias at the base target (the state axis).

    copy answer = V[t_base]; alias answer = mut_char. flip_target for a state
    patch is the other frame's answer, as in the list-frame sem twins.
    """
    recs = []
    for c in cores:
        sc_copy = c.scenario("copy", c.t_base)
        sc_alias = c.scenario("alias", c.t_base)
        # identical except the b-line ('b = dict(a)' vs 'b = a')
        assert sc_copy.replace("b = dict(a)", "b = a") == sc_alias
        q = c.question(qmode, c.t_base, pos_style)
        copy_ans, alias_ans = c.slot_values[c.t_base], c.mut_char
        assert copy_ans != alias_ans
        recs.append(dict(
            family="sem", qmode=qmode, semantics="copy_vs_alias",
            core_id=c.core_id, t_base=c.t_base, t_exp=c.t_base,
            p_base=c.p_base, p_exp=c.p_base,
            key_base=c.key_of(c.t_base), key_exp=c.key_of(c.t_base),
            pad_len=c.pad_len, perm="".join(map(str, c.perm)),
            slot_values="".join(c.slot_values), mut_char=c.mut_char,
            keys_ordered="".join(c.keys_ordered),
            base_scenario=sc_copy, exp_scenario=sc_alias,
            base_question=q, exp_question=q,
            base_answer=copy_ans, exp_answer=alias_ans,
            flip_target=copy_ans,
        ))
    return recs


def shuffled_referents(rec: dict, donor: dict) -> dict:
    """The coordinate-diagnostic bookkeeping for a foreign-donor row.

    Returns the TWO candidate referents the donor's address picks out in the
    RECIPIENT'S dict: the recipient's value under the donor's queried KEY,
    and the recipient's value at the donor's queried POSITION. Because key
    and position are decorrelated across items, these generally differ;
    where the patched mass lands adjudicates which coordinate travels.
    """
    keys = list(rec["keys_ordered"])
    vals_by_pos = {}
    slot_vals = list(rec["slot_values"])
    perm = [int(x) for x in rec["perm"]]
    for s in range(4):
        vals_by_pos[perm[s]] = slot_vals[s]
    key_ref = slot_vals[SLOT_KEYS.index(donor["key_base"])]
    pos_ref = vals_by_pos[donor["p_base"]]
    return dict(donor_key=donor["key_base"], donor_pos=donor["p_base"],
                key_referent=key_ref, pos_referent=pos_ref,
                referents_differ=key_ref != pos_ref)


# ---------------------------------------------------------------------------
# capture records (RSA / geometry) and design check
# ---------------------------------------------------------------------------

def capture_records(cores: list[CACore], sems=("copy", "alias"),
                    qmodes=("key", "pos"), pos_style: str = "expr",
                    targets=("base",)) -> list[dict]:
    """One row per unique (core, sem, qmode, target) prompt, with the design
    factors needed for the model RDMs: queried KEY identity, queried
    POSITION, semantics, pad. Key-identity and position models are
    decorrelated across the set by the balanced perms (design_check)."""
    rows = []
    for c in cores:
        for sem in sems:
            for qmode in qmodes:
                for which in targets:
                    t = c.t_base if which == "base" else c.t_exp
                    rows.append(dict(
                        core_id=c.core_id, sem=sem, qmode=qmode, target=t,
                        queried_key=c.key_of(t), queried_pos=c.perm[t],
                        pad_len=c.pad_len, perm="".join(map(str, c.perm)),
                        scenario=c.scenario(sem, t),
                        question=c.question(qmode, t, pos_style),
                        answer=c.answer(sem, t),
                    ))
    return rows


def design_check(cores: list[CACore], hard: bool = True) -> dict:
    """Key<->position balance and queried-key vs queried-position
    independence. Hard-fails (like the compiler) rather than warning."""
    import math
    n = len(cores)
    # (slot, position) contingency: each core assigns each slot one position
    cont = [[0] * 4 for _ in range(4)]
    for c in cores:
        for s in range(4):
            cont[s][c.perm[s] - 1] += 1
    exp_cell = n / 4
    max_dev = max(abs(cont[s][p] - exp_cell) / exp_cell
                  for s in range(4) for p in range(4))
    # queried coordinate independence (base target): key vs position
    q = [[0] * 4 for _ in range(4)]
    for c in cores:
        q[c.t_base][c.p_base - 1] += 1
    tot = sum(map(sum, q))
    chi2 = 0.0
    for s in range(4):
        for p in range(4):
            e = sum(q[s]) * sum(row[p] for row in q) / tot
            if e > 0:
                chi2 += (q[s][p] - e) ** 2 / e
    cramers_v = math.sqrt(chi2 / (tot * 3)) if tot else 0.0
    report = dict(n_cores=n, keypos_max_rel_dev=round(max_dev, 4),
                  queried_key_pos_cramers_v=round(cramers_v, 4))
    if hard:
        assert max_dev <= 0.20, f"key<->position imbalance: {report}"
        assert cramers_v <= 0.15, f"queried key/pos correlated: {report}"
    return report


if __name__ == "__main__":
    cores = build_cores(n_per_cell=4, seed=0)     # 12*2*4 = 96, 96 % 24 == 0
    print("design check:", design_check(cores))
    for qmode in ("key", "pos"):
        recs = addressing_pairs(cores, sem="copy", qmode=qmode)
        print(f"{qmode} arm: {len(recs)} addressing pairs (copy)")
    twins = semantics_twins(cores, qmode="key")
    print(f"sem twins (key arm): {len(twins)}")
    caps = capture_records(cores)
    print(f"capture records: {len(caps)}")

    r = addressing_pairs(cores, sem="copy", qmode="key")[0]
    print(f"\nexample (key arm, copy, slot {r['t_base']}->{r['t_exp']}, "
          f"keys {r['key_base']}->{r['key_exp']}, "
          f"pos {r['p_base']}->{r['p_exp']}):")
    print("BASE:\n" + r["base_scenario"] + "\n# " + r["base_question"]
          + f"  -> '{r['base_answer']}'")
    print("EXP:\n" + r["exp_scenario"] + "\n# " + r["exp_question"]
          + f"  -> '{r['exp_answer']}'  (flip target: '{r['flip_target']}')")
    rp = addressing_pairs(cores, sem="copy", qmode="pos")[0]
    print("\nsame core, pos arm questions: "
          f"{rp['base_question']!r} / {rp['exp_question']!r}")
    # coordinate-diagnostic smoke: foreign donor referents
    recs = addressing_pairs(cores, sem="copy", qmode="key")
    diff = sum(shuffled_referents(recs[i], recs[(i + 7) % len(recs)])
               ["referents_differ"] for i in range(len(recs)))
    print(f"\nforeign-donor key/pos referents differ in {diff}/{len(recs)} "
          "rows (want a solid majority for the coordinate diagnostic)")
