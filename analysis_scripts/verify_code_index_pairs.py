"""
Verify the Sept 2026 code CMA pairs by EXECUTING each generated snippet:
two-mutation pairs (code_index, code_index_flip, code_abstract_two_mut) and
mutate-b pairs (code_veracity_only, code_site_only, code_site_only_rev), and
the one-element position pilot (code_pos_matched, code_pos_unmatched; needs
--tokenizer, since filler sizing is tokenizer-dependent).

Per pair, asserts:
  - base / exp correct answers match what Python computes
  - the causal answer is what the claimed transfer would produce in exp:
      code_index*            exp evaluated at base's index
      code_abstract_two_mut  exp evaluated with base's copy/alias kind
  - code_index*: copy/alias kind identical on both sides; base and exp differ
    only in the two answer letters' places and the question index
  - code_index_flip (alias): base's index sits on a different mutation line
    than in exp, and exp's line at base's line number holds the exp answer
    (a transferred line position predicts NO change)
  - code_veracity_only: copy -> alias; BOTH answers are read from the
    mutation line (so a transferred read site predicts no change); causal =
    exp evaluated with base's copy status
  - code_site_only(_rev): both copy; read sites differ (list <-> mutation
    line); causal = the letter exp holds at base's read site
  - code_pos_matched: answer-token index, total length and distance to the
    end are EQUAL on the fully formatted prompt (system prompt + chat template
    + code fence); code_pos_unmatched: answer-token index differs
  - each answer letter occurs exactly once per snippet
  - with --tokenizer: base and exp tokenize to the same length, and differ at
    a small fixed number of token positions

Usage:
    python analysis_scripts/verify_code_index_pairs.py --n 2000
    python analysis_scripts/verify_code_index_pairs.py --n 500 --tokenizer Qwen/Qwen2.5-14B-Instruct --show 1
"""

import argparse
import ast
import random
import re
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from prompt_generators.code_templates import CodeTemplateSystem, filter_answer_pool  # noqa: E402
from prompt_generators.code_generator import CodeGenerator  # noqa: E402
from cma_config import CMACondition, PromptConfig  # noqa: E402

TEMPLATE = "indexed_list"
CONDITIONS = ("code_index", "code_index_flip", "code_abstract_two_mut")
MUTATE_B_CONDITIONS = ("code_veracity_only", "code_site_only", "code_site_only_rev")
MUT_LINE = r"[ab]\[\d+\] = "
POS_CONDITIONS = ("code_pos_matched", "code_pos_unmatched")


class _Cfg:
    def __init__(self, prompts):
        self.prompts = prompts


def _code_block(prompt: str) -> str:
    return re.search(r"```python\n(.*?)```", prompt, re.S).group(1).rstrip("\n")


def _answer_token_index(tokenizer, prompt: str, ch: str):
    m = re.search(r"```python\n(.*?)```", prompt, re.S)
    pos = m.start(1) + m.group(1).index(ch)
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    idx = next(i for i, (a, b) in enumerate(enc["offset_mapping"]) if a <= pos < b)
    return idx, len(enc["input_ids"])


def verify_pos_pilot(condition: str, rule: str, n: int, tokenizer, show: int) -> Counter:
    """Runs through the real CodeGenerator so alignment is checked on the
    prompt cma.py will actually tokenize."""
    pc = PromptConfig(base_rule=rule, context_type="code", prompt_num=n,
                      condition=CMACondition(condition), template_names=[TEMPLATE])
    pairs, answers, causals = CodeGenerator(_Cfg(pc), tokenizer).generate_prompts()
    stats = Counter()
    for k, ((base, exp), (base_ans, exp_ans), causal) in enumerate(zip(pairs, answers, causals)):
        cb, ce = _code_block(base), _code_block(exp)
        assert run_snippet(cb, 0) == base_ans and run_snippet(ce, 0) == exp_ans
        assert base_ans == exp_ans and causal != exp_ans
        assert kind_of(cb) != kind_of(ce) and kind_of(cb) == ("copy" if rule == "ABA" else "alias")
        assert run_snippet(flip_kind(ce), 0) == causal, (ce, causal)
        for code in (cb, ce):
            for letter in (exp_ans[1], causal[1]):
                assert code.count(letter) == 1, (code, letter)
        (ib, lb), (ie, le) = _answer_token_index(tokenizer, base, exp_ans[1]), _answer_token_index(tokenizer, exp, exp_ans[1])
        n_blocks = (cb.count("\n#") + cb.startswith("#"), ce.count("\n#") + ce.startswith("#"))
        assert n_blocks == (2, 2), n_blocks  # prefix + suffix comment in every prompt
        if condition == "code_pos_matched":
            assert ib == ie and lb == le, (ib, ie, lb, le)
        else:
            assert ib != ie
            stats[f"offset={ie - ib}"] += 1
        stats[f"len={lb}/{le}"] += 1
        if k < show:
            print(f"--- {condition} {rule} | ans={exp_ans} causal={causal} | answer tok {ib}->{ie}, len {lb}->{le}")
            print(f"[base]\n{cb}\n[exp]\n{ce}\n")
        stats["ok"] += 1
    return stats


def run_snippet(snippet: str, idx: int) -> str:
    env = {}
    exec(snippet, {}, env)
    return f"'{env['b'][idx]}'"


def q_index(question: str) -> int:
    return int(re.search(r"b\[(\d+)\]", question).group(1))


def kind_of(snippet: str) -> str:
    return "copy" if "b = a.copy()" in snippet else "alias"


def flip_kind(snippet: str) -> str:
    if kind_of(snippet) == "copy":
        return snippet.replace("b = a.copy()", "b = a")
    return re.sub(r"b = a(?!\.)", "b = a.copy()", snippet)


def mutation_line_of(snippet: str, idx: int) -> int:
    lines = [l for l in snippet.split("\n") if re.match(MUT_LINE, l)]
    return next(n for n, l in enumerate(lines) if l.startswith(f"a[{idx}]"))


def mutation_value_at_line(snippet: str, line_no: int) -> str:
    lines = [l for l in snippet.split("\n") if re.match(MUT_LINE, l)]
    return lines[line_no].split(" = ")[1]


def site_values(snippet: str, idx: int) -> dict:
    """Quoted letter held at each candidate read site for index idx."""
    init = ast.literal_eval(snippet.split("\n")[0].split(" = ", 1)[1])
    return {"list": f"'{init[idx]}'", "mutation_line": mutation_value_at_line(snippet, 0)}


def read_site(snippet: str, idx: int, answer: str) -> str:
    sites = [k for k, v in site_values(snippet, idx).items() if v == answer]
    assert len(sites) == 1, (snippet, answer)
    return sites[0]


def verify_mutate_b(condition: str, n: int, tokenizer, show: int) -> Counter:
    ts = CodeTemplateSystem(
        answer_pool=filter_answer_pool(tokenizer) if tokenizer is not None else None)
    stats = Counter()
    for k in range(n):
        if condition == "code_veracity_only":
            pair = ts.generate_veracity_only_pair(TEMPLATE, "ABA")
        else:
            pair = ts.generate_site_only_pair(
                TEMPLATE, "ABA", reverse=condition == "code_site_only_rev")
        base, exp, base_ans, exp_ans, causal, question = pair
        idx = q_index(question)

        assert run_snippet(base, idx) == base_ans, (base, base_ans)
        assert run_snippet(exp, idx) == exp_ans, (exp, exp_ans)
        assert base_ans == exp_ans and causal != exp_ans
        base_site = read_site(base, idx, base_ans)
        exp_site = read_site(exp, idx, exp_ans)

        if condition == "code_veracity_only":
            assert (kind_of(base), kind_of(exp)) == ("copy", "alias")
            assert base_site == exp_site == "mutation_line"
            assert run_snippet(flip_kind(exp), idx) == causal, (exp, causal)
        else:
            assert kind_of(base) == kind_of(exp) == "copy"
            expected = ("mutation_line", "list") if condition.endswith("_rev") else ("list", "mutation_line")
            assert (base_site, exp_site) == expected
            assert site_values(exp, idx)[base_site] == causal, (exp, causal)
        stats[f"{base_site}->{exp_site}"] += 1

        for snippet in (base, exp):
            for letter in (exp_ans, causal):
                assert snippet.count(letter) == 1, (snippet, letter)

        if tokenizer is not None:
            ids_b = tokenizer(f"{base}\n{question}", add_special_tokens=False)["input_ids"]
            ids_e = tokenizer(f"{exp}\n{question}", add_special_tokens=False)["input_ids"]
            if condition != "code_veracity_only":  # .copy() legitimately changes length there
                assert len(ids_b) == len(ids_e), (base, exp)
                stats[f"n_diff_tokens={sum(a != b for a, b in zip(ids_b, ids_e))}"] += 1

        if k < show:
            print(f"--- {condition} ABA | base_ans={base_ans} exp_ans={exp_ans} causal={causal}")
            print(f"[base]\n{base}\n{question}\n[exp]\n{exp}\n{question}\n")
        stats["ok"] += 1
    return stats


def verify(condition: str, rule: str, n: int, tokenizer, show: int) -> Counter:
    ts = CodeTemplateSystem(
        answer_pool=filter_answer_pool(tokenizer) if tokenizer is not None else None)
    stats = Counter()
    for k in range(n):
        if condition == "code_abstract_two_mut":
            base, exp, base_ans, exp_ans, causal, q_base = (
                ts.generate_abstract_two_mut_pair(TEMPLATE, rule))
            q_exp = q_base
        else:
            base, exp, base_ans, exp_ans, causal, q_base, q_exp = ts.generate_index_pair(
                TEMPLATE, rule, flip_order=condition == "code_index_flip")
        p, q = q_index(q_base), q_index(q_exp)

        assert run_snippet(base, p) == base_ans, (base, q_base, base_ans)
        assert run_snippet(exp, q) == exp_ans, (exp, q_exp, exp_ans)
        assert base_ans == exp_ans and causal != exp_ans

        if condition == "code_abstract_two_mut":
            assert kind_of(base) != kind_of(exp)
            assert kind_of(base) == ("copy" if rule == "ABA" else "alias")
            assert run_snippet(flip_kind(exp), q) == causal, (exp, causal)
        else:
            assert p != q
            assert kind_of(base) == kind_of(exp) == ("copy" if rule == "ABA" else "alias")
            assert run_snippet(exp, p) == causal, (exp, p, causal)
            if kind_of(base) == "alias":
                same_line = mutation_line_of(base, p) == mutation_line_of(exp, p)
                if condition == "code_index_flip":
                    assert not same_line
                    # transferred line position -> exp's answer -> no change
                    assert mutation_value_at_line(exp, mutation_line_of(base, p)) == exp_ans
                else:
                    assert same_line
            stats["p<q"] += p < q

        for snippet in (base, exp):
            for letter in (exp_ans, causal):
                assert snippet.count(letter) == 1, (snippet, letter)

        if tokenizer is not None:
            ids_b = tokenizer(f"{base}\n{q_base}", add_special_tokens=False)["input_ids"]
            ids_e = tokenizer(f"{exp}\n{q_exp}", add_special_tokens=False)["input_ids"]
            if condition != "code_abstract_two_mut":  # .copy() legitimately changes length there
                assert len(ids_b) == len(ids_e), (base, exp)
                stats[f"n_diff_tokens={sum(a != b for a, b in zip(ids_b, ids_e))}"] += 1

        if k < show:
            print(f"--- {condition} {rule} | base_ans={base_ans} exp_ans={exp_ans} causal={causal}")
            print(f"[base]\n{base}\n{q_base}\n[exp]\n{exp}\n{q_exp}\n")
        stats["ok"] += 1
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="HF tokenizer name; enables answer-pool filtering and token-alignment checks")
    parser.add_argument("--show", type=int, default=0, help="print the first k pairs per cell")
    args = parser.parse_args()

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else None
    for condition in CONDITIONS:
        for rule in ("ABA", "ABB"):
            stats = verify(condition, rule, args.n, tokenizer, args.show)
            print(f"{condition:24s} {rule}  {dict(stats)}")
    for condition in MUTATE_B_CONDITIONS:
        stats = verify_mutate_b(condition, args.n, tokenizer, args.show)
        print(f"{condition:24s} ABA  {dict(stats)}")
    if tokenizer is not None:
        for condition in POS_CONDITIONS:
            for rule in ("ABA", "ABB"):
                stats = verify_pos_pilot(condition, rule, args.n, tokenizer, args.show)
                print(f"{condition:24s} {rule}  {dict(stats)}")
    else:
        print("[skipped code_pos_* checks: pass --tokenizer]")
    for bad in MUTATE_B_CONDITIONS[:2]:  # ABB must be rejected, not silently generated
        gen = (CodeTemplateSystem().generate_veracity_only_pair if bad == "code_veracity_only"
               else CodeTemplateSystem().generate_site_only_pair)
        try:
            gen(TEMPLATE, "ABB")
        except ValueError:
            print(f"{bad:24s} ABB  correctly rejected")
        else:
            raise AssertionError(f"{bad} accepted base_rule=ABB")


if __name__ == "__main__":
    main()
