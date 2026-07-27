"""
diagnose library_book validity failures.

for each (template, run) under abstract_belief at end-patch position, parse
the ans_*.txt file, the model_responses_*_base.txt and *_exp.txt files,
and check for each prompt whether base / exp would pass the locations_match
correctness check (the threshold is 0.9 over a single sample, so 'correct' = pass).
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "codebase" / "tasks" / "causal_analysis"))
from prompt_generators.base import locations_match

RESULTS_ROOT = Path("results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct")


def parse_ans(path: Path):
    """ans_{N}.txt: '<i>: base_ans: <a>, exp_ans: <b>, causal_ans: <c>'"""
    rows = []
    pat = re.compile(r"^(\d+):\s*base_ans:\s*([^,]+),\s*exp_ans:\s*([^,]+),\s*causal_ans:\s*(.+)$")
    for line in path.read_text().splitlines():
        m = pat.match(line.strip())
        if m:
            rows.append({
                "idx": int(m.group(1)),
                "base_ans": m.group(2).strip(),
                "exp_ans": m.group(3).strip(),
                "causal_ans": m.group(4).strip(),
            })
    return rows


def parse_responses(path: Path):
    """model_responses_*_base.txt: blocks of 'response (sample 0): X' separated by ---"""
    text = path.read_text()
    # one response per prompt block
    return [m.group(1).strip() for m in re.finditer(r"response \(sample 0\):\s*(.+?)(?:\n---|\Z)", text, re.DOTALL)]


def diagnose_run(run_dir: Path, condition: str, template: str, rule_pair: str):
    seed_dir = run_dir / "z_seed_0_shuffle_False"
    if not seed_dir.exists():
        return None
    ans_files = list(seed_dir.glob("ans_*.txt"))
    if not ans_files:
        return None
    ans_path = ans_files[0]
    ans = parse_ans(ans_path)

    # find responses (under any sample_num_*_*/ folder)
    sample_dirs = list((seed_dir / "logit").glob("sample_num_*_gen_acc_*"))
    if not sample_dirs:
        return None
    sample_dir = sample_dirs[0]
    base_resps = parse_responses(sample_dir / "model_responses_50_base.txt")
    exp_resps = parse_responses(sample_dir / "model_responses_50_exp.txt")

    if len(base_resps) != len(ans) or len(exp_resps) != len(ans):
        print(f"  warn: count mismatch ans={len(ans)} base={len(base_resps)} exp={len(exp_resps)}")

    n = min(len(ans), len(base_resps), len(exp_resps))
    base_correct = []
    exp_correct = []
    for i in range(n):
        base_correct.append(locations_match(base_resps[i], ans[i]["base_ans"]))
        exp_correct.append(locations_match(exp_resps[i], ans[i]["exp_ans"]))

    n_base = sum(base_correct)
    n_exp = sum(exp_correct)
    n_both = sum(b and e for b, e in zip(base_correct, exp_correct))

    failures_base = [(i, base_resps[i], ans[i]["base_ans"]) for i in range(n) if not base_correct[i]]
    failures_exp = [(i, exp_resps[i], ans[i]["exp_ans"]) for i in range(n) if not exp_correct[i]]

    return {
        "condition": condition, "template": template, "rule": rule_pair,
        "n": n, "n_base": n_base, "n_exp": n_exp, "n_both": n_both,
        "failures_base": failures_base, "failures_exp": failures_exp,
    }


def main():
    rows = []
    conditions = ["abstract_context", "answer_changes_belief", "answer_changes_photo",
                  "control_context", "photo_context", "abstract_belief", "abstract_photo_new",
                  "answer_changes_photo_new", "control"]
    for cond in conditions:
        cond_root = RESULTS_ROOT / cond
        if not cond_root.exists():
            continue
        for tpl_dir in sorted(cond_root.glob("template_*")):
            for rule_dir in sorted(tpl_dir.glob("base_rule_*")):
                run_dir = rule_dir / "patch_before_movement"
                if not run_dir.exists():
                    continue
                r = diagnose_run(run_dir, cond, tpl_dir.name, rule_dir.name)
                if r:
                    rows.append(r)

    print(f"\n{'condition':<20} {'template':<25} {'rule':<35} {'n':>4} {'base':>6} {'exp':>6} {'both':>6}")
    print("-" * 110)
    for r in rows:
        print(f"{r['condition']:<20} {r['template']:<25} {r['rule']:<35} {r['n']:>4} {r['n_base']:>6} {r['n_exp']:>6} {r['n_both']:>6}")

    # show library_book failure details
    print("\n\n=== library_book failures (first 15 each) ===")
    for r in rows:
        if "library_book" not in r["template"]:
            continue
        print(f"\n[{r['template']} {r['rule']}] base failures ({len(r['failures_base'])}):")
        for i, resp, expect in r["failures_base"][:15]:
            print(f"  [{i:>2}] expected={expect!r:<30} got={resp[:60]!r}")
        print(f"[{r['template']} {r['rule']}] exp failures ({len(r['failures_exp'])}):")
        for i, resp, expect in r["failures_exp"][:15]:
            print(f"  [{i:>2}] expected={expect!r:<30} got={resp[:60]!r}")


if __name__ == "__main__":
    main()
