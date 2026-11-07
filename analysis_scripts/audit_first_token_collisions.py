"""
audit tom_datasets vocab files for first-token collisions under the Qwen2.5 tokenizer.

CMA measures the logit-diff at the first generated token. if two entries in the same
vocab file tokenize to the same first BPE token, sampling them as a pair (e.g.
street_1 vs street_2) produces base_ans and exp_ans with identical first tokens, so
the CMA signal at that position collapses to ~zero.

we tokenize each entry with a leading space (" {word}") because that is how the
answer typically appears in the model's continuation. for completeness we also
report the no-leading-space form.
"""
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

VOCAB_DIR = Path(__file__).resolve().parents[1] / "cma_scripts" / "tom_datasets"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"

# files actually used by the active templates (skip "copy" backups + unused files)
ACTIVE_FILES = [
    "streets.txt",        # food_truck
    "foods.txt",
    "hair_colors.txt",    # hair_styling
    "gift_items.txt",
    "library_sections.txt",  # library_book
    "book_types.txt",
    # also-present, in case used elsewhere
    "restaurants.txt",
    "times.txt",
    "reasons.txt",
    "people.txt",
    "locations.txt",
    "locations_short.txt",
    "hair_lengths.txt",
]

# variables paired in same prompt (need disjoint first tokens). source: tom_templates.py
PAIRED_VARS_PER_TEMPLATE = {
    "food_truck": [("street_1", "street_2", "streets.txt")],
    "hair_styling": [("color_1", "color_2", "hair_colors.txt")],
    "library_book": [("section_1", "section_2", "library_sections.txt")],
}


def first_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if ids else -1


def audit_file(tok, path: Path):
    entries = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    by_first_lead = defaultdict(list)  # leading-space form
    by_first_no = defaultdict(list)    # no-leading-space form

    for e in entries:
        t_lead = first_token(tok, " " + e)
        t_no = first_token(tok, e)
        by_first_lead[t_lead].append(e)
        by_first_no[t_no].append(e)

    n = len(entries)
    n_unique_lead = len(by_first_lead)
    n_unique_no = len(by_first_no)
    collisions_lead = {tid: words for tid, words in by_first_lead.items() if len(words) > 1}

    return {
        "path": path,
        "n_entries": n,
        "n_unique_first_token_leadingspace": n_unique_lead,
        "n_unique_first_token_nospace": n_unique_no,
        "collisions_lead": collisions_lead,
    }


def main():
    print(f"loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    print(f"\n{'file':<28} {'n':>4} {'uniq_first(lead)':>18} {'uniq_first(no)':>16} {'collision_groups':>18}")
    print("-" * 95)

    results = {}
    for fname in ACTIVE_FILES:
        path = VOCAB_DIR / fname
        if not path.exists():
            print(f"{fname:<28} <missing>")
            continue
        r = audit_file(tok, path)
        results[fname] = r
        print(f"{fname:<28} {r['n_entries']:>4} {r['n_unique_first_token_leadingspace']:>18} "
              f"{r['n_unique_first_token_nospace']:>16} {len(r['collisions_lead']):>18}")

    # show the actual collisions for files used by active templates
    print("\n=== collision detail (leading-space form, files used by active templates) ===")
    used_files = {trip[2] for trips in PAIRED_VARS_PER_TEMPLATE.values() for trip in trips}
    for fname in sorted(used_files):
        r = results.get(fname)
        if not r:
            continue
        print(f"\n[{fname}] {r['n_entries']} entries -> {r['n_unique_first_token_leadingspace']} unique first tokens")
        if not r["collisions_lead"]:
            print("  no collisions -- sampling 2 distinct entries always yields distinct first tokens")
            continue
        for tid, words in sorted(r["collisions_lead"].items(), key=lambda x: -len(x[1])):
            tok_str = tok.decode([tid])
            print(f"  token_id={tid} {tok_str!r:<14} <- {words}")

        # estimate p(collision) when sampling 2 distinct entries uniformly without replacement
        n = r["n_entries"]
        p_collide = sum(len(w) * (len(w) - 1) for w in r["collisions_lead"].values()) / (n * (n - 1))
        print(f"  p(first-token collision when sampling 2 distinct) = {p_collide:.3f} ({p_collide*100:.1f}%)")


if __name__ == "__main__":
    main()
