"""
expand the three active tom_datasets vocab files to a target size with all
distinct first tokens under the Qwen2.5 tokenizer.

inputs:  tom_datasets/{streets,library_sections,hair_colors}.txt
outputs: overwrites the same files (preserves existing entries, drops dupes,
         appends new candidates that don't collide on first token).

run:
    python scripts/expand_tom_vocabs.py --dry_run    # preview only
    python scripts/expand_tom_vocabs.py              # write changes
"""
import argparse
from pathlib import Path
from transformers import AutoTokenizer

VOCAB_DIR = Path(__file__).resolve().parents[1] / "tom_datasets"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
TARGET_SIZE = 50

# candidate additions per file. only those that pass first-token uniqueness
# (against existing entries AND each other) are added, in the order listed,
# until TARGET_SIZE is reached.
CANDIDATES = {
    "streets.txt": [
        # extras chosen to give variety in starting letters
        "Birch Lane", "Willow Way", "Sycamore Drive", "Beech Avenue", "Aspen Court",
        "Magnolia Road", "Hickory Street", "Walnut Avenue", "Poplar Drive", "Linden Lane",
        "Sunset Boulevard", "Highland Avenue", "Lakeview Drive", "Broadway", "Madison Avenue",
        "Jefferson Street", "Lincoln Road", "Washington Street", "Franklin Avenue", "Adams Drive",
        "Roosevelt Boulevard", "Kennedy Lane", "Riverside Drive", "Harbor Road", "Cliff Drive",
        "Meadow Lane", "Brook Street", "Creek Road", "Falls Drive", "Stone Street",
        "Brick Lane", "Iron Avenue", "Steel Drive", "Copper Road", "Ocean Avenue",
        "Bay Street", "Crystal Drive", "Diamond Street", "Ruby Lane", "Emerald Drive",
        "Onyx Road", "Quartz Lane", "Granite Drive", "Marble Street", "Ivy Lane",
        "Holly Avenue", "Rose Drive", "Daisy Lane", "Tulip Avenue", "Orchid Drive",
        "Jasmine Road", "Lotus Lane", "Apple Street", "Plum Lane", "Berry Road",
        "Vine Street", "Grove Avenue", "Mountain Road", "Canyon Lane", "Ridge Street",
        "Summit Avenue", "Plateau Drive", "Mesa Lane", "Prairie Road", "Desert Avenue",
        "Woodland Street", "Glen Avenue", "Dale Lane", "Heath Road", "Moor Drive",
        "North Street", "South Avenue", "East Drive", "West Road", "Central Boulevard",
        "Union Street", "Liberty Avenue", "Independence Drive", "Heritage Lane", "Founders Way",
        "Victory Boulevard", "Memorial Drive", "Capitol Street", "Federal Avenue", "Industrial Road",
        "Commerce Street", "Trade Lane", "Mill Road", "Foundry Avenue", "Granary Drive",
        "Tannery Lane", "Cobblestone Street", "Boulder Avenue", "Pebble Road", "Sandstone Drive",
    ],
    "library_sections.txt": [
        "psychology", "sociology", "anthropology", "economics", "business",
        "philosophy", "religion", "mythology", "linguistics", "journalism",
        "sports", "gardening", "crafts", "hobbies", "drama",
        "essays", "romance", "mystery", "thriller", "fantasy",
        "horror", "classics", "memoir", "geography", "nature",
        "environment", "agriculture", "astronomy", "physics", "chemistry",
        "biology", "mathematics", "statistics", "programming", "robotics",
        "electronics", "automotive", "aviation", "military", "politics",
        "government", "law", "ethics", "languages", "literature",
        "magazines", "newspapers", "periodicals", "manga", "juvenile",
        "audiobooks", "ebooks", "criminology", "archaeology", "graphic novels",
        "picture books", "atlases", "encyclopedias", "dictionaries", "almanacs",
    ],
    "hair_colors.txt": [
        # already 76 entries, just dedup; no candidates needed but listed for completeness
        "honey", "strawberry", "espresso", "platinum blonde", "salt and pepper",
        "ash blonde", "dirty blonde", "dark brown", "light brown", "midnight black",
    ],
}


def first_token_id(tok, s: str) -> int:
    ids = tok.encode(" " + s, add_special_tokens=False)
    return ids[0] if ids else -1


def process_file(tok, path: Path, candidates: list, target: int, dry_run: bool):
    existing = [line.strip() for line in path.read_text().splitlines() if line.strip()]

    seen_words = set()
    seen_first_tokens = set()
    kept = []
    dropped_dup = []
    dropped_collision = []

    for w in existing:
        if w in seen_words:
            dropped_dup.append(w)
            continue
        tid = first_token_id(tok, w)
        if tid in seen_first_tokens:
            dropped_collision.append((w, tid))
            continue
        seen_words.add(w)
        seen_first_tokens.add(tid)
        kept.append(w)

    n_after_dedup = len(kept)

    added = []
    skipped_cand = []
    for w in candidates:
        if len(kept) >= target:
            break
        if w in seen_words:
            skipped_cand.append((w, "duplicate of existing"))
            continue
        tid = first_token_id(tok, w)
        if tid in seen_first_tokens:
            collider = next((k for k in kept if first_token_id(tok, k) == tid), "?")
            skipped_cand.append((w, f"first-token collision with {collider!r}"))
            continue
        seen_words.add(w)
        seen_first_tokens.add(tid)
        kept.append(w)
        added.append(w)

    print(f"\n[{path.name}]")
    print(f"  existing: {len(existing)}  -> after dedup/collision-prune: {n_after_dedup}  -> after expansion: {len(kept)}")
    if dropped_dup:
        print(f"  dropped duplicates: {dropped_dup}")
    if dropped_collision:
        print(f"  dropped first-token collisions inside existing: {dropped_collision}")
    if added:
        print(f"  added {len(added)}: {added}")
    n_collision_skips = sum(1 for _, r in skipped_cand if r.startswith("first-token"))
    if n_collision_skips:
        print(f"  candidates skipped due to first-token collision: {n_collision_skips}")
    if len(kept) < target:
        print(f"  WARN: only reached {len(kept)}/{target} -- need more candidate words")

    if not dry_run:
        path.write_text("\n".join(kept) + "\n")
        print(f"  wrote {len(kept)} entries to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--target", type=int, default=TARGET_SIZE)
    args = parser.parse_args()

    print(f"loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    for fname, cands in CANDIDATES.items():
        path = VOCAB_DIR / fname
        if not path.exists():
            print(f"skip missing {path}")
            continue
        process_file(tok, path, cands, args.target, args.dry_run)


if __name__ == "__main__":
    main()
