"""
Per-head final-token attention by prompt region, with region sizes, over ALL prompts.

For every captured head (the top-50 CMA heads stored by capture_attention_patterns.py,
significant or not) in every story cell, partitions the final-token ("lookback")
attention row into regions that cover the whole prompt. Regions are aligned with
the code domain (see region_regions_code.py) so the three domains share columns:

  orig_loc        every mention of the ORIGINAL location in the story
                  (the answer in false-belief / outdated-photo prompts)
  current_loc     every mention of the CURRENT (post-move) location in the story
  orig_context    rest of the sentence that first mentions the original location
  access_cue      rest of the access-event sentence: the info-loss sentence (belief),
                  or the photo-taking sentence (photo; the one holding char_q)
  change_context  rest of the sentence that first mentions the current location (the move)
  other_story     every other story token
  question        the question, from its line start up to and including "?"
  system          everything before the story (system turn, chat header)
  end_instruction the answer instruction after "?" ("Answer with only the location, ...")
  end_chat        chat-template tokens closing the user turn / opening the assistant turn
  end_self        the final token (the query position itself)
  (share_end / frac_end: the three end parts summed, for older plotting code)

Precedence when a token qualifies twice: locations > question > access_cue >
change_context > orig_context > other_story.

For each (cell, head, region) the CSV holds
  share_{region}: mean over prompts of the head's attention mass in the region
  frac_{region}:  mean over prompts of the region's share of the prompt's tokens
The ratio share/frac is attention relative to uniform (1 = no preference).

Location strings: the capture stored only the first token of each location. The
string is recovered by matching the longest entry of the template vocab files at
that token (the template's own list in tom_datasets); all whole-word occurrences of it in the story are then labelled. If
no entry matches, that single token is used and the prompt counts in n_loc_fallback.

CPU only; reads {capture_dir}/attention.pt. Writes one long CSV plus an example
labelling per cell (--examples) for checking.
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("region_attention_shares")

REGIONS = ["orig_loc", "current_loc", "orig_context", "access_cue", "change_context",
           "other_story", "question", "system", "end_instruction", "end_chat", "end_self"]
END_PARTS = ["end_instruction", "end_chat", "end_self"]  # also written summed as share_end / frac_end
# Token roles inside the state-change ("move") sentence, the sentence holding the first
# mention of the updated location. Labelled separately from the regions above (a token has
# one region and, if it is in that sentence, one role); written as share_role_* / frac_role_*.
ROLES = ["agent", "verb", "object", "prep", "newloc", "other"]
ROLE_WORDS = {  # per template, from the move sentences in tom_templates.py
    "food_truck":   {"agent": ["Bob"], "verb": ["drives"], "object": ["truck"]},
    "hair_styling": {"agent": ["Alice", "she", "She"], "verb": ["dyes"], "object": ["hair"]},
    "library_book": {"agent": ["librarian"], "verb": ["returns", "moves", "move"], "object": ["book", "it"]},
}
PREPOSITIONS = {"to", "in"}  # only the word directly before the new location
# each template draws both locations from one vocab list
TEMPLATE_VOCAB = {"food_truck": "streets.txt", "hair_styling": "hair_colors.txt",
                  "library_book": "library_sections.txt"}
USER_TAG = "<|im_start|>user\n"
# true_belief_marker of each template, cma_scripts/tasks/causal_analysis/prompt_generators/tom_templates.py
TRUE_BELIEF_MARKERS = ["Alice sees Bob moving the truck", "immediately calls Jennifer to tell her",
                       "watches the librarian take the book"]


def load_vocab(vocab_dir: Path) -> dict[str, list[str]]:
    """template -> its location strings, longest first."""
    out = {}
    for template, f in TEMPLATE_VOCAB.items():
        words = {w.strip() for w in (vocab_dir / f).read_text().splitlines() if w.strip()}
        out[template] = sorted(words, key=len, reverse=True)
    return out


def tok_start(prompt: str, offsets, i: int) -> int:
    s, e = offsets[i]
    while s < e and prompt[s].isspace():
        s += 1
    return s


def match_location(prompt: str, offsets, tok: int, vocab: list[str]) -> str | None:
    s = tok_start(prompt, offsets, tok)
    for w in vocab:
        if prompt.startswith(w, s):
            return w
    return None


def label_prompt(prompt: str, pos: dict, tokenizer, vocab: list[str], context: str,
                 template: str | None = None):
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    T = len(offsets)
    assert T == pos["seq_len"], f"token count {T} != stored {pos['seq_len']}"

    u0 = prompt.find(USER_TAG)
    u0 = u0 + len(USER_TAG) if u0 >= 0 else 0
    q1 = prompt.rfind("?") + 1
    q0 = prompt.rfind("\n", u0, q1 - 1) + 1
    has_q = q1 > 0 and q0 > u0
    if not has_q:
        q0 = q1 = prompt.find("<|im_end|>", u0)
    story = (u0, q0)

    def toks(c0, c1):
        return [i for i, (s, e) in enumerate(offsets) if e > s and e > c0 and s < c1]

    lab = np.full(T, "", dtype=object)
    story_toks = toks(*story)
    first_story = min(story_toks) if story_toks else T
    last_q = max(toks(q0, q1)) if has_q else first_story - 1
    lab[:first_story] = "system"
    # end of prompt, split: the answer instruction after "?", the chat-template tokens
    # closing the user turn and opening the assistant turn, and the final token itself
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = enc["input_ids"]
    close = max((i for i in range(last_q + 1, T) if ids[i] == im_end), default=T - 1)
    for i in range(last_q + 1, T):
        lab[i] = "end_self" if i == T - 1 else ("end_instruction" if i < close else "end_chat")
    for i in story_toks:
        lab[i] = "other_story"

    # sentences of the story, as char spans
    sents = [(story[0] + m.start(), story[0] + m.end())
             for m in re.finditer(r"[^.!?\n]+[.!?]*", prompt[story[0]:story[1]])]

    def sentence_of(tok_idx):
        if tok_idx is None or not (0 <= tok_idx < T):
            return None
        c = tok_start(prompt, offsets, tok_idx)
        return next((sp for sp in sents if sp[0] <= c < sp[1]), None)

    # Location strings. The stored anchors come from prompt.find(loc_str), which can
    # hit a substring elsewhere ("art" inside "<|im_start|>"), so they are only a
    # hint: the correct location is base_ans; the other is the vocab entry at its
    # anchor, else the single remaining whole-word vocab entry in the story.
    story_txt = prompt[story[0]:story[1]]

    def occurrences(w):
        return [(story[0] + m.start(), story[0] + m.end())
                for m in re.finditer(r"(?<!\w)" + re.escape(w) + r"(?!\w)", story_txt)]

    correct_is_first = pos["state_correct"] == pos["state_first"]
    ans = (pos.get("base_ans") or "").strip()
    strings, fallback = {}, 0
    for name, key in (("orig_loc", "state_first"), ("current_loc", "state_second")):
        is_correct = (name == "orig_loc") == correct_is_first
        w = ans if is_correct and ans and occurrences(ans) else \
            (match_location(prompt, offsets, pos[key], vocab) if 0 <= pos[key] < T else None)
        strings[name] = w
    for name in strings:
        if strings[name] is None or not occurrences(strings[name]):
            other = strings["current_loc" if name == "orig_loc" else "orig_loc"]
            cands = [w for w in vocab if occurrences(w) and w != other]
            cands = [w for w in cands if not any(w != v and w in v for v in cands)]
            if len(cands) > 1:  # e.g. "a history book from the art section": prefer "<w> section"
                cands = [w for w in cands
                         if re.search(r"(?<!\w)" + re.escape(w) + r" section", story_txt)] or cands
            strings[name] = cands[0] if len(cands) == 1 else None
            fallback += 1
    first_tok = {n: (toks(*occurrences(w)[0])[0] if w and occurrences(w) else None)
                 for n, w in strings.items()}

    # Access event: where the agent's epistemic access is decided.
    #   photo: the photo-taking sentence (holds char_q, "photographer")
    #   belief, false-belief: the info-loss sentence ("...is unable to see the truck")
    #   belief, true-belief: the sentence where the agent observes the change (belief_form,
    #   e.g. "Alice sees Bob moving the truck."); an info-loss sentence earlier in a
    #   true-belief story stays in other_story
    true_belief = pos["state_correct"] == pos["state_second"]
    if context == "photo":
        # by text, not the char_q anchor: captures made before the photo char_q
        # lookup existed store char_q = -1
        m = re.search(r"(?<!\w)photographer(?!\w)", prompt[story[0]:story[1]])
        access_tok = toks(story[0] + m.start(), story[0] + m.end())[0] if m else -1
    elif true_belief:
        # The stored belief_form anchor is unreliable here: the detector checks the
        # false-belief marker first, and food_truck true-belief stories also contain
        # the info-loss sentence. Find the template's true-belief marker directly.
        m = next((i for i in (prompt.find(mk, story[0], story[1]) for mk in TRUE_BELIEF_MARKERS)
                  if i >= 0), -1)
        access_tok = toks(m, m + 1)[0] if m >= 0 else -1
    else:
        access_tok = pos.get("info_loss", -1)
    for name, tok_idx in (("orig_context", first_tok["orig_loc"]),
                          ("change_context", first_tok["current_loc"]),
                          ("access_cue", access_tok)):
        sp = sentence_of(tok_idx)
        if sp:
            for i in toks(*sp):
                lab[i] = name
    if has_q:
        for i in toks(q0, q1):
            lab[i] = "question"

    for name, w in strings.items():
        if w is None:
            continue
        for c0, c1 in occurrences(w):
            for i in toks(c0, c1):
                lab[i] = name
    assert (lab != "").all(), "unlabelled tokens"

    role = np.full(T, "", dtype=object)
    move = sentence_of(first_tok["current_loc"])
    if move and template in ROLE_WORDS:
        m0, m1 = move
        for i in toks(m0, m1):
            role[i] = "other"
        for kind, words in ROLE_WORDS[template].items():
            for w in words:
                for m in re.finditer(r"(?<!\w)" + re.escape(w) + r"(?!\w)", prompt[m0:m1]):
                    for i in toks(m0 + m.start(), m0 + m.end()):
                        role[i] = kind
        loc = [(c0, c1) for c0, c1 in occurrences(strings["current_loc"]) if m0 <= c0 < m1] \
            if strings["current_loc"] else []
        for c0, c1 in loc:
            for i in toks(c0, c1):
                role[i] = "newloc"
        if loc:
            before = toks(loc[0][0], loc[0][1])[0] - 1
            word = lambda i: prompt[offsets[i][0]:offsets[i][1]].strip()
            if before >= 1 and word(before) == "the":  # "returns it to the history section"
                before -= 1
            if before >= 0 and word(before) in PREPOSITIONS:
                role[before] = "prep"
    return lab, fallback, has_q, enc["input_ids"], role


def analyze_cell(capture_dir: Path, tokenizer, vocab, examples: list) -> pd.DataFrame:
    payload = torch.load(capture_dir / "attention.pt", map_location="cpu", weights_only=False)
    attn = payload["attn"].float()
    q_idx = payload.get("query_pos_names", ["lookback"]).index("lookback")
    heads = pd.DataFrame(payload["heads"])
    meta = payload.get("meta", {})
    cond = meta.get("condition", capture_dir.parts[-4])
    context = "photo" if "photo" in cond else "belief"
    template = meta.get("template", capture_dir.parts[-3].replace("template_", ""))
    H = attn.shape[1]

    share = np.zeros((H, len(REGIONS)))
    frac = np.zeros(len(REGIONS))
    rshare = np.zeros((H, len(ROLES)))
    rfrac = np.zeros(len(ROLES))
    n_ok = n_fb = n_noq = 0
    answer_slot = set()
    for pi, (prompt, pos) in enumerate(zip(payload["prompts"], payload["positions"])):
        lab, fb, has_q, ids, role = label_prompt(prompt, pos, tokenizer, vocab[template], context, template)
        rows = attn[pi, :, q_idx, :pos["seq_len"]].numpy()
        if not np.all(np.isfinite(rows)):
            continue
        for r, name in enumerate(REGIONS):
            m = lab == name
            share[:, r] += rows[:, m].sum(1)
            frac[r] += m.mean()
        for r, name in enumerate(ROLES):
            m = role == name
            rshare[:, r] += rows[:, m].sum(1)
            rfrac[r] += m.mean()
        if pi == 0:
            examples.append(f"\n=== {capture_dir}\n" + " ".join(
                f"{tokenizer.decode([t])!r}:{l}" for t, l in zip(ids, lab)))
        n_ok += 1
        n_fb += fb > 0
        n_noq += not has_q
        answer_slot.add("original" if pos["state_correct"] == pos["state_first"] else "current")

    share /= max(n_ok, 1)
    frac /= max(n_ok, 1)
    rshare /= max(n_ok, 1)
    rfrac /= max(n_ok, 1)
    out = heads[["layer", "head"]].copy()
    for col in ("observed_score", "p_value", "significant", "groups"):
        if col in heads.columns:
            out[col] = heads[col].values
    for r, name in enumerate(REGIONS):
        out[f"share_{name}"] = share[:, r]
        out[f"frac_{name}"] = frac[r]
    for r, name in enumerate(ROLES):
        out[f"share_role_{name}"] = rshare[:, r]
        out[f"frac_role_{name}"] = rfrac[r]
    out["share_end"] = out[[f"share_{n}" for n in END_PARTS]].sum(1)
    out["frac_end"] = out[[f"frac_{n}" for n in END_PARTS]].sum(1)
    out["condition"] = cond
    out["template"] = template
    out["base_rule"] = meta.get("base_rule")
    out["exp_rule"] = meta.get("exp_rule")
    out["answer_slot"] = "/".join(sorted(answer_slot))
    out["n_prompts"] = n_ok
    out["n_loc_fallback"] = n_fb
    out["n_no_question"] = n_noq
    rs = share.sum(1)
    log.info(f"{capture_dir}: {n_ok} prompts, loc fallback {n_fb}, no question {n_noq}, "
             f"answer {answer_slot}, row sums {rs.min():.3f}-{rs.max():.3f}, "
             f"token fractions " + " ".join(f"{n}={f:.3f}" for n, f in zip(REGIONS, frac)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/lookback_analysis"))
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--vocab_dir", type=Path, default=Path("tom_datasets"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = load_vocab(args.vocab_dir)
    log.info("location vocab sizes: " + str({k: len(v) for k, v in vocab.items()}))
    frames, examples = [], []
    for cond in args.conditions:
        for pt in sorted((args.root / cond).glob("template_*/base_rule_*/patch_before_movement/attention.pt")):
            frames.append(analyze_cell(pt.parent, tok, vocab, examples))
    df = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    args.out.with_name(args.out.stem + "_example_labels.txt").write_text("\n".join(examples))
    log.info(f"-> {args.out}  ({len(df)} rows)")
    log.info("\n" + df.groupby(["condition", "template", "base_rule"])[
        ["n_prompts", "n_loc_fallback", "n_no_question"]].first().to_string())


if __name__ == "__main__":
    main()
