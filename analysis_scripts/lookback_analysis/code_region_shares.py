"""
Code-domain counterpart of region_attention_shares.py: per-head final-token
attention by prompt region, with region sizes, for the K1 (code_abstract,
divergence) and K5 (code_control, retrieval) heads.

Reads the full attention rows from capture_code_attention_rows.py and labels each
token of the padtwins prompts with the same region names as the story domains:

  orig_loc        the queried slot's ORIGINAL value in a's list literal
                  (the answer for copy queries)
  current_loc     the mutated value in the a[m] = 'X' line (the answer for alias queries)
  orig_context    rest of the `a = [...]` line
  access_cue      the copy / alias assignment lines (b = a.copy(), c = a, t = a, ...)
  change_context  rest of the `a[m] = 'X'` line
  other_story     pad lines (pad = [...], pad2 = [...])
  question        "What is b[3]?"
  system          everything before the code (system turn, chat header)
  end_instruction text between "?" and the end of the user turn (empty for these prompts)
  end_chat        chat-template tokens closing the user turn / opening the assistant turn
  end_self        the final token

Only queries of the mutated slot are kept (there the original and updated values
compete as answers). By default only unpadded stimuli are used, matching the CMA prompts.

Output CSV columns match region_attention_shares.py (share_*, frac_*), one row per
(head set, head, queried object).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
REGIONS = ["orig_loc", "current_loc", "orig_context", "access_cue", "change_context",
           "other_story", "question", "system", "end_instruction", "end_chat", "end_self"]
END_PARTS = ["end_instruction", "end_chat", "end_self"]  # also written summed as share_end / frac_end


# Token roles inside the mutation line `a[m] = 'X'`, the code counterpart of the story
# move-sentence roles in region_attention_shares.py; written as share_role_* / frac_role_*.
CODE_ROLES = ["var", "index", "assign", "quote", "newloc", "other"]


def code_roles(prompt: str, offsets, stim: dict) -> np.ndarray:
    """Role of each token in the a[m] = 'X' line ('' outside it)."""
    T = len(offsets)
    role = np.full(T, "", dtype=object)
    base = prompt.find(stim["scenario"])
    pos = base
    for line in stim["scenario"].split("\n"):
        if line.startswith("a["):
            c0, c1 = pos, pos + len(line)
            mut = base + int(stim["mut_char_pos"])
            close_bracket = line.index("]")
            for i, (s, e) in enumerate(offsets):
                if not (e > s and e > c0 and s < c1):
                    continue
                if s <= mut < e:
                    role[i] = "newloc"
                elif s < c0 + 1:
                    role[i] = "var"                       # the list name a
                elif c0 + 2 <= s < c0 + close_bracket:
                    role[i] = "index"                     # the digits inside [ ]
                elif "=" in prompt[s:e]:
                    role[i] = "assign"
                elif prompt[s:e].strip() == "'" and s < mut:
                    role[i] = "quote"                     # opening quote of the new value
                else:
                    role[i] = "other"                     # [ ] and the closing quote
        pos += len(line) + 1
    return role


def label_code_prompt(prompt: str, offsets, stim: dict) -> np.ndarray:
    T = len(offsets)
    base = prompt.find(stim["scenario"])
    assert base >= 0, "scenario not found in prompt"
    q0 = prompt.find(stim["question"], base + len(stim["scenario"]))
    q1 = q0 + len(stim["question"])
    assert q0 >= 0

    def toks(c0, c1):
        return [i for i, (s, e) in enumerate(offsets) if e > s and e > c0 and s < c1]

    lab = np.full(T, "", dtype=object)
    code_toks = toks(base, base + len(stim["scenario"]))
    q_toks = toks(q0, q1)
    lab[:min(code_toks)] = "system"
    # end of prompt: code prompts carry no answer instruction after the question (it is
    # in the system prompt), so this is the chat-template tail plus the final token
    close = prompt.rfind("<|im_end|>")
    for i in range(max(q_toks) + 1, T):
        lab[i] = "end_self" if i == T - 1 else (
            "end_instruction" if offsets[i][0] < close and offsets[i][1] > offsets[i][0] else "end_chat")
    for i in range(min(code_toks), max(q_toks) + 1):
        lab[i] = "question"  # the newline between code and question goes with the question

    pos = base
    for line in stim["scenario"].split("\n"):
        c0, c1 = pos, pos + len(line)
        if line.startswith("pad"):
            name = "other_story"
        elif line.startswith("a = ["):
            name = "orig_context"
        elif line.startswith("a["):
            name = "change_context"
        else:
            name = "access_cue"
        for i in toks(c0, c1 + 1):  # + the line's trailing newline
            lab[i] = name
        pos = c1 + 1
    for i in toks(q0, q1):
        lab[i] = "question"
    slot = base + int(stim["slot_char_pos"])
    mut = base + int(stim["mut_char_pos"])
    for i in toks(slot, slot + 1):
        lab[i] = "orig_loc"
    for i in toks(mut, mut + 1):
        lab[i] = "current_loc"
    assert (lab != "").all(), "unlabelled tokens"
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path,
                    default=ROOT / "results/rsa_head_outputs/activations_code_attn_rows/code_attention_rows.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "results/lookback_analysis/_regions/code_region_shares.csv")
    ap.add_argument("--include_padded", action="store_true")
    args = ap.parse_args()

    p = torch.load(args.rows, map_location="cpu", weights_only=False)
    attn = p["attn"].float().numpy()
    stim = pd.DataFrame(p["stim"])
    keep = stim.queried_slot_type == "mutated"
    if not args.include_padded:
        keep &= (stim.pad_len == 0) & (stim.suffix_pad_len == 0)
    idx = np.where(keep)[0]
    scores = {name: pd.read_csv(ROOT / f"results/aggregated/code_head_sets/{name}_head_list.csv")
              .set_index(["layer", "head"])["mean_score"] for name in ("code_abstract", "code_control")}

    labels, roles = {}, {}
    for si in idx:
        labels[si] = label_code_prompt(p["prompts"][si], p["offsets"][si], p["stim"][si])
        roles[si] = code_roles(p["prompts"][si], p["offsets"][si], p["stim"][si])

    rows = []
    for hi, h in enumerate(p["heads"]):
        for qobj in ("copy", "alias"):
            sel = [si for si in idx if stim.queried_object[si] == qobj]
            share = np.zeros(len(REGIONS))
            frac = np.zeros(len(REGIONS))
            rshare = np.zeros(len(CODE_ROLES))
            rfrac = np.zeros(len(CODE_ROLES))
            for si in sel:
                n = p["n_tokens"][si]
                row = attn[si, hi, :n]
                for r, name in enumerate(REGIONS):
                    m = labels[si] == name
                    share[r] += row[m].sum()
                    frac[r] += m.mean()
                for r, name in enumerate(CODE_ROLES):
                    m = roles[si] == name
                    rshare[r] += row[m].sum()
                    rfrac[r] += m.mean()
            share /= len(sel)
            frac /= len(sel)
            rshare /= len(sel)
            rfrac /= len(sel)
            for s in h["sets"].split("+"):
                rows.append({"condition": s, "layer": h["layer"], "head": h["head"],
                             "observed_score": scores[s].get((h["layer"], h["head"]), np.nan),
                             "significant": True, "in_both_sets": "+" in h["sets"],
                             "queried": qobj, "n_prompts": len(sel),
                             **{f"share_{n}": v for n, v in zip(REGIONS, share)},
                             **{f"frac_{n}": v for n, v in zip(REGIONS, frac)},
                             **{f"share_role_{n}": v for n, v in zip(CODE_ROLES, rshare)},
                             **{f"frac_role_{n}": v for n, v in zip(CODE_ROLES, rfrac)},
                             "share_end": sum(share[REGIONS.index(n)] for n in END_PARTS),
                             "frac_end": sum(frac[REGIONS.index(n)] for n in END_PARTS)})
    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    sums = df[[f"share_{n}" for n in REGIONS]].sum(1)
    print(f"-> {args.out}  ({len(df)} rows, {len(idx)} stimuli, row sums {sums.min():.3f}-{sums.max():.3f})")
    print("token fractions (copy):", df[df.queried == "copy"].iloc[0][[f"frac_{n}" for n in REGIONS]].round(3).to_dict())


if __name__ == "__main__":
    main()
