"""Offline position check for draft2 Figure 6; no model inference.

Primary test: within-template association of old-location distance to the
final prompt token with the change in old-location attention under TB
rank-1 injection at alpha=4. Two-sided permutation null: no association
within a template. This is observational; position and wording covary.
"""

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen2.5-14B-Instruct"
CAP = ROOT / "results/circuit/stage1_capture" / MODEL
RUN = ROOT / "results/circuit/stage2_steering" / MODEL / "L41_main"
OUT = ROOT / "results/aggregated/circuit/steering_position_check"
TEMPLATES = ("food_truck", "hair_styling", "library_book")
KEY = ["prompt_set", "template", "pair_idx"]
CONFIG = KEY + ["mode", "direction", "dose"]


def location_positions(prompt, location, offsets, narrative_only=False):
    spans = []
    start = 0
    while (start := prompt.find(location, start)) >= 0:
        spans.append((start, start + len(location)))
        start += 1
    if narrative_only:
        user_start = prompt.index("<|im_start|>user\n") + len("<|im_start|>user\n")
        question_start = prompt.rfind("\n", user_start, prompt.index("?", user_start))
        assert question_start > user_start
        spans = [m.span() for m in re.finditer(r"(?<!\w)" + re.escape(location) + r"(?!\w)", prompt)
                 if m.start() >= user_start and m.end() <= question_start]
    indices = [i for i, (a, b) in enumerate(offsets)
               if any(a < end and b > begin for begin, end in spans)]
    assert indices, (location, prompt)
    return dict(first=indices[0], last=indices[-1],
                distance=len(offsets) - 1 - indices[-1],
                mentions=len(spans), tokens=len(indices))


def within_template_test(frame, xcol, ycol, seed=20261022, n_perm=4999):
    """Correlate centered within-template ranks; shuffle y within template."""
    xs, ys, groups = [], [], []
    for _, group in frame.groupby("template", sort=True):
        x, y = rankdata(group[xcol]), rankdata(group[ycol])
        groups.append(np.arange(len(xs), len(xs) + len(group)))
        xs.extend(x - x.mean())
        ys.extend(y - y.mean())
    x, y = np.array(xs), np.array(ys)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return dict(x=xcol, y=ycol, n=len(frame), correlation=None, p=None)
    observed = float(x @ y / denom)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        yp = np.concatenate([rng.permutation(y[g]) for g in groups])
        exceed += abs(x @ yp / denom) >= abs(observed)
    return dict(x=xcol, y=ycol, n=len(frame), correlation=observed,
                p=(exceed + 1) / (n_perm + 1), permutations=n_perm)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/" + MODEL, local_files_only=True)
    directions = np.load(ROOT / "results/aggregated/circuit/stage1/directions_L41.npz")
    rows = []
    for template in TEMPLATES:
        cap = torch.load(CAP / f"{template}.pt", map_location="cpu", weights_only=False)
        hold = set(directions[f"hold_idx_{template}"].tolist())
        fit = set(directions[f"fit_idx_{template}"].tolist())
        assert not hold & fit
        for variant in ("false_belief", "true_belief"):
            behavior = {b["idx"]: b for b in cap["behavior"][variant]}
            assert hold | fit == set(behavior)
            for idx, prompt in enumerate(cap["prompts"][variant]):
                b = behavior[idx]
                old = b["correct"] if variant == "false_belief" else b["alt"]
                current = b["alt"] if variant == "false_belief" else b["correct"]
                encoded = tokenizer(prompt, add_special_tokens=False,
                                    return_offsets_mapping=True)
                assert len(encoded["input_ids"]) == b["token_len"]
                row = dict(prompt_set=variant, template=template, pair_idx=idx,
                           split="heldout" if idx in hold else "fit",
                           length=b["token_len"])
                for role, location in (("old", old), ("current", current)):
                    original = location_positions(prompt, location, encoded["offset_mapping"])
                    story = location_positions(prompt, location, encoded["offset_mapping"], True)
                    row.update({f"{role}_{k}": v for k, v in original.items()})
                    row.update({f"story_{role}_{k}": v for k, v in story.items()})
                    row[f"{role}_extra_matches"] = original["tokens"] != story["tokens"]
                rows.append(row)
        del cap
    pos = pd.DataFrame(rows)
    assert len(pos) == 900 and not pos[KEY].duplicated().any()
    pos.to_csv(OUT / "positions.csv", index=False)
    summary = pos.groupby(["split", "prompt_set", "template"])[
        ["length", "old_first", "current_first", "old_distance", "current_distance",
         "story_old_distance", "story_current_distance"]
    ].agg(["mean", "std", "min", "max"])
    summary.to_csv(OUT / "position_summary.csv")
    print("POSITION CHECK: 900 prompts; all token lengths match saved capture;")
    print("450 held-out prompts, 75 per variant/template. Positions are zero-based.")
    print(summary.loc["heldout"].round(2).to_string())

    att = pd.read_csv(RUN / "attention.csv").drop_duplicates(CONFIG + ["layer", "head"])
    clean = att.direction.eq("clean")
    selected = att.direction.eq("target_rank1") & att.dose.eq(4) & (
        (att.prompt_set.eq("true_belief") & att["mode"].eq("inject")) |
        (att.prompt_set.eq("false_belief") & att["mode"].eq("delete")))
    att = att[clean | selected].copy()
    assert att.groupby(CONFIG).size().eq(26).all()
    att["state"] = np.where(att.direction.eq("clean"), "clean", "steered")
    averaged = att.groupby(KEY + ["state"])[["mass_believed", "mass_current"]].mean()
    wide = averaged.unstack("state")
    wide.columns = [f"{metric}_{state}" for metric, state in wide.columns]
    wide = wide.reset_index()
    for role, metric in (("old", "mass_believed"), ("current", "mass_current")):
        wide[f"delta_{role}"] = wide[f"{metric}_steered"] - wide[f"{metric}_clean"]
    merged = pos[pos.split.eq("heldout")].merge(wide, on=KEY, validate="one_to_one")
    assert len(merged) == 450 and not merged.isna().any().any()

    behavior = pd.read_csv(RUN / "per_prompt.csv").drop_duplicates(CONFIG)
    behavior = behavior[behavior.direction.eq("clean") | (
        behavior.direction.eq("target_rank1") & behavior.dose.eq(4) & (
            (behavior.prompt_set.eq("true_belief") & behavior["mode"].eq("inject")) |
            (behavior.prompt_set.eq("false_belief") & behavior["mode"].eq("delete"))))].copy()
    behavior["state"] = np.where(behavior.direction.eq("clean"), "clean", "steered")
    bw = behavior.pivot(index=KEY, columns="state", values="flip_score")
    bw["delta_alt_probability"] = bw.steered - bw.clean
    merged = merged.merge(bw[["delta_alt_probability"]], on=KEY, validate="one_to_one")
    assert len(merged) == 450 and not merged.isna().any().any()
    merged.to_csv(OUT / "heldout_effects.csv", index=False)
    effects = merged.groupby(["prompt_set", "template"])[
        ["delta_old", "delta_current", "delta_alt_probability"]].mean()
    effects.to_csv(OUT / "effects_by_template.csv")
    print("\nMEAN CHANGES (attention and conditional alternative probability, range -1 to 1):")
    print(effects.round(5).to_string())
    tests = []
    for variant, distance, effect in (
        ("true_belief", "old_distance", "delta_old"),
        ("false_belief", "current_distance", "delta_current")):
        frame = merged[merged.prompt_set.eq(variant)]
        for outcome in (effect, "delta_alt_probability"):
            result = within_template_test(frame, distance, outcome)
            result.update(prompt_set=variant,
                          primary=variant == "true_belief" and outcome == "delta_old")
            tests.append(result)
    # Original attention contains all literal matches. Exclude prompts where
    # these differ from whole-word narrative mentions; saved aggregate masses
    # cannot be corrected without another attention capture.
    valid = merged[~(merged.old_extra_matches | merged.current_extra_matches)]
    print("\nEXCLUDED FROM SENSITIVITY CHECK:")
    print(merged.loc[merged.old_extra_matches | merged.current_extra_matches,
                     KEY + ["old_extra_matches", "current_extra_matches"]].to_string(index=False))
    for variant, distance, effect in (
        ("true_belief", "story_old_distance", "delta_old"),
        ("false_belief", "story_current_distance", "delta_current")):
        frame = valid[valid.prompt_set.eq(variant)]
        for outcome in (effect, "delta_alt_probability"):
            result = within_template_test(frame, distance, outcome)
            result.update(prompt_set=variant, primary=False,
                          sensitivity="exclude non-narrative or substring matches")
            tests.append(result)
    for template, frame in valid[valid.prompt_set.eq("true_belief")].groupby("template"):
        result = within_template_test(frame, "story_old_distance", "delta_old")
        result.update(prompt_set="true_belief", primary=False, template=template,
                      sensitivity="per-template, exclude extra matches")
        tests.append(result)
    report = dict(
        primary_hypothesis="Old location farther from prompt end predicts less attention gained under true-belief injection (negative correlation).",
        null="No association within story template; two-sided within-template permutation test.",
        attention="Sum over all tokens of all mentions, averaged over 26 heads; changes in fraction units.",
        behavior="Change in p(alternative)/(p(alternative)+p(correct)); not accuracy.",
        limitations=["Observational associations do not establish a positional cause.",
                     "Positions were not experimentally varied; wording and length covary.",
                     "No direct test of information lost by averaging the steering direction.",
                     "Original attention includes substring and answer-instruction matches for some prompts; sensitivity excludes those prompts.",
                     "Only the attention test for true-belief injection is primary; other tests are exploratory."],
        tests=tests)
    (OUT / "tests.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nWITHIN-TEMPLATE RANK CORRELATIONS (range -1 to 1; two-sided permutation p):")
    print(pd.DataFrame(tests).to_string(index=False))
    print("\nSaved:", OUT)


if __name__ == "__main__":
    main()
