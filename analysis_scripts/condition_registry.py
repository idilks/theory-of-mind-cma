"""Canonical C1-C7 condition registry + shared CMA result-path resolution.

Single source of truth for the condition tables and seed/source-resolution
logic that were hand-rolled across the heatmap / aggregation scripts. Extracted
verbatim from build_pi_heatmaps.py / build_avg_heatmaps.py (see Refactor Debt
items 1-2 in CLAUDE.md) -- pure extraction, zero behavior change.

Canonical numbering == scripts/aggregate_conditions.py == the thesis:

    C1 abstract_belief                     (belief false<->true, answer SAME)
    C2 abstract_photo                      (photo  false<->true, answer SAME)
    C3 answer_changes_belief               (belief false<->true, answer DIFFERENT)
    C4 answer_changes_photo                (photo  false<->true, answer DIFFERENT)
    C5 answer_changes_location_only        (location only, ToM frame, answer DIFFERENT)
    C6 answer_changes_location_only_photo  (location only, photo frame, answer DIFFERENT)
    C7 state_only_no_belief                (state change, no belief, answer DIFFERENT)

Folder names are derived from `CMACondition` in the LLMSymbMech codebase; the
flip-vs-keep rule-pair split is derived from `CMACondition.within_context`
(within-context conditions keep the rule label, across-context flip it).

Other C# numberings exist in legacy scripts (build_pi_results.py,
compare_bf_conditions.py, visualize_conditions.py) -- do NOT trust those.

Run `python scripts/condition_registry.py` for a self-check that the registry
folder names match what is actually on disk under the Qwen results tree.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Mirror the sys.path pattern used by scripts/combine_per_sample_runs.py so we
# can import the canonical condition enum from the LLMSymbMech codebase.
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))
from cma_config import CMACondition  # noqa: E402

# ---------------------------------------------------------------------------
# Rule-pair sets (base_rule, exp_rule)
# ---------------------------------------------------------------------------
# C1-C4 patch ACROSS contexts (false_template <-> true_template) -> rule flips.
# C5-C7 patch WITHIN one context (no template flip; only subs differ) -> rule kept.
FLIP_RULES = {("ABA", "ABB"), ("ABB", "ABA")}
KEEP_RULES = {("ABA", "ABA"), ("ABB", "ABB")}


@dataclass(frozen=True)
class Condition:
    """One canonical condition cell in the C1-C7 registry."""
    cnum: str                  # "C1" .. "C7"
    condition: CMACondition    # backing enum member (source of folder name)
    label: str                 # human-readable label

    @property
    def folder(self) -> str:
        """Results-tree folder name (e.g. 'abstract_belief')."""
        return self.condition.value

    @property
    def key(self) -> str:
        """Manifest / aggregate_conditions key (e.g. 'C1_abstract_belief')."""
        return f"{self.cnum}_{self.folder}"

    @property
    def rule_pairs(self) -> set:
        """Allowed (base_rule, exp_rule) pairs: FLIP for across-context, KEEP for within."""
        return KEEP_RULES if self.condition.within_context else FLIP_RULES


REGISTRY: list[Condition] = [
    Condition("C1", CMACondition.ABSTRACT_BELIEF, "abstract belief"),
    Condition("C2", CMACondition.ABSTRACT_PHOTO, "abstract photo"),
    Condition("C3", CMACondition.ANSWER_CHANGES_BELIEF, "answer-changes belief"),
    Condition("C4", CMACondition.ANSWER_CHANGES_PHOTO, "answer-changes photo"),
    Condition("C5", CMACondition.LOCATION_ONLY_MATCHED_BELIEF, "answer-changes location-only (ToM frame)"),
    Condition("C6", CMACondition.LOCATION_ONLY_MATCHED_PHOTO, "answer-changes location-only (photo frame)"),
    Condition("C7", CMACondition.STATE_ONLY_NO_BELIEF, "state-only, no belief"),
]

BY_CNUM: dict[str, Condition] = {c.cnum: c for c in REGISTRY}
BY_KEY: dict[str, Condition] = {c.key: c for c in REGISTRY}
BY_FOLDER: dict[str, Condition] = {c.folder: c for c in REGISTRY}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------
def ordered_keys() -> list[str]:
    """['C1_abstract_belief', ...] in canonical order."""
    return [c.key for c in REGISTRY]


def short_labels() -> dict[str, str]:
    """key -> 'C1' short label (used for plot ticks)."""
    return {c.key: c.cnum for c in REGISTRY}


def across_context_keys() -> list[str]:
    """Keys that patch across contexts (rule flips): C1-C4."""
    return [c.key for c in REGISTRY if not c.condition.within_context]


def within_context_keys() -> list[str]:
    """Keys that patch within one context (rule kept): C5-C7."""
    return [c.key for c in REGISTRY if c.condition.within_context]


def heatmap_condition_table() -> dict:
    """key -> (folder, [alias_folders], allowed_rule_pairs).

    Reproduces the CONDITIONS dict in build_pi_heatmaps.py / build_avg_heatmaps.py.
    """
    return {c.key: (c.folder, [c.folder], c.rule_pairs) for c in REGISTRY}


# ---------------------------------------------------------------------------
# Result-path resolution (seed priority + sample_num_* glob)
# ---------------------------------------------------------------------------
SEED_PRIORITY = {"combined": 0, "1": 1, "0": 2}  # lower = preferred
SAMPLE_RE = re.compile(r"sample_num_(\d+)_(.+)")
RULE_PAIR_RE = re.compile(r"base_rule_(\w+)_exp_rule_(\w+)")


def resolve_best_source(
    results_root,
    folders,
    allowed_rule_pairs,
    patch_position: str = "patch_before_movement",
    seed_priority: dict = SEED_PRIORITY,
    require_per_sample: bool = False,
) -> dict:
    """Return {(template_dir, rule_dir): entry} picking the best source per cell.

    Best = lowest seed-priority (combined > seed_1 > seed_0), tie-broken by
    largest sample count. Walks `<results_root>/<folder>/.../causal_scores.pt`
    at the requested patch position, keeping only shuffle_False seeds whose tag
    is in `seed_priority` and whose (base_rule, exp_rule) is in
    `allowed_rule_pairs`.

    entry keys: scores_pt, per_sample_pt, csv, n, seed_tag, priority, source_dir.

    This is the shared implementation behind build_pi_heatmaps.find_best_source,
    build_avg_heatmaps.find_best_source, and the Gemma variant. Set
    `require_per_sample=True` to skip cells lacking causal_scores_per_sample.pt
    (Gemma tolerance behavior).
    """
    results_root = Path(results_root)
    best: dict[tuple[str, str], dict] = {}
    for folder in folders:
        cond_root = results_root / folder
        if not cond_root.exists():
            continue
        for scores_pt in cond_root.rglob("causal_scores.pt"):
            try:
                sample_dir = scores_pt.parent.name
                logit_dir = scores_pt.parent.parent.name
                seed_dir = scores_pt.parent.parent.parent.name
                patch_dir = scores_pt.parent.parent.parent.parent.name
                rule_dir = scores_pt.parent.parent.parent.parent.parent.name
                template_dir = scores_pt.parent.parent.parent.parent.parent.parent.name
            except IndexError:
                continue
            if logit_dir != "logit" or patch_dir != patch_position:
                continue
            if not seed_dir.startswith("z_seed_") or not seed_dir.endswith("_shuffle_False"):
                continue
            seed_tag = seed_dir[len("z_seed_"):-len("_shuffle_False")]
            if seed_tag not in seed_priority:
                continue
            m = SAMPLE_RE.match(sample_dir)
            if not m:
                continue
            rp = RULE_PAIR_RE.match(rule_dir)
            if not rp or (rp.group(1), rp.group(2)) not in allowed_rule_pairs:
                continue
            per_sample_pt = scores_pt.parent / "causal_scores_per_sample.pt"
            if require_per_sample and not per_sample_pt.exists():
                continue
            n = int(m.group(1))
            key = (template_dir, rule_dir)
            entry = {
                "scores_pt": scores_pt,
                "per_sample_pt": per_sample_pt,
                "csv": scores_pt.parent / "statistical_test_results.csv",
                "n": n,
                "seed_tag": seed_tag,
                "priority": seed_priority[seed_tag],
                "source_dir": scores_pt.parent,
            }
            cur = best.get(key)
            if cur is None or entry["priority"] < cur["priority"] or (
                entry["priority"] == cur["priority"] and entry["n"] > cur["n"]
            ):
                best[key] = entry
    return best


def pick_sample_dir(logit_dir, require_per_sample: bool = True):
    """Return (sample_dir, n_samples, eval_suffix) with the largest n under `logit_dir`.

    Globs `sample_num_<N>_<suffix>` children of a logit/ directory and returns
    the one with the most samples, or None if none qualify. Shared primitive
    behind combine_aliased_c2_c3.find_sample_dir. With `require_per_sample=True`
    only sample dirs containing causal_scores_per_sample.pt are considered.
    """
    logit_dir = Path(logit_dir)
    if not logit_dir.exists():
        return None
    candidates = []
    for d in logit_dir.iterdir():
        m = SAMPLE_RE.match(d.name)
        if not m:
            continue
        if require_per_sample and not (d / "causal_scores_per_sample.pt").exists():
            continue
        candidates.append((d, int(m.group(1)), m.group(2)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0]


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
def _self_check() -> int:
    print("Canonical condition registry (C1-C7):")
    for c in REGISTRY:
        rp = "FLIP" if c.rule_pairs is FLIP_RULES else "KEEP"
        print(f"  {c.cnum}  {c.folder:38s}  rules={rp}  key={c.key}")

    qwen = ROOT / "results/causal_analysis/cma/Qwen/Qwen2.5-14B-Instruct"
    print(f"\nchecking folder names against {qwen} ...")
    if not qwen.exists():
        print("  (results tree not present; skipping disk check)")
        return 0
    on_disk = {p.name for p in qwen.iterdir() if p.is_dir()}
    missing = [c.folder for c in REGISTRY if c.folder not in on_disk]
    for c in REGISTRY:
        mark = "OK" if c.folder in on_disk else "MISSING"
        print(f"  [{mark}] {c.folder}")
    assert not missing, f"registry folders missing on disk: {missing}"

    # cross-check against CMACondition context split
    assert across_context_keys() == [
        "C1_abstract_belief", "C2_abstract_photo",
        "C3_answer_changes_belief", "C4_answer_changes_photo",
    ], across_context_keys()
    assert within_context_keys() == [
        "C5_answer_changes_location_only",
        "C6_answer_changes_location_only_photo",
        "C7_state_only_no_belief",
    ], within_context_keys()
    print("\nself-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
