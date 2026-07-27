"""Aggregate cross-model behavioral performance on the ToM stimuli.

Answers "how well did each model we tested actually do the task?" from data
already on disk. Four independent sources:

  1. raw   -- results/tom_performance/raw_responses/*.json
              Per-item records with belief_correct / world_correct / malformed.
              Uniform across models, so this is the comparable sweep. Pools all
              runs sharing (model, condition, format, temperature); runs used
              different system-prompt styles, which the raw filenames do NOT
              record, so these are style-mixed numbers.

  2. style -- results/tom_performance/*_behavioral_results.json
              The only source that separates the "detailed" vs "standard"
              system-prompt styles. Three schema generations exist; we read the
              Sept 2025 one keyed {cond}_{fmt}_{style}_T{temp} and skip the rest.

  3. gate  -- results/causal_analysis/cma/**/sample_num_{K}_gen_acc_0.9/
              CMA's own behavioral gate. K = pairs where the model answered BOTH
              base and exp correctly; the requested count comes from the sibling
              model_responses_{N}_base.txt. K/N is a paired accuracy on the exact
              CMA stimuli -- the only ToM/photo metric available for Gemma-2-27B.

  4. rsa   -- results/rsa_head_outputs/activations_code_*_full/{model}/*.pt
              Free-generation accuracy on the RSA code grids, saved because the
              captures ran with --decode_answers. The only Qwen-vs-Gemma code
              comparison we have. Opt-in (--table rsa): loads big tensors.

Usage:
    python scripts/aggregate_model_performance.py                  # all but rsa
    python scripts/aggregate_model_performance.py --table gate_summary
    python scripts/aggregate_model_performance.py --table rsa
    python scripts/aggregate_model_performance.py --out results/aggregated/model_performance
"""

import argparse
import collections
import csv
import glob
import json
import os
import re

RAW_RE = re.compile(
    r"(?P<model>.+?)_T(?P<temp>[\d.]+)_(?P<cond>.+?)_"
    r"(?P<fmt>direct|multiple_choice|unknown)_responses_"
)
STYLE_RE = re.compile(
    r"^(?P<cond>true_belief|false_belief)_(?P<fmt>direct|multiple_choice)_"
    r"(?P<variant>abstract_)?(?P<style>detailed|standard)_T(?P<temp>[\d.]+)$"
)
GATE_RE = re.compile(r"sample_num_(\d+)_gen_acc")
REQ_RE = re.compile(r"model_responses_(\d+)_base")

# Condition -> stimulus family. "legacy" folders are duplicate aliases or
# content-invalid (see CLAUDE.md, "Legacy _context folders") and are dropped
# unless --include-legacy is passed.
CONDITION_FAMILY = {
    "abstract_belief": "tom",
    "answer_changes_belief": "tom",
    "answer_changes_location_only": "tom",
    "state_only_no_belief": "tom",
    "abstract_photo": "photo",
    "abstract_photo_new": "photo",
    "answer_changes_photo": "photo",
    "answer_changes_photo_new": "photo",
    "answer_changes_location_only_photo": "photo",
    "code_abstract": "code",
    "code_answer_changes": "code",
    "code_control": "code",
    "abstract_context": "legacy",
    "control_context": "legacy",
    "photo_context": "legacy",
    "control": "legacy",
}


def collect_raw(root):
    """Per-item accuracy pooled over runs, from raw_responses/."""
    agg = collections.defaultdict(collections.Counter)
    for path in glob.glob(os.path.join(root, "tom_performance", "raw_responses", "*_responses_*.json")):
        m = RAW_RE.match(os.path.basename(path))
        if not m or m.group("cond") == "unknown":
            continue
        with open(path) as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            continue
        key = (m.group("model"), m.group("cond"), m.group("fmt"), float(m.group("temp")))
        counter = agg[key]
        counter["runs"] += 1
        for rec in records:
            counter["n"] += 1
            counter["belief"] += bool(rec.get("belief_correct"))
            counter["world"] += bool(rec.get("world_correct"))
            counter["malformed"] += bool(rec.get("malformed"))

    rows = []
    for (model, cond, fmt, temp), c in sorted(agg.items()):
        n = c["n"] or 1
        rows.append({
            "model": model, "condition": cond, "format": fmt, "temperature": temp,
            "runs": c["runs"], "n": c["n"],
            "belief_acc": round(c["belief"] / n, 4),
            "world_acc": round(c["world"] / n, 4),
            "malformed_rate": round(c["malformed"] / n, 4),
        })
    return rows


def collect_style(root):
    """Prompt-style-resolved accuracy from the Sept 2025 aggregate JSONs."""
    agg = collections.defaultdict(list)
    pattern = os.path.join(root, "tom_performance", "*_behavioral_results.json")
    for path in sorted(glob.glob(pattern)):
        base = os.path.basename(path)
        model = base.split("_2025")[0]
        if model == "combined_results":  # nested-by-model duplicate of the per-model files
            continue
        with open(path) as fh:
            payload = json.load(fh)
        for key, val in payload.items():
            m = STYLE_RE.match(key)
            if not m or not isinstance(val, dict) or "belief_accuracy" not in val:
                continue
            agg[(model, m.group("cond"), m.group("fmt"),
                 "abstract" if m.group("variant") else "template",
                 m.group("style"), float(m.group("temp")))].append(val)

    rows = []
    for (model, cond, fmt, variant, style, temp), vals in sorted(agg.items()):
        k = len(vals)
        rows.append({
            "model": model, "condition": cond, "format": fmt, "stimuli": variant,
            "prompt_style": style, "temperature": temp, "runs": k,
            "n_per_run": vals[0]["total_responses"],
            "belief_acc": round(sum(v["belief_accuracy"] for v in vals) / k, 4),
            "world_acc": round(sum(v["world_accuracy"] for v in vals) / k, 4),
            "malformed_rate": round(sum(v["malformed_rate"] for v in vals) / k, 4),
        })
    return rows


def collect_gate(root, include_legacy=False, exclude_models=()):
    """Paired base+exp pass rate implied by the CMA behavioral gate."""
    agg = collections.defaultdict(collections.Counter)
    pattern = os.path.join(root, "causal_analysis", "cma", "**", "model_responses_*_base.txt")
    for path in glob.glob(pattern, recursive=True):
        req = REQ_RE.search(os.path.basename(path))
        parent = os.path.dirname(path)
        kept = GATE_RE.search(parent)
        if not (req and kept):
            continue
        tail = parent.split(os.sep + "cma" + os.sep, 1)[1].split(os.sep)
        if len(tail) < 3:
            continue
        model, cond = "/".join(tail[:2]), tail[2]
        family = CONDITION_FAMILY.get(cond, "unknown")
        if family == "legacy" and not include_legacy:
            continue
        if any(x in model for x in exclude_models):
            continue
        c = agg[(model, family, cond)]
        c["kept"] += int(kept.group(1))
        c["requested"] += int(req.group(1))
        c["cells"] += 1

    rows = []
    for (model, family, cond), c in sorted(agg.items()):
        rows.append({
            "model": model, "family": family, "condition": cond, "cells": c["cells"],
            "pairs_kept": c["kept"], "pairs_requested": c["requested"],
            "pass_rate": round(c["kept"] / c["requested"], 4),
        })
    return rows


def summarize_gate(rows):
    """Roll the gate table up to model x family."""
    agg = collections.defaultdict(collections.Counter)
    for r in rows:
        c = agg[(r["model"], r["family"])]
        c["kept"] += r["pairs_kept"]
        c["requested"] += r["pairs_requested"]
        c["cells"] += r["cells"]
    return [{
        "model": model, "family": family, "cells": c["cells"],
        "pairs_kept": c["kept"], "pairs_requested": c["requested"],
        "pass_rate": round(c["kept"] / c["requested"], 4),
    } for (model, family), c in sorted(agg.items())]


def collect_rsa(root):
    """Free-generation accuracy on the RSA code grids, from --decode_answers.

    capture_head_outputs_full.py stores per-item `correct` when --decode_answers
    is passed. The Gemma code grids were captured with it, so this is the only
    Qwen-vs-Gemma comparison on code stimuli. Note these are the RSA grids
    (code_ledger / code_position / code_padtwins), NOT the CMA code_* prompt
    pairs, and it is single-prompt accuracy rather than paired base+exp.

    Loads the full payloads (up to ~400 MB each); import torch lazily so the
    other tables stay usable without it.
    """
    import numpy as np
    import torch

    rows = []
    pattern = os.path.join(root, "rsa_head_outputs", "activations_code_*_full", "*", "*.pt")
    for path in sorted(glob.glob(pattern)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "correct" not in payload:
            continue
        model = path.split("_full" + os.sep)[1].split(os.sep)[0]
        grid = os.path.splitext(os.path.basename(path))[0]
        correct = np.asarray(payload["correct"], dtype=float)
        stim = payload.get("stim") or []

        def cell(mask=None):
            sel = correct if mask is None else correct[mask]
            return round(float(sel.mean()), 4), int(sel.sum()), len(sel)

        acc, n_ok, n = cell()
        row = {"model": model, "grid": grid, "split": "all", "level": "-",
               "acc": acc, "correct": n_ok, "n": n}
        rows.append(row)
        if stim and "role_b" in stim[0]:
            for level in sorted({s["role_b"] for s in stim}):
                mask = np.array([s["role_b"] == level for s in stim])
                acc, n_ok, n = cell(mask)
                rows.append({"model": model, "grid": grid, "split": "role_b",
                             "level": level, "acc": acc, "correct": n_ok, "n": n})
        del payload
    return rows


def as_markdown(rows):
    if not rows:
        return "_(no rows)_"
    cols = list(rows[0])
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    out = ["| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="results",
                    help="tree to scan; 'other_results' holds the HPC-pulled mirror")
    ap.add_argument("--table", choices=["raw", "style", "gate", "gate_summary", "rsa", "all"], default="all",
                    help="'rsa' loads multi-hundred-MB tensors and is excluded from 'all'")
    ap.add_argument("--out", default=None, help="directory to write CSVs into")
    ap.add_argument("--include-legacy", action="store_true",
                    help="keep the _context alias / content-invalid condition folders")
    ap.add_argument("--exclude-models", nargs="*", default=["phi-4"],
                    help="substrings of model dirs to drop (default: phi-4, no longer an active arm)")
    args = ap.parse_args()

    def gate(root):
        return collect_gate(root, args.include_legacy, args.exclude_models)

    builders = {
        "raw": collect_raw,
        "style": collect_style,
        "gate": gate,
        "gate_summary": lambda root: summarize_gate(gate(root)),
        "rsa": collect_rsa,
    }
    # rsa is opt-in: it deserializes the full activation payloads.
    wanted = [k for k in builders if k != "rsa"] if args.table == "all" else [args.table]

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    for name in wanted:
        rows = builders[name](args.results_root)
        print(f"\n## {name}  ({len(rows)} rows, root={args.results_root})\n")
        print(as_markdown(rows))
        if args.out and rows:
            dest = os.path.join(args.out, f"model_performance_{name}.csv")
            with open(dest, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print(f"\n-> {dest}")


if __name__ == "__main__":
    main()
