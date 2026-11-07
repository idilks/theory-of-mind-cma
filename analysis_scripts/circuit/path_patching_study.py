"""Frozen 41-sender study and matched controls; one SLURM array task per cell.

Each task loads the model once and measures every target and control sender
in its cell, in two receiver groups (layer-46 senders read layer 47).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
from pathlib import Path
import random
from typing import Any

from path_patching import (ROOT, atomic_json, digest, export_pairs, head_label,
                           pair_provenance, provenance, read_heads, run)

# Receiver groups: (scope, plan key for receivers, whether a sender pair belongs).
GROUPS = (("existing_candidates", "candidate_receivers", lambda layer: layer != 46),
          ("exploratory_layer47", "late_receivers", lambda layer: layer == 46))

DEFAULT_PLAN = ROOT / "results/aggregated/path_patching_setup/plan.json"
MODEL_REVISION = "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8"


def build_plan(membership: Path, readers: Path, seed: int) -> dict[str, Any]:
    with membership.open() as f:
        rows = sorted(csv.DictReader(f), key=lambda r: (int(r["layer"]), int(r["head"])))
    selected = {(int(r["layer"]), int(r["head"])) for r in rows}
    if len(rows) != 41 or len(selected) != 41:
        raise ValueError("Expected the frozen 41 distinct selected heads")
    if any(not (0 <= l < 48 and 0 <= h < 40) for l, h in selected):
        raise ValueError("Selected head outside Qwen bounds")
    receiver_heads = read_heads(readers)
    if any(not (0 <= l < 48 and 0 <= h < 40) for l, h in receiver_heads):
        raise ValueError("Receiver outside Qwen bounds")
    rng = random.Random(seed)
    controls = {}
    for layer in sorted({l for l, _ in selected}):
        targets = sorted(h for l, h in selected if l == layer)
        available = [h for h in range(40) if (layer, h) not in selected]
        if len(available) < len(targets):
            raise ValueError(f"Not enough distinct layer-matched controls at layer {layer}")
        for target, control in zip(targets, rng.sample(available, len(targets))):
            controls[layer, target] = [layer, control]
    sender_pairs = [{"target": [int(r["layer"]), int(r["head"])],
                     "control": controls[int(r["layer"]), int(r["head"])],
                     "membership": r["membership"]} for r in rows]
    cells = []
    for condition in ("C1", "C2", "K1"):
        templates = ("indexed_list",) if condition == "K1" else (
            "food_truck", "hair_styling", "library_book")
        for template in templates:
            for rule in ("ABA", "ABB"):
                cells.append({"id": f"{condition}_{template}_{rule}", "condition": condition,
                              "template": template, "base_rule": rule})
    return {"schema": "union41_matched_controls_v1", "model": "Qwen/Qwen2.5-14B-Instruct",
            "revision": MODEL_REVISION, "control_seed": seed, "prompt_seed": seed,
            "control_rule": "Uniform without replacement within layer; exclude entire 41-head union",
            "sender_pairs": sender_pairs, "cells": cells,
            "candidate_receivers": [list(h) for h in receiver_heads],
            "late_receivers": [[47, h] for h in range(40)],
            "late_receiver_scope": "Exploratory all-layer-47 search for layer-46 senders only",
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (membership, readers)}}


def task_spec(plan: dict[str, Any], task_id: int) -> dict[str, Any]:
    if not 0 <= task_id < len(plan["cells"]):
        raise ValueError("Task index outside full study grid")
    groups = []
    for scope, key, member in GROUPS:
        pairs = [p for p in plan["sender_pairs"] if member(p["target"][0])]
        senders = sorted(h for p in pairs for h in (p["target"], p["control"]))
        receivers = plan[key]
        if not pairs:
            continue
        if not receivers or min(r[0] for r in receivers) <= max(l for l, _ in senders):
            raise ValueError("Study contains a non-forward or empty path selection")
        groups.append({"scope": scope, "sender_pairs": pairs, "senders": senders,
                       "receivers": receivers})
    if sum(len(g["sender_pairs"]) for g in groups) != len(plan["sender_pairs"]):
        raise ValueError("Every sender pair must belong to exactly one receiver group")
    return {"task_id": task_id, "cell": plan["cells"][task_id], "groups": groups}


def freeze_plan(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing different frozen settings at {path}; use a new output root")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, value)


def prepare(path: Path, seed: int) -> None:
    plan = build_plan(ROOT / "results/aggregated/head_evidence_inventory/head_membership.csv",
                      ROOT / "results/aggregated/circuit/reader_heads.csv", seed)
    tasks = [task_spec(plan, i) for i in range(len(plan["cells"]))]
    freeze_plan(path, plan)
    with path.with_name("sender_control_map.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["target", "control", "layer", "membership", "receiver_scope"])
        for pair in plan["sender_pairs"]:
            writer.writerow([head_label([pair["target"]]), head_label([pair["control"]]),
                             pair["target"][0], pair["membership"],
                             "exploratory_layer47" if pair["target"][0] == 46 else "existing_candidates"])
    print(f"Frozen {len(plan['sender_pairs'])} targets + 41 controls; "
          f"{len(plan['cells'])} cells; {len(tasks)} array tasks -> {path}")


def execute_task(args: argparse.Namespace) -> None:
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    spec = task_spec(plan, args.task_id)
    if args.pairs_per_cell < 1:
        raise ValueError("pairs-per-cell must be positive")
    if args.dry_run:
        print(json.dumps({**spec, "pairs_per_cell": args.pairs_per_cell, "batch_size": args.batch_size,
                          "revision": plan["revision"]}, indent=2))
        return
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT / "analysis_scripts/circuit/path_patching.py",
               ROOT / "analysis_scripts/condition_registry.py",
               ROOT / "cma_scripts/tasks/causal_analysis/cma_config.py"]
    sources += sorted((ROOT / "cma_scripts/tasks/causal_analysis/prompt_generators").glob("*.py"))
    # All tasks in a study must use the same source/dependencies/sample settings.
    # Blocking lock is held only for a tiny manifest read/write, never for inference.
    with (output_root / ".study.lock").open("a") as study_lock:
        fcntl.flock(study_lock, fcntl.LOCK_EX)
        freeze_plan(output_root / "study.json", {
            "plan": plan, "plan_sha256": digest(plan_path),
            "pairs_per_cell": args.pairs_per_cell, "batch_size": args.batch_size,
            "versions": provenance()["versions"],
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources}})
    task_dir = output_root / spec["cell"]["id"]
    task_dir.mkdir(parents=True, exist_ok=True)
    with (task_dir / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another job is already using {task_dir}") from exc
        frozen = {"task": spec, "plan_sha256": digest(plan_path),
                  "study_script_sha256": digest(Path(__file__)), "pairs_per_cell": args.pairs_per_cell,
                  "batch_size": args.batch_size,
                  "model": plan["model"], "revision": plan["revision"],
                  "prompt_seed": plan["prompt_seed"]}
        freeze_plan(task_dir / "task.json", frozen)
        pairs_path = task_dir / "pairs.jsonl"
        common = {"model": plan["model"], "revision": plan["revision"],
                  "seed": plan["prompt_seed"], "allow_downloads": False}
        if not pairs_path.exists():
            export_pairs(argparse.Namespace(**common, **{k: spec["cell"][k] for k in
                ("condition", "template", "base_rule")}, n=args.pairs_per_cell, output=str(pairs_path)))
        export_metadata = pair_provenance(pairs_path, plan["model"], plan["revision"])
        if export_metadata["status"] != "validated_export":
            raise ValueError("Study requires a complete frozen export with its metadata sidecar")
        pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
        if len(pairs) != args.pairs_per_cell:
            raise ValueError("Frozen pair count does not match requested sample size")
        model = None
        for group in spec["groups"]:
            print(f"Task {args.task_id}: {spec['cell']['id']}; {group['scope']}: "
                  f"{len(group['senders'])} senders ({len(group['sender_pairs'])} target/control pairs); "
                  f"{len(group['receivers'])} receivers; {len(pairs)} pairs; "
                  f"pairs_sha256={digest(pairs_path)}", flush=True)
            model = run(argparse.Namespace(**common, command="run", pairs=str(pairs_path),
                senders=",".join(head_label([h]) for h in group["senders"]),
                sender_csv=None, receivers=",".join(head_label([h]) for h in group["receivers"]),
                receiver_csv=None, combine_senders=False, combine_receivers=False,
                components=["q", "k", "v"], allow_grouped_kv=True,
                device="cuda", n_devices=1, dtype="bfloat16", tolerance=1e-4,
                batch_size=args.batch_size, output=str(task_dir / group["scope"])), model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--prepare", action="store_true", help="Freeze selection locally, without model loading")
    parser.add_argument("--seed", type=int, default=20261022, help="Selection/export seed for --prepare")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--pairs-per-cell", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8, help="Receiver plans per forward pass")
    parser.add_argument("--output-root", default=str(ROOT / "results/circuit/path_patching/union41_v2"))
    parser.add_argument("--dry-run", action="store_true", help="Print one task without exporting or loading weights")
    args = parser.parse_args()
    if args.prepare:
        prepare(Path(args.plan), args.seed)
    elif args.task_id is None:
        parser.error("--task-id is required unless --prepare is used")
    else:
        execute_task(args)


if __name__ == "__main__":
    main()
