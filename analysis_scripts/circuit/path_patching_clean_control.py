"""Replay clean receiver transplants using the completed study's exact batches.

No prompt generation, sender perturbation, or alteration of the original results.
Numerical failures are measurements: save them and continue through the full grid.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
CORE = "analysis_scripts/circuit/path_patching.py"
FIELDS = ("receivers", "receiver_aliases", "component", "routes")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_core(source):
    path = source / "code_snapshot" / CORE
    expected = read(source / "study.json")["source_sha256"][CORE]
    if sha(path) != expected:
        raise ValueError("Original executed path-patching script hash mismatch")
    spec = importlib.util.spec_from_file_location("original_path_patching", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def layouts(rows, batch_size, expected_senders):
    """Deduplicate identical full chunks; retain exact original row mapping."""
    if batch_size < 1:
        raise ValueError("Invalid source batch size")
    by_sender = defaultdict(list)
    for index, row in enumerate(rows):
        by_sender[row["senders"]].append((index, row))
    if set(by_sender) != set(expected_senders):
        raise ValueError("Source checkpoint does not cover the expected senders")
    unique: dict[str, dict[str, Any]] = {}
    for sender, entries in by_sender.items():
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            plans = [{k: row[k] for k in FIELDS} for _, row in chunk]
            key = json.dumps(plans, sort_keys=True)
            if key not in unique:
                unique[key] = {"plans": plans, "source_rows": [[] for _ in plans]}
            for offset, (index, _) in enumerate(chunk):
                unique[key]["source_rows"][offset].append({"sender": sender, "row_index": index})
    return list(unique.values())


def inspect_task(source, task_id, core):
    study = read(source / "study.json")
    cell = study["plan"]["cells"][task_id]["id"]
    directory = source / cell
    task = read(directory / "task.json")
    if task["task"]["task_id"] != task_id or task["batch_size"] != study["batch_size"]:
        raise ValueError("Source task/study mismatch")
    pairs_path = directory / "pairs.jsonl"
    pairs = [json.loads(s) for s in pairs_path.read_text().splitlines() if s.strip()]
    if len(pairs) != study["pairs_per_cell"]:
        raise ValueError("Source pair count mismatch")
    metadata = core.pair_provenance(pairs_path, study["plan"]["model"], study["plan"]["revision"])
    if metadata["status"] != "validated_export":
        raise ValueError("Missing validated original prompt export")
    files = [source / "study.json", directory / "task.json", pairs_path,
             pairs_path.with_suffix(".meta.json")]
    jobs = []
    for group in task["task"]["groups"]:
        folder = directory / group["scope"]
        manifest = folder / "manifest.json"
        identity = read(manifest)["identity"]
        args = identity["arguments"]
        if (identity["protocol"] != "frozen_writers_final_token_v3_batched"
                or identity["script_sha256"] != study["source_sha256"][CORE]
                or identity["pairs_sha256"] != sha(pairs_path)
                or identity["versions"] != study["versions"]
                or args["batch_size"] != study["batch_size"]
                or args["revision"] != study["plan"]["revision"]
                or args["model"] != study["plan"]["model"]
                or identity["senders"] != group["senders"]
                or identity["receivers"] != group["receivers"]):
            raise ValueError(f"Inconsistent original manifest: {manifest}")
        files += [manifest, folder / "model_config.json"]
        config = read(folder / "model_config.json")["config"]
        shape_model = SimpleNamespace(cfg=SimpleNamespace(**config))
        expected: dict[str, dict[str, Any]] = {}
        for receiver in group["receivers"]:
            for component in args["components"]:
                routes = core.route_spec(shape_model, group["senders"], [receiver], component, True)
                key = json.dumps(routes, sort_keys=True)
                label = core.head_label([receiver])
                if key in expected:
                    expected[key]["receiver_aliases"].append(label)
                else:
                    expected[key] = dict(receivers=label, receiver_aliases=[label], component=component, routes=routes)
        senders = [core.head_label([h]) for h in group["senders"]]
        for index, pair in enumerate(pairs):
            file = folder / f"pair_{index:06d}.json"
            saved = read(file)
            if saved["pair"] != pair:
                raise ValueError(f"Source pair mismatch: {file}")
            chunks = layouts(saved["rows"], args["batch_size"], senders)
            # Every sender must have the same complete receiver route ordering.
            canonical = [{k: r[k] for k in FIELDS} for r in saved["rows"] if r["senders"] == senders[0]]
            if canonical != list(expected.values()):
                raise ValueError(f"Source routes differ from frozen receiver grid: {file}")
            for sender in senders:
                if [{k: r[k] for k in FIELDS} for r in saved["rows"] if r["senders"] == sender] != canonical:
                    raise ValueError(f"Incomplete/inconsistent source routes: {file} {sender}")
            files.append(file)
            jobs.append({"file": file, "scope": group["scope"], "index": index,
                         "pair": pair, "chunks": chunks, "arguments": args,
                         "model_config": config,
                         "original_rows": len(saved["rows"])})
    return study, task, jobs, {str(p.relative_to(source)): sha(p) for p in files}


@torch.inference_mode()
def measure(core, model, tokens, pair, chunks, tolerance):
    target, original = pair["causal_token_id"], pair["original_token_id"]
    if target == original:
        raise ValueError("Zero-signal answer readout")
    names = {f"blocks.{r['layer']}.attn.hook_{r['component']}"
             for chunk in chunks for p in chunk["plans"] for r in p["routes"]}
    writers = {f"blocks.{layer}.{suffix}" for layer in range(model.cfg.n_layers)
               for suffix in ("attn.hook_z", "hook_mlp_out")}
    single_logits, clean = core.capture(model, tokens, names | writers)
    frozen_logits, frozen = core.capture(model, tokens, names, core.freeze_writers(clean, clean, []))
    frozen_error = float((frozen_logits - single_logits).abs().max())
    # Clean and frozen-clean caches sometimes differ; measure both if needed.
    same_cache = all(torch.equal(clean[n], frozen[n]) for n in names)
    baselines = {}
    for width in sorted({len(c["plans"]) + 1 for c in chunks}):
        baselines[width] = core.capture(model, tokens.repeat(width, 1), names)
    rows = []
    for chunk_id, chunk in enumerate(chunks):
        plans = chunk["plans"]
        width = len(plans) + 1
        baseline, natural = baselines[width]
        routes = [p["routes"] for p in plans]
        transplanted = core.evaluate(model, tokens.repeat(width, 1),
                                    core.patch_receivers_batched(frozen, routes))
        clean_transplanted = transplanted if same_cache else core.evaluate(
            model, tokens.repeat(width, 1), core.patch_receivers_batched(clean, routes))
        for row, plan in enumerate(plans, 1):
            effect = core.margin(transplanted, target, original, row) - core.margin(transplanted, target, original)
            error = float((transplanted[row] - transplanted[0]).abs().max())
            clean_effect = core.margin(clean_transplanted, target, original, row) - core.margin(clean_transplanted, target, original)
            clean_error = float((clean_transplanted[row] - clean_transplanted[0]).abs().max())
            deltas = []
            for route in plan["routes"]:
                name = f"blocks.{route['layer']}.attn.hook_{route['component']}"
                idx = route["hook_index"]
                deltas.append(float((frozen[name][0, -1, idx].float() - natural[name][row, -1, idx].float()).norm()))
            baseline_error = float((baseline[row] - baseline[0]).abs().max())
            reference_error = float((transplanted[0] - baseline[0]).abs().max())
            rows.append({**plan, "chunk_id": chunk_id, "batch_width": width, "batch_row": row,
                "source_rows": chunk["source_rows"][row - 1],
                "control_effect": effect, "max_logit_error": error,
                "clean_cache_control_effect": clean_effect, "clean_cache_max_logit_error": clean_error,
                "single_to_batch_receiver_delta_l2": deltas,
                "batch_clean_margin": core.margin(transplanted, target, original),
                "batch_row_max_logit_error": baseline_error,
                "untouched_reference_max_logit_error": reference_error,
                "batch_vs_single_max_logit_error": float((baseline[0] - single_logits[0]).abs().max()),
                "passes_tolerance": max(abs(effect), error, abs(clean_effect), clean_error,
                                        baseline_error, reference_error, frozen_error) <= tolerance})
    return {"rows": rows, "single_clean_margin": core.margin(single_logits, target, original),
            "clean_and_frozen_receiver_caches_identical": same_cache,
            "frozen_clean_max_logit_error": frozen_error}


def freeze(core, path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Changed inputs/settings: {path}; use a new output root")
    else:
        core.atomic_json(path, value)


def execute(args):
    source, output = Path(args.source_root).resolve(), Path(args.output_root).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Control output must be separate from original study")
    core = load_core(source)
    study, task, jobs, hashes = inspect_task(source, args.task_id, core)
    counts = {"cell": task["task"]["cell"]["id"], "source_path_rows": sum(j["original_rows"] for j in jobs),
              "clean_control_rows": sum(len(c["plans"]) for j in jobs for c in j["chunks"]),
              "selected_heads": sum(len(g["sender_pairs"]) for g in task["task"]["groups"]),
              "matched_controls": sum(len(g["sender_pairs"]) for g in task["task"]["groups"])}
    print(json.dumps(counts), flush=True)
    if args.preflight:
        return counts
    versions = core.provenance()["versions"]
    if versions != study["versions"]:
        raise ValueError(f"Dependency drift from original study: {versions} != {study['versions']}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".study.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        freeze(core, output / "study.json", {"protocol": "clean_receiver_transplant_v1",
            "source_root": str(source), "source_study_sha256": sha(source / "study.json"),
            "control_script_sha256": sha(Path(__file__)), "versions": versions,
            "tolerance": args.tolerance, "source_plan": study["plan"]})
    folder = output / counts["cell"]
    folder.mkdir(exist_ok=True)
    with (folder / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(core, folder / "manifest.json", {"source_sha256": hashes, "coverage": counts})
        model = None
        summaries = []
        for job in jobs:
            destination = folder / job["scope"]
            destination.mkdir(exist_ok=True)
            checkpoint = destination / job["file"].name
            if checkpoint.exists():
                saved = read(checkpoint)
                if saved["pair"] != job["pair"] or saved["source_sha256"] != sha(job["file"]):
                    raise ValueError(f"Checkpoint mismatch: {checkpoint}")
            else:
                if model is None:
                    model = core.prepare_model(argparse.Namespace(**job["arguments"]))
                    core.atomic_json(folder / "runtime.json", {"provenance": core.provenance(),
                        "gpu": torch.cuda.get_device_name(), "config": model.cfg.to_dict()})
                actual_config = json.loads(json.dumps(model.cfg.to_dict(), default=str))
                for key in ("n_layers", "n_heads", "n_key_value_heads", "d_model", "d_head",
                            "dtype", "ungroup_grouped_query_attention", "normalization_type",
                            "positional_embedding_type", "rotary_dim", "eps"):
                    if actual_config.get(key) != job["model_config"].get(key):
                        raise ValueError(f"Model configuration drift: {key}")
                pair = job["pair"]
                for answer, key in (("causal_ans", "causal_token_id"), ("exp_ans", "original_token_id")):
                    if model.to_tokens(pair[answer], prepend_bos=False)[0, 0].item() != pair[key]:
                        raise ValueError("Tokenizer/readout drift")
                tokens = model.tokenizer(pair["recipient_prompt"], return_tensors="pt").input_ids.to(job["arguments"]["device"])
                print(f"{job['scope']} pair {job['index']}: tokens={tuple(tokens.shape)} dtype={model.cfg.dtype}", flush=True)
                started = time.monotonic()
                saved = measure(core, model, tokens, pair, job["chunks"], args.tolerance)
                original_rows = read(job["file"])["rows"]
                saved["original_single_clean_margin"] = original_rows[0]["clean_margin"]
                saved["single_clean_margin_drift_from_original"] = saved["single_clean_margin"] - original_rows[0]["clean_margin"]
                for row in saved["rows"]:
                    row["original_batch_clean_margins"] = sorted({original_rows[m["row_index"]]["batch_clean_margin"] for m in row["source_rows"]})
                saved.update(pair=pair, source_sha256=sha(job["file"]), measurement_seconds=time.monotonic()-started)
                core.atomic_json(checkpoint, saved)
            bad = sum(not r["passes_tolerance"] for r in saved["rows"])
            summaries.append({"checkpoint": str(checkpoint.relative_to(folder)), "failed_routes": bad,
                              "routes": len(saved["rows"])})
            print(f"{checkpoint.name} {job['scope']}: {bad}/{len(saved['rows'])} failed clean-transplant tolerance", flush=True)
        core.atomic_json(folder / "summary.json", {"coverage": counts, "checkpoints": summaries,
            "status": "numerical_failures_detected" if any(s["failed_routes"] for s in summaries) else "passed"})
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=str(ROOT / "results/circuit/path_patching/union41_v2"))
    parser.add_argument("--output-root", default=str(ROOT / "results/circuit/path_patching/union41_v2_clean_control_v1"))
    parser.add_argument("--task-id", type=int, required=True, choices=range(14))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--preflight", action="store_true", help="Validate completed source and coverage; no model loading or writes")
    args = parser.parse_args()
    if not 0 <= args.tolerance < float("inf"):
        parser.error("Tolerance must be finite and nonnegative")
    execute(args)


if __name__ == "__main__":
    main()
