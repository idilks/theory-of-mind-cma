"""New frozen-writer direct path assay; not the lost historical experiment.

All edits are at the final token; the protocol is summarised in analysis_scripts/README.md.
"""
from __future__ import annotations

import argparse
import csv
from enum import Enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "analysis_scripts"), str(ROOT / "cma_scripts"),
                str(ROOT / "cma_scripts/tasks/causal_analysis")]

Head = tuple[int, int]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict[str, Any]:
    # The Andes copy is rsynced without .git (push.sh), and study tasks run from
    # a code snapshot; git info is recorded when available, never required.
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                                  capture_output=True, text=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"git_revision": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain"),
            "script_sha256": digest(Path(__file__)),
            "command": sys.argv,
            "versions": {p: importlib.metadata.version(p) for p in
                         ("torch", "transformer-lens", "transformers")}}


def atomic_json(path: Path, obj: Any) -> None:
    def serialize(value: Any) -> str:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (torch.dtype, torch.device, Path)):
            return str(value)
        raise TypeError(f"Cannot serialize {type(value).__name__}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(obj, f, indent=2, allow_nan=False, default=serialize)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read_heads(path: Path) -> list[Head]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    heads = sorted({(int(row["layer"]), int(row["head"])) for row in rows})
    if not heads:
        raise ValueError(f"Empty head file: {path}")
    return heads


def parse_heads(value: str) -> list[Head]:
    import re
    heads = []
    for item in value.split(","):
        match = re.fullmatch(r"L(\d+)H(\d+)", item.strip())
        if not match:
            raise ValueError(f"Expected L29H31,L35H30; got {item!r}")
        heads.append((int(match[1]), int(match[2])))
    return sorted(set(heads))


def head_label(heads: list[Head]) -> str:
    return "+".join(f"L{layer}H{head}" for layer, head in heads)


def route_spec(model: Any, senders: list[Head], receivers: list[Head],
               component: str, allow_grouped_kv: bool = False) -> list[dict[str, Any]]:
    if component not in ("q", "k", "v") or not senders or not receivers:
        raise ValueError("Nonempty senders/receivers and component q/k/v required")
    for layer, head in senders + receivers:
        if not (0 <= layer < model.cfg.n_layers and 0 <= head < model.cfg.n_heads):
            raise ValueError(f"Head out of bounds: L{layer}H{head}")
    if max(layer for layer, _ in senders) >= min(layer for layer, _ in receivers):
        raise ValueError("Every sender must be strictly earlier than every receiver")
    if model.cfg.ungroup_grouped_query_attention:
        raise ValueError("Native grouped K/V required; disable ungroup_grouped_query_attention")
    n_heads = model.cfg.n_heads
    n_kv = model.cfg.n_key_value_heads or n_heads
    grouped = component != "q" and n_heads != n_kv
    if grouped and not allow_grouped_kv:
        raise ValueError("K/V edits affect a native GQA group: pass --allow-grouped-kv explicitly")
    group_size = n_heads // n_kv
    routes: dict[tuple[int, int], dict[str, Any]] = {}
    for layer, head in receivers:
        index = head // group_size if component != "q" else head
        siblings = list(range(index * group_size, (index + 1) * group_size)) if component != "q" else [head]
        routes[layer, index] = {"layer": layer, "component": component,
                                "hook_index": index, "query_heads_affected": siblings,
                                "native_kv_group": grouped}
    return list(routes.values())


def capture(model: Any, tokens: torch.Tensor, names: set[str],
            hooks: list | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Cache on the model device (a few hundred MB per prompt), so freeze hooks
    do not copy every writer from CPU on each forward; keep only final logits."""
    with model.hooks(fwd_hooks=hooks or []):
        logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names,
                                             return_cache_object=False)
    last = logits[:, -1].detach().float().cpu()
    del logits
    return last, cache


def evaluate(model: Any, tokens: torch.Tensor, hooks: list) -> torch.Tensor:
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    last = logits[:, -1].detach().float().cpu()
    del logits
    return last


def patch_senders(cache: dict[str, torch.Tensor], senders: list[Head]) -> list:
    by_layer: dict[int, list[int]] = {}
    for layer, head in senders:
        by_layer.setdefault(layer, []).append(head)
    hooks = []
    for layer, heads in by_layer.items():
        name = f"blocks.{layer}.attn.hook_z"
        def replace(value, hook, heads=heads, name=name):
            result = value.clone()
            result[:, -1, heads] = cache[name][:, -1, heads].to(value)
            return result
        hooks.append((name, replace))
    return hooks


def freeze_writers(clean: dict[str, torch.Tensor], donor: dict[str, torch.Tensor],
                   senders: list[Head]) -> list:
    """Freeze every z and MLP write at all tokens, except donor sender z at -1."""
    by_name: dict[str, list[int]] = {}
    for layer, head in senders:
        by_name.setdefault(f"blocks.{layer}.attn.hook_z", []).append(head)
    hooks = []
    for name in clean:
        if not (name.endswith("hook_z") or name.endswith("hook_mlp_out")):
            continue
        def freeze(value, hook, name=name):
            result = clean[name].to(value).clone()
            heads = by_name.get(name, [])
            if heads:
                result[:, -1, heads] = donor[name][:, -1, heads].to(value)
            return result
        hooks.append((name, freeze))
    return hooks


def patch_receivers(cache: dict[str, torch.Tensor], routes: list[dict[str, Any]]) -> list:
    by_name: dict[str, list[int]] = {}
    for route in routes:
        name = f"blocks.{route['layer']}.attn.hook_{route['component']}"
        by_name.setdefault(name, []).append(route["hook_index"])
    hooks = []
    for name, indices in by_name.items():
        def replace(value, hook, name=name, indices=indices):
            result = value.clone()
            if max(indices) >= value.shape[2]:
                raise ValueError(f"Unexpected native hook width: {name} {tuple(value.shape)}")
            result[:, -1, indices] = cache[name][:, -1, indices].to(value)
            return result
        hooks.append((name, replace))
    return hooks


def patch_receivers_batched(cache: dict[str, torch.Tensor],
                            plan_routes: list[list[dict[str, Any]]]) -> list:
    """Row r + 1 receives plan r's routes; row 0 stays unpatched as the in-batch clean reference."""
    by_name: dict[str, list[tuple[int, int]]] = {}
    for row, routes in enumerate(plan_routes, start=1):
        for route in routes:
            name = f"blocks.{route['layer']}.attn.hook_{route['component']}"
            by_name.setdefault(name, []).append((row, route["hook_index"]))
    hooks = []
    for name, edits in by_name.items():
        def replace(value, hook, name=name, edits=edits):
            result = value.clone()
            for row, index in edits:
                if index >= value.shape[2]:
                    raise ValueError(f"Unexpected native hook width: {name} {tuple(value.shape)}")
                result[row, -1, index] = cache[name][0, -1, index].to(value)
            return result
        hooks.append((name, replace))
    return hooks


def margin(logits: torch.Tensor, target: int, original: int, row: int = 0) -> float:
    return float(logits[row, target] - logits[row, original])


@torch.inference_mode()
def run_pair(model: Any, recipient: torch.Tensor, donor: torch.Tensor,
             target: int, original: int, sender_sets: list[list[Head]],
             receiver_sets: list[list[Head]], components: list[str],
             allow_grouped_kv: bool = False, tolerance: float = 1e-4,
             batch_size: int = 8) -> list[dict[str, Any]]:
    if target == original:
        raise ValueError("Causal and original answers have identical first token: zero-signal readout")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Query heads can name the same native K/V intervention: retain aliases,
    # but execute each distinct intervention just once.
    unique_plans: dict[tuple, tuple] = {}
    for ss in sender_sets:
        for rr in receiver_sets:
            for c in components:
                routes = route_spec(model, ss, rr, c, allow_grouped_kv)
                key = (tuple(ss), c, tuple(sorted((r["layer"], r["hook_index"]) for r in routes)))
                if key in unique_plans:
                    unique_plans[key][4].append(head_label(rr))
                else:
                    unique_plans[key] = (ss, rr, c, routes, [head_label(rr)])
    plans = list(unique_plans.values())
    writer_names = {f"blocks.{layer}.{suffix}" for layer in range(model.cfg.n_layers)
                    for suffix in ("attn.hook_z", "hook_mlp_out")}
    receiver_names = {f"blocks.{r['layer']}.attn.hook_{r['component']}"
                      for _, _, _, routes, _ in plans for r in routes}
    clean_logits, clean = capture(model, recipient, writer_names | receiver_names)
    _, donor_cache = capture(model, donor, {f"blocks.{layer}.attn.hook_z"
                                           for ss in sender_sets for layer, _ in ss})
    clean_margin = margin(clean_logits, target, original)

    # Deterministic identity checks, once per pair; they do not depend on which
    # sender is patched, so the union of all senders/routes is the stronger test.
    all_senders = sorted({h for ss in sender_sets for h in ss})
    self_logits = evaluate(model, recipient, patch_senders(clean, all_senders))
    self_error = float((self_logits - clean_logits).abs().max())
    if self_error > tolerance:
        raise RuntimeError(f"Sender self-patch failed: max logit error {self_error}")
    _, self_route_cache = capture(model, recipient, receiver_names, freeze_writers(clean, clean, []))
    noop_error = route_self_error = 0.0
    for component in components:
        routes = [r for _, _, c, rs, _ in plans if c == component for r in rs]
        if not routes:
            continue
        noop = evaluate(model, recipient, patch_receivers(clean, routes))
        self_path = evaluate(model, recipient, patch_receivers(self_route_cache, routes))
        noop_error = max(noop_error, float((noop - clean_logits).abs().max()))
        route_self_error = max(route_self_error, float((self_path - clean_logits).abs().max()))
    if max(noop_error, route_self_error) > tolerance:
        raise RuntimeError(f"Receiver no-op/self-path failed: {noop_error}, {route_self_error}")
    # Path effects are read against an unpatched row of the same batch. Identical
    # rows must give identical logits, or batching would add numerical noise.
    width = min(batch_size, max(1, len(plans))) + 1
    replicated = evaluate(model, recipient.repeat(width, 1), [])
    batch_error = float((replicated - replicated[:1]).abs().max())
    if batch_error > tolerance:
        raise RuntimeError(f"Batch rows are not numerically independent: {batch_error}")
    batch_vs_single = float((replicated[:1] - clean_logits).abs().max())
    del self_route_cache

    rows = []
    for senders in sender_sets:
        whole = evaluate(model, recipient, patch_senders(donor_cache, senders))
        _, route_cache = capture(model, recipient, receiver_names,
                                 freeze_writers(clean, donor_cache, senders))
        mine = [p for p in plans if p[0] == senders]
        for start in range(0, len(mine), batch_size):
            chunk = mine[start:start + batch_size]
            logits = evaluate(model, recipient.repeat(len(chunk) + 1, 1),
                              patch_receivers_batched(route_cache, [p[3] for p in chunk]))
            batch_clean_margin = margin(logits, target, original, 0)
            for row, (_, receivers, component, routes, aliases) in enumerate(chunk, start=1):
                delta_norms = []
                for route in routes:
                    name = f"blocks.{route['layer']}.attn.hook_{component}"
                    index = route["hook_index"]
                    difference = route_cache[name][:, -1, index].float() - clean[name][:, -1, index].float()
                    delta_norms.append(float(difference.norm()))
                path_margin = margin(logits, target, original, row)
                rows.append({"senders": head_label(senders), "receivers": head_label(receivers),
                             "receiver_aliases": aliases,
                             "component": component, "routes": routes,
                             "clean_margin": clean_margin,
                             "whole_sender_margin": margin(whole, target, original),
                             "whole_sender_effect": margin(whole, target, original) - clean_margin,
                             "batch_clean_margin": batch_clean_margin,
                             "path_margin": path_margin,
                             "path_effect": path_margin - batch_clean_margin,
                             "receiver_delta_l2": delta_norms,
                             "sender_self_max_logit_error": self_error,
                             "receiver_noop_max_logit_error": noop_error,
                             "self_path_max_logit_error": route_self_error,
                             "batch_row_max_logit_error": batch_error,
                             "batch_vs_single_max_logit_error": batch_vs_single})
        del route_cache
    return rows


def export_pairs(args: argparse.Namespace) -> None:
    from dataclasses import asdict
    import numpy as np
    from transformers import AutoTokenizer
    from condition_registry import lookup
    from cma_config import (ExperimentConfig, ModelConfig, PromptConfig, GenerationConfig,
                            PatchingConfig, EvaluationConfig, WandbConfig)
    from prompt_generators import get_prompt_generator
    cond = lookup(args.condition)
    if cond.cnum not in ("C1", "C2", "K1"):
        raise ValueError("Initial assay supports canonical divergence conditions C1/C2/K1")
    template = args.template or ("indexed_list" if cond.domain == "code" else "food_truck")
    if template not in (("indexed_list",) if cond.domain == "code" else
                        ("food_truck", "hair_styling", "library_book")):
        raise ValueError("Template incompatible with condition")
    if args.n < 1:
        raise ValueError("n must be positive")
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to replace frozen pairs: {out}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              local_files_only=not args.allow_downloads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config = ExperimentConfig(ModelConfig(args.model), GenerationConfig(),
        PromptConfig(args.base_rule, cond.condition.context_type, args.n,
            condition=cond.condition, use_template_system=True, template_names=[template],
            tom_locations_file=str(ROOT / "cma_scripts/tom_datasets/locations.txt")),
        PatchingConfig(activation_name="z"), EvaluationConfig(), WandbConfig(enabled=False),
        seed=args.seed)
    generator = get_prompt_generator(config, tokenizer)
    pairs, answers, causal = generator.generate_prompts()
    rows = []
    for i, ((base, exp), (base_ans, exp_ans), causal_ans) in enumerate(zip(pairs, answers, causal)):
        if config.prompts.code_quote_prefill:
            raise ValueError("Qwen merged quoted-token protocol required; quote-prefill unsupported")
        token_ids = {key: tokenizer.encode(value, add_special_tokens=False)[0] for key, value in
                     {"causal_token_id": causal_ans, "original_token_id": exp_ans}.items()}
        if token_ids["causal_token_id"] == token_ids["original_token_id"]:
            raise ValueError(f"Answer-token collision in pair {i}")
        rows.append({"pair_id": f"{cond.cnum}_{template}_{args.base_rule}_{args.seed}_{i}",
                     "condition": cond.folder, "template": template, "base_rule": args.base_rule,
                     "donor_prompt": base, "recipient_prompt": exp, "base_ans": base_ans,
                     "exp_ans": exp_ans, "causal_ans": causal_ans, **token_ids})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    atomic_json(out.with_suffix(".meta.json"), {**provenance(), "config": asdict(config),
                "pairs_sha256": digest(out), "n_pairs": len(rows),
                "tokenizer_requested_revision": args.revision,
                "generator_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in
                    sorted((ROOT / "cma_scripts/tasks/causal_analysis/prompt_generators").glob("*.py"))},
                "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                "chat_template": tokenizer.chat_template,
                "selection": "Unfiltered generated pairs; not the historical generation-filtered CMA cohort"})
    print(f"Exported {len(rows)} frozen {cond.cnum} pairs to {out}")


def validate_resume(out: Path, identity: dict[str, Any]) -> None:
    manifest = out / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("Resume identity mismatch: inputs/config/source/model differ")
    elif any(out.glob("pair_*.json")):
        raise ValueError("Orphan checkpoints without manifest; use a new output directory")
    else:
        atomic_json(manifest, {"identity": identity, "provenance": provenance()})


def load_pinned_model(args: argparse.Namespace) -> Any:
    """Pass the explicitly pinned tokenizer; TL does not propagate all kwargs to it."""
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=not args.allow_downloads)
    model = HookedTransformer.from_pretrained(
        args.model, tokenizer=tokenizer, device=args.device,
        n_devices=args.n_devices, dtype=getattr(torch, args.dtype), fold_ln=False,
        center_writing_weights=False, center_unembed=False, revision=args.revision,
        local_files_only=not args.allow_downloads)
    return model


def pair_provenance(path: Path, model: str, revision: str) -> dict[str, Any]:
    """Validate export provenance before loading weights, or flag manual input."""
    sidecar = path.with_suffix(".meta.json")
    if not sidecar.exists():
        return {"status": "unverified_manual_pairs", "sidecar_sha256": None}
    metadata = json.loads(sidecar.read_text())
    if metadata.get("pairs_sha256") != digest(path):
        raise ValueError("Export sidecar pair hash mismatch")
    requested = metadata.get("tokenizer_requested_revision")
    if requested is None:
        # Older exports recorded the immutable revision in their exact command.
        command = metadata.get("command", [])
        if "--revision" in command:
            index = command.index("--revision")
            if index + 1 < len(command):
                requested = command[index + 1]
    resolved = metadata.get("tokenizer_revision")
    if requested != revision or (resolved is not None and resolved != revision):
        raise ValueError("Export tokenizer revision does not match requested model revision")
    exported_model = metadata.get("config", {}).get("model", {}).get("model_type")
    if exported_model != model:
        raise ValueError("Export tokenizer model does not match requested model")
    return {"status": "validated_export", "sidecar_sha256": digest(sidecar),
            "tokenizer_requested_revision": requested, "tokenizer_resolved_revision": resolved}


def prepare_model(args: argparse.Namespace) -> Any:
    torch.manual_seed(args.seed)
    model = load_pinned_model(args)
    model.eval()
    model.set_ungroup_grouped_query_attention(False)
    print(f"Model layers={model.cfg.n_layers} heads={model.cfg.n_heads} "
          f"kv_groups={model.cfg.n_key_value_heads} dtype={model.cfg.dtype} device={model.cfg.device}", flush=True)
    return model


def run(args: argparse.Namespace, model: Any = None) -> Any:
    """Measure all pending pairs; reuses ``model`` when given and returns the (possibly new) model."""
    if args.model != "Qwen/Qwen2.5-14B-Instruct":
        raise ValueError("CLI is validated for Qwen/Qwen2.5-14B-Instruct only")
    pairs_path = Path(args.pairs).resolve()
    export_provenance = pair_provenance(pairs_path, args.model, args.revision)
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if not pairs or len({p["pair_id"] for p in pairs}) != len(pairs):
        raise ValueError("Pairs must be nonempty with unique pair_id values")
    senders = parse_heads(args.senders) if args.senders else read_heads(Path(args.sender_csv))
    receivers = parse_heads(args.receivers) if args.receivers else read_heads(Path(args.receiver_csv))
    # Each sender is a separate intervention unless explicitly combined.
    sender_sets = [senders] if args.combine_senders else [[h] for h in senders]
    receiver_sets = [receivers] if args.combine_receivers else [[h] for h in receivers]
    if max(l for l, _ in senders) >= min(l for l, _ in receivers):
        raise ValueError("Select receivers strictly later than ALL selected senders; no paths silently skipped")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.tolerance < 0:
        raise ValueError("Tolerance must be nonnegative")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = {"protocol": "frozen_writers_final_token_v3_batched", "pairs_sha256": digest(pairs_path),
                "pair_provenance": export_provenance,
                "script_sha256": digest(Path(__file__)), "senders": senders, "receivers": receivers,
                "arguments": {k: v for k, v in vars(args).items() if k not in ("output",)},
                "head_csv_sha256": {key: digest(Path(getattr(args, key))) for key in
                                    ("sender_csv", "receiver_csv") if getattr(args, key)},
                "versions": provenance()["versions"]}
    # Canonicalize tuple/list representation before equality checks on restart.
    identity = json.loads(json.dumps(identity))
    validate_resume(out, identity)
    completed = []
    for i, pair in enumerate(pairs):
        file = out / f"pair_{i:06d}.json"
        if file.exists():
            saved = json.loads(file.read_text())
            if saved["pair"] != pair:
                raise ValueError(f"Checkpoint pair mismatch: {file}")
            completed.append(i)
    if len(completed) == len(pairs):
        print(f"All {len(pairs)} pairs already complete")
        return model
    if model is None:
        model = prepare_model(args)
    atomic_json(out / "model_config.json", {"config": model.cfg.to_dict(),
                 "requested_revision": args.revision,
                 "tokenizer_requested_revision": args.revision,
                 "tokenizer_resolved_revision": model.tokenizer.init_kwargs.get("_commit_hash")})
    for i, pair in enumerate(pairs):
        if i in completed:
            continue
        tokens = [model.tokenizer(pair[key], return_tensors="pt").input_ids.to(args.device)
                  for key in ("recipient_prompt", "donor_prompt")]
        for key, answer in (("causal_token_id", "causal_ans"), ("original_token_id", "exp_ans")):
            actual = model.to_tokens(pair[answer], prepend_bos=False)[0, 0].item()
            if actual != pair[key]:
                raise ValueError(f"Tokenizer/readout mismatch: pair {pair['pair_id']} {key}")
        print(f"Pair {i + 1}/{len(pairs)} token shapes={[tuple(t.shape) for t in tokens]} "
              f"readout={pair['causal_token_id']}-{pair['original_token_id']}", flush=True)
        started = time.monotonic()
        rows = run_pair(model, tokens[0], tokens[1], pair["causal_token_id"],
            pair["original_token_id"], sender_sets, receiver_sets, args.components,
            args.allow_grouped_kv, args.tolerance, args.batch_size)
        elapsed = time.monotonic() - started
        atomic_json(out / f"pair_{i:06d}.json", {"pair": pair, "rows": rows,
                    "measurement_seconds": elapsed})
        print(f"Saved {len(rows)} path measurements for {pair['pair_id']} in {elapsed:.1f}s", flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="Freeze pairs from canonical CMA generators")
    export.add_argument("--condition", required=True)
    export.add_argument("--template")
    export.add_argument("--base-rule", choices=["ABA", "ABB"], default="ABA")
    export.add_argument("--n", type=int, default=10)
    execute = sub.add_parser("run", help="Run direct paths on frozen pairs")
    execute.add_argument("--pairs", required=True)
    sender = execute.add_mutually_exclusive_group(required=True)
    sender.add_argument("--senders")
    sender.add_argument("--sender-csv")
    receiver = execute.add_mutually_exclusive_group(required=True)
    receiver.add_argument("--receivers")
    receiver.add_argument("--receiver-csv")
    execute.add_argument("--combine-senders", action="store_true")
    execute.add_argument("--combine-receivers", action="store_true")
    execute.add_argument("--components", nargs="+", choices=["q", "k", "v"], default=["q"])
    execute.add_argument("--allow-grouped-kv", action="store_true")
    execute.add_argument("--device", default="cuda")
    execute.add_argument("--n-devices", type=int, default=1)
    execute.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    execute.add_argument("--tolerance", type=float, default=1e-4,
                         help="Maximum absolute logit error for deterministic no-op controls")
    execute.add_argument("--batch-size", type=int, default=8,
                         help="Receiver plans per forward pass (plus one in-batch clean row)")
    for command in (export, execute):
        command.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
        command.add_argument("--revision", required=True,
                             help="Immutable Hugging Face model/tokenizer commit SHA")
        command.add_argument("--seed", type=int, default=20261022)
        command.add_argument("--allow-downloads", action="store_true")
        command.add_argument("--output", required=True)
    args = parser.parse_args()
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character Hugging Face commit SHA")
    if args.command == "export":
        export_pairs(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
