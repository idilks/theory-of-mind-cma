"""Support for stage2_steering.py --head_residual_bridge (not a standalone runner).

Head-axis perturbations are followed by restoration of the SAME prompt's clean
residual coordinate. All inference is single-prompt, with paired readouts.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch

from circuit_common import BAND_LAYERS, CAPTURE_ROOT, REPO, STAGE1_OUT, TEMPLATES

log = logging.getLogger("steering-bridge")
FB, TB = "false_belief", "true_belief"
SITE = "blocks.42.hook_resid_pre"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def save_raw_states(path, records, heads):
    """Rows align with checkpoint rows; retain actual edits and full readouts."""
    arrays = {key: np.stack([r[key].numpy() for r in records])
              for key in ("residual", "direct_write", "logits", "attention")}
    example = next(delta for r in records for delta in r["head_deltas"].values())
    zero_delta = torch.zeros_like(example)
    arrays["head_deltas"] = np.stack([
        np.stack([r["head_deltas"].get(h, zero_delta).numpy() for h in heads])
        for r in records])
    arrays["heads"] = np.asarray(heads)
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    return {"file": path.name, "sha256": sha256(path)}


def validate_checkpoint(path, prompt):
    saved = json.loads(path.read_text())
    if saved["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
        raise ValueError("Checkpoint prompt mismatch")
    raw = path.parent / saved["raw_states"]["file"]
    if not raw.is_file() or sha256(raw) != saved["raw_states"]["sha256"]:
        raise ValueError("Checkpoint raw states missing or changed")
    return saved


def write_identity(out, identity):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError("Resume identity mismatch: inputs, code, model or configuration changed")
    elif any(out.glob("prompt_*.json")):
        raise ValueError("Orphan checkpoints without run identity")
    else:
        atomic_json(path, identity)


def validate_partition(fit, hold, n):
    if (fit.ndim != 1 or hold.ndim != 1 or not len(fit) or not len(hold)
            or not np.issubdtype(fit.dtype, np.integer)
            or not np.issubdtype(hold.dtype, np.integer)
            or sorted(np.concatenate([fit, hold]).tolist()) != list(range(n))):
        raise ValueError("Fit/hold indices must be a disjoint, exhaustive, nonempty partition")


def select_code_heads(capture, heads):
    z = capture["z"]
    if z.ndim == 4:
        selected = torch.stack([z[:, l, h] for l, h in heads], dim=1)
    elif z.ndim == 3 and "heads" in capture:
        mapping = {(int(r["layer"]), int(r["head"])): i
                   for i, r in enumerate(capture["heads"])}
        if len(mapping) != z.shape[1] or any(h not in mapping for h in heads):
            raise ValueError("Code capture has missing or duplicate selected heads")
        selected = z[:, [mapping[h] for h in heads]]
    else:
        raise ValueError("Code capture needs explicit head dimensions or a complete head map")
    return selected.double()


def normalized_rows(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    norms = x.norm(dim=-1, keepdim=True)
    if not torch.isfinite(x).all() or torch.any(norms <= 1e-12):
        raise ValueError("Nonfinite or zero state direction")
    return x / norms


def narrative_mask(tokenizer, prompt, location, n_tokens):
    """Whole location mentions in the narrative only, not headers/instructions."""
    match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", prompt, re.S)
    if not match or not location:
        raise ValueError("Cannot identify narrative in the frozen chat prompt")
    # These frozen templates put their question on a subsequent line.
    narrative = match.group(1).split("\n", 1)[0]
    spans = [(match.start(1) + m.start(), match.start(1) + m.end())
             for m in re.finditer(r"(?<!\w)" + re.escape(location) + r"(?!\w)", narrative)]
    if not spans:
        raise ValueError(f"No whole narrative mention of {location!r}")
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    if len(offsets) != n_tokens:
        raise ValueError("Attention masks and model tokens differ")
    mask = np.array([b > a and any(a < e and b > s for s, e in spans)
                     for a, b in offsets], dtype=bool)
    if not mask.any():
        raise ValueError("Location spans have no tokens")
    return mask


def prepare_inputs(args):
    """Read-only preflight; axes/scales use FIT prompts, evaluation uses HOLD."""
    if args.model_type != "Qwen2.5-14B-Instruct" or args.domain != "tom" or args.site != "L41":
        raise ValueError("Bridge currently supports Qwen2.5-14B-Instruct, tom, L41 only")
    if args.bridge_pairs < 1 or args.bridge_control_draws < 2:
        raise ValueError("Positive pilot pairs and at least two control draws required")
    if not re.fullmatch(r"[0-9a-f]{40}", args.bridge_revision):
        raise ValueError("Bridge requires a pinned 40-character model revision")
    capture_dir = args.capture_dir or CAPTURE_ROOT / args.model_type
    direction_path = args.directions or STAGE1_OUT / "directions_L41.npz"
    code_path = args.bridge_code_capture or (REPO / "results/rsa_head_outputs/"
        "activations_code_position_full/Qwen2.5-14B-Instruct/code_position.pt")
    fp = REPO / "results/aggregated/fingerprints"
    f1 = fp / "C1_abstract_belief_clamped.npy"
    f5 = fp / "C5_answer_changes_location_only_clamped.npy"
    files = [direction_path, code_path, f1, f5, capture_dir / "weights.pt"]
    files += [capture_dir / f"{t}.pt" for t in TEMPLATES]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing/empty bridge input: {path}")
    a, b = np.load(f1), np.load(f5)
    if a.shape != (1920,) or b.shape != (1920,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Unexpected Qwen fingerprint dimensions/values")
    flat = sorted(set(np.argsort(a)[-32:].tolist()) - set(np.argsort(b)[-22:].tolist()))
    heads = [(h // 40, h % 40) for h in flat]
    if len(heads) != 22 or any(l not in BAND_LAYERS for l, _ in heads):
        raise ValueError("The frozen selection must contain the 22 upstream heads")
    with np.load(direction_path) as data:
        dirs = {k: data[k].copy() for k in data.files}
    bases = {"rank1": torch.from_numpy(dirs["v_rank1"]).double()[None],
             "subspace": torch.from_numpy(dirs["U_topk"]).double()}
    for basis in bases.values():
        if basis.ndim != 2 or basis.shape[1] != 5120 or not torch.isfinite(basis).all():
            raise ValueError("Invalid residual basis")
        torch.testing.assert_close(basis @ basis.T, torch.eye(len(basis), dtype=torch.float64),
                                   atol=1e-8, rtol=0)
    prompts, fit_prompt_texts, hold_prompt_texts = [], set(), set()
    belief_sum, n_fit = torch.zeros(22, 128, dtype=torch.float64), 0
    resid_sum = torch.zeros(5120, dtype=torch.float64)
    partitions = {}
    for template in TEMPLATES:
        cap = torch.load(capture_dir / f"{template}.pt", map_location="cpu", weights_only=False)
        meta = cap["meta"]
        if (meta["model_type"] != args.model_type or meta["template"] != template
                or meta["band_layers"] != BAND_LAYERS or meta["position"] != -1):
            raise ValueError("Capture model, template, layer or token convention mismatch")
        n = int(meta["n_pairs"])
        fit, hold = dirs[f"fit_idx_{template}"], dirs[f"hold_idx_{template}"]
        validate_partition(fit, hold, n)
        partitions[template] = {"fit": fit.tolist(), "hold": hold.tolist()}
        if args.bridge_phase == "pilot" and len(hold) < args.bridge_pairs:
            raise ValueError("Not enough held-out pairs for pilot")
        chosen = hold[:args.bridge_pairs] if args.bridge_phase == "pilot" else hold
        z = {}
        for variant in (FB, TB):
            if cap["z"][variant].shape != (n, 13, 40, 128):
                raise ValueError("Unexpected capture z shape")
            z[variant] = torch.stack([cap["z"][variant][fit, BAND_LAYERS.index(l), h]
                                      for l, h in heads], dim=1).double().sum(0)
            fit_prompt_texts.update(cap["prompts"][variant][int(i)] for i in fit)
            hold_prompt_texts.update(cap["prompts"][variant][int(i)] for i in hold)
            behavior = {int(r["idx"]): r for r in cap["behavior"][variant]}
            for index in chosen:
                row = behavior[int(index)]
                if row["first_token_collision"]:
                    raise ValueError("Frozen evaluation contains an answer-token collision")
                correct, alt = row["correct"], row["alt"]
                prompts.append({"prompt": cap["prompts"][variant][int(index)],
                    "template": template, "pair_idx": int(index), "variant": variant,
                    "correct": correct, "alt": alt, "token_len": int(row["token_len"]),
                    "believed": correct if variant == FB else alt,
                    "current": alt if variant == FB else correct})
        belief_sum += z[TB] - z[FB]
        resid_sum += (cap["resid"][FB][fit, 1].double().sum(0)
                      - cap["resid"][TB][fit, 1].double().sum(0))
        n_fit += len(fit)
        del cap
    if fit_prompt_texts & hold_prompt_texts:
        raise ValueError("Identical prompt text occurs in fit and held-out partitions")
    # Check that the supplied residual axis is the one fitted on these captures.
    fitted_resid = normalized_rows(resid_sum / n_fit)
    if float(fitted_resid @ bases["rank1"][0]) < .99999:
        raise ValueError("Residual direction does not match capture fit partition")
    native_delta = belief_sum / n_fit
    scales = native_delta.norm(dim=-1)
    code = torch.load(code_path, map_location="cpu", weights_only=False)
    if code["meta"].get("model") != args.model_type:
        raise ValueError("Code capture model mismatch")
    code_z = select_code_heads(code, heads)
    semantics = pd.DataFrame(code["stim"])["semantics"].astype(str)
    if set(semantics) != {"alias", "copy"}:
        raise ValueError("Code labels must contain alias and copy")
    alias = torch.tensor(semantics.eq("alias").to_numpy())
    code_delta = code_z[alias].mean(0) - code_z[~alias].mean(0)
    axes = {"code": normalized_rows(code_delta), "belief": normalized_rows(native_delta)}
    del code, code_z
    weights = torch.load(capture_dir / "weights.pt", map_location="cpu", weights_only=False)
    if weights["fold_ln"] or weights["band_layers"] != BAND_LAYERS:
        raise ValueError("Incompatible saved weight conventions")
    readers = [(int(r["layer"]), int(r["head"])) for r in weights["readers"]
               if 42 <= int(r["layer"]) <= 46]
    if len(readers) != 26 or len(set(readers)) != 26:
        raise ValueError("Expected 26 distinct saved reader heads")
    selected_weights = torch.stack([weights["W_O_band"][BAND_LAYERS.index(l), h]
                                    for l, h in heads])
    del weights
    sources = [Path(__file__), Path(__file__).with_name("stage2_steering.py"),
               Path(__file__).with_name("circuit_common.py")]
    identity = {"version": 1, "model": args.model_type, "revision": args.bridge_revision,
        "dtype": args.bridge_dtype, "phase": args.bridge_phase, "pilot_pairs": args.bridge_pairs,
        "seed": args.bridge_seed, "control_draws": args.bridge_control_draws,
        "heads": [list(h) for h in heads], "readers": [list(h) for h in readers],
        "partitions": partitions, "n_prompts": len(prompts), "alphas": [-1, 1],
        "inputs": {str(p.resolve()): sha256(p) for p in files},
        "source": {p.name: sha256(p) for p in sources},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "numpy", "pandas", "transformers", "transformer-lens")},
        "scope": "new paired mediation test; native axes and scales refit on Stage-1 fit half"}
    summary = {"heads": len(heads), "readers": len(readers), "prompts": len(prompts),
               "fit_pairs": n_fit, "head_axes_shape": list(axes["code"].shape),
               "residual_rank": len(bases["subspace"]), "input_validation": "passed",
               "axis_dtype": str(axes["code"].dtype), "axis_device": str(axes["code"].device)}
    print(json.dumps(summary, indent=2), flush=True)
    return dict(heads=heads, readers=readers, prompts=prompts, axes=axes, scales=scales,
                bases=bases, weights=selected_weights, identity=identity, sources=sources)


def bridge_forward(model, tokens, head_deltas, site, readers, edit_hook=None):
    """Capture only final-token state/attention, avoiding full-pattern caches."""
    site_layer = int(site.split(".")[1])
    if any(l >= site_layer for l, _ in head_deltas) or any(l < site_layer for l, _ in readers):
        raise ValueError("Senders must precede the site; readers must follow it")
    by_layer = {}
    for (layer, head), delta in head_deltas.items():
        by_layer.setdefault(layer, {})[head] = delta
    recorded, actual_deltas, attention = {}, {}, {}
    hooks = []
    for layer, changes in by_layer.items():
        def edit_z(value, hook, layer=layer, changes=changes):
            value = value.clone()
            for head, delta in changes.items():
                before = value[0, -1, head].clone()
                value[0, -1, head] = before + delta.to(value.device, value.dtype)
                actual_deltas[(layer, head)] = (value[0, -1, head].double()
                                               - before.double()).detach().cpu()
            return value
        hooks.append((f"blocks.{layer}.attn.hook_z", edit_z))
    if edit_hook is not None:
        hooks.append((site, edit_hook))
    def save_residual(value, hook):
        recorded["residual"] = value[0, -1].detach().double().cpu().clone()
    hooks.append((site, save_residual))
    for layer in sorted({l for l, _ in readers}):
        hs = [h for l, h in readers if l == layer]
        def save_attention(value, hook, layer=layer, hs=hs):
            for h in hs:
                attention[(layer, h)] = value[0, h, -1].detach().float().cpu().clone()
        hooks.append((f"blocks.{layer}.attn.hook_pattern", save_attention))
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)[0, -1].detach().float().cpu()
    direct = torch.zeros_like(recorded["residual"])
    for (layer, head), delta in actual_deltas.items():
        weight = model.blocks[layer].attn.W_O[head].detach()
        direct += (delta.to(weight.device) @ weight.double()).cpu()
    return dict(logits=logits, residual=recorded["residual"],
                attention=torch.stack([attention[h] for h in readers]),
                direct_write=direct, head_deltas=actual_deltas)


def bridge_conditions(model, tokens, head_deltas, site, readers, bases, clean,
                      steer_factory, control_draws=3, seed=0):
    """Use the existing stage-2 swap operator for prompt-specific restoration."""
    steered = bridge_forward(model, tokens, head_deltas, site, readers)
    results = {"head_only": steered}
    device = tokens.device
    delta = steered["residual"] - clean["residual"]
    rng = torch.Generator().manual_seed(seed)
    for name, basis in bases.items():
        basis = basis.double().cpu()
        component = (delta @ basis.T) @ basis
        fn = steer_factory("swap", 1.0, [], U=basis.to(device),
                           mu=clean["residual"].to(device))
        results[f"{name}/restore"] = bridge_forward(model, tokens, head_deltas,
                                                    site, readers, fn)
        add = steer_factory("add", 1.0, [], vec=component.to(device), scale=1.0)
        results[f"{name}/component_only"] = bridge_forward(model, tokens, {}, site, readers, add)
        for draw in range(control_draws):
            noise = torch.randn(len(delta), generator=rng, dtype=torch.float64)
            noise -= (noise @ basis.T) @ basis
            if noise.norm() < 1e-12:
                raise ValueError("No orthogonal control direction available")
            noise = noise / noise.norm() * component.norm()
            fn = steer_factory("add", 1.0, [], vec=noise.to(device), scale=1.0)
            results[f"{name}/control_{draw}"] = bridge_forward(model, tokens, head_deltas,
                                                              site, readers, fn)
        # Validate the implemented edit against exact geometry, with only the
        # rounding error of the destination dtype allowed (not model-effect size).
        dtype = next(model.parameters()).dtype
        expected = steered["residual"] - component
        rounded = expected.to(dtype).double()
        actual = results[f"{name}/restore"]["residual"]
        error = float((actual - rounded).abs().max())
        if error > 1e-6:
            raise ValueError(f"Restoration differs from dtype-rounded target: {error}")
        results[f"{name}/restore"]["validation"] = {
            "max_error_from_rounded_target": error,
            "coordinate_error": float(((actual - clean["residual"]) @ basis.T).norm()),
            "rounding_error_norm": float((rounded - expected).norm()),
            "removed_component_norm": float(component.norm()),
        }
    return results


def answer_token(tokenizer, text, prompt):
    # Match the saved capture's compute_answer_token_id convention.
    ids = tokenizer.encode((" " if prompt.endswith(" ") else "") + text,
                           add_special_tokens=False)
    if not ids:
        raise ValueError("Empty answer tokenization")
    return int(ids[0])


def summarize_outputs(out):
    rows, attn, contrasts = [], [], []
    for path in sorted(out.glob("prompt_*.json")):
        checkpoint = json.loads(path.read_text())
        rows.extend(checkpoint["rows"])
        attn.extend(checkpoint["attention"])
        contrasts.extend(checkpoint["contrasts"])
    if not rows:
        return
    for name, data in [("per_prompt", rows), ("attention", attn), ("contrasts", contrasts)]:
        temp = out / f"{name}.csv.tmp"
        pd.DataFrame(data).to_csv(temp, index=False)
        temp.replace(out / f"{name}.csv")
    # Pilot summaries are descriptive; paired uncertainty belongs to the full analysis.
    summary = pd.DataFrame(rows).groupby(["variant", "axis", "alpha", "condition"],
        dropna=False).agg(n=("margin_effect", "size"),
                         mean_margin_effect=("margin_effect", "mean"),
                         mean_coordinate_shift=("coordinate_shift", "mean")).reset_index()
    summary.to_csv(out / "summary.csv", index=False)


def run_bridge(args, steer_factory):
    prepared = prepare_inputs(args)
    if args.bridge_preflight:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Bridge inference requires CUDA on A100 or H200; use --bridge_preflight locally")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if not gpu_names or any(not any(kind in n.upper() for kind in ("H200", "A100")) for n in gpu_names):
        raise RuntimeError(f"A100 or H200 required; visible GPUs: {gpu_names}")
    from transformers import AutoTokenizer
    from transformer_lens import HookedTransformer
    out = args.bridge_output or REPO / f"results/circuit/head_residual_bridge/v1_{args.bridge_phase}"
    out = out.resolve()
    # Reject collisions with original steering/capture directories.
    if out == REPO or any(p.name in {"stage1_capture", "stage2_steering"}
                          for p in (out, *out.parents)):
        raise ValueError("Use a separate bridge output directory")
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out / ".run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = prepared["identity"]
        identity["gpu_names"] = gpu_names
        write_identity(out, identity)
        snapshots = out / "code_snapshot"
        snapshots.mkdir(exist_ok=True)
        for source in prepared["sources"]:
            dest = snapshots / source.name
            if dest.exists() and sha256(dest) != sha256(source):
                raise ValueError("Saved source snapshot differs")
            if not dest.exists():
                dest.write_bytes(source.read_bytes())
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct",
            revision=args.bridge_revision, local_files_only=True)
        # Validate every prompt before the first expensive forward pass.
        for rec in prepared["prompts"]:
            ids = tokenizer.encode(rec["prompt"], add_special_tokens=False)
            if len(ids) != rec["token_len"]:
                raise ValueError("Frozen prompt tokenization differs from saved capture")
            rec["tokens"] = ids
            rec["correct_id"] = answer_token(tokenizer, rec["correct"], rec["prompt"])
            rec["alt_id"] = answer_token(tokenizer, rec["alt"], rec["prompt"])
            if rec["correct_id"] == rec["alt_id"]:
                raise ValueError("Answer-token collision")
            rec["masks"] = [narrative_mask(tokenizer, rec["prompt"], rec[key], len(ids))
                            for key in ("believed", "current")]
            if np.any(rec["masks"][0] & rec["masks"][1]):
                raise ValueError("Believed/current narrative masks overlap")
        model = HookedTransformer.from_pretrained("Qwen/Qwen2.5-14B-Instruct",
            tokenizer=tokenizer, revision=args.bridge_revision, local_files_only=True,
            device="cuda", n_devices=1, dtype=getattr(torch, args.bridge_dtype),
            fold_ln=False, center_writing_weights=False, center_unembed=False).eval()
        if (model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head, model.cfg.d_model) != (48, 40, 128, 5120):
            raise ValueError("Unexpected model dimensions")
        atomic_json(out / "model_config.json", json.loads(json.dumps(model.cfg.to_dict(), default=str)))
        # Historical captures lack revision metadata. Verify their actual saved
        # write matrices against the loaded model, instead of inventing provenance.
        for i, (l, h) in enumerate(prepared["heads"]):
            current = model.blocks[l].attn.W_O[h].detach().cpu().half()
            if not torch.equal(current, prepared["weights"][i]):
                raise ValueError(f"Loaded model W_O differs from capture at L{l}H{h}")
        print(f"Model: dtype={next(model.parameters()).dtype}, device=cuda; "
              f"selected W_O compatibility passed; GPU={gpu_names}", flush=True)
        rng = np.random.default_rng(args.bridge_seed)
        order = rng.permutation(len(prepared["heads"]))
        while np.any(order == np.arange(len(order))):
            order = rng.permutation(len(order))
        axes = dict(prepared["axes"])
        axes["shuffled_code"] = axes["code"][order]
        random_axes = torch.randn(axes["code"].shape, dtype=torch.float64,
                                  generator=torch.Generator().manual_seed(args.bridge_seed))
        axes["random_heads"] = normalized_rows(random_axes)
        np.savez(out / "frozen_axes.npz", heads=np.array(prepared["heads"]),
                 scales=prepared["scales"].numpy(), permutation=order,
                 **{k: v.numpy() for k, v in axes.items()})
        started = time.monotonic()
        for prompt_number, rec in enumerate(prepared["prompts"]):
            dest = out / f"prompt_{prompt_number:04d}.json"
            if dest.exists():
                validate_checkpoint(dest, rec["prompt"])
                continue
            tick = time.monotonic()
            tokens = torch.tensor([rec["tokens"]], device="cuda")
            heads, readers, bases = prepared["heads"], prepared["readers"], prepared["bases"]
            clean = bridge_forward(model, tokens, {}, SITE, readers)
            zero = bridge_conditions(model, tokens, {h: torch.zeros(128) for h in heads},
                SITE, readers, {"rank1": bases["rank1"]}, clean, steer_factory,
                control_draws=0, seed=args.bridge_seed)
            for condition in zero.values():
                if not torch.equal(condition["logits"], clean["logits"]):
                    raise ValueError("Zero intervention/restoration changes clean logits")
            base = {k: rec[k] for k in ("template", "pair_idx", "variant")}
            rows, attn, contrasts, raw_records = [], [], [], []
            clean_logp = clean["logits"].log_softmax(0)
            clean_margin = float(clean["logits"][rec["alt_id"]] - clean["logits"][rec["correct_id"]])
            def record(result, axis, alpha, name):
                raw_records.append(result)
                logits = result["logits"]
                margin = float(logits[rec["alt_id"]] - logits[rec["correct_id"]])
                shift = result["residual"] - clean["residual"]
                lp = logits.log_softmax(0)
                row = dict(base, axis=axis, alpha=alpha, condition=name,
                    opposite_state=(alpha == (1 if rec["variant"] == FB else -1)),
                    margin=margin, margin_effect=margin-clean_margin,
                    p_correct=float(lp[rec["correct_id"]].exp()),
                    p_alt=float(lp[rec["alt_id"]].exp()),
                    top1_correct=int(logits.argmax()) == rec["correct_id"],
                    kl_from_clean=float((lp.exp() * (lp-clean_logp)).sum()),
                    coordinate_shift=float(shift @ bases["rank1"][0]),
                    residual_shift_norm=float(shift.norm()),
                    direct_write_coordinate=float(result["direct_write"] @ bases["rank1"][0]))
                row.update(result.get("validation", {}))
                rows.append(row)
                for i, (layer, head) in enumerate(readers):
                    attn.append(dict(base, axis=axis, alpha=alpha, condition=name,
                        layer=layer, head=head,
                        mass_believed=float(result["attention"][i, rec["masks"][0]].sum()),
                        mass_current=float(result["attention"][i, rec["masks"][1]].sum())))
                return margin-clean_margin
            record(clean, "clean", 0, "clean")
            for axis, vectors in axes.items():
                for alpha in (-1, 1):
                    deltas = {h: alpha * vectors[i] * prepared["scales"][i]
                              for i, h in enumerate(heads)}
                    if axis in ("code", "belief"):
                        conditions = bridge_conditions(model, tokens, deltas, SITE, readers,
                            bases, clean, steer_factory, args.bridge_control_draws,
                            args.bridge_seed + prompt_number)
                    else:
                        conditions = {"head_only": bridge_forward(model, tokens, deltas, SITE, readers)}
                    effects = {name: record(result, axis, alpha, name)
                               for name, result in conditions.items()}
                    for target in bases if axis in ("code", "belief") else []:
                        controls = [effects[f"{target}/control_{j}"] for j in range(args.bridge_control_draws)]
                        contrasts.append(dict(base, axis=axis, alpha=alpha, target=target,
                            head_effect=effects["head_only"],
                            restored_effect=effects[f"{target}/restore"],
                            component_only_effect=effects[f"{target}/component_only"],
                            restoration_attenuation=effects["head_only"]-effects[f"{target}/restore"],
                            control_mean_attenuation=effects["head_only"]-float(np.mean(controls)),
                            attenuation_beyond_control=float(np.mean(controls))-effects[f"{target}/restore"]))
                    del conditions
            raw_states = save_raw_states(dest.with_suffix(".npz"), raw_records, heads)
            atomic_json(dest, dict(prompt_sha256=hashlib.sha256(rec["prompt"].encode()).hexdigest(),
                prompt=rec["prompt"], rows=rows, attention=attn, contrasts=contrasts,
                raw_states=raw_states, tokens=rec["tokens"],
                answer_tokens={"correct": rec["correct_id"], "alternative": rec["alt_id"]},
                location_tokens={key: np.flatnonzero(mask).tolist()
                    for key, mask in zip(("believed", "current"), rec["masks"])},
                seconds=time.monotonic()-tick, zero_checks="passed",
                max_memory_bytes=int(torch.cuda.max_memory_allocated())))
            print(f"Checkpoint {prompt_number+1}/{len(prepared['prompts'])}: "
                  f"{time.monotonic()-tick:.1f}s; {dest.name}", flush=True)
        summarize_outputs(out)
        print(f"Bridge complete: {out}; elapsed this invocation {time.monotonic()-started:.1f}s", flush=True)
