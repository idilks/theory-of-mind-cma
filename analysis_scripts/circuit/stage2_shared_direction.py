"""Steer with the shared cross-domain wire components (GPU job).

Tests whether the mirror effect (code direction steering ToM) is mediated
by the shared component the geometry identifies, and whether a single
direction steers BOTH domains:

  proj_code_in_tom  code wire projected into ToM's k=8 subspace, ToM legs,
                    full dose grid — the mediation test. If it reproduces
                    the mirror effect at matched dose, the transfer runs
                    through ToM's own subspace.
  proj_tom_in_code  symmetric test on the code legs.
  v_shared          principal-angle bisector, both domains (pre-test says
                    weak: reader-Q ratio 1.51, near-orthogonal to both
                    rank-1 wires).
  u1_tom / w1_code  top principal pair, dose 2 only (exploratory).

All steering is additive rank-1 style at the RECEIVING domain's natural
scale (+ for inject on the non-divergent set, - for delete on the
divergent set). Prompts are the same held-out sets as the main Stage-2
runs, so paired comparisons against those runs' controls line up on
(template, pair_idx). Checkpoints per config; resumable.

Output: results/circuit/stage2_steering/{model}/L41_shared/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "circuit"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from circuit_common import (  # noqa: E402
    AGG_DIR, CAPTURE_ROOT, DOMAINS, INJECTION_HOOKS, READER_DLA_LAYERS,
    SITE_KEYS, STAGE1_OUT, STAGE2_OUT_ROOT, TEMPLATES,
)
from stage2_steering import (  # noqa: E402
    eval_config, heldout_capture_prompts, make_steer_fn, render_figures,
)

log = logging.getLogger("circuit-shared-steer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# (prompt_set, mode, sign) per domain; sign + pushes toward the divergent side
DOMAIN_LEGS = {
    "tom": [("true_belief", "inject", +1.0), ("false_belief", "delete", -1.0)],
    "code": [("code_alias", "inject", +1.0), ("code_copy", "delete", -1.0)],
}
DIR_PLAN = [
    ("proj_code_in_tom", ["tom"], [0.5, 1.0, 2.0, 4.0]),
    ("proj_tom_in_code", ["code"], [0.5, 1.0, 2.0, 4.0]),
    ("v_shared", ["tom", "code"], [1.0, 2.0]),
    ("u1_tom", ["tom", "code"], [2.0]),
    ("w1_code", ["tom", "code"], [2.0]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=SITE_KEYS, default="L41")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--run_label", default="shared")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="cuda")
    args = ap.parse_args()

    out_dir = STAGE2_OUT_ROOT / args.model_type / f"{args.site}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        render_figures(out_dir)
        return

    shared = np.load(AGG_DIR / f"shared_direction_{args.site}.npz")
    scales = {"tom": float(shared["tom_scale"]), "code": float(shared["code_scale"])}
    hook_name = INJECTION_HOOKS[args.site]
    tom_root = CAPTURE_ROOT / args.model_type
    npz = {"tom": np.load(STAGE1_OUT / f"directions_{args.site}.npz"),
           "code": np.load(STAGE1_OUT.parent / "stage1_code"
                           / f"directions_{args.site}.npz")}

    configs = []  # (pset, mode, direction, dose, dom, sign)
    for dname, doms, doses in DIR_PLAN:
        for dom in doms:
            for pset, mode, sign in DOMAIN_LEGS[dom]:
                for dose in doses:
                    configs.append((pset, mode, dname, dose, dom, sign))

    per_path = out_dir / "per_prompt.csv"
    attn_path = out_dir / "attention.csv"
    done = set()
    if per_path.exists():
        prev = pd.read_csv(per_path)
        done = set(zip(prev["prompt_set"], prev["mode"], prev["direction"],
                       prev["dose"]))
        log.info(f"resuming: {len(done)} configs already present")

    if args.dry_run:
        for c in configs:
            print("config:", c)
        return

    # deferred past the --dry-run early return above: pulls in transformer_lens
    # and the full model stack, which --dry-run deliberately does not require
    from tasks.causal_analysis.cma_config import (
        EvaluationConfig, ExperimentConfig, GenerationConfig, ModelConfig,
        PatchingConfig, PromptConfig, WandbConfig)
    from tasks.causal_analysis.models import ModelLoader

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type, n_devices=args.n_devices,
                          device_map=args.device_map),
        generation=GenerationConfig(), evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=TEMPLATES),
        wandb=WandbConfig(enabled=False))
    model, tokenizer, *_ = ModelLoader(cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()
    device = next(model.parameters()).device

    weights = torch.load(tom_root / "weights.pt", map_location="cpu",
                         weights_only=False)
    readers = [r for r in weights["readers"]
               if READER_DLA_LAYERS[0] <= r["layer"] <= READER_DLA_LAYERS[1]]
    pattern_names = {f"blocks.{r['layer']}.attn.hook_pattern" for r in readers}

    dir_vecs = {n: torch.tensor(shared[n], dtype=torch.float64, device=device)
                for n in ("v_shared", "u1_tom", "w1_code",
                          "proj_code_in_tom", "proj_tom_in_code")}

    prompt_sets: dict[str, list] = {}

    def get_prompts(pset, dom):
        if pset not in prompt_sets:
            d = DOMAINS[dom]
            cap_dir = tom_root / d["capture_subdir"]
            prompt_sets[pset] = heldout_capture_prompts(
                cap_dir, npz[dom], pset, d["templates"],
                divergent=d["variants"][0])
            log.info(f"prompt set {pset}: {len(prompt_sets[pset])} prompts")
        return prompt_sets[pset]

    clean_cache: dict[str, list] = {}
    applied_norms: list[float] = []

    def append_rows(rows, pset, mode, direction, dose, attn_rows):
        df = pd.DataFrame(rows)
        df.insert(0, "prompt_set", pset)
        df.insert(1, "mode", mode)
        df.insert(2, "direction", direction)
        df.insert(3, "dose", dose)
        df.to_csv(per_path, mode="a", header=not per_path.exists(), index=False)
        if attn_rows:
            adf = pd.DataFrame(attn_rows)
            adf.insert(0, "prompt_set", pset)
            adf.insert(1, "mode", mode)
            adf.insert(2, "direction", direction)
            adf.insert(3, "dose", dose)
            adf.to_csv(attn_path, mode="a", header=not attn_path.exists(),
                       index=False)

    def run_clean(pset, dom):
        if pset in clean_cache:
            return
        log.info(f"clean pass: {pset}")
        rows, attn_rows, logprobs = eval_config(
            model, tokenizer, get_prompts(pset, dom), [], applied_norms, None,
            True, readers, pattern_names, device)
        clean_cache[pset] = logprobs
        if (pset, "clean", "clean", 0.0) not in done:
            append_rows(rows, pset, "clean", "clean", 0.0, attn_rows)
            done.add((pset, "clean", "clean", 0.0))

    for pset, mode, dname, dose, dom, sign in configs:
        run_clean(pset, dom)
        key = (pset, mode, dname, dose)
        if key in done:
            continue
        log.info(f"config: {key} (domain={dom}, scale={scales[dom]:.1f})")
        fn = make_steer_fn("add", dose, applied_norms, vec=dir_vecs[dname],
                           scale=scales[dom], sign=sign)
        rows, attn_rows, _ = eval_config(
            model, tokenizer, get_prompts(pset, dom), [(hook_name, fn)],
            applied_norms, clean_cache[pset], True, readers, pattern_names,
            device)
        append_rows(rows, pset, mode, dname, dose, attn_rows)
        done.add(key)

    (out_dir / "config.json").write_text(json.dumps(dict(
        vars(args), scales=scales, hook=hook_name,
        shared_npz=str(AGG_DIR / f"shared_direction_{args.site}.npz"),
    ), indent=2, default=str))
    render_figures(out_dir)
    log.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
