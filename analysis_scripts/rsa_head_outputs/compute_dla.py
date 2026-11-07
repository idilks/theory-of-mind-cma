"""Compute direct logit attribution (DLA) for every (layer, head) on the
balanced RSA stimulus set, using the per-head z activations captured at the
final input token by capture_head_outputs_full.py.

For each stimulus and each head, projects the head's z output through W_O
and then onto the unembedding row for the first token of the answer and the
distractor. The logit-difference (answer - distractor) is the headline
metric: positive = head pushes the model toward the correct answer.

Inputs (per template):
  results/rsa_head_outputs/activations_full/{template}.pt
    keys: z (n_stim, n_layers, n_heads, d_head), stim, meta

Output (per template):
  results/rsa_head_outputs/dla/{template}_dla.csv
  Long format: one row per (stim, layer, head) with:
    template, prompt_id, belief_veracity, answer_state, distractor_state,
    answer_tok, distractor_tok, layer, head,
    dla_answer, dla_distractor, logit_diff

Usage (HPC):
  python analysis_scripts/rsa_head_outputs/compute_dla.py --template food_truck
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from cma_config import (  # noqa: E402
    EvaluationConfig, GenerationConfig, ModelConfig, PatchingConfig,
    PromptConfig, WandbConfig, ExperimentConfig,
)
from models import ModelLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dla")

DEFAULT_ACT_DIR = ROOT / "results/rsa_head_outputs/activations_full"
DEFAULT_OUT_DIR = ROOT / "results/rsa_head_outputs/dla"


def first_token_with_leading_space(tokenizer, text: str) -> int:
    """First-token id of " {text}" (Qwen BPE; leading space matches mid-sentence position)."""
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenization for {text!r}")
    return ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--act_dir", type=Path, default=DEFAULT_ACT_DIR)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    act_path = args.act_dir / f"{args.template}.pt"
    log.info(f"loading {act_path}")
    payload = torch.load(act_path, map_location="cpu", weights_only=False)
    z = payload["z"]                # (n_stim, n_layers, n_heads, d_head) fp16
    stim = payload["stim"]
    meta = payload["meta"]
    n_stim, n_layers, n_heads, d_head = z.shape
    log.info(f"z shape = {tuple(z.shape)}  meta={meta}")

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type),
        generation=GenerationConfig(),
        evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=[args.template]),
        wandb=WandbConfig(enabled=False),
    )
    log.info("loading model + tokenizer")
    model, tokenizer, *_ = ModelLoader(cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()

    device = next(model.parameters()).device
    # Keep native dtype (bf16) to fit on 40 GB GPUs — fp32 promotion of W_O
    # alone is ~5 GB and OOMs alongside the 30 GB model. bf16 precision is
    # plenty for the logit-diff magnitudes we care about.
    W_O = model.W_O.detach().to(device).clone()           # (n_layers, n_heads, d_head, d_model)
    W_U = model.W_U.detach().to(device).clone()           # (d_model, d_vocab)
    d_model = W_O.shape[-1]
    log.info(f"W_O {tuple(W_O.shape)} dtype={W_O.dtype}  W_U {tuple(W_U.shape)} dtype={W_U.dtype}  d_model={d_model}")
    if (n_layers, n_heads, d_head) != tuple(W_O.shape[:3]):
        raise RuntimeError(f"z dims {(n_layers, n_heads, d_head)} != W_O dims {tuple(W_O.shape[:3])}")

    # We only need W_O / W_U + tokenizer from here on — drop the model to free ~28 GB.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Per-stim answer / distractor first-token ids
    ans_tok_ids = []
    dist_tok_ids = []
    distractor_states = []
    for s in stim:
        ans = s["answer_state"]
        loc_a, loc_b = s["loc_a"], s["loc_b"]
        if ans == loc_a:
            dist = loc_b
        elif ans == loc_b:
            dist = loc_a
        else:
            raise ValueError(f"answer_state {ans!r} does not match loc_a/loc_b ({loc_a!r}/{loc_b!r})")
        ans_tok_ids.append(first_token_with_leading_space(tokenizer, ans))
        dist_tok_ids.append(first_token_with_leading_space(tokenizer, dist))
        distractor_states.append(dist)

    ans_tok_t = torch.tensor(ans_tok_ids, device=device)         # (n_stim,)
    dist_tok_t = torch.tensor(dist_tok_ids, device=device)       # (n_stim,)

    # Per-stim unembedding columns: (n_stim, d_model), same dtype as W_U.
    U_ans = W_U[:, ans_tok_t].T.contiguous()                     # (n_stim, d_model)
    U_dist = W_U[:, dist_tok_t].T.contiguous()                   # (n_stim, d_model)

    # DLA: project z through W_O then dot with unembedding.
    # z: (S, L, H, D)  W_O: (L, H, D, M)  -> out: (S, L, H, M)
    # dla: out @ U (per-stim) -> (S, L, H)
    # Match z dtype to W_O so the einsum stays in bf16 (≈1 GB for `out` vs 2 GB in fp32).
    z_dev = z.to(device).to(W_O.dtype)
    log.info(f"computing z @ W_O in {W_O.dtype} (this is the heavy step)")
    out = torch.einsum("slhd,lhdm->slhm", z_dev, W_O)            # (S, L, H, M)
    log.info("projecting onto answer / distractor unembedding")
    dla_ans = torch.einsum("slhm,sm->slh", out, U_ans).float()    # (S, L, H), cast for CSV
    dla_dist = torch.einsum("slhm,sm->slh", out, U_dist).float()  # (S, L, H), cast for CSV
    logit_diff = dla_ans - dla_dist                               # (S, L, H)
    log.info(f"dla_ans range: [{dla_ans.min().item():.3f}, {dla_ans.max().item():.3f}]")
    log.info(f"logit_diff range: [{logit_diff.min().item():.3f}, {logit_diff.max().item():.3f}]")

    # Materialise as long-format DataFrame.
    dla_ans_cpu = dla_ans.cpu().numpy()
    dla_dist_cpu = dla_dist.cpu().numpy()
    ld_cpu = logit_diff.cpu().numpy()

    rows = []
    for si, s in enumerate(stim):
        for L in range(n_layers):
            for H in range(n_heads):
                rows.append({
                    "template": args.template,
                    "prompt_id": s.get("prompt_id", si),
                    "belief_veracity": s.get("belief_veracity", ""),
                    "answer_state": s["answer_state"],
                    "distractor_state": distractor_states[si],
                    "answer_tok_id": ans_tok_ids[si],
                    "distractor_tok_id": dist_tok_ids[si],
                    "layer": L,
                    "head": H,
                    "dla_answer": float(dla_ans_cpu[si, L, H]),
                    "dla_distractor": float(dla_dist_cpu[si, L, H]),
                    "logit_diff": float(ld_cpu[si, L, H]),
                })

    df = pd.DataFrame(rows)
    out_csv = args.out_dir / f"{args.template}_dla.csv"
    df.to_csv(out_csv, index=False)
    log.info(f"wrote {out_csv}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
