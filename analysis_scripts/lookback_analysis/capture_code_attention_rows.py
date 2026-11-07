"""
Full final-token attention rows for the code divergence (K1) and retrieval (K5) heads.

The padtwins capture (capture_head_outputs_full.py --capture_attn_summary) kept only
three summed masses per head (queried slot, mutated char, whole code block), so
"rest of code" could not be split. This forward-only pass stores the whole
final-token attention row for the heads in the K1 / K5 head lists, plus the
token offsets, so any region (list literal, .copy() line, alias line, mutation
line, question, system turn ...) can be defined afterwards without the GPU.

Same stimuli and prompt construction as the padtwins capture:
results/rsa_head_outputs/stimuli_code_padtwins/code_padtwins.csv, CODE_SYSTEM_PROMPT,
build_chat_prompt() from capture_head_outputs_full.py.

Output: results/rsa_head_outputs/activations_code_attn_rows/code_attention_rows.pt
(rsa_head_outputs/ is the tree pull_results.sh copies with its .pt files)
  attn:    (n_stim, n_heads_kept, max_T) float16, NaN-padded
  heads:   list of {layer, head, sets}   (sets: "code_abstract", "code_control", or both)
  prompts, offsets, n_tokens, stim (stimulus CSV rows), meta
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
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))

from cma_config import (  # noqa: E402
    EvaluationConfig, GenerationConfig, ModelConfig, PatchingConfig,
    PromptConfig, WandbConfig, ExperimentConfig,
)
from models import ModelLoader  # noqa: E402
from prompt_generators.code_templates import CODE_SYSTEM_PROMPT  # noqa: E402
from capture_head_outputs_full import build_chat_prompt, supports_system_role  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("code-attn-rows")

HEAD_SETS = ROOT / "results/aggregated/code_head_sets"


def load_heads() -> list[dict]:
    heads: dict[tuple[int, int], set] = {}
    for name in ("code_abstract", "code_control"):
        df = pd.read_csv(HEAD_SETS / f"{name}_head_list.csv")
        for L, H in zip(df["layer"].astype(int), df["head"].astype(int)):
            heads.setdefault((L, H), set()).add(name)
    return [{"layer": L, "head": H, "sets": "+".join(sorted(s))} for (L, H), s in sorted(heads.items())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim_csv", type=Path,
                    default=ROOT / "results/rsa_head_outputs/stimuli_code_padtwins/code_padtwins.csv")
    ap.add_argument("--out_dir", type=Path,  # rsa_head_outputs/ is pulled WITH .pt files
                    default=ROOT / "results/rsa_head_outputs/activations_code_attn_rows")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--unpadded_only", action="store_true",
                    help="keep only pad_len == suffix_pad_len == 0 stimuli")
    args = ap.parse_args()

    stim = pd.read_csv(args.stim_csv)
    if args.unpadded_only:
        stim = stim[(stim.pad_len == 0) & (stim.suffix_pad_len == 0)].reset_index(drop=True)
    heads = load_heads()
    log.info(f"{len(stim)} stimuli, {len(heads)} heads: "
             + ", ".join(f"L{h['layer']}H{h['head']}" for h in heads))

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=args.model_type),
        generation=GenerationConfig(), evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1, template_names=["code_padtwins"]),
        wandb=WandbConfig(enabled=False),
    )
    model, tokenizer, *_ = ModelLoader(cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()
    sys_ok = supports_system_role(tokenizer)

    layers = sorted({h["layer"] for h in heads})
    names = {f"blocks.{L}.attn.hook_pattern" for L in layers}

    prompts, offsets, n_tok = [], [], []
    for _, row in stim.iterrows():
        p = build_chat_prompt(row["scenario"], row["question"], tokenizer,
                              system_prompt=CODE_SYSTEM_PROMPT, system_role_ok=sys_ok)
        enc = tokenizer(p, return_offsets_mapping=True, add_special_tokens=False)
        prompts.append(p)
        offsets.append([tuple(o) for o in enc["offset_mapping"]])
        n_tok.append(len(enc["input_ids"]))
    max_T = max(n_tok)
    attn = torch.full((len(stim), len(heads), max_T), float("nan"), dtype=torch.float16)

    for si, p in enumerate(prompts):
        tokens = tokenizer(p, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            next(model.parameters()).device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in names)
        for hi, h in enumerate(heads):
            row = cache[f"blocks.{h['layer']}.attn.hook_pattern"][0, h["head"], -1]  # final-token query
            attn[si, hi, :n_tok[si]] = row.float().cpu().to(torch.float16)
        del cache
        if (si + 1) % 50 == 0:
            log.info(f"  {si + 1}/{len(stim)}")

    sums = attn.float().nansum(-1)
    log.info(f"row-sum check: {sums.min():.3f} - {sums.max():.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "code_attention_rows.pt"
    torch.save({"attn": attn, "heads": heads, "prompts": prompts, "offsets": offsets,
                "n_tokens": n_tok, "stim": stim.to_dict("records"),
                "meta": {"model": args.model_type, "system_prompt": "code",
                         "stim_csv": str(args.stim_csv), "unpadded_only": args.unpadded_only}}, out)
    log.info(f"saved {out}  attn={tuple(attn.shape)}")


if __name__ == "__main__":
    main()
