"""Shared runner for the local patching experiments (Mac / MPS friendly).

Used by cross_patch_experiment.py (set-level patches) and scoped_idx_cma.py
(per-head sweep over a scoped list). Loads the model once via the repo's
ModelLoader, builds chat prompts with CODE_SYSTEM_PROMPT, and provides:

  load_model(device)                  -> (model, tokenizer)
  probe_id(tokenizer, ch)             -> first token id of "'{ch}'" (merged 'X)
  final_z(model, tokenizer, prompt)   -> (n_layers, n_heads, head_dim) at the
                                         last position, plus clean last logits
  patched_logits(model, tokens, subs) -> last-position logits with hook_z
                                         substituted at the final position for
                                         {(layer, head): vector} entries

Head-set derivation from perhead_geometry.csv also lives here so both
experiments select identical populations:

  address     top-K by queried_slot_rel
  divergence  top-K by fp_C1  (--divergence_by semantics uses the RSA
              semantics loading instead; note in the analysis that fp_C1 is
              the causal choice and semantics the representational one)
  random      K heads sampled outside both, seeded
  layer_ctrl  for each address head's layer, the head in that layer with the
              smallest |queried_slot_rel| not already claimed (the
              "same depth, wrong heads" control, in the spirit of the
              ablation control curves)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from code_idx_pairs import CODE_SYSTEM_PROMPT  # noqa: E402

# Qwen defaults; load_model() overwrites these from the loaded checkpoint so
# the same code runs on Gemma (46x32) without touching call sites. See dims.py.
N_LAYERS, N_HEADS, HEAD_DIM = 48, 40, 128
PERHEAD_CSV = ROOT / "results/rsa_head_outputs/code_position_analysis/perhead_geometry.csv"


def load_model(device: str, model_type: str = "Qwen2.5-14B-Instruct"):
    from cma_config import (EvaluationConfig, GenerationConfig, ModelConfig,
                            PatchingConfig, PromptConfig, WandbConfig,
                            ExperimentConfig)
    from models import ModelLoader
    cfg = ExperimentConfig(
        model=ModelConfig(model_type=model_type, device_map="cpu",
                          device=device, n_devices=1),
        generation=GenerationConfig(), evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=["food_truck"]),
        wandb=WandbConfig(enabled=False),
    )
    model, tokenizer, *_ = ModelLoader(
        cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()
    global N_LAYERS, N_HEADS, HEAD_DIM
    import dims as _dims
    d = _dims.check(model, model_type)
    N_LAYERS, N_HEADS, HEAD_DIM = d.n_layers, d.n_heads, d.head_dim
    return model, tokenizer


def chat_prompt(scenario: str, question: str, tokenizer) -> str:
    import chat_compat
    return chat_compat.chat_prompt(tokenizer, CODE_SYSTEM_PROMPT,
                                   f"{scenario}\n{question}")


def encode(tokenizer, prompt: str, device: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt",
                     add_special_tokens=False)["input_ids"].to(device)


def probe_id(tokenizer, ch: str) -> int:
    ids = tokenizer(f"'{ch}'", add_special_tokens=False)["input_ids"]
    tok0 = tokenizer.decode(ids[:1])
    assert tok0.startswith("'"), (
        f"probe char {ch!r} does not tokenize to a merged 'X token ({tok0!r}) "
        "— it is outside the filtered answer pool for this tokenizer")
    return int(ids[0])


def answer_variant_ids(tokenizer, ch: str) -> set[int]:
    """Token ids that all realise answer char `ch` at the final position.

    probe_id returns the merged 'X token, but the model often puts its argmax
    on the bare X token instead — same character, different id. Scoring clean
    accuracy against the merged id alone counts those as wrong. Used only for
    the clean-accuracy diagnostic; the patched-vs-clean effect measure stays
    on probe_id so its two logits remain commensurate.
    """
    ids = {probe_id(tokenizer, ch)}
    for form in (ch, f" {ch}"):
        enc = tokenizer(form, add_special_tokens=False)["input_ids"]
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return ids


def run_clean(model, tokens) -> tuple[dict, torch.Tensor]:
    """One forward pass: returns ({(L,H): z_vec at final pos}, last logits)."""
    def nf(name): return name.endswith(".attn.hook_z")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=nf)
    zs = {}
    for L in range(N_LAYERS):
        key = f"blocks.{L}.attn.hook_z"
        if key in cache:
            z = cache[key][0, -1]           # (n_heads, head_dim)
            for H in range(N_HEADS):
                zs[(L, H)] = z[H].detach().clone()
    del cache
    return zs, logits[0, -1].detach().float().cpu()


def patched_logits(model, tokens, substitutions: dict) -> torch.Tensor:
    """Forward pass with hook_z[(L,H)] at the final position replaced.

    substitutions: {(layer, head): tensor(head_dim)}
    """
    by_layer: dict[int, dict[int, torch.Tensor]] = {}
    for (L, H), v in substitutions.items():
        by_layer.setdefault(L, {})[H] = v

    def make_hook(subs_for_layer):
        def hook(value, hook=None):
            for H, v in subs_for_layer.items():
                value[:, -1, H, :] = v.to(value.dtype).to(value.device)
            return value
        return hook

    fwd_hooks = [(f"blocks.{L}.attn.hook_z", make_hook(subs))
                 for L, subs in by_layer.items()]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
    return logits[0, -1].detach().float().cpu()


# ---------------------------------------------------------------------------
# head sets
# ---------------------------------------------------------------------------

def derive_head_sets(perhead_csv: Path = PERHEAD_CSV, k: int = 32,
                     seed: int = 0, divergence_by: str = "fp_C1"
                     ) -> dict[str, list[tuple[int, int]]]:
    d = pd.read_csv(perhead_csv)
    rng = np.random.default_rng(seed)

    def hs(rows): return [(int(r.layer), int(r.head))
                          for r in rows.itertuples()]

    address = hs(d.nlargest(k, "queried_slot_rel"))
    divergence = hs(d.nlargest(k, divergence_by))
    claimed = set(address) | set(divergence)

    pool = [(int(r.layer), int(r.head)) for r in d.itertuples()
            if (int(r.layer), int(r.head)) not in claimed]
    random_set = [pool[i] for i in rng.choice(len(pool), size=k,
                                              replace=False)]
    claimed |= set(random_set)

    layer_ctrl = []
    d = d.assign(absrel=d["queried_slot_rel"].abs())
    for (L, _) in address:
        cand = d[(d["layer"] == L)].sort_values("absrel")
        for r in cand.itertuples():
            key = (int(r.layer), int(r.head))
            if key not in claimed and key not in layer_ctrl:
                layer_ctrl.append(key)
                break
    return {"address": address, "divergence": divergence,
            "random": random_set, "layer_ctrl": layer_ctrl}
