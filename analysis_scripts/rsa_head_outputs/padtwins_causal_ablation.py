#!/usr/bin/env python
"""Deconfound step 3 (GPU): is the pad carrier CAUSALLY USED?

Steps 1 and 2 are observational. The random-head baseline showed the pad
response is generic (arbitrary heads show ~95% of the CMA sets' pad
response), and the position-prediction control asks whether it is inherited.
Neither says whether the model USES the carrier. This does: project the k=2
pad subspace out of the divergence heads' z in a live forward pass and see
whether copy/alias behaviour degrades.

This also closes the projection section's missing causal test — the
projection analysis showed the veracity tag survives pad-subspace removal in
the RDMs; here the same removal is applied online.

Protocol
  pass 1   clean forward over the pad-twin stimuli; cache hook_z at the final
           token for every head in the union of the head sets, and record the
           clean answer-token log-probs.
  fit      per head, the pad subspace = top-k right singular vectors of the
           within-twin single-pad displacement matrix (same estimator as
           padtwins_projection_test.pad_subspace). Fit on THIS run's
           activations, not the on-disk capture, so the subspace and the
           intervention are guaranteed to refer to the same prompts — this
           matters for --quote_prefill, which changes the final token.
  pass 2+  one forward per condition, hooking hook_z at the final token only
           (the position the subspace was estimated at):
             pad_projected     z <- z - (z U) U^T           [the test]
             random_projected  same with k random orthonormal dirs [rank control]
             zero_ablated      z <- 0                       [upper bound]

Conditions are run for each requested head set. Add --random_heads to also
run a size-matched draw of never-CMA-selected heads.

Readouts per stimulus: p(answer), p(alt), flip_score = p_alt/(p_alt+p_answer),
KL(ablated || clean) at the answer position, top-1 token.
  - mutated-slot queries carry a real veracity alternative (the copy answer
    is the pre-mutation char, the alias answer the post-mutation char), so
    flip_score is meaningful there;
  - untouched-slot queries have the same answer under both semantics, so
    only accuracy is meaningful. They are the built-in specificity control:
    a pad-subspace ablation that damages retrieval in general will hurt both.
Rows carry veracity_pair so the analysis can split them.

Predictions
  carrier is used        pad_projected degrades mutated-slot behaviour well
                         beyond random_projected at matched rank
  carrier is a byproduct pad_projected ~ random_projected; only zero_ablated
                         moves anything

Output: results/rsa_head_outputs/causal_pad_ablation/{model}/
  per_prompt.csv, summary.csv, pad_subspace_stats.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
# importing ablation_experiment os.chdir()s into cma_scripts/, so remember where
# we started and resolve any relative CLI path against it
CWD = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))

from cma_config import (  # noqa: E402
    EvaluationConfig, ExperimentConfig, GenerationConfig, ModelConfig,
    PatchingConfig, PromptConfig, WandbConfig)
from models import ModelLoader  # noqa: E402
from prompt_generators.code_templates import CODE_SYSTEM_PROMPT  # noqa: E402
from tasks.causal_analysis.ablation_experiment import (  # noqa: E402
    get_first_code_answer_token_id)
from capture_head_outputs_full import (  # noqa: E402
    build_chat_prompt, supports_system_role)
from analyze_code_ledger_rdms import (  # noqa: E402
    HEADS_CSV, GEMMA_HEADSET_DIR, HEAD_SETS, GEMMA_HEAD_SETS)
from padtwins_projection_test import twin_groups, displacement_pairs  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pad-causal")

STIM_CSV = (ROOT / "results/rsa_head_outputs/stimuli_code_padtwins/"
            "code_padtwins.csv")
OUT_ROOT = ROOT / "results/rsa_head_outputs/causal_pad_ablation"
MODEL_NAMES = {"qwen": "Qwen2.5-14B-Instruct", "gemma": "gemma-2-27b-it"}
CONDITIONS = ("pad_projected", "random_projected", "zero_ablated")


# --------------------------------------------------------------------- setup

def head_sets_for(model: str) -> dict[str, list[tuple[int, int]]]:
    if model == "gemma":
        frames = {s: pd.read_csv(GEMMA_HEADSET_DIR / f"{s}_head_list.csv")
                  for s in GEMMA_HEAD_SETS}
    else:
        heads = pd.read_csv(HEADS_CSV)
        frames = {s: heads[heads["sources"].str.contains(s)] for s in HEAD_SETS}
    return {s: [(int(r.layer), int(r.head)) for r in df.itertuples()]
            for s, df in frames.items()}


def answer_pair(row) -> tuple[str, str, bool]:
    """(answer char, alternative char, is a real veracity pair).

    Mutated slot: copy reads the pre-mutation char, alias the post-mutation
    char — the two are the veracity alternatives. Untouched slot: both
    semantics read the same init char, so there is no veracity alternative;
    the mutated char is used as a salient distractor for the flip readout
    but the row is flagged so it can be excluded."""
    ans = str(row["answer"]).strip("'")
    if row["queried_slot_type"] == "mutated":
        other = (row["mut_char"] if ans == row["orig_m"] else row["orig_m"])
        return ans, str(other), True
    return ans, str(row["mut_char"]), False


# ----------------------------------------------------------------- subspaces

def fit_pad_subspaces(z_clean, stim, members, k, rng, stats_rows, set_name):
    """Per head: (dim, k) pad basis + (dim, k) matched-rank random basis."""
    pairs = displacement_pairs(stim, twin_groups(stim))
    pad_U, rnd_U = {}, {}
    for (L, H) in members:
        zh = z_clean[:, L, H, :].astype(np.float64)
        disp = np.stack([zh[j] - zh[i] for i, j in pairs])
        _, s, vt = np.linalg.svd(disp, full_matrices=False)
        evr = float((s[:k] ** 2).sum() / (s ** 2).sum())
        half = len(pairs) // 2
        _, _, va = np.linalg.svd(disp[:half], full_matrices=False)
        _, _, vb = np.linalg.svd(disp[half:], full_matrices=False)
        stats_rows.append({"head_set": set_name, "layer": L, "head": H,
                           "evr_topk": evr,
                           "pc1_split_half_cos": float(abs(va[0] @ vb[0]))})
        pad_U[(L, H)] = vt[:k].T
        q, _ = np.linalg.qr(rng.normal(size=(zh.shape[1], k)))
        rnd_U[(L, H)] = q[:, :k]
    return pad_U, rnd_U


def group_by_layer(basis: dict, device, dtype=torch.float64):
    """{layer: (head index tensor, stacked basis (n_heads, dim, k))}."""
    out = {}
    for (L, H), U in basis.items():
        out.setdefault(L, ([], []))
        out[L][0].append(H)
        out[L][1].append(U)
    return {L: (torch.tensor(hs, device=device),
                torch.tensor(np.stack(Us), device=device, dtype=dtype))
            for L, (hs, Us) in out.items()}


def make_project_hook(by_layer):
    """Remove span(U) from each target head's z at the FINAL token only —
    the position the subspace was estimated at."""
    def hook(z, hook):
        L = hook.layer()
        if L not in by_layer:
            return z
        idx, U = by_layer[L]
        v = z[:, -1, :, :].index_select(1, idx).to(torch.float64)
        coef = torch.einsum("bhd,hdk->bhk", v, U)
        v = v - torch.einsum("bhk,hdk->bhd", coef, U)
        z[:, -1, idx, :] = v.to(z.dtype)
        return z
    return hook


def make_zero_hook(by_layer):
    def hook(z, hook):
        L = hook.layer()
        if L in by_layer:
            z[:, -1, by_layer[L][0], :] = 0
        return z
    return hook


# ---------------------------------------------------------------- evaluation

def answer_token_ids(tokenizer, ans, alt, prefill):
    f = lambda c: get_first_code_answer_token_id(  # noqa: E731
        tokenizer, f"'{c}'", quote_prefilled=prefill)
    return f(ans), f(alt)


def run_pass(model, tokenizer, prompts, records, fwd_hooks, clean_lp,
             device, capture_z=None):
    """One forward pass over all stimuli. Returns (rows, clean_logprobs)."""
    is_clean = clean_lp is None
    kept = [] if is_clean else None
    names_filter = (lambda n: n.endswith(".attn.hook_z")) if is_clean else None
    rows = []
    for i, (prompt, rec) in enumerate(zip(prompts, records)):
        tokens = tokenizer(prompt, return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            if is_clean:
                logits, cache = model.run_with_cache(
                    tokens, names_filter=names_filter)
                for L in range(model.cfg.n_layers):
                    key = f"blocks.{L}.attn.hook_z"
                    if key in cache:
                        capture_z[i, L] = cache[key][0, -1].float().cpu()
                del cache
            else:
                logits = model(tokens)
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu()
        probs = lp.exp()
        p_ans = float(probs[rec["id_ans"]])
        p_alt = float(probs[rec["id_alt"]])
        kl = (0.0 if is_clean else
              float(torch.sum(lp.exp() * (lp - clean_lp[i].float()))))
        rows.append(dict(
            prompt_id=rec["prompt_id"], twin_id=rec["twin_id"],
            queried_object=rec["queried_object"],
            queried_slot_type=rec["queried_slot_type"],
            pad_len=rec["pad_len"], suffix_pad_len=rec["suffix_pad_len"],
            veracity_pair=rec["veracity_pair"],
            p_answer=p_ans, p_alt=p_alt,
            flip_score=p_alt / (p_alt + p_ans + 1e-12),
            kl_from_clean=kl,
            top1=tokenizer.decode([int(probs.argmax())])))
        if is_clean:
            kept.append(lp.half())
        if (i + 1) % 200 == 0:
            log.info(f"    {i + 1}/{len(prompts)}")
    return rows, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen", "gemma"], default="qwen")
    ap.add_argument("--k", type=int, default=2, help="pad-subspace rank")
    ap.add_argument("--head_sets", nargs="+", default=None,
                    help="default: all sets defined for the model")
    ap.add_argument("--random_heads", action="store_true",
                    help="also run a size-matched never-CMA-selected draw")
    ap.add_argument("--quote_prefill", action="store_true",
                    help="append the opening quote so the readout is the bare "
                         "answer char. REQUIRED for Gemma, whose tokenizer has "
                         "no merged 'X tokens (both answers would share the "
                         "bare ' first token and the readout would be "
                         "degenerate). Qwen runs unprefilled by default so it "
                         "matches the RDM captures.")
    ap.add_argument("--stim_csv", type=Path, default=STIM_CSV)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N stimuli")
    args = ap.parse_args()

    args.stim_csv = (args.stim_csv if args.stim_csv.is_absolute()
                     else (CWD / args.stim_csv))
    model_name = MODEL_NAMES[args.model]
    if args.model == "gemma" and not args.quote_prefill:
        log.warning("gemma without --quote_prefill: the two answer chars share "
                    "the bare ' first token, so flip_score is degenerate")
    out_dir = OUT_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    stim_df = pd.read_csv(args.stim_csv)
    if args.limit:
        stim_df = stim_df.head(args.limit)
    stim = stim_df.to_dict("records")
    log.info(f"{len(stim)} stimuli from {args.stim_csv}")

    cfg = ExperimentConfig(
        model=ModelConfig(model_type=model_name,
                          ungroup_grouped_query_attention=(args.model == "gemma")),
        generation=GenerationConfig(), evaluation=EvaluationConfig(),
        patching=PatchingConfig(activation_name="z"),
        prompts=PromptConfig(use_template_system=True, context_type="abstract",
                             base_rule="ABA", prompt_num=1,
                             template_names=["code_padtwins"]),
        wandb=WandbConfig(enabled=False))
    model, tokenizer, *_ = ModelLoader(
        cfg.model, cfg.generation, cfg.prompts).load_model_and_tokenizer()
    model.eval()
    device = next(model.parameters()).device
    n_layers, n_heads, d_head = (model.cfg.n_layers, model.cfg.n_heads,
                                 model.cfg.d_head)

    role_ok = supports_system_role(tokenizer)
    prompts, records = [], []
    for r in stim:
        p = build_chat_prompt(r["scenario"], r["question"], tokenizer,
                              system_prompt=CODE_SYSTEM_PROMPT,
                              system_role_ok=role_ok)
        if args.quote_prefill:
            p = p + "'"
        ans, alt, vpair = answer_pair(r)
        id_ans, id_alt = answer_token_ids(tokenizer, ans, alt,
                                          args.quote_prefill)
        if id_ans == id_alt:
            raise SystemExit(
                f"answer/alt collide on the first token for {ans!r}/{alt!r} "
                "— use --quote_prefill (or a tokenizer-filtered answer pool)")
        prompts.append(p)
        records.append({**{k: r[k] for k in (
            "prompt_id", "twin_id", "queried_object", "queried_slot_type",
            "pad_len", "suffix_pad_len")},
            "veracity_pair": vpair, "id_ans": id_ans, "id_alt": id_alt})

    # ---- pass 1: clean, with z capture
    log.info("pass 1/N: clean")
    z_clean = torch.zeros(len(stim), n_layers, n_heads, d_head,
                          dtype=torch.float32)
    clean_rows, clean_lp = run_pass(model, tokenizer, prompts, records, [],
                                    None, device, capture_z=z_clean)
    z_np = z_clean.numpy()
    del z_clean

    sets = head_sets_for(args.model)
    if args.head_sets:
        sets = {s: sets[s] for s in args.head_sets}
    if args.random_heads:
        union = {m for v in sets.values() for m in v}
        pool = [(l, h) for l in range(n_layers) for h in range(n_heads)
                if (l, h) not in union]
        rng0 = np.random.default_rng(args.seed + 1)
        size = int(np.median([len(v) for v in sets.values()]))
        sets["random_heads"] = [pool[i] for i in
                                rng0.choice(len(pool), size, replace=False)]

    rng = np.random.default_rng(args.seed)
    all_rows = [dict(r, head_set="-", condition="clean") for r in clean_rows]
    stats_rows = []
    for set_name, members in sets.items():
        log.info(f"fitting pad subspaces for {set_name} ({len(members)} heads)")
        pad_U, rnd_U = fit_pad_subspaces(z_np, stim, members, args.k, rng,
                                         stats_rows, set_name)
        hooks_for = {
            "pad_projected": [(lambda n: n.endswith(".attn.hook_z"),
                               make_project_hook(group_by_layer(pad_U, device)))],
            "random_projected": [(lambda n: n.endswith(".attn.hook_z"),
                                  make_project_hook(group_by_layer(rnd_U, device)))],
            "zero_ablated": [(lambda n: n.endswith(".attn.hook_z"),
                              make_zero_hook(group_by_layer(pad_U, device)))],
        }
        for cond in CONDITIONS:
            log.info(f"  {set_name} / {cond}")
            rows, _ = run_pass(model, tokenizer, prompts, records,
                               hooks_for[cond], clean_lp, device)
            all_rows += [dict(r, head_set=set_name, condition=cond)
                         for r in rows]

    per = pd.DataFrame(all_rows)
    per.to_csv(out_dir / "per_prompt.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out_dir / "pad_subspace_stats.csv",
                                    index=False)

    ver = per[per["veracity_pair"]]
    summary = (per.groupby(["head_set", "condition", "veracity_pair"])
               .agg(p_answer=("p_answer", "mean"),
                    flip_score=("flip_score", "mean"),
                    flip_sem=("flip_score", "sem"),
                    kl=("kl_from_clean", "mean"),
                    n=("flip_score", "size")).reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)
    log.info("\n=== veracity-pair (mutated-slot) rows ===\n" +
             summary[summary["veracity_pair"]].to_string(index=False))
    log.info("\n=== untouched-slot rows (specificity) ===\n" +
             summary[~summary["veracity_pair"]].to_string(index=False))
    base = ver[ver["condition"] == "clean"]["flip_score"].mean()
    log.info(f"\nclean flip (veracity rows) = {base:.4f}; "
             "pad_projected must beat random_projected at matched rank "
             "for the carrier to count as causally used")
    log.info(f"-> {out_dir}")


if __name__ == "__main__":
    main()
