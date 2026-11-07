"""Reader-head restoration and transplantation under residual steering.

Question: when the L41 residual edit of stage2_steering.py changes the
answer, does that change run through the later answer heads' attention?
Two versions of "attention": the heads' final-token QUERIES (the input to
the attention scores) and their final-token attention PATTERN rows (the
post-softmax weights that multiply the values). Keys and values are
included as sanity rows (see caveat below).

Per held-out prompt (same prompts, direction, dose and 26 readers as the
Figure "steer-attn" run: rank-1 direction, alpha = 4, resid_pre L42):

  clean                 no edit; record readers' final-token q/k/v/pattern
  steered               residual edit on; record the same activations
  restore_{site}        edit on; readers' {site} at the final token
                        replaced by that prompt's CLEAN values
  transplant_{site}     edit off; readers' {site} replaced by the STEERED
                        values
  restore_{site}_ctl{d} edit on; a layer-matched random non-reader set of
                        the same size has its {site} restored instead

Restoration measures necessity (does blocking the attention route shrink
the steering effect?); transplantation measures sufficiency (does the
attention change alone reproduce it?). Interactions can make the two
unequal, so no "fraction mediated" is derived here.

Readouts per run: p(correct), p(alt), flip_score = p_alt/(p_alt+p_correct),
margin = logit(alt) - logit(correct) (the unbounded score the restoration
null is stated on: zero mean steered-minus-restored difference), KL from
clean, top-1 token, and the believed/current location attention mass of
each reader (attention.csv).

Caveats baked into the design
  * The steer edits only the final-token residual, so keys and values at
    earlier positions are unchanged; restoring K or V at the final token
    affects only the final token's attention to itself. The informative
    sites are q and pattern; k/v rows are kept as a check that this is so.
  * Qwen2.5-14B shares each K/V across n_heads/n_kv_heads query heads, so
    K/V restoration acts on the shared group, not on single readers.
  * `pattern` restoration swaps post-softmax rows (TransformerLens
    hook_pattern), so normalization is preserved without any
    redistribution rule. It restores the whole row, not only the two
    location totals; attributing mediation to location switching needs a
    further, more targeted edit.
  * Restoring readers' q/pattern blocks every route into those activations
    (direct from L41 and indirect through L42-46). Routes to the logits
    that bypass the readers' attention stay open.

Identity checks: restoring an activation from the run it came from must
reproduce that run's logits (clean<-clean, steered<-steered); the first
--n_identity prompts record the max |logit diff| in identity.csv, and
every restoration run records the max |post-hook - target| error.

Checkpointing: rows for a prompt are appended only once all its runs are
done, so every prompt in per_prompt.csv is complete; finished prompts are
skipped on restart.
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
sys.path.insert(0, str(ROOT / "analysis_scripts" / "circuit"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cma_scripts"))
sys.path.insert(0, str(ROOT / "cma_scripts" / "tasks" / "causal_analysis"))

from circuit_common import (  # noqa: E402
    CAPTURE_ROOT, DOMAINS, INJECTION_HOOKS, READER_DLA_LAYERS, SITE_KEYS,
    STAGE1_OUT, STAGE2_OUT_ROOT, TEMPLATES,
)
from stage2_steering import (  # noqa: E402
    heldout_capture_prompts, location_token_mask, make_steer_fn,
)
# ablation_experiment chdir()s to cma_scripts/ on import; every path here is
# absolute (ROOT-based), so that is harmless.
from tasks.causal_analysis.ablation_experiment import compute_answer_token_id  # noqa: E402
from tasks.causal_analysis.cma_config import (  # noqa: E402
    EvaluationConfig, ExperimentConfig, GenerationConfig, ModelConfig,
    PatchingConfig, PromptConfig, WandbConfig)
from tasks.causal_analysis.models import ModelLoader  # noqa: E402

RESTORE_SITES = ["q", "k", "v", "pattern"]
HOOK_OF = {"q": "hook_q", "k": "hook_k", "v": "hook_v", "pattern": "hook_pattern"}

log = logging.getLogger("reader-restore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# head sets
# ---------------------------------------------------------------------------

def hook_name(layer: int, site: str) -> str:
    return f"blocks.{layer}.attn.{HOOK_OF[site]}"


def kv_groups(heads: list[int], n_heads: int, n_kv_heads: int) -> list[int]:
    """K/V group indices used by `heads` (identity when not grouped)."""
    ratio = n_heads // n_kv_heads
    return sorted({h // ratio for h in heads})


def by_layer(head_set: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for L, H in head_set:
        out.setdefault(L, []).append(H)
    return {L: sorted(hs) for L, hs in sorted(out.items())}


def layer_matched_control_sets(readers: list[tuple[int, int]], n_heads: int,
                               n_draws: int, seed: int) -> list[list[tuple[int, int]]]:
    """Random non-reader heads with the readers' per-layer counts."""
    rng = np.random.default_rng(seed)
    reader_set = set(readers)
    draws = []
    for _ in range(n_draws):
        chosen = []
        for L, hs in by_layer(readers).items():
            pool = [h for h in range(n_heads) if (L, h) not in reader_set]
            if len(pool) < len(hs):
                raise ValueError(f"layer {L}: not enough non-reader heads")
            chosen += [(L, int(h)) for h in rng.choice(pool, len(hs), replace=False)]
        draws.append(sorted(chosen))
    return draws


def indices_for(site: str, heads: list[int], n_heads: int, n_kv_heads: int) -> list[int]:
    """Head (q/pattern) or K/V-group (k/v) indices a restoration touches."""
    if site in ("k", "v"):
        return kv_groups(heads, n_heads, n_kv_heads)
    return sorted(heads)


# ---------------------------------------------------------------------------
# hooks
# ---------------------------------------------------------------------------

def final_token_slice(site: str, value: torch.Tensor) -> torch.Tensor:
    """q/k/v: [batch, pos, head, d_head] -> [head, d_head] at the last
    position; pattern: [batch, head, q_pos, k_pos] -> [head, k_pos] last row."""
    if site == "pattern":
        return value[0, :, -1, :]
    return value[0, -1, :, :]


def make_record_hooks(layers: list[int], sites: list[str], store: dict):
    hooks = []
    for L in layers:
        for site in sites:
            def rec(value, hook, L=L, site=site):
                store[(L, site)] = final_token_slice(site, value).detach().clone()
                return value
            hooks.append((hook_name(L, site), rec))
    return hooks


def make_restore_hooks(site: str, head_set: list[tuple[int, int]], source: dict,
                       n_heads: int, n_kv_heads: int, errors: list[float]):
    """Replace the final-token `site` activation of `head_set` with
    `source[(layer, site)]` (recorded by make_record_hooks in another run).
    Appends max |post-hook - target| per layer to `errors`."""
    hooks = []
    for L, hs in by_layer(head_set).items():
        idx = indices_for(site, hs, n_heads, n_kv_heads)
        def restore(value, hook, L=L, idx=idx):
            target = source[(L, site)][idx].to(value.device, value.dtype)
            value = value.clone()
            if site == "pattern":
                value[0, idx, -1, :] = target
                written = value[0, idx, -1, :]
            else:
                value[0, -1, idx, :] = target
                written = value[0, -1, idx, :]
            errors.append(float((written.float() - target.float()).abs().max()))
            return value
        hooks.append((hook_name(L, site), restore))
    return hooks


def forward(model, tokens, hooks, record_layers, record_sites):
    """Run with `hooks` (restoration/steering first) and record the
    final-token activations at `record_layers` AFTER those hooks apply.
    Returns (final logits fp32 cpu, store)."""
    store: dict = {}
    all_hooks = list(hooks) + make_record_hooks(record_layers, record_sites, store)
    with torch.inference_mode():
        logits = model.run_with_hooks(tokens, fwd_hooks=all_hooks)
    return logits[0, -1].detach().float().cpu(), store


# ---------------------------------------------------------------------------
# readouts
# ---------------------------------------------------------------------------

def readouts(logits: torch.Tensor, clean_logits: torch.Tensor | None,
             cid: int, aid: int) -> dict:
    logprobs = torch.log_softmax(logits, dim=-1)
    probs = logprobs.exp()
    p_c, p_a = float(probs[cid]), float(probs[aid])
    top1 = int(probs.argmax())
    if clean_logits is None:
        kl, agree = 0.0, 1.0
    else:
        clean_lp = torch.log_softmax(clean_logits, dim=-1)
        kl = float(torch.sum(probs * (logprobs - clean_lp)))
        agree = float(top1 == int(clean_lp.argmax()))
    return dict(p_correct=p_c, p_alt=p_a,
                flip_score=p_a / (p_a + p_c + 1e-12),
                margin=float(logits[aid] - logits[cid]),
                kl_from_clean=kl, top1=top1, top1_agree=agree,
                p_top1=float(probs[top1]))


def attention_mass(store: dict, readers: list[tuple[int, int]],
                   m_bel: np.ndarray, m_cur: np.ndarray) -> list[dict]:
    rows = []
    for L, H in readers:
        pat = store[(L, "pattern")][H].float().cpu().numpy()
        rows.append(dict(layer=L, head=H,
                         mass_believed=float(pat[m_bel].sum()),
                         mass_current=float(pat[m_cur].sum())))
    return rows


# ---------------------------------------------------------------------------
# per-prompt driver
# ---------------------------------------------------------------------------

def run_prompt(model, tokens, steer_hook, readers, controls, sites,
               n_heads, n_kv_heads, cid, aid, m_bel, m_cur,
               applied_norms, do_identity: bool):
    """All runs for one prompt. Returns (rows, attn_rows, identity_rows)."""
    layers = sorted({L for L, _ in readers})
    rows, attn_rows, ident = [], [], []

    def add(run, site, head_set_name, logits, store, clean_logits, errors):
        applied = float(np.mean(applied_norms)) if applied_norms else 0.0
        rows.append(dict(run=run, restore_site=site, head_set=head_set_name,
                         restore_max_err=(max(errors) if errors else float("nan")),
                         applied_norm=applied,
                         **readouts(logits, clean_logits, cid, aid)))
        for r in attention_mass(store, readers, m_bel, m_cur):
            attn_rows.append(dict(run=run, restore_site=site,
                                  head_set=head_set_name, **r))

    applied_norms.clear()
    clean_logits, clean_store = forward(model, tokens, [], layers, sites)
    add("clean", "", "none", clean_logits, clean_store, None, [])
    applied_norms.clear()
    steer_logits, steer_store = forward(model, tokens, [steer_hook], layers, sites)
    add("steered", "", "none", steer_logits, steer_store, clean_logits, [])

    for site in sites:
        err: list[float] = []
        applied_norms.clear()
        lg, st = forward(model, tokens,
                         [steer_hook] + make_restore_hooks(site, readers, clean_store,
                                                           n_heads, n_kv_heads, err),
                         layers, sites)
        add(f"restore_{site}", site, "readers", lg, st, clean_logits, err)

        err = []
        applied_norms.clear()
        lg, st = forward(model, tokens,
                         make_restore_hooks(site, readers, steer_store,
                                            n_heads, n_kv_heads, err),
                         layers, sites)
        add(f"transplant_{site}", site, "readers", lg, st, clean_logits, err)

        for d, ctl in enumerate(controls):
            err = []
            applied_norms.clear()
            lg, st = forward(model, tokens,
                             [steer_hook] + make_restore_hooks(site, ctl, clean_store,
                                                               n_heads, n_kv_heads, err),
                             layers, sites)
            add(f"restore_{site}_ctl{d}", site, f"ctl{d}", lg, st, clean_logits, err)

        if do_identity:
            err = []
            lg, _ = forward(model, tokens,
                            make_restore_hooks(site, readers, clean_store,
                                               n_heads, n_kv_heads, err), layers, sites)
            ident.append(dict(site=site, source="clean",
                              max_abs_logit_diff=float((lg - clean_logits).abs().max()),
                              restore_max_err=max(err)))
            err = []
            applied_norms.clear()
            lg, _ = forward(model, tokens,
                            [steer_hook] + make_restore_hooks(site, readers, steer_store,
                                                              n_heads, n_kv_heads, err),
                            layers, sites)
            ident.append(dict(site=site, source="steered",
                              max_abs_logit_diff=float((lg - steer_logits).abs().max()),
                              restore_max_err=max(err)))
    return rows, attn_rows, ident


def expected_runs(sites: list[str], n_controls: int) -> set[str]:
    runs = {"clean", "steered"}
    for s in sites:
        runs |= {f"restore_{s}", f"transplant_{s}"}
        runs |= {f"restore_{s}_ctl{d}" for d in range(n_controls)}
    return runs


def load_done(per_path: Path, runs: set[str]) -> set[tuple]:
    """Prompts whose run set is complete; incomplete ones are dropped from
    the CSV so they are re-run whole."""
    if not per_path.exists():
        return set()
    df = pd.read_csv(per_path)
    key = ["prompt_set", "template", "pair_idx"]
    have = df.groupby(key)["run"].apply(set)
    complete = {k for k, s in have.items() if s >= runs}
    if len(complete) < len(have):
        keep = df[[tuple(r) in complete for r in df[key].itertuples(index=False)]]
        keep.to_csv(per_path, index=False)
        log.info(f"dropped {len(have) - len(complete)} incomplete prompts from {per_path.name}")
    return complete


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=list(DOMAINS), default="tom")
    ap.add_argument("--site", choices=SITE_KEYS, default="L41")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--direction", choices=["target_rank1", "target_topk"],
                    default="target_rank1")
    ap.add_argument("--dose", type=float, default=4.0)
    ap.add_argument("--restore", nargs="+", choices=RESTORE_SITES,
                    default=["q", "pattern"],
                    help="activations to restore/transplant (k/v are sanity rows)")
    ap.add_argument("--control_draws", type=int, default=3)
    ap.add_argument("--control_seed", type=int, default=99)
    ap.add_argument("--n_identity", type=int, default=3,
                    help="prompts per variant that also run the identity checks")
    ap.add_argument("--limit", type=int, default=None,
                    help="pilot: first N held-out prompts per variant")
    ap.add_argument("--run_label", default="reader_restore")
    ap.add_argument("--capture_dir", type=Path, default=None)
    ap.add_argument("--directions", type=Path, default=None)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="cuda")
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    FBD, TBD = domain["variants"]
    tom_root = CAPTURE_ROOT / args.model_type
    capture_dir = args.capture_dir or (tom_root / domain["capture_subdir"])
    weights_path = tom_root / "weights.pt"
    stage1_dir = (STAGE1_OUT if args.domain == "tom"
                  else STAGE1_OUT.parent / "stage1_code")
    dirs_path = args.directions or (stage1_dir / f"directions_{args.site}.npz")
    out_dir = STAGE2_OUT_ROOT / args.model_type / f"{args.site}_{args.run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_path, attn_path, ident_path = (out_dir / "per_prompt.csv",
                                       out_dir / "attention.csv",
                                       out_dir / "identity.csv")

    dz = np.load(dirs_path)
    steer_hook_name = INJECTION_HOOKS[args.site]
    scale = float(dz["natural_scale"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    readers = sorted((int(r["layer"]), int(r["head"])) for r in weights["readers"]
                     if READER_DLA_LAYERS[0] <= r["layer"] <= READER_DLA_LAYERS[1])
    n_heads = int(weights["n_heads"])
    controls = layer_matched_control_sets(readers, n_heads, args.control_draws,
                                          args.control_seed)
    runs = expected_runs(args.restore, args.control_draws)
    log.info(f"readers: {len(readers)} in layers {sorted({L for L, _ in readers})}; "
             f"controls: {args.control_draws} draws; runs/prompt: {len(runs)}; "
             f"direction={args.direction} dose={args.dose} hook={steer_hook_name}")
    (out_dir / "controls.json").write_text(json.dumps(dict(
        readers=readers, controls=controls, seed=args.control_seed), indent=2))
    config = dict(vars(args), capture_dir=str(capture_dir), directions=str(dirs_path),
                  out_dir=str(out_dir), natural_scale=scale, hook=steer_hook_name,
                  runs=sorted(runs), n_heads=n_heads)
    if args.dry_run:
        print(json.dumps(config, indent=2, default=str))
        return

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
    if model.cfg.n_heads != n_heads:
        sys.exit(f"weights.pt n_heads={n_heads} but the model has {model.cfg.n_heads}")
    # The runtime K/V head count is what hook_k/hook_v carry: the grouped
    # count unless GQA was ungrouped. (weights.pt's n_kv_heads is read off
    # the repeated per-head W_V and says 40 for Qwen, which is wrong here.)
    grouped = (model.cfg.n_key_value_heads is not None
               and not getattr(model.cfg, "ungroup_grouped_query_attention", False))
    n_kv = int(model.cfg.n_key_value_heads) if grouped else n_heads
    for site in args.restore:
        if site in ("k", "v"):
            groups = {L: kv_groups(hs, n_heads, n_kv) for L, hs in by_layer(readers).items()}
            log.info(f"{site}: K/V groups restored per layer {groups} "
                     f"(n_heads={n_heads}, n_kv_heads={n_kv})")
    config["n_kv_heads"] = n_kv
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))

    tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    applied_norms: list[float] = []

    def steer_fn(mode):
        if args.direction == "target_topk":
            mu = dz["mean_FB_fit"] if mode == "inject" else dz["mean_TB_fit"]
            return make_steer_fn("swap", args.dose, applied_norms,
                                 U=tt(dz["U_topk"]), mu=tt(mu))
        return make_steer_fn("add", args.dose, applied_norms, vec=tt(dz["v_rank1"]),
                             scale=scale, sign=(+1.0 if mode == "inject" else -1.0))

    done = load_done(per_path, runs)
    for variant, mode in ((FBD, "delete"), (TBD, "inject")):
        prompts = heldout_capture_prompts(capture_dir, dz, variant,
                                          domain["templates"], divergent=FBD)
        if args.limit:
            prompts = prompts[:args.limit]
        log.info(f"{variant} ({mode}): {len(prompts)} held-out prompts")
        hook = (steer_hook_name, steer_fn(mode))
        n_done_variant = 0
        for i, rec in enumerate(prompts):
            key = (variant, rec["template"], rec["pair_idx"])
            if key in done:
                n_done_variant += 1
                continue
            tokens = tokenizer(rec["prompt"], return_tensors="pt",
                               add_special_tokens=False)["input_ids"].to(device)
            cid = compute_answer_token_id(tokenizer, variant, rec["correct"], rec["prompt"])
            aid = compute_answer_token_id(tokenizer, variant, rec["alt"], rec["prompt"])
            n_tok = tokens.shape[1]
            m_bel = location_token_mask(tokenizer, rec["prompt"], rec["believed"], n_tok)
            m_cur = location_token_mask(tokenizer, rec["prompt"], rec["current"], n_tok)
            rows, attn_rows, ident = run_prompt(
                model, tokens, hook, readers, controls, args.restore, n_heads, n_kv,
                cid, aid, m_bel, m_cur, applied_norms,
                do_identity=(n_done_variant + i) < args.n_identity)
            meta = dict(prompt_set=variant, mode=mode, direction=args.direction,
                        dose=args.dose, template=rec["template"],
                        pair_idx=rec["pair_idx"])
            for path, rs in ((per_path, rows), (attn_path, attn_rows), (ident_path, ident)):
                if rs:
                    pd.DataFrame([{**meta, **r} for r in rs]).to_csv(
                        path, mode="a", header=not path.exists(), index=False)
            done.add(key)
            if (i + 1) % 10 == 0:
                log.info(f"    {variant}: {i + 1}/{len(prompts)}")
    log.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
