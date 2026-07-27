"""Stage 2 of the layered-lookback circuit: causal steering (GPU job).

Interventions at the injection site (resid_pre L42 == measured resid_post
L41; position -1 only), on HELD-OUT prompts only (split indices come from
the Stage-1 directions file — fit-half prompts are never steered):

  inject  on true_belief prompts  -> prediction: answer flips TO the
          believed/stale location (rerouting, not damage)
  delete  on false_belief prompts -> prediction: answer reverts to reality

Steering operators
  target_topk  : x -> x + alpha * P_k(mu_ref - x)   (subspace-coordinate
                 swap; mu_ref = fit-half FB mean for inject, TB mean for
                 delete). NOTE the design's "delete: project out" is
                 operationalized as swap-to-TB-mean, not coordinate
                 zeroing: the clean resid's coordinate along the wire
                 contains large shared (non-divergence) content, so
                 zeroing overshoots past TB; the fit-half TB mean is the
                 correct reference point. At k=1 this reduces to vector
                 addition toward the reference.
  target_rank1 : x -> x +/- alpha * s * v_rank1     (s = natural scale =
                 ||Dresid_fit||; + for inject, - for delete)
  controls (all additive at the SAME alpha*s magnitude, i.e.
  magnitude-matched at every dose):
    random       magnitude-matched random unit direction
    heads_ctl    unit sum of fit-half Delta-writes from layer-matched
                 non-writer heads (same count, same layers, not in T_C1)
    shuffled     Dresid recomputed with FB/TB labels permuted within the
                 fit half (the "any two prompt halves differ" null)

Readouts per prompt: p(correct)/p(alt) over the two first answer tokens,
flip_score = p_alt/(p_alt+p_correct) (same definition as the ablation
work), KL(steered || clean) at -1 (collateral-damage meter), top-1 token,
applied-delta norm; reader-head (L42-46) attention mass at query -1 on
believed vs current location tokens (secondary readout; by default
captured for clean + target directions only — pass
--attn_all_directions to capture it for the CONTROL directions too,
which is what turns the attention readout from "reliably shifts" into
"shifts more than a magnitude-matched non-wire edit").

Specificity / negative-control legs (all predicted no-ops, and all
magnitude- AND operator-matched, which the orthogonal-direction controls
are not):
  basic_ab   under inject  -> no belief variable to move
  TB prompts under delete  -> the FB component it removes is absent
  FB prompts under inject  -> already at the FB pole; nothing to add
The last two are sign-reversal controls: SAME direction, same norms,
opposite sign, on prompts where the operator's precondition fails.

Cross-domain leg: inject the ToM direction into code_alias prompts
(CODE_SYSTEM_PROMPT + quote prefill) -> does the model report the stale
pre-mutation value? Cross-domain uses additive steering only
(P_k(mu_FB^ToM - x) on code prompts would inject ToM content wholesale,
not just the divergence coordinates). The mirror (code-derived direction
into true_belief) needs a code-domain Stage-1 capture first.

Checkpointing: per-config rows append to per_prompt.csv; finished configs
are skipped on restart. --plot_only re-renders figures from the CSVs on a
laptop (no GPU).

Run via slurm_scripts/circuit_stage2_steering.sh after Stage 1's GO.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "circuit"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "codebase"))
sys.path.insert(0, str(ROOT / "codebase" / "tasks" / "causal_analysis"))

from circuit_common import (  # noqa: E402
    BAND_LAYERS, CAPTURE_ROOT, DOMAINS, EVAL_VARIANTS, INJECTION_HOOKS,
    READER_DLA_LAYERS, SITE_KEYS, STAGE1_OUT, STAGE2_OUT_ROOT,
    SYSTEM_PROMPT_MATCHED, TEMPLATES, WRITER_CSV, load_head_csv,
    split_pair_indices,
)

FB, TB = EVAL_VARIANTS  # ToM names kept for the tom-domain legs
DOSES = [0.5, 1.0, 2.0, 4.0]
CONTROL_DIRECTIONS = ["random", "heads_ctl", "shuffled"]
TARGET_DIRECTIONS = ["target_topk", "target_rank1"]
# Prompt sets steered with a direction fit in the OTHER domain get additive
# steering only (subspace swap toward a foreign domain mean would inject that
# domain's content wholesale), at the receiving domain's natural scale.
# Which sets those are depends on the run's domain — see cross_sets in main.

log = logging.getLogger("circuit-steer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# prompt sets
# ---------------------------------------------------------------------------

def heldout_capture_prompts(capture_dir: Path, dirs_npz, variant: str,
                            templates: list, divergent: str) -> list[dict]:
    """Held-out matched prompts from a Stage-1 capture (never re-generated,
    so they are byte-identical to what the directions were NOT fit on).
    `divergent` names the FB-analog variant; "believed" = the stale value."""
    rows = []
    for t in templates:
        p = capture_dir / f"{t}.pt"
        if not p.exists():
            continue
        cap = torch.load(p, map_location="cpu", weights_only=False)
        hold = dirs_npz[f"hold_idx_{t}"]
        beh = {b["idx"]: b for b in cap["behavior"][variant]}
        for i in hold:
            b = beh[int(i)]
            correct, alt = b["correct"], b["alt"]
            believed = correct if variant == divergent else alt
            current = alt if variant == divergent else correct
            rows.append(dict(prompt=cap["prompts"][variant][int(i)],
                             template=t, pair_idx=int(i), variant=variant,
                             correct=correct, alt=alt,
                             believed=believed, current=current))
    return rows


def fresh_prompt_set(tokenizer, variant: str, n: int, seed: int) -> list[dict]:
    """Generate specificity / cross-domain prompts (not tied to the capture)."""
    # deferred: importing ablation_experiment os.chdir()s to codebase/, which
    # would break the relative output paths this script resolves at startup
    from tasks.causal_analysis.ablation_experiment import generate_eval_prompts
    random.seed(seed)
    np.random.seed(seed)
    templates = TEMPLATES if not variant.startswith("code_") else ["indexed_list"]
    ps = generate_eval_prompts(tokenizer, templates, n,
                               SYSTEM_PROMPT_MATCHED, [variant],
                               prefill_mode="per_variant")[variant]
    rows = []
    for i, prompt in enumerate(ps["prompts"]):
        rows.append(dict(prompt=prompt, template=templates[0] if len(templates) == 1
                         else TEMPLATES[i // max(1, len(ps["prompts"]) // len(TEMPLATES))],
                         pair_idx=i, variant=variant,
                         correct=ps["correct"][i], alt=ps["alt"][i],
                         believed="", current=""))
    return rows


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------

def build_control_directions(capture_dir: Path, weights_path: Path,
                             split_seed: int, seed: int, site_i: int,
                             d_model: int, variants: list, templates: list,
                             heads_ctl_set: str) -> dict[str, np.ndarray]:
    """Unit control directions computed from the fit half of the capture."""
    fb_v, tb_v = variants
    rng = np.random.default_rng(seed)
    out = {"random": None, "heads_ctl": None, "shuffled": None}

    v = rng.standard_normal(d_model)
    out["random"] = v / np.linalg.norm(v)

    # shuffled-pair Dresid + fit-half Delta-z (for heads_ctl), per template
    dz_sum, shuf_sum, n_tot = None, None, 0
    for t in templates:
        p = capture_dir / f"{t}.pt"
        if not p.exists():
            continue
        cap = torch.load(p, map_location="cpu", weights_only=False)
        n = cap["meta"]["n_pairs"]
        fit, _ = split_pair_indices(n, t, split_seed)
        k = len(fit)
        rf = cap["resid"][fb_v][fit, site_i].to(torch.float64).numpy()
        rt = cap["resid"][tb_v][fit, site_i].to(torch.float64).numpy()
        pool = np.concatenate([rf, rt])
        flip = rng.permutation(2 * k)[:k]
        signs = np.full(2 * k, -1.0)
        signs[flip] = 1.0
        shuf = (pool * signs[:, None]).sum(0) / k
        shuf_sum = shuf * k if shuf_sum is None else shuf_sum + shuf * k
        dz = (cap["z"][fb_v][fit].to(torch.float64).mean(0)
              - cap["z"][tb_v][fit].to(torch.float64).mean(0)).numpy()
        dz_sum = dz * k if dz_sum is None else dz_sum + dz * k
        n_tot += k
    shuf = shuf_sum / n_tot
    out["shuffled"] = shuf / np.linalg.norm(shuf)

    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    w_o = weights["W_O_band"].to(torch.float64).numpy()
    dz = dz_sum / n_tot
    writers = load_head_csv(WRITER_CSV, head_set=heads_ctl_set)
    wset = set(zip(writers["layer"], writers["head"]))
    bi = {L: i for i, L in enumerate(BAND_LAYERS)}
    n_heads = weights["n_heads"]
    ctl_sum = np.zeros(d_model)
    used: set[tuple[int, int]] = set()
    for L, _H in sorted(wset):
        options = [h for h in range(n_heads)
                   if (L, h) not in wset and (L, h) not in used]
        h_ctl = int(rng.choice(options))
        used.add((L, h_ctl))
        ctl_sum += dz[bi[L], h_ctl] @ w_o[bi[L], h_ctl]
    out["heads_ctl"] = ctl_sum / np.linalg.norm(ctl_sum)
    log.info("control directions built (layer-matched heads: "
             f"{len(used)} sampled)")
    return out


# ---------------------------------------------------------------------------
# steering hooks
# ---------------------------------------------------------------------------

def make_steer_fn(mode_kind, dose, applied_norms, *, U=None, mu=None,
                  vec=None, scale=None, sign=1.0):
    """mode_kind 'swap' (needs U (k,d), mu (d,)) or 'add' (vec unit, scale)."""

    def steer(resid, hook):
        x = resid[:, -1, :].to(torch.float64)
        if mode_kind == "swap":
            delta = dose * ((mu - x) @ U.T) @ U
        else:
            delta = (sign * dose * scale) * vec.unsqueeze(0)
        applied_norms.append(float(delta.norm(dim=-1).mean()))
        resid[:, -1, :] = (x + delta).to(resid.dtype)
        return resid

    return steer


# ---------------------------------------------------------------------------
# attention readout helpers
# ---------------------------------------------------------------------------

def location_token_mask(tokenizer, prompt: str, needle: str, n_tokens: int):
    """Boolean mask over token positions overlapping any occurrence of
    `needle` in the prompt text."""
    mask = np.zeros(n_tokens, dtype=bool)
    if not needle:
        return mask
    enc = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    spans = []
    start = 0
    while True:
        j = prompt.find(needle, start)
        if j < 0:
            break
        spans.append((j, j + len(needle)))
        start = j + 1
    for ti, (a, b) in enumerate(enc["offset_mapping"][:n_tokens]):
        if any(a < e and b > s for s, e in spans):
            mask[ti] = True
    return mask


# ---------------------------------------------------------------------------
# evaluation core
# ---------------------------------------------------------------------------

def eval_config(model, tokenizer, prompts, fwd_hooks, applied_norms,
                clean_logprobs, capture_attn, readers, pattern_names, device):
    """One pass over a prompt set under (possibly empty) steering hooks.

    Returns (rows, attn_rows, logprobs_list or None).
    clean_logprobs: list of fp32 CPU tensors (vocab,) or None when this IS
    the clean pass (then logprobs are returned for reuse).
    """
    from tasks.causal_analysis.ablation_experiment import compute_answer_token_id

    is_clean = clean_logprobs is None
    kept_logprobs = [] if is_clean else None
    rows, attn_rows = [], []
    names_filter = (lambda n: n in pattern_names) if capture_attn else None

    for i, rec in enumerate(prompts):
        tokens = tokenizer(rec["prompt"], return_tensors="pt",
                           add_special_tokens=False)["input_ids"].to(device)
        applied_norms.clear()
        with torch.no_grad(), model.hooks(fwd_hooks=fwd_hooks):
            if capture_attn:
                logits, cache = model.run_with_cache(tokens, names_filter=names_filter)
            else:
                logits = model(tokens)
                cache = None

        logprobs = torch.log_softmax(logits[0, -1].float(), dim=-1).cpu()
        probs = logprobs.exp()
        cid = compute_answer_token_id(tokenizer, rec["variant"], rec["correct"], rec["prompt"])
        aid = compute_answer_token_id(tokenizer, rec["variant"], rec["alt"], rec["prompt"])
        p_c, p_a = float(probs[cid]), float(probs[aid])
        # KL(steered || clean) at the answer position (design readout 3)
        kl = (0.0 if is_clean else float(
            torch.sum(logprobs.exp() * (logprobs - clean_logprobs[i]))))
        rows.append(dict(
            template=rec["template"], pair_idx=rec["pair_idx"],
            variant=rec["variant"], p_correct=p_c, p_alt=p_a,
            flip_score=p_a / (p_a + p_c + 1e-12),
            kl_from_clean=kl,
            top1_str=tokenizer.decode([int(probs.argmax())]),
            applied_norm=float(np.mean(applied_norms)) if applied_norms else 0.0,
        ))
        if is_clean:
            kept_logprobs.append(logprobs)

        if cache is not None and rec.get("believed"):
            n_tok = tokens.shape[1]
            m_bel = location_token_mask(tokenizer, rec["prompt"], rec["believed"], n_tok)
            m_cur = location_token_mask(tokenizer, rec["prompt"], rec["current"], n_tok)
            for r in readers:
                L, H = r["layer"], r["head"]
                pat = cache[f"blocks.{L}.attn.hook_pattern"][0, H, -1].float().cpu().numpy()
                attn_rows.append(dict(
                    template=rec["template"], pair_idx=rec["pair_idx"],
                    variant=rec["variant"], layer=L, head=H,
                    mass_believed=float(pat[m_bel].sum()),
                    mass_current=float(pat[m_cur].sum()),
                ))
        del cache
        if (i + 1) % 50 == 0:
            log.info(f"    {i + 1}/{len(prompts)}")
    return rows, attn_rows, kept_logprobs


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def render_figures(out_dir: Path):
    per = pd.read_csv(out_dir / "per_prompt.csv")
    # preemptable-partition requeues can run overlapping attempts that
    # append the same configs more than once — keep the first occurrence
    key = ["prompt_set", "mode", "direction", "dose", "template", "pair_idx"]
    n0 = len(per)
    per = per.drop_duplicates(subset=key, keep="first")
    if len(per) < n0:
        log.info(f"dropped {n0 - len(per)} duplicate rows (requeued-job appends)")
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    summary = (per.groupby(["prompt_set", "mode", "direction", "dose"])
               .agg(flip_score=("flip_score", "mean"),
                    flip_sem=("flip_score", "sem"),
                    p_correct=("p_correct", "mean"),
                    p_alt=("p_alt", "mean"),
                    kl=("kl_from_clean", "mean"),
                    applied_norm=("applied_norm", "mean"),
                    n=("flip_score", "size"))
               .reset_index())
    summary.to_csv(out_dir / "summary.csv", index=False)

    # data-driven grid: one panel per observed (prompt_set, mode) leg,
    # one figure per metric (works for both tom- and code-domain runs)
    combos = (summary[summary["direction"] != "clean"]
              [["prompt_set", "mode"]].drop_duplicates().values.tolist())
    for metric in ("flip_score", "p_correct", "kl"):
        ncols = 3
        nrows = max(1, -(-len(combos) // ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        for ax in axes.flat:
            ax.set_axis_off()
        for ax, (pset, mode) in zip(axes.flat, combos):
            ax.set_axis_on()
            sub = summary[(summary["prompt_set"] == pset)
                          & (summary["mode"] == mode)]
            clean = summary[(summary["prompt_set"] == pset)
                            & (summary["direction"] == "clean")]
            if not clean.empty and metric != "kl":
                ax.axhline(clean[metric].iloc[0], color="0.6", ls=":", lw=1,
                           label="clean")
            for d, g in sub[sub["direction"] != "clean"].groupby("direction"):
                g = g.sort_values("dose")
                ax.plot(g["dose"], g[metric], marker="o", label=d)
            ax.set_title(f"{pset} / {mode}", fontsize=9)
            ax.set_xlabel(r"steering coefficient $\alpha$ (× natural scale)")
            ax.set_ylabel(metric)
            ax.set_xscale("log")
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / f"dose_{metric}.png", dpi=150)
        plt.close(fig)

    attn_path = out_dir / "attention.csv"
    if attn_path.exists():
        at = pd.read_csv(attn_path)
        at = at.drop_duplicates(subset=key + ["layer", "head"], keep="first")
        at["shift"] = at["mass_believed"] - at["mass_current"]
        piv = (at.groupby(["prompt_set", "mode", "direction", "dose"])["shift"]
               .mean().reset_index())
        fig, ax = plt.subplots(figsize=(7, 4))
        for (pset, mode, d), g in piv[piv["direction"] != "clean"].groupby(
                ["prompt_set", "mode", "direction"]):
            g = g.sort_values("dose")
            ax.plot(g["dose"], g["shift"], marker="o",
                    label=f"{pset}/{mode}/{d}")
        for (pset, mode), g in piv[piv["direction"] == "clean"].groupby(
                ["prompt_set", "mode"]):
            ax.axhline(g["shift"].iloc[0], ls=":", lw=1,
                       label=f"clean {pset}")
        ax.set_xscale("log")
        ax.set_xlabel(r"steering coefficient $\alpha$")
        ax.set_ylabel("reader attn mass: believed − current")
        ax.set_title("reader-head attention shift (secondary readout)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / "reader_attention_shift.png", dpi=150)
        plt.close(fig)
    log.info(f"figures -> {fig_dir}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=list(DOMAINS), default="tom")
    ap.add_argument("--capture_dir", type=Path, default=None,
                    help="default: stage1_capture/{model}[/code_domain]")
    ap.add_argument("--directions", type=Path, default=None,
                    help="default: the domain's stage1 directions_{site}.npz")
    ap.add_argument("--site", choices=SITE_KEYS, default="L41")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    ap.add_argument("--doses", type=float, nargs="+", default=DOSES)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--control_seed", type=int, default=99)
    ap.add_argument("--n_code", type=int, default=150)
    ap.add_argument("--n_basic", type=int, default=225)
    ap.add_argument("--skip_cross_domain", action="store_true")
    ap.add_argument("--only_cross_domain", action="store_true",
                    help="run ONLY the cross-domain legs (rescale reruns)")
    ap.add_argument("--cross_scale", choices=["default", "native", "other"],
                    default="default",
                    help="scale for cross-set injections: default = native "
                    "for tom runs / receiving-domain for code runs (the "
                    "original conventions); native/other force it")
    ap.add_argument("--skip_specificity", action="store_true")
    ap.add_argument("--only_controls", action="store_true",
                    help="run ONLY the control-direction legs (random / "
                         "heads_ctl / shuffled). Pair with "
                         "--attn_all_directions and a fresh --run_label to "
                         "produce the attention null for an existing run: "
                         "the finished-config skip logic would otherwise "
                         "skip those configs in the original output dir")
    ap.add_argument("--attn_all_directions", action="store_true",
                    help="capture the reader-head attention readout for "
                         "control directions too (default: target only)")
    ap.add_argument("--run_label", default=None,
                    help="default: main (tom) / code_main (code)")
    ap.add_argument("--plot_only", action="store_true")
    ap.add_argument("--dry_run", action="store_true",
                    help="build directions + config list, no model")
    ap.add_argument("--n_devices", type=int, default=1)
    ap.add_argument("--device_map", type=str, default="cuda")
    args = ap.parse_args()

    domain = DOMAINS[args.domain]
    FBD, TBD = domain["variants"]  # divergent, non-divergent
    run_label = args.run_label or ("main" if args.domain == "tom" else "code_main")
    tom_root = CAPTURE_ROOT / args.model_type
    capture_dir = args.capture_dir or (tom_root / domain["capture_subdir"])
    weights_path = tom_root / "weights.pt"
    stage1_dir = (STAGE1_OUT if args.domain == "tom"
                  else STAGE1_OUT.parent / "stage1_code")

    out_dir = STAGE2_OUT_ROOT / args.model_type / f"{args.site}_{run_label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        render_figures(out_dir)
        return

    dirs_path = args.directions or (stage1_dir / f"directions_{args.site}.npz")
    dz = np.load(dirs_path)
    site_i = SITE_KEYS.index(args.site)
    hook_name = INJECTION_HOOKS[args.site]
    v_rank1 = dz["v_rank1"]
    U_topk = dz["U_topk"]
    mu_fb, mu_tb = dz["mean_FB_fit"], dz["mean_TB_fit"]
    scale = float(dz["natural_scale"])
    d_model = v_rank1.shape[0]
    log.info(f"directions: {dirs_path.name} k={int(dz['k'])} "
             f"natural_scale={scale:.2f} hook={hook_name}")

    # cross-set injections run at the RECEIVING domain's natural scale when
    # that domain's stage1 file exists (else fall back to the native scale)
    cross_sets = {"code_alias"} if args.domain == "tom" else {"mirror_tom_tb"}
    other_stage1 = (STAGE1_OUT.parent / "stage1_code" if args.domain == "tom"
                    else STAGE1_OUT)
    other_npz_path = other_stage1 / f"directions_{args.site}.npz"
    dz_other = np.load(other_npz_path) if other_npz_path.exists() else None
    if args.cross_scale == "native":
        cross_scale = scale
    elif args.cross_scale == "other":
        if dz_other is None:
            sys.exit(f"--cross_scale other needs {other_npz_path}")
        cross_scale = float(dz_other["natural_scale"])
    else:  # original conventions: tom runs used native scale (keeps resumed
        # runs consistent); code runs use the receiving (ToM) scale
        cross_scale = (scale if args.domain == "tom" or dz_other is None
                       else float(dz_other["natural_scale"]))

    controls = build_control_directions(
        capture_dir, weights_path, args.split_seed, args.control_seed,
        site_i, d_model, domain["variants"], domain["templates"],
        heads_ctl_set=("primary" if args.domain == "tom" else "code_abstract"))

    # ---- config grid ------------------------------------------------------
    # (prompt_set, mode, direction) legs; dose grid applied to each.
    main_dirs = (CONTROL_DIRECTIONS if args.only_controls
                 else TARGET_DIRECTIONS + CONTROL_DIRECTIONS)
    legs = [(TBD, "inject", d) for d in main_dirs]
    legs += [(FBD, "delete", d) for d in main_dirs]
    if args.only_cross_domain:
        legs = []
    if not args.skip_specificity and not (args.only_cross_domain
                                          or args.only_controls):
        if args.domain == "tom":
            legs += [("basic_ab", "inject", d) for d in TARGET_DIRECTIONS]
        # sign-reversal negative controls: same direction and norms, operator
        # applied where its precondition fails (nothing to delete on TB /
        # nothing to add on FB). Predicted no-ops; a flip here would mean the
        # operators damage rather than move the wire coordinate.
        legs += [(TBD, "delete", d) for d in TARGET_DIRECTIONS]
        legs += [(FBD, "inject", d) for d in TARGET_DIRECTIONS]
    if not args.skip_cross_domain and not args.only_controls:
        if args.domain == "tom":
            legs += [("code_alias", "inject", d) for d in TARGET_DIRECTIONS]
        else:
            # mirror: code direction into held-out ToM true_belief prompts
            # (pre-registered prediction: null, given wire cos = 0.0997)
            legs += [("mirror_tom_tb", "inject", d) for d in TARGET_DIRECTIONS]

    per_path = out_dir / "per_prompt.csv"
    attn_path = out_dir / "attention.csv"
    done = set()
    if per_path.exists():
        prev = pd.read_csv(per_path)
        done = set(zip(prev["prompt_set"], prev["mode"], prev["direction"],
                       prev["dose"]))
        log.info(f"resuming: {len(done)} configs already in {per_path.name}")

    if args.dry_run:
        for leg in legs:
            print("leg:", leg, "doses:", args.doses)
        print(f"controls: { {k: v[:3].round(4).tolist() for k, v in controls.items()} }")
        return

    # ---- model ------------------------------------------------------------
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

    weights = torch.load(weights_path, map_location="cpu", weights_only=False)
    readers = [r for r in weights["readers"]
               if READER_DLA_LAYERS[0] <= r["layer"] <= READER_DLA_LAYERS[1]]
    pattern_names = {f"blocks.{r['layer']}.attn.hook_pattern" for r in readers}

    tt = lambda a: torch.tensor(a, dtype=torch.float64, device=device)
    U_t, mu_fb_t, mu_tb_t = tt(U_topk), tt(mu_fb), tt(mu_tb)
    dir_vecs = {"target_rank1": tt(v_rank1),
                "cross_topk_delta": tt((U_topk.T @ (U_topk @ (mu_fb - mu_tb)))
                                       / np.linalg.norm(U_topk @ (mu_fb - mu_tb))),
                **{k: tt(v) for k, v in controls.items()}}

    # ---- prompt sets (built lazily, clean pass cached per set) ------------
    prompt_sets: dict[str, list] = {}

    def get_prompts(pset):
        if pset not in prompt_sets:
            if pset in (FBD, TBD):
                prompt_sets[pset] = heldout_capture_prompts(
                    capture_dir, dz, pset, domain["templates"], divergent=FBD)
            elif pset == "mirror_tom_tb":
                if dz_other is None:
                    sys.exit("mirror leg needs the ToM stage1 directions file "
                             f"({other_npz_path}) for held-out indices")
                prompt_sets[pset] = heldout_capture_prompts(
                    tom_root, dz_other, TB, DOMAINS["tom"]["templates"],
                    divergent=FB)
            elif pset == "basic_ab":
                prompt_sets[pset] = fresh_prompt_set(tokenizer, "basic_ab",
                                                     args.n_basic, args.control_seed)
            elif pset == "code_alias":
                prompt_sets[pset] = fresh_prompt_set(tokenizer, "code_alias",
                                                     args.n_code, args.control_seed)
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

    def run_clean(pset):
        if pset in clean_cache:
            return
        prompts = get_prompts(pset)
        capture_attn = pset in (FBD, TBD, "mirror_tom_tb")
        log.info(f"clean pass: {pset}")
        rows, attn_rows, logprobs = eval_config(
            model, tokenizer, prompts, [], applied_norms, None,
            capture_attn, readers, pattern_names, device)
        clean_cache[pset] = logprobs
        if (pset, "clean", "clean", 0.0) not in done:
            append_rows(rows, pset, "clean", "clean", 0.0, attn_rows)
            done.add((pset, "clean", "clean", 0.0))

    for pset, mode, direction in legs:
        if all((pset, mode, direction, d) in done for d in args.doses):
            continue  # fully resumed leg: don't pay for its clean pass
        run_clean(pset)
        for dose in args.doses:
            key = (pset, mode, direction, dose)
            if key in done:
                continue
            log.info(f"config: {key}")
            use_scale = cross_scale if pset in cross_sets else scale
            if direction == "target_topk":
                if pset in cross_sets:
                    fn = make_steer_fn("add", dose, applied_norms,
                                       vec=dir_vecs["cross_topk_delta"],
                                       scale=use_scale, sign=+1.0)
                else:
                    mu = mu_fb_t if mode == "inject" else mu_tb_t
                    fn = make_steer_fn("swap", dose, applied_norms,
                                       U=U_t, mu=mu)
            else:
                vec = dir_vecs["target_rank1" if direction == "target_rank1"
                               else direction]
                sign = +1.0 if mode == "inject" else -1.0
                fn = make_steer_fn("add", dose, applied_norms, vec=vec,
                                   scale=use_scale, sign=sign)
            capture_attn = (pset in (FBD, TBD, "mirror_tom_tb")
                            and (direction in TARGET_DIRECTIONS
                                 or args.attn_all_directions))
            rows, attn_rows, _ = eval_config(
                model, tokenizer, get_prompts(pset), [(hook_name, fn)],
                applied_norms, clean_cache[pset], capture_attn, readers,
                pattern_names, device)
            append_rows(rows, pset, mode, direction, dose, attn_rows)
            done.add(key)

    (out_dir / "config.json").write_text(json.dumps(dict(
        vars(args), domain=args.domain, capture_dir=str(capture_dir),
        directions=str(dirs_path), out_dir=str(out_dir), doses=args.doses,
        k=int(dz["k"]), natural_scale=scale, cross_scale=cross_scale,
        hook=hook_name, writer_set=str(dz["writer_set"]),
    ), indent=2, default=str))
    render_figures(out_dir)
    log.info(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
