"""Steering-coefficient extension analysis (Nov 2026, camera-ready).

Merges the frozen paper run (L41_main, alpha <= 4) with the extension run
(L41_dose_ext: alpha in {6, 8, 12, 16} on the belief legs, plus the generic
belief-free leg over the full grid) and writes the tables the appendix
figure (scripts/rebuttal_figures/fig14_dose_extension.py) and the paper
text quote. No inference; saved CSVs only.

Outputs (results/aggregated/circuit/dose_extension/):
  clean_reproduction.csv  per prompt-set agreement of the two runs' clean
                          passes (same prompts, different GPU/run)
  belief_outcomes.csv     per (prompt_set, mode, direction, dose): first-token
                          accuracy, alternative-location rate, off-manifold
                          rate, p(correct), KL, n — belief legs, all alphas
  merged_summary.csv      per-config means for every prompt set in either run
  generic_summary.csv     generic leg: KL, top-1 agreement, p(top-1)
  generic_by_family.csv   top-1 agreement per prompt family (target inject)
  REPORT.md               the tables above as markdown plus the quoted numbers

Definitions (the paper's): accuracy = argmax next token begins the correct
answer and not the alternative; alternative rate = it begins the alternative
and not the correct answer (an on-manifold answer change); off-manifold =
neither. KL = KL(steered || clean) of the next-token distribution at the
edited final token, nats. top1_agree = argmax equals the clean model's.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results/circuit/stage2_steering/Qwen2.5-14B-Instruct"
CAPTURE = ROOT / "results/circuit/stage1_capture/Qwen2.5-14B-Instruct"
OUT = ROOT / "results/aggregated/circuit/dose_extension"
TEMPLATES = ("food_truck", "hair_styling", "library_book")
KEY = ["prompt_set", "mode", "direction", "dose"]
PROMPT_KEY = ["prompt_set", "template", "pair_idx"]
BELIEF_SETS = ("false_belief", "true_belief")


def load_run(label: str) -> pd.DataFrame:
    df = pd.read_csv(RUN_ROOT / f"L41_{label}" / "per_prompt.csv")
    # requeued preemptable jobs can append a config twice; keep the first
    df = df.drop_duplicates(subset=KEY + ["template", "pair_idx"], keep="first")
    for col in ("top1_agree", "p_top1"):
        if col not in df.columns:
            df[col] = np.nan
    return df.assign(run=label)


def label_belief_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Attach correct/alt strings to held-out belief rows and classify the
    argmax token (same rule as fig5_steering.accuracy_table)."""
    beh = pd.concat(pd.read_csv(CAPTURE / f"{t}_behavior.csv").assign(template=t)
                    for t in TEMPLATES)
    assert not beh.first_token_collision.any()
    b = df[df.prompt_set.isin(BELIEF_SETS)].merge(
        beh[["template", "idx", "variant", "correct", "alt"]],
        left_on=["template", "pair_idx", "variant"],
        right_on=["template", "idx", "variant"], how="inner")
    assert len(b) == df.prompt_set.isin(BELIEF_SETS).sum(), "unlabelled belief rows"
    t1 = b.top1_str.astype(str).str.strip()
    hit_c = np.array([len(t) > 0 and c.startswith(t) for t, c in zip(t1, b.correct)])
    hit_a = np.array([len(t) > 0 and a.startswith(t) for t, a in zip(t1, b.alt)])
    b["acc"] = (hit_c & ~hit_a).astype(float)
    b["alt_rate"] = (hit_a & ~hit_c).astype(float)
    b["off_manifold"] = (~hit_c & ~hit_a).astype(float)
    return b


def clean_reproduction(main: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    cm = main[main.direction == "clean"]
    ce = ext[ext.direction == "clean"]
    m = cm.merge(ce, on=PROMPT_KEY, suffixes=("_main", "_ext"))
    d = m.p_correct_ext - m.p_correct_main
    out = (m.assign(d=d, same_top1=(m.top1_str_main == m.top1_str_ext))
            .groupby("prompt_set")
            .agg(n=("d", "size"), mean_signed_dp=("d", "mean"),
                 mean_abs_dp=("d", lambda s: s.abs().mean()),
                 max_abs_dp=("d", lambda s: s.abs().max()),
                 n_abs_dp_gt_0_1=("d", lambda s: int((s.abs() > 0.1).sum())),
                 top1_agreement=("same_top1", "mean"))
            .reset_index())
    return out


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pivot(df, index, value, doses=None):
    p = df.pivot_table(index=index, columns="dose", values=value)
    if doses is not None:
        p = p.reindex(columns=[d for d in doses if d in p.columns])
    p.columns = [f"a={c:g}" for c in p.columns]
    return p.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    main_run = load_run("main")
    ext = load_run("dose_ext")
    repro = clean_reproduction(main_run, ext)
    repro.to_csv(args.out / "clean_reproduction.csv", index=False)

    # The merge rule: alpha <= 4 rows come from the paper run (L41_main),
    # alpha >= 6 and the generic set from the extension. The extension's
    # own clean rows are kept only for the generic set (its clean pass is the
    # reference for its KL/top-1 columns); belief/basic/code clean rows come
    # from L41_main so the alpha <= 4 cells are exactly the paper's.
    ext_keep = ext[(ext.prompt_set == "generic") | (ext.dose >= 6)]
    both = pd.concat([main_run, ext_keep], ignore_index=True)
    both = both[both.direction != "target_topk"]
    dup = both.duplicated(subset=KEY + ["template", "pair_idx"])
    assert not dup.any(), f"{dup.sum()} overlapping rows between runs"

    merged = (both.groupby(KEY)
              .agg(p_correct=("p_correct", "mean"), p_alt=("p_alt", "mean"),
                   flip_score=("flip_score", "mean"), kl=("kl_from_clean", "mean"),
                   kl_sem=("kl_from_clean", "sem"), top1_agree=("top1_agree", "mean"),
                   p_top1=("p_top1", "mean"), applied_norm=("applied_norm", "mean"),
                   n=("kl_from_clean", "size"), run=("run", "first"))
              .reset_index())
    merged.to_csv(args.out / "merged_summary.csv", index=False)

    lab = label_belief_rows(both)
    outcomes = (lab.groupby(KEY)
                .agg(accuracy=("acc", "mean"), acc_sem=("acc", "sem"),
                     alt_rate=("alt_rate", "mean"), alt_sem=("alt_rate", "sem"),
                     off_manifold=("off_manifold", "mean"), off_sem=("off_manifold", "sem"),
                     p_correct=("p_correct", "mean"), kl=("kl_from_clean", "mean"),
                     n=("acc", "size"))
                .reset_index())
    outcomes.to_csv(args.out / "belief_outcomes.csv", index=False)

    gen = both[both.prompt_set == "generic"]
    generic = (gen.groupby(KEY)
               .agg(kl=("kl_from_clean", "mean"), kl_sem=("kl_from_clean", "sem"),
                    top1_agree=("top1_agree", "mean"), top1_sem=("top1_agree", "sem"),
                    p_top1=("p_top1", "mean"), applied_norm=("applied_norm", "mean"),
                    n=("kl_from_clean", "size"))
               .reset_index())
    generic.to_csv(args.out / "generic_summary.csv", index=False)
    fam = (gen[(gen["mode"] == "inject") & (gen.direction == "target_rank1")]
           .assign(family=lambda d: d.template.str.replace("generic:", "", regex=False))
           .groupby(["family", "dose"]).top1_agree.mean().reset_index())
    fam.to_csv(args.out / "generic_by_family.csv", index=False)
    clean_gen = gen[gen.direction == "clean"]

    # ---- report -----------------------------------------------------------
    def leg(ps, mode):
        return outcomes[(outcomes.prompt_set == ps)
                        & (outcomes["mode"].isin([mode, "clean"]))]

    def q(ps, mode, direction, dose, col, table=outcomes):
        r = table[(table.prompt_set == ps) & (table["mode"] == mode)
                  & (table.direction == direction) & (table.dose == dose)]
        return float(r[col].iloc[0])

    tb_alt = [q("true_belief", "inject", "target_rank1", a, "alt_rate")
              for a in (1, 2, 4, 6, 8)]
    tb_off = [q("true_belief", "inject", "target_rank1", a, "off_manifold")
              for a in (4, 6, 8)]
    g_t = {a: (q("generic", "inject", "target_rank1", a, "kl", generic),
               q("generic", "inject", "target_rank1", a, "top1_agree", generic))
           for a in (2, 4, 6, 8)}
    g_r = {a: (q("generic", "inject", "random", a, "kl", generic),
               q("generic", "inject", "random", a, "top1_agree", generic))
           for a in (2, 4, 6, 8)}

    lines = [
        "# Steering-coefficient extension (layer-41 rank-1 residual edit, Qwen2.5-14B-Instruct)",
        "",
        "Source runs: `L41_main` (paper, alpha <= 4) and `L41_dose_ext` (alpha in {6, 8, 12, 16}; "
        "generic leg over {0.5, ..., 16}). Same held-out prompts, same `--control_seed 99`. "
        "Rank-1 direction only; the 8-dim swap operator is off-manifold by alpha = 4 and was not extended.",
        "",
        "Metric definitions are in the module docstring of `scripts/circuit/analyze_dose_extension.py`. "
        f"Applied edit norm is alpha x {merged[merged.dose == 1].applied_norm.iloc[0]:.1f}; "
        "the clean residual norm at this site is about 390 (stage-1 capture, final token).",
        "",
        "## Quoted numbers",
        "",
        "- True-belief injection, alternative-location rate (on-manifold flip) at alpha 1/2/4/6/8: "
        + " / ".join(f"{100 * v:.1f}%" for v in tb_alt) + ". Peak at alpha = 4.",
        "- True-belief injection, off-manifold rate at alpha 4/6/8: "
        + " / ".join(f"{100 * v:.1f}%" for v in tb_off) + ".",
        f"- Generic prompts (n = {int(generic.n.iloc[0])}), target direction inject, KL / top-1 unchanged: "
        + "; ".join(f"alpha {a}: {k:.3f} nats / {100 * t:.1f}%" for a, (k, t) in g_t.items()) + ".",
        "- Generic prompts, norm-matched random direction: "
        + "; ".join(f"alpha {a}: {k:.3f} nats / {100 * t:.1f}%" for a, (k, t) in g_r.items()) + ".",
        f"- Generic clean confidence: median p(top-1) = {clean_gen.p_top1.median():.3f}, "
        f"fraction with p(top-1) > 0.5 = {(clean_gen.p_top1 > 0.5).mean():.3f}.",
        "",
        "## Clean-pass reproduction (L41_dose_ext vs L41_main, same prompts)",
        "",
        md_table(repro, "{:.4f}"),
        "",
    ]
    for ps, mode, title in [("true_belief", "inject", "True belief, inject"),
                            ("false_belief", "delete", "False belief, delete"),
                            ("true_belief", "delete", "True belief, delete (precondition absent)"),
                            ("false_belief", "inject", "False belief, inject (precondition absent)")]:
        L = leg(ps, mode)
        lines += [f"## {title}", "", "Accuracy (%)", "",
                  md_table(pivot(L.assign(accuracy=100 * L.accuracy), "direction", "accuracy"), "{:.1f}"),
                  "", "Alternative-location rate (%)", "",
                  md_table(pivot(L.assign(alt_rate=100 * L.alt_rate), "direction", "alt_rate"), "{:.1f}"),
                  "", "Off-manifold rate (%)", "",
                  md_table(pivot(L.assign(off_manifold=100 * L.off_manifold), "direction", "off_manifold"), "{:.1f}"),
                  "", "KL(steered || clean), nats", "",
                  md_table(pivot(L, "direction", "kl"), "{:.3f}"), ""]
    for ps in ("basic_ab", "code_alias"):
        L = merged[merged.prompt_set == ps]
        lines += [f"## {ps} (belief-free, target inject)", "", "p(correct)", "",
                  md_table(pivot(L, "direction", "p_correct")), "", "KL", "",
                  md_table(pivot(L, "direction", "kl")), "", "top-1 unchanged (extension doses only)", "",
                  md_table(pivot(L, "direction", "top1_agree")), ""]
    G = generic.assign(mode_dir=generic["mode"] + "/" + generic.direction)
    lines += ["## Generic belief-free prompts", "", "KL(steered || clean), nats", "",
              md_table(pivot(G, "mode_dir", "kl")), "", "Top-1 unchanged (fraction)", "",
              md_table(pivot(G, "mode_dir", "top1_agree")), "", "Mean p(top-1)", "",
              md_table(pivot(G, "mode_dir", "p_top1")), "",
              "Top-1 unchanged by family, target inject", "",
              md_table(pivot(fam, "family", "top1_agree"), "{:.2f}"), ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:16]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
