"""Summarize a stage2_reader_restoration.py run.

Per prompt and site, with margin = logit(alt) - logit(correct):
  effect       = steered - clean                   (the steering effect)
  recovery     = steered - restore_{site}          (necessity: >0 means
                 restoring the readers' site undoes part of the effect)
  ctl_recovery = steered - restore_{site}_ctl{d}   (same for a layer-matched
                 random set; averaged over draws)
  transplant   = transplant_{site} - clean         (sufficiency: the steered
                 activation alone, in a clean run)
Fractions (recovery/effect, transplant/effect) are reported as ratios of
MEANS, not means of ratios, and only as descriptives: interactions make
restoration and transplantation non-additive.

Tests: two-sided paired Wilcoxon signed-rank across prompts within each
variant, for recovery vs 0, recovery vs ctl_recovery (readers minus
control, paired on prompt), and transplant vs 0. The same for the
readers' believed-minus-current attention mass (attention.csv, averaged
over the 26 readers per prompt).

Writes summary.csv (one row per variant x site x quantity) and
per_prompt_wide.csv next to the run, plus a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_reader_restore"
KEY = ["prompt_set", "template", "pair_idx"]


def wide(df: pd.DataFrame, value: str) -> pd.DataFrame:
    return df.pivot_table(index=KEY, columns="run", values=value)


def paired_stats(x: np.ndarray, y: np.ndarray | None = None) -> dict:
    """x (or x - y) vs 0: mean, SEM, fraction positive, Wilcoxon p."""
    d = x if y is None else x - y
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return dict(n=len(d), mean=float(d.mean()) if len(d) else np.nan,
                    sem=np.nan, frac_pos=np.nan, p=1.0)
    return dict(n=len(d), mean=float(d.mean()), sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                frac_pos=float((d > 0).mean()), p=float(wilcoxon(d).pvalue))


def summarize(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    per = pd.read_csv(run_dir / "per_prompt.csv")
    att = pd.read_csv(run_dir / "attention.csv")
    sites = sorted({s for s in per["restore_site"].dropna().unique() if s})
    n_ctl = per["head_set"].str.startswith("ctl").sum() // max(1, len(sites) * per.groupby(KEY).ngroups)
    M = wide(per, "margin")
    F = wide(per, "flip_score")
    att["mass_diff"] = att["mass_believed"] - att["mass_current"]
    A = att.groupby(KEY + ["run"])["mass_diff"].mean().unstack("run")

    rows, wide_rows = [], M[["clean", "steered"]].copy()
    wide_rows.columns = ["margin_clean", "margin_steered"]
    wide_rows["effect"] = M["steered"] - M["clean"]
    wide_rows["flip_clean"], wide_rows["flip_steered"] = F["clean"], F["steered"]
    wide_rows["attn_clean"], wide_rows["attn_steered"] = A["clean"], A["steered"]
    for site in sites:
        ctl_cols = [c for c in M.columns if c.startswith(f"restore_{site}_ctl")]
        wide_rows[f"recovery_{site}"] = M["steered"] - M[f"restore_{site}"]
        wide_rows[f"ctl_recovery_{site}"] = M["steered"] - M[ctl_cols].mean(axis=1)
        wide_rows[f"transplant_{site}"] = M[f"transplant_{site}"] - M["clean"]
        wide_rows[f"flip_restore_{site}"] = F[f"restore_{site}"]
        wide_rows[f"flip_transplant_{site}"] = F[f"transplant_{site}"]
        wide_rows[f"attn_recovery_{site}"] = A["steered"] - A[f"restore_{site}"]
        wide_rows[f"attn_ctl_recovery_{site}"] = A["steered"] - A[ctl_cols].mean(axis=1)
        wide_rows[f"attn_transplant_{site}"] = A[f"transplant_{site}"] - A["clean"]

    for variant, g in wide_rows.groupby(level="prompt_set"):
        eff = g["effect"].to_numpy()
        rows.append(dict(variant=variant, site="", quantity="effect (steered-clean)",
                         **paired_stats(eff), frac_of_effect=1.0,
                         flip_clean=g["flip_clean"].mean(), flip_steered=g["flip_steered"].mean()))
        rows.append(dict(variant=variant, site="", quantity="attn effect (believed-current)",
                         **paired_stats((g["attn_steered"] - g["attn_clean"]).to_numpy()),
                         frac_of_effect=np.nan))
        for site in sites:
            rec = g[f"recovery_{site}"].to_numpy()
            ctl = g[f"ctl_recovery_{site}"].to_numpy()
            tr = g[f"transplant_{site}"].to_numpy()
            rows.append(dict(variant=variant, site=site, quantity="recovery (steered-restored)",
                             **paired_stats(rec), frac_of_effect=rec.mean() / eff.mean(),
                             flip_restored=g[f"flip_restore_{site}"].mean()))
            rows.append(dict(variant=variant, site=site, quantity="ctl recovery (random set)",
                             **paired_stats(ctl), frac_of_effect=ctl.mean() / eff.mean()))
            rows.append(dict(variant=variant, site=site, quantity="recovery minus ctl",
                             **paired_stats(rec, ctl), frac_of_effect=(rec - ctl).mean() / eff.mean()))
            rows.append(dict(variant=variant, site=site, quantity="transplant (transplanted-clean)",
                             **paired_stats(tr), frac_of_effect=tr.mean() / eff.mean(),
                             flip_transplanted=g[f"flip_transplant_{site}"].mean()))
            a_eff = (g["attn_steered"] - g["attn_clean"]).to_numpy()
            for q, col in (("attn recovery", f"attn_recovery_{site}"),
                           ("attn ctl recovery", f"attn_ctl_recovery_{site}"),
                           ("attn transplant", f"attn_transplant_{site}")):
                v = g[col].to_numpy()
                rows.append(dict(variant=variant, site=site, quantity=q, **paired_stats(v),
                                 frac_of_effect=v.mean() / a_eff.mean()))
            v = g[f"attn_recovery_{site}"].to_numpy()
            c = g[f"attn_ctl_recovery_{site}"].to_numpy()
            rows.append(dict(variant=variant, site=site, quantity="attn recovery minus ctl",
                             **paired_stats(v, c), frac_of_effect=(v - c).mean() / a_eff.mean()))
    return pd.DataFrame(rows), wide_rows.reset_index()


def fmt(summary: pd.DataFrame, which: str) -> str:
    sel = summary[~summary["quantity"].str.startswith("attn")] if which == "margin" \
        else summary[summary["quantity"].str.startswith("attn")]
    out = [f"| variant | site | quantity | mean ± SEM | frac>0 | p (Wilcoxon) | /effect |",
           "|---|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        sem = "" if not np.isfinite(r.sem) else f" ± {r.sem:.3f}"
        fp = "" if not np.isfinite(r.frac_pos) else f"{r.frac_pos:.2f}"
        fr = "" if not np.isfinite(r.frac_of_effect) else f"{r.frac_of_effect:.2f}"
        out.append(f"| {r.variant} | {r.site} | {r.quantity} | {r.mean:.3f}{sem} | {fp} | {r.p:.2e} | {fr} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    summary, wide_rows = summarize(args.run_dir)
    summary.to_csv(args.run_dir / "summary.csv", index=False)
    wide_rows.to_csv(args.run_dir / "per_prompt_wide.csv", index=False)
    print(f"n prompts: {wide_rows.groupby('prompt_set').size().to_dict()}\n")
    print("## Answer margin (logit alt - logit correct)\n")
    print(fmt(summary, "margin"))
    print("\n## Reader attention (believed - current mass, mean over 26 readers)\n")
    print(fmt(summary, "attn"))
    print(f"\nwrote {args.run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
