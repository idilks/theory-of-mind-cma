"""Summarize a stage2_route_decomposition.py run.

Per prompt, v(S) = margin(config with routes in S steered, others clean)
- margin(clean), for S a subset of {direct, attn, mlp}; the 8 factorial
runs give every v(S). Shapley share of route i:
    phi_i = sum_{S not containing i} |S|!(3-|S|-1)!/3! [v(S+i) - v(S)]
Shares sum to the full effect v({direct,attn,mlp}) exactly, interactions
included. Reported as means over prompts with SEM and as fractions of
the mean effect, per variant and per template. Also: single-route
necessity (steered minus the run with that route alone restored),
single-route sufficiency (that route alone steered), the all-heads
attention-WEIGHT route (pattern_all / pattern_all_tr), and per-layer
attn/mlp necessity. Checks exact.csv first.
"""

from __future__ import annotations

import argparse
import itertools
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "results/circuit/stage2_steering/Qwen2.5-14B-Instruct/L41_route_decomp"
KEY = ["prompt_set", "template", "pair_idx"]
ROUTES = ["direct", "attn", "mlp"]


def shapley_three(v: dict[str, float]) -> dict[str, float]:
    """v keyed by 'dam' strings (c/s per route in ROUTES order)."""
    n = len(ROUTES)
    phi = {}
    for i, route in enumerate(ROUTES):
        total = 0.0
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                key_without = "".join("s" if j in S else "c" for j in range(n))
                key_with = "".join("s" if (j in S or j == i) else "c" for j in range(n))
                w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
                total += w * (v[key_with] - v[key_without])
        phi[route] = total
    return phi


def stats(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    if len(d) == 0 or np.allclose(d, 0):
        return dict(n=len(d), mean=float(d.mean()) if len(d) else np.nan, sem=np.nan, p=1.0)
    return dict(n=len(d), mean=float(d.mean()), sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                p=float(wilcoxon(d).pvalue))


def summarize(run_dir: Path):
    per = pd.read_csv(run_dir / "per_prompt.csv")
    ex = pd.read_csv(run_dir / "exact.csv")
    M = per.pivot_table(index=KEY, columns="run", values="margin")
    clean = M["fac_ccc"]
    v = {}
    for d, a, m in itertools.product("cs", repeat=3):
        v[f"{d}{a}{m}"] = (M[f"fac_{d}{a}{m}"] - clean).to_numpy()
    n = len(clean)
    phis = {r: np.zeros(n) for r in ROUTES}
    for i in range(n):
        phi = shapley_three({k: val[i] for k, val in v.items()})
        for r in ROUTES:
            phis[r][i] = phi[r]
    W = pd.DataFrame(index=M.index)
    W["effect"] = v["sss"]
    for r in ROUTES:
        W[f"shapley_{r}"] = phis[r]
    W["suff_direct"], W["suff_attn"], W["suff_mlp"] = v["scc"], v["csc"], v["ccs"]
    W["nec_direct"] = v["sss"] - v["css"]
    W["nec_attn"] = v["sss"] - v["scs"]
    W["nec_mlp"] = v["sss"] - v["ssc"]
    W["nec_pattern_all"] = v["sss"] - (M["pattern_all"] - clean).to_numpy()
    W["suff_pattern_all"] = (M["pattern_all_tr"] - clean).to_numpy()
    layer_cols = sorted(c for c in M.columns if c.startswith(("attn_L", "mlp_L")))
    for c in layer_cols:
        W[f"nec_{c}"] = v["sss"] - (M[c] - clean).to_numpy()
    W = W.reset_index()

    rows = []
    for (variant, template), g in list(W.groupby(["prompt_set", "template"])) + \
            [((variant, "all"), g) for variant, g in W.groupby("prompt_set")]:
        eff = g["effect"].mean()
        for col in [c for c in W.columns if c not in KEY]:
            s = stats(g[col].to_numpy())
            rows.append(dict(variant=variant, template=template, quantity=col,
                             **s, frac_of_effect=s["mean"] / eff))
    return pd.DataFrame(rows), W, ex


def table(summary: pd.DataFrame, variant: str, quantities: list[str]) -> str:
    sel = summary[(summary.variant == variant) & (summary.template == "all")
                  & summary.quantity.isin(quantities)].set_index("quantity").loc[quantities]
    out = [f"| {variant} | mean ± SEM (logits) | /effect | p |", "|---|---|---|---|"]
    for q, r in sel.iterrows():
        sem = "" if not np.isfinite(r.sem) else f" ± {r.sem:.3f}"
        out.append(f"| {q} | {r['mean']:.3f}{sem} | {r.frac_of_effect:.2f} | {r.p:.1e} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, default=DEFAULT_RUN)
    args = ap.parse_args()
    summary, W, ex = summarize(args.run_dir)
    summary.to_csv(args.run_dir / "summary.csv", index=False)
    W.to_csv(args.run_dir / "per_prompt_wide.csv", index=False)
    print(f"n prompts: {W.groupby('prompt_set').size().to_dict()}")
    print(f"exactness: max scc_max_abs_err = {ex.scc_max_abs_err.max():.3e}, "
          f"max scc_rel_err = {ex.scc_rel_err.max():.3e}; "
          f"delta norm {ex.delta_norm.mean():.1f}, resid_final shift norm {ex.resid_final_shift_norm.mean():.1f}\n")
    main_q = ["effect", "shapley_direct", "shapley_attn", "shapley_mlp",
              "suff_direct", "suff_attn", "suff_mlp", "nec_direct", "nec_attn", "nec_mlp",
              "nec_pattern_all", "suff_pattern_all"]
    layer_q = sorted(q for q in summary.quantity.unique() if q.startswith("nec_attn_L") or q.startswith("nec_mlp_L"))
    for variant in sorted(W.prompt_set.unique()):
        print(f"## {variant}: Shapley shares, single-route sufficiency/necessity\n")
        print(table(summary, variant, main_q))
        print(f"\n## {variant}: per-layer necessity (steered minus that layer's output restored)\n")
        print(table(summary, variant, layer_q))
        print()
    print("## Shapley fraction by template\n")
    sel = summary[summary.quantity.str.startswith("shapley_") & (summary.template != "all")]
    print(sel.pivot_table(index=["variant", "template"], columns="quantity",
                          values="frac_of_effect").round(2).to_string())
    print(f"\nwrote {args.run_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
