"""Adjust the 16 saved geometric-projection comparisons using Holm's method.

The family is all rows in s1a_projection.csv: two residual sites, four
template scopes, and two subspace types. This does not adjust other analyses
or rerun model inference. Original Monte Carlo probabilities are preserved.
"""

import csv
from pathlib import Path

import numpy as np
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/aggregated/circuit/stage1/s1a_projection.csv"
OUTPUT = SOURCE.with_name("s1a_projection_holm.csv")


def main():
    with SOURCE.open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    keys = {(row["site"], row["scope"], row["subspace"]) for row in rows}
    expected = {
        (site, scope, subspace)
        for site in ("L40", "L41")
        for scope in ("food_truck", "hair_styling", "library_book", "pooled")
        for subspace in ("writers", "mlp")
    }
    assert keys == expected and len(rows) == 16, "Review the comparison family"
    p = np.array([float(row["p_perm"]) for row in rows], dtype=np.float64)
    assert np.isfinite(p).all() and ((p > 0) & (p <= 1)).all()
    reject, adjusted, _, _ = multipletests(p, alpha=0.05, method="holm")
    # Check the adjustment independently using the ordered Holm formula.
    order = np.argsort(p)
    check = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    np.testing.assert_allclose(adjusted[order], check)
    print(f"Input: {SOURCE}; {len(rows)} comparisons; dtype={p.dtype}")
    for row, value, significant in zip(rows, adjusted, reject):
        row.update(p_holm=float(value), family_size=len(rows), reject_0_05=bool(significant))
        if row["scope"] == "pooled" and row["subspace"] == "writers":
            print(f"{row['site']} pooled heads: overlap={float(row['frac_hold']):.6f}, "
                  f"p={float(row['p_perm']):.9f}, Holm p={value:.9f}")
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields + ["p_holm", "family_size", "reject_0_05"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
