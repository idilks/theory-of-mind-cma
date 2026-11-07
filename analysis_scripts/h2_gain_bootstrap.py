"""Bootstrap CI for the H2 additivity gain (draft2 Results, "Modularized representations").

The hypothesis is r(C3, C1 + C5) > max(r(C3, C1), r(C3, C5)). This script
resamples the 1920 heads with replacement (10,000 draws, seed 0) and reports the
95% percentile interval of two paired differences:
  gain_vs_best_part = r(C3, C1+C5) - max(r(C3, C1), r(C3, C5))
  gain_vs_C2+C6     = r(C3, C1+C5) - r(C3, C2+C6)   (axis-matched substitute)
Heads are not independent samples, so this measures how much the gain depends
on which heads dominate the correlation, not sampling error over prompts.

Inputs: results/aggregated/fingerprints/*.npy (the arrays behind r = 0.971).
Output: results/summary/h2_robustness/stats_gain_bootstrap.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FP_DIR = REPO / "results/aggregated/fingerprints"
OUT = REPO / "results/summary/h2_robustness/stats_gain_bootstrap.csv"
N_BOOT = 10_000


def r(a, b):
    return np.corrcoef(a, b)[0, 1]


def gains(c1, c2, c3, c5, c6):
    s = r(c3, c1 + c5)
    return s - max(r(c3, c1), r(c3, c5)), s - r(c3, c2 + c6)


def main():
    rows = []
    for key in ("clamped", "raw"):
        fp = {c: np.load(FP_DIR / f"{name}_{key}.npy").ravel().astype(np.float64)
              for c, name in [("c1", "C1_abstract_belief"), ("c2", "C2_abstract_photo"),
                              ("c3", "C3_answer_changes_belief"),
                              ("c5", "C5_answer_changes_location_only"),
                              ("c6", "C6_answer_changes_location_only_photo")]}
        obs = gains(**fp)
        rng = np.random.default_rng(0)
        n = len(fp["c1"])
        boot = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            boot.append(gains(**{k: v[i] for k, v in fp.items()}))
        boot = np.array(boot)
        for j, name in enumerate(("gain_vs_best_part", "gain_vs_C2+C6")):
            rows.append({"key": key, "quantity": name, "observed": obs[j],
                         "ci_lo": np.percentile(boot[:, j], 2.5),
                         "ci_hi": np.percentile(boot[:, j], 97.5),
                         "frac_boot_le_0": np.mean(boot[:, j] <= 0),
                         "n_heads": n, "n_boot": N_BOOT})
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(df.round(4).to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
