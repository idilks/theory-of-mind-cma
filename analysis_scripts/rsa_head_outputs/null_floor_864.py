"""What is the Mantel shuffle-null floor on the 864-stimulus Exp 3 grid?

Earlier notes quote ~0.04-0.06, a number imported from the 144-stimulus grids.
Permutation nulls narrow with n, so that floor is almost certainly too high
here. Measure it directly on the projected code_abstract RDM x the
queried_slot_abs model (the cell the bolding question is about).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis_scripts" / "rsa_head_outputs"))
from rsa_utils import cosine_rdm  # noqa: E402
from analyze_code_ledger_rdms import build_model_rdms, upper_tri  # noqa: E402
from analyze_code_padtwins_rdms import load  # noqa: E402
from padtwins_projection_test import (  # noqa: E402
    displacement_pairs, pad_subspace, project_out, twin_groups)

d, sets = load("qwen")
z = d["z"].float().numpy()
stim = d["stim"]
pairs = displacement_pairs(stim, twin_groups(stim))
hdf = sets["code_abstract"]
proj = []
for r in hdf.itertuples():
    zh = z[:, int(r.layer), int(r.head), :]
    U, _, _ = pad_subspace(zh, pairs, 2)
    proj.append(cosine_rdm(project_out(zh, U)))
R = np.mean(proj, axis=0)

models = build_model_rdms(stim, "Qwen/Qwen2.5-14B-Instruct")
flat = upper_tri(R)
rng = np.random.default_rng(0)
n = R.shape[0]
for name in ("queried_slot_abs", "object_divergence"):
    M = models[name]
    obs = float(spearmanr(flat, upper_tri(M)).correlation)
    null = np.empty(1000)
    for i in range(1000):
        p = rng.permutation(n)
        null[i] = float(spearmanr(flat, upper_tri(M[p][:, p])).correlation)
    print(f"{name}: observed {obs:+.4f} | null max|r| {np.abs(null).max():.4f} "
          f"p95|r| {np.quantile(np.abs(null), 0.95):.4f} sd {null.std():.4f}")
