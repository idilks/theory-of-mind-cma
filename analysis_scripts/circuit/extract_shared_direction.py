"""Extract the shared cross-domain wire component (ToM ∩ code).

Takes the two Stage-1 k=8 subspaces, computes the principal-angle
decomposition of U_tom @ U_code^T, and saves the top principal pair:
  u1       — the ToM-side principal direction (in ToM's subspace)
  w1       — the code-side principal direction (in code's subspace)
  v_shared — their normalized bisector (the "shared component" candidate)

Sign convention: the principal pair automatically satisfies u1·w1 =
sigma_1 > 0; the global sign is chosen so u1 points toward the ToM
false-belief side (u1 · v_rank1_tom >= 0). w1's polarity w.r.t. the code
divergent side is then REPORTED, not enforced — an anti-aligned w1 would
mean the shared axis has opposite meanings in the two domains.

Also pre-tests the reader survival of v_shared (S1c machinery): if the
shared component is the functionally potent sliver, it should project
into the reader heads' Q subspaces at least as strongly as the full
rank-1 wires do.

Output: results/aggregated/circuit/shared_direction_{site}.npz
Steering test: analysis_scripts/circuit/stage2_shared_direction.py (GPU).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from circuit_common import AGG_DIR, CAPTURE_ROOT, STAGE1_OUT  # noqa: E402
from stage1_geometric_screen import reader_read_norms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="L41")
    ap.add_argument("--model_type", default="Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    tom = np.load(STAGE1_OUT / f"directions_{args.site}.npz")
    code = np.load(STAGE1_OUT.parent / "stage1_code" / f"directions_{args.site}.npz")
    Ua, Ub = tom["U_topk"], code["U_topk"]          # (k, d) orthonormal rows

    P, S, Qt = np.linalg.svd(Ua @ Ub.T)
    u = P[:, 0] @ Ua                                 # ToM-side principal (d,)
    w = Qt[0, :] @ Ub                                # code-side principal (d,)
    if u @ tom["v_rank1"] < 0:                       # point toward ToM FB side
        u, w = -u, -w
    v_shared = (u + w) / np.linalg.norm(u + w)

    print(f"principal cosines: {np.round(S, 3).tolist()}")
    print(f"cos(u1, w1) = {u @ w:.3f} (should equal sigma_1 = {S[0]:.3f})")
    print(f"u1 · v_rank1_tom  = {u @ tom['v_rank1']:+.3f} (sign-fixed >= 0)")
    print(f"w1 · v_rank1_code = {w @ code['v_rank1']:+.3f} "
          "(reported: negative => opposite polarity across domains)")
    print(f"v_shared · v_rank1_tom  = {v_shared @ tom['v_rank1']:+.3f}")
    print(f"v_shared · v_rank1_code = {v_shared @ code['v_rank1']:+.3f}")

    # mediation candidates: the part of each domain's rank-1 wire that lives
    # inside the OTHER domain's k=8 subspace (the component that could have
    # carried the mirror effect)
    proj_code_in_tom = Ua.T @ (Ua @ code["v_rank1"])
    proj_code_in_tom /= np.linalg.norm(proj_code_in_tom)
    proj_tom_in_code = Ub.T @ (Ub @ tom["v_rank1"])
    proj_tom_in_code /= np.linalg.norm(proj_tom_in_code)
    print(f"proj_code_in_tom · v_rank1_tom = {proj_code_in_tom @ tom['v_rank1']:+.3f}")
    print(f"proj_tom_in_code · v_rank1_code = {proj_tom_in_code @ code['v_rank1']:+.3f}")

    # reader-survival pre-test of the shared component vs the full wires
    weights = torch.load(CAPTURE_ROOT / args.model_type / "weights.pt",
                         map_location="cpu", weights_only=False)
    dirs = np.vstack([v_shared, u, w, tom["v_rank1"], code["v_rank1"],
                      proj_code_in_tom, proj_tom_in_code])
    names = ["v_shared", "u1_tom", "w1_code", "v_rank1_tom", "v_rank1_code",
             "proj_code_in_tom", "proj_tom_in_code"]
    reads = reader_read_norms(dirs, weights)
    rng = np.random.default_rng(0)
    rand = rng.standard_normal((1000, dirs.shape[1]))
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    rand_reads = reader_read_norms(rand, weights)
    for fam in ("Q", "V"):
        med = np.median(rand_reads[fam], axis=0)
        ratios = np.median(reads[fam] / med, axis=1)
        print(f"reader read-ratio medians [{fam}]: "
              + ", ".join(f"{n}={r:.2f}" for n, r in zip(names, ratios)))

    out = AGG_DIR / f"shared_direction_{args.site}.npz"
    np.savez(out, v_shared=v_shared, u1_tom=u, w1_code=w,
             proj_code_in_tom=proj_code_in_tom,
             proj_tom_in_code=proj_tom_in_code,
             principal_cosines=S,
             tom_scale=float(tom["natural_scale"]),
             code_scale=float(code["natural_scale"]),
             site=args.site)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
