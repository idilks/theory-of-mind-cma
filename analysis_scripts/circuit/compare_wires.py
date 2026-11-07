"""Compare wire directions across domains (ToM vs code).

Loads two Stage-1 directions files and reports, per site:
  - cos(v_rank1_a, v_rank1_b): alignment of the rank-1 wires
  - subspace overlap: mean squared singular value of U_a U_b^T (average
    squared cosine of principal angles) — 1.0 = identical subspaces,
    ~k_b/d_model = random-subspace baseline (printed for reference)
  - projection of each rank-1 wire onto the OTHER domain's top-k subspace

Discriminates the cross-domain steering-null explanations:
  domain-indexed wire content -> code wire exists but cos ~ 0
  scale/manifold mismatch     -> high alignment (failure was operational)

Usage:
  python analysis_scripts/circuit/compare_wires.py \
      --a results/aggregated/circuit/stage1/directions_L41.npz \
      --b results/aggregated/circuit/stage1_code/directions_L41.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path):
    z = np.load(path)
    return dict(v=z["v_rank1"], U=z["U_topk"], k=int(z["k"]),
                scale=float(z["natural_scale"]),
                domain=str(z["domain"]) if "domain" in z else "?",
                label=path.parent.name + "/" + path.stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=Path, required=True)
    ap.add_argument("--b", type=Path, required=True)
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    d = A["v"].shape[0]
    print(f"A: {A['label']} (domain={A['domain']}, k={A['k']}, "
          f"natural_scale={A['scale']:.2f})")
    print(f"B: {B['label']} (domain={B['domain']}, k={B['k']}, "
          f"natural_scale={B['scale']:.2f})")

    cos = float(A["v"] @ B["v"])
    print(f"\ncos(v_rank1_A, v_rank1_B) = {cos:.4f}")

    sv = np.linalg.svd(A["U"] @ B["U"].T, compute_uv=False)
    overlap = float(np.mean(sv ** 2))
    print(f"subspace overlap (mean cos^2 of principal angles, "
          f"k_A={A['k']} x k_B={B['k']}) = {overlap:.4f}")
    print(f"  random-subspace baseline ~ k/d = {max(A['k'], B['k']) / d:.4f}")
    print(f"  principal-angle cosines: {np.round(sv, 3).tolist()}")

    pa_in_b = float(np.sum((B["U"] @ A["v"]) ** 2))
    pb_in_a = float(np.sum((A["U"] @ B["v"]) ** 2))
    print(f"||P_(U_B)(v_A)||^2 = {pa_in_b:.4f}   "
          f"||P_(U_A)(v_B)||^2 = {pb_in_a:.4f}   (random ~ {B['k'] / d:.4f})")

    # 1000-random-unit-vector null for the rank-1 cosine
    rng = np.random.default_rng(0)
    r = rng.standard_normal((1000, d))
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    null = np.abs(r @ B["v"])
    print(f"|cos| null (1000 random v): median={np.median(null):.4f}, "
          f"p95={np.quantile(null, 0.95):.4f} -> observed |cos|="
          f"{abs(cos):.4f} is {'ABOVE' if abs(cos) > np.quantile(null, 0.95) else 'within'} null")


if __name__ == "__main__":
    main()
