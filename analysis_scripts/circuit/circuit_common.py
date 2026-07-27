"""Shared constants and helpers for the layered-lookback circuit analysis.

Design: docs/design_layered_lookback_circuit.md. Stage 1 = geometric screen
(writers -> resid wire -> readers), Stage 2 = causal steering (gated on
Stage 1's go/no-go).

Conventions pinned here so capture (HPC) and screen (laptop) agree:
  - band layers 28..40 inclusive: hook_z + hook_mlp_out capture range and
    the pool for random-head-set nulls (matches the design's L28-40).
  - bus sites: resid_post at L40 and L41, position -1. resid_post L41 ==
    resid_pre L42 in TransformerLens, so the Stage-2 injection site
    (resid_pre L42) is numerically identical to the L41 measurement site.
  - model loaded with fold_ln=False (repo default), so ln1.w is a real
    per-layer scale and reader projections must apply it explicitly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

# --- head-set sources (local trees; the builder bakes them into AGG_DIR,
#     which sync_to_hpc.sh pushes, so HPC jobs never read other_results/) ---
T_C1_PREFILL_CSV = REPO / "other_results/ablation/belief_only_sysprompt_prefill/head_list.csv"
T_C5_CSV = REPO / "other_results/ablation/loc_belief_heads/head_list.csv"
DLA_DIR = REPO / "results/rsa_head_outputs/dla"

# --- baked artifacts (synced to HPC via results/aggregated) ---
AGG_DIR = REPO / "results/aggregated/circuit"
WRITER_CSV = AGG_DIR / "writer_heads.csv"
READER_CSV = AGG_DIR / "reader_heads.csv"

# --- capture outputs (pulled from HPC via pull_results.sh) ---
CAPTURE_ROOT = REPO / "results/circuit/stage1_capture"

# --- stage-1 screen outputs ---
STAGE1_OUT = AGG_DIR / "stage1"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]

BAND_LAYERS = list(range(28, 41))          # z + mlp capture band, null pool
WRITER_MAX_LAYER = 40                      # writers must sit below the wire
SITES = {"L40": "blocks.40.hook_resid_post",
         "L41": "blocks.41.hook_resid_post"}
SITE_KEYS = list(SITES.keys())

READER_DLA_LAYERS = (42, 46)               # DLA writer band (inclusive)
READER_TC5_MIN_LAYER = 41                  # T_C5 heads at/above this layer

# Stage-2 injection site per measurement site: resid_pre of the next block
# is the SAME tensor as the measured resid_post (TL identity), so
# measurement and intervention compose exactly (design "Fixed choices").
INJECTION_HOOKS = {"L40": "blocks.41.hook_resid_pre",
                   "L41": "blocks.42.hook_resid_pre"}
STAGE2_OUT_ROOT = REPO / "results/circuit/stage2_steering"

# The system prompt CMA selected heads under — the primary eval protocol
# (matched prompt + prefill; TB baseline p_correct 0.986).
SYSTEM_PROMPT_MATCHED = "You are a helpful assistant reasoning about agents' beliefs."

EVAL_VARIANTS = ["false_belief", "true_belief"]

# Domain registry: variants ordered (divergent, non-divergent) so the
# FB/TB roles map onto (code_copy, code_alias) — copy is the false-belief
# analog (state diverges), alias the true-belief analog. Code prompts
# always use CODE_SYSTEM_PROMPT (generator enforces it) and the single
# indexed_list template.
DOMAINS = {
    "tom": dict(variants=EVAL_VARIANTS, templates=TEMPLATES,
                capture_subdir="", default_writer_set="primary"),
    "code": dict(variants=["code_copy", "code_alias"],
                 templates=["indexed_list"], capture_subdir="code_domain",
                 default_writer_set="code_abstract"),
}


def load_head_csv(path: Path, head_set: str | None = None) -> pd.DataFrame:
    """Load a circuit head CSV (columns layer, head, ... [, set])."""
    df = pd.read_csv(path)
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)
    if head_set is not None:
        df = df[df["set"] == head_set].reset_index(drop=True)
    return df


def split_pair_indices(n_pairs: int, template: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-template half split over matched FB/TB pair indices.

    Directions are fit on the fit half; all Stage-1 statistics and all
    Stage-2 steering are reported on the held-out half (design: no
    self-fit steering). Stratification = splitting within each template.
    """
    ti = (TEMPLATES.index(template) if template in TEMPLATES
          else 100 + sum(template.encode()) % 1000)  # stable id for non-ToM templates
    rng = np.random.default_rng([seed, ti])
    perm = rng.permutation(n_pairs)
    half = n_pairs // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def rms_normalize(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RMSNorm without scale: x / rms(x), last-dim. Qwen2 uses RMSNorm."""
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps)
    return x / rms


def orthonormal_basis(vectors: np.ndarray, rel_tol: float = 1e-6) -> np.ndarray:
    """Orthonormal basis (rows) of span{vectors} via SVD, rank-truncated."""
    _, s, vt = np.linalg.svd(vectors.astype(np.float64), full_matrices=False)
    keep = s > s[0] * rel_tol if len(s) else np.array([], dtype=bool)
    return vt[keep]


def projection_fraction(basis: np.ndarray, target: np.ndarray) -> float:
    """||P_span(target)||^2 / ||target||^2 for an orthonormal basis (rows)."""
    t = target.astype(np.float64)
    denom = float(t @ t)
    if denom == 0.0 or basis.size == 0:
        return float("nan")
    coeffs = basis @ t
    return float(coeffs @ coeffs) / denom
