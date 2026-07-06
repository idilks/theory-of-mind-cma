"""RDM analysis for head-output RSA.

Loads per-template head activations from results/rsa_head_outputs/activations/{T}.pt,
sorts stimuli by (belief_veracity, answer_state) so the RDMs read clearly,
then for each head builds a (n_stim, n_stim) RDM = 1 - cosine_sim between
head-output vectors. Compares those data RDMs to two model RDMs:

  - belief_veracity model: 0 if same belief value, 1 otherwise
  - answer_state model:    0 if same answer string, 1 otherwise

Per-condition CMA-weighted average RDMs are computed for top-k C1 and top-k C5
(weights = signed CMA score at that head, clipped at 0). Finally, Mantel
correlations (Spearman, 5000 permutations) test whether each weighted-avg
data RDM matches its predicted model RDM.

Outputs:
  results_for_pi/aggregation/rsa_head_output_rdms.png
  results/rsa_head_outputs/mantel_results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
ACT_DIR = ROOT / "results/rsa_head_outputs/activations"
TOPK_DIR = ROOT / "results/rsa_head_outputs/topk_heads"
FIG_OUT = ROOT / "results_for_pi/aggregation/rsa_head_output_rdms.png"
CSV_OUT = ROOT / "results/rsa_head_outputs/mantel_results.csv"

TEMPLATES = ["food_truck", "hair_styling", "library_book"]


from rsa_utils import cosine_rdm


def model_rdm_belief(stim: pd.DataFrame) -> np.ndarray:
    v = stim["belief_veracity"].to_numpy()
    return (v[:, None] != v[None, :]).astype(float)


def model_rdm_answer(stim: pd.DataFrame) -> np.ndarray:
    a = stim["answer_state"].to_numpy()
    return (a[:, None] != a[None, :]).astype(float)


def upper_tri(M: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr
    r, _ = spearmanr(x, y)
    return float(r)


def mantel(data_rdm: np.ndarray, model_rdm: np.ndarray, n_perm: int = 5000,
           rng: np.random.Generator | None = None) -> tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    obs = spearman(upper_tri(data_rdm), upper_tri(model_rdm))
    n = data_rdm.shape[0]
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(n)
        permuted = model_rdm[perm][:, perm]
        null[i] = spearman(upper_tri(data_rdm), upper_tri(permuted))
    p = float((np.abs(null) >= np.abs(obs)).mean())
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_perm", type=int, default=5000)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    # Stack activations across templates: stims indexed first by template,
    # then by within-template order. Heads must match across templates.
    payloads = {t: torch.load(ACT_DIR / f"{t}.pt", weights_only=False) for t in TEMPLATES}

    head_lists = [[(h["layer"], h["head"]) for h in payloads[t]["heads"]] for t in TEMPLATES]
    assert all(hl == head_lists[0] for hl in head_lists), "head order mismatch across templates"
    heads = head_lists[0]

    z_all = np.concatenate([payloads[t]["z"].numpy().astype(np.float32) for t in TEMPLATES], axis=0)
    stim = pd.concat([pd.DataFrame(payloads[t]["stim"]) for t in TEMPLATES], ignore_index=True)
    n_stim = stim.shape[0]
    n_heads = len(heads)
    print(f"loaded {n_stim} stimuli x {n_heads} heads")

    # Sort stimuli by (belief, answer) so RDM blocks read clearly.
    sort_idx = stim.sort_values(["belief_veracity", "answer_state"]).index.to_numpy()
    z_all = z_all[sort_idx]
    stim = stim.loc[sort_idx].reset_index(drop=True)

    # Per-head RDMs.
    head_rdms = np.stack([cosine_rdm(z_all[:, h, :]) for h in range(n_heads)], axis=0)

    # Top-k head sets + weights.
    c1 = pd.read_csv(TOPK_DIR / f"C1_abstract_belief_top{args.k}.csv")
    c5 = pd.read_csv(TOPK_DIR / f"C5_answer_changes_location_only_top{args.k}.csv")
    head_to_idx = {(L, H): i for i, (L, H) in enumerate(heads)}
    c1_idx = np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c1.itertuples()])
    c5_idx = np.array([head_to_idx[(int(r.layer), int(r.head))] for r in c5.itertuples()])
    w1 = np.clip(c1["score"].to_numpy(), 0, None); w1 /= w1.sum()
    w5 = np.clip(c5["score"].to_numpy(), 0, None); w5 /= w5.sum()

    rdm_c1 = np.tensordot(w1, head_rdms[c1_idx], axes=1)
    rdm_c5 = np.tensordot(w5, head_rdms[c5_idx], axes=1)

    rdm_belief = model_rdm_belief(stim)
    rdm_answer = model_rdm_answer(stim)

    rng = np.random.default_rng(0)
    rows = []
    for label, data in [("topk_C1_abstract_belief", rdm_c1), ("topk_C5_answer_changes_location_only", rdm_c5)]:
        for mlabel, mdl in [("belief_veracity", rdm_belief), ("answer_state", rdm_answer)]:
            r, p = mantel(data, mdl, n_perm=args.n_perm, rng=rng)
            rows.append({"head_set": label, "model": mlabel, "spearman_r": r, "p_perm": p})
            print(f"  {label} vs {mlabel}: r={r:.3f}, p={p:.4f}")
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    print(f"mantel results -> {CSV_OUT}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, M, title in [
        (axes[0, 0], rdm_c1, f"top-{args.k} C1 (belief) weighted RDM"),
        (axes[0, 1], rdm_c5, f"top-{args.k} C5 (ToM control) weighted RDM"),
        (axes[1, 0], rdm_belief, "model RDM: belief veracity"),
        (axes[1, 1], rdm_answer, "model RDM: answer state"),
    ]:
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=M.max())
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_OUT, dpi=150)
    print(f"figure -> {FIG_OUT}")


if __name__ == "__main__":
    main()
