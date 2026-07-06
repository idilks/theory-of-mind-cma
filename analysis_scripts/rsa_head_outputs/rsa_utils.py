"""Shared utilities for the RSA head-output analyses.

Holds the single canonical cosine-distance RDM used across every analysis in
this directory. Previously this function was copy-pasted (byte-for-byte
equivalent, only variable names / intermediate temporaries differed) into 9
scripts under the names cosine_rdm / cosine_dist / cosine_dist_matrix. See
Refactor Debt item 3 in CLAUDE.md.
"""
from __future__ import annotations

import numpy as np


def cosine_rdm(z: np.ndarray) -> np.ndarray:
    """z: (n_stim, head_dim) -> (n_stim, n_stim) RDM = 1 - cosine similarity.

    Row norms of zero are guarded to 1 (leaves those rows at zero vectors),
    and the similarity is clipped to [-1, 1] before subtracting from 1.
    """
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1
    zn = z / norms
    return 1 - np.clip(zn @ zn.T, -1, 1)
