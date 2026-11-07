"""Shared helpers for the CMA pipeline: environment config, seeding, model naming, plotting.

Imported by `tasks/causal_analysis/cma.py` after `pyrootutils.setup_root()` has put
the package root on `sys.path`.
"""
from __future__ import annotations

import logging
import os
import random

import matplotlib
matplotlib.use("Agg")  # no display on compute nodes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

logger: logging.Logger = logging.getLogger(__name__)

# --- environment -------------------------------------------------------------
# Nothing here is hardcoded; see the "Setup" section of the top-level README.md.
#   HF_HOME        HuggingFace cache location (default: ~/.cache/huggingface)
#   HF_TOKEN       required for gated repos (Llama, Gemma)
#   WANDB_API_KEY  optional; only used when W&B logging is enabled
HF_HOME = os.environ.get("HF_HOME", "")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN", "")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")

LINE_SEP = "---------------------\n"

# TransformerLens activation names that `--activation_name` accepts.
HEAD_ACTIVATIONS = ["q", "k", "v", "z", "pattern", "attn_scores", "rot_q", "rot_k"]

# Model family -> (HuggingFace org prefix, substring that identifies the family).
# Order matters only in that each substring must be unique across families.
_MODEL_FAMILIES: list[tuple[str, str]] = [
    ("Llama-3", "meta-llama"),
    ("Qwen2.5", "Qwen"),
    ("gemma-2", "google"),
    ("gpt-oss", "openai"),
    ("phi-4", "microsoft"),
    ("gpt2", ""),  # gpt2 checkpoints are published without an org prefix
]


def get_model_id_family(model_type: str) -> tuple[str, str]:
    """Resolve a short model name to its (HuggingFace repo id, family key).

    A name that already contains "/" is treated as a fully-qualified repo id and
    passed through unchanged.

        >>> get_model_id_family("Qwen2.5-14B-Instruct")
        ('Qwen/Qwen2.5-14B-Instruct', 'Qwen2.5')

    Raises ValueError for an unrecognised name, listing the families that are known.
    """
    for family, org in _MODEL_FAMILIES:
        if family not in model_type:
            continue
        if "/" in model_type or not org:
            return model_type, family
        return f"{org}/{model_type}", family

    known = ", ".join(family for family, _ in _MODEL_FAMILIES)
    raise ValueError(
        f"Unrecognised model type {model_type!r}. Known families: {known}. "
        "Add an entry to _MODEL_FAMILIES to support a new one."
    )


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and all visible GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot(
    result_array,
    save_folder: str,
    metric_name: str = "Test Accuracy",
    xlabel_name: str = "Head Index",
    ylabel_name: str = "Layer Index",
    invert_y_axis: bool = False,
    fig_w: float = 30,
    fig_h: float = 22,
    colormap: str = "viridis",
    **_ignored,
) -> str:
    """Render a (layers x heads) score grid as a heatmap and save it as a PNG.

    Writes `{metric_name}_heatmap.png` into `save_folder` — callers rely on that
    exact name to locate the figure afterwards. Returns the path written.

    A 1-D input is treated as a single column, so per-layer scores plot without
    the caller having to reshape. Extra keyword arguments are accepted and ignored
    so that call sites can pass presentation options this implementation does not use.
    """
    os.makedirs(save_folder, exist_ok=True)

    grid = np.asarray(result_array, dtype=float)
    if grid.ndim == 1:
        grid = grid[:, None]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(grid, cmap=colormap, ax=ax, cbar_kws={"label": metric_name})
    ax.set_xlabel(xlabel_name)
    ax.set_ylabel(ylabel_name)
    ax.set_title(metric_name)
    if invert_y_axis:
        ax.invert_yaxis()

    out_path = os.path.join(save_folder, f"{metric_name}_heatmap.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("wrote heatmap: %s", out_path)
    return out_path
