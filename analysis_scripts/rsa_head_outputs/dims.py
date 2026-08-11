"""Model dimensions, parameterised — so the pipeline runs on more than Qwen.

Every script in this pipeline hardcoded `N_LAYERS, N_HEADS, HEAD_DIM = 48, 40,
128` and the flat head index `layer * 40 + head`. That is correct for
Qwen2.5-14B-Instruct and wrong for anything else. Gemma-2-27b-it is 46 x 32,
so NH = 1472 rather than 1920 and every flat index shifts.

Design rule: **the Qwen values remain the defaults everywhere**, so existing
invocations reproduce byte-identically. Nothing already in the results doc
changes as a result of this refactor.

Preferred source of truth is the loaded model (`from_model`); the preset table
exists so analysis-only scripts, which never load a model, can still resolve
dimensions from a `--model_type` string.

GQA note: `n_heads` here is the number of QUERY heads, which is what hook_z is
indexed by and what the flat head index means. Gemma-2-27b has 32 query heads
and 16 KV heads; the Gemma CMA was run with --ungroup_grouped_query_attention,
which expands K/V to 32 for indexing purposes. Head lists derived from that run
are therefore in query-head space and align with hook_z. Do not assume the same
for a run made without that flag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dims:
    n_layers: int
    n_heads: int
    head_dim: int

    @property
    def nh(self) -> int:
        """Total heads = the length of a flat per-head vector."""
        return self.n_layers * self.n_heads

    def flat(self, layer: int, head: int) -> int:
        return int(layer) * self.n_heads + int(head)

    def unflat(self, idx: int) -> tuple[int, int]:
        return divmod(int(idx), self.n_heads)

    def __str__(self) -> str:
        return (f"{self.n_layers}L x {self.n_heads}H x {self.head_dim}d "
                f"(NH={self.nh})")


PRESETS = {
    "Qwen2.5-14B-Instruct": Dims(48, 40, 128),
    "gemma-2-27b-it": Dims(46, 32, 128),
    "gemma-2-27b-it_ungroup_gqa": Dims(46, 32, 128),
}

QWEN = PRESETS["Qwen2.5-14B-Instruct"]


def for_model_type(model_type: str) -> Dims:
    if model_type in PRESETS:
        return PRESETS[model_type]
    for k, v in PRESETS.items():
        if model_type.endswith(k) or k in model_type:
            return v
    raise KeyError(
        f"no dimension preset for {model_type!r}; add one to dims.PRESETS or "
        "resolve from a loaded model with dims.from_model()")


def from_model(model) -> Dims:
    """Authoritative: read the dimensions off the loaded model."""
    cfg = model.cfg
    return Dims(int(cfg.n_layers), int(cfg.n_heads), int(cfg.d_head))


def check(model, model_type: str | None = None) -> Dims:
    """Resolve from the model and, if a preset exists, assert they agree.

    Catches the silent-drift case where a preset is stale relative to the
    checkpoint actually loaded.
    """
    d = from_model(model)
    if model_type:
        try:
            p = for_model_type(model_type)
        except KeyError:
            return d
        assert d == p, (
            f"dimension mismatch for {model_type!r}: model reports {d}, "
            f"preset says {p}. Fix dims.PRESETS before trusting head indices.")
    return d
