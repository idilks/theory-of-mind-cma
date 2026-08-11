"""Chat-template compatibility: models whose template rejects a system role.

Gemma-2's chat template raises `TemplateError: System role not supported`, so
the `[{system}, {user}]` message list used throughout this pipeline fails.

The CMA that produced the Gemma head lists already handles this, in
`codebase/tasks/causal_analysis/prompt_generators/unified_tom.py`:

    if self._supports_system_role:
        return [{"role": "system", ...}, {"role": "user", ...}]
    combined_user_prompt = f"{system_prompt}\\n\\n{user_prompt}"
    return [{"role": "user", "content": combined_user_prompt}]

This module replicates that byte-for-byte, INCLUDING the double newline
separator. That fidelity is not cosmetic: the Gemma head lists were derived
from activations captured under this exact prompt format, so a capture using
any other format would be selecting heads by one prompt and measuring them
under another.

For Qwen (whose template does support a system role) the probe succeeds and the
original two-message form is returned unchanged, so every existing Qwen result
reproduces byte-identically.
"""

from __future__ import annotations

_CACHE: dict[int, bool] = {}


def supports_system_role(tokenizer) -> bool:
    """Probe the tokenizer's chat template, exactly as the CMA does."""
    key = id(tokenizer)
    if key in _CACHE:
        return _CACHE[key]
    try:
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "x"},
             {"role": "user", "content": "y"}],
            tokenize=False, add_generation_prompt=True)
        ok = True
    except Exception as exc:                     # noqa: BLE001
        if "System role not supported" in str(exc):
            ok = False
        else:
            raise
    _CACHE[key] = ok
    return ok


def build_messages(tokenizer, system_prompt: str, user_prompt: str) -> list:
    if supports_system_role(tokenizer):
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}]
    return [{"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}]


def chat_prompt(tokenizer, system_prompt: str, user_prompt: str) -> str:
    """Rendered prompt string, add_generation_prompt=True."""
    return tokenizer.apply_chat_template(
        build_messages(tokenizer, system_prompt, user_prompt),
        tokenize=False, add_generation_prompt=True)
