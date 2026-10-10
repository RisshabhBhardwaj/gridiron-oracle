"""
ml/product_mode.py

Single resolver for the PRODUCT_MODE switch, shared by backend config, ml/ and
scripts/. Shell callers keep a literal default that matches DEFAULT_PRODUCT_MODE.

artifact_backed is the default. It serves and trains only manifest-pinned
artifacts and fails closed. graceful_fallback is an explicit opt-in for local
development, and any output it produces must be labelled as a fallback.

An unknown value raises ValueError. It never falls back silently.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

ENV_VAR = "PRODUCT_MODE"
ARTIFACT_BACKED = "artifact_backed"
GRACEFUL_FALLBACK = "graceful_fallback"
DEFAULT_PRODUCT_MODE = ARTIFACT_BACKED
VALID_PRODUCT_MODES = frozenset({ARTIFACT_BACKED, GRACEFUL_FALLBACK})


def resolve_product_mode(environ: Mapping[str, str] | None = None) -> str:
    """Return the active product mode.

    Reads the environment on every call, so tests can monkeypatch it. Unset or
    blank means DEFAULT_PRODUCT_MODE. Values are stripped and lowercased. Any
    other value raises ValueError.
    """
    env = os.environ if environ is None else environ
    raw = env.get(ENV_VAR, "")
    value = raw.strip().lower()
    if not value:
        return DEFAULT_PRODUCT_MODE
    if value not in VALID_PRODUCT_MODES:
        valid = ", ".join(sorted(VALID_PRODUCT_MODES))
        raise ValueError(f"PRODUCT_MODE={raw!r} is not valid; use one of: {valid}")
    return value
