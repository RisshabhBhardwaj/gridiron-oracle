"""Disclose whether a served cell is a Ridge stack or an LGBM identity passthrough."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT = _ROOT / "releases" / "candidates" / "causal_20260810" / "CONSTRAINED_STACK_SELECTION.json"


@lru_cache(maxsize=4)
def _decisions(path: str) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("decisions") or {}
    return {str(key): dict(value) for key, value in raw.items()}


def served_learner(stat: str, position: str, *, path: Path | None = None) -> str:
    """Return ``ridge_stack``, ``lgbm_identity``, or ``unknown`` for a cell."""
    manifest = path or _DEFAULT
    if not manifest.is_file():
        return "unknown"
    key = f"{stat}:{str(position).upper()}"
    entry = _decisions(str(manifest)).get(key) or {}
    selection = entry.get("selection")
    return str(selection) if selection else "unknown"
