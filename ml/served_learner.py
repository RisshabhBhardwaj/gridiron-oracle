"""Disclose whether a served cell is a Ridge stack or an LGBM identity passthrough."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from ml.artifact_manifest import ManifestError, get_manifest, verify_file_digest


def default_selection_path() -> Optional[Path]:
    """The selection file the release manifest pins, digest-checked.

    Read from ``artifacts.selection_policy`` in the manifest (honouring
    ``BASELINE_MANIFEST_PATH``), so a re-freeze moves disclosure with the pin.
    Returns None when the manifest or its pin is missing or altered; disclosure
    then reports ``unknown`` rather than naming a learner from another release.
    """
    try:
        manifest = get_manifest()
        artifacts = manifest.raw.get("artifacts") or {}
        relpath = str((artifacts.get("selection_policy") or [None])[0] or "")
        if not relpath or relpath not in manifest.digests:
            return None
        path = manifest.root / relpath
        verify_file_digest(path, manifest.digests[relpath])
    except (ManifestError, OSError, AttributeError, TypeError):
        return None
    return path


@lru_cache(maxsize=4)
def _decisions(path: str) -> dict[str, dict]:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("decisions") or {}
    return {str(key): dict(value) for key, value in raw.items()}


def served_learner(stat: str, position: str, *, path: Path | None = None) -> str:
    """Return ``ridge_stack``, ``lgbm_identity``, or ``unknown`` for a cell."""
    manifest = path or default_selection_path()
    if manifest is None or not manifest.is_file():
        return "unknown"
    key = f"{stat}:{str(position).upper()}"
    entry = _decisions(str(manifest)).get(key) or {}
    selection = entry.get("selection")
    return str(selection) if selection else "unknown"
