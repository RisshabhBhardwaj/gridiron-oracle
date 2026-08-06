"""
Runtime mode helpers shared by backend and ML-facing services.
"""

from __future__ import annotations

from backend.app.core.config import settings


class ArtifactRequiredError(RuntimeError):
    """Raised when artifact-backed mode forbids a fallback code path."""


def artifact_backed_mode() -> bool:
    """Return True when the API is configured for strict artifact-backed serving."""
    return settings.product_mode == "artifact_backed"


def fallbacks_allowed() -> bool:
    """Return True when synthetic or proxy outputs are permitted."""
    return not artifact_backed_mode()
