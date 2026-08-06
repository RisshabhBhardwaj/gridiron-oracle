"""Backend tests for draft board ADP API helpers."""

from __future__ import annotations

from backend.app.api.draft import _normalize_name


def test_normalize_name_strips_suffixes() -> None:
    assert _normalize_name("A.J. Brown Jr.") == "aj brown"
    assert _normalize_name("  Justin  Jefferson ") == "justin jefferson"
