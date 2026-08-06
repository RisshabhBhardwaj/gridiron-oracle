"""
Pro-Football-Reference adapter — RETIRED.

This module previously raised NotImplementedError on every call while still
looking like an available data source. Gridiron Oracle uses nflverse/nflreadpy
as the free stats source. PFR is not on the critical path.

If historical pre-2019 PFR data is needed later, implement a dedicated adapter
in a new module rather than resurrecting this stub.
"""

from __future__ import annotations


class ProFootballRefAdapter:
    """Retired stub. Do not call."""

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError(
            "ProFootballRefAdapter is retired. Use nflreadpy adapters instead "
            "(scraper.adapters.nflreadpy_adapter)."
        )


__all__ = ["ProFootballRefAdapter"]
