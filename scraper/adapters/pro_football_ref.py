"""
Pro-Football-Reference *scraping* adapter — RETIRED.

This module previously raised NotImplementedError on every call while still
looking like an available data source. It has zero callers and now raises on
construction.

CAUTION — "PFR is gone" is not true of the project as a whole, and this file
used to imply otherwise by describing PFR as off the project's critical path.
PFR-derived data is still ingested, via nflverse rather than by scraping
(audit C-25):

    pipeline/pbp_pipeline.py::_load_pfr_drop_rates
        → nflreadpy.load_pfr_advstats(seasons=..., stat_type="rec")
        → the `drop_rate` column on `player_pbp_features`

What is retired is *direct scraping of pro-football-reference.com*. What is live
is nflverse's redistributed copy of PFR advanced receiving stats, reached
through `nflreadpy` like every other free source here. Do not describe the
project as PFR-free while that call exists — see `docs/DATA_SOURCES.md`.

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
