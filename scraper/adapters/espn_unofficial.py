"""
scraper/adapters/espn_unofficial.py

ESPN Unofficial API adapter — scrapes ESPN's internal endpoints for:
  - Practice participation / injury status (participation reports)
  - Depth chart changes
  - Game-day inactives

NOTE: ESPN doesn't have a public API. This adapter uses their unofficial
internal endpoints (same ones powering ESPN.com). Endpoints may break
without notice when ESPN updates their frontend.

Uses the fully-implemented EspnAdapter in espn_adapter.py for the
practice report / injury feed. This module re-exports that class
under the EspnUnofficialAdapter alias for backwards compatibility
with any code that imports from this module.

Primary adapter reference: scraper/adapters/espn_adapter.py
"""

from __future__ import annotations

from scraper.adapters.espn_adapter import EspnAdapter as EspnUnofficialAdapter  # noqa: F401

__all__ = ["EspnUnofficialAdapter"]
