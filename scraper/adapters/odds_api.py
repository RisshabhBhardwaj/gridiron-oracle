"""
scraper/adapters/odds_api.py

The Odds API adapter re-export.

The full implementation lives in scraper/adapters/odds_adapter.py.
This module re-exports OddsAdapter under the OddsApiAdapter alias
so that imports from either module name work correctly.

Usage:
    from scraper.adapters.odds_api import OddsApiAdapter
    adapter = OddsApiAdapter(api_key=os.environ["ODDS_API_KEY"])
    markets = adapter.fetch_markets(sport="americanfootball_nfl")
"""

from __future__ import annotations

from scraper.adapters.odds_adapter import OddsAdapter as OddsApiAdapter  # noqa: F401

__all__ = ["OddsApiAdapter"]
