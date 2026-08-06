"""
scraper/adapters/pro_football_ref.py

Pro-Football-Reference (PFR) scraper adapter.

PFR provides historical game logs, advanced rushing/passing/receiving stats,
draft data, and career stats. It requires HTML scraping (no public API).

STATUS: Not yet implemented.
  - nflreadpy covers the primary data needs (2019–present).
  - PFR is planned as a supplementary historical data source (pre-2019).
  - Gated behind Phase 3 roadmap item in CLAUDE.md §11.

When implemented, this adapter should:
  - Use requests + BeautifulSoup (or playwright) to scrape pfr.
  - Respect robots.txt rate limits (1 req/s as per PFR ToS).
  - Target tables: game_log, advanced_stats.
  - Output: List[dict] compatible with staging.StagingNflReadPy structure.
"""

from __future__ import annotations


class ProFootballRefAdapter:
    """
    Scrapes Pro-Football-Reference for supplementary historical data.

    Not yet implemented — raises NotImplementedError on all method calls.
    See module docstring for implementation plan.
    """

    def __init__(self, rate_limit_secs: float = 1.0) -> None:
        self._rate_limit = rate_limit_secs

    def fetch_player_game_log(
        self,
        player_pfr_id: str,
        season: int,
    ) -> list[dict]:
        """Fetch one player's game log for a season from PFR."""
        raise NotImplementedError(
            "ProFootballRefAdapter is not yet implemented. "
            "See scraper/adapters/pro_football_ref.py for the roadmap."
        )

    def fetch_team_stats(self, season: int) -> list[dict]:
        """Fetch team-level advanced stats for a season from PFR."""
        raise NotImplementedError(
            "ProFootballRefAdapter is not yet implemented. "
            "See scraper/adapters/pro_football_ref.py for the roadmap."
        )


__all__ = ["ProFootballRefAdapter"]
