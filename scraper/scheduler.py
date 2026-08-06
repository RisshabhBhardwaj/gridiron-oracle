"""
scraper/scheduler.py

APScheduler-based background scheduler for the Gridiron Oracle scraper layer.

Schedules periodic data ingestion jobs:
  - nflreadpy_adapter:   weekly (Tuesdays at 06:00 UTC, after game results available)
  - espn_adapter:        every 4 hours (practice reports, injury updates)
  - weather_adapter:     every 2 hours (real-time game-time weather)
  - odds_adapter:        every 1 hour (line movements, market updates)

Run standalone:
    python -m scraper.scheduler

Or imported by the orchestrator:
    from scraper.scheduler import ScraperScheduler
    scheduler = ScraperScheduler()
    scheduler.start()
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class ScraperScheduler:
    """
    Manages APScheduler jobs for periodic data scraping.

    Uses AsyncIOScheduler (preferred when running inside FastAPI lifespan)
    or BlockingScheduler (when running standalone via __main__).
    """

    def __init__(
        self,
        db_url: str | None = None,
        odds_api_key: str | None = None,
        openweather_api_key: str | None = None,
    ) -> None:
        self._db_url = db_url or os.environ.get(
            "DATABASE_URL",
            "postgresql://oracle:oracle@localhost:5432/oracle",
        )
        self._odds_key = odds_api_key or os.environ.get("ODDS_API_KEY", "")
        self._weather_key = openweather_api_key or os.environ.get("OPENWEATHER_API_KEY", "")
        self._scheduler = None

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self, async_mode: bool = False) -> None:
        """
        Start the scheduler.

        Args:
            async_mode: True when running inside asyncio event loop (FastAPI).
                        False when running as a standalone blocking process.
        """
        try:
            if async_mode:
                from apscheduler.schedulers.asyncio import AsyncIOScheduler
                self._scheduler = AsyncIOScheduler()
            else:
                from apscheduler.schedulers.blocking import BlockingScheduler
                self._scheduler = BlockingScheduler()
        except ImportError as exc:
            logger.error("apscheduler not installed: %s — scheduler disabled", exc)
            return

        self._register_jobs()
        self._scheduler.start()
        logger.info(
            "ScraperScheduler started with %d jobs at %s",
            len(self._scheduler.get_jobs()),
            datetime.now(timezone.utc).isoformat(),
        )

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("ScraperScheduler stopped")

    # ------------------------------------------------------------------
    # Job Registration
    # ------------------------------------------------------------------

    def _register_jobs(self) -> None:
        """Register all scraping jobs with the scheduler."""
        if self._scheduler is None:
            return

        # nflreadpy: weekly on Tuesdays at 06:00 UTC (game results typically
        # finalized Monday night; Tuesday morning is safe)
        self._scheduler.add_job(
            self._run_nflreadpy,
            trigger="cron",
            day_of_week="tue",
            hour=6,
            minute=0,
            id="nflreadpy_weekly",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        # ESPN injury / practice reports: every 4 hours
        self._scheduler.add_job(
            self._run_espn,
            trigger="interval",
            hours=4,
            id="espn_injury_reports",
            replace_existing=True,
            misfire_grace_time=600,
        )

        # Weather: every 2 hours (real-time conditions for upcoming games)
        if self._weather_key:
            self._scheduler.add_job(
                self._run_weather,
                trigger="interval",
                hours=2,
                id="weather_update",
                replace_existing=True,
                misfire_grace_time=600,
            )
        else:
            logger.warning("OPENWEATHER_API_KEY not set — weather job disabled")

        # Odds / market data: every hour
        if self._odds_key:
            self._scheduler.add_job(
                self._run_odds,
                trigger="interval",
                hours=1,
                id="odds_update",
                replace_existing=True,
                misfire_grace_time=300,
            )
        else:
            logger.warning("ODDS_API_KEY not set — odds job disabled")

    # ------------------------------------------------------------------
    # Job implementations
    # ------------------------------------------------------------------

    def _run_nflreadpy(self) -> None:
        """Weekly: ingest latest nflreadpy stats + schedule data."""
        logger.info("[scheduler] Starting nflreadpy weekly ingest")
        try:
            from datetime import datetime, timezone
            from pipeline.orchestrator import Orchestrator
            Orchestrator(self._db_url).run(seasons=[datetime.now(timezone.utc).year])
            logger.info("[scheduler] nflreadpy weekly ingest complete")
        except Exception as exc:
            logger.error("[scheduler] nflreadpy ingest failed: %s", exc, exc_info=True)

    def _run_espn(self) -> None:
        """Every 4h: scrape ESPN for injury/practice participation updates."""
        logger.info("[scheduler] Starting ESPN injury/practice report scrape")
        try:
            from scraper.adapters.espn_adapter import EspnAdapter
            adapter = EspnAdapter()
            # ESPN adapter publishes directly to DB via its internal session
            adapter.ingest_injury_reports(db_url=self._db_url)
            logger.info("[scheduler] ESPN scrape complete")
        except Exception as exc:
            logger.error("[scheduler] ESPN scrape failed: %s", exc, exc_info=True)

    def _run_weather(self) -> None:
        """Every 2h: update weather conditions for upcoming games."""
        logger.info("[scheduler] Starting weather update")
        try:
            from scraper.adapters.weather_adapter import enrich_games_precipitation
            enrich_games_precipitation(db_url=self._db_url, api_key=self._weather_key)
            logger.info("[scheduler] Weather update complete")
        except Exception as exc:
            logger.error("[scheduler] Weather update failed: %s", exc, exc_info=True)

    def _run_odds(self) -> None:
        """Every 1h: pull latest market odds from The Odds API."""
        logger.info("[scheduler] Starting odds update")
        try:
            from scraper.adapters.odds_adapter import fetch_and_store_props
            fetch_and_store_props(db_url=self._db_url, api_key=self._odds_key)
            logger.info("[scheduler] Odds update complete")
        except Exception as exc:
            logger.error("[scheduler] Odds update failed: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    logger.info("Starting ScraperScheduler in blocking mode")
    s = ScraperScheduler()
    try:
        s.start(async_mode=False)
    except (KeyboardInterrupt, SystemExit):
        s.stop()
        sys.exit(0)
