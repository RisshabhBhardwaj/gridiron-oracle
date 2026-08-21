"""
pipeline/orchestrator.py

Single entry point that wires the three pipeline steps in sequence:
  1. Ingest  — nflreadpy_adapter: fetch & validate, write to staging_nflreadpy
  2. Normalize — normalize.py: staging → production tables (teams/players/games/game_logs)
  3. Feature Engineer — feature_engineer.py: game_logs → feature_matrix

In --dry-run mode, all three steps execute in memory using nflreadpy's
24h filesystem cache. No DB connection is required for dry-run.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DRY-RUN DATA FLOW (no DB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All three steps share one nflreadpy load (cached after first fetch):

  Step 1 Ingest:          Polars DataFrames → Pydantic validation → counts
  Step 2 Normalize:       Validated dicts  → pure transform functions → in-memory
                          player_dicts + game_dicts + log_dicts
  Step 3 Feature Eng:     log_dicts (sample of DRY_RUN_SAMPLE players) →
                          build_feature_row() → FeatureRow count

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ERROR HANDLING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Per-step failures are caught and logged. In live mode a dead_letter row is
written so no failure is silent. If ingest or normalize fail, the remaining
steps for that season are skipped and processing continues with the next
season in the --seasons list.

Standalone usage:
  python pipeline/orchestrator.py --seasons 2025 --dry-run
  python pipeline/orchestrator.py --seasons 2024 2025      # live, requires DB
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError

from scraper.adapters.nflreadpy_adapter import (
    NFLReadPyAdapter,
    PlayerStatsRow,
    RosterRow,
    ScheduleRow,
    _coerce_row,
    _psycopg2_dsn,
)
from pipeline.normalize import (
    Normalizer,
)
from pipeline.feature_engineer import (
    FeatureEngineer,
    build_feature_row,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# In dry-run mode, sample at most this many unique players for the feature
# engineer step to avoid O(n²) matchup computation on a full season.
DRY_RUN_SAMPLE = 500


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Timing and outcome for one pipeline step."""
    name: str
    elapsed_s: float
    ok: bool
    rows_in: int = 0
    rows_out: int = 0
    error: Optional[str] = None


@dataclass
class SeasonResult:
    """Aggregated results for one season's run."""
    season: int
    dry_run: bool
    steps: list[StepResult] = field(default_factory=list)
    n_players: int = 0
    n_games: int = 0
    n_feature_rows: int = 0


@dataclass
class OrchestratorSummary:
    """Top-level summary returned by Orchestrator.run()."""
    seasons: list[int]
    dry_run: bool
    season_results: list[SeasonResult]
    total_players: int = 0
    total_games: int = 0
    total_feature_rows: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None


# ── Dry-run pipeline (no DB required) ────────────────────────────────────────

def _dry_run_season(season: int) -> SeasonResult:
    """
    Full dry-run for one season: ingest + normalize + feature_engineer in memory.
    All three steps share one nflreadpy load (24h filesystem cache means the
    second and third load calls are near-instant disk reads).

    No database connection required.
    """
    result = SeasonResult(season=season, dry_run=True)

    try:
        import nflreadpy as nfl
    except ImportError:
        result.steps.append(StepResult(
            name="ingest", elapsed_s=0.0, ok=False,
            error="nflreadpy not installed. Run: pip install nflreadpy",
        ))
        return result

    # ─── Step 1: Ingest ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("[season=%d] Step 1: ingest (dry-run)…", season)

    try:
        stats_df  = nfl.load_player_stats(seasons=season)
        sched_df  = nfl.load_schedules(seasons=season)
        roster_df = nfl.load_rosters(seasons=season)

        stat_rows:   list[dict] = []
        sched_rows:  list[dict] = []
        roster_rows: list[dict] = []
        ingest_fail = 0

        for raw in stats_df.to_dicts():
            try:
                stat_rows.append(
                    PlayerStatsRow.model_validate(_coerce_row(raw)).model_dump()
                )
            except ValidationError:
                ingest_fail += 1

        for raw in sched_df.to_dicts():
            try:
                sched_rows.append(
                    ScheduleRow.model_validate(_coerce_row(raw)).model_dump()
                )
            except ValidationError:
                ingest_fail += 1

        for raw in roster_df.to_dicts():
            try:
                roster_rows.append(
                    RosterRow.model_validate(_coerce_row(raw)).model_dump()
                )
            except ValidationError:
                ingest_fail += 1

        rows_in  = len(stats_df) + len(sched_df) + len(roster_df)
        rows_out = len(stat_rows) + len(sched_rows) + len(roster_rows)
        result.steps.append(StepResult(
            name="ingest", elapsed_s=time.monotonic() - t0, ok=True,
            rows_in=rows_in, rows_out=rows_out,
        ))
        logger.info(
            "[season=%d] ingest done: %d valid rows, %d failed (%.2fs)",
            season, rows_out, ingest_fail, result.steps[-1].elapsed_s,
        )

    except Exception as exc:
        result.steps.append(StepResult(
            name="ingest", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] ingest FAILED: %s", season, exc)
        return result  # can't continue without data

    # ─── Step 2: Normalize ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("[season=%d] Step 2: normalize (dry-run)…", season)

    try:
        # game_id → ScheduleRow dict (has all venue/context fields used downstream)
        game_map: dict[str, dict] = {g["game_id"]: g for g in sched_rows}

        # (season, week, team) → game_id fallback for 2024 player_stats (game_id=None)
        team_week_to_game: dict[tuple, str] = {}
        for g in sched_rows:
            wk = int(g.get("week") or 0)
            team_week_to_game[(season, wk, g["home_team"])] = g["game_id"]
            team_week_to_game[(season, wk, g["away_team"])] = g["game_id"]

        # Count unique players with a usable primary key
        n_players = len({r["gsis_id"] for r in roster_rows if r.get("gsis_id")})
        n_games   = len(game_map)

        # Build game_log dicts from stat_rows; patch missing game_ids via schedule
        log_dicts: list[dict] = []
        for r in stat_rows:
            game_id = r.get("game_id")
            if not game_id:
                wk = int(r.get("week") or 0)
                game_id = team_week_to_game.get((season, wk, r.get("team") or ""))
            if not game_id:
                continue  # no FK target — skip (counts as skipped in normalize)
            log_dicts.append({
                "player_id":          r["player_id"],
                "game_id":            game_id,
                "season":             r["season"],
                "week":               r["week"],
                "position":           r.get("position"),
                "team":               r.get("team"),
                "opponent_team":      r.get("opponent_team"),
                "receiving_yards":    r.get("receiving_yards"),
                "receiving_tds":      r.get("receiving_tds"),
                "targets":            r.get("targets"),
                "receptions":         r.get("receptions"),
                "target_share":       r.get("target_share"),
                "air_yards_share":    r.get("air_yards_share"),
                "fantasy_points_ppr": r.get("fantasy_points_ppr"),
                "carries":            r.get("carries"),
                "rushing_yards":      r.get("rushing_yards"),
                "rushing_tds":        r.get("rushing_tds"),
                "attempts":           r.get("attempts"),
                "passing_yards":      r.get("passing_yards"),
                "passing_tds":        r.get("passing_tds"),
                "completions":        r.get("completions"),
            })

        result.n_players = n_players
        result.n_games   = n_games
        result.steps.append(StepResult(
            name="normalize",
            elapsed_s=time.monotonic() - t0,
            ok=True,
            rows_in=len(stat_rows) + len(sched_rows) + len(roster_rows),
            rows_out=n_players + n_games + len(log_dicts),
        ))
        logger.info(
            "[season=%d] normalize done: %d players, %d games, %d game_logs (%.2fs)",
            season, n_players, n_games, len(log_dicts), result.steps[-1].elapsed_s,
        )

    except Exception as exc:
        result.steps.append(StepResult(
            name="normalize", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] normalize FAILED: %s", season, exc)
        return result

    # ─── Step 3: Feature Engineer ──────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info(
        "[season=%d] Step 3: feature_engineer (dry-run, sample=%d)…",
        season, DRY_RUN_SAMPLE,
    )

    try:
        # Sample: collect rows from the first DRY_RUN_SAMPLE unique players.
        # This keeps O(n) matchup computation tractable on large seasons.
        player_ids_seen: set[str] = set()
        sampled: list[dict] = []
        for row in log_dicts:
            player_ids_seen.add(row["player_id"])
            sampled.append(row)
            if len(player_ids_seen) >= DRY_RUN_SAMPLE:
                break

        # Group sampled rows by player, sort each player's games by week
        player_groups: dict[str, list[dict]] = defaultdict(list)
        for row in sampled:
            player_groups[row["player_id"]].append(row)

        feature_count = 0
        build_errors  = 0
        for pid, p_rows in player_groups.items():
            p_sorted = sorted(p_rows, key=lambda r: int(r.get("week") or 0))
            for i, target_row in enumerate(p_sorted):
                prior = p_sorted[:i]
                # Look up venue/context from the schedule (same field names)
                game = game_map.get(target_row.get("game_id") or "", {})
                try:
                    build_feature_row(target_row, prior, game, sampled)
                    feature_count += 1
                except Exception as fe_exc:
                    build_errors += 1
                    logger.debug(
                        "build_feature_row skipped (pid=%s week=%s): %s",
                        pid, target_row.get("week"), fe_exc,
                    )

        result.n_feature_rows = feature_count
        result.steps.append(StepResult(
            name="feature_engineer",
            elapsed_s=time.monotonic() - t0,
            ok=True,
            rows_in=len(sampled),
            rows_out=feature_count,
        ))
        logger.info(
            "[season=%d] feature_engineer done: %d rows built, %d skipped (%.2fs)",
            season, feature_count, build_errors, result.steps[-1].elapsed_s,
        )

    except Exception as exc:
        result.steps.append(StepResult(
            name="feature_engineer", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] feature_engineer FAILED: %s", season, exc)

    return result


# ── Live pipeline (DB required) ───────────────────────────────────────────────

def _write_dead_letter(db_url: str, source: str, error: str, season: int) -> None:
    """
    Write a pipeline-level failure to the dead_letter table.
    Called only in live mode. Silently swallows write errors so the
    orchestrator never crashes on a dead_letter write failure.
    """
    try:
        import psycopg2
        import psycopg2.extras
        dsn = _psycopg2_dsn(db_url)
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO dead_letter
                        (source, error_message, raw_payload, ingested_at)
                    VALUES (%s, %s, %s, NOW())
                    """,
                    (source, error, psycopg2.extras.Json({"season": season})),
                )
            conn.commit()
    except Exception as write_exc:
        logger.warning("Could not write to dead_letter: %s", write_exc)


def _live_season(season: int, db_url: str) -> SeasonResult:
    """Full live run for one season: writes to DB at each step."""
    result = SeasonResult(season=season, dry_run=False)

    # ─── Step 1: Ingest ────────────────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("[season=%d] Step 1: ingest (live)…", season)
    try:
        with NFLReadPyAdapter(db_url) as adapter:
            ingest_results = adapter.run_full_ingest(seasons=[season])
        total_ok   = sum(ok   for ok,   _ in ingest_results.values())
        total_fail = sum(fail for _,  fail in ingest_results.values())
        result.steps.append(StepResult(
            name="ingest", elapsed_s=time.monotonic() - t0,
            ok=True, rows_out=total_ok,
        ))
        logger.info(
            "[season=%d] ingest done: %d rows staged, %d dead-letter (%.2fs)",
            season, total_ok, total_fail, result.steps[-1].elapsed_s,
        )
    except Exception as exc:
        result.steps.append(StepResult(
            name="ingest", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] ingest FAILED: %s", season, exc)
        _write_dead_letter(db_url, "orchestrator/ingest", str(exc), season)
        return result

    # ─── Step 2: Normalize ─────────────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("[season=%d] Step 2: normalize (live)…", season)
    try:
        with Normalizer(db_url) as norm:
            summary = norm.run(seasons=[season])
        result.n_players = summary.players_upserted
        result.n_games   = summary.games_upserted
        result.steps.append(StepResult(
            name="normalize", elapsed_s=time.monotonic() - t0, ok=True,
            rows_out=(summary.players_upserted + summary.games_upserted
                      + summary.game_logs_upserted),
        ))
        logger.info(
            "[season=%d] normalize done: %d players, %d games, %d logs (%.2fs)",
            season, summary.players_upserted, summary.games_upserted,
            summary.game_logs_upserted, result.steps[-1].elapsed_s,
        )
    except Exception as exc:
        result.steps.append(StepResult(
            name="normalize", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] normalize FAILED: %s", season, exc)
        _write_dead_letter(db_url, "orchestrator/normalize", str(exc), season)
        return result

    # ─── Step 2b: Weather enrichment (precipitation_bucket for outdoor games) ─
    try:
        from scraper.adapters.weather_adapter import enrich_games_precipitation
        n = enrich_games_precipitation(db_url)
        if n > 0:
            logger.info("[season=%d] weather enrichment: %d games updated", season, n)
    except Exception as wexc:
        logger.debug("Weather enrichment skipped: %s", wexc)

    # ─── Step 3: Feature Engineer ──────────────────────────────────────────────
    t0 = time.monotonic()
    logger.info("[season=%d] Step 3: feature_engineer (live)…", season)
    try:
        with FeatureEngineer(db_url) as fe:
            n_features = fe.run(seasons=[season])
        result.n_feature_rows = n_features
        result.steps.append(StepResult(
            name="feature_engineer", elapsed_s=time.monotonic() - t0,
            ok=True, rows_out=n_features,
        ))
        logger.info(
            "[season=%d] feature_engineer done: %d rows (%.2fs)",
            season, n_features, result.steps[-1].elapsed_s,
        )
    except Exception as exc:
        result.steps.append(StepResult(
            name="feature_engineer", elapsed_s=time.monotonic() - t0,
            ok=False, error=str(exc),
        ))
        logger.error("[season=%d] feature_engineer FAILED: %s", season, exc)
        _write_dead_letter(db_url, "orchestrator/feature_engineer", str(exc), season)

    return result


# ── Orchestrator class ────────────────────────────────────────────────────────

class Orchestrator:
    """
    Wires the full NFL data pipeline: ingest → normalize → feature_engineer.

    Live mode (requires DATABASE_URL):
        orch = Orchestrator(os.environ["DATABASE_URL"])
        summary = orch.run(seasons=[2025])

    Dry-run mode (no DB required):
        orch = Orchestrator()
        summary = orch.run(seasons=[2025], dry_run=True)
    """

    def __init__(self, db_url: str = "") -> None:
        self._db_url = db_url

    def run(
        self,
        seasons: list[int],
        dry_run: bool = False,
    ) -> OrchestratorSummary:
        """
        Run all three pipeline steps for each requested season.

        Season failures are isolated: if one season fails at any step, the
        remaining seasons continue processing. The per-season step that failed
        is recorded in SeasonResult.steps with ok=False.

        Args:
            seasons:  List of NFL season years (e.g. [2024, 2025]).
            dry_run:  If True, validate and transform in memory only (no DB).

        Returns:
            OrchestratorSummary with per-season results and totals.
        """
        if not dry_run and not self._db_url:
            raise ValueError(
                "DATABASE_URL is required for live mode. "
                "Pass db_url to Orchestrator() or use dry_run=True."
            )

        start = datetime.utcnow()
        season_results: list[SeasonResult] = []

        for season in seasons:
            logger.info(
                "━━━ Orchestrator: season=%d  dry_run=%s ━━━", season, dry_run
            )
            if dry_run:
                sr = _dry_run_season(season)
            else:
                sr = _live_season(season, self._db_url)
            season_results.append(sr)

        if not dry_run:
            _run_global_enrichment(self._db_url)

        end = datetime.utcnow()
        summary = OrchestratorSummary(
            seasons=seasons,
            dry_run=dry_run,
            season_results=season_results,
            total_players=sum(sr.n_players for sr in season_results),
            total_games=sum(sr.n_games   for sr in season_results),
            total_feature_rows=sum(sr.n_feature_rows for sr in season_results),
            start_time=start,
            end_time=end,
        )
        _print_orchestrator_summary(summary)
        return summary


def _run_global_enrichment(db_url: str) -> None:
    """
    Elo/embedding enrichment and the PBP pipeline write directly into
    feature_matrix across all seasons in one pass (not per-season, no
    season argument) — previously only pipeline/run_full_etl.sh called
    them, so any orchestrator-only live run silently skipped Elo columns
    and Bucket 11 (PBP) features. Best-effort, matching run_full_etl.sh:
    a failure here is logged and does not fail the ingest/normalize/
    feature_engineer work already committed for the requested seasons.
    """
    os.environ.setdefault("DATABASE_URL", db_url)

    logger.info("Post-ETL: Elo/embedding enrichment…")
    try:
        from pipeline.enrich_elo_embeddings import main as run_elo_enrichment
        run_elo_enrichment()
    except Exception as exc:
        logger.warning("Elo/embedding enrichment failed (continuing): %s", exc)
        _write_dead_letter(db_url, "orchestrator/enrich_elo_embeddings", str(exc), 0)

    logger.info("Post-ETL: PBP pipeline…")
    try:
        from pipeline.pbp_pipeline import main as run_pbp_pipeline
        run_pbp_pipeline()
    except Exception as exc:
        logger.warning("PBP pipeline failed (continuing): %s", exc)
        _write_dead_letter(db_url, "orchestrator/pbp_pipeline", str(exc), 0)


# ── Print helper ──────────────────────────────────────────────────────────────

def _print_orchestrator_summary(summary: OrchestratorSummary) -> None:
    mode = "(DRY-RUN)" if summary.dry_run else "(LIVE)"
    elapsed = (
        (summary.end_time - summary.start_time).total_seconds()
        if summary.end_time else 0.0
    )
    print(f"\n{'═' * 72}")
    print(f"  ORCHESTRATOR SUMMARY  {mode}")
    print(f"{'═' * 72}")

    for sr in summary.season_results:
        print(f"\n  Season {sr.season}")
        for step in sr.steps:
            status = "✓" if step.ok else "✗"
            err_str = f"  ERROR: {step.error}" if step.error else ""
            print(
                f"    {status} {step.name:<20}  {step.elapsed_s:>6.2f}s"
                f"  in={step.rows_in:>6}  out={step.rows_out:>6}{err_str}"
            )
        print(
            f"    Players: {sr.n_players}"
            f"  Games: {sr.n_games}"
            f"  Features: {sr.n_feature_rows}"
        )

    print(f"\n{'─' * 72}")
    print(
        f"  TOTAL: {summary.total_players} players | "
        f"{summary.total_games} games | "
        f"{summary.total_feature_rows} feature rows | "
        f"{elapsed:.1f}s"
    )
    print(f"{'═' * 72}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full NFL data pipeline: ingest → normalize → feature_engineer."
    )
    parser.add_argument(
        "--seasons", nargs="+", type=int, metavar="YEAR",
        help="Season year(s) to process (e.g. --seasons 2024 2025). Default: current.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full chain in memory without writing to DB.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve seasons — default to current NFL season (context ETL; may be empty logs)
    if args.seasons:
        seasons = args.seasons
    else:
        try:
            import nflreadpy as nfl
            seasons = [nfl.get_current_season()]
            logger.info("Resolved current season: %d", seasons[0])
        except Exception:
            from ml.season_constants import CURRENT_SEASON

            seasons = [CURRENT_SEASON]
            logger.info(
                "Could not resolve current season; defaulting to CURRENT_SEASON=%d.",
                CURRENT_SEASON,
            )

    db_url = os.environ.get("DATABASE_URL", "")

    if args.dry_run:
        orch = Orchestrator(db_url="")
        orch.run(seasons=seasons, dry_run=True)
    else:
        if not db_url:
            logger.error(
                "DATABASE_URL not set. Use --dry-run for no-DB testing."
            )
            raise SystemExit(1)
        orch = Orchestrator(db_url=db_url)
        orch.run(seasons=seasons, dry_run=False)


if __name__ == "__main__":
    main()
