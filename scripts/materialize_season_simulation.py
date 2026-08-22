#!/usr/bin/env python3
"""
Materialize a SeasonSimulator run into season_simulations for one
(season, start_week) — the rest-of-season player-projection surface behind
/projections/season/{n}, run offline because a full run (real Phase 4 model
fit + per-week Ridge predict + autoregressive Monte Carlo across the whole
roster) takes tens of seconds, too slow for a synchronous request under the
endpoint's rate limit.

Reuses ProjectionService._load_season_feature_rows for the roster + the same
prior_games/prior_active_games/depth_rank/p_active metadata the flat-rate
path already carries (so the cold-start guard — ml.playing_time.
assert_cold_start_qb_not_in_top24 — sees the same inputs either way), and
builds prior_game_rows from game_logs for SeasonSimulator's own Kalman
initialization (a per-player list of individual weekly game rows — a
different shape than the season-average rows _load_season_feature_rows
returns, and something no prior caller in this codebase built, since
SeasonSimulator had no production consumer before Phase 7).

Writing a row here does NOT make it servable: get_season_projections only
reads season_simulations rows whose pipeline_run_id is in the baseline
manifest's approved_pipeline_run_ids — pinning a new run there is a
deliberate, separate release step (see scripts/freeze_baseline.py), same
discipline as every other materialized artifact in this codebase.

Usage:
  DATABASE_URL=... python scripts/materialize_season_simulation.py \
      --season 2026 --start-week 5 --end-week 18 --n-simulations 300
"""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.extras

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.kalman_tracker import KALMAN_STATS, STAT_SOURCE_COL
from ml.season_simulator import SeasonSimulator
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL

logger = logging.getLogger(__name__)

_SERVED_STATS = ["passing_yards", "rushing_yards", "receiving_yards", "fantasy_ppr"]
_DEFAULT_POSITIONS = ["QB", "RB", "WR", "TE"]

# Raw game_logs source columns needed for _SERVED_STATS's Kalman fit (via
# STAT_SOURCE_COL). Not the full KALMAN_STATS set — compute_kalman_form
# gracefully cold-starts any stat whose source column is absent from a row
# (see ml/kalman_tracker.py), and unrequested stats are never read back out.
_SOURCE_COLS = sorted({STAT_SOURCE_COL[s] for s in _SERVED_STATS if s in KALMAN_STATS} | {
    STAT_SOURCE_COL.get("fantasy_ppr", "fantasy_points_ppr"),
})


def _load_roster(db_url: str, season: int, start_week: int, positions: list[str]) -> list[dict]:
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService(db_url)
    return svc._load_season_feature_rows(season, start_week, [p.upper() for p in positions])


def _load_prior_game_rows(
    db_url: str, season: int, start_week: int, player_ids: list[str]
) -> dict[str, list[dict]]:
    """
    {player_id: [game row dicts, oldest first]} for every game strictly
    before (season, start_week) — SeasonSimulator._compute_initial_kalman's
    expected input shape, built here for the first time since no prior
    caller in this codebase ran SeasonSimulator against real data.
    """
    if not player_ids:
        return {}
    cols = ", ".join(_SOURCE_COLS)
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT player_id, season, week, {cols}
                FROM game_logs
                WHERE player_id = ANY(%s)
                  AND (season < %s OR (season = %s AND week < %s))
                ORDER BY player_id, season, week
                """,
                (player_ids, season, season, start_week),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    out: dict[str, list[dict]] = {pid: [] for pid in player_ids}
    for r in rows:
        out[str(r["player_id"])].append(r)
    return out


def _load_schedule_gate(db_url: str, season: int, start_week: int, end_week: int):
    """
    One cheap query for real [week, home_team, away_team] rows — run()'s
    schedule_df is only a per-week gate plus the team_win_accum key source
    (see SeasonSimulator.run's docstring); _resolve_week_games pulls its own
    authoritative per-week frame from the DB regardless of what's in here.
    Deliberately NOT ml.season_simulator.load_real_schedule, which calls the
    much heavier build_team_game_forward_frame once per week (Elo/coach/
    dome joins) — wasted work for a frame we only use for gating.
    """
    import pandas as pd

    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT week, home_team, away_team FROM games
                WHERE season = %s AND week BETWEEN %s AND %s
                """,
                (season, start_week, end_week),
            )
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return pd.DataFrame(rows, columns=["week", "home_team", "away_team"])


def materialize(
    season: int,
    start_week: int,
    end_week: int = 18,
    n_simulations: int = 300,
    positions: list[str] | None = None,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
) -> int:
    positions = positions or _DEFAULT_POSITIONS
    roster_rows = _load_roster(database_url, season, start_week, positions)
    if not roster_rows:
        raise ValueError(f"No roster rows for season={season} start_week={start_week} positions={positions}")

    import pandas as pd

    players_df = pd.DataFrame([
        {"player_id": str(r["player_id"]), "position": r.get("position") or "", "team": r.get("team")}
        for r in roster_rows
    ])
    meta_by_player = {str(r["player_id"]): r for r in roster_rows}

    prior_game_rows = _load_prior_game_rows(
        database_url, season, start_week, list(players_df["player_id"])
    )
    schedule_df = _load_schedule_gate(database_url, season, start_week, end_week)

    from ml.playing_time import p_active_from_priors

    # Computed BEFORE sim.run() so it can gate the simulation itself (each
    # simulated week draws its own Bernoulli(p_active) "did this player play"
    # outcome — see SeasonSimulator.run's player_active_prob docstring) rather
    # than only being stored as metadata alongside a simulation that ignored
    # it. prior_games is the player's OWN participation count (correct for
    # cold-start detection elsewhere) — NOT a valid denominator for an
    # active-rate calculation. team_games_played (the player's team's own
    # completed-game count over the same window) is; see
    # ml.playing_time.attach_playing_time's docstring for the same fix.
    p_active_by_player = {
        str(pid): p_active_from_priors(
            meta.get("prior_snap_share"), meta.get("team_games_played"),
            depth_rank=meta.get("depth_rank"), prior_active_games=meta.get("prior_active_games"),
        )
        for pid, meta in meta_by_player.items()
    }

    sim = SeasonSimulator(
        season=season, start_week=start_week, end_week=end_week,
        n_simulations=n_simulations, stats=_SERVED_STATS,
        database_url=database_url,
    )
    result = sim.run(
        players_df=players_df, prior_game_rows=prior_game_rows,
        schedule_df=schedule_df, player_active_prob=p_active_by_player, rng_seed=0,
    )

    model_run_id = (
        f"season_sim_{season}_{start_week}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    rows = []
    for player_id, stat_dict in result.player_season_totals.items():
        meta = meta_by_player.get(player_id, {})
        p_active = p_active_by_player.get(player_id, 0.0)
        for stat in _SERVED_STATS:
            d = stat_dict.get(stat)
            if d is None:
                continue
            rows.append((
                season, start_week, player_id, stat,
                meta.get("player_name") or player_id, meta.get("position") or "", meta.get("team"),
                float(d["mean"]), float(d["p10"]), float(d["p50"]), float(d["p90"]),
                p_active, meta.get("prior_games"), meta.get("prior_active_games"), meta.get("depth_rank"),
                False, "season_simulator_mc", model_run_id,
            ))

    if not rows:
        raise ValueError("SeasonSimulator produced no player_season_totals rows")

    # Same run's week_by_week breakdown, persisted alongside the season
    # totals under the SAME pipeline_run_id — this is what makes "summed
    # weekly equals season" a checkable fact about one model's own output
    # (see test_season_simulation_serving.py's coherence test) rather than
    # an assertion about two independent pipelines that were never meant to
    # reconcile (this simulator vs. the stack models behind
    # /projections/week/{n} — see test_season_coherence.py).
    week_rows = [
        (
            season, start_week, int(w["week"]), str(w["player_id"]), w["stat"],
            float(w["mean"]), float(w["p10"]), float(w["p50"]), float(w["p90"]),
            model_run_id,
        )
        for w in result.week_by_week
    ]

    # team_win_totals is computed from the SAME per-path team score draws
    # used for player-stat coupling (_draw_team_score_paths), but was
    # discarded after run() returned — nothing served it, so there was no
    # surface for the plan's "team win totals match the game-by-game
    # surface" verify criterion (the Vikings test) to run against.
    win_rows = [
        (
            season, start_week, team,
            float(w["wins_mean"]), float(w.get("wins_p10") or 0.0), float(w.get("wins_p90") or 0.0),
            model_run_id,
        )
        for team, w in result.team_win_totals.items()
    ]

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO season_simulations
                    (season, start_week, player_id, stat, player_name, position, team,
                     mean, p10, p50, p90, p_active, prior_games, prior_active_games,
                     depth_rank, degraded, interval_method, pipeline_run_id)
                VALUES %s
                ON CONFLICT (season, start_week, player_id, stat) DO UPDATE SET
                    player_name = EXCLUDED.player_name, position = EXCLUDED.position,
                    team = EXCLUDED.team, mean = EXCLUDED.mean, p10 = EXCLUDED.p10,
                    p50 = EXCLUDED.p50, p90 = EXCLUDED.p90, p_active = EXCLUDED.p_active,
                    prior_games = EXCLUDED.prior_games,
                    prior_active_games = EXCLUDED.prior_active_games,
                    depth_rank = EXCLUDED.depth_rank, degraded = EXCLUDED.degraded,
                    interval_method = EXCLUDED.interval_method,
                    pipeline_run_id = EXCLUDED.pipeline_run_id, created_at = now()
                """,
                rows,
            )
            if week_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO season_simulation_weeks
                        (season, start_week, week, player_id, stat, mean, p10, p50, p90, pipeline_run_id)
                    VALUES %s
                    ON CONFLICT (season, start_week, week, player_id, stat) DO UPDATE SET
                        mean = EXCLUDED.mean, p10 = EXCLUDED.p10, p50 = EXCLUDED.p50,
                        p90 = EXCLUDED.p90, pipeline_run_id = EXCLUDED.pipeline_run_id,
                        created_at = now()
                    """,
                    week_rows,
                )
            if win_rows:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO season_team_wins
                        (season, start_week, team, wins_mean, wins_p10, wins_p90, pipeline_run_id)
                    VALUES %s
                    ON CONFLICT (season, start_week, team) DO UPDATE SET
                        wins_mean = EXCLUDED.wins_mean, wins_p10 = EXCLUDED.wins_p10,
                        wins_p90 = EXCLUDED.wins_p90, pipeline_run_id = EXCLUDED.pipeline_run_id,
                        created_at = now()
                    """,
                    win_rows,
                )
        conn.commit()
    finally:
        conn.close()

    logger.info(
        "Wrote %d season_simulations rows + %d season_simulation_weeks rows + "
        "%d season_team_wins rows (model_run_id=%s). NOT yet approved for "
        "serving — add %r to releases/current_baseline.json's "
        "projection_policy.approved_pipeline_run_ids and re-freeze to serve it.",
        len(rows), len(week_rows), len(win_rows), model_run_id, model_run_id,
    )
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--start-week", type=int, required=True)
    parser.add_argument("--end-week", type=int, default=18)
    parser.add_argument("--n-simulations", type=int, default=300)
    parser.add_argument("--positions", nargs="*", default=_DEFAULT_POSITIONS)
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    n = materialize(
        args.season, args.start_week, args.end_week, args.n_simulations,
        args.positions, args.database_url,
    )
    print(f"Wrote {n} season_simulations rows for season={args.season} start_week={args.start_week}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
