"""
Live-DB coherence test for scripts/materialize_season_simulation.py: the
plan's "summed weekly equals season" verify criterion, made checkable by
persisting SeasonSimulator's own week_by_week breakdown
(season_simulation_weeks) alongside its season totals (season_simulations)
under the SAME pipeline_run_id — one model's own output, not two
independent pipelines (see test_season_coherence.py for why the flat-rate
path and /projections/week/{n} were never meant to reconcile with each
other).

Skips cleanly if no database is reachable, matching this repo's convention
for tests needing real infrastructure (e.g. test_season_simulator_game_
resolution.py's TestLiveDbIntegration).
"""
from __future__ import annotations

import pytest


def _db_available() -> str | None:
    try:
        import psycopg2

        from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
        conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
        conn.close()
        return DEFAULT_HOST_DATABASE_URL
    except Exception:
        return None


class TestSummedWeeklyEqualsSeason:
    def test_materializer_week_and_season_rows_agree_for_the_same_player(self) -> None:
        db_url = _db_available()
        if db_url is None:
            pytest.skip("database unavailable")

        import psycopg2
        import psycopg2.extras

        from scripts.materialize_season_simulation import materialize

        season, start_week, end_week = 2026, 17, 18
        try:
            n = materialize(
                season=season, start_week=start_week, end_week=end_week,
                n_simulations=30, positions=["TE"], database_url=db_url,
            )
        except ValueError as exc:
            pytest.skip(f"no data to materialize for this slice: {exc}")

        assert n > 0
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT pipeline_run_id FROM season_simulations "
                    "WHERE season=%s AND start_week=%s ORDER BY created_at DESC LIMIT 1",
                    (season, start_week),
                )
                run_id_row = cur.fetchone()
                assert run_id_row is not None
                run_id = run_id_row["pipeline_run_id"]

                cur.execute(
                    "SELECT player_id, stat, mean, pipeline_run_id FROM season_simulations "
                    "WHERE season=%s AND start_week=%s AND pipeline_run_id=%s",
                    (season, start_week, run_id),
                )
                season_rows = {(r["player_id"], r["stat"]): r for r in cur.fetchall()}
                cur.execute(
                    "SELECT player_id, stat, week, mean FROM season_simulation_weeks "
                    "WHERE season=%s AND start_week=%s AND pipeline_run_id=%s",
                    (season, start_week, run_id),
                )
                week_rows = cur.fetchall()
        finally:
            conn.close()

        assert season_rows
        assert week_rows

        from collections import defaultdict

        summed: dict[tuple[str, str], float] = defaultdict(float)
        weeks_seen: dict[tuple[str, str], set] = defaultdict(set)
        for r in week_rows:
            key = (r["player_id"], r["stat"])
            summed[key] += float(r["mean"])
            weeks_seen[key].add(r["week"])

        checked = 0
        for key, season_row in season_rows.items():
            if key not in summed:
                continue
            # Same weeks covered as the simulation's own [start_week, end_week] range.
            assert weeks_seen[key] == set(range(start_week, end_week + 1)), (
                f"{key}: week coverage mismatch, got {sorted(weeks_seen[key])}"
            )
            assert summed[key] == pytest.approx(float(season_row["mean"]), rel=1e-6, abs=1e-6), (
                f"{key}: summed weekly mean={summed[key]} != season mean={season_row['mean']}"
            )
            checked += 1
        assert checked > 0, "no (player, stat) pairs present in both tables to compare"

        # Clean up only THIS test's own run_id — never a blanket delete on
        # (season, start_week), which could destroy a real candidate run a
        # human materialized at the same key.
        conn = psycopg2.connect(db_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM season_simulations WHERE pipeline_run_id=%s", (run_id,)
                )
                cur.execute(
                    "DELETE FROM season_simulation_weeks WHERE pipeline_run_id=%s", (run_id,)
                )
            conn.commit()
        finally:
            conn.close()
