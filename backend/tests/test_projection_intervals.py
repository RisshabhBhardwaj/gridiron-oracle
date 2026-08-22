from pathlib import Path

import pytest
from fastapi import HTTPException

from backend.app.services.projection import (
    _interval_method,
    _interval_value,
    load_approved_pipeline_run_ids,
)


def test_stored_conformal_method_exposes_floor() -> None:
    row = {
        "interval_method": "causal_oof_conformal_90",
        "floor": 12.0,
        "ceiling": 40.0,
        "fantasy_floor": -3.0,
        "position": "WR",
        "posterior_samples": None,
    }
    assert _interval_method(row) == "causal_oof_conformal_90"
    assert _interval_value(row, "floor") == 12.0
    assert _interval_value(row, "ceiling") == 40.0
    assert _interval_value(row, "fantasy_floor") == 0.0


def test_unavailable_method_hides_legacy_bands() -> None:
    row = {"interval_method": "unavailable", "floor": 1.0, "ceiling": 99.0, "posterior_samples": None}
    assert _interval_method(row) == "unavailable"
    assert _interval_value(row, "floor") is None


def test_empty_approved_runs_fail_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "baseline.json"
    manifest.write_text('{"projection_policy": {"approved_pipeline_run_ids": []}}')
    with pytest.raises(HTTPException) as exc:
        load_approved_pipeline_run_ids(manifest)
    assert exc.value.status_code == 503


def test_season_projections_db_error_is_fail_closed() -> None:
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService("postgresql://unused")
    with pytest.raises(HTTPException) as exc:
        svc.get_season_projections(season=2025, start_week=1)
    assert exc.value.status_code == 503


def test_season_projections_rank_by_playing_time(monkeypatch) -> None:
    from backend.app.services import projection as projection_mod
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run"}))
    monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *_a, **_k: [])

    def features(_season, _start_week, _positions):
        rows = []
        for i in range(24):
            rows.append({
                "player_id": f"wr{i}",
                "player_name": f"WR{i}",
                "position": "WR",
                "team": "MIN",
                "prior_snap_share": 0.75,
                "prior_games": 16,
                "seas_avg_fantasy_ppr": 14.0,
                "seas_avg_passing_yards": 0.0,
                "seas_avg_rushing_yards": 5.0,
                "seas_avg_receiving_yards": 70.0,
                "depth_rank": 1.0,
            })
        rows.append({
            "player_id": "cold-qb",
            "player_name": "Backup",
            "position": "QB",
            "team": "KC",
            "prior_snap_share": 0.0,
            "prior_games": 0,
            "prior_active_games": 0,
            "seas_avg_fantasy_ppr": 20.0,
            "seas_avg_passing_yards": 250.0,
            "seas_avg_rushing_yards": 0.0,
            "seas_avg_receiving_yards": 0.0,
            "depth_rank": 2.0,
        })
        rows.append({
            "player_id": "starter-qb",
            "player_name": "Starter",
            "position": "QB",
            "team": "KC",
            "prior_snap_share": 0.98,
            "prior_games": 17,
            "seas_avg_fantasy_ppr": 22.0,
            "seas_avg_passing_yards": 280.0,
            "seas_avg_rushing_yards": 20.0,
            "seas_avg_receiving_yards": 0.0,
            "depth_rank": 1.0,
        })
        return rows

    monkeypatch.setattr(svc, "_load_season_feature_rows", features)
    monkeypatch.setattr(svc, "_load_weekly_rates", lambda *_args, **_kwargs: {})
    results = svc.get_season_projections(season=2025, start_week=1)
    assert results
    assert all(row.get("interval_method") == "playing_time_gaussian_mc" for row in results)
    assert all("degraded" in row for row in results)
    top24 = [row["player_id"] for row in results if int(row["ros_rank"]) <= 24]
    assert "cold-qb" not in top24
    assert "starter-qb" in [row["player_id"] for row in results]


def test_load_season_feature_rows_differentiates_active_rate_by_real_availability() -> None:
    """
    Regression test for the prior_active_games/team_games_played fix.

    Two earlier attempts got this wrong in opposite directions:
      1. prior_active_games always None -> active-rate collapses toward 0
         for well-established players (a healthy 90%-snap-share, 18-game
         starter came out at p_active=0.15).
      2. prior_active_games aliased to the SAME column as the denominator
         -> active rate is mathematically always ~1.0 regardless of real
         missed-game history (400/400 sampled rows had active==games).

    The fix must produce BOTH a plausible high floor for durable players AND
    real separation for a player who missed a meaningful fraction of their
    team's games — this test checks the actual distribution, not just a
    single floor value either prior attempt's own test would have passed.
    """
    try:
        import psycopg2

        from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
        conn = psycopg2.connect(DEFAULT_HOST_DATABASE_URL)
        conn.close()
    except Exception as exc:
        pytest.skip(f"PostgreSQL not reachable ({exc})")

    from ml.playing_time import attach_playing_time
    from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
    from backend.app.services.projection import ProjectionService

    svc = ProjectionService(DEFAULT_HOST_DATABASE_URL)
    rows = svc._load_season_feature_rows(2025, 15, ["WR", "RB"])
    if not rows:
        pytest.skip("no season feature rows available for 2025 week<15")

    assert all("team_games_played" in row for row in rows)
    assert all("prior_active_games" in row for row in rows)

    sample = [r for r in rows if (r.get("prior_games") or 0) >= 5]
    if len(sample) < 20:
        pytest.skip("not enough rows with meaningful history for this check")

    attached = attach_playing_time(sample)
    p_active_vals = [r["p_active"] for r in attached]

    # Not everyone collapsed to the same value (the "always ==" bug).
    assert max(p_active_vals) - min(p_active_vals) > 0.2, (
        "p_active shows almost no spread across players — team_games_played "
        "may be equal to prior_active_games again"
    )

    # Someone who missed a meaningful share of their team's games scores
    # meaningfully lower than someone who didn't.
    missed_games = [
        r for r in attached
        if r.get("team_games_played") and r.get("prior_active_games") is not None
        and r["team_games_played"] > 0
        and r["prior_active_games"] < r["team_games_played"] * 0.6
    ]
    full_attendance = [
        r for r in attached
        if r.get("team_games_played") and r.get("prior_active_games") is not None
        and r["team_games_played"] > 0
        and r["prior_active_games"] >= r["team_games_played"] * 0.95
    ]
    if missed_games and full_attendance:
        avg_missed = sum(r["p_active"] for r in missed_games) / len(missed_games)
        avg_full = sum(r["p_active"] for r in full_attendance) / len(full_attendance)
        assert avg_full > avg_missed, (
            f"full-attendance players (avg p_active={avg_full:.3f}) should score "
            f"higher than players who missed games (avg p_active={avg_missed:.3f})"
        )
