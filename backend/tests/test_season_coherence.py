"""Phase 7 coherence detector.

The Coherent Prediction Hierarchy plan's Phase 7 verify criterion is:
"summed weekly equals season; team win totals match the game-by-game surface."

`ProjectionService.get_season_projections` now has two paths:

- A materialized-simulation path: reads `season_simulations`
  (scripts/materialize_season_simulation.py's output — a real
  `ml.season_simulator.SeasonSimulator` run using the Phase 4 team-game
  model, autoregressive Elo, and player stats coupled to their team's
  simulated score) when an approved pipeline_run_id exists for
  (season, start_week). See test_season_simulator_game_resolution.py and its
  TestPickensCoherence for what that simulator itself guarantees.
- A flat-rate fallback: draws Bernoulli(p_active) x N(rate, residual_scale)
  `remaining` times from a single flat per-player rate, with no per-week
  schedule signal folded in. This path remains exactly as limited as before
  — it's what serves whenever no approved simulation exists for the
  requested (season, start_week), which is the common case until someone
  runs the materializer and approves its output (see
  scripts/materialize_season_simulation.py's module docstring for why
  approval is a deliberate, separate release step).

test_flat_rate_fallback_draws_one_flat_rate documents the fallback path's
remaining limitation. test_prefers_materialized_simulation_when_approved
(and test_season_simulation_serving.py generally) documents that the
materialized path is real and reachable — the "SeasonSimulator has no
production API consumer" gap this module used to document is now closed for
the read side; the write side (running + approving a materialization) is
still a manual, deliberate step, not an automatic one.
"""
import pytest

from backend.app.services.projection import ProjectionService
from backend.app.services import projection as projection_mod


def test_flat_rate_fallback_draws_one_flat_rate_not_per_week_values(monkeypatch) -> None:
    """When no approved season_simulations row exists, get_season_projections
    falls back to drawing every remaining week from the SAME flat rate, with
    no per-week schedule signal (opponent, home/away, injury changes
    week-to-week) folded in — the pre-Phase-7 limitation, still present in
    this fallback path by design (see scripts/materialize_season_simulation.py
    for the real, schedule-aware alternative)."""
    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run"}))
    monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *_a, **_k: [])

    captured_rates = []

    def features(_season, _start_week, _positions):
        return [{
            "player_id": "wr1",
            "player_name": "WR One",
            "position": "WR",
            "team": "MIN",
            "prior_games": 10,
            "seas_avg_fantasy_ppr": 14.0,
            "seas_avg_passing_yards": 0.0,
            "seas_avg_rushing_yards": 5.0,
            "seas_avg_receiving_yards": 70.0,
            "depth_rank": 1.0,
        }]

    monkeypatch.setattr(svc, "_load_season_feature_rows", features)
    monkeypatch.setattr(svc, "_load_weekly_rates", lambda *_a, **_k: {})

    import ml.playing_time as playing_time_mod

    original = playing_time_mod.simulate_season_paths

    def spy(weekly_rate, p_active, remaining_weeks, **kwargs):
        captured_rates.append((weekly_rate, remaining_weeks))
        return original(weekly_rate, p_active, remaining_weeks, **kwargs)

    monkeypatch.setattr(playing_time_mod, "simulate_season_paths", spy)

    results = svc.get_season_projections(season=2025, start_week=10, stats=["receiving_yards"])
    assert results

    # One call per stat per player — one flat rate applied uniformly across
    # every remaining week, not `remaining_weeks` distinct per-week values.
    assert len(captured_rates) == 1
    rate, remaining = captured_rates[0]
    assert remaining == 19 - 10
    assert rate == pytest.approx(70.0)


def test_prefers_materialized_simulation_when_approved(monkeypatch) -> None:
    """The other half of what used to be the gap: when a materialized
    SeasonSimulator run IS approved for (season, start_week),
    get_season_projections serves it directly and never touches the
    flat-rate loaders at all — see test_season_simulation_serving.py for
    the fuller test surface on this path."""
    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run"}))
    monkeypatch.setattr(svc, "_warn_if_depth_chart_stale", lambda season: None)
    monkeypatch.setattr(svc, "_load_season_simulation_rows", lambda *_a, **_k: [{
        "player_id": "wr1", "player_name": "WR One", "position": "WR", "team": "MIN",
        "prior_games": 10, "prior_active_games": 9, "depth_rank": 1.0, "p_active": 0.9,
        "degraded": False, "interval_method": "season_simulator_mc",
        "receiving_yards": {"mean": 900.0, "p10": 700.0, "p50": 900.0, "p90": 1100.0},
        "mean": 0.0,
    }])
    monkeypatch.setattr(svc, "_load_season_feature_rows", lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("flat-rate path should not run when a simulation is approved")))

    results = svc.get_season_projections(season=2025, start_week=10, stats=["receiving_yards"])
    assert results[0]["interval_method"] == "season_simulator_mc"
    assert results[0]["receiving_yards"]["mean"] == 900.0
