"""Phase 7 coherence detector.

The Coherent Prediction Hierarchy plan's Phase 7 verify criterion is:
"summed weekly equals season; team win totals match the game-by-game surface."

That criterion cannot pass today for a structural reason, not a modeling one:
`ProjectionService.get_season_projections` and `ml.season_simulator.SeasonSimulator`
are two independent code paths that never call each other and share no data.

- `get_season_projections` (used by the real `/projections/season/{n}` API) draws
  Bernoulli(p_active) x N(rate, residual_scale) `remaining` times from a single
  flat per-player rate (a weekly-materialized rate if present, else a season
  average) — it never reads the per-week `projections` table for future weeks,
  so it cannot reconcile with what `/projections/week/{n}` would serve for those
  same weeks.
- `SeasonSimulator` (schedule- and Elo-aware, but with its own game-resolution
  defects per the plan) has no production API consumer at all — only tests
  reference it (`grep -rn "SeasonSimulator" --include="*.py" .` outside its own
  module and `ml.season_simulator_bridge`/`ml.win_eval` turns up test files only).

These tests document that gap rather than paper over it. They should keep failing
(or keep asserting the disconnect) until a real reconciliation is built — see the
plan file's Phase 7 section for what that would require.
"""
import pytest

from backend.app.services.projection import ProjectionService
from backend.app.services import projection as projection_mod


def test_season_projections_draw_one_flat_rate_not_per_week_values(monkeypatch) -> None:
    """get_season_projections cannot equal a sum of real per-week projections:
    it draws every remaining week from the SAME rate, with no per-week schedule
    signal (opponent, home/away, injury changes week-to-week) folded in."""
    svc = ProjectionService("postgresql://unused")
    monkeypatch.setattr(projection_mod, "load_approved_pipeline_run_ids", lambda: frozenset({"run"}))

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


def test_season_simulator_has_no_production_api_consumer() -> None:
    """Documents the other half of the Phase 7 gap: `SeasonSimulator` is schedule-
    and Elo-aware but nothing in `backend/app` calls it — only tests do. Confirmed
    by grepping every `.py` file outside `ml/season_simulator.py` itself for
    `SeasonSimulator`/`season_simulator` and checking each hit's file: as of this
    test's writing, the only non-test, non-`ml/` hits are `ml/win_eval.py` (borrows
    the `_AFC`/`_NFC` conference sets, doesn't run the simulator) and
    `ml/season_simulator_bridge.py` (a C++-fast-path wrapper around it that is
    itself uncalled from `backend/app`). If this test starts failing because a
    route now imports `SeasonSimulator`, that's good news — update this docstring
    and reconsider whether `test_season_projections_draw_one_flat_rate_not_per_week_values`
    above is still the live gap or whether `get_season_projections` has been
    replaced by the real simulator per the Phase 7 plan."""
    import ast
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    api_dir = repo_root / "backend" / "app"
    hits = []
    for path in api_dir.rglob("*.py"):
        text = path.read_text()
        if "SeasonSimulator" in text or "season_simulator" in text:
            hits.append(path.relative_to(repo_root))
    assert hits == [], (
        f"backend/app now references the season simulator: {hits}. "
        "The season/week coherence gap this module documents may be resolved — "
        "verify get_season_projections actually delegates to it before updating this test."
    )
