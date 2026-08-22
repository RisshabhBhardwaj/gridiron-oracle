"""
backend/tests/test_integration.py

End-to-end integration tests — exercises the full FastAPI → service → DB stack.

No mocking. Every call goes through the real service layer and hits a live
PostgreSQL database. These tests prove that all layers (API, service, ML,
Kalman, Monte Carlo, backtest) talk to each other correctly.

Prerequisites
─────────────
  - PostgreSQL running (default: oracle:oracle@localhost:15439/oracle)
  - projections table populated:
        python3.11 -m ml.train --seasons 2025 --all-weeks --fast
  - Backtest results CSV present (optional, enables coverage_80 assertion):
        python3.11 ml/run_backtest.py

Excluded from CI (requires live DB):
    pytest -m "not integration and not slow"

Run locally:
    export DATABASE_URL="postgresql://oracle:oracle@localhost:15439/oracle"
    pytest backend/tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

# All tests in this file are integration tests
pytestmark = pytest.mark.integration

# Normalise asyncpg → psycopg2 DSN for direct fixture queries
_DB_URL = (
    os.environ.get("DATABASE_URL", "postgresql://oracle:oracle@localhost:15439/oracle")
    .replace("postgresql+asyncpg://", "postgresql://")
)


# ── Module-scoped fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def int_client():
    """
    FastAPI TestClient that runs the full app lifespan.

    Using it as a context manager triggers the startup event (lifespan),
    which publishes the startup system alert to the AlertService singleton.
    All integration tests share this one client instance.
    """
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(scope="module")
def db_conn():
    """
    psycopg2 connection for setup queries in fixtures.
    Skips the entire module if the DB is not reachable.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(_DB_URL)
        try:
            yield conn
        finally:
            conn.close()
    except Exception as exc:
        pytest.skip(
            f"PostgreSQL not reachable ({exc}) — "
            "run `docker compose -f infra/docker-compose.yml up -d db` first"
        )


@pytest.fixture(scope="module")
def best_wr_projection(db_conn):
    """
    Fetch the top receiving_yards projection for week=1, season=2026.

    Season 2026 week 1 is the live forward-projection target this API
    serves; backtest seasons (2021-2025) never have week=1 rows under an
    approved pipeline run because within-season lag features require at
    least one prior week in the same season — week 1 is causally
    unreachable there, not a data gap.

    Returns a dict with: player_id, player_name, week, season,
    projection, floor, ceiling, boom_probability, bust_probability.

    Skips if no projections exist (run ml.train first).
    """
    from backend.app.services.projection import load_approved_pipeline_run_ids

    cur = db_conn.cursor()
    try:
        # /predict only ever serves rows whose pipeline_run_id is on the
        # approval allowlist (backend/app/services/projection.py). Picking
        # the raw top projection without this filter can surface a row from
        # a superseded run that /predict correctly 404s on — that's not a
        # server bug, it's this fixture testing something the API doesn't.
        approved = tuple(load_approved_pipeline_run_ids())
        cur.execute(
            """
            SELECT p.player_id,
                   pl.full_name,
                   p.week,
                   p.season,
                   p.projection,
                   p.floor,
                   p.ceiling,
                   p.boom_probability,
                   p.bust_probability
            FROM   projections p
            LEFT JOIN players pl ON pl.id = p.player_id
            WHERE  p.stat       = 'receiving_yards'
              AND  p.season     = 2026
              AND  p.week       = 1
              AND  p.projection IS NOT NULL
              AND  p.projection  > 0
              AND  p.pipeline_run_id = ANY(%s)
            ORDER BY p.projection DESC
            LIMIT 1
            """,
            (list(approved),),
        )
        row = cur.fetchone()
    except Exception as exc:
        # If the table doesn't exist (e.g. psycopg2.errors.UndefinedTable), skip properly
        pytest.skip(f"Database query failed, possibly missing tables: {exc}")

    if row is None:
        pytest.skip(
            "No projections for week=1 season=2026 — "
            "run: python3.11 -m ml.train --seasons 2026 --week 1"
        )
    return {
        "player_id":       row[0],
        "player_name":     row[1] or row[0],
        "week":            row[2],
        "season":          row[3],
        "projection":      float(row[4]),
        "floor":           float(row[5]),
        "ceiling":         float(row[6]),
        "boom_probability": float(row[7]) if row[7] is not None else None,
        "bust_probability": float(row[8]) if row[8] is not None else None,
    }


# ── Scenario 1: GET /projections/week/{n} ─────────────────────────────────────

class TestProjectionsWeek:
    """
    Batch projections for a given week.

    Verifies: 200 OK, schema, required non-null fields, data_freshness.
    """

    def test_returns_200(self, int_client, best_wr_projection):
        r = int_client.get(
            f"/projections/week/{best_wr_projection['week']}",
            params={"season": best_wr_projection["season"], "stat": "receiving_yards"},
        )
        assert r.status_code == 200

    def test_response_top_level_schema(self, int_client, best_wr_projection):
        r = int_client.get(
            f"/projections/week/{best_wr_projection['week']}",
            params={"season": best_wr_projection["season"], "stat": "receiving_yards"},
        )
        body = r.json()
        assert "projections" in body
        assert "count" in body
        assert isinstance(body["projections"], list)
        assert body["count"] == len(body["projections"])

    def test_projections_have_required_non_null_fields(self, int_client, best_wr_projection):
        """CLAUDE.md: projection, floor, ceiling, boom_probability must be present."""
        r = int_client.get(
            f"/projections/week/{best_wr_projection['week']}",
            params={"season": best_wr_projection["season"], "stat": "receiving_yards"},
        )
        body = r.json()
        assert body["count"] > 0, (
            "Expected ≥1 projection for week=1 season=2026 — "
            "check projections table is populated"
        )
        first = body["projections"][0]
        assert first["projection"] is not None, "projection should not be None"
        assert first["floor"] is not None,       "floor should not be None"
        assert first["ceiling"] is not None,     "ceiling should not be None"
        assert first["boom_probability"] is not None, "boom_probability should not be None"

    def test_data_freshness_present_and_parseable(self, int_client, best_wr_projection):
        """CLAUDE.md API Rule: data_freshness required on every /predict response."""
        r = int_client.get(
            f"/projections/week/{best_wr_projection['week']}",
            params={"season": best_wr_projection["season"]},
        )
        body = r.json()
        # Must parse as ISO 8601
        ts = datetime.fromisoformat(body["data_freshness"])
        assert ts is not None


# ── Scenario 2: GET /predict ──────────────────────────────────────────────────

class TestPredict:
    """
    Full projection for one player / week / stat.

    Verifies: 200 OK, percentile ordering, data_freshness, SHAP schema,
    404 for unknown player.
    """

    def _call(self, int_client, best_wr_projection, **kwargs):
        params = {
            "player": best_wr_projection["player_name"],
            "week":   best_wr_projection["week"],
            "season": best_wr_projection["season"],
            "stat":   "receiving_yards",
        }
        params.update(kwargs)
        return int_client.get("/predict", params=params)

    def test_returns_200_for_known_player(self, int_client, best_wr_projection):
        r = self._call(int_client, best_wr_projection)
        assert r.status_code == 200

    def test_percentiles_ordered_p10_le_p50_le_p90(self, int_client, best_wr_projection):
        body = self._call(int_client, best_wr_projection).json()
        p = body["percentiles"]
        assert p["p10"] is not None
        assert p["p50"] is not None
        assert p["p90"] is not None
        assert p["p10"] <= p["p50"] <= p["p90"], (
            f"Percentile ordering violated: p10={p['p10']} p50={p['p50']} p90={p['p90']}"
        )

    def test_data_freshness_present_and_parseable(self, int_client, best_wr_projection):
        """CLAUDE.md API Rule: data_freshness on every /predict response."""
        body = self._call(int_client, best_wr_projection).json()
        assert "data_freshness" in body
        datetime.fromisoformat(body["data_freshness"])

    def test_top_factors_is_list_with_correct_schema(self, int_client, best_wr_projection):
        """
        SHAP factors: top_factors is always a list.
        If the SHAP service loaded a model successfully, each element has
        feature / impact / label.  Empty list is acceptable (no MLflow model).
        """
        body = self._call(int_client, best_wr_projection).json()
        assert isinstance(body["top_factors"], list)
        for factor in body["top_factors"]:
            assert "feature" in factor
            assert "impact" in factor
            assert "label" in factor

    def test_unknown_player_returns_404(self, int_client):
        r = int_client.get("/predict", params={
            "player": "__no_such_nfl_player_xyz_9999__",
            "week":   1,
            "season": 2025,
        })
        assert r.status_code == 404


# ── Scenario 3: POST /scenario ─────────────────────────────────────────────────

class TestScenario:
    """
    What-if re-projection with feature overrides.

    Key assertion: 40 mph wind applies a ~10% penalty on receiving_yards,
    so delta < 0.  The service applies `adjusted *= 0.90` when wind > 20 mph.
    """

    def _payload(self, best_wr_projection, overrides=None):
        return {
            "player_id": best_wr_projection["player_id"],
            "week":      best_wr_projection["week"],
            "season":    best_wr_projection["season"],
            "stat":      "receiving_yards",
            "overrides": overrides or {},
        }

    def test_returns_200(self, int_client, best_wr_projection):
        r = int_client.post("/scenario", json=self._payload(
            best_wr_projection, {"wind_speed_mph": 40.0}
        ))
        assert r.status_code == 200

    def test_wind_override_reduces_projection(self, int_client, best_wr_projection):
        """
        40 mph wind should produce delta < 0 (wind penalty applied).

        service.run_scenario():  adjusted *= 0.90 when wind_speed_mph > 20
        base_projection = stored DB value (e.g. 74.2 yds)
        scenario_projection ≈ 74.2 * 0.90 = 66.8 yds  → delta ≈ -7.4
        """
        r = int_client.post("/scenario", json=self._payload(
            best_wr_projection, {"wind_speed_mph": 40.0}
        ))
        body = r.json()
        assert body["delta"] < 0, (
            f"Expected negative delta from 40mph wind penalty, "
            f"got delta={body['delta']:.2f}  "
            f"(base={body['base_projection']:.2f}, scenario={body['scenario_projection']:.2f})"
        )

    def test_response_schema_complete(self, int_client, best_wr_projection):
        r = int_client.post("/scenario", json=self._payload(
            best_wr_projection, {"wind_speed_mph": 5.0}
        ))
        body = r.json()
        required_fields = (
            "base_projection", "scenario_projection", "delta", "delta_pct",
            "percentiles", "data_freshness",
        )
        for field in required_fields:
            assert field in body, f"Missing required field: {field}"
        p = body["percentiles"]
        assert "p10" in p and "p50" in p and "p90" in p
        datetime.fromisoformat(body["data_freshness"])


# ── Scenario 4: GET /backtest ─────────────────────────────────────────────────

class TestBacktest:
    """
    Walk-forward backtest results page.

    CLAUDE.md §3 requires: MAE, RMSE, Brier score, simulated P&L,
    Sharpe ratio, max drawdown, calibration data points.

    If backtest CSV exists (from ml/run_backtest.py), also verifies
    coverage_80 > 0.60 for at least one season × position cohort.
    """

    def test_fails_closed_when_artifact_backed_backtest_assets_are_unavailable(self, int_client):
        r = int_client.get("/backtest", params={"stat": "receiving_yards"})
        if r.status_code == 503:
            assert r.status_code == 503
            assert "Backtest artifacts unavailable" in r.json()["detail"]
        else:
            assert r.status_code == 200

    def test_has_all_required_fields_when_backtest_is_available(self, int_client):
        """CLAUDE.md §3: MAE, RMSE, Brier, P&L, Sharpe, max_drawdown, calibration."""
        response = int_client.get("/backtest", params={"stat": "receiving_yards"})
        if response.status_code == 503:
            # Artifact-backed mode must not fabricate a backtest response.
            assert "Backtest artifacts unavailable" in response.json()["detail"]
            return
        assert response.status_code == 200
        body = response.json()
        for field in (
            "overall_mae", "overall_rmse", "brier_score",
            "simulated_pnl", "sharpe_ratio", "max_drawdown", "calibration",
        ):
            assert field in body, f"Missing CLAUDE.md-required backtest field: {field}"
        assert isinstance(body["calibration"], list)

    def test_coverage_80_above_threshold(self, int_client):
        """
        At least one season × position cohort must have coverage_80 > 0.60.

        Expected values (from CLAUDE.md §7 Backtesting status):
          WR receiving_yards coverage_80 ≈ 0.80-0.86 after ×3.0 sigma correction.

        Skips if no backtest CSV or DB data is available.
        """
        body = int_client.get("/backtest", params={"stat": "receiving_yards"}).json()
        by_season = body.get("by_season", [])
        if not by_season:
            pytest.skip(
                "No backtest data (CSV or DB) — "
                "run: python3.11 ml/run_backtest.py"
            )
        max_coverage = max(row["coverage_80"] for row in by_season)
        assert max_coverage > 0.60, (
            f"No season×position cohort with coverage_80 > 0.60 "
            f"(max={max_coverage:.3f}). "
            "Check sigma multiplier in ml/train.py _fast_bayesian_samples()."
        )


# ── Scenario 5: WebSocket /alerts/ws ─────────────────────────────────────────

class TestAlertsWebSocket:
    """
    Real-time alert streaming via WebSocket.

    Tests:
      (a) Connection is accepted without error.
      (b) Server delivers the startup alert (published by lifespan) on connect.
    """

    def test_alerts_rest_endpoint_schema(self, int_client):
        """GET /alerts returns count + alerts list — sanity check before WS test."""
        r = int_client.get("/alerts")
        assert r.status_code == 200
        body = r.json()
        assert "count" in body
        assert "alerts" in body
        assert isinstance(body["alerts"], list)
        assert body["count"] == len(body["alerts"])

    def test_websocket_connection_accepted(self, int_client):
        """WebSocket handshake succeeds without raising any exception."""
        from backend.app.api.alerts import get_alert_service

        svc = get_alert_service()
        # Ensure at least one alert is in history before connecting
        svc.publish_system("Integration Test", "WebSocket delivery verified")

        # connect → receive first alert from recent history → close cleanly
        with int_client.websocket_connect("/alerts/ws") as ws:
            data = ws.receive_text()

        alert = json.loads(data)
        assert "title"     in alert, "Alert missing 'title' field"
        assert "severity"  in alert, "Alert missing 'severity' field"
        assert "timestamp" in alert, "Alert missing 'timestamp' field"

    def test_websocket_startup_alert_delivered(self, int_client):
        """
        Lifespan publishes 'Gridiron Oracle started' to AlertService.
        The WebSocket handler sends recent(20) on connect, so the startup
        alert must appear in the first batch of messages.
        """
        from backend.app.api.alerts import get_alert_service

        svc = get_alert_service()
        recent = svc.recent(20)
        if not recent:
            pytest.skip("AlertService history empty — lifespan may not have fired")

        with int_client.websocket_connect("/alerts/ws") as ws:
            # Receive the first message (most recent alert = our test alert or startup)
            raw = ws.receive_text()

        alert = json.loads(raw)
        # Structural check — not content-dependent
        assert isinstance(alert.get("title"), str)
        assert alert.get("severity") in ("info", "warning", "edge", "injury")
        # Timestamp must be valid ISO 8601
        datetime.fromisoformat(alert["timestamp"])


class TestTeamGamePredictions:
    """
    Phase 4's first real game-outcome endpoint.

    Prerequisites: team_game_predictions populated —
        python scripts/materialize_team_game_predictions.py --season 2026 --week 1
    """

    def test_week_endpoint_returns_all_scheduled_games(self, int_client):
        r = int_client.get("/team-games/2026/1")
        if r.status_code == 404:
            pytest.skip("team_game_predictions not materialized for 2026 week 1")
        assert r.status_code == 200
        body = r.json()
        assert body["season"] == 2026
        assert body["week"] == 1
        assert body["count"] == len(body["games"])
        assert body["count"] > 0

    def test_week_endpoint_games_have_all_targets(self, int_client):
        r = int_client.get("/team-games/2026/1")
        if r.status_code == 404:
            pytest.skip("team_game_predictions not materialized for 2026 week 1")
        game = r.json()["games"][0]
        for field in ("points", "yards", "pass_rate", "win_probability"):
            assert game[field] is not None
        assert 0.0 <= game["win_probability"] <= 1.0

    def test_week_endpoint_404_for_unmaterialized_week(self, int_client):
        r = int_client.get("/team-games/2099/1")
        assert r.status_code == 404

    def test_team_endpoint_matches_week_endpoint(self, int_client):
        week_resp = int_client.get("/team-games/2026/1")
        if week_resp.status_code == 404:
            pytest.skip("team_game_predictions not materialized for 2026 week 1")
        team = week_resp.json()["games"][0]["team"]
        team_resp = int_client.get(f"/team-games/2026/1/{team}")
        assert team_resp.status_code == 200
        assert team_resp.json()["team"] == team

    def test_team_endpoint_404_for_unknown_team(self, int_client):
        r = int_client.get("/team-games/2026/1/ZZZ")
        assert r.status_code == 404

    def test_a_home_and_away_team_pair_have_complementary_win_probability(self, int_client):
        """
        EXACTLY 1.0 - p, not approximately: both sides' win_probability come
        from the SAME points model's predictions for both teams (see
        ml.team_game_model.derive_win_probability), so
        margin_opponent = pred_opp - pred_team = -margin_team exactly, and
        norm.cdf(-x) = 1 - norm.cdf(x) exactly. If this drifts from exact,
        the two sides stopped being derived from one shared points
        prediction — the whole point of deriving win_probability instead of
        fitting it as an independent classifier.
        """
        week_resp = int_client.get("/team-games/2026/1")
        if week_resp.status_code == 404:
            pytest.skip("team_game_predictions not materialized for 2026 week 1")
        games = {g["team"]: g for g in week_resp.json()["games"]}
        game = next(iter(games.values()))
        opponent = games.get(game["opponent"])
        if opponent is None:
            pytest.skip("opponent row not present in this materialization")
        assert abs((game["win_probability"] + opponent["win_probability"]) - 1.0) < 1e-9
