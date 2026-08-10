"""
backend/tests/test_api.py

FastAPI endpoint tests using TestClient (no DB required).

Strategy:
  - All tests mock the service layer so no PostgreSQL is needed.
  - Tests verify: routing, response schema, Pydantic validation, HTTP codes.
  - data_freshness presence is asserted on every /predict response.
  - /settings weight validation (must sum to 100) is tested.
  - /backtest response contains all CLAUDE.md required fields.

Run with:
    pytest backend/tests/test_api.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# A. Health check
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "ok",
                "fallback_allowed": True,
                "artifact_mode_ready": None,
            },
        ):
            r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "ok",
                "fallback_allowed": True,
                "artifact_mode_ready": None,
            },
        ):
            body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_health_has_timestamp(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "ok",
                "fallback_allowed": True,
                "artifact_mode_ready": None,
            },
        ):
            body = client.get("/health").json()
        # Must be parseable as ISO 8601
        datetime.fromisoformat(body["timestamp"])

    def test_health_has_model_version(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "ok",
                "fallback_allowed": True,
                "artifact_mode_ready": None,
            },
        ):
            body = client.get("/health").json()
        assert "model_version" in body

    def test_health_has_product_mode(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "ok",
                "fallback_allowed": True,
                "artifact_mode_ready": None,
            },
        ):
            body = client.get("/health").json()
        assert body["product_mode"] in {"artifact_backed", "graceful_fallback"}

    def test_health_has_fallback_metadata(self):
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value={
                "overall_status": "degraded",
                "fallback_allowed": False,
                "artifact_mode_ready": False,
            },
        ):
            body = client.get("/health").json()
        assert body["fallback_allowed"] is False
        assert body["artifact_mode_ready"] is False

    def test_integrity_returns_report(self):
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": "ok",
            "product_mode": "graceful_fallback",
            "model_version": "test-v1",
            "fallback_allowed": True,
            "artifact_mode_ready": None,
            "checks": {"database": {"status": "ok", "detail": "reachable"}},
        }
        with patch(
            "backend.app.main.RuntimeStatusService.build_report",
            return_value=report,
        ):
            r = client.get("/integrity")
        assert r.status_code == 200
        assert r.json()["checks"]["database"]["status"] == "ok"

    def test_root_returns_200(self):
        assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# B. /predict endpoint
# ---------------------------------------------------------------------------

def _mock_projection_result(player_name: str = "Justin Jefferson") -> MagicMock:
    """Build a realistic ProjectionResult mock."""
    from backend.app.services.projection import ProjectionResult, PropComparison

    return ProjectionResult(
        player_id="00-0035228",
        player_name=player_name,
        position="WR",
        team="MIN",
        week=12,
        season=2025,
        stat="receiving_yards",
        projection=74.2,
        floor=38.5,
        ceiling=121.8,
        boom_probability=0.28,
        bust_probability=0.15,
        fantasy_projection=15.4,
        fantasy_floor=7.9,
        fantasy_ceiling=25.6,
        kalman_ability_estimate=88.1,
        kalman_uncertainty=6.4,
        confidence_score=0.72,
        prop_comparison=None,
        data_freshness=datetime(2025, 11, 19, 9, 14, tzinfo=timezone.utc),
        model_version="test-v1",
    )


class TestPredict:
    def test_predict_returns_200_with_mock(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=_mock_projection_result(),
        ), patch(
            "backend.app.api.predict._build_shap_factors",
            return_value=[],
        ), patch(
            "backend.app.api.predict.ProjectionService.get_feature_dict",
            return_value={},
        ):
            r = client.get("/predict?player=Jefferson&week=12&season=2025")
        assert r.status_code == 200

    def test_predict_response_has_data_freshness(self):
        """CLAUDE.md §3: data_freshness required on every /predict response."""
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=_mock_projection_result(),
        ), patch("backend.app.api.predict._build_shap_factors", return_value=[]), \
           patch("backend.app.api.predict.ProjectionService.get_feature_dict", return_value={}):
            body = client.get("/predict?player=Jefferson&week=12&season=2025").json()
        assert "data_freshness" in body
        datetime.fromisoformat(body["data_freshness"])

    def test_predict_response_schema(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=_mock_projection_result(),
        ), patch("backend.app.api.predict._build_shap_factors", return_value=[]), \
           patch("backend.app.api.predict.ProjectionService.get_feature_dict", return_value={}):
            body = client.get("/predict?player=Jefferson&week=12&season=2025").json()

        required = ["player", "player_id", "week", "season", "position",
                    "projection", "percentiles", "model_version", "data_freshness"]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_predict_percentiles_p10_lt_p90(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=_mock_projection_result(),
        ), patch("backend.app.api.predict._build_shap_factors", return_value=[]), \
           patch("backend.app.api.predict.ProjectionService.get_feature_dict", return_value={}):
            body = client.get("/predict?player=Jefferson&week=12&season=2025").json()
        p = body["percentiles"]
        assert p["p10"] < p["p50"] < p["p90"]

    def test_predict_404_when_player_not_found(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=None,
        ):
            r = client.get("/predict?player=Nonexistent&week=12&season=2025")
        assert r.status_code == 404

    def test_predict_requires_player_param(self):
        r = client.get("/predict?week=12&season=2025")
        assert r.status_code == 422

    def test_predict_requires_week_param(self):
        r = client.get("/predict?player=Jefferson&season=2025")
        assert r.status_code == 422

    def test_predict_requires_season_param(self):
        r = client.get("/predict?player=Jefferson&week=12")
        assert r.status_code == 422

    def test_predict_week_out_of_range(self):
        r = client.get("/predict?player=Jefferson&week=25&season=2025")
        assert r.status_code == 422

    def test_predict_has_top_factors_list(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=_mock_projection_result(),
        ), patch(
            "backend.app.api.predict._build_shap_factors",
            return_value=[{"feature": "kalman_est_receiving_yards",
                           "impact": 4.2, "label": "Kalman estimate"}],
        ), patch("backend.app.api.predict.ProjectionService.get_feature_dict", return_value={}):
            body = client.get("/predict?player=Jefferson&week=12&season=2025").json()
        assert isinstance(body["top_factors"], list)


# ---------------------------------------------------------------------------
# C. /projections/week/{n}
# ---------------------------------------------------------------------------

class TestWeekProjections:
    def test_week_returns_200(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_week_projections",
            return_value=[],
        ):
            r = client.get("/projections/week/12?season=2025")
        assert r.status_code == 200

    def test_week_response_schema(self):
        with patch(
            "backend.app.api.predict.ProjectionService.get_week_projections",
            return_value=[],
        ):
            body = client.get("/projections/week/12?season=2025").json()
        assert "week" in body
        assert "season" in body
        assert "projections" in body
        assert "count" in body
        assert body["count"] == 0

    def test_week_season_required(self):
        r = client.get("/projections/week/12")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# D. /explain/{player_id}
# ---------------------------------------------------------------------------

class TestExplain:
    def test_explain_returns_200(self):
        with patch(
            "backend.app.api.explain.ProjectionService.get_feature_dict",
            return_value={"kalman_est_receiving_yards": 74.2},
        ), patch(
            "backend.app.api.explain.ProjectionService._player_meta",
            return_value=("Justin Jefferson", "WR", "MIN"),
        ), patch(
            "backend.app.api.explain.ProjectionService._load_kalman",
            return_value=(74.2, 120.0),
        ), patch(
            "ml.shap_service.SHAPService.explain",
            return_value=type("Explanation", (), {
                "source": "synthetic_fallback",
                "attributions": [],
            })(),
        ):
            r = client.get("/explain/00-0035228?week=12&season=2025")
        assert r.status_code == 200

    def test_explain_schema(self):
        with patch(
            "backend.app.api.explain.ProjectionService.get_feature_dict",
            return_value={"kalman_est_receiving_yards": 74.2},
        ), patch(
            "backend.app.api.explain.ProjectionService._player_meta",
            return_value=("Justin Jefferson", "WR", "MIN"),
        ), patch(
            "backend.app.api.explain.ProjectionService._load_kalman",
            return_value=(74.2, 120.0),
        ), patch(
            "ml.shap_service.SHAPService.explain",
            return_value=type("Explanation", (), {
                "source": "synthetic_fallback",
                "attributions": [],
            })(),
        ):
            body = client.get("/explain/00-0035228?week=12&season=2025").json()

        required = ["player_id", "week", "season", "stat", "position",
                    "base_value", "top_factors", "data_freshness"]
        for field in required:
            assert field in body, f"Missing field: {field}"

    def test_explain_top_factors_have_label(self):
        """All SHAP factors must have plain-English labels (CLAUDE.md §3)."""
        with patch(
            "backend.app.api.explain.ProjectionService.get_feature_dict",
            return_value={"kalman_est_receiving_yards": 74.2, "seas_targets_avg": 8.0},
        ), patch(
            "backend.app.api.explain.ProjectionService._player_meta",
            return_value=("Justin Jefferson", "WR", "MIN"),
        ), patch(
            "backend.app.api.explain.ProjectionService._load_kalman",
            return_value=(74.2, 120.0),
        ), patch(
            "ml.shap_service.SHAPService.explain",
            return_value=type("Explanation", (), {
                "source": "synthetic_fallback",
                "attributions": [
                    type("Attr", (), {
                        "feature": "kalman_est_receiving_yards",
                        "label": "Estimated: Kalman estimated receiving yards (filtered ability)",
                        "impact": 4.2,
                        "value": 74.2,
                    })(),
                ],
            })(),
        ):
            body = client.get("/explain/00-0035228?week=12&season=2025").json()

        for factor in body["top_factors"]:
            assert "label" in factor
            assert len(factor["label"]) > 0
            # Label must not be a raw feature name (no underscores as separator)
            # (plain English labels have spaces)


# ---------------------------------------------------------------------------
# E. /scenario
# ---------------------------------------------------------------------------

class TestScenario:
    def _mock_scenario_result(self):
        from backend.app.services.projection import ProjectionResult
        from datetime import datetime, timezone
        return ProjectionResult(
            player_id="00-0035228",
            player_name="Justin Jefferson",
            position="WR",
            team="MIN",
            week=12,
            season=2025,
            stat="receiving_yards",
            projection=62.0,
            floor=28.0,
            ceiling=104.0,
            boom_probability=0.22,
            bust_probability=0.18,
            fantasy_projection=12.2,
            fantasy_floor=5.8,
            fantasy_ceiling=21.4,
            kalman_ability_estimate=88.1,
            kalman_uncertainty=6.4,
            confidence_score=0.65,
            prop_comparison=None,
            data_freshness=datetime.now(timezone.utc),
            model_version="test-v1",
        )

    def test_scenario_returns_200(self):
        with patch(
            "backend.app.api.scenario.ProjectionService._load_projection_row",
            return_value={"projection": 74.2},
        ), patch(
            "backend.app.api.scenario.ProjectionService.run_scenario",
            return_value=self._mock_scenario_result(),
        ):
            r = client.post("/scenario", json={
                "player_id": "00-0035228",
                "week": 12,
                "season": 2025,
                "stat": "receiving_yards",
                "overrides": {"wind_speed_mph": 20.0},
            })
        assert r.status_code == 200

    def test_scenario_response_has_delta(self):
        with patch(
            "backend.app.api.scenario.ProjectionService._load_projection_row",
            return_value={"projection": 74.2},
        ), patch(
            "backend.app.api.scenario.ProjectionService.run_scenario",
            return_value=self._mock_scenario_result(),
        ):
            body = client.post("/scenario", json={
                "player_id": "00-0035228",
                "week": 12,
                "season": 2025,
                "stat": "receiving_yards",
                "overrides": {"wind_speed_mph": 20.0},
            }).json()
        assert "delta" in body
        assert "delta_pct" in body
        assert "base_projection" in body
        assert "scenario_projection" in body

    def test_scenario_requires_all_fields(self):
        r = client.post("/scenario", json={"player_id": "00-0035228"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# F. /backtest
# ---------------------------------------------------------------------------

class TestBacktest:
    def _mock_summary(self):
        from backend.app.services.backtest import BacktestSummary, CalibrationPoint
        return BacktestSummary(
            model_version="test-v1",
            seasons=[2022, 2023, 2024],
            positions=["WR"],
            stat="receiving_yards",
            overall_mae=18.4,
            overall_rmse=24.7,
            overall_crps=12.1,
            brier_score=0.22,
            simulated_pnl=3.5,
            sharpe_ratio=0.82,
            max_drawdown=1.2,
            calibration=[CalibrationPoint(0.1, 0.09, 50)],
            by_season=[],
            data_source="test_fixture",
            metric_notes={},
            data_freshness=datetime.now(timezone.utc),
        )

    def test_backtest_returns_200(self):
        with patch(
            "backend.app.api.backtest.BacktestService.get_summary",
            return_value=self._mock_summary(),
        ):
            r = client.get("/backtest")
        assert r.status_code == 200

    def test_backtest_has_all_required_fields(self):
        """CLAUDE.md §3: /backtest must include all these fields."""
        with patch(
            "backend.app.api.backtest.BacktestService.get_summary",
            return_value=self._mock_summary(),
        ):
            body = client.get("/backtest").json()

        required = [
            "overall_mae",   # MAE
            "overall_rmse",  # RMSE
            "brier_score",   # Brier score
            "simulated_pnl", # simulated P&L
            "sharpe_ratio",  # Sharpe ratio
            "max_drawdown",  # max drawdown
            "calibration",   # calibration data points
            "data_source",
            "metric_notes",
        ]
        for field in required:
            assert field in body, f"Missing CLAUDE.md required field: {field}"

    def test_backtest_calibration_is_list(self):
        with patch(
            "backend.app.api.backtest.BacktestService.get_summary",
            return_value=self._mock_summary(),
        ):
            body = client.get("/backtest").json()
        assert isinstance(body["calibration"], list)

    def test_backtest_has_data_freshness(self):
        with patch(
            "backend.app.api.backtest.BacktestService.get_summary",
            return_value=self._mock_summary(),
        ):
            body = client.get("/backtest").json()
        assert "data_freshness" in body

    def test_backtest_returns_503_when_artifact_mode_requires_assets(self):
        from backend.app.core.runtime_mode import ArtifactRequiredError

        with patch(
            "backend.app.api.backtest.BacktestService.get_summary",
            side_effect=ArtifactRequiredError("artifact-backed backtest assets missing"),
        ):
            r = client.get("/backtest?stat=receiving_yards")
        assert r.status_code == 503

    def test_csv_sort_key_is_deterministic_not_mtime(self):
        """
        BacktestService._load_csv must sort CSVs by a deterministic scalar key.

        Two regressions guarded here:

        1. The key must not be the raw ``stat_result`` struct — sorting those
           raises TypeError once more than one CSV exists. (Original intent.)
        2. The key must not be ``st_mtime`` either. A filesystem timestamp is not
           a statement about which artifact is correct: ``touch`` reorders them,
           and a fresh clone stamps every file with the checkout time, so the
           winner becomes arbitrary. Selection is by filename, which carries the
           run stamp.
        """
        import inspect
        from backend.app.services.backtest import BacktestService
        src = inspect.getsource(BacktestService._load_csv)
        # Compare on code only, so the explanatory comments in _load_csv (which
        # name the rejected approach) do not trip the negative assertion.
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "key=lambda p: p.name" in code, (
            "_load_csv must sort by filename (which carries the run stamp)"
        )
        assert "st_mtime" not in code, (
            "_load_csv must not select artifacts by mtime — a touched or freshly "
            "cloned file would change which backtest is served"
        )


class TestExplainArtifactMode:
    def test_explain_returns_503_when_artifact_mode_requires_real_shap(self):
        from backend.app.core.runtime_mode import ArtifactRequiredError

        with patch(
            "backend.app.api.explain.ProjectionService.get_feature_dict",
            return_value={"kalman_est_receiving_yards": 74.2},
        ), patch(
            "backend.app.api.explain.ProjectionService._player_meta",
            return_value=("Justin Jefferson", "WR", "MIN"),
        ), patch(
            "ml.shap_service.SHAPService.explain",
            side_effect=ArtifactRequiredError("shap artifacts missing"),
        ):
            r = client.get("/explain/00-0035228?week=12&season=2025")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# G. /settings
# ---------------------------------------------------------------------------

class TestSettings:
    def test_get_settings_returns_200(self):
        r = client.get("/settings")
        assert r.status_code == 200

    def test_get_settings_schema(self):
        body = client.get("/settings").json()
        assert "weights" in body
        assert "fantasy_scoring" in body
        assert "engine_exposure_cap" in body

    def test_weights_sum_to_100(self):
        body = client.get("/settings").json()
        weights = body["weights"]
        total = sum(weights.values())
        assert abs(total - 100.0) < 0.1

    def test_put_settings_returns_200(self):
        valid_payload = {
            "weights": {
                "kalman_form": 35.0,
                "seasonal_baseline": 20.0,
                "matchup": 20.0,
                "weather_venue": 10.0,
                "team_context": 5.0,
                "roster_injury": 5.0,
                "rule_meta": 5.0,
            },
            "fantasy_scoring": "ppr",
        }
        r = client.put("/settings", json=valid_payload)
        assert r.status_code == 200

    def test_put_settings_rejects_weights_not_summing_to_100(self):
        bad_payload = {
            "weights": {
                "kalman_form": 50.0,   # too high
                "seasonal_baseline": 20.0,
                "matchup": 20.0,
                "weather_venue": 10.0,
                "team_context": 5.0,
                "roster_injury": 5.0,
                "rule_meta": 5.0,      # total = 115, not 100
            }
        }
        r = client.put("/settings", json=bad_payload)
        assert r.status_code == 422

    def test_put_settings_rejects_invalid_fantasy_scoring(self):
        r = client.put("/settings", json={"fantasy_scoring": "fantasy_draftkings"})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# H. /alerts
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_alerts_returns_200(self):
        assert client.get("/alerts").status_code == 200

    def test_alerts_schema(self):
        body = client.get("/alerts").json()
        assert "alerts" in body
        assert "count" in body
        assert isinstance(body["alerts"], list)

    def test_alerts_n_param_respected(self):
        # n=1 should work and not error
        r = client.get("/alerts?n=1")
        assert r.status_code == 200

    def test_alerts_n_out_of_range_rejected(self):
        r = client.get("/alerts?n=500")
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# I. SHAP label quality (CLAUDE.md requirement)
# ---------------------------------------------------------------------------

class TestSHAPLabels:
    def test_feature_labels_dict_exists(self):
        from ml.shap_service import FEATURE_LABELS
        assert isinstance(FEATURE_LABELS, dict)
        assert len(FEATURE_LABELS) > 50  # comprehensive

    def test_all_kalman_stats_have_labels(self):
        from ml.shap_service import FEATURE_LABELS
        from ml.kalman_tracker import KALMAN_STATS
        for stat in KALMAN_STATS:
            est_key = f"kalman_est_{stat}"
            assert est_key in FEATURE_LABELS, f"Missing label for {est_key}"

    def test_labels_are_plain_english(self):
        from ml.shap_service import FEATURE_LABELS
        for key, lbl in FEATURE_LABELS.items():
            assert isinstance(lbl, str)
            assert len(lbl) > 0
            # Labels must not be identical to their raw key
            assert lbl != key, f"Label for {key!r} is same as key"
            # Labels start with a capital letter
            assert lbl[0].isupper(), f"Label for {key!r} does not start with capital: {lbl!r}"

    def test_label_function_fallback(self):
        from ml.shap_service import label
        # Unknown feature should still return a formatted string
        result = label("some_unknown_feature_xyz")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_every_feature_labels_key_exists_in_fm_cols(self):
        """
        Every key in FEATURE_LABELS must be a real column in _FM_COLS or an
        identifier column. This prevents SHAP explanations from referencing
        phantom features that don't exist in the feature_matrix table.
        """
        from ml.shap_service import FEATURE_LABELS
        from pipeline.feature_engineer import _FM_COLS
        # Identifiers in _FM_COLS + synthetic fallback columns also valid as labels
        fm_cols_set = set(_FM_COLS)
        for key in FEATURE_LABELS:
            assert key in fm_cols_set, (
                f"FEATURE_LABELS key {key!r} is not in _FM_COLS. "
                f"Either update the label key or add the column to feature_matrix."
            )


# ---------------------------------------------------------------------------
# J. Stat whitelist — SQL injection guard (CRITICAL fix)
# ---------------------------------------------------------------------------

class TestStatWhitelist:
    """VALID_STATS whitelist must reject malicious or invalid stat values
    before they reach any f-string SQL query."""

    def test_valid_stat_receiving_yards_accepted(self):
        from backend.app.services.projection import VALID_STATS
        assert "receiving_yards" in VALID_STATS

    def test_valid_stat_passing_yards_accepted(self):
        from backend.app.services.projection import VALID_STATS
        assert "passing_yards" in VALID_STATS

    def test_valid_stats_covers_all_skill_positions(self):
        from backend.app.services.projection import VALID_STATS
        expected = {
            "passing_yards", "rushing_yards", "receiving_yards",
            "receptions", "passing_tds", "rushing_tds", "receiving_tds",
            "carries", "targets",
        }
        assert expected.issubset(VALID_STATS)

    def test_injection_payload_not_in_whitelist(self):
        from backend.app.services.projection import VALID_STATS
        assert "1; DROP TABLE projections--" not in VALID_STATS
        assert "receiving_yards; DROP TABLE players--" not in VALID_STATS

    def test_predict_invalid_stat_returns_422(self):
        r = client.get("/predict?player=Jefferson&week=12&season=2025&stat=INVALID")
        assert r.status_code == 422

    def test_predict_injection_stat_returns_422(self):
        r = client.get(
            "/predict?player=Jefferson&week=12&season=2025"
            "&stat=receiving_yards%3B+DROP+TABLE+projections--"
        )
        assert r.status_code == 422

    def test_projections_week_invalid_stat_returns_422(self):
        r = client.get("/projections/week/1?season=2025&stat=NOT_A_STAT")
        assert r.status_code == 422

    def test_kalman_valid_stats_whitelist_matches(self):
        from backend.app.services.kalman import VALID_STATS as KS
        from backend.app.services.projection import VALID_STATS as PS
        # Both services must use identical whitelists
        assert KS == PS


# ---------------------------------------------------------------------------
# K. Projection importable from app.models (MEDIUM 1 fix)
# ---------------------------------------------------------------------------

class TestProjectionImport:
    def test_projection_importable_from_app_models(self):
        from backend.app.models import Projection
        assert Projection is not None


# ---------------------------------------------------------------------------
# L. AlertService thread safety (MEDIUM 2 fix)
# ---------------------------------------------------------------------------

class TestAlertServiceThreadSafety:
    def test_alert_service_concurrent_emit(self):
        import threading
        from backend.app.services.alert import AlertService
        # Reset singleton for test isolation
        AlertService._instance = None
        svc = AlertService()
        threads = [
            threading.Thread(target=svc.publish_system, args=("t", "b"))
            for _ in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert svc._counter == 50
        assert len(svc._history) == 50

    def teardown_method(self, _method):
        # Restore singleton after isolation test so other tests still work.
        from backend.app.services.alert import AlertService
        AlertService._instance = None


# ---------------------------------------------------------------------------
# M. /season/current endpoint (MEDIUM 5 fix)
# ---------------------------------------------------------------------------

class TestSeasonCurrent:
    def test_season_current_returns_200(self):
        r = client.get("/season/current")
        assert r.status_code == 200

    def test_season_current_returns_valid_shape(self):
        r = client.get("/season/current")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body["season"], int)
        assert isinstance(body["week"], int)

    def test_season_current_season_is_reasonable(self):
        body = client.get("/season/current").json()
        # Should be a plausible NFL season year
        assert 2019 <= body["season"] <= 2030

    def test_season_current_week_is_reasonable(self):
        body = client.get("/season/current").json()
        # Must be a valid NFL week (1-22)
        assert 1 <= body["week"] <= 22


# ---------------------------------------------------------------------------
# Z. API key middleware
# ---------------------------------------------------------------------------

class TestAPIKeyMiddleware:
    """
    Tests for the opt-in X-API-Key guard (APIKeyMiddleware in main.py).

    The middleware is a no-op when settings.api_key == "".
    When a key is configured, all non-exempt paths require the header.
    """

    def test_no_key_configured_allows_all_requests(self):
        with patch("backend.app.main.settings.api_key", ""):
            r = client.get("/season/current")
        assert r.status_code != 401

    def test_key_configured_missing_header_returns_401(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            r = client.get("/season/current")
        assert r.status_code == 401
        assert "X-API-Key" in r.json()["detail"]

    def test_key_configured_wrong_header_returns_401(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            r = client.get("/season/current", headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 401

    def test_key_configured_correct_header_passes(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            r = client.get("/season/current", headers={"X-API-Key": "test-secret-key"})
        assert r.status_code != 401

    def test_health_exempt_without_key_header(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            with patch(
                "backend.app.main.RuntimeStatusService.build_report",
                return_value={
                    "overall_status": "ok",
                    "fallback_allowed": True,
                    "artifact_mode_ready": None,
                },
            ):
                r = client.get("/health")
        assert r.status_code == 200

    def test_root_exempt_without_key_header(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            r = client.get("/")
        assert r.status_code == 200

    def test_integrity_exempt_without_key_header(self):
        with patch("backend.app.main.settings.api_key", "test-secret-key"):
            with patch(
                "backend.app.main.RuntimeStatusService.build_report",
                return_value={
                    "overall_status": "ok",
                    "fallback_allowed": True,
                    "artifact_mode_ready": None,
                },
            ):
                r = client.get("/integrity")
        assert r.status_code == 200
