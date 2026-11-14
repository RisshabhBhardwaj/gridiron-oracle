"""
backend/tests/test_weighting_service.py

End-to-end proof for the Settings layer (weights-08 stream, item 6):

- Default settings reproduce the stored row exactly (identity), carrying
  weighting "default".
- Non-default bucket weights move the numbers in the expected direction
  once attributions exist (the hook), and leave canonical numbers
  untouched while they do not (reason "no_attribution").
- fantasy_scoring ppr returns the stored fantasy number; half_ppr and
  standard re-score components (a 10-reception line differs by exactly 5).
- "standard" maps to STANDARD_NON_PPR, never STANDARD_PPR.
- engine_exposure_cap maps to kelly_fraction with the hard cap fixed.
"""

from __future__ import annotations

import pytest

from backend.app.services import weighting
from backend.app.services.weighting import (
    apply_settings_to_row,
    engine_kelly_config,
)
from ml import bucket_weights as bw
from ml.scoring import (
    STANDARD_NON_PPR,
    STANDARD_PPR,
    rules_for_scoring,
)


def _default_settings(**overrides) -> dict:
    cfg = {
        "weights": dict(bw.DEFAULT_BUCKET_WEIGHTS),
        "fantasy_scoring": "ppr",
        "engine_exposure_cap": 0.25,
        "weight_preset_name": "Default",
    }
    cfg.update(overrides)
    return cfg


def _row_kwargs(**overrides):
    kw = {
        "projection": 17.5,
        "floor": 8.0,
        "ceiling": 29.0,
        "fantasy_projection": 12.0,
        "component_stats": None,
        "player_id": "00-0034796",
        "week": 12,
        "season": 2025,
        "stat": "receiving_yards",
        "settings": _default_settings(),
    }
    kw.update(overrides)
    return kw


class TestRowIdentityAtDefaults:
    def test_default_settings_reproduce_stored_row_exactly(self):
        out = apply_settings_to_row(**_row_kwargs())
        assert out.projection == 17.5
        assert out.floor == 8.0
        assert out.ceiling == 29.0
        assert out.fantasy_projection == 12.0
        assert out.weighting == "default"
        assert out.weights_reason == "defaults"
        assert out.fantasy_scoring == "ppr"

    def test_identity_holds_with_none_intervals(self):
        out = apply_settings_to_row(
            **_row_kwargs(floor=None, ceiling=None, fantasy_projection=None)
        )
        assert out.projection == 17.5
        assert out.floor is None
        assert out.ceiling is None
        assert out.fantasy_projection is None


class TestBucketWeightHook:
    def test_non_default_weights_without_attributions_leave_numbers(self):
        settings = _default_settings(
            weights={**bw.DEFAULT_BUCKET_WEIGHTS, "kalman_form": 50.0}
        )
        out = apply_settings_to_row(**_row_kwargs(settings=settings))
        assert out.projection == 17.5
        assert out.floor == 8.0
        assert out.ceiling == 29.0
        assert out.weighting == "default"
        assert out.weights_reason == "no_attribution"

    def test_non_default_weights_with_attributions_move_numbers(
        self, monkeypatch
    ):
        phi = {key: 0.0 for key in bw.BUCKET_KEYS}
        phi["kalman_form"] = 4.0
        monkeypatch.setattr(
            weighting, "load_bucket_attributions", lambda *a: dict(phi)
        )
        settings = _default_settings(
            weights={**bw.DEFAULT_BUCKET_WEIGHTS, "kalman_form": 50.0}
        )
        out = apply_settings_to_row(**_row_kwargs(settings=settings))
        delta = (50.0 / 35.0 - 1) * 4.0
        assert out.projection == pytest.approx(17.5 + delta)
        assert out.floor == pytest.approx(8.0 + delta)
        assert out.ceiling == pytest.approx(29.0 + delta)
        assert out.weighting == "user"
        assert out.weights_reason is None


class TestFantasyScoring:
    COMPONENTS_10_RECEPTIONS = {
        "receptions": 10.0,
        "receiving_yards": 95.0,
        "receiving_tds": 1.0,
        "rushing_yards": 0.0,
    }

    def test_ppr_returns_stored_fantasy_number(self):
        out = apply_settings_to_row(
            **_row_kwargs(component_stats=self.COMPONENTS_10_RECEPTIONS)
        )
        assert out.fantasy_projection == 12.0
        assert out.fantasy_scoring == "ppr"

    def test_standard_ignores_receptions(self):
        out = apply_settings_to_row(
            **_row_kwargs(
                component_stats=self.COMPONENTS_10_RECEPTIONS,
                settings=_default_settings(fantasy_scoring="standard"),
            )
        )
        assert out.fantasy_projection == pytest.approx(95.0 * 0.1 + 6.0)
        assert out.fantasy_scoring == "standard"

    def test_half_and_standard_differ_by_five_for_ten_receptions(self):
        kw = _row_kwargs(component_stats=self.COMPONENTS_10_RECEPTIONS)
        half = apply_settings_to_row(
            **{**kw, "settings": _default_settings(fantasy_scoring="half_ppr")}
        )
        std = apply_settings_to_row(
            **{**kw, "settings": _default_settings(fantasy_scoring="standard")}
        )
        assert half.fantasy_projection - std.fantasy_projection == pytest.approx(5.0)

    def test_missing_components_keep_stored_value(self):
        out = apply_settings_to_row(
            **_row_kwargs(
                component_stats=None,
                settings=_default_settings(fantasy_scoring="standard"),
            )
        )
        assert out.fantasy_projection == 12.0


class TestScoringMapping:
    def test_standard_is_non_ppr_not_ppr(self):
        assert rules_for_scoring("standard") is STANDARD_NON_PPR
        assert rules_for_scoring("standard").pts_per_reception == 0.0
        assert rules_for_scoring("ppr") is STANDARD_PPR

    def test_unknown_scoring_raises(self):
        with pytest.raises(KeyError):
            rules_for_scoring("super_ppr")


class TestEngineMapping:
    def test_default_cap_equals_engine_default(self):
        cfg = engine_kelly_config(0.25)
        assert cfg["kelly_fraction"] == 0.25
        # The hard cap is fixed: this mapping must never loosen it.
        assert cfg["max_bet_fraction"] == pytest.approx(0.05)

    def test_custom_cap_flows_to_kelly_fraction_only(self):
        cfg = engine_kelly_config(0.5)
        assert cfg["kelly_fraction"] == 0.5
        assert cfg["max_bet_fraction"] == pytest.approx(0.05)


class TestPredictCarriesWeightingLabel:
    def _mock_result(self, **overrides):
        from datetime import datetime, timezone

        from backend.app.services.projection import ProjectionResult

        kw = {
            "player_id": "00-0034796",
            "player_name": "Test Player",
            "position": "WR",
            "team": "MIN",
            "week": 12,
            "season": 2025,
            "stat": "receiving_yards",
            "projection": 17.5,
            "floor": 8.0,
            "ceiling": 29.0,
            "boom_probability": None,
            "bust_probability": None,
            "fantasy_projection": 12.0,
            "fantasy_floor": None,
            "fantasy_ceiling": None,
            "kalman_ability_estimate": None,
            "kalman_uncertainty": None,
            "confidence_score": None,
            "prop_comparison": None,
            "data_freshness": datetime(2025, 11, 19, 9, 14, tzinfo=timezone.utc),
            "model_version": "test-v1",
        }
        kw.update(overrides)
        return ProjectionResult(**kw)

    def test_predict_labels_default_output(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from backend.app.main import app

        client = TestClient(app, raise_server_exceptions=True)
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=self._mock_result(),
        ), patch(
            "backend.app.api.predict._build_shap_factors", return_value=[]
        ):
            body = client.get("/predict?player=Test&week=12&season=2025").json()
        assert body["weighting"] == "default"
        assert body["fantasy_scoring"] == "ppr"
        assert body["projection"]["receiving_yards"] == pytest.approx(17.5)

    def test_predict_passes_user_weighting_through(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from backend.app.main import app

        client = TestClient(app, raise_server_exceptions=True)
        with patch(
            "backend.app.api.predict.ProjectionService.get_projection",
            return_value=self._mock_result(
                projection=18.8, weighting="user", weights_reason=None
            ),
        ), patch(
            "backend.app.api.predict._build_shap_factors", return_value=[]
        ):
            body = client.get("/predict?player=Test&week=12&season=2025").json()
        assert body["weighting"] == "user"
        assert body["weights_reason"] is None
        assert body["projection"]["receiving_yards"] == pytest.approx(18.8)


class _FakeCursor:
    def __init__(self, row):
        self._row = row
        self.executed = []

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self.cursor_obj = _FakeCursor(row)

    def cursor(self):
        return self.cursor_obj

    def close(self):
        pass


def _stored(monkeypatch, row):
    """Point the real loader at a fake database returning ``row``."""
    conns: list[_FakeConn] = []

    def connect(_url):
        conns.append(_FakeConn(row))
        return conns[-1]

    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", connect)
    from backend.app.services import projection

    monkeypatch.setattr(projection, "load_approved_pipeline_run_ids", lambda *a: frozenset({"run_a"}))
    return conns


def _db_row(projection=17.5, kalman_form=4.0):
    # projection, base, unweighted, then BUCKET_KEYS in order
    buckets = [kalman_form, 1.0, 2.0, 0.5, 0.25, 0.25, 0.0]
    base = 10.0
    unweighted = projection - base - sum(buckets)
    return (projection, base, unweighted, *buckets)


class TestStoredAttributionLoader:
    def test_loader_returns_buckets_keyed_by_name(self, monkeypatch):
        _stored(monkeypatch, _db_row())
        got = weighting.load_bucket_attributions("p", 12, 2025, "receiving_yards")
        assert list(got) == list(bw.BUCKET_KEYS)
        assert got["kalman_form"] == 4.0

    def test_query_requires_matching_run_id_and_approved_run(self, monkeypatch):
        conns = _stored(monkeypatch, _db_row())
        weighting.load_bucket_attributions("p", 12, 2025, "receiving_yards")
        sql, params = conns[0].cursor_obj.executed[0]
        assert "a.pipeline_run_id = p.pipeline_run_id" in sql
        assert "p.pipeline_run_id = ANY" in sql
        assert ["run_a"] in params

    def test_no_row_means_no_attribution(self, monkeypatch):
        _stored(monkeypatch, None)
        assert weighting.load_bucket_attributions("p", 12, 2025, "receiving_yards") is None

    def test_drifted_row_is_ignored_not_used(self, monkeypatch):
        row = list(_db_row())
        row[1] += 0.01  # base no longer adds up to the served projection
        _stored(monkeypatch, tuple(row))
        assert weighting.load_bucket_attributions("p", 12, 2025, "receiving_yards") is None

    def test_database_failure_means_no_attribution(self, monkeypatch):
        import psycopg2

        def boom(_url):
            raise psycopg2.OperationalError("down")

        monkeypatch.setattr(psycopg2, "connect", boom)
        assert weighting.load_bucket_attributions("p", 12, 2025, "receiving_yards") is None


class TestStoredAttributionsChangeServedNumbers:
    def test_defaults_are_identity_and_skip_the_lookup(self, monkeypatch):
        conns = _stored(monkeypatch, _db_row())
        out = apply_settings_to_row(**_row_kwargs())
        assert out.projection == 17.5
        assert out.weighting == "default"
        assert conns == []

    def test_non_default_weights_move_the_number_by_the_stored_share(self, monkeypatch):
        _stored(monkeypatch, _db_row(projection=17.5, kalman_form=4.0))
        settings = _default_settings(
            weights={**bw.DEFAULT_BUCKET_WEIGHTS, "kalman_form": 50.0}
        )
        out = apply_settings_to_row(**_row_kwargs(settings=settings))
        assert out.projection == pytest.approx(17.5 + (50.0 / 35.0 - 1) * 4.0)
        assert out.weighting == "user"

    def test_weights_raising_one_bucket_and_lowering_another_net_out(self, monkeypatch):
        _stored(monkeypatch, _db_row())
        weights = {**bw.DEFAULT_BUCKET_WEIGHTS, "kalman_form": 45.0, "seasonal_baseline": 10.0}
        out = apply_settings_to_row(**_row_kwargs(settings=_default_settings(weights=weights)))
        expected = 17.5 + (45 / 35 - 1) * 4.0 + (10 / 20 - 1) * 1.0
        assert out.projection == pytest.approx(expected)

    def test_non_default_weights_fall_back_when_no_row_is_stored(self, monkeypatch):
        _stored(monkeypatch, None)
        settings = _default_settings(
            weights={**bw.DEFAULT_BUCKET_WEIGHTS, "kalman_form": 50.0}
        )
        out = apply_settings_to_row(**_row_kwargs(settings=settings))
        assert out.projection == 17.5
        assert out.weights_reason == "no_attribution"
