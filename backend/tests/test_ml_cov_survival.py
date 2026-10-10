"""
backend/tests/test_ml_cov_survival.py

Coverage tests for ml/survival_model.py (player availability survival model).

Offline only. `lifelines` is not installed in every environment, so the
CoxPH wrapper logic is exercised through a stand-in module injected into
sys.modules (monkeypatch restores it). One test uses
pytest.importorskip("lifelines") to run the real CoxPHFitter when present.
"""

from __future__ import annotations

import sys
import types

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ml import survival_model as sm
from ml.survival_model import (
    POSITION_ENCODING,
    PlayerSurvivalModel,
    _heuristic_return_probabilities,
    build_survival_dataset,
    get_survival_model,
    reset_survival_model,
)


# ── Stand-in for lifelines.CoxPHFitter ────────────────────────────────────────
# Module-level so instances pickle (save/load round-trip). Behaviour is driven
# by _STUB, which the fixture resets for every test.

_STUB: dict = {}


def _reset_stub() -> None:
    _STUB.clear()
    _STUB.update({
        "survival_frame": None,
        "fit_calls": [],
        "predict_calls": [],
        "fail_fit": False,
        "fail_predict": False,
    })


class _StubCoxPHFitter:
    def __init__(self, penalizer: float = 0.0) -> None:
        self.penalizer = penalizer
        self.concordance_index_ = 0.71
        self.summary = pd.DataFrame({"coef": [0.1, -0.2]}, index=["age", "snap_pct_off"])
        self.baseline_survival_ = pd.Series(
            [1.0, 0.8, 0.5], index=pd.Index([0.0, 2.0, 4.0], name="timeline"), name="baseline",
        )

    def fit(self, df, duration_col, event_col, show_progress=False):
        if _STUB["fail_fit"]:
            raise RuntimeError("synthetic convergence failure")
        _STUB["fit_calls"].append({
            "rows": len(df),
            "duration_col": duration_col,
            "event_col": event_col,
            "penalizer": self.penalizer,
            "columns": list(df.columns),
        })
        return self

    def predict_survival_function(self, covariate_df):
        _STUB["predict_calls"].append(covariate_df.copy())
        if _STUB["fail_predict"]:
            raise ValueError("synthetic prediction failure")
        return _STUB["survival_frame"]


@pytest.fixture
def fake_lifelines(monkeypatch):
    _reset_stub()
    mod = types.ModuleType("lifelines")
    mod.CoxPHFitter = _StubCoxPHFitter
    monkeypatch.setitem(sys.modules, "lifelines", mod)
    yield _STUB
    _reset_stub()


def _survival_training_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "duration": rng.integers(1, 6, n),
        "event_observed": rng.integers(0, 2, n),
        "games_missed_streak": rng.integers(1, 4, n),
        "injury_status_encoded": rng.integers(0, 5, n),
        "snap_pct_off": rng.uniform(0.2, 0.9, n),
        "age": rng.uniform(22, 34, n),
        "position_encoded": rng.integers(0, 4, n),
    })


# ── Heuristic fallback ────────────────────────────────────────────────────────

class TestHeuristicFallback:
    def test_out_status_cumulative_return_probability(self) -> None:
        # p_week = 0.05 for Out. P(returned by week t) = 1 - 0.95**t.
        probs = _heuristic_return_probabilities(0, 3)
        assert probs == pytest.approx([1 - 0.95 ** t for t in (1, 2, 3)])
        assert probs == pytest.approx([0.05, 0.0975, 0.142625])

    def test_full_status_is_high_return_rate(self) -> None:
        assert _heuristic_return_probabilities(4, 2) == pytest.approx([0.95, 1 - 0.05 ** 2])

    def test_unknown_status_uses_questionable_default(self) -> None:
        # Status 9 is not in the table, so the 0.50 default applies.
        assert _heuristic_return_probabilities(9, 3) == pytest.approx([0.5, 0.75, 0.875])

    def test_zero_weeks_returns_empty(self) -> None:
        assert _heuristic_return_probabilities(2, 0) == []

    def test_probabilities_are_monotone_non_decreasing(self) -> None:
        probs = _heuristic_return_probabilities(3, 6)
        assert all(b >= a for a, b in zip(probs, probs[1:]))
        assert all(0.0 <= p <= 1.0 for p in probs)


# ── PlayerSurvivalModel without a fitted model ───────────────────────────────

class TestUnfittedModel:
    def test_starts_unfitted_with_covariates(self) -> None:
        model = PlayerSurvivalModel()
        assert model.is_fitted is False
        assert model._covariates == [
            "games_missed_streak", "injury_status_encoded", "snap_pct_off", "age", "position_encoded",
        ]

    def test_predict_uses_heuristic_when_unfitted(self) -> None:
        probs = PlayerSurvivalModel().predict_return_probability(
            games_missed_streak=2, injury_status_encoded=2, n_weeks=2,
        )
        assert probs == pytest.approx(_heuristic_return_probabilities(2, 2))

    def test_predict_availability_for_week_returns_last_week(self) -> None:
        # Full status, two weeks out: P(available in week 2) = 1 - 0.05**2.
        value = PlayerSurvivalModel().predict_availability_for_week(
            games_missed_streak=1, injury_status_encoded=4, weeks_until_target=2,
        )
        assert value == pytest.approx(1 - 0.05 ** 2)

    def test_predict_availability_zero_weeks_is_zero(self) -> None:
        assert PlayerSurvivalModel().predict_availability_for_week(1, 0, weeks_until_target=0) == 0.0

    def test_summary_none_and_save_is_noop_when_unfitted(self, tmp_path, caplog) -> None:
        model = PlayerSurvivalModel()
        assert model.summary() is None
        target = tmp_path / "unfitted.pkl"
        with caplog.at_level("WARNING", logger=sm.logger.name):
            model.save(str(target))
        assert not target.exists()
        assert "Cannot save unfitted model" in caplog.text

    def test_plot_is_noop_when_unfitted(self, tmp_path) -> None:
        out = tmp_path / "surv.png"
        PlayerSurvivalModel().plot_survival_function(out_path=str(out))
        assert not out.exists()


# ── fit() guard rails ─────────────────────────────────────────────────────────

class TestFitGuards:
    def test_fit_falls_back_when_lifelines_missing(self, monkeypatch) -> None:
        # A None entry in sys.modules makes `from lifelines import ...` raise ImportError.
        monkeypatch.setitem(sys.modules, "lifelines", None)
        model = PlayerSurvivalModel()
        assert model.fit(_survival_training_frame()) is model
        assert model.is_fitted is False

    def test_fit_falls_back_on_missing_columns(self, fake_lifelines) -> None:
        df = _survival_training_frame().drop(columns=["age"])
        model = PlayerSurvivalModel().fit(df)
        assert model.is_fitted is False
        assert fake_lifelines["fit_calls"] == []

    def test_fit_falls_back_below_minimum_rows(self, fake_lifelines) -> None:
        df = _survival_training_frame(n=sm._MIN_TRAINING_ROWS - 1)
        model = PlayerSurvivalModel().fit(df)
        assert model.is_fitted is False
        assert fake_lifelines["fit_calls"] == []

    def test_fit_drops_nan_rows_before_threshold(self, fake_lifelines) -> None:
        # 60 rows, 11 with a NaN covariate -> 49 complete rows, below the 50 minimum.
        df = _survival_training_frame(n=60)
        df.loc[df.index[:11], "snap_pct_off"] = np.nan
        model = PlayerSurvivalModel().fit(df)
        assert model.is_fitted is False
        assert fake_lifelines["fit_calls"] == []

    def test_fit_failure_falls_back_to_heuristic(self, fake_lifelines) -> None:
        fake_lifelines["fail_fit"] = True
        model = PlayerSurvivalModel().fit(_survival_training_frame())
        assert model.is_fitted is False
        assert model.predict_return_probability(0, 0, n_weeks=1) == pytest.approx([0.05])


# ── fit() success path with the stand-in CoxPH ───────────────────────────────

class TestFittedModel:
    def test_fit_passes_clean_frame_and_penalizer(self, fake_lifelines) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [1.0, 0.8, 0.5]}, index=[0.0, 2.0, 4.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))

        assert model.is_fitted is True
        assert len(fake_lifelines["fit_calls"]) == 1
        call = fake_lifelines["fit_calls"][0]
        assert call["rows"] == 60
        assert call["duration_col"] == "duration"
        assert call["event_col"] == "event_observed"
        assert call["penalizer"] == pytest.approx(0.1)
        assert set(call["columns"]) == {"duration", "event_observed", *model._covariates}

    def test_predict_interpolates_survival_by_searchsorted(self, fake_lifelines) -> None:
        # Survival defined at durations 0, 2, 4 with S = 1.0, 0.8, 0.5.
        # For week t the code takes the last index with duration <= t:
        #   t=1 -> S=1.0 (return 0.0); t=2 -> S=0.8 (0.2); t=3 -> S=0.8 (0.2); t=4 -> S=0.5 (0.5).
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [1.0, 0.8, 0.5]}, index=[0.0, 2.0, 4.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        probs = model.predict_return_probability(games_missed_streak=2, injury_status_encoded=2, n_weeks=4)
        assert probs == pytest.approx([0.0, 0.2, 0.2, 0.5])

    def test_predict_clips_survival_above_one(self, fake_lifelines) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [1.2, 1.2]}, index=[0.0, 3.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        assert model.predict_return_probability(0, 2, n_weeks=2) == pytest.approx([0.0, 0.0])

    def test_position_encoding_passed_to_covariates(self, fake_lifelines) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [0.9]}, index=[0.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        model.predict_return_probability(3, 1, snap_pct_off=0.5, age=30.0, position="te", n_weeks=1)
        model.predict_return_probability(3, 1, position="K", n_weeks=1)
        te_row, k_row = (c.iloc[0].to_dict() for c in fake_lifelines["predict_calls"])
        assert te_row["position_encoded"] == POSITION_ENCODING["TE"] == 3
        assert te_row["games_missed_streak"] == 3
        assert te_row["snap_pct_off"] == pytest.approx(0.5)
        assert te_row["age"] == pytest.approx(30.0)
        # Unknown positions default to WR (2).
        assert k_row["position_encoded"] == 2

    def test_prediction_error_falls_back_to_heuristic(self, fake_lifelines) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [0.9]}, index=[0.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        fake_lifelines["fail_predict"] = True
        assert model.predict_return_probability(1, 3, n_weeks=2) == pytest.approx(
            _heuristic_return_probabilities(3, 2)
        )

    def test_summary_returns_fitted_coefficients(self, fake_lifelines) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [0.9]}, index=[0.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        summary = model.summary()
        assert list(summary.index) == ["age", "snap_pct_off"]
        assert summary.loc["age", "coef"] == pytest.approx(0.1)

    def test_plot_writes_png(self, fake_lifelines, tmp_path) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [0.9]}, index=[0.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        out = tmp_path / "nested" / "survival.png"
        model.plot_survival_function(out_path=str(out))
        assert out.exists()
        assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    def test_save_and_load_round_trip(self, fake_lifelines, tmp_path) -> None:
        fake_lifelines["survival_frame"] = pd.DataFrame({0: [1.0, 0.8, 0.5]}, index=[0.0, 2.0, 4.0])
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=60))
        path = tmp_path / "sub" / "survival.pkl"
        model.save(str(path))
        assert path.exists()

        restored = PlayerSurvivalModel().load(str(path))
        assert restored.is_fitted is True
        assert restored.predict_return_probability(2, 2, n_weeks=4) == pytest.approx([0.0, 0.2, 0.2, 0.5])

    def test_fit_with_real_lifelines_when_available(self) -> None:
        pytest.importorskip("lifelines")
        model = PlayerSurvivalModel().fit(_survival_training_frame(n=120))
        assert model.is_fitted is True
        probs = model.predict_return_probability(2, 2, snap_pct_off=0.6, age=27.0, position="WR", n_weeks=3)
        assert len(probs) == 3
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert probs == sorted(probs)


# ── build_survival_dataset ────────────────────────────────────────────────────

class TestBuildSurvivalDataset:
    def _inputs(self):
        feature_matrix = pd.DataFrame([
            # QB A: active week 1, injured weeks 2-3, returns in week 5.
            {"player_id": "A", "season": 2024, "week": 1, "games_missed_streak": 0,
             "injury_status_encoded": 4, "snap_pct_off": 0.9, "age": 30.0, "position": "QB"},
            {"player_id": "A", "season": 2024, "week": 2, "games_missed_streak": 1,
             "injury_status_encoded": 1, "snap_pct_off": 0.9, "age": 30.0, "position": "QB"},
            {"player_id": "A", "season": 2024, "week": 3, "games_missed_streak": 2,
             "injury_status_encoded": 1, "snap_pct_off": 0.9, "age": 30.0, "position": "QB"},
            # TE B: injured week 4 and never plays again this season.
            {"player_id": "B", "season": 2024, "week": 4, "games_missed_streak": 1,
             "injury_status_encoded": 2, "snap_pct_off": 0.5, "age": 24.0, "position": "TE"},
        ])
        game_logs = pd.DataFrame([
            {"player_id": "A", "season": 2024, "week": 1},
            {"player_id": "A", "season": 2024, "week": 5},
            {"player_id": "B", "season": 2024, "week": 3},
        ])
        return feature_matrix, game_logs

    def test_builds_one_row_per_injury_entry_with_event_flags(self) -> None:
        feature_matrix, game_logs = self._inputs()
        out = build_survival_dataset(feature_matrix, game_logs)

        assert len(out) == 3  # active week-1 row is skipped
        assert list(out.columns) == [
            "duration", "event_observed", "games_missed_streak", "injury_status_encoded",
            "snap_pct_off", "age", "position_encoded",
        ]
        # Player A returned in week 5 -> both injury entries are observed events.
        a_rows = out[out["position_encoded"] == POSITION_ENCODING["QB"]]
        assert list(a_rows["duration"]) == [1, 2]
        assert list(a_rows["event_observed"]) == [1, 1]
        assert list(a_rows["injury_status_encoded"]) == [1, 1]

        # Player B never returned -> censored (event 0).
        b_row = out[out["position_encoded"] == POSITION_ENCODING["TE"]].iloc[0]
        assert b_row["event_observed"] == 0
        assert b_row["duration"] == 1
        assert b_row["age"] == pytest.approx(24.0)

    def test_no_injury_entries_gives_empty_frame(self) -> None:
        # Every row is an active week (streak 0), so no survival records are built.
        feature_matrix = pd.DataFrame([
            {"player_id": "D", "season": 2024, "week": 1, "games_missed_streak": 0,
             "injury_status_encoded": 4, "snap_pct_off": 0.9, "age": 25.0, "position": "RB"},
        ])
        _, game_logs = self._inputs()
        assert build_survival_dataset(feature_matrix, game_logs).empty

    def test_out_status_zero_is_preserved(self) -> None:
        feature_matrix = pd.DataFrame([
            {"player_id": "A", "season": 2024, "week": 2, "games_missed_streak": 1,
             "injury_status_encoded": 0, "snap_pct_off": 0.9, "age": 30.0, "position": "QB"},
        ])
        game_logs = pd.DataFrame([{"player_id": "A", "season": 2024, "week": 5}])
        out = build_survival_dataset(feature_matrix, game_logs)
        assert list(out["injury_status_encoded"]) == [0]

    def test_missing_position_defaults_to_wr(self) -> None:
        feature_matrix = pd.DataFrame([
            {"player_id": "C", "season": 2023, "week": 6, "games_missed_streak": 3,
             "injury_status_encoded": 3, "snap_pct_off": 0.4, "age": 26.0},
        ])
        game_logs = pd.DataFrame([{"player_id": "C", "season": 2023, "week": 1}])
        out = build_survival_dataset(feature_matrix, game_logs)
        assert out.iloc[0]["position_encoded"] == POSITION_ENCODING["WR"]
        assert out.iloc[0]["event_observed"] == 0
        assert out.iloc[0]["duration"] == 3


# ── Module singleton ──────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_returns_same_instance_until_reset(self) -> None:
        reset_survival_model()
        first = get_survival_model()
        assert get_survival_model() is first
        reset_survival_model()
        assert get_survival_model() is not first
        reset_survival_model()
