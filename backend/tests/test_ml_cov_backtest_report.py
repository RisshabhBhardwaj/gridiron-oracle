"""
backend/tests/test_ml_cov_backtest_report.py

Coverage tests for ml/backtest_report.py (OOF metrics and HTML report).

Every metric is checked against a value computed by hand on a tiny frame.
Files are written under tmp_path only.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ml import backtest_report as br
from ml.backtest_report import (
    BacktestReport,
    ReportMetrics,
    _bias,
    _coverage_at_threshold,
    _crps_empirical,
    _mae,
    _rmse,
)


def _oof(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestMathHelpers:
    def test_mae_rmse_bias_known_values(self) -> None:
        actual = np.array([10.0, 20.0])
        predicted = np.array([12.0, 18.0])
        assert _mae(actual, predicted) == pytest.approx(2.0)
        assert _rmse(actual, predicted) == pytest.approx(2.0)
        # Signed error predicted - actual is +2 and -2, mean 0.
        assert _bias(actual, predicted) == pytest.approx(0.0)

    def test_bias_sign_positive_when_over_predicting(self) -> None:
        assert _bias(np.array([1.0, 1.0]), np.array([3.0, 4.0])) == pytest.approx(2.5)

    def test_crps_zero_for_perfect_point_samples(self) -> None:
        actual = np.array([5.0, 5.0, 5.0])
        samples = np.full((3, 4), 5.0)
        assert _crps_empirical(actual, posterior_samples=samples) == pytest.approx(0.0)

    def test_crps_constant_forecast_equals_absolute_error(self) -> None:
        # With every draw equal to c, E|X-y| = |c-y| and E|X-X'| = 0, so CRPS = |c-y|.
        actual = np.array([10.0])
        samples = np.full((1, 6), 13.0)
        assert _crps_empirical(actual, posterior_samples=samples) == pytest.approx(3.0)

    def test_crps_gaussian_approximation_at_zero_error(self) -> None:
        # For z=0 the Normal CRPS is sigma * (2*phi(0) - 1/sqrt(pi)).
        expected = 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi)
        got = _crps_empirical(np.array([7.0]), predicted=np.array([7.0]), std_proxy=1.0)
        assert got == pytest.approx(expected, rel=1e-9)

    def test_crps_nan_without_samples_or_prediction(self) -> None:
        assert math.isnan(_crps_empirical(np.array([1.0])))

    def test_crps_gaussian_uses_mae_sigma_when_no_proxy(self) -> None:
        actual = np.array([0.0, 0.0])
        predicted = np.array([2.0, -2.0])  # MAE = 2, so sigma = 2
        z = (actual - predicted) / 2.0
        from scipy.special import ndtr

        per_row = 2.0 * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
        assert _crps_empirical(actual, predicted=predicted) == pytest.approx(per_row.mean())

    def test_coverage_counts_actuals_inside_central_interval(self) -> None:
        # sigma=1, level=0.80 -> half-width z=1.2816. Offsets 0 and 1 are inside, 2 and -2 outside.
        predicted = np.zeros(4)
        actual = np.array([0.0, 1.0, 2.0, -2.0])
        assert _coverage_at_threshold(actual, predicted, sigma=1.0, level=0.80) == pytest.approx(0.5)

    def test_coverage_full_for_wide_interval(self) -> None:
        predicted = np.zeros(3)
        actual = np.array([0.5, -0.5, 1.0])
        assert _coverage_at_threshold(actual, predicted, sigma=10.0, level=0.90) == pytest.approx(1.0)


class TestReportMetrics:
    def test_to_dict_rounds_and_nulls_nan_crps(self) -> None:
        m = ReportMetrics(
            position="WR", stat="receiving_yards", n_rows=4, mae=2.123456, rmse=3.0,
            bias=-0.0004, crps=float("nan"), coverage_50=0.5, coverage_80=0.75, coverage_90=1.0,
        )
        d = m.to_dict()
        assert d["mae"] == 2.123
        assert d["bias"] == -0.0
        assert d["crps"] is None
        assert d["coverage_80"] == 0.75

    def test_to_dict_keeps_finite_crps(self) -> None:
        m = ReportMetrics("RB", "rushing_yards", 2, 1.0, 1.0, 0.0, 0.123456, 0.5, 0.8, 0.9)
        assert m.to_dict()["crps"] == pytest.approx(0.123)


class TestLoadOofs:
    def test_loads_csvs_and_infers_stat_from_filename(self, tmp_path) -> None:
        (tmp_path / "receiving_yards_xgb_oof.csv").write_text(
            "player_id,season,week,actual,predicted\nA,2024,1,10,12\n"
        )
        (tmp_path / "fantasy_ppr_lgbm.csv").write_text(
            "player_id,season,week,actual,predicted\nB,2024,2,8,6\n"
        )
        df = BacktestReport().load_oofs(str(tmp_path))._oof_df
        assert len(df) == 2
        assert sorted(df["stat"]) == ["fantasy_ppr", "receiving_yards"]

    def test_explicit_stat_column_is_kept(self, tmp_path) -> None:
        (tmp_path / "anything.csv").write_text(
            "player_id,season,week,stat,actual,predicted\nA,2024,1,targets,5,4\n"
        )
        df = BacktestReport().load_oofs(str(tmp_path))._oof_df
        assert list(df["stat"]) == ["targets"]

    def test_unreadable_file_is_skipped(self, tmp_path, caplog) -> None:
        (tmp_path / "receiving_yards_bad.csv").write_text("")  # empty file -> EmptyDataError
        (tmp_path / "rushing_yards_ok.csv").write_text(
            "player_id,season,week,actual,predicted\nA,2024,1,3,4\n"
        )
        with caplog.at_level("WARNING", logger=br.logger.name):
            df = BacktestReport().load_oofs(str(tmp_path))._oof_df
        assert list(df["stat"]) == ["rushing_yards"]
        assert "Failed to load" in caplog.text

    def test_empty_directory_gives_empty_frame(self, tmp_path) -> None:
        report = BacktestReport().load_oofs(str(tmp_path))
        assert report._oof_df.empty
        assert report.compute_metrics() == []


class TestComputeMetrics:
    def test_per_position_stat_metrics_on_known_frame(self) -> None:
        report = BacktestReport()
        report._oof_df = _oof([
            {"player_id": "A", "season": 2024, "week": 1, "stat": "receiving_yards",
             "position": "WR", "actual": 10.0, "predicted": 12.0},
            {"player_id": "B", "season": 2024, "week": 1, "stat": "receiving_yards",
             "position": "WR", "actual": 20.0, "predicted": 18.0},
            {"player_id": "C", "season": 2024, "week": 1, "stat": "rushing_yards",
             "position": "RB", "actual": 50.0, "predicted": 50.0},
        ])
        metrics = report.compute_metrics()
        assert [(m.stat, m.position) for m in metrics] == [
            ("receiving_yards", "WR"), ("rushing_yards", "RB"),
        ]
        wr = metrics[0]
        assert wr.n_rows == 2
        assert wr.mae == pytest.approx(2.0)
        assert wr.rmse == pytest.approx(2.0)
        assert wr.bias == pytest.approx(0.0)
        # sigma = max(MAE, 1) = 2, so the 80% half-width is 1.2816 * 2 = 2.563; both errors (+/-2) fall inside.
        assert wr.coverage_80 == pytest.approx(1.0)
        assert wr.crps > 0

        rb = metrics[1]
        assert rb.mae == pytest.approx(0.0)
        assert rb.n_rows == 1

    def test_missing_position_column_uses_all_label(self) -> None:
        report = BacktestReport()
        report._oof_df = _oof([
            {"player_id": "A", "stat": "targets", "actual": 6.0, "predicted": 5.0},
            {"player_id": "B", "stat": "targets", "actual": 4.0, "predicted": 5.0},
        ])
        (only,) = report.compute_metrics()
        assert only.position == "ALL"
        assert only.stat == "targets"
        assert only.mae == pytest.approx(1.0)
        assert only.bias == pytest.approx(0.0)

    def test_rows_with_non_numeric_values_are_dropped(self) -> None:
        report = BacktestReport()
        report._oof_df = _oof([
            {"player_id": "A", "stat": "targets", "actual": 6.0, "predicted": 5.0},
            {"player_id": "B", "stat": "targets", "actual": "n/a", "predicted": 5.0},
            {"player_id": "C", "stat": "targets", "actual": 4.0, "predicted": 5.0},
        ])
        (only,) = report.compute_metrics()
        assert only.n_rows == 2
        assert only.mae == pytest.approx(1.0)

    def test_no_data_returns_empty_list(self) -> None:
        report = BacktestReport()
        assert report.compute_metrics() == []
        report._oof_df = pd.DataFrame()
        assert report.compute_metrics() == []


class TestGenerate:
    def _write_oofs(self, root):
        oof_dir = root / "oof"
        oof_dir.mkdir()
        rows = []
        for week in (1, 2, 3):
            rows.append(f"A{week},2024,{week},receiving_yards,WR,{10 + week},{12 + week}")
            rows.append(f"B{week},2024,{week},receiving_yards,WR,{20 + week},{18 + week}")
        (oof_dir / "receiving_yards_xgb.csv").write_text(
            "player_id,season,week,stat,position,actual,predicted\n" + "\n".join(rows) + "\n"
        )
        return oof_dir

    def test_generate_writes_summary_html_and_plot(self, tmp_path) -> None:
        oof_dir = self._write_oofs(tmp_path)
        out_dir = tmp_path / "reports"
        html_path = BacktestReport().generate(str(oof_dir), str(out_dir), title="Test Run")

        assert html_path == out_dir / "report.html"
        summary = pd.read_csv(out_dir / "summary.csv")
        assert list(summary["stat"]) == ["receiving_yards"]
        assert summary.loc[0, "n_rows"] == 6
        assert summary.loc[0, "mae"] == pytest.approx(2.0)
        assert summary.loc[0, "bias"] == pytest.approx(0.0)

        html = html_path.read_text(encoding="utf-8")
        assert "<title>Test Run</title>" in html
        assert "<td>receiving_yards</td>" in html
        assert "Metric Definitions" in html
        assert (out_dir / "mae_by_week.png").exists()

    def test_generate_with_no_metrics_returns_out_dir_and_writes_nothing(self, tmp_path) -> None:
        out_dir = tmp_path / "reports"
        result = BacktestReport().generate(str(tmp_path / "missing"), str(out_dir))
        assert result == out_dir
        assert not out_dir.exists()

    def test_build_html_contains_one_row_per_metric(self) -> None:
        metrics = [
            ReportMetrics("WR", "receiving_yards", 10, 2.0, 2.5, 0.5, 1.25, 0.5, 0.8, 0.9),
            ReportMetrics("RB", "rushing_yards", 4, 3.0, 3.5, -1.0, 2.0, 0.4, 0.7, 0.95),
        ]
        summary = pd.DataFrame([m.to_dict() for m in metrics])
        html = BacktestReport()._build_html(metrics, summary, "Title")
        assert html.count("<tr><td>") == 2
        assert "<td>+0.50</td>" in html
        assert "<td>80.0%</td>" in html
        assert "Rows: 14" in html


class TestMaeByWeekPlot:
    def test_skipped_without_week_column(self, tmp_path) -> None:
        report = BacktestReport()
        report._oof_df = pd.DataFrame({"actual": [1.0], "predicted": [2.0]})
        report._plot_mae_by_week(str(tmp_path))
        assert not (tmp_path / "mae_by_week.png").exists()

    def test_skipped_when_no_frame_loaded(self, tmp_path) -> None:
        report = BacktestReport()
        report._plot_mae_by_week(str(tmp_path))
        assert not (tmp_path / "mae_by_week.png").exists()

    def test_writes_png_for_weekly_frame(self, tmp_path) -> None:
        report = BacktestReport()
        report._oof_df = pd.DataFrame({
            "week": [1, 1, 2, 2],
            "actual": [10.0, 12.0, 5.0, 9.0],
            "predicted": [11.0, 10.0, 5.0, 7.0],
        })
        report._plot_mae_by_week(str(tmp_path))
        png = tmp_path / "mae_by_week.png"
        assert png.exists()
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
