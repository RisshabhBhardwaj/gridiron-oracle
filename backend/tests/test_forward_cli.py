"""Decision 0012 commit 6: PipelineRunner leaves artifact_backed serving, the CLI is thin.

In ``artifact_backed`` mode the old MLflow/ONNX path must not run at all; it points at
``ml.forward_serving``. ``scripts/materialize_forward_projections.py`` only opens a
connection and calls ``run_forward``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from unittest.mock import patch

import pandas as pd
import pytest

from backend.tests._fake_db import FakeConn

POINTER = "ml.forward_serving"


@pytest.fixture
def artifact_backed(monkeypatch) -> None:
    monkeypatch.setenv("PRODUCT_MODE", "artifact_backed")


def test_run_in_artifact_backed_stops_before_any_io_and_names_the_producer(artifact_backed) -> None:
    from ml.train import PipelineRunner

    runner = PipelineRunner(mlflow_tracking_uri="http://fake:9999")
    with patch.object(runner, "_load_from_db") as load, patch.object(runner, "_log_mlflow") as log:
        with pytest.raises(RuntimeError, match=POINTER):
            runner.run(2026, 11, ["WR"], ["receiving_yards"])
    load.assert_not_called()
    log.assert_not_called()


def test_stacking_step_in_artifact_backed_never_reaches_mlflow_or_onnx(artifact_backed) -> None:
    from ml.train import PipelineRunner

    runner = PipelineRunner(mlflow_tracking_uri="http://fake:9999")
    df = pd.DataFrame({"player_id": ["p1"], "kalman_est_rushing_yards": [15.0]})
    with patch.object(runner, "_load_and_run_stacking") as stack, patch("urllib.request.urlopen") as ping:
        with pytest.raises(RuntimeError, match=POINTER):
            runner._run_stacking_step(df, "rushing_yards", "RB")
    stack.assert_not_called()
    ping.assert_not_called()


def test_graceful_fallback_mode_and_dry_runs_are_unchanged(monkeypatch) -> None:
    from ml.train import PipelineRunner

    monkeypatch.setenv("PRODUCT_MODE", "graceful_fallback")
    df = pd.DataFrame({"player_id": ["p1"], "kalman_est_rushing_yards": [15.0]})
    assert list(PipelineRunner()._run_stacking_step(df, "rushing_yards", "RB")) == [15.0]
    monkeypatch.setenv("PRODUCT_MODE", "artifact_backed")
    assert not PipelineRunner().dry_run(2026, 11, ["RB"], ["rushing_yards"]).empty


# ── the CLI ───────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def cli(monkeypatch):
    import scripts.materialize_forward_projections as cli

    conn = FakeConn()
    monkeypatch.setattr(cli.psycopg2, "connect", lambda *a, **k: conn)
    cli.fake_conn = conn
    return cli


def test_cli_passes_its_arguments_to_run_forward_and_prints_the_report(cli, capsys, monkeypatch) -> None:
    seen = {}

    def fake_run(conn, season, week, **kw):
        seen.update(conn=conn, season=season, week=week, **kw)
        return {"projection_rows": 3, "dry_run": kw["dry_run"]}

    monkeypatch.setattr(cli, "run_forward", fake_run)
    code = cli.main(["--season", "2026", "--week", "11", "--feature-run-id", "fwd_1",
                     "--database-url", "postgresql://x/y", "--dry-run"])
    assert code == 0
    assert seen["conn"] is cli.fake_conn
    assert (seen["season"], seen["week"], seen["feature_run_id"], seen["dry_run"]) == (2026, 11, "fwd_1", True)
    assert seen["allow_no_availability"] is False
    assert json.loads(capsys.readouterr().out)["projection_rows"] == 3


def test_cli_refuses_a_guard_failure_with_exit_1_and_the_reason(cli, capsys, monkeypatch) -> None:
    from ml.forward_serving import ForwardGuardError

    def refuse(*a, **k):
        raise ForwardGuardError("G2", "week 11 already has scores")

    monkeypatch.setattr(cli, "run_forward", refuse)
    code = cli.main(["--season", "2026", "--week", "11", "--database-url", "postgresql://x/y"])
    assert code == 1
    assert "[G2] week 11 already has scores" in capsys.readouterr().err


def test_cli_in_graceful_fallback_mode_is_refused_by_g1_before_any_query(cli, capsys, monkeypatch) -> None:
    monkeypatch.setenv("PRODUCT_MODE", "graceful_fallback")
    code = cli.main(["--season", "2026", "--week", "11", "--database-url", "postgresql://x/y"])
    assert code == 1
    assert cli.fake_conn.log == []
    assert "artifact_backed" in capsys.readouterr().err


def test_cli_imports_neither_the_pipeline_runner_nor_the_tracking_stack() -> None:
    code = (
        "import sys, scripts.materialize_forward_projections\n"
        "bad = [m for m in ('ml.train','mlflow','pymc','torch','onnxruntime','ml.monte_carlo') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
