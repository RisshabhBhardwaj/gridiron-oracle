"""PRODUCT_MODE default and fail-closed parsing (ml-01, part A).

artifact_backed is the default in every consumer (backend config, ml/, pipeline/,
scripts/). An unknown value raises instead of silently becoming graceful_fallback.
graceful_fallback remains available, but only when it is set explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- backend config -----------------------------------------------------------


def test_settings_default_product_mode_is_artifact_backed(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.delenv("PRODUCT_MODE", raising=False)
    assert Settings().product_mode == "artifact_backed"


def test_settings_unknown_product_mode_raises(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv("PRODUCT_MODE", "artifact-backed")  # typo: hyphen, not underscore
    with pytest.raises(ValueError, match="PRODUCT_MODE"):
        Settings()


def test_settings_explicit_graceful_fallback_is_honoured(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv("PRODUCT_MODE", " Graceful_Fallback ")
    assert Settings().product_mode == "graceful_fallback"


def test_settings_blank_product_mode_means_default(monkeypatch):
    from backend.app.core.config import Settings

    monkeypatch.setenv("PRODUCT_MODE", "   ")
    assert Settings().product_mode == "artifact_backed"


# --- shared resolver (ml/product_mode.py) -------------------------------------


def test_resolve_product_mode_default_is_artifact_backed():
    from ml.product_mode import resolve_product_mode

    assert resolve_product_mode({}) == "artifact_backed"


def test_resolve_product_mode_rejects_unknown_value():
    from ml.product_mode import resolve_product_mode

    with pytest.raises(ValueError, match="PRODUCT_MODE"):
        resolve_product_mode({"PRODUCT_MODE": "artifact-backed"})


def test_resolve_product_mode_honours_explicit_graceful_fallback():
    from ml.product_mode import resolve_product_mode

    assert resolve_product_mode({"PRODUCT_MODE": " Graceful_Fallback "}) == "graceful_fallback"


def test_resolve_product_mode_blank_means_default():
    from ml.product_mode import resolve_product_mode

    assert resolve_product_mode({"PRODUCT_MODE": "  "}) == "artifact_backed"


# --- ml pipeline sites fail closed by default ---------------------------------


def test_stacking_without_mlflow_fails_closed_by_default(monkeypatch):
    from ml.train import PipelineRunner

    monkeypatch.delenv("PRODUCT_MODE", raising=False)
    runner = PipelineRunner()  # mlflow_tracking_uri defaults to ""
    df = pd.DataFrame({"player_id": ["p1"], "kalman_est_rushing_yards": [15.0]})
    with pytest.raises(RuntimeError, match="requires MLFLOW_TRACKING_URI"):
        runner._run_stacking_step(df, "rushing_yards", "RB")


def test_stacking_without_mlflow_explicit_graceful_fallback_uses_kalman(monkeypatch):
    from ml.train import PipelineRunner

    monkeypatch.setenv("PRODUCT_MODE", "graceful_fallback")
    runner = PipelineRunner()
    df = pd.DataFrame({"player_id": ["p1"], "kalman_est_rushing_yards": [15.0]})
    assert list(runner._run_stacking_step(df, "rushing_yards", "RB")) == [15.0]


def _no_pinned_manifest():
    from ml.artifact_manifest import ManifestError

    raise ManifestError("no pinned stack for this cell")


def test_conformal_yardage_without_pinned_oof_fails_closed_by_default(monkeypatch):
    from ml import train as train_mod

    monkeypatch.delenv("PRODUCT_MODE", raising=False)
    monkeypatch.setattr(train_mod, "get_manifest", _no_pinned_manifest)
    runner = train_mod.PipelineRunner()
    mc_df = pd.DataFrame({"projection": [1.0]}, index=pd.Index(["p1"], name="player_id"))
    with pytest.raises(RuntimeError, match="No stacked OOF residuals"):
        runner._apply_conformal_yardage_intervals(
            mc_df, np.array([10.0]), stat="rushing_yards", position="RB",
        )


def test_conformal_yardage_explicit_graceful_fallback_keeps_bayesian_intervals(monkeypatch):
    from ml import train as train_mod

    monkeypatch.setenv("PRODUCT_MODE", "graceful_fallback")
    monkeypatch.setattr(train_mod, "get_manifest", _no_pinned_manifest)
    runner = train_mod.PipelineRunner()
    mc_df = pd.DataFrame({"projection": [1.0]}, index=pd.Index(["p1"], name="player_id"))
    out = runner._apply_conformal_yardage_intervals(
        mc_df, np.array([10.0]), stat="rushing_yards", position="RB",
    )
    assert out is mc_df


# --- scripts ------------------------------------------------------------------


def test_verify_artifact_prerequisites_defaults_to_artifact_backed(monkeypatch, capsys):
    import scripts.verify_artifact_prerequisites as vap

    monkeypatch.delenv("PRODUCT_MODE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    # Default is artifact_backed, so the gate runs and blocks without a database
    # instead of printing "skipped" and exiting 0.
    assert vap.main() == 1
    assert json.loads(capsys.readouterr().out)["overall_status"] == "blocked"


# --- static guard: no default anywhere in ml/, pipeline/, scripts/ -------------


def test_no_ml_pipeline_or_script_defaults_to_graceful_fallback():
    offenders: list[str] = []
    for root in ("ml", "pipeline", "scripts"):
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if "PRODUCT_MODE" in line and "graceful_fallback" in line:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")
    assert offenders == [], "graceful_fallback must not be a default: " + ", ".join(offenders)
