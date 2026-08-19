"""Regression guards for C-04, C-09, C-10, and C-11 serving contracts."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_materializer_declares_the_complete_serving_matrix():
    from ml.artifact_manifest import REQUIRED_SERVING_CELLS
    from scripts.materialize_stack_projections import CELLS

    assert tuple(CELLS) == REQUIRED_SERVING_CELLS
    assert len(CELLS) == 15


def test_materializer_uses_prior_seasons_for_conformal_bounds_not_percentiles(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_rebuilt.csv"
    pd.DataFrame({
        "player_id": [f"p{i}" for i in range(102)],
        "game_id": [f"g{i}" for i in range(102)],
        "season": [2024] * 100 + [2025, 2025],
        "week": [1] * 102,
        "y_pred": [50.0] * 50 + [100.0] * 50 + [50.0, 100.0],
        "y_true": [40.0] * 50 + [130.0] * 50 + [0.0, 0.0],
        "max_train_season": [2023] * 100 + [2024, 2024],
    }).to_csv(artifact, index=False)
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    frame = materializer._load_cell("receiving_yards", "WR")

    # First season has no prior calibration history; its bounds are absent.
    assert frame.loc[frame["season"] == 2024, "floor"].isna().all()
    # 2025 values use 2024 residuals and differ by prediction bucket.  Their
    # own y_true values are deliberately nonsensical, proving they were not
    # used to form their interval.
    current = frame.loc[frame["season"] == 2025]
    assert current["floor"].notna().all()
    assert current["ceiling"].notna().all()
    assert current.iloc[0]["floor"] != current.iloc[1]["floor"]
    assert frame["p25"].isna().all()
    assert frame["p75"].isna().all()
    assert set(current["interval_method"]) == {"causal_oof_conformal_90"}


def test_materializer_requires_artifact_provenance(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_rebuilt.csv"
    pd.DataFrame({
        "player_id": ["p1"], "game_id": ["g1"], "season": [2025],
        "week": [1], "y_pred": [50.0],
    }).to_csv(artifact, index=False)
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    try:
        materializer._load_cell("receiving_yards", "WR")
    except ValueError as exc:
        assert "max_train_season" in str(exc)
    else:
        raise AssertionError("materializer accepted a stack without artifact provenance")


def test_materializer_refuses_pre_02_stack_names(tmp_path, monkeypatch):
    from scripts import materialize_stack_projections as materializer

    artifact = tmp_path / "stack_receiving_yards_WR_20260809.csv"
    artifact.write_text("")
    monkeypatch.setattr(materializer, "_pinned_stack", lambda *_: artifact)

    try:
        materializer._load_cell("receiving_yards", "WR")
    except ValueError as exc:
        assert "pre-02 legacy stack" in str(exc)
    else:
        raise AssertionError("materializer accepted a pre-02 stack")


def test_materializer_conflict_update_clears_stale_uncertainty_fields():
    from scripts.materialize_stack_projections import _upsert
    import inspect

    sql_source = inspect.getsource(_upsert)
    assert "boom_probability = EXCLUDED.boom_probability" in sql_source
    assert "bust_probability = EXCLUDED.bust_probability" in sql_source
    assert "posterior_samples = EXCLUDED.posterior_samples" in sql_source


def test_materialization_report_distinguishes_conformal_coverage_from_posterior_percentiles():
    from scripts import materialize_stack_projections as materializer
    import inspect

    source = inspect.getsource(materializer.main)
    assert '"causal_oof_conformal_90"' in source
    assert '"posterior_percentiles_materialized": False' in source


def _clean_manifest_copy(tmp_path):
    payload = json.loads(Path("releases/current_baseline.json").read_text())
    payload["git_dirty"] = False
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(payload))
    return path


def test_readiness_rejects_a_manifest_at_a_different_commit(tmp_path):
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=["not-the-manifest-head\n", ""],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "does not match HEAD" in result["detail"]


def test_readiness_rejects_a_dirty_runtime_worktree(tmp_path):
    from backend.app.core.config import settings
    from backend.app.services.runtime_status import RuntimeStatusService

    service = RuntimeStatusService("postgresql://unused", "http://unused", "latest")
    manifest = _clean_manifest_copy(tmp_path)
    with patch.object(settings, "baseline_manifest_path", str(manifest)), patch(
        "backend.app.services.runtime_status.subprocess.check_output",
        side_effect=["a4fb0e96995de1de943842afaa196777e0a194cb\n", " M ml/train.py\n"],
    ):
        result = service._check_baseline_manifest()

    assert result["status"] == "error"
    assert "worktree is dirty" in result["detail"]
