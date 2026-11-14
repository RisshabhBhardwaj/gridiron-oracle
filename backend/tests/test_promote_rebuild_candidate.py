"""scripts/promote_rebuild_candidate.py: complete rebuild -> pinned, loadable release."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.artifact_manifest import REQUIRED_SERVING_CELLS, load_manifest  # noqa: E402
from ml.bucket_attribution import ATTR_COLUMNS, write_attribution_file  # noqa: E402
from scripts import promote_rebuild_candidate as prc  # noqa: E402


def _write_cell(directory: Path, stat: str, position: str, *, residual: float = 0.0) -> None:
    stack = pd.DataFrame({
        "player_id": ["a", "b", "c"],
        "game_id": ["g1", "g1", "g2"],
        "season": [2024] * 3,
        "week": [1, 1, 2],
        "y_true": [1.0, 2.0, 3.0],
        "y_pred": [1.5, 2.5, 3.5],
        "lgbm_pred": [1.4, 2.4, 3.4],
        "max_train_season": [2023] * 3,
    })
    path = directory / f"stack_{stat}_{position}_abc123.csv"
    stack.to_csv(path, index=False)
    attr = stack[["player_id", "game_id", "season", "week", "y_pred"]].copy()
    for column in ATTR_COLUMNS:
        attr[column] = 0.0
    attr["base"] = stack["y_pred"].to_numpy() - residual
    write_attribution_file(attr, path)
    (directory / f"ridge_{stat}_{position}_coefs.json").write_text(
        json.dumps({"learner_order": ["lgbm", "catboost"], "weights": {"lgbm": 0.5, "catboost": 0.5}, "intercept": 0.0})
    )


def _rebuild(tmp_path: Path, *, skip: tuple | None = None, bad: tuple | None = None) -> Path:
    directory = tmp_path / "rebuild"
    directory.mkdir()
    for stat, position in sorted(REQUIRED_SERVING_CELLS):
        if (stat, position) == skip:
            continue
        _write_cell(directory, stat, position, residual=0.5 if (stat, position) == bad else 0.0)
    (directory / prc.SELECTION_FILE).write_text("{}\n")
    (directory / prc.REFERENCE_DIR).mkdir()
    (directory / prc.REFERENCE_DIR / "lgbm_reference_x_WR.csv").write_text("a\n")
    return directory


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "releases").mkdir(parents=True)
    (root / "ml" / "oof").mkdir(parents=True)
    drive = root / "ml" / "oof" / "transitions_by_game_state.csv"
    drive.write_text("x\n")
    (root / "releases" / "current_baseline.json").write_text(json.dumps({
        "artifacts": {
            "stacks": ["releases/candidates/old/stack_x_WR_1.csv"],
            "serving_parity": ["releases/candidates/old/serving_divergence_x_WR.json"],
            "drive_transitions": ["ml/oof/transitions_by_game_state.csv"],
        },
        "artifact_digests": {},
        "artifact_pins_missing": ["x"],
        "projection_policy": {
            "approved_pipeline_run_ids": ["stack_materialize_20260822T144443Z", "weekly_auto_2026"],
            "require_interval_columns": True,
        },
        "known_issues": ["kept"],
    }))
    return root


def test_promote_pins_all_cells_with_repo_relative_digests(tmp_path):
    rebuild, root = _rebuild(tmp_path), _root(tmp_path)
    summary = prc.promote(rebuild, "causal_test", root=root)

    assert summary["cells"] == 32
    payload = json.loads((root / "releases" / "current_baseline.json").read_text())
    art = payload["artifacts"]
    assert len(art["stacks"]) == len(art["ridge_coefficients"]) == len(art["bucket_attributions"]) == 32
    assert "serving_parity" not in art
    assert art["drive_transitions"] == ["ml/oof/transitions_by_game_state.csv"]
    assert all(not k.startswith("/") for k in payload["artifact_digests"])
    assert "artifact_pins_missing" not in payload
    assert payload["known_issues"] == ["kept"]
    assert payload["projection_policy"]["approved_pipeline_run_ids"] == ["weekly_auto_2026"]
    assert summary["freeze_env"].endswith(",weekly_auto_2026")
    assert (root / "releases/candidates/causal_test" / prc.REFERENCE_DIR / "lgbm_reference_x_WR.csv").is_file()

    manifest = load_manifest(root=root)
    assert len(manifest.stack_index) == 32
    stat, position = sorted(REQUIRED_SERVING_CELLS)[0]
    assert manifest.resolve_stack(stat, position).is_file()
    assert manifest.resolve_attribution(stat, position).is_file()


def test_dry_run_writes_nothing(tmp_path):
    rebuild, root = _rebuild(tmp_path), _root(tmp_path)
    before = (root / "releases" / "current_baseline.json").read_text()
    prc.promote(rebuild, "causal_test", root=root, dry_run=True)
    assert (root / "releases" / "current_baseline.json").read_text() == before
    assert not (root / "releases" / "candidates").exists()


def test_missing_cell_refuses(tmp_path):
    cell = sorted(REQUIRED_SERVING_CELLS)[3]
    rebuild, root = _rebuild(tmp_path, skip=cell), _root(tmp_path)
    with pytest.raises(prc.PromotionError, match=cell[0]):
        prc.promote(rebuild, "causal_test", root=root)
    assert not (root / "releases" / "candidates").exists()


def test_attribution_that_does_not_add_up_refuses(tmp_path):
    cell = sorted(REQUIRED_SERVING_CELLS)[0]
    rebuild, root = _rebuild(tmp_path, bad=cell), _root(tmp_path)
    with pytest.raises(prc.PromotionError, match="add up"):
        prc.promote(rebuild, "causal_test", root=root)


def test_existing_candidate_is_never_overwritten(tmp_path):
    rebuild, root = _rebuild(tmp_path), _root(tmp_path)
    (root / "releases" / "candidates" / "causal_test").mkdir(parents=True)
    with pytest.raises(prc.PromotionError, match="already exists"):
        prc.promote(rebuild, "causal_test", root=root)


def test_selection_report_is_required(tmp_path):
    rebuild, root = _rebuild(tmp_path), _root(tmp_path)
    (rebuild / prc.SELECTION_FILE).unlink()
    with pytest.raises(prc.PromotionError, match="select_nonregressing_stacks"):
        prc.promote(rebuild, "causal_test", root=root)


def test_refuses_when_no_non_stack_approved_id_would_remain(tmp_path):
    rebuild, root = _rebuild(tmp_path), _root(tmp_path)
    manifest = root / "releases" / "current_baseline.json"
    payload = json.loads(manifest.read_text())
    payload["projection_policy"]["approved_pipeline_run_ids"] = ["stack_materialize_20260822T144443Z"]
    manifest.write_text(json.dumps(payload))
    with pytest.raises(prc.PromotionError, match="fails closed"):
        prc.promote(rebuild, "causal_test", root=root)
    assert not (root / "releases" / "candidates").exists()
