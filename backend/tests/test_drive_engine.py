from pathlib import Path

import pytest

from ml.drive_engine import require_transitions, transitions_artifact_path
from ml.bdb_tracking import load_tracking_week, tracking_available


def test_drive_mcmc_fails_closed_without_transitions(tmp_path: Path) -> None:
    missing = tmp_path / "transitions.csv"
    with pytest.raises(FileNotFoundError):
        require_transitions(missing)


def test_default_transitions_path_is_oof_artifact() -> None:
    assert transitions_artifact_path().name == "transitions.csv"
    assert "ml/oof" in str(transitions_artifact_path())


def test_bdb_tracking_fails_closed(tmp_path: Path) -> None:
    assert tracking_available(tmp_path) is False
    with pytest.raises(FileNotFoundError):
        load_tracking_week(tmp_path / "tracking_week_1.csv")
