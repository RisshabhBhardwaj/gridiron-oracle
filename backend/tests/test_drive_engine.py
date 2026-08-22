from pathlib import Path

import pytest

from ml.bdb_tracking import load_tracking_week, tracking_available
from ml.drive_engine import load_drive_engine, require_transitions, transitions_artifact_path


def test_drive_mcmc_fails_closed_without_transitions(tmp_path: Path) -> None:
    missing = tmp_path / "transitions_by_game_state.csv"
    with pytest.raises(FileNotFoundError):
        require_transitions(missing)


def test_default_transitions_path_is_game_state_oof_artifact() -> None:
    """The C++ engine now consumes the 5D (fp, down, ytg, score, quarter) table."""
    assert transitions_artifact_path().name == "transitions_by_game_state.csv"
    assert "ml/oof" in str(transitions_artifact_path())


def test_bdb_tracking_fails_closed(tmp_path: Path) -> None:
    assert tracking_available(tmp_path) is False
    with pytest.raises(FileNotFoundError):
        load_tracking_week(tmp_path / "tracking_week_1.csv")


class TestDriveMCMCGameState:
    """
    Exercises the real compiled C++ engine (Phase 5) end to end, skipping if
    the shared library or the fitted artifact isn't present locally rather
    than failing — CI/dev environments may not always have engine/build/ or
    a materialized transitions_by_game_state.csv.
    """

    @pytest.fixture
    def engine(self):
        try:
            return load_drive_engine()
        except (FileNotFoundError, RuntimeError) as exc:
            pytest.skip(f"DriveMCMC engine unavailable: {exc}")

    def test_leading_team_passes_less_than_trailing_team_in_q4(self, engine) -> None:
        """
        The plan's Phase 5 acceptance criterion: leading teams run more
        (pass less) in Q4, reproduced through the compiled engine loaded
        with real fitted transitions — not just the Python-side fit.
        """
        leading = engine.simulate(field_pos=50, down=1, yards_to_go=10, score_differential=20, quarter=4)
        trailing = engine.simulate(field_pos=50, down=1, yards_to_go=10, score_differential=-20, quarter=4)
        assert leading["expected_pass_rate"] < trailing["expected_pass_rate"] - 0.15

    def test_simulate_outputs_are_valid_probabilities(self, engine) -> None:
        res = engine.simulate(field_pos=25, down=1, yards_to_go=10, score_differential=0, quarter=1)
        assert 0.0 <= res["expected_pass_rate"] <= 1.0
        assert 0.0 <= res["p_touchdown"] <= 1.0
        assert 0.0 <= res["p_field_goal"] <= 1.0
