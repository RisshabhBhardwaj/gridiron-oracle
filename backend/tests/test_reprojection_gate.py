"""Wave-04 promotion gate: prior-release evidence or no promotion."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "reprojection_gate", ROOT / "scripts" / "reprojection_gate.py"
)
assert _SPEC and _SPEC.loader
gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(gate)


def _candidate(path: Path, prediction: float) -> Path:
    pd.DataFrame({
        "season": [2024], "y_true": [10.0], "y_pred": [prediction],
    }).to_csv(path, index=False)
    return path


def _baseline(path: Path, *, mae: float, digest: str = "not-the-candidate") -> Path:
    path.write_text(json.dumps({
        "frozen_baseline": {
            "fantasy_ppr:WR:2024": {
                "mae": mae, "oof_sha256": digest, "release_id": "prior-promoted-release",
            }
        }
    }))
    return path


def test_gate_rejects_regressed_candidate_against_distinct_prior_release(tmp_path: Path) -> None:
    report = gate.run_gate(
        holdout_season=2024,
        position="WR",
        target="fantasy_ppr",
        candidate_oof=_candidate(tmp_path / "candidate.csv", 20.0),
        baseline_manifest=_baseline(tmp_path / "baseline.json", mae=1.0),
    )
    assert report["promote"] is False
    assert report["checks"][0]["name"] == "vs_frozen_prior_release"


def test_gate_rejects_five_percent_worse_mae(tmp_path: Path) -> None:
    report = gate.run_gate(
        holdout_season=2024,
        position="WR",
        target="fantasy_ppr",
        candidate_oof=_candidate(tmp_path / "candidate.csv", 10.6),
        baseline_manifest=_baseline(tmp_path / "baseline.json", mae=0.5),
    )
    assert report["promote"] is False


def test_first_release_requires_human_override(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")
    with pytest.raises(ValueError, match="no frozen_baseline block"):
        gate.run_gate(
            holdout_season=2024,
            position="WR",
            target="fantasy_ppr",
            candidate_oof=_candidate(tmp_path / "candidate.csv", 10.0),
            baseline_manifest=legacy,
        )
    report = gate.run_gate(
        holdout_season=2024,
        position="WR",
        target="fantasy_ppr",
        candidate_oof=_candidate(tmp_path / "candidate.csv", 10.0),
        baseline_manifest=legacy,
        approval=gate.FIRST_RELEASE_OVERRIDE,
    )
    assert report["promote"] is True


def test_gate_dsr_can_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Deflated Sharpe"):
        gate.run_gate(
            holdout_season=2024,
            position="WR",
            target="fantasy_ppr",
            candidate_oof=_candidate(tmp_path / "candidate.csv", 10.0),
            baseline_manifest=_baseline(tmp_path / "baseline.json", mae=10.0),
            n_trials=500,
            observed_sharpe=0.1,
            n_observations=20,
        )


def test_gate_fails_closed_without_frozen_prior_release_evidence(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}")
    with pytest.raises(ValueError, match="no frozen_baseline block"):
        gate.run_gate(
            holdout_season=2024,
            position="WR",
            target="fantasy_ppr",
            candidate_oof=_candidate(tmp_path / "candidate.csv", 10.0),
            baseline_manifest=legacy,
        )


def test_gate_refuses_candidate_self_comparison(tmp_path: Path) -> None:
    candidate = _candidate(tmp_path / "candidate.csv", 10.0)
    from ml.artifact_manifest import sha256_of

    with pytest.raises(ValueError, match="self-comparison"):
        gate.run_gate(
            holdout_season=2024,
            position="WR",
            target="fantasy_ppr",
            candidate_oof=candidate,
            baseline_manifest=_baseline(tmp_path / "baseline.json", mae=1.0, digest=sha256_of(candidate)),
        )
