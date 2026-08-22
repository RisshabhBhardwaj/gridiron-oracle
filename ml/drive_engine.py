"""Wire DriveMarkov transitions into C++ DriveMCMC, fail-closed without the artifact.

Do not size DFS exposure off this engine. Kelly / lock-free exposure code is
unbacktested against real prop odds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
# The C++ DriveMCMC engine (Phase 5) consumes the game-state-aware 5D table
# (fp, down, ytg, score_diff_bucket, quarter) — not the older 3D
# transitions.csv, which load_transitions_from_csv no longer accepts.
TRANSITIONS_PATH = ROOT / "ml" / "oof" / "transitions_by_game_state.csv"


def transitions_artifact_path() -> Path:
    return TRANSITIONS_PATH


def require_transitions(path: Path | None = None) -> Path:
    artifact = path or TRANSITIONS_PATH
    if not artifact.is_file():
        raise FileNotFoundError(
            f"{artifact} is missing. Fit DriveMarkovModel and call "
            "export_transitions_by_game_state() (see "
            "scripts/export_drive_transitions.py) before DriveMCMC."
        )
    return artifact


def load_drive_engine(path: Path | None = None, *, n_simulations: int = 2000):
    """Load C++ DriveMCMC with the fitted CSV. Raises if either piece is missing."""
    artifact = require_transitions(path)
    from engine.python_bindings import LIB, CppDriveMCMC

    if LIB is None:
        raise RuntimeError("C++ engine shared library not found; build engine/ first")
    engine = CppDriveMCMC(n_simulations=n_simulations)
    if not engine.load_transitions_csv(str(artifact)):
        raise RuntimeError(f"DriveMCMC rejected transitions CSV at {artifact}")
    return engine


def copula_preferred_for_correlation(oof: Any) -> bool:
    """True when pairwise residual dependence is large enough to prefer the copula."""
    from ml.copula_eval import independence_overstates_joint, pairwise_residual_correlation

    if oof is None or getattr(oof, "empty", True):
        return False
    players = list(oof["player_id"].astype(str).unique())
    if len(players) < 2:
        return False
    corr = pairwise_residual_correlation(oof, player_a=players[0], player_b=players[1])
    if corr != corr:  # NaN
        return False
    independent = 0.25
    empirical = max(0.0, min(1.0, 0.25 + 0.5 * abs(float(corr))))
    return independence_overstates_joint(0.5, 0.5, empirical)
