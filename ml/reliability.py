"""Reproducibility, promotion, and interval-calibration contracts for ML runs.

This module intentionally has no database or model-library dependency.  It is
used by trainers, reports, and release automation to make the evidence behind a
projection immutable and reviewable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


YARDAGE_STATS = frozenset({"passing_yards", "rushing_yards", "receiving_yards"})


def sha256_json(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible evidence."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataframe_hash(df: pd.DataFrame, columns: Iterable[str]) -> str:
    """Hash an ordered, explicit dataframe slice without serialising its index."""
    cols = list(columns)
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"Cannot hash missing dataframe columns: {missing}")
    payload = df.loc[:, cols].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TrainingManifest:
    """Immutable provenance for one target/position training cohort."""

    created_at: str
    target: str
    position: str
    seasons: list[int]
    row_count: int
    feature_columns: list[str]
    feature_schema_hash: str
    training_data_hash: str
    target_summary: dict[str, float]
    null_rates: dict[str, float]
    source_contract_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_training_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    position: str,
    target_col: str,
    feature_columns: list[str],
    source_contract: dict[str, Any],
) -> TrainingManifest:
    """Create a manifest after target/position filtering and before fitting."""
    if df.empty:
        raise ValueError("Cannot create a training manifest for an empty cohort")
    target_values = pd.to_numeric(df[target_col], errors="coerce")
    data_columns = ["player_id", "game_id", "season", "week", target_col, *feature_columns]
    present = [col for col in data_columns if col in df.columns]
    return TrainingManifest(
        created_at=datetime.now(timezone.utc).isoformat(),
        target=target,
        position=position,
        seasons=sorted(int(x) for x in df["season"].dropna().unique()),
        row_count=int(len(df)),
        feature_columns=feature_columns,
        feature_schema_hash=sha256_json(feature_columns),
        training_data_hash=dataframe_hash(df, present),
        target_summary={
            "min": float(target_values.min()),
            "max": float(target_values.max()),
            "mean": float(target_values.mean()),
            "std": float(target_values.std(ddof=0)),
            "zero_rate": float((target_values <= 0).mean()),
        },
        null_rates={col: float(df[col].isna().mean()) for col in feature_columns},
        source_contract_hash=sha256_json(source_contract),
    )


def write_manifest(manifest: TrainingManifest, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def promotion_gate(
    *,
    candidate_mae: float,
    incumbent_mae: float | None,
    min_improvement_pct: float = 0.0,
) -> tuple[bool, str]:
    """Make stat-specific promotion impossible without held-out improvement."""
    if not np.isfinite(candidate_mae) or candidate_mae < 0:
        return False, "candidate MAE is not finite"
    if incumbent_mae is None or not np.isfinite(incumbent_mae):
        return False, "no valid incumbent held-out MAE is available"
    if incumbent_mae <= 0:
        return False, "incumbent MAE must be positive"
    improvement = (incumbent_mae - candidate_mae) / incumbent_mae * 100.0
    if improvement <= min_improvement_pct:
        return False, f"held-out improvement {improvement:.2f}% does not exceed {min_improvement_pct:.2f}%"
    return True, f"held-out improvement {improvement:.2f}% exceeds {min_improvement_pct:.2f}%"


@dataclass(frozen=True)
class SplitConformalInterval:
    lower: np.ndarray
    upper: np.ndarray
    radius: float
    coverage: float


def split_conformal_interval(
    calibration_actual: np.ndarray,
    calibration_prediction: np.ndarray,
    prediction: np.ndarray,
    *,
    coverage: float = 0.80,
    lower_bound: float = 0.0,
) -> SplitConformalInterval:
    """Distribution-free symmetric interval using strictly held-out residuals.

    The caller must supply out-of-fold predictions only.  This avoids using a
    model's in-sample residuals to make its uncertainty appear narrower.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one")
    actual = np.asarray(calibration_actual, dtype=float)
    fitted = np.asarray(calibration_prediction, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(fitted)
    if mask.sum() < 20:
        raise ValueError("split conformal calibration requires at least 20 finite OOF residuals")
    residuals = np.abs(actual[mask] - fitted[mask])
    rank = int(np.ceil((len(residuals) + 1) * coverage))
    radius = float(np.partition(residuals, min(rank - 1, len(residuals) - 1))[min(rank - 1, len(residuals) - 1)])
    return SplitConformalInterval(
        lower=np.maximum(lower_bound, pred - radius),
        upper=np.maximum(lower_bound, pred + radius),
        radius=radius,
        coverage=coverage,
    )
