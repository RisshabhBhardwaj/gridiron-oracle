"""
backend/app/services/backtest.py

Backtest evidence for the /backtest endpoint and the Backtest and Season Review
pages.

Source: the served release's pinned causal evals (``artifacts.causal_evals`` in
``releases/current_baseline.json``): ``causal_baseline_beats.json`` plus one
``eval_causal_<stat>_<position>.csv`` per cell. Every file is read through
``ml.artifact_manifest`` and checked against its SHA-256. There is no fallback:
a missing pin, a missing file or a digest mismatch raises ``EvalsUnavailable``
and the API answers 503. The older ml-folder CSV and the database results table
are no longer read; they do not describe the served release.

The evals measure, per (stat, position, season), a model score against the
simple method (last game), trailing-3 and Marcel baselines, on a causal cohort.
They do not measure interval coverage, CRPS, Brier, P&L or calibration, so none
of those is returned (see ``NOT_MEASURED``).
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from backend.app.core.runtime_mode import ArtifactRequiredError
from ml.artifact_manifest import ArtifactManifest, ManifestError, get_manifest

logger = logging.getLogger(__name__)

GROUP_KEY = "causal_evals"
BEATS_NAME = "causal_baseline_beats.json"
CSV_PREFIX = "eval_causal_"

#: Plain-words statement of what the evals do not cover. Keys are stable ids.
NOT_MEASURED: dict[str, str] = {
    "coverage": "We don't yet publish how often outcomes land inside our range.",
    "crps": "We don't yet publish how well our full ranges match outcomes.",
    "brier": "We don't yet publish scores for yes/no events.",
    "pnl": "We don't yet publish simulated betting results.",
    "calibration": "We don't yet publish whether our stated chances match what happened.",
}

#: The only message a caller should show a user when the evidence is unavailable.
UNAVAILABLE_MESSAGE = "Backtest results for the current release aren't available right now."


class EvalsUnavailable(ArtifactRequiredError):
    """The pinned evals cannot be read or do not verify. The API answers 503."""


class UnknownStat(ValueError):
    """The release has no eval for the requested stat."""


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalRow:
    stat: str
    position: str
    season: int
    metric: str
    metric_label: str
    n: int
    cohort_rule: str
    model_score: float
    naive_score: float
    trailing3_score: float
    marcel_score: float
    beats_naive: bool
    beats_trailing3: bool
    beats_marcel: bool
    causal_ok: bool

    @property
    def improvement_vs_naive_pct(self) -> Optional[float]:
        """Percent the model score sits below the simple method's (positive = better)."""
        if not self.naive_score:
            return None
        return (self.naive_score - self.model_score) / self.naive_score * 100.0


@dataclass(frozen=True)
class Cell:
    stat: str
    position: str
    metric: str
    metric_label: str


@dataclass(frozen=True)
class Verdict:
    cell_seasons: int
    beat_naive: int
    beat_trailing3: int
    beat_marcel: int
    beat_both: int
    causal_check_passed: int
    mean_improvement_vs_naive_pct: Optional[float]


@dataclass(frozen=True)
class ReleaseEvals:
    release: str
    generated_at: Optional[str]
    rows: tuple[EvalRow, ...]
    cells: tuple[Cell, ...]
    cohort_rules: tuple[str, ...]


@dataclass
class BacktestSummary:
    release: str
    generated_at: Optional[str]
    seasons: list[int]
    stat: Optional[str]
    positions: list[str]
    verdict: Verdict
    release_verdict: Verdict
    cohort_rules: list[str]
    rows: list[EvalRow]
    available: list[Cell]
    not_measured: dict[str, str] = field(default_factory=lambda: dict(NOT_MEASURED))


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def metric_label(stat: str, metric: str) -> str:
    """Plain words for how a cell is scored. Deviance is never called an 'average miss'."""
    if metric == "mae":
        unit = "points" if stat in {"fantasy_ppr", "ppr_points"} else "yards" if stat.endswith("_yards") else "units"
        return f"Average miss, in {unit} (lower is better)"
    if metric == "poisson_deviance":
        return "Count error score, Poisson deviance (lower is better)"
    return f"{metric.replace('_', ' ').capitalize()} (lower is better)"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS = (
    "stat", "metric", "n", "cohort_rule", "model_score", "naive_score", "trailing3_score",
    "marcel_score", "beats_naive", "beats_trailing3", "beats_marcel",
    "max_train_season_ok", "season", "position",
)

_lock = threading.Lock()
_PARSED: dict[tuple[tuple[str, str], ...], ReleaseEvals] = {}


def clear_cache() -> None:
    """Drop parsed evals (tests, deliberate re-freeze)."""
    with _lock:
        _PARSED.clear()


def _default_manifest() -> ArtifactManifest:
    return get_manifest()


def _flag(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"expected True/False, got {value!r}")
    return value == "True"


def _read_csv(path: Path) -> list[EvalRow]:
    out: list[EvalRow] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in _REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path.name} is missing columns {missing}")
        for r in reader:
            out.append(EvalRow(
                stat=r["stat"],
                position=r["position"].upper(),
                season=int(r["season"]),
                metric=r["metric"],
                metric_label=metric_label(r["stat"], r["metric"]),
                n=int(r["n"]),
                cohort_rule=r["cohort_rule"],
                model_score=float(r["model_score"]),
                naive_score=float(r["naive_score"]),
                trailing3_score=float(r["trailing3_score"]),
                marcel_score=float(r["marcel_score"]),
                beats_naive=_flag(r["beats_naive"]),
                beats_trailing3=_flag(r["beats_trailing3"]),
                beats_marcel=_flag(r["beats_marcel"]),
                causal_ok=_flag(r["max_train_season_ok"]),
            ))
    return out


def _both(row: EvalRow) -> bool:
    return row.beats_naive and row.beats_trailing3


def _check_against_beats(rows: list[EvalRow], beats: dict) -> None:
    """The summary JSON and the CSVs are pinned together; they must agree."""
    headline = beats.get("headline") or {}
    both = sum(_both(r) for r in rows)
    if headline.get("beat_both") != both or headline.get("cell_seasons") != len(rows):
        raise ValueError(
            f"beats headline {headline} disagrees with the rows: "
            f"{both}/{len(rows)} beat both"
        )
    per_cell: dict[tuple[str, str], list[EvalRow]] = {}
    for r in rows:
        per_cell.setdefault((r.stat, r.position), []).append(r)
    for c in beats.get("cells", []):
        mine = per_cell.get((c["stat"], c["position"]), [])
        if (
            len(mine) != c["cell_seasons"]
            or sum(_both(r) for r in mine) != c["beat_both_count"]
            or any(r.metric != c["metric"] for r in mine)
        ):
            raise ValueError(f"beats cell {c['stat']}/{c['position']} disagrees with its rows")


def load_release_evals(manifest: Optional[ArtifactManifest] = None) -> ReleaseEvals:
    """Read and verify the pinned evals. Raises ``EvalsUnavailable`` on any problem."""
    try:
        manifest = manifest or _default_manifest()
        files = manifest.resolve_group(GROUP_KEY)
        key = tuple(sorted((rel, manifest.digests[rel]) for rel in files))
        with _lock:
            cached = _PARSED.get(key)
        if cached is not None:
            return cached

        beats_rel = next((r for r in files if Path(r).name == BEATS_NAME), None)
        if beats_rel is None:
            raise ValueError(f"{GROUP_KEY} pins no {BEATS_NAME}")
        beats = json.loads(files[beats_rel].read_text())
        rows: list[EvalRow] = []
        for rel, path in files.items():
            if Path(rel).name.startswith(CSV_PREFIX) and rel.endswith(".csv"):
                rows.extend(_read_csv(path))
        if not rows:
            raise ValueError(f"{GROUP_KEY} pins no eval CSVs")
        _check_against_beats(rows, beats)

        rows.sort(key=lambda r: (r.stat, r.position, r.season))
        cells = tuple(sorted(
            {(r.stat, r.position, r.metric): Cell(r.stat, r.position, r.metric, r.metric_label)
             for r in rows}.values(),
            key=lambda c: (c.stat, c.position),
        ))
        evals = ReleaseEvals(
            release=Path(beats_rel).parent.parent.name,
            generated_at=beats.get("generated_at"),
            rows=tuple(rows),
            cells=cells,
            cohort_rules=tuple(sorted({r.cohort_rule for r in rows})),
        )
    except (ManifestError, OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("causal evals unavailable: %s", exc)
        raise EvalsUnavailable(str(exc)) from exc
    with _lock:
        _PARSED[key] = evals
    return evals


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

def verdict_of(rows: list[EvalRow] | tuple[EvalRow, ...]) -> Verdict:
    gains = [g for g in (r.improvement_vs_naive_pct for r in rows) if g is not None]
    return Verdict(
        cell_seasons=len(rows),
        beat_naive=sum(r.beats_naive for r in rows),
        beat_trailing3=sum(r.beats_trailing3 for r in rows),
        beat_marcel=sum(r.beats_marcel for r in rows),
        beat_both=sum(_both(r) for r in rows),
        causal_check_passed=sum(r.causal_ok for r in rows),
        mean_improvement_vs_naive_pct=(sum(gains) / len(gains)) if gains else None,
    )


class BacktestService:
    """Aggregates the served release's pinned causal evals for the API."""

    def __init__(self, manifest: Optional[ArtifactManifest] = None) -> None:
        self._manifest = manifest

    def evals(self) -> ReleaseEvals:
        return load_release_evals(self._manifest)

    def get_summary(
        self,
        stat: Optional[str] = None,
        positions: Optional[list[str]] = None,
    ) -> BacktestSummary:
        """
        Release-wide verdict plus the rows for the selection.

        ``stat=None`` selects every stat. ``positions=None`` selects every
        position. An unknown stat raises ``UnknownStat``; a known stat with no
        row for the chosen positions returns an empty selection.
        """
        ev = self.evals()
        if stat is not None and stat not in {c.stat for c in ev.cells}:
            raise UnknownStat(stat)
        wanted = {p.upper() for p in positions} if positions else None
        rows = [
            r for r in ev.rows
            if (stat is None or r.stat == stat) and (wanted is None or r.position in wanted)
        ]
        return BacktestSummary(
            release=ev.release,
            generated_at=ev.generated_at,
            seasons=sorted({r.season for r in ev.rows}),
            stat=stat,
            positions=sorted(wanted) if wanted else sorted({c.position for c in ev.cells}),
            verdict=verdict_of(rows),
            release_verdict=verdict_of(ev.rows),
            cohort_rules=list(ev.cohort_rules),
            rows=rows,
            available=list(ev.cells),
        )
