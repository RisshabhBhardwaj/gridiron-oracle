"""
backend/app/api/backtest.py

GET /backtest and GET /backtest/table.csv

Backtest evidence for the served release, read from its pinned causal evals
(see ``backend/app/services/backtest.py``). Only what the release measures is
returned: model vs simple-method, trailing-3 and Marcel scores per player-stat
season. Anything it does not measure is listed under ``not_measured``.
"""

from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from backend.app.core.rate_limit import limiter
from backend.app.core.runtime_mode import ArtifactRequiredError
from backend.app.services.backtest import (
    UNAVAILABLE_MESSAGE,
    BacktestService,
    EvalRow,
    UnknownStat,
)

router = APIRouter(prefix="", tags=["backtest"])

TABLE_PATH = "/backtest/table.csv"


class Verdict(BaseModel):
    cell_seasons: int = Field(description="Player-stat seasons compared")
    beat_naive: int = Field(description="How many beat the simple method (last game)")
    beat_trailing3: int = Field(description="How many beat the trailing-3 average")
    beat_marcel: int = Field(description="How many beat the Marcel baseline")
    beat_both: int = Field(description="How many beat both the simple method and trailing-3")
    causal_check_passed: int = Field(description="How many passed the no-future-data check")
    mean_improvement_vs_naive_pct: Optional[float] = Field(
        default=None,
        description="Average percent the model score sits below the simple method's (positive = better)",
    )


class EvalRowOut(BaseModel):
    stat: str
    position: str
    season: int
    metric: str = Field(description="mae or poisson_deviance")
    metric_label: str = Field(description="Plain-words name of the score; lower is better")
    n: int = Field(description="Player-games in the comparison")
    cohort_rule: str
    model_score: float
    naive_score: float = Field(description="Simple method: the player's last game")
    trailing3_score: float
    marcel_score: float
    beats_naive: bool
    beats_trailing3: bool
    beats_marcel: bool
    improvement_vs_naive_pct: Optional[float]


class CellOut(BaseModel):
    stat: str
    position: str
    metric: str
    metric_label: str


class BacktestResponse(BaseModel):
    release: str
    generated_at: Optional[str]
    seasons: list[int]
    stat: Optional[str]
    positions: list[str]
    verdict: Verdict = Field(description="Counts for the selected stat and positions")
    release_verdict: Verdict = Field(description="Counts for the whole release")
    cohort_rules: list[str]
    rows: list[EvalRowOut]
    available: list[CellOut] = Field(description="Every stat and position the release measures")
    not_measured: dict[str, str]
    table_url: str


def _row_out(r: EvalRow) -> EvalRowOut:
    return EvalRowOut(
        stat=r.stat, position=r.position, season=r.season, metric=r.metric,
        metric_label=r.metric_label, n=r.n, cohort_rule=r.cohort_rule,
        model_score=r.model_score, naive_score=r.naive_score,
        trailing3_score=r.trailing3_score, marcel_score=r.marcel_score,
        beats_naive=r.beats_naive, beats_trailing3=r.beats_trailing3,
        beats_marcel=r.beats_marcel, improvement_vs_naive_pct=r.improvement_vs_naive_pct,
    )


def _unavailable() -> HTTPException:
    # The cause (path, digest) is logged by the service; callers get one safe line.
    return HTTPException(
        status_code=503,
        detail={"message": UNAVAILABLE_MESSAGE, "code": "evidence_unavailable"},
    )


@router.get("/backtest", response_model=BacktestResponse)
@limiter.limit("30/minute")
def get_backtest(
    request: Request,
    stat: Optional[str] = Query(None, max_length=64, description="Stat to show; omit for all"),
    positions: Optional[list[str]] = Query(default=None, description="Positions to show; omit for all"),
) -> BacktestResponse:
    """Served-release evidence: model vs simple method, trailing-3 and Marcel."""
    try:
        s = BacktestService().get_summary(stat=stat, positions=list(positions) if positions else None)
    except UnknownStat as exc:
        raise HTTPException(status_code=404, detail=f"No results for stat {exc}") from exc
    except ArtifactRequiredError as exc:
        raise _unavailable() from exc

    return BacktestResponse(
        release=s.release,
        generated_at=s.generated_at,
        seasons=s.seasons,
        stat=s.stat,
        positions=s.positions,
        verdict=Verdict(**vars(s.verdict)),
        release_verdict=Verdict(**vars(s.release_verdict)),
        cohort_rules=s.cohort_rules,
        rows=[_row_out(r) for r in s.rows],
        available=[CellOut(**vars(c)) for c in s.available],
        not_measured=s.not_measured,
        table_url=TABLE_PATH,
    )


_TABLE_HEADER = [
    "stat", "position", "season", "metric", "metric_label", "n",
    "model_score", "naive_score", "trailing3_score", "marcel_score",
    "beats_naive", "beats_trailing3", "beats_marcel", "improvement_vs_naive_pct", "cohort_rule",
]


@router.get(TABLE_PATH)
@limiter.limit("10/minute")
def get_backtest_table(request: Request) -> Response:
    """Every player-stat season of the release as a downloadable table."""
    try:
        svc = BacktestService()
        ev = svc.evals()
    except ArtifactRequiredError as exc:
        raise _unavailable() from exc

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(_TABLE_HEADER)
    for r in ev.rows:
        gain = r.improvement_vs_naive_pct
        w.writerow([
            r.stat, r.position, r.season, r.metric, r.metric_label, r.n,
            r.model_score, r.naive_score, r.trailing3_score, r.marcel_score,
            r.beats_naive, r.beats_trailing3, r.beats_marcel,
            "" if gain is None else f"{gain:.2f}", r.cohort_rule,
        ])
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="gridiron-backtest-{ev.release}.csv"'},
    )
