"""
Offline coverage tests for pipeline/orchestrator.py.

The dry-run path monkeypatches the nflreadpy loaders with in-memory frames.
The live path replaces the adapter, Normalizer, and FeatureEngineer with
recording fakes. dead_letter writes go through FakeConn. No network, no DB.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import nflreadpy
import psycopg2
import pytest

import pipeline.orchestrator as orch_mod
from backend.tests._pipeline_fakes import FakeConn, FakeConnFactory
from pipeline.normalize import NormalizeSummary
from pipeline.orchestrator import (
    Orchestrator,
    OrchestratorSummary,
    SeasonResult,
    StepResult,
    _print_orchestrator_summary,
    main,
)

DB_URL = "postgresql+asyncpg://user:pw@db.test:5432/oracle"


class _Frame:
    """Stands in for a polars DataFrame: only to_dicts() and len() are used."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._rows]

    def __len__(self) -> int:
        return len(self._rows)


def _install_nflreadpy(monkeypatch, stats=(), sched=(), roster=()) -> None:
    monkeypatch.setattr(nflreadpy, "load_player_stats", lambda seasons: _Frame(stats))
    monkeypatch.setattr(nflreadpy, "load_schedules", lambda seasons: _Frame(sched))
    monkeypatch.setattr(nflreadpy, "load_rosters", lambda seasons: _Frame(roster))


def _stat(pid: str, week: int, team: str, **extra: Any) -> dict[str, Any]:
    return {"player_id": pid, "season": 2025, "week": week, "team": team, **extra}


SCHED_KC_BAL = {"game_id": "2025_01_KC_BAL", "season": 2025, "week": 1,
                "home_team": "KC", "away_team": "BAL"}


class _Recorder:
    """Callable that records its arguments and returns a fixed value."""

    def __init__(self, returns: Any = None, raises: BaseException | None = None) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        self._returns = returns
        self._raises = raises

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        if self._raises is not None:
            raise self._raises
        return self._returns


class _FakeAdapter:
    instances: list["_FakeAdapter"] = []
    ingest_results: dict = {}
    raise_on_run: BaseException | None = None

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.run_calls: list[dict] = []
        _FakeAdapter.instances.append(self)

    def __enter__(self) -> "_FakeAdapter":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def run_full_ingest(self, seasons, sources=None) -> dict:
        self.run_calls.append({"seasons": seasons, "sources": sources})
        if _FakeAdapter.raise_on_run is not None:
            raise _FakeAdapter.raise_on_run
        return _FakeAdapter.ingest_results


@pytest.fixture(autouse=True)
def _no_weather_network(monkeypatch):
    """Weather enrichment talks to a forecast API and the DB; stub it for every test here."""
    monkeypatch.setattr(
        "scraper.adapters.weather_adapter.enrich_games_precipitation",
        lambda db_url: 0,
    )


@pytest.fixture
def fake_adapter(monkeypatch):
    _FakeAdapter.instances = []
    _FakeAdapter.ingest_results = {"player_stats": (10, 1), "rosters": (5, 0)}
    _FakeAdapter.raise_on_run = None
    monkeypatch.setattr(orch_mod, "NFLReadPyAdapter", _FakeAdapter)
    return _FakeAdapter


class _FakeNormalizer:
    instances: list["_FakeNormalizer"] = []
    summary = NormalizeSummary(players_upserted=3, games_upserted=2, game_logs_upserted=7)
    raise_on_run: BaseException | None = None

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.run_seasons: list | None = None
        _FakeNormalizer.instances.append(self)

    def __enter__(self) -> "_FakeNormalizer":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def run(self, seasons=None) -> NormalizeSummary:
        self.run_seasons = seasons
        if _FakeNormalizer.raise_on_run is not None:
            raise _FakeNormalizer.raise_on_run
        return _FakeNormalizer.summary


@pytest.fixture
def fake_normalizer(monkeypatch):
    _FakeNormalizer.instances = []
    _FakeNormalizer.summary = NormalizeSummary(players_upserted=3, games_upserted=2, game_logs_upserted=7)
    _FakeNormalizer.raise_on_run = None
    monkeypatch.setattr(orch_mod, "Normalizer", _FakeNormalizer)
    return _FakeNormalizer


class _FakeFeatureEngineer:
    instances: list["_FakeFeatureEngineer"] = []
    result = 42
    raise_on_run: BaseException | None = None

    def __init__(self, db_url: str) -> None:
        self.db_url = db_url
        self.run_seasons: list | None = None
        _FakeFeatureEngineer.instances.append(self)

    def __enter__(self) -> "_FakeFeatureEngineer":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def run(self, seasons=None) -> int:
        self.run_seasons = seasons
        if _FakeFeatureEngineer.raise_on_run is not None:
            raise _FakeFeatureEngineer.raise_on_run
        return _FakeFeatureEngineer.result


@pytest.fixture
def fake_features(monkeypatch):
    _FakeFeatureEngineer.instances = []
    _FakeFeatureEngineer.result = 42
    _FakeFeatureEngineer.raise_on_run = None
    monkeypatch.setattr(orch_mod, "FeatureEngineer", _FakeFeatureEngineer)
    return _FakeFeatureEngineer


@pytest.fixture
def dead_letters(monkeypatch) -> list[tuple[str, str, int]]:
    written: list[tuple[str, str, int]] = []

    def _record(db_url: str, source: str, error: str, season: int) -> None:
        written.append((source, error, season))

    monkeypatch.setattr(orch_mod, "_write_dead_letter", _record)
    return written


# ── dry-run path ──────────────────────────────────────────────────────────────

def test_dry_run_season_counts_rows_through_all_three_steps(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        stats=[
            _stat("00-1", 1, "KC", game_id="2025_01_KC_BAL", position="WR"),
            _stat("00-2", 1, "BAL", game_id=None, position="RB"),  # derived from schedule
            _stat("00-3", 2, "ZZZ", game_id=None),  # no schedule match: dropped
            {"season": 2025, "week": 1, "team": "KC"},  # no player id: ingest failure
        ],
        sched=[SCHED_KC_BAL],
        roster=[
            {"season": 2025, "gsis_id": "00-1"},
            {"season": 2025, "gsis_id": "00-2"},
            {"season": 2025, "gsis_id": None},
        ],
    )
    seen: list[tuple[str, int, int]] = []

    def fake_build(target, prior, game, sampled):
        seen.append((target["player_id"], target["week"], len(prior)))
        return object()

    monkeypatch.setattr(orch_mod, "build_feature_row", fake_build)
    result = orch_mod._dry_run_season(2025)

    assert [s.name for s in result.steps] == ["ingest", "normalize", "feature_engineer"]
    assert all(s.ok for s in result.steps)

    ingest, normalize, features = result.steps
    assert ingest.rows_in == 8  # 4 stats + 1 schedule + 3 rosters, including the invalid one
    assert ingest.rows_out == 7  # 3 valid stats + 1 schedule + 3 rosters

    assert result.n_players == 2  # unique non-null gsis ids
    assert result.n_games == 1
    assert normalize.rows_out == 2 + 1 + 2  # players + games + game_logs (two matched)

    assert features.rows_in == 2
    assert features.rows_out == 2
    assert result.n_feature_rows == 2
    assert sorted(seen) == [("00-1", 1, 0), ("00-2", 1, 0)]


def test_dry_run_builds_prior_weeks_and_counts_feature_errors(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        stats=[
            _stat("00-1", 2, "KC", game_id="2025_02_KC_X"),
            _stat("00-1", 1, "KC", game_id="2025_01_KC_BAL"),
        ],
        sched=[SCHED_KC_BAL],
        roster=[{"season": 2025, "gsis_id": "00-1"}],
    )
    calls: list[tuple[int, int]] = []

    def fake_build(target, prior, game, sampled):
        calls.append((target["week"], len(prior)))
        if target["week"] == 2:
            raise ValueError("no history")
        return object()

    monkeypatch.setattr(orch_mod, "build_feature_row", fake_build)
    result = orch_mod._dry_run_season(2025)

    # Weeks are processed in order; week 2 sees week 1 as its only prior game.
    assert calls == [(1, 0), (2, 1)]
    assert result.n_feature_rows == 1
    assert result.steps[-1].ok is True  # per-row build errors do not fail the step


def test_dry_run_sample_cap_limits_feature_rows_to_first_unique_players(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        stats=[
            _stat("00-1", 1, "KC", game_id="2025_01_KC_BAL"),
            _stat("00-2", 1, "BAL", game_id="2025_01_KC_BAL"),
        ],
        sched=[SCHED_KC_BAL],
        roster=[{"season": 2025, "gsis_id": "00-1"}],
    )
    monkeypatch.setattr(orch_mod, "DRY_RUN_SAMPLE", 1)
    built: list[str] = []
    monkeypatch.setattr(
        orch_mod, "build_feature_row",
        lambda target, prior, game, sampled: built.append(target["player_id"]) or object(),
    )
    result = orch_mod._dry_run_season(2025)

    assert built == ["00-1"]
    assert result.n_feature_rows == 1


def test_dry_run_ingest_counts_invalid_schedule_and_roster_rows_out_of_valid_output(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        stats=[_stat("00-1", 1, "KC", game_id="2025_01_KC_BAL")],
        sched=[SCHED_KC_BAL, {"game_id": "2025_01_BAD", "season": 2025, "week": 1}],  # no teams
        roster=[{"season": 2025, "gsis_id": "00-1"}, {"gsis_id": "00-9"}],  # no season
    )
    monkeypatch.setattr(orch_mod, "build_feature_row", lambda *a: object())
    result = orch_mod._dry_run_season(2025)

    ingest = result.steps[0]
    assert ingest.ok is True
    assert ingest.rows_in == 1 + 2 + 2
    assert ingest.rows_out == 1 + 1 + 1  # the invalid schedule and roster rows are dropped
    assert result.n_games == 1
    assert result.n_players == 1


def test_dry_run_feature_step_failure_is_recorded_without_failing_the_season(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        stats=[_stat("00-1", 1, "KC", game_id="2025_01_KC_BAL")],
        sched=[SCHED_KC_BAL],
        roster=[{"season": 2025, "gsis_id": "00-1"}],
    )
    # A non-numeric sample cap makes the sampling loop itself raise, outside build_feature_row's guard.
    monkeypatch.setattr(orch_mod, "DRY_RUN_SAMPLE", None)
    result = orch_mod._dry_run_season(2025)

    features = result.steps[-1]
    assert features.name == "feature_engineer"
    assert features.ok is False
    assert features.error
    assert result.n_feature_rows == 0
    assert [s.ok for s in result.steps] == [True, True, False]


def test_dry_run_season_reports_missing_nflreadpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "nflreadpy", None)  # forces ImportError on import
    result = orch_mod._dry_run_season(2025)

    assert len(result.steps) == 1
    step = result.steps[0]
    assert (step.name, step.ok) == ("ingest", False)
    assert "nflreadpy not installed" in step.error


def test_dry_run_ingest_failure_stops_before_normalize(monkeypatch):
    def boom(seasons):
        raise RuntimeError("cache miss")

    monkeypatch.setattr(nflreadpy, "load_player_stats", boom)
    monkeypatch.setattr(nflreadpy, "load_schedules", lambda seasons: _Frame([]))
    monkeypatch.setattr(nflreadpy, "load_rosters", lambda seasons: _Frame([]))
    result = orch_mod._dry_run_season(2025)

    assert [s.name for s in result.steps] == ["ingest"]
    assert result.steps[0].ok is False
    assert result.steps[0].error == "cache miss"


def test_dry_run_normalize_failure_stops_before_feature_engineer(monkeypatch):
    _install_nflreadpy(
        monkeypatch,
        sched=[SCHED_KC_BAL],
        stats=[],
        roster=[],
    )
    # A schedule dict without home_team passes ingest but breaks the team-week map.
    monkeypatch.setattr(
        orch_mod, "ScheduleRow",
        type("_BadSchedule", (), {
            "model_validate": staticmethod(
                lambda raw: type("_V", (), {"model_dump": lambda self: {"game_id": "G", "week": 1}})()
            )
        }),
    )
    result = orch_mod._dry_run_season(2025)

    assert [s.name for s in result.steps] == ["ingest", "normalize"]
    assert result.steps[1].ok is False
    assert "home_team" in result.steps[1].error


# ── dead letter ───────────────────────────────────────────────────────────────

def test_write_dead_letter_inserts_source_error_and_season(monkeypatch):
    conn = FakeConn()
    monkeypatch.setattr(psycopg2, "connect", FakeConnFactory(conn))

    orch_mod._write_dead_letter(DB_URL, "orchestrator/ingest", "network down", 2025)

    inserts = conn.statements_containing("INSERT INTO dead_letter")
    assert len(inserts) == 1
    _, params = inserts[0]
    assert params[0] == "orchestrator/ingest"
    assert params[1] == "network down"
    assert params[2].adapted == {"season": 2025}
    assert conn.commits == 1
    assert conn.connect_dsns == ["postgresql://user:pw@db.test:5432/oracle"]


def test_write_dead_letter_swallows_write_failure_and_logs(monkeypatch, caplog):
    def refuse(dsn, *args, **kwargs):
        raise psycopg2.OperationalError("connection refused")

    monkeypatch.setattr(psycopg2, "connect", refuse)
    with caplog.at_level(logging.WARNING, logger="pipeline.orchestrator"):
        orch_mod._write_dead_letter(DB_URL, "orchestrator/ingest", "x", 2025)  # must not raise

    assert any("Could not write to dead_letter" in r.getMessage() for r in caplog.records)


# ── live season ───────────────────────────────────────────────────────────────

def test_live_season_runs_each_step_and_records_outcomes(
    fake_adapter, fake_normalizer, fake_features, monkeypatch
):
    enriched: list[str] = []
    monkeypatch.setattr(
        "scraper.adapters.weather_adapter.enrich_games_precipitation",
        lambda db_url: enriched.append(db_url) or 4,
    )
    result = orch_mod._live_season(2025, DB_URL, sources=["player_stats"])

    assert [s.name for s in result.steps] == ["ingest", "normalize", "feature_engineer"]
    assert all(s.ok for s in result.steps)
    assert result.steps[0].rows_out == 15  # 10 ok + 5 ok across sources
    assert result.n_players == 3
    assert result.n_games == 2
    assert result.steps[1].rows_out == 3 + 2 + 7
    assert result.n_feature_rows == 42

    assert fake_adapter.instances[0].run_calls == [{"seasons": [2025], "sources": ["player_stats"]}]
    assert fake_normalizer.instances[0].run_seasons == [2025]
    assert fake_features.instances[0].run_seasons == [2025]
    assert enriched == [DB_URL]


def test_live_season_weather_enrichment_failure_does_not_fail_the_season(
    fake_adapter, fake_normalizer, fake_features, monkeypatch
):
    def weather_down(db_url):
        raise RuntimeError("forecast API unavailable")

    monkeypatch.setattr("scraper.adapters.weather_adapter.enrich_games_precipitation", weather_down)
    result = orch_mod._live_season(2025, DB_URL)

    assert all(s.ok for s in result.steps)
    assert result.n_feature_rows == 42


def test_live_season_ingest_failure_dead_letters_and_skips_the_rest(
    fake_adapter, fake_normalizer, fake_features, dead_letters
):
    fake_adapter.raise_on_run = RuntimeError("network down")
    result = orch_mod._live_season(2025, DB_URL)

    assert [(s.name, s.ok) for s in result.steps] == [("ingest", False)]
    assert result.steps[0].error == "network down"
    assert dead_letters == [("orchestrator/ingest", "network down", 2025)]
    assert fake_normalizer.instances == []
    assert fake_features.instances == []


def test_live_season_normalize_failure_dead_letters_and_skips_features(
    fake_adapter, fake_normalizer, fake_features, dead_letters
):
    fake_normalizer.raise_on_run = RuntimeError("FK violation")
    result = orch_mod._live_season(2025, DB_URL)

    assert [(s.name, s.ok) for s in result.steps] == [("ingest", True), ("normalize", False)]
    assert dead_letters == [("orchestrator/normalize", "FK violation", 2025)]
    assert fake_features.instances == []


def test_live_season_feature_failure_is_recorded_and_dead_lettered(
    fake_adapter, fake_normalizer, fake_features, dead_letters
):
    fake_features.raise_on_run = RuntimeError("matrix write failed")
    result = orch_mod._live_season(2025, DB_URL)

    assert result.steps[-1].name == "feature_engineer"
    assert result.steps[-1].ok is False
    assert result.n_feature_rows == 0
    assert dead_letters == [("orchestrator/feature_engineer", "matrix write failed", 2025)]


# ── global enrichment ─────────────────────────────────────────────────────────

def test_run_global_enrichment_runs_elo_then_pbp_and_exports_database_url(monkeypatch):
    order: list[str] = []
    import pipeline.enrich_elo_embeddings as elo
    import pipeline.pbp_pipeline as pbp

    monkeypatch.setattr(elo, "main", lambda: order.append("elo"))
    monkeypatch.setattr(pbp, "main", lambda: order.append("pbp"))
    monkeypatch.setenv("DATABASE_URL", "tmp")
    monkeypatch.delenv("DATABASE_URL")

    orch_mod._run_global_enrichment(DB_URL)

    assert order == ["elo", "pbp"]
    assert os.environ["DATABASE_URL"] == DB_URL


def test_run_global_enrichment_keeps_an_existing_database_url(monkeypatch):
    import pipeline.enrich_elo_embeddings as elo
    import pipeline.pbp_pipeline as pbp

    monkeypatch.setattr(elo, "main", lambda: None)
    monkeypatch.setattr(pbp, "main", lambda: None)
    monkeypatch.setenv("DATABASE_URL", "postgresql://already-set")

    orch_mod._run_global_enrichment(DB_URL)

    assert os.environ["DATABASE_URL"] == "postgresql://already-set"


def test_run_global_enrichment_elo_failure_dead_letters_and_pbp_still_runs(
    monkeypatch, dead_letters
):
    import pipeline.enrich_elo_embeddings as elo
    import pipeline.pbp_pipeline as pbp

    ran: list[str] = []
    monkeypatch.setattr(elo, "main", lambda: (_ for _ in ()).throw(RuntimeError("elo broke")))
    monkeypatch.setattr(pbp, "main", lambda: ran.append("pbp"))
    monkeypatch.setenv("DATABASE_URL", DB_URL)

    orch_mod._run_global_enrichment(DB_URL)

    assert ran == ["pbp"]
    assert dead_letters == [("orchestrator/enrich_elo_embeddings", "elo broke", 0)]


def test_run_global_enrichment_pbp_failure_dead_letters(monkeypatch, dead_letters):
    import pipeline.enrich_elo_embeddings as elo
    import pipeline.pbp_pipeline as pbp

    monkeypatch.setattr(elo, "main", lambda: None)
    monkeypatch.setattr(pbp, "main", lambda: (_ for _ in ()).throw(RuntimeError("pbp broke")))
    monkeypatch.setenv("DATABASE_URL", DB_URL)

    orch_mod._run_global_enrichment(DB_URL)

    assert dead_letters == [("orchestrator/pbp_pipeline", "pbp broke", 0)]


# ── Orchestrator.run ──────────────────────────────────────────────────────────

def test_run_live_without_database_url_raises_value_error():
    with pytest.raises(ValueError, match="DATABASE_URL is required"):
        Orchestrator().run(seasons=[2025], dry_run=False)


def test_run_dry_run_delegates_per_season_and_skips_global_enrichment(monkeypatch, capsys):
    calls: list[int] = []

    def fake_dry(season: int) -> SeasonResult:
        calls.append(season)
        return SeasonResult(season=season, dry_run=True, n_players=5, n_games=2, n_feature_rows=9)

    enrich_calls: list[str] = []
    monkeypatch.setattr(orch_mod, "_dry_run_season", fake_dry)
    monkeypatch.setattr(orch_mod, "_run_global_enrichment", lambda url: enrich_calls.append(url))

    summary = Orchestrator().run(seasons=[2024, 2025], dry_run=True)

    assert calls == [2024, 2025]
    assert enrich_calls == []
    assert summary.total_players == 10
    assert summary.total_games == 4
    assert summary.total_feature_rows == 18
    assert summary.dry_run is True
    assert "ORCHESTRATOR SUMMARY  (DRY-RUN)" in capsys.readouterr().out


def test_run_live_isolates_season_failures_and_runs_global_enrichment_once(monkeypatch):
    calls: list[tuple[int, Any]] = []

    def fake_live(season, db_url, sources=None):
        calls.append((season, sources))
        ok = season != 2024
        return SeasonResult(
            season=season, dry_run=False,
            steps=[StepResult(name="ingest", elapsed_s=0.0, ok=ok, error=None if ok else "bad")],
            n_players=1 if ok else 0, n_games=1 if ok else 0, n_feature_rows=2 if ok else 0,
        )

    enrich: list[str] = []
    monkeypatch.setattr(orch_mod, "_live_season", fake_live)
    monkeypatch.setattr(orch_mod, "_run_global_enrichment", lambda url: enrich.append(url))

    summary = Orchestrator(db_url=DB_URL).run(seasons=[2024, 2025], dry_run=False, sources=["rosters"])

    assert calls == [(2024, ["rosters"]), (2025, ["rosters"])]  # 2024 failure does not stop 2025
    assert enrich == [DB_URL]
    assert [sr.season for sr in summary.season_results] == [2024, 2025]
    assert summary.total_players == 1
    assert summary.end_time is not None


# ── printing ──────────────────────────────────────────────────────────────────

def test_print_orchestrator_summary_shows_failed_step_error_and_totals(capsys):
    summary = OrchestratorSummary(
        seasons=[2025],
        dry_run=False,
        season_results=[
            SeasonResult(
                season=2025, dry_run=False, n_players=3, n_games=1, n_feature_rows=4,
                steps=[
                    StepResult(name="ingest", elapsed_s=1.5, ok=True, rows_in=10, rows_out=9),
                    StepResult(name="normalize", elapsed_s=0.2, ok=False, error="boom"),
                ],
            )
        ],
        total_players=3,
        total_games=1,
        total_feature_rows=4,
    )
    _print_orchestrator_summary(summary)
    out = capsys.readouterr().out

    assert "ORCHESTRATOR SUMMARY  (LIVE)" in out
    assert "✓ ingest" in out
    assert "✗ normalize" in out and "ERROR: boom" in out
    assert "Players: 3  Games: 1  Features: 4" in out
    assert "TOTAL: 3 players | 1 games | 4 feature rows | 0.0s" in out


# ── CLI ───────────────────────────────────────────────────────────────────────

class _RecordingOrchestrator:
    instances: list["_RecordingOrchestrator"] = []

    def __init__(self, db_url: str = "") -> None:
        self.db_url = db_url
        self.run_args: tuple | None = None
        _RecordingOrchestrator.instances.append(self)

    def run(self, seasons, dry_run=False, sources=None):
        self.run_args = (seasons, dry_run)


@pytest.fixture
def recording_orchestrator(monkeypatch):
    _RecordingOrchestrator.instances = []
    monkeypatch.setattr(orch_mod, "Orchestrator", _RecordingOrchestrator)
    return _RecordingOrchestrator


def test_main_dry_run_uses_no_database(monkeypatch, recording_orchestrator):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--seasons", "2025", "--dry-run"])
    main()

    orch = recording_orchestrator.instances[0]
    assert orch.db_url == ""
    assert orch.run_args == ([2025], True)


def test_main_live_requires_database_url(monkeypatch, recording_orchestrator):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--seasons", "2025"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert recording_orchestrator.instances == []


def test_main_live_passes_database_url_and_seasons(monkeypatch, recording_orchestrator):
    monkeypatch.setenv("DATABASE_URL", DB_URL)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--seasons", "2024", "2025"])
    main()

    orch = recording_orchestrator.instances[0]
    assert orch.db_url == DB_URL
    assert orch.run_args == ([2024, 2025], False)


def test_main_defaults_to_current_season_from_nflreadpy(monkeypatch, recording_orchestrator):
    monkeypatch.setattr(nflreadpy, "get_current_season", lambda: 2026)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--dry-run"])
    main()
    assert recording_orchestrator.instances[0].run_args == ([2026], True)


def test_main_falls_back_to_configured_season_when_nflreadpy_fails(monkeypatch, recording_orchestrator):
    from ml.season_constants import CURRENT_SEASON

    def unavailable():
        raise RuntimeError("no network")

    monkeypatch.setattr(nflreadpy, "get_current_season", unavailable)
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--dry-run"])
    main()
    assert recording_orchestrator.instances[0].run_args == ([CURRENT_SEASON], True)


def test_main_verbose_raises_root_logger_to_debug(monkeypatch, recording_orchestrator):
    monkeypatch.setattr(sys, "argv", ["orchestrator.py", "--seasons", "2025", "--dry-run", "--verbose"])
    root = logging.getLogger()
    original = root.level
    try:
        main()
        assert root.level == logging.DEBUG
    finally:
        root.setLevel(original)
