"""
Offline coverage tests for pipeline/ingest.py and pipeline/schema.py.

The ingest CLI is tested with the Orchestrator replaced by a recording fake.
Alembic's upgrade and stamp commands are replaced, so no migration runs.
No database, no network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import pipeline.ingest as ingest
import pipeline.orchestrator as orchestrator_mod
import pipeline.schema as schema
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.orchestrator import SeasonResult, StepResult

# ── ingest.main ───────────────────────────────────────────────────────────────


class _FakeOrchestrator:
    instances: list["_FakeOrchestrator"] = []
    summary = None
    raise_on_run: BaseException | None = None

    def __init__(self, db_url: str = "") -> None:
        self.db_url = db_url
        self.run_args: tuple | None = None
        _FakeOrchestrator.instances.append(self)

    def run(self, seasons=None, dry_run=False, sources=None):
        self.run_args = (seasons, dry_run)
        if _FakeOrchestrator.raise_on_run is not None:
            raise _FakeOrchestrator.raise_on_run
        return _FakeOrchestrator.summary


@pytest.fixture
def fake_orch(monkeypatch):
    _FakeOrchestrator.instances = []
    _FakeOrchestrator.raise_on_run = None
    _FakeOrchestrator.summary = orchestrator_mod.OrchestratorSummary(
        seasons=[2025], dry_run=True,
        season_results=[SeasonResult(season=2025, dry_run=True, steps=[
            StepResult(name="ingest", elapsed_s=0.1, ok=True),
        ])],
    )
    monkeypatch.setattr(orchestrator_mod, "Orchestrator", _FakeOrchestrator)
    return _FakeOrchestrator


def test_ingest_main_returns_zero_when_every_step_succeeds(fake_orch, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    code = ingest.main(["--seasons", "2025", "--dry-run"])

    assert code == 0
    orch = fake_orch.instances[0]
    assert orch.db_url == "postgresql://oracle:oracle@localhost:15439/oracle"
    assert orch.run_args == ([2025], True)


def test_ingest_main_returns_one_when_any_step_failed(fake_orch):
    fake_orch.summary.season_results[0].steps.append(
        StepResult(name="normalize", elapsed_s=0.2, ok=False, error="FK violation"),
    )
    assert ingest.main(["--seasons", "2025", "--db-url", "postgresql://h/db"]) == 1
    assert fake_orch.instances[0].db_url == "postgresql://h/db"


def test_ingest_main_returns_one_when_the_orchestrator_raises(fake_orch, caplog):
    fake_orch.raise_on_run = RuntimeError("no DATABASE_URL")
    with caplog.at_level(logging.ERROR, logger="pipeline.ingest"):
        code = ingest.main([])

    assert code == 1
    assert any("Ingest failed: no DATABASE_URL" in r.getMessage() for r in caplog.records)


def test_ingest_main_uses_database_url_from_env_when_no_override(fake_orch, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://env-host/db")
    ingest.main([])
    assert fake_orch.instances[0].db_url == "postgresql://env-host/db"
    assert fake_orch.instances[0].run_args == (None, False)


def test_ingest_main_rejects_a_non_integer_season(fake_orch):
    with pytest.raises(SystemExit) as exc:
        ingest.main(["--seasons", "twenty"])
    assert exc.value.code == 2
    assert fake_orch.instances == []


# ── schema helpers ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("postgresql+asyncpg://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgresql+psycopg2://u:p@h/db", "postgresql://u:p@h/db"),
        ("postgresql://u:p@h/db", "postgresql://u:p@h/db"),
    ],
)
def test_normalize_dsn_strips_driver_suffixes(raw, expected):
    assert schema.normalize_dsn(raw) == expected


def test_database_url_from_env_prefers_the_environment(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://from-env/db")
    assert schema.database_url_from_env() == "postgresql://from-env/db"


def test_database_url_from_env_falls_back_to_the_host_default(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert schema.database_url_from_env() == DEFAULT_HOST_DATABASE_URL


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, str]] = []

    def make(self, name: str):
        def _record(cfg, revision):
            self.calls.append((name, cfg, revision))
        return _record


@pytest.fixture
def alembic_calls(monkeypatch):
    import alembic.command as alembic_command

    rec = _Recorder()
    monkeypatch.setattr(alembic_command, "upgrade", rec.make("upgrade"))
    monkeypatch.setattr(alembic_command, "stamp", rec.make("stamp"))
    return rec


def test_upgrade_head_targets_repo_alembic_ini_with_normalized_url(alembic_calls):
    schema.upgrade_head("postgresql+asyncpg://u:p@h/db")

    (name, cfg, revision), = alembic_calls.calls
    assert (name, revision) == ("upgrade", "head")
    assert Path(cfg.config_file_name) == schema.REPO_ROOT / "alembic.ini"
    assert cfg.get_main_option("sqlalchemy.url") == "postgresql://u:p@h/db"


def test_upgrade_head_defaults_to_the_environment_url(alembic_calls, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg2://env/db")
    schema.upgrade_head()

    (_, cfg, _), = alembic_calls.calls
    assert cfg.get_main_option("sqlalchemy.url") == "postgresql://env/db"


def test_stamp_head_marks_head_without_running_ddl(alembic_calls):
    schema.stamp_head("postgresql://u:p@h/db")

    (name, cfg, revision), = alembic_calls.calls
    assert (name, revision) == ("stamp", "head")
    assert cfg.get_main_option("sqlalchemy.url") == "postgresql://u:p@h/db"


@pytest.mark.parametrize("helper", ["upgrade_head", "stamp_head"])
def test_helpers_accept_a_percent_encoded_password(alembic_calls, helper):
    # A URL-encoded password (`%40` is `@`) used to crash configparser
    # interpolation inside Config.set_main_option before any SQL ran.
    url = "postgresql://u:p%40ss%25w@h:5432/db"
    getattr(schema, helper)(url)

    (_, cfg, _), = alembic_calls.calls
    assert cfg.get_main_option("sqlalchemy.url") == url
    assert cfg.attributes[schema.EXPLICIT_DB_URL] == url
