"""A psycopg2-shaped connection over in-memory sqlite, for tests that must run real SQL.

``FakeConn`` (``_fake_db.py``) answers routed statements; it cannot say which rows a relabel or a delete
leaves behind. Tests about *which rows get served* need a table. sqlite runs the statements the weekly
refresh sends once ``%s`` becomes ``?`` and the few Postgres functions it calls exist. A statement that
needs Postgres-only syntax (``= ANY(%s)``) fails here on purpose rather than being guessed at.
"""
from __future__ import annotations

import json
import sqlite3

SCHEMA = """
CREATE TABLE games (season INT, week INT, kickoff_at TEXT);
CREATE TABLE game_logs (season INT, week INT);
CREATE TABLE projections (
    player_id TEXT, game_id TEXT, stat TEXT, season INT, week INT, pipeline_run_id TEXT,
    availability_status TEXT, UNIQUE (player_id, game_id, stat));
CREATE TABLE projections_replaced (
    player_id TEXT, game_id TEXT, stat TEXT, season INT, week INT, pipeline_run_id TEXT,
    availability_status TEXT, archived_at TEXT);
CREATE TABLE season_simulation_weeks (season INT, start_week INT, pipeline_run_id TEXT, k TEXT);
CREATE TABLE season_simulations (season INT, start_week INT, pipeline_run_id TEXT, k TEXT);
CREATE TABLE season_team_wins (season INT, start_week INT, pipeline_run_id TEXT, k TEXT);
CREATE TABLE team_game_predictions (team TEXT, season INT, week INT, model_run_id TEXT,
    PRIMARY KEY (team, season, week));
CREATE TABLE forecast_runs (pipeline_run_id TEXT PRIMARY KEY, season INT, week INT, source_summary TEXT);
"""


class _Cursor:
    def __init__(self, db: "SqlitePg") -> None:
        self._db = db
        self._cur = db.raw.cursor()
        self._boolean = False
        self.rowcount = -1

    def execute(self, sql: str, params=()) -> None:
        if "ANY(" in sql:
            raise NotImplementedError("Postgres-only ANY(...) is not emulated; avoid it in code under test")
        self._db.statements.append(sql)
        self._cur.execute(sql.replace("%s", "?"), tuple(params or ()))
        self.rowcount = self._cur.rowcount
        self._boolean = "pg_try_advisory_xact_lock" in sql  # Postgres returns a bool here, sqlite an int

    def fetchone(self):
        row = self._cur.fetchone()
        return (bool(row[0]),) if row is not None and self._boolean else row

    def fetchall(self):
        return self._cur.fetchall()

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc) -> None:
        return None


class SqlitePg:
    """One shared connection: ``close()`` does nothing, ``with conn`` commits or rolls back like psycopg2."""

    autocommit = False

    def __init__(self) -> None:
        self.raw = sqlite3.connect(":memory:", check_same_thread=False)
        self.raw.create_function("starts_with", 2, lambda s, p: int(s is not None and str(s).startswith(p)))
        self.raw.create_function("NOW", 0, lambda: "2026-09-25T00:00:00")
        self.raw.create_function("pg_try_advisory_xact_lock", 2, self._try_lock)
        self.raw.executescript(SCHEMA)
        self.statements: list[str] = []
        self.held_elsewhere: set[tuple[int, int]] = set()  # advisory keys another session holds
        self.lock_attempts: list[tuple[int, int]] = []

    def _try_lock(self, namespace: int, key: int) -> int:
        self.lock_attempts.append((namespace, key))
        return 0 if (namespace, key) in self.held_elsewhere else 1

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def commit(self) -> None:
        self.raw.commit()

    def rollback(self) -> None:
        self.raw.rollback()

    def close(self) -> None:
        return None

    def __enter__(self) -> "SqlitePg":
        return self

    def __exit__(self, exc_type, *_rest) -> None:
        self.raw.rollback() if exc_type else self.raw.commit()

    # ── test helpers (not part of the psycopg2 surface) ──────────────────────────────────────
    def run(self, sql: str, params=()) -> list[tuple]:
        cur = self.raw.execute(sql, params)
        rows = cur.fetchall()
        self.raw.commit()
        return rows

    def add_projection(self, player: str, run_id: str, *, week: int = 3, season: int = 2026,
                       game: str | None = None, stat: str = "fantasy_points") -> None:
        """Upsert like the producer: a row for the same (player, game, stat) takes the new run id."""
        self.raw.execute(
            "INSERT INTO projections (player_id, game_id, stat, season, week, pipeline_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (player_id, game_id, stat) "
            "DO UPDATE SET pipeline_run_id = excluded.pipeline_run_id",
            (player, game or f"{season}_{week:02d}_AAA_BBB", stat, season, week, run_id),
        )
        self.raw.commit()

    def add_forecast_run(self, key: str, pending_id: str, *, season: int = 2026, week: int = 3,
                         kind: str = "forward_projections") -> None:
        summary = json.dumps({"kind": kind, "pipeline_run_id": pending_id})
        self.raw.execute(
            "INSERT INTO forecast_runs VALUES (?, ?, ?, ?)", (key, season, week, summary))
        self.raw.commit()

    def served(self, approved: str = "weekly_auto_2026") -> set[str]:
        return {r[0] for r in self.run(
            "SELECT player_id FROM projections WHERE pipeline_run_id = ?", (approved,))}
