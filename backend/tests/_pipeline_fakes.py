"""
Offline fakes for the pipeline coverage tests (test_pipeline_cov_*.py).

No database, no network. ``FakeConn`` records every statement and every
``execute_values`` batch so tests can assert what would have been written,
and it lets a test script the rows a SELECT returns or force a statement
to raise.

Not a test module (no ``test_`` prefix), so pytest does not collect it.
"""

from __future__ import annotations

from typing import Any, Optional


class FakeCursor:
    def __init__(self, conn: "FakeConn", cursor_factory: Any = None) -> None:
        self._conn = conn
        self.cursor_factory = cursor_factory
        self.rowcount: int = 0
        self._rows: list[Any] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.split())
        self._conn.statements.append((normalized, params))
        for needle, exc in self._conn.raise_on.items():
            if needle in normalized:
                raise exc
        self._rows = []
        self.rowcount = 0
        for needle, rows, rowcount in self._conn.responses:
            if needle in normalized:
                self._rows = [_copy_row(r) for r in rows]
                self.rowcount = len(self._rows) if rowcount is None else rowcount
                break

    def fetchall(self) -> list[Any]:
        return list(self._rows)

    def fetchone(self) -> Optional[Any]:
        return self._rows[0] if self._rows else None


def _copy_row(row: Any) -> Any:
    return dict(row) if isinstance(row, dict) else tuple(row)


class FakeConn:
    def __init__(self) -> None:
        self.closed = False
        self.autocommit = False
        self.commits = 0
        self.rollbacks = 0
        self.statements: list[tuple[str, Any]] = []
        # Each batch handed to execute_values: (normalized sql, list of row tuples).
        self.values_calls: list[tuple[str, list[Any]]] = []
        # (needle, rows to return, rowcount override or None), matched in order.
        self.responses: list[tuple[str, Any, Optional[int]]] = []
        # needle -> exception raised when a statement containing needle runs.
        self.raise_on: dict[str, BaseException] = {}
        self.cursors: list[FakeCursor] = []
        # DSNs passed to connect() when this conn is handed out by FakeConnFactory.
        self.connect_dsns: list[str] = []

    def respond(self, needle: str, rows: Any = (), rowcount: Optional[int] = None) -> None:
        self.responses.append((needle, list(rows), rowcount))

    def cursor(self, cursor_factory: Any = None, **_: Any) -> FakeCursor:
        cur = FakeCursor(self, cursor_factory)
        self.cursors.append(cur)
        return cur

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    # psycopg2 semantics: `with conn:` rolls back on error and does not close.
    def __enter__(self) -> "FakeConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            self.rollback()
        return False

    def execute_values(
        self, cur: Any, sql: str, argslist: Any, template: Any = None,
        page_size: int = 100, fetch: bool = False,
    ) -> None:
        normalized = " ".join(sql.split())
        rows = [tuple(r) if isinstance(r, (list, tuple)) else r for r in argslist]
        for needle, exc in self.raise_on.items():
            if needle in normalized:
                raise exc
        self.values_calls.append((normalized, rows))

    # ── Query helpers for assertions ──────────────────────────────────────

    def values_for(self, table_fragment: str) -> list[list[Any]]:
        """Row lists of every execute_values batch whose SQL mentions table_fragment."""
        return [rows for sql, rows in self.values_calls if table_fragment in sql]

    def statements_containing(self, fragment: str) -> list[tuple[str, Any]]:
        return [(sql, params) for sql, params in self.statements if fragment in sql]


class FakeConnFactory:
    """Callable that stands in for psycopg2.connect and records the DSN on the conn."""

    def __init__(self, conn: FakeConn) -> None:
        self.conn = conn

    def __call__(self, dsn: str, *args: Any, **kwargs: Any) -> FakeConn:
        self.conn.connect_dsns.append(dsn)
        return self.conn
