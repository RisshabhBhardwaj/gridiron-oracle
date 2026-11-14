"""A recording fake of a psycopg2 connection, for tests that must prove what was written and when.

Reads are answered by marker: a route is ``(marker, result)`` and the first route whose
marker appears in the SQL text answers. ``result`` is ``(columns, rows)`` or a callable
taking ``(sql, params)`` and returning that. Every statement, batch, commit and rollback is
appended to ``conn.log`` in order, so a test can assert "nothing was written" or "the
snapshot read came before the upsert".
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

WRITE_RE = re.compile(r"^\s*(INSERT|UPDATE|DELETE|ALTER|CREATE|DROP|TRUNCATE)\b", re.IGNORECASE)


class FakeCursor:
    def __init__(self, conn: "FakeConn") -> None:
        self.conn = conn
        self.description: Optional[list[tuple]] = None
        self._rows: list[tuple] = []
        self.rowcount = 0

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, params: Any = None) -> None:
        self.conn.log.append(("execute", sql, params))
        result = self.conn.answer(sql, params)
        if result is None:
            self.description, self._rows = None, []
            return
        columns, rows = result
        self.description = [(c,) for c in columns]
        self._rows = [tuple(r) for r in rows]
        self.rowcount = len(self._rows)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows

    def close(self) -> None:
        pass


class FakeConn:
    def __init__(self, routes: list[tuple[str, Any]] | None = None) -> None:
        self.routes = list(routes or [])
        self.log: list[tuple] = []

    def cursor(self, *args: Any, **kwargs: Any) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.log.append(("commit",))

    def rollback(self) -> None:
        self.log.append(("rollback",))

    def close(self) -> None:
        pass

    def answer(self, sql: str, params: Any):
        for marker, result in self.routes:
            if marker in sql:
                return result(sql, params) if callable(result) else result
        return None

    # ---- assertions helpers -------------------------------------------------
    @property
    def writes(self) -> list[tuple]:
        """Every statement or batch that changes data, in order."""
        out = []
        for entry in self.log:
            if entry[0] == "batch":
                out.append(entry)
            elif entry[0] == "execute" and WRITE_RE.match(entry[1]):
                out.append(entry)
        return out

    @property
    def commits(self) -> int:
        return sum(1 for e in self.log if e[0] == "commit")

    def kinds(self) -> list[str]:
        return [e[0] for e in self.log]


def install_batch_recorder(monkeypatch) -> None:
    """Make ``psycopg2.extras.execute_batch`` record into the cursor's connection log."""
    import psycopg2.extras

    def _record(cur, sql, argslist, page_size=100):
        rows = list(argslist)
        cur.conn.log.append(("batch", sql, rows))

    monkeypatch.setattr(psycopg2.extras, "execute_batch", _record)
