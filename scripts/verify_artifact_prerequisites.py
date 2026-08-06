#!/usr/bin/env python3
"""Fail artifact-backed training when required feature sources are absent.

Graceful-fallback development remains possible without credentials.  A run that
will be promoted, however, must prove that its major feature groups are present
in the database rather than silently training on their NULL fallbacks.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class Check:
    source: str
    status: str
    rows: int | None
    detail: str


def _check(cur, source: str, query: str, minimum: int = 1) -> Check:
    try:
        cur.execute(query)
        rows = int(cur.fetchone()[0] or 0)
        return Check(source, "ok" if rows >= minimum else "error", rows,
                     f"requires at least {minimum} qualifying rows")
    except Exception as exc:  # database/schema failures are prerequisite failures
        return Check(source, "error", None, str(exc))


def build_report(db_url: str) -> dict:
    import psycopg2

    conn = psycopg2.connect(db_url.replace("postgresql+asyncpg://", "postgresql://"))
    try:
        with conn.cursor() as cur:
            checks = [
                _check(cur, "pbp", "SELECT COUNT(*) FROM pbp_features"),
                _check(cur, "participation_routes", "SELECT COUNT(*) FROM participation_player_game WHERE routes_run IS NOT NULL"),
                _check(cur, "depth_chart", "SELECT COUNT(*) FROM depth_charts WHERE depth_rank IS NOT NULL"),
                _check(cur, "weather", "SELECT COUNT(*) FROM games WHERE temp IS NOT NULL AND wind IS NOT NULL"),
                _check(cur, "historical_props", "SELECT COUNT(*) FROM prop_odds WHERE game_id IS NOT NULL AND line_value IS NOT NULL AND over_odds IS NOT NULL"),
            ]
    finally:
        conn.close()
    return {"overall_status": "ok" if all(c.status == "ok" for c in checks) else "blocked",
            "checks": [asdict(c) for c in checks]}


def main() -> int:
    if os.environ.get("PRODUCT_MODE", "graceful_fallback") != "artifact_backed":
        print(json.dumps({"overall_status": "skipped", "detail": "PRODUCT_MODE is not artifact_backed"}))
        return 0
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print(json.dumps({"overall_status": "blocked", "detail": "DATABASE_URL is required"}))
        return 1
    try:
        report = build_report(db_url)
    except Exception as exc:
        report = {"overall_status": "blocked", "detail": f"database unavailable: {exc}"}
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["overall_status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
