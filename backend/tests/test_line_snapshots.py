"""Causal line snapshots (decision 0008, item 3). DB-off tests.

Snapshots preserve what the line was before kickoff; readers take the latest
snapshot captured strictly before as-of (and before kickoff when known). With
no qualifying snapshot the line is implied by pregame Elo only — never a fixed
median. normalize appends snapshots; it never rewrites history.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

import pipeline.normalize as normalize_mod
from backend.tests._pipeline_fakes import FakeConn
from ml.team_elo import TeamEloSystem
from pipeline.feature_engineer import FeatureEngineer
from pipeline.line_snapshots import (
    GameLine,
    causal_line_for,
    elo_implied_line,
    get_line_as_of,
    snapshot_batch,
)

GAME = "2025_01_KC_BAL"
AS_OF = datetime(2025, 9, 4, 12, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2025, 9, 4, 20, 20, tzinfo=timezone.utc)


def _snap_rows(rows: list[tuple]) -> list[dict]:
    return [
        {"game_id": GAME, "spread_line": spread, "total_line": total,
         "captured_at": captured}
        for spread, total, captured in rows
    ]


class _SnapshotCursor:
    """Applies the as-of/kickoff bounds from the query params, the way
    Postgres evaluates the WHERE clause. Lets behavior tests author both
    qualifying and disqualified snapshots in one scenario."""

    def __init__(self, conn: "_SnapshotConn") -> None:
        self._conn = conn
        self._row: dict | None = None

    def __enter__(self) -> "_SnapshotCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        assert isinstance(params, (list, tuple))
        game_id, as_of = params[0], params[1]
        kickoff = params[2] if len(params) > 2 else None
        self._conn.statements.append((" ".join(sql.split()), params))
        eligible = [
            r for r in self._conn.snapshots
            if r["game_id"] == game_id
            and r["captured_at"] < as_of
            and (kickoff is None or r["captured_at"] < kickoff)
        ]
        eligible.sort(key=lambda r: r["captured_at"], reverse=True)
        self._row = eligible[0] if eligible else None

    def fetchone(self) -> dict | None:
        return self._row


class _SnapshotConn(FakeConn):
    def __init__(self, snapshots: list[dict]) -> None:
        super().__init__()
        self.snapshots = snapshots

    def cursor(self, **_kwargs: object) -> _SnapshotCursor:
        return _SnapshotCursor(self)


def _conn_with_snapshots(rows: list[dict]) -> _SnapshotConn:
    return _SnapshotConn(rows)


def _fitted_elo() -> TeamEloSystem:
    """KC dominant in 2024, BAL weak; nothing played in 2025 (all pregame)."""
    elo = TeamEloSystem()
    results = []
    for week in range(1, 6):
        results.append({
            "season": 2024, "week": week,
            "home_team": "KC", "away_team": "BAL",
            "home_score": 35, "away_score": 10,
        })
    elo.fit(pd.DataFrame(results))
    return elo


class TestGetLineAsOf:
    def test_latest_before_as_of_wins(self) -> None:
        conn = _conn_with_snapshots(_snap_rows([
            (-3.0, 46.5, datetime(2025, 9, 1, tzinfo=timezone.utc)),
            (-4.5, 47.0, datetime(2025, 9, 3, tzinfo=timezone.utc)),
            (-2.5, 44.0, datetime(2025, 8, 25, tzinfo=timezone.utc)),
        ]))
        line = get_line_as_of(conn, GAME, AS_OF)
        assert line == GameLine(spread_line=-4.5, total_line=47.0)

    def test_snapshot_at_or_after_as_of_is_ignored(self) -> None:
        conn = _conn_with_snapshots(_snap_rows([
            (-3.0, 46.5, datetime(2025, 9, 1, tzinfo=timezone.utc)),
            (-9.5, 50.0, datetime(2025, 9, 5, tzinfo=timezone.utc)),  # after as-of
            (-8.5, 49.0, AS_OF),  # exactly at as-of: strictly-before only
        ]))
        line = get_line_as_of(conn, GAME, AS_OF)
        assert line == GameLine(spread_line=-3.0, total_line=46.5)

    def test_snapshot_at_or_after_kickoff_is_never_returned(self) -> None:
        conn = _conn_with_snapshots(_snap_rows([
            (-3.0, 46.5, datetime(2025, 9, 1, tzinfo=timezone.utc)),
            (-9.5, 50.0, datetime(2025, 9, 4, 19, 0, tzinfo=timezone.utc)),
            (-10.5, 51.0, KICKOFF),  # exactly at kickoff: excluded
        ]))
        line = get_line_as_of(conn, GAME, AS_OF, kickoff_at=KICKOFF)
        assert line == GameLine(spread_line=-3.0, total_line=46.5)

    def test_no_qualifying_snapshot_returns_none(self) -> None:
        conn = _conn_with_snapshots(_snap_rows([
            (-9.5, 50.0, datetime(2025, 9, 5, tzinfo=timezone.utc)),
        ]))
        assert get_line_as_of(conn, GAME, AS_OF) is None

    def test_tuple_rows_also_read(self) -> None:
        class _TupleCursor:
            def __enter__(self) -> "_TupleCursor":
                return self

            def __exit__(self, *exc: object) -> bool:
                return False

            def execute(self, sql: str, params: object = None) -> None:
                pass

            def fetchone(self) -> tuple:
                return (-3.0, 46.5)

        class _TupleConn:
            def cursor(self, **_kwargs: object) -> _TupleCursor:
                return _TupleCursor()

        assert get_line_as_of(_TupleConn(), GAME, AS_OF) == GameLine(-3.0, 46.5)

    def test_query_uses_strict_bounds_and_latest_first(self) -> None:
        """The emitted SQL itself must exclude boundary snapshots."""
        conn = _conn_with_snapshots([])
        get_line_as_of(conn, GAME, AS_OF, kickoff_at=KICKOFF)
        (sql, params), = conn.statements
        assert sql.count("captured_at < %s") == 2
        assert "captured_at <=" not in sql
        assert "ORDER BY captured_at DESC LIMIT 1" in sql
        assert list(params) == [GAME, AS_OF, KICKOFF]


class TestEloImpliedLine:
    def test_no_snapshot_falls_back_to_elo_not_median(self) -> None:
        conn = _conn_with_snapshots([])
        line = causal_line_for(
            conn, game_id=GAME, as_of=AS_OF,
            home_team="KC", away_team="BAL", season=2025, week=1,
            elo=_fitted_elo(),
        )
        assert line.spread_line is not None and line.total_line is not None
        assert (line.spread_line, line.total_line) != (2.5, 45.0)
        # A KC side that dominated 2024 is favored at home.
        assert line.spread_line < 0.0

    def test_fallback_uses_pregame_elo_only(self) -> None:
        """Results from the target week (or later) must not move its own line.

        The week-1 line from a system fitted on later weeks must equal the
        line from a system that never saw those games.
        """
        fitted = _fitted_elo()
        pristine = TeamEloSystem()
        assert (
            elo_implied_line("KC", "BAL", 2024, 1, fitted)
            == elo_implied_line("KC", "BAL", 2024, 1, pristine)
        )

    def test_snapshot_beats_elo_when_present(self) -> None:
        conn = _conn_with_snapshots(_snap_rows([
            (-3.0, 46.5, datetime(2025, 9, 1, tzinfo=timezone.utc)),
        ]))
        line = causal_line_for(
            conn, game_id=GAME, as_of=AS_OF,
            home_team="KC", away_team="BAL", season=2025, week=1,
            elo=_fitted_elo(),
        )
        assert line == GameLine(spread_line=-3.0, total_line=46.5)

    def test_partial_snapshot_fills_missing_leg_from_elo(self) -> None:
        conn = _conn_with_snapshots(
            [{"game_id": GAME, "spread_line": -3.0, "total_line": None,
              "captured_at": datetime(2025, 9, 1, tzinfo=timezone.utc)}]
        )
        line = causal_line_for(
            conn, game_id=GAME, as_of=AS_OF,
            home_team="KC", away_team="BAL", season=2025, week=1,
            elo=_fitted_elo(),
        )
        assert line.spread_line == -3.0
        assert line.total_line is not None


class TestSnapshotBatch:
    def test_games_without_lines_contribute_nothing(self) -> None:
        assert snapshot_batch([{"id": "G1", "spread_line": None, "total_line": None}], AS_OF) == []

    def test_batch_carries_game_lines_and_capture_time(self) -> None:
        rows = snapshot_batch(
            [{"id": "G1", "spread_line": -3.0, "total_line": 46.5}], AS_OF
        )
        assert rows == [("G1", -3.0, 46.5, AS_OF, "nflreadpy_schedule")]


@pytest.fixture()
def norm_conn(monkeypatch: pytest.MonkeyPatch) -> FakeConn:
    conn = FakeConn()
    monkeypatch.setattr(normalize_mod.psycopg2, "connect", lambda *a, **k: conn)
    monkeypatch.setattr(normalize_mod, "execute_values", conn.execute_values)
    return conn


class TestNormalizeAppendsHistory:
    def test_upsert_games_appends_snapshot_batch(self, norm_conn: FakeConn) -> None:
        from pipeline.normalize import Normalizer

        norm = Normalizer("postgresql://unused")
        norm._conn = norm_conn
        norm._upsert_games([{
            "id": "G1", "season": 2025, "week": 1,
            "home_team": "KC", "away_team": "BAL",
            "spread_line": -3.0, "total_line": 46.5,
        }])
        batches = norm_conn.values_for("game_line_snapshots")
        assert len(batches) == 1
        game_id, spread, total, captured, source = batches[0][0]
        assert (game_id, spread, total) == ("G1", -3.0, 46.5)
        assert captured is not None and source == "nflreadpy_schedule"

    def test_second_upsert_appends_instead_of_overwriting(
        self, norm_conn: FakeConn
    ) -> None:
        """A line move adds a second snapshot row; the first is untouched."""
        from pipeline.normalize import Normalizer

        norm = Normalizer("postgresql://unused")
        norm._conn = norm_conn
        game = {
            "id": "G1", "season": 2025, "week": 1,
            "home_team": "KC", "away_team": "BAL",
        }
        norm._upsert_games([dict(game, spread_line=-3.0, total_line=46.5)])
        norm._upsert_games([dict(game, spread_line=-5.5, total_line=44.0)])
        batches = norm_conn.values_for("game_line_snapshots")
        assert len(batches) == 2
        assert batches[0][0][1:] == (-3.0, 46.5, batches[0][0][3], "nflreadpy_schedule")
        assert batches[1][0][1] == -5.5 and batches[1][0][2] == 44.0
        # The games upsert still overwrites the live row (unchanged behaviour).
        assert len(norm_conn.values_for("INSERT INTO games")) == 2

    def test_upsert_without_lines_writes_no_snapshot(
        self, norm_conn: FakeConn
    ) -> None:
        from pipeline.normalize import Normalizer

        norm = Normalizer("postgresql://unused")
        norm._conn = norm_conn
        norm._upsert_games([{"id": "G1", "season": 2025, "week": 1,
                             "home_team": "KC", "away_team": "BAL"}])
        assert norm_conn.values_for("game_line_snapshots") == []


class TestForwardWeekUsesCausalLines:
    """build_forward_week resolves a missing line from snapshots/Elo (DB-off)."""

    def _run(self, candidates: list[dict], priors: list[dict],
             elo: TeamEloSystem) -> dict:
        import pipeline.feature_engineer as fe_mod

        fe = FeatureEngineer("postgresql://unused")
        conn = FakeConn()
        conn.respond("AS played", [{"played": False}])
        conn.respond("FROM depth_charts", candidates)
        conn.respond("FROM player_season_profiles", [])
        conn.respond("FROM game_logs", [])
        conn.respond("FROM feature_matrix", priors)
        conn.respond("game_line_snapshots", [])
        fe._conn = conn
        captured: list = []
        fe._upsert_feature_rows = lambda rows: (captured.extend(rows), len(rows))[1]
        fe._fill_prior_routes_run = lambda season: None
        fe._fill_prior_pbp_ngs_features = lambda season: None
        fe._fill_lagged_depth_chart_rank = lambda season: None
        fe.build_forward_week(
            season=2026, week=1,
            as_of=datetime(2026, 9, 9, tzinfo=timezone.utc),
            line_elo=elo,
        )
        (row,) = [r for r in captured if r.player_id == "P_H0"]
        return {name: getattr(row, name) for name in fe_mod._FM_COLS}

    def _candidates(self, total: float | None, spread: float | None) -> list[dict]:
        rows: list[dict] = []
        for g in range(16):
            home, away = f"H{g}", f"A{g}"
            for team in (home, away):
                rows.append({
                    "game_id": f"2026_01_{home}_{away}",
                    "home_team": home, "away_team": away,
                    "roof": "outdoors", "surface": "grass",
                    "total_line": total if g == 0 else 45.0,
                    "spread_line": spread if g == 0 else -3.0,
                    "home_rest": 7, "away_rest": 7,
                    "kickoff_at": datetime(2026, 9, 10, tzinfo=timezone.utc),
                    "player_id": f"P_{team}", "position": "WR", "team": team,
                    "player_height": 74.0, "player_weight": 200.0,
                    "draft_round": 2, "draft_number": 40,
                })
        return rows

    def test_missing_line_is_elo_implied_not_median(self) -> None:
        elo = TeamEloSystem()  # pristine: every team at 1500
        row = self._run(self._candidates(None, None), [], elo)
        expected = elo_implied_line("H0", "A0", 2026, 1, elo)
        assert row["game_total_line"] == expected.total_line
        assert row["spread_line"] == expected.spread_line
        assert row["spread_line"] != 2.5

    def test_present_line_is_not_replaced(self) -> None:
        row = self._run(self._candidates(52.5, -3.0), [], TeamEloSystem())
        assert row["game_total_line"] == 52.5
        assert row["spread_line"] == -3.0
