"""Forward (pre-kickoff) feature rows take game context from the target game.

build_forward_week takes the lines, rest, and venue of a row from the target
game, never from the player's prior-season row, which describes a different
game. Player history still comes from the prior row. A missing target line
resolves through the causal line history (latest snapshot known before as-of,
else a pregame-Elo implied line — decision 0008, item 3), never a median,
because training never saw a missing line.

These tests run without a database: a fake connection serves the SELECTs,
and the upsert and post-upsert fills are captured or stubbed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pipeline.feature_engineer as fe_mod
from ml.team_elo import TeamEloSystem
from pipeline.feature_engineer import FeatureEngineer
from pipeline.line_snapshots import elo_implied_line

PRIOR_TOTAL = 40.0
TARGET_TOTAL = 50.0
TARGET_PID = "P_H0"
AS_OF = datetime(2026, 9, 9, tzinfo=timezone.utc)


class _Cursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self._conn = conn
        self._rows: list[dict] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        if "game_line_snapshots" in sql:
            self._rows = list(self._conn.snapshots)
        elif "JOIN LATERAL" in sql:
            self._rows = list(self._conn.candidates)
        elif "FROM feature_matrix" in sql:
            self._rows = list(self._conn.priors)
        else:
            self._rows = []

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None


class _FakeConn:
    def __init__(self, candidates: list[dict], priors: list[dict],
                 snapshots: list[dict] | None = None) -> None:
        self.candidates = candidates
        self.priors = priors
        self.snapshots = snapshots or []

    def cursor(self, **_kwargs: object) -> _Cursor:
        return _Cursor(self)


def _candidates(total: float | None = TARGET_TOTAL, spread: float | None = -3.0) -> list[dict]:
    """16 games covering 32 teams (the forward-roster invariant).

    The target player is the home team (H0) in game 0. Its line is `total` and
    `spread`. Every other game is at 45.0.
    """
    rows: list[dict] = []
    for g in range(16):
        home, away = f"H{g}", f"A{g}"
        game_total = total if g == 0 else 45.0
        game_spread = spread if g == 0 else -3.0
        for team in (home, away):
            rows.append({
                "game_id": f"2026_01_{home}_{away}",
                "home_team": home, "away_team": away,
                "roof": "outdoors", "surface": "grass",
                "total_line": game_total, "spread_line": game_spread,
                "home_rest": 7, "away_rest": 7,
                "player_id": f"P_{team}", "position": "WR", "team": team,
                "player_height": 74.0, "player_weight": 200.0,
                "draft_round": 2, "draft_number": 40,
            })
    return rows


def _prior_row(**overrides: object) -> dict:
    """A feature_matrix row for the target player's last prior-season game."""
    row: dict = {name: None for name in fe_mod._FM_COLS}
    row.update({
        "player_id": TARGET_PID, "game_id": "2025_18_H0_A0", "season": 2025,
        "week": 18, "position": "WR", "team": "H0", "opponent_team": "A0",
        "is_home": 1,
        "game_total_line": PRIOR_TOTAL, "spread_line": -7.0,
        "days_rest": 6, "is_short_week": 1, "is_bye_prior": 0,
        "is_dome": 1, "surface_turf": 1,
        "kalman_est_receiving_yards": 77.0, "seas_avg_targets": 6.5,
    })
    row.update(overrides)
    return row


def _forward_row(prior: dict | None, *, total: float | None = TARGET_TOTAL,
                 spread: float | None = -3.0,
                 snapshots: list[dict] | None = None,
                 line_elo: TeamEloSystem | None = None) -> dict:
    """Run build_forward_week for one week and return the target player's row."""
    fe = FeatureEngineer("postgresql://unused")
    fe._conn = _FakeConn(_candidates(total, spread), [] if prior is None else [prior],
                         snapshots)
    captured: list = []
    fe._upsert_feature_rows = lambda rows: (captured.extend(rows), len(rows))[1]
    fe._fill_prior_routes_run = lambda season: None
    fe._fill_prior_pbp_ngs_features = lambda season: None
    fe._fill_lagged_depth_chart_rank = lambda season: None
    fe.build_forward_week(season=2026, week=1, as_of=AS_OF,
                          line_elo=line_elo or TeamEloSystem())
    (row,) = [r for r in captured if r.player_id == TARGET_PID]
    return {name: getattr(row, name) for name in fe_mod._FM_COLS}


class TestTargetGameContextWins:
    def test_target_game_lines_replace_prior_game_lines(self) -> None:
        row = _forward_row(_prior_row())
        assert row["game_total_line"] == TARGET_TOTAL
        assert row["spread_line"] == -3.0

    def test_rest_comes_from_target_game_not_prior_row(self) -> None:
        # Prior row was a short week (days_rest=6, is_short_week=1). The
        # target game has home_rest=7 and is not short.
        row = _forward_row(_prior_row())
        assert row["days_rest"] == 7
        assert row["is_short_week"] == 0
        assert row["is_bye_prior"] == 0

    def test_venue_comes_from_target_game_not_prior_row(self) -> None:
        # Prior row was a dome on turf. The target game is outdoors on grass.
        row = _forward_row(_prior_row())
        assert row["is_dome"] == 0
        assert row["surface_turf"] == 0

    def test_player_history_still_comes_from_prior_row(self) -> None:
        row = _forward_row(_prior_row())
        assert row["kalman_est_receiving_yards"] == 77.0
        assert row["seas_avg_targets"] == 6.5

    def test_player_without_prior_still_gets_target_game_context(self) -> None:
        row = _forward_row(None)
        assert row["game_total_line"] == TARGET_TOTAL
        assert row["days_rest"] == 7


class TestMissingLineFallback:
    def test_missing_line_takes_snapshot_before_as_of(self) -> None:
        # A total of None read as 0 by LightGBM would look like a 0-point game.
        snapshots = [
            {"spread_line": -4.5, "total_line": 47.5,
             "captured_at": datetime(2026, 9, 8, tzinfo=timezone.utc)},
        ]
        row = _forward_row(_prior_row(), total=None, spread=None,
                           snapshots=snapshots)
        assert row["game_total_line"] == 47.5
        assert row["spread_line"] == -4.5

    def test_missing_line_without_snapshot_is_elo_implied(self) -> None:
        elo = TeamEloSystem()
        row = _forward_row(_prior_row(), total=None, spread=None, line_elo=elo)
        expected = elo_implied_line("H0", "A0", 2026, 1, elo)
        assert row["game_total_line"] == expected.total_line
        assert row["spread_line"] == expected.spread_line
        assert row["spread_line"] != 2.5

    def test_present_line_is_not_replaced(self) -> None:
        row = _forward_row(_prior_row(), total=52.5)
        assert row["game_total_line"] == 52.5


class TestSourcePartition:
    def test_target_game_columns_are_all_real_feature_fields(self) -> None:
        assert fe_mod._FWD_TARGET_GAME_COLS <= set(fe_mod._FM_COLS)

    def test_no_fixed_median_fallback_remains(self) -> None:
        assert not hasattr(fe_mod, "_FWD_LINE_FALLBACK")
