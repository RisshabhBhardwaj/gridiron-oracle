"""Forward (pre-kickoff) feature rows are built from real prior games.

build_forward_week builds each candidate's row with build_feature_row, the builder
the historical path uses, from only what is known before the target game: the
season's completed games with week < target that kicked off before as_of, for the
player and for the league (opponent and positional context). Profile values come
from the latest profile at or before the season. Game context (lines, rest,
venue) comes from the target game. A missing target line resolves through the
causal line history (latest snapshot known before as-of and kickoff, else a
pregame-Elo implied line — decision 0008, item 3), never a median, because
training never saw a missing line.

These tests run without a database. A fake connection serves the guard, roster,
profile, history, and snapshot SELECTs, and the upsert and post-upsert fills are
captured or stubbed. The fake ignores WHERE clauses on purpose, so the
Python-side cut is what the tests exercise.
"""

from __future__ import annotations

import math
from dataclasses import fields as dc_fields
from datetime import date, datetime, timedelta, timezone

import pytest

import pipeline.feature_engineer as fe_mod
from ml.team_elo import TeamEloSystem
from pipeline.feature_engineer import FeatureEngineer
from pipeline.features.buckets import build_season_feature_context
from pipeline.features.feature_row import FeatureRow
from pipeline.line_snapshots import elo_implied_line

UTC = timezone.utc
SEASON = 2026
TARGET_WEEK = 3
# Week 2 kicked off Sep 13 and week 3 kicks off Sep 20, so Sep 17 sits between them.
AS_OF = datetime(2026, 9, 17, tzinfo=UTC)
PRIOR_TOTAL = 40.0
TARGET_TOTAL = 50.0
TARGET_SPREAD = -3.0
TARGET_PID = "P_H0"
BIRTH = date(1998, 5, 1)


def _kickoff(week: int) -> datetime:
    return datetime(2026, 9, 6, 17, tzinfo=UTC) + timedelta(days=7 * (week - 1))


def _games(week: int, *, target_total: float | None = TARGET_TOTAL,
           target_spread: float | None = TARGET_SPREAD) -> list[dict]:
    """Sixteen games. Game 0 (H0 vs A0) is a dome game with a short week in prior weeks."""
    games = []
    for g in range(16):
        game = {
            "game_id": f"{SEASON}_{week:02d}_H{g}_A{g}", "home_team": f"H{g}", "away_team": f"A{g}",
            "roof": "outdoors", "surface": "grass", "total_line": 47.5, "spread_line": -3.0,
            "home_rest": 7, "away_rest": 7, "kickoff_at": _kickoff(week),
        }
        if g == 0 and week == TARGET_WEEK:
            game.update(total_line=target_total, spread_line=target_spread)
        elif g == 0:
            game.update(roof="dome", surface="turf", home_rest=6, total_line=PRIOR_TOTAL, spread_line=-7.0)
        games.append(game)
    return games


def _teams() -> list[str]:
    return [f"H{g}" for g in range(16)] + [f"A{g}" for g in range(16)]


# Two receivers per team. P_ has 6 targets a game, P2_ has 3. Yardage is set so the
# target-week rows are recognisable: a 999 for P_H0 and a 400 for P2_H0 in week 3.
_YARDS = {("P_H0", 1): 80.0, ("P_H0", 2): 60.0, ("P_H0", 3): 999.0, ("P_H0", 4): 500.0, ("P2_H0", 3): 400.0}


def _stat_row(pid: str, team: str, opponent: str, week: int, game: dict, targets: int) -> dict:
    yards = _YARDS.get((pid, week), 30.0 if pid.startswith("P2_") else 50.0)
    return {
        "player_id": pid, "game_id": game["game_id"], "season": SEASON, "week": week,
        "position": "WR", "team": team, "opponent_team": opponent,
        "receiving_yards": yards, "receiving_tds": 0, "targets": targets, "receptions": 4,
        "target_share": 0.25, "air_yards_share": 0.2, "fantasy_points_ppr": yards / 10 + 4,
        "carries": 0, "rushing_yards": 0.0, "rushing_tds": 0, "attempts": 0, "passing_yards": 0.0,
        "passing_tds": 0, "completions": 0, "passing_interceptions": 0, "rushing_fumbles": 0,
        "receiving_fumbles": 0, "sack_fumbles": 0, "passing_cpoe": None, "passing_epa": None,
        "receiving_epa": None, "rushing_epa": None, "racr": None, "wopr": None,
        "passing_air_yards": None, "receiving_yards_after_catch": 40.0, "offense_pct": 70.0,
        "home_team": game["home_team"], "away_team": game["away_team"], "roof": game["roof"],
        "surface": game["surface"], "temp": None, "wind": None, "total_line": game["total_line"],
        "spread_line": game["spread_line"], "home_rest": game["home_rest"], "away_rest": game["away_rest"],
        "precipitation_bucket": None, "kickoff_at": game["kickoff_at"],
        "player_height": 74.0, "player_weight": 200.0, "draft_round": 2, "entry_year": 2020,
        "birth_date": BIRTH,
        "opp_zone_pct": None, "opp_man_pct": None, "opp_blitz_rate": None, "opp_pressure_rate_pbp": None,
    }


def _season_rows(weeks: tuple[int, ...] = (1, 2, 3, 4)) -> list[dict]:
    """Every game-log row of the season, including the target week and later weeks."""
    rows = []
    for week in weeks:
        for game in _games(week):
            for team, opponent in ((game["home_team"], game["away_team"]), (game["away_team"], game["home_team"])):
                rows.append(_stat_row(f"P_{team}", team, opponent, week, game, 6))
                rows.append(_stat_row(f"P2_{team}", team, opponent, week, game, 3))
    return rows


def _candidates(week: int = TARGET_WEEK, *, target_total: float | None = TARGET_TOTAL,
                target_spread: float | None = TARGET_SPREAD, extra: tuple = ()) -> list[dict]:
    rows = []
    for game in _games(week, target_total=target_total, target_spread=target_spread):
        for team in (game["home_team"], game["away_team"]):
            for pid in (f"P_{team}", f"P2_{team}"):
                rows.append({
                    "game_id": game["game_id"], "home_team": game["home_team"], "away_team": game["away_team"],
                    "roof": game["roof"], "surface": game["surface"], "total_line": game["total_line"],
                    "spread_line": game["spread_line"], "home_rest": game["home_rest"],
                    "away_rest": game["away_rest"], "kickoff_at": game["kickoff_at"],
                    "player_id": pid, "position": "WR", "team": team,
                    "birth_date": BIRTH, "entry_year": 2020, "draft_round": 2, "draft_number": 40,
                    "player_height": 74.0, "player_weight": 200.0,
                })
    return rows + list(extra)


def _profiles(extra: tuple = ()) -> list[dict]:
    rows = [
        {"player_id": pid, "effective_season": 2025, "height_inches": 74.0, "weight_lbs": 200.0}
        for team in _teams() for pid in (f"P_{team}", f"P2_{team}")
    ]
    return rows + list(extra)


class _FakeConn:
    def __init__(self, candidates: list[dict], season_rows: list[dict], profiles: list[dict],
                 snapshots: list[dict] | None = None, *, played: bool = False) -> None:
        self.candidates = candidates
        self.season_rows = season_rows
        self.profiles = profiles
        self.snapshots = snapshots or []
        self.played = played
        self.statements: list[str] = []

    def cursor(self, **_kwargs: object) -> "_Cursor":
        return _Cursor(self)


class _Cursor:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn
        self._rows: list[dict] = []

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: object = None) -> None:
        self._conn.statements.append(sql)
        if "feature_matrix" in sql:
            raise AssertionError("forward rows must not read feature_matrix")
        if "AS played" in sql:
            self._rows = [{"played": self._conn.played}]
        elif "FROM depth_charts" in sql:
            self._rows = [dict(row) for row in self._conn.candidates]
        elif "FROM player_season_profiles" in sql:
            self._rows = [dict(row) for row in self._conn.profiles]
        elif "FROM game_logs" in sql:
            self._rows = [dict(row) for row in self._conn.season_rows]
        elif "game_line_snapshots" in sql:
            assert isinstance(params, (list, tuple))
            game_id, as_of = params[0], params[1]
            kickoff = params[2] if len(params) > 2 else None
            eligible = [
                r for r in self._conn.snapshots
                if r["game_id"] == game_id
                and r["captured_at"] < as_of
                and (kickoff is None or r["captured_at"] < kickoff)
            ]
            eligible.sort(key=lambda r: r["captured_at"], reverse=True)
            self._rows = [eligible[0]] if eligible else []
        else:
            raise AssertionError(f"unexpected SQL: {sql.strip()[:80]}")

    def fetchall(self) -> list[dict]:
        return list(self._rows)

    def fetchone(self) -> dict | None:
        return self._rows[0] if self._rows else None


def _run_forward(*, week: int = TARGET_WEEK, as_of: datetime = AS_OF, candidates: list[dict] | None = None,
                 season_rows: list[dict] | None = None, profiles: list[dict] | None = None,
                 snapshots: list[dict] | None = None, played: bool = False,
                 line_elo: TeamEloSystem | None = None,
                 ) -> tuple[dict[str, FeatureRow], _FakeConn]:
    fe = FeatureEngineer("postgresql://unused")
    conn = _FakeConn(
        _candidates(week) if candidates is None else candidates,
        _season_rows() if season_rows is None else season_rows,
        _profiles() if profiles is None else profiles,
        snapshots, played=played,
    )
    fe._conn = conn
    captured: list[FeatureRow] = []
    fe._upsert_feature_rows = lambda rows: (captured.extend(rows), len(rows))[1]
    fe._fill_prior_routes_run = lambda season: None
    fe._fill_prior_pbp_ngs_features = lambda season: None
    fe._fill_lagged_depth_chart_rank = lambda season: None
    written = fe.build_forward_week(season=SEASON, week=week, as_of=as_of,
                                    line_elo=line_elo or TeamEloSystem())
    assert written == len(captured)
    return {row.player_id: row for row in captured}, conn


def _as_dict(row: FeatureRow) -> dict:
    return {f.name: getattr(row, f.name) for f in dc_fields(FeatureRow)}


def _historical_rows(pids: list[str]) -> dict[str, FeatureRow]:
    """The historical builder's week-TARGET_WEEK row, computed over the full season.

    The full season includes the target week and later, so any use of future games
    by the historical path would show up as a difference from the forward row.
    """
    rows = _season_rows()
    game_map: dict[str, dict] = {}
    for row in rows:
        game_map.setdefault(row["game_id"], {key: row[key] for key in (
            "home_team", "away_team", "roof", "surface", "temp", "wind",
            "total_line", "spread_line", "home_rest", "away_rest", "precipitation_bucket",
        )})
    league = build_season_feature_context(rows)
    out: dict[str, FeatureRow] = {}
    for pid in pids:
        player_rows = sorted((r for r in rows if r["player_id"] == pid), key=lambda r: r["week"])
        built = fe_mod._compute_player_features(pid, player_rows, game_map, rows, league)
        out[pid] = next(fr for fr in built if fr.week == TARGET_WEEK)
    return out


def _differences(forward: FeatureRow, historical: FeatureRow) -> dict:
    out = {}
    for f in dc_fields(FeatureRow):
        if f.name.startswith("actual_") or f.name == "as_of":
            continue
        a, b = getattr(forward, f.name), getattr(historical, f.name)
        same = a == b or (
            isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        )
        if not same:
            out[f.name] = (a, b)
    return out


class TestPriorGamesOfTheSeason:
    def test_rolling_features_come_from_games_earlier_this_season(self) -> None:
        # No carried feature_matrix row: the player's own week 1 and week 2 games
        # feed the row.
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["seas_games_played"] == 2
        assert row["seas_avg_receiving_yards"] == 70.0  # (80 + 60) / 2
        assert row["prior_snap_share"] == 0.7

    def test_target_week_game_is_not_used(self) -> None:
        # The fixture holds a week-3 log of 999 yards for the target player. It must not count.
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["seas_games_played"] == 2
        assert row["seas_avg_receiving_yards"] == 70.0

    def test_game_after_as_of_is_not_used(self) -> None:
        season = _season_rows()
        for row in season:
            if row["week"] == 2:
                row["kickoff_at"] = datetime(2026, 9, 18, 17, tzinfo=UTC)
        rows, _ = _run_forward(season_rows=season)
        row = _as_dict(rows[TARGET_PID])
        assert row["seas_games_played"] == 1
        assert row["seas_avg_receiving_yards"] == 80.0

    def test_prior_week_without_kickoff_fails_closed(self) -> None:
        season = _season_rows()
        for row in season:
            if row["week"] == 2:
                row["kickoff_at"] = None
        with pytest.raises(ValueError, match="no kickoff_at"):
            _run_forward(season_rows=season)

    def test_opponent_stats_come_from_the_opponents_prior_games(self) -> None:
        # A0 allowed P_H0 and P2_H0 in weeks 1 and 2: 80 + 30 then 60 + 30 yards.
        # The week-3 rows (400 and 999 yards) must not count.
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["opp_avg_receiving_yards_allowed"] == 100.0
        assert row["opp_avg_targets_allowed"] is not None

    def test_positional_rank_uses_teammates_prior_games(self) -> None:
        rows, _ = _run_forward()
        assert rows[TARGET_PID].team_pos_rank == 1.0  # 12 prior targets vs 6
        assert rows["P2_H0"].team_pos_rank == 2.0

    def test_no_row_from_feature_matrix_is_read(self) -> None:
        _, conn = _run_forward()
        assert conn.statements
        assert not any("feature_matrix" in sql for sql in conn.statements)


class TestNoPriorGames:
    def test_week_one_gets_the_builders_first_season_defaults(self) -> None:
        # Every candidate is in this state in week 1. The values are the builder's
        # own first-season defaults, the same ones training gives a week-1 row.
        rows, _ = _run_forward(week=1, as_of=datetime(2026, 9, 1, tzinfo=UTC), candidates=_candidates(1))
        row = _as_dict(rows[TARGET_PID])
        assert row["seas_games_played"] == 0
        assert row["seas_avg_receiving_yards"] is None
        assert row["games_missed_streak"] == 0
        assert row["team_pos_rank"] == 1.0
        assert row["career_games"] == 0.0
        assert row["prior_snap_share"] is None

    def test_candidate_with_no_games_so_far_gets_the_debut_values(self) -> None:
        # A player with no game log yet this season gets the value the builder
        # gives a debut row in the same week: missed weeks so far, and last
        # place among teammates with games (their rank is teammates + 1).
        extra = tuple(dict(row, player_id="P_NEW") for row in _candidates() if row["team"] == "H5")[:1]
        rows, _ = _run_forward(candidates=_candidates(extra=extra))
        row = _as_dict(rows["P_NEW"])
        assert row["seas_games_played"] == 0
        assert row["games_missed_streak"] == TARGET_WEEK - 1
        assert row["team_pos_rank"] == 3.0  # after P_H5 (rank 1) and P2_H5 (rank 2)


class TestCausalParityWithHistoricalBuilder:
    def test_forward_rows_equal_historical_rows_for_the_same_games(self) -> None:
        pids = [TARGET_PID, "P2_H0", "P_A0", "P_H5"]
        forward, _ = _run_forward()
        historical = _historical_rows(pids)
        for pid in pids:
            assert _differences(forward[pid], historical[pid]) == {}, pid


class TestTargetGameContext:
    def test_lines_rest_and_venue_come_from_the_target_game(self) -> None:
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["game_total_line"] == TARGET_TOTAL
        assert row["spread_line"] == TARGET_SPREAD
        assert row["days_rest"] == 7
        assert row["is_short_week"] == 0
        assert row["is_bye_prior"] == 0
        assert row["is_dome"] == 0
        assert row["surface_turf"] == 0
        assert row["is_home"] == 1
        assert row["as_of"] == AS_OF

    def test_prior_games_context_is_not_carried(self) -> None:
        # Prior H0 games were dome games on turf with 6 days of rest and a 40-point total.
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["game_total_line"] != PRIOR_TOTAL
        assert row["is_dome"] != 1


class TestMissingLineFallback:
    def test_missing_line_takes_snapshot_before_as_of(self) -> None:
        game_id = f"{SEASON}_{TARGET_WEEK:02d}_H0_A0"
        snapshots = [
            {"game_id": game_id, "spread_line": -4.5, "total_line": 47.5,
             "captured_at": datetime(2026, 9, 8, tzinfo=UTC)},
            {"game_id": game_id, "spread_line": -9.5, "total_line": 50.0,
             "captured_at": datetime(2026, 9, 18, tzinfo=UTC)},  # after as-of: ignored
        ]
        rows, _ = _run_forward(candidates=_candidates(target_total=None, target_spread=None),
                               snapshots=snapshots)
        assert rows[TARGET_PID].game_total_line == 47.5
        assert rows[TARGET_PID].spread_line == -4.5

    def test_missing_line_without_snapshot_is_elo_implied(self) -> None:
        elo = TeamEloSystem()
        rows, _ = _run_forward(candidates=_candidates(target_total=None, target_spread=None),
                               line_elo=elo)
        expected = elo_implied_line("H0", "A0", SEASON, TARGET_WEEK, elo)
        assert rows[TARGET_PID].game_total_line == expected.total_line
        assert rows[TARGET_PID].spread_line == expected.spread_line
        assert rows[TARGET_PID].spread_line != 2.5

    def test_present_line_is_not_replaced(self) -> None:
        rows, _ = _run_forward(candidates=_candidates(target_total=52.5))
        assert rows[TARGET_PID].game_total_line == 52.5

    def test_no_fixed_median_fallback_remains(self) -> None:
        assert not hasattr(fe_mod, "_FWD_LINE_FALLBACK")


class TestColumnSources:
    def test_disabled_columns_are_null_on_forward_rows(self) -> None:
        # Training clears these to NULL. Weather and injury stay NULL as well, so
        # forward rows do not show the model a source it never trained on.
        rows, _ = _run_forward()
        fields = {f.name for f in dc_fields(FeatureRow)}
        disabled = sorted(fe_mod._DISABLED_FEATURE_COLUMNS & fields)
        assert disabled, "the disabled set must overlap the row fields"
        for pid, row in rows.items():
            for name in disabled:
                assert getattr(row, name) is None, (pid, name)

    def test_target_game_columns_are_model_fields(self) -> None:
        assert fe_mod._FWD_TARGET_GAME_COLS <= set(fe_mod._FM_COLS)


class TestProfileValues:
    def test_height_and_weight_are_the_latest_profile_at_or_before_the_season(self) -> None:
        future = {"player_id": TARGET_PID, "effective_season": 2027, "height_inches": 80.0, "weight_lbs": 250.0}
        rows, _ = _run_forward(profiles=_profiles(extra=(future,)))
        assert rows[TARGET_PID].height == 74.0
        assert rows[TARGET_PID].weight == 200.0

    def test_age_and_experience_come_from_the_player_dates(self) -> None:
        rows, _ = _run_forward()
        row = _as_dict(rows[TARGET_PID])
        assert row["age"] == 28.0  # 2026 - 1998
        assert row["years_exp"] == 6.0  # 2026 - 2020
        assert row["exp_bucket"] == 2.0
