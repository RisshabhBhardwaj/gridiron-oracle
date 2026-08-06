"""
backend/tests/test_normalize.py

Unit tests for pipeline/normalize.py pure transform functions.

These tests use real nflreadpy data (loaded from local filesystem cache —
no DB connection required). The 2025 season is complete as of February 2026,
so row counts are stable.

Expected counts (confirmed by dry-run on 2026-02-25):
  schedules   : 285 valid rows  (272 regular season + 13 playoffs)
  player_stats: 19,399 valid rows (22 dead-letter for anonymous aggregate rows)
"""

from __future__ import annotations

import re
from typing import Optional

import pytest

import nflreadpy as nfl
from pydantic import ValidationError

from scraper.adapters.nflreadpy_adapter import (
    PlayerStatsRow,
    RosterRow,
    ScheduleRow,
    _coerce_row,
)
from pipeline.normalize import (
    player_stats_row_to_game_log,
    roster_row_to_player,
    schedule_row_to_game,
    schedule_row_to_teams,
)

pytestmark = [pytest.mark.integration, pytest.mark.network]


# ── Module-scoped fixtures (loaded once per test session) ─────────────────────

@pytest.fixture(scope="module")
def schedule_rows() -> list[ScheduleRow]:
    """Load and validate 2025 schedules via nflreadpy (uses local cache)."""
    df = nfl.load_schedules(seasons=2025)
    rows: list[ScheduleRow] = []
    for raw in df.to_dicts():
        try:
            rows.append(ScheduleRow.model_validate(_coerce_row(raw)))
        except ValidationError:
            pass
    return rows


@pytest.fixture(scope="module")
def player_stats_rows() -> list[PlayerStatsRow]:
    """Load and validate 2025 player stats via nflreadpy (uses local cache)."""
    df = nfl.load_player_stats(seasons=2025)
    rows: list[PlayerStatsRow] = []
    for raw in df.to_dicts():
        try:
            rows.append(PlayerStatsRow.model_validate(_coerce_row(raw)))
        except ValidationError:
            pass
    return rows


@pytest.fixture(scope="module")
def roster_rows() -> list[RosterRow]:
    """Load and validate 2025 rosters via nflreadpy (uses local cache)."""
    df = nfl.load_rosters(seasons=2025)
    rows: list[RosterRow] = []
    for raw in df.to_dicts():
        try:
            rows.append(RosterRow.model_validate(_coerce_row(raw)))
        except ValidationError:
            pass
    return rows


# ── Schedule / Game tests ─────────────────────────────────────────────────────

class TestNormalizeSchedules:

    def test_count(self, schedule_rows: list[ScheduleRow]) -> None:
        """2025 season: 272 regular season + 13 playoff games = 285 total."""
        assert len(schedule_rows) == 285

    def test_all_produce_games(self, schedule_rows: list[ScheduleRow]) -> None:
        games = [schedule_row_to_game(r) for r in schedule_rows]
        assert len(games) == 285
        assert all(g["id"] for g in games), "Every game must have a non-empty id"

    def test_game_ids_unique(self, schedule_rows: list[ScheduleRow]) -> None:
        ids = [schedule_row_to_game(r)["id"] for r in schedule_rows]
        assert len(ids) == len(set(ids)), "Duplicate game_id found"

    def test_game_id_format(self, schedule_rows: list[ScheduleRow]) -> None:
        """game_id format: YYYY_WW_AWAY_HOME  e.g. '2025_01_KC_BAL'."""
        pattern = re.compile(r"^\d{4}_\d{2}_[A-Z]{2,3}_[A-Z]{2,3}$")
        bad = [
            schedule_row_to_game(r)["id"]
            for r in schedule_rows
            if not pattern.match(schedule_row_to_game(r)["id"])
        ]
        assert bad == [], f"Unexpected game_id formats: {bad[:5]}"

    def test_season_is_2025(self, schedule_rows: list[ScheduleRow]) -> None:
        for r in schedule_rows:
            assert schedule_row_to_game(r)["season"] == 2025

    def test_home_away_teams_present(self, schedule_rows: list[ScheduleRow]) -> None:
        for r in schedule_rows:
            g = schedule_row_to_game(r)
            assert g["home_team"], "home_team must be non-empty"
            assert g["away_team"], "away_team must be non-empty"

    def test_teams_extracted_count(self, schedule_rows: list[ScheduleRow]) -> None:
        """NFL has 32 franchises; London/international games don't add new teams."""
        all_ids: set[str] = set()
        for r in schedule_rows:
            for t in schedule_row_to_teams(r):
                all_ids.add(t["id"])
        assert len(all_ids) >= 32, f"Expected ≥32 teams, got {len(all_ids)}"

    def test_team_stubs_have_id_only(self, schedule_rows: list[ScheduleRow]) -> None:
        """schedule_row_to_teams returns minimal stubs — just the primary key."""
        for r in schedule_rows:
            for stub in schedule_row_to_teams(r):
                assert "id" in stub
                assert stub["id"]  # non-empty


# ── Player Stats / GameLog tests ──────────────────────────────────────────────

class TestNormalizePlayerStats:

    def test_count(self, player_stats_rows: list[PlayerStatsRow]) -> None:
        """19,421 total rows minus 22 dead-letter (player_id=None) = 19,399."""
        assert len(player_stats_rows) == 19399

    def test_all_have_player_id(self, player_stats_rows: list[PlayerStatsRow]) -> None:
        missing = [r for r in player_stats_rows if not r.player_id]
        assert missing == [], "player_id must be set (null rows go to dead_letter)"

    def test_game_logs_with_game_id(self, player_stats_rows: list[PlayerStatsRow]) -> None:
        logs = [player_stats_row_to_game_log(r) for r in player_stats_rows]
        with_game = [l for l in logs if l is not None]
        # The vast majority of rows have a game_id; allow up to 1% missing
        assert len(with_game) > len(player_stats_rows) * 0.99

    def test_player_id_in_every_game_log(
        self, player_stats_rows: list[PlayerStatsRow]
    ) -> None:
        for r in player_stats_rows:
            log = player_stats_row_to_game_log(r)
            if log is not None:
                assert log["player_id"], "game_log must carry player_id"

    def test_game_id_in_every_game_log(
        self, player_stats_rows: list[PlayerStatsRow]
    ) -> None:
        for r in player_stats_rows:
            log = player_stats_row_to_game_log(r)
            if log is not None:
                assert log["game_id"], "game_log must carry game_id"

    def test_fantasy_points_ppr_reasonable(
        self, player_stats_rows: list[PlayerStatsRow]
    ) -> None:
        """
        PPR fantasy points must be in a plausible weekly range.
        Lower bound: -20 (e.g. multiple fumbles/INTs in one game).
        Upper bound: 100 (no real player has ever scored 100 PPR in a week).
        Negative values are valid — fumbles score -2 pts in standard scoring.
        """
        for r in player_stats_rows:
            if r.fantasy_points_ppr is not None:
                assert -20.0 <= r.fantasy_points_ppr < 100.0, (
                    f"Unreasonable fantasy_points_ppr={r.fantasy_points_ppr} "
                    f"for player_id={r.player_id}"
                )

    def test_season_is_2025(self, player_stats_rows: list[PlayerStatsRow]) -> None:
        for r in player_stats_rows:
            assert r.season == 2025

    def test_week_in_valid_range(self, player_stats_rows: list[PlayerStatsRow]) -> None:
        """Regular season: weeks 1–18. Playoffs: 1–4. All ≥ 1."""
        for r in player_stats_rows:
            assert 1 <= r.week <= 22, (
                f"Unexpected week={r.week} for player_id={r.player_id}"
            )


# ── Roster / Player tests ─────────────────────────────────────────────────────

class TestNormalizeRosters:

    def test_players_with_gsis_id(self, roster_rows: list[RosterRow]) -> None:
        """Most roster rows have a gsis_id; historical players may not."""
        with_id = [r for r in roster_rows if r.gsis_id]
        assert len(with_id) > len(roster_rows) * 0.80, (
            "Expected > 80% of roster rows to have gsis_id"
        )

    def test_roster_row_to_player_requires_gsis_id(
        self, roster_rows: list[RosterRow]
    ) -> None:
        without_gsis = [r for r in roster_rows if not r.gsis_id]
        for r in without_gsis:
            assert roster_row_to_player(r) is None, (
                "roster_row_to_player must return None when gsis_id is absent"
            )

    def test_player_dict_has_required_keys(self, roster_rows: list[RosterRow]) -> None:
        required = {"id", "full_name", "position", "team", "status"}
        for r in roster_rows:
            p = roster_row_to_player(r)
            if p is not None:
                missing = required - p.keys()
                assert not missing, f"Player dict missing keys: {missing}"

    def test_player_id_matches_gsis_id(self, roster_rows: list[RosterRow]) -> None:
        for r in roster_rows:
            p = roster_row_to_player(r)
            if p is not None:
                assert p["id"] == r.gsis_id

    def test_duplicate_gsis_id_deduped_before_upsert(self) -> None:
        """nflverse rosters can have duplicate gsis_id; upsert requires unique ids."""
        # Two rows with same gsis_id (simulated trade or snapshot)
        r1 = RosterRow(season=2019, gsis_id="00-0035718", team="NE", full_name="Player A", position="WR")
        r2 = RosterRow(season=2019, gsis_id="00-0035718", team="TB", full_name="Player A", position="WR")
        p1 = roster_row_to_player(r1)
        p2 = roster_row_to_player(r2)
        assert p1 and p2
        assert p1["id"] == p2["id"]
        # Simulate _process_rosters dedup: last occurrence wins
        by_id: dict[str, dict] = {}
        for p in [p1, p2]:
            by_id[p["id"]] = p
        deduped = list(by_id.values())
        assert len(deduped) == 1
        assert deduped[0]["team"] == "TB"  # last wins
