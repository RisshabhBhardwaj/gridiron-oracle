"""Tests for scripts.materialize_stack_projections's Phase 6 (L4 scoring) wiring."""
from __future__ import annotations

import pandas as pd
import pytest

from ml.scoring import score_fantasy
from scripts.materialize_stack_projections import _attach_derived_fantasy_projection


def _row(player_id, game_id, stat, position, projection):
    return {
        "player_id": player_id, "game_id": game_id, "stat": stat,
        "position": position, "projection": projection,
    }


class TestAttachDerivedFantasyProjection:
    def test_derived_total_reconciles_exactly_to_score_fantasy(self):
        """
        A WR with receiving_yards/receptions/receiving_tds rows for the same
        game gets one derived_fantasy_projection equal to score_fantasy() on
        that exact stat dict — the plan's "reconcile exactly" verify line,
        now proven through the materializer's own composition path.
        """
        df = pd.DataFrame([
            _row("p1", "g1", "receiving_yards", "WR", 80.0),
            _row("p1", "g1", "receptions", "WR", 6.0),
            _row("p1", "g1", "receiving_tds", "WR", 1.0),
        ])
        out = _attach_derived_fantasy_projection(df)
        expected = score_fantasy({"receiving_yards": 80.0, "receptions": 6.0, "receiving_tds": 1.0})
        for value in out["derived_fantasy_projection"]:
            assert value == pytest.approx(expected, abs=1e-9)

    def test_every_row_for_a_player_game_gets_the_same_total(self):
        """derived_fantasy_projection lands on every stat row, not just one."""
        df = pd.DataFrame([
            _row("p1", "g1", "rushing_yards", "RB", 90.0),
            _row("p1", "g1", "rushing_tds", "RB", 1.0),
            _row("p1", "g1", "receptions", "RB", 2.0),
        ])
        out = _attach_derived_fantasy_projection(df)
        assert out["derived_fantasy_projection"].nunique() == 1

    def test_different_players_get_independent_totals(self):
        df = pd.DataFrame([
            _row("p1", "g1", "receiving_yards", "WR", 100.0),
            _row("p2", "g1", "receiving_yards", "WR", 20.0),
        ])
        out = _attach_derived_fantasy_projection(df)
        p1_total = out.loc[out["player_id"] == "p1", "derived_fantasy_projection"].iloc[0]
        p2_total = out.loc[out["player_id"] == "p2", "derived_fantasy_projection"].iloc[0]
        assert p1_total == pytest.approx(10.0, abs=1e-9)
        assert p2_total == pytest.approx(2.0, abs=1e-9)

    def test_fantasy_ppr_own_projection_column_is_unaffected(self):
        """
        fantasy_ppr's `projection` column (the direct regression, later
        copied verbatim into `fantasy_projection` by _load_cell) must not be
        touched by attaching derived_fantasy_projection — the two stay
        independently comparable on the same row.
        """
        df = pd.DataFrame([
            _row("p1", "g1", "fantasy_ppr", "WR", 18.5),
            _row("p1", "g1", "receiving_yards", "WR", 80.0),
            _row("p1", "g1", "receptions", "WR", 6.0),
        ])
        out = _attach_derived_fantasy_projection(df)
        ppr_row = out[out["stat"] == "fantasy_ppr"].iloc[0]
        assert ppr_row["projection"] == pytest.approx(18.5, abs=1e-9)
        # derived total from receiving_yards/receptions only (fantasy_ppr itself
        # is an unscored key in ml.scoring's SCORED_STATS)
        assert ppr_row["derived_fantasy_projection"] == pytest.approx(80 * 0.1 + 6 * 1.0, abs=1e-9)

    def test_missing_component_stats_sum_only_whats_present(self):
        """A player with only one modeled cell for their game still gets a valid (partial) total."""
        df = pd.DataFrame([_row("p1", "g1", "receiving_yards", "WR", 55.0)])
        out = _attach_derived_fantasy_projection(df)
        assert out["derived_fantasy_projection"].iloc[0] == pytest.approx(5.5, abs=1e-9)


class TestServingMatrixIncludesScoringComponents:
    def test_required_serving_cells_has_every_position_td_and_reception(self):
        """
        Regression guard for the Phase 6 baseline expansion: without these,
        score_fantasy composition silently omits every TD/reception.
        """
        from ml.artifact_manifest import REQUIRED_SERVING_CELLS

        cells = set(REQUIRED_SERVING_CELLS)
        for pos in ("RB", "TE", "WR"):
            assert ("receptions", pos) in cells
            assert ("receiving_tds", pos) in cells
        for pos in ("QB", "RB", "TE", "WR"):
            assert ("rushing_tds", pos) in cells
            assert ("fumbles", pos) in cells
        assert ("passing_tds", "QB") in cells
        assert ("interceptions", "QB") in cells
        assert ("completions", "QB") in cells
        assert len(cells) == 32
