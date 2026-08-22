"""Tests for ml.scoring.score_fantasy (Phase 6, L4)."""
from __future__ import annotations

import pytest

from ml.scoring import (
    HALF_PPR,
    SCORED_STATS,
    STANDARD_NON_PPR,
    STANDARD_PPR,
    LeagueRules,
    _STAT_TO_RULE_FIELD,
    score_fantasy,
)


class TestScoreFantasyExactReconciliation:
    """
    The plan's Phase 6 verify line: "fantasy points reconcile exactly to
    component stats." Every assertion here is exact arithmetic, not a
    tolerance — score_fantasy on a known stat dict is deterministic, so a
    loose margin would hide a real coefficient bug.
    """

    def test_wr_receiving_line_standard_ppr(self):
        # 7 receptions, 95 receiving yards, 1 TD.
        stats = {"receptions": 7, "receiving_yards": 95, "receiving_tds": 1}
        expected = 7 * 1.0 + 95 * 0.1 + 1 * 6.0
        assert score_fantasy(stats, STANDARD_PPR) == pytest.approx(expected, abs=1e-9)
        assert expected == pytest.approx(22.5, abs=1e-9)

    def test_rb_full_stat_line_with_negative_events(self):
        stats = {
            "carries": 18, "rushing_yards": 82, "rushing_tds": 1,
            "receptions": 3, "receiving_yards": 24, "receiving_tds": 0,
            "fumbles": 1,
        }
        expected = (
            18 * 0.0 + 82 * 0.1 + 1 * 6.0
            + 3 * 1.0 + 24 * 0.1 + 0 * 6.0
            + 1 * -2.0
        )
        assert score_fantasy(stats, STANDARD_PPR) == pytest.approx(expected, abs=1e-9)

    def test_qb_passing_and_rushing_line(self):
        stats = {
            "pass_attempts": 34, "completions": 24, "passing_yards": 278,
            "passing_tds": 2, "interceptions": 1,
            "carries": 5, "rushing_yards": 22, "rushing_tds": 0,
        }
        expected = (
            34 * 0.0 + 24 * 0.0 + 278 * 0.04 + 2 * 4.0 + 1 * -2.0
            + 5 * 0.0 + 22 * 0.1 + 0 * 6.0
        )
        assert score_fantasy(stats, STANDARD_PPR) == pytest.approx(expected, abs=1e-9)

    def test_half_ppr_only_changes_reception_weight(self):
        stats = {"receptions": 6, "receiving_yards": 50}
        full = score_fantasy(stats, STANDARD_PPR)
        half = score_fantasy(stats, HALF_PPR)
        assert full - half == pytest.approx(6 * 0.5, abs=1e-9)

    def test_non_ppr_zeroes_out_receptions(self):
        stats = {"receptions": 10, "receiving_yards": 0}
        assert score_fantasy(stats, STANDARD_NON_PPR) == pytest.approx(0.0, abs=1e-9)

    def test_empty_stats_scores_zero(self):
        assert score_fantasy({}, STANDARD_PPR) == 0.0

    def test_unrecognized_keys_are_ignored_not_scored(self):
        """
        A caller frequently passes a full projections row, which mixes
        scored stats with helper/composite columns (fantasy_ppr itself,
        target_share, etc.) — those must contribute 0, not raise.
        """
        stats = {"receiving_yards": 50, "target_share": 0.25, "fantasy_ppr": 999.0}
        expected = 50 * 0.1
        assert score_fantasy(stats, STANDARD_PPR) == pytest.approx(expected, abs=1e-9)

    def test_none_values_treated_as_zero_contribution(self):
        stats = {"receiving_yards": 40, "receiving_tds": None}
        assert score_fantasy(stats, STANDARD_PPR) == pytest.approx(4.0, abs=1e-9)

    def test_custom_league_rules_are_honored_exactly(self):
        custom = LeagueRules(pts_per_reception=1.5, pts_per_receiving_td=7.0)
        stats = {"receptions": 4, "receiving_tds": 1}
        assert score_fantasy(stats, custom) == pytest.approx(4 * 1.5 + 1 * 7.0, abs=1e-9)


class TestScoredStatsCoverage:
    def test_scored_stats_matches_mapping_keys(self):
        assert SCORED_STATS == frozenset(_STAT_TO_RULE_FIELD)

    def test_every_league_rules_field_is_reachable_by_some_stat(self):
        """Every scoring weight in LeagueRules must be addressable by at least one stat key."""
        mapped_fields = set(_STAT_TO_RULE_FIELD.values())
        assert mapped_fields == set(LeagueRules.__dataclass_fields__)
