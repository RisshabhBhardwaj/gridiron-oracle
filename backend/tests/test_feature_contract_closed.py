"""The as-of contract fails closed on unknown feature names.

An unclassified name must raise instead of silently joining PRIOR_GAMES:
a new feature earns its classification in ml/feature_contract.py before
any model frame may carry it. Without the fix, ``_spec_for`` returned a
PRIOR_GAMES spec for every unknown name and the tests below fail.
"""

from __future__ import annotations

import pytest

from ml.feature_contract import (
    AsOf,
    FORBIDDEN_MODEL_FIELDS,
    _spec_for,
    assert_model_frame_contract,
    assert_model_input_columns,
    build_feature_registry,
)
from ml.utils import FEATURE_COLS, FEATURE_GROUPS


class TestUnknownNamesFailClosed:
    def test_spec_for_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="no as-of classification"):
            _spec_for("brand_new_feature")

    def test_spec_for_typo_of_legal_name_raises(self) -> None:
        with pytest.raises(KeyError, match="no as-of classification"):
            _spec_for("kalman_est_reciving_yards")

    def test_build_registry_rejects_unknown(self) -> None:
        with pytest.raises(KeyError, match="no as-of classification"):
            build_feature_registry([*FEATURE_COLS, "mystery_column"])

    def test_build_registry_rejects_forbidden(self) -> None:
        with pytest.raises(AssertionError, match="Illegal features"):
            build_feature_registry([*FEATURE_COLS, "fantasy_points_exp"])

    def test_assert_inputs_rejects_forbidden(self) -> None:
        with pytest.raises(AssertionError, match="prohibited feature"):
            assert_model_input_columns([*FEATURE_COLS, "snap_pct_off"], consumer="test")


class TestExplicitClassification:
    def test_every_served_column_is_explicit(self) -> None:
        registry = build_feature_registry(FEATURE_COLS)
        assert set(registry) == set(FEATURE_COLS)

    def test_legal_group_columns_are_explicit(self) -> None:
        names = {c for g in FEATURE_GROUPS.values() for c in g} - set(FORBIDDEN_MODEL_FIELDS)
        registry = build_feature_registry(names)
        assert set(registry) == names

    def test_expected_source_classes(self) -> None:
        assert _spec_for("kalman_est_targets").as_of is AsOf.PRIOR_GAMES
        assert _spec_for("prior_epa_per_play").as_of is AsOf.PRIOR_GAMES
        assert _spec_for("elo_implied_win_prob").as_of is AsOf.PRIOR_GAMES
        assert _spec_for("team_pace").as_of is AsOf.PRIOR_GAMES
        # Pace/script expectations mix prior pace with the target-game line,
        # so the line's pregame provenance governs them.
        assert _spec_for("expected_pass_attempts").as_of is AsOf.PREGAME
        assert _spec_for("neutral_script_flag").as_of is AsOf.PREGAME
        assert _spec_for("spread_line").as_of is AsOf.PREGAME
        assert _spec_for("height").as_of is AsOf.SEASON_PROFILE
        assert _spec_for("age").as_of is AsOf.SEASON_PROFILE

    def test_contemporaneous_xfp_is_illegal(self) -> None:
        for name in (
            "fantasy_points_exp",
            "rec_fantasy_points_exp",
            "targets_exp",
            "receiving_yards_exp",
        ):
            assert name in FORBIDDEN_MODEL_FIELDS
            assert _spec_for(name).as_of is AsOf.ILLEGAL

    def test_model_frame_contract_still_guards_columns(self) -> None:
        import pandas as pd

        frame = pd.DataFrame({c: [1.0] for c in FEATURE_COLS})
        assert_model_frame_contract(frame, FEATURE_COLS, consumer="test")
        with pytest.raises(AssertionError, match="missing registered features"):
            assert_model_frame_contract(frame, [*FEATURE_COLS, "mystery_column"], consumer="test")
