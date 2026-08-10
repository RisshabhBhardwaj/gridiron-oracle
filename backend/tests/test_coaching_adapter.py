"""
Coaching seed validation — audit finding C-24.

The deleted `data/coaching/coaching_2026.csv` had a former wide receiver as a
defensive coordinator, two coordinators each holding the same role on two teams,
and `verify` in every row's notes. Nothing rejected it. These tests lock the
validation that now does, using fixtures that reproduce those exact shapes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ml.team_elo import _ALL_NFL_TEAMS
from scraper.adapters.coaching_adapter import (
    CANONICAL_TEAMS,
    CoachingValidationError,
    load_coaching_csv,
    validate_coaching,
)

SEASON = 2026


def _valid_frame() -> pd.DataFrame:
    """A structurally valid 32-team frame. Names are placeholders by design —
    these tests exercise validation logic, not real staff assignments."""
    return pd.DataFrame(
        [
            {
                "season": SEASON,
                "team": team,
                "head_coach": f"HC {team}",
                "offensive_coordinator": f"OC {team}",
                "defensive_coordinator": f"DC {team}",
                "dual_role": "false",
                "scheme_pass_rate_prior": 0.55,
                "source": "https://example.com/team-staff",
                "verified_by": "test",
                "verified_on": "2026-08-01",
                "notes": "",
            }
            for team in sorted(_ALL_NFL_TEAMS)
        ]
    )


def test_valid_frame_passes() -> None:
    validate_coaching(_valid_frame(), SEASON)


def test_canonical_teams_is_the_32_franchises() -> None:
    assert len(CANONICAL_TEAMS) == 32


# ── the defects the deleted seed actually had ────────────────────────────────


def test_coordinator_on_two_teams_is_rejected() -> None:
    """Kingsbury was listed at SEA and WAS; Schwartz at TEN and CLE."""
    df = _valid_frame()
    df.loc[df["team"] == "SEA", "offensive_coordinator"] = "Kliff Kingsbury"
    df.loc[df["team"] == "WAS", "offensive_coordinator"] = "Kliff Kingsbury"

    with pytest.raises(CoachingValidationError) as exc:
        validate_coaching(df, SEASON)
    assert "multiple teams" in str(exc.value)
    assert "Kliff Kingsbury" in str(exc.value)


def test_defensive_coordinator_on_two_teams_is_rejected() -> None:
    df = _valid_frame()
    df.loc[df["team"] == "TEN", "defensive_coordinator"] = "Jim Schwartz"
    df.loc[df["team"] == "CLE", "defensive_coordinator"] = "Jim Schwartz"

    with pytest.raises(CoachingValidationError, match="multiple teams"):
        validate_coaching(df, SEASON)


@pytest.mark.parametrize("placeholder", ["verify", "TBD", "TBA", "n/a", "?", "", "todo"])
def test_placeholder_names_are_rejected(placeholder: str) -> None:
    df = _valid_frame()
    df.loc[df["team"] == "NE", "defensive_coordinator"] = placeholder

    with pytest.raises(CoachingValidationError, match="placeholder"):
        validate_coaching(df, SEASON)


def test_undeclared_head_coach_dual_role_is_rejected() -> None:
    df = _valid_frame()
    df.loc[df["team"] == "DAL", "offensive_coordinator"] = df.loc[
        df["team"] == "DAL", "head_coach"
    ].item()

    with pytest.raises(CoachingValidationError, match="dual_role"):
        validate_coaching(df, SEASON)


def test_declared_dual_role_is_accepted() -> None:
    df = _valid_frame()
    mask = df["team"] == "DAL"
    df.loc[mask, "offensive_coordinator"] = df.loc[mask, "head_coach"].item()
    df.loc[mask, "dual_role"] = "true"

    validate_coaching(df, SEASON)


# ── structural checks ────────────────────────────────────────────────────────


def test_unknown_team_code_is_rejected() -> None:
    df = _valid_frame()
    df.loc[df["team"] == "LAR", "team"] = "STL"

    with pytest.raises(CoachingValidationError, match="unknown team codes"):
        validate_coaching(df, SEASON)


def test_missing_team_is_rejected() -> None:
    df = _valid_frame()
    df = df[df["team"] != "KC"]

    with pytest.raises(CoachingValidationError, match="missing teams"):
        validate_coaching(df, SEASON)


def test_duplicate_team_is_rejected() -> None:
    df = _valid_frame()
    df = pd.concat([df, df[df["team"] == "KC"]], ignore_index=True)

    with pytest.raises(CoachingValidationError, match="duplicate rows"):
        validate_coaching(df, SEASON)


@pytest.mark.parametrize("bad", [0.0, 1.0, 1.4, -0.2, "n/a"])
def test_scheme_prior_outside_unit_interval_is_rejected(bad: object) -> None:
    df = _valid_frame()
    df["scheme_pass_rate_prior"] = df["scheme_pass_rate_prior"].astype(object)
    df.loc[df["team"] == "KC", "scheme_pass_rate_prior"] = bad

    with pytest.raises(CoachingValidationError, match="scheme_pass_rate_prior"):
        validate_coaching(df, SEASON)


# ── provenance is required, not documented-and-hoped-for ─────────────────────


@pytest.mark.parametrize("column", ["source", "verified_by", "verified_on"])
def test_missing_provenance_column_is_rejected(column: str) -> None:
    df = _valid_frame().drop(columns=[column])

    with pytest.raises(CoachingValidationError, match="missing required columns"):
        validate_coaching(df, SEASON)


@pytest.mark.parametrize("column", ["source", "verified_by"])
def test_blank_provenance_value_is_rejected(column: str) -> None:
    df = _valid_frame()
    df.loc[df["team"] == "KC", column] = "verify"

    with pytest.raises(CoachingValidationError, match="placeholder"):
        validate_coaching(df, SEASON)


def test_unparseable_verified_on_is_rejected() -> None:
    df = _valid_frame()
    df.loc[df["team"] == "KC", "verified_on"] = "soon"

    with pytest.raises(CoachingValidationError, match="verified_on"):
        validate_coaching(df, SEASON)


def test_future_verified_on_is_rejected() -> None:
    df = _valid_frame()
    df.loc[df["team"] == "KC", "verified_on"] = "2099-01-01"

    with pytest.raises(CoachingValidationError, match="future"):
        validate_coaching(df, SEASON)


# ── no unverified seed ships ─────────────────────────────────────────────────


def test_no_seed_csv_is_committed() -> None:
    """A seed nobody verified must not sit in the tree looking authoritative."""
    from data.coaching import TEMPLATE_PATH, seed_path

    assert TEMPLATE_PATH.exists(), "template must ship"
    assert not seed_path(2026).exists(), (
        "an unverified coaching seed was re-added; see data/coaching/README.md"
    )


def test_missing_seed_raises_a_useful_error() -> None:
    with pytest.raises(FileNotFoundError, match="coaching_TEMPLATE.csv"):
        load_coaching_csv(2026)


def test_template_header_matches_required_columns() -> None:
    from data.coaching import TEMPLATE_PATH

    header = TEMPLATE_PATH.read_text().strip().split(",")
    for column in ("season", "team", "source", "verified_by", "verified_on"):
        assert column in header
