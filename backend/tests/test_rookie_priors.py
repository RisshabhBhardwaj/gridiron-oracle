import pytest

from ml.rookie_priors import label_projection_basis, rookie_multiplier, vacated_opportunity_share
import pandas as pd


def test_vacated_opportunity_ignores_returning_players() -> None:
    logs = pd.DataFrame([
        {"player_id": "gone", "season": 2025, "team": "AAA", "position": "WR", "fantasy_points_ppr": 80.0},
        {"player_id": "stay", "season": 2025, "team": "AAA", "position": "WR", "fantasy_points_ppr": 20.0},
    ])
    roster = pd.DataFrame([
        {"player_id": "stay", "team": "AAA", "position": "WR"},
        {"player_id": "rook", "team": "AAA", "position": "WR"},
    ])
    shares = vacated_opportunity_share(logs, roster, season=2026)
    assert shares[("AAA", "WR")] == pytest.approx(0.8)


def test_first_round_exceeds_seventh_round() -> None:
    assert rookie_multiplier(1, 0.4) > rookie_multiplier(7, 0.4)


def test_projection_basis_labels_rookies() -> None:
    assert label_projection_basis(historical_games=0, is_rookie=True) == "rookie_draft_capital_vacated_opportunity"
    assert label_projection_basis(historical_games=12, is_rookie=False) == "historical_ppr_x_games_prior"
