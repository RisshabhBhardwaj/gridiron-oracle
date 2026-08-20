import pandas as pd

from ml.baselines import attach_marcel


def test_marcel_ignores_same_season_actuals() -> None:
    history = pd.DataFrame([
        {"player_id": "p1", "season": 2022, "week": 1, "position": "WR", "receiving_yards": 40.0},
        {"player_id": "p1", "season": 2023, "week": 1, "position": "WR", "receiving_yards": 50.0},
        {"player_id": "p1", "season": 2024, "week": 1, "position": "WR", "receiving_yards": 60.0},
        {"player_id": "p1", "season": 2025, "week": 1, "position": "WR", "receiving_yards": 900.0},
        {"player_id": "p2", "season": 2024, "week": 1, "position": "WR", "receiving_yards": 55.0},
    ])
    eval_rows = pd.DataFrame([
        {"player_id": "p1", "season": 2025, "week": 2, "position": "WR"},
    ])
    inflated = attach_marcel(eval_rows, history, stat_col="receiving_yards")
    cleaned = attach_marcel(
        eval_rows,
        history[history["season"] < 2025],
        stat_col="receiving_yards",
    )
    assert inflated["marcel_baseline"].notna().all()
    assert inflated["marcel_baseline"].iloc[0] == cleaned["marcel_baseline"].iloc[0]
    assert inflated["marcel_baseline"].iloc[0] < 200.0
