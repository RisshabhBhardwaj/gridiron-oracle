from ml.served_learner import served_learner


def test_served_learner_discloses_lgbm_identity_cells() -> None:
    assert served_learner("fantasy_ppr", "QB") == "lgbm_identity"
    assert served_learner("fantasy_ppr", "WR") == "lgbm_identity"
    assert served_learner("receiving_yards", "WR") == "lgbm_identity"


def test_served_learner_discloses_ridge_stack_cells() -> None:
    assert served_learner("carries", "RB") == "ridge_stack"
