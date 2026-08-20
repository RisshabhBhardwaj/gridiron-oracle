import pytest

from ml.multiplicity import deflated_sharpe_ratio, effective_n_trials, require_dsr_pass


def test_effective_n_trials_is_the_product() -> None:
    assert effective_n_trials(15, 1, 4, n_feature_groups=2, n_alphas=3) == 360


def test_more_trials_raise_deflated_pvalue() -> None:
    one = deflated_sharpe_ratio(0.8, n_trials=1, n_observations=200)
    many = deflated_sharpe_ratio(0.8, n_trials=60, n_observations=200)
    assert many > one


def test_dsr_rejects_a_weak_selection() -> None:
    with pytest.raises(ValueError, match="Deflated Sharpe"):
        require_dsr_pass(0.05, n_trials=80, n_observations=40, max_pvalue=0.05)
