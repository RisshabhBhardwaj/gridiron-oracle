"""Tests for ADP Spearman eval."""

from __future__ import annotations

import pandas as pd

from ml.adp_eval import _normalize_name, spearman_vs_adp


def test_spearman_perfect_agreement() -> None:
    model = pd.DataFrame({
        "name_key": ["a", "b", "c", "d", "e"],
        "position": ["WR"] * 5,
        "fantasy_ppr": [100, 80, 60, 40, 20],
    })
    adp = pd.DataFrame({
        "name_key": ["a", "b", "c", "d", "e"],
        "position": ["WR"] * 5,
        "adp": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    rho, pval, n, by_pos = spearman_vs_adp(model, adp)
    assert n == 5
    assert abs(rho - 1.0) < 1e-9
    assert "WR" in by_pos


def test_normalize_name() -> None:
    assert _normalize_name("CeeDee Lamb") == "ceedee lamb"
