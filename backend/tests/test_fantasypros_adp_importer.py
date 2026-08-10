"""
FantasyPros ADP importer column resolution — audit finding C-23.

A standard FantasyPros PPR ADP export carries BOTH a `Rank` column (1..N, the
site's own ordering) and an `AVG` column (the actual average draft position).
The old alias table mapped both onto `adp`, so the importer silently stored
Rank as ADP — wrong by an order of magnitude on the documented 2026 import
path — and the `required - set(columns)` guard could not see it because the
rename produced two columns both named `adp`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scraper.adapters.fantasypros_adp_importer import _normalize_columns

# Real FantasyPros ADP export header shape (PPR, overall).
FP_HEADER = "Rank,Player,Team,Bye,POS,ESPN,Sleeper,NFL,RTSports,FFC,AVG"
FP_ROWS = [
    "1,Ja'Marr Chase,CIN,10,WR1,1.3,1.0,1.2,1.5,1.1,1.2",
    "2,Bijan Robinson,ATL,5,RB1,2.4,2.6,2.1,2.8,2.3,2.4",
    "3,Justin Jefferson,MIN,6,WR2,3.9,4.1,3.4,4.0,3.6,3.8",
    "40,Tony Pollard,TEN,10,RB18,44.1,43.0,45.2,42.8,43.9,43.8",
    "120,Adam Thielen,CAR,14,WR52,131.0,126.4,129.9,133.1,127.6,129.6",
]


@pytest.fixture()
def fp_export(tmp_path: Path) -> Path:
    path = tmp_path / "ppr_2026.csv"
    path.write_text("\n".join([FP_HEADER, *FP_ROWS]) + "\n")
    return path


def test_adp_comes_from_avg_not_rank(fp_export: Path) -> None:
    raw = pd.read_csv(fp_export)
    out = _normalize_columns(raw)

    # The value actually imported must be AVG, not Rank.
    assert out["adp"].tolist() == raw["AVG"].tolist()
    assert out["adp"].tolist() != raw["Rank"].tolist()

    # Rank is preserved under its own name, never as a draft position.
    assert out["fp_rank"].tolist() == raw["Rank"].tolist()


def test_exactly_one_adp_column(fp_export: Path) -> None:
    """The old rename produced two columns labelled `adp`; `itertuples` then
    picked one positionally, which is how Rank got stored."""
    out = _normalize_columns(pd.read_csv(fp_export))
    assert list(out.columns).count("adp") == 1


def test_row_values_match_avg_through_itertuples(fp_export: Path) -> None:
    """Exercise the same access path `import_csv` uses to build DB rows."""
    raw = pd.read_csv(fp_export)
    out = _normalize_columns(raw)
    values = [float(r.adp) for r in out.itertuples(index=False)]
    assert values == [float(v) for v in raw["AVG"].tolist()]
    # Thielen: ADP ~129.6, rank 120. A rank-as-ADP regression would give 120.0.
    assert values[-1] == pytest.approx(129.6)


def test_player_and_position_resolve(fp_export: Path) -> None:
    out = _normalize_columns(pd.read_csv(fp_export))
    assert out["player_name"].iloc[0] == "Ja'Marr Chase"
    assert out["position"].iloc[0] == "WR1"
    assert out["team"].iloc[0] == "CIN"


def test_rank_only_export_is_rejected(tmp_path: Path) -> None:
    """A CSV with no average-draft-position column has no ADP. It must raise
    rather than import the rank."""
    path = tmp_path / "rank_only.csv"
    path.write_text("Rank,Player,Team,POS\n1,Ja'Marr Chase,CIN,WR1\n")

    with pytest.raises(ValueError, match="missing columns"):
        _normalize_columns(pd.read_csv(path))


def test_explicit_adp_column_wins_over_avg(tmp_path: Path) -> None:
    path = tmp_path / "both.csv"
    path.write_text("Player,ADP,AVG\nJa'Marr Chase,1.2,99.9\n")

    out = _normalize_columns(pd.read_csv(path))
    assert out["adp"].tolist() == [1.2]


def test_player_name_alias_priority(tmp_path: Path) -> None:
    path = tmp_path / "names.csv"
    path.write_text("Name,Player Name,AVG\nwrong,Ja'Marr Chase,1.2\n")

    out = _normalize_columns(pd.read_csv(path))
    assert out["player_name"].tolist() == ["Ja'Marr Chase"]
