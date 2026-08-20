from pipeline.adp_resolution import resolve_rows
from scraper.adapters.sleeper_adp import aggregate_adp


def test_resolver_uses_canonical_id_and_normalizes_vendor_position() -> None:
    resolved = resolve_rows(
        [{"player_name": "Hollywood Brown", "position": "WR42", "team": "KC", "adp": 99.0}],
        [{"id": "00-003", "full_name": "Hollywood Brown", "position": "WR", "team": "KC"}],
    )
    assert resolved[0]["player_id"] == "00-003"
    assert resolved[0]["position"] == "WR"
    assert resolved[0]["match_status"] == "matched"


def test_resolver_reports_ambiguous_rows_instead_of_selecting_by_name() -> None:
    resolved = resolve_rows(
        [{"player_name": "Josh Palmer", "position": "WR", "team": None, "adp": 180.0}],
        [
            {"id": "a", "full_name": "Josh Palmer", "position": "WR", "team": "LAC"},
            {"id": "b", "full_name": "Josh Palmer", "position": "WR", "team": "BUF"},
        ],
    )
    assert resolved[0]["player_id"] is None
    assert resolved[0]["match_method"] == "ambiguous_name"
    assert resolved[0]["candidate_player_ids"] == ["a", "b"]


def test_resolver_prefers_fantasy_player_ids_before_name() -> None:
    resolved = resolve_rows(
        [{"player_name": "Wrong Name", "sleeper_id": "s1", "adp": 12.0}],
        [{"id": "00-001", "full_name": "Canonical", "position": "WR", "team": "KC"}],
        id_maps=[{"gsis_id": "00-001", "sleeper_id": "s1"}],
    )
    assert resolved[0]["player_id"] == "00-001"
    assert resolved[0]["match_method"] == "fantasy_player_ids:sleeper_id"


def test_empty_sleeper_draft_set_returns_a_shaped_empty_frame() -> None:
    frame = aggregate_adp([])
    assert frame.empty
    assert "adp" in frame.columns
