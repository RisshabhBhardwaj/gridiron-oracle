from scraper.adapters.ffc_adp import DEFAULT_TEAMS, adp_url


def test_ffc_url_locks_eight_team_leagues() -> None:
    assert DEFAULT_TEAMS == 8
    url = adp_url(2025)
    assert "teams=8" in url
    assert "year=2025" in url
