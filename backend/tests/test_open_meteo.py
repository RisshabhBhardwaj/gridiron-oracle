from scraper.adapters.open_meteo import OPEN_METEO, PROVIDER, forecast_url


def test_open_meteo_url_and_provider() -> None:
    url = forecast_url(44.97, -93.25)
    assert url.startswith(OPEN_METEO)
    assert "latitude=44.97" in url
    assert PROVIDER == "open_meteo"


def test_scheduler_registers_keyless_weather_and_id_jobs() -> None:
    from pathlib import Path

    source = Path("scraper/scheduler.py").read_text()
    assert "_run_open_meteo" in source
    assert "_run_ff_playerids" in source
    assert "_run_sleeper_consensus" in source
    assert "open_meteo_forecast" in source
