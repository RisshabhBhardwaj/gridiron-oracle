"""
backend/tests/test_settings_storage.py

Tests for the /settings weight store: test isolation from the owner's live
user_config.json, atomic writes, and the provenance of values returned when
the stored file cannot be used.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.app.api import settings as settings_api
from backend.app.main import app

client = TestClient(app, raise_server_exceptions=True)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "user_config.json"


def _file_state(path: Path):
    """Inode, mtime and size of a file; None when it does not exist."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


# ---------------------------------------------------------------------------
# 1. Test isolation
# ---------------------------------------------------------------------------

def test_put_settings_never_touches_repo_user_config():
    before = _file_state(REPO_CONFIG)
    r = client.put("/settings", json={"fantasy_scoring": "ppr"})
    assert r.status_code == 200
    assert _file_state(REPO_CONFIG) == before


def test_settings_store_path_is_per_test_tmp(tmp_path):
    from backend.app.api import settings as settings_api

    assert settings_api._config_path() == tmp_path / "user_config.json"


# ---------------------------------------------------------------------------
# 2. Atomic write
# ---------------------------------------------------------------------------

def test_save_config_crash_mid_write_keeps_previous_file_intact(monkeypatch):
    from backend.app.api import settings as settings_api

    path = settings_api._config_path()
    previous = json.dumps(settings_api._DEFAULT_CONFIG, indent=2)
    path.write_text(previous)

    def crash_after_partial_write(obj, fp, **kwargs):
        fp.write(json.dumps(obj, indent=2)[:40])
        fp.flush()
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(json, "dump", crash_after_partial_write)

    with pytest.raises(HTTPException) as exc_info:
        settings_api._save_config({**settings_api._DEFAULT_CONFIG, "fantasy_scoring": "half_ppr"})
    assert exc_info.value.status_code == 500

    assert path.read_text() == previous
    assert json.loads(path.read_text()) == settings_api._DEFAULT_CONFIG
    leftovers = sorted(
        p.name for p in path.parent.iterdir()
        if p.name != path.name and not p.name.endswith(".lock")
    )
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_save_config_keeps_existing_file_mode(monkeypatch):
    from backend.app.api import settings as settings_api

    path = settings_api._config_path()
    path.write_text(json.dumps(settings_api._DEFAULT_CONFIG, indent=2))
    path.chmod(0o640)

    settings_api._save_config({**settings_api._DEFAULT_CONFIG, "fantasy_scoring": "standard"})

    assert stat.S_IMODE(path.stat().st_mode) == 0o640
    assert json.loads(path.read_text())["fantasy_scoring"] == "standard"


# ---------------------------------------------------------------------------
# 3. Provenance of the values returned (settings_source)
# ---------------------------------------------------------------------------

_DEFAULTS = settings_api._DEFAULT_CONFIG
_EXISTING_KEYS = {"weights", "fantasy_scoring", "engine_exposure_cap", "weight_preset_name"}

_BAD_FILE_TEXT = {
    "invalid_json": "{not json",
    "not_an_object": "[1, 2, 3]",
    "weights_not_summing_to_100": json.dumps(
        {**_DEFAULTS, "weights": {**_DEFAULTS["weights"], "kalman_form": 50.0}}
    ),
    "unknown_fantasy_scoring": json.dumps({**_DEFAULTS, "fantasy_scoring": "draftkings"}),
    "exposure_cap_out_of_range": json.dumps({**_DEFAULTS, "engine_exposure_cap": 5.0}),
}


@pytest.fixture
def cfg_path():
    """The per-test store path set up by the autouse fixture in conftest.py."""
    return settings_api._config_path()


def _assert_defaults(body):
    assert body["weights"] == _DEFAULTS["weights"]
    assert body["fantasy_scoring"] == _DEFAULTS["fantasy_scoring"]
    assert body["engine_exposure_cap"] == _DEFAULTS["engine_exposure_cap"]
    assert body["weight_preset_name"] == _DEFAULTS["weight_preset_name"]


def _deny_reads_of(path: Path, monkeypatch) -> None:
    real_read_text = Path.read_text

    def deny(self, *args, **kwargs):
        if self == path:
            raise PermissionError("denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", deny)


def test_get_settings_missing_file_reports_default_missing(cfg_path):
    assert not cfg_path.exists()
    r = client.get("/settings")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == _EXISTING_KEYS | {"settings_source"}
    assert body["settings_source"] == "default_missing"
    _assert_defaults(body)


@pytest.mark.parametrize("name", sorted(_BAD_FILE_TEXT))
def test_get_settings_invalid_file_reports_default_invalid(cfg_path, name):
    cfg_path.write_text(_BAD_FILE_TEXT[name])
    r = client.get("/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["settings_source"] == "default_invalid"
    _assert_defaults(body)


def test_get_settings_unreadable_file_reports_default_invalid(cfg_path, monkeypatch):
    cfg_path.write_text(json.dumps(_DEFAULTS))
    _deny_reads_of(cfg_path, monkeypatch)
    r = client.get("/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["settings_source"] == "default_invalid"
    _assert_defaults(body)


def test_get_settings_valid_file_reports_file_and_its_values(cfg_path):
    cfg_path.write_text(
        json.dumps({**_DEFAULTS, "fantasy_scoring": "half_ppr", "weight_preset_name": "Balanced"}, indent=2)
    )
    body = client.get("/settings").json()
    assert body["settings_source"] == "file"
    assert body["fantasy_scoring"] == "half_ppr"
    assert body["weight_preset_name"] == "Balanced"


def test_repo_user_config_loads_as_file_with_its_values(cfg_path):
    """Guard: the owner's live file must load as valid, with no fallback to defaults."""
    if not REPO_CONFIG.exists():
        pytest.skip("no live user_config.json in this checkout")
    live_text = REPO_CONFIG.read_text()
    stored = json.loads(live_text)
    cfg_path.write_text(live_text)
    body = client.get("/settings").json()
    assert body["settings_source"] == "file"
    assert body["weights"] == stored["weights"]
    assert body["fantasy_scoring"] == stored["fantasy_scoring"]
    assert body["engine_exposure_cap"] == stored["engine_exposure_cap"]
    assert body["weight_preset_name"] == stored["weight_preset_name"]


def test_put_settings_missing_file_reports_default_missing(cfg_path):
    r = client.put("/settings", json={"fantasy_scoring": "standard"})
    assert r.status_code == 200
    body = r.json()
    assert body["settings_source"] == "default_missing"
    assert body["fantasy_scoring"] == "standard"
    assert body["weights"] == _DEFAULTS["weights"]


@pytest.mark.parametrize("name", sorted(_BAD_FILE_TEXT))
def test_put_settings_invalid_file_reports_default_invalid(cfg_path, name):
    cfg_path.write_text(_BAD_FILE_TEXT[name])
    r = client.put("/settings", json={"weight_preset_name": "Balanced"})
    assert r.status_code == 200
    body = r.json()
    assert body["settings_source"] == "default_invalid"
    assert body["weight_preset_name"] == "Balanced"
    assert body["weights"] == _DEFAULTS["weights"]
    assert body["fantasy_scoring"] == _DEFAULTS["fantasy_scoring"]


def test_put_settings_unreadable_file_reports_default_invalid(cfg_path, monkeypatch):
    cfg_path.write_text(json.dumps(_DEFAULTS))
    _deny_reads_of(cfg_path, monkeypatch)
    r = client.put("/settings", json={"weight_preset_name": "Balanced"})
    assert r.status_code == 200
    body = r.json()
    assert body["settings_source"] == "default_invalid"
    assert body["weight_preset_name"] == "Balanced"


def test_put_settings_valid_file_reports_file(cfg_path):
    cfg_path.write_text(json.dumps({**_DEFAULTS, "fantasy_scoring": "half_ppr"}, indent=2))
    body = client.put("/settings", json={"weight_preset_name": "Balanced"}).json()
    assert body["settings_source"] == "file"
    assert body["fantasy_scoring"] == "half_ppr"
    assert body["weight_preset_name"] == "Balanced"
