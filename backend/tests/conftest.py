"""
backend/tests/conftest.py

Shared fixtures for the backend test suite.
"""

from __future__ import annotations

import pytest

from backend.tests.fail_on_skip import pytest_runtest_makereport  # noqa: F401  (hook; see module)


@pytest.fixture(autouse=True)
def _isolate_user_settings_file(tmp_path, monkeypatch):
    """Point the /settings weight store at a per-test file.

    Without this, any test that PUTs /settings overwrites the owner's live
    user_config.json at the repo root.
    """
    from backend.app.core.config import settings as app_settings

    path = tmp_path / "user_config.json"
    monkeypatch.setattr(app_settings, "settings_file", str(path))
    monkeypatch.setenv("SETTINGS_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def _reset_rate_limits():
    """Start every test with empty rate-limit buckets.

    The limiter is process-wide, so PUT /settings (10/minute) would otherwise
    start returning 429 once enough tests have written settings.
    """
    from backend.app.core.rate_limit import limiter

    limiter.reset()
    yield
