"""The API key guard must judge the path the router dispatches on, not a path
rebuilt from the Host header. A crafted Host (for example "x/health#") used to
make the guard see "/health" (exempt) while the router served "/settings"."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

from backend.app.main import app

client = TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize("host", ["x/health#", "x?", "x/health?admin=1"])
@pytest.mark.parametrize(("method", "path"), [("PUT", "/settings"), ("GET", "/integrity")])
def test_crafted_host_header_does_not_skip_the_key_check(host: str, method: str, path: str) -> None:
    with patch("backend.app.main.settings.api_key", "k1"), patch(
        "backend.app.main.settings.admin_api_key", "adm"
    ):
        r = client.request(method, path, headers={"host": host}, json={} if method == "PUT" else None)
    assert r.status_code == 401


def test_exempt_path_still_open_with_key_configured() -> None:
    with patch("backend.app.main.settings.api_key", "k1"):
        r = client.get("/")
    assert r.status_code != 401
