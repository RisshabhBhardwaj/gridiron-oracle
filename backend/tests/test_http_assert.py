"""
e2e smoke-test assertions — audit finding C-32.

`e2e/smoke_test.sh` called `/predict` with obsolete `player_id`/`position`
parameters instead of the required `player`, so it could not exercise the
current contract, and its assertions were `grep -qE '\\[|\\]'` — which passes on
an empty array and on most error payloads. `bash -n` passing is not a test.

These tests drive `e2e/lib/http_assert.sh` against a real HTTP server with
canned payloads, so the assertion helper is verified to fail when it should.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIB = REPO_ROOT / "e2e" / "lib" / "http_assert.sh"
SMOKE = REPO_ROOT / "e2e" / "smoke_test.sh"

# Payloads keyed by request path.
ROUTES: dict[str, tuple[int, object]] = {
    "/good": (200, {"season": 2025, "week": 3}),
    "/empty-list": (200, {"projections": [], "count": 0}),
    "/wrong-shape": (200, {"season": "not-an-int", "week": 3}),
    "/error-with-brackets": (200, {"detail": ["validation failed"]}),
    "/not-found": (404, {"detail": "nope"}),
    "/unprocessable": (422, {"detail": "field required"}),
}


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 — stdlib API
        status, payload = ROUTES.get(self.path, (404, {"detail": "unknown route"}))
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def _run(snippet: str) -> subprocess.CompletedProcess[str]:
    script = f"""
set -uo pipefail
PYTHON={sys.executable!r}
PASS=0
FAIL=0
source {str(LIB)!r}
{snippet}
echo "PASS=$PASS FAIL=$FAIL"
http_assert_cleanup
"""
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, cwd=REPO_ROOT
    )


def _counts(result: subprocess.CompletedProcess[str]) -> tuple[int, int]:
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("PASS=")][-1]
    passed, failed = line.split()
    return int(passed.split("=")[1]), int(failed.split("=")[1])


pytestmark = pytest.mark.skipif(
    shutil.which("curl") is None, reason="curl not available"
)


# ── api_check ────────────────────────────────────────────────────────────────


def test_valid_shape_passes(base_url: str) -> None:
    result = _run(
        f"""api_check "ok" "{base_url}/good" "
assert isinstance(d['season'], int), d
assert isinstance(d['week'], int), d
" """
    )
    assert _counts(result) == (1, 0), result.stdout + result.stderr


def test_empty_array_fails(base_url: str) -> None:
    """The old `grep -qE '\\[|\\]'` check passed on exactly this payload."""
    result = _run(
        f"""api_check "rows" "{base_url}/empty-list" "
rows = d['projections']
assert rows, 'empty projections list'
" """
    )
    assert _counts(result) == (0, 1), result.stdout + result.stderr
    assert "schema assertion failed" in result.stdout


def test_error_payload_containing_brackets_fails(base_url: str) -> None:
    """`{'detail': ['validation failed']}` contains brackets and would have
    satisfied the old assertion."""
    result = _run(
        f"""api_check "predict" "{base_url}/error-with-brackets" "
assert 'projection' in d, sorted(d)
" """
    )
    assert _counts(result) == (0, 1), result.stdout + result.stderr


def test_wrong_field_type_fails(base_url: str) -> None:
    result = _run(
        f"""api_check "season" "{base_url}/wrong-shape" "
assert isinstance(d['season'], int), d
" """
    )
    assert _counts(result) == (0, 1), result.stdout + result.stderr


def test_non_200_fails_before_assertions_run(base_url: str) -> None:
    result = _run(
        f"""api_check "missing" "{base_url}/not-found" "
assert True
" """
    )
    assert _counts(result) == (0, 1), result.stdout + result.stderr
    assert "HTTP 404" in result.stdout


def test_unreachable_host_fails() -> None:
    result = _run(
        """api_check "dead" "http://127.0.0.1:1/nothing" "
assert True
" """
    )
    assert _counts(result) == (0, 1), result.stdout + result.stderr


# ── status_check ─────────────────────────────────────────────────────────────


def test_status_check_matches_expected(base_url: str) -> None:
    result = _run(f'status_check "rejects" "{base_url}/unprocessable" 422')
    assert _counts(result) == (1, 0), result.stdout + result.stderr


def test_status_check_rejects_unexpected_success(base_url: str) -> None:
    """A request shape that must be rejected but returns 200 is a failure."""
    result = _run(f'status_check "rejects" "{base_url}/good" 422')
    assert _counts(result) == (0, 1), result.stdout + result.stderr


# ── the smoke test itself ────────────────────────────────────────────────────


def test_smoke_test_uses_the_current_predict_contract() -> None:
    source = SMOKE.read_text()
    assert "player=" in source, "/predict must be called with the `player` parameter"
    # The obsolete parameters may appear only in the negative check that asserts
    # they are rejected.
    for obsolete in ("player_id=", "position=WR"):
        occurrences = source.count(obsolete)
        assert occurrences <= 1, f"{obsolete} appears {occurrences} times"
        if occurrences:
            assert "422" in source, (
                "the obsolete parameter shape may only appear in a check that "
                "asserts it is rejected"
            )


def test_smoke_test_has_no_bracket_grep_assertions() -> None:
    source = SMOKE.read_text()
    assert "grep -qE '\\[|\\]'" not in source, (
        "bracket-presence grep is not a schema assertion (audit C-32)"
    )


def test_assertion_library_is_tracked() -> None:
    """`.gitignore` had an unanchored `lib/`, which matched `e2e/lib/` and kept
    this file out of every clone — the smoke test would have failed to source
    it on a fresh checkout."""
    tracked = subprocess.run(
        ["git", "ls-files", "e2e/lib/http_assert.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert tracked, "e2e/lib/http_assert.sh is not tracked by git"

    ignored = subprocess.run(
        ["git", "check-ignore", "e2e/lib/http_assert.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert ignored.returncode != 0, f"still ignored by: {ignored.stdout.strip()}"


def test_smoke_test_is_syntactically_valid() -> None:
    result = subprocess.run(
        ["bash", "-n", str(SMOKE)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
