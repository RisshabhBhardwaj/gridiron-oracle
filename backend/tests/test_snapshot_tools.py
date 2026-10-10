"""
Offline tests for the baseline and parity tooling:

  scripts/capture_api_snapshot.py  capture API responses, diff two capture dirs
  scripts/db_parity.py             row counts and md5 checksums, diff two snapshots

No test reaches the network or a database. HTTP is replaced at
``capture_api_snapshot._http_get`` / ``capture_api_snapshot._build_opener``, and
psycopg2 is replaced with an in-memory fake that records the SQL it receives.
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import urllib.error
from pathlib import Path

import pytest
from psycopg2 import sql

from scripts import capture_api_snapshot as capture
from scripts import db_parity

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

API_KEY_SENTINEL = "TEST-API-KEY-SENTINEL-7c1f"  # gitleaks:allow (test sentinel, not a credential)
DSN_SENTINEL_PW = "TEST-DSN-PASSWORD-SENTINEL-9a2e"
DSN_SENTINEL = f"postgresql://parity:{DSN_SENTINEL_PW}@db.example.test:5432/gridiron"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _has_main_guard(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return True
    return False


def _imports_argparse(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "argparse" for alias in node.names):
            return True
    return False


def _write_capture_file(directory: Path, name: str, status: int, body) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    capture.write_json(directory / name, {"status": status, "body": body})


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalise_drops_volatile_fields_and_sorts_keys():
    raw = {
        "z": 1,
        "a": {
            "timestamp": "2026-10-10T12:00:00+00:00",
            "value": 2,
            "data_freshness": "2026-10-09T00:00:00+00:00",
            "pipeline_run_id": "run-123",
        },
        "alerts": [
            {
                "id": "sys-7",
                "title": "Gridiron Oracle started",
                "body": "API online at 2026-10-10T12:00:00.123456+00:00",
            }
        ],
    }

    out = capture.normalise(raw)

    assert out == {
        "a": {"value": 2},
        "alerts": [{"body": "API online at <datetime>", "title": "Gridiron Oracle started"}],
        "z": 1,
    }
    assert list(out) == ["a", "alerts", "z"]
    assert list(out["a"]) == ["value"]


def test_normalise_drops_alert_id_only_inside_alerts_list():
    raw = {
        "alerts": [{"id": "sys-1", "severity": "system"}],
        "player": {"id": "00-0030000", "name": "Example Player"},
    }

    out = capture.normalise(raw)

    assert out["alerts"] == [{"severity": "system"}]
    assert out["player"] == {"id": "00-0030000", "name": "Example Player"}


def test_normalise_keeps_list_order_and_non_volatile_values():
    raw = {"players": [{"adp": 9.5}, {"adp": 1.0}, {"adp": None}], "flag": True}

    out = capture.normalise(raw)

    assert out == {"players": [{"adp": 9.5}, {"adp": 1.0}, {"adp": None}], "flag": True}


def test_volatile_constant_documents_each_key_it_drops():
    # The constant is the single source of truth for what gets stripped.
    for key in ("timestamp", "data_freshness", "as_of", "pipeline_run_id", "computed_at", "generated_at", "git_commit"):
        assert key in capture.VOLATILE_KEYS
    assert "id" in capture.VOLATILE_SCOPED_KEYS["alerts"]


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def test_first_difference_reports_nested_json_path():
    a = {"body": {"projection": {"receiving_yards": 61.5, "ppr_points": 9.0}}}
    b = {"body": {"projection": {"receiving_yards": 62.0, "ppr_points": 9.0}}}

    assert capture.first_difference(a, b) == "$.body.projection.receiving_yards"


def test_first_difference_reports_list_index_path():
    a = {"body": {"players": [{"adp": 1}, {"adp": 2}, {"adp": 3}, {"adp": 4.0}]}}
    b = {"body": {"players": [{"adp": 1}, {"adp": 2}, {"adp": 3}, {"adp": 5}]}}

    assert capture.first_difference(a, b) == "$.body.players[3].adp"


def test_first_difference_reports_first_element_present_on_one_side_only():
    a = {"players": [1, 2, 3]}
    b = {"players": [1, 2]}

    assert capture.first_difference(a, b) == "$.players[2]"


def test_first_difference_treats_int_and_float_equal_and_bool_distinct():
    assert capture.first_difference({"n": 4}, {"n": 4.0}) is None
    assert capture.first_difference({"flag": True}, {"flag": 1}) == "$.flag"


def test_first_difference_quotes_keys_that_are_not_identifiers():
    assert capture.first_difference({"a b": 1}, {"a b": 2}) == '$["a b"]'


def test_compare_directories_reports_match_mismatch_and_missing(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_capture_file(a, "health.json", 200, {"status": "ok"})
    _write_capture_file(b, "health.json", 200, {"status": "ok"})
    _write_capture_file(a, "predict_x.json", 200, {"projection": {"receiving_yards": 61.5}})
    _write_capture_file(b, "predict_x.json", 200, {"projection": {"receiving_yards": 62.0}})
    _write_capture_file(a, "alerts.json", 200, {"alerts": []})

    results = {name: (match, detail) for name, match, detail in capture.compare_directories(a, b)}

    assert results["health.json"] == (True, "")
    assert results["predict_x.json"] == (False, "$.body.projection.receiving_yards")
    assert results["alerts.json"] == (False, "only in A")


def test_compare_directories_ignores_manifest(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_capture_file(a, "health.json", 200, {"status": "ok"})
    _write_capture_file(b, "health.json", 200, {"status": "ok"})
    capture.write_json(a / "manifest.json", {"base_host": "one.example.test"})
    capture.write_json(b / "manifest.json", {"base_host": "two.example.test"})

    names = [name for name, _match, _detail in capture.compare_directories(a, b)]

    assert names == ["health.json"]


def test_diff_subcommand_exit_codes(tmp_path, capsys):
    a = tmp_path / "a"
    b = tmp_path / "b"
    _write_capture_file(a, "root.json", 200, {"message": "hi"})
    _write_capture_file(b, "root.json", 200, {"message": "hi"})

    assert capture.main(["diff", str(a), str(b)]) == 0
    assert "MATCH" in capsys.readouterr().out

    _write_capture_file(b, "root.json", 200, {"message": "changed"})
    assert capture.main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "MISMATCH" in out and "$.body.message" in out


# ---------------------------------------------------------------------------
# Manifest and writing
# ---------------------------------------------------------------------------

def test_write_json_is_sorted_and_redacts_secrets(tmp_path):
    path = tmp_path / "x.json"

    capture.write_json(path, {"b": "value-KEY-PLACEHOLDER", "a": 1}, secrets=["KEY-PLACEHOLDER"])

    text = path.read_text()
    assert "KEY-PLACEHOLDER" not in text
    assert "<redacted>" in text
    assert text.index('"a"') < text.index('"b"')
    assert text.endswith("\n")


def test_base_host_is_host_and_port_only():
    assert capture.base_host("https://user.example.test/api/v1/") == "user.example.test"
    assert capture.base_host("http://127.0.0.1:8000/") == "127.0.0.1:8000"


def test_base_url_with_credentials_is_rejected():
    with pytest.raises(ValueError):
        capture.validated_base_url("https://user:pw@api.example.test")


def test_build_requests_uses_real_required_params():
    reqs = dict(capture.build_requests(["Josh Allen", "Ja'Marr Chase"], season=2026, week=6, stat="receiving_yards"))

    assert reqs["root.json"] == "/"
    assert reqs["health.json"] == "/health"
    assert reqs["settings.json"] == "/settings"
    assert reqs["season_current.json"] == "/season/current"
    assert reqs["alerts.json"] == "/alerts"
    assert reqs["draft_board.json"] == "/draft/board?season=2026"
    assert reqs["odds_player_reception_yards.json"] == "/odds?markets=player_reception_yards"
    assert reqs["predict_josh_allen.json"] == "/predict?player=Josh+Allen&week=6&season=2026&stat=receiving_yards"
    assert reqs["predict_ja_marr_chase.json"].startswith("/predict?player=Ja%27Marr+Chase&")


def test_build_requests_rejects_players_that_collide_after_slugging():
    with pytest.raises(ValueError):
        capture.build_requests(["Mike Evans", "Mike-Evans"], season=2026, week=6, stat="receiving_yards")


def test_parse_players_from_csv_and_file(tmp_path):
    players_file = tmp_path / "players.txt"
    players_file.write_text("# comment\nJosh Allen\n\n Ja'Marr Chase \n")

    assert capture.parse_players("Josh Allen, Ja'Marr Chase", None) == ["Josh Allen", "Ja'Marr Chase"]
    assert capture.parse_players(None, players_file) == ["Josh Allen", "Ja'Marr Chase"]


# ---------------------------------------------------------------------------
# API capture against a faked HTTP layer
# ---------------------------------------------------------------------------

class _FakeHttp:
    """Stands in for capture._http_get. Records every call and returns scripted bodies."""

    def __init__(self, clock: int = 0, echo: str | None = None):
        self.calls: list[tuple[str, dict, float]] = []
        self.clock = clock
        self.echo = echo

    def __call__(self, url: str, headers: dict, timeout: float):
        self.calls.append((url, dict(headers), timeout))
        path = url[len("https://api.example.test"):]
        stamp = f"2026-10-10T12:00:{self.clock:02d}+00:00"
        if path == "/health":
            return 503, json.dumps({"status": "blocked", "timestamp": stamp}).encode()
        if path.startswith("/predict"):
            body = {
                "player_id": "00-0036389",
                "projection": {"receiving_yards": 61.5},
                "data_freshness": stamp,
                "pipeline_run_id": f"run-{self.clock}",
            }
            if self.echo:
                body["echoed"] = self.echo
            return 200, json.dumps(body).encode()
        if path == "/alerts":
            return 200, json.dumps({
                "count": 1,
                "alerts": [{"id": f"sys-{self.clock}", "title": "started",
                            "body": f"API online at {stamp}", "timestamp": stamp}],
            }).encode()
        if path == "/odds?markets=player_reception_yards":
            return 404, b'{"detail":"Not Found"}'
        return 200, json.dumps({"path": path}).encode()


@pytest.fixture
def capture_env(monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://api.example.test/")
    monkeypatch.setenv("API_KEY", API_KEY_SENTINEL)


def test_capture_requests_every_endpoint_and_records_status(tmp_path, monkeypatch, capture_env):
    fake = _FakeHttp()
    monkeypatch.setattr(capture, "_http_get", fake)
    out = tmp_path / "snap"

    rc = capture.main([
        "capture", "--out", str(out), "--players", "Josh Allen,Ja'Marr Chase",
        "--season", "2026", "--week", "6",
    ])

    assert rc == 0
    names = sorted(p.name for p in out.glob("*.json"))
    assert names == sorted([
        "alerts.json", "draft_board.json", "health.json", "odds_player_reception_yards.json",
        "predict_ja_marr_chase.json", "predict_josh_allen.json", "root.json", "season_current.json",
        "settings.json", "manifest.json",
    ])
    health = json.loads((out / "health.json").read_text())
    assert health == {"body": {"status": "blocked"}, "status": 503}
    assert json.loads((out / "odds_player_reception_yards.json").read_text())["status"] == 404


def test_capture_sends_api_key_only_as_header_when_set(tmp_path, monkeypatch, capture_env):
    fake = _FakeHttp()
    monkeypatch.setattr(capture, "_http_get", fake)

    capture.main(["capture", "--out", str(tmp_path / "with"), "--players", "Josh Allen",
                  "--season", "2026", "--week", "6"])

    for url, headers, _timeout in fake.calls:
        assert headers.get("X-API-Key") == API_KEY_SENTINEL
        assert API_KEY_SENTINEL not in url


def test_capture_omits_api_key_header_when_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://api.example.test")
    monkeypatch.setenv("API_KEY", "   ")
    fake = _FakeHttp()
    monkeypatch.setattr(capture, "_http_get", fake)

    capture.main(["capture", "--out", str(tmp_path / "nokey"), "--players", "Josh Allen",
                  "--season", "2026", "--week", "6"])

    assert fake.calls
    for _url, headers, _timeout in fake.calls:
        assert "X-API-Key" not in headers


def test_capture_predict_passes_player_week_season_stat(tmp_path, monkeypatch, capture_env):
    fake = _FakeHttp()
    monkeypatch.setattr(capture, "_http_get", fake)

    capture.run_capture(
        out_dir=tmp_path / "p", base_url="https://api.example.test", api_key="",
        players=["Josh Allen"], season=2026, week=6, stat="receiving_yards", timeout=5.0,
    )

    predict_urls = [url for url, _h, _t in fake.calls if "/predict" in url]
    assert predict_urls == [
        "https://api.example.test/predict?player=Josh+Allen&week=6&season=2026&stat=receiving_yards"
    ]


def test_capture_records_transport_error_without_exception_text(tmp_path, monkeypatch, capture_env):
    def boom(url, headers, timeout):
        raise urllib.error.URLError("connection refused at 10.0.0.9 with a secret reason")

    monkeypatch.setattr(capture, "_http_get", boom)
    out = tmp_path / "down"

    rc = capture.run_capture(
        out_dir=out, base_url="https://api.example.test", api_key="",
        players=[], season=2026, week=6, stat="receiving_yards", timeout=5.0,
    )

    assert rc == 1
    record = json.loads((out / "root.json").read_text())
    assert record == {"body": None, "error": "URLError", "status": None}
    assert "secret reason" not in (out / "root.json").read_text()


def test_http_get_maps_http_error_to_status_and_body(monkeypatch):
    class _RaisingOpener:
        def open(self, request, timeout=None):
            raise urllib.error.HTTPError(
                request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b'{"status":"blocked"}')
            )

    monkeypatch.setattr(capture, "_build_opener", lambda: _RaisingOpener())

    status, body = capture._http_get("https://api.example.test/health", {}, 5.0)

    assert status == 503
    assert json.loads(body) == {"status": "blocked"}


def test_redirects_are_not_followed():
    handler = capture._NoRedirect()

    assert handler.redirect_request(None, None, 302, "Found", {}, "https://elsewhere.example.test/") is None


def test_manifest_hashes_are_stable_across_runs_despite_volatile_fields(tmp_path, monkeypatch, capture_env):
    # Two runs with different clocks: timestamps, data_freshness, pipeline_run_id
    # and the alert id all differ. The normalised files and manifest must not.
    monkeypatch.setattr(capture, "_http_get", _FakeHttp(clock=1))
    run_a = tmp_path / "run_a"
    capture.main(["capture", "--out", str(run_a), "--players", "Josh Allen", "--season", "2026", "--week", "6"])

    monkeypatch.setattr(capture, "_http_get", _FakeHttp(clock=42))
    run_b = tmp_path / "run_b"
    capture.main(["capture", "--out", str(run_b), "--players", "Josh Allen", "--season", "2026", "--week", "6"])

    manifest_a = json.loads((run_a / "manifest.json").read_text())
    manifest_b = json.loads((run_b / "manifest.json").read_text())
    assert manifest_a == manifest_b
    assert (run_a / "manifest.json").read_bytes() == (run_b / "manifest.json").read_bytes()

    for name, digest in manifest_a["files"].items():
        assert hashlib.sha256((run_a / name).read_bytes()).hexdigest() == digest
    assert all(match for _name, match, _detail in capture.compare_directories(run_a, run_b))


def test_manifest_has_host_only_and_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("BASE_URL", "https://api.example.test:8443/v1/")
    monkeypatch.setenv("API_KEY", API_KEY_SENTINEL)
    monkeypatch.setattr(capture, "_http_get", _FakeHttp())
    out = tmp_path / "m"

    capture.main(["capture", "--out", str(out), "--players", "Josh Allen", "--season", "2026", "--week", "6"])

    text = (out / "manifest.json").read_text()
    manifest = json.loads(text)
    assert manifest["base_host"] == "api.example.test:8443"
    assert "/v1" not in text
    assert API_KEY_SENTINEL not in text
    assert manifest["inputs"] == {"players": ["Josh Allen"], "season": 2026, "stat": "receiving_yards", "week": 6}
    assert sorted(manifest["files"]) == sorted(
        p.name for p in out.glob("*.json") if p.name != "manifest.json"
    )


def test_api_key_never_appears_in_written_files_or_stdout(tmp_path, monkeypatch, capture_env, capsys):
    # The server echoes the key back in a body. The tool must redact it.
    monkeypatch.setattr(capture, "_http_get", _FakeHttp(echo=API_KEY_SENTINEL))
    out = tmp_path / "leak"

    capture.main(["capture", "--out", str(out), "--players", "Josh Allen", "--season", "2026", "--week", "6"])

    for path in out.rglob("*"):
        if path.is_file():
            assert API_KEY_SENTINEL not in path.read_text(), path.name
    assert API_KEY_SENTINEL not in capsys.readouterr().out
    assert "<redacted>" in (out / "predict_josh_allen.json").read_text()


def test_capture_rejects_missing_base_url(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("BASE_URL", raising=False)

    rc = capture.main(["capture", "--out", str(tmp_path / "x"), "--players", "Josh Allen",
                       "--season", "2026", "--week", "6"])

    assert rc == 2
    assert "BASE_URL is not set" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


# ---------------------------------------------------------------------------
# db_parity: fake psycopg2 connection
# ---------------------------------------------------------------------------

def _render(query) -> str:
    if isinstance(query, sql.Composed):
        return "".join(_render(part) for part in query.seq)
    if isinstance(query, sql.Identifier):
        return '"' + query.strings[0] + '"'
    if isinstance(query, sql.SQL):
        return query.string
    return str(query)


class _FakeCursor:
    def __init__(self, conn: "_FakeConnection"):
        self._conn = conn
        self._rows: list = []

    def execute(self, query, params=None):
        text = _render(query)
        self._conn.log.append(text)
        self._rows = self._conn.dispatch(text)

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]


class _FakeConnection:
    def __init__(self, tables: dict[str, int], checksums: dict[str, str | None]):
        self.tables = tables
        self.checksums = checksums
        self.log: list[str] = []
        self.session: dict = {}
        self.closed = False

    def cursor(self):
        return _FakeCursor(self)

    def set_session(self, **kwargs):
        self.session.update(kwargs)

    def rollback(self):
        pass

    def close(self):
        self.closed = True

    def dispatch(self, text: str) -> list:
        import re

        if text.startswith("SET LOCAL"):
            return []
        if "information_schema.tables" in text:
            return [(name,) for name in sorted(self.tables)]
        if text.startswith("SELECT count(*) FROM"):
            name = re.search(r'FROM "([^"]+)"', text).group(1)
            return [(self.tables[name],)]
        if "md5(" in text:
            name = re.search(r'FROM "([^"]+)" t', text).group(1)
            return [(self.checksums.get(name),)]
        raise AssertionError(f"unexpected SQL in fake: {text}")


# ---------------------------------------------------------------------------
# db_parity: snapshot and diff
# ---------------------------------------------------------------------------

def test_build_snapshot_counts_every_public_table_and_checksums_two_tables():
    conn = _FakeConnection(
        tables={"alembic_version": 1, "feature_matrix": 0, "projections": 3, "teams": 32},
        checksums={"projections": "abc123", "feature_matrix": None},
    )

    snap = db_parity.build_snapshot(conn)

    assert snap["row_counts"] == {"alembic_version": 1, "feature_matrix": 0, "projections": 3, "teams": 32}
    assert snap["checksums"] == {"feature_matrix": None, "projections": "abc123"}
    assert conn.session == {"readonly": True, "isolation_level": "REPEATABLE READ", "autocommit": False}
    assert conn.log[0] == "SET LOCAL TIME ZONE 'UTC'"
    assert (
        "SELECT md5(string_agg(t::text, '|' ORDER BY t::text COLLATE \"C\")) FROM \"projections\" t" in conn.log
    )


def test_build_snapshot_gives_null_checksum_for_missing_table():
    conn = _FakeConnection(tables={"teams": 32}, checksums={})

    snap = db_parity.build_snapshot(conn)

    assert snap["checksums"] == {"feature_matrix": None, "projections": None}
    assert not any('"projections" t' in stmt for stmt in conn.log)


def test_diff_snapshots_on_hand_written_files_reports_each_difference(tmp_path, capsys):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    base = {
        "schema_version": 1,
        "row_counts": {"projections": 10, "teams": 32},
        "checksums": {"feature_matrix": "f1", "projections": "p1"},
    }
    a.write_text(json.dumps(base))
    b.write_text(json.dumps(base))

    assert db_parity.main(["diff", str(a), str(b)]) == 0
    assert "MATCH" in capsys.readouterr().out

    changed = json.loads(json.dumps(base))
    changed["row_counts"]["projections"] = 11
    changed["row_counts"]["extra"] = 5
    changed["checksums"]["projections"] = "p2"
    b.write_text(json.dumps(changed))

    assert db_parity.main(["diff", str(a), str(b)]) == 1
    out = capsys.readouterr().out
    assert "row_counts.projections" in out
    assert "row_counts.extra" in out and "only in B" in out
    assert "checksums.projections" in out


def test_diff_snapshots_reports_table_only_in_a(tmp_path):
    left = {"row_counts": {"teams": 32, "old_table": 1}, "checksums": {}}
    right = {"row_counts": {"teams": 32}, "checksums": {}}

    diffs = db_parity.diff_snapshots(left, right)

    assert diffs == ["row_counts.old_table: only in A"]


def test_snapshot_writes_json_without_dsn(tmp_path, monkeypatch):
    import psycopg2

    seen: dict = {}

    def fake_connect(dsn):
        seen["dsn"] = dsn
        return _FakeConnection(tables={"projections": 2}, checksums={"projections": "abc"})

    monkeypatch.setenv("DATABASE_URL", DSN_SENTINEL)
    monkeypatch.setattr(psycopg2, "connect", fake_connect)
    out = tmp_path / "snap.json"

    rc = db_parity.main(["snapshot", "--out", str(out)])

    assert rc == 0
    assert seen["dsn"] == DSN_SENTINEL
    text = out.read_text()
    assert DSN_SENTINEL_PW not in text
    assert "db.example.test" not in text
    assert json.loads(text)["row_counts"] == {"projections": 2}


def test_snapshot_failure_prints_exception_class_only(tmp_path, monkeypatch, capsys):
    import psycopg2

    def failing_connect(dsn):
        raise RuntimeError(f"could not connect using {dsn}")

    monkeypatch.setenv("DATABASE_URL", DSN_SENTINEL)
    monkeypatch.setattr(psycopg2, "connect", failing_connect)
    out = tmp_path / "never.json"

    rc = db_parity.main(["snapshot", "--out", str(out)])

    assert rc == 2
    assert not out.exists()
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.out + captured.err
    assert DSN_SENTINEL_PW not in captured.out + captured.err


def test_snapshot_requires_database_url(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DATABASE_URL", raising=False)

    rc = db_parity.main(["snapshot", "--out", str(tmp_path / "x.json")])

    assert rc == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_psycopg2_dsn_strips_sqlalchemy_driver_suffix():
    assert db_parity._psycopg2_dsn("postgresql+asyncpg://u:p@h/db") == "postgresql://u:p@h/db"
    assert db_parity._psycopg2_dsn("postgresql://u:p@h/db") == "postgresql://u:p@h/db"


# ---------------------------------------------------------------------------
# Script shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["capture_api_snapshot.py", "db_parity.py"])
def test_scripts_have_main_guard_and_argparse(script):
    path = SCRIPTS / script

    assert _has_main_guard(path)
    assert _imports_argparse(path)


def test_capture_parser_accepts_capture_and_diff_subcommands():
    parser = capture.build_parser()

    args = parser.parse_args(["capture", "--out", "d", "--players", "A,B", "--season", "2026", "--week", "6"])
    assert args.command == "capture" and args.season == 2026 and args.stat == "receiving_yards"

    args = parser.parse_args(["diff", "a", "b"])
    assert args.command == "diff" and args.dir_a == "a" and args.dir_b == "b"
