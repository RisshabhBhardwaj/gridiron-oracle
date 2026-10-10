#!/usr/bin/env python3
"""
Capture a normalised snapshot of the public API, or diff two captures.

For baseline and parity checks: run the same capture against two deployments,
or before and after a change, then diff the two directories. Only the owner
runs this against live services. The test suite never does.

    BASE_URL=https://api.example.com API_KEY=... \\
      python scripts/capture_api_snapshot.py capture --out DIR \\
        --players "Josh Allen,Ja'Marr Chase" --season 2026 --week 6

    python scripts/capture_api_snapshot.py diff DIR_A DIR_B

Environment:
    BASE_URL  required. Scheme and host, optional path prefix, no credentials.
    API_KEY   optional. Sent only as the X-API-Key header, and only when non-empty.
              Never printed or written. Any copy in a response body is redacted.

Each endpoint is written to <name>.json as {"status": <int or null>, "body": ...}.
manifest.json holds the SHA-256 of every file, the host (no path, no key), and
the capture inputs. Volatile fields are removed before writing; see
VOLATILE_KEYS below.

Exit codes. capture: 0 all endpoints answered, 1 at least one endpoint had no
HTTP response, 2 configuration error. diff: 0 all match, 1 any mismatch, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_STAT = "receiving_yards"  # api/predict.py:216 default for `stat`
ODDS_MARKET = "player_reception_yards"

# Keys dropped at any depth, because they change between runs or hosts for
# reasons unrelated to the model. Evidence is given for each entry, with paths
# under backend/app/ unless noted.
VOLATILE_KEYS: frozenset[str] = frozenset({
    # Wall-clock time. /health body: main.py:455. AlertItem.timestamp: api/alerts.py:45,
    # set from datetime.now() at services/alert.py:282, 333 and 347.
    "timestamp",
    # /predict freshness. api/predict.py:102. Falls back to now() when there is no
    # feature_matrix row: services/projection.py:1294, called from projection.py:315.
    "data_freshness",
    # /draft/board as_of. api/draft.py:53. Resolved as MAX(as_of) with
    # as_of <= COALESCE(param, CURRENT_DATE), api/draft.py:146-148, so it moves with the calendar.
    "as_of",
    # Approved pipeline run that served a /predict row. api/predict.py:93,
    # set from services/projection.py:1174. Changes when a run is replaced.
    "pipeline_run_id",
    # Model run on team-game rows. api/team_game.py:57. Not in a captured body today.
    "model_run_id",
    # Build and runtime-status fields. Only /integrity (main.py:465) carries them,
    # and that route is not captured today: services/runtime_status.py:96 generated_at,
    # :158 computed_at, :260 git_commit (main.py:55 documents the commit field).
    "generated_at",
    "computed_at",
    "git_commit",
    # Request ids. Only a response header: middleware.py:49 and :71. No captured body
    # carries one today. Kept so the list still holds if a body ever does.
    "request_id",
})

# Keys dropped only inside the items of the named list. alerts[].id is a
# per-process counter ("sys-N", services/alert.py:238-242), so it restarts with
# the API and would differ between any two runs.
VOLATILE_SCOPED_KEYS: dict[str, frozenset[str]] = {
    "alerts": frozenset({"id"}),
}

# Full ISO-8601 datetimes inside string values, such as the startup alert body
# "API online at <time>" (main.py:176). They are replaced with a placeholder so
# the rest of the text still compares.
_ISO_DATETIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?")
_IDENTIFIER_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_NON_SLUG = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Normalisation and comparison
# ---------------------------------------------------------------------------

def normalise(value: Any, *, drop: frozenset[str] = frozenset()) -> Any:
    """Return a copy with volatile fields removed, keys sorted, and datetimes scrubbed."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value, key=str):
            if key in VOLATILE_KEYS or key in drop:
                continue
            child = value[key]
            if isinstance(child, list):
                scoped = VOLATILE_SCOPED_KEYS.get(key, frozenset())
                out[key] = [normalise(item, drop=scoped) for item in child]
            else:
                out[key] = normalise(child)
        return out
    if isinstance(value, list):
        return [normalise(item) for item in value]
    if isinstance(value, str):
        return _ISO_DATETIME.sub("<datetime>", value)
    return value


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def _key_segment(key: Any) -> str:
    key = str(key)
    return f".{key}" if _IDENTIFIER_KEY.match(key) else f"[{json.dumps(key)}]"


def first_difference(a: Any, b: Any, path: str = "$") -> str | None:
    """Return the JSON path of the first difference between a and b, or None if equal."""
    if _kind(a) != _kind(b):
        return path
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b), key=str):
            child_path = f"{path}{_key_segment(key)}"
            if key not in a or key not in b:
                return child_path
            found = first_difference(a[key], b[key], child_path)
            if found is not None:
                return found
        return None
    if isinstance(a, list):
        for index, (left, right) in enumerate(zip(a, b)):
            found = first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        if len(a) != len(b):
            # The first element that exists on one side only.
            return f"{path}[{min(len(a), len(b))}]"
        return None
    return None if a == b else path


def _capture_names(directory: Path) -> set[str]:
    return {p.name for p in directory.glob("*.json") if p.name != MANIFEST_NAME}


def compare_directories(dir_a: Path, dir_b: Path) -> list[tuple[str, bool, str]]:
    """Compare two capture directories. Returns (file, matches, detail) per file, sorted by name."""
    names_a = _capture_names(dir_a)
    names_b = _capture_names(dir_b)
    results: list[tuple[str, bool, str]] = []
    for name in sorted(names_a | names_b):
        if name not in names_b:
            results.append((name, False, "only in A"))
        elif name not in names_a:
            results.append((name, False, "only in B"))
        else:
            left = json.loads((dir_a / name).read_text(encoding="utf-8"))
            right = json.loads((dir_b / name).read_text(encoding="utf-8"))
            path = first_difference(left, right)
            results.append((name, path is None, "" if path is None else path))
    return results


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _redact(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "<redacted>")
        return value
    if isinstance(value, dict):
        return {key: _redact(child, secrets) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact(child, secrets) for child in value]
    return value


def write_json(path: Path, payload: Any, secrets: Iterable[str] = ()) -> None:
    """Write sorted, indented JSON with a trailing newline. Secret values are redacted from string values."""
    clean_secrets = tuple(s for s in secrets if s)
    text = json.dumps(_redact(payload, clean_secrets), sort_keys=True, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Return 3xx responses as they are, so X-API-Key is never sent to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_opener():
    return urllib.request.build_opener(_NoRedirect())


def _http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    """GET url. Returns (status, body) for any HTTP response, including 4xx, 5xx and 3xx.

    Transport failures (OSError, http.client.HTTPException) propagate to the caller.
    """
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with _build_opener().open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def validated_base_url(raw: str) -> str:
    """Return BASE_URL without a trailing slash. Reject non-http(s), no host, or embedded credentials."""
    cleaned = raw.strip()
    parts = urllib.parse.urlsplit(cleaned)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("BASE_URL must be an http or https URL with a host")
    if parts.username or parts.password:
        raise ValueError("BASE_URL must not contain credentials")
    return cleaned.rstrip("/")


def base_host(base_url: str) -> str:
    """Host and optional port only. The path and any credentials are dropped."""
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    port = parts.port
    return f"{host}:{port}" if port else host


def _slug(name: str) -> str:
    slug = _NON_SLUG.sub("_", name.lower()).strip("_")
    if not slug:
        raise ValueError("player name has no letters or digits")
    return slug


def parse_players(csv_value: str | None, players_file: Path | None) -> list[str]:
    if players_file is not None:
        raw_names = [line.strip() for line in players_file.read_text(encoding="utf-8").splitlines()]
        names = [n for n in raw_names if n and not n.startswith("#")]
    else:
        names = [part.strip() for part in (csv_value or "").split(",")]
    names = [n for n in names if n]
    if not names:
        raise ValueError("no players given")
    return names


def build_requests(players: list[str], *, season: int, week: int, stat: str) -> list[tuple[str, str]]:
    """Return (output file name, path with query) for every endpoint to capture.

    /predict takes a player's full name (or partial name), not an ID:
    api/predict.py:213 for `player`, resolved by name at services/projection.py:1110-1124.
    """
    requests: list[tuple[str, str]] = [
        ("root.json", "/"),                                              # main.py:475
        ("health.json", "/health"),                                      # main.py:436
        ("settings.json", "/settings"),                                  # api/settings.py:100
        ("season_current.json", "/season/current"),                      # api/season.py:23
        ("draft_board.json", "/draft/board?" + urllib.parse.urlencode({"season": season})),  # api/draft.py:232-234
        ("alerts.json", "/alerts"),                                      # api/alerts.py:57-58
        # /odds is not defined on this branch (no route under backend/app). Captured so the
        # missing route shows up as a recorded 404 rather than being silently absent.
        ("odds_player_reception_yards.json", "/odds?" + urllib.parse.urlencode({"markets": ODDS_MARKET})),
    ]
    seen_slugs: set[str] = set()
    for player in players:
        slug = _slug(player)
        if slug in seen_slugs:
            raise ValueError(f"two players map to the same file name: {slug}")
        seen_slugs.add(slug)
        query = urllib.parse.urlencode({"player": player, "week": week, "season": season, "stat": stat})
        requests.append((f"predict_{slug}.json", "/predict?" + query))   # api/predict.py:209-216
    return requests


def _decode_body(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def run_capture(
    *,
    out_dir: Path,
    base_url: str,
    api_key: str,
    players: list[str],
    season: int,
    week: int,
    stat: str,
    timeout: float,
) -> int:
    base = validated_base_url(base_url)
    secrets = (api_key,) if api_key else ()
    headers = {"Accept": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    requests = build_requests(players, season=season, week=week, stat=stat)

    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, path in requests:
        try:
            status, raw = _http_get(base + path, headers, timeout)
        except (OSError, http.client.HTTPException) as exc:
            # Record the exception class only. Its text can contain the URL or host.
            failures += 1
            record: dict[str, Any] = {"status": None, "body": None, "error": type(exc).__name__}
        else:
            record = {"status": status, "body": normalise(_decode_body(raw))}
        write_json(out_dir / name, record, secrets)
        shown = record["status"] if record["status"] is not None else f"no response ({record['error']})"
        print(f"{name}  {shown}")

    files = {
        name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest()
        for name, _path in requests
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_host": base_host(base),
        "inputs": {"players": players, "season": season, "stat": stat, "week": week},
        "files": dict(sorted(files.items())),
    }
    write_json(out_dir / MANIFEST_NAME, manifest, secrets)
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a normalised API snapshot, or diff two captures.")
    sub = parser.add_subparsers(dest="command", required=True)

    cap = sub.add_parser("capture", help="Capture the API using BASE_URL and API_KEY from the environment.")
    cap.add_argument("--out", required=True, type=Path, help="Output directory.")
    source = cap.add_mutually_exclusive_group(required=True)
    source.add_argument("--players", help="Comma-separated player names as /predict expects them (not IDs).")
    source.add_argument("--players-file", type=Path, help="One player name per line. # starts a comment.")
    cap.add_argument("--season", required=True, type=int)
    cap.add_argument("--week", required=True, type=int)
    cap.add_argument("--stat", default=DEFAULT_STAT)
    cap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)

    diff = sub.add_parser("diff", help="Compare two capture directories file by file.")
    diff.add_argument("dir_a")
    diff.add_argument("dir_b")
    return parser


def _run_capture_command(args: argparse.Namespace) -> int:
    base_url = os.environ.get("BASE_URL", "")
    api_key = os.environ.get("API_KEY", "").strip()
    if not base_url.strip():
        print("error: BASE_URL is not set", file=sys.stderr)
        return 2
    try:
        players = parse_players(args.players, args.players_file)
        return run_capture(
            out_dir=args.out,
            base_url=base_url,
            api_key=api_key,
            players=players,
            season=args.season,
            week=args.week,
            stat=args.stat,
            timeout=args.timeout,
        )
    except (ValueError, OSError) as exc:
        # ValueError messages here never contain the URL or the key.
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run_diff(dir_a: Path, dir_b: Path) -> int:
    for directory in (dir_a, dir_b):
        if not directory.is_dir():
            print(f"error: not a directory: {directory}", file=sys.stderr)
            return 2
    results = compare_directories(dir_a, dir_b)
    if not results:
        print("error: no capture files found", file=sys.stderr)
        return 2
    mismatches = 0
    for name, matches, detail in results:
        if matches:
            print(f"MATCH     {name}")
        else:
            mismatches += 1
            print(f"MISMATCH  {name}  {detail}")
    print(f"{len(results) - mismatches} match, {mismatches} mismatch")
    return 1 if mismatches else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "diff":
        return _run_diff(Path(args.dir_a), Path(args.dir_b))
    return _run_capture_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
