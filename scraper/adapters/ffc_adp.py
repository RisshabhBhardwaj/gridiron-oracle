"""Fantasy Football Calculator ADP REST client.

Free for personal and commercial use; attribution requested.
League lock: teams=8 (see prediction-surfaces rebuild design).
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
_HIST_DIR = ROOT / "data" / "adp" / "historical"
FFC_BASE = "https://fantasyfootballcalculator.com/api/v1/adp"
DEFAULT_TEAMS = 8
DEFAULT_FORMAT = "ppr"

logger = logging.getLogger(__name__)


def adp_url(
    year: int,
    *,
    scoring: str = DEFAULT_FORMAT,
    teams: int = DEFAULT_TEAMS,
    position: str | None = None,
) -> str:
    params: dict[str, Any] = {"teams": teams, "year": year}
    if position:
        params["position"] = position
    return f"{FFC_BASE}/{scoring}?{urlencode(params)}"


def fetch_adp(
    year: int,
    *,
    scoring: str = DEFAULT_FORMAT,
    teams: int = DEFAULT_TEAMS,
    timeout_s: float = 30.0,
) -> list[dict[str, Any]]:
    """Return player rows. Does not hammer the API — callers should cache daily."""
    import json

    url = adp_url(year, scoring=scoring, teams=teams)
    request = Request(url, headers={"User-Agent": "gridiron-oracle/ffc-adp"})
    with urlopen(request, timeout=timeout_s) as response:
        payload = json.loads(response.read().decode("utf-8"))
    players = payload.get("players") or payload.get("adp") or []
    if not isinstance(players, list):
        raise ValueError(f"Unexpected FFC payload keys: {sorted(payload)}")
    rows = []
    for raw in players:
        name = raw.get("name") or raw.get("player_name")
        adp = raw.get("adp") or raw.get("average")
        if not name or adp is None:
            continue
        rows.append(
            {
                "player_name": str(name),
                "position": raw.get("position"),
                "team": raw.get("team"),
                "adp": float(adp),
                "player_id": raw.get("player_id") or raw.get("gsis_id"),
                "sleeper_id": raw.get("sleeper_id"),
                "espn_id": raw.get("espn_id"),
            }
        )
    return rows


def write_historical_csv(year: int, rows: list[dict[str, Any]], dest: Path | None = None) -> Path:
    path = dest or (_HIST_DIR / f"adp_{year}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["player_name", "position", "team", "adp", "player_id", "sleeper_id", "espn_id"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Fetch 8-team PPR ADP from FFC")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--teams", type=int, default=DEFAULT_TEAMS)
    parser.add_argument("--scoring", default=DEFAULT_FORMAT)
    parser.add_argument("--sleep", type=float, default=1.0, help="politeness pause after fetch")
    args = parser.parse_args()
    rows = fetch_adp(args.year, scoring=args.scoring, teams=args.teams)
    path = write_historical_csv(args.year, rows)
    logger.info("Wrote %d rows to %s (teams=%d)", len(rows), path, args.teams)
    time.sleep(max(0.0, args.sleep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
