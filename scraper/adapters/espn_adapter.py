"""
scraper/adapters/espn_adapter.py

ESPN unofficial injury adapter.

Fetches weekly practice participation reports from ESPN's unofficial API:
  https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/injuries

No API key required — this is a public unofficial endpoint.
Respects rate limits with a 1.5-second delay between team requests.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLAYER NAME → player_id MAPPING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ESPN uses display names ("Tyreek Hill") while our players table stores
gsis_ids ("00-0032765"). The adapter fuzzy-matches display names via
rapidfuzz (WRatio scorer, threshold=80) against names loaded from the DB.

Players whose names score below the threshold are logged at DEBUG level
and returned with player_id=None. Callers should filter these out.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT SCHEMA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DataFrame columns returned by fetch_injury_report():
  player_id       str | None  gsis_id from players table (None = unmatched)
  player_name     str         ESPN display name
  espn_team       str         NFL team abbreviation (e.g. "KC")
  practice_status str         "full" | "limited" | "dnp" | "questionable" |
                              "doubtful" | "out"
  injury_type     str         body part / injury description (e.g. "knee")
  week            int         NFL week number passed to fetch_injury_report()
  season          int         NFL season year passed to fetch_injury_report()

Status encoding (for feature_engineer.py):
  out=0  doubtful=1  questionable=2  limited=3  full=4
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── ESPN numeric team IDs (abbreviation → ESPN ID) ────────────────────────────
# Used in the unofficial API URL.  IDs are stable but unofficial.
ESPN_TEAM_IDS: dict[str, int] = {
    "ARI": 22, "ATL": 1,  "BAL": 33, "BUF": 2,  "CAR": 29,
    "CHI": 3,  "CIN": 4,  "CLE": 5,  "DAL": 6,  "DEN": 7,
    "DET": 8,  "GB":  9,  "HOU": 34, "IND": 11, "JAX": 30,
    "KC":  12, "LAC": 24, "LAR": 14, "LV":  13, "MIA": 15,
    "MIN": 16, "NE":  17, "NO":  18, "NYG": 19, "NYJ": 20,
    "PHI": 21, "PIT": 23, "SEA": 26, "SF":  25, "TB":  27,
    "TEN": 10, "WSH": 28,
}

# ── Practice / injury status normalisation ────────────────────────────────────
# ESPN uses varied casing and phrasing; normalise to one of 6 canonical values.
_STATUS_ALIASES: dict[str, str] = {
    # "full" practice — player is healthy
    "full":         "full",
    "full practice": "full",
    "active":       "full",
    # "limited" practice
    "limited":      "limited",
    "limited practice": "limited",
    "lp":           "limited",
    # "dnp" — did not participate
    "dnp":          "dnp",
    "did not participate": "dnp",
    "did not practice":    "dnp",
    "no practice":  "dnp",
    "np":           "dnp",
    # Injury report statuses
    "questionable": "questionable",
    "doubtful":     "doubtful",
    "out":          "out",
    "ir":           "out",
    "pup":          "out",
}

# Numeric encoding used by feature_engineer.py
STATUS_ENCODING: dict[str, int] = {
    "out":          0,
    "doubtful":     1,
    "questionable": 2,
    "limited":      3,
    "full":         4,
    "dnp":          0,   # did-not-participate treated same as "out" for encoding
}

# Minimum fuzzy-match score (0-100) for name→player_id resolution.
_FUZZY_THRESHOLD: int = 80

# Seconds between requests to the ESPN API (polite rate limiting).
_REQUEST_DELAY_S: float = 1.5

_ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl"
    "/teams/{team_id}/injuries"
)


# ── EspnAdapter ───────────────────────────────────────────────────────────────

class EspnAdapter:
    """
    Fetches weekly NFL injury / practice participation data from ESPN.

    Args:
        db_url: PostgreSQL connection URL for the players table lookup.
                Pass None to skip DB lookup (player_id will always be None).
    """

    def __init__(self, db_url: Optional[str] = None) -> None:
        self._db_url = db_url
        # {display_name_lower: player_id} — loaded lazily once from DB.
        self._name_lookup: Optional[dict[str, str]] = None
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "gridiron-oracle/1.0 (research project)",
            "Accept": "application/json",
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch_injury_report(self, week: int, season: int) -> pd.DataFrame:
        """
        Fetch injury / practice status for all 32 teams for the given week.

        Makes one HTTP request per team with a 1.5-second delay between
        requests.  Teams that return HTTP errors are skipped with a WARNING.

        Args:
            week:   NFL week number (1-18 for regular season).
            season: NFL season year.

        Returns:
            DataFrame with columns: player_id, player_name, espn_team,
            practice_status, injury_type, week, season.
            Rows with unresolved player_id are included (player_id=None).
        """
        self._ensure_name_lookup()
        records: list[dict] = []

        for abbr, team_id in ESPN_TEAM_IDS.items():
            try:
                rows = self._fetch_team_injuries(abbr, team_id, week, season)
                records.extend(rows)
            except Exception as exc:
                logger.warning("ESPN fetch failed for team=%s: %s", abbr, exc)
            time.sleep(_REQUEST_DELAY_S)

        if not records:
            logger.warning(
                "ESPN: no injury records returned for week=%d season=%d",
                week, season,
            )
            return _empty_df()

        df = pd.DataFrame(records)
        logger.info(
            "ESPN injury report: %d records, %d matched player_ids "
            "(week=%d season=%d)",
            len(df),
            df["player_id"].notna().sum(),
            week, season,
        )
        return df

    def _map_player_name_to_id(self, name: str) -> Optional[str]:
        """
        Fuzzy-match ``name`` against player display names in the players table.

        Uses rapidfuzz WRatio scorer (handles transpositions, initials, etc.).
        Returns the matched gsis_id string, or None if no match exceeds
        _FUZZY_THRESHOLD (80 by default).

        Args:
            name: ESPN display name, e.g. "T. Hill" or "Tyreek Hill".
        """
        if not name or self._name_lookup is None:
            return None

        try:
            from rapidfuzz import process as rfp, fuzz
        except ImportError:
            logger.debug(
                "rapidfuzz not installed — name lookup disabled. "
                "Run: pip install rapidfuzz"
            )
            return None

        result = rfp.extractOne(
            name.lower(),
            self._name_lookup.keys(),
            scorer=fuzz.WRatio,
            score_cutoff=_FUZZY_THRESHOLD,
        )
        if result is None:
            logger.debug(
                "No fuzzy match for ESPN name '%s' (threshold=%d)", name, _FUZZY_THRESHOLD
            )
            return None

        matched_name, score, _ = result
        player_id = self._name_lookup[matched_name]
        logger.debug(
            "Matched '%s' → '%s' (player_id=%s, score=%d)",
            name, matched_name, player_id, score,
        )
        return player_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fetch_team_injuries(
        self,
        abbr: str,
        team_id: int,
        week: int,
        season: int,
    ) -> list[dict]:
        """
        Fetch and parse injury data for one team from the ESPN unofficial API.

        Returns a list of dicts suitable for DataFrame construction.
        """
        url = _ESPN_INJURIES_URL.format(team_id=team_id)
        resp = self._session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        injuries = data.get("injuries", [])
        rows = []
        for item in injuries:
            athlete = item.get("athlete", {})
            raw_name = athlete.get("displayName", "")
            if not raw_name:
                continue

            raw_status = (
                item.get("status", "")
                or item.get("type", {}).get("description", "")
            ).lower().strip()
            practice_status = _STATUS_ALIASES.get(raw_status, raw_status or "unknown")

            injury_type = (
                item.get("type", {}).get("text", "")
                or item.get("shortComment", "")
            ).lower().strip()

            player_id = self._map_player_name_to_id(raw_name)

            rows.append({
                "player_id":       player_id,
                "player_name":     raw_name,
                "espn_team":       abbr,
                "practice_status": practice_status,
                "injury_type":     injury_type,
                "week":            week,
                "season":          season,
            })

        logger.debug("ESPN team=%s: %d injury records", abbr, len(rows))
        return rows

    def _ensure_name_lookup(self) -> None:
        """
        Load {full_name_lower: player_id} from the players table.
        Idempotent — called once, result cached in self._name_lookup.
        """
        if self._name_lookup is not None:
            return

        if not self._db_url:
            self._name_lookup = {}
            return

        try:
            import psycopg2
            import psycopg2.extras
            from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn

            conn = psycopg2.connect(_psycopg2_dsn(self._db_url))
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT id, full_name FROM players WHERE full_name IS NOT NULL")
                    rows = cur.fetchall()
            finally:
                conn.close()

            self._name_lookup = {
                row["full_name"].lower(): row["id"]
                for row in rows
                if row["full_name"]
            }
            logger.info(
                "ESPN adapter: loaded %d player names for fuzzy matching",
                len(self._name_lookup),
            )
        except Exception as exc:
            logger.warning(
                "ESPN adapter: failed to load player name lookup: %s — "
                "player_id will be None for all rows",
                exc,
            )
            self._name_lookup = {}


# ── Module-level helpers ──────────────────────────────────────────────────────

def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical injury report schema."""
    return pd.DataFrame(columns=[
        "player_id", "player_name", "espn_team",
        "practice_status", "injury_type", "week", "season",
    ])
