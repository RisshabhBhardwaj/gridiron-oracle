"""
Coaching / scheme adapter — free, manual CSV seed, validated on load.

Loads `data/coaching/coaching_{season}.csv` into the `team_coaching` table.
No paid APIs.

**No seed CSV ships with this repository.** The 2026 seed that used to live
here was provably wrong — it listed a former wide receiver as a defensive
coordinator and had two coordinators each holding the same role on two teams,
and every row's own `notes` field said "verify" (audit C-24). It was loaded by
nothing, so shipping it bought nothing and asserted facts nobody had checked.

Supply your own verified CSV using `data/coaching/coaching_TEMPLATE.csv`, then:

  python -m scraper.adapters.coaching_adapter --season 2026

`load_coaching_csv` refuses to return a frame that fails `validate_coaching`,
so the semantic errors in the old seed are now load-time failures rather than
silently-upserted rows.
"""

from __future__ import annotations

import argparse
import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from ml.team_elo import _ALL_NFL_TEAMS
from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "coaching"

# Single source of truth for team codes, shared with the Elo registry.
CANONICAL_TEAMS = frozenset(_ALL_NFL_TEAMS)

COACH_COLUMNS = ("head_coach", "offensive_coordinator", "defensive_coordinator")

# Provenance is required, not documented-and-hoped-for. A row that cannot say
# where it came from and who checked it does not load.
PROVENANCE_COLUMNS = ("source", "verified_by", "verified_on")

REQUIRED_COLUMNS = ("season", "team", *COACH_COLUMNS, *PROVENANCE_COLUMNS)

# Text that means "nobody checked this". The old seed shipped 32 rows of it.
_PLACEHOLDER = re.compile(
    r"^\s*$"                                          # blank
    r"|^\s*(tbd|tba|unknown|verify|n/?a|todo)\b"      # leading placeholder word
    r"|^\s*[?-]+\s*$",                                # ? / ?? / -
    re.IGNORECASE,
)

CREATE_TEAM_COACHING = """
CREATE TABLE IF NOT EXISTS team_coaching (
    season INTEGER NOT NULL,
    team TEXT NOT NULL,
    head_coach TEXT,
    offensive_coordinator TEXT,
    defensive_coordinator TEXT,
    scheme_pass_rate_prior FLOAT,
    notes TEXT,
    source TEXT,
    verified_by TEXT,
    verified_on DATE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, team)
)
"""


class CoachingValidationError(ValueError):
    """Raised when a coaching CSV is internally inconsistent or unverified."""


def _is_placeholder(value: object) -> bool:
    return pd.isna(value) or bool(_PLACEHOLDER.match(str(value)))


def validate_coaching(df: pd.DataFrame, season: int) -> None:
    """
    Semantic validation. Raises `CoachingValidationError` listing every problem.

    Deliberately limited to what is checkable without a coach registry — there
    is no authoritative coach table in this repo, so "is this a real person who
    holds this job" is not something code can answer. What code *can* answer:

      - team codes are among the 32 current franchises, exactly once each
      - no placeholder / "verify" text in any name field
      - no coordinator holds the same role on two teams
      - a head coach doubling as coordinator is declared, not implied
      - the pass-rate prior is a probability
      - provenance is present and the verification date is real and not future
    """
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CoachingValidationError(
            f"coaching CSV for {season} missing required columns: {missing}"
        )

    # ── teams ────────────────────────────────────────────────────────────────
    teams = df["team"].astype(str).str.strip().str.upper()
    unknown = sorted(set(teams) - CANONICAL_TEAMS)
    if unknown:
        problems.append(f"unknown team codes: {unknown}")
    duplicated = sorted(teams[teams.duplicated()].unique())
    if duplicated:
        problems.append(f"duplicate rows for teams: {duplicated}")
    absent = sorted(CANONICAL_TEAMS - set(teams))
    if absent:
        problems.append(f"missing teams (all 32 required): {absent}")

    # ── names ────────────────────────────────────────────────────────────────
    for col in COACH_COLUMNS:
        bad = [
            f"{t}={v!r}"
            for t, v in zip(teams, df[col])
            if _is_placeholder(v)
        ]
        if bad:
            problems.append(f"{col} unset or placeholder for: {bad}")

    # ── one coordinator per team, no coordinator on two teams ────────────────
    for col in ("offensive_coordinator", "defensive_coordinator"):
        holders: dict[str, list[str]] = {}
        for team, name in zip(teams, df[col]):
            if _is_placeholder(name):
                continue
            holders.setdefault(str(name).strip(), []).append(team)
        for name, owners in sorted(holders.items()):
            if len(owners) > 1:
                problems.append(
                    f"{col} {name!r} is listed for multiple teams: {sorted(owners)}"
                )

    # ── HC/OC or HC/DC dual roles must be declared ───────────────────────────
    dual_declared = (
        df["dual_role"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        if "dual_role" in df.columns
        else pd.Series([False] * len(df), index=df.index)
    )
    for team, hc, oc, dc, declared in zip(
        teams,
        df["head_coach"],
        df["offensive_coordinator"],
        df["defensive_coordinator"],
        dual_declared,
    ):
        same = (str(hc).strip() == str(oc).strip()) or (str(hc).strip() == str(dc).strip())
        if same and not declared:
            problems.append(
                f"{team}: head coach {str(hc).strip()!r} also holds a coordinator "
                "role; set dual_role=true to declare it deliberately"
            )

    # ── scheme prior ─────────────────────────────────────────────────────────
    if "scheme_pass_rate_prior" in df.columns:
        prior = pd.to_numeric(df["scheme_pass_rate_prior"], errors="coerce")
        bad_prior = [
            f"{team}={raw!r}"
            for team, raw, value in zip(teams, df["scheme_pass_rate_prior"], prior)
            if pd.notna(raw) and (pd.isna(value) or not 0.0 < float(value) < 1.0)
        ]
        if bad_prior:
            problems.append(f"scheme_pass_rate_prior outside (0, 1): {bad_prior}")

    # ── provenance ───────────────────────────────────────────────────────────
    for col in ("source", "verified_by"):
        bad = [t for t, v in zip(teams, df[col]) if _is_placeholder(v)]
        if bad:
            problems.append(f"{col} unset or placeholder for teams: {sorted(bad)}")

    verified_on = pd.to_datetime(df["verified_on"], errors="coerce")
    unparsed = sorted(teams[verified_on.isna()].tolist())
    if unparsed:
        problems.append(f"verified_on not a parseable date for teams: {unparsed}")
    today = pd.Timestamp(date.today())
    future = sorted(teams[verified_on.notna() & (verified_on > today)].tolist())
    if future:
        problems.append(f"verified_on is in the future for teams: {future}")

    # ── season ───────────────────────────────────────────────────────────────
    wrong_season = sorted(set(df["season"].astype(int)) - {season})
    if wrong_season:
        problems.append(f"rows for other seasons present: {wrong_season}")

    if problems:
        raise CoachingValidationError(
            f"coaching CSV for {season} failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def load_coaching_csv(season: int) -> pd.DataFrame:
    path = DATA_DIR / f"coaching_{season}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing coaching seed {path}. No seed ships with this repository — "
            f"copy {DATA_DIR / 'coaching_TEMPLATE.csv'} and fill it in with "
            "verified staff, including source / verified_by / verified_on. "
            "See data/coaching/README.md."
        )
    df = pd.read_csv(path)
    if "season" not in df.columns:
        raise CoachingValidationError(f"{path} missing column 'season'")
    df = df[df["season"] == season].copy()
    if df.empty:
        raise CoachingValidationError(f"{path} has no rows for season={season}")
    validate_coaching(df, season)
    return df


def upsert_coaching(db_url: str, season: int) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    df = load_coaching_csv(season)
    conn = psycopg2.connect(normalize_dsn(db_url))
    try:
        with conn.cursor() as cur:
            cur.execute(CREATE_TEAM_COACHING)
            rows = [
                (
                    int(r.season),
                    str(r.team),
                    r.head_coach,
                    r.offensive_coordinator,
                    r.defensive_coordinator,
                    float(r.scheme_pass_rate_prior)
                    if pd.notna(getattr(r, "scheme_pass_rate_prior", None))
                    else None,
                    getattr(r, "notes", None),
                    r.source,
                    r.verified_by,
                    str(r.verified_on),
                )
                for r in df.itertuples(index=False)
            ]
            execute_values(
                cur,
                """
                INSERT INTO team_coaching
                    (season, team, head_coach, offensive_coordinator,
                     defensive_coordinator, scheme_pass_rate_prior, notes,
                     source, verified_by, verified_on)
                VALUES %s
                ON CONFLICT (season, team) DO UPDATE SET
                    head_coach = EXCLUDED.head_coach,
                    offensive_coordinator = EXCLUDED.offensive_coordinator,
                    defensive_coordinator = EXCLUDED.defensive_coordinator,
                    scheme_pass_rate_prior = EXCLUDED.scheme_pass_rate_prior,
                    notes = EXCLUDED.notes,
                    source = EXCLUDED.source,
                    verified_by = EXCLUDED.verified_by,
                    verified_on = EXCLUDED.verified_on,
                    updated_at = NOW()
                """,
                rows,
            )
        conn.commit()
        logger.info("Upserted %d coaching rows for season %d", len(rows), season)
        return len(rows)
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--db-url", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the CSV and exit without touching the database.",
    )
    args = parser.parse_args()
    import os

    if args.validate_only:
        df = load_coaching_csv(args.season)
        print(f"OK: {len(df)} validated coaching rows for {args.season}")
        return 0

    db_url = args.db_url or os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)
    n = upsert_coaching(db_url, args.season)
    print(f"upserted {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
