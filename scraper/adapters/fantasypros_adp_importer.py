"""
FantasyPros ADP importer — manual CSV export only.

FantasyPros terms prohibit scraping, and this repository is public.
Download the PPR ADP CSV from FantasyPros and place it under
`data/adp/fantasypros/`, then run:

  python -m scraper.adapters.fantasypros_adp_importer --season 2026 --path data/adp/fantasypros/ppr_2026.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.adp_resolution import resolve_and_audit
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)

# Common FantasyPros export headers → canonical names, in PRIORITY ORDER.
#
# A standard FantasyPros ADP export looks like:
#   Rank,Player,Team,Bye,POS,ESPN,Sleeper,NFL,RTSports,FFC,AVG
# `AVG` is the average draft position; `Rank` is FantasyPros' own ordering and
# is a different quantity — an integer 1..N rather than a pick number. Mapping
# both onto `adp` (audit C-23) silently imported Rank as ADP, wrong by an order
# of magnitude on the documented 2026 import path, and the old
# `required - set(out.columns)` check could not detect it because the rename
# produced two columns both labelled `adp`.
#
# `rank` is deliberately NOT an ADP alias. A CSV carrying only Rank has no ADP
# and must raise rather than import a rank as a draft position.
_ALIAS_PRIORITY: dict[str, tuple[str, ...]] = {
    "player_name": ("player name", "player", "name"),
    "adp": ("adp", "avg adp", "adp avg", "avg", "average"),
    "position": ("pos", "position"),
    "team": ("team", "tm"),
    "fp_rank": ("rank", "#"),
}

REQUIRED_COLUMNS = ("player_name", "adp")


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve FantasyPros export headers to canonical names.

    Each canonical name takes the highest-priority source column present, and
    every canonical name binds to at most one source column — so a header set
    containing both `AVG` and `Rank` resolves `adp` to `AVG` and never emits
    duplicate `adp` columns.
    """
    lowered = {str(col).strip().lower(): col for col in df.columns}
    rename: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for canonical, candidates in _ALIAS_PRIORITY.items():
        for candidate in candidates:
            source = lowered.get(candidate)
            if source is not None and source not in rename:
                rename[source] = canonical
                resolved[canonical] = str(source)
                break

    missing = [c for c in REQUIRED_COLUMNS if c not in resolved]
    if missing:
        raise ValueError(
            f"FantasyPros CSV missing columns {missing}. Found: {list(df.columns)}. "
            "ADP must come from an average-draft-position column (AVG/ADP); "
            "a Rank column is not an ADP."
        )

    out = df.rename(columns=rename)
    if list(out.columns).count("adp") != 1:
        raise ValueError(
            f"Ambiguous ADP column in {list(df.columns)} — resolved to "
            f"{out.columns.tolist().count('adp')} columns named 'adp'."
        )
    logger.info(
        "FantasyPros column mapping: %s",
        ", ".join(f"{v!r}→{k}" for k, v in sorted(resolved.items())),
    )
    return out


def import_csv(
    db_url: str,
    *,
    path: Path,
    season: int,
    scoring: str = "ppr",
    source: str = "fantasypros",
) -> int:
    import psycopg2
    from psycopg2.extras import execute_values

    df = _normalize_columns(pd.read_csv(path))
    conn = psycopg2.connect(normalize_dsn(db_url))
    try:
        with conn.cursor() as cur:
            raw_rows = [
                {"player_name": str(r.player_name), "position": getattr(r, "position", None),
                 "team": getattr(r, "team", None), "adp": float(r.adp)}
                for r in df.itertuples(index=False)
                if pd.notna(r.adp)
            ]
            resolved = resolve_and_audit(conn, season=season, source=source, scoring=scoring, rows=raw_rows)
            unmatched = [row for row in resolved if not row["player_id"]]
            if unmatched:
                logger.warning(
                    "%d FantasyPros ADP rows were not imported; inspect adp_player_matches for audited candidates",
                    len(unmatched),
                )
            rows = [
                (season, source, scoring, row["player_name"], row.get("position"), row.get("team"),
                 row["adp"], row["player_id"])
                for row in resolved if row["player_id"]
            ]
            execute_values(
                cur,
                """
                INSERT INTO fantasy_adp
                    (season, source, scoring, player_name, position, team, adp, player_id)
                VALUES %s
                ON CONFLICT (season, source, scoring, player_id) DO UPDATE SET
                    player_name = EXCLUDED.player_name,
                    position = EXCLUDED.position,
                    team = EXCLUDED.team,
                    adp = EXCLUDED.adp,
                    imported_at = NOW()
                """,
                rows,
            )
        conn.commit()
        logger.info("Imported %d FantasyPros ADP rows from %s", len(rows), path)
        return len(rows)
    finally:
        conn.close()


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()
    import os

    db_url = args.db_url or os.environ.get("DATABASE_URL", DEFAULT_HOST_DATABASE_URL)
    n = import_csv(db_url, path=args.path, season=args.season, scoring=args.scoring)
    print(f"imported {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
