"""
ml/adp_eval.py

Spearman rank correlation of model season ranks vs ADP.

Ranks fantasy_ppr projections (higher = better) against ADP (lower = better).
Reports overall and per-position Spearman ρ + n matched players.

Usage:
  python -m ml.adp_eval --season 2024 --source historical
  python -m ml.adp_eval --season 2024 --source historical --from-actuals
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / "experiments" / "adp_eval"


@dataclass
class AdpEvalResult:
    season: int
    source: str
    scoring: str
    mode: str  # "projections" | "actuals"
    n_matched: int
    spearman_rho: float
    spearman_pvalue: float
    by_position: dict[str, dict]
    created_at: str


def _normalize_name(name: str) -> str:
    s = (name or "").lower().strip()
    for ch in (".", "'", "-", " jr", " sr", " iii", " ii", " iv"):
        s = s.replace(ch, "")
    return " ".join(s.split())


def load_adp(
    conn,
    season: int,
    source: str = "historical",
    scoring: str = "ppr",
) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT player_name, position, team, adp, player_id
            FROM fantasy_adp
            WHERE season = %s AND source = %s AND scoring = %s
            """,
            (season, source, scoring),
        )
        rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["name_key"] = df["player_name"].map(_normalize_name)
    return df


def load_season_actual_ppr(conn, season: int) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT gl.player_id,
                   COALESCE(p.full_name, gl.player_id) AS player_name,
                   gl.position,
                   MAX(gl.team) AS team,
                   SUM(gl.fantasy_points_ppr) AS fantasy_ppr
            FROM game_logs gl
            LEFT JOIN players p ON gl.player_id = p.id
            WHERE gl.season = %s
            GROUP BY gl.player_id, p.full_name, gl.position
            HAVING SUM(gl.fantasy_points_ppr) IS NOT NULL
            """,
            (season,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["name_key"] = df["player_name"].map(_normalize_name)
    return df


def load_season_projected_ppr(conn, season: int) -> pd.DataFrame:
    """Mean projected fantasy_ppr from projections table when available."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'projections'
            """
        )
        cols = {r["column_name"] for r in cur.fetchall()}
        if not cols:
            return pd.DataFrame()

        # Prefer season-total style rows if present; else sum weekly means.
        has_stat = "stat" in cols
        has_mean = "mean" in cols or "projected_mean" in cols
        mean_col = "mean" if "mean" in cols else ("projected_mean" if "projected_mean" in cols else None)
        if not has_stat or not mean_col:
            return pd.DataFrame()

        cur.execute(
            f"""
            SELECT pr.player_id,
                   COALESCE(p.full_name, pr.player_id) AS player_name,
                   COALESCE(pr.position, p.position) AS position,
                   COALESCE(pr.team, p.team) AS team,
                   AVG(pr.{mean_col}) AS fantasy_ppr
            FROM projections pr
            LEFT JOIN players p ON pr.player_id = p.id
            WHERE pr.season = %s AND pr.stat = 'fantasy_ppr'
            GROUP BY pr.player_id, p.full_name, pr.position, p.position, pr.team, p.team
            """,
            (season,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["name_key"] = df["player_name"].map(_normalize_name)
    return df


def spearman_vs_adp(
    model_df: pd.DataFrame,
    adp_df: pd.DataFrame,
    value_col: str = "fantasy_ppr",
) -> tuple[float, float, int, dict[str, dict]]:
    """
    Higher model value = better rank (rank 1 = best).
    Lower ADP = better rank. Correlate those ranks.
    """
    if model_df.empty or adp_df.empty:
        return float("nan"), float("nan"), 0, {}

    merged = model_df.merge(
        adp_df[["name_key", "adp", "position"]].rename(columns={"position": "adp_pos"}),
        on="name_key",
        how="inner",
    )
    merged = merged.dropna(subset=[value_col, "adp"])
    if len(merged) < 5:
        return float("nan"), float("nan"), int(len(merged)), {}

    # model rank: denser high values → rank 1
    merged["model_rank"] = merged[value_col].rank(ascending=False, method="average")
    merged["adp_rank"] = merged["adp"].rank(ascending=True, method="average")
    rho, pval = spearmanr(merged["model_rank"], merged["adp_rank"])

    by_pos: dict[str, dict] = {}
    for pos, grp in merged.groupby(merged["position"].astype(str).str.upper()):
        if len(grp) < 5:
            continue
        r, pv = spearmanr(
            grp[value_col].rank(ascending=False, method="average"),
            grp["adp"].rank(ascending=True, method="average"),
        )
        by_pos[str(pos)] = {"n": int(len(grp)), "spearman_rho": float(r), "pvalue": float(pv)}

    return float(rho), float(pval), int(len(merged)), by_pos


def load_stack_season_ppr(season: int, oof_dir: Path | None = None) -> pd.DataFrame:
    """Sum weekly stack OOF y_pred → season fantasy_ppr ranks (model, not actuals)."""
    root = oof_dir or (Path(__file__).resolve().parent / "oof")
    frames: list[pd.DataFrame] = []
    for pos in ("QB", "RB", "WR", "TE"):
        paths = sorted(p for p in root.glob(f"stack_fantasy_ppr_{pos}_*.csv") if "_archive" not in str(p))
        if not paths:
            continue
        path = max(paths, key=lambda p: p.stat().st_mtime)
        df = pd.read_csv(path)
        sub = df[df["season"].astype(int) == int(season)].copy()
        if sub.empty:
            continue
        sub["position"] = pos
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    stack = pd.concat(frames, ignore_index=True)
    out = (
        stack.groupby("player_id", as_index=False)
        .agg(fantasy_ppr=("y_pred", "sum"), position=("position", "first"))
    )
    # Names filled by caller via DB join when available; name_key from id fallback
    out["player_name"] = out["player_id"].astype(str)
    out["name_key"] = out["player_name"].map(_normalize_name)
    return out


def evaluate(
    season: int,
    source: str = "historical",
    scoring: str = "ppr",
    from_actuals: bool = False,
    from_stack_oof: bool = False,
    database_url: str = DEFAULT_HOST_DATABASE_URL,
) -> AdpEvalResult:
    dsn = normalize_dsn(database_url)
    with psycopg2.connect(dsn) as conn:
        adp = load_adp(conn, season, source, scoring)
        if from_stack_oof:
            model = load_stack_season_ppr(season)
            mode = "stack_oof"
            if not model.empty:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT id, full_name FROM players")
                    names = {r["id"]: r["full_name"] for r in cur.fetchall()}
                model["player_name"] = model["player_id"].map(lambda i: names.get(i) or str(i))
                model["name_key"] = model["player_name"].map(_normalize_name)
            elif model.empty:
                logger.warning("No stack OOF for %s", season)
        elif from_actuals:
            model = load_season_actual_ppr(conn, season)
            mode = "actuals"
        else:
            model = load_season_projected_ppr(conn, season)
            mode = "projections"
            if model.empty:
                # Prefer stack OOF over actuals for draft-claim scoring
                model = load_stack_season_ppr(season)
                if not model.empty:
                    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                        cur.execute("SELECT id, full_name FROM players")
                        names = {r["id"]: r["full_name"] for r in cur.fetchall()}
                    model["player_name"] = model["player_id"].map(lambda i: names.get(i) or str(i))
                    model["name_key"] = model["player_name"].map(_normalize_name)
                    mode = "stack_oof_fallback"
                else:
                    logger.warning("No projections/stack for %s; refusing actuals fallback in evaluate()", season)
                    mode = "unavailable"

    rho, pval, n, by_pos = spearman_vs_adp(model, adp)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, float) and (obj != obj):  # NaN
            return None
        return obj

    return AdpEvalResult(
        season=season,
        source=source,
        scoring=scoring,
        mode=mode,
        n_matched=n,
        spearman_rho=rho if rho == rho else float("nan"),
        spearman_pvalue=pval if pval == pval else float("nan"),
        by_position=_clean(by_pos),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Spearman eval: model ranks vs ADP")
    p.add_argument("--season", type=int, required=True)
    p.add_argument("--source", default="historical")
    p.add_argument("--scoring", default="ppr")
    p.add_argument("--from-actuals", action="store_true")
    p.add_argument(
        "--from-stack-oof",
        action="store_true",
        help="Rank from stack fantasy_ppr OOF season sums (preferred for model-vs-ADP).",
    )
    p.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = p.parse_args(argv)

    result = evaluate(
        season=args.season,
        source=args.source,
        scoring=args.scoring,
        from_actuals=args.from_actuals,
        from_stack_oof=args.from_stack_oof,
        database_url=args.database_url,
    )
    _OUT.mkdir(parents=True, exist_ok=True)
    path = _OUT / f"adp_{result.season}_{result.source}_{result.mode}.json"
    payload = asdict(result)
    if payload["spearman_rho"] != payload["spearman_rho"]:
        payload["spearman_rho"] = None
    if payload["spearman_pvalue"] != payload["spearman_pvalue"]:
        payload["spearman_pvalue"] = None
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"saved={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
