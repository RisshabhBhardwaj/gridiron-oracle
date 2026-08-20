"""Causal draft-rank diagnostics with oracle and prior-season comparators.

ADP agreement is descriptive, not a model-performance claim.  In particular,
this module refuses target-season OOF rows: they condition on realized games.
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import psycopg2
import psycopg2.extras
from scipy.stats import spearmanr

from pipeline.db_defaults import DEFAULT_HOST_DATABASE_URL
from pipeline.schema import normalize_dsn
from ml.consensus_baseline import points_lost_vs_optimal, top_n_hit_rate

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parent / "experiments" / "adp_eval"


@dataclass
class AdpEvalResult:
    season: int
    source: str
    scoring: str
    mode: str
    n_matched: int
    spearman_rho: float
    spearman_pvalue: float
    by_position: dict[str, dict]
    actuals_oracle_rho: float | None
    prev_season_baseline_rho: float | None
    gap_vs_actuals_oracle: float | None
    gap_vs_prev_season_baseline: float | None
    created_at: str
    top24_hit_rate: float | None = None
    points_lost_vs_optimal: float | None = None


def _normalize_name(name: str) -> str:
    value = (name or "").lower().strip()
    for token in (".", "'", "-", " jr", " sr", " iii", " ii", " iv"):
        value = value.replace(token, "")
    return " ".join(value.split())


def _frame(conn, sql: str, params: tuple) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return pd.DataFrame([dict(row) for row in cur.fetchall()])


def load_adp(conn, season: int, source: str = "historical", scoring: str = "ppr") -> pd.DataFrame:
    return _frame(conn, """
        SELECT player_id, player_name, position, team, adp FROM fantasy_adp
        WHERE season = %s AND source = %s AND scoring = %s AND player_id IS NOT NULL
    """, (season, source, scoring))


def load_season_actual_ppr(conn, season: int) -> pd.DataFrame:
    return _frame(conn, """
        SELECT gl.player_id, COALESCE(p.full_name, gl.player_id) AS player_name,
               gl.position, MAX(gl.team) AS team, SUM(gl.fantasy_points_ppr) AS fantasy_ppr
        FROM game_logs gl LEFT JOIN players p ON p.id = gl.player_id
        WHERE gl.season = %s AND gl.week BETWEEN 1 AND 18
        GROUP BY gl.player_id, p.full_name, gl.position
        HAVING SUM(gl.fantasy_points_ppr) IS NOT NULL
    """, (season,))


def load_season_projected_ppr(conn, season: int) -> pd.DataFrame:
    """Read the latest explicitly-preseason projection run; never a weekly OOF sum."""
    return _frame(conn, """
        WITH latest AS (
            SELECT MAX(as_of) AS as_of FROM draft_preseason_projections WHERE season = %s
        )
        SELECT player_id, player_name, position, team, projection AS fantasy_ppr
        FROM draft_preseason_projections JOIN latest USING (as_of) WHERE season = %s
    """, (season, season))


def spearman_vs_adp(model_df: pd.DataFrame, adp_df: pd.DataFrame, value_col: str = "fantasy_ppr") -> tuple[float, float, int, dict[str, dict]]:
    if model_df.empty or adp_df.empty:
        return float("nan"), float("nan"), 0, {}
    # Legacy unit fixtures may use name_key; production data must use player_id.
    key = "player_id" if "player_id" in model_df and "player_id" in adp_df else "name_key"
    merged = model_df.merge(adp_df[[key, "adp", "position"]].rename(columns={"position": "adp_pos"}), on=key, how="inner")
    merged = merged.dropna(subset=[value_col, "adp"])
    if len(merged) < 5:
        return float("nan"), float("nan"), int(len(merged)), {}
    merged["model_rank"] = merged[value_col].rank(ascending=False, method="average")
    merged["adp_rank"] = merged["adp"].rank(ascending=True, method="average")
    rho, pvalue = spearmanr(merged["model_rank"], merged["adp_rank"])
    by_position: dict[str, dict] = {}
    for position, group in merged.groupby(merged["position"].astype(str).str.upper()):
        if len(group) < 5:
            continue
        value, p = spearmanr(group[value_col].rank(ascending=False), group["adp"].rank(ascending=True))
        by_position[str(position)] = {"n": int(len(group)), "spearman_rho": float(value), "pvalue": float(p)}
    return float(rho), float(pvalue), int(len(merged)), by_position


def _finite(value: float) -> float | None:
    return float(value) if value == value else None


def evaluate(season: int, source: str = "historical", scoring: str = "ppr", from_actuals: bool = False,
             from_stack_oof: bool = False, database_url: str = DEFAULT_HOST_DATABASE_URL) -> AdpEvalResult:
    if from_stack_oof:
        raise ValueError("Target-season stack OOF is forbidden for draft evidence")
    with psycopg2.connect(normalize_dsn(database_url)) as conn:
        adp = load_adp(conn, season, source, scoring)
        model = load_season_actual_ppr(conn, season) if from_actuals else load_season_projected_ppr(conn, season)
        actuals = load_season_actual_ppr(conn, season)
        previous = load_season_actual_ppr(conn, season - 1)
    rho, pvalue, n, by_position = spearman_vs_adp(model, adp)
    oracle_rho, _, _, _ = spearman_vs_adp(actuals, adp)
    previous_rho, _, _, _ = spearman_vs_adp(previous, adp)
    clean_rho, clean_oracle, clean_previous = _finite(rho), _finite(oracle_rho), _finite(previous_rho)
    ranked = model.merge(adp[["player_id", "adp"]], on="player_id", how="inner") if "player_id" in model.columns else pd.DataFrame()
    hit_rate = None
    lost_points = None
    if not ranked.empty and "fantasy_ppr" in ranked.columns:
        ranked = ranked.merge(
            actuals[["player_id", "fantasy_ppr"]].rename(columns={"fantasy_ppr": "realized"}),
            on="player_id",
            how="inner",
        )
        if not ranked.empty:
            model_rank = ranked["fantasy_ppr"].rank(ascending=False, method="average")
            market_rank = ranked["adp"].rank(ascending=True, method="average")
            hit_rate = _finite(top_n_hit_rate(model_rank.to_numpy(), market_rank.to_numpy(), n=24))
            lost_points = _finite(points_lost_vs_optimal(
                ranked["fantasy_ppr"].to_numpy(), ranked["realized"].to_numpy(), slots=24
            ))
    return AdpEvalResult(
        season=season, source=source, scoring=scoring,
        mode="actuals_oracle" if from_actuals else "preseason_projections", n_matched=n,
        spearman_rho=rho, spearman_pvalue=pvalue, by_position=by_position,
        actuals_oracle_rho=clean_oracle, prev_season_baseline_rho=clean_previous,
        gap_vs_actuals_oracle=(clean_rho - clean_oracle) if clean_rho is not None and clean_oracle is not None else None,
        gap_vs_prev_season_baseline=(clean_rho - clean_previous) if clean_rho is not None and clean_previous is not None else None,
        top24_hit_rate=hit_rate,
        points_lost_vs_optimal=lost_points,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--source", default="historical")
    parser.add_argument("--scoring", default="ppr")
    parser.add_argument("--from-actuals", action="store_true", help="Oracle comparator only; never a model claim.")
    parser.add_argument("--from-stack-oof", action="store_true", help="Rejected: target-season OOF is forbidden.")
    parser.add_argument("--database-url", default=DEFAULT_HOST_DATABASE_URL)
    args = parser.parse_args(argv)
    result = evaluate(args.season, args.source, args.scoring, args.from_actuals, args.from_stack_oof, args.database_url)
    _OUT.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    for key in ("spearman_rho", "spearman_pvalue"):
        if payload[key] != payload[key]:
            payload[key] = None
    path = _OUT / f"adp_{result.season}_{result.source}_{result.mode}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
