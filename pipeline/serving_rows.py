"""Row preparation and writing shared by the history and forward writers.

History rows (``scripts/materialize_stack_projections.py``) and forward rows
(``ml/forward_serving.py``) must look the same to every consumer: same interval
method, same derived fantasy total, same upsert guard. One code path keeps them
from drifting (decision 0012, section 5).

Moved from ``scripts/materialize_stack_projections.py`` without behaviour
change; ``backend/tests/test_serving_rows.py`` holds a golden file that proves it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import psycopg2.extras

from ml.bucket_attribution import ATTR_COLUMNS
from ml.scoring import SCORED_STATS, score_fantasy
from pipeline.run_guard import (
    approved_array_literal,
    approved_pipeline_run_ids,
    skip_rows_served_by_another_approved_run,
)

CONFORMAL_LEVEL = 0.90
MIN_CONFORMAL_HISTORY = 100


def attach_causal_conformal_intervals(df: pd.DataFrame, only_seasons=None) -> pd.DataFrame:
    """Attach 90% empirical intervals using only earlier OOF seasons.

    Stack OOF files contain realised outcomes because they are validation
    evidence.  Those outcomes must never calibrate their own row (or any row
    in the same target season).  For each season we therefore fit residual
    quantiles from strictly earlier seasons.  Prediction-conditioned buckets
    make the width responsive to the forecast level; sparse buckets safely
    fall back to the prior-season pooled residual distribution.

    ``floor`` and ``ceiling`` are lower/upper prediction bounds, not posterior
    percentiles.  ``p25``/``p75`` remain NULL until serving has genuine
    per-player posterior samples.

    ``only_seasons`` limits which seasons receive intervals (the forward producer
    calibrates one forward season from the pinned history and skips the rest);
    ``None`` keeps the history behaviour: every season.
    """
    out = df.copy()
    out["floor"] = np.nan
    out["ceiling"] = np.nan
    out["interval_method"] = "unavailable"
    if "y_true" not in out.columns:
        return out

    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["residual"] = out["y_true"] - out["projection"]
    finite = out["residual"].notna() & np.isfinite(out["residual"])
    alpha = (1.0 - CONFORMAL_LEVEL) / 2.0

    wanted = None if only_seasons is None else {int(x) for x in only_seasons}
    for season in sorted(out["season"].unique()):
        if wanted is not None and int(season) not in wanted:
            continue
        target = out["season"] == season
        history = out[(out["season"] < season) & finite].copy()
        if len(history) < MIN_CONFORMAL_HISTORY:
            continue

        # Use stable, explicit prediction-value cut points rather than qcut's
        # labels, which can collapse when the model emits ties.
        edges = np.unique(history["projection"].quantile([0.2, 0.4, 0.6, 0.8]).to_numpy())
        history["_bin"] = np.searchsorted(edges, history["projection"].to_numpy(), side="right")
        target_bins = np.searchsorted(edges, out.loc[target, "projection"].to_numpy(), side="right")
        pooled = history["residual"]
        lower: list[float] = []
        upper: list[float] = []
        for bucket in target_bins:
            bucket_residuals = history.loc[history["_bin"] == bucket, "residual"]
            calibration = bucket_residuals if len(bucket_residuals) >= MIN_CONFORMAL_HISTORY else pooled
            lower.append(float(calibration.quantile(alpha)))
            upper.append(float(calibration.quantile(1.0 - alpha)))

        predictions = out.loc[target, "projection"].to_numpy(dtype=float)
        # All declared targets are non-negative counting/yards statistics.
        out.loc[target, "floor"] = np.maximum(0.0, predictions + np.asarray(lower))
        out.loc[target, "ceiling"] = np.maximum(0.0, predictions + np.asarray(upper))
        out.loc[target, "interval_method"] = "causal_oof_conformal_90"

    return out.drop(columns=["residual"], errors="ignore")


def attach_derived_fantasy_projection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute one derived fantasy total per (player_id, game_id) from every
    scored component-stat cell available for that player-game, via
    ml.scoring.score_fantasy, and attach it to every row for that
    player-game — including the fantasy_ppr row, whose OWN
    `fantasy_projection` stays the direct regression's untouched output
    (the plan's calibration anchor). The two numbers living on the same
    fantasy_ppr row is what makes them comparable.

    This sums whatever REQUIRED_SERVING_CELLS rows actually exist for a
    given player-game — not a schema-guaranteed-complete set. A player
    with no receiving_tds row for some rare game (a model wasn't computed
    for them that week) gets a total missing that component, same as any
    real serving system built from independently-fit per-stat models.
    """
    pivot = df.pivot_table(index=["player_id", "game_id"], columns="stat", values="projection", aggfunc="first")
    scored_cols = [c for c in pivot.columns if c in SCORED_STATS]
    derived = pivot[scored_cols].apply(lambda row: score_fantasy(row.dropna().to_dict()), axis=1)
    derived = derived.rename("derived_fantasy_projection").reset_index()
    return df.merge(derived, on=["player_id", "game_id"], how="left")


def upsert_attributions(cur, attr: pd.DataFrame, approved_literal: str) -> int:
    """Write the attribution rows beside the projections they explain."""
    sql = """
        INSERT INTO projection_attributions
            (player_id, game_id, stat, pipeline_run_id, base,
             kalman_form, seasonal_baseline, matchup, weather_venue,
             team_context, roster_injury, rule_meta, unweighted, max_residual)
        VALUES (
            %(player_id)s, %(game_id)s, %(stat)s, %(pipeline_run_id)s, %(base)s,
            %(kalman_form)s, %(seasonal_baseline)s, %(matchup)s, %(weather_venue)s,
            %(team_context)s, %(roster_injury)s, %(rule_meta)s, %(unweighted)s,
            %(max_residual)s
        )
        ON CONFLICT (player_id, game_id, stat) DO UPDATE SET
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            base = EXCLUDED.base,
            kalman_form = EXCLUDED.kalman_form,
            seasonal_baseline = EXCLUDED.seasonal_baseline,
            matchup = EXCLUDED.matchup,
            weather_venue = EXCLUDED.weather_venue,
            team_context = EXCLUDED.team_context,
            roster_injury = EXCLUDED.roster_injury,
            rule_meta = EXCLUDED.rule_meta,
            unweighted = EXCLUDED.unweighted,
            max_residual = EXCLUDED.max_residual,
            created_at = now()
        __GUARD__
    """
    columns = ["player_id", "game_id", "stat", "pipeline_run_id", *ATTR_COLUMNS, "max_residual"]
    rows = [
        {
            key: (value if isinstance(value, str) else float(value))
            for key, value in zip(columns, record)
        }
        for record in attr[columns].itertuples(index=False, name=None)
    ]
    guard = skip_rows_served_by_another_approved_run(
        "projection_attributions", "pipeline_run_id", approved_literal
    )
    psycopg2.extras.execute_batch(cur, sql.replace("__GUARD__", guard), rows, page_size=500)
    return len(rows)


def count_rows_served_by_another_approved_run(cur, df: pd.DataFrame) -> int:
    """How many of ``df``'s rows ``upsert_projections`` would skip (review L1).

    A row is skipped when the same (player, game, stat) is already served under another approved
    run id. ``execute_batch`` reports no row counts, so the writer cannot count them; this does, in
    the caller's transaction, for one forward week.
    """
    if df.empty:
        return 0
    approved_literal = approved_array_literal(approved_pipeline_run_ids())
    season, week = int(df["season"].iloc[0]), int(df["week"].iloc[0])
    cur.execute(
        "-- sr:skipped\nSELECT count(*) FROM projections p JOIN unnest(%s::text[], %s::text[], %s::text[], %s::text[]) "
        "AS k(player_id, game_id, stat, run_id) "
        "ON p.player_id::text = k.player_id AND p.game_id::text = k.game_id AND p.stat::text = k.stat "
        "WHERE p.season = %s AND p.week = %s "
        f"AND COALESCE(p.pipeline_run_id = ANY({approved_literal}), false) "
        "AND p.pipeline_run_id IS DISTINCT FROM k.run_id",
        (
            [str(v) for v in df["player_id"]], [str(v) for v in df["game_id"]],
            [str(v) for v in df["stat"]], [str(v) for v in df["pipeline_run_id"]], season, week,
        ),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def upsert_projections(conn, df: pd.DataFrame, attributions: pd.DataFrame | None = None) -> int:
    """Write this run's rows. A row already served under another approved run is never changed.

    Attributions, when given, go in the same transaction as the projections
    they explain, so a failure leaves neither half behind.
    """
    # Forward rows carry the availability status known at the snapshot; history rows
    # have no such column and their SQL is unchanged.
    with_status = "availability_status" in df.columns
    sql = """
        INSERT INTO projections
            (player_id, game_id, season, week, stat, position,
             projection, floor, ceiling, p25, p75,
             boom_probability, bust_probability,
             fantasy_projection, fantasy_floor, fantasy_ceiling,
             derived_fantasy_projection,
             pipeline_run_id, posterior_samples, max_train_season, interval_method__STATUS_COLUMN__)
        VALUES (
            %(player_id)s, %(game_id)s, %(season)s, %(week)s, %(stat)s, %(position)s,
            %(projection)s, %(floor)s, %(ceiling)s, %(p25)s, %(p75)s,
            NULL, NULL,
            %(fantasy_projection)s, %(fantasy_floor)s, %(fantasy_ceiling)s,
            %(derived_fantasy_projection)s,
            %(pipeline_run_id)s, NULL, %(max_train_season)s, %(interval_method)s__STATUS_VALUE__
        )
        ON CONFLICT (player_id, game_id, stat) DO UPDATE SET
            season = EXCLUDED.season,
            week = EXCLUDED.week,
            position = EXCLUDED.position,
            projection = EXCLUDED.projection,
            floor = EXCLUDED.floor,
            ceiling = EXCLUDED.ceiling,
            p25 = EXCLUDED.p25,
            p75 = EXCLUDED.p75,
            fantasy_projection = EXCLUDED.fantasy_projection,
            fantasy_floor = EXCLUDED.fantasy_floor,
            fantasy_ceiling = EXCLUDED.fantasy_ceiling,
            derived_fantasy_projection = EXCLUDED.derived_fantasy_projection,
            pipeline_run_id = EXCLUDED.pipeline_run_id,
            boom_probability = EXCLUDED.boom_probability,
            bust_probability = EXCLUDED.bust_probability,
            posterior_samples = EXCLUDED.posterior_samples,
            max_train_season = EXCLUDED.max_train_season,
            interval_method = EXCLUDED.interval_method__STATUS_UPDATE__
        __GUARD__
    """
    sql = (
        sql.replace("__STATUS_COLUMN__", ", availability_status" if with_status else "")
        .replace("__STATUS_VALUE__", ", %(availability_status)s" if with_status else "")
        .replace(
            "__STATUS_UPDATE__",
            ",\n            availability_status = EXCLUDED.availability_status" if with_status else "",
        )
    )
    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            {
                "player_id": str(r.player_id),
                "game_id": str(r.game_id),
                "season": int(r.season),
                "week": int(r.week),
                "stat": str(r.stat),
                "position": str(r.position),
                "projection": float(r.projection) if pd.notna(r.projection) else None,
                "floor": float(r.floor) if pd.notna(r.floor) else None,
                "ceiling": float(r.ceiling) if pd.notna(r.ceiling) else None,
                "p25": float(r.p25) if pd.notna(r.p25) else None,
                "p75": float(r.p75) if pd.notna(r.p75) else None,
                "fantasy_projection": float(r.fantasy_projection)
                if r.fantasy_projection is not None and pd.notna(r.fantasy_projection)
                else None,
                "fantasy_floor": float(r.fantasy_floor)
                if r.fantasy_floor is not None and pd.notna(r.fantasy_floor)
                else None,
                "fantasy_ceiling": float(r.fantasy_ceiling)
                if r.fantasy_ceiling is not None and pd.notna(r.fantasy_ceiling)
                else None,
                "derived_fantasy_projection": float(r.derived_fantasy_projection)
                if r.derived_fantasy_projection is not None and pd.notna(r.derived_fantasy_projection)
                else None,
                "pipeline_run_id": str(r.pipeline_run_id),
                "max_train_season": int(r.max_train_season),
                "interval_method": str(r.interval_method) if getattr(r, "interval_method", None) else "unavailable",
            }
        )
        if with_status:
            status = getattr(r, "availability_status", None)
            rows[-1]["availability_status"] = str(status) if status is not None and pd.notna(status) else None
    approved_literal = approved_array_literal(approved_pipeline_run_ids())
    try:
        with conn.cursor() as cur:
            guard = skip_rows_served_by_another_approved_run(
                "projections", "pipeline_run_id", approved_literal
            )
            psycopg2.extras.execute_batch(cur, sql.replace("__GUARD__", guard), rows, page_size=500)
            if attributions is not None:
                upsert_attributions(cur, attributions, approved_literal)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)
