"""Component-level draft evaluation with confidence intervals.

Why this exists
---------------
``ml.draft_sim.evaluate_board`` answers the question that matters — how many
starting-lineup points does a board win — but it answers it once per season, and
only six seasons of resolved ADP exist. Measured 2026-08-19, the season-to-season
standard deviation of the model board is 170.6 points, so the mean gap against ADP
carries a standard error of about 66. Differences of 15-70 points are inside one
standard error; **roughly 21 seasons would be needed to call the observed gap
significant.** Every component decision made on that metric is a decision made on
noise, and two were.

This module evaluates the same projections over ~1000 player-seasons instead of
six board scores, which is enough to resolve component questions: is this rate
better, is this rookie prior better, does this multiplier help.

It is deliberately *not* a replacement for the board sim. Rank correlation ignores
positional scarcity, which is the whole reason VOR exists. Use this to choose
components; use the board sim, reported with its interval, for "should I draft
from this".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

TOP_N = 24

# League-wide starting slots (8 teams). Any "top N" taken across positions is
# measured against an oracle that is 9-11 quarterbacks -- an illegal roster --
# so a board is rewarded for the very quarterback-loading that VOR removes.
# That defect has now appeared three times in this repo: in the board oracle,
# in points_lost_vs_optimal, and in cross-position top-N precision, where it
# reversed the sign of the comparison against ADP.
POSITION_SLOTS = {"QB": 8, "RB": 16, "WR": 16, "TE": 8}


def _within_season_pct(frame: pd.DataFrame, column: str, *, ascending: bool) -> pd.Series:
    """Percentile rank inside each season, so seasons pool on a common scale."""
    return frame.groupby("season")[column].rank(ascending=ascending, pct=True)


def pooled_spearman(
    frame: pd.DataFrame,
    projection_col: str,
    *,
    realized_col: str = "realized",
    higher_is_better: bool = True,
) -> float:
    """Rank correlation of projection against realized, pooled on within-season ranks."""
    work = frame.dropna(subset=[projection_col, realized_col])
    if len(work) < 10:
        return float("nan")
    proj = _within_season_pct(work, projection_col, ascending=not higher_is_better)
    real = _within_season_pct(work, realized_col, ascending=False)
    return float(np.corrcoef(proj, real)[0, 1])


def calibration_ratio(frame: pd.DataFrame, projection_col: str, *, realized_col: str = "realized") -> float:
    """Total projected over total realized. 1.0 is unbiased; the rookie bug read 0.24."""
    work = frame.dropna(subset=[projection_col, realized_col])
    denominator = float(work[realized_col].sum())
    if denominator <= 0:
        return float("nan")
    return float(work[projection_col].sum() / denominator)


def top_n_precision(
    frame: pd.DataFrame,
    projection_col: str,
    *,
    realized_col: str = "realized",
    n: int = TOP_N,
    higher_is_better: bool = True,
) -> float:
    """Share of the projected top ``n`` that finished in the realized top ``n``."""
    hits, seasons = 0, 0
    for _season, group in frame.groupby("season"):
        if len(group) < n:
            continue
        picked = group.nlargest(n, projection_col) if higher_is_better else group.nsmallest(n, projection_col)
        actual = set(group.nlargest(n, realized_col)["player_id"])
        hits += len(set(picked["player_id"]) & actual)
        seasons += 1
    return float(hits / (seasons * n)) if seasons else float("nan")


def within_position_precision(
    frame: pd.DataFrame,
    projection_col: str,
    *,
    realized_col: str = "realized",
    position_col: str = "position",
    higher_is_better: bool = True,
) -> float:
    """Top-N precision inside each position, against a roster the league can field.

    This is the precision metric to report. :func:`top_n_precision` compares
    against the raw-points top N, which in a 1-QB league is a quarterback pile;
    measured 2026-08-20, that version showed the board beating ADP 0.451 to
    0.396 while this one shows ADP ahead 0.556 to 0.486. Same projections,
    opposite conclusion.
    """
    hits = total = 0
    for (_season, position), group in frame.groupby(["season", position_col]):
        slots = POSITION_SLOTS.get(str(position).upper())
        if not slots or len(group) < slots:
            continue
        picked = (
            group.nlargest(slots, projection_col)
            if higher_is_better
            else group.nsmallest(slots, projection_col)
        )
        actual = set(group.nlargest(slots, realized_col)["player_id"])
        hits += len(set(picked["player_id"]) & actual)
        total += slots
    return float(hits / total) if total else float("nan")


def bootstrap_ci(
    frame: pd.DataFrame,
    statistic,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    cluster_col: str = "player_id",
) -> tuple[float, float, float]:
    """(point estimate, lo, hi) at 95%, resampling clusters so a player's seasons move together."""
    point = float(statistic(frame))
    clusters = frame[cluster_col].to_numpy()
    unique = np.unique(clusters)
    if len(unique) < 20:
        return point, float("nan"), float("nan")
    index_by_cluster = {value: np.where(clusters == value)[0] for value in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index_by_cluster[value] for value in picked])
        try:
            draws[i] = float(statistic(frame.iloc[rows]))
        except Exception:
            draws[i] = np.nan
    draws = draws[np.isfinite(draws)]
    if draws.size < n_boot // 4:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def season_level_ci(
    frame: pd.DataFrame,
    statistic,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(estimate, lo, hi) resampling whole seasons.

    Top-N precision cannot use the cluster bootstrap: resampling players with
    replacement lets one player occupy several of the N slots, so the metric is
    no longer the quantity it names. It is a per-season statistic and must be
    resampled per season -- which means roughly six independent observations and
    correspondingly wide intervals. That is the honest width, not a defect.
    """
    point = float(statistic(frame))
    seasons = frame["season"].unique()
    if len(seasons) < 3:
        return point, float("nan"), float("nan")
    per_season = []
    for season in seasons:
        try:
            per_season.append(float(statistic(frame[frame["season"] == season])))
        except Exception:
            continue
    values = np.array([v for v in per_season if np.isfinite(v)])
    if values.size < 3:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def evaluate_component(
    frame: pd.DataFrame,
    projection_col: str,
    *,
    higher_is_better: bool = True,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """All component metrics for one projection column, with 95% CIs."""
    out: dict = {}
    spearman = bootstrap_ci(
        frame,
        lambda f: pooled_spearman(f, projection_col, higher_is_better=higher_is_better),
        n_boot=n_boot,
        seed=seed,
    )
    out["spearman"] = {"est": round(spearman[0], 4), "lo": round(spearman[1], 4), "hi": round(spearman[2], 4)}
    precision = season_level_ci(
        frame,
        lambda f: within_position_precision(f, projection_col, higher_is_better=higher_is_better),
        seed=seed,
    )
    out["within_position_precision"] = {
        "est": round(precision[0], 4), "lo": round(precision[1], 4), "hi": round(precision[2], 4),
        "resampling": "seasons (n~6) -- low power by construction",
    }
    blind = season_level_ci(
        frame,
        lambda f: top_n_precision(f, projection_col, higher_is_better=higher_is_better),
        seed=seed,
    )
    out[f"top{TOP_N}_precision_position_blind"] = {
        "est": round(blind[0], 4), "lo": round(blind[1], 4), "hi": round(blind[2], 4),
        "warning": "oracle is the raw-points top N (9-11 QBs). Diagnostic only, never an acceptance metric.",
    }
    if higher_is_better:
        ratio = bootstrap_ci(frame, lambda f: calibration_ratio(f, projection_col), n_boot=n_boot, seed=seed)
        out["calibration_ratio"] = {"est": round(ratio[0], 4), "lo": round(ratio[1], 4), "hi": round(ratio[2], 4)}
        for label, mask in (
            ("rookie", frame.get("is_rookie", pd.Series(False, index=frame.index)).astype(bool)),
            ("veteran", ~frame.get("is_rookie", pd.Series(False, index=frame.index)).astype(bool)),
        ):
            subset = frame[mask]
            if len(subset) >= 30:
                out[f"calibration_ratio_{label}"] = round(calibration_ratio(subset, projection_col), 4)
                out[f"spearman_{label}"] = round(
                    pooled_spearman(subset, projection_col, higher_is_better=higher_is_better), 4
                )
    return out


def paired_difference(
    frame: pd.DataFrame,
    column_a: str,
    column_b: str,
    statistic,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    by_season: bool = False,
) -> dict:
    """Bootstrap the A-minus-B difference on the same resampled clusters.

    Pairing matters: two columns evaluated on independent bootstraps would carry
    the shared season-quality variance twice and hide a real difference.
    """
    def _diff(f: pd.DataFrame) -> float:
        return float(statistic(f, column_a)) - float(statistic(f, column_b))

    if by_season:
        point, lo, hi = season_level_ci(frame, _diff, seed=seed)
    else:
        point, lo, hi = bootstrap_ci(frame, _diff, n_boot=n_boot, seed=seed)
    return {
        "diff": round(point, 4),
        "lo": round(lo, 4),
        "hi": round(hi, 4),
        "significant": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
    }
