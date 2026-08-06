"""
ml/parquet_cache.py

Parquet-backed feature matrix cache for fast weekly retrains.

PROBLEM
-------
load_feature_matrix() executes a large PostgreSQL query (all feature_matrix
rows for 7 seasons) on every training run. For the full 2019-2026 dataset
this is ~130k rows × 120 columns — a cold DB query takes 30-90 seconds and
produces a DataFrame that is IDENTICAL between runs unless the underlying
game_logs or feature_matrix table has changed.

SOLUTION
--------
Cache the feature matrix to a Parquet file keyed by (seasons, row_count_hash).
On weekly retrains where no new game data has been ingested, the entire DB
query is replaced by a local Parquet read (~1-3 seconds).

Cache invalidation:
  - Seasons list changes (new season added to training run)
  - Row count in feature_matrix changes (ETL ingested new games)
  - Cache file older than MAX_CACHE_AGE_HOURS (safety valve)

Cache location: ml/cache/feature_matrix_{cache_key}.parquet
Key:            SHA256(sorted_seasons_str + "_" + row_count)

USAGE
-----
    from ml.parquet_cache import FeatureCache

    cache = FeatureCache()

    # Load from cache or DB
    df = cache.load_or_fetch(
        seasons=[2022, 2023, 2025],
        db_url=os.environ["DATABASE_URL"],
        position_filter="WR",
        no_cache=False,   # set True to always hit DB
    )
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Cache files older than this are always invalidated (safety valve).
MAX_CACHE_AGE_HOURS: float = 24.0

_DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"


class FeatureCache:
    """
    Parquet cache layer for feature_matrix DB queries.

    Args:
        cache_dir: Directory for .parquet cache files. Created if absent.
    """

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_or_fetch(
        self,
        seasons: list[int],
        db_url: str,
        position_filter: Optional[str] = None,
        no_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Return feature matrix for the given seasons.

        Checks the Parquet cache first; falls back to load_feature_matrix()
        on cache miss or invalidation.

        Args:
            seasons:         List of NFL season years to include.
            db_url:          PostgreSQL connection URL.
            position_filter: Optional position filter (e.g. "WR").
            no_cache:        If True, skip cache entirely and always query DB.

        Returns:
            DataFrame with all feature_matrix columns (same as load_feature_matrix).
        """
        from ml.utils import load_feature_matrix

        if no_cache:
            logger.info("FeatureCache: no_cache=True — fetching from DB.")
            return load_feature_matrix(db_url, seasons, position_filter)

        cache_key = self._make_key(seasons, position_filter, db_url)
        cache_path = self.cache_dir / f"feature_matrix_{cache_key}.parquet"

        if self._is_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(
                    "FeatureCache: HIT %s (%d rows, %d cols).",
                    cache_path.name, len(df), len(df.columns),
                )
                return df
            except Exception as exc:
                logger.warning("FeatureCache: read failed (%s) — fetching from DB.", exc)

        # Cache miss — query DB and save.
        logger.info("FeatureCache: MISS — fetching from DB for seasons=%s.", seasons)
        df = load_feature_matrix(db_url, seasons, position_filter)
        self._save(df, cache_path)
        return df

    def invalidate(
        self,
        seasons: Optional[list[int]] = None,
        position_filter: Optional[str] = None,
    ) -> int:
        """
        Delete matching cache files. Pass seasons=None to purge all.

        Returns:
            Number of files deleted.
        """
        deleted = 0
        for f in self.cache_dir.glob("feature_matrix_*.parquet"):
            if seasons is None or any(str(s) in f.stem for s in seasons):
                try:
                    f.unlink()
                    deleted += 1
                    logger.info("FeatureCache: invalidated %s", f.name)
                except OSError:
                    pass
        return deleted

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_key(
        self,
        seasons: list[int],
        position_filter: Optional[str],
        db_url: str,
    ) -> str:
        """
        Build a cache key from seasons + DB row count + position filter.

        Row count is fetched cheaply (COUNT(*) query) and captures any
        new ETL ingestion without requiring a full table scan.
        """
        row_count = self._fetch_row_count(db_url, seasons)
        pos_tag = position_filter or "all"
        raw = f"{sorted(seasons)}_{pos_tag}_{row_count}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _fetch_row_count(self, db_url: str, seasons: list[int]) -> int:
        """Fast COUNT(*) to detect new ETL ingestion."""
        try:
            import psycopg2
            from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn
            dsn = _psycopg2_dsn(db_url)
            conn = psycopg2.connect(dsn)
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM feature_matrix WHERE season = ANY(%s)",
                        (seasons,),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else 0
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("FeatureCache: COUNT(*) failed (%s) — using 0.", exc)
            return 0

    def _is_valid(self, path: Path) -> bool:
        """Return True if the cache file exists and is not too old."""
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > MAX_CACHE_AGE_HOURS:
            logger.info(
                "FeatureCache: expired (%.1f h old, limit %s h).",
                age_hours, MAX_CACHE_AGE_HOURS,
            )
            return False
        return True

    def _save(self, df: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to Parquet with snappy compression."""
        try:
            df.to_parquet(path, index=False, compression="snappy")
            size_mb = path.stat().st_size / 1e6
            logger.info(
                "FeatureCache: saved %s (%.1f MB, %d rows).",
                path.name, size_mb, len(df),
            )
        except Exception as exc:
            logger.warning("FeatureCache: save failed (%s).", exc)
