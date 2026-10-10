"""
backend/tests/test_ml_cov_parquet_cache.py

Coverage tests for ml/parquet_cache.py (feature-matrix Parquet cache).

Offline only: psycopg2.connect is replaced with an in-test fake (the real
driver stays importable and no socket is opened), and the loader
`ml.utils.load_feature_matrix` is replaced by a counting stub. Cache files
are written under tmp_path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time

import pandas as pd
import pytest

import ml.utils as ml_utils
from ml import parquet_cache as pc
from ml.parquet_cache import FeatureCache


def _frame(n: int = 3, tag: str = "a") -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [f"{tag}{i}" for i in range(n)],
        "season": [2023] * n,
        "seas_avg_fantasy_ppr": [10.0 + i for i in range(n)],
    })


@pytest.fixture
def db_counts(monkeypatch):
    """Replace the DB loader and the COUNT(*) probe with deterministic stubs."""
    state = {"loader_calls": [], "row_count": 42, "frame": _frame()}

    def fake_loader(db_url, seasons, position_filter=None, feature_cols=None):
        state["loader_calls"].append((db_url, list(seasons), position_filter))
        return state["frame"].copy()

    monkeypatch.setattr(ml_utils, "load_feature_matrix", fake_loader)
    monkeypatch.setattr(FeatureCache, "_fetch_row_count", lambda self, db_url, seasons: state["row_count"])
    return state


def _expected_key(seasons, position, row_count) -> str:
    raw = f"{sorted(seasons)}_{position or 'all'}_{row_count}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TestLoadOrFetch:
    def test_no_cache_always_hits_db_and_writes_nothing(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        out = cache.load_or_fetch([2023], "postgresql://x", position_filter="WR", no_cache=True)
        assert list(out["player_id"]) == ["a0", "a1", "a2"]
        assert db_counts["loader_calls"] == [("postgresql://x", [2023], "WR")]
        assert list(cache.cache_dir.glob("*.parquet")) == []

    def test_miss_writes_parquet_then_second_call_hits(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        first = cache.load_or_fetch([2024, 2023], "postgresql://x", position_filter="WR")
        assert len(db_counts["loader_calls"]) == 1

        path = cache.cache_dir / f"feature_matrix_{_expected_key([2023, 2024], 'WR', 42)}.parquet"
        assert path.exists()

        second = cache.load_or_fetch([2024, 2023], "postgresql://x", position_filter="WR")
        assert len(db_counts["loader_calls"]) == 1  # served from cache
        pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))

    def test_different_position_filter_is_a_separate_cache_entry(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        cache.load_or_fetch([2023], "postgresql://x", position_filter="WR")
        cache.load_or_fetch([2023], "postgresql://x", position_filter="RB")
        cache.load_or_fetch([2023], "postgresql://x", position_filter=None)
        assert len(db_counts["loader_calls"]) == 3
        names = sorted(p.name for p in cache.cache_dir.glob("*.parquet"))
        assert names == sorted(
            f"feature_matrix_{_expected_key([2023], pos, 42)}.parquet" for pos in ("WR", "RB", None)
        )

    def test_row_count_change_invalidates_key(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        cache.load_or_fetch([2023], "postgresql://x")
        db_counts["row_count"] = 43  # ETL ingested new rows
        cache.load_or_fetch([2023], "postgresql://x")
        assert len(db_counts["loader_calls"]) == 2

    def test_expired_cache_file_is_refetched(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        cache.load_or_fetch([2023], "postgresql://x", position_filter="WR")
        path = next(cache.cache_dir.glob("*.parquet"))
        old = time.time() - (pc.MAX_CACHE_AGE_HOURS + 1) * 3600
        os.utime(path, (old, old))

        cache.load_or_fetch([2023], "postgresql://x", position_filter="WR")
        assert len(db_counts["loader_calls"]) == 2

    def test_unreadable_cache_file_falls_back_to_db(self, tmp_path, db_counts) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        bad = cache.cache_dir / f"feature_matrix_{_expected_key([2023], None, 42)}.parquet"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not a parquet file")

        out = cache.load_or_fetch([2023], "postgresql://x")
        assert len(db_counts["loader_calls"]) == 1
        assert list(out["player_id"]) == ["a0", "a1", "a2"]
        # The bad file is overwritten with a valid Parquet copy.
        assert pd.read_parquet(bad).shape == (3, 3)

    def test_save_failure_is_logged_not_raised(self, tmp_path, db_counts, caplog) -> None:
        # Arbitrary Python objects cannot be converted to an Arrow type, so the write fails.
        db_counts["frame"] = pd.DataFrame({"player_id": ["x"], "bad": [object()]})
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        with caplog.at_level(logging.WARNING, logger=pc.logger.name):
            out = cache.load_or_fetch([2023], "postgresql://x")
        assert list(out["player_id"]) == ["x"]
        assert list(cache.cache_dir.glob("*.parquet")) == []
        assert "save failed" in caplog.text


def _install_fake_connect(monkeypatch, connect) -> None:
    """Replace psycopg2.connect so no socket is opened; the real driver stays importable."""
    import psycopg2

    monkeypatch.setattr(psycopg2, "connect", connect)


class TestRowCountProbe:
    def test_fetch_row_count_returns_count_and_closes_connection(self, tmp_path, monkeypatch) -> None:
        seen: dict = {}

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql, params):
                seen["sql"] = sql
                seen["params"] = params

            def fetchone(self):
                return (1234,)

        class _Conn:
            def cursor(self):
                return _Cursor()

            def close(self):
                seen["closed"] = True

        def connect(dsn):
            seen["dsn"] = dsn
            return _Conn()

        _install_fake_connect(monkeypatch, connect)

        cache = FeatureCache(cache_dir=tmp_path / "cache")
        count = cache._fetch_row_count("postgresql+asyncpg://u:p@h/db", [2023, 2024])
        assert count == 1234
        assert seen["dsn"] == "postgresql://u:p@h/db"
        assert seen["params"] == ([2023, 2024],)
        assert "COUNT(*) FROM feature_matrix" in seen["sql"]
        assert seen["closed"] is True

    def test_fetch_row_count_returns_zero_on_connection_error(self, tmp_path, monkeypatch) -> None:
        def boom(dsn):
            raise OSError("connection refused")

        _install_fake_connect(monkeypatch, boom)
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        assert cache._fetch_row_count("postgresql://x", [2023]) == 0


class TestInvalidateAndInit:
    def test_init_creates_directory(self, tmp_path) -> None:
        target = tmp_path / "nested" / "cache"
        FeatureCache(cache_dir=target)
        assert target.is_dir()

    def test_invalidate_by_season_then_all(self, tmp_path) -> None:
        cache = FeatureCache(cache_dir=tmp_path / "cache")
        for stem in ("feature_matrix_k2023.parquet", "feature_matrix_k2024.parquet", "feature_matrix_kz.parquet"):
            (cache.cache_dir / stem).write_bytes(b"x")
        (cache.cache_dir / "unrelated.parquet").write_bytes(b"x")

        assert cache.invalidate(seasons=[2023]) == 1
        assert not (cache.cache_dir / "feature_matrix_k2023.parquet").exists()
        assert (cache.cache_dir / "feature_matrix_k2024.parquet").exists()

        assert cache.invalidate() == 2  # purge all feature_matrix_* files
        assert (cache.cache_dir / "unrelated.parquet").exists()
