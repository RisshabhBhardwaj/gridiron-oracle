"""
backend/tests/test_ml_cov_player_embeddings.py

Coverage tests for ml/player_embeddings.py (physical-profile embeddings).

Known-answer checks use the closed-form speed score and the one-hot position
encoding. Embedding invariants checked: 32-dim output, tanh bounds, zero vectors
for unknown or unfitted players, batch-vs-single agreement, and orthonormal
columns for the random-projection fallback.

The database is replaced by an in-test psycopg2 fake; no socket is opened.
The MLX branch is not exercised because mlx is not installed here.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
import pytest

from ml import player_embeddings as pe
from ml.player_embeddings import (
    EMBED_DIM,
    POSITIONS,
    EmbeddingStore,
    PlayerProfile,
    PlayerProfileEmbedder,
    _position_onehot,
    _speed_score,
    get_embedding_store,
    reset_embedding_store,
)


def _roster(n: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    positions = ["QB", "RB", "WR", "TE", "WR", "RB"]
    rows = []
    for i in range(n):
        rows.append({
            "player_id": f"P{i:02d}",
            "position": positions[i % len(positions)],
            "height": float(70 + rng.integers(0, 8)),
            "weight": float(190 + rng.integers(0, 40)),
            "draft_round": float(rng.integers(1, 8)),
            "age": float(22 + rng.integers(0, 10)),
            "years_experience": float(rng.integers(0, 10)),
            "forty_time": float(round(4.3 + rng.random() * 0.4, 2)),
        })
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def _clean_global_store():
    reset_embedding_store()
    yield
    reset_embedding_store()


# ── helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_speed_score_closed_form(self) -> None:
        # (200^2 / 4.0^4) * 100 = 40000 / 256 * 100 = 15625.
        assert _speed_score(200.0, 4.0) == pytest.approx(15625.0)

    def test_speed_score_nonpositive_time_is_zero(self) -> None:
        assert _speed_score(210.0, 0.0) == 0.0
        assert _speed_score(210.0, -1.0) == 0.0

    def test_position_onehot_is_case_and_space_insensitive(self) -> None:
        np.testing.assert_array_equal(_position_onehot(" wr "), [0, 0, 1, 0])
        np.testing.assert_array_equal(_position_onehot("QB"), [1, 0, 0, 0])
        assert POSITIONS == ["QB", "RB", "WR", "TE"]

    def test_unknown_or_missing_position_is_all_zero(self) -> None:
        np.testing.assert_array_equal(_position_onehot("K"), np.zeros(4))
        np.testing.assert_array_equal(_position_onehot(""), np.zeros(4))
        np.testing.assert_array_equal(_position_onehot(None), np.zeros(4))

    def test_safe_float_rejects_non_finite_and_bad_values(self) -> None:
        assert PlayerProfileEmbedder._safe_float("2.5") == pytest.approx(2.5)
        assert PlayerProfileEmbedder._safe_float(float("nan")) is None
        assert PlayerProfileEmbedder._safe_float(float("inf")) is None
        assert PlayerProfileEmbedder._safe_float("abc") is None
        assert PlayerProfileEmbedder._safe_float(None) is None


# ── PlayerProfile ─────────────────────────────────────────────────────────────

class TestPlayerProfile:
    def test_speed_score_uses_position_default_forty_time(self) -> None:
        p = PlayerProfile(player_id="x", position="QB", weight=220.0)
        assert p.speed_score() == pytest.approx(_speed_score(220.0, 4.85))

    def test_speed_score_uses_recorded_forty_and_default_weight(self) -> None:
        p = PlayerProfile(player_id="x", position="WR", forty_time=4.4)
        assert p.speed_score() == pytest.approx(_speed_score(210.0, 4.4))

    def test_speed_score_unknown_position_uses_tight_end_default(self) -> None:
        p = PlayerProfile(player_id="x", position="K", weight=250.0)
        assert p.speed_score() == pytest.approx(_speed_score(250.0, 4.65))

    def test_raw_features_fall_back_to_documented_defaults(self) -> None:
        p = PlayerProfile(player_id="x", position="TE")
        raw = p.to_raw_features()
        assert raw.shape == (10,)
        np.testing.assert_allclose(raw[:5], [72.0, 210.0, 9.0, 25.0, 2.0], rtol=1e-6)
        assert raw[5] == pytest.approx(_speed_score(210.0, 4.65), rel=1e-5)
        np.testing.assert_array_equal(raw[6:], [0, 0, 0, 1])

    def test_raw_features_use_present_values(self) -> None:
        p = PlayerProfile(player_id="x", position="RB", height=70.0, weight=205.0,
                          draft_round=3.0, age=24.0, years_experience=2.0)
        raw = p.to_raw_features()
        np.testing.assert_allclose(raw[:5], [70.0, 205.0, 3.0, 24.0, 2.0], rtol=1e-6)
        np.testing.assert_array_equal(raw[6:], [0, 1, 0, 0])


# ── PlayerProfileEmbedder ─────────────────────────────────────────────────────

class TestEmbedderUnfitted:
    def test_empty_roster_leaves_embedder_unfitted(self, caplog) -> None:
        embedder = PlayerProfileEmbedder()
        with caplog.at_level(logging.WARNING, logger=pe.logger.name):
            embedder.fit(pd.DataFrame(columns=["player_id", "position"]))
        assert embedder._fitted is False
        assert "empty roster_df" in caplog.text
        assert embedder.embed("anyone").shape == (EMBED_DIM,)
        assert np.all(embedder.embed("anyone") == 0)

    def test_unfitted_batch_is_zeros(self) -> None:
        out = PlayerProfileEmbedder().embed_batch(["a", "b", "c"])
        assert out.shape == (3, EMBED_DIM)
        assert not out.any()

    def test_repr_reports_state(self) -> None:
        embedder = PlayerProfileEmbedder()
        assert repr(embedder) == f"PlayerProfileEmbedder(fitted=False, players=0, embed_dim={EMBED_DIM})"


class TestEmbedderFitted:
    def test_fit_pads_to_embed_dim_and_bounds_output(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(6))
        assert embedder._fitted is True
        assert embedder._W.shape == (10, EMBED_DIM)
        # 6 players -> at most 5 PCA components; columns past that are zero padding.
        assert not embedder._W[:, 5:].any()
        vec = embedder.embed("P00")
        assert vec.shape == (EMBED_DIM,)
        assert vec.dtype == np.float32
        assert np.all(np.abs(vec) <= 1.0)

    def test_skips_rows_without_player_id(self) -> None:
        roster = _roster(4)
        roster.loc[roster.index[0], "player_id"] = ""
        embedder = PlayerProfileEmbedder().fit(roster)
        assert sorted(embedder.known_players()) == ["P01", "P02", "P03"]

    def test_unknown_player_gets_zero_embedding(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(5))
        assert not embedder.embed("NOPE").any()

    def test_batch_matches_single_player_embedding(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(6))
        ids = ["P03", "P00", "P05"]
        batch = embedder.embed_batch(ids)
        assert batch.shape == (3, EMBED_DIM)
        for row, pid in enumerate(ids):
            np.testing.assert_allclose(batch[row], embedder.embed(pid), atol=1e-6)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Production bug: embed_batch documents zeros for players outside the profile store, "
            "but it normalises a zero raw vector, which yields a non-zero embedding. embed() does "
            "return zeros. Fix in ml/player_embeddings.py, then this xfail flips to a pass."
        ),
    )
    def test_batch_unknown_player_is_zero_as_documented(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(6))
        assert not embedder.embed_batch(["NOPE"]).any()

    def test_project_equals_tanh_of_normalized_times_w(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(6))
        profile = embedder._profiles["P02"]
        normalized = (profile.to_raw_features() - embedder._mean) / embedder._std
        expected = np.tanh(normalized @ embedder._W).astype(np.float32)
        np.testing.assert_allclose(embedder._project(profile), expected, atol=1e-6)

    def test_update_profile_adds_player_without_refit(self) -> None:
        embedder = PlayerProfileEmbedder().fit(_roster(5))
        weight_before = embedder._W.copy()
        embedder.update_profile(PlayerProfile(player_id="NEW", position="WR", weight=215.0))
        assert "NEW" in embedder.known_players()
        assert embedder.embed("NEW").shape == (EMBED_DIM,)
        np.testing.assert_array_equal(embedder._W, weight_before)
        assert "players=6" in repr(embedder)

    def test_random_projection_fallback_without_sklearn(self, monkeypatch) -> None:
        # A None entry makes `from sklearn.decomposition import PCA` raise ImportError.
        monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
        embedder = PlayerProfileEmbedder().fit(_roster(8))
        assert embedder._fitted is True
        basis = embedder._W[:, :7]
        # QR-derived columns are orthonormal, so the Gram matrix is the identity.
        np.testing.assert_allclose(basis.T @ basis, np.eye(7), atol=1e-5)


# ── EmbeddingStore ────────────────────────────────────────────────────────────

class TestEmbeddingStore:
    def test_build_caches_every_known_player(self) -> None:
        store = EmbeddingStore().build(_roster(6))
        assert len(store) == 6
        assert store.get("P04").shape == (EMBED_DIM,)
        assert "players=6" in repr(store)

    def test_get_unknown_player_returns_zeros_and_caches_them(self) -> None:
        store = EmbeddingStore().build(_roster(4))
        vec = store.get("GHOST")
        assert not vec.any()
        assert len(store) == 5  # the zero vector is now cached

    def test_enrich_features_adds_named_columns_matching_batch(self) -> None:
        store = EmbeddingStore().build(_roster(6))
        df = pd.DataFrame({"player_id": ["P01", "GHOST", "P03"], "season": [2025, 2025, 2025]})
        out = store.enrich_features(df)

        assert list(out.columns[:2]) == ["player_id", "season"]
        emb_cols = [f"player_emb_{i}" for i in range(EMBED_DIM)]
        assert list(out.columns[2:]) == emb_cols
        assert len(out) == 3  # no rows dropped
        expected = store._embedder.embed_batch(["P01", "GHOST", "P03"])
        np.testing.assert_allclose(out[emb_cols].to_numpy(), expected, atol=1e-6)
        # The unknown GHOST row is covered by the single-player path only: see the xfail above.
        assert store._embedder.embed("GHOST").tolist() == [0.0] * EMBED_DIM

    def test_enrich_without_player_id_returns_frame_unchanged(self, caplog) -> None:
        store = EmbeddingStore().build(_roster(4))
        df = pd.DataFrame({"season": [2025]})
        with caplog.at_level(logging.WARNING, logger=pe.logger.name):
            out = store.enrich_features(df)
        assert out is df
        assert "no player_id column" in caplog.text

    def test_save_and_load_round_trip(self, tmp_path) -> None:
        store = EmbeddingStore().build(_roster(5))
        path = tmp_path / "emb" / "store.npz"
        store.save(str(path))
        assert path.exists()

        restored = EmbeddingStore().load(str(path))
        assert len(restored) == 5
        for pid in store._cache:
            np.testing.assert_allclose(restored.get(pid), store.get(pid), atol=1e-6)

    def test_build_from_db_uses_rows_from_query(self, monkeypatch) -> None:
        import psycopg2

        roster = _roster(4).rename(columns={"player_id": "player_id"})
        rows = roster.to_dict(orient="records")
        seen: dict = {"closed": False}

        class _Cursor:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def execute(self, sql):
                seen["sql"] = sql

            def fetchall(self):
                return rows

        class _Conn:
            def cursor(self, cursor_factory=None):
                seen["cursor_factory"] = cursor_factory
                return _Cursor()

            def close(self):
                seen["closed"] = True

        monkeypatch.setattr(psycopg2, "connect", lambda dsn: (seen.__setitem__("dsn", dsn), _Conn())[1])
        store = EmbeddingStore().build_from_db("postgresql+asyncpg://u@h/db")
        assert seen["dsn"] == "postgresql://u@h/db"
        assert seen["closed"] is True
        assert "FROM players" in seen["sql"]
        assert len(store) == 4

    def test_build_from_db_failure_returns_empty_store(self, monkeypatch, caplog) -> None:
        import psycopg2

        def boom(dsn):
            raise OSError("db unreachable")

        monkeypatch.setattr(psycopg2, "connect", boom)
        with caplog.at_level(logging.WARNING, logger=pe.logger.name):
            store = EmbeddingStore().build_from_db("postgresql://u@h/db")
        assert len(store) == 0
        assert "using empty store" in caplog.text


class TestSingleton:
    def test_roster_builds_singleton_once(self) -> None:
        first = get_embedding_store(roster_df=_roster(4))
        second = get_embedding_store(roster_df=_roster(8))
        assert first is second
        assert len(first) == 4

    def test_load_path_takes_precedence_when_it_exists(self, tmp_path) -> None:
        source = EmbeddingStore().build(_roster(3))
        path = tmp_path / "saved.npz"
        source.save(str(path))
        store = get_embedding_store(load_path=str(path), roster_df=_roster(9))
        assert len(store) == 3

    def test_reset_clears_singleton(self) -> None:
        first = get_embedding_store()
        assert len(first) == 0
        reset_embedding_store()
        assert get_embedding_store() is not first
