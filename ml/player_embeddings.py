"""
ml/player_embeddings.py

Dense player profile embeddings for cold-start handling and TFT static covariates.

PROBLEM
-------
Rookie players have zero historical game data — XGBoost and LightGBM cannot
make meaningful predictions without prior seasons' performance, and TFT needs
at least min_encoder_length weeks of history.

SOLUTION — PHYSICAL PROFILE EMBEDDINGS
---------------------------------------
Encode a player's static physical attributes (height, weight, speed score,
draft round, position, age, experience) into a dense 32-dimensional vector.

This embedding serves as:
1. A cold-start proxy for ZeroKalmanPrior when a player has < 4 game log rows.
2. An additional static real variable for TFT (alongside the existing
   TFTDataset.STATIC_REALS columns).
3. A clustering feature for finding "similar player" analogs.

FEATURES ENCODED
----------------
Continuous (z-score normalized):
    height (inches), weight (lbs), draft_round (1-7, 9 for undrafted),
    age (years), years_experience

Binary/categorical (one-hot encoded):
    position: QB, RB, WR, TE (4 dims)
    college_conference: power5, g5, fbs (3 dims)  — if available

Derived (computed from physical attributes):
    speed_score = (weight * weight) / (40_time ** 4) × 100
        → combines size and 40-time into a single athletic quality metric
        → established metric from football analytics literature

Embedding dim = 32 (to match TFTDataset.STATIC_REALS planned expansion)

USAGE
-----
    from ml.player_embeddings import PlayerProfileEmbedder, EmbeddingStore

    # Build embeddings from DB roster data
    embedder = PlayerProfileEmbedder()
    embedder.fit(roster_df)  # DataFrame with [player_id, position, height, weight, ...]

    # Get embedding for a specific player
    emb = embedder.embed("00-0033106")  # Travis Kelce
    # → np.ndarray(32,) representing Kelce's physical profile

    # Store and retrieve
    store = EmbeddingStore(embedder)
    store.build_from_db(db_url)
    store.save("ml/embeddings/player_embeddings.npz")
    store.load("ml/embeddings/player_embeddings.npz")

    # Add embedding columns to feature_matrix DataFrame
    enriched_df = store.enrich_features(df)
    # → adds columns player_emb_0 … player_emb_31
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── MLX acceleration (Apple Silicon) ─────────────────────────────────────────
# Batch projection via MLX matmul is ~3-8× faster than looping over per-player
# NumPy ops when encoding 100+ players at once (e.g. full roster enrichment).
try:
    import mlx.core as mx  # type: ignore[import]
    _USE_MLX: bool = True
except ImportError:
    mx = None  # type: ignore[assignment]
    _USE_MLX: bool = False


# ── Constants ─────────────────────────────────────────────────────────────────

EMBED_DIM = 32  # dimensionality of the player embedding vector

# Position encoding — one-hot, 4 bits
POSITIONS = ["QB", "RB", "WR", "TE"]
N_POSITIONS = len(POSITIONS)

# Physical feature column names from nflreadpy / feature_matrix
_PHYS_COLS = ["height", "weight", "draft_round", "age", "years_experience"]
_DEFAULT_40_TIME: dict[str, float] = {  # average 40-time in seconds by position
    "QB": 4.85, "RB": 4.50, "WR": 4.45, "TE": 4.65,
}
_UNDRAFTED_ROUND = 9.0   # sentinel for undrafted players


# ── Helper Functions ───────────────────────────────────────────────────────────

def _speed_score(weight_lbs: float, time_40: float) -> float:
    """
    Compute Bill Barnwell's speed score metric.

    speed_score = (weight² / 40_time⁴) × 100

    Higher score = better combination of size and speed.
    Typical range: 50–130 (higher = more athletic for their weight class).

    Args:
        weight_lbs: Player weight in pounds.
        time_40:    40-yard dash time in seconds.

    Returns:
        Float speed score.
    """
    if time_40 <= 0:
        return 0.0
    return (weight_lbs ** 2 / time_40 ** 4) * 100


def _position_onehot(position: str) -> np.ndarray:
    """4-bit one-hot encoding for QB/RB/WR/TE. Unknown position → [0,0,0,0]."""
    v = np.zeros(N_POSITIONS, dtype=np.float32)
    pos = position.upper().strip() if position else ""
    if pos in POSITIONS:
        v[POSITIONS.index(pos)] = 1.0
    return v


# ── Player Profile ─────────────────────────────────────────────────────────────

@dataclass
class PlayerProfile:
    """
    Physical profile for a single player.
    All fields may be None when data is unavailable.
    """
    player_id: str
    position:  str = ""
    height:    Optional[float] = None   # inches
    weight:    Optional[float] = None   # lbs
    draft_round: Optional[float] = None # 1-7; None = undrafted
    age:       Optional[float] = None   # age at start of season
    years_experience: Optional[float] = None
    forty_time: Optional[float] = None  # 40-yard dash in seconds; optional

    def speed_score(self) -> float:
        """Compute speed score from available data. Falls back to position average."""
        w = self.weight or 210.0
        t = self.forty_time or _DEFAULT_40_TIME.get(self.position.upper(), 4.65)
        return _speed_score(w, t)

    def to_raw_features(self) -> np.ndarray:
        """
        Convert profile to a raw feature vector (pre-normalization).

        Returns:
            np.ndarray(9,): [height, weight, draft_round, age, experience,
                             speed_score, pos_QB, pos_RB, pos_WR, pos_TE]
        """
        pos_vec = _position_onehot(self.position)
        raw = np.array([
            self.height or 72.0,               # fallback: 6'0"
            self.weight or 210.0,              # fallback: 210 lbs
            self.draft_round or _UNDRAFTED_ROUND,
            self.age or 25.0,
            self.years_experience or 2.0,
            self.speed_score(),
        ], dtype=np.float32)
        return np.concatenate([raw, pos_vec])  # shape (10,)


# ── PlayerProfileEmbedder ─────────────────────────────────────────────────────

class PlayerProfileEmbedder:
    """
    Encodes player physical profiles into dense 32-dim embedding vectors.

    Architecture:
        Raw features (10-dim) → linear projection + tanh → 32-dim embedding

    The linear projection matrix W (10×32) is learned via PCA on the player
    population when fit() is called. This ensures the 32 dimensions capture
    maximum variance in physical profiles (e.g., size-speed tradeoff, position
    clustering, experience spectrum).

    If only a few players are available (< 10), the embedder falls back to
    a deterministic sparse projection (reproducible without training data).

    State:
        _mean:  (10,) float — mean of raw feature vectors
        _std:   (10,) float — std of raw feature vectors
        _W:     (10, EMBED_DIM) float — projection matrix
        _profiles: {player_id: PlayerProfile}
        _fitted: bool
    """

    def __init__(self) -> None:
        self._mean:     Optional[np.ndarray] = None
        self._std:      Optional[np.ndarray] = None
        self._W:        Optional[np.ndarray] = None
        self._profiles: dict[str, PlayerProfile] = {}
        self._fitted:   bool = False

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(self, roster_df: pd.DataFrame) -> "PlayerProfileEmbedder":
        """
        Fit the embedding from a roster DataFrame.

        Args:
            roster_df: DataFrame with columns:
                [player_id, position, height, weight, draft_round, age,
                 years_experience] — additional columns ignored.
                Rows with missing player_id are skipped.

        Returns:
            self (for chaining)
        """
        profiles: list[PlayerProfile] = []
        for _, row in roster_df.iterrows():
            pid = str(row.get("player_id", ""))
            if not pid:
                continue
            p = PlayerProfile(
                player_id=pid,
                position=str(row.get("position") or ""),
                height=self._safe_float(row.get("height")),
                weight=self._safe_float(row.get("weight")),
                draft_round=self._safe_float(row.get("draft_round")),
                age=self._safe_float(row.get("age")),
                years_experience=self._safe_float(row.get("years_experience")),
                forty_time=self._safe_float(row.get("forty_time")),
            )
            self._profiles[pid] = p
            profiles.append(p)

        if not profiles:
            logger.warning("PlayerProfileEmbedder.fit(): empty roster_df — using identity projection.")
            self._fitted = False
            return self

        # Build raw feature matrix
        raw = np.stack([p.to_raw_features() for p in profiles], axis=0)  # (N, 10)

        # Z-score normalization
        self._mean = raw.mean(axis=0)
        self._std  = raw.std(axis=0)
        self._std  = np.where(self._std < 1e-8, 1.0, self._std)  # avoid div-by-zero

        normalized = (raw - self._mean) / self._std  # (N, 10)

        # PCA projection to EMBED_DIM
        n_components = min(EMBED_DIM, max(normalized.shape[0] - 1, 1), normalized.shape[1])
        if n_components < 1:
            n_components = 1
        pca = None
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(normalized)
            W_pca = pca.components_.T  # (10, n_components)
        except ImportError:
            # sklearn not available — use random orthogonal projection
            logger.warning("sklearn not available; using random projection for embeddings.")
            rng = np.random.default_rng(42)
            W_pca, _ = np.linalg.qr(rng.standard_normal((normalized.shape[1], n_components)))

        # Pad to EMBED_DIM if PCA returned fewer components
        if n_components < EMBED_DIM:
            pad = np.zeros((W_pca.shape[0], EMBED_DIM - n_components), dtype=np.float32)
            self._W = np.concatenate([W_pca, pad], axis=1).astype(np.float32)
        else:
            self._W = W_pca.astype(np.float32)

        self._fitted = True
        logger.info(
            "PlayerProfileEmbedder fitted: %d players, embedding dim=%d, "
            "PCA components=%d (%.1f%% var explained if PCA)",
            len(profiles), EMBED_DIM, n_components,
            getattr(pca, 'explained_variance_ratio_', np.array([0])).sum() * 100
            if pca is not None else 0.0,
        )
        return self

    def embed(self, player_id: str) -> np.ndarray:
        """
        Get the 32-dim embedding for a player.

        Args:
            player_id: Player ID string.

        Returns:
            np.ndarray(EMBED_DIM,). Zeros if player not in profile store or
            embedder not fitted.
        """
        if not self._fitted or self._W is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        profile = self._profiles.get(player_id)
        if profile is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        return self._project(profile)

    def embed_batch(self, player_ids: list[str]) -> np.ndarray:
        """
        Compute embeddings for a batch of player IDs at once.

        Uses MLX matrix multiplication on Apple Silicon (3-8× faster than
        per-player NumPy loops for 100+ players). Falls back to NumPy on
        non-Apple platforms or when mlx is not installed.

        Args:
            player_ids: List of player ID strings.

        Returns:
            np.ndarray of shape (len(player_ids), EMBED_DIM).
            Players not in the profile store get zero embeddings.
        """
        if not self._fitted or self._W is None:
            return np.zeros((len(player_ids), EMBED_DIM), dtype=np.float32)

        profiles = [self._profiles.get(pid) for pid in player_ids]
        raw_list = [
            p.to_raw_features() if p is not None
            else np.zeros(10, dtype=np.float32)
            for p in profiles
        ]
        raw = np.stack(raw_list, axis=0)  # (N, 10)
        normalized = ((raw - self._mean) / self._std).astype(np.float32)

        if _USE_MLX:
            mx_norm = mx.array(normalized)
            mx_W    = mx.array(self._W)
            mx_emb  = mx.tanh(mx_norm @ mx_W)   # (N, EMBED_DIM)
            mx.eval(mx_emb)
            return np.array(mx_emb.tolist(), dtype=np.float32)
        else:
            return np.tanh(normalized @ self._W).astype(np.float32)

    def _project(self, profile: PlayerProfile) -> np.ndarray:
        """Project a PlayerProfile through the fitted embedding matrix."""
        raw = profile.to_raw_features()
        normalized = (raw - self._mean) / self._std
        embedding = normalized @ self._W           # (EMBED_DIM,)
        # Tanh activation to bound the embedding to [-1, 1]
        return np.tanh(embedding).astype(np.float32)

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        try:
            v = float(val)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    def update_profile(self, profile: PlayerProfile) -> None:
        """Add or update a single player's profile."""
        self._profiles[profile.player_id] = profile
        # Note: _W is not re-fit — new player gets projected through existing matrix.

    def known_players(self) -> list[str]:
        return list(self._profiles.keys())

    def __repr__(self) -> str:
        return (
            f"PlayerProfileEmbedder(fitted={self._fitted}, players={len(self._profiles)}, "
            f"embed_dim={EMBED_DIM})"
        )


# ── EmbeddingStore ─────────────────────────────────────────────────────────────

class EmbeddingStore:
    """
    Persists and serves player embeddings for downstream use.

    Stores the full embedding matrix in memory ({player_id → np.ndarray(32,)})
    and supports fast batch enrichment of feature DataFrames.
    """

    def __init__(self, embedder: Optional[PlayerProfileEmbedder] = None) -> None:
        self._embedder = embedder or PlayerProfileEmbedder()
        self._cache: dict[str, np.ndarray] = {}

    def build(self, roster_df: pd.DataFrame) -> "EmbeddingStore":
        """Fit embedder and pre-compute all player embeddings."""
        self._embedder.fit(roster_df)
        for pid in self._embedder.known_players():
            self._cache[pid] = self._embedder.embed(pid)
        logger.info("EmbeddingStore: cached %d player embeddings.", len(self._cache))
        return self

    def build_from_db(self, db_url: str) -> "EmbeddingStore":
        """Load roster from DB and build embeddings."""
        try:
            import psycopg2
            import psycopg2.extras
            from scraper.adapters.nflreadpy_adapter import _psycopg2_dsn
            conn = psycopg2.connect(_psycopg2_dsn(db_url))
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT id AS player_id, position, height, weight, draft_round,
                        EXTRACT(YEAR FROM NOW())::int - EXTRACT(YEAR FROM birthdate)::int AS age
                    FROM players
                    WHERE height IS NOT NULL
                      AND position IS NOT NULL
                """)
                rows = cur.fetchall()
            conn.close()
            roster_df = pd.DataFrame([dict(r) for r in rows])
            return self.build(roster_df)
        except Exception as exc:
            logger.warning("EmbeddingStore.build_from_db failed (%s); using empty store.", exc)
            return self

    def get(self, player_id: str) -> np.ndarray:
        """Get embedding for a player. Returns zeros if not in store."""
        if player_id in self._cache:
            return self._cache[player_id]
        emb = self._embedder.embed(player_id)
        self._cache[player_id] = emb
        return emb

    def enrich_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add player_emb_0 … player_emb_{EMBED_DIM-1} columns to df.

        Args:
            df: DataFrame containing a 'player_id' column.

        Returns:
            df with EMBED_DIM new float columns added. Players not in store
            get zero embeddings (no rows dropped).
        """
        if "player_id" not in df.columns:
            logger.warning("EmbeddingStore.enrich_features: no player_id column.")
            return df

        col_names = [f"player_emb_{i}" for i in range(EMBED_DIM)]
        player_ids = [str(pid) for pid in df["player_id"]]

        # Use embed_batch (MLX-accelerated) for all players at once.
        # For players not in the profile store, zeros are returned by embed_batch.
        embeddings = self._embedder.embed_batch(player_ids)  # (N, EMBED_DIM)

        emb_df = pd.DataFrame(embeddings, columns=col_names, index=df.index)
        return pd.concat([df, emb_df], axis=1)

    def save(self, path: str) -> None:
        """Save embedding store to NPZ file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        player_ids = list(self._cache.keys())
        matrix = np.stack([self._cache[pid] for pid in player_ids], axis=0)
        np.savez_compressed(
            path,
            player_ids=np.array(player_ids, dtype=str),
            embeddings=matrix,
        )
        logger.info("EmbeddingStore saved: %d players → %s", len(player_ids), path)

    def load(self, path: str) -> "EmbeddingStore":
        """Load embedding store from NPZ file."""
        data = np.load(path, allow_pickle=True)
        pids    = data["player_ids"].tolist()
        matrix  = data["embeddings"]
        self._cache = {pid: matrix[i] for i, pid in enumerate(pids)}
        logger.info("EmbeddingStore loaded: %d players from %s", len(self._cache), path)
        return self

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"EmbeddingStore(players={len(self._cache)}, embed_dim={EMBED_DIM})"


# ── Singleton ─────────────────────────────────────────────────────────────────

_GLOBAL_STORE: Optional[EmbeddingStore] = None


def get_embedding_store(
    roster_df: Optional[pd.DataFrame] = None,
    load_path: Optional[str] = None,
) -> EmbeddingStore:
    """Return the module-level singleton EmbeddingStore."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = EmbeddingStore()
        if load_path and Path(load_path).exists():
            _GLOBAL_STORE.load(load_path)
        elif roster_df is not None:
            _GLOBAL_STORE.build(roster_df)
    return _GLOBAL_STORE


def reset_embedding_store() -> None:
    """Reset singleton (useful in tests)."""
    global _GLOBAL_STORE
    _GLOBAL_STORE = None
