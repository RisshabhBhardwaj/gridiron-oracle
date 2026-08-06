"""
backend/tests/test_gnn_matchup.py

Unit tests for ml/gnn_matchup.py.

All tests run without a database connection and without torch_geometric
installed (tests rely on the identity-fallback path by default, with
optional torch_geometric tests skipped when unavailable).
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from ml.gnn_matchup import (
    OFFENSIVE_POSITIONS,
    DEFENSIVE_POSITIONS,
    OFF_NODE_FEATURES,
    DEF_NODE_FEATURES,
    N_OFF_FEATURES,
    N_DEF_FEATURES,
    HIDDEN_DIM,
    N_GAT_LAYERS,
    MatchupEdge,
    MatchupGraph,
    GNNMatchupModel,
    get_gnn_model,
    reset_gnn_model,
)

# Detect torch_geometric availability for conditional skips
try:
    import torch_geometric  # noqa: F401
    _HAS_PYGEOM = True
except ImportError:
    _HAS_PYGEOM = False

requires_pygeom = pytest.mark.skipif(
    not _HAS_PYGEOM, reason="torch_geometric not installed"
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_off_df(n: int = 4) -> pd.DataFrame:
    """Build a minimal offensive player DataFrame."""
    _pos_cycle = ["WR", "WR", "RB", "TE", "QB", "WR", "RB", "TE"]
    return pd.DataFrame({
        "player_id":  [f"off_{i}" for i in range(n)],
        "position":   (_pos_cycle * ((n // len(_pos_cycle)) + 1))[:n],
        **{col: np.random.default_rng(0).uniform(0, 1, n) for col in OFF_NODE_FEATURES},
    })


def _make_def_df(n: int = 3) -> pd.DataFrame:
    """Build a minimal defensive player DataFrame."""
    return pd.DataFrame({
        "player_id":  [f"def_{i}" for i in range(n)],
        "position":   ["CB", "CB", "S"][:n],
        **{col: np.random.default_rng(1).uniform(0, 1, n) for col in DEF_NODE_FEATURES},
    })


def _make_matchup_df(off_ids, def_ids) -> pd.DataFrame:
    """Build a small coverage matchup DataFrame."""
    return pd.DataFrame([
        {"off_player_id": off_ids[0], "def_player_id": def_ids[0],
         "snap_overlap": 0.7, "is_primary": True},
        {"off_player_id": off_ids[1], "def_player_id": def_ids[1],
         "snap_overlap": 0.5, "is_primary": True},
        {"off_player_id": off_ids[0], "def_player_id": def_ids[2],
         "snap_overlap": 0.2, "is_primary": False},
    ])


# ── TestMatchupEdge ───────────────────────────────────────────────────────────

class TestMatchupEdge:
    def test_weight_primary(self):
        edge = MatchupEdge("o1", "d1", snap_overlap=0.5, is_primary=True)
        assert edge.weight == pytest.approx(1.0)  # 0.5 * 2.0

    def test_weight_non_primary(self):
        edge = MatchupEdge("o1", "d1", snap_overlap=0.5, is_primary=False)
        assert edge.weight == pytest.approx(0.5)

    def test_weight_zero_overlap(self):
        edge = MatchupEdge("o1", "d1", snap_overlap=0.0, is_primary=True)
        assert edge.weight == pytest.approx(0.0)

    def test_fields_stored(self):
        edge = MatchupEdge("off_42", "def_7", snap_overlap=0.3, is_primary=False)
        assert edge.off_player_id == "off_42"
        assert edge.def_player_id == "def_7"
        assert edge.snap_overlap == pytest.approx(0.3)
        assert edge.is_primary is False


# ── TestMatchupGraph ──────────────────────────────────────────────────────────

class TestMatchupGraph:
    def test_from_game_data_basic(self):
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_)
        assert g.n_off == 4
        assert g.n_def == 3

    def test_node_feature_shapes(self):
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_)
        assert g.off_features.shape == (4, N_OFF_FEATURES)
        assert g.def_features.shape == (3, N_DEF_FEATURES)

    def test_feature_dtype_float32(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        assert g.off_features.dtype == np.float32
        assert g.def_features.dtype == np.float32

    def test_coverage_matchup_edges_loaded(self):
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        matchups = _make_matchup_df(off["player_id"].tolist(), def_["player_id"].tolist())
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        assert len(g.edges) == 3

    def test_edges_filter_unknown_players(self):
        """Edges referencing unknown player IDs should be silently dropped."""
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        bad_matchups = pd.DataFrame([
            {"off_player_id": "UNKNOWN", "def_player_id": def_["player_id"][0],
             "snap_overlap": 0.5, "is_primary": True},
        ])
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=bad_matchups)
        assert len(g.edges) == 0

    def test_fallback_bipartite_when_no_matchups(self):
        """Without matchup data, builds complete bipartite graph."""
        off = _make_off_df(2)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=None)
        # 2 off × 3 def = 6 edges
        assert len(g.edges) == 6

    def test_fallback_bipartite_uniform_weight(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=None)
        for edge in g.edges:
            assert edge.snap_overlap == pytest.approx(0.1)
            assert edge.is_primary is False

    def test_missing_features_filled_with_zeros(self):
        """Columns missing from the input DF should default to 0."""
        off = pd.DataFrame({"player_id": ["o1", "o2"], "position": ["WR", "RB"]})
        def_ = pd.DataFrame({"player_id": ["d1"], "position": ["CB"]})
        g = MatchupGraph.from_game_data(off, def_)
        assert np.all(g.off_features == 0.0)
        assert np.all(g.def_features == 0.0)

    def test_game_id_stored(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_, game_id="2025_01_KC_LAC")
        assert g.game_id == "2025_01_KC_LAC"

    def test_player_id_order_preserved(self):
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_)
        assert g.off_player_ids == off["player_id"].tolist()
        assert g.def_player_ids == def_["player_id"].tolist()

    def test_empty_off_df(self):
        off = _make_off_df(0).iloc[:0]
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        assert g.n_off == 0
        assert g.off_features.shape == (0, N_OFF_FEATURES)

    def test_empty_matchup_df_ignored(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        empty_matchups = pd.DataFrame(
            columns=["off_player_id", "def_player_id", "snap_overlap", "is_primary"]
        )
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=empty_matchups)
        # Empty matchup df → fall back to bipartite (len > 0 check is False)
        assert len(g.edges) == 4  # 2 × 2 bipartite

    def test_nan_features_filled(self):
        """NaN values in off_players features should become 0.0."""
        off = _make_off_df(2)
        off[OFF_NODE_FEATURES[0]] = np.nan
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        assert np.all(np.isfinite(g.off_features))


# ── TestGNNMatchupModelFallback ───────────────────────────────────────────────

class TestGNNMatchupModelFallback:
    """
    Tests that work regardless of whether torch_geometric is installed.
    When torch_geometric is absent the model returns identity uplifts (1.0).
    When torch_geometric IS installed the untrained model also returns values
    in [0.5, 1.5] range (tanh bounded).
    """

    def test_forward_returns_dict_for_all_off_players(self):
        reset_gnn_model()
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_)
        model = GNNMatchupModel()
        result = model.forward(g)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(off["player_id"].tolist())

    def test_forward_identity_when_no_edges(self):
        """A graph with no valid edges always returns 1.0 multipliers."""
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        # Build graph but replace edges with empty list
        g = MatchupGraph.from_game_data(off, def_)
        g = MatchupGraph(
            off_player_ids=g.off_player_ids,
            def_player_ids=g.def_player_ids,
            off_features=g.off_features,
            def_features=g.def_features,
            edges=[],  # no edges
        )
        model = GNNMatchupModel()
        result = model.forward(g)
        for v in result.values():
            assert v == pytest.approx(1.0)

    def test_forward_empty_graph_returns_empty_dict(self):
        off = _make_off_df(0).iloc[:0]
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        model = GNNMatchupModel()
        result = model.forward(g)
        assert result == {}

    def test_forward_multipliers_are_finite(self):
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        matchups = _make_matchup_df(off["player_id"].tolist(), def_["player_id"].tolist())
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        model = GNNMatchupModel()
        result = model.forward(g)
        for v in result.values():
            assert np.isfinite(v)

    def test_forward_multipliers_bounded(self):
        """Multipliers must be in (0.5, 1.5] = 1 ± 0.5*tanh(anything)."""
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        matchups = _make_matchup_df(off["player_id"].tolist(), def_["player_id"].tolist())
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        model = GNNMatchupModel()
        result = model.forward(g)
        for v in result.values():
            assert 0.4 <= v <= 1.6, f"Multiplier {v} out of expected bounds"

    def test_train_model_no_op_when_few_graphs(self):
        """train_model() returns self without crashing when < 10 graphs provided."""
        model = GNNMatchupModel()
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        targets = np.zeros(2, dtype=np.float32)
        result = model.train_model([(g, targets)])
        assert result is model  # returns self

    def test_repr_contains_status(self):
        model = GNNMatchupModel()
        r = repr(model)
        assert "GNNMatchupModel" in r

    def test_save_no_crash_when_no_torch(self, tmp_path):
        """save() should not raise even if model is None."""
        model = GNNMatchupModel()
        model._model = None  # simulate no-torch path
        model.save(str(tmp_path / "model.pt"))  # should log warning, not crash


# ── TestGetGnnModel (singleton) ───────────────────────────────────────────────

class TestGetGnnModel:
    def test_singleton_returns_same_instance(self):
        reset_gnn_model()
        m1 = get_gnn_model()
        m2 = get_gnn_model()
        assert m1 is m2

    def test_reset_clears_singleton(self):
        reset_gnn_model()
        m1 = get_gnn_model()
        reset_gnn_model()
        m2 = get_gnn_model()
        assert m1 is not m2

    def test_get_gnn_model_returns_gnn_instance(self):
        reset_gnn_model()
        m = get_gnn_model()
        assert isinstance(m, GNNMatchupModel)


# ── TestConstants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_n_off_features_matches_list(self):
        assert N_OFF_FEATURES == len(OFF_NODE_FEATURES)

    def test_n_def_features_matches_list(self):
        assert N_DEF_FEATURES == len(DEF_NODE_FEATURES)

    def test_hidden_dim_positive(self):
        assert HIDDEN_DIM > 0

    def test_gat_layers_is_three(self):
        assert N_GAT_LAYERS == 3

    def test_offensive_positions_non_empty(self):
        assert len(OFFENSIVE_POSITIONS) > 0

    def test_defensive_positions_non_empty(self):
        assert len(DEFENSIVE_POSITIONS) > 0

    def test_no_position_overlap(self):
        shared = set(OFFENSIVE_POSITIONS) & set(DEFENSIVE_POSITIONS)
        assert shared == set(), f"Overlapping positions: {shared}"


# ── TestMatchupGraphEdgeWeights ───────────────────────────────────────────────

class TestMatchupGraphEdgeWeights:
    def test_primary_edge_weight_doubled(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        matchups = pd.DataFrame([{
            "off_player_id": off["player_id"][0],
            "def_player_id": def_["player_id"][0],
            "snap_overlap": 0.4,
            "is_primary": True,
        }])
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        assert len(g.edges) == 1
        assert g.edges[0].weight == pytest.approx(0.8)  # 0.4 × 2.0

    def test_non_primary_edge_weight_unchanged(self):
        off = _make_off_df(2)
        def_ = _make_def_df(2)
        matchups = pd.DataFrame([{
            "off_player_id": off["player_id"][0],
            "def_player_id": def_["player_id"][0],
            "snap_overlap": 0.3,
            "is_primary": False,
        }])
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        assert g.edges[0].weight == pytest.approx(0.3)


# ── TestGNNForwardWithTorch (optional) ────────────────────────────────────────

@requires_pygeom
class TestGNNForwardWithTorch:
    """Tests that run only when torch_geometric is available."""

    def test_model_is_initialized(self):
        model = GNNMatchupModel()
        assert model._model is not None
        assert model._torch is not None

    def test_forward_untrained_runs_without_error(self):
        model = GNNMatchupModel()
        off = _make_off_df(4)
        def_ = _make_def_df(3)
        matchups = _make_matchup_df(off["player_id"].tolist(), def_["player_id"].tolist())
        g = MatchupGraph.from_game_data(off, def_, coverage_matchups=matchups)
        result = model.forward(g)
        assert len(result) == 4

    def test_forward_returns_float_values(self):
        model = GNNMatchupModel()
        off = _make_off_df(3)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        result = model.forward(g)
        for v in result.values():
            assert isinstance(v, float)

    def test_forward_tanh_bounded_output(self):
        """Untrained model: output ∈ (0.5, 1.5) since 1 ± 0.5*tanh(any)."""
        model = GNNMatchupModel()
        off = _make_off_df(5)
        def_ = _make_def_df(3)
        g = MatchupGraph.from_game_data(off, def_)
        result = model.forward(g)
        for v in result.values():
            assert 0.49 < v < 1.51, f"Expected 1 ± 0.5*tanh, got {v}"

    def test_save_and_load(self, tmp_path):
        import torch
        model = GNNMatchupModel()
        save_path = str(tmp_path / "gnn.pt")
        model.save(save_path)
        assert (tmp_path / "gnn.pt").exists()

        # Load into a new model
        loaded = GNNMatchupModel(pretrained_path=save_path)
        assert loaded._fitted is True
        assert loaded._model is not None

    def test_forward_with_loaded_model(self, tmp_path):
        import torch
        model = GNNMatchupModel()
        save_path = str(tmp_path / "gnn.pt")
        model.save(save_path)

        loaded = GNNMatchupModel(pretrained_path=save_path)
        off = _make_off_df(3)
        def_ = _make_def_df(2)
        g = MatchupGraph.from_game_data(off, def_)
        result = loaded.forward(g)
        assert set(result.keys()) == set(off["player_id"].tolist())

    def test_train_model_with_sufficient_graphs(self, tmp_path):
        """train_model() runs without error when given ≥10 training graphs."""
        model = GNNMatchupModel()
        training_data = []
        for _ in range(12):
            off = _make_off_df(3)
            def_ = _make_def_df(2)
            g = MatchupGraph.from_game_data(off, def_)
            targets = np.random.default_rng(0).normal(0, 5, size=3).astype(np.float32)
            training_data.append((g, targets))

        result = model.train_model(training_data, n_epochs=2, patience=1)
        assert result is model

    def test_trained_model_fitted_flag(self):
        model = GNNMatchupModel()
        training_data = []
        for _ in range(10):
            off = _make_off_df(2)
            def_ = _make_def_df(2)
            g = MatchupGraph.from_game_data(off, def_)
            targets = np.zeros(2, dtype=np.float32)
            training_data.append((g, targets))

        model.train_model(training_data, n_epochs=2, patience=1)
        assert model._fitted is True
