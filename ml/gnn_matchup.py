"""
ml/gnn_matchup.py

Graph Neural Network Matchup Model Architecture.

PROBLEM
-------
Column-based models (XGBoost, LightGBM) cannot reason about the structure of
a game directly. They see each player in isolation rather than modeling the
interdependencies between:
    - QB ←→ WR1/WR2/TE (passing game connectivity)
    - Offensive line ←→ QB (protection network)
    - WR ←→ Corner/Safety (individual coverage matchups)
    - OC/HC ←→ scheme tendencies (play-call graph)

A Graph Attention Network (GAT) can model these relationships explicitly by
treating players as nodes and coverage matchups/snap overlaps as edges.

OVERVIEW
--------
Architecture:

    Heterogeneous Graph:
        Nodes:  offensive players (n_off), defensive players (n_def)
        Edges:  coverage matchup edges (WR → CB), pass-blocking edges (OL → DE)
                Weighted by snap overlap fraction from PBP data.

    3-Layer GAT Encoder:
        Layer 1: raw node features → 64-dim hidden
        Layer 2: 64-dim hidden → 64-dim (attended)
        Layer 3: 64-dim → 1-dim uplift scalar (per offensive player)

    Output:
        Per-player uplift multiplier applied to Kalman/XGB baseline:
        adjusted_projection = baseline × (1 + gnn_uplift)

    Training:
        Supervised on OOF residuals:
            target = actual - xgb_predicted   (the "unexplained variance")
        The GNN learns the structural context missed by the tree model.

DEPENDENCIES
------------
    - torch
    - torch_geometric  (`pip install torch-geometric`)

    Falls back to an identity multiplier (no-op) if torch_geometric unavailable.

USAGE
-----
    from ml.gnn_matchup import MatchupGraph, GNNMatchupModel, get_gnn_model

    # Build graph for a single game
    graph = MatchupGraph.from_game_data(
        off_players=off_player_df,
        def_players=def_player_df,
        coverage_matchups=matchup_df,   # from PBP snap/target tracking
    )

    # Run model (after training)
    model = get_gnn_model()
    uplifts = model(graph)
    # → Dict[player_id → float] uplift multipliers

NOTE: Training requires play-by-play matchup data (Phase 4). The model is
defined here as a skeleton — forward() can be called once PBP data is wired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from ml.feature_contract import assert_model_input_columns

logger = logging.getLogger(__name__)


# ── Node Feature Encoding ─────────────────────────────────────────────────────

OFFENSIVE_POSITIONS = ["QB", "RB", "WR", "TE", "OL"]
DEFENSIVE_POSITIONS = ["DT", "DE", "LB", "CB", "S", "FS", "SS", "ILB", "OLB"]

# Offensive node features (from feature_matrix + nflreadpy)
OFF_NODE_FEATURES = [
    "kalman_est_receiving_yards",   # form estimate
    "kalman_est_targets",           # target volume
    "prior_snap_share",             # completed prior-game usage
    "height",                       # physical profile
    "weight",
    "draft_round",
    "years_experience",
    "seas_avg_receiving_yards",     # season baseline
    "seas_avg_target_share",
]

# Defensive node features (from PBP / nflreadpy; Phase 4 data wired here)
DEF_NODE_FEATURES = [
    "career_coverage_grade",        # PFF-style feature placeholder
    "snap_pct_def",
    "opp_zone_pct",
    "opp_man_pct",
    "opp_blitz_rate",
    "tackles_per_game",
    "pbes",                         # pass breakups + expected stops
    "target_allowed_per_game",
]

N_OFF_FEATURES = len(OFF_NODE_FEATURES)
N_DEF_FEATURES = len(DEF_NODE_FEATURES)
HIDDEN_DIM   = 64
EMBED_DIM    = 64
N_GAT_LAYERS = 3


# ── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class MatchupEdge:
    """
    Directed edge from an offensive player to a defensive player.
    Weight = fraction of snaps the two players were matched up.
    """
    off_player_id: str
    def_player_id: str
    snap_overlap:  float   # [0, 1] — fraction of snaps
    is_primary:    bool    # True = primary coverage matchup

    @property
    def weight(self) -> float:
        """Edge weight: snap_overlap × 2.0 for primary matchups."""
        return self.snap_overlap * (2.0 if self.is_primary else 1.0)


@dataclass
class MatchupGraph:
    """
    Heterogeneous game matchup graph.

    Attributes:
        off_player_ids:  Ordered list of offensive player IDs (node indices)
        def_player_ids:  Ordered list of defensive player IDs
        off_features:    np.ndarray (n_off, N_OFF_FEATURES)
        def_features:    np.ndarray (n_def, N_DEF_FEATURES)
        edges:           list of MatchupEdge
        game_id:         Optional game identifier
    """
    off_player_ids: list[str]
    def_player_ids: list[str]
    off_features:   np.ndarray
    def_features:   np.ndarray
    edges:          list[MatchupEdge]
    game_id:        Optional[str] = None

    @property
    def n_off(self) -> int:
        return len(self.off_player_ids)

    @property
    def n_def(self) -> int:
        return len(self.def_player_ids)

    @classmethod
    def from_game_data(
        cls,
        off_players: pd.DataFrame,
        def_players: pd.DataFrame,
        coverage_matchups: Optional[pd.DataFrame] = None,
        game_id: Optional[str] = None,
    ) -> "MatchupGraph":
        """
        Build a MatchupGraph from DataFrames.

        Args:
            off_players:       DataFrame with [player_id, position, feature cols]
            def_players:       DataFrame with [player_id, position, feature cols]
            coverage_matchups: DataFrame with [off_player_id, def_player_id,
                               snap_overlap, is_primary]. If None, builds complete
                               bipartite graph (uniform weight=0.1).
            game_id:           Optional game identifier.

        Returns:
            MatchupGraph
        """
        assert_model_input_columns(OFF_NODE_FEATURES, consumer="GNN matchup graph")
        off_ids = off_players["player_id"].tolist()
        def_ids = def_players["player_id"].tolist()

        # Build node feature matrices (fill NaN with 0)
        def _extract_features(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
            missing = [c for c in cols if c not in df.columns]
            for c in missing:
                df = df.copy()
                df[c] = 0.0
            return df[cols].fillna(0.0).values.astype(np.float32)

        off_feat = _extract_features(off_players, OFF_NODE_FEATURES)
        def_feat = _extract_features(def_players, DEF_NODE_FEATURES)

        # Build edges
        edges: list[MatchupEdge] = []
        if coverage_matchups is not None and len(coverage_matchups) > 0:
            for _, row in coverage_matchups.iterrows():
                oid = str(row.get("off_player_id", ""))
                did = str(row.get("def_player_id", ""))
                if oid not in off_ids or did not in def_ids:
                    continue
                edges.append(MatchupEdge(
                    off_player_id=oid,
                    def_player_id=did,
                    snap_overlap=float(row.get("snap_overlap", 0.5)),
                    is_primary=bool(row.get("is_primary", False)),
                ))
        else:
            # Fallback: sparse bipartite graph (each off player → all def players)
            # with low uniform weight. This is the pre-Phase-4 fallback.
            for oid in off_ids:
                for did in def_ids:
                    edges.append(MatchupEdge(
                        off_player_id=oid,
                        def_player_id=did,
                        snap_overlap=0.1,
                        is_primary=False,
                    ))

        logger.debug(
            "MatchupGraph: %d off nodes, %d def nodes, %d edges",
            len(off_ids), len(def_ids), len(edges),
        )
        return cls(
            off_player_ids=off_ids,
            def_player_ids=def_ids,
            off_features=off_feat,
            def_features=def_feat,
            edges=edges,
            game_id=game_id,
        )


# ── GNN Model ─────────────────────────────────────────────────────────────────

class GNNMatchupModel:
    """
    3-layer Graph Attention Network (GAT) for per-player uplift prediction.

    Architecture:
        Input:
            off_nodes:  (n_off, N_OFF_FEATURES) — offensive player features
            def_nodes:  (n_def, N_DEF_FEATURES) — defensive player features
            edge_index: (2, n_edges) — connectivity (off_idx → def_idx)
            edge_attr:  (n_edges,) — edge weights

        Processing:
            Linear projection → combined node space (HIDDEN_DIM)
            3 × GATConv with multi-head attention (4 heads, concat)
            Global mean pool → game context vector
            Player-level: concat(node_emb, game_ctx) → linear → uplift scalar

        Output:
            uplift: (n_off,) — per-player multiplier (centered at 0)
                    adjusted_projection = baseline × (1 + tanh(uplift) × 0.5)

    Training regime (post Phase-4):
        Loss: MSE(baseline + baseline × tanh(uplift) × 0.5, actual)
        Optimizer: Adam lr=1e-3, weight_decay=1e-4
        Epochs: 50 with early stopping (patience=5)
    """

    def __init__(self, pretrained_path: Optional[str] = None) -> None:
        self._model: Any = None
        self._fitted = False
        self._device = "cpu"
        self._pretrained_path = pretrained_path
        self._try_load_torch()

    def _try_load_torch(self) -> None:
        """Attempt to import torch and torch_geometric. Fail gracefully."""
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GATConv, global_mean_pool

            torch.set_num_threads(1)  # prevent thread conflicts with XGB/LGB

            class _GATNet(nn.Module):
                def __init__(self):
                    super().__init__()
                    N_OFF_FEATURES + N_DEF_FEATURES
                    self.proj = nn.Linear(N_OFF_FEATURES, HIDDEN_DIM)
                    self.def_proj = nn.Linear(N_DEF_FEATURES, HIDDEN_DIM)
                    # 3 GAT layers, 4 heads each (concat → hidden_dim*heads)
                    self.gat1 = GATConv(HIDDEN_DIM, HIDDEN_DIM // 4, heads=4, concat=True,
                                        add_self_loops=True)
                    self.gat2 = GATConv(HIDDEN_DIM, HIDDEN_DIM // 4, heads=4, concat=True,
                                        add_self_loops=True)
                    self.gat3 = GATConv(HIDDEN_DIM, HIDDEN_DIM,      heads=1, concat=False,
                                        add_self_loops=True)
                    self.dropout = nn.Dropout(0.1)
                    # Final player-level head: node_emb (hidden) + game_ctx (hidden) → 1
                    self.head = nn.Sequential(
                        nn.Linear(HIDDEN_DIM * 2, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    )
                    self.act = nn.ELU()

                def forward(self, off_x, def_x, edge_index, edge_attr, batch=None):
                    """
                    Args:
                        off_x:      (n_off, N_OFF_FEATURES)
                        def_x:      (n_def, N_DEF_FEATURES)
                        edge_index: (2, n_edges) — [off_src, def_dst+n_off]
                        edge_attr:  (n_edges,)
                        batch:      (n_off,) batch assignment (optional)

                    Returns:
                        uplifts: (n_off,) — centered uplift scalars
                    """
                    import torch
                    # Project both node types into shared HIDDEN_DIM space
                    off_h = self.act(self.proj(off_x))      # (n_off, HIDDEN_DIM)
                    def_h = self.act(self.def_proj(def_x))  # (n_def, HIDDEN_DIM)
                    # Concatenate into single node tensor: [off | def]
                    x = torch.cat([off_h, def_h], dim=0)    # (n_off+n_def, HIDDEN_DIM)

                    # 3 GAT layers
                    x = self.act(self.gat1(x, edge_index, edge_attr))
                    x = self.dropout(x)
                    x = self.act(self.gat2(x, edge_index, edge_attr))
                    x = self.dropout(x)
                    x = self.gat3(x, edge_index, edge_attr)  # (n_nodes, HIDDEN_DIM)

                    # Separate off/def embeddings
                    n_off = off_x.size(0)
                    off_emb = x[:n_off]   # (n_off, HIDDEN_DIM)

                    # Game-level context: mean pool over off nodes
                    if batch is None:
                        batch = torch.zeros(n_off, dtype=torch.long, device=off_x.device)
                    game_ctx = global_mean_pool(off_emb, batch)  # (n_graphs, HIDDEN_DIM)
                    game_ctx_expanded = game_ctx[batch]          # (n_off, HIDDEN_DIM)

                    # Per-player uplift
                    combined = torch.cat([off_emb, game_ctx_expanded], dim=1)  # (n_off, HIDDEN_DIM*2)
                    uplifts = self.head(combined).squeeze(-1)     # (n_off,)
                    return uplifts

            self._GATNet = _GATNet
            self._torch = torch

            if self._pretrained_path and Path(self._pretrained_path).exists():
                self._model = _GATNet()
                self._model.load_state_dict(torch.load(self._pretrained_path, map_location="cpu"))
                self._model.eval()
                self._fitted = True
                logger.info("GNNMatchupModel loaded from %s", self._pretrained_path)
            else:
                self._model = _GATNet()
                self._model.eval()
                logger.info(
                    "GNNMatchupModel architecture initialized (not fitted). "
                    "Call .train_model() after Phase-4 PBP data is available."
                )

        except ImportError as exc:
            logger.info(
                "torch_geometric not installed (%s). GNNMatchupModel will return "
                "identity uplifts (1.0 multiplier). Install: pip install torch-geometric",
                exc,
            )

    def forward(self, graph: MatchupGraph) -> dict[str, float]:
        """
        Compute per-player uplift multipliers for one game's matchup graph.

        Args:
            graph: MatchupGraph for the game being projected.

        Returns:
            Dict[player_id → float] : uplift multipliers.
            If model not fitted or torch_geometric unavailable, returns 1.0 for all.
        """
        if self._model is None or self._GATNet is None:
            # Graceful fallback: identity uplifts
            return {pid: 1.0 for pid in graph.off_player_ids}

        import torch

        # Convert numpy features to tensors
        off_x = torch.tensor(graph.off_features, dtype=torch.float32)  # (n_off, N_OFF_FEAT)
        def_x = torch.tensor(graph.def_features, dtype=torch.float32)  # (n_def, N_DEF_FEAT)
        n_off  = graph.n_off

        # Build edge index and attrs
        off_id_to_idx = {pid: i for i, pid in enumerate(graph.off_player_ids)}
        def_id_to_idx = {pid: i + n_off for i, pid in enumerate(graph.def_player_ids)}

        srcs, dsts, weights = [], [], []
        for edge in graph.edges:
            if edge.off_player_id in off_id_to_idx and edge.def_player_id in def_id_to_idx:
                srcs.append(off_id_to_idx[edge.off_player_id])
                dsts.append(def_id_to_idx[edge.def_player_id])
                weights.append(edge.weight)

        if not srcs:
            # No edges — return identity
            return {pid: 1.0 for pid in graph.off_player_ids}

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)   # (2, n_edges)
        edge_attr  = torch.tensor(weights, dtype=torch.float32)      # (n_edges,)

        with torch.no_grad():
            raw_uplifts = self._model(off_x, def_x, edge_index, edge_attr)  # (n_off,)
            # Map to bounded multipliers: 1.0 ± 0.5 × tanh(raw)
            multipliers = 1.0 + 0.5 * torch.tanh(raw_uplifts)
            multipliers = multipliers.numpy()

        return {
            pid: float(multipliers[i])
            for i, pid in enumerate(graph.off_player_ids)
        }

    def train_model(
        self,
        training_graphs: list[tuple[MatchupGraph, np.ndarray]],
        n_epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5,
    ) -> "GNNMatchupModel":
        """
        Train the GNN on (MatchupGraph, residual_targets) pairs.

        Args:
            training_graphs: List of (graph, residuals) where residuals is
                np.ndarray(n_off,) of (actual - xgb_predicted) for each player.
            n_epochs:        Maximum training epochs.
            lr:              Adam learning rate.
            patience:        Early stopping patience.

        Returns:
            self

        NOTE: Requires Phase-4 PBP matchup data. This method is a stub.
        """
        if self._model is None:
            logger.warning("Cannot train: torch_geometric not available.")
            return self

        if len(training_graphs) < 10:
            logger.warning(
                "Only %d training graphs provided. Need PBP matchup data (Phase 4).",
                len(training_graphs),
            )
            return self

        import torch
        import torch.nn as nn
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0

        self._model.train()
        for epoch in range(n_epochs):
            total_loss = 0.0
            for graph, targets in training_graphs:
                off_x = torch.tensor(graph.off_features, dtype=torch.float32)
                def_x = torch.tensor(graph.def_features, dtype=torch.float32)
                y     = torch.tensor(targets, dtype=torch.float32)

                # Build edge tensors (same as forward())
                n_off = graph.n_off
                off_id_to_idx = {p: i for i, p in enumerate(graph.off_player_ids)}
                def_id_to_idx = {p: i + n_off for i, p in enumerate(graph.def_player_ids)}

                srcs, dsts, wts = [], [], []
                for edge in graph.edges:
                    if edge.off_player_id in off_id_to_idx and edge.def_player_id in def_id_to_idx:
                        srcs.append(off_id_to_idx[edge.off_player_id])
                        dsts.append(def_id_to_idx[edge.def_player_id])
                        wts.append(edge.weight)

                if not srcs:
                    continue

                edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
                edge_attr  = torch.tensor(wts, dtype=torch.float32)

                optimizer.zero_grad()
                uplifts = self._model(off_x, def_x, edge_index, edge_attr)
                loss = criterion(uplifts, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / max(len(training_graphs), 1)

            if epoch % 5 == 0:
                logger.info("GNN epoch %d/%d | loss=%.4f", epoch + 1, n_epochs, avg_loss)

            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("GNN early stopping at epoch %d.", epoch + 1)
                    break

        self._model.eval()
        self._fitted = True
        logger.info("GNNMatchupModel training complete. Best loss: %.4f", best_loss)
        return self

    def save(self, path: str) -> None:
        """Save model state_dict."""
        if self._model is None:
            logger.warning("Cannot save: torch not available or model not initialized.")
            return
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)
        logger.info("GNNMatchupModel saved → %s", path)

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "architecture only (requires Phase-4 PBP data)"
        return f"GNNMatchupModel(status={status})"


# ── Singleton ─────────────────────────────────────────────────────────────────

_GLOBAL_GNN: Optional[GNNMatchupModel] = None


def get_gnn_model(pretrained_path: Optional[str] = None) -> GNNMatchupModel:
    global _GLOBAL_GNN
    if _GLOBAL_GNN is None:
        _GLOBAL_GNN = GNNMatchupModel(pretrained_path=pretrained_path)
    return _GLOBAL_GNN


def reset_gnn_model() -> None:
    global _GLOBAL_GNN
    _GLOBAL_GNN = None
