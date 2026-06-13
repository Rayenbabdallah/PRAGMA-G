"""PRAGMA-G fusion + classification head (ARCHITECTURE.md §4.3).

Fuses per-account temporal embeddings (from PRAGMA-Mini) with graph-enriched
embeddings (from GraphSAGE) and scores each *transaction* (graph edge) for
AML risk — `data.y` and `data.{train,val,test}_mask` from
`build_transaction_graph` are already edge-level with the correct ~5%
positive rate, so this is the natural prediction target (unlike a per-node
score, which would need an ad-hoc node-level label derived from edges).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.graph.graphsage import GraphSAGEEncoder


class PRAGMAGClassifier(nn.Module):
    """`(N, d_model)` temporal embeddings + graph -> `(E,)` AML risk logits."""

    def __init__(
        self,
        d_model: int = 192,
        graph_hidden_channels: int = 256,
        graph_n_layers: int = 3,
        graph_aggregation: str = "mean",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graphsage = GraphSAGEEncoder(
            in_channels=d_model,
            hidden_channels=graph_hidden_channels,
            out_channels=d_model,
            n_layers=graph_n_layers,
            aggregation=graph_aggregation,
            dropout=dropout,
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, z_temporal: Tensor, edge_index: Tensor, edge_attr: Tensor, use_graph: bool = True
    ) -> Tensor:
        """
        z_temporal: (N, d_model) — PRAGMA-Mini per-account embeddings
        edge_index: (2, E), edge_attr: (E, edge_feature_dim)
        use_graph: if False, skips GraphSAGE (the "PRAGMA-Mini only" baseline)
        Returns: (E,) raw logits — one per transaction (apply `sigmoid` for
        risk scores in [0, 1]).
        """
        if use_graph:
            z_graph = self.graphsage(z_temporal, edge_index, edge_attr)
        else:
            z_graph = torch.zeros_like(z_temporal)

        src, dst = edge_index[0], edge_index[1]
        temporal_edge = z_temporal[src] + z_temporal[dst]
        graph_edge = z_graph[src] + z_graph[dst]
        z_fused = torch.cat([temporal_edge, graph_edge], dim=-1)
        return self.fusion(z_fused).squeeze(-1)
