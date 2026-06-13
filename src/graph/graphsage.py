"""GraphSAGE encoder for the PRAGMA-G graph extension (§4.2).

Inductive GNN over the account-to-account transaction graph, using
PRAGMA-Mini embeddings as node features and injecting edge features
(amount, time delta, payment format, currency) via mean aggregation into
each node before message passing. Mean aggregation (`SAGEConv` default) is
the most stable choice for financial graphs with high-degree hub accounts;
3 layers cover a 2-hop neighbourhood, enough to capture fan-in/fan-out
laundering patterns.
"""
from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter

from src.graph.graph_builder import EDGE_FEATURE_DIM


class GraphSAGEEncoder(nn.Module):
    """Inductive 3-layer GraphSAGE encoder: `(N, in_channels) -> (N, out_channels)`."""

    def __init__(
        self,
        in_channels: int = 192,
        hidden_channels: int = 256,
        out_channels: int = 192,
        n_layers: int = 3,
        edge_feature_dim: int = EDGE_FEATURE_DIM,
        dropout: float = 0.1,
        aggregation: str = "mean",
    ):
        super().__init__()
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2, got {n_layers}")

        self.edge_encoder = nn.Linear(edge_feature_dim, in_channels)
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggregation))
        for _ in range(n_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggregation))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggregation))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        x: (N, in_channels) — PRAGMA-Mini node embeddings
        edge_index: (2, E) — directed transaction edges
        edge_attr: (E, edge_feature_dim) — edge features
        Returns: (N, out_channels) — graph-enriched node embeddings
        """
        edge_emb = self.edge_encoder(edge_attr)
        edge_context = scatter(edge_emb, edge_index[1], dim=0, dim_size=x.size(0), reduce="mean")
        x = x + edge_context

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
