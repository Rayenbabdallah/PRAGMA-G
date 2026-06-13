"""Unit tests for the transaction graph builder and GraphSAGE encoder (PLAN.md Week 4)."""
from __future__ import annotations

import torch

from src.graph.graph_builder import build_transaction_graph, temporal_split_masks
from src.graph.graphsage import GraphSAGEEncoder
from src.tokenizer.vocab import Vocab

D_MODEL = 192


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_build_transaction_graph_shapes(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    n_edges = len(synthetic_transactions_df)
    n_nodes = pd_unique_accounts(synthetic_transactions_df)

    assert data.x.shape == (n_nodes, D_MODEL)
    assert data.edge_index.shape == (2, n_edges)
    assert data.edge_attr.shape == (n_edges, 4)
    assert data.y.shape == (n_edges,)
    assert data.edge_index.max().item() < n_nodes


def pd_unique_accounts(df):
    import pandas as pd

    return len(pd.unique(df[["Account", "Account.1"]].values.ravel()))


def test_edge_features_are_normalised(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab)

    assert torch.all(data.edge_attr >= 0.0)
    assert torch.all(data.edge_attr <= 1.0 + 1e-6)


def test_temporal_split_masks_are_60_20_20():
    n = 1000
    train_mask, val_mask, test_mask = temporal_split_masks(n)

    assert train_mask.sum().item() == 600
    assert val_mask.sum().item() == 200
    assert test_mask.sum().item() == 200
    # masks are mutually exclusive and exhaustive
    assert torch.all((train_mask.int() + val_mask.int() + test_mask.int()) == 1)


def test_temporal_split_is_chronological(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab)

    train_idx = data.train_mask.nonzero().flatten()
    test_idx = data.test_mask.nonzero().flatten()
    # All train edges precede all test edges in sorted (chronological) order
    assert train_idx.max() < test_idx.min()


# ---------------------------------------------------------------------------
# GraphSAGE encoder
# ---------------------------------------------------------------------------


def test_graphsage_output_shape(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    encoder = GraphSAGEEncoder(in_channels=D_MODEL, hidden_channels=256, out_channels=D_MODEL)
    out = encoder(data.x, data.edge_index, data.edge_attr)

    assert out.shape == (data.x.shape[0], D_MODEL)


def test_graphsage_rejects_too_few_layers():
    import pytest

    with pytest.raises(ValueError):
        GraphSAGEEncoder(n_layers=1)


def test_graphsage_inductive_on_unseen_node_subset(synthetic_transactions_df):
    """GraphSAGE must run on a node/edge subset not seen during 'training'."""
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)
    encoder = GraphSAGEEncoder(in_channels=D_MODEL, hidden_channels=256, out_channels=D_MODEL)
    encoder.eval()

    # Full graph forward pass
    full_out = encoder(data.x, data.edge_index, data.edge_attr)
    assert full_out.shape == (data.x.shape[0], D_MODEL)

    # Subgraph restricted to a held-out set of nodes/edges (e.g. test split)
    test_idx = data.test_mask.nonzero().flatten()
    sub_edge_index = data.edge_index[:, test_idx]
    sub_edge_attr = data.edge_attr[test_idx]
    sub_nodes = torch.unique(sub_edge_index)

    # New node features for an "unseen" subset (e.g. newly-appeared accounts)
    sub_x = torch.randn(data.x.shape[0], D_MODEL)
    sub_out = encoder(sub_x, sub_edge_index, sub_edge_attr)

    assert sub_out.shape == (data.x.shape[0], D_MODEL)
    assert sub_nodes.numel() > 0
