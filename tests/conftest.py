"""Shared pytest fixtures for PRAGMA-G tests."""
from __future__ import annotations

import pandas as pd
import pytest

from src.training.dataset import make_synthetic_transactions_df


@pytest.fixture
def synthetic_transactions_df() -> pd.DataFrame:
    """Synthetic IBM-AML-shaped transactions dataframe for fast unit tests."""
    return make_synthetic_transactions_df(n=2000, n_accounts=50, seed=42)
