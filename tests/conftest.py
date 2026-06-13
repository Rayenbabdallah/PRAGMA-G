"""Shared pytest fixtures for PRAGMA-G tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_transactions_df() -> pd.DataFrame:
    """Synthetic IBM-AML-shaped transactions dataframe for fast unit tests."""
    rng = np.random.default_rng(42)
    n = 2000
    n_accounts = 50

    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="3min"),
            "From Bank": rng.choice(["10", "20", "30", "40"], n),
            "Account": [f"ACC_{i % n_accounts:04d}" for i in range(n)],
            "To Bank": rng.choice(["10", "20", "30", "40"], n),
            "Account.1": [f"ACC_{(i + 7) % n_accounts:04d}" for i in range(n)],
            "Amount Received": np.round(rng.lognormal(4, 1.5, n), 2),
            "Receiving Currency": rng.choice(["US Dollar", "Euro", "Bitcoin"], n),
            "Amount Paid": np.round(rng.lognormal(4, 1.5, n), 2),
            "Payment Currency": rng.choice(["US Dollar", "Euro", "Bitcoin"], n),
            "Payment Format": rng.choice(
                ["Reinvestment", "Wire", "Cheque", "Credit Card", "Cash", "Bitcoin"], n
            ),
            "Is Laundering": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )
