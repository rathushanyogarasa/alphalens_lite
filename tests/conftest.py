from __future__ import annotations

import pytest


@pytest.fixture
def make_synthetic_inputs():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=80)
    tickers = [f"T{i:02d}" for i in range(10)]

    technical_rows = []
    sentiment_rows = []
    market_rows = []

    for d in dates:
        for t in tickers:
            technical_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "momentum_z": rng.normal(),
                    "volatility_z": rng.normal(),
                }
            )
            sentiment_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "sentiment_z": rng.normal(),
                }
            )
            market_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "ret_1d": rng.normal(0.0002, 0.01),
                    "fwd_ret_1d": rng.normal(0.0002, 0.01),
                }
            )

    technical = pd.DataFrame(technical_rows)
    sentiment = pd.DataFrame(sentiment_rows)
    market_panel = pd.DataFrame(market_rows)
    return technical, sentiment, market_panel
