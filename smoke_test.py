"""Fast smoke test for AlphaLens Lite core pipeline.

Runs synthetic data through:
    factor_engine -> backtest -> evaluation
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.evaluation import run_evaluation
from src.factor_engine import build_alpha_factors


def make_synthetic_inputs(n_days: int = 80, n_tickers: int = 10):
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

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

    return pd.DataFrame(technical_rows), pd.DataFrame(sentiment_rows), pd.DataFrame(market_rows)


def main() -> None:
    technical, sentiment, market_panel = make_synthetic_inputs()
    alpha = build_alpha_factors(technical, sentiment)
    daily, positions = run_backtest(alpha_df=alpha, market_panel=market_panel, save=False)
    metrics = run_evaluation(daily=daily, alpha_df=alpha, market_panel=market_panel, save=False)

    summary = {
        "alpha_rows": int(len(alpha)),
        "daily_rows": int(len(daily)),
        "positions_rows": int(len(positions)),
        "sharpe": float(metrics["sharpe"]),
        "ic_mean": float(metrics.get("ic_mean", 0.0)),
    }
    print("SMOKE_TEST_OK")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
