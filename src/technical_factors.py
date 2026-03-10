"""src/technical_factors.py - Momentum and volatility factor computation."""

from __future__ import annotations

import logging

import pandas as pd

import config
from src.market_data import load_prices

logger = logging.getLogger(__name__)

_TECHNICAL_FACTOR = config.PROCESSED_DIR / "technical_factors.csv"


def _cross_section_zscore(long_df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    stats = long_df.groupby("date")[col].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sigma"})
    out = long_df.join(stats, on="date")
    out[out_col] = (out[col] - out["mu"]) / out["sigma"].replace(0, pd.NA)
    out[out_col] = out[out_col].fillna(0.0)
    return out.drop(columns=["mu", "sigma"])


def compute_technical_factors(
    prices: pd.DataFrame,
    momentum_window: int = config.MOMENTUM_WINDOW,
    volatility_window: int = config.VOLATILITY_WINDOW,
    min_obs: int = config.MIN_FACTOR_OBS,
) -> pd.DataFrame:
    """Build daily momentum and low-volatility factors."""
    rets = prices.pct_change()

    momentum = prices / prices.shift(momentum_window) - 1.0
    vol = rets.rolling(volatility_window, min_periods=min_obs).std()
    low_vol = -vol

    factors = (
        pd.concat(
            {
                "momentum_raw": momentum.stack(),
                "volatility_raw": low_vol.stack(),
            },
            axis=1,
        )
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )

    factors["date"] = pd.to_datetime(factors["date"]).dt.normalize()
    factors["ticker"] = factors["ticker"].astype(str).str.upper().str.strip()

    factors = _cross_section_zscore(factors, "momentum_raw", "momentum_z")
    factors = _cross_section_zscore(factors, "volatility_raw", "volatility_z")

    return factors.sort_values(["date", "ticker"]).reset_index(drop=True)


def run_technical_factors(
    prices: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh_prices: bool = False,
    save: bool = True,
) -> pd.DataFrame:
    """Load prices (if needed) and compute technical factor table."""
    if prices is None:
        prices = load_prices(
            tickers=tickers or config.TICKERS,
            start_date=start_date or config.MARKET_START_DATE,
            end_date=end_date or config.MARKET_END_DATE,
            refresh=refresh_prices,
        )

    factors = compute_technical_factors(prices)
    if save:
        factors.to_csv(_TECHNICAL_FACTOR, index=False)
        logger.info("Saved technical factors -> %s", _TECHNICAL_FACTOR)

    return factors
