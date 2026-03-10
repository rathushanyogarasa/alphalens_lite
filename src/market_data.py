"""src/market_data.py - Price data ingestion and return panel builder."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

import config

logger = logging.getLogger(__name__)


_RAW_PRICE_CACHE = config.RAW_DIR / "market_prices.csv"
_PROCESSED_PANEL = config.PROCESSED_DIR / "market_panel.csv"


def _normalize_tickers(tickers: list[str] | None) -> list[str]:
    values = tickers or config.TICKERS
    out = sorted({str(t).strip().upper() for t in values if str(t).strip()})
    if not out:
        raise ValueError("No tickers were provided.")
    return out


def _default_end_date(end_date: str | None) -> str:
    if end_date:
        return end_date
    return date.today().strftime("%Y-%m-%d")


def _fetch_prices_yfinance(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except Exception as exc:
        raise RuntimeError(
            "yfinance is required for market data fetch. Install it with `pip install yfinance`."
        ) from exc

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("No market data returned from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(1):
            wide = raw.xs("Adj Close", level=1, axis=1)
        elif "Close" in raw.columns.get_level_values(1):
            wide = raw.xs("Close", level=1, axis=1)
        else:
            raise RuntimeError("Could not find Adj Close or Close columns in yfinance output.")
    else:
        series_name = tickers[0] if len(tickers) == 1 else "PRICE"
        wide = raw.rename(series_name).to_frame() if isinstance(raw, pd.Series) else raw[["Adj Close"]]
        if "Adj Close" in wide.columns:
            wide.columns = [series_name]

    wide.index = pd.to_datetime(wide.index).tz_localize(None)
    wide = wide.sort_index().dropna(how="all")
    wide.columns = [str(c).upper() for c in wide.columns]
    return wide


def load_prices(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Load wide price matrix indexed by date, columns=tickers."""
    tickers = _normalize_tickers(tickers)
    start_date = start_date or config.MARKET_START_DATE
    end_date = _default_end_date(end_date or config.MARKET_END_DATE)

    if _RAW_PRICE_CACHE.exists() and not refresh:
        cached = pd.read_csv(_RAW_PRICE_CACHE, parse_dates=["date"]).set_index("date")
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        have = [t for t in tickers if t in cached.columns]
        if have:
            logger.info("Loaded market cache: %s", _RAW_PRICE_CACHE)
            return cached[have].sort_index()

    if config.MARKET_PROVIDER.lower() != "yfinance":
        raise ValueError(f"Unsupported MARKET_PROVIDER='{config.MARKET_PROVIDER}'")

    prices = _fetch_prices_yfinance(tickers, start_date, end_date)
    _save_prices_cache(prices)
    return prices


def _save_prices_cache(prices: pd.DataFrame) -> None:
    df = prices.copy().reset_index().rename(columns={"index": "date"})
    df.to_csv(_RAW_PRICE_CACHE, index=False)
    logger.info("Saved price cache -> %s", _RAW_PRICE_CACHE)


def build_market_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert wide prices to long panel with ret_1d and fwd_ret_1d."""
    rets = prices.pct_change()
    panel = (
        prices.stack()
        .rename("close")
        .to_frame()
        .join(rets.stack().rename("ret_1d"))
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker"})
    )
    panel["fwd_ret_1d"] = panel.groupby("ticker")["ret_1d"].shift(-1)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].str.upper()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def run_market_data(
    tickers: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    refresh: bool = False,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch/load prices and build market panel.

    Returns:
        prices_wide, market_panel_long
    """
    prices = load_prices(tickers=tickers, start_date=start_date, end_date=end_date, refresh=refresh)
    panel = build_market_panel(prices)

    if save:
        panel.to_csv(_PROCESSED_PANEL, index=False)
        logger.info("Saved market panel -> %s", _PROCESSED_PANEL)

    return prices, panel
