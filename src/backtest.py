"""src/backtest.py - Fast non-overlapping long/short backtest for AlphaLens Lite."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

_ALPHA_FACTOR_PATH = config.PROCESSED_DIR / "alpha_factors.csv"
_MARKET_PANEL_PATH = config.PROCESSED_DIR / "market_panel.csv"
_BACKTEST_DAILY_PATH = config.METRICS_DIR / "backtest_daily.csv"
_BACKTEST_POSITIONS_PATH = config.METRICS_DIR / "positions.csv"


_REQUIRED_ALPHA_COLS = {"date", "ticker", "alpha_score"}
_REQUIRED_MARKET_COLS = {"date", "ticker", "ret_1d"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _load_default_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _ALPHA_FACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing alpha factors: {_ALPHA_FACTOR_PATH}")
    if not _MARKET_PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing market panel: {_MARKET_PANEL_PATH}")

    alpha = pd.read_csv(_ALPHA_FACTOR_PATH)
    market_panel = pd.read_csv(_MARKET_PANEL_PATH)
    return alpha, market_panel


def _prepare_frames(alpha_df: pd.DataFrame, market_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_columns(alpha_df, _REQUIRED_ALPHA_COLS, "alpha_df")
    _validate_columns(market_panel, _REQUIRED_MARKET_COLS, "market_panel")

    alpha = alpha_df.copy()
    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    alpha["ticker"] = alpha["ticker"].astype(str).str.upper().str.strip()

    panel = market_panel.copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()
    panel["ret_1d"] = pd.to_numeric(panel["ret_1d"], errors="coerce")

    return alpha, panel


def _select_quantile_positions(day_df: pd.DataFrame, cutoff: float) -> tuple[list[str], list[str]]:
    ranked = day_df.sort_values("alpha_score", ascending=False).reset_index(drop=True)
    n = len(ranked)
    n_each = max(1, int(np.floor(n * cutoff)))
    longs = ranked.head(n_each)["ticker"].tolist()
    shorts = ranked.tail(n_each)["ticker"].tolist()
    return longs, shorts


def run_backtest(
    alpha_df: pd.DataFrame | None = None,
    market_panel: pd.DataFrame | None = None,
    hold_days: int | None = None,
    quantile_cutoff: float | None = None,
    transaction_cost_bps: float | None = None,
    slippage_bps: float | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a non-overlapping cross-sectional long/short backtest.

    Signals from day T are applied from day T+1 onward for `hold_days`.
    """
    if alpha_df is None or market_panel is None:
        alpha_df, market_panel = _load_default_inputs()

    alpha, panel = _prepare_frames(alpha_df, market_panel)

    hold_days = int(hold_days or config.HOLD_DAYS)
    quantile_cutoff = float(quantile_cutoff or config.QUANTILE_CUTOFF)
    tc_bps = float(transaction_cost_bps if transaction_cost_bps is not None else config.TRANSACTION_COST_BPS)
    slip_bps = float(slippage_bps if slippage_bps is not None else config.SLIPPAGE_BPS)
    one_way_cost = (tc_bps + slip_bps) / 10_000.0

    returns_wide = panel.pivot(index="date", columns="ticker", values="ret_1d").sort_index()
    all_tickers = returns_wide.columns.tolist()

    alpha_dates = sorted(alpha["date"].unique())
    trading_dates = returns_wide.index

    prev_weights = pd.Series(0.0, index=all_tickers)
    rows: list[dict] = []
    positions_rows: list[dict] = []

    for rebalance_date in alpha_dates[::hold_days]:
        day_scores = alpha.loc[alpha["date"] == rebalance_date, ["ticker", "alpha_score"]].dropna()
        day_scores = day_scores[day_scores["ticker"].isin(all_tickers)]
        if len(day_scores) < 4:
            continue

        longs, shorts = _select_quantile_positions(day_scores, quantile_cutoff)
        if not longs or not shorts:
            continue

        weights = pd.Series(0.0, index=all_tickers)
        long_w = 0.5 / len(longs)
        short_w = -0.5 / len(shorts)
        weights.loc[longs] = long_w
        weights.loc[shorts] = short_w

        turnover = float((weights - prev_weights).abs().sum())
        cost_today = turnover * one_way_cost

        pos = int(trading_dates.searchsorted(rebalance_date))
        start = pos + 1
        end = min(start + hold_days, len(trading_dates))
        if start >= end:
            prev_weights = weights
            continue

        signal_map = day_scores.set_index("ticker")["alpha_score"].to_dict()
        for t in longs:
            positions_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "ticker": t,
                    "side": "long",
                    "weight": float(long_w),
                    "alpha_score": float(signal_map.get(t, np.nan)),
                }
            )
        for t in shorts:
            positions_rows.append(
                {
                    "rebalance_date": rebalance_date,
                    "ticker": t,
                    "side": "short",
                    "weight": float(short_w),
                    "alpha_score": float(signal_map.get(t, np.nan)),
                }
            )

        for i in range(start, end):
            d = trading_dates[i]
            day_ret = returns_wide.loc[d].fillna(0.0)
            port_ret = float((weights * day_ret).sum())
            bench_ret = float(day_ret.mean()) if len(day_ret) else 0.0

            if i == start:
                port_ret -= cost_today

            rows.append(
                {
                    "date": d,
                    "rebalance_date": rebalance_date,
                    "portfolio_ret": port_ret,
                    "benchmark_ret": bench_ret,
                    "turnover": turnover if i == start else 0.0,
                }
            )

        prev_weights = weights

    if not rows:
        raise RuntimeError("No backtest rows were produced. Check alpha coverage and market data range.")

    daily = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    daily["portfolio_nav"] = (1.0 + daily["portfolio_ret"]).cumprod()
    daily["benchmark_nav"] = (1.0 + daily["benchmark_ret"]).cumprod()
    daily["drawdown"] = daily["portfolio_nav"] / daily["portfolio_nav"].cummax() - 1.0

    positions = pd.DataFrame(positions_rows).sort_values(["rebalance_date", "side", "ticker"]).reset_index(drop=True)

    if save:
        daily.to_csv(_BACKTEST_DAILY_PATH, index=False)
        positions.to_csv(_BACKTEST_POSITIONS_PATH, index=False)
        logger.info("Saved backtest daily -> %s", _BACKTEST_DAILY_PATH)
        logger.info("Saved positions -> %s", _BACKTEST_POSITIONS_PATH)

    return daily, positions
