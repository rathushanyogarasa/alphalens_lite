"""src/evaluation.py - Lightweight performance analytics for AlphaLens Lite."""

import logging
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from src.utils import validate_columns

logger = logging.getLogger(__name__)

_EVAL_METRICS_PATH = config.METRICS_DIR / "performance_metrics.csv"
_IC_METRICS_PATH = config.METRICS_DIR / "ic_metrics.csv"
_EQUITY_PLOT_PATH = config.PLOTS_DIR / "equity_curve.png"


_REQUIRED_DAILY_COLS = {"date", "portfolio_ret", "benchmark_ret", "portfolio_nav", "drawdown"}


def compute_performance_metrics(daily: pd.DataFrame, risk_free_rate: float = config.RISK_FREE_RATE) -> dict[str, float]:
    """Compute core strategy and benchmark metrics."""
    validate_columns(daily, _REQUIRED_DAILY_COLS, "daily")

    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    port = df["portfolio_ret"].astype(float)
    bench = df["benchmark_ret"].astype(float)

    n = len(df)
    if n == 0:
        raise ValueError("No rows in backtest daily DataFrame.")

    ann_factor = 252.0 / n

    ann_return = float((1.0 + port).prod() ** ann_factor - 1.0)
    ann_vol = float(port.std(ddof=0) * math.sqrt(252.0))

    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_mean = float((port - daily_rf).mean())
    sharpe = float((excess_mean / port.std(ddof=0)) * math.sqrt(252.0)) if port.std(ddof=0) > 0 else 0.0

    benchmark_ann_return = float((1.0 + bench).prod() ** ann_factor - 1.0)
    benchmark_ann_vol = float(bench.std(ddof=0) * math.sqrt(252.0))

    max_drawdown = float(df["drawdown"].min())
    hit_rate = float((port > 0).mean())
    avg_turnover = float(df["turnover"].fillna(0.0).mean()) if "turnover" in df.columns else 0.0

    return {
        "n_days": float(n),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
        "avg_turnover": avg_turnover,
        "benchmark_ann_return": benchmark_ann_return,
        "benchmark_ann_vol": benchmark_ann_vol,
    }


def compute_information_coefficient(alpha_df: pd.DataFrame, market_panel: pd.DataFrame) -> dict[str, float]:
    """Compute daily cross-sectional rank IC between alpha_score and fwd_ret_1d."""
    required_alpha = {"date", "ticker", "alpha_score"}
    required_panel = {"date", "ticker", "fwd_ret_1d"}
    validate_columns(alpha_df, required_alpha, "alpha_df")
    validate_columns(market_panel, required_panel, "market_panel")

    alpha = alpha_df[["date", "ticker", "alpha_score"]].copy()
    panel = market_panel[["date", "ticker", "fwd_ret_1d"]].copy()

    alpha["date"] = pd.to_datetime(alpha["date"]).dt.normalize()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    alpha["ticker"] = alpha["ticker"].astype(str).str.upper().str.strip()
    panel["ticker"] = panel["ticker"].astype(str).str.upper().str.strip()

    merged = alpha.merge(panel, on=["date", "ticker"], how="inner").dropna(subset=["alpha_score", "fwd_ret_1d"])
    if merged.empty:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_tstat": 0.0, "ic_n": 0.0}

    daily_ic = []
    for _, g in merged.groupby("date"):
        if len(g) < 4:
            continue
        rank_alpha = g["alpha_score"].rank(method="average")
        rank_fwd = g["fwd_ret_1d"].rank(method="average")
        ic = rank_alpha.corr(rank_fwd)
        if pd.notna(ic):
            daily_ic.append(float(ic))

    if not daily_ic:
        return {"ic_mean": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_tstat": 0.0, "ic_n": 0.0}

    arr = np.array(daily_ic, dtype=float)
    ic_mean = float(arr.mean())
    ic_std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ic_ir = float(ic_mean / ic_std) if ic_std > 0 else 0.0
    ic_tstat = float(ic_mean / (ic_std / math.sqrt(len(arr)))) if ic_std > 0 and len(arr) > 1 else 0.0

    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_ir": ic_ir,
        "ic_tstat": ic_tstat,
        "ic_n": float(len(arr)),
    }


def plot_equity_curve(daily: pd.DataFrame, save_path=config.PLOTS_DIR / "equity_curve.png") -> None:
    """Save portfolio vs benchmark equity curve plot."""
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    plt.figure(figsize=(10, 4.5))
    plt.plot(df["date"], df["portfolio_nav"], label="AlphaLens Lite", linewidth=2)
    plt.plot(df["date"], df["benchmark_nav"], label="Equal-Weight Benchmark", linewidth=1.5)
    plt.title("Portfolio Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_evaluation(
    daily: pd.DataFrame,
    alpha_df: pd.DataFrame | None = None,
    market_panel: pd.DataFrame | None = None,
    save: bool = True,
) -> dict[str, float]:
    """Run performance metrics (+ optional IC metrics) and save outputs."""
    metrics = compute_performance_metrics(daily)

    if alpha_df is not None and market_panel is not None:
        metrics.update(compute_information_coefficient(alpha_df=alpha_df, market_panel=market_panel))

    plot_equity_curve(daily, save_path=_EQUITY_PLOT_PATH)

    if save:
        pd.DataFrame([metrics]).to_csv(_EVAL_METRICS_PATH, index=False)
        logger.info("Saved performance metrics -> %s", _EVAL_METRICS_PATH)
        if "ic_mean" in metrics:
            pd.DataFrame([
                {
                    "ic_mean": metrics["ic_mean"],
                    "ic_std": metrics["ic_std"],
                    "ic_ir": metrics["ic_ir"],
                    "ic_tstat": metrics["ic_tstat"],
                    "ic_n": metrics["ic_n"],
                }
            ]).to_csv(_IC_METRICS_PATH, index=False)
            logger.info("Saved IC metrics -> %s", _IC_METRICS_PATH)

    return metrics
