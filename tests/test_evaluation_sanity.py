from __future__ import annotations

import math

from src.backtest import run_backtest
from src.evaluation import compute_performance_metrics, run_evaluation
from src.factor_engine import build_alpha_factors


def test_evaluation_metrics_are_finite(make_synthetic_inputs):
    technical, sentiment, market_panel = make_synthetic_inputs
    alpha = build_alpha_factors(technical, sentiment)
    daily, _ = run_backtest(alpha_df=alpha, market_panel=market_panel, hold_days=2, save=False)

    metrics = compute_performance_metrics(daily)
    for key, value in metrics.items():
        assert math.isfinite(float(value)), f"Metric {key} is not finite: {value}"


def test_run_evaluation_includes_ic(make_synthetic_inputs):
    technical, sentiment, market_panel = make_synthetic_inputs
    alpha = build_alpha_factors(technical, sentiment)
    daily, _ = run_backtest(alpha_df=alpha, market_panel=market_panel, hold_days=2, save=False)

    metrics = run_evaluation(daily=daily, alpha_df=alpha, market_panel=market_panel, save=False)
    for key in ("ic_mean", "ic_std", "ic_ir", "ic_tstat", "ic_n"):
        assert key in metrics
