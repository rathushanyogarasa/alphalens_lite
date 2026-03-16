import pytest

from src.backtest import run_backtest
from src.factor_engine import build_alpha_factors


def test_backtest_weight_neutrality_and_timing(make_synthetic_inputs):
    technical, sentiment, market_panel = make_synthetic_inputs
    alpha = build_alpha_factors(technical, sentiment)

    daily, positions = run_backtest(alpha_df=alpha, market_panel=market_panel, hold_days=3, save=False)

    assert (daily["date"] > daily["rebalance_date"]).all()

    net_by_reb = positions.groupby("rebalance_date")["weight"].sum().abs().max()
    assert net_by_reb < 1e-10

    assert len(daily) > 0
    assert daily["portfolio_ret"].notna().all()


def test_backtest_requires_coverage(make_synthetic_inputs):
    technical, sentiment, market_panel = make_synthetic_inputs
    alpha = build_alpha_factors(technical, sentiment)

    empty_panel = market_panel.iloc[0:0].copy()
    with pytest.raises(RuntimeError):
        run_backtest(alpha_df=alpha, market_panel=empty_panel, save=False)
