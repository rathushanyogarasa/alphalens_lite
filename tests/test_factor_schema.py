from src.factor_engine import build_alpha_factors


def test_alpha_factor_schema(make_synthetic_inputs):
    technical, sentiment, _ = make_synthetic_inputs
    alpha = build_alpha_factors(technical, sentiment)

    expected_cols = {
        "date",
        "ticker",
        "sentiment_z",
        "momentum_z",
        "volatility_z",
        "alpha_score",
    }
    assert expected_cols.issubset(set(alpha.columns))
    assert not alpha.duplicated(["date", "ticker"]).any()
    assert alpha["alpha_score"].notna().all()
