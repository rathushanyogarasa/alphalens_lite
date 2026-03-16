"""src/sentiment_factor.py - Headline scoring and daily sentiment factor."""

import logging

import pandas as pd

import config
from src.gdelt_fetcher import fetch_gdelt
from src.model import FinBERTClassifier, SIGNAL_MAP
from src.utils import cross_section_zscore

logger = logging.getLogger(__name__)

_SENTIMENT_HEADLINES = config.PROCESSED_DIR / "headlines_scored.csv"
_SENTIMENT_FACTOR = config.PROCESSED_DIR / "sentiment_factor.csv"


def _ensure_columns(df: pd.DataFrame) -> None:
    required = {"date", "ticker", "headline"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required headline columns: {sorted(missing)}")


def _row_score(pred: dict) -> float:
    probs = pred.get("probabilities", {})
    p_pos = float(probs.get("positive", 0.0))
    p_neg = float(probs.get("negative", 0.0))
    return p_neg - p_pos  # contrarian direction


def score_headlines(headlines: pd.DataFrame, model: FinBERTClassifier | None = None) -> pd.DataFrame:
    """Run FinBERT inference over headlines."""
    _ensure_columns(headlines)
    if model is None:
        model = FinBERTClassifier.load(config.MODEL_DIR)

    df = headlines.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df.dropna(subset=["date"]) \
        .loc[df["headline"] != ""] \
        .reset_index(drop=True)

    preds = model.predict(df["headline"].tolist())
    scored = pd.DataFrame(preds)

    df["label_name"] = scored["label_name"].values
    df["confidence"] = scored["confidence"].values
    df["signal"] = df["label_name"].map(SIGNAL_MAP).fillna(0).astype(float)
    df["sentiment_raw"] = [
        _row_score(pred) for pred in preds
    ]

    return df[["date", "ticker", "headline", "label_name", "confidence", "signal", "sentiment_raw"]]


def aggregate_daily_sentiment(scored_headlines: pd.DataFrame) -> pd.DataFrame:
    """Aggregate article-level sentiment to daily ticker factor."""
    daily = (
        scored_headlines.groupby(["date", "ticker"], as_index=False)
        .agg(
            sentiment_raw=("sentiment_raw", "mean"),
            sentiment_signal=("signal", "mean"),
            article_count=("headline", "count"),
            mean_confidence=("confidence", "mean"),
        )
    )
    daily = cross_section_zscore(daily, "sentiment_raw", "sentiment_z")
    return daily.sort_values(["date", "ticker"]).reset_index(drop=True)


def run_sentiment_factor(
    headlines: pd.DataFrame | None = None,
    tickers: list[str] | None = None,
    lookback_days: int | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full sentiment pipeline: fetch headlines, score, aggregate.

    Returns:
        scored_headlines, daily_sentiment_factor
    """
    if headlines is None:
        headlines = fetch_gdelt(
            tickers=tickers or config.TICKERS,
            lookback_days=lookback_days or config.GDELT_LOOKBACK_DAYS,
        )

    scored = score_headlines(headlines=headlines)
    daily = aggregate_daily_sentiment(scored)

    if save:
        scored.to_csv(_SENTIMENT_HEADLINES, index=False)
        daily.to_csv(_SENTIMENT_FACTOR, index=False)
        logger.info("Saved scored headlines -> %s", _SENTIMENT_HEADLINES)
        logger.info("Saved sentiment factor -> %s", _SENTIMENT_FACTOR)

    return scored, daily
