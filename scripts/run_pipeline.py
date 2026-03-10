"""Run AlphaLens Lite end-to-end without notebooks.

Examples:
  python scripts/run_pipeline.py --lite
  python scripts/run_pipeline.py --full --skip-sentiment
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd

# Ensure imports work when executed as python scripts/run_pipeline.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.backtest import run_backtest
from src.evaluation import run_evaluation
from src.factor_engine import run_factor_engine
from src.market_data import run_market_data
from src.sentiment_factor import run_sentiment_factor
from src.technical_factors import run_technical_factors

logger = logging.getLogger("alphalens_lite.pipeline")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run AlphaLens Lite pipeline")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--lite", action="store_true", help="Use smaller/faster settings")
    mode.add_argument("--full", action="store_true", help="Use default full settings")
    p.add_argument("--skip-sentiment", action="store_true", help="Use cached sentiment_factor.csv")
    p.add_argument("--refresh-prices", action="store_true", help="Force refresh market download")
    p.add_argument("--tickers", nargs="*", help="Optional explicit ticker override")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def _load_cached_sentiment() -> pd.DataFrame:
    path = config.PROCESSED_DIR / "sentiment_factor.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run notebook 02 or rerun pipeline without --skip-sentiment."
        )
    return pd.read_csv(path)


def _write_summary(metrics: dict[str, float], path: Path) -> None:
    lines = ["# AlphaLens Lite Run Summary", ""]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- {k}: {v:.6f}")
        else:
            lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)

    is_lite = args.lite or not args.full
    tickers = args.tickers or (config.TICKERS[:8] if is_lite else config.TICKERS)

    if is_lite:
        start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        lookback_days = 120
        logger.info("Running in LITE mode | tickers=%d | start=%s | lookback=%d", len(tickers), start_date, lookback_days)
    else:
        start_date = config.MARKET_START_DATE
        lookback_days = config.GDELT_LOOKBACK_DAYS
        logger.info("Running in FULL mode | tickers=%d", len(tickers))

    prices, market_panel = run_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=config.MARKET_END_DATE,
        refresh=args.refresh_prices,
        save=True,
    )
    logger.info("Market data ready | prices=%s | panel=%s", prices.shape, market_panel.shape)

    technical = run_technical_factors(prices=prices, save=True)
    logger.info("Technical factors ready | shape=%s", technical.shape)

    if args.skip_sentiment:
        sentiment = _load_cached_sentiment()
        logger.info("Using cached sentiment factor | shape=%s", sentiment.shape)
    else:
        try:
            _, sentiment = run_sentiment_factor(
                tickers=tickers,
                lookback_days=lookback_days,
                save=True,
            )
            logger.info("Sentiment factors ready | shape=%s", sentiment.shape)
        except Exception as exc:
            logger.warning("Sentiment build failed (%s); trying cached sentiment_factor.csv", exc)
            sentiment = _load_cached_sentiment()

    alpha = run_factor_engine(technical_df=technical, sentiment_df=sentiment, save=True)
    logger.info("Alpha factors ready | shape=%s", alpha.shape)

    daily, positions = run_backtest(alpha_df=alpha, market_panel=market_panel, save=True)
    logger.info("Backtest complete | daily=%s | positions=%s", daily.shape, positions.shape)

    metrics = run_evaluation(daily=daily, alpha_df=alpha, market_panel=market_panel, save=True)
    logger.info("Evaluation complete | metrics=%s", metrics)

    summary_path = config.METRICS_DIR / "run_summary.md"
    _write_summary(metrics, summary_path)
    logger.info("Summary written -> %s", summary_path)


if __name__ == "__main__":
    main()

