"""src/gdelt_fetcher.py — GDELT 2.0 news connector.

Fetches financial headlines from the GDELT Document API.
  - No API key required
  - Updated every 15 minutes
  - Up to 250 articles per ticker per query
  - History up to ~2 years

Rate limit: ~1 request per 5-10 seconds (use GDELT_SLEEP_SECS >= 6).

Usage:
    from src.gdelt_fetcher import fetch_gdelt
    df = fetch_gdelt(config.TICKERS, lookback_days=365)
    # -> DataFrame with columns: date, ticker, headline, source
"""

import logging
import time

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Company names give better GDELT query recall than raw ticker symbols
_COMPANY_MAP: dict[str, str] = {
    "AAPL": "Apple",           "MSFT": "Microsoft",       "NVDA": "NVIDIA",
    "JPM":  "JPMorgan",        "GS":   "Goldman Sachs",    "BAC":  "Bank of America",
    "UNH":  "UnitedHealth",    "JNJ":  "Johnson Johnson",  "LLY":  "Eli Lilly",
    "PG":   "Procter Gamble",  "KO":   "Coca-Cola",        "WMT":  "Walmart",
    "AMZN": "Amazon",          "TSLA": "Tesla",             "NKE":  "Nike",
    "XOM":  "ExxonMobil",      "CVX":  "Chevron",           "COP":  "ConocoPhillips",
    "CAT":  "Caterpillar",     "HON":  "Honeywell",         "BA":   "Boeing",
    "GOOGL":"Alphabet",        "META": "Meta Platforms",    "NFLX": "Netflix",
    "LIN":  "Linde",           "SHW":  "Sherwin-Williams",  "FCX":  "Freeport McMoRan",
    "NEE":  "NextEra Energy",  "DUK":  "Duke Energy",       "SO":   "Southern Company",
}


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "ticker", "headline", "source"])


def _fetch_one(ticker: str, timespan: str, max_records: int) -> list[dict]:
    """Fetch up to max_records articles for a single ticker."""
    query = _COMPANY_MAP.get(ticker, ticker)
    params = {
        "query":      query,
        "mode":       "ArtList",
        "maxrecords": max_records,
        "timespan":   timespan,
        "sourcelang": "english",
        "format":     "json",
    }
    for attempt in range(3):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=25)
            if resp.status_code == 429:
                wait = 15 * (attempt + 1)
                logger.warning("GDELT 429 for %s (attempt %d) — sleeping %ds",
                               ticker, attempt + 1, wait)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                logger.warning("GDELT HTTP %d for %s", resp.status_code, ticker)
                return []
            articles = resp.json().get("articles") or []
            break
        except Exception as exc:
            logger.warning("GDELT request failed for %s: %s", ticker, exc)
            return []
    else:
        return []

    rows = []
    for art in articles:
        title = (art.get("title") or "").strip()
        pub   = art.get("seendate", "")
        if not title or title.lower() == "[removed]":
            continue
        try:
            date = pd.Timestamp(pub, tz="UTC").tz_localize(None)
        except Exception:
            try:
                date = pd.Timestamp(pub[:8])
            except Exception:
                continue
        rows.append({"date": date, "ticker": ticker,
                     "headline": title, "source": "gdelt"})
    return rows


def fetch_gdelt(
    tickers: list[str] | None = None,
    lookback_days: int | None = None,
    max_records: int | None = None,
    sleep_secs: float | None = None,
) -> pd.DataFrame:
    """Fetch GDELT headlines for all tickers and return a unified DataFrame.

    Args:
        tickers:      Tickers to fetch (defaults to config.TICKERS).
        lookback_days: Days of history (defaults to config.GDELT_LOOKBACK_DAYS).
        max_records:  Max articles per ticker (defaults to config.GDELT_MAX_RECORDS).
        sleep_secs:   Seconds between requests (defaults to config.GDELT_SLEEP_SECS).

    Returns:
        DataFrame with columns: date (datetime), ticker, headline, source='gdelt'.
        Sorted by date ascending, duplicates on (ticker, headline) removed.
    """
    tickers      = tickers      or config.TICKERS
    lookback_days= lookback_days or config.GDELT_LOOKBACK_DAYS
    max_records  = min(max_records or config.GDELT_MAX_RECORDS, 250)
    sleep_secs   = sleep_secs   if sleep_secs is not None else config.GDELT_SLEEP_SECS

    if lookback_days <= 30:
        timespan = f"{lookback_days}d"
    else:
        timespan = f"{max(1, round(lookback_days / 30))}months"

    print(f"  Fetching GDELT | {len(tickers)} tickers | timespan={timespan} | "
          f"max={max_records}/ticker | ~{len(tickers)*sleep_secs/60:.0f} min")

    all_rows: list[dict] = []
    for i, ticker in enumerate(tickers):
        rows = _fetch_one(ticker, timespan, max_records)
        all_rows.extend(rows)
        print(f"  [{i+1:3d}/{len(tickers)}] {ticker:<6}  {len(rows):>3} articles  "
              f"(total: {len(all_rows)})", end="\r")
        time.sleep(sleep_secs)

    print()  # newline after progress line
    if not all_rows:
        logger.warning("GDELT returned no articles")
        return _empty_df()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = (df.dropna(subset=["date", "headline"])
            .sort_values("date")
            .drop_duplicates(subset=["ticker", "headline"])
            .reset_index(drop=True))

    print(f"  GDELT done: {len(df):,} articles | "
          f"{df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")
    return df
