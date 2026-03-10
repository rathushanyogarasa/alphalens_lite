"""src/factor_engine.py - Lightweight multi-factor fusion for AlphaLens Lite."""

from __future__ import annotations

import logging

import pandas as pd

import config

logger = logging.getLogger(__name__)

_ALPHA_FACTOR_PATH = config.PROCESSED_DIR / "alpha_factors.csv"
_SENTIMENT_PATH = config.PROCESSED_DIR / "sentiment_factor.csv"
_TECHNICAL_PATH = config.PROCESSED_DIR / "technical_factors.csv"


_REQUIRED_TECH_COLS = {"date", "ticker", "momentum_z", "volatility_z"}
_REQUIRED_SENT_COLS = {"date", "ticker", "sentiment_z"}


def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _load_default_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _TECHNICAL_PATH.exists():
        raise FileNotFoundError(f"Missing technical factors: {_TECHNICAL_PATH}")
    if not _SENTIMENT_PATH.exists():
        raise FileNotFoundError(f"Missing sentiment factor: {_SENTIMENT_PATH}")

    technical = pd.read_csv(_TECHNICAL_PATH)
    sentiment = pd.read_csv(_SENTIMENT_PATH)
    return technical, sentiment


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = float(sum(weights.values()))
    if total <= 0:
        raise ValueError("Factor weights must sum to a positive value.")
    return {k: float(v) / total for k, v in weights.items()}


def build_alpha_factors(
    technical_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Merge sentiment + technical factors and compute a composite alpha score."""
    _validate_columns(technical_df, _REQUIRED_TECH_COLS, "technical_df")
    _validate_columns(sentiment_df, _REQUIRED_SENT_COLS, "sentiment_df")

    w = _normalize_weights(weights or config.FACTOR_WEIGHTS)

    tech = technical_df.copy()
    sent = sentiment_df[["date", "ticker", "sentiment_z"]].copy()

    tech["date"] = pd.to_datetime(tech["date"]).dt.normalize()
    tech["ticker"] = tech["ticker"].astype(str).str.upper().str.strip()
    sent["date"] = pd.to_datetime(sent["date"]).dt.normalize()
    sent["ticker"] = sent["ticker"].astype(str).str.upper().str.strip()

    panel = tech.merge(sent, on=["date", "ticker"], how="left")
    panel["sentiment_z"] = panel["sentiment_z"].fillna(0.0)

    panel["alpha_score"] = (
        w.get("sentiment", 0.0) * panel["sentiment_z"]
        + w.get("momentum", 0.0) * panel["momentum_z"]
        + w.get("volatility", 0.0) * panel["volatility_z"]
    )

    out_cols = [
        "date",
        "ticker",
        "sentiment_z",
        "momentum_z",
        "volatility_z",
        "alpha_score",
    ]
    return panel[out_cols].sort_values(["date", "ticker"]).reset_index(drop=True)


def run_factor_engine(
    technical_df: pd.DataFrame | None = None,
    sentiment_df: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Orchestrator for alpha factor construction."""
    if technical_df is None or sentiment_df is None:
        technical_df, sentiment_df = _load_default_inputs()

    alpha = build_alpha_factors(technical_df=technical_df, sentiment_df=sentiment_df, weights=weights)

    if save:
        alpha.to_csv(_ALPHA_FACTOR_PATH, index=False)
        logger.info("Saved alpha factors -> %s", _ALPHA_FACTOR_PATH)

    return alpha
