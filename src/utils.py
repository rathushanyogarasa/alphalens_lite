"""src/utils.py - Shared utilities for AlphaLens Lite."""

import pandas as pd


def cross_section_zscore(df: pd.DataFrame, col: str, out_col: str) -> pd.DataFrame:
    stats = df.groupby("date")[col].agg(["mean", "std"]).rename(columns={"mean": "mu", "std": "sigma"})
    out = df.join(stats, on="date")
    out[out_col] = (out[col] - out["mu"]) / out["sigma"].replace(0, pd.NA)
    out[out_col] = out[out_col].fillna(0.0)
    return out.drop(columns=["mu", "sigma"])


def validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
