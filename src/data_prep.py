"""src/data_prep.py — Training dataset preparation.

Loads FinancialPhraseBank + FiQA from HuggingFace, merges them,
performs a stratified 70/15/15 train/val/test split, and saves CSVs
to data/processed/ for use in Notebook 01.

If HuggingFace is offline, a small synthetic fallback is used so the
notebook still runs end-to-end (you'll just train on a toy dataset).
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

LABEL_MAP    = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_NAME  = {v: k for k, v in LABEL_MAP.items()}


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_phrasebank() -> pd.DataFrame:
    """Load FinancialPhraseBank from HuggingFace (nickmuchi/financial-classification).

    Returns DataFrame with columns: text, label (int), label_name, source.
    Falls back to 300-row synthetic set if HuggingFace is unavailable.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        ds = load_dataset("nickmuchi/financial-classification")
        for split in ds.values():
            for item in split:
                lbl = int(item["labels"])
                rows.append({"text": str(item["text"]).strip(), "label": lbl,
                             "label_name": INT_TO_NAME[lbl], "source": "phrasebank"})
        logger.info("PhraseBank loaded: %d rows", len(rows))
    except Exception as exc:
        logger.warning("PhraseBank unavailable (%s) -- using synthetic fallback", exc)
        seeds = [
            ("Company beat earnings estimates and raised full-year guidance", 2),
            ("Results were broadly in line with analyst expectations", 1),
            ("Firm missed revenue targets and issued a profit warning", 0),
        ]
        for i in range(300):
            text, lbl = seeds[i % 3]
            rows.append({"text": f"{text} [{i}]", "label": lbl,
                         "label_name": INT_TO_NAME[lbl], "source": "phrasebank"})

    df = pd.DataFrame(rows)
    return df[df["text"] != ""].reset_index(drop=True)


def load_fiqa() -> pd.DataFrame:
    """Load FiQA-2018 sentiment from HuggingFace (pauri32/fiqa-2018).

    Continuous scores are discretised: <-0.1 = negative, >0.1 = positive.
    Returns DataFrame with columns: text, label (int), label_name, source.
    """
    from datasets import load_dataset

    rows: list[dict] = []
    try:
        ds = load_dataset("pauri32/fiqa-2018", trust_remote_code=True)
        for split in ds.values():
            for item in split:
                text  = item.get("sentence") or item.get("text") or ""
                score = float(item.get("sentiment_score", item.get("score", 0.0)))
                lbl   = 0 if score < -0.1 else (2 if score > 0.1 else 1)
                rows.append({"text": str(text).strip(), "label": lbl,
                             "label_name": INT_TO_NAME[lbl], "source": "fiqa"})
        logger.info("FiQA loaded: %d rows", len(rows))
    except Exception as exc:
        logger.warning("FiQA unavailable (%s) -- using synthetic fallback", exc)
        seeds = [
            ("Stock plunged after weak quarterly guidance", 0),
            ("Management reiterated full-year targets unchanged", 1),
            ("Profit margins expanded significantly beyond expectations", 2),
        ]
        for i in range(300):
            text, lbl = seeds[i % 3]
            rows.append({"text": f"{text} [{i}]", "label": lbl,
                         "label_name": INT_TO_NAME[lbl], "source": "fiqa"})

    df = pd.DataFrame(rows)
    return df[df["text"] != ""].reset_index(drop=True)


# ── Merge & split ────────────────────────────────────────────────────────────

def merge_datasets(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate, deduplicate on text, and shuffle."""
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset="text")
    combined = combined.sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
    logger.info("Merged: %d rows | %s",
                len(combined), combined["label_name"].value_counts().to_dict())
    return combined


def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 train/val/test split."""
    val_test = config.VAL_SPLIT + config.TEST_SPLIT
    train_df, vt = train_test_split(df, test_size=val_test,
                                    stratify=df["label"], random_state=config.RANDOM_SEED)
    val_df, test_df = train_test_split(vt, test_size=config.TEST_SPLIT / val_test,
                                       stratify=vt["label"], random_state=config.RANDOM_SEED)
    logger.info("Split: train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ── Persistence ──────────────────────────────────────────────────────────────

def save_splits(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    """Write train/val/test CSVs to data/processed/."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        path = config.PROCESSED_DIR / f"{name}.csv"
        df.to_csv(path, index=False)
        logger.info("Saved %s.csv (%d rows) -> %s", name, len(df), path)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load previously saved train/val/test CSVs."""
    dfs = []
    for name in ("train", "val", "test"):
        path = config.PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run run_data_prep() first.")
        dfs.append(pd.read_csv(path))
    return tuple(dfs)


# ── Orchestrator ─────────────────────────────────────────────────────────────

def run_data_prep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full pipeline: load -> merge -> split -> save.

    Called from Notebook 01.  Returns (train_df, val_df, test_df).
    """
    phrasebank = load_phrasebank()
    fiqa       = load_fiqa()
    merged     = merge_datasets([phrasebank, fiqa])
    train, val, test = split_dataset(merged)
    save_splits(train, val, test)
    return train, val, test
