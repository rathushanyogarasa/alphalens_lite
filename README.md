# AlphaLens Lite

AlphaLens Lite is a compact, reproducible version of AlphaLens for an undergraduate deep-learning project.

## What It Does
- Fine-tunes FinBERT for financial headline sentiment
- Builds daily sentiment, momentum, and volatility factors
- Combines factors into a composite `alpha_score`
- Runs a fast cross-sectional long/short backtest
- Reports core portfolio metrics and rank-IC diagnostics

## Repository Layout
- `config.py`: all constants and paths
- `src/`: pipeline modules
- `notebooks/`: end-to-end workflow notebooks
- `data/processed/`: cached factor/market tables
- `results/metrics/` and `results/plots/`: outputs

## Quick Start
1. Create environment and install dependencies:
   - `python -m venv .venv`
   - Windows: `.\.venv\Scripts\activate`
   - `pip install -r requirements.txt`
2. Run notebooks in order:
   - `notebooks/01_train_finbert.ipynb`
   - `notebooks/02_build_sentiment_signal.ipynb`
   - `notebooks/03_construct_alpha.ipynb`
   - `notebooks/04_backtest.ipynb`
   - `notebooks/05_evaluate.ipynb`

## Runtime Notes
- Use cache-first defaults (`refresh=False`) to avoid repeated data downloads.
- Model training is the most expensive step; reuse `results/model/weights.pt` when possible.
- Backtest/evaluation are designed to run quickly on CPU.
