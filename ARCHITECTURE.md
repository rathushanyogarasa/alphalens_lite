# AlphaLens Lite Architecture Plan

## 1. Scope
AlphaLens Lite is a compact, reproducible equity alpha research pipeline for an undergraduate deep learning project.

Primary objective:
- Build and evaluate a news-sentiment-enhanced long/short strategy on a fixed US large-cap universe.

Design constraints:
- Minimal code surface, notebook-friendly, deterministic runs.
- Clear separation between data, modeling, factor construction, backtesting, and evaluation.
- Fast enough to run a lite demonstration on CPU (with optional GPU acceleration for model training).

## 2. Current Status
Already implemented:
- `config.py`: global constants, universe, training/backtest hyperparameters, directories.
- `src/model.py`: FinBERT classifier wrapper + inference API + save/load.
- `src/data_prep.py`: FinancialPhraseBank + FiQA loading, merge, split, persistence.
- `src/train.py`: fine-tuning loop, validation F1 checkpointing, training curves.
- `src/gdelt_fetcher.py`: GDELT headline collection for ticker universe.

Missing core modules for end-to-end AlphaLens Lite:
- Price/returns data loader.
- Factor engineering and factor fusion.
- Portfolio construction and backtest engine.
- Performance analytics and report generation.
- Notebook pipeline that stitches all phases.

## 3. Target System Architecture
```text
                    +----------------------+
                    |      config.py       |
                    | (constants + paths)  |
                    +----------+-----------+
                               |
                               v
+------------------+   +------------------+   +------------------+
| src/data_prep.py |   | src/model.py     |   | src/train.py     |
| sentiment train  +---> FinBERT model    +---> fine-tune + save |
| data processing  |   | + predict API    |   | best checkpoint  |
+------------------+   +------------------+   +------------------+
                               |
                               v
                      +--------------------+
                      | src/gdelt_fetcher  |
                      | headline ingest     |
                      +----------+---------+
                                 |
                                 v
                   +-----------------------------+
                   | src/sentiment_factor.py     |
                   | headline -> daily signal    |
                   +-------------+---------------+
                                 |
          +----------------------+----------------------+
          |                                             |
          v                                             v
+-------------------------+                 +-------------------------+
| src/market_data.py      |                 | src/technical_factors.py |
| OHLCV + returns panel   |                 | momentum + volatility    |
+------------+------------+                 +------------+------------+
             |                                           |
             +-------------------+-----------------------+
                                 v
                     +--------------------------+
                     | src/factor_engine.py     |
                     | normalize + blend score  |
                     +------------+-------------+
                                  |
                                  v
                     +--------------------------+
                     | src/backtest.py          |
                     | rebalance + costs + PnL  |
                     +------------+-------------+
                                  |
                                  v
                     +--------------------------+
                     | src/evaluation.py        |
                     | IC, Sharpe, DD, plots    |
                     +--------------------------+
```

## 4. Proposed Module Contracts
1. `src/market_data.py`
- Responsibility: fetch/load daily adjusted close (and optional OHLCV), align to trading calendar.
- Input: `tickers`, `start_date`, `end_date`.
- Output: DataFrame indexed by `date`, columns as tickers (prices + returns).

2. `src/sentiment_factor.py`
- Responsibility: run trained FinBERT on headlines, aggregate headline scores into daily ticker sentiment.
- Input: GDELT headlines (`date`, `ticker`, `headline`) and trained model checkpoint.
- Output: DataFrame with `date`, `ticker`, `sentiment_raw`, `sentiment_z`.

3. `src/technical_factors.py`
- Responsibility: compute momentum and volatility factors from price returns.
- Output schema: `date`, `ticker`, `momentum_z`, `volatility_z`.

4. `src/factor_engine.py`
- Responsibility: merge factors, cross-section normalize, apply `FACTOR_WEIGHTS`, produce `alpha_score`.
- Output schema: `date`, `ticker`, `alpha_score`.

5. `src/backtest.py`
- Responsibility: quantile long/short selection, holding-period rebalancing, transaction/slippage costs, equity curve.
- Outputs: daily portfolio returns, cumulative NAV, turnover series, positions matrix.

6. `src/evaluation.py`
- Responsibility: metrics and diagnostics: annualized return/volatility, Sharpe, max drawdown, win rate, optional IC/RankIC.

## 5. Data Contracts (Canonical Schema)
Use these canonical columns consistently:
- Headlines: `date`, `ticker`, `headline`, `source`
- Factors: `date`, `ticker`, `<factor_name>_z`
- Combined alpha panel: `date`, `ticker`, `alpha_score`
- Returns panel: `date`, `ticker`, `ret_1d`, `fwd_ret_1d` (or matrix equivalent)
- Backtest output: `date`, `portfolio_ret`, `nav`, `turnover`

Conventions:
- `date` is timezone-naive pandas `datetime64[ns]` normalized to date boundary.
- `ticker` uppercased, validated against `config.TICKERS`.
- One row per (`date`, `ticker`) in long-form factor tables.

## 6. Notebook Flow (Lite)
1. `notebooks/01_train_finbert.ipynb`
- Run `run_data_prep()`, train model, save checkpoint/curves.

2. `notebooks/02_build_sentiment_signal.ipynb`
- Pull GDELT headlines, infer sentiment, aggregate daily sentiment factor.

3. `notebooks/03_construct_alpha.ipynb`
- Load market data, compute momentum/volatility, blend with sentiment.

4. `notebooks/04_backtest.ipynb`
- Run portfolio simulation with costs/slippage.

5. `notebooks/05_evaluate.ipynb`
- Summarize metrics, generate plots and short conclusions for report.

## 7. Engineering Standards
- Keep all tunables in `config.py`.
- Pure transformation functions where possible; isolate IO in orchestrators.
- Deterministic seed usage for any randomized step.
- Save intermediate artifacts under `data/processed` and metrics under `results/metrics`.
- Each module exposes one high-level `run_*` function for notebook usage.

## 8. Immediate Build Order
Phase 1 (data + factors):
1. Implement `src/market_data.py` with a stable provider interface and cache to `data/raw`.
2. Implement `src/sentiment_factor.py` for headline scoring + daily aggregation.
3. Implement `src/technical_factors.py` for momentum/volatility features.

Phase 2 (portfolio + evaluation):
4. Implement `src/factor_engine.py` for factor merge and weighted alpha score.
5. Implement `src/backtest.py` for execution logic and net returns.
6. Implement `src/evaluation.py` for metrics and visual diagnostics.

Phase 3 (project packaging):
7. Add `requirements.txt` and `README.md` with run instructions.
8. Create minimal notebook skeletons wired to `run_*` entry points.

## 9. Risks and Mitigations
- Data provider/API instability: local CSV cache + fallback path.
- Long FinBERT training time: checkpoint reuse and smaller demo subset.
- Sparse headline coverage: neutral fill and confidence thresholding.
- Backtest overfitting: separate tuning and final evaluation windows.

## 10. Definition of Done (Lite)
AlphaLens Lite is complete when:
1. Running notebooks 01-05 produces a full reproducible pipeline.
2. Final outputs include: trained model checkpoint, alpha factor series, equity curve, metrics in `results/metrics`.
3. A short report-ready summary can be generated from notebook outputs.
