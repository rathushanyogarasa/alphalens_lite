"""config.py — AlphaLens Lite
Single source of truth for all constants.
Change things here, not inside src/ modules.
"""

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Universe (30 tickers: 3 per sector across 10 GICS sectors) ─────────────
TICKERS: list[str] = [
    "AAPL", "MSFT", "NVDA",    # Technology
    "JPM",  "GS",   "BAC",     # Financials
    "UNH",  "JNJ",  "LLY",     # Healthcare
    "PG",   "KO",   "WMT",     # Consumer Staples
    "AMZN", "TSLA", "NKE",     # Consumer Discretionary
    "XOM",  "CVX",  "COP",     # Energy
    "CAT",  "HON",  "BA",      # Industrials
    "GOOGL","META", "NFLX",    # Communication Services
    "LIN",  "SHW",  "FCX",     # Materials
    "NEE",  "DUK",  "SO",      # Utilities
]

# ── FinBERT model ────────────────────────────────────────────────────────────
MODEL_NAME           = "ProsusAI/finbert"
BATCH_SIZE           = 16       # reduce to 8 if you run out of GPU memory
EPOCHS               = 3
LEARNING_RATE        = 2e-5
WEIGHT_DECAY         = 0.01
WARMUP_RATIO         = 0.1      # fraction of total steps used for linear warmup
MAX_LENGTH           = 128      # max tokens per headline
CONFIDENCE_THRESHOLD = 0.70     # below this the model returns "uncertain"

# ── Training dataset splits ──────────────────────────────────────────────────
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15

# ── Champion strategy ────────────────────────────────────────────────────────
# These values were derived via ablation study (Notebook 04).
# Do NOT change here unless you have re-run the ablation.
FACTOR_WEIGHTS: dict[str, float] = {
    "sentiment":  0.30,
    "momentum":   0.50,
    "volatility": 0.20,
}
HOLD_DAYS            = 2        # rebalance every 2 trading days
QUANTILE_CUTOFF      = 0.20     # long top 20%, short bottom 20%
TRANSACTION_COST_BPS = 10.0     # one-way cost in basis points
SLIPPAGE_BPS         = 5.0      # one-way slippage in basis points
RISK_FREE_RATE       = 0.045    # annualised, used for Sharpe calculation

# ── GDELT data fetcher ───────────────────────────────────────────────────────
GDELT_LOOKBACK_DAYS  = 365      # history to fetch per ticker (free, no limit)
GDELT_MAX_RECORDS    = 250      # articles per ticker per query (GDELT hard cap)
GDELT_SLEEP_SECS     = 6.0      # seconds between requests (GDELT rate limit)

# ── Directories (created automatically on import) ────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = BASE_DIR / "results"
MODEL_DIR     = RESULTS_DIR / "model"
PLOTS_DIR     = RESULTS_DIR / "plots"
METRICS_DIR   = RESULTS_DIR / "metrics"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, PLOTS_DIR, METRICS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
