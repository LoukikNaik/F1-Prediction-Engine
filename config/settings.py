"""Central configuration for the F1 Prediction Engine."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
KAGGLE_DIR = RAW_DIR / "kaggle"
FASTF1_CACHE_DIR = RAW_DIR / "fastf1_cache"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
PREDICTIONS_DIR = PROCESSED_DIR / "predictions"
DB_PATH = DATA_DIR / "f1_engine.db"
MODELS_DIR = PROJECT_ROOT / "models" / "trained"

# Database
DB_URL = f"sqlite:///{DB_PATH}"

# Current season
CURRENT_SEASON = 2026

# Data source URLs
OPENF1_BASE_URL = "https://api.openf1.org/v1"
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"

# FastF1 settings
FASTF1_CACHE_ENABLED = True

# Era weights for model training
ERA_WEIGHTS = {
    (2022, 2026): 1.0,    # Current ground-effect era
    (2017, 2021): 0.6,    # Recent hybrid
    (2014, 2016): 0.3,    # Early hybrid
    (2006, 2013): 0.15,   # V8 era
    (1950, 2005): 0.05,   # Older eras
}

# ML settings
MONTE_CARLO_SIMULATIONS = 10_000
OPTUNA_TRIALS = 100
ENSEMBLE_MODELS = ["lightgbm", "xgboost", "logistic_regression"]
RANDOM_SEED = 42

# Feature engineering
ROLLING_WINDOWS = [3, 5, 10]
ELO_K_FACTOR = 32
ELO_INITIAL_RATING = 1500

# Scheduler
PIPELINE_CRON_DAY = "thu"
PIPELINE_CRON_HOUR = 18
PIPELINE_TIMEZONE = "UTC"

# Live prediction settings
LIVE_POLL_INTERVAL = 90
LIVE_MC_SIMULATIONS = 5_000
LIVE_DASHBOARD_REFRESH = 30
LIVE_PREDICTIONS_DIR = PREDICTIONS_DIR / "live"

# Backfill settings
BACKFILL_DEFAULT_LAP_INTERVAL = 3

# Live data source: "openf1" (needs paid auth), "scraper" (web scraping, free)
LIVE_DATA_SOURCE = "scraper"
# Optional Firecrawl API for JS-rendered pages (self-hosted or cloud)
FIRECRAWL_API_URL = None  # e.g. "http://localhost:3002" or "https://api.firecrawl.dev"
FIRECRAWL_API_KEY = None  # Required for cloud, not for self-hosted

# Claude API key for LLM-based HTML parsing fallback (uses ANTHROPIC_API_KEY env var if unset)
CLAUDE_API_KEY = None
