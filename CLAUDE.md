# F1 Prediction Engine

## Overview
ML-powered F1 race prediction engine. Generates 20x20 probability matrices (driver × finishing position) using an ensemble of LightGBM, XGBoost, Logistic Regression + Monte Carlo simulations.

## Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Key Commands
- `make pipeline` — Run full prediction pipeline for next race
- `make dashboard` — Launch Streamlit dashboard
- `make train` — Train ML models on historical data
- `make test` — Run pytest suite
- `make lint` — Run ruff check + format
- `make fetch-kaggle` — Seed database from Kaggle CSVs

## Architecture
- **Database**: SQLite via SQLAlchemy (`data/f1_engine.db`)
- **Dashboard**: Streamlit multi-page app (`dashboard/`)
- **ML Models**: LightGBM + XGBoost + LogReg ensemble, Monte Carlo sims
- **Data Sources**: Kaggle (historical), FastF1 + OpenF1 (current season), Open-Meteo (weather)
- **Feature Store**: Parquet files in `data/processed/features/`

## Project Structure
- `src/pipeline/collectors/` — Data fetching from APIs and Kaggle
- `src/pipeline/processors/` — Cleaning, feature engineering, aggregation
- `src/models/` — ML training, prediction, ELO, Monte Carlo, SHAP
- `src/database/` — SQLAlchemy ORM, queries, migrations
- `src/utils/` — Constants, logging, validators, F1 calendar
- `dashboard/` — Streamlit app with 5 pages
- `config/` — Settings, circuit metadata, race schedule

## Conventions
- Python 3.14, type hints on all functions
- Google-style docstrings for public functions
- snake_case for functions/variables, PascalCase for classes
- Use loguru for logging, pathlib for paths
- Parquet for intermediate data, not CSV
- SQLAlchemy Session context managers for DB access
