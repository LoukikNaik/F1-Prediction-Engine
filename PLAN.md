# F1 Race Prediction Engine - Implementation Plan

## Context

Build a complete F1 race prediction engine from scratch. The system will collect historical and current-season data, engineer features, train an ensemble of ML models, generate probability matrices for driver/team finishing positions, and display everything in an interactive web dashboard. An automated pipeline runs before each race weekend.

**User choices:** Python, Kaggle + free APIs for data, ensemble ML, full system design, all available history, maximum features, dashboard + automated pipeline.

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Language | Python 3.11+ | Best ML/data ecosystem |
| Dashboard | Streamlit (multi-page) | Fastest dev cycle, native Plotly, free hosting |
| ML Models | LightGBM + XGBoost + Logistic Regression + Monte Carlo | Ensemble for robust probability estimates |
| Database | SQLite (via SQLAlchemy) | Zero config, right-sized for F1 data |
| Feature Store | Parquet files | Columnar, fast for large feature matrices |
| Scheduling | APScheduler | Pure Python, persistent jobs, cron support |
| CLI | Typer | Modern CLI with auto-generated help |
| Weather | Open-Meteo API | Free, no API key, global coverage |
| F1 Data (historical) | Kaggle datasets | Pre-cleaned CSV, covers 1950-2025 |
| F1 Data (current) | FastF1 + OpenF1 API | Comprehensive, free, no API key needed |
| Explainability | SHAP | Feature importance for dashboard |
| Hyperparameter Tuning | Optuna | Bayesian optimization |

---

## Data Sources

### Historical (Kaggle - load once)
- **Formula 1 Race Data 1950-present** (`jtrotman/formula-1-race-data`) - all-time results
- **Comprehensive F1 Dataset 2020-2025** (`vshreekamalesh/comprehensive-formula-1-dataset-2020-2025`) - detailed recent data
- **F1 Pit Stop Dataset** (`akashrane2609/formula-1-pit-stop-dataset`) - pit stop times

### Current Season (APIs - per race weekend)
- **FastF1** (Python library) - qualifying, practice, race lap times, telemetry (2018+)
- **OpenF1 API** (openf1.org) - real-time position, weather, pit data, radio, no auth needed
- **Jolpica-F1 API** (Ergast successor) - standings, schedule, results (1950+), 200 req/hr
- **Open-Meteo** - weather forecasts by circuit lat/lon, free, no API key

---

## Project Structure

```
f1-prediction-engine/
├── pyproject.toml                     # Dependencies & project metadata
├── requirements.txt                   # Pinned deps
├── .env.example                       # API key template (Open-Meteo doesn't need one)
├── .gitignore
├── Makefile                           # make pipeline, make dashboard, make train, make test
├── CLAUDE.md                          # Project conventions for Claude Code
│
├── config/
│   ├── settings.py                    # Central config: paths, URLs, season year, hyperparams
│   ├── circuits.json                  # Circuit metadata (lat/lon, type, length)
│   └── schedule.json                  # Race calendar
│
├── data/
│   ├── raw/
│   │   ├── kaggle/                    # Downloaded CSVs (races.csv, results.csv, etc.)
│   │   └── fastf1_cache/             # FastF1 session cache
│   ├── processed/
│   │   ├── features/                  # Engineered feature matrices (Parquet)
│   │   └── predictions/              # Prediction outputs (Parquet)
│   └── f1_engine.db                   # SQLite database
│
├── src/
│   ├── __init__.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── run.py                     # CLI entry: python -m src.pipeline.run
│   │   ├── scheduler.py              # APScheduler cron triggers
│   │   ├── orchestrator.py           # fetch → process → predict → store
│   │   │
│   │   ├── collectors/
│   │   │   ├── __init__.py
│   │   │   ├── kaggle_loader.py      # Load Kaggle CSVs into DB
│   │   │   ├── fastf1_collector.py   # Qualifying, practice, race data
│   │   │   ├── openf1_collector.py   # Real-time session data
│   │   │   ├── weather_collector.py  # Open-Meteo forecasts
│   │   │   └── standings_collector.py # Current WDC/WCC standings
│   │   │
│   │   └── processors/
│   │       ├── __init__.py
│   │       ├── cleaner.py            # Missing values, name normalization
│   │       ├── feature_engineer.py   # Core feature engineering
│   │       └── aggregator.py         # Merge sources into feature matrix
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Train LightGBM/XGBoost/LogReg ensemble
│   │   ├── predictor.py             # Generate 20×20 probability matrix
│   │   ├── evaluator.py             # Backtest accuracy, log-loss, calibration
│   │   ├── explainer.py             # SHAP feature importance
│   │   ├── registry.py              # Model save/load with versioning
│   │   ├── calibrator.py            # Probability calibration (Platt/isotonic)
│   │   ├── elo.py                   # ELO rating system for drivers
│   │   └── monte_carlo.py          # Monte Carlo race simulation
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py            # SQLAlchemy engine/session
│   │   ├── models.py                # ORM: Driver, Team, Circuit, Race, Result, Prediction
│   │   ├── queries.py               # Named queries
│   │   └── migrations.py            # Schema creation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── constants.py             # Enums, column names
│       ├── logger.py                # Loguru config
│       ├── validators.py            # Pandera/Pydantic schemas
│       └── f1_calendar.py           # Next race, timezone handling
│
├── dashboard/
│   ├── app.py                        # Streamlit main + page router
│   ├── components/
│   │   ├── __init__.py
│   │   ├── header.py                # Race countdown, last-updated
│   │   ├── filters.py               # Sidebar filters
│   │   └── charts.py                # Plotly chart builders
│   │
│   └── pages/
│       ├── 1_Race_Predictions.py    # Probability heatmap, winner card, constructor probs
│       ├── 2_Feature_Importance.py  # SHAP plots, per-driver breakdowns
│       ├── 3_Season_Standings.py    # WDC/WCC tables, points progression
│       ├── 4_Historical_Results.py  # Browse past results, driver comparison
│       └── 5_Model_Performance.py   # Backtesting, calibration, accuracy trends
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── test_collectors.py
│   ├── test_processors.py
│   ├── test_models.py
│   └── test_database.py
│
└── scripts/
    ├── download_kaggle_data.sh      # kaggle CLI download
    ├── seed_database.py             # Populate DB from CSVs
    └── backfill_predictions.py      # Historical backtest runs
```

---

## ML Architecture

### Feature Engineering (`feature_engineer.py`)

| Feature | Description | Source |
|---------|-------------|--------|
| `driver_elo` | Custom ELO rating, updated race-by-race | Historical results |
| `rolling_avg_finish_3` | Avg finishing position (last 3 races) | Historical |
| `rolling_avg_finish_5` | Avg finishing position (last 5 races) | Historical |
| `quali_position` | Qualifying grid position | FastF1/OpenF1 |
| `quali_gap_to_pole` | Time delta to pole (seconds) | FastF1 |
| `circuit_driver_avg` | Driver's historical avg finish at circuit | Historical |
| `circuit_type` | Categorical: street/high-speed/technical | circuits.json |
| `team_reliability` | Team DNF rate in last N races | Historical |
| `championship_position` | Current WDC standing | Standings API |
| `championship_points` | Current points total | Standings API |
| `teammate_delta` | Qualifying gap to teammate | FastF1 |
| `rain_probability` | Race day rain forecast | Open-Meteo |
| `air_temperature` | Forecast temperature | Open-Meteo |
| `wet_performance` | Driver's historical wet-race performance | Historical |
| `season_momentum` | Weighted recent form (exponential decay) | Historical |
| `pit_stop_avg` | Team's avg pit stop time | FastF1/Kaggle |
| `practice_pace_rank` | FP long-run pace ranking | FastF1 |
| `driver_age` | Driver's age at race time | Driver metadata |
| `experience_races` | Total career race starts | Historical |
| `penalty_count_recent` | Penalties in last 5 races | Historical |
| `safety_car_prob` | Circuit's historical safety car rate | Historical |
| `grid_to_finish_circuit` | Avg positions gained/lost at circuit | Historical |
| `constructor_budget_tier` | Team budget proxy (1-5 tier, manual) | Config |

### Era Weighting Strategy

Different F1 regulation eras have different relevance to current performance:

| Era | Years | Weight | Notes |
|-----|-------|--------|-------|
| Current regs | 2022-present | 1.0 | Ground-effect era, most relevant |
| Recent hybrid | 2017-2021 | 0.6 | Different aero regs but similar power units |
| Early hybrid | 2014-2016 | 0.3 | Turbo-hybrid intro |
| V8 era | 2006-2013 | 0.15 | Different engine formula |
| Older | 1950-2005 | 0.05 | Minimal relevance, useful for circuit/driver longevity stats |

Applied as sample weights during model training.

### Ensemble Architecture

```
                    ┌─────────────────┐
                    │  Feature Matrix  │
                    │  (per race entry)│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
    ┌─────────▼──────┐ ┌────▼─────┐ ┌──────▼──────────┐
    │   LightGBM     │ │ XGBoost  │ │ Logistic Reg    │
    │  (multi-class)  │ │(multi-cl)│ │ (ordinal proxy) │
    │  softprob output│ │ softprob │ │ calibrated probs│
    └────────┬───────┘ └────┬─────┘ └──────┬──────────┘
             │              │              │
             └──────────────┼──────────────┘
                            │
                   ┌────────▼────────┐
                   │ Weighted Average │  (optimized weights via Optuna)
                   │   Ensemble       │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │  Probability     │
                   │  Calibration     │  (isotonic regression)
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ Monte Carlo Sim  │  (10,000 race simulations)
                   │ using calibrated │  → finishing position distribution
                   │ probabilities    │
                   └────────┬────────┘
                            │
                   ┌────────▼────────┐
                   │ 20×20 Probability│
                   │    Matrix        │  drivers × positions
                   └─────────────────┘
```

### Monte Carlo Simulation (`monte_carlo.py`)

1. Take calibrated per-driver probability vectors from ensemble
2. For each simulation (N=10,000):
   - Sample qualifying performance with noise
   - Apply circuit-specific overtaking difficulty
   - Simulate first-lap incidents (historical probability)
   - Simulate safety car events (circuit-specific rate)
   - Apply DNF probability per driver/team
   - Simulate pit strategy variants (1-stop vs 2-stop)
   - Resolve finishing order
3. Aggregate 10,000 simulated races → empirical probability matrix

### Training Strategy

- **Cross-validation:** Leave-one-season-out CV (train on all seasons except one, validate on held-out season)
- **Also:** Leave-last-3-races-out for within-season validation
- **Hyperparameter tuning:** Optuna with 100 trials, optimizing log-loss
- **Class imbalance:** 20 classes (positions), naturally imbalanced. Use sample weights (era weights × recency) rather than oversampling
- **Target encoding:** Finish position (1-20) as multi-class target. DNFs encoded as position 20

---

## Database Schema (SQLite)

```sql
drivers       (id, code, full_name, nationality, date_of_birth)
teams         (id, name, engine_supplier, budget_tier)
circuits      (id, name, country, city, latitude, longitude, circuit_type, length_km)
races         (id, season, round, circuit_id FK, race_date, race_name)
results       (id, race_id FK, driver_id FK, team_id FK, grid, position, points,
               status, fastest_lap_rank, qualifying_time_ms)
standings     (id, season, round, driver_id FK, team_id FK, points, position, type)
predictions   (id, race_id FK, driver_id FK, predicted_position, probability,
               model_version, created_at)
weather       (id, race_id FK, air_temp, track_temp, rain_prob, wind_speed, humidity)
```

---

## Pipeline Flow

```
[Thursday 18:00 UTC — APScheduler cron trigger]
  │
  ├─ 1. Determine next race (f1_calendar.py → FastF1 schedule)
  │
  ├─ 2. Collect data (parallel)
  │     ├─ standings_collector → current WDC/WCC
  │     ├─ fastf1_collector → qualifying, practice times
  │     ├─ weather_collector → 3-day forecast for circuit
  │     └─ kaggle_loader → historical data (if not loaded)
  │
  ├─ 3. Process & engineer features
  │     ├─ cleaner.py → normalize, handle missing data
  │     ├─ feature_engineer.py → ELO, rolling avgs, circuit stats, weather features
  │     └─ aggregator.py → unified feature matrix (Parquet)
  │
  ├─ 4. Generate predictions
  │     ├─ predictor.py → load model, predict
  │     ├─ calibrator.py → calibrate probabilities
  │     ├─ monte_carlo.py → 10K simulations
  │     └─ Output: 20×20 probability matrix
  │
  └─ 5. Store results → SQLite + Parquet
```

**CLI Commands:**
```bash
python -m src.pipeline.run pipeline              # Run for next race
python -m src.pipeline.run pipeline --round 5    # Specific round
python -m src.pipeline.run fetch-kaggle          # Download Kaggle data
python -m src.pipeline.run train --seasons 2018-2025  # Train model
python -m src.pipeline.run scheduler-start       # Start cron daemon
streamlit run dashboard/app.py                   # Launch dashboard
```

---

## Dashboard Pages

### Page 1: Race Predictions
- Next race info card with countdown
- Predicted winner with confidence %
- Predicted podium (top 3) with probability bars
- **Driver probability heatmap**: Plotly `imshow` — drivers (Y) × positions 1-20 (X), color = probability
- **Constructor probability heatmap**: aggregated team probabilities
- Sortable prediction table

### Page 2: Feature Importance
- SHAP beeswarm plot (global)
- Per-driver SHAP waterfall (dropdown to select driver)
- Feature contribution breakdown

### Page 3: Season Standings
- Current WDC/WCC standings tables
- Points progression chart (cumulative, race-by-race)
- Season filter

### Page 4: Historical Results
- Browse past results by season/round
- Driver comparison tool (select 2+ drivers)
- Circuit history viewer

### Page 5: Model Performance
- Backtesting accuracy over past races
- Calibration plot (predicted vs actual probability)
- Log-loss trend
- Top-3 prediction accuracy

---

## Implementation Phases

### Phase 1: Foundation
1. Create project structure, `pyproject.toml`, install dependencies
2. Implement `config/settings.py` with all configuration
3. Implement `database/models.py` (SQLAlchemy ORM) and `database/connection.py`
4. Implement `database/migrations.py` (create tables)
5. Write `scripts/download_kaggle_data.sh` and `collectors/kaggle_loader.py`
6. Run `scripts/seed_database.py` to populate historical data
7. Set up `.gitignore`, `CLAUDE.md`, `Makefile`

### Phase 2: Data Pipeline
8. Implement `collectors/fastf1_collector.py`
9. Implement `collectors/openf1_collector.py`
10. Implement `collectors/weather_collector.py` (Open-Meteo)
11. Implement `collectors/standings_collector.py`
12. Implement `processors/cleaner.py`
13. Implement `processors/feature_engineer.py` (all 23+ features)
14. Implement `processors/aggregator.py`
15. Implement `pipeline/orchestrator.py`

### Phase 3: ML Models
16. Implement `models/elo.py` (ELO rating system)
17. Implement `models/trainer.py` (LightGBM + XGBoost + LogReg ensemble)
18. Implement `models/calibrator.py` (probability calibration)
19. Implement `models/monte_carlo.py` (10K simulations)
20. Implement `models/predictor.py` (prediction pipeline → 20×20 matrix)
21. Implement `models/explainer.py` (SHAP integration)
22. Implement `models/evaluator.py` (backtesting)
23. Implement `models/registry.py` (model save/load)

### Phase 4: Dashboard
24. Build `dashboard/app.py` and `components/` (header, filters, charts)
25. Build Page 1: Race Predictions (heatmap, winner card)
26. Build Page 2: Feature Importance (SHAP plots)
27. Build Page 3: Season Standings
28. Build Page 4: Historical Results
29. Build Page 5: Model Performance

### Phase 5: Automation & Polish
30. Implement `pipeline/scheduler.py` (APScheduler)
31. Implement `pipeline/run.py` (Typer CLI)
32. Write tests for collectors, processors, models
33. Create notebooks for exploration and prototyping

---

## Key Dependencies

```
# Core
pandas>=2.2, numpy>=1.26, pyarrow>=15.0

# F1 Data
fastf1>=3.8, requests>=2.31

# Database
sqlalchemy>=2.0

# ML
lightgbm>=4.3, xgboost>=2.0, scikit-learn>=1.4
shap>=0.45, optuna>=3.5

# Dashboard
streamlit>=1.40, plotly>=5.22

# Pipeline
apscheduler>=3.10, typer>=0.12

# Utils
python-dotenv>=1.0, loguru>=0.7, pandera>=0.20, pydantic>=2.6

# Testing
pytest>=8.0, pytest-cov>=5.0, responses>=0.25
```

---

## Claude Code Setup & Features for This Project

### 1. CLAUDE.md (create at project root — loaded every session)
Acts as persistent project memory. Will contain:
- Python version, venv activation command
- Key commands: `make pipeline`, `make dashboard`, `make test`
- Database location and reset instructions
- Architecture decisions (SQLite, Streamlit, LightGBM)
- Data source URLs and refresh procedures
- File structure overview and naming conventions

### 2. Skills to use during development
- **`/commit`** — Quick version control commits with Sonnet (faster, cheaper)
- **`webapp-testing`** — Test the Streamlit dashboard via Playwright (screenshots, UI behavior, browser logs)
- **`simplify`** — Review code for quality, reuse, and efficiency after each phase

### 3. MCP Servers (powerful — extend Claude's tools)
- **SQLite MCP server** — Let Claude directly query the F1 database during development. Install with: `npx @anthropic-ai/create-mcp --name sqlite` or use the community `mcp-server-sqlite`. This means Claude can run `SELECT * FROM results WHERE ...` directly instead of writing Python scripts to inspect data.
- **Filesystem MCP** — Already have access, but worth noting for data file inspection

### 4. Hooks (auto-run commands on tool events)
Configure in `.claude/settings.json`:
- **Pre-commit hook** — Auto-run `ruff check` and `ruff format` before every commit
- **Post-edit hook** — Auto-run type checking (`mypy`) after file edits
- Example: Ensure code quality stays high automatically without manual linting

### 5. Worktrees (isolated experimentation)
- Use `/worktree` when experimenting with different ML approaches (e.g., try a neural network branch vs gradient boosting branch)
- Each worktree gets its own copy of the repo — safe to experiment without affecting main code
- Great for A/B testing model architectures

### 6. Background Agents
- Run long-running model training as a background agent while continuing to work on dashboard code
- Example: kick off `python -m src.pipeline.run train` in background, keep building dashboard pages

### 7. Auto Memory (already active)
- Claude remembers project patterns, debugging insights, and your preferences across sessions
- Will automatically build up knowledge about this project's conventions as we work

### 8. Plan Mode (what we're using now)
- Use for each major phase before coding — ensures alignment on approach
- Especially useful before ML model design and feature engineering decisions

### 9. Explore Agents
- When debugging data issues or understanding FastF1/OpenF1 API behavior, Claude can launch exploration agents to search docs and code simultaneously
- Useful for "why is this feature returning NaN" type investigations

### 10. Git integration
- Initialize repo with `git init` early
- Claude can create branches, review diffs, manage PRs
- Recommend committing after each implementation phase

---

## Verification Plan

1. **Data pipeline**: Run `python -m src.pipeline.run fetch-kaggle` → verify CSVs downloaded and DB populated with `sqlite3 data/f1_engine.db "SELECT COUNT(*) FROM results;"`
2. **Feature engineering**: Run `python -m src.pipeline.run pipeline --round 1` for a past race → verify Parquet output in `data/processed/features/`
3. **Model training**: Run `python -m src.pipeline.run train --seasons 2018-2025` → verify model saved in registry, print accuracy metrics
4. **Predictions**: Run predictor for a known past race → verify 20×20 matrix sums to ~1.0 per driver, top prediction matches reasonable expectations
5. **Dashboard**: Run `streamlit run dashboard/app.py` → verify all 5 pages load, heatmap renders, filters work
6. **Scheduler**: Start scheduler, verify it triggers at configured time (test with short interval first)
7. **End-to-end**: Run full pipeline for upcoming race → view predictions in dashboard
