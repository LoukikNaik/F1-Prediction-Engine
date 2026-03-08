"""Pipeline orchestrator: fetch -> process -> predict -> store."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import CURRENT_SEASON, PREDICTIONS_DIR
from src.database.connection import get_session
from src.database.models import (
    Circuit,
    Prediction,
    PredictionMatrix,
    Race,
    Result,
    Weather,
)
from src.database.queries import get_all_results_df, get_races_for_season
from src.models.predictor import build_race_features, generate_predictions
from src.models.trainer import F1Ensemble
from src.pipeline.collectors.weather_collector import (
    get_race_weekend_weather,
    get_weather_forecast,
)
from src.pipeline.stage_detector import detect_stage
from src.utils.constants import PredictionStage
from src.utils.logger import logger


def train_model(min_season: int = 2000) -> F1Ensemble:
    """Train the ensemble model on all historical data.

    Args:
        min_season: Earliest season to include in training.

    Returns:
        Trained F1Ensemble model.
    """
    logger.info(f"Training model on data from {min_season} onwards...")

    with get_session() as session:
        results_df = get_all_results_df(session, min_season=min_season)

    if results_df.empty:
        raise ValueError("No results data found in database. Run seed_database.py first.")

    logger.info(f"Loaded {len(results_df)} results for training")

    # Engineer features
    from src.pipeline.processors.feature_engineer import engineer_features

    featured_df = engineer_features(results_df)

    # Remove rows with NaN in critical columns
    feature_cols = F1Ensemble().feature_columns
    featured_df = featured_df.dropna(subset=feature_cols + ["position_filled"])

    # Train the ensemble
    model = F1Ensemble()
    metrics = model.train(featured_df)

    # Save model
    model.save()

    logger.info(f"Model trained. Metrics: {metrics}")
    return model


def predict_race(
    season: int,
    round_num: int,
    stage: PredictionStage | None = None,
    model: F1Ensemble | None = None,
) -> pd.DataFrame:
    """Generate predictions for a specific race at a given stage.

    Args:
        season: Race season year.
        round_num: Race round number.
        stage: Prediction stage. If None, auto-detected from schedule.
        model: Trained model. If None, loads from disk.

    Returns:
        Probability matrix DataFrame.
    """
    if model is None:
        model = F1Ensemble.load()

    with get_session() as session:
        results_df = get_all_results_df(session, min_season=2000)

        # Auto-detect stage if not specified (must be inside session)
        if stage is None:
            race = (
                session.query(Race)
                .filter(Race.season == season, Race.round == round_num)
                .first()
            )
            if race:
                stage = detect_stage(race)
            else:
                stage = PredictionStage.PRE_WEEKEND

    if results_df.empty:
        raise ValueError("No results data found.")

    logger.info(f"Predicting {season} R{round_num} at stage: {stage.value}")

    # Build features for the target race
    race_features = build_race_features(results_df, season, round_num, stage=stage)

    if race_features.empty:
        raise ValueError(f"No data for season {season} round {round_num}")

    # Generate predictions
    prob_matrix, ensemble_proba = generate_predictions(model, race_features)

    # Save matrix to parquet
    matrix_path = _save_matrix(prob_matrix, season, round_num, stage)

    # Store predictions in database
    _store_predictions(season, round_num, stage, prob_matrix, race_features, matrix_path)

    # Fetch and store weather
    _store_weather(season, round_num, stage)

    logger.info(f"Predictions saved for {season} R{round_num} stage={stage.value}")
    return prob_matrix


def predict_season(
    season: int,
    model: F1Ensemble | None = None,
) -> dict[int, pd.DataFrame]:
    """Predict all upcoming races in a season.

    Args:
        season: Target season.
        model: Trained model. If None, loads from disk.

    Returns:
        Dict mapping round_num to probability matrix.
    """
    if model is None:
        model = F1Ensemble.load()

    with get_session() as session:
        races = get_races_for_season(session, season)

    if not races:
        raise ValueError(f"No races found for season {season}")

    results = {}
    for race in races:
        try:
            prob_matrix = predict_race(
                season, race.round, stage=PredictionStage.PRE_WEEKEND, model=model
            )
            results[race.round] = prob_matrix
            logger.info(f"Predicted R{race.round} ({race.race_name})")
        except Exception as e:
            logger.warning(f"Failed to predict R{race.round} ({race.race_name}): {e}")

    logger.info(f"Season prediction complete: {len(results)}/{len(races)} races predicted")
    return results


def run_full_pipeline(
    season: int = CURRENT_SEASON,
    round_num: int | None = None,
    retrain: bool = False,
) -> pd.DataFrame:
    """Run the complete prediction pipeline.

    Args:
        season: Target season.
        round_num: Target round. If None, auto-detects next upcoming race.
        retrain: Whether to retrain the model.

    Returns:
        Probability matrix DataFrame.
    """
    # Auto-detect next race if round not specified
    if round_num is None:
        round_num = _find_next_round(season)
        logger.info(f"Auto-detected next race: season {season} round {round_num}")

    logger.info(f"Running pipeline for season {season}, round {round_num}...")

    if retrain:
        model = train_model()
    else:
        try:
            model = F1Ensemble.load()
        except FileNotFoundError:
            logger.info("No saved model found, training from scratch...")
            model = train_model()

    return predict_race(season, round_num, model=model)


def _find_next_round(season: int) -> int:
    """Find the next upcoming race round for a season.

    Returns the first round that has no race results yet,
    or the last round if all are completed.
    """
    from datetime import date

    with get_session() as session:
        races = get_races_for_season(session, season)
        if not races:
            return 1

        for race in races:
            # Check if this race has results
            result_count = (
                session.query(Result)
                .filter(Result.race_id == race.id)
                .filter(Result.position.isnot(None))
                .count()
            )
            if result_count == 0:
                return race.round

        # All races completed, return last one
        return races[-1].round


def _save_matrix(
    prob_matrix: pd.DataFrame,
    season: int,
    round_num: int,
    stage: PredictionStage,
) -> Path:
    """Save probability matrix to parquet file."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"matrix_{season}_r{round_num}_{stage.value}.parquet"
    path = PREDICTIONS_DIR / filename
    prob_matrix.to_parquet(path)

    # Also save as latest_matrix for backward compat
    latest_path = PREDICTIONS_DIR / "latest_matrix.parquet"
    prob_matrix.to_parquet(latest_path)

    logger.info(f"Matrix saved to {path}")
    return path


def _store_predictions(
    season: int,
    round_num: int,
    stage: PredictionStage,
    prob_matrix: pd.DataFrame,
    race_features: pd.DataFrame,
    matrix_path: Path,
) -> None:
    """Store predictions in the database, upserting by (race_id, driver_id, stage)."""
    with get_session() as session:
        race = (
            session.query(Race)
            .filter(Race.season == season, Race.round == round_num)
            .first()
        )
        if not race:
            return

        # Delete existing predictions for this race+stage (upsert)
        session.query(Prediction).filter(
            Prediction.race_id == race.id,
            Prediction.stage == stage.value,
        ).delete()

        for driver_name, row in prob_matrix.iterrows():
            driver_row = race_features[race_features["driver_name"] == driver_name]
            if driver_row.empty:
                continue

            driver_id = int(driver_row["driver_id"].iloc[0])
            pred_pos = int(row["expected_position"])
            win_prob = float(row["win_prob"])

            prediction = Prediction(
                race_id=race.id,
                driver_id=driver_id,
                predicted_position=pred_pos,
                probability=win_prob,
                stage=stage.value,
                model_version="ensemble_v2",
            )
            session.add(prediction)

        # Store matrix path reference
        session.query(PredictionMatrix).filter(
            PredictionMatrix.race_id == race.id,
            PredictionMatrix.stage == stage.value,
        ).delete()
        session.add(PredictionMatrix(
            race_id=race.id,
            stage=stage.value,
            matrix_parquet_path=str(matrix_path),
        ))


def _store_weather(
    season: int, round_num: int, stage: PredictionStage
) -> None:
    """Fetch and store weather for a race."""
    try:
        with get_session() as session:
            race = (
                session.query(Race)
                .filter(Race.season == season, Race.round == round_num)
                .first()
            )
            if not race:
                return

            circuit = session.query(Circuit).filter(Circuit.id == race.circuit_id).first()
            if not circuit or not circuit.latitude or not circuit.longitude or not race.race_date:
                return

            # Fetch weather for all sessions if we have timing info
            if race.qualifying_date:
                weather_data = get_race_weekend_weather(
                    circuit.latitude,
                    circuit.longitude,
                    race.qualifying_date,
                    race.race_date,
                    race.sprint_date,
                )
                for forecast_type, data in weather_data.items():
                    if data:
                        # Check if weather already exists
                        existing = (
                            session.query(Weather)
                            .filter(
                                Weather.race_id == race.id,
                                Weather.forecast_type == forecast_type,
                            )
                            .first()
                        )
                        if not existing:
                            session.add(Weather(
                                race_id=race.id,
                                forecast_type=forecast_type,
                                fetched_at=datetime.utcnow(),
                                **data,
                            ))
            else:
                # Fallback: single forecast for race day
                weather = get_weather_forecast(
                    circuit.latitude, circuit.longitude, race.race_date
                )
                if weather:
                    existing = (
                        session.query(Weather)
                        .filter(Weather.race_id == race.id)
                        .first()
                    )
                    if not existing:
                        session.add(Weather(
                            race_id=race.id,
                            forecast_type="race_day",
                            fetched_at=datetime.utcnow(),
                            **weather,
                        ))
    except Exception as e:
        logger.warning(f"Weather storage failed: {e}")
