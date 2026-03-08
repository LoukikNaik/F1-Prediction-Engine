"""Generate race predictions using the trained ensemble + Monte Carlo."""

import numpy as np
import pandas as pd

from src.models.monte_carlo import run_monte_carlo
from src.models.trainer import F1Ensemble
from src.pipeline.processors.feature_engineer import engineer_features, get_feature_columns
from src.utils.constants import PredictionStage
from src.utils.logger import logger


def generate_predictions(
    model: F1Ensemble,
    race_features_df: pd.DataFrame,
    use_monte_carlo: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate predictions for an upcoming race.

    Args:
        model: Trained F1Ensemble model.
        race_features_df: Feature-engineered DataFrame for the race entries.
        use_monte_carlo: Whether to use Monte Carlo simulation.

    Returns:
        Tuple of (probability_matrix_df, raw_ensemble_proba).
    """
    logger.info(f"Generating predictions for {len(race_features_df)} drivers...")

    # Get driver names
    driver_names = race_features_df["driver_name"].tolist()

    # Get ensemble probabilities
    ensemble_proba = model.predict_proba(race_features_df)

    if use_monte_carlo:
        # Get grid positions and DNF probabilities for Monte Carlo
        grid_positions = race_features_df["grid_filled"].values.astype(int)
        dnf_probs = race_features_df["team_reliability"].values.clip(0, 0.3)

        prob_matrix = run_monte_carlo(
            ensemble_proba=ensemble_proba,
            driver_names=driver_names,
            grid_positions=grid_positions,
            dnf_probs=dnf_probs,
        )
    else:
        # Use raw ensemble probabilities
        columns = [f"P{i}" for i in range(1, 21)]
        prob_matrix = pd.DataFrame(
            ensemble_proba,
            index=driver_names,
            columns=columns,
        )
        positions = np.arange(1, 21)
        prob_matrix["expected_position"] = (ensemble_proba * positions).sum(axis=1)
        prob_matrix["win_prob"] = ensemble_proba[:, 0]
        prob_matrix["podium_prob"] = ensemble_proba[:, :3].sum(axis=1)
        prob_matrix["top5_prob"] = ensemble_proba[:, :5].sum(axis=1)
        prob_matrix = prob_matrix.sort_values("expected_position")

    logger.info("Predictions generated successfully")
    return prob_matrix, ensemble_proba


def build_race_features(
    all_results_df: pd.DataFrame,
    race_season: int,
    race_round: int,
    stage: PredictionStage = PredictionStage.PRE_WEEKEND,
) -> pd.DataFrame:
    """Build feature matrix for a specific upcoming race.

    Uses historical data up to (but not including) the target race.
    Feature adjustments based on prediction stage:
    - PRE_WEEKEND: grid_filled replaced with rolling avg qualifying position (proxy)
    - POST_QUALIFYING+: actual grid from DB used as-is
    - POST_SPRINT: sprint results can refine form features

    Args:
        all_results_df: All historical results.
        race_season: Target race season.
        race_round: Target race round.
        stage: Prediction stage.

    Returns:
        DataFrame with features for drivers in the target race.
    """
    # Split: training history vs target race
    target_mask = (
        (all_results_df["season"] == race_season) & (all_results_df["round"] == race_round)
    )

    # Engineer features on all data (features use lagged values, so no leakage)
    all_featured = engineer_features(all_results_df)

    # Return only the target race rows
    target_df = all_featured[target_mask].copy()

    if target_df.empty:
        logger.warning(f"No data found for season {race_season} round {race_round}")
        return target_df

    # Stage-specific adjustments
    if stage == PredictionStage.PRE_WEEKEND:
        # Before qualifying: use rolling average of recent qualifying positions as grid proxy
        if "rolling_avg_finish_3" in target_df.columns:
            target_df["grid_filled"] = target_df["rolling_avg_finish_3"].clip(1, 22)
            # Recompute grid_delta with proxy
            target_df["grid_delta"] = target_df["grid_filled"] - target_df["position_filled"]

    # POST_QUALIFYING and later: actual grid from DB is already in the data
    # POST_SPRINT: sprint results would have already been incorporated into
    # rolling features if the data was refreshed

    logger.info(
        f"Built features for {len(target_df)} drivers in round {race_round} "
        f"(stage: {stage.value})"
    )
    return target_df
