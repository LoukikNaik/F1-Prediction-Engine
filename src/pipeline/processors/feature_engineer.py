"""Feature engineering for F1 race prediction."""

from datetime import date, datetime

import numpy as np
import pandas as pd

from config.settings import ERA_WEIGHTS, ROLLING_WINDOWS
from src.models.elo import EloRating
from src.utils.constants import DNF_POSITION
from src.utils.logger import logger


def compute_era_weight(season: int) -> float:
    """Get era weight for a given season."""
    for (start, end), weight in ERA_WEIGHTS.items():
        if start <= season <= end:
            return weight
    return 0.05


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling average finishing position features per driver."""
    df = df.sort_values(["season", "round"])

    for window in ROLLING_WINDOWS:
        col_name = f"rolling_avg_finish_{window}"
        df[col_name] = (
            df.groupby("driver_id")["position_filled"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )

    # Season momentum (exponential weighted)
    df["season_momentum"] = (
        df.groupby("driver_id")["position_filled"]
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    return df


def add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add qualifying/grid position related features."""
    df["grid_filled"] = df["grid"].fillna(DNF_POSITION).clip(upper=DNF_POSITION)

    # Positions gained/lost from grid
    df["grid_delta"] = df["grid_filled"] - df["position_filled"]

    # Historical grid-to-finish at this circuit
    df["circuit_grid_delta_avg"] = (
        df.groupby(["driver_id", "circuit_name"])["grid_delta"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    return df


def add_circuit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add circuit-specific performance features."""
    # Driver's historical average finish at this circuit
    df["circuit_driver_avg"] = (
        df.groupby(["driver_id", "circuit_name"])["position_filled"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Circuit safety car rate (approximated by DNF rate)
    circuit_dnf = df.groupby("circuit_name").apply(
        lambda g: (g["status"] != "Finished").mean(), include_groups=False
    ).to_dict()
    df["safety_car_prob"] = df["circuit_name"].map(circuit_dnf).fillna(0.3)

    return df


def add_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team/constructor features."""
    df = df.sort_values(["season", "round"])

    # Team reliability (DNF rate in recent races)
    df["is_dnf"] = (~df["status"].isin(["Finished", "+1 Lap", "+2 Laps", "+3 Laps"])).astype(int)
    df["team_reliability"] = (
        df.groupby("team_id")["is_dnf"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # Teammate delta (qualifying gap)
    # Group by race and team, compute relative qualifying performance
    def teammate_delta(group):
        if len(group) < 2:
            return pd.Series(0.0, index=group.index)
        times = group["qualifying_time_ms"].values
        if pd.isna(times).all():
            return pd.Series(0.0, index=group.index)
        team_mean = np.nanmean(times)
        if team_mean == 0:
            return pd.Series(0.0, index=group.index)
        return (group["qualifying_time_ms"] - team_mean) / team_mean

    df["teammate_delta"] = df.groupby(["race_id", "team_id"]).apply(
        teammate_delta, include_groups=False
    ).reset_index(level=[0, 1], drop=True)

    return df


def add_driver_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add driver metadata features (age, experience)."""
    # Experience: cumulative race count
    df["experience_races"] = df.groupby("driver_id").cumcount()

    # Driver age at race time
    def compute_age(row):
        try:
            if pd.isna(row.get("date_of_birth")) or pd.isna(row.get("race_date")):
                return None
            dob = datetime.strptime(str(row["date_of_birth"]), "%Y-%m-%d")
            race = datetime.strptime(str(row["race_date"]), "%Y-%m-%d")
            return (race - dob).days / 365.25
        except (ValueError, TypeError):
            return None

    df["driver_age"] = df.apply(compute_age, axis=1)

    return df


def add_championship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add championship standings features."""
    df = df.sort_values(["season", "round"])

    # Cumulative points within the season
    df["championship_points"] = df.groupby(["season", "driver_id"])["points"].cumsum()

    # Championship position within the season (rank by cumulative points)
    df["championship_position"] = (
        df.groupby(["season", "round"])["championship_points"]
        .rank(method="min", ascending=False)
    )

    return df


def add_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ELO rating features."""
    elo = EloRating()
    elo_df = elo.compute_all_ratings(
        df[["season", "round", "driver_id", "position"]].copy()
    )

    df = df.merge(elo_df, on=["season", "round", "driver_id"], how="left")
    df["elo_rating"] = df["elo_rating"].fillna(1500)

    return df


def engineer_features(results_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Args:
        results_df: Raw results DataFrame from database query.

    Returns:
        Feature-enriched DataFrame ready for ML training.
    """
    logger.info(f"Engineering features for {len(results_df)} results...")

    df = results_df.copy()

    # Fill position for DNFs
    df["position_filled"] = df["position"].fillna(DNF_POSITION).clip(upper=DNF_POSITION)

    # Sort chronologically
    df = df.sort_values(["season", "round"]).reset_index(drop=True)

    # Add all feature groups
    df = add_rolling_features(df)
    df = add_grid_features(df)
    df = add_circuit_features(df)
    df = add_team_features(df)
    df = add_driver_meta_features(df)
    df = add_championship_features(df)
    df = add_elo_features(df)
    df = add_weather_features(df)

    # Era weight
    df["era_weight"] = df["season"].apply(compute_era_weight)

    # Fill remaining NaN feature columns with sensible defaults
    feature_cols = [
        "rolling_avg_finish_3", "rolling_avg_finish_5", "season_momentum",
        "grid_filled", "grid_delta", "circuit_grid_delta_avg",
        "circuit_driver_avg", "safety_car_prob",
        "team_reliability", "teammate_delta",
        "experience_races", "driver_age",
        "championship_points", "championship_position",
        "elo_rating", "era_weight",
        "rain_prob", "air_temp", "wind_speed", "is_wet", "temp_normalized",
    ]

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-related features.

    Handles missing weather data gracefully — most historical races won't have it.
    """
    # If weather columns aren't present, fill with defaults
    if "rain_prob" not in df.columns:
        df["rain_prob"] = np.nan
    if "air_temp" not in df.columns:
        df["air_temp"] = np.nan
    if "wind_speed" not in df.columns:
        df["wind_speed"] = np.nan

    # Binary wet flag
    rain = pd.to_numeric(df["rain_prob"], errors="coerce").fillna(0)
    df["is_wet"] = (rain > 0.5).astype(float)

    # Normalized temperature (z-score across all races)
    air = pd.to_numeric(df["air_temp"], errors="coerce")
    temp_mean = air.mean()
    temp_std = air.std()
    if pd.notna(temp_mean) and pd.notna(temp_std) and temp_std > 0:
        df["temp_normalized"] = (air - temp_mean) / temp_std
    else:
        df["temp_normalized"] = 0.0

    # Fill NaNs with sensible defaults
    df["rain_prob"] = pd.to_numeric(df["rain_prob"], errors="coerce").fillna(0.1)
    df["air_temp"] = pd.to_numeric(df["air_temp"], errors="coerce").fillna(25.0)
    df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce").fillna(10.0)

    return df


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used for ML training."""
    return [
        "grid_filled",
        "rolling_avg_finish_3",
        "rolling_avg_finish_5",
        "season_momentum",
        "circuit_driver_avg",
        "circuit_grid_delta_avg",
        "safety_car_prob",
        "team_reliability",
        "teammate_delta",
        "experience_races",
        "driver_age",
        "championship_points",
        "championship_position",
        "elo_rating",
        "era_weight",
        "rain_prob",
        "air_temp",
        "wind_speed",
        "is_wet",
        "temp_normalized",
    ]
