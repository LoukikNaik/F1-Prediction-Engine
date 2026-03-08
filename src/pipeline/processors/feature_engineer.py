"""Feature engineering for F1 race prediction."""

from datetime import datetime

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
    df = df.sort_values(["season", "round"])

    # Driver's historical average finish at this circuit
    df["circuit_driver_avg"] = (
        df.groupby(["driver_id", "circuit_name"])["position_filled"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Best ever finish at this circuit (prior visits)
    df["circuit_driver_best"] = (
        df.groupby(["driver_id", "circuit_name"])["position_filled"]
        .transform(lambda x: x.shift(1).expanding().min())
    )

    # Number of times raced at this circuit (track experience)
    df["circuit_driver_races"] = (
        df.groupby(["driver_id", "circuit_name"]).cumcount()
    )

    # Most recent finish at this circuit (last visit, could be last year)
    df["circuit_driver_last_finish"] = (
        df.groupby(["driver_id", "circuit_name"])["position_filled"]
        .transform(lambda x: x.shift(1))
    )

    # Team's historical average at this circuit
    df["circuit_team_avg"] = (
        df.groupby(["team_id", "circuit_name"])["position_filled"]
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
    df["is_dnf"] = (~df["status"].isin(["Finished", "+1 Lap", "+2 Laps", "+3 Laps", "Scheduled", "Pending"])).astype(int)
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


def add_prev_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous-season summary features.

    Gives the model explicit access to a driver's prior-year performance,
    which is critical early in a new season when rolling windows are thin.
    """
    df = df.sort_values(["season", "round"])

    # Build per-driver, per-season summaries from the data itself
    season_stats = (
        df.groupby(["driver_id", "season"])
        .agg(
            season_avg_finish=("position_filled", "mean"),
            season_best_finish=("position_filled", "min"),
            season_total_points=("points", "sum"),
            season_win_count=("position", lambda x: (x == 1).sum()),
            season_podium_count=("position", lambda x: (x <= 3).sum()),
            season_races=("position_filled", "count"),
        )
        .reset_index()
    )

    # Shift by one season: each row gets the *previous* season's stats
    season_stats = season_stats.sort_values(["driver_id", "season"])
    for col in ["season_avg_finish", "season_best_finish", "season_total_points",
                "season_win_count", "season_podium_count", "season_races"]:
        season_stats[f"prev_{col}"] = (
            season_stats.groupby("driver_id")[col].shift(1)
        )

    prev_cols = [c for c in season_stats.columns if c.startswith("prev_")]
    merge_cols = ["driver_id", "season"] + prev_cols
    df = df.merge(season_stats[merge_cols], on=["driver_id", "season"], how="left")

    # Rename for clarity
    df = df.rename(columns={
        "prev_season_avg_finish": "prev_season_avg_finish",
        "prev_season_best_finish": "prev_season_best_finish",
        "prev_season_total_points": "prev_season_points",
        "prev_season_win_count": "prev_season_wins",
        "prev_season_podium_count": "prev_season_podiums",
        "prev_season_races": "prev_season_races",
    })

    # Win rate from previous season
    races = df["prev_season_races"].replace(0, np.nan)
    df["prev_season_win_rate"] = df["prev_season_wins"] / races
    df["prev_season_podium_rate"] = df["prev_season_podiums"] / races

    return df


def add_championship_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add championship standings features.

    Uses shifted cumsum so each row sees standings *before* that race,
    preventing data leakage from the current race's points.
    """
    df = df.sort_values(["season", "round"])

    # Cumulative points within the season, shifted by 1 so each row
    # reflects standings BEFORE the current race (no leakage)
    df["championship_points"] = (
        df.groupby(["season", "driver_id"])["points"]
        .transform(lambda x: x.cumsum().shift(1).fillna(0))
    )

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
    df = add_prev_season_features(df)
    df = add_championship_features(df)
    df = add_elo_features(df)
    df = add_weather_features(df)

    # Era weight
    df["era_weight"] = df["season"].apply(compute_era_weight)

    # Fill remaining NaN feature columns with sensible defaults
    feature_cols = [
        "rolling_avg_finish_3", "rolling_avg_finish_5", "rolling_avg_finish_10",
        "season_momentum",
        "grid_filled", "grid_delta", "circuit_grid_delta_avg",
        "circuit_driver_avg", "circuit_driver_best", "circuit_driver_races",
        "circuit_driver_last_finish", "circuit_team_avg", "safety_car_prob",
        "team_reliability", "teammate_delta",
        "experience_races", "driver_age",
        "prev_season_avg_finish", "prev_season_best_finish",
        "prev_season_points", "prev_season_wins",
        "prev_season_podiums", "prev_season_win_rate", "prev_season_podium_rate",
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
        "rolling_avg_finish_10",
        "season_momentum",
        "circuit_driver_avg",
        "circuit_driver_best",
        "circuit_driver_races",
        "circuit_driver_last_finish",
        "circuit_team_avg",
        "circuit_grid_delta_avg",
        "safety_car_prob",
        "team_reliability",
        "teammate_delta",
        "experience_races",
        "driver_age",
        "prev_season_avg_finish",
        "prev_season_best_finish",
        "prev_season_points",
        "prev_season_wins",
        "prev_season_podiums",
        "prev_season_win_rate",
        "prev_season_podium_rate",
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
