"""Named database query functions."""

from typing import Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.models import (
    Circuit,
    Driver,
    Prediction,
    PredictionMatrix,
    Race,
    Result,
    Standing,
    Team,
)


def get_driver_by_name(session: Session, name: str) -> Optional[Driver]:
    """Find a driver by full name (case-insensitive)."""
    return session.query(Driver).filter(Driver.full_name.ilike(f"%{name}%")).first()


def get_driver_by_ergast_id(session: Session, ergast_id: str) -> Optional[Driver]:
    """Find a driver by Ergast ID."""
    return session.query(Driver).filter(Driver.ergast_id == ergast_id).first()


def get_team_by_name(session: Session, name: str) -> Optional[Team]:
    """Find a team by name (case-insensitive)."""
    return session.query(Team).filter(Team.name.ilike(f"%{name}%")).first()


def get_team_by_ergast_id(session: Session, ergast_id: str) -> Optional[Team]:
    """Find a team by Ergast ID."""
    return session.query(Team).filter(Team.ergast_id == ergast_id).first()


def get_circuit_by_name(session: Session, name: str) -> Optional[Circuit]:
    """Find a circuit by name."""
    return session.query(Circuit).filter(Circuit.name.ilike(f"%{name}%")).first()


def get_race(session: Session, season: int, round_num: int) -> Optional[Race]:
    """Get a specific race by season and round."""
    return session.query(Race).filter(Race.season == season, Race.round == round_num).first()


def get_all_results_df(session: Session, min_season: int = 1950) -> pd.DataFrame:
    """Get all race results as a DataFrame, with weather data joined."""
    query = text("""
        SELECT r.id as result_id, r.race_id, race.season, race.round, race.race_name,
               race.race_date,
               c.name as circuit_name, c.circuit_type, c.latitude, c.longitude,
               d.full_name as driver_name, d.code as driver_code, d.nationality,
               d.date_of_birth, d.id as driver_id,
               t.name as team_name, t.id as team_id,
               r.grid, r.position, r.points, r.status,
               r.fastest_lap_rank, r.qualifying_time_ms, r.laps, r.milliseconds,
               w.air_temp, w.rain_prob, w.wind_speed, w.humidity
        FROM results r
        JOIN races race ON r.race_id = race.id
        LEFT JOIN circuits c ON race.circuit_id = c.id
        JOIN drivers d ON r.driver_id = d.id
        LEFT JOIN teams t ON r.team_id = t.id
        LEFT JOIN weather w ON w.race_id = race.id
            AND (w.forecast_type = 'race_day' OR w.forecast_type IS NULL)
        WHERE race.season >= :min_season
        ORDER BY race.season, race.round, r.position
    """)
    return pd.read_sql(query, session.bind, params={"min_season": min_season})


def get_standings_df(
    session: Session, season: int, standing_type: str = "driver"
) -> pd.DataFrame:
    """Get championship standings for a season."""
    query = text("""
        SELECT s.season, s.round, s.points, s.position, s.standing_type,
               d.full_name as driver_name, d.code as driver_code,
               t.name as team_name
        FROM standings s
        LEFT JOIN drivers d ON s.driver_id = d.id
        LEFT JOIN teams t ON s.team_id = t.id
        WHERE s.season = :season AND s.standing_type = :standing_type
        ORDER BY s.round, s.position
    """)
    return pd.read_sql(
        query, session.bind, params={"season": season, "standing_type": standing_type}
    )


def get_driver_circuit_history(
    session: Session, driver_id: int, circuit_id: int
) -> pd.DataFrame:
    """Get a driver's historical results at a specific circuit."""
    query = text("""
        SELECT race.season, race.round, r.grid, r.position, r.points, r.status
        FROM results r
        JOIN races race ON r.race_id = race.id
        WHERE r.driver_id = :driver_id AND race.circuit_id = :circuit_id
        ORDER BY race.season
    """)
    return pd.read_sql(
        query, session.bind, params={"driver_id": driver_id, "circuit_id": circuit_id}
    )


def get_recent_results(
    session: Session, driver_id: int, n_races: int = 5
) -> pd.DataFrame:
    """Get a driver's most recent N race results."""
    query = text("""
        SELECT race.season, race.round, race.race_name,
               r.grid, r.position, r.points, r.status
        FROM results r
        JOIN races race ON r.race_id = race.id
        WHERE r.driver_id = :driver_id AND r.position IS NOT NULL
        ORDER BY race.season DESC, race.round DESC
        LIMIT :n_races
    """)
    return pd.read_sql(
        query, session.bind, params={"driver_id": driver_id, "n_races": n_races}
    )


def get_predictions_by_stage(
    session: Session, race_id: int, stage: str
) -> pd.DataFrame:
    """Get predictions for a race at a specific stage."""
    query = text("""
        SELECT p.predicted_position, p.probability, p.stage,
               d.full_name as driver_name, d.code as driver_code
        FROM predictions p
        JOIN drivers d ON p.driver_id = d.id
        WHERE p.race_id = :race_id AND p.stage = :stage
        ORDER BY p.predicted_position
    """)
    return pd.read_sql(
        query, session.bind, params={"race_id": race_id, "stage": stage}
    )


def get_available_stages(session: Session, race_id: int) -> list[str]:
    """Get list of prediction stages available for a race."""
    rows = (
        session.query(Prediction.stage)
        .filter(Prediction.race_id == race_id)
        .distinct()
        .all()
    )
    stage_order = ["pre_weekend", "post_qualifying", "post_sprint", "race_eve"]
    stages = [r[0] for r in rows if r[0] in stage_order]
    return sorted(stages, key=lambda s: stage_order.index(s))


def get_races_for_season(session: Session, season: int) -> list[Race]:
    """Get all races for a season, ordered by round."""
    return (
        session.query(Race)
        .filter(Race.season == season)
        .order_by(Race.round)
        .all()
    )


def get_prediction_matrix_path(
    session: Session, race_id: int, stage: str
) -> Optional[str]:
    """Get the parquet path for a prediction matrix."""
    row = (
        session.query(PredictionMatrix.matrix_parquet_path)
        .filter(PredictionMatrix.race_id == race_id, PredictionMatrix.stage == stage)
        .order_by(PredictionMatrix.created_at.desc())
        .first()
    )
    return row[0] if row else None


def get_live_snapshots(
    session: Session, race_id: int, lap_number: int | None = None
) -> pd.DataFrame:
    """Get live prediction snapshots for a race.

    Args:
        session: SQLAlchemy session.
        race_id: Race ID.
        lap_number: Optional specific lap number. If None, returns all laps.

    Returns:
        DataFrame with live prediction snapshot data.
    """
    sql = """
        SELECT lap_number, driver_name, current_position, predicted_position,
               win_prob, podium_prob, tire_compound, tire_age,
               safety_car_active, created_at
        FROM live_prediction_snapshots
        WHERE race_id = :race_id
    """
    params: dict = {"race_id": race_id}
    if lap_number is not None:
        sql += " AND lap_number = :lap_number"
        params["lap_number"] = lap_number
    sql += " ORDER BY lap_number, predicted_position"
    return pd.read_sql(text(sql), session.bind, params=params)


def get_race_weather(session: Session, race_id: int) -> dict | None:
    """Get weather data for a race."""
    from src.database.models import Weather
    weather = (
        session.query(Weather)
        .filter(Weather.race_id == race_id)
        .order_by(Weather.fetched_at.desc())
        .first()
    )
    if not weather:
        return None
    return {
        "air_temp": weather.air_temp,
        "track_temp": weather.track_temp,
        "rain_prob": weather.rain_prob,
        "wind_speed": weather.wind_speed,
        "humidity": weather.humidity,
        "forecast_type": weather.forecast_type,
    }
