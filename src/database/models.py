"""SQLAlchemy ORM models for the F1 Prediction Engine."""

from datetime import date, datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Driver(Base):
    __tablename__ = "drivers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(3), nullable=True)
    full_name = Column(String(100), nullable=False)
    nationality = Column(String(50), nullable=True)
    date_of_birth = Column(String(10), nullable=True)
    ergast_id = Column(String(50), unique=True, nullable=True)

    results = relationship("Result", back_populates="driver")
    standings = relationship("Standing", back_populates="driver")
    predictions = relationship("Prediction", back_populates="driver")
    championship_predictions = relationship("ChampionshipPrediction", back_populates="driver")


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    engine_supplier = Column(String(50), nullable=True)
    budget_tier = Column(Integer, default=3)
    ergast_id = Column(String(50), unique=True, nullable=True)

    results = relationship("Result", back_populates="team")
    standings = relationship("Standing", back_populates="team")
    championship_predictions = relationship("ChampionshipPrediction", back_populates="team")


class Circuit(Base):
    __tablename__ = "circuits"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    country = Column(String(50), nullable=True)
    city = Column(String(50), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    circuit_type = Column(String(20), default="hybrid")
    length_km = Column(Float, nullable=True)
    ergast_id = Column(String(50), unique=True, nullable=True)

    races = relationship("Race", back_populates="circuit")


class Race(Base):
    __tablename__ = "races"

    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    circuit_id = Column(Integer, ForeignKey("circuits.id"), nullable=True)
    race_date = Column(String(10), nullable=True)
    race_name = Column(String(100), nullable=False)
    qualifying_date = Column(String(10), nullable=True)
    qualifying_time = Column(String(8), nullable=True)
    sprint_date = Column(String(10), nullable=True)
    sprint_time = Column(String(8), nullable=True)
    race_time = Column(String(8), nullable=True)
    is_sprint_weekend = Column(Integer, default=0)

    __table_args__ = (UniqueConstraint("season", "round", name="uq_race_season_round"),)

    circuit = relationship("Circuit", back_populates="races")
    results = relationship("Result", back_populates="race")
    predictions = relationship("Prediction", back_populates="race")
    weather = relationship("Weather", back_populates="race")
    prediction_matrices = relationship("PredictionMatrix", back_populates="race")


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    grid = Column(Integer, nullable=True)
    position = Column(Integer, nullable=True)
    points = Column(Float, default=0)
    status = Column(String(50), nullable=True)
    fastest_lap_rank = Column(Integer, nullable=True)
    qualifying_time_ms = Column(Float, nullable=True)
    laps = Column(Integer, nullable=True)
    milliseconds = Column(Integer, nullable=True)

    race = relationship("Race", back_populates="results")
    driver = relationship("Driver", back_populates="results")
    team = relationship("Team", back_populates="results")


class Standing(Base):
    __tablename__ = "standings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False)
    round = Column(Integer, nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    points = Column(Float, default=0)
    position = Column(Integer, nullable=True)
    standing_type = Column(String(20), nullable=False)

    driver = relationship("Driver", back_populates="standings")
    team = relationship("Team", back_populates="standings")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=False)
    predicted_position = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    stage = Column(String(20), default="pre_weekend")
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("race_id", "driver_id", "stage", name="uq_prediction_race_driver_stage"),
    )

    race = relationship("Race", back_populates="predictions")
    driver = relationship("Driver", back_populates="predictions")


class Weather(Base):
    __tablename__ = "weather"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    air_temp = Column(Float, nullable=True)
    track_temp = Column(Float, nullable=True)
    rain_prob = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    humidity = Column(Float, nullable=True)
    forecast_type = Column(String(20), default="race_day")
    fetched_at = Column(DateTime, default=datetime.utcnow)

    race = relationship("Race", back_populates="weather")


class PredictionMatrix(Base):
    __tablename__ = "prediction_matrices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    stage = Column(String(20), nullable=False)
    matrix_parquet_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    race = relationship("Race", back_populates="prediction_matrices")


class ChampionshipPrediction(Base):
    __tablename__ = "championship_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(Integer, nullable=False)
    after_round = Column(Integer, nullable=False)
    driver_id = Column(Integer, ForeignKey("drivers.id"), nullable=True)
    team_id = Column(Integer, ForeignKey("teams.id"), nullable=True)
    prediction_type = Column(String(10), nullable=False)  # "wdc" or "wcc"
    predicted_position = Column(Integer, nullable=True)
    probability = Column(Float, nullable=True)
    predicted_points = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    driver = relationship("Driver", back_populates="championship_predictions")
    team = relationship("Team", back_populates="championship_predictions")


class LivePredictionSnapshot(Base):
    __tablename__ = "live_prediction_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Integer, ForeignKey("races.id"), nullable=False)
    lap_number = Column(Integer, nullable=False)
    driver_name = Column(String(100), nullable=False)
    current_position = Column(Integer, nullable=True)
    predicted_position = Column(Float, nullable=True)
    win_prob = Column(Float, nullable=True)
    podium_prob = Column(Float, nullable=True)
    tire_compound = Column(String(20), nullable=True)
    tire_age = Column(Integer, nullable=True)
    safety_car_active = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "race_id", "lap_number", "driver_name",
            name="uq_live_snap_race_lap_driver",
        ),
    )

    race = relationship("Race")
