"""Database schema creation, initialization, and migrations."""

from sqlalchemy import inspect, text

from src.database.connection import engine
from src.database.models import Base
from src.utils.logger import logger


def create_all_tables() -> None:
    """Create all database tables."""
    Base.metadata.create_all(engine)
    logger.info("All database tables created successfully")


def drop_all_tables() -> None:
    """Drop all database tables (use with caution)."""
    Base.metadata.drop_all(engine)
    logger.info("All database tables dropped")


def _column_exists(conn, table: str, column: str) -> bool:
    """Check if a column exists in a SQLite table."""
    result = conn.execute(text(f"PRAGMA table_info({table})"))
    columns = [row[1] for row in result.fetchall()]
    return column in columns


def _table_exists(conn, table: str) -> bool:
    """Check if a table exists in SQLite."""
    result = conn.execute(
        text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": table},
    )
    return result.fetchone() is not None


def migrate_v2() -> None:
    """Run v2 migration: add stage support, timing columns, new tables.

    Safe to run multiple times — skips columns/tables that already exist.
    """
    logger.info("Running v2 migration...")

    with engine.connect() as conn:
        # -- Race table: add session timing columns --
        race_columns = {
            "qualifying_date": "TEXT",
            "qualifying_time": "TEXT",
            "sprint_date": "TEXT",
            "sprint_time": "TEXT",
            "race_time": "TEXT",
            "is_sprint_weekend": "INTEGER DEFAULT 0",
        }
        for col_name, col_type in race_columns.items():
            if not _column_exists(conn, "races", col_name):
                conn.execute(text(f"ALTER TABLE races ADD COLUMN {col_name} {col_type}"))
                logger.info(f"Added races.{col_name}")

        # -- Prediction table: add stage column --
        if not _column_exists(conn, "predictions", "stage"):
            conn.execute(
                text("ALTER TABLE predictions ADD COLUMN stage TEXT DEFAULT 'pre_weekend'")
            )
            logger.info("Added predictions.stage")

        # -- Weather table: add forecast_type and fetched_at --
        if not _column_exists(conn, "weather", "forecast_type"):
            conn.execute(
                text("ALTER TABLE weather ADD COLUMN forecast_type TEXT DEFAULT 'race_day'")
            )
            logger.info("Added weather.forecast_type")
        if not _column_exists(conn, "weather", "fetched_at"):
            conn.execute(text("ALTER TABLE weather ADD COLUMN fetched_at DATETIME"))
            logger.info("Added weather.fetched_at")

        conn.commit()

    # Create new tables (PredictionMatrix, ChampionshipPrediction)
    # create_all is safe — it skips existing tables
    Base.metadata.create_all(engine)
    logger.info("v2 migration complete")


def migrate_v3() -> None:
    """Run v3 migration: add live_prediction_snapshots table.

    Safe to run multiple times — create_all skips existing tables.
    """
    logger.info("Running v3 migration...")
    Base.metadata.create_all(engine)
    logger.info("v3 migration complete — live_prediction_snapshots table ready")


if __name__ == "__main__":
    create_all_tables()
