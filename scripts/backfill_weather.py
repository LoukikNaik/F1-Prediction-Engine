"""Backfill historical weather data for races from 2022-2025.

Uses Open-Meteo archive API (free, no key needed) to fetch actual weather
for past race dates and store in the Weather table.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

from src.database.connection import get_session
from src.database.models import Circuit, Race, Weather
from src.pipeline.collectors.weather_collector import get_historical_weather
from src.utils.logger import logger


def backfill_weather(min_season: int = 2022, max_season: int = 2025) -> None:
    """Fetch and store historical weather for past races.

    Args:
        min_season: Earliest season to backfill.
        max_season: Latest season to backfill.
    """
    with get_session() as session:
        races = (
            session.query(Race)
            .filter(Race.season >= min_season, Race.season <= max_season)
            .order_by(Race.season, Race.round)
            .all()
        )

        logger.info(f"Found {len(races)} races to backfill weather for ({min_season}-{max_season})")

        added = 0
        skipped = 0

        for race in races:
            # Skip if weather already exists
            existing = (
                session.query(Weather)
                .filter(Weather.race_id == race.id)
                .first()
            )
            if existing:
                skipped += 1
                continue

            # Need circuit coords and race date
            circuit = session.query(Circuit).filter(Circuit.id == race.circuit_id).first()
            if not circuit or not circuit.latitude or not circuit.longitude:
                logger.warning(f"No circuit coords for {race.race_name} ({race.season} R{race.round})")
                continue
            if not race.race_date:
                logger.warning(f"No race date for {race.race_name} ({race.season} R{race.round})")
                continue

            weather_data = get_historical_weather(
                circuit.latitude, circuit.longitude, race.race_date
            )

            if weather_data:
                weather_record = Weather(
                    race_id=race.id,
                    forecast_type="race_day",
                    fetched_at=datetime.utcnow(),
                    **weather_data,
                )
                session.add(weather_record)
                added += 1
                logger.info(
                    f"Added weather for {race.race_name} ({race.season} R{race.round}): "
                    f"temp={weather_data['air_temp']:.1f}C, rain={weather_data['rain_prob']:.0%}"
                )
            else:
                logger.warning(f"No weather data for {race.race_name} ({race.season} R{race.round})")

            # Rate limit: Open-Meteo allows ~600 req/min, be polite
            time.sleep(0.2)

    logger.info(f"Weather backfill complete: {added} added, {skipped} skipped (already existed)")


if __name__ == "__main__":
    backfill_weather()
