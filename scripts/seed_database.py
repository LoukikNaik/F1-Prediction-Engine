"""Seed the database with historical F1 data from the Jolpica-F1 API.

Usage: python scripts/seed_database.py [--start-year 2000] [--end-year 2025]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.connection import get_session
from src.database.migrations import create_all_tables
from src.database.models import Circuit, Driver, Race, Result, Standing, Team
from src.pipeline.collectors import ergast_collector as ergast
from src.utils.logger import logger


def parse_position(pos_str: str | None) -> int | None:
    """Parse position string to int, handling DNF/DNS etc."""
    if pos_str is None:
        return None
    try:
        return int(pos_str)
    except (ValueError, TypeError):
        return None


def parse_time_to_ms(time_str: str | None) -> float | None:
    """Parse qualifying time string (M:SS.mmm) to milliseconds."""
    if not time_str:
        return None
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return (minutes * 60 + seconds) * 1000
        return float(parts[0]) * 1000
    except (ValueError, TypeError):
        return None


def seed_circuits(session) -> dict[str, int]:
    """Load all circuits into the database. Returns ergast_id -> db_id mapping."""
    logger.info("Loading circuits...")
    circuits_data = ergast.get_all_circuits()
    mapping = {}

    for c in circuits_data:
        ergast_id = c["circuitId"]
        existing = session.query(Circuit).filter(Circuit.ergast_id == ergast_id).first()
        if existing:
            mapping[ergast_id] = existing.id
            continue

        location = c.get("Location", {})
        circuit = Circuit(
            name=c["circuitName"],
            country=location.get("country", ""),
            city=location.get("locality", ""),
            latitude=float(location.get("lat", 0)),
            longitude=float(location.get("long", 0)),
            ergast_id=ergast_id,
        )
        session.add(circuit)
        session.flush()
        mapping[ergast_id] = circuit.id

    session.commit()
    logger.info(f"Loaded {len(mapping)} circuits")
    return mapping


def seed_season(session, season: int, circuit_map: dict) -> None:
    """Load a full season's data: drivers, teams, races, results, standings."""
    logger.info(f"Loading season {season}...")

    # Get all race results for the season in one call
    races_data = ergast.get_season_results(season)
    if not races_data:
        logger.warning(f"No data for season {season}")
        return

    driver_map: dict[str, int] = {}
    team_map: dict[str, int] = {}

    for race_data in races_data:
        round_num = int(race_data["round"])
        race_name = race_data["raceName"]
        race_date = race_data.get("date", "")
        circuit_ergast_id = race_data["Circuit"]["circuitId"]

        # Ensure circuit exists
        circuit_id = circuit_map.get(circuit_ergast_id)
        if not circuit_id:
            location = race_data["Circuit"].get("Location", {})
            circuit = Circuit(
                name=race_data["Circuit"]["circuitName"],
                country=location.get("country", ""),
                city=location.get("locality", ""),
                latitude=float(location.get("lat", 0)),
                longitude=float(location.get("long", 0)),
                ergast_id=circuit_ergast_id,
            )
            session.add(circuit)
            session.flush()
            circuit_id = circuit.id
            circuit_map[circuit_ergast_id] = circuit_id

        # Create or get race
        race = (
            session.query(Race)
            .filter(Race.season == season, Race.round == round_num)
            .first()
        )
        if not race:
            race = Race(
                season=season,
                round=round_num,
                circuit_id=circuit_id,
                race_date=race_date,
                race_name=race_name,
            )
            session.add(race)
            session.flush()

        # Process results
        for result in race_data.get("Results", []):
            # Driver
            driver_info = result["Driver"]
            driver_ergast = driver_info["driverId"]
            if driver_ergast not in driver_map:
                existing = (
                    session.query(Driver)
                    .filter(Driver.ergast_id == driver_ergast)
                    .first()
                )
                if existing:
                    driver_map[driver_ergast] = existing.id
                else:
                    driver = Driver(
                        code=driver_info.get("code", ""),
                        full_name=f"{driver_info.get('givenName', '')} {driver_info.get('familyName', '')}",
                        nationality=driver_info.get("nationality", ""),
                        date_of_birth=driver_info.get("dateOfBirth", ""),
                        ergast_id=driver_ergast,
                    )
                    session.add(driver)
                    session.flush()
                    driver_map[driver_ergast] = driver.id

            # Team/Constructor
            constructor = result.get("Constructor", {})
            team_ergast = constructor.get("constructorId", "")
            if team_ergast and team_ergast not in team_map:
                existing = (
                    session.query(Team)
                    .filter(Team.ergast_id == team_ergast)
                    .first()
                )
                if existing:
                    team_map[team_ergast] = existing.id
                else:
                    team = Team(
                        name=constructor.get("name", ""),
                        ergast_id=team_ergast,
                    )
                    session.add(team)
                    session.flush()
                    team_map[team_ergast] = team.id

            # Check if result already exists
            existing_result = (
                session.query(Result)
                .filter(
                    Result.race_id == race.id,
                    Result.driver_id == driver_map[driver_ergast],
                )
                .first()
            )
            if existing_result:
                continue

            # Parse qualifying time from result
            quali_time = None
            q_data = result.get("Q1") or result.get("Q2") or result.get("Q3")
            if q_data:
                quali_time = parse_time_to_ms(q_data)

            result_obj = Result(
                race_id=race.id,
                driver_id=driver_map[driver_ergast],
                team_id=team_map.get(team_ergast),
                grid=parse_position(result.get("grid")),
                position=parse_position(result.get("position")),
                points=float(result.get("points", 0)),
                status=result.get("status", ""),
                fastest_lap_rank=parse_position(
                    result.get("FastestLap", {}).get("rank")
                ),
                qualifying_time_ms=quali_time,
                laps=parse_position(result.get("laps")),
                milliseconds=parse_position(
                    result.get("Time", {}).get("millis")
                ),
            )
            session.add(result_obj)

    session.commit()

    # Load driver standings (end of season)
    try:
        driver_standings = ergast.get_driver_standings(season)
        for ds in driver_standings:
            driver_ergast = ds["Driver"]["driverId"]
            if driver_ergast not in driver_map:
                continue
            standing = Standing(
                season=season,
                round=len(races_data),
                driver_id=driver_map[driver_ergast],
                points=float(ds.get("points", 0)),
                position=int(ds.get("position", 0)),
                standing_type="driver",
            )
            session.add(standing)

        constructor_standings = ergast.get_constructor_standings(season)
        for cs in constructor_standings:
            team_ergast = cs["Constructor"]["constructorId"]
            if team_ergast not in team_map:
                continue
            standing = Standing(
                season=season,
                round=len(races_data),
                team_id=team_map[team_ergast],
                points=float(cs.get("points", 0)),
                position=int(cs.get("position", 0)),
                standing_type="constructor",
            )
            session.add(standing)

        session.commit()
    except Exception as e:
        logger.warning(f"Could not load standings for {season}: {e}")

    logger.info(
        f"Season {season}: {len(races_data)} races loaded"
    )


def main():
    parser = argparse.ArgumentParser(description="Seed F1 database")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    args = parser.parse_args()

    create_all_tables()

    with get_session() as session:
        circuit_map = seed_circuits(session)

        for year in range(args.start_year, args.end_year + 1):
            try:
                seed_season(session, year, circuit_map)
            except Exception as e:
                logger.error(f"Failed to load season {year}: {e}")
                continue

    logger.info("Database seeding complete!")


if __name__ == "__main__":
    main()
