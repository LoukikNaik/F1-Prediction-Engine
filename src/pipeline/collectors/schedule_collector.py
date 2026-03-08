"""Fetch F1 race calendar with session timing for stage auto-detection."""

import requests
from sqlalchemy.orm import Session

from src.database.models import Circuit, Race
from src.utils.logger import logger


# Jolpica-F1 API (Ergast successor)
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def fetch_season_schedule(season: int) -> list[dict]:
    """Fetch the race schedule for a season from Jolpica API.

    Args:
        season: Season year.

    Returns:
        List of race info dicts.
    """
    url = f"{JOLPICA_BASE}/{season}.json"
    params = {"limit": 30}

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        races = data["MRData"]["RaceTable"]["Races"]
        logger.info(f"Fetched {len(races)} races for {season} season")
        return races
    except Exception as e:
        logger.error(f"Failed to fetch {season} schedule: {e}")
        return []


def populate_race_schedule(session: Session, season: int) -> int:
    """Fetch and store race schedule with timing info.

    Updates existing Race rows with qualifying/sprint/race times,
    or creates new Race rows if they don't exist.

    Args:
        session: Database session.
        season: Season year.

    Returns:
        Number of races updated/created.
    """
    races_data = fetch_season_schedule(season)
    count = 0

    for race_data in races_data:
        round_num = int(race_data["round"])
        race_name = race_data["raceName"]
        race_date = race_data.get("date")
        race_time = race_data.get("time", "").replace("Z", "").rstrip(":") or None

        # Qualifying info
        quali = race_data.get("Qualifying", {})
        quali_date = quali.get("date")
        quali_time = quali.get("time", "").replace("Z", "").rstrip(":") or None

        # Sprint info
        sprint = race_data.get("Sprint", {})
        sprint_date = sprint.get("date")
        sprint_time = sprint.get("time", "").replace("Z", "").rstrip(":") or None
        is_sprint = 1 if sprint_date else 0

        # Find or create race
        race = (
            session.query(Race)
            .filter(Race.season == season, Race.round == round_num)
            .first()
        )

        if race:
            # Update timing columns
            race.qualifying_date = quali_date
            race.qualifying_time = quali_time
            race.sprint_date = sprint_date
            race.sprint_time = sprint_time
            race.race_time = race_time
            race.is_sprint_weekend = is_sprint
            if race_date and not race.race_date:
                race.race_date = race_date
        else:
            # Find circuit
            circuit_data = race_data.get("Circuit", {})
            circuit_id = None
            if circuit_data.get("circuitId"):
                circuit = (
                    session.query(Circuit)
                    .filter(Circuit.ergast_id == circuit_data["circuitId"])
                    .first()
                )
                if circuit:
                    circuit_id = circuit.id

            race = Race(
                season=season,
                round=round_num,
                race_name=race_name,
                race_date=race_date,
                circuit_id=circuit_id,
                qualifying_date=quali_date,
                qualifying_time=quali_time,
                sprint_date=sprint_date,
                sprint_time=sprint_time,
                race_time=race_time,
                is_sprint_weekend=is_sprint,
            )
            session.add(race)

        count += 1

    logger.info(f"Updated {count} races for {season} season schedule")
    return count
