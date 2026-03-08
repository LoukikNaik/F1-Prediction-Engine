"""Collect historical F1 data from the Jolpica-F1 API (Ergast successor)."""

import time

import requests

from src.utils.logger import logger



BASE_URL = "https://api.jolpi.ca/ergast/f1"


def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """Make a rate-limited GET request to the Jolpica API with retry."""
    url = f"{BASE_URL}/{endpoint}.json"
    if params is None:
        params = {}
    params.setdefault("limit", 1000)

    for attempt in range(retries):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            logger.warning(f"Rate limited, waiting {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        time.sleep(1.0)  # Respect rate limits (200/hr)
        return resp.json()["MRData"]

    resp.raise_for_status()
    return {}


def get_all_races(season: int) -> list[dict]:
    """Get all races for a season."""
    data = _get(f"{season}")
    return data.get("RaceTable", {}).get("Races", [])


def get_race_results(season: int, round_num: int) -> list[dict]:
    """Get results for a specific race."""
    data = _get(f"{season}/{round_num}/results")
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0].get("Results", [])
    return []


def get_season_results(season: int) -> list[dict]:
    """Get all race results for a full season, handling pagination."""
    all_races: dict[str, dict] = {}
    offset = 0
    limit = 100

    while True:
        data = _get(f"{season}/results", params={"limit": limit, "offset": offset})
        races = data.get("RaceTable", {}).get("Races", [])
        total = int(data.get("total", 0))

        for race in races:
            round_key = race["round"]
            if round_key not in all_races:
                all_races[round_key] = race
            else:
                all_races[round_key].setdefault("Results", []).extend(
                    race.get("Results", [])
                )

        offset += limit
        if offset >= total or not races:
            break

    return list(all_races.values())


def get_qualifying_results(season: int, round_num: int) -> list[dict]:
    """Get qualifying results for a specific race."""
    data = _get(f"{season}/{round_num}/qualifying")
    races = data.get("RaceTable", {}).get("Races", [])
    if races:
        return races[0].get("QualifyingResults", [])
    return []


def get_driver_standings(season: int, round_num: int | None = None) -> list[dict]:
    """Get driver standings for a season (optionally after a specific round)."""
    endpoint = f"{season}/driverStandings"
    if round_num:
        endpoint = f"{season}/{round_num}/driverStandings"
    data = _get(endpoint)
    standings_lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    if standings_lists:
        return standings_lists[0].get("DriverStandings", [])
    return []


def get_constructor_standings(season: int, round_num: int | None = None) -> list[dict]:
    """Get constructor standings for a season."""
    endpoint = f"{season}/constructorStandings"
    if round_num:
        endpoint = f"{season}/{round_num}/constructorStandings"
    data = _get(endpoint)
    standings_lists = data.get("StandingsTable", {}).get("StandingsLists", [])
    if standings_lists:
        return standings_lists[0].get("ConstructorStandings", [])
    return []


def get_all_circuits() -> list[dict]:
    """Get all circuits."""
    data = _get("circuits", params={"limit": 100})
    return data.get("CircuitTable", {}).get("Circuits", [])


def get_all_drivers(season: int) -> list[dict]:
    """Get all drivers for a season."""
    data = _get(f"{season}/drivers")
    return data.get("DriverTable", {}).get("Drivers", [])


def get_all_constructors(season: int) -> list[dict]:
    """Get all constructors for a season."""
    data = _get(f"{season}/constructors")
    return data.get("ConstructorTable", {}).get("Constructors", [])
