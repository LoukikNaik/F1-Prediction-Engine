"""Collect weather data from Open-Meteo API (free, no API key needed)."""

from datetime import datetime

import requests

from config.settings import OPEN_METEO_BASE_URL
from src.utils.logger import logger

# Open-Meteo historical weather endpoint
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def get_weather_forecast(
    latitude: float, longitude: float, race_date: str
) -> dict | None:
    """Get weather forecast for a circuit location and date.

    Args:
        latitude: Circuit latitude.
        longitude: Circuit longitude.
        race_date: Date string in YYYY-MM-DD format.

    Returns:
        Dict with weather data or None if unavailable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max,relative_humidity_2m_mean",
        "start_date": race_date,
        "end_date": race_date,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        if not daily or not daily.get("temperature_2m_max"):
            return None

        return {
            "air_temp": (daily["temperature_2m_max"][0] + daily["temperature_2m_min"][0]) / 2,
            "track_temp": daily["temperature_2m_max"][0] + 15,  # rough track temp estimate
            "rain_prob": daily["precipitation_probability_max"][0] / 100.0,
            "wind_speed": daily["wind_speed_10m_max"][0],
            "humidity": daily["relative_humidity_2m_mean"][0],
        }
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return None


def get_session_weather(
    latitude: float, longitude: float, date: str, time: str | None = None
) -> dict | None:
    """Get hourly weather for a specific session time.

    Args:
        latitude: Circuit latitude.
        longitude: Circuit longitude.
        date: Date string in YYYY-MM-DD format.
        time: Time string in HH:MM:SS format (optional).

    Returns:
        Dict with weather data or None if unavailable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation_probability,wind_speed_10m,relative_humidity_2m",
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        hourly = data.get("hourly", {})
        if not hourly or not hourly.get("temperature_2m"):
            return None

        # Pick the hour closest to session time, or 14:00 (typical race start)
        target_hour = 14
        if time:
            try:
                target_hour = int(time.split(":")[0])
            except (ValueError, IndexError):
                pass

        idx = min(target_hour, len(hourly["temperature_2m"]) - 1)

        air_temp = hourly["temperature_2m"][idx]
        return {
            "air_temp": air_temp,
            "track_temp": air_temp + 15,
            "rain_prob": hourly["precipitation_probability"][idx] / 100.0,
            "wind_speed": hourly["wind_speed_10m"][idx],
            "humidity": hourly["relative_humidity_2m"][idx],
        }
    except Exception as e:
        logger.warning(f"Session weather fetch failed: {e}")
        return None


def get_race_weekend_weather(
    latitude: float,
    longitude: float,
    quali_date: str | None,
    race_date: str,
    sprint_date: str | None = None,
) -> dict[str, dict | None]:
    """Get weather for all sessions in a race weekend.

    Args:
        latitude: Circuit latitude.
        longitude: Circuit longitude.
        quali_date: Qualifying date (YYYY-MM-DD).
        race_date: Race date (YYYY-MM-DD).
        sprint_date: Sprint date if applicable (YYYY-MM-DD).

    Returns:
        Dict mapping forecast_type to weather data.
    """
    results = {}

    # Pre-weekend forecast (race day)
    results["pre_weekend"] = get_weather_forecast(latitude, longitude, race_date)

    # Qualifying day
    if quali_date:
        results["qualifying_day"] = get_session_weather(latitude, longitude, quali_date)

    # Race day (hourly)
    results["race_day"] = get_session_weather(latitude, longitude, race_date)

    # Sprint day
    if sprint_date:
        results["sprint_day"] = get_session_weather(latitude, longitude, sprint_date)

    return results


def get_historical_weather(
    latitude: float, longitude: float, date: str
) -> dict | None:
    """Get historical weather data from Open-Meteo archive API.

    Args:
        latitude: Circuit latitude.
        longitude: Circuit longitude.
        date: Date string in YYYY-MM-DD format.

    Returns:
        Dict with weather data or None if unavailable.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,relative_humidity_2m_mean",
        "start_date": date,
        "end_date": date,
        "timezone": "auto",
    }

    try:
        resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        daily = data.get("daily", {})
        if not daily or not daily.get("temperature_2m_max"):
            return None

        precip = daily.get("precipitation_sum", [0])[0] or 0
        return {
            "air_temp": (daily["temperature_2m_max"][0] + daily["temperature_2m_min"][0]) / 2,
            "track_temp": daily["temperature_2m_max"][0] + 15,
            "rain_prob": min(1.0, precip / 5.0),  # rough estimate: 5mm+ = 100% wet
            "wind_speed": daily["wind_speed_10m_max"][0],
            "humidity": daily.get("relative_humidity_2m_mean", [50])[0],
        }
    except Exception as e:
        logger.warning(f"Historical weather fetch failed for {date}: {e}")
        return None
