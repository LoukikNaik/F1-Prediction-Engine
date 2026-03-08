"""Live race data collector using OpenF1 API."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import requests

from config.settings import OPENF1_BASE_URL
from src.utils.logger import logger


@dataclass
class DriverLiveState:
    """Live state for a single driver during a race."""

    driver_number: int
    name_acronym: str
    position: int | None = None
    lap_number: int = 0
    lap_times: list[float] = field(default_factory=list)
    sector_times: list[float] = field(default_factory=list)
    tire_compound: str | None = None
    tire_age: int = 0
    pit_stops: int = 0
    retired: bool = False


@dataclass
class RaceControlEvent:
    """A race control message (safety car, flags, etc.)."""

    category: str  # "SafetyCar", "Flag", etc.
    flag: str | None = None  # "GREEN", "YELLOW", "RED", "CHEQUERED"
    message: str = ""
    lap_number: int = 0


@dataclass
class LiveRaceState:
    """Aggregated live state for the entire race."""

    session_key: int
    drivers: dict[int, DriverLiveState] = field(default_factory=dict)
    leader_lap: int = 0
    total_laps: int = 0
    safety_car_active: bool = False
    virtual_safety_car: bool = False
    red_flag: bool = False
    race_started: bool = False
    race_finished: bool = False
    race_control_events: list[RaceControlEvent] = field(default_factory=list)
    weather: dict | None = None


class OpenF1Client:
    """Client for the OpenF1 API (free, no key needed).

    Follows the same resilient pattern as weather_collector.py:
    never crashes, returns empty data on failure.
    """

    def __init__(self, base_url: str = OPENF1_BASE_URL) -> None:
        self.base_url = base_url
        self.session = requests.Session()

    def _get(
        self, endpoint: str, params: dict | None = None, retries: int = 3
    ) -> list[dict]:
        """Make a GET request to OpenF1. Returns [] on failure."""
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(retries):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else []
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"OpenF1 rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                logger.warning(f"OpenF1 HTTP error on {endpoint}: {e}")
                return []
            except Exception as e:
                logger.warning(f"OpenF1 request failed ({endpoint}, attempt {attempt + 1}): {e}")
                if attempt < retries - 1:
                    time.sleep(1)
        return []

    def get_session_key(self, season: int, round_num: int) -> int | None:
        """Find the race session key for a given season and round.

        Args:
            season: Season year.
            round_num: Round number.

        Returns:
            Session key integer or None if not found.
        """
        # OpenF1 uses meeting_key and session_name, not round directly.
        # First get meetings for the season, then find the Race session.
        meetings = self._get("meetings", {"year": season})
        if not meetings or round_num > len(meetings):
            return None

        # Meetings are ordered by date; round_num maps to index
        meeting = meetings[round_num - 1] if round_num <= len(meetings) else None
        if not meeting:
            return None

        meeting_key = meeting.get("meeting_key")
        sessions = self._get("sessions", {
            "meeting_key": meeting_key,
            "session_name": "Race",
        })
        if sessions:
            return sessions[0].get("session_key")
        return None

    def get_drivers(self, session_key: int) -> list[dict]:
        """Get driver list for a session.

        Returns:
            List of driver dicts with driver_number, name_acronym, etc.
        """
        return self._get("drivers", {"session_key": session_key})

    def get_latest_positions(self, session_key: int) -> dict[int, int]:
        """Get latest position for each driver.

        Returns:
            Mapping of driver_number -> position.
        """
        data = self._get("position", {"session_key": session_key})
        if not data:
            return {}

        # Take the latest entry per driver
        positions: dict[int, int] = {}
        for entry in data:
            dn = entry.get("driver_number")
            pos = entry.get("position")
            if dn is not None and pos is not None:
                positions[dn] = pos
        return positions

    def get_latest_laps(self, session_key: int) -> dict[int, dict]:
        """Get latest lap data for each driver.

        Returns:
            Mapping of driver_number -> {lap_number, lap_duration, ...}.
        """
        data = self._get("laps", {"session_key": session_key})
        if not data:
            return {}

        laps: dict[int, dict] = {}
        for entry in data:
            dn = entry.get("driver_number")
            if dn is not None:
                laps[dn] = entry
        return laps

    def get_lap_times(self, session_key: int, driver_number: int) -> list[float]:
        """Get all lap times for a specific driver.

        Returns:
            List of lap durations in seconds.
        """
        data = self._get("laps", {
            "session_key": session_key,
            "driver_number": driver_number,
        })
        times = []
        for entry in data:
            duration = entry.get("lap_duration")
            if duration is not None and duration > 0:
                times.append(float(duration))
        return times

    def get_current_stints(self, session_key: int) -> dict[int, dict]:
        """Get current tire stint data for each driver.

        Returns:
            Mapping of driver_number -> {compound, stint_number, lap_start, lap_end, ...}.
        """
        data = self._get("stints", {"session_key": session_key})
        if not data:
            return {}

        # Take the latest stint per driver
        stints: dict[int, dict] = {}
        for entry in data:
            dn = entry.get("driver_number")
            if dn is not None:
                stints[dn] = entry
        return stints

    def get_pit_stops(self, session_key: int) -> dict[int, int]:
        """Get pit stop counts per driver.

        Returns:
            Mapping of driver_number -> pit stop count.
        """
        data = self._get("pit", {"session_key": session_key})
        counts: dict[int, int] = {}
        for entry in data:
            dn = entry.get("driver_number")
            if dn is not None:
                counts[dn] = counts.get(dn, 0) + 1
        return counts

    def get_race_control(self, session_key: int) -> list[RaceControlEvent]:
        """Get race control messages.

        Returns:
            List of RaceControlEvent objects.
        """
        data = self._get("race_control", {"session_key": session_key})
        events = []
        for entry in data:
            events.append(RaceControlEvent(
                category=entry.get("category", ""),
                flag=entry.get("flag"),
                message=entry.get("message", ""),
                lap_number=entry.get("lap_number", 0),
            ))
        return events

    def get_weather(self, session_key: int) -> dict | None:
        """Get latest weather data for the session.

        Returns:
            Dict with air_temp, track_temp, humidity, rain, wind_speed or None.
        """
        data = self._get("weather", {"session_key": session_key})
        if not data:
            return None
        latest = data[-1]
        return {
            "air_temp": latest.get("air_temperature"),
            "track_temp": latest.get("track_temperature"),
            "humidity": latest.get("humidity"),
            "rainfall": latest.get("rainfall", 0),
            "wind_speed": latest.get("wind_speed"),
        }

    def is_race_started(self, events: list[RaceControlEvent]) -> bool:
        """Check if the race has started based on race control events."""
        for event in events:
            if event.flag == "GREEN" or "GREEN LIGHT" in event.message.upper():
                return True
        return False

    def is_race_finished(self, events: list[RaceControlEvent]) -> bool:
        """Check if the race has finished based on race control events."""
        for event in events:
            if event.flag == "CHEQUERED" or "CHEQUERED" in event.message.upper():
                return True
        return False


def build_live_race_state(
    client: OpenF1Client,
    session_key: int,
    total_laps: int = 0,
) -> LiveRaceState:
    """Aggregate all OpenF1 endpoints into a single LiveRaceState.

    Gracefully handles partial data — missing endpoints don't crash
    the state build, they just leave fields empty/default.

    Args:
        client: OpenF1Client instance.
        session_key: OpenF1 session key for the race.
        total_laps: Scheduled race distance in laps (0 if unknown).

    Returns:
        Populated LiveRaceState dataclass.
    """
    state = LiveRaceState(session_key=session_key, total_laps=total_laps)

    # Get race control events first (needed for flags)
    events = client.get_race_control(session_key)
    state.race_control_events = events
    state.race_started = client.is_race_started(events)
    state.race_finished = client.is_race_finished(events)

    # Check for active safety car / VSC / red flag from events
    for event in reversed(events):
        msg = event.message.upper()
        if "SAFETY CAR" in msg and "VIRTUAL" not in msg and "IN THIS LAP" not in msg:
            state.safety_car_active = True
            break
        if "VIRTUAL SAFETY CAR" in msg:
            state.virtual_safety_car = True
            break
        if event.flag == "RED":
            state.red_flag = True
            break
        if event.flag == "GREEN" or "SAFETY CAR IN" in msg:
            # SC/VSC ended
            break

    # Get driver list
    driver_data = client.get_drivers(session_key)
    driver_acronyms: dict[int, str] = {}
    for d in driver_data:
        dn = d.get("driver_number")
        acronym = d.get("name_acronym", f"D{dn}")
        if dn is not None:
            driver_acronyms[dn] = acronym
            state.drivers[dn] = DriverLiveState(
                driver_number=dn,
                name_acronym=acronym,
            )

    # Positions
    positions = client.get_latest_positions(session_key)
    for dn, pos in positions.items():
        if dn in state.drivers:
            state.drivers[dn].position = pos

    # Latest laps (for leader lap count)
    latest_laps = client.get_latest_laps(session_key)
    for dn, lap_data in latest_laps.items():
        if dn in state.drivers:
            lap_num = lap_data.get("lap_number", 0)
            state.drivers[dn].lap_number = lap_num
            if lap_num > state.leader_lap:
                state.leader_lap = lap_num

    # All lap times per driver
    for dn in list(state.drivers.keys()):
        times = client.get_lap_times(session_key, dn)
        state.drivers[dn].lap_times = times

    # Tire stints
    stints = client.get_current_stints(session_key)
    for dn, stint_data in stints.items():
        if dn in state.drivers:
            state.drivers[dn].tire_compound = stint_data.get("compound")
            lap_start = stint_data.get("lap_start", 0) or 0
            state.drivers[dn].tire_age = max(0, state.drivers[dn].lap_number - lap_start)

    # Pit stops
    pit_counts = client.get_pit_stops(session_key)
    for dn, count in pit_counts.items():
        if dn in state.drivers:
            state.drivers[dn].pit_stops = count

    # Weather
    state.weather = client.get_weather(session_key)

    # Detect retired drivers (position gone or no recent laps while others advance)
    if state.leader_lap > 3:
        for dn, ds in state.drivers.items():
            if ds.lap_number < state.leader_lap - 3 and ds.position is None:
                ds.retired = True

    logger.debug(
        f"Live state built: lap {state.leader_lap}/{state.total_laps}, "
        f"{len(state.drivers)} drivers, SC={state.safety_car_active}"
    )
    return state
