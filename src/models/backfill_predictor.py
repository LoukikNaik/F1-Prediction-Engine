"""Backfill live lap predictions from historical FastF1 race data.

Replays a completed race lap-by-lap, feeding masked state through
the existing LivePredictor.update() pipeline. At each lap the model
sees only data up to that lap — it does not know the final result.
"""

from __future__ import annotations

from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from config.settings import (
    BACKFILL_DEFAULT_LAP_INTERVAL,
    FASTF1_CACHE_DIR,
    LIVE_PREDICTIONS_DIR,
)
from src.database.connection import get_session
from src.database.models import Driver, LivePredictionSnapshot
from src.database.queries import get_race
from src.models.live_predictor import LivePredictor
from src.pipeline.collectors.live_collector import (
    DriverLiveState,
    LiveRaceState,
    RaceControlEvent,
)
from src.utils.logger import logger


class FastF1RaceReplay:
    """Loads a completed FastF1 session and produces masked LiveRaceState snapshots.

    Args:
        season: Season year (e.g. 2025).
        round_num: Round number within the season.
    """

    def __init__(self, season: int, round_num: int) -> None:
        self.season = season
        self.round_num = round_num
        self.session = None
        self.laps_df: pd.DataFrame | None = None
        self.rc_messages: pd.DataFrame | None = None
        self.total_laps: int = 0

        # FastF1 abbreviation -> driver number
        self.code_to_number: dict[str, int] = {}
        # FastF1 abbreviation -> our DB full_name
        self.code_to_name: dict[str, str] = {}

    def load(self) -> None:
        """Download / load the race session from FastF1 cache."""
        FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))

        logger.info(f"Loading FastF1 session: {self.season} R{self.round_num} Race")
        self.session = fastf1.get_session(self.season, self.round_num, "R")
        self.session.load()

        self.laps_df = self.session.laps
        self.rc_messages = getattr(self.session, "race_control_messages", None)
        self.total_laps = int(self.laps_df["LapNumber"].max())

        self._build_driver_mappings()
        logger.info(
            f"Loaded {self.total_laps} laps, "
            f"{len(self.code_to_name)} drivers mapped"
        )

    def _build_driver_mappings(self) -> None:
        """Map FastF1 Abbreviation -> DB Driver.code -> full_name."""
        # Get unique drivers from FastF1 laps
        f1_drivers = self.laps_df[["Driver", "DriverNumber"]].drop_duplicates()

        # Build code -> number from FastF1
        for _, row in f1_drivers.iterrows():
            abbrev = str(row["Driver"])
            try:
                num = int(row["DriverNumber"])
            except (ValueError, TypeError):
                continue
            self.code_to_number[abbrev] = num

        # Match abbreviations to DB driver.code -> full_name
        with get_session() as session:
            for abbrev in self.code_to_number:
                driver = (
                    session.query(Driver)
                    .filter(Driver.code == abbrev)
                    .first()
                )
                if driver:
                    self.code_to_name[abbrev] = driver.full_name
                else:
                    logger.debug(f"No DB match for FastF1 abbreviation '{abbrev}'")

    def get_state_at_lap(self, target_lap: int) -> LiveRaceState:
        """Build a masked LiveRaceState showing only data up to target_lap.

        Args:
            target_lap: The lap number to mask up to (inclusive).

        Returns:
            LiveRaceState as if we were watching the race live at that lap.
        """
        masked = self.laps_df[self.laps_df["LapNumber"] <= target_lap]
        drivers_at_lap = masked.groupby("Driver")

        state = LiveRaceState(
            session_key=0,  # not used during backfill
            leader_lap=target_lap,
            total_laps=self.total_laps,
            race_started=True,
            race_finished=(target_lap >= self.total_laps),
        )

        # Determine leader's actual lap for retirement detection
        leader_max_lap = target_lap

        for abbrev, group in drivers_at_lap:
            abbrev = str(abbrev)
            driver_num = self.code_to_number.get(abbrev)
            if driver_num is None:
                continue

            # Sort by lap number
            group = group.sort_values("LapNumber")
            last_row = group.iloc[-1]

            # Lap times: convert timedelta to seconds, skip NaT
            lap_times: list[float] = []
            for _, lap_row in group.iterrows():
                lt = lap_row["LapTime"]
                if pd.notna(lt):
                    lap_times.append(lt.total_seconds())

            # Position at target lap
            position = None
            if pd.notna(last_row.get("Position")):
                position = int(last_row["Position"])

            # Tire info
            tire_compound = None
            if pd.notna(last_row.get("Compound")):
                tire_compound = str(last_row["Compound"])

            tire_age = 0
            if pd.notna(last_row.get("TyreLife")):
                tire_age = int(last_row["TyreLife"])

            # Pit stops = unique stints minus 1
            stint_col = "Stint"
            if stint_col in group.columns:
                n_stints = group[stint_col].dropna().nunique()
                pit_stops = max(0, n_stints - 1)
            else:
                pit_stops = 0

            # Retired detection: driver's last lap is 3+ behind leader
            driver_last_lap = int(last_row["LapNumber"])
            retired = False
            if driver_last_lap < leader_max_lap - 2:
                retired = True

            state.drivers[driver_num] = DriverLiveState(
                driver_number=driver_num,
                name_acronym=abbrev,
                position=position,
                lap_number=driver_last_lap,
                lap_times=lap_times,
                tire_compound=tire_compound,
                tire_age=tire_age,
                pit_stops=pit_stops,
                retired=retired,
            )

        # Race control events up to target_lap
        if self.rc_messages is not None and len(self.rc_messages) > 0:
            rc_df = self.rc_messages
            # Filter to events at or before target lap
            if "Lap" in rc_df.columns:
                rc_filtered = rc_df[rc_df["Lap"] <= target_lap]
            else:
                rc_filtered = rc_df

            for _, row in rc_filtered.iterrows():
                category = str(row.get("Category", ""))
                flag = row.get("Flag")
                if pd.isna(flag):
                    flag = None
                else:
                    flag = str(flag)
                message = str(row.get("Message", ""))
                lap_num = int(row.get("Lap", 0)) if pd.notna(row.get("Lap")) else 0

                state.race_control_events.append(RaceControlEvent(
                    category=category,
                    flag=flag,
                    message=message,
                    lap_number=lap_num,
                ))

            # Determine SC / VSC / red flag from events (walk in reverse)
            for event in reversed(state.race_control_events):
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
                    break

        return state

    def get_prediction_laps(
        self, interval: int = BACKFILL_DEFAULT_LAP_INTERVAL,
    ) -> list[int]:
        """Return lap numbers at which to generate predictions.

        Always includes lap 1 and the final lap.

        Args:
            interval: Gap between prediction laps.

        Returns:
            Sorted list of lap numbers.
        """
        laps = set(range(1, self.total_laps + 1, interval))
        laps.add(1)
        laps.add(self.total_laps)
        return sorted(laps)


def _clear_existing_live_data(season: int, round_num: int, race_id: int | None) -> None:
    """Delete existing backfill artefacts for a race.

    Removes per-lap parquets, evolution parquet, and DB LivePredictionSnapshot rows.
    """
    # Parquet files
    pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
    for f in LIVE_PREDICTIONS_DIR.glob(pattern):
        f.unlink()
        logger.debug(f"Deleted {f}")

    evo_path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
    if evo_path.exists():
        evo_path.unlink()
        logger.debug(f"Deleted {evo_path}")

    # DB rows
    if race_id is not None:
        try:
            with get_session() as session:
                deleted = (
                    session.query(LivePredictionSnapshot)
                    .filter(LivePredictionSnapshot.race_id == race_id)
                    .delete()
                )
                logger.info(f"Deleted {deleted} DB snapshot rows for race_id={race_id}")
        except Exception as e:
            logger.warning(f"Failed to clear DB snapshots: {e}")


def run_backfill(
    season: int,
    round_num: int,
    lap_interval: int = BACKFILL_DEFAULT_LAP_INTERVAL,
    n_sims: int = 5_000,
    clear_existing: bool = False,
) -> bool:
    """Replay a completed race and generate lap-by-lap predictions.

    Args:
        season: Season year.
        round_num: Round number.
        lap_interval: Laps between predictions.
        n_sims: Monte Carlo simulations per prediction.
        clear_existing: Delete existing live data before backfilling.

    Returns:
        True if backfill completed successfully.
    """
    # 1. Load FastF1 session
    replay = FastF1RaceReplay(season, round_num)
    try:
        replay.load()
    except Exception as e:
        logger.error(f"Failed to load FastF1 session for {season} R{round_num}: {e}")
        return False

    # 2. Initialize LivePredictor
    predictor = LivePredictor(season, round_num, n_simulations=n_sims)
    if not predictor.initialize():
        logger.error(f"No pre-race matrix for {season} R{round_num} — run predictions first")
        return False

    # 3. Set number_to_name from replay mappings (bypasses OpenF1)
    for abbrev, driver_num in replay.code_to_number.items():
        full_name = replay.code_to_name.get(abbrev)
        if full_name and full_name in predictor.driver_names:
            predictor.number_to_name[driver_num] = full_name

    matched = len(predictor.number_to_name)
    logger.info(f"Mapped {matched}/{len(predictor.driver_names)} drivers from FastF1")

    # 4. Optionally clear existing data
    if clear_existing:
        _clear_existing_live_data(season, round_num, predictor.race_id)

    # 5. Run predictions at each sampled lap
    prediction_laps = replay.get_prediction_laps(lap_interval)
    logger.info(
        f"Backfilling {len(prediction_laps)} laps for {season} R{round_num} "
        f"(total: {replay.total_laps})"
    )

    for lap in prediction_laps:
        state = replay.get_state_at_lap(lap)
        predictor.update(state)

    # 6. Save evolution parquet
    evolution = predictor.get_prediction_evolution()
    if not evolution.empty:
        LIVE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        evo_path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
        evolution.to_parquet(evo_path, index=False)
        logger.info(f"Saved prediction evolution to {evo_path}")

    logger.info(f"Backfill complete for {season} R{round_num}")
    return True
