"""Live race predictor — adjusts Monte Carlo params from OpenF1 telemetry."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    LIVE_DATA_SOURCE,
    LIVE_MC_SIMULATIONS,
    LIVE_POLL_INTERVAL,
    LIVE_PREDICTIONS_DIR,
    PREDICTIONS_DIR,
)
from src.database.connection import get_session
from src.database.models import LivePredictionSnapshot
from src.database.queries import get_race
from src.models.monte_carlo import run_monte_carlo
from src.pipeline.collectors.live_collector import (
    LiveRaceState,
    OpenF1Client,
    build_live_race_state,
)
from src.utils.constants import GRID_SIZE, LIVE_MIN_LAPS_FOR_PACE
from src.utils.logger import logger


class LivePredictor:
    """Generates live predictions by re-running Monte Carlo with adjusted params.

    The ensemble model stays frozen — only the MC simulation inputs change
    based on live telemetry (current positions, pace deltas, incidents).
    """

    def __init__(
        self,
        season: int,
        round_num: int,
        n_simulations: int = LIVE_MC_SIMULATIONS,
    ) -> None:
        self.season = season
        self.round_num = round_num
        self.n_simulations = n_simulations

        # Loaded during initialize()
        self.base_matrix: pd.DataFrame | None = None
        self.base_strengths: np.ndarray | None = None
        self.driver_names: list[str] = []
        self.race_id: int | None = None

        # Maps OpenF1 driver_number -> driver_name (from base matrix)
        self.number_to_name: dict[int, str] = {}

        # Prediction history for evolution charts
        self.history: list[dict] = []

    def initialize(self) -> bool:
        """Load pre-race prediction matrix and build driver mappings.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        # Find the latest pre-race matrix
        matrix_path = self._find_base_matrix()
        if matrix_path is None:
            logger.error("No pre-race prediction matrix found — run `make predict-race` first")
            return False

        self.base_matrix = pd.read_parquet(matrix_path)
        self.driver_names = list(self.base_matrix.index)
        logger.info(f"Loaded base matrix from {matrix_path} ({len(self.driver_names)} drivers)")

        # Compute base strengths from the matrix probabilities
        pos_cols = [f"P{i}" for i in range(1, GRID_SIZE + 1)]
        available_cols = [c for c in pos_cols if c in self.base_matrix.columns]
        proba = self.base_matrix[available_cols].values
        positions = np.arange(1, len(available_cols) + 1)
        expected = (proba * positions).sum(axis=1)
        self.base_strengths = GRID_SIZE - expected  # higher = faster

        # Get race_id from DB
        with get_session() as session:
            race = get_race(session, self.season, self.round_num)
            if race:
                self.race_id = race.id

        return True

    def build_driver_map(self, client: OpenF1Client, session_key: int) -> None:
        """Build mapping from OpenF1 driver_number to our driver_name.

        Matches by 3-letter driver code (e.g., VER -> Max Verstappen).
        """
        openf1_drivers = client.get_drivers(session_key)

        # Build acronym -> driver_number lookup
        acronym_to_number: dict[str, int] = {}
        for d in openf1_drivers:
            acr = d.get("name_acronym", "")
            dn = d.get("driver_number")
            if acr and dn is not None:
                acronym_to_number[acr] = dn

        # Match our driver names to OpenF1 numbers via DB driver codes
        with get_session() as session:
            from src.database.models import Driver

            for name in self.driver_names:
                driver = (
                    session.query(Driver)
                    .filter(Driver.full_name.ilike(f"%{name}%"))
                    .first()
                )
                if driver and driver.code and driver.code in acronym_to_number:
                    self.number_to_name[acronym_to_number[driver.code]] = name

        logger.info(f"Matched {len(self.number_to_name)}/{len(self.driver_names)} drivers to OpenF1")

    def build_driver_map_from_scraped(self, state: LiveRaceState) -> None:
        """Build driver mapping from scraped data by fuzzy-matching names.

        For web scraping, driver_number is just an index — we match by
        comparing the scraped acronym against our DB driver codes.
        """
        with get_session() as session:
            from src.database.models import Driver

            for dn, ds in state.drivers.items():
                acr = ds.name_acronym.upper()
                # Try exact code match first
                driver = (
                    session.query(Driver)
                    .filter(Driver.code == acr)
                    .first()
                )
                if driver and driver.full_name in self.driver_names:
                    self.number_to_name[dn] = driver.full_name
                    continue

                # Try matching acronym against last name prefix
                for name in self.driver_names:
                    last_name = name.split()[-1].upper()
                    if last_name[:3] == acr or acr in last_name:
                        self.number_to_name[dn] = name
                        break

        logger.info(
            f"Matched {len(self.number_to_name)}/{len(self.driver_names)} "
            f"drivers from scraped data"
        )

    def update(self, live_state: LiveRaceState) -> pd.DataFrame:
        """Run a live prediction cycle.

        Translates live telemetry into MC parameters and runs simulation.

        Args:
            live_state: Current LiveRaceState from OpenF1.

        Returns:
            Updated probability matrix DataFrame.
        """
        n_drivers = len(self.driver_names)
        laps_completed = live_state.leader_lap
        total_laps = live_state.total_laps or 57  # default

        # --- 1. Driver strengths: blend base with live pace ---
        strengths = self.base_strengths.copy()
        live_weight = min(0.6, (laps_completed / total_laps) * 0.8)

        if laps_completed >= LIVE_MIN_LAPS_FOR_PACE:
            live_paces = self._compute_live_paces(live_state)
            if live_paces:
                # Normalize pace deltas to strength adjustments
                median_pace = np.median(list(live_paces.values()))
                for dn, pace in live_paces.items():
                    name = self.number_to_name.get(dn)
                    if name and name in self.driver_names:
                        idx = self.driver_names.index(name)
                        # Faster pace (lower time) = positive delta
                        pace_delta = (median_pace - pace) / median_pace * 10
                        strengths[idx] = (
                            (1 - live_weight) * self.base_strengths[idx]
                            + live_weight * (self.base_strengths[idx] + pace_delta)
                        )

        # --- 2. Grid positions: replace with current race positions ---
        grid_positions = np.full(n_drivers, n_drivers, dtype=int)
        for dn, ds in live_state.drivers.items():
            name = self.number_to_name.get(dn)
            if name and name in self.driver_names and ds.position is not None:
                idx = self.driver_names.index(name)
                grid_positions[idx] = ds.position

        # --- 3. DNF probabilities ---
        laps_remaining_frac = max(0, (total_laps - laps_completed)) / total_laps
        dnf_probs = np.full(n_drivers, 0.03 * laps_remaining_frac)

        for dn, ds in live_state.drivers.items():
            name = self.number_to_name.get(dn)
            if name and name in self.driver_names:
                idx = self.driver_names.index(name)
                if ds.retired:
                    dnf_probs[idx] = 1.0
                    strengths[idx] = -999

        # Check for incident-involved drivers from race control
        incident_drivers = self._get_incident_drivers(live_state)
        for dn in incident_drivers:
            name = self.number_to_name.get(dn)
            if name and name in self.driver_names:
                idx = self.driver_names.index(name)
                if dnf_probs[idx] < 1.0:
                    dnf_probs[idx] = min(dnf_probs[idx] + 0.15, 0.5)

        # --- 4. Safety car probability ---
        if live_state.safety_car_active:
            sc_prob = 0.0  # Already under SC
        elif live_state.virtual_safety_car:
            sc_prob = 0.3
        else:
            sc_prob = 0.55 * laps_remaining_frac

        # --- 5. Build ensemble proba from strengths (for MC input) ---
        race_progress = laps_completed / total_laps
        ensemble_proba = self._strengths_to_proba(strengths, n_drivers, race_progress)

        # --- 6. Run Monte Carlo (with race_progress to scale variance) ---
        result_df = run_monte_carlo(
            ensemble_proba=ensemble_proba,
            driver_names=self.driver_names,
            grid_positions=grid_positions,
            dnf_probs=dnf_probs,
            n_simulations=self.n_simulations,
            race_progress=race_progress,
        )

        # --- 7. Save outputs ---
        self._save_parquet(result_df, laps_completed)
        self._save_db_snapshot(result_df, live_state)
        self._record_history(result_df, laps_completed, live_state)

        logger.info(
            f"Lap {laps_completed}/{total_laps}: "
            f"predicted winner = {result_df.index[0]} ({result_df.iloc[0]['win_prob']:.1%})"
        )
        return result_df

    def get_prediction_evolution(self) -> pd.DataFrame:
        """Get prediction history as a DataFrame for timeline charts.

        Returns:
            DataFrame with columns: lap, driver_name, win_prob, expected_position.
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    # --- Private helpers ---

    def _find_base_matrix(self) -> Path | None:
        """Find the most recent pre-race prediction matrix."""
        # Try stage-specific files first (race_eve > post_sprint > post_qualifying > pre_weekend)
        for stage in ["race_eve", "post_sprint", "post_qualifying", "pre_weekend"]:
            path = PREDICTIONS_DIR / f"matrix_{self.season}_r{self.round_num}_{stage}.parquet"
            if path.exists():
                return path

        # Fallback to latest_matrix.parquet
        latest = PREDICTIONS_DIR / "latest_matrix.parquet"
        if latest.exists():
            return latest
        return None

    def _compute_live_paces(self, state: LiveRaceState) -> dict[int, float]:
        """Compute EWM-smoothed pace for each driver.

        Returns:
            Mapping of driver_number -> smoothed pace (seconds).
        """
        paces: dict[int, float] = {}
        for dn, ds in state.drivers.items():
            if ds.retired or len(ds.lap_times) < LIVE_MIN_LAPS_FOR_PACE:
                continue
            # Exponentially weighted mean (alpha=0.3 = recent laps matter more)
            times = pd.Series(ds.lap_times)
            # Filter out outliers (pit laps, SC laps) — keep within 110% of median
            median = times.median()
            clean = times[times < median * 1.10]
            if len(clean) >= LIVE_MIN_LAPS_FOR_PACE:
                paces[dn] = clean.ewm(alpha=0.3).mean().iloc[-1]
        return paces

    def _get_incident_drivers(self, state: LiveRaceState) -> set[int]:
        """Find drivers involved in recent incidents from race control."""
        incident_drivers: set[int] = set()
        recent_lap = state.leader_lap - 3

        for event in state.race_control_events:
            if event.lap_number >= recent_lap:
                msg = event.message.upper()
                if any(kw in msg for kw in ["INCIDENT", "COLLISION", "CONTACT", "OFF TRACK"]):
                    # Try to extract driver number from message
                    for dn in state.drivers:
                        acr = state.drivers[dn].name_acronym
                        if acr and acr in msg:
                            incident_drivers.add(dn)
        return incident_drivers

    def _strengths_to_proba(
        self, strengths: np.ndarray, n_drivers: int, race_progress: float = 0.0,
    ) -> np.ndarray:
        """Convert strength scores to a pseudo-probability matrix for MC input.

        Each driver's distribution is centered at their strength-based rank.
        As race_progress increases, distributions become sharper (near-deterministic).

        Args:
            strengths: Per-driver strength scores.
            n_drivers: Number of drivers.
            race_progress: 0.0 = broad distribution, 1.0 = near-deterministic.
        """
        proba = np.zeros((n_drivers, GRID_SIZE))

        # Sharpness increases with race progress:
        # At start: 0.5 (broad), at end: 4.0 (extremely sharp)
        sharpness = 0.5 + 3.5 * race_progress

        # Rank drivers by strength (1 = strongest) — excludes retired
        active_mask = strengths > -900
        ranks = np.full(n_drivers, n_drivers, dtype=int)
        if active_mask.any():
            active_order = np.argsort(-strengths[active_mask])
            active_indices = np.where(active_mask)[0]
            for rank, idx in enumerate(active_indices[active_order]):
                ranks[idx] = rank + 1  # 1-indexed

        for i in range(n_drivers):
            if strengths[i] < -900:
                proba[i, GRID_SIZE - 1] = 1.0
                continue
            # Laplace-like distribution centered at this driver's rank
            center = ranks[i] - 1  # 0-indexed
            distances = np.abs(np.arange(GRID_SIZE) - center)
            scores = -sharpness * distances
            exp_scores = np.exp(scores - scores.max())
            proba[i] = exp_scores / exp_scores.sum()

        return proba

    def _save_parquet(self, matrix: pd.DataFrame, lap_number: int) -> None:
        """Save prediction matrix to parquet files."""
        LIVE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

        # Per-lap snapshot
        lap_path = LIVE_PREDICTIONS_DIR / f"matrix_{self.season}_r{self.round_num}_lap{lap_number}.parquet"
        matrix.to_parquet(lap_path)

        # Always update latest
        latest_path = LIVE_PREDICTIONS_DIR / "latest_live_matrix.parquet"
        matrix.to_parquet(latest_path)

    def _save_db_snapshot(self, matrix: pd.DataFrame, state: LiveRaceState) -> None:
        """Save prediction snapshot to database."""
        if self.race_id is None:
            return

        sc_active = 1 if state.safety_car_active else (2 if state.virtual_safety_car else 0)

        try:
            with get_session() as session:
                for driver_name in matrix.index:
                    row = matrix.loc[driver_name]
                    # Find current position and tire info from live state
                    current_pos = None
                    tire_compound = None
                    tire_age = None
                    for dn, ds in state.drivers.items():
                        if self.number_to_name.get(dn) == driver_name:
                            current_pos = ds.position
                            tire_compound = ds.tire_compound
                            tire_age = ds.tire_age
                            break

                    # Check if snapshot already exists (upsert)
                    existing = (
                        session.query(LivePredictionSnapshot)
                        .filter_by(
                            race_id=self.race_id,
                            lap_number=state.leader_lap,
                            driver_name=driver_name,
                        )
                        .first()
                    )
                    if existing:
                        existing.current_position = current_pos
                        existing.predicted_position = row.get("expected_position")
                        existing.win_prob = row.get("win_prob")
                        existing.podium_prob = row.get("podium_prob")
                        existing.tire_compound = tire_compound
                        existing.tire_age = tire_age
                        existing.safety_car_active = sc_active
                    else:
                        session.add(LivePredictionSnapshot(
                            race_id=self.race_id,
                            lap_number=state.leader_lap,
                            driver_name=driver_name,
                            current_position=current_pos,
                            predicted_position=row.get("expected_position"),
                            win_prob=row.get("win_prob"),
                            podium_prob=row.get("podium_prob"),
                            tire_compound=tire_compound,
                            tire_age=tire_age,
                            safety_car_active=sc_active,
                        ))
        except Exception as e:
            logger.warning(f"Failed to save DB snapshot: {e}")

    def _record_history(
        self, matrix: pd.DataFrame, lap: int, state: LiveRaceState
    ) -> None:
        """Record prediction snapshot for evolution charts."""
        for driver_name in matrix.index:
            row = matrix.loc[driver_name]
            self.history.append({
                "lap": lap,
                "driver_name": driver_name,
                "win_prob": row.get("win_prob", 0),
                "podium_prob": row.get("podium_prob", 0),
                "expected_position": row.get("expected_position", 0),
                "safety_car": state.safety_car_active,
            })


def _run_openf1_loop(
    predictor: LivePredictor,
    season: int,
    round_num: int,
    poll_interval: int,
    total_laps: int,
) -> None:
    """Run the live loop using OpenF1 API (requires paid auth for live data)."""
    client = OpenF1Client()

    logger.info(f"Looking up OpenF1 session key for {season} R{round_num}...")
    session_key = client.get_session_key(season, round_num)
    if session_key is None:
        logger.error("Could not find OpenF1 session key — is the race weekend active?")
        return

    logger.info(f"OpenF1 session_key = {session_key}")
    predictor.build_driver_map(client, session_key)

    # Wait for race start
    logger.info("Waiting for race start...")
    while True:
        events = client.get_race_control(session_key)
        if client.is_race_started(events):
            logger.info("Race started!")
            break
        if client.is_race_finished(events):
            logger.info("Race already finished")
            return
        time.sleep(30)

    # Main loop
    _prediction_loop(
        predictor,
        lambda: build_live_race_state(client, session_key, total_laps),
        poll_interval,
    )


def _run_scraper_loop(
    predictor: LivePredictor,
    season: int,
    round_num: int,
    poll_interval: int,
    total_laps: int,
) -> None:
    """Run the live loop using web scraping (free, no auth needed)."""
    from src.pipeline.collectors.web_scraper import LiveTimingScraper

    scraper = LiveTimingScraper(season, round_num, total_laps=total_laps)
    driver_map_built = False

    logger.info("Waiting for race data from web sources...")
    while True:
        state = scraper.fetch_live_state()
        if state and state.drivers:
            if not driver_map_built:
                predictor.build_driver_map_from_scraped(state)
                driver_map_built = True
            logger.info(f"Race data found: {len(state.drivers)} drivers")
            break
        logger.info("No race data yet — retrying in 60s...")
        time.sleep(60)

    # Main loop
    _prediction_loop(
        predictor,
        lambda: scraper.fetch_live_state(),
        poll_interval,
    )


def _prediction_loop(
    predictor: LivePredictor,
    fetch_state: callable,
    poll_interval: int,
) -> None:
    """Shared prediction loop — works with any data source.

    Args:
        predictor: Initialized LivePredictor.
        fetch_state: Callable that returns a LiveRaceState.
        poll_interval: Seconds between cycles.
    """
    last_leader_lap = -1
    stale_count = 0

    while True:
        try:
            state = fetch_state()
            if state is None:
                logger.warning("Failed to fetch live state — retrying")
                time.sleep(poll_interval)
                continue

            if state.race_finished:
                logger.info("Race finished — running final prediction")
                predictor.update(state)
                break

            # Skip if leader lap hasn't changed
            if state.leader_lap == last_leader_lap:
                stale_count += 1
                if stale_count > 5:
                    logger.warning(
                        f"Leader lap unchanged for {stale_count * poll_interval}s — "
                        "data may be stale"
                    )
            else:
                stale_count = 0
                last_leader_lap = state.leader_lap
                predictor.update(state)

            sleep_time = 300 if state.red_flag else poll_interval
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Live prediction loop interrupted by user")
            break
        except Exception as e:
            logger.error(f"Live prediction cycle failed: {e}")
            time.sleep(poll_interval)


def run_live_prediction_loop(
    season: int,
    round_num: int,
    poll_interval: int = LIVE_POLL_INTERVAL,
    n_sims: int = LIVE_MC_SIMULATIONS,
    total_laps: int = 0,
    data_source: str | None = None,
) -> None:
    """Main entry point: poll for live data, update predictions, repeat.

    Supports two data sources:
    - "openf1": OpenF1 API (requires paid auth for live data)
    - "scraper": Web scraping from public F1 results pages (free)

    Args:
        season: Race season year.
        round_num: Round number.
        poll_interval: Seconds between prediction cycles.
        n_sims: Monte Carlo simulations per cycle.
        total_laps: Scheduled race distance (0 = auto-detect).
        data_source: "openf1" or "scraper". Defaults to config setting.
    """
    source = data_source or LIVE_DATA_SOURCE
    predictor = LivePredictor(season, round_num, n_simulations=n_sims)

    if not predictor.initialize():
        return

    logger.info(f"Starting live predictions for {season} R{round_num} (source: {source})")

    if source == "openf1":
        _run_openf1_loop(predictor, season, round_num, poll_interval, total_laps)
    else:
        _run_scraper_loop(predictor, season, round_num, poll_interval, total_laps)

    # Save evolution data
    evolution = predictor.get_prediction_evolution()
    if not evolution.empty:
        evo_path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
        LIVE_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        evolution.to_parquet(evo_path, index=False)
        logger.info(f"Saved prediction evolution to {evo_path}")

    logger.info("Live prediction loop finished")
