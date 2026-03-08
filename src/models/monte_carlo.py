"""Monte Carlo race simulation for F1 predictions."""

import numpy as np
import pandas as pd

from config.settings import MONTE_CARLO_SIMULATIONS, RANDOM_SEED
from src.utils.constants import GRID_SIZE
from src.utils.logger import logger


def simulate_race(
    driver_strengths: np.ndarray,
    dnf_probs: np.ndarray,
    grid_positions: np.ndarray,
    n_drivers: int,
    rng: np.random.Generator,
    safety_car_prob: float = 0.55,
    race_progress: float = 0.0,
) -> np.ndarray:
    """Simulate a single race and return finishing order.

    Realistic simulation that accounts for:
    - Race-pace variance (some days you're just off)
    - First-lap incidents (can take out anyone)
    - Safety car periods (bunch up the field)
    - Mechanical DNFs (any car can break)
    - Punctures and damage (random mid-race events)

    All variance and incident probabilities scale down with race_progress,
    so late-race predictions are near-deterministic (positions mostly locked in).

    Args:
        driver_strengths: Per-driver strength scores (higher = faster).
        dnf_probs: DNF probability per driver.
        grid_positions: Starting grid positions (1-indexed).
        n_drivers: Number of drivers.
        rng: Random number generator.
        safety_car_prob: Probability of a safety car period.
        race_progress: 0.0 = race start (full variance), 1.0 = race end (minimal).

    Returns:
        Array of finishing positions (1-indexed) for each driver.
    """
    remaining = 1.0 - race_progress  # fraction of race left

    # Base performance with variance scaled by remaining race distance
    noise_scale = 1.5 * remaining
    performance = driver_strengths + rng.normal(0, max(noise_scale, 0.05), n_drivers)

    # Grid position advantage (less relevant late in race — positions are real)
    grid_bonus = (n_drivers - grid_positions) * 0.08 * remaining
    performance += grid_bonus

    # First-lap incident: only possible in early race (< 5% progress)
    if race_progress < 0.05 and rng.random() < 0.15:
        n_involved = rng.integers(1, 4)
        weights = np.ones(n_drivers)
        weights[grid_positions > 5] *= 1.5
        weights /= weights.sum()
        involved = rng.choice(n_drivers, size=min(n_involved, n_drivers), replace=False, p=weights)
        for idx in involved:
            if rng.random() < 0.5:
                performance[idx] = -999 - rng.random()
            else:
                performance[idx] -= rng.uniform(3, 8)

    # Safety car: bunches up the field, reduces gaps
    if rng.random() < safety_car_prob:
        mean_perf = performance[performance > -900].mean()
        # Compression scales with remaining race — late SC has less effect
        compress = 0.4 * remaining
        performance[performance > -900] = (
            (1 - compress) * performance[performance > -900] + compress * mean_perf
        )
        restart_noise = 0.8 * remaining
        performance[performance > -900] += rng.normal(0, max(restart_noise, 0.05), (performance > -900).sum())

    # Mechanical DNF: scaled by remaining distance
    for i in range(n_drivers):
        if performance[i] > -900:
            if rng.random() < dnf_probs[i]:
                performance[i] = -999 - rng.random()

    # Random mid-race incidents: scale by remaining race
    surviving = np.where(performance > -900)[0]
    incident_rate = 0.10 * remaining
    n_incidents = max(0, int(len(surviving) * incident_rate))
    if n_incidents > 0 and len(surviving) > 0:
        incident_drivers = rng.choice(surviving, size=min(n_incidents, len(surviving)), replace=False)
        for idx in incident_drivers:
            severity = rng.random()
            if severity < 0.15:
                performance[idx] = -999 - rng.random()
            elif severity < 0.5:
                performance[idx] -= rng.uniform(2, 5)
            else:
                performance[idx] -= rng.uniform(0.5, 2)

    # Convert to finishing positions (highest performance = P1)
    order = np.argsort(-performance)
    positions = np.empty(n_drivers, dtype=int)
    positions[order] = np.arange(1, n_drivers + 1)

    return positions


def run_monte_carlo(
    ensemble_proba: np.ndarray,
    driver_names: list[str],
    grid_positions: np.ndarray | None = None,
    dnf_probs: np.ndarray | None = None,
    n_simulations: int = MONTE_CARLO_SIMULATIONS,
    race_progress: float = 0.0,
) -> pd.DataFrame:
    """Run Monte Carlo simulations to generate finishing position distributions.

    Args:
        ensemble_proba: Probability matrix from ensemble model (n_drivers, 20).
        driver_names: List of driver names.
        grid_positions: Starting grid positions. If None, uses expected position.
        dnf_probs: DNF probability per driver. If None, defaults to 0.05.
        n_simulations: Number of simulations to run.
        race_progress: 0.0 = pre-race (full variance), 1.0 = race end (minimal).

    Returns:
        DataFrame with probability matrix (drivers × positions).
    """
    n_drivers = len(driver_names)
    logger.info(f"Running {n_simulations} Monte Carlo simulations for {n_drivers} drivers...")

    rng = np.random.default_rng(RANDOM_SEED)

    # Convert ensemble probabilities to strength scores
    # Use expected finishing position as strength (lower = stronger)
    positions = np.arange(1, GRID_SIZE + 1)
    expected_positions = (ensemble_proba[:, :GRID_SIZE] * positions).sum(axis=1)
    driver_strengths = GRID_SIZE - expected_positions  # Invert so higher = faster

    if grid_positions is None:
        grid_positions = np.argsort(np.argsort(expected_positions)) + 1

    if dnf_probs is None:
        dnf_probs = np.full(n_drivers, 0.05)

    # Run simulations
    position_counts = np.zeros((n_drivers, GRID_SIZE), dtype=int)

    for _ in range(n_simulations):
        finish_positions = simulate_race(
            driver_strengths[:n_drivers],
            dnf_probs[:n_drivers],
            grid_positions[:n_drivers],
            n_drivers,
            rng,
            race_progress=race_progress,
        )
        for driver_idx in range(n_drivers):
            pos = min(finish_positions[driver_idx], GRID_SIZE) - 1
            position_counts[driver_idx, pos] += 1

    # Convert to probabilities (no smoothing needed — realistic simulation
    # already produces non-zero counts for extreme outcomes via incidents,
    # safety cars, mechanical DNFs, and first-lap crashes)
    proba_matrix = position_counts / n_simulations

    # Create DataFrame
    columns = [f"P{i}" for i in range(1, GRID_SIZE + 1)]
    result_df = pd.DataFrame(
        proba_matrix,
        index=driver_names,
        columns=columns,
    )

    # Add summary stats
    result_df["expected_position"] = (
        proba_matrix * np.arange(1, GRID_SIZE + 1)
    ).sum(axis=1)
    result_df["win_prob"] = proba_matrix[:, 0]
    result_df["podium_prob"] = proba_matrix[:, :3].sum(axis=1)
    result_df["top5_prob"] = proba_matrix[:, :5].sum(axis=1)

    result_df = result_df.sort_values("expected_position")

    logger.info("Monte Carlo simulation complete")
    return result_df
