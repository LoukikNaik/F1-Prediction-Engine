"""Championship prediction via Monte Carlo simulation of remaining season."""

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from config.settings import PREDICTIONS_DIR, RANDOM_SEED
from src.database.connection import get_session
from src.database.models import (
    ChampionshipPrediction,
    Driver,
    Race,
    Result,
    Team,
)
from src.database.queries import get_races_for_season
from src.utils.constants import POINTS_SYSTEM, SPRINT_POINTS_SYSTEM
from src.utils.logger import logger


def predict_championship(
    season: int,
    n_simulations: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict end-of-season WDC and WCC standings.

    Algorithm:
    1. Get current actual standings (points through last completed round)
    2. For each remaining race, load/generate position probability distributions
    3. Run N simulations of the remaining season
    4. Tally championship position counts -> probability matrix
    5. Store results in ChampionshipPrediction table

    Args:
        season: Target season year.
        n_simulations: Number of Monte Carlo simulations.

    Returns:
        Tuple of (wdc_df, wcc_df) DataFrames with championship probabilities.
    """
    logger.info(f"Predicting {season} championship with {n_simulations} simulations...")

    with get_session() as session:
        races = get_races_for_season(session, season)
        if not races:
            raise ValueError(f"No races found for season {season}")

        # Determine completed vs remaining rounds
        completed_rounds, remaining_races = _split_completed_remaining(session, races)

        # Get current standings
        driver_points, driver_names, driver_team_map = _get_current_standings(
            session, season, completed_rounds
        )
        team_points, team_names = _get_team_standings(
            session, season, driver_points, driver_team_map
        )

        # Load prediction matrices for remaining races
        race_predictions = _load_race_predictions(remaining_races)

    if not remaining_races:
        logger.info("Season complete, no remaining races to simulate")
        # Return actual final standings as 100% probability
        wdc_df = _standings_to_df(driver_points, driver_names)
        wcc_df = _standings_to_df(team_points, team_names)
        return wdc_df, wcc_df

    # Run simulations
    wdc_df, wcc_df = _simulate_remaining_season(
        driver_points,
        driver_names,
        driver_team_map,
        team_points,
        team_names,
        race_predictions,
        remaining_races,
        n_simulations,
    )

    # Store results
    after_round = max(completed_rounds) if completed_rounds else 0
    with get_session() as session:
        _store_championship_predictions(session, season, after_round, wdc_df, wcc_df)

    logger.info("Championship prediction complete")
    return wdc_df, wcc_df


def predict_championship_evolution(
    season: int,
    n_simulations: int = 2000,
) -> dict[str, list[dict]]:
    """Compute championship win probability after each round.

    Returns:
        Dict with 'wdc' and 'wcc' keys, each mapping entity name to
        list of {round, p1_prob, expected_position, expected_points}.
    """
    logger.info(f"Computing {season} championship evolution ({n_simulations} sims/round)...")

    with get_session() as session:
        races = get_races_for_season(session, season)
        if not races:
            return {"wdc": {}, "wcc": {}}

        # Find rounds that have results
        rounds_with_results = []
        for race in races:
            count = session.query(Result).filter(Result.race_id == race.id).count()
            if count > 0:
                rounds_with_results.append(race.round)

        # Load prediction matrices while session is open (Race objects need it)
        race_predictions = _load_race_predictions(races)

        # Eagerly extract race metadata so we can use it outside session
        race_meta = [
            {"round": r.round, "is_sprint_weekend": bool(r.is_sprint_weekend)}
            for r in races
        ]

    if not rounds_with_results:
        return {"wdc": {}, "wcc": {}}

    wdc_evo: dict[str, list[dict]] = {}
    wcc_evo: dict[str, list[dict]] = {}

    # At each cutoff, simulate remaining season
    for cutoff_round in rounds_with_results:
        completed = [r for r in rounds_with_results if r <= cutoff_round]
        remaining = [r for r in race_meta if r["round"] > cutoff_round]

        with get_session() as session:
            driver_points, driver_names, driver_team_map = _get_current_standings(
                session, season, completed
            )
            team_points, team_names = _get_team_standings(
                session, season, driver_points, driver_team_map
            )

        if not remaining:
            # Season over at this cutoff — winner gets 100%
            sorted_drivers = sorted(driver_points.items(), key=lambda x: -x[1])
            for rank, (d_id, pts) in enumerate(sorted_drivers):
                name = driver_names[d_id]
                wdc_evo.setdefault(name, []).append({
                    "round": cutoff_round,
                    "p1_prob": 1.0 if rank == 0 else 0.0,
                    "expected_position": rank + 1,
                    "expected_points": round(pts, 1),
                })
            sorted_teams = sorted(team_points.items(), key=lambda x: -x[1])
            for rank, (t_id, pts) in enumerate(sorted_teams):
                name = team_names[t_id]
                wcc_evo.setdefault(name, []).append({
                    "round": cutoff_round,
                    "p1_prob": 1.0 if rank == 0 else 0.0,
                    "expected_position": rank + 1,
                    "expected_points": round(pts, 1),
                })
            continue

        # Filter predictions to only remaining races
        remaining_predictions = {r["round"]: race_predictions.get(r["round"]) for r in remaining}

        wdc_df, wcc_df = _simulate_remaining_season(
            driver_points, driver_names, driver_team_map,
            team_points, team_names,
            remaining_predictions, remaining, n_simulations,
        )

        # Extract P1 prob for each driver
        for name, row in wdc_df.iterrows():
            wdc_evo.setdefault(name, []).append({
                "round": cutoff_round,
                "p1_prob": round(float(row.get("P1_prob", 0)), 4),
                "expected_position": round(float(row["expected_position"]), 1),
                "expected_points": round(float(row["expected_points"]), 1),
            })

        # Extract P1 prob for each team
        for name, row in wcc_df.iterrows():
            wcc_evo.setdefault(name, []).append({
                "round": cutoff_round,
                "p1_prob": round(float(row.get("P1_prob", 0)), 4),
                "expected_position": round(float(row["expected_position"]), 1),
                "expected_points": round(float(row["expected_points"]), 1),
            })

        logger.info(f"  R{cutoff_round}: done")

    return {"wdc": wdc_evo, "wcc": wcc_evo}


def _split_completed_remaining(
    session: Session, races: list[Race]
) -> tuple[list[int], list[Race]]:
    """Split races into completed and remaining."""
    completed = []
    remaining = []

    for race in races:
        result_count = (
            session.query(Result)
            .filter(Result.race_id == race.id)
            .count()
        )
        if result_count > 0:
            completed.append(race.round)
        else:
            remaining.append(race)

    return completed, remaining


def _get_current_standings(
    session: Session,
    season: int,
    completed_rounds: list[int],
) -> tuple[dict[int, float], dict[int, str], dict[int, int]]:
    """Get current driver points from actual results.

    Returns:
        Tuple of (driver_id -> points, driver_id -> name, driver_id -> team_id).
    """
    driver_points: dict[int, float] = {}
    driver_names: dict[int, str] = {}
    driver_team_map: dict[int, int] = {}

    if not completed_rounds:
        # No completed races — get driver list from latest round entries
        latest_race = (
            session.query(Race)
            .filter(Race.season == season)
            .order_by(Race.round)
            .first()
        )
        if latest_race:
            # Try to get drivers from results of previous season
            prev_results = (
                session.query(Result)
                .join(Race, Result.race_id == Race.id)
                .filter(Race.season == season - 1)
                .all()
            )
            seen = set()
            for r in prev_results:
                if r.driver_id not in seen:
                    seen.add(r.driver_id)
                    driver = session.query(Driver).get(r.driver_id)
                    if driver:
                        driver_points[r.driver_id] = 0.0
                        driver_names[r.driver_id] = driver.full_name
                        if r.team_id:
                            driver_team_map[r.driver_id] = r.team_id
        return driver_points, driver_names, driver_team_map

    # Sum points from completed race results
    races = (
        session.query(Race)
        .filter(Race.season == season, Race.round.in_(completed_rounds))
        .all()
    )
    race_ids = [r.id for r in races]

    results = (
        session.query(Result)
        .filter(Result.race_id.in_(race_ids))
        .all()
    )

    for r in results:
        if r.driver_id not in driver_points:
            driver_points[r.driver_id] = 0.0
            driver = session.query(Driver).get(r.driver_id)
            driver_names[r.driver_id] = driver.full_name if driver else f"Driver {r.driver_id}"
        driver_points[r.driver_id] += r.points or 0
        if r.team_id:
            driver_team_map[r.driver_id] = r.team_id

    return driver_points, driver_names, driver_team_map


def _get_team_standings(
    session: Session,
    season: int,
    driver_points: dict[int, float],
    driver_team_map: dict[int, int],
) -> tuple[dict[int, float], dict[int, str]]:
    """Aggregate driver points into team standings."""
    team_points: dict[int, float] = {}
    team_names: dict[int, str] = {}

    for driver_id, points in driver_points.items():
        team_id = driver_team_map.get(driver_id)
        if team_id is None:
            continue
        team_points[team_id] = team_points.get(team_id, 0) + points
        if team_id not in team_names:
            team = session.query(Team).get(team_id)
            team_names[team_id] = team.name if team else f"Team {team_id}"

    return team_points, team_names


def _load_race_predictions(remaining_races: list[Race]) -> dict[int, pd.DataFrame | None]:
    """Load prediction matrices for remaining races from parquet files."""
    predictions = {}

    for race in remaining_races:
        # Try to find a prediction matrix file
        pattern = f"matrix_{race.season}_r{race.round}_*.parquet"
        found = list(PREDICTIONS_DIR.glob(pattern)) if PREDICTIONS_DIR.exists() else []

        if found:
            # Use the most recent one
            latest = max(found, key=lambda p: p.stat().st_mtime)
            try:
                predictions[race.round] = pd.read_parquet(latest)
                continue
            except Exception:
                pass

        predictions[race.round] = None

    return predictions


def _simulate_remaining_season(
    driver_points: dict[int, float],
    driver_names: dict[int, str],
    driver_team_map: dict[int, int],
    team_points: dict[int, float],
    team_names: dict[int, str],
    race_predictions: dict[int, pd.DataFrame | None],
    remaining_races: list[Race] | list[dict],
    n_simulations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo simulation of remaining season for WDC and WCC.

    Returns:
        Tuple of (wdc_df, wcc_df).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    driver_ids = sorted(driver_points.keys())
    n_drivers = len(driver_ids)

    # Track final championship positions across simulations
    position_counts = np.zeros((n_drivers, n_drivers), dtype=int)
    total_points_all = np.zeros((n_drivers, n_simulations))

    # Track team results
    team_ids = sorted(team_points.keys())
    n_teams = len(team_ids)
    team_position_counts = np.zeros((n_teams, n_teams), dtype=int)
    team_total_points_all = np.zeros((n_teams, n_simulations))

    for sim in range(n_simulations):
        # Start with current actual points
        sim_points = {d: driver_points[d] for d in driver_ids}

        for race in remaining_races:
            rnd = race["round"] if isinstance(race, dict) else race.round
            is_sprint_raw = race.get("is_sprint_weekend", False) if isinstance(race, dict) else race.is_sprint_weekend
            pred_matrix = race_predictions.get(rnd)
            is_sprint = bool(is_sprint_raw)

            if pred_matrix is not None:
                # Sample finishing positions from probability distribution
                _sample_race_results(
                    sim_points, driver_ids, driver_names, pred_matrix, rng, is_sprint
                )
            else:
                # No predictions available — use uniform random for this race
                positions = rng.permutation(n_drivers) + 1
                for idx, d_id in enumerate(driver_ids):
                    pos = int(positions[idx])
                    sim_points[d_id] += POINTS_SYSTEM.get(pos, 0)
                    if is_sprint:
                        sprint_pos = int(rng.permutation(n_drivers)[idx]) + 1
                        sim_points[d_id] += SPRINT_POINTS_SYSTEM.get(sprint_pos, 0)

        # Rank drivers by total points (descending)
        sorted_drivers = sorted(driver_ids, key=lambda d: sim_points[d], reverse=True)
        for rank, d_id in enumerate(sorted_drivers):
            idx = driver_ids.index(d_id)
            position_counts[idx, rank] += 1
            total_points_all[idx, sim] = sim_points[d_id]

        # Aggregate to team points and rank teams
        sim_team_points = {t: team_points[t] for t in team_ids}
        for d_id in driver_ids:
            t_id = driver_team_map.get(d_id)
            if t_id in sim_team_points:
                sim_team_points[t_id] += sim_points[d_id] - driver_points[d_id]

        sorted_teams = sorted(team_ids, key=lambda t: sim_team_points[t], reverse=True)
        for rank, t_id in enumerate(sorted_teams):
            idx = team_ids.index(t_id)
            team_position_counts[idx, rank] += 1
            team_total_points_all[idx, sim] = sim_team_points[t_id]

    # Build WDC DataFrame
    prob_matrix = position_counts / n_simulations
    columns = {f"P{i+1}_prob": prob_matrix[:, i] for i in range(n_drivers)}
    columns["expected_points"] = total_points_all.mean(axis=1)
    columns["expected_position"] = (
        prob_matrix * np.arange(1, n_drivers + 1)
    ).sum(axis=1)

    wdc_df = pd.DataFrame(columns, index=[driver_names[d] for d in driver_ids])
    wdc_df = wdc_df.sort_values("expected_position")

    # Build WCC DataFrame
    team_prob_matrix = team_position_counts / n_simulations
    t_columns = {f"P{i+1}_prob": team_prob_matrix[:, i] for i in range(n_teams)}
    t_columns["expected_points"] = team_total_points_all.mean(axis=1)
    t_columns["expected_position"] = (
        team_prob_matrix * np.arange(1, n_teams + 1)
    ).sum(axis=1)

    wcc_df = pd.DataFrame(t_columns, index=[team_names[t] for t in team_ids])
    wcc_df = wcc_df.sort_values("expected_position")

    return wdc_df, wcc_df


def _sample_race_results(
    sim_points: dict[int, float],
    driver_ids: list[int],
    driver_names: dict[int, str],
    pred_matrix: pd.DataFrame,
    rng: np.random.Generator,
    is_sprint: bool,
) -> None:
    """Sample finishing positions from a prediction matrix and award points."""
    pos_cols = [c for c in pred_matrix.columns if c.startswith("P") and c[1:].isdigit()]

    for d_id in driver_ids:
        name = driver_names[d_id]
        if name in pred_matrix.index:
            probs = pred_matrix.loc[name, pos_cols].values.astype(float)
            probs = np.maximum(probs, 0)
            total = probs.sum()
            if total > 0:
                probs /= total
                pos = rng.choice(len(probs), p=probs) + 1
            else:
                pos = rng.integers(1, len(driver_ids) + 1)
        else:
            pos = rng.integers(1, len(driver_ids) + 1)

        sim_points[d_id] += POINTS_SYSTEM.get(pos, 0)

        if is_sprint:
            # Simplified: sprint position correlated with race prediction
            sprint_pos = max(1, pos + rng.integers(-3, 4))
            sim_points[d_id] += SPRINT_POINTS_SYSTEM.get(sprint_pos, 0)



def _standings_to_df(
    points: dict[int, float],
    names: dict[int, str],
) -> pd.DataFrame:
    """Convert standings dict to DataFrame for completed season."""
    sorted_ids = sorted(points.keys(), key=lambda d: points[d], reverse=True)
    data = {
        "expected_points": [points[d] for d in sorted_ids],
        "expected_position": list(range(1, len(sorted_ids) + 1)),
        "P1_prob": [1.0 if i == 0 else 0.0 for i in range(len(sorted_ids))],
    }
    return pd.DataFrame(data, index=[names[d] for d in sorted_ids])


def _store_championship_predictions(
    session: Session,
    season: int,
    after_round: int,
    wdc_df: pd.DataFrame,
    wcc_df: pd.DataFrame,
) -> None:
    """Store championship predictions in the database."""
    # Clear existing predictions for this season+round
    session.query(ChampionshipPrediction).filter(
        ChampionshipPrediction.season == season,
        ChampionshipPrediction.after_round == after_round,
    ).delete()

    # Store WDC predictions
    for rank, (driver_name, row) in enumerate(wdc_df.iterrows(), 1):
        driver = (
            session.query(Driver)
            .filter(Driver.full_name == driver_name)
            .first()
        )
        if not driver:
            continue

        p1_prob = row.get("P1_prob", 0.0)
        session.add(ChampionshipPrediction(
            season=season,
            after_round=after_round,
            driver_id=driver.id,
            prediction_type="wdc",
            predicted_position=rank,
            probability=float(p1_prob),
            predicted_points=float(row["expected_points"]),
            model_version="championship_v1",
        ))

    # Store WCC predictions
    for rank, (team_name, row) in enumerate(wcc_df.iterrows(), 1):
        team = (
            session.query(Team)
            .filter(Team.name == team_name)
            .first()
        )
        if not team:
            continue

        p1_prob = row.get("P1_prob", 0.0)
        session.add(ChampionshipPrediction(
            season=season,
            after_round=after_round,
            team_id=team.id,
            prediction_type="wcc",
            predicted_position=rank,
            probability=float(p1_prob),
            predicted_points=float(row["expected_points"]),
            model_version="championship_v1",
        ))
