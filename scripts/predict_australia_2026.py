"""Generate predictions for the 2026 Australian Grand Prix.

Uses actual qualifying results from March 7, 2026.
Now uses the unified orchestrator pipeline instead of custom logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config.settings import PREDICTIONS_DIR
from src.database.connection import get_session
from src.database.models import Circuit, Driver, Race, Result, Team
from src.pipeline.orchestrator import predict_race, train_model
from src.utils.constants import PredictionStage
from src.utils.logger import logger

# Actual 2026 Australian GP qualifying grid (March 7, 2026)
QUALIFYING_GRID_2026 = [
    # (driver_name, team_name, grid_position)
    ("George Russell", "Mercedes", 1),
    ("Andrea Kimi Antonelli", "Mercedes", 2),
    ("Isack Hadjar", "Red Bull", 3),
    ("Charles Leclerc", "Ferrari", 4),
    ("Oscar Piastri", "McLaren", 5),
    ("Lando Norris", "McLaren", 6),
    ("Lewis Hamilton", "Ferrari", 7),
    ("Liam Lawson", "Racing Bulls", 8),
    ("Arvid Lindblad", "Racing Bulls", 9),
    ("Gabriel Bortoleto", "Audi", 10),
    ("Nico Hulkenberg", "Audi", 11),
    ("Oliver Bearman", "Haas F1 Team", 12),
    ("Esteban Ocon", "Haas F1 Team", 13),
    ("Pierre Gasly", "Alpine F1 Team", 14),
    ("Alexander Albon", "Williams", 15),
    ("Franco Colapinto", "Alpine F1 Team", 16),
    ("Fernando Alonso", "Aston Martin", 17),
    ("Sergio Perez", "Cadillac", 18),
    ("Valtteri Bottas", "Cadillac", 19),
    ("Max Verstappen", "Red Bull", 20),  # Crashed in Q1
    ("Carlos Sainz", "Williams", 21),  # No qualifying time
    ("Lance Stroll", "Aston Martin", 22),  # No qualifying time
]


def _get_or_create_driver(session, name: str) -> Driver:
    """Find or create a driver by name."""
    driver = session.query(Driver).filter(Driver.full_name == name).first()
    if driver:
        return driver
    driver = session.query(Driver).filter(Driver.full_name.ilike(f"%{name}%")).first()
    if driver:
        return driver
    code = "".join(w[0:3].upper() for w in name.split()[-1:])
    driver = Driver(
        code=code,
        full_name=name,
        nationality="",
        date_of_birth="",
        ergast_id=name.lower().replace(" ", "_"),
    )
    session.add(driver)
    session.flush()
    logger.info(f"Created new driver: {name}")
    return driver


def _get_or_create_team(session, name: str) -> Team:
    """Find or create a team by name."""
    team = session.query(Team).filter(Team.name == name).first()
    if team:
        return team
    team = session.query(Team).filter(Team.name.ilike(f"%{name}%")).first()
    if team:
        return team
    team = Team(
        name=name,
        ergast_id=name.lower().replace(" ", "_"),
    )
    session.add(team)
    session.flush()
    logger.info(f"Created new team: {name}")
    return team


def create_2026_australian_gp():
    """Create/update 2026 Australian GP with actual qualifying grid."""
    with get_session() as session:
        circuit = session.query(Circuit).filter(
            Circuit.ergast_id == "albert_park"
        ).first()
        if not circuit:
            circuit = session.query(Circuit).filter(
                Circuit.name.ilike("%albert%")
            ).first()
        if not circuit:
            circuit = Circuit(
                name="Albert Park Grand Prix Circuit",
                country="Australia",
                city="Melbourne",
                latitude=-37.8497,
                longitude=144.968,
                circuit_type="hybrid",
                length_km=5.278,
                ergast_id="albert_park",
            )
            session.add(circuit)
            session.flush()

        existing_race = session.query(Race).filter(
            Race.season == 2026, Race.round == 1
        ).first()

        if existing_race:
            # Only recreate results, keep race record
            session.query(Result).filter(Result.race_id == existing_race.id).delete()
            race = existing_race
        else:
            race = Race(
                season=2026,
                round=1,
                circuit_id=circuit.id,
                race_date="2026-03-08",
                race_name="Australian Grand Prix",
                qualifying_date="2026-03-07",
                qualifying_time="06:00:00",
                race_time="04:00:00",
            )
            session.add(race)
            session.flush()

        for driver_name, team_name, grid_pos in QUALIFYING_GRID_2026:
            driver = _get_or_create_driver(session, driver_name)
            team = _get_or_create_team(session, team_name)

            result = Result(
                race_id=race.id,
                driver_id=driver.id,
                team_id=team.id,
                grid=grid_pos,
                position=None,
                points=0,
                status="Pending",
            )
            session.add(result)

    logger.info(f"Created 2026 Australian GP with {len(QUALIFYING_GRID_2026)} drivers")


def predict_2026_australian_gp():
    """Generate predictions using the unified pipeline."""
    logger.info("=" * 60)
    logger.info("2026 AUSTRALIAN GRAND PRIX - PREDICTION ENGINE")
    logger.info("Post-Qualifying Predictions (actual grid)")
    logger.info("=" * 60)

    # Ensure race data exists
    create_2026_australian_gp()

    # Train model
    model = train_model(min_season=1950)

    # Use orchestrator to predict (post-qualifying stage since we have the grid)
    prob_matrix = predict_race(
        season=2026,
        round_num=1,
        stage=PredictionStage.POST_QUALIFYING,
        model=model,
    )

    # Print results
    print("\n" + "=" * 80)
    print("2026 AUSTRALIAN GRAND PRIX - POST-QUALIFYING PREDICTIONS")
    print("=" * 80)

    print("\nPREDICTED PODIUM:")
    for i, (name, row) in enumerate(prob_matrix.head(3).iterrows(), 1):
        medals = {1: "P1", 2: "P2", 3: "P3"}
        print(f"  {medals[i]}: {name:25s}  Win: {row['win_prob']:6.1%}  Podium: {row['podium_prob']:6.1%}")

    print(f"\n{'Driver':<25} {'Win%':>7} {'Podium%':>8} {'Top5%':>7} {'Exp.Pos':>8}")
    print("-" * 60)
    for name, row in prob_matrix.iterrows():
        print(f"{name:<25} {row['win_prob']:6.1%} {row['podium_prob']:7.1%} {row['top5_prob']:6.1%} {row['expected_position']:7.1f}")

    logger.info("Predictions saved to data/processed/predictions/")
    return prob_matrix


if __name__ == "__main__":
    predict_2026_australian_gp()
