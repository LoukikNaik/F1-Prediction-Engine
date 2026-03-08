"""CLI entry point for the F1 prediction pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import typer

from src.utils.logger import logger

app = typer.Typer(help="F1 Prediction Engine CLI")


@app.command()
def pipeline(
    season: int = typer.Option(2026, help="Target season year"),
    round: int = typer.Option(None, help="Target round number"),
    retrain: bool = typer.Option(False, help="Retrain model before predicting"),
):
    """Run the prediction pipeline for a race."""
    from src.pipeline.orchestrator import run_full_pipeline

    result = run_full_pipeline(season=season, round_num=round, retrain=retrain)
    print("\n" + "=" * 80)
    print(f"PREDICTIONS - Season {season}, Round {round}")
    print("=" * 80)
    print(result[["win_prob", "podium_prob", "expected_position"]].to_string())


@app.command()
def predict(
    season: int = typer.Option(2026, help="Target season year"),
    round: int = typer.Option(..., help="Target round number"),
    stage: str = typer.Option(None, help="Prediction stage: pre_weekend, post_qualifying, post_sprint, race_eve"),
):
    """Generate predictions for a specific race and stage."""
    from src.pipeline.orchestrator import predict_race
    from src.utils.constants import PredictionStage

    stage_enum = None
    if stage:
        stage_enum = PredictionStage(stage)

    result = predict_race(season=season, round_num=round, stage=stage_enum)
    print("\n" + "=" * 80)
    stage_label = stage or "auto-detected"
    print(f"PREDICTIONS - Season {season}, Round {round} (stage: {stage_label})")
    print("=" * 80)
    print(result[["win_prob", "podium_prob", "expected_position"]].to_string())


@app.command(name="predict-season")
def predict_season(
    season: int = typer.Option(2026, help="Target season year"),
):
    """Predict all upcoming races in a season."""
    from src.pipeline.orchestrator import predict_season as _predict_season

    results = _predict_season(season=season)
    print(f"\nPredicted {len(results)} races for {season} season")
    for round_num, matrix in sorted(results.items()):
        winner = matrix.iloc[0]
        print(f"  R{round_num}: Predicted winner = {winner.name} ({winner['win_prob']:.1%})")


@app.command()
def championship(
    season: int = typer.Option(2026, help="Target season year"),
    simulations: int = typer.Option(5000, help="Number of Monte Carlo simulations"),
):
    """Generate WDC/WCC championship predictions."""
    from src.models.championship_predictor import predict_championship

    wdc_df, wcc_df = predict_championship(season=season, n_simulations=simulations)

    print("\n" + "=" * 80)
    print(f"WDC CHAMPIONSHIP PREDICTION - {season}")
    print("=" * 80)
    display_cols = ["expected_points", "expected_position", "P1_prob"]
    available = [c for c in display_cols if c in wdc_df.columns]
    print(wdc_df[available].head(10).to_string())

    print("\n" + "=" * 80)
    print(f"WCC CHAMPIONSHIP PREDICTION - {season}")
    print("=" * 80)
    available = [c for c in display_cols if c in wcc_df.columns]
    print(wcc_df[available].head(10).to_string())


@app.command()
def train(
    min_season: int = typer.Option(2000, help="Earliest season for training"),
):
    """Train the ML model on historical data."""
    from src.pipeline.orchestrator import train_model

    model = train_model(min_season=min_season)
    logger.info("Training complete!")


@app.command(name="fetch-schedule")
def fetch_schedule(
    season: int = typer.Option(2026, help="Season to fetch schedule for"),
):
    """Fetch race schedule with session timing for a season."""
    from src.database.connection import get_session
    from src.pipeline.collectors.schedule_collector import populate_race_schedule

    with get_session() as session:
        count = populate_race_schedule(session, season)
    print(f"Updated {count} races for {season} season")


@app.command()
def live(
    season: int = typer.Option(2026, help="Target season year"),
    round: int = typer.Option(1, help="Target round number"),
    poll_interval: int = typer.Option(90, help="Seconds between prediction cycles"),
    simulations: int = typer.Option(5000, help="Monte Carlo simulations per cycle"),
    total_laps: int = typer.Option(0, help="Scheduled race laps (0 = auto)"),
    source: str = typer.Option(None, help="Data source: 'openf1' or 'scraper' (default: config)"),
):
    """Run live lap-by-lap predictions during a race."""
    from src.models.live_predictor import run_live_prediction_loop

    logger.info(f"Starting live predictions for {season} R{round}...")
    run_live_prediction_loop(
        season=season,
        round_num=round,
        poll_interval=poll_interval,
        n_sims=simulations,
        total_laps=total_laps,
        data_source=source,
    )


@app.command()
def migrate():
    """Run database migrations (v2 + v3 schema)."""
    from src.database.migrations import migrate_v2, migrate_v3

    migrate_v2()
    migrate_v3()
    print("Migration complete")


@app.command()
def info():
    """Show database statistics."""
    from sqlalchemy import func

    from src.database.connection import get_session
    from src.database.models import Circuit, Driver, Race, Result, Team

    with get_session() as session:
        print(f"Drivers:  {session.query(Driver).count()}")
        print(f"Teams:    {session.query(Team).count()}")
        print(f"Circuits: {session.query(Circuit).count()}")
        print(f"Races:    {session.query(Race).count()}")
        print(f"Results:  {session.query(Result).count()}")

        min_year = session.query(func.min(Race.season)).scalar()
        max_year = session.query(func.max(Race.season)).scalar()
        print(f"Seasons:  {min_year} - {max_year}")


if __name__ == "__main__":
    app()
