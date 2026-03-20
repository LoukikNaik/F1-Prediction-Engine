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
def export(
    season: int = typer.Option(2026, help="Season to export"),
    round: int = typer.Option(None, help="Specific round to export (default: all)"),
    push: bool = typer.Option(False, help="Git add, commit, and push after export"),
):
    """Export prediction data as static JSON for GitHub Pages."""
    import subprocess

    from config.settings import PROJECT_ROOT
    from src.export.json_exporter import export_round as _export_round
    from src.export.json_exporter import export_season as _export_season

    output_dir = PROJECT_ROOT / "frontend" / "data"

    if round is not None:
        _export_round(season, round, output_dir)
    else:
        _export_season(season, output_dir)

    if push:
        data_path = str(output_dir)
        subprocess.run(["git", "add", data_path], check=True)
        msg = f"Update predictions: {season}"
        if round is not None:
            msg += f" R{round}"
        subprocess.run(["git", "commit", "-m", msg], check=True)
        subprocess.run(["git", "push"], check=True)
        logger.info("Pushed to remote")


@app.command()
def schedule(
    races: int = typer.Option(2, help="Number of upcoming weekends to schedule"),
):
    """Schedule automatic prediction runs for upcoming race weekends via launchd."""
    from scripts.schedule_weekend import schedule_weekends

    schedule_weekends(n_races=races)


@app.command()
def backfill(
    season: int = typer.Option(2025, help="Target season year"),
    round: int = typer.Option(..., help="Target round number"),
    lap_interval: int = typer.Option(3, help="Laps between predictions"),
    simulations: int = typer.Option(5000, help="Monte Carlo simulations per lap"),
    clear: bool = typer.Option(False, help="Clear existing live data before backfilling"),
    export: bool = typer.Option(True, help="Export JSON for frontend after backfill"),
):
    """Backfill live lap predictions from historical FastF1 data."""
    from src.models.backfill_predictor import run_backfill

    success = run_backfill(
        season=season,
        round_num=round,
        lap_interval=lap_interval,
        n_sims=simulations,
        clear_existing=clear,
    )
    if success and export:
        from config.settings import PROJECT_ROOT
        from src.export.json_exporter import export_round as _export_round

        output_dir = PROJECT_ROOT / "frontend" / "data"
        _export_round(season, round, output_dir)
        logger.info(f"Exported JSON for {season} R{round}")


@app.command(name="backfill-season")
def backfill_season(
    season: int = typer.Option(2025, help="Target season year"),
    start_round: int = typer.Option(1, help="First round to backfill"),
    end_round: int = typer.Option(24, help="Last round to backfill"),
    lap_interval: int = typer.Option(5, help="Laps between predictions"),
    simulations: int = typer.Option(5000, help="Monte Carlo simulations per lap"),
    export: bool = typer.Option(True, help="Export JSON for frontend after each round"),
):
    """Backfill live predictions for all completed rounds in a season."""
    from config.settings import PREDICTIONS_DIR, PROJECT_ROOT
    from src.models.backfill_predictor import run_backfill

    output_dir = PROJECT_ROOT / "frontend" / "data"
    backfilled = 0

    for rnd in range(start_round, end_round + 1):
        # Check if a pre-race matrix exists for this round
        has_matrix = any(
            (PREDICTIONS_DIR / f"matrix_{season}_r{rnd}_{stage}.parquet").exists()
            for stage in ["race_eve", "post_sprint", "post_qualifying", "pre_weekend"]
        )
        if not has_matrix:
            logger.debug(f"Skipping R{rnd} — no pre-race matrix found")
            continue

        logger.info(f"Backfilling {season} R{rnd}...")
        success = run_backfill(
            season=season,
            round_num=rnd,
            lap_interval=lap_interval,
            n_sims=simulations,
            clear_existing=True,
        )
        if success:
            backfilled += 1
            if export:
                from src.export.json_exporter import export_round as _export_round

                _export_round(season, rnd, output_dir)

    logger.info(f"Backfilled {backfilled} rounds for {season}")


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
