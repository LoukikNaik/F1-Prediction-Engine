"""Export prediction data as static JSON files for GitHub Pages."""

import json
from pathlib import Path

import pandas as pd
from sqlalchemy import text as sql_text

from config.settings import LIVE_PREDICTIONS_DIR, PREDICTIONS_DIR
from src.database.connection import get_session
from src.utils.logger import logger
from src.utils.serialization import matrix_to_records, to_json_safe

STAGE_PRIORITY = ("race_eve", "post_sprint", "post_qualifying", "pre_weekend")


def _write_json(path: Path, data: dict | list) -> None:
    """Write data as formatted JSON, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"Wrote {path}")


def _export_races(season: int, output_dir: Path) -> list[dict]:
    """Export races.json and return the race list."""
    with get_session() as session:
        rows = session.execute(
            sql_text("""
                SELECT r.round, r.race_name, c.name, c.country
                FROM races r
                LEFT JOIN circuits c ON r.circuit_id = c.id
                WHERE r.season = :season
                ORDER BY r.round
            """),
            {"season": season},
        ).fetchall()

    races = []
    for row in rows:
        round_num = int(row[0])
        available_stages = []
        for stage in STAGE_PRIORITY:
            path = PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_{stage}.parquet"
            if path.exists():
                available_stages.append(stage)
        live_pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
        has_live = any(LIVE_PREDICTIONS_DIR.glob(live_pattern))
        races.append({
            "round": round_num,
            "race_name": str(row[1]) if row[1] else f"Round {round_num}",
            "circuit_name": str(row[2]) if row[2] else None,
            "country": str(row[3]) if row[3] else None,
            "available_stages": available_stages,
            "has_live": has_live,
        })

    _write_json(output_dir / str(season) / "races.json", {"races": races})
    return races


def _export_teams(season: int, output_dir: Path) -> None:
    """Export teams.json with driver -> team mapping."""
    with get_session() as session:
        rows = session.execute(
            sql_text("""
                SELECT d.full_name, t.name
                FROM results r
                JOIN drivers d ON r.driver_id = d.id
                JOIN teams t ON r.team_id = t.id
                JOIN races ra ON r.race_id = ra.id
                WHERE ra.season = :season
                ORDER BY ra.round DESC
            """),
            {"season": season},
        ).fetchall()

    mapping: dict[str, str] = {}
    for name, team in rows:
        if name not in mapping:
            mapping[name] = team

    if not mapping:
        # Fallback to previous season
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                    SELECT d.full_name, t.name
                    FROM results r
                    JOIN drivers d ON r.driver_id = d.id
                    JOIN teams t ON r.team_id = t.id
                    JOIN races ra ON r.race_id = ra.id
                    WHERE ra.season = :season
                    ORDER BY ra.round DESC
                """),
                {"season": season - 1},
            ).fetchall()
        for name, team in rows:
            if name not in mapping:
                mapping[name] = team

    _write_json(output_dir / str(season) / "teams.json", {"mapping": mapping})


def _export_standings(season: int, output_dir: Path) -> None:
    """Export standings.json for the season."""
    from src.database.queries import get_standings_df

    with get_session() as session:
        df = get_standings_df(session, season, "driver")

    if not df.empty:
        latest = df[df["round"] == df["round"].max()]
        records = latest[["driver_name", "driver_code", "position", "points"]].to_dict("records")
        for rec in records:
            for k, v in list(rec.items()):
                rec[k] = to_json_safe(v)
        _write_json(output_dir / str(season) / "standings.json", {"records": records, "season": season})
    else:
        # No standings — build 0-point grid
        with get_session() as session:
            rows = session.execute(
                sql_text("""
                    SELECT DISTINCT d.full_name, d.code
                    FROM results r
                    JOIN drivers d ON r.driver_id = d.id
                    JOIN races ra ON r.race_id = ra.id
                    WHERE ra.season = :season
                    ORDER BY d.full_name
                """),
                {"season": season},
            ).fetchall()

        if rows:
            records = [
                {"driver_name": name, "driver_code": code, "position": i + 1, "points": 0}
                for i, (name, code) in enumerate(rows)
            ]
        else:
            records = []
        _write_json(output_dir / str(season) / "standings.json", {"records": records, "season": season})


def _export_stage_matrix(season: int, round_num: int, stage: str, output_dir: Path) -> bool:
    """Export a single stage prediction matrix. Returns True if file existed."""
    path = PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_{stage}.parquet"
    if not path.exists():
        return False

    df = pd.read_parquet(path)
    rows = matrix_to_records(df)
    data = {
        "lap": None,
        "drivers": df.index.tolist(),
        "rows": rows,
    }
    _write_json(output_dir / str(season) / f"r{round_num}" / f"{stage}.json", data)
    return True


def _export_circuit_history(season: int, round_num: int, output_dir: Path) -> None:
    """Export bundled circuit history for all drivers in a round."""
    with get_session() as session:
        # Get all drivers who have results this season (or previous if no results yet)
        drivers = session.execute(
            sql_text("""
                SELECT DISTINCT d.full_name
                FROM results r
                JOIN drivers d ON r.driver_id = d.id
                JOIN races ra ON r.race_id = ra.id
                WHERE ra.season = :season OR ra.season = :prev
                ORDER BY d.full_name
            """),
            {"season": season, "prev": season - 1},
        ).fetchall()

        all_records: dict[str, list[dict]] = {}
        for (driver_name,) in drivers:
            rows = session.execute(
                sql_text("""
                    SELECT ra.season, r.grid, r.position, r.status, r.points, r.laps
                    FROM results r
                    JOIN drivers d ON r.driver_id = d.id
                    JOIN races ra ON r.race_id = ra.id
                    WHERE d.full_name = :driver
                      AND ra.circuit_id = (
                          SELECT circuit_id FROM races
                          WHERE season = :season AND round = :round
                      )
                      AND ra.season < :season
                    ORDER BY ra.season DESC
                """),
                {"driver": driver_name, "season": season, "round": round_num},
            ).fetchall()

            records = []
            for row in rows:
                records.append({
                    "season": int(row[0]),
                    "grid": to_json_safe(row[1]),
                    "position": to_json_safe(row[2]),
                    "status": str(row[3]) if row[3] else None,
                    "points": to_json_safe(row[4]),
                    "laps": to_json_safe(row[5]),
                })
            if records:
                all_records[driver_name] = records

    _write_json(
        output_dir / str(season) / f"r{round_num}" / "circuit_history.json",
        all_records,
    )


def _export_live_data(season: int, round_num: int, output_dir: Path) -> None:
    """Export live prediction data: laps list, evolution, and per-lap matrices."""
    live_dir = output_dir / str(season) / f"r{round_num}" / "live"

    # Find all lap parquet files
    pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
    files = sorted(LIVE_PREDICTIONS_DIR.glob(pattern))
    laps: list[int] = []
    for f in files:
        try:
            lap = int(f.stem.split("_lap")[-1])
            laps.append(lap)
        except ValueError:
            continue
    laps.sort()

    if not laps:
        return

    # laps.json
    _write_json(live_dir / "laps.json", {"laps": laps})

    # Per-lap combined files (matrix + snapshots)
    for lap in laps:
        lap_path = LIVE_PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_lap{lap}.parquet"
        if not lap_path.exists():
            continue
        df = pd.read_parquet(lap_path)
        rows = matrix_to_records(df)

        # Try to get snapshots from DB
        snapshot_records = []
        try:
            from src.database.queries import get_live_snapshots, get_race

            with get_session() as session:
                race = get_race(session, season, round_num)
                if race:
                    snap_df = get_live_snapshots(session, race.id, lap_number=lap)
                    if not snap_df.empty:
                        snapshot_records = snap_df.to_dict("records")
                        for rec in snapshot_records:
                            for k, v in list(rec.items()):
                                rec[k] = to_json_safe(v)
        except Exception:
            pass

        lap_data = {
            "lap": lap,
            "drivers": df.index.tolist(),
            "rows": rows,
            "snapshots": snapshot_records,
        }
        _write_json(live_dir / f"lap_{lap}.json", lap_data)

    # Evolution data
    evo_path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
    if evo_path.exists():
        df = pd.read_parquet(evo_path)
        records = []
        for _, row in df.iterrows():
            rec: dict = {}
            for col in df.columns:
                rec[col] = to_json_safe(row[col])
            records.append(rec)
        _write_json(live_dir / "evolution.json", {"records": records})


def export_round(season: int, round_num: int, output_dir: Path) -> None:
    """Export all data for a single round."""
    logger.info(f"Exporting round {round_num} for {season}...")

    # Stage matrices
    for stage in STAGE_PRIORITY:
        _export_stage_matrix(season, round_num, stage, output_dir)

    # Circuit history (bundled for all drivers)
    _export_circuit_history(season, round_num, output_dir)

    # Live data
    _export_live_data(season, round_num, output_dir)


def export_season(season: int, output_dir: Path) -> None:
    """Export all data for a full season: races, teams, standings, and all rounds."""
    logger.info(f"Exporting season {season} to {output_dir}...")

    races = _export_races(season, output_dir)
    _export_teams(season, output_dir)
    _export_standings(season, output_dir)

    # Export each round that has data
    for race in races:
        round_num = race["round"]
        if race["available_stages"] or race["has_live"]:
            export_round(season, round_num, output_dir)

    logger.info(f"Season {season} export complete.")
