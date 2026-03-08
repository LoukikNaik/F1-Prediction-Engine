"""FastAPI backend for F1 Prediction Engine dashboard."""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import LIVE_PREDICTIONS_DIR, PREDICTIONS_DIR

app = FastAPI(title="F1 Prediction Engine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://f1.loukik.dev",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "null",  # local file:// origin
    ],
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_json_safe(val):
    """Convert a single value to a JSON-serialisable Python type."""
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 6)
    if isinstance(val, float):
        return round(val, 6)
    if isinstance(val, int):
        return val
    if hasattr(val, "isoformat"):
        return val.isoformat()
    return str(val)


def _matrix_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a prediction matrix DataFrame to JSON-friendly records."""
    records = []
    pos_cols = [f"P{i}" for i in range(1, 23) if f"P{i}" in df.columns]
    summary_cols = [c for c in ("expected_position", "win_prob", "podium_prob", "top5_prob") if c in df.columns]
    for driver_name in df.index:
        row = df.loc[driver_name]
        rec: dict = {"driver_name": driver_name}
        for col in pos_cols + summary_cols:
            rec[col] = _to_json_safe(row[col])
        records.append(rec)
    return records


STAGE_PRIORITY = ("race_eve", "post_sprint", "post_qualifying", "pre_weekend")


def _find_matrix_path(
    season: int,
    round_num: int,
    lap: int | None = None,
    stage: str | None = None,
) -> Path | None:
    """Locate a prediction matrix parquet file."""
    if lap is not None:
        path = LIVE_PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_lap{lap}.parquet"
        if path.exists():
            return path
    # If an explicit stage is requested, return only that exact file
    if stage is not None:
        path = PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_{stage}.parquet"
        return path if path.exists() else None
    # Fallback through stages in priority order
    for s in STAGE_PRIORITY:
        path = PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_{s}.parquet"
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/laps")
def get_laps(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
) -> dict:
    """Return available lap numbers that have prediction snapshots."""
    pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
    files = sorted(LIVE_PREDICTIONS_DIR.glob(pattern))
    laps: list[int] = []
    for f in files:
        try:
            lap = int(f.stem.split("_lap")[-1])
            laps.append(lap)
        except ValueError:
            continue
    return {"laps": sorted(laps)}


@app.get("/api/races")
def get_races(
    season: int = Query(...),
) -> JSONResponse:
    """Return races for a season with available prediction stages per round."""
    try:
        from sqlalchemy import text as sql_text

        from src.database.connection import get_session

        with get_session() as session:
            rows = session.execute(sql_text("""
                SELECT r.round, r.race_name, c.name, c.country
                FROM races r
                LEFT JOIN circuits c ON r.circuit_id = c.id
                WHERE r.season = :season
                ORDER BY r.round
            """), {"season": season}).fetchall()

            races = []
            for row in rows:
                round_num = int(row[0])
                # Check which stage parquet files exist
                available_stages = []
                for stage in STAGE_PRIORITY:
                    path = PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_{stage}.parquet"
                    if path.exists():
                        available_stages.append(stage)
                # Check if live lap files exist
                live_pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
                has_live = bool(list(LIVE_PREDICTIONS_DIR.glob(live_pattern)))

                races.append({
                    "round": round_num,
                    "race_name": str(row[1]) if row[1] else f"Round {round_num}",
                    "circuit_name": str(row[2]) if row[2] else None,
                    "country": str(row[3]) if row[3] else None,
                    "available_stages": available_stages,
                    "has_live": has_live,
                })
            return JSONResponse({"races": races})
    except Exception as e:
        return JSONResponse({"races": [], "error": str(e)})


@app.get("/api/matrix")
def get_matrix(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
    lap: int | None = Query(None),
    stage: str | None = Query(None),
) -> JSONResponse:
    """Return the prediction matrix for a given lap (or race_eve fallback)."""
    path = _find_matrix_path(season, round_num, lap, stage)
    if path is None:
        return JSONResponse({"error": "Matrix not found"}, status_code=404)
    df = pd.read_parquet(path)

    # Inject any missing drivers (e.g. DNS) with zeroed-out probabilities
    try:
        from sqlalchemy import text as sql_text
        from src.database.connection import get_session

        with get_session() as session:
            all_drivers = [r[0] for r in session.execute(sql_text("""
                SELECT DISTINCT d.full_name
                FROM results r
                JOIN drivers d ON r.driver_id = d.id
                JOIN races ra ON r.race_id = ra.id
                WHERE ra.season = :season AND ra.round = :round
            """), {"season": season, "round": round_num}).fetchall()]

            missing = [d for d in all_drivers if d not in df.index]
            if missing:
                pos_cols = [c for c in df.columns if c.startswith("P")]
                zero_row = {c: 0.0 for c in df.columns}
                zero_row["expected_position"] = float(len(df) + 1)
                for name in missing:
                    df.loc[name] = zero_row
    except Exception:
        pass

    drivers = df.index.tolist()
    rows = _matrix_to_records(df)
    return JSONResponse({
        "lap": lap,
        "drivers": drivers,
        "rows": rows,
    })


@app.get("/api/driver-evolution")
def get_driver_evolution(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
    driver: str = Query(...),
) -> JSONResponse:
    """Return a single driver's position probabilities across all laps."""
    pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
    files = sorted(LIVE_PREDICTIONS_DIR.glob(pattern))
    records = []
    for f in files:
        try:
            lap = int(f.stem.split("_lap")[-1])
        except ValueError:
            continue
        df = pd.read_parquet(f)
        if driver not in df.index:
            continue
        row = df.loc[driver]
        rec: dict = {"lap": lap}
        for col in df.columns:
            rec[col] = _to_json_safe(row[col])
        records.append(rec)
    records.sort(key=lambda r: r["lap"])
    return JSONResponse({"driver": driver, "records": records})


@app.get("/api/evolution")
def get_evolution(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
) -> JSONResponse:
    """Return prediction evolution data across laps."""
    path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
    if not path.exists():
        return JSONResponse({"records": []})
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        rec: dict = {}
        for col in df.columns:
            rec[col] = _to_json_safe(row[col])
        records.append(rec)
    return JSONResponse({"records": records})


@app.get("/api/snapshots")
def get_snapshots(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
    lap: int | None = Query(None),
) -> JSONResponse:
    """Return live prediction snapshots from the database."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_live_snapshots, get_race

        with get_session() as session:
            race = get_race(session, season, round_num)
            if not race:
                return JSONResponse({"records": []})
            df = get_live_snapshots(session, race.id, lap_number=lap)
            if df.empty:
                return JSONResponse({"records": []})
            records = df.to_dict("records")
            for rec in records:
                for k, v in list(rec.items()):
                    rec[k] = _to_json_safe(v)
            return JSONResponse({"records": records})
    except Exception as e:
        return JSONResponse({"records": [], "error": str(e)})


@app.get("/api/standings")
def get_standings(
    season: int = Query(...),
) -> JSONResponse:
    """Return championship standings for a season."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_standings_df

        with get_session() as session:
            df = get_standings_df(session, season, "driver")
            if not df.empty:
                latest = df[df["round"] == df["round"].max()]
                records = latest[["driver_name", "driver_code", "position", "points"]].to_dict("records")
                for rec in records:
                    for k, v in list(rec.items()):
                        rec[k] = _to_json_safe(v)
                return JSONResponse({"records": records, "season": season})

            # No standings yet — build 0-point grid from this season's drivers
            from sqlalchemy import text as sql_text

            rows = session.execute(sql_text("""
                SELECT DISTINCT d.full_name, d.code
                FROM results r
                JOIN drivers d ON r.driver_id = d.id
                JOIN races ra ON r.race_id = ra.id
                WHERE ra.season = :season
                ORDER BY d.full_name
            """), {"season": season}).fetchall()
            if rows:
                records = [
                    {"driver_name": name, "driver_code": code, "position": i + 1, "points": 0}
                    for i, (name, code) in enumerate(rows)
                ]
                return JSONResponse({"records": records, "season": season})

            return JSONResponse({"records": [], "season": season})
    except Exception as e:
        return JSONResponse({"records": [], "error": str(e)})


@app.get("/api/circuit-history")
def get_circuit_history(
    season: int = Query(...),
    round_num: int = Query(..., alias="round"),
    driver: str = Query(...),
) -> JSONResponse:
    """Return a driver's historical results at the circuit for this round."""
    try:
        from sqlalchemy import text as sql_text
        from src.database.connection import get_session

        with get_session() as session:
            rows = session.execute(sql_text("""
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
            """), {"driver": driver, "season": season, "round": round_num}).fetchall()

            records = []
            for row in rows:
                records.append({
                    "season": int(row[0]),
                    "grid": _to_json_safe(row[1]),
                    "position": _to_json_safe(row[2]),
                    "status": str(row[3]) if row[3] else None,
                    "points": _to_json_safe(row[4]),
                    "laps": _to_json_safe(row[5]),
                })
            return JSONResponse({"driver": driver, "records": records})
    except Exception as e:
        return JSONResponse({"driver": driver, "records": [], "error": str(e)})


@app.get("/api/teams")
def get_teams(
    season: int = Query(...),
) -> JSONResponse:
    """Return driver -> team mapping for a season."""
    try:
        from sqlalchemy import text as sql_text

        from src.database.connection import get_session

        with get_session() as session:
            rows = session.execute(sql_text("""
                SELECT d.full_name, t.name
                FROM results r
                JOIN drivers d ON r.driver_id = d.id
                JOIN teams t ON r.team_id = t.id
                JOIN races ra ON r.race_id = ra.id
                WHERE ra.season = :season
                ORDER BY ra.round DESC
            """), {"season": season}).fetchall()
            mapping: dict[str, str] = {}
            for name, team in rows:
                if name not in mapping:
                    mapping[name] = team
            if not mapping:
                # Fallback to previous season
                rows = session.execute(sql_text("""
                    SELECT d.full_name, t.name
                    FROM results r
                    JOIN drivers d ON r.driver_id = d.id
                    JOIN teams t ON r.team_id = t.id
                    JOIN races ra ON r.race_id = ra.id
                    WHERE ra.season = :season
                    ORDER BY ra.round DESC
                """), {"season": season - 1}).fetchall()
                for name, team in rows:
                    if name not in mapping:
                        mapping[name] = team
            return JSONResponse({"mapping": mapping})
    except Exception as e:
        return JSONResponse({"mapping": {}, "error": str(e)})
