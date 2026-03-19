"""Generate and install launchd plists for automatic race weekend predictions."""

import plistlib
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text as sql_text

from src.database.connection import get_session
from src.utils.logger import logger

PROJECT_ROOT = Path(__file__).parent.parent
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PREFIX = "com.f1engine"
VENV_PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")


def _parse_datetime(date_str: str, time_str: str | None) -> datetime | None:
    """Parse race date + time strings into a datetime."""
    if not date_str:
        return None
    try:
        if time_str:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


def _make_plist(
    label: str,
    trigger_time: datetime,
    round_num: int,
    stage: str,
    season: int,
    extra_cmds: list[str] | None = None,
) -> dict:
    """Build a launchd plist dict for a prediction job."""
    cmds = [
        f"cd {PROJECT_ROOT}",
        f"{VENV_PYTHON} -m src.pipeline.run predict --round {round_num} --stage {stage} --season {season}",
        f"{VENV_PYTHON} -m src.pipeline.run export --season {season} --round {round_num} --push",
    ]
    if extra_cmds:
        # Insert extra commands before the export step
        cmds = cmds[:1] + extra_cmds + cmds[1:]

    script = " && ".join(cmds)

    return {
        "Label": label,
        "ProgramArguments": ["/bin/zsh", "-c", script],
        "StartCalendarInterval": {
            "Month": trigger_time.month,
            "Day": trigger_time.day,
            "Hour": trigger_time.hour,
            "Minute": trigger_time.minute,
        },
        "StandardOutPath": str(PROJECT_ROOT / "logs" / f"{label}.log"),
        "StandardErrorPath": str(PROJECT_ROOT / "logs" / f"{label}.err"),
        "WorkingDirectory": str(PROJECT_ROOT),
    }


def _unload_existing() -> None:
    """Unload any existing f1engine launchd jobs."""
    for plist in LAUNCH_AGENTS_DIR.glob(f"{PLIST_PREFIX}.*.plist"):
        try:
            subprocess.run(
                ["launchctl", "unload", str(plist)],
                capture_output=True,
            )
            plist.unlink()
            logger.info(f"Removed old job: {plist.name}")
        except Exception:
            pass


def _install_plist(label: str, plist_data: dict) -> None:
    """Write plist file and load it."""
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = LAUNCH_AGENTS_DIR / f"{label}.plist"
    with open(path, "wb") as f:
        plistlib.dump(plist_data, f)
    subprocess.run(["launchctl", "load", str(path)], check=True)
    logger.info(f"Loaded: {path.name}")


def schedule_weekends(n_races: int = 2) -> None:
    """Query DB for upcoming races and install launchd jobs."""
    today = datetime.now().strftime("%Y-%m-%d")

    with get_session() as session:
        races = session.execute(
            sql_text("""
                SELECT r.season, r.round, r.race_name, r.race_date, r.race_time,
                       r.qualifying_date, r.qualifying_time,
                       r.sprint_date, r.sprint_time, r.is_sprint_weekend
                FROM races r
                WHERE r.race_date >= :today
                ORDER BY r.race_date
                LIMIT :limit
            """),
            {"today": today, "limit": n_races},
        ).fetchall()

    if not races:
        logger.warning("No upcoming races found in the database.")
        return

    # Ensure logs dir exists
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)

    # Remove old jobs
    _unload_existing()

    jobs: list[tuple[str, str, datetime]] = []  # (label, description, trigger_time)

    for row in races:
        season, round_num, _race_name = int(row[0]), int(row[1]), str(row[2])
        race_dt = _parse_datetime(row[3], row[4])
        quali_dt = _parse_datetime(row[5], row[6])
        sprint_dt = _parse_datetime(row[7], row[8])
        is_sprint = bool(row[9])

        # Post-qualifying: 2 hours after qualifying
        if quali_dt:
            trigger = quali_dt + timedelta(hours=2)
            if trigger > datetime.now():
                label = f"{PLIST_PREFIX}.r{round_num}_post_qualifying"
                plist = _make_plist(label, trigger, round_num, "post_qualifying", season)
                _install_plist(label, plist)
                jobs.append((label, f"R{round_num} Post-Qualifying", trigger))

        # Post-sprint: 1 hour after sprint (sprint weekends only)
        if is_sprint and sprint_dt:
            trigger = sprint_dt + timedelta(hours=1)
            if trigger > datetime.now():
                label = f"{PLIST_PREFIX}.r{round_num}_post_sprint"
                plist = _make_plist(label, trigger, round_num, "post_sprint", season)
                _install_plist(label, plist)
                jobs.append((label, f"R{round_num} Post-Sprint", trigger))

        # Race eve: 3 hours before race
        if race_dt:
            trigger = race_dt - timedelta(hours=3)
            if trigger > datetime.now():
                label = f"{PLIST_PREFIX}.r{round_num}_race_eve"
                plist = _make_plist(label, trigger, round_num, "race_eve", season)
                _install_plist(label, plist)
                jobs.append((label, f"R{round_num} Race Eve", trigger))

        # Post-race: 3 hours after race (export standings update)
        if race_dt:
            trigger = race_dt + timedelta(hours=3)
            if trigger > datetime.now():
                label = f"{PLIST_PREFIX}.r{round_num}_post_race"
                # Post-race just re-exports standings + full season data
                plist_data = _make_plist(label, trigger, round_num, "race_eve", season)
                # Override command to just export (no re-predict)
                export_script = " && ".join([
                    f"cd {PROJECT_ROOT}",
                    f"{VENV_PYTHON} -m src.pipeline.run export --season {season} --push",
                ])
                plist_data["ProgramArguments"] = ["/bin/zsh", "-c", export_script]
                _install_plist(label, plist_data)
                jobs.append((label, f"R{round_num} Post-Race Export", trigger))

    # Print summary
    if jobs:
        print("\n" + "=" * 70)
        print(f"Scheduled {len(jobs)} jobs for {n_races} race weekend(s)")
        print("=" * 70)
        for label, desc, trigger in sorted(jobs, key=lambda x: x[2]):
            print(f"  {desc:<30} {trigger:%Y-%m-%d %H:%M}  ({label})")
        print()
    else:
        print("No future jobs to schedule (all triggers are in the past).")
