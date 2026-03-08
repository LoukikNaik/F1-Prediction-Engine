"""Auto-detect which prediction stage to use based on race schedule."""

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.database.models import Prediction, Race
from src.utils.constants import PredictionStage
from src.utils.logger import logger


def detect_stage(race: Race) -> PredictionStage:
    """Detect the appropriate prediction stage based on current time vs race schedule.

    Args:
        race: Race object with timing info.

    Returns:
        The most appropriate PredictionStage.
    """
    now = datetime.now(timezone.utc)

    # Parse race datetime
    race_dt = _parse_session_dt(race.race_date, race.race_time)
    quali_dt = _parse_session_dt(race.qualifying_date, race.qualifying_time)
    sprint_dt = _parse_session_dt(race.sprint_date, race.sprint_time)

    # If race has started or passed, return race_eve (latest available)
    if race_dt and now >= race_dt:
        return PredictionStage.RACE_EVE

    # If sprint is done (sprint weekends only)
    if sprint_dt and now >= sprint_dt:
        return PredictionStage.POST_SPRINT

    # If qualifying is done
    if quali_dt and now >= quali_dt:
        return PredictionStage.POST_QUALIFYING

    return PredictionStage.PRE_WEEKEND


def get_available_stages(session: Session, race_id: int) -> list[PredictionStage]:
    """Get list of stages that have stored predictions for a race.

    Args:
        session: Database session.
        race_id: Race ID.

    Returns:
        List of PredictionStage values with stored predictions.
    """
    rows = (
        session.query(Prediction.stage)
        .filter(Prediction.race_id == race_id)
        .distinct()
        .all()
    )
    stages = []
    for (stage_val,) in rows:
        try:
            stages.append(PredictionStage(stage_val))
        except ValueError:
            continue
    # Return in chronological order
    stage_order = list(PredictionStage)
    return sorted(stages, key=lambda s: stage_order.index(s))


def _parse_session_dt(date_str: str | None, time_str: str | None) -> datetime | None:
    """Parse a date + time string into a UTC datetime."""
    if not date_str:
        return None
    try:
        if time_str:
            return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None
