"""Season Standings page - WDC/WCC tables and charts."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Season Standings", page_icon="🏆", layout="wide")
st.title("🏆 Season Standings")


@st.cache_data(ttl=300)
def load_standings(season: int):
    """Load standings for a season."""
    from src.database.connection import get_session
    from src.database.queries import get_standings_df

    with get_session() as session:
        driver_standings = get_standings_df(session, season, "driver")
        constructor_standings = get_standings_df(session, season, "constructor")

    return driver_standings, constructor_standings


@st.cache_data(ttl=300)
def load_available_seasons():
    """Get list of seasons with data."""
    from src.database.connection import get_session
    from src.database.models import Race

    with get_session() as session:
        seasons = (
            session.query(Race.season)
            .distinct()
            .order_by(Race.season.desc())
            .all()
        )
    return [s[0] for s in seasons]


@st.cache_data(ttl=300)
def load_predicted_standings(season: int):
    """Load predicted final standings from championship predictions."""
    from src.database.connection import get_session
    from src.database.models import ChampionshipPrediction, Driver, Team

    with get_session() as session:
        # WDC
        wdc_rows = (
            session.query(
                ChampionshipPrediction.predicted_position,
                ChampionshipPrediction.predicted_points,
                ChampionshipPrediction.probability,
                Driver.full_name,
            )
            .join(Driver, ChampionshipPrediction.driver_id == Driver.id)
            .filter(
                ChampionshipPrediction.season == season,
                ChampionshipPrediction.prediction_type == "wdc",
            )
            .order_by(ChampionshipPrediction.predicted_position)
            .all()
        )

        # WCC
        wcc_rows = (
            session.query(
                ChampionshipPrediction.predicted_position,
                ChampionshipPrediction.predicted_points,
                ChampionshipPrediction.probability,
                Team.name,
            )
            .join(Team, ChampionshipPrediction.team_id == Team.id)
            .filter(
                ChampionshipPrediction.season == season,
                ChampionshipPrediction.prediction_type == "wcc",
            )
            .order_by(ChampionshipPrediction.predicted_position)
            .all()
        )

    wdc_pred = [
        {"Pos": r.predicted_position, "Driver": r.full_name,
         "Predicted Points": r.predicted_points, "Title Prob": r.probability}
        for r in wdc_rows
    ] if wdc_rows else None

    wcc_pred = [
        {"Pos": r.predicted_position, "Constructor": r.name,
         "Predicted Points": r.predicted_points, "Title Prob": r.probability}
        for r in wcc_rows
    ] if wcc_rows else None

    return wdc_pred, wcc_pred


seasons = load_available_seasons()
if not seasons:
    st.warning("No data loaded yet.")
    st.stop()

selected_season = st.sidebar.selectbox("Season", seasons, index=0)

driver_standings, constructor_standings = load_standings(selected_season)

# Driver Championship
st.subheader(f"{selected_season} Driver Championship")
if not driver_standings.empty:
    # Get latest round standings
    latest_round = driver_standings["round"].max()
    latest = driver_standings[driver_standings["round"] == latest_round].sort_values("position")

    st.dataframe(
        latest[["position", "driver_name", "team_name", "points"]].rename(
            columns={
                "position": "Pos",
                "driver_name": "Driver",
                "team_name": "Team",
                "points": "Points",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(f"No driver standings data for {selected_season}")

st.divider()

# Constructor Championship
st.subheader(f"{selected_season} Constructor Championship")
if not constructor_standings.empty:
    latest_round = constructor_standings["round"].max()
    latest = constructor_standings[constructor_standings["round"] == latest_round].sort_values("position")

    st.dataframe(
        latest[["position", "team_name", "points"]].rename(
            columns={
                "position": "Pos",
                "team_name": "Constructor",
                "points": "Points",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info(f"No constructor standings data for {selected_season}")

# --- Predicted Final Standings ---
st.divider()
st.subheader(f"Predicted Final Standings ({selected_season})")

import pandas as pd

wdc_pred, wcc_pred = load_predicted_standings(selected_season)

if wdc_pred:
    st.markdown("**Predicted Driver Championship**")
    wdc_pred_df = pd.DataFrame(wdc_pred)
    st.dataframe(
        wdc_pred_df.style.format({
            "Predicted Points": "{:.0f}",
            "Title Prob": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.caption("No predicted standings available. Run: `python -m src.pipeline.run championship`")

if wcc_pred:
    st.markdown("**Predicted Constructor Championship**")
    wcc_pred_df = pd.DataFrame(wcc_pred)
    st.dataframe(
        wcc_pred_df.style.format({
            "Predicted Points": "{:.0f}",
            "Title Prob": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
