"""Historical Results page - browse past race results."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Historical Results", page_icon="📜", layout="wide")
st.title("📜 Historical Results")


@st.cache_data(ttl=300)
def load_race_results(season: int):
    """Load all race results for a season."""
    from src.database.connection import get_session
    from src.database.queries import get_all_results_df

    with get_session() as session:
        df = get_all_results_df(session, min_season=season)

    return df[df["season"] == season]


@st.cache_data(ttl=300)
def load_seasons():
    """Get available seasons."""
    from sqlalchemy import func
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


seasons = load_seasons()
if not seasons:
    st.warning("No data loaded.")
    st.stop()

selected_season = st.sidebar.selectbox("Season", seasons, index=0)

results_df = load_race_results(selected_season)

if results_df.empty:
    st.info(f"No results for {selected_season}")
    st.stop()

# Race selector
races = results_df.groupby(["round", "race_name"]).size().reset_index()
race_options = {f"R{row['round']}: {row['race_name']}": row["round"] for _, row in races.iterrows()}

selected_race = st.sidebar.selectbox("Race", list(race_options.keys()))
selected_round = race_options[selected_race]

race_results = results_df[results_df["round"] == selected_round].sort_values("position")

st.subheader(f"{selected_race} ({selected_season})")

# Results table
display_cols = ["position", "driver_name", "team_name", "grid", "points", "status", "laps"]
available_cols = [c for c in display_cols if c in race_results.columns]

st.dataframe(
    race_results[available_cols].rename(columns={
        "position": "Pos",
        "driver_name": "Driver",
        "team_name": "Team",
        "grid": "Grid",
        "points": "Points",
        "status": "Status",
        "laps": "Laps",
    }),
    use_container_width=True,
    hide_index=True,
)

# Grid vs Finish visualization
st.subheader("Grid vs Finish Position")
race_viz = race_results.dropna(subset=["grid", "position"])
if not race_viz.empty:
    fig = px.scatter(
        race_viz,
        x="grid",
        y="position",
        text="driver_name",
        color="points",
        color_continuous_scale="YlOrRd",
        labels={"grid": "Grid Position", "position": "Finish Position", "points": "Points"},
    )
    # Add diagonal reference line
    max_pos = max(race_viz["grid"].max(), race_viz["position"].max())
    fig.add_shape(type="line", x0=1, y0=1, x1=max_pos, y1=max_pos,
                  line=dict(color="gray", dash="dash"))
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(height=500, yaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
