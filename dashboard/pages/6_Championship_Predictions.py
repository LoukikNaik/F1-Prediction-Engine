"""Championship Predictions page - WDC/WCC forecasts with probability heatmaps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Championship Predictions", page_icon="🏆", layout="wide")
st.title("🏆 Championship Predictions")


@st.cache_data(ttl=300)
def load_championship_predictions(season: int):
    """Load championship predictions from database."""
    from src.database.connection import get_session
    from src.database.models import ChampionshipPrediction, Driver, Team

    with get_session() as session:
        # WDC predictions
        wdc_rows = (
            session.query(
                ChampionshipPrediction.predicted_position,
                ChampionshipPrediction.probability,
                ChampionshipPrediction.predicted_points,
                ChampionshipPrediction.after_round,
                Driver.full_name,
                Driver.code,
            )
            .join(Driver, ChampionshipPrediction.driver_id == Driver.id)
            .filter(
                ChampionshipPrediction.season == season,
                ChampionshipPrediction.prediction_type == "wdc",
            )
            .order_by(ChampionshipPrediction.predicted_position)
            .all()
        )

        wdc_data = [
            {
                "driver": r.full_name,
                "code": r.code,
                "predicted_position": r.predicted_position,
                "title_probability": r.probability,
                "predicted_points": r.predicted_points,
                "after_round": r.after_round,
            }
            for r in wdc_rows
        ]

        # WCC predictions
        wcc_rows = (
            session.query(
                ChampionshipPrediction.predicted_position,
                ChampionshipPrediction.probability,
                ChampionshipPrediction.predicted_points,
                ChampionshipPrediction.after_round,
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

        wcc_data = [
            {
                "team": r.name,
                "predicted_position": r.predicted_position,
                "title_probability": r.probability,
                "predicted_points": r.predicted_points,
                "after_round": r.after_round,
            }
            for r in wcc_rows
        ]

    return pd.DataFrame(wdc_data) if wdc_data else None, pd.DataFrame(wcc_data) if wcc_data else None


@st.cache_data(ttl=300)
def load_available_seasons():
    """Get seasons with championship predictions."""
    from src.database.connection import get_session
    from src.database.models import ChampionshipPrediction

    with get_session() as session:
        seasons = (
            session.query(ChampionshipPrediction.season)
            .distinct()
            .order_by(ChampionshipPrediction.season.desc())
            .all()
        )
    return [s[0] for s in seasons]


# Sidebar
seasons = load_available_seasons()

if not seasons:
    st.info("No championship predictions available yet. Generate them with:")
    st.code("python -m src.pipeline.run championship --season 2026", language="bash")

    # Offer to run predictions
    if st.sidebar.button("Generate Predictions"):
        with st.spinner("Running championship simulation..."):
            try:
                from src.models.championship_predictor import predict_championship

                wdc_df, wcc_df = predict_championship(season=2026, n_simulations=5000)
                st.success("Championship predictions generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    st.stop()

selected_season = st.sidebar.selectbox("Season", seasons, index=0)

# Simulation controls
st.sidebar.divider()
sim_count = st.sidebar.slider("Simulations", min_value=1000, max_value=10000, value=5000, step=1000)

if st.sidebar.button("Recalculate"):
    with st.spinner(f"Running {sim_count} simulations..."):
        try:
            from src.models.championship_predictor import predict_championship

            wdc_df, wcc_df = predict_championship(
                season=selected_season, n_simulations=sim_count
            )
            st.success("Predictions updated!")
            load_championship_predictions.clear()
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

# Load predictions
wdc_df, wcc_df = load_championship_predictions(selected_season)

# --- WDC Section ---
st.subheader(f"{selected_season} World Drivers' Championship")

if wdc_df is not None and not wdc_df.empty:
    after_round = wdc_df["after_round"].iloc[0]
    st.caption(f"Predictions based on data through Round {after_round}")

    # Summary table
    display_df = wdc_df[["driver", "predicted_points", "title_probability", "predicted_position"]].copy()
    display_df.columns = ["Driver", "Predicted Points", "Title Probability", "Predicted Pos"]

    st.dataframe(
        display_df.style.format({
            "Predicted Points": "{:.0f}",
            "Title Probability": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Title probability bar chart (top 8)
    top_drivers = wdc_df.head(8)
    fig = px.bar(
        top_drivers,
        x="driver",
        y="title_probability",
        color="title_probability",
        color_continuous_scale="RdYlGn_r",
        labels={"title_probability": "Title Probability", "driver": "Driver"},
        title="WDC Title Probability",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Points prediction chart
    fig2 = px.bar(
        wdc_df.head(10),
        x="driver",
        y="predicted_points",
        color="predicted_points",
        color_continuous_scale="Blues",
        labels={"predicted_points": "Predicted Points", "driver": "Driver"},
        title="Predicted Season Points",
    )
    fig2.update_layout(xaxis_tickangle=-45, height=400, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("No WDC predictions available")

st.divider()

# --- WCC Section ---
st.subheader(f"{selected_season} World Constructors' Championship")

if wcc_df is not None and not wcc_df.empty:
    display_wcc = wcc_df[["team", "predicted_points", "title_probability", "predicted_position"]].copy()
    display_wcc.columns = ["Constructor", "Predicted Points", "Title Probability", "Predicted Pos"]

    st.dataframe(
        display_wcc.style.format({
            "Predicted Points": "{:.0f}",
            "Title Probability": "{:.1%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Title probability bar chart
    fig = px.bar(
        wcc_df,
        x="team",
        y="title_probability",
        color="title_probability",
        color_continuous_scale="RdYlGn_r",
        labels={"title_probability": "Title Probability", "team": "Constructor"},
        title="WCC Title Probability",
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No WCC predictions available")
