"""Race Predictions page - probability heatmap and winner predictions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Race Predictions", page_icon="🏁", layout="wide")
st.title("🏁 Race Predictions")


@st.cache_data(ttl=300)
def load_available_races():
    """Get races with predictions grouped by season."""
    from src.database.connection import get_session
    from src.database.models import Prediction, Race

    with get_session() as session:
        # Get seasons that have predictions
        races_with_preds = (
            session.query(Race.season, Race.round, Race.race_name, Race.id)
            .join(Prediction, Prediction.race_id == Race.id)
            .distinct()
            .order_by(Race.season.desc(), Race.round)
            .all()
        )

        if not races_with_preds:
            return {}, []

        seasons = sorted(set(r.season for r in races_with_preds), reverse=True)
        race_map = {}
        for r in races_with_preds:
            if r.season not in race_map:
                race_map[r.season] = []
            race_map[r.season].append({
                "round": r.round,
                "name": r.race_name,
                "id": r.id,
                "label": f"R{r.round}: {r.race_name}",
            })

    return race_map, seasons


@st.cache_data(ttl=300)
def load_stages_for_race(race_id: int):
    """Get available prediction stages for a race."""
    from src.database.connection import get_session
    from src.database.queries import get_available_stages

    with get_session() as session:
        return get_available_stages(session, race_id)


@st.cache_data(ttl=300)
def load_predictions(race_id: int, stage: str | None = None):
    """Load predictions for a specific race and stage."""
    from src.database.connection import get_session
    from src.database.models import Driver, Prediction, Race

    with get_session() as session:
        race = session.query(Race).filter(Race.id == race_id).first()
        if not race:
            return None, None

        query = (
            session.query(
                Prediction.predicted_position,
                Prediction.probability,
                Prediction.stage,
                Driver.full_name,
                Driver.code,
            )
            .join(Driver, Prediction.driver_id == Driver.id)
            .filter(Prediction.race_id == race_id)
        )
        if stage:
            query = query.filter(Prediction.stage == stage)
        else:
            # Get the latest stage available
            query = query.order_by(Prediction.created_at.desc())

        predictions = query.all()
        if not predictions:
            return None, None

        pred_data = [
            {
                "driver": p.full_name,
                "code": p.code,
                "predicted_position": p.predicted_position,
                "win_probability": p.probability,
                "stage": p.stage,
            }
            for p in predictions
        ]

        race_info = {
            "name": race.race_name,
            "season": race.season,
            "round": race.round,
            "date": race.race_date,
        }

        return pd.DataFrame(pred_data), race_info


@st.cache_data(ttl=300)
def load_probability_matrix(race_id: int, stage: str | None = None):
    """Load probability matrix from parquet."""
    from src.database.connection import get_session
    from src.database.models import Race
    from src.database.queries import get_prediction_matrix_path

    with get_session() as session:
        race = session.query(Race).filter(Race.id == race_id).first()
        if not race:
            return None

        if stage:
            path = get_prediction_matrix_path(session, race_id, stage)
            if path and Path(path).exists():
                return pd.read_parquet(path)

    # Fallback to latest_matrix
    matrix_path = Path("data/processed/predictions/latest_matrix.parquet")
    if matrix_path.exists():
        return pd.read_parquet(matrix_path)
    return None


@st.cache_data(ttl=300)
def load_weather(race_id: int):
    """Load weather data for a race."""
    from src.database.connection import get_session
    from src.database.queries import get_race_weather

    with get_session() as session:
        return get_race_weather(session, race_id)


# --- Sidebar: Race & Stage Selection ---
race_map, seasons = load_available_races()

if not race_map:
    st.info("No predictions available yet. Run the prediction pipeline first:")
    st.code("make pipeline", language="bash")
    st.stop()

selected_season = st.sidebar.selectbox("Season", seasons, index=0)
race_options = race_map.get(selected_season, [])
race_labels = [r["label"] for r in race_options]
selected_race_idx = st.sidebar.selectbox(
    "Race", range(len(race_labels)), format_func=lambda i: race_labels[i]
)
selected_race = race_options[selected_race_idx]

# Stage selector
stages = load_stages_for_race(selected_race["id"])
stage_labels = {
    "pre_weekend": "Pre-Weekend",
    "post_qualifying": "Post-Qualifying",
    "post_sprint": "Post-Sprint",
    "race_eve": "Race Eve",
}

if stages:
    stage_display = [stage_labels.get(s, s) for s in stages]
    selected_stage_idx = st.sidebar.radio(
        "Prediction Stage", range(len(stages)), format_func=lambda i: stage_display[i]
    )
    selected_stage = stages[selected_stage_idx]
else:
    selected_stage = None

# Load data
pred_df, race_info = load_predictions(selected_race["id"], selected_stage)
prob_matrix = load_probability_matrix(selected_race["id"], selected_stage)
weather = load_weather(selected_race["id"])

if pred_df is not None and race_info is not None:
    # Race info header
    stage_label = stage_labels.get(selected_stage, "") if selected_stage else ""
    st.markdown(
        f"### {race_info['name']} ({race_info['season']} Season, Round {race_info['round']})"
    )
    if race_info.get("date"):
        st.markdown(f"**Race Date:** {race_info['date']}")
    if stage_label:
        st.markdown(f"**Stage:** {stage_label}")

    # Weather panel
    if weather:
        st.divider()
        wcol1, wcol2, wcol3, wcol4 = st.columns(4)
        wcol1.metric("Temperature", f"{weather['air_temp']:.1f} °C")
        wcol2.metric("Rain Probability", f"{weather['rain_prob']:.0%}")
        wcol3.metric("Wind Speed", f"{weather['wind_speed']:.1f} km/h")
        wcol4.metric("Humidity", f"{weather.get('humidity', 0):.0f}%")

    st.divider()

    # Sort by win probability
    pred_df = pred_df.sort_values("win_probability", ascending=False)

    # Winner prediction
    col1, col2, col3 = st.columns(3)
    if len(pred_df) >= 3:
        with col1:
            winner = pred_df.iloc[0]
            st.markdown("### 🥇 Predicted Winner")
            st.markdown(f"## {winner['driver']}")
            st.markdown(f"**Win Probability: {winner['win_probability']:.1%}**")

        with col2:
            p2 = pred_df.iloc[1]
            st.markdown("### 🥈 P2")
            st.markdown(f"## {p2['driver']}")
            st.markdown(f"**Win Probability: {p2['win_probability']:.1%}**")

        with col3:
            p3 = pred_df.iloc[2]
            st.markdown("### 🥉 P3")
            st.markdown(f"## {p3['driver']}")
            st.markdown(f"**Win Probability: {p3['win_probability']:.1%}**")

    st.divider()

    # Stage comparison delta (if we have multiple stages)
    if selected_stage and selected_stage != "pre_weekend" and len(stages) > 1:
        prev_stage = stages[stages.index(selected_stage) - 1]
        prev_df, _ = load_predictions(selected_race["id"], prev_stage)
        if prev_df is not None:
            st.subheader(f"Changes from {stage_labels.get(prev_stage, prev_stage)}")
            merged = pred_df.merge(
                prev_df[["driver", "win_probability", "predicted_position"]],
                on="driver",
                suffixes=("", "_prev"),
            )
            if not merged.empty:
                merged["prob_delta"] = merged["win_probability"] - merged["win_probability_prev"]
                merged["pos_delta"] = merged["predicted_position_prev"] - merged["predicted_position"]
                delta_display = merged[["driver", "win_probability", "prob_delta", "pos_delta"]].copy()
                delta_display.columns = ["Driver", "Win Prob", "Win Prob Delta", "Position Change"]
                st.dataframe(
                    delta_display.sort_values("Win Prob", ascending=False).style.format({
                        "Win Prob": "{:.2%}",
                        "Win Prob Delta": "{:+.2%}",
                        "Position Change": "{:+.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
            st.divider()

    # Win probability bar chart
    st.subheader("Win Probabilities")
    fig = px.bar(
        pred_df,
        x="driver",
        y="win_probability",
        color="win_probability",
        color_continuous_scale="RdYlGn_r",
        labels={"win_probability": "Win Probability", "driver": "Driver"},
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_tickformat=".0%",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability matrix heatmap
    if prob_matrix is not None:
        st.subheader("Position Probability Matrix")
        st.markdown("Each cell shows the probability of a driver finishing in that position.")

        # Only show P1-P10 for clarity
        display_cols = [f"P{i}" for i in range(1, 11) if f"P{i}" in prob_matrix.columns]
        if display_cols:
            heatmap_data = prob_matrix[display_cols].head(15)

            fig = px.imshow(
                heatmap_data.values,
                x=display_cols,
                y=heatmap_data.index.tolist(),
                color_continuous_scale="YlOrRd",
                aspect="auto",
                labels={"color": "Probability"},
            )
            fig.update_layout(height=500, xaxis_title="Position", yaxis_title="Driver")
            st.plotly_chart(fig, use_container_width=True)

    # Full predictions table
    st.subheader("All Predictions")
    st.dataframe(
        pred_df[["driver", "predicted_position", "win_probability"]].style.format(
            {"win_probability": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

else:
    st.info("No predictions available yet. Run the prediction pipeline first:")
    st.code("make pipeline", language="bash")
    st.markdown("Or run predictions for a specific race:")
    st.code("python -m src.pipeline.run predict --round 1", language="bash")
