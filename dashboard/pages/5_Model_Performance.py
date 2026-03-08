"""Model Performance page - backtesting and accuracy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
st.title("📈 Model Performance")
st.markdown("Evaluate model accuracy through backtesting on historical races.")


@st.cache_data(ttl=600)
def run_backtest(season: int):
    """Run backtest on a season and return accuracy metrics."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_all_results_df
        from src.models.predictor import build_race_features
        from src.models.trainer import F1Ensemble
        from src.pipeline.processors.feature_engineer import (
            engineer_features,
            get_feature_columns,
        )

        model = F1Ensemble.load()

        with get_session() as session:
            results_df = get_all_results_df(session, min_season=2000)

        if results_df.empty:
            return None

        season_races = results_df[results_df["season"] == season]["round"].unique()
        metrics = []

        all_featured = engineer_features(results_df)
        feature_cols = get_feature_columns()

        for round_num in sorted(season_races):
            race_df = all_featured[
                (all_featured["season"] == season) & (all_featured["round"] == round_num)
            ].copy()

            race_df = race_df.dropna(subset=feature_cols + ["position_filled"])
            if len(race_df) < 5:
                continue

            proba = model.predict_proba(race_df)
            pred_positions = proba.argmax(axis=1) + 1
            actual = race_df["position_filled"].values

            exact = (pred_positions == actual).mean()
            top3 = np.mean([
                actual[i] in np.argsort(proba[i])[-3:] + 1
                for i in range(len(actual))
            ])
            top5 = np.mean([
                actual[i] in np.argsort(proba[i])[-5:] + 1
                for i in range(len(actual))
            ])

            race_name = race_df["race_name"].iloc[0] if "race_name" in race_df.columns else f"R{round_num}"
            metrics.append({
                "round": round_num,
                "race": race_name,
                "exact_accuracy": exact,
                "top3_accuracy": top3,
                "top5_accuracy": top5,
            })

        import pandas as pd
        return pd.DataFrame(metrics) if metrics else None
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None


season = st.sidebar.number_input("Backtest Season", min_value=2000, max_value=2025, value=2024)

if st.button("Run Backtest"):
    with st.spinner("Running backtest..."):
        metrics_df = run_backtest(season)

    if metrics_df is not None and not metrics_df.empty:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Exact Accuracy", f"{metrics_df['exact_accuracy'].mean():.1%}")
        col2.metric("Avg Top-3 Accuracy", f"{metrics_df['top3_accuracy'].mean():.1%}")
        col3.metric("Avg Top-5 Accuracy", f"{metrics_df['top5_accuracy'].mean():.1%}")

        st.divider()

        # Accuracy by race
        st.subheader("Accuracy by Race")
        fig = px.bar(
            metrics_df,
            x="race",
            y=["exact_accuracy", "top3_accuracy", "top5_accuracy"],
            barmode="group",
            labels={"value": "Accuracy", "variable": "Metric", "race": "Race"},
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            yaxis_tickformat=".0%",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        st.subheader("Detailed Results")
        st.dataframe(
            metrics_df.style.format({
                "exact_accuracy": "{:.1%}",
                "top3_accuracy": "{:.1%}",
                "top5_accuracy": "{:.1%}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("No backtest results. Ensure model is trained and data is loaded.")
else:
    st.info("Click 'Run Backtest' to evaluate model performance on a season.")
