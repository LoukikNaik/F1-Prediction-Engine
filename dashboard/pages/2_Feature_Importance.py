"""Feature Importance page - SHAP analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Feature Importance", page_icon="📊", layout="wide")
st.title("📊 Feature Importance")
st.markdown("What factors drive the model's predictions?")


@st.cache_data(ttl=600)
def compute_importance():
    """Compute feature importance from the trained model."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_all_results_df
        from src.models.explainer import get_feature_importance
        from src.models.trainer import F1Ensemble
        from src.pipeline.processors.feature_engineer import (
            engineer_features,
            get_feature_columns,
        )

        model = F1Ensemble.load()

        with get_session() as session:
            results_df = get_all_results_df(session, min_season=2020)

        if results_df.empty:
            return None

        featured_df = engineer_features(results_df)
        feature_cols = get_feature_columns()
        featured_df = featured_df.dropna(subset=feature_cols)

        # Sample for speed
        if len(featured_df) > 500:
            featured_df = featured_df.sample(500, random_state=42)

        importance_df = get_feature_importance(model, featured_df)
        return importance_df
    except Exception as e:
        st.error(f"Could not compute feature importance: {e}")
        return None


importance_df = compute_importance()

if importance_df is not None:
    # Bar chart
    fig = px.bar(
        importance_df,
        x="importance_pct",
        y="feature",
        orientation="h",
        color="importance_pct",
        color_continuous_scale="Viridis",
        labels={"importance_pct": "Importance (%)", "feature": "Feature"},
    )
    fig.update_layout(
        height=500,
        yaxis={"categoryorder": "total ascending"},
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.subheader("Feature Details")
    st.dataframe(
        importance_df.style.format({
            "importance": "{:.4f}",
            "importance_pct": "{:.1f}%",
        }),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("Train a model first to see feature importance.")
    st.code("python -m src.pipeline.run train", language="bash")
