"""F1 Prediction Engine - Streamlit Dashboard."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="F1 Prediction Engine",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏎️ F1 Race Prediction Engine")
st.markdown("""
**ML-powered race predictions** using an ensemble of LightGBM, XGBoost,
Logistic Regression + Monte Carlo simulations.

Features multi-stage predictions (pre-weekend through race eve),
live lap-by-lap race tracking, weather integration, and full-season
WDC/WCC championship forecasts.

Use the sidebar to navigate between 7 pages.
""")

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.markdown("Select a page above to explore predictions, standings, and more.")

# Quick stats
try:
    from src.database.connection import get_session
    from src.database.models import Driver, Race, Result, Team

    with get_session() as session:
        n_races = session.query(Race).count()
        n_drivers = session.query(Driver).count()
        n_results = session.query(Result).count()
        n_teams = session.query(Team).count()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Races", f"{n_races:,}")
    col2.metric("Drivers", f"{n_drivers:,}")
    col3.metric("Results", f"{n_results:,}")
    col4.metric("Teams", f"{n_teams:,}")
except Exception as e:
    st.warning(f"Database not initialized. Run `make fetch-kaggle` first. Error: {e}")
