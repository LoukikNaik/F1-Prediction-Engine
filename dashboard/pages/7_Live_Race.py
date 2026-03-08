"""Live Race Predictions — browse predictions at each lap with evolution charts."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import LIVE_DASHBOARD_REFRESH, LIVE_PREDICTIONS_DIR

st.set_page_config(page_title="Live Race", page_icon="📡", layout="wide")
st.title("📡 Live Race Predictions")

# --- F1 Team colors and logos (2025/2026 season) ---
TEAM_COLORS: dict[str, str] = {
    "Red Bull": "#3671C6",
    "Mercedes": "#27F4D2",
    "Ferrari": "#E8002D",
    "McLaren": "#FF8000",
    "Aston Martin": "#229971",
    "Alpine F1 Team": "#FF87BC",
    "Alpine": "#FF87BC",
    "Williams": "#64C4FF",
    "RB F1 Team": "#6692FF",
    "Racing Bulls": "#6692FF",
    "Haas F1 Team": "#B6BABD",
    "Sauber": "#52E252",
    "Audi": "#52E252",
    "Cadillac": "#C0C0C0",
}

_F1_LOGO_BASE = (
    "https://media.formula1.com/image/upload/c_lfill,w_48/q_auto"
    "/v1740000000/common/f1/2026"
)
TEAM_LOGOS: dict[str, str] = {
    "Red Bull": f"{_F1_LOGO_BASE}/redbullracing/2026redbullracinglogowhite.webp",
    "Mercedes": f"{_F1_LOGO_BASE}/mercedes/2026mercedeslogowhite.webp",
    "Ferrari": f"{_F1_LOGO_BASE}/ferrari/2026ferrarilogowhite.webp",
    "McLaren": f"{_F1_LOGO_BASE}/mclaren/2026mclarenlogowhite.webp",
    "Aston Martin": f"{_F1_LOGO_BASE}/astonmartin/2026astonmartinlogowhite.webp",
    "Alpine F1 Team": f"{_F1_LOGO_BASE}/alpine/2026alpinelogowhite.webp",
    "Alpine": f"{_F1_LOGO_BASE}/alpine/2026alpinelogowhite.webp",
    "Williams": f"{_F1_LOGO_BASE}/williams/2026williamslogowhite.webp",
    "RB F1 Team": f"{_F1_LOGO_BASE}/racingbulls/2026racingbullslogowhite.webp",
    "Racing Bulls": f"{_F1_LOGO_BASE}/racingbulls/2026racingbullslogowhite.webp",
    "Haas F1 Team": f"{_F1_LOGO_BASE}/haasf1team/2026haasf1teamlogowhite.webp",
    "Sauber": f"{_F1_LOGO_BASE}/audi/2026audilogowhite.webp",
    "Audi": f"{_F1_LOGO_BASE}/audi/2026audilogowhite.webp",
    "Cadillac": f"{_F1_LOGO_BASE}/cadillac/2026cadillaclogowhite.webp",
}


@st.cache_data(ttl=300)
def get_driver_team_map(season: int) -> dict[str, str]:
    """Get driver_name -> team_name mapping from latest results."""
    try:
        from sqlalchemy import text
        from src.database.connection import get_session

        with get_session() as session:
            rows = session.execute(text("""
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
            return mapping
    except Exception:
        return {}


@st.cache_data(ttl=300)
def get_standings_order(season: int) -> list[dict]:
    """Get drivers in championship standings order."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_standings_df

        with get_session() as session:
            df = get_standings_df(session, season, "driver")
            if df.empty:
                df = get_standings_df(session, season - 1, "driver")
            if df.empty:
                return []
            latest = df[df["round"] == df["round"].max()]
            return latest[["driver_name", "driver_code", "position", "points"]].to_dict("records")
    except Exception:
        return []


def driver_html(driver_name: str, team_name: str, position: int | None = None) -> str:
    """Render a driver row with team logo, color bar, and name."""
    color = TEAM_COLORS.get(team_name, "#888888")
    logo_url = TEAM_LOGOS.get(team_name, "")

    pos_str = f"<span style='color:#999;margin-right:8px;min-width:24px;display:inline-block;text-align:right;'>{position}.</span>" if position else ""
    logo_img = (
        f"<img src='{logo_url}' width='24' height='24' "
        f"style='vertical-align:middle;margin-right:6px;' "
        f"onerror=\"this.style.display='none'\" />"
        if logo_url else ""
    )
    color_bar = (
        f"<span style='display:inline-block;width:4px;height:22px;"
        f"border-radius:2px;background:{color};margin-right:8px;"
        f"vertical-align:middle;'></span>"
    )
    name_span = f"<span style='vertical-align:middle;'>{driver_name}</span>"
    team_span = (
        f"<span style='color:#999;font-size:0.85em;margin-left:8px;"
        f"vertical-align:middle;'>{team_name}</span>"
    )

    return (
        f"<div style='display:flex;align-items:center;padding:4px 0;'>"
        f"{pos_str}{color_bar}{logo_img}{name_span}{team_span}</div>"
    )


# --- Auto-refresh logic ---
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = time.time()
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "live_stop_event" not in st.session_state:
    st.session_state.live_stop_event = threading.Event()

elapsed = time.time() - st.session_state.last_refresh
if elapsed > LIVE_DASHBOARD_REFRESH and st.session_state.live_running:
    st.session_state.last_refresh = time.time()
    st.rerun()


@st.cache_data(ttl=15)
def get_available_laps(season: int, round_num: int) -> list[int]:
    """Get list of laps that have prediction snapshots."""
    pattern = f"matrix_{season}_r{round_num}_lap*.parquet"
    files = sorted(LIVE_PREDICTIONS_DIR.glob(pattern))
    laps = []
    for f in files:
        try:
            lap = int(f.stem.split("_lap")[-1])
            laps.append(lap)
        except ValueError:
            continue
    return sorted(laps)


@st.cache_data(ttl=15)
def load_lap_matrix(season: int, round_num: int, lap: int) -> pd.DataFrame | None:
    """Load prediction matrix for a specific lap."""
    path = LIVE_PREDICTIONS_DIR / f"matrix_{season}_r{round_num}_lap{lap}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=15)
def load_evolution(season: int, round_num: int) -> pd.DataFrame | None:
    """Load prediction evolution data."""
    path = LIVE_PREDICTIONS_DIR / f"evolution_{season}_r{round_num}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


@st.cache_data(ttl=15)
def load_db_snapshots(race_id: int, lap: int | None = None) -> pd.DataFrame | None:
    """Load live snapshots from database."""
    try:
        from src.database.connection import get_session
        from src.database.queries import get_live_snapshots

        with get_session() as session:
            return get_live_snapshots(session, race_id, lap_number=lap)
    except Exception:
        return None


# --- Sidebar ---
st.sidebar.header("Live Tracking")
season = st.sidebar.number_input("Season", value=2026, min_value=2020, max_value=2030)
round_num = st.sidebar.number_input("Round", value=1, min_value=1, max_value=24)


def start_live_tracking() -> None:
    st.session_state.live_stop_event.clear()
    st.session_state.live_running = True

    def _run() -> None:
        from src.models.live_predictor import run_live_prediction_loop
        try:
            run_live_prediction_loop(season=season, round_num=round_num)
        except Exception as e:
            from src.utils.logger import logger
            logger.error(f"Live tracking failed: {e}")
        finally:
            st.session_state.live_running = False

    threading.Thread(target=_run, daemon=True).start()


def stop_live_tracking() -> None:
    st.session_state.live_stop_event.set()
    st.session_state.live_running = False


if st.session_state.live_running:
    st.sidebar.button("Stop Live Tracking", on_click=stop_live_tracking, type="primary")
    st.sidebar.success("Live tracking is running")
else:
    st.sidebar.button("Start Live Tracking", on_click=start_live_tracking, type="primary")
    st.sidebar.info("Live tracking is stopped")

st.sidebar.caption(f"Auto-refresh every {LIVE_DASHBOARD_REFRESH}s when tracking is active")

# --- Sidebar: Championship standings ---
_sb_standings = get_standings_order(season)
if not _sb_standings:
    _sb_standings = get_standings_order(season - 1)
_sb_teams = get_driver_team_map(season)
if not _sb_teams:
    _sb_teams = get_driver_team_map(season - 1)

if _sb_standings:
    st.sidebar.divider()
    st.sidebar.header("Championship Standings")
    _sb_html = ""
    for s in _sb_standings:
        name = s["driver_name"]
        team = _sb_teams.get(name, "")
        color = TEAM_COLORS.get(team, "#888888")
        logo_url = TEAM_LOGOS.get(team, "")
        logo = (
            f"<img src='{logo_url}' width='16' height='16' "
            f"style='vertical-align:middle;margin-right:4px;' "
            f"onerror=\"this.style.display='none'\" />"
            if logo_url else ""
        )
        bar = (
            f"<span style='display:inline-block;width:3px;height:16px;"
            f"border-radius:1px;background:{color};margin-right:6px;"
            f"vertical-align:middle;'></span>"
        )
        _sb_html += (
            f"<div style='display:flex;align-items:center;padding:2px 0;"
            f"font-size:0.85em;'>"
            f"<span style='color:#999;min-width:20px;text-align:right;"
            f"margin-right:6px;'>{s['position']}</span>"
            f"{bar}{logo}"
            f"<span style='flex:1;'>{name}</span>"
            f"<span style='color:#999;margin-left:4px;'>{s['points']:.0f}pts</span>"
            f"</div>"
        )
    st.sidebar.markdown(_sb_html, unsafe_allow_html=True)

# --- Check for data ---
available_laps = get_available_laps(season, round_num)

if not available_laps:
    st.info(
        "No live predictions available yet. Start live tracking from the sidebar, "
        "or run from the CLI:"
    )
    st.code(f"make live SEASON={season} ROUND={round_num}", language="bash")
    st.stop()

# --- Lap selector ---
st.sidebar.divider()
st.sidebar.header("Lap Selector")

selected_lap = st.sidebar.slider(
    "Select Lap",
    min_value=available_laps[0],
    max_value=available_laps[-1],
    value=available_laps[-1],
    step=1,
)

# Snap to nearest available lap if slider is between snapshots
if selected_lap not in available_laps:
    selected_lap = min(available_laps, key=lambda x: abs(x - selected_lap))

st.sidebar.caption(
    f"Showing lap {selected_lap} of {available_laps[-1]} "
    f"({len(available_laps)} snapshots available)"
)

# Play through laps button
if "playing" not in st.session_state:
    st.session_state.playing = False
if "play_lap_idx" not in st.session_state:
    st.session_state.play_lap_idx = 0

# Load matrix for selected lap
matrix = load_lap_matrix(season, round_num, selected_lap)

if matrix is None:
    st.warning(f"No data for lap {selected_lap}")
    st.stop()

# --- Row 1: Status bar ---
st.divider()

# Get race_id for DB queries
race_id = None
try:
    from src.database.connection import get_session
    from src.database.queries import get_race
    with get_session() as session:
        race = get_race(session, season, round_num)
        race_id = race.id if race else None
except Exception:
    pass

# Load DB snapshot for this lap (has current positions, tire info, SC status)
db_snap = load_db_snapshots(race_id, selected_lap) if race_id else None

sc_status = ""
if db_snap is not None and not db_snap.empty and "safety_car_active" in db_snap.columns:
    sc_val = db_snap["safety_car_active"].iloc[0]
    if sc_val == 1:
        sc_status = "SAFETY CAR"
    elif sc_val == 2:
        sc_status = "VSC"

status_cols = st.columns(5)
status_cols[0].metric("Lap", f"{selected_lap} / {available_laps[-1]}")
status_cols[1].metric("Active Drivers", f"{len(matrix)}")

if sc_status:
    status_cols[2].metric("Race Status", sc_status)
else:
    status_cols[2].metric("Race Status", "GREEN" if selected_lap < available_laps[-1] else "FINISHED")

if "win_prob" in matrix.columns:
    leader = matrix.index[0]
    leader_prob = matrix.iloc[0]["win_prob"]
    status_cols[3].metric("Predicted Winner", leader)
    status_cols[4].metric("Win Probability", f"{leader_prob:.1%}")

# --- Load team data ---
driver_team = get_driver_team_map(season)
if not driver_team:
    driver_team = get_driver_team_map(season - 1)
standings = get_standings_order(season)
if not standings:
    standings = get_standings_order(season - 1)

# Build standings lookup: driver_name -> position
standings_pos = {s["driver_name"]: s["position"] for s in standings}
standings_pts = {s["driver_name"]: s["points"] for s in standings}

# --- Row 2: Podium cards with team logos ---
st.divider()
st.subheader("Predicted Podium")

if len(matrix) >= 3 and "win_prob" in matrix.columns:
    pod_cols = st.columns(3)
    medals = ["🥇", "🥈", "🥉"]
    labels = ["P1", "P2", "P3"]

    for i, col in enumerate(pod_cols):
        driver = matrix.index[i]
        row = matrix.iloc[i]
        team = driver_team.get(driver, "")
        logo_url = TEAM_LOGOS.get(team, "")
        color = TEAM_COLORS.get(team, "#888888")
        with col:
            st.markdown(f"### {medals[i]} {labels[i]}")
            if logo_url:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:10px;'>"
                    f"<img src='{logo_url}' width='32' height='32' "
                    f"onerror=\"this.style.display='none'\" />"
                    f"<span style='font-size:1.6em;font-weight:bold;'>{driver}</span>"
                    f"</div>"
                    f"<span style='color:{color};font-size:0.9em;'>{team}</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f"## {driver}")
            prob_col1, prob_col2 = st.columns(2)
            prob_col1.metric("Win", f"{row.get('win_prob', 0):.1%}")
            prob_col2.metric("Podium", f"{row.get('podium_prob', 0):.1%}")

# --- Row 3: Driver standings + predictions with team logos ---
st.divider()
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader(f"Race Order — Lap {selected_lap}")

    # Build driver list sorted by current race position (from DB or matrix)
    race_drivers: list[dict] = []
    if db_snap is not None and not db_snap.empty:
        snap_sorted = db_snap.sort_values("current_position", na_position="last")
        for _, r in snap_sorted.iterrows():
            name = r["driver_name"]
            race_drivers.append({
                "name": name,
                "pos": int(r["current_position"]) if pd.notna(r["current_position"]) else None,
                "pred": r.get("predicted_position"),
                "win": r.get("win_prob", 0),
            })
    else:
        for name in matrix.index:
            row = matrix.loc[name]
            race_drivers.append({
                "name": name,
                "pos": None,
                "pred": row.get("expected_position"),
                "win": row.get("win_prob", 0),
            })

    # Render driver list with team logos
    html_rows = []
    for d in race_drivers:
        name = d["name"]
        team = driver_team.get(name, "")
        color = TEAM_COLORS.get(team, "#888888")
        logo_url = TEAM_LOGOS.get(team, "")
        pos = d["pos"]
        pred = d["pred"]
        win = d["win"]

        pos_label = f"P{pos}" if pos else ""
        pred_label = f"{pred:.1f}" if pred else "-"
        win_label = f"{win:.1%}" if win else "-"

        logo_img = (
            f"<img src='{logo_url}' width='20' height='20' "
            f"style='vertical-align:middle;' "
            f"onerror=\"this.style.display='none'\" />"
            if logo_url else ""
        )

        html_rows.append(
            f"<tr>"
            f"<td style='padding:6px 8px;font-weight:bold;color:#ccc;width:40px;'>{pos_label}</td>"
            f"<td style='padding:6px 4px;width:6px;'>"
            f"<span style='display:inline-block;width:4px;height:20px;"
            f"border-radius:2px;background:{color};'></span></td>"
            f"<td style='padding:6px 4px;width:28px;'>{logo_img}</td>"
            f"<td style='padding:6px 4px;'>{name}</td>"
            f"<td style='padding:6px 8px;color:#999;font-size:0.85em;'>{team}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{pred_label}</td>"
            f"<td style='padding:6px 8px;text-align:right;'>{win_label}</td>"
            f"</tr>"
        )

    table_html = (
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='padding:6px 8px;text-align:left;color:#888;font-size:0.8em;'>POS</th>"
        "<th></th><th></th>"
        "<th style='padding:6px 4px;text-align:left;color:#888;font-size:0.8em;'>DRIVER</th>"
        "<th style='padding:6px 8px;text-align:left;color:#888;font-size:0.8em;'>TEAM</th>"
        "<th style='padding:6px 8px;text-align:right;color:#888;font-size:0.8em;'>PRED</th>"
        "<th style='padding:6px 8px;text-align:right;color:#888;font-size:0.8em;'>WIN %</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(html_rows) + "</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

with chart_col2:
    st.subheader("Win Probabilities")
    if "win_prob" in matrix.columns:
        win_df = pd.DataFrame({
            "Driver": matrix.index,
            "Win Probability": matrix["win_prob"].values,
        }).sort_values("Win Probability", ascending=True).tail(10)

        fig = px.bar(
            win_df,
            x="Win Probability",
            y="Driver",
            orientation="h",
            color="Win Probability",
            color_continuous_scale="RdYlGn_r",
        )
        fig.update_layout(height=400, xaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# --- Row 4: Probability heatmap ---
st.divider()
st.subheader(f"Position Probability Matrix — Lap {selected_lap}")

pos_cols = [f"P{i}" for i in range(1, 23) if f"P{i}" in matrix.columns]
if pos_cols:
    heatmap_data = matrix[pos_cols]

    fig = px.imshow(
        heatmap_data.values,
        x=pos_cols,
        y=heatmap_data.index.tolist(),
        color_continuous_scale="YlOrRd",
        aspect="auto",
        labels={"color": "Probability"},
    )
    fig.update_layout(height=700, xaxis_title="Position", yaxis_title="Driver")
    st.plotly_chart(fig, use_container_width=True)

# --- Row 5: Full predictions table ---
st.divider()
st.subheader(f"Full Predictions — Lap {selected_lap}")
display_cols = ["expected_position", "win_prob", "podium_prob", "top5_prob"]
pos_detail_cols = [f"P{i}" for i in range(1, 23) if f"P{i}" in matrix.columns]
available = [c for c in display_cols if c in matrix.columns]
if available:
    full_display = matrix[available + pos_detail_cols].copy()
    st.dataframe(
        full_display.style.format({
            c: "{:.1%}" if "prob" in c or c.startswith("P") else "{:.1f}"
            for c in full_display.columns
        }),
        use_container_width=True,
        height=600,
    )

# --- Row 6: Lap comparison ---
if len(available_laps) >= 2:
    st.divider()
    st.subheader("Lap Comparison")

    compare_options = [l for l in available_laps if l != selected_lap]
    compare_lap = st.selectbox(
        "Compare with lap",
        compare_options,
        index=0,
        key=f"compare_lap_{selected_lap}",
    )

    st.caption(f"Showing changes from **Lap {compare_lap}** to **Lap {selected_lap}**")

    compare_matrix = load_lap_matrix(season, round_num, compare_lap)
    if compare_matrix is not None and "win_prob" in matrix.columns:
        current = matrix[["win_prob", "expected_position"]].copy()
        current.columns = ["win_prob_now", "exp_pos_now"]
        previous = compare_matrix[["win_prob", "expected_position"]].copy()
        previous.columns = ["win_prob_prev", "exp_pos_prev"]

        merged = current.join(previous, how="outer").fillna(0)
        merged["win_prob_delta"] = merged["win_prob_now"] - merged["win_prob_prev"]
        merged["pos_delta"] = merged["exp_pos_prev"] - merged["exp_pos_now"]
        merged = merged.sort_values("win_prob_now", ascending=False).head(15)

        # Delta bar chart
        delta_col1, delta_col2 = st.columns(2)

        with delta_col1:
            delta_df = pd.DataFrame({
                "Driver": merged.index,
                "Win % Change": merged["win_prob_delta"].values,
            }).sort_values("Win % Change")

            colors = ["#ef4444" if v < 0 else "#22c55e" for v in delta_df["Win % Change"]]
            fig = go.Figure(go.Bar(
                x=delta_df["Win % Change"],
                y=delta_df["Driver"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1%}" for v in delta_df["Win % Change"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Win Probability Change (Lap {compare_lap} → {selected_lap})",
                height=450,
                xaxis_tickformat=".0%",
                xaxis_title="Change",
                yaxis_title="",
                margin=dict(l=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        with delta_col2:
            pos_df = pd.DataFrame({
                "Driver": merged.index,
                "Position Change": merged["pos_delta"].values,
            }).sort_values("Position Change", ascending=False)

            colors = ["#22c55e" if v > 0 else "#ef4444" if v < 0 else "#6b7280"
                       for v in pos_df["Position Change"]]
            fig = go.Figure(go.Bar(
                x=pos_df["Position Change"],
                y=pos_df["Driver"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}" for v in pos_df["Position Change"]],
                textposition="outside",
            ))
            fig.update_layout(
                title=f"Expected Position Change (Lap {compare_lap} → {selected_lap})",
                height=450,
                xaxis_title="Positions gained",
                yaxis_title="",
                margin=dict(l=0),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Comparison table
        display = merged[["win_prob_prev", "win_prob_now", "win_prob_delta",
                          "exp_pos_prev", "exp_pos_now", "pos_delta"]].copy()
        display.columns = [
            f"Win % (Lap {compare_lap})",
            f"Win % (Lap {selected_lap})",
            "Win % Change",
            f"Exp Pos (Lap {compare_lap})",
            f"Exp Pos (Lap {selected_lap})",
            "Pos Change",
        ]

        def _color_delta(val: float) -> str:
            if val > 0.001:
                return "color: #22c55e"
            elif val < -0.001:
                return "color: #ef4444"
            return ""

        st.dataframe(
            display.style
            .format({
                display.columns[0]: "{:.1%}",
                display.columns[1]: "{:.1%}",
                display.columns[2]: "{:+.1%}",
                display.columns[3]: "{:.1f}",
                display.columns[4]: "{:.1f}",
                display.columns[5]: "{:+.1f}",
            })
            .map(_color_delta, subset=[display.columns[2], display.columns[5]]),
            use_container_width=True,
        )

        # Side-by-side position probability heatmaps
        cmp_pos_cols = [f"P{i}" for i in range(1, 23)
                        if f"P{i}" in matrix.columns and f"P{i}" in compare_matrix.columns]
        if cmp_pos_cols:
            # Use same driver order (sorted by current expected position)
            top_drivers_cmp = merged.index.tolist()

            hm_col1, hm_col2 = st.columns(2)
            with hm_col1:
                cmp_data = compare_matrix.loc[
                    compare_matrix.index.isin(top_drivers_cmp), cmp_pos_cols
                ].reindex(top_drivers_cmp)
                fig_hm1 = px.imshow(
                    cmp_data.values,
                    x=cmp_pos_cols,
                    y=cmp_data.index.tolist(),
                    color_continuous_scale="YlOrRd",
                    aspect="auto",
                    zmin=0, zmax=1,
                    labels={"color": "Probability"},
                )
                fig_hm1.update_layout(
                    title=f"Position Probabilities — Lap {compare_lap}",
                    height=450,
                    xaxis_title="Position",
                    yaxis_title="",
                )
                st.plotly_chart(fig_hm1, use_container_width=True)

            with hm_col2:
                cur_data = matrix.loc[
                    matrix.index.isin(top_drivers_cmp), cmp_pos_cols
                ].reindex(top_drivers_cmp)
                fig_hm2 = px.imshow(
                    cur_data.values,
                    x=cmp_pos_cols,
                    y=cur_data.index.tolist(),
                    color_continuous_scale="YlOrRd",
                    aspect="auto",
                    zmin=0, zmax=1,
                    labels={"color": "Probability"},
                )
                fig_hm2.update_layout(
                    title=f"Position Probabilities — Lap {selected_lap}",
                    height=450,
                    xaxis_title="Position",
                    yaxis_title="",
                )
                st.plotly_chart(fig_hm2, use_container_width=True)

# --- Row 7: Prediction evolution (timeline across all laps) ---
st.divider()
st.subheader("Prediction Evolution")

evolution = load_evolution(season, round_num)
if evolution is not None and not evolution.empty:
    latest_lap = evolution["lap"].max()
    top_drivers = (
        evolution[evolution["lap"] == latest_lap]
        .nlargest(5, "win_prob")["driver_name"]
        .tolist()
    )

    evo_filtered = evolution[evolution["driver_name"].isin(top_drivers)]

    fig = px.line(
        evo_filtered,
        x="lap",
        y="win_prob",
        color="driver_name",
        labels={"lap": "Lap", "win_prob": "Win Probability", "driver_name": "Driver"},
    )
    fig.add_vline(
        x=selected_lap,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
        annotation_text=f"Lap {selected_lap}",
        annotation_position="top",
    )
    fig.update_layout(height=400, yaxis_tickformat=".0%", legend_title="Driver")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Expected Finishing Position")
    fig2 = px.line(
        evo_filtered,
        x="lap",
        y="expected_position",
        color="driver_name",
        labels={
            "lap": "Lap",
            "expected_position": "Expected Position",
            "driver_name": "Driver",
        },
    )
    fig2.add_vline(
        x=selected_lap,
        line_dash="dash",
        line_color="white",
        opacity=0.7,
    )
    fig2.update_layout(
        height=400,
        yaxis_autorange="reversed",
        legend_title="Driver",
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.caption("Evolution chart will appear after multiple laps of data are collected.")
