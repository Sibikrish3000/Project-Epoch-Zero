import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update
import dash_bootstrap_components as dbc
from dash_iconify import DashIconify
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import json
import math

# --- PROJECT IMPORTS ---
try:
    from src.deployer import OrbitDeployer
    deployer = OrbitDeployer()
    MODEL_LOADED = True
    print("[+] PINN Model loaded successfully.")
except Exception as e:
    print(f"[!] Warning: Model not loaded. {e}")
    MODEL_LOADED = False
    class MockDeployer:
        def predict(self, *args):
            r = np.random.randn(3) * 100 + np.array([6800, 0, 0])
            return (r, np.zeros(3)), (r + np.random.randn(3)*50, np.zeros(3)), (r + np.random.randn(3)*30, np.random.randn(3)*0.1)
        def get_trajectory(self, l1, l2, target, steps=100):
            t = np.linspace(0, 2*np.pi, steps)
            a = 6800
            return np.column_stack([a*np.cos(t), a*np.sin(t), np.zeros(steps) + np.random.randn(steps)*100])
    deployer = MockDeployer()

# --- APP CONFIG ---
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Rajdhani:wght@300;400;500;600;700&family=Share+Tech+Mono&display=swap"
    ],
    suppress_callback_exceptions=True
)
app.title = "EPOCH ZERO | Orbital Defense System"

# --- SATELLITE DATABASE ---
SAT_DATABASE = {
    "50803": {"name": "STARLINK-3321", "norad": "50803", "type": "PAYLOAD", "country": "US", "launch": "2022-01-06",
              "tle1": "1 50803U 22001A   22011.83334491  .00918374  26886-3 -20449-2 0  9990",
              "tle2": "2 50803  53.2176 175.5863 0053823 179.7175 211.9048 15.94459142  2073"},
    "99999": {"name": "COSMOS-1408 DEB", "norad": "99999", "type": "DEBRIS", "country": "CIS", "launch": "N/A",
              "tle1": "1 99999U 22001A   22011.83334491  .00918374  26886-3 -20449-2 0  9990",
              "tle2": "2 99999  53.2100 175.6000 0053823 179.7175 211.9048 15.94459142 42073"},
    "25544": {"name": "ISS (ZARYA)", "norad": "25544", "type": "PAYLOAD", "country": "ISS", "launch": "1998-11-20",
              "tle1": "1 25544U 98067A   22011.50000000  .00006730  00000-0  12500-3 0  9990",
              "tle2": "2 25544  51.6435 200.1234 0006828 300.1234  59.8765 15.48919755370000"},
    "48274": {"name": "STARLINK-2305", "norad": "48274", "type": "PAYLOAD", "country": "US", "launch": "2021-03-24",
              "tle1": "1 48274U 21024A   22011.50000000  .00012000  00000-0  80000-4 0  9990",
              "tle2": "2 48274  53.0540 170.2345 0001234 85.6789 274.4321 15.06400000 40000"},
}

SAT_OPTIONS = [{"label": f"{v['name']} [{v['norad']}]", "value": k} for k, v in SAT_DATABASE.items()]

# --- PRECOMPUTED ASSETS ---
np.random.seed(42)
NUM_STARS = 800
STAR_R = 35000
s_theta = np.random.uniform(0, 2*np.pi, NUM_STARS)
s_phi = np.random.uniform(0, np.pi, NUM_STARS)
s_sizes = np.random.choice([1, 1.5, 2, 2.5, 3], NUM_STARS, p=[0.4, 0.25, 0.2, 0.1, 0.05])
stars_x = STAR_R * np.sin(s_phi) * np.cos(s_theta)
stars_y = STAR_R * np.sin(s_phi) * np.sin(s_theta)
stars_z = STAR_R * np.cos(s_phi)

STAR_TRACE = go.Scatter3d(
    x=stars_x, y=stars_y, z=stars_z,
    mode='markers', marker=dict(size=s_sizes, color='white', opacity=0.7),
    hoverinfo='skip', name='Stars', showlegend=False,
)

# Earth sphere
N_EARTH = 50
u_e = np.linspace(0, 2*np.pi, N_EARTH)
v_e = np.linspace(0, np.pi, N_EARTH)
R_EARTH = 6371
EX = R_EARTH * np.outer(np.cos(u_e), np.sin(v_e))
EY = R_EARTH * np.outer(np.sin(u_e), np.sin(v_e))
EZ = R_EARTH * np.outer(np.ones(np.size(u_e)), np.cos(v_e))

earth_colors = [[0, '#0a1628'], [0.2, '#0d2137'], [0.4, '#0f3460'], [0.6, '#16537e'], [0.8, '#1a759f'], [1.0, '#34a0a4']]

EARTH_TRACE = go.Surface(
    x=EX, y=EY, z=EZ,
    colorscale=earth_colors, showscale=False,
    lighting=dict(ambient=0.3, diffuse=0.6, specular=0.2, roughness=0.8, fresnel=0.2),
    lightposition=dict(x=10000, y=5000, z=8000),
    hoverinfo='skip', opacity=0.95, name='Earth',
)

ATM_SCALE = 1.015
ATMO_TRACE = go.Surface(
    x=EX*ATM_SCALE, y=EY*ATM_SCALE, z=EZ*ATM_SCALE,
    colorscale=[[0, 'rgba(0,170,255,0.05)'], [1, 'rgba(0,100,255,0.12)']],
    showscale=False, opacity=0.15, hoverinfo='skip', name='Atmosphere',
    lighting=dict(ambient=0.8, diffuse=0.2, specular=0.0),
)

# Grid lines on earth
def make_grid_lines():
    traces = []
    for lat in range(-60, 90, 30):
        theta = np.linspace(0, 2*np.pi, 80)
        r = R_EARTH * np.cos(np.radians(lat)) * 1.002
        z = R_EARTH * np.sin(np.radians(lat)) * 1.002
        traces.append(go.Scatter3d(
            x=r*np.cos(theta), y=r*np.sin(theta), z=np.full(80, z),
            mode='lines', line=dict(color='rgba(0,200,255,0.15)', width=1),
            hoverinfo='skip', showlegend=False
        ))
    for lon in range(0, 360, 30):
        phi = np.linspace(0, np.pi, 80)
        traces.append(go.Scatter3d(
            x=R_EARTH*1.002*np.sin(phi)*np.cos(np.radians(lon)),
            y=R_EARTH*1.002*np.sin(phi)*np.sin(np.radians(lon)),
            z=R_EARTH*1.002*np.cos(phi),
            mode='lines', line=dict(color='rgba(0,200,255,0.1)', width=1),
            hoverinfo='skip', showlegend=False
        ))
    return traces

GRID_TRACES = make_grid_lines()


# =============================================================================
# COMPONENT BUILDERS
# =============================================================================

def make_icon(icon, size=18, color="#00d4ff"):
    return DashIconify(icon=icon, width=size, color=color, style={"marginRight": "8px"})

def make_stat_card(icon, label, value_id, unit="", color="#00d4ff"):
    return html.Div([
        html.Div([
            make_icon(icon, 16, color),
            html.Span(label, className="stat-label"),
        ], className="stat-header"),
        html.Div([
            html.Span("---", id=value_id, className="stat-value", style={"color": color}),
            html.Span(f" {unit}", className="stat-unit") if unit else html.Span(),
        ], className="stat-body"),
    ], className="stat-card")

def make_sidebar_section(title, icon, children):
    return html.Div([
        html.Div([
            make_icon(icon, 16, "#00d4ff"),
            html.Span(title, className="section-title"),
        ], className="section-header"),
        html.Div(children, className="section-body"),
    ], className="sidebar-section")


# =============================================================================
# HEADER BAR
# =============================================================================
header = html.Div([
    html.Div([
        html.Div([
            DashIconify(icon="mdi:satellite-variant", width=28, color="#00d4ff"),
            html.Span("EPOCH", style={"color": "#00d4ff", "fontWeight": "800", "fontSize": "20px", "fontFamily": "'Orbitron', sans-serif"}),
            html.Span("ZERO", style={"color": "#ff6b35", "fontWeight": "800", "fontSize": "20px", "fontFamily": "'Orbitron', sans-serif", "marginLeft": "4px"}),
        ], className="header-brand"),
        html.Div([
            html.Div([
                html.Div(className="status-dot status-dot-green"),
                html.Span("SYS NOMINAL", className="status-text"),
            ], className="status-badge"),
            html.Div([
                html.Div(className="status-dot status-dot-blue"),
                html.Span("PINN ONLINE" if MODEL_LOADED else "PINN OFFLINE", className="status-text",
                          style={"color": "#00d4ff"} if MODEL_LOADED else {"color": "#ff4444"}),
            ], className="status-badge"),
            html.Div([
                DashIconify(icon="mdi:clock-outline", width=14, color="#888"),
                html.Span(id="header-clock", className="status-text"),
            ], className="status-badge"),
        ], className="header-status"),
        html.Div([
            html.Span("ORBITAL DEFENSE SYSTEM", className="header-subtitle"),
            html.Span("v2.0 | MISSION CONTROL", className="header-version"),
        ], className="header-right"),
    ], className="header-inner"),
], className="app-header")


# =============================================================================
# LEFT SIDEBAR - MISSION CONTROL
# =============================================================================
left_sidebar = html.Div([
    make_sidebar_section("OBJECT SELECTION", "mdi:target", [
        html.Label("PRIMARY OBJECT", className="input-label"),
        dcc.Dropdown(
            id="sat-a-select", options=SAT_OPTIONS, value="50803",
            className="dark-dropdown", clearable=False,
            style={"marginBottom": "10px"}
        ),
        html.Label("SECONDARY OBJECT", className="input-label"),
        dcc.Dropdown(
            id="sat-b-select", options=SAT_OPTIONS, value="99999",
            className="dark-dropdown", clearable=False,
        ),
    ]),

    make_sidebar_section("PREDICTION WINDOW", "mdi:clock-fast", [
        html.Label("TARGET DATE", className="input-label"),
        dcc.DatePickerSingle(
            id="target-date", date="2022-01-12",
            display_format="YYYY-MM-DD",
            className="dark-datepicker",
            style={"width": "100%"},
        ),
        html.Label("TARGET TIME (UTC)", className="input-label", style={"marginTop": "8px"}),
        dcc.Input(
            id="target-time", value="20:00:00", type="text",
            className="dark-input", placeholder="HH:MM:SS",
        ),
        html.Label("PROPAGATION WINDOW", className="input-label", style={"marginTop": "8px"}),
        dcc.Slider(
            id="prop-window", min=30, max=180, step=10, value=100,
            marks={30: "30m", 60: "1h", 120: "2h", 180: "3h"},
            className="dark-slider",
            tooltip={"placement": "bottom", "always_visible": False},
        ),
    ]),

    make_sidebar_section("ANIMATION", "mdi:motion-play-outline", [
        html.Label("PLAYBACK SPEED", className="input-label"),
        dcc.Slider(
            id="anim-speed", min=50, max=500, step=50, value=200,
            marks={50: "0.5x", 100: "1x", 200: "2x", 500: "5x"},
            className="dark-slider",
        ),
        html.Div([
            dbc.Button([make_icon("mdi:play", 16, "#000"), "RUN"], id="btn-run", className="btn-action btn-primary-action", n_clicks=0),
            dbc.Button([make_icon("mdi:pause", 16, "#fff"), ""], id="btn-pause", className="btn-action btn-secondary-action", n_clicks=0),
            dbc.Button([make_icon("mdi:skip-forward", 16, "#fff"), ""], id="btn-step", className="btn-action btn-secondary-action", n_clicks=0),
        ], className="btn-group-controls"),
    ]),

    make_sidebar_section("VISUAL LAYERS", "mdi:layers-triple-outline", [
        dbc.Checklist(
            id="visual-toggles",
            options=[
                {"label": " Orbit Trajectories", "value": "rails"},
                {"label": " Kill Zone / LOS", "value": "killzone"},
                {"label": " Atmosphere", "value": "atmo"},
                {"label": " Star Field", "value": "stars"},
                {"label": " Grid Lines", "value": "grid"},
                {"label": " Velocity Vectors", "value": "vectors"},
            ],
            value=["rails", "killzone", "atmo", "stars"],
            className="visual-checklist", switch=True,
        ),
    ]),

    html.Div([
        dbc.Button([
            DashIconify(icon="mdi:radar", width=20, color="#000"),
            html.Span(" INITIATE CONJUNCTION SCAN", style={"marginLeft": "8px"}),
        ], id="btn-scan", className="btn-scan", n_clicks=0),
    ], className="scan-btn-wrap"),

], className="left-sidebar", id="left-sidebar")


# =============================================================================
# RIGHT SIDEBAR - TELEMETRY
# =============================================================================
right_sidebar = html.Div([
    make_sidebar_section("CONJUNCTION ASSESSMENT", "mdi:shield-alert-outline", [
        html.Div([
            html.Div([
                html.Span("MISS DISTANCE", className="metric-label"),
                html.Span("---", id="metric-miss", className="metric-value metric-primary"),
                html.Span(" km", className="metric-unit"),
            ], className="metric-row"),
            html.Div([
                html.Span("COLLISION PROBABILITY", className="metric-label"),
                html.Span("---", id="metric-prob", className="metric-value metric-warn"),
            ], className="metric-row"),
            html.Div([
                html.Span("TIME TO TCA", className="metric-label"),
                html.Span("---", id="metric-tca", className="metric-value"),
            ], className="metric-row"),
        ], className="metrics-block"),
        html.Div([
            html.Div("THREAT LEVEL", className="threat-label"),
            html.Div("STANDBY", id="threat-level", className="threat-badge threat-standby"),
        ], className="threat-block"),
        dbc.Progress(value=0, id="threat-bar", className="threat-progress", style={"height": "4px"}),
    ]),

    make_sidebar_section("PRIMARY TELEMETRY", "mdi:satellite-uplink", [
        html.Div(id="sat-a-info", children=[
            make_stat_card("mdi:crosshairs-gps", "POSITION (PINN)", "stat-pos-a", "km", "#00d4ff"),
            make_stat_card("mdi:speedometer", "VELOCITY", "stat-vel-a", "km/s", "#00d4ff"),
            make_stat_card("mdi:arrow-expand-vertical", "ALTITUDE", "stat-alt-a", "km", "#00d4ff"),
        ]),
    ]),

    make_sidebar_section("SECONDARY TELEMETRY", "mdi:alert-octagon-outline", [
        html.Div(id="sat-b-info", children=[
            make_stat_card("mdi:crosshairs-gps", "POSITION (PINN)", "stat-pos-b", "km", "#ff6b35"),
            make_stat_card("mdi:speedometer", "VELOCITY", "stat-vel-b", "km/s", "#ff6b35"),
            make_stat_card("mdi:arrow-expand-vertical", "ALTITUDE", "stat-alt-b", "km", "#ff6b35"),
        ]),
    ]),

    make_sidebar_section("PINN CORRECTION", "mdi:brain", [
        html.Div([
            html.Div([
                html.Span("SGP4 → PINN Δr (A)", className="metric-label"),
                html.Span("---", id="pinn-dr-a", className="metric-value", style={"color": "#00d4ff"}),
                html.Span(" km", className="metric-unit"),
            ], className="metric-row"),
            html.Div([
                html.Span("SGP4 → PINN Δr (B)", className="metric-label"),
                html.Span("---", id="pinn-dr-b", className="metric-value", style={"color": "#ff6b35"}),
                html.Span(" km", className="metric-unit"),
            ], className="metric-row"),
            html.Div([
                html.Span("MODEL CONFIDENCE", className="metric-label"),
                html.Span("---", id="pinn-confidence", className="metric-value", style={"color": "#a855f7"}),
            ], className="metric-row"),
        ], className="metrics-block"),
    ]),

    make_sidebar_section("EVENT LOG", "mdi:format-list-bulleted", [
        html.Div(id="event-log", className="event-log", children=[
            html.Div([
                html.Span("SYS", className="log-tag log-tag-info"),
                html.Span(" System initialized. Awaiting scan.", className="log-msg"),
            ], className="log-entry"),
        ]),
    ]),

], className="right-sidebar", id="right-sidebar")


# =============================================================================
# CENTER - 3D VIEWPORT
# =============================================================================
center_viewport = html.Div([
    html.Div([
        html.Div([
            html.Span("3D ORBITAL VIEWPORT", className="viewport-title"),
            html.Span(" | REAL-TIME PROPAGATION", className="viewport-subtitle"),
        ]),
        html.Div([
            html.Span("FRAME: ", style={"color": "#666", "fontSize": "11px"}),
            html.Span("0/0", id="frame-counter", style={"color": "#00d4ff", "fontSize": "11px", "fontFamily": "'Share Tech Mono', monospace"}),
        ]),
    ], className="viewport-header"),

    dcc.Graph(
        id="globe-graph",
        style={"height": "100%", "width": "100%"},
        config={'displayModeBar': False, 'scrollZoom': True},
    ),

    dcc.Interval(id="anim-interval", interval=200, n_intervals=0, disabled=True),
    dcc.Store(id="sim-store", data=None),
    dcc.Store(id="anim-frame", data=0),
    dcc.Store(id="anim-playing", data=False),
], className="center-viewport")


# =============================================================================
# BOTTOM BAR
# =============================================================================
bottom_bar = html.Div([
    html.Div([
        DashIconify(icon="mdi:information-outline", width=14, color="#555"),
        html.Span(" EPOCH ZERO v2.0 | ", style={"color": "#555"}),
        html.Span("PINN-Enhanced Orbital Prediction", style={"color": "#666"}),
    ], className="bottom-left"),
    html.Div([
        html.Span("MODEL: ", style={"color": "#555"}),
        html.Span("GatedPINN v3.1.1", style={"color": "#00d4ff" if MODEL_LOADED else "#ff4444"}),
        html.Span(" | ", style={"color": "#333"}),
        html.Span("ENGINE: ", style={"color": "#555"}),
        html.Span("SGP4 + J2/Drag", style={"color": "#888"}),
        html.Span(" | ", style={"color": "#333"}),
        html.Span("PROPAGATOR: ", style={"color": "#555"}),
        html.Span("ACTIVE", style={"color": "#00ff88"}),
    ], className="bottom-right"),
], className="bottom-bar")


# =============================================================================
# MAIN LAYOUT
# =============================================================================
app.layout = html.Div([
    header,
    html.Div([
        left_sidebar,
        center_viewport,
        right_sidebar,
    ], className="main-body"),
    bottom_bar,
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),
], className="app-root")


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(Output("header-clock", "children"), Input("clock-interval", "n_intervals"))
def update_clock(_):
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


@callback(
    [Output("sim-store", "data"),
     Output("anim-interval", "disabled"),
     Output("anim-frame", "data"),
     Output("anim-playing", "data"),
     Output("event-log", "children")],
    [Input("btn-scan", "n_clicks"), Input("btn-run", "n_clicks")],
    [State("sat-a-select", "value"), State("sat-b-select", "value"),
     State("target-date", "date"), State("target-time", "value"),
     State("prop-window", "value"), State("event-log", "children")],
    prevent_initial_call=True
)
def run_scan(scan_clicks, run_clicks, sat_a_id, sat_b_id, date_val, time_val, prop_window, current_log):
    if not scan_clicks and not run_clicks:
        return no_update, no_update, no_update, no_update, no_update

    target_str = f"{date_val} {time_val}"
    sat_a = SAT_DATABASE.get(sat_a_id, SAT_DATABASE["50803"])
    sat_b = SAT_DATABASE.get(sat_b_id, SAT_DATABASE["99999"])

    log_entries = list(current_log) if current_log else []
    log_entries.insert(0, html.Div([
        html.Span("CMD", className="log-tag log-tag-cmd"),
        html.Span(f" Scan initiated: {sat_a['name']} vs {sat_b['name']}", className="log-msg"),
    ], className="log-entry"))

    try:
        res_A = deployer.predict(sat_a["tle1"], sat_a["tle2"], target_str)
        res_B = deployer.predict(sat_b["tle1"], sat_b["tle2"], target_str)

        if res_A[0] is None or res_B[0] is None:
            raise ValueError("SGP4 propagation error")

        (r0_A, v0_A), (r_sgp4_A, v_sgp4_A), (r_pinn_A, v_pinn_A) = res_A
        (r0_B, v0_B), (r_sgp4_B, v_sgp4_B), (r_pinn_B, v_pinn_B) = res_B

        traj_A = deployer.get_trajectory(sat_a["tle1"], sat_a["tle2"], target_str, steps=200)
        traj_B = deployer.get_trajectory(sat_b["tle1"], sat_b["tle2"], target_str, steps=200)

        miss_dist = float(np.linalg.norm(r_pinn_A - r_pinn_B))
        miss_sgp4 = float(np.linalg.norm(np.array(r_sgp4_A) - np.array(r_sgp4_B)))
        dr_a = float(np.linalg.norm(r_pinn_A - np.array(r_sgp4_A)))
        dr_b = float(np.linalg.norm(r_pinn_B - np.array(r_sgp4_B)))

        sim_data = {
            "traj_a": traj_A.tolist(),
            "traj_b": traj_B.tolist(),
            "pos_a": r_pinn_A.tolist(),
            "pos_b": r_pinn_B.tolist(),
            "vel_a": v_pinn_A.tolist(),
            "vel_b": v_pinn_B.tolist(),
            "pos_sgp4_a": list(r_sgp4_A),
            "pos_sgp4_b": list(r_sgp4_B),
            "miss_dist": miss_dist,
            "miss_sgp4": miss_sgp4,
            "dr_a": dr_a,
            "dr_b": dr_b,
            "sat_a_name": sat_a["name"],
            "sat_b_name": sat_b["name"],
            "target": target_str,
            "total_frames": len(traj_A),
        }

        log_entries.insert(0, html.Div([
            html.Span("OK", className="log-tag log-tag-ok"),
            html.Span(f" Scan complete. Miss distance: {miss_dist:.2f} km", className="log-msg"),
        ], className="log-entry"))

        return sim_data, False, 0, True, log_entries[:20]

    except Exception as e:
        log_entries.insert(0, html.Div([
            html.Span("ERR", className="log-tag log-tag-err"),
            html.Span(f" {str(e)}", className="log-msg"),
        ], className="log-entry"))
        return None, True, 0, False, log_entries[:20]


@callback(
    [Output("anim-interval", "disabled", allow_duplicate=True),
     Output("anim-playing", "data", allow_duplicate=True)],
    [Input("btn-pause", "n_clicks")],
    [State("anim-playing", "data")],
    prevent_initial_call=True
)
def toggle_pause(n, playing):
    if n:
        return playing, not playing
    return no_update, no_update


@callback(
    Output("anim-interval", "interval"),
    Input("anim-speed", "value"),
)
def update_speed(speed):
    return max(50, 600 - speed)


@callback(
    [Output("globe-graph", "figure"),
     Output("anim-frame", "data", allow_duplicate=True),
     Output("frame-counter", "children"),
     Output("metric-miss", "children"),
     Output("metric-prob", "children"),
     Output("metric-tca", "children"),
     Output("threat-level", "children"),
     Output("threat-level", "className"),
     Output("threat-bar", "value"),
     Output("threat-bar", "color"),
     Output("stat-pos-a", "children"),
     Output("stat-vel-a", "children"),
     Output("stat-alt-a", "children"),
     Output("stat-pos-b", "children"),
     Output("stat-vel-b", "children"),
     Output("stat-alt-b", "children"),
     Output("pinn-dr-a", "children"),
     Output("pinn-dr-b", "children"),
     Output("pinn-confidence", "children"),
    ],
    [Input("anim-interval", "n_intervals"),
     Input("sim-store", "data"),
     Input("visual-toggles", "value")],
    [State("anim-frame", "data"),
     State("anim-playing", "data")],
    prevent_initial_call='initial_duplicate',
)
def render_frame(n_intervals, sim_data, toggles, current_frame, playing):
    fig = go.Figure()

    defaults = ("---", "---", "---", "STANDBY", "threat-badge threat-standby", 0, "info",
                "---", "---", "---", "---", "---", "---", "---", "---", "---")

    if toggles and "stars" in toggles:
        fig.add_trace(STAR_TRACE)

    fig.add_trace(EARTH_TRACE)

    if toggles and "atmo" in toggles:
        fig.add_trace(ATMO_TRACE)

    if toggles and "grid" in toggles:
        for t in GRID_TRACES:
            fig.add_trace(t)

    frame_text = "0/0"
    telem = defaults

    if sim_data:
        traj_a = np.array(sim_data["traj_a"])
        traj_b = np.array(sim_data["traj_b"])
        total = sim_data["total_frames"]

        frame = current_frame if current_frame else 0
        if playing:
            frame = (frame + 1) % total
        frame = min(frame, total - 1)

        frame_text = f"{frame+1}/{total}"

        pos_a_now = traj_a[frame]
        pos_b_now = traj_b[frame]

        if toggles and "rails" in toggles:
            fig.add_trace(go.Scatter3d(
                x=traj_a[:frame+1, 0], y=traj_a[:frame+1, 1], z=traj_a[:frame+1, 2],
                mode='lines', line=dict(color='#00d4ff', width=3),
                hoverinfo='skip', showlegend=False, name='Trail A',
            ))
            fig.add_trace(go.Scatter3d(
                x=traj_a[frame:, 0], y=traj_a[frame:, 1], z=traj_a[frame:, 2],
                mode='lines', line=dict(color='rgba(0,212,255,0.2)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False, name='Future A',
            ))
            fig.add_trace(go.Scatter3d(
                x=traj_b[:frame+1, 0], y=traj_b[:frame+1, 1], z=traj_b[:frame+1, 2],
                mode='lines', line=dict(color='#ff6b35', width=3),
                hoverinfo='skip', showlegend=False, name='Trail B',
            ))
            fig.add_trace(go.Scatter3d(
                x=traj_b[frame:, 0], y=traj_b[frame:, 1], z=traj_b[frame:, 2],
                mode='lines', line=dict(color='rgba(255,107,53,0.2)', width=1, dash='dot'),
                hoverinfo='skip', showlegend=False, name='Future B',
            ))

        fig.add_trace(go.Scatter3d(
            x=[pos_a_now[0]], y=[pos_a_now[1]], z=[pos_a_now[2]],
            mode='markers+text',
            marker=dict(size=7, color='#00d4ff', symbol='diamond', line=dict(width=1, color='white')),
            text=[sim_data.get("sat_a_name", "SAT-A")], textposition="top center",
            textfont=dict(color="#00d4ff", size=10, family="Rajdhani"),
            showlegend=False, name='SAT-A',
        ))
        fig.add_trace(go.Scatter3d(
            x=[pos_b_now[0]], y=[pos_b_now[1]], z=[pos_b_now[2]],
            mode='markers+text',
            marker=dict(size=7, color='#ff6b35', symbol='circle', line=dict(width=1, color='white')),
            text=[sim_data.get("sat_b_name", "SAT-B")], textposition="top center",
            textfont=dict(color="#ff6b35", size=10, family="Rajdhani"),
            showlegend=False, name='SAT-B',
        ))

        dist_now = np.linalg.norm(pos_a_now - pos_b_now)
        if toggles and "killzone" in toggles:
            line_color = '#ff0040' if dist_now < 1000 else '#ffaa00' if dist_now < 3000 else 'rgba(255,255,255,0.15)'
            fig.add_trace(go.Scatter3d(
                x=[pos_a_now[0], pos_b_now[0]], y=[pos_a_now[1], pos_b_now[1]], z=[pos_a_now[2], pos_b_now[2]],
                mode='lines', line=dict(color=line_color, width=2, dash='dot'),
                hoverinfo='skip', showlegend=False, name='LOS',
            ))

        if toggles and "vectors" in toggles:
            vel_a = np.array(sim_data["vel_a"])
            vel_b = np.array(sim_data["vel_b"])
            scale = 500
            fig.add_trace(go.Scatter3d(
                x=[pos_a_now[0], pos_a_now[0] + vel_a[0]*scale],
                y=[pos_a_now[1], pos_a_now[1] + vel_a[1]*scale],
                z=[pos_a_now[2], pos_a_now[2] + vel_a[2]*scale],
                mode='lines', line=dict(color='#00ff88', width=3),
                hoverinfo='skip', showlegend=False,
            ))
            fig.add_trace(go.Scatter3d(
                x=[pos_b_now[0], pos_b_now[0] + vel_b[0]*scale],
                y=[pos_b_now[1], pos_b_now[1] + vel_b[1]*scale],
                z=[pos_b_now[2], pos_b_now[2] + vel_b[2]*scale],
                mode='lines', line=dict(color='#ff4444', width=3),
                hoverinfo='skip', showlegend=False,
            ))

        # --- Compute Telemetry ---
        miss = sim_data["miss_dist"]
        alt_a = np.linalg.norm(pos_a_now) - R_EARTH
        alt_b = np.linalg.norm(pos_b_now) - R_EARTH
        vel_a_mag = np.linalg.norm(sim_data["vel_a"])
        vel_b_mag = np.linalg.norm(sim_data["vel_b"])

        if miss < 100:
            prob = "> 1e-2"
            threat_txt = "CRITICAL"
            threat_cls = "threat-badge threat-critical"
            threat_val = 100
            threat_col = "danger"
        elif miss < 500:
            prob = "~1e-3"
            threat_txt = "HIGH"
            threat_cls = "threat-badge threat-high"
            threat_val = 75
            threat_col = "danger"
        elif miss < 2000:
            prob = "~1e-5"
            threat_txt = "WARNING"
            threat_cls = "threat-badge threat-warn"
            threat_val = 40
            threat_col = "warning"
        else:
            prob = "< 1e-7"
            threat_txt = "LOW"
            threat_cls = "threat-badge threat-low"
            threat_val = 10
            threat_col = "success"

        pos_a_str = f"[{pos_a_now[0]:.1f}, {pos_a_now[1]:.1f}, {pos_a_now[2]:.1f}]"
        pos_b_str = f"[{pos_b_now[0]:.1f}, {pos_b_now[1]:.1f}, {pos_b_now[2]:.1f}]"

        telem = (
            f"{miss:.2f}", prob, sim_data.get("target", "---"),
            threat_txt, threat_cls, threat_val, threat_col,
            pos_a_str, f"{vel_a_mag:.4f}", f"{alt_a:.1f}",
            pos_b_str, f"{vel_b_mag:.4f}", f"{alt_b:.1f}",
            f"{sim_data['dr_a']:.3f}", f"{sim_data['dr_b']:.3f}",
            "HIGH" if (sim_data['dr_a'] < 5 and sim_data['dr_b'] < 5) else "MODERATE",
        )

        mid = (pos_a_now + pos_b_now) / 2
        angle = (frame / total) * 2 * math.pi * 0.3
        cam_dist = 2.2
        eye = dict(
            x=cam_dist * math.cos(angle),
            y=cam_dist * math.sin(angle),
            z=0.5
        )
    else:
        frame = 0
        eye = dict(x=2.5, y=0.5, z=0.5)

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(visible=False, showgrid=False, zeroline=False, backgroundcolor='rgba(0,0,0,0)'),
            camera=dict(eye=eye, center=dict(x=0, y=0, z=0), up=dict(x=0, y=0, z=1)),
            dragmode='turntable',
            aspectmode='data',
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision='constant',
    )

    return (fig, frame, frame_text, *telem)


if __name__ == "__main__":
    app.run(debug=True, port=8051, host='127.0.0.1')

