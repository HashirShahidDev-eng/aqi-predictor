

import os
import time
from datetime import timedelta, timezone

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Lahore AQI Forecast",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

CITY = "Lahore"
PKT = timezone(timedelta(hours=5))
HORIZONS = [24, 48, 72]

FEATURE_GROUP_NAME = "aqi_features_history"
FEATURE_GROUP_VERSION = 7
MODEL_NAME = "aqi_forecast_model"

# Cross-validated results from the v7 training run (5-fold rolling origin).
# "persistence" is the AQI-in-H-hours-equals-AQI-now baseline the model beats.
METRICS = {
    24: {"rmse": 20.22, "mae": 15.91, "r2": 0.642, "persistence": 23.46},
    48: {"rmse": 26.82, "mae": 21.59, "r2": 0.402, "persistence": 31.26},
    72: {"rmse": 28.75, "mae": 22.99, "r2": 0.319, "persistence": 34.48},
}

# Must match training_pipeline.py exactly, or the models receive columns in a
# different order than they were fitted on.
BASE_FEATURES = [
    "hour", "day_of_week", "month",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "co", "no2", "o3", "so2", "pm2_5", "pm10",
    "temperature", "humidity", "pressure", "wind_speed", "precipitation",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24",
    "aqi_lag_48", "aqi_lag_72",
    "pm2_5_lag_1", "pm2_5_lag_3", "pm2_5_lag_6", "pm2_5_lag_12", "pm2_5_lag_24",
    "aqi_change_1h", "aqi_change_24h",
    "aqi_rolling_avg_24h", "aqi_rolling_std_24h",
    "pm2_5_rolling_avg_6h", "pm2_5_rolling_avg_24h",
]

FORECAST_FEATURE_STEMS = [
    "temp_fc", "humidity_fc", "pressure_fc", "wind_fc", "precip_fc",
    "wind_mean_next", "wind_max_next", "precip_sum_next", "temp_mean_next",
    "temp_delta", "pressure_delta",
]


def feature_columns(horizon):
    """Features for one horizon: shared base plus that horizon's weather block."""
    return BASE_FEATURES + [f"{stem}_{horizon}" for stem in FORECAST_FEATURE_STEMS]


# Human labels for the SHAP panel. {h} is filled with the horizon.
FEATURE_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "o3": "Ozone", "no2": "Nitrogen dioxide",
    "so2": "Sulfur dioxide", "co": "Carbon monoxide",
    "temperature": "Temperature", "humidity": "Humidity", "pressure": "Pressure",
    "wind_speed": "Wind speed", "precipitation": "Rainfall",
    "hour": "Hour", "month": "Month", "day_of_week": "Day of week",
    "hour_sin": "Time of day", "hour_cos": "Time of day",
    "month_sin": "Season", "month_cos": "Season",
    "aqi_lag_1": "AQI (1 hour ago)", "aqi_lag_3": "AQI (3 hours ago)",
    "aqi_lag_6": "AQI (6 hours ago)", "aqi_lag_12": "AQI (12 hours ago)",
    "aqi_lag_24": "AQI (1 day ago)", "aqi_lag_48": "AQI (2 days ago)",
    "aqi_lag_72": "AQI (3 days ago)",
    "pm2_5_lag_1": "PM2.5 (1 hour ago)", "pm2_5_lag_3": "PM2.5 (3 hours ago)",
    "pm2_5_lag_6": "PM2.5 (6 hours ago)", "pm2_5_lag_12": "PM2.5 (12 hours ago)",
    "pm2_5_lag_24": "PM2.5 (1 day ago)",
    "aqi_change_1h": "AQI change (1h)", "aqi_change_24h": "AQI change (24h)",
    "aqi_rolling_avg_24h": "AQI average (24h)",
    "aqi_rolling_std_24h": "AQI volatility (24h)",
    "pm2_5_rolling_avg_6h": "PM2.5 average (6h)",
    "pm2_5_rolling_avg_24h": "PM2.5 average (24h)",
}

FORECAST_LABELS = {
    "wind_mean_next": "Forecast wind, next {h}h",
    "wind_max_next": "Forecast peak wind, next {h}h",
    "precip_sum_next": "Forecast rainfall, next {h}h",
    "temp_mean_next": "Forecast temperature, next {h}h",
    "temp_delta": "Temperature change over {h}h",
    "pressure_delta": "Pressure change over {h}h",
    "wind_fc": "Wind at +{h}h",
    "precip_fc": "Rainfall at +{h}h",
    "temp_fc": "Temperature at +{h}h",
    "humidity_fc": "Humidity at +{h}h",
    "pressure_fc": "Pressure at +{h}h",
}


def label_for(column):
    if column in FEATURE_LABELS:
        return FEATURE_LABELS[column]
    stem, _, horizon = column.rpartition("_")
    if stem in FORECAST_LABELS:
        return FORECAST_LABELS[stem].format(h=horizon)
    return column.replace("_", " ").capitalize()


# ============================================================================
# 2. AQI DOMAIN
# ============================================================================


CATEGORIES = [
    (0, 50, "Good", "#4ade80"),
    (51, 100, "Moderate", "#fbbf24"),
    (101, 150, "Sensitive groups", "#fb923c"),
    (151, 200, "Unhealthy", "#f87171"),
    (201, 300, "Very unhealthy", "#a78bfa"),
    (301, 500, "Hazardous", "#e11d48"),
]
CATEGORY_COLOR = {name: color for _, _, name, color in CATEGORIES}

BREAKPOINTS = {
    "pm2_5": [(0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
              (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
              (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500)],
    "pm10": [(0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
             (255, 354, 151, 200), (355, 424, 201, 300), (425, 504, 301, 400),
             (505, 604, 401, 500)],
    "o3": [(0.0, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
           (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300)],
    "co": [(0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
           (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400),
           (40.5, 50.4, 401, 500)],
    "so2": [(0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
            (186, 304, 151, 200), (305, 604, 201, 300), (605, 804, 301, 400),
            (805, 1004, 401, 500)],
    "no2": [(0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
            (361, 649, 151, 200), (650, 1249, 201, 300), (1250, 1649, 301, 400),
            (1650, 2049, 401, 500)],
}

# Open-Meteo reports everything in ug/m3, but EPA defines ozone and CO in ppm
# and NO2/SO2 in ppb. Factors are for 25 C and 1 atm.
UNIT_FACTOR = {
    "pm2_5": 1.0, "pm10": 1.0,
    "o3": 0.5094 / 1000,     # ug/m3 -> ppm
    "co": 0.000873,          # ug/m3 -> ppm
    "so2": 0.3816,           # ug/m3 -> ppb
    "no2": 0.5314,           # ug/m3 -> ppb
}

POLLUTANT_NAMES = {"pm2_5": "PM2.5", "pm10": "PM10", "o3": "Ozone",
                   "co": "CO", "so2": "SO₂", "no2": "NO₂"}


def piecewise_aqi(value, breakpoints):
    """Linear interpolation between EPA breakpoints."""
    if value is None or pd.isna(value) or value < 0:
        return None
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= value <= c_high:
            return ((i_high - i_low) / (c_high - c_low)) * (value - c_low) + i_low
    return 500.0 if value > breakpoints[-1][1] else None


def dominant_pollutant(row):
    """Pollutant with the highest AQI ."""
    scores = {}
    for key, breakpoints in BREAKPOINTS.items():
        value = row[key] * UNIT_FACTOR[key]
        if key == "pm2_5":
            value = int(value * 10) / 10        # EPA truncates to one decimal
        index = piecewise_aqi(value, breakpoints)
        if index is not None:
            scores[POLLUTANT_NAMES[key]] = index
    return max(scores, key=scores.get) if scores else "PM2.5"


def category(aqi):
    for low, high, name, color in CATEGORIES:
        if low <= aqi <= high:
            return name, color
    return "Hazardous", "#e11d48"


def health_advice(aqi):
    
    if aqi <= 50:
        return "The air is clean. Good day to be outside."
    if aqi <= 100:
        return ("The air is okay for most people. If smoke or dust usually bothers "
                "you, take it easy outdoors.")
    if aqi <= 150:
        return ("Not safe for everyone. Children, older people, and anyone with "
                "asthma or a heart problem should stay inside or keep outdoor time "
                "short.")
    if aqi <= 200:
        return ("Bad air. Most people will feel it. Stay indoors where you can, "
                "shut the windows, and skip outdoor exercise.")
    if aqi <= 300:
        return ("Very bad air. Do not go outside unless you have to. Wear an N95 "
                "mask if you do, and run an air purifier at home.")
    return ("Dangerous air. Stay inside. Keep every window and door shut. Do not "
            "go out or exercise outdoors at all.")


# ============================================================================
# 3. PRESENTATION
# ============================================================================

INK = "#eef2f6"
MUTED = "#6f7d8c"
LABEL = "#8593a2"
SURFACE = "#161b22"
LINE = "#232a35"
GRID = "#1e242d"
TEAL = "#5eead4"
AMBER = "#fb923c"

st.markdown(f"""
<style>
.stApp {{ background: #0e1116; }}
  .block-container {{ padding-top: 2.5rem; max-width: 1260px; }}
  h1, h2, h3 {{ color: {INK}; }}
  hr {{ border-color: {LINE}; }}

  .app-title {{ font-size: 2rem; font-weight: 700; color: {INK};
                letter-spacing: -.02em; margin-bottom: .15rem; }}
  .app-sub {{ color: {MUTED}; font-size: .92rem; }}

  .card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 12px;
           padding: 1rem 1.15rem; height: 100%; }}
  .card-value {{ font-size: 1.7rem; font-weight: 620; color: {INK}; line-height: 1.15; }}
  .card-unit {{ font-size: .78rem; color: {MUTED}; margin-left: .3rem; font-weight: 400; }}
  .card-label {{ font-size: .82rem; color: {LABEL}; margin-top: .3rem; }}

  .section-title {{ font-size: 1.2rem; font-weight: 620; color: {INK};
                    margin: 1.9rem 0 .15rem; }}
  .section-sub {{ font-size: .87rem; color: {MUTED}; margin-bottom: .85rem; }}

  .banner {{ border-radius: 12px; padding: .95rem 1.2rem; margin-bottom: 1.2rem;
             border-left: 3px solid; }}
  .banner-title {{ font-weight: 620; font-size: .98rem; margin-bottom: .2rem; }}
  .banner-body {{ font-size: .86rem; color: #c4cdd6; }}

  .fc-when {{ font-size: .76rem; color: {MUTED}; }}
  .fc-aqi {{ font-size: 2.4rem; font-weight: 700; line-height: 1.2;
             margin: .35rem 0 .1rem; }}
  .fc-meta {{ font-size: .8rem; color: {LABEL}; margin-top: .5rem; }}
  .pill {{ border-radius: 999px; padding: .18rem .6rem; font-size: .74rem;
           font-weight: 600; display: inline-block; }}

  .shap-row {{ display: flex; justify-content: space-between; font-size: .88rem;
               color: #c4cdd6; margin: .7rem 0 .28rem; }}
  .shap-track {{ background: {GRID}; border-radius: 3px; }}
  .shap-bar {{ height: 6px; border-radius: 3px; }}

  .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {LINE}; }}
  .stTabs [data-baseweb="tab"] {{ background: transparent; color: {MUTED};
                                  font-size: .95rem; padding: .4rem 0; }}
  .stTabs [aria-selected="true"] {{ color: {INK}; }}
</style>
""", unsafe_allow_html=True)

CHART_BASE = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=LABEL, size=12),
    margin=dict(l=10, r=10, t=30, b=10),
)


def heading(title, subtitle=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def cards(items):
    """Row of stat cards. items: (value, unit, label)."""
    for col, (value, unit, lab) in zip(st.columns(len(items)), items):
        col.markdown(
            f'<div class="card"><div class="card-value">{value}'
            f'<span class="card-unit">{unit}</span></div>'
            f'<div class="card-label">{lab}</div></div>',
            unsafe_allow_html=True)


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def rgba(hex_color, alpha):
    """Plotly rejects 8-digit hex (#rrggbbaa) that CSS accepts, so convert."""
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def gauge_chart(aqi):
    _, color = category(aqi)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi,
        number=dict(font=dict(size=54, color=INK)),
        gauge=dict(
            axis=dict(range=[0, 400], tickwidth=1, tickcolor="#3a4450",
                      tickvals=[0, 100, 200, 300, 400]),
            bar=dict(color=color, thickness=0.7),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[low, min(high, 400)], color=rgba(c, 0.13))
                   for low, high, _, c in CATEGORIES if low < 400],
            threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=aqi),
        ),
    ))
    fig.update_layout(**{**CHART_BASE, "height": 250,
                         "margin": dict(l=20, r=20, t=10, b=0)})
    return fig


def render_gauge(slot, aqi, animate):
    
    if animate:
        frames = 18
        for i in range(1, frames):
            progress = 1 - (1 - i / frames) ** 3
            slot.plotly_chart(gauge_chart(aqi * progress),
                              use_container_width=True,
                              config={"displayModeBar": False},
                              key=f"gauge_frame_{i}")
            time.sleep(0.025)
    slot.plotly_chart(gauge_chart(aqi), use_container_width=True,
                      config={"displayModeBar": False}, key="gauge_final")


def category_bands(y_max):
    """EPA bands shaded behind a trend line."""
    return [dict(type="rect", xref="paper", yref="y", x0=0, x1=1,
                 y0=low, y1=min(high, y_max), fillcolor=color,
                 opacity=0.07, layer="below", line_width=0)
            for low, high, _, color in CATEGORIES if low <= y_max]


def trend_chart(x, series, forecast=None, height=330):
    values = [v for v in series if v is not None]
    if forecast:
        values += [v for v in forecast if v is not None]
    y_max = max(values) * 1.2

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=series, name="Observed", mode="lines",
        line=dict(color=TEAL, width=2.4),
        fill="tozeroy", fillcolor="rgba(94,234,212,0.08)"))
    if forecast is not None:
        fig.add_trace(go.Scatter(
            x=x, y=forecast, name="Forecast", mode="lines+markers",
            line=dict(color=AMBER, width=2.4, dash="dot"),
            marker=dict(size=8, line=dict(width=0))))
    fig.update_layout(**{
        **CHART_BASE, "height": height, "shapes": category_bands(y_max),
        "xaxis": dict(gridcolor=GRID),
        "yaxis": dict(title="AQI", range=[0, y_max], gridcolor=GRID),
        "showlegend": forecast is not None,
        "legend": dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        "hovermode": "x unified"})
    return fig


def bar_chart(x, y, colors, x_title=None, y_title=None, horizontal=False, height=300):
    fig = go.Figure(go.Bar(
        x=y if horizontal else x, y=x if horizontal else y,
        orientation="h" if horizontal else "v", marker_color=colors))
    fig.update_layout(**{
        **CHART_BASE, "height": height,
        "xaxis": dict(title=x_title, gridcolor=GRID),
        "yaxis": dict(title=y_title, gridcolor=GRID)})
    return fig


# ============================================================================
# 4. DATA ACCESS
# ============================================================================

def get_secret(name, default=None):
    
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


@st.cache_resource(show_spinner=False)
def connect():
    import hopsworks
    return hopsworks.login(
        api_key_value=get_secret("HOPSWORKS_API_KEY"),
        project=get_secret("HOPSWORKS_PROJECT"),
    )


@st.cache_data(ttl=1800, show_spinner=False)
def load_features():
    """Cached for 30 minutes — fg.read() pulls the whole table and is by far
    the slowest thing on this page."""
    fs = connect().get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_models():
  
    if all(os.path.exists(f"model_dir/model_h{h}.pkl") for h in HORIZONS):
        return {h: joblib.load(f"model_dir/model_h{h}.pkl") for h in HORIZONS}, "local files"

    model = connect().get_model_registry().get_model(MODEL_NAME)
    path = model.download()
    models = {h: joblib.load(os.path.join(path, f"model_h{h}.pkl")) for h in HORIZONS}
    return models, f"registry v{model.version}"


def as_row(row, horizon):
    return row[feature_columns(horizon)].to_frame().T.astype("float64")


def predict(models, row):
    return {h: float(models[h].predict(as_row(row, h))[0]) for h in HORIZONS}


def shap_contributions(model, row, horizon, top_n=9):
    """Per-prediction attribution: how each feature moved THIS forecast, which
    is a different question from the global summary saved during training."""
    values = shap.TreeExplainer(model).shap_values(as_row(row, horizon))[0]
    s = pd.Series(values, index=feature_columns(horizon))
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)


def to_pkt(ts):
    return ts.tz_convert(PKT)


# --- Load once, before the tabs render --------------------------------------

header, refresh = st.columns([5, 1])
header.markdown(
    f'<div class="app-title">{CITY} air quality forecast</div>'
    '<div class="app-sub">Three days ahead, updated hourly</div>',
    unsafe_allow_html=True)
if refresh.button("Refresh", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Loading…"):
        df = load_features()
        models, model_source = load_models()
except Exception as error:
    st.error(f"Could not load data or models: {error}")
    st.caption("Check that HOPSWORKS_API_KEY and HOPSWORKS_PROJECT are set and that "
               "the training pipeline has registered a model.")
    st.stop()

latest = df.iloc[-1]
current_aqi = float(latest["us_aqi"])
predictions = predict(models, latest)

tab_forecast, tab_performance, tab_explain, tab_data = st.tabs(
    ["Forecast", "Model performance", "Explainability", "Data analysis"])


# ============================================================================
# 5A. FORECAST
# ============================================================================

with tab_forecast:
    peak_horizon = max(predictions, key=predictions.get)
    peak = max(current_aqi, predictions[peak_horizon])

    if peak > 100:
        peak_name, peak_color = category(peak)
        when = "right now" if peak == current_aqi else f"within {peak_horizon} hours"
        st.markdown(
            f'<div class="banner" style="background:{peak_color}14;'
            f'border-color:{peak_color}">'
            f'<div class="banner-title" style="color:{peak_color}">'
            f'{peak_name} — AQI {peak:.0f} expected {when}</div>'
            f'<div class="banner-body">{health_advice(peak)}</div></div>',
            unsafe_allow_html=True)

    gauge_col, detail_col = st.columns([1.05, 1.95])

    with gauge_col:
        current_name, current_color = category(current_aqi)
        first_view = not st.session_state.get("gauge_played", False)
        st.session_state["gauge_played"] = True
        render_gauge(st.empty(), current_aqi, animate=first_view)
        st.markdown(
            f'<div style="text-align:center;margin-top:-1.1rem">'
            f'<span class="pill" style="background:{current_color}1f;'
            f'color:{current_color}">{current_name}</span>'
            f'<div class="app-sub" style="margin-top:.55rem">Observed '
            f'{to_pkt(latest["timestamp"]).strftime("%d %b %Y, %H:%M")} PKT</div></div>',
            unsafe_allow_html=True)

    with detail_col:
        cards([(dominant_pollutant(latest), "", "Dominant pollutant"),
               (f"{latest['pm2_5']:.1f}", "µg/m³", "PM2.5"),
               (f"{latest['pm10']:.1f}", "µg/m³", "PM10")])
        st.write("")
        cards([(f"{latest['o3']:.0f}", "µg/m³", "Ozone"),
               (f"{latest['no2']:.1f}", "µg/m³", "Nitrogen dioxide"),
               (f"{latest['so2']:.1f}", "µg/m³", "Sulfur dioxide")])
        st.write("")
        cards([(f"{latest['temperature']:.1f}", "°C", "Temperature"),
               (f"{latest['humidity']:.0f}", "%", "Humidity"),
               (f"{latest['wind_speed']:.1f}", "m/s", "Wind")])

    heading("Three-day forecast")

    for col, horizon in zip(st.columns(3), HORIZONS):
        aqi = predictions[horizon]
        name, color = category(aqi)
        delta = aqi - current_aqi
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        delta_color = "#f87171" if delta > 3 else "#4ade80" if delta < -3 else LABEL
        when = to_pkt(latest["timestamp"] + pd.Timedelta(hours=horizon))
        col.markdown(
            f'<div class="card">'
            f'<div class="fc-when">+{horizon}h · '
            f'{when.strftime("%a %d %b, %H:%M")} PKT</div>'
            f'<div class="fc-aqi" style="color:{color}">{aqi:.0f}</div>'
            f'<span class="pill" style="background:{delta_color}1f;color:{delta_color}">'
            f'{arrow} {delta:+.0f} vs now</span>'
            f'<div class="fc-meta">{name} · ± {METRICS[horizon]["rmse"]:.0f} RMSE</div>'
            f'</div>', unsafe_allow_html=True)

    st.write("")
    history = df.tail(24)
    axis = list(history["timestamp"].map(to_pkt)) + [
        to_pkt(latest["timestamp"] + pd.Timedelta(hours=h)) for h in HORIZONS]
    observed = history["us_aqi"].tolist() + [None] * 3
    # Repeat the current value so the forecast line starts where observation ends.
    projected = [None] * (len(history) - 1) + [current_aqi] + \
                [predictions[h] for h in HORIZONS]
    show(trend_chart(axis, observed, projected))

    heading("Recent history")
    window = st.radio("Window", ["24 hours", "3 days", "7 days", "30 days"],
                      index=1, horizontal=True, label_visibility="collapsed")
    recent = df.tail({"24 hours": 24, "3 days": 72,
                      "7 days": 168, "30 days": 720}[window])
    cards([(f"{recent['us_aqi'].iloc[-1]:.0f}", "", "Latest"),
           (f"{recent['us_aqi'].mean():.0f}", "", "Average"),
           (f"{recent['us_aqi'].min():.0f}", "", "Lowest"),
           (f"{recent['us_aqi'].max():.0f}", "", "Highest")])
    st.write("")
    show(trend_chart(recent["timestamp"].map(to_pkt), recent["us_aqi"].tolist()))


# ============================================================================
# 5B. MODEL PERFORMANCE
# ============================================================================

with tab_performance:
    heading("Model performance")

    st.dataframe(pd.DataFrame([{
        "Horizon": f"+{h}h",
        "RMSE": round(m["rmse"], 2),
        "MAE": round(m["mae"], 2),
        "R²": round(m["r2"], 3),
        "Persistence RMSE": round(m["persistence"], 2),
        "Skill": f"{(1 - m['rmse'] / m['persistence']) * 100:.0f}%",
    } for h, m in METRICS.items()]), hide_index=True, use_container_width=True)

    comparison = go.Figure()
    comparison.add_trace(go.Bar(
        x=[f"+{h}h" for h in HORIZONS],
        y=[METRICS[h]["persistence"] for h in HORIZONS],
        name="Persistence baseline", marker_color="#3a4450"))
    comparison.add_trace(go.Bar(
        x=[f"+{h}h" for h in HORIZONS],
        y=[METRICS[h]["rmse"] for h in HORIZONS],
        name="XGBoost", marker_color=TEAL))
    comparison.update_layout(**{
        **CHART_BASE, "height": 330, "barmode": "group",
        "xaxis": dict(gridcolor=GRID),
        "yaxis": dict(title="RMSE, AQI points", gridcolor=GRID),
        "legend": dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)")})
    show(comparison)

    heading("How it was built")
    st.markdown("""
- - 
""")


# ============================================================================
# 5C. EXPLAINABILITY
# ============================================================================

with tab_explain:
    heading("Key factors behind the forecast")
    explain_horizon = st.radio(
        "Horizon", HORIZONS, index=2, horizontal=True,
        format_func=lambda h: f"+{h} hours", label_visibility="collapsed")

    contributions = shap_contributions(models[explain_horizon], latest, explain_horizon)
    st.markdown(
        f'<div class="section-sub">SHAP values for the +{explain_horizon}h prediction '
        f'of <strong style="color:{INK}">{predictions[explain_horizon]:.0f}</strong>. '
        f'Each bar is that feature\'s contribution in AQI points. Amber pushed the '
        f'forecast up, teal pushed it down.</div>', unsafe_allow_html=True)

    widest = contributions.abs().max()
    for column, value in contributions.items():
        color = AMBER if value > 0 else TEAL
        st.markdown(
            f'<div class="shap-row"><span>{label_for(column)}</span>'
            f'<span style="color:{color}">{value:+.2f}</span></div>'
            f'<div class="shap-track"><div class="shap-bar" style="'
            f'width:{abs(value) / widest * 100:.1f}%;background:{color}"></div></div>',
            unsafe_allow_html=True)

    st.write("")
    st.markdown(
        '<div class="section-sub">This is a local explanation — it describes this one '
        'prediction, not the model in general. The training pipeline also writes a '
        'global SHAP summary across the full test set.</div>', unsafe_allow_html=True)


# ============================================================================
# 5D. DATA ANALYSIS
# ============================================================================

with tab_data:
    heading("Two years readings")
    cards([(f"{len(df):,}", "", "Hourly rows"),
           (f"{df['us_aqi'].mean():.0f}", "", "Mean AQI"),
           (f"{df['us_aqi'].max():.0f}", "", "Peak AQI"),
           (f"{(df['us_aqi'] > 150).mean() * 100:.0f}", "%", "Hours above 150")])

    heading("Seasonal pattern",
            "Monthly mean AQI. The winter peak is Lahore's smog season  "
            )
    by_month = df.groupby("month")["us_aqi"].mean()
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    show(bar_chart([month_names[m - 1] for m in by_month.index], by_month.values,
                   [category(v)[1] for v in by_month.values], y_title="Mean AQI"))

    left, right = st.columns(2)

    with left:
        heading("Wind and pollution", "Mean AQI by wind speed.")
        buckets = pd.cut(df["wind_speed"], bins=[0, 1, 2, 3, 4, 5, 100],
                         labels=["0–1", "1–2", "2–3", "3–4", "4–5", "5+"])
        by_wind = df.groupby(buckets, observed=True)["us_aqi"].mean()
        show(bar_chart(list(by_wind.index.astype(str)), by_wind.values, "#818cf8",
                       x_title="Wind speed (m/s)", y_title="Mean AQI", height=290))

    with right:
        heading("AQI Category Distribution",
                "Share of all recorded hours .")
        counts = df["us_aqi"].apply(lambda v: category(v)[0]).value_counts()
        order = [name for _, _, name, _ in CATEGORIES if name in counts.index]
        show(bar_chart(order, [counts[o] / len(df) * 100 for o in order],
                       [CATEGORY_COLOR[o] for o in order],
                       x_title="% of hours", horizontal=True, height=290))


st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    f"Open-Meteo (CAMS pollutants, ERA5 weather) · Hopsworks feature store "
    f"v{FEATURE_GROUP_VERSION} and model registry · GitHub Actions runs the hourly "
    f"feature pipeline and daily retraining · models loaded from {model_source}"
)