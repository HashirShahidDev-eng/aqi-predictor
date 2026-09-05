

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Lahore AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FEATURE_GROUP_NAME = "aqi_features_history"
FEATURE_GROUP_VERSION = 7
MODEL_NAME = "aqi_forecast_model"
HORIZONS = [24, 48, 72]
CITY = "Lahore"


MODEL_RMSE = {24: 20.22, 48: 26.82, 72: 28.75}

# Feature lists must match training_pipeline.py exactly. Defined here rather
# than only read from metadata.pkl so the dashboard still works if the model
# artifact is loaded from a registry version that predates that file.
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


def forecast_features(h):
    return [
        f"temp_fc_{h}", f"humidity_fc_{h}", f"pressure_fc_{h}",
        f"wind_fc_{h}", f"precip_fc_{h}",
        f"wind_mean_next_{h}", f"wind_max_next_{h}",
        f"precip_sum_next_{h}", f"temp_mean_next_{h}",
        f"temp_delta_{h}", f"pressure_delta_{h}",
    ]


def feature_columns(h):
    return BASE_FEATURES + forecast_features(h)


# Readable labels for the SHAP panel — raw column names are not user-facing.
DISPLAY_NAMES = {
    "pm2_5": "PM2.5", "pm10": "PM10", "o3": "Ozone (O₃)",
    "no2": "Nitrogen Dioxide (NO₂)", "so2": "Sulfur Dioxide (SO₂)",
    "co": "Carbon Monoxide (CO)",
    "temperature": "Temperature", "humidity": "Humidity",
    "pressure": "Pressure", "wind_speed": "Wind Speed",
    "precipitation": "Precipitation",
    "aqi_lag_24": "AQI (1 day ago)", "aqi_lag_48": "AQI (2 days ago)",
    "aqi_lag_72": "AQI (3 days ago)",
    "aqi_rolling_avg_24h": "AQI average (24h)",
    "aqi_rolling_std_24h": "AQI volatility (24h)",
    "aqi_change_24h": "AQI change (24h)",
    "pm2_5_rolling_avg_24h": "PM2.5 average (24h)",
    "hour_cos": "Time of day", "hour_sin": "Time of day",
    "month_cos": "Season", "month_sin": "Season",
    "day_of_week": "Day of week",
}


def label_for(col):
    if col in DISPLAY_NAMES:
        return DISPLAY_NAMES[col]
    for h in HORIZONS:
        if col == f"wind_mean_next_{h}":
            return f"Forecast wind (next {h}h)"
        if col == f"wind_max_next_{h}":
            return f"Forecast peak wind ({h}h)"
        if col == f"precip_sum_next_{h}":
            return f"Forecast rain (next {h}h)"
        if col == f"temp_mean_next_{h}":
            return f"Forecast temp (next {h}h)"
        if col == f"temp_delta_{h}":
            return f"Temp change over {h}h"
        if col == f"pressure_delta_{h}":
            return f"Pressure change over {h}h"
        if col == f"wind_fc_{h}":
            return f"Wind at +{h}h"
        if col == f"precip_fc_{h}":
            return f"Rain at +{h}h"
        if col == f"temp_fc_{h}":
            return f"Temperature at +{h}h"
        if col == f"humidity_fc_{h}":
            return f"Humidity at +{h}h"
        if col == f"pressure_fc_{h}":
            return f"Pressure at +{h}h"
    return col.replace("_", " ").title()


# --- US-EPA AQI categories (0-500) -----------------------------------------
# These bands are why the project stays on the 0-500 scale: the thresholds
# that trigger advisories in Punjab are defined on it.
CATEGORIES = [
    (0, 50, "Good", "#00e400"),
    (51, 100, "Moderate", "#ffff00"),
    (101, 150, "Unhealthy for Sensitive Groups", "#ff7e00"),
    (151, 200, "Unhealthy", "#ff0000"),
    (201, 300, "Very Unhealthy", "#8f3f97"),
    (301, 500, "Hazardous", "#7e0023"),
]


def category(aqi):
    for low, high, name, color in CATEGORIES:
        if low <= aqi <= high:
            return name, color
    return "Hazardous", "#7e0023"


def health_advice(aqi):
    if aqi <= 50:
        return "Air quality is good. No precautions needed."
    if aqi <= 100:
        return "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion."
    if aqi <= 150:
        return "Sensitive groups — children, older adults, people with asthma or heart conditions — should reduce prolonged outdoor exertion."
    if aqi <= 200:
        return "Everyone may begin to feel effects. Limit outdoor exertion and keep windows closed."
    if aqi <= 300:
        return "Health alert. Avoid outdoor activity, run an air purifier indoors, and wear an N95 if you must go out."
    return "Health emergency. Stay indoors with windows sealed. Avoid all outdoor exertion."



st.markdown("""
<style>
  .stApp { background: #0b0f14; }
  .block-container { padding-top: 3rem; max-width: 1250px; }

  .card {
    background: #141a22; border: 1px solid #1f2833; border-radius: 14px;
    padding: 1.1rem 1.25rem; height: 100%;
  }
  .card-value { font-size: 1.9rem; font-weight: 650; color: #f0f4f8; line-height: 1.1; }
  .card-unit  { font-size: .8rem; color: #7d8b9a; margin-left: .3rem; font-weight: 400; }
  .card-label { font-size: .85rem; color: #8b99a7; margin-top: .35rem; }

  .app-title { font-size: 2.1rem; font-weight: 800; color: #ffffff; line-height: 1.3; padding-top: .2rem; margin-bottom: .1rem; }
  .section-title { font-size: 1.35rem; font-weight: 650; color: #f0f4f8; margin: 2rem 0 .2rem; }
  .section-sub   { font-size: .9rem; color: #7d8b9a; margin-bottom: .9rem; }

  .banner { border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; border-left: 4px solid; }
  .banner-title { font-weight: 650; font-size: 1rem; margin-bottom: .2rem; }
  .banner-body  { font-size: .88rem; opacity: .9; }

  .fc-head { display: flex; justify-content: space-between; align-items: center; }
  .fc-when { font-size: .78rem; color: #7d8b9a; }
  .fc-day  { font-size: 1.05rem; font-weight: 600; color: #f0f4f8; margin-top: .1rem; }
  .fc-aqi  { font-size: 2.6rem; font-weight: 700; line-height: 1.15; margin-top: .5rem; }
  .fc-note { font-size: .78rem; color: #7d8b9a; margin-top: .3rem; }
  .pill { border-radius: 999px; padding: .2rem .7rem; font-size: .75rem; font-weight: 600; }

  .shap-row { display: flex; justify-content: space-between; font-size: .9rem;
              color: #d3dae2; margin: .75rem 0 .3rem; }
  .shap-bar { height: 6px; border-radius: 3px; }

  hr { border-color: #1f2833; }
</style>
""", unsafe_allow_html=True)



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
    """Read the feature group. Cached for 30 minutes — fg.read() pulls the
    whole table and is the slowest thing on this page."""
    project = connect()
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_models():
    """Prefer the local model_dir (fast, present after a training run);
    fall back to downloading the latest registry version."""
    models = {}
    if all(os.path.exists(f"model_dir/model_h{h}.pkl") for h in HORIZONS):
        for h in HORIZONS:
            models[h] = joblib.load(f"model_dir/model_h{h}.pkl")
        return models, "local model_dir"

    project = connect()
    mr = project.get_model_registry()
    model = mr.get_model(MODEL_NAME)
    path = model.download()
    for h in HORIZONS:
        models[h] = joblib.load(os.path.join(path, f"model_h{h}.pkl"))
    return models, f"registry v{model.version}"


def predict(models, row):
    out = {}
    for h in HORIZONS:
        X = row[feature_columns(h)].to_frame().T.astype("float64")
        out[h] = float(models[h].predict(X)[0])
    return out


def local_shap(model, row, h, top_n=8):
    """Per-prediction explanation: how each feature moved THIS forecast, as
    opposed to the global SHAP summary saved during training."""
    X = row[feature_columns(h)].to_frame().T.astype("float64")
    values = shap.TreeExplainer(model).shap_values(X)[0]
    s = pd.Series(values, index=feature_columns(h))
    return s.reindex(s.abs().sort_values(ascending=False).index).head(top_n)


def band_shapes(y_max):
    """EPA category bands drawn behind the trend lines."""
    shapes = []
    for low, high, _, color in CATEGORIES:
        if low > y_max:
            break
        shapes.append(dict(
            type="rect", xref="paper", yref="y", x0=0, x1=1,
            y0=low, y1=min(high, y_max), fillcolor=color,
            opacity=0.10, layer="below", line_width=0,
        ))
    return shapes


def trend_chart(x, y, y2=None, name="AQI", name2=None):
    vals = [v for v in y if v is not None]
    if y2:
        vals += [v for v in y2 if v is not None]
    y_max = max(vals) * 1.15
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines",
        line=dict(color="#ff5c5c", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,92,92,0.12)",
    ))
    if y2 is not None:
        fig.add_trace(go.Scatter(
            x=x, y=y2, name=name2, mode="lines+markers",
            line=dict(color="#4da3ff", width=2.5, dash="dot"),
            marker=dict(size=8),
        ))
    fig.update_layout(
        shapes=band_shapes(y_max),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), height=340,
        yaxis=dict(title="US-EPA AQI", range=[0, y_max], gridcolor="#1f2833"),
        xaxis=dict(gridcolor="#1f2833"),
        showlegend=y2 is not None,
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified",
    )
    return fig


def cards(items):
    """Render a row of stat cards. items: (value, unit, label) tuples."""
    cols = st.columns(len(items))
    for col, (value, unit, lab) in zip(cols, items):
        col.markdown(
            f'<div class="card"><div class="card-value">{value}'
            f'<span class="card-unit">{unit}</span></div>'
            f'<div class="card-label">{lab}</div></div>',
            unsafe_allow_html=True,
        )


# --- Page ------------------------------------------------------------------

head_l, head_r = st.columns([5, 1])
head_l.markdown(
    f'<div class="app-title">{CITY} AQI Predictor</div>'
    "<span style='color:#7d8b9a'>Three-day air quality forecast, US-EPA scale</span>",
    unsafe_allow_html=True,
)
if head_r.button("Refresh", width="stretch"):
    st.cache_data.clear()
    st.rerun()

try:
    with st.spinner("Loading features from Hopsworks…"):
        df = load_features()
    with st.spinner("Loading models…"):
        models, model_source = load_models()
except Exception as e:
    st.error(f"Could not load data or models: {e}")
    st.caption(
        "Check that HOPSWORKS_API_KEY and HOPSWORKS_PROJECT are set, and that "
        "the training pipeline has registered a model."
    )
    st.stop()

latest = df.iloc[-1]
current_aqi = float(latest["us_aqi"])
cur_name, cur_color = category(current_aqi)
preds = predict(models, latest)

# Alert banner — the "implement alerts for hazardous AQI levels" requirement.
if current_aqi > 150 or max(preds.values()) > 150:
    peak = max(current_aqi, max(preds.values()))
    peak_name, peak_color = category(peak)
    st.markdown(
        f'<div class="banner" style="background:{peak_color}22;border-color:{peak_color}">'
        f'<div class="banner-title" style="color:{peak_color}">'
        f'Air quality alert — {peak_name} (AQI {peak:.0f})</div>'
        f'<div class="banner-body">{health_advice(peak)}</div></div>',
        unsafe_allow_html=True,
    )

age_h = (pd.Timestamp.now(tz="UTC") - latest["timestamp"]).total_seconds() / 3600
st.caption(
    f"Latest observation {latest['timestamp'].strftime('%Y-%m-%d %H:%M UTC')} "
    f"({age_h:.0f}h ago) · {len(df):,} hourly rows · models from {model_source}"
)

# --- Current AQI + pollutants ---
st.markdown('<div class="section-title">Current air quality</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">AQI from the 24-hour mean PM2.5 concentration, '
    f'per US-EPA method</div>', unsafe_allow_html=True)

aqi_col, poll_col = st.columns([1, 3])
aqi_col.markdown(
    f'<div class="card" style="border-color:{cur_color}55">'
    f'<div class="fc-aqi" style="color:{cur_color}">{current_aqi:.0f}</div>'
    f'<div class="pill" style="background:{cur_color}22;color:{cur_color};'
    f'display:inline-block;margin-top:.4rem">{cur_name}</div></div>',
    unsafe_allow_html=True,
)
with poll_col:
    cards([
        (f"{latest['pm2_5']:.1f}", "µg/m³", "PM2.5"),
        (f"{latest['pm10']:.1f}", "µg/m³", "PM10"),
        (f"{latest['o3']:.0f}", "µg/m³", "Ozone (O₃)"),
    ])
    st.write("")
    cards([
        (f"{latest['no2']:.1f}", "µg/m³", "Nitrogen Dioxide (NO₂)"),
        (f"{latest['so2']:.1f}", "µg/m³", "Sulfur Dioxide (SO₂)"),
        (f"{latest['co']:.0f}", "µg/m³", "Carbon Monoxide (CO)"),
    ])

# --- Recent trend ---
st.markdown('<div class="section-title">Recent AQI trend</div>', unsafe_allow_html=True)
window = st.radio("Window", ["24H", "48H", "72H"], index=0,
                  horizontal=True, label_visibility="collapsed")
hours = {"24H": 24, "48H": 48, "72H": 72}[window]
recent = df.tail(hours)

cards([
    (f"{recent['us_aqi'].iloc[-1]:.0f}", "", "Current"),
    (f"{recent['us_aqi'].mean():.0f}", "", "Average"),
    (f"{recent['us_aqi'].min():.0f}", "", "Minimum"),
    (f"{recent['us_aqi'].max():.0f}", "", "Maximum"),
])
st.write("")
st.plotly_chart(
    trend_chart(recent["timestamp"], recent["us_aqi"].tolist()),
    width="stretch", config={"displayModeBar": False},
)

# --- Current weather ---
st.markdown('<div class="section-title">Weather conditions</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Measured at the latest observation hour</div>',
            unsafe_allow_html=True)
cards([
    (f"{latest['temperature']:.1f}", "°C", "Temperature"),
    (f"{latest['humidity']:.0f}", "%", "Humidity"),
    (f"{latest['pressure']:.0f}", "hPa", "Pressure"),
    (f"{latest['wind_speed']:.1f}", "m/s", "Wind Speed"),
])

# --- Forecast ---
st.markdown('<div class="section-title">Three-day forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">One XGBoost model per horizon, trained on '
            'pollutant history, weather, and forecast weather</div>',
            unsafe_allow_html=True)

fc_cols = st.columns(3)
for col, h in zip(fc_cols, HORIZONS):
    aqi = preds[h]
    name, color = category(aqi)
    when = (latest["timestamp"] + pd.Timedelta(hours=h)).strftime("%a %d %b, %H:%M")
    col.markdown(
        f'<div class="card">'
        f'<div class="fc-head"><span class="fc-when">+{h}h · {when} UTC</span>'
        f'<span class="pill" style="background:{color}22;color:{color}">{name}</span></div>'
        f'<div class="fc-day">Day {HORIZONS.index(h) + 1}</div>'
        f'<div class="fc-aqi" style="color:{color}">{aqi:.0f}</div>'
        f'<div class="fc-note">± {MODEL_RMSE[h]:.1f} model RMSE</div></div>',
        unsafe_allow_html=True,
    )

delta = preds[72] - current_aqi
if delta < -10:
    verdict, vcolor = "Expected to improve", "#00e400"
elif delta > 10:
    verdict, vcolor = "Expected to worsen", "#ff5c5c"
else:
    verdict, vcolor = "Expected to hold steady", "#8b99a7"

st.write("")
label_l, label_r = st.columns([3, 1])
label_l.markdown('<div class="section-sub">Observed AQI and the forecast path '
                 'across the next three days</div>', unsafe_allow_html=True)
label_r.markdown(
    f'<div style="text-align:right;color:{vcolor};font-weight:600">{verdict} '
    f'({delta:+.0f})</div>', unsafe_allow_html=True)

hist = df.tail(24)
fx = list(hist["timestamp"]) + [latest["timestamp"] + pd.Timedelta(hours=h) for h in HORIZONS]
observed = hist["us_aqi"].tolist() + [None] * 3
forecast = [None] * (len(hist) - 1) + [current_aqi] + [preds[h] for h in HORIZONS]

st.plotly_chart(
    trend_chart(fx, observed, forecast, name="Observed", name2="Forecast"),
    width="stretch", config={"displayModeBar": False},
)

# --- Forecast breakdown ---
st.markdown('<div class="section-title">Forecast Breakdown</div>', unsafe_allow_html=True)
exp_h = st.radio("Horizon", HORIZONS, index=2, horizontal=True,
                 format_func=lambda h: f"+{h}h", label_visibility="collapsed")

contrib = local_shap(models[exp_h], latest, exp_h)
st.markdown(
    f'<div class="section-sub">SHAP contributions for the +{exp_h}h forecast of '
    f'{preds[exp_h]:.0f}. Red pushed the prediction up, teal pushed it down; '
    f'bar length is the size of the effect in AQI points.</div>',
    unsafe_allow_html=True,
)

widest = contrib.abs().max()
for col_name, val in contrib.items():
    color = "#ff5c5c" if val > 0 else "#2dd4bf"
    width = abs(val) / widest * 100
    st.markdown(
        f'<div class="shap-row"><span>{label_for(col_name)}</span>'
        f'<span style="color:{color}">{val:+.2f}</span></div>'
        f'<div style="background:#1f2833;border-radius:3px">'
        f'<div class="shap-bar" style="width:{width:.1f}%;background:{color}"></div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption(
    "Data: Open-Meteo (CAMS pollutants, ERA5 weather) · Feature store and model "
    f"registry: Hopsworks · Pipelines automated with GitHub Actions · "
    f"Forecast horizons {HORIZONS} hours. "
    "Forward-weather features were trained on reanalysis, so live accuracy is "
    "somewhat below the cross-validated figures shown."
)
