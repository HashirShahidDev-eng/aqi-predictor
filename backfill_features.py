

import os
import time
from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
LAT = float(os.getenv("CITY_LAT", "31.5204"))
LON = float(os.getenv("CITY_LON", "74.3587"))

DAYS_BACK = 730
FEATURE_GROUP_VERSION = 7
HORIZONS = [24, 48, 72]

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

POLLUTANT_VARS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
                  "sulphur_dioxide", "ozone", "us_aqi_pm2_5"]
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                "wind_speed_10m", "precipitation"]

POLLUTANT_RENAME = {"carbon_monoxide": "co", "nitrogen_dioxide": "no2",
                    "sulphur_dioxide": "so2", "ozone": "o3",
                    "us_aqi_pm2_5": "us_aqi_openmeteo"}
WEATHER_RENAME = {"temperature_2m": "temperature", "relative_humidity_2m": "humidity",
                  "surface_pressure": "pressure", "wind_speed_10m": "wind_speed"}

INTERPOLATE_COLS = ["co", "no2", "o3", "so2", "pm2_5", "pm10",
                    "temperature", "humidity", "pressure", "wind_speed", "precipitation"]


def calculate_us_aqi_pm25(pm25):
    """US-EPA PM2.5 AQI (0-500), pre-2024 breakpoints. EPA truncates the
    concentration to one decimal before lookup — without that, values like
    12.05 fall between breakpoints and hit the fallthrough."""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    pm25 = int(pm25 * 10) / 10.0
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low)
    return 500 if pm25 > 500.4 else 0


def _hourly_to_df(payload, rename):
    df = pd.DataFrame(payload["hourly"]).rename(columns={"time": "timestamp", **rename})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # Open-Meteo returns some fields (e.g. humidity) as whole numbers, which
    # pandas infers as int64 — but the feature group schema was fixed as
    # 'double' by the original backfill. Force float64 here so every fetch
    # (archive, forecast tail, hourly pipeline) matches it regardless of what
    # a given endpoint/response happens to look like.
    value_cols = [c for c in df.columns if c != "timestamp"]
    df[value_cols] = df[value_cols].astype("float64")
    return df


def _get(url, params):
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def fetch_chunked(url, extra_params, vars_, rename, start, end, label):
    frames = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=365), end)
        print(f"  {label} {cur.date()} -> {nxt.date()}")
        payload = _get(url, {
            "latitude": LAT, "longitude": LON,
            "start_date": cur.strftime("%Y-%m-%d"),
            "end_date": nxt.strftime("%Y-%m-%d"),
            "hourly": ",".join(vars_), "timezone": "UTC", **extra_params,
        })
        frames.append(_hourly_to_df(payload, rename))
        cur = nxt
        time.sleep(1.0)
    return (pd.concat(frames).drop_duplicates(subset="timestamp")
              .sort_values("timestamp").reset_index(drop=True))


def fetch_weather_recent(past_days=92):
    print(f"  forecast API (past_days={past_days}) for the archive tail")
    return _hourly_to_df(_get(FORECAST_URL, {
        "latitude": LAT, "longitude": LON, "hourly": ",".join(WEATHER_VARS),
        "past_days": past_days, "forecast_days": 1,
        "timezone": "UTC", "wind_speed_unit": "ms",
    }), WEATHER_RENAME)


def build_weather(start, end):
    archive = fetch_chunked(ARCHIVE_URL, {"wind_speed_unit": "ms"}, WEATHER_VARS,
                            WEATHER_RENAME, start, end, "archive")
    recent = fetch_weather_recent()
    tail = recent[recent["timestamp"] > archive["timestamp"].max()]
    print(f"  filling {len(tail)} tail hours from the forecast API")
    return (pd.concat([archive, tail]).drop_duplicates(subset="timestamp", keep="first")
              .sort_values("timestamp").reset_index(drop=True))


def add_forecast_features(df):
    """Forward-looking weather, one block per horizon.

    The forward-window aggregate uses shift(-H).rolling(H): after shifting,
    position i holds the value from i+H, so a backward rolling window of H at
    position i covers original positions i+1 .. i+H — exactly the interval
    between "now" and the target hour, and never including the present row.
    """
    for h in HORIZONS:
        # Point values at the target hour.
        df[f"temp_fc_{h}"] = df["temperature"].shift(-h)
        df[f"humidity_fc_{h}"] = df["humidity"].shift(-h)
        df[f"pressure_fc_{h}"] = df["pressure"].shift(-h)
        df[f"wind_fc_{h}"] = df["wind_speed"].shift(-h)
        df[f"precip_fc_{h}"] = df["precipitation"].shift(-h)

        # Aggregates over the intervening window — usually the stronger signal.
        # Total rain over three days scrubs PM2.5 far more than the rain rate
        # at one particular hour does.
        df[f"wind_mean_next_{h}"] = df["wind_speed"].shift(-h).rolling(h, min_periods=1).mean()
        df[f"wind_max_next_{h}"] = df["wind_speed"].shift(-h).rolling(h, min_periods=1).max()
        df[f"precip_sum_next_{h}"] = df["precipitation"].shift(-h).rolling(h, min_periods=1).sum()
        df[f"temp_mean_next_{h}"] = df["temperature"].shift(-h).rolling(h, min_periods=1).mean()

        # Change from now to the target hour — captures frontal passage.
        df[f"temp_delta_{h}"] = df[f"temp_fc_{h}"] - df["temperature"]
        df[f"pressure_delta_{h}"] = df[f"pressure_fc_{h}"] - df["pressure"]
    return df


def add_derived_features(df):
    """Turn a raw pollutant+weather hourly frame into the full v7 feature set:
    AQI target, calendar features, backward-looking lags/rolling stats, and
    the forward-looking weather block per horizon. Shared by the historical
    backfill and the hourly incremental pipeline so both always produce the
    same columns for the same kind of input."""
    df["pm2_5_24h"] = df["pm2_5"].rolling(24, min_periods=18).mean()
    df["us_aqi"] = df["pm2_5_24h"].apply(calculate_us_aqi_pm25)
    df["unix_time"] = (df["timestamp"].astype("int64") // 10**9).astype("int64")

    if "us_aqi_openmeteo" in df.columns:
        both = df[["us_aqi", "us_aqi_openmeteo"]].dropna()
        if len(both):
            diff = (both["us_aqi"] - both["us_aqi_openmeteo"]).abs()
            print(f"AQI cross-check vs Open-Meteo: mean abs diff={diff.mean():.2f}, "
                  f"max={diff.max():.1f}")

    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    # Backward-looking: lags and rolling, all shifted by >=1 (leak-free).
    for lag in [1, 3, 6, 12, 24, 48, 72]:
        df[f"aqi_lag_{lag}"] = df["us_aqi"].shift(lag)
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    df["aqi_change_1h"] = df["us_aqi"].shift(1) - df["us_aqi"].shift(2)
    df["aqi_change_24h"] = df["us_aqi"].shift(1) - df["us_aqi"].shift(25)
    df["aqi_rolling_avg_24h"] = df["us_aqi"].shift(1).rolling(24, min_periods=1).mean()
    df["aqi_rolling_std_24h"] = df["us_aqi"].shift(1).rolling(24, min_periods=1).std()
    df["pm2_5_rolling_avg_6h"] = df["pm2_5"].shift(1).rolling(6, min_periods=1).mean()
    df["pm2_5_rolling_avg_24h"] = df["pm2_5"].shift(1).rolling(24, min_periods=1).mean()

    # Forward-looking weather.
    df = add_forecast_features(df)

    drop_cols = ["pm2_5_24h"] + (["us_aqi_openmeteo"] if "us_aqi_openmeteo" in df.columns else [])
    return df.drop(columns=drop_cols)


def build_history_df():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS_BACK)

    print(f"Fetching {DAYS_BACK} days of pollutants...")
    df_p = fetch_chunked(AIR_QUALITY_URL, {"domains": "cams_global"}, POLLUTANT_VARS,
                         POLLUTANT_RENAME, start, end, "air-quality")
    print(f"Pollutant rows: {len(df_p)}")

    print("Fetching weather...")
    df_w = build_weather(start, end)
    print(f"Weather rows: {len(df_w)}")

    df = pd.merge(df_p, df_w, on="timestamp", how="inner")
    print(f"Joined rows: {len(df)}")

    # Hourly grid — every shift below counts ROWS, so a missing hour would
    # silently misalign both the lags and the new forward features.
    before = len(df)
    df = df.set_index("timestamp").asfreq("h")
    print(f"\nGrid: {before} joined rows -> {len(df)} hourly slots")
    print(f"Missing pm2_5 slots: {df['pm2_5'].isna().sum()}")
    df[INTERPOLATE_COLS] = df[INTERPOLATE_COLS].interpolate(limit=2)
    df = df.reset_index()

    print("\nAdding derived features (AQI target, calendar, lags, forecast)...")
    df = add_derived_features(df)

    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    # Rows lost at BOTH ends now: the lag warm-up at the start (~89) and the
    # forward-feature window at the end (72), since the last 72 hours have no
    # future weather to look at yet.
    print(f"After dropna: {len(df)} rows (dropped {before_drop - len(df)})")
    if len(df):
        print(f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
        print(f"us_aqi mean={df['us_aqi'].mean():.1f} std={df['us_aqi'].std():.1f}")
        print(f"Columns: {len(df.columns)}")
    return df


def push_to_feature_store(df):
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features_history",
        version=FEATURE_GROUP_VERSION,
        primary_key=["unix_time"],
        event_time="timestamp",
        description=("v6 features plus forward-looking weather (point values and "
                     "window aggregates) for 24/48/72h horizons. Forward features "
                     "come from ERA5 reanalysis, so CV results are an upper bound "
                     "relative to live forecast-driven inference."),
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Inserted {len(df)} rows into 'aqi_features_history' v{FEATURE_GROUP_VERSION}.")


if __name__ == "__main__":
    df = build_history_df()
    print("\nPreview:")
    print(df[["timestamp", "us_aqi", "wind_mean_next_72",
              "precip_sum_next_72", "temp_delta_72"]].head())
    push_to_feature_store(df)