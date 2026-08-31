"""
Hourly feature pipeline (v7) — keeps 'aqi_features_history' current between
daily backfills.

Reuses the same feature-engineering code as backfill_features.py (imported,
not duplicated) so the hourly and historical paths can never drift into
different schemas — this replaces the old version of this file, which wrote
a completely different feature group/schema (aqi_features v1, OpenWeatherMap
fields) that the training pipeline never read from.

Each run re-fetches a trailing window of recent pollutant/weather data plus a
forward weather forecast, recomputes the full feature set, and upserts it
into the feature group. Upserting the trailing window rather than a single
row means a missed hour self-heals on the next run, and rows near the edge
of the forecast horizon get refreshed as better forecast data comes in.

Intended to run every hour (cron / GitHub Actions / Airflow).
"""

import backfill_features as bf
import pandas as pd

PAST_DAYS = 7       # backward window: covers the 72h max lag + 24h rolling, with margin
FORECAST_DAYS = 4   # forward window: covers the 72h max horizon, with margin


def fetch_recent_pollutants():
    payload = bf._get(bf.AIR_QUALITY_URL, {
        "latitude": bf.LAT, "longitude": bf.LON, "hourly": ",".join(bf.POLLUTANT_VARS),
        "past_days": PAST_DAYS, "forecast_days": 1,
        "timezone": "UTC", "domains": "cams_global",
    })
    return bf._hourly_to_df(payload, bf.POLLUTANT_RENAME)


def fetch_recent_weather():
    payload = bf._get(bf.FORECAST_URL, {
        "latitude": bf.LAT, "longitude": bf.LON, "hourly": ",".join(bf.WEATHER_VARS),
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "UTC", "wind_speed_unit": "ms",
    })
    return bf._hourly_to_df(payload, bf.WEATHER_RENAME)


def build_recent_df():
    df_p = fetch_recent_pollutants()
    df_w = fetch_recent_weather()
    df = pd.merge(df_p, df_w, on="timestamp", how="inner")

    before = len(df)
    df = df.set_index("timestamp").asfreq("h")
    df[bf.INTERPOLATE_COLS] = df[bf.INTERPOLATE_COLS].interpolate(limit=2)
    df = df.reset_index()
    print(f"Grid: {before} joined rows -> {len(df)} hourly slots")

    df = bf.add_derived_features(df)

    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"After dropna: {len(df)} rows (dropped {before_drop - len(df)})")
    return df


def push_to_feature_store(df):
    project = bf.hopsworks.login(api_key_value=bf.HOPSWORKS_API_KEY, project=bf.HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name="aqi_features_history",
        version=bf.FEATURE_GROUP_VERSION,
        primary_key=["unix_time"],
        event_time="timestamp",
        description=("v6 features plus forward-looking weather for 24/48/72h "
                     "horizons. This hourly pipeline upserts a trailing window "
                     "using live Open-Meteo forecasts for the forward block, "
                     "vs. backfill_features.py's one-time ERA5-based history."),
        time_travel_format="HUDI",
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"Upserted {len(df)} row(s) into 'aqi_features_history' v{bf.FEATURE_GROUP_VERSION}.")


if __name__ == "__main__":
    df = build_recent_df()
    if df.empty:
        raise RuntimeError("No complete rows to upsert — check the API responses above "
                            "(e.g. a gap the interpolate(limit=2) couldn't fill).")
    print(f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    push_to_feature_store(df)
