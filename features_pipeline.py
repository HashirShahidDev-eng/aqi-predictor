
import backfill_features as bf
import pandas as pd

PAST_DAYS = 7       # backward window: covers the 72h max lag + 24h rolling, with margin
FORECAST_DAYS = 4   # forward window: covers the 72h max horizon, with margin


def fetch_recent_pollutants():
    payload = bf._get(bf.AIR_QUALITY_URL, {
        "latitude": bf.LAT, "longitude": bf.LON,
        "hourly": ",".join(bf.POLLUTANT_VARS),
        "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        "timezone": "UTC", "domains": "cams_global",
    })
    return bf._hourly_to_df(payload, bf.POLLUTANT_RENAME)


def fetch_recent_weather():
    payload = bf._get(bf.FORECAST_URL, {
        "latitude": bf.LAT, "longitude": bf.LON,
        "hourly": ",".join(bf.WEATHER_VARS),
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
    print(f"Grid: {before} joined rows -> {len(df)} hourly slots "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()})")

    # Features are built over the FULL series, including the forecast tail, so
    # that rows near "now" can look forward. The tail is discarded below.
    df = bf.add_derived_features(df)

    required = [f"wind_mean_next_{h}" for h in bf.HORIZONS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Forward features absent: {missing}. "
                       "add_derived_features must call add_forecast_features.")

    now = pd.Timestamp.now(tz="UTC").floor("h")
    before_trim = len(df)
    df = df[df["timestamp"] <= now]
    print(f"Trimmed {before_trim - len(df)} forecast row(s) after {now} — "
          "only observed pollutant rows are stored")

    before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"After dropna: {len(df)} rows (dropped {before_drop - len(df)})")

    if len(df):
        staleness = (now - df["timestamp"].max()).total_seconds() / 3600
        print(f"Newest row: {df['timestamp'].max()} ({staleness:.0f}h behind now)")
        # More than a few hours behind means the forward window is still eating
        # recent rows — inference would be predicting from stale inputs.
        if staleness > 6:
            print(f"  WARNING: {staleness:.0f}h stale. Check that the "
                  "air-quality API actually returned forecast_days data.")
    return df


def push_to_feature_store(df):
    project = bf.hopsworks.login(api_key_value=bf.HOPSWORKS_API_KEY,
                                 project=bf.HOPSWORKS_PROJECT)
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
    # wait_for_job=False: on an hourly schedule, blocking on materialization
    # risks colliding with the previous run's job. unix_time is the primary
    # key, so a re-run upserts rather than duplicating.
    fg.insert(df, write_options={"wait_for_job": False})
    print(f"Upserted {len(df)} row(s) into 'aqi_features_history' "
          f"v{bf.FEATURE_GROUP_VERSION}.")


if __name__ == "__main__":
    df = build_recent_df()
    if df.empty:
        raise RuntimeError("No complete rows to upsert — check the API responses "
                           "above (e.g. a gap interpolate(limit=2) could not fill).")
    print(f"Range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    push_to_feature_store(df)