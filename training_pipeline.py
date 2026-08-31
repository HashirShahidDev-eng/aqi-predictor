"""
Training pipeline (v7) — with weather forecast features.

Same structure as the clean v6 trainer, one important difference: the feature
set is now HORIZON-SPECIFIC. The +24h model gets the forward-weather block for
24 hours ahead, the +48h model gets the 48h block, and so on. Giving the +24h
model the 72h forward weather would hand it information about a period past its
own target — not leakage of the answer, but a mismatch with what inference can
supply, so each horizon sees only its own block.

The forecast features are why v7 exists. In v6 the model knew the wind speed
*now* but nothing about what the atmosphere would do between now and the target
hour, which is most of what determines AQI three days out. That was a
missing-input problem, and it is why +72h R2 sat at 0.086.

CAVEAT, restated from the backfill: forward features here come from ERA5
reanalysis — what the weather ACTUALLY did. Live inference will use Open-Meteo
forecasts, which carry their own error. These CV numbers are therefore an
upper bound, and the write-up must say so.
"""

import os

import hopsworks
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FEATURE_GROUP_VERSION = 7
TARGET_COLUMN = "us_aqi"
HORIZONS = [24, 48, 72]
PRIMARY_HORIZON = 72
N_SPLITS = 5

# Everything available regardless of horizon: current conditions, calendar,
# and backward-looking lags / rolling stats.
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


def forecast_features(horizon):
    """Forward-looking weather for one horizon. Point values at the target
    hour, aggregates over the intervening window, and now-to-then deltas."""
    h = horizon
    return [
        f"temp_fc_{h}", f"humidity_fc_{h}", f"pressure_fc_{h}",
        f"wind_fc_{h}", f"precip_fc_{h}",
        f"wind_mean_next_{h}", f"wind_max_next_{h}",
        f"precip_sum_next_{h}", f"temp_mean_next_{h}",
        f"temp_delta_{h}", f"pressure_delta_{h}",
    ]


def feature_columns(horizon):
    return BASE_FEATURES + forecast_features(horizon)


def make_model():
    return XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )


def load_training_data():
    project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY, project=HOPSWORKS_PROJECT)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features_history", version=FEATURE_GROUP_VERSION)
    df = fg.read().sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows  |  {df['timestamp'].min().date()} -> "
          f"{df['timestamp'].max().date()}")

    # Fail loudly if the v7 backfill has not run — a missing column here would
    # otherwise surface as a confusing KeyError deep in the fold loop.
    missing = [c for h in HORIZONS for c in forecast_features(h) if c not in df.columns]
    if missing:
        raise KeyError(
            f"{len(missing)} forecast column(s) absent from feature group "
            f"v{FEATURE_GROUP_VERSION}, e.g. {missing[:4]}. "
            "Run backfill_features_v7.py first."
        )
    return df, project


def build_target(df, horizon):
    df = df.copy()
    df["target_aqi"] = df[TARGET_COLUMN].shift(-horizon)
    return df.dropna(subset=["target_aqi"]).reset_index(drop=True)


def evaluate(y_true, y_pred):
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def persistence_rmse(test_df):
    """'AQI in H hours == AQI now.' Kept as a single sanity number: if the
    model cannot beat this, the new features did not help, whatever R2 says."""
    return float(mean_squared_error(test_df["target_aqi"],
                                    test_df[TARGET_COLUMN]) ** 0.5)


def cross_validate(df, horizon):
    feats = feature_columns(horizon)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=horizon)
    fold_metrics, persist = [], []
    last_test_X = None

    for tr_idx, te_idx in tscv.split(df):
        train_df, test_df = df.iloc[tr_idx], df.iloc[te_idx]
        model = make_model()
        model.fit(train_df[feats], train_df["target_aqi"])
        fold_metrics.append(evaluate(test_df["target_aqi"],
                                     model.predict(test_df[feats])))
        persist.append(persistence_rmse(test_df))
        last_test_X = test_df[feats]

    return fold_metrics, float(np.mean(persist)), last_test_X


def mean_metrics(fold_metrics):
    return {k: float(np.mean([m[k] for m in fold_metrics]))
            for k in ("rmse", "mae", "r2")}


def shape(model, X_sample, out_path="shap_summary_v7.png"):
    shap_values = shap.TreeExplainer(model).shap_values(X_sample)
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to {out_path}")


def save_and_register(project, models_by_horizon, metrics):
    os.makedirs("model_dir", exist_ok=True)
    for horizon, model in models_by_horizon.items():
        joblib.dump(model, os.path.join("model_dir", f"model_h{horizon}.pkl"))
    joblib.dump({
        "base_features": BASE_FEATURES,
        "feature_columns": {h: feature_columns(h) for h in HORIZONS},
        "horizons": HORIZONS,
        "target": TARGET_COLUMN,
        "model": "xgboost",
        "feature_group_version": FEATURE_GROUP_VERSION,
        "note": ("Forward weather features trained on ERA5 reanalysis; live "
                 "inference must supply Open-Meteo forecast values in the same "
                 "column order. CV metrics are an upper bound."),
    }, os.path.join("model_dir", "metadata.pkl"))

    mr = project.get_model_registry()
    aqi_model = mr.python.create_model(
        name="aqi_forecast_model",
        metrics=metrics,
        description=("AQI 3-day forecast (XGBoost) with weather forecast features. "
                     f"US-EPA 0-500 from 24h-average PM2.5, horizons {HORIZONS}h, "
                     f"{N_SPLITS}-fold rolling-origin CV."),
    )
    aqi_model.save("model_dir")
    print("Model registered in Hopsworks Model Registry as 'aqi_forecast_model'.")


if __name__ == "__main__":
    raw_df, project = load_training_data()

    results, models_by_horizon = {}, {}
    primary_test_X = None

    for horizon in HORIZONS:
        feats = feature_columns(horizon)
        print(f"Training +{horizon}h ... ({len(feats)} features: "
              f"{len(BASE_FEATURES)} base + {len(forecast_features(horizon))} forecast)")

        df = build_target(raw_df, horizon)
        fold_metrics, persist_rmse, last_test_X = cross_validate(df, horizon)
        avg = mean_metrics(fold_metrics)
        avg["persistence_rmse"] = persist_rmse
        results[horizon] = avg

        final = make_model()
        final.fit(df[feats], df["target_aqi"])
        models_by_horizon[horizon] = final

        if horizon == PRIMARY_HORIZON:
            primary_test_X = last_test_X

    print("\n" + "=" * 62)
    print(f"FINAL RESULTS  (mean over {N_SPLITS} folds)")
    print("=" * 62)
    for horizon in HORIZONS:
        r = results[horizon]
        print(f"  +{horizon}h    RMSE = {r['rmse']:7.3f}    "
              f"MAE = {r['mae']:7.3f}    R2 = {r['r2']:6.3f}")

    print("\n  vs v6 (no forecast features):  R2 was "
          "0.559 / 0.190 / 0.086 at 24/48/72h")
    print("  vs persistence baseline (RMSE):")
    for horizon in HORIZONS:
        r = results[horizon]
        print(f"    +{horizon}h  model {r['rmse']:.2f}  "
              f"persistence {r['persistence_rmse']:.2f}")

    pd.DataFrame([{"horizon": h, **m} for h, m in results.items()]
                 ).to_csv("model_results_v7.csv", index=False)
    print("\nResults written to model_results_v7.csv")

    shape(models_by_horizon[PRIMARY_HORIZON],
          primary_test_X.sample(min(200, len(primary_test_X)),
                                random_state=42))

    save_and_register(project, models_by_horizon,
                      {k: v for k, v in results[PRIMARY_HORIZON].items()
                       if k in ("rmse", "mae", "r2")})