#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


TRAIN_USERS = {1, 2, 3, 5, 6, 8, 9, 10}
TEST_USERS = {4, 7}

DATA_DIR = Path("../data_collection")
OUTPUT_DIR = Path("output")

TARGET_COL = "cpu_usage"

# Keep only useful/contributing features for CPU utilization.
FEATURE_COLS = [
    "request_rate",
    "replicas",
    "configured_replicas",
    "cpu_limit",
    "cpu_request",
    "memory_limit",
    "memory_request",
    "experiment_users",
    "request_rate_lag_1",
    "request_rate_lag_2",
    "cpu_usage_lag_1",
    "cpu_usage_lag_2",
]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def parse_user_from_filename(path: Path) -> int | None:
    # Expected format: taskX_<users>_<replicas>.csv
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def load_data(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    frames = []
    for file_path in files:
        user = parse_user_from_filename(file_path)
        if user is None:
            continue

        df = pd.read_csv(file_path)
        parts = file_path.stem.split("_")

        configured_replicas = np.nan
        if len(parts) >= 3:
            try:
                configured_replicas = int(parts[2])
            except ValueError:
                pass

        df["experiment_users"] = user
        df["configured_replicas"] = configured_replicas
        df["source_file"] = file_path.name
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid CSV files matched expected naming pattern.")

    data = pd.concat(frames, ignore_index=True)

    numeric_cols = [
        "request_rate",
        "replicas",
        "cpu_limit",
        "cpu_request",
        "memory_limit",
        "memory_request",
        "cpu_usage",
        "experiment_users",
        "configured_replicas",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Minimal lag features for CPU prediction.
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    data["request_rate_lag_1"] = data.groupby("source_file")["request_rate"].shift(1)
    data["request_rate_lag_2"] = data.groupby("source_file")["request_rate"].shift(2)
    data["cpu_usage_lag_1"] = data.groupby("source_file")["cpu_usage"].shift(1)
    data["cpu_usage_lag_2"] = data.groupby("source_file")["cpu_usage"].shift(2)

    return data


def train_and_evaluate(data: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL, "experiment_users"]).copy()

    train_df = data[data["experiment_users"].isin(TRAIN_USERS)].copy()
    test_df = data[data["experiment_users"].isin(TEST_USERS)].copy()

    if train_df.empty:
        raise RuntimeError("Training set is empty for requested TRAIN_USERS.")
    if test_df.empty:
        raise RuntimeError("Test set is empty for requested TEST_USERS.")

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    model.fit(X_train, y_train)

    # Case 1: users 4/7 (within training range)
    pred_case1 = model.predict(X_test)
    mae_case1 = float(mean_absolute_error(y_test, pred_case1))
    mape_case1 = mape(y_test, pred_case1)

    # Case 2: simulated unseen higher user intensity (user=15)
    # We alter user count and traffic-related features to mimic extrapolation.
    X_case2 = X_test.copy()
    avg_test_user = float(np.maximum(X_test["experiment_users"].mean(), 1.0))
    intensity_scale = 15.0 / avg_test_user
    X_case2["experiment_users"] = 15
    for col in ["request_rate", "request_rate_lag_1", "request_rate_lag_2"]:
        X_case2[col] = X_case2[col] * intensity_scale
    pred_case2 = model.predict(X_case2)
    mae_case2 = float(mean_absolute_error(y_test, pred_case2))
    mape_case2 = mape(y_test, pred_case2)

    results = {
        "target": TARGET_COL,
        "train_users": sorted(TRAIN_USERS),
        "test_users": sorted(TEST_USERS),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "metrics": {
            "case1_users_4_7": {"mae": mae_case1, "mape": mape_case1},
            "case2_simulated_user_15": {"mae": mae_case2, "mape": mape_case2},
        },
        "note": "Case 2 uses simulated user=15 feature values because no user-15 CSV is currently loaded.",
    }

    pred_out = test_df[["timestamp", "service", "source_file", "experiment_users", TARGET_COL]].copy()
    pred_out["pred_case1"] = pred_case1
    pred_out["pred_case2_user15"] = pred_case2
    pred_out.to_csv(OUTPUT_DIR / "cpu_predictions_test.csv", index=False)

    with open(OUTPUT_DIR / "cpu_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Training complete (simple regression mode).")
    print(f"Train users: {sorted(TRAIN_USERS)}")
    print(f"Test users: {sorted(TEST_USERS)}")
    print(f"Case 1 (users 4/7)    -> MAE: {mae_case1:.6f}, MAPE: {mape_case1:.4f}%")
    print(f"Case 2 (sim user=15)  -> MAE: {mae_case2:.6f}, MAPE: {mape_case2:.4f}%")
    print(f"Saved: {(OUTPUT_DIR / 'cpu_eval_summary.json').resolve()}")
    print(f"Saved: {(OUTPUT_DIR / 'cpu_predictions_test.csv').resolve()}")


def main() -> None:
    data = load_data(DATA_DIR)
    train_and_evaluate(data)


if __name__ == "__main__":
    main()
