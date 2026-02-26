#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


TARGET_COLS = ["avg_latency", "throughput", "cpu_usage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression-only pipeline for service performance prediction.")
    parser.add_argument("--data-dir", default="data_collection", help="Directory containing CSV files.")
    parser.add_argument("--output-dir", default="training_regression/output_regression", help="Output directory.")
    parser.add_argument("--config-test-size", type=float, default=0.25, help="Held-out config ratio.")
    parser.add_argument("--interp-test-size", type=float, default=0.2, help="Interpolation test ratio.")
    parser.add_argument("--extrap-quantile", type=float, default=0.8, help="Quantile threshold for extrapolation test.")
    return parser.parse_args()


def load_dataset(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    frames = []
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        parts = file_path.stem.split("_")

        service = parts[0] if len(parts) >= 1 else "unknown"
        users = np.nan
        configured_replicas = np.nan
        run_id = np.nan

        if len(parts) >= 3:
            try:
                users = int(parts[1])
                configured_replicas = int(parts[2])
            except ValueError:
                pass

        if len(parts) >= 4:
            try:
                run_id = int(parts[3])
            except ValueError:
                pass

        df["service_from_file"] = service
        df["experiment_users"] = users
        df["configured_replicas"] = configured_replicas
        df["run_id"] = run_id
        df["source_file"] = file_path.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy().drop_duplicates().reset_index(drop=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"]).copy()

    numeric_cols = [
        "replicas",
        "cpu_limit",
        "cpu_request",
        "memory_limit",
        "memory_request",
        "request_rate",
        "p50_latency",
        "p95_latency",
        "p99_latency",
        "avg_latency",
        "throughput",
        "cpu_usage",
        "experiment_users",
        "configured_replicas",
        "run_id",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["service"] = data["service"].fillna(data["service_from_file"]).astype(str)
    data["endpoint"] = data["endpoint"].fillna("unknown").astype(str)

    minute_of_day = data["timestamp"].dt.hour * 60 + data["timestamp"].dt.minute
    day_of_week = data["timestamp"].dt.dayofweek

    # Fourier features for daily + weekly periodicity.
    data["fourier_day_sin_1"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    data["fourier_day_cos_1"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    data["fourier_day_sin_2"] = np.sin(4 * np.pi * minute_of_day / 1440.0)
    data["fourier_day_cos_2"] = np.cos(4 * np.pi * minute_of_day / 1440.0)
    data["fourier_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    data["fourier_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    data = data.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    # Lag features and rolling statistics from traffic/system history.
    lag_base = ["request_rate", "avg_latency", "throughput", "cpu_usage"]
    for feat in lag_base:
        for lag in [1, 2, 3]:
            data[f"{feat}_lag_{lag}"] = data.groupby("source_file")[feat].shift(lag)

    roll_base = ["request_rate", "throughput", "cpu_usage"]
    for feat in roll_base:
        shifted = data.groupby("source_file")[feat].shift(1)
        data[f"{feat}_roll_mean_3"] = shifted.groupby(data["source_file"]).rolling(window=3).mean().reset_index(level=0, drop=True)
        data[f"{feat}_roll_std_3"] = shifted.groupby(data["source_file"]).rolling(window=3).std().reset_index(level=0, drop=True)

    data["config_signature"] = (
        data["service"].astype(str)
        + "|rep="
        + data["configured_replicas"].astype(str)
        + "|cpu="
        + data["cpu_limit"].astype(str)
        + "|mem="
        + data["memory_limit"].astype(str)
    )

    # Load scenario labels for phase-4 analysis.
    q_low = data["request_rate"].quantile(0.33)
    q_high = data["request_rate"].quantile(0.66)
    data["load_scenario"] = pd.cut(
        data["request_rate"],
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["low", "medium", "high"],
    ).astype(str)

    return data


def choose_feature_columns(data: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    traffic_features = [
        "request_rate",
        "request_rate_lag_1",
        "request_rate_lag_2",
        "request_rate_lag_3",
        "avg_latency_lag_1",
        "avg_latency_lag_2",
        "avg_latency_lag_3",
        "throughput_lag_1",
        "throughput_lag_2",
        "throughput_lag_3",
        "cpu_usage_lag_1",
        "cpu_usage_lag_2",
        "cpu_usage_lag_3",
        "request_rate_roll_mean_3",
        "request_rate_roll_std_3",
        "throughput_roll_mean_3",
        "throughput_roll_std_3",
        "cpu_usage_roll_mean_3",
        "cpu_usage_roll_std_3",
        "fourier_day_sin_1",
        "fourier_day_cos_1",
        "fourier_day_sin_2",
        "fourier_day_cos_2",
        "fourier_week_sin",
        "fourier_week_cos",
    ]

    config_features = [
        "replicas",
        "configured_replicas",
        "cpu_limit",
        "cpu_request",
        "memory_limit",
        "memory_request",
        "experiment_users",
    ]

    categorical_features = ["service"]

    required = set(traffic_features + config_features + categorical_features + TARGET_COLS + ["config_signature", "request_rate", "load_scenario"]) 
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return traffic_features, config_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def within_tolerance_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol_pct: float = 10.0) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    pct_err = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.mean(pct_err <= tol_pct) * 100.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, targets: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    maes, mapes, r2s = [], [], []
    for i, t in enumerate(targets):
        mae_i = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape_i = mape(y_true[:, i], y_pred[:, i])
        r2_i = r2_score(y_true[:, i], y_pred[:, i])
        acc_i = within_tolerance_accuracy(y_true[:, i], y_pred[:, i], tol_pct=10.0)

        out[f"{t}_mae"] = float(mae_i)
        out[f"{t}_mape"] = float(mape_i)
        out[f"{t}_r2"] = float(r2_i)
        out[f"{t}_acc_10pct"] = float(acc_i)

        maes.append(mae_i)
        mapes.append(mape_i)
        r2s.append(r2_i)

    out["mae_mean"] = float(np.mean(maes))
    out["mape_mean"] = float(np.mean(mapes))
    out["r2_mean"] = float(np.mean(r2s))
    return out


def split_config_generalization(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_train, idx_test = next(gss.split(df, groups=df["config_signature"]))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


def split_interpolation(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Unseen traffic patterns within distribution: central quantile band.
    q10 = df["request_rate"].quantile(0.10)
    q90 = df["request_rate"].quantile(0.90)
    pool = df[(df["request_rate"] >= q10) & (df["request_rate"] <= q90)].copy()

    if len(pool) < 20:
        pool = df.copy()

    test = pool.sample(frac=test_size, random_state=RANDOM_STATE)
    train = df.drop(index=test.index)
    return train.copy(), test.copy()


def split_extrapolation(df: pd.DataFrame, quantile_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Train on lower traffic intensity, test on higher unseen intensity region.
    thr = df["request_rate"].quantile(quantile_threshold)
    train = df[df["request_rate"] <= thr].copy()
    test = df[df["request_rate"] > thr].copy()

    if len(test) < 20:
        thr = df["request_rate"].quantile(0.75)
        train = df[df["request_rate"] <= thr].copy()
        test = df[df["request_rate"] > thr].copy()

    return train, test


def train_and_predict(
    model_name: str,
    estimator,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, Pipeline]:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLS].to_numpy(dtype=float)
    X_test = test_df[feature_cols]

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    return pred, pipeline


def evaluate_load_scenarios(test_df: pd.DataFrame, y_pred: np.ndarray, model_name: str, scenario_name: str) -> pd.DataFrame:
    rows = []
    y_true = test_df[TARGET_COLS].to_numpy(dtype=float)

    for load_name in ["low", "medium", "high"]:
        mask = test_df["load_scenario"].astype(str) == load_name
        if mask.sum() == 0:
            continue

        yt = y_true[mask.to_numpy()]
        yp = y_pred[mask.to_numpy()]
        metrics = evaluate(yt, yp, TARGET_COLS)

        for t in TARGET_COLS:
            rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "load_scenario": load_name,
                    "target": t,
                    "mae": metrics[f"{t}_mae"],
                    "mape": metrics[f"{t}_mape"],
                    "r2": metrics[f"{t}_r2"],
                    "acc_10pct": metrics[f"{t}_acc_10pct"],
                    "n_samples": int(mask.sum()),
                }
            )

    return pd.DataFrame(rows)


def plot_core_eda(df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, t in enumerate(TARGET_COLS):
        sns.histplot(df[t], bins=30, kde=True, ax=axes[i])
        axes[i].set_title(f"Distribution: {t}")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distributions.png", dpi=160)
    plt.close(fig)

    num_df = df.select_dtypes(include=["number"]).copy()
    corr = num_df.corr(numeric_only=True)
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(plots_dir / "correlation_heatmap.png", dpi=160)
    plt.close()


def plot_phase4_summary(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    view = metrics_df[["scenario", "model", "mae_mean", "mape_mean", "r2_mean"]].drop_duplicates()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=view, x="scenario", y="mae_mean", hue="model")
    plt.title("MAE (mean across targets) by Scenario")
    plt.tight_layout()
    plt.savefig(plots_dir / "phase4_mae_by_scenario.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=view, x="scenario", y="mape_mean", hue="model")
    plt.title("MAPE (mean across targets) by Scenario")
    plt.tight_layout()
    plt.savefig(plots_dir / "phase4_mape_by_scenario.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=view, x="scenario", y="r2_mean", hue="model")
    plt.title("R2 (mean across targets) by Scenario")
    plt.tight_layout()
    plt.savefig(plots_dir / "phase4_r2_by_scenario.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset(Path(args.data_dir))
    data = clean_and_engineer(raw)

    traffic_features, config_features, categorical_features = choose_feature_columns(data)
    feature_cols = traffic_features + config_features + categorical_features
    numeric_cols = traffic_features + config_features

    data = data.dropna(subset=feature_cols + TARGET_COLS).copy()

    plot_core_eda(data, out_dir)

    # Phase 4 scenarios.
    cfg_train, cfg_test = split_config_generalization(data, test_size=args.config_test_size)
    interp_train, interp_test = split_interpolation(cfg_train, test_size=args.interp_test_size)
    extrap_train, extrap_test = split_extrapolation(cfg_train, quantile_threshold=args.extrap_quantile)

    scenarios = {
        "configuration_generalization": (cfg_train, cfg_test),
        "interpolation": (interp_train, interp_test),
        "extrapolation": (extrap_train, extrap_test),
    }

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=400,
            random_state=RANDOM_STATE,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
    }

    metric_rows = []
    load_rows = []
    artifact_models: Dict[str, Pipeline] = {}

    for scenario_name, (train_df, test_df) in scenarios.items():
        if len(train_df) < 20 or len(test_df) < 10:
            continue

        y_true = test_df[TARGET_COLS].to_numpy(dtype=float)

        for model_name, estimator in models.items():
            pred, fitted = train_and_predict(
                model_name,
                estimator,
                train_df,
                test_df,
                feature_cols=feature_cols,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_features,
            )

            metrics = evaluate(y_true, pred, TARGET_COLS)
            metric_rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    **metrics,
                }
            )

            pred_df = test_df[["timestamp", "service", "source_file", "request_rate", "load_scenario", "config_signature"]].copy()
            for i, t in enumerate(TARGET_COLS):
                pred_df[f"actual_{t}"] = y_true[:, i]
                pred_df[f"pred_{t}"] = pred[:, i]
                pred_df[f"err_{t}"] = pred_df[f"actual_{t}"] - pred_df[f"pred_{t}"]
            pred_df.to_csv(out_dir / f"predictions_{scenario_name}_{model_name}.csv", index=False)

            load_df = evaluate_load_scenarios(test_df, pred, model_name, scenario_name)
            load_rows.append(load_df)

            if scenario_name == "configuration_generalization":
                artifact_models[model_name] = fitted

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        raise RuntimeError("No scenario produced enough train/test samples. Check split parameters.")

    load_metrics_df = pd.concat(load_rows, ignore_index=True) if load_rows else pd.DataFrame()

    # Select best regression model by configuration-generalization MAPE mean.
    cfg_eval = metrics_df[metrics_df["scenario"] == "configuration_generalization"].copy()
    if cfg_eval.empty:
        cfg_eval = metrics_df.copy()

    best_row = cfg_eval.sort_values(["mape_mean", "mae_mean"], ascending=[True, True]).iloc[0]
    best_model_name = str(best_row["model"])

    if best_model_name in artifact_models:
        joblib.dump(artifact_models[best_model_name], out_dir / "best_model.joblib")

    metrics_df.to_csv(out_dir / "phase4_metrics_summary.csv", index=False)
    if not load_metrics_df.empty:
        load_metrics_df.to_csv(out_dir / "phase4_load_scenario_metrics.csv", index=False)

    plot_phase4_summary(metrics_df, out_dir)

    summary = {
        "pipeline": "Regression only",
        "targets": TARGET_COLS,
        "features": {
            "traffic": traffic_features,
            "configuration": config_features,
            "categorical": categorical_features,
        },
        "splits": {
            "configuration_generalization": {
                "train_rows": int(len(cfg_train)),
                "test_rows": int(len(cfg_test)),
            },
            "interpolation": {
                "train_rows": int(len(interp_train)),
                "test_rows": int(len(interp_test)),
            },
            "extrapolation": {
                "train_rows": int(len(extrap_train)),
                "test_rows": int(len(extrap_test)),
                "extrapolation_quantile": float(args.extrap_quantile),
            },
        },
        "selected_model": best_model_name,
        "selection_basis": "Lowest MAPE mean on configuration-generalization scenario",
        "best_row": best_row.to_dict(),
    }

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Regression pipeline complete.")
    print(f"Selected model: {best_model_name}")
    print(metrics_df.to_string(index=False))
    print(f"Artifacts: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
