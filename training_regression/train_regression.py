from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

RANDOM_STATE = 42

TRAIN_USERS = {1, 2, 3, 5, 6, 8, 9, 10}
TEST_USERS = {4, 7}

DATA_DIR = Path("../data_collection")
OUTPUT_DIR = Path("output")

TARGET_COL = "cpu_usage"


FEATURE_COLS = [
    "request_rate",
    "request_rate_lag_1",
    "request_rate_lag_2",
    "cpu_usage_lag_1",
    "cpu_usage_lag_2",
]


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def round_float(x: float, ndigits: int = 2) -> float:
    return float(np.round(x, ndigits))


def parse_user_from_filename(path: Path) -> int | None:
    parts = path.stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


# Loading all the csv files, parsing out the user from the filename,
# and concatenating into a single DataFrame.
def load_data(data_dir: Path) -> pd.DataFrame:
    print("[1/6] Loading CSV files...")
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    frames = []
    for file_path in files:
        user = parse_user_from_filename(file_path)
        if user is None:
            continue

        df = pd.read_csv(file_path)
        df["concurrent_users"] = user
        df["source_file"] = file_path.name
        frames.append(df)
    
    if not frames:
        raise RuntimeError("No valid CSV files matched expected naming pattern.")

    data = pd.concat(frames, ignore_index=True)
    print(f"  Loaded rows: {len(data)} from {len(frames)} files")

    # Numeric coercion: convert invalid entries to NaN (e.g., "invalid" in request_rate).
    for col in ["request_rate", "cpu_usage"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # Timestamp coercion: convert invalid timestamps to NaT (NaN for datetime).
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data


def preprocess_and_feature_engineer(data: pd.DataFrame) -> pd.DataFrame:
    print("[2/6] Preprocessing + feature engineering...")
    before = len(data)
    
    # Drop completely empty rows (e.g., trailing ",,,,,").
    data = data.dropna(how="all").copy()
    dropped_all_empty = before - len(data)
    if dropped_all_empty > 0:
        print(f"  Dropped fully empty rows: {dropped_all_empty}")
    before = len(data)

    # Drop invalid timestamp/target rows.
    data = data.dropna(subset=["timestamp", TARGET_COL]).copy()
    dropped = before - len(data)
    print(f"  Dropped rows with invalid timestamp/target: {dropped}")

    # Handle likely invalid shutdown/collector rows where all core metrics are zero.
    # Strategy: keep rows, convert these zero blocks to NaN, then fill reasonably.
    zero_cols = [
        "cpu_limit",
        "cpu_request",
        "memory_limit",
        "memory_request",
        "request_rate",
        "throughput",
        "cpu_usage",
    ]
    available_zero_cols = [c for c in zero_cols if c in data.columns]
    if available_zero_cols:
        zero_mask = (data[available_zero_cols].fillna(0.0) == 0.0).all(axis=1)
        zero_rows = int(zero_mask.sum())
        print(f"  Found all-zero metric rows: {zero_rows}")

        if zero_rows > 0:
            # Convert only the suspicious all-zero rows to NaN for selected cols.
            data.loc[zero_mask, available_zero_cols] = np.nan

            # Fill within each source file first (temporal continuity).
            data[available_zero_cols] = (
                data.groupby("source_file", group_keys=False)[available_zero_cols]
                .apply(lambda g: g.ffill().bfill())
            )

            # Global fallback for any remaining NaN (e.g., full-file zero blocks).
            global_medians = data[available_zero_cols].median(numeric_only=True)
            data[available_zero_cols] = data[available_zero_cols].fillna(global_medians)
            print("  Filled all-zero rows using per-file ffill/bfill + global medians")

    # Sort by time inside each source file, then create lag features.
    data = data.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    # Lag definitions:
    # - lag_1 = value observed one 2-minute interval earlier.
    # - lag_2 = value observed two 2-minute intervals earlier.
    # Why we add these:
    # CPU usage responds with short temporal dependency; recent traffic/CPU
    # values improve one-step-ahead prediction versus using only current traffic.
    data["request_rate_lag_1"] = data.groupby("source_file")["request_rate"].shift(1)
    data["request_rate_lag_2"] = data.groupby("source_file")["request_rate"].shift(2)
    data["cpu_usage_lag_1"] = data.groupby("source_file")["cpu_usage"].shift(1)
    data["cpu_usage_lag_2"] = data.groupby("source_file")["cpu_usage"].shift(2)

    # Drop unnecessary columns early to keep the pipeline clean and focused.
    keep_cols = [
        "timestamp",
        "service",
        "source_file",
        "concurrent_users",
        TARGET_COL,
        *FEATURE_COLS,
    ]
    data = data[[c for c in keep_cols if c in data.columns]].copy()

    print(f"\n\nColumns kept for modeling:\n{data.columns.tolist()}\n")
    return data


def analyze_data(data: pd.DataFrame) -> None:
    print("[3/7] Data stats + EDA plots...")
    eda_dir = OUTPUT_DIR / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Total rows: {len(data)}")
    print("  Missing values per selected column:")
    selected = FEATURE_COLS + [TARGET_COL, "concurrent_users"]
    miss = data[selected].isna().sum().to_dict()
    for k, v in miss.items():
        print(f"    - {k}: {v}")

    desc = data[selected].describe(include="all").transpose().round(2)
    desc.to_csv(eda_dir / "summary_stats.csv")
    print(f"  Saved summary stats: {(eda_dir / 'summary_stats.csv').resolve()}")

    corr_cols = FEATURE_COLS + [TARGET_COL, "concurrent_users"]
    corr_df = data[corr_cols].copy()
    corr = corr_df.corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap (Features + cpu_usage)")
    plt.tight_layout()
    plt.savefig(eda_dir / "correlation_heatmap.png", dpi=160)
    plt.close()
    print(f"  Saved plot: {(eda_dir / 'correlation_heatmap.png').resolve()}")

    plt.figure(figsize=(8, 4))
    ax = sns.histplot(data[TARGET_COL], bins=30, kde=True, stat="count")
    plt.title("cpu_usage Distribution")
    plt.ylabel("Count")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{int(h)}", (p.get_x() + p.get_width() / 2, h), ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(eda_dir / "cpu_usage_distribution.png", dpi=160)
    plt.close()
    print(f"  Saved plot: {(eda_dir / 'cpu_usage_distribution.png').resolve()}")

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=data, x="request_rate", y="cpu_usage", alpha=0.5)
    plt.title("request_rate vs cpu_usage")
    plt.tight_layout()
    plt.savefig(eda_dir / "request_rate_vs_cpu_usage.png", dpi=160)
    plt.close()
    print(f"  Saved plot: {(eda_dir / 'request_rate_vs_cpu_usage.png').resolve()}")


def plot_model_comparison(metrics_df: pd.DataFrame) -> None:
    charts_dir = OUTPUT_DIR / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_df = metrics_df.copy()
    plot_df[["mae", "mape"]] = plot_df[["mae", "mape"]].round(2)
    for metric in ["mae", "mape"]:
        plt.figure(figsize=(9, 4))
        sns.barplot(data=plot_df, x="model", y=metric, hue="case")
        plt.title(f"Model Comparison: {metric.upper()} by Case")
        plt.tight_layout()
        plt.savefig(charts_dir / f"model_comparison_{metric}.png", dpi=160)
        plt.close()

    print(f"  Saved comparison charts under: {charts_dir.resolve()}")


def infer_saved_models_on_new_points(model_dir: Path) -> None:
    
    print("[9/9] Running saved-model inference on hardcoded new points...")

    # unseen-user points (not users 4 or 7) for simple demo inference.
    new_points = pd.DataFrame(
        [
            {
                "scenario": "new_user11_normal",
                "concurrent_users": 11,
                "request_rate": 4.00,
                "request_rate_lag_1": 3.90,
                "request_rate_lag_2": 3.80,
                "cpu_usage_lag_1": 0.55,
                "cpu_usage_lag_2": 0.52,
            },
            {
                "scenario": "new_user12_medium",
                "concurrent_users": 12,
                "request_rate": 7.00,
                "request_rate_lag_1": 6.80,
                "request_rate_lag_2": 6.60,
                "cpu_usage_lag_1": 0.95,
                "cpu_usage_lag_2": 0.90,
            },
            {
                "scenario": "new_user15_high",
                "concurrent_users": 15,
                "request_rate": 12.00,
                "request_rate_lag_1": 11.70,
                "request_rate_lag_2": 11.20,
                "cpu_usage_lag_1": 1.60,
                "cpu_usage_lag_2": 1.52,
            },
        ]
    )

    infer_rows = []
    for model_path in sorted(model_dir.glob("*.joblib")):
        model_name = model_path.stem
        loaded_model = joblib.load(model_path)
        y_new = loaded_model.predict(new_points[FEATURE_COLS])
        for i, pred in enumerate(y_new):
            infer_rows.append(
                {
                    "model": model_name,
                    "scenario": new_points.iloc[i]["scenario"],
                    "concurrent_users": int(new_points.iloc[i]["concurrent_users"]),
                    "predicted_cpu_usage": round_float(float(pred)),
                }
            )

    new_pred_df = pd.DataFrame(infer_rows)
    new_pred_df.to_csv(OUTPUT_DIR / "new_point_predictions.csv", index=False)
    with open(OUTPUT_DIR / "new_point_predictions.json", "w", encoding="utf-8") as f:
        json.dump(new_pred_df.to_dict(orient="records"), f, indent=2)

    print(f"Saved: { 'new_point_predictions.csv'}")


def train_and_evaluate(data: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[4/8] Train/test split by users...")
    train_df = data[data["concurrent_users"].isin(TRAIN_USERS)].copy()
    test_df = data[data["concurrent_users"].isin(TEST_USERS)].copy()
    print(f"  Train users: {sorted(TRAIN_USERS)} -> rows: {len(train_df)}")
    print(f"  Test users:  {sorted(TEST_USERS)} -> rows: {len(test_df)}")

    if train_df.empty:
        raise RuntimeError("Training set is empty for requested TRAIN_USERS.")
    if test_df.empty:
        raise RuntimeError("Test set is empty for requested TEST_USERS.")

    print("[5/8] Missing-value handling (drop/fill)...")

    # Dropping rows missing core input signal in each split.
    train_before, test_before = len(train_df), len(test_df)
    train_df = train_df.dropna(subset=["request_rate"]).copy()
    test_df = test_df.dropna(subset=["request_rate"]).copy()
    print(f"  Dropped from train (missing request_rate): {train_before - len(train_df)}")
    print(f"  Dropped from test  (missing request_rate): {test_before - len(test_df)}")

    # Filling remaining feature missing values using train medians : Only the numeric features used for modeling
    train_medians = train_df[FEATURE_COLS].median(numeric_only=True)
    train_missing_before = int(train_df[FEATURE_COLS].isna().sum().sum())
    
    test_missing_before = int(test_df[FEATURE_COLS].isna().sum().sum())

    train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(train_medians)
    test_df[FEATURE_COLS] = test_df[FEATURE_COLS].fillna(train_medians)
    
    train_missing_after = int(train_df[FEATURE_COLS].isna().sum().sum())
    test_missing_after = int(test_df[FEATURE_COLS].isna().sum().sum())

    print(f"  Train missing features: {train_missing_before} -> {train_missing_after}")
    print(f"  Test missing features:  {test_missing_before} -> {test_missing_after}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    # Case 2 uses scaled traffic to simulate unseen higher load (user ~15).
    X_case2 = X_test.copy()
    avg_test_user = float(np.maximum(test_df["concurrent_users"].mean(), 1.0))
    intensity_scale = 15.0 / avg_test_user
    for col in ["request_rate", "request_rate_lag_1", "request_rate_lag_2"]:
        X_case2[col] = X_case2[col] * intensity_scale

    print("[6/8] Training models and evaluating...")
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    metric_rows: list[dict] = []
    trained_models: dict[str, object] = {}
    pred_out = test_df[["timestamp", "service", "source_file", "concurrent_users", TARGET_COL]].copy()

    for model_name, model in models.items():
        print(f"  Training: {model_name}")
        model.fit(X_train, y_train)
        trained_models[model_name] = model

        pred_case1 = model.predict(X_test)
        mae_case1 = round_float(mean_absolute_error(y_test, pred_case1))
        mape_case1 = round_float(mape(y_test, pred_case1))

        pred_case2 = model.predict(X_case2)
        mae_case2 = round_float(mean_absolute_error(y_test, pred_case2))
        mape_case2 = round_float(mape(y_test, pred_case2))

        metric_rows.append(
            {"model": model_name, "case": "case1_users_4_7", "mae": mae_case1, "mape": mape_case1}
        )
        metric_rows.append(
            {
                "model": model_name,
                "case": "case2_simulated_user15_traffic_scale",
                "mae": mae_case2,
                "mape": mape_case2,
            }
        )

        pred_out[f"pred_case1_{model_name}"] = np.round(pred_case1, 2)
        pred_out[f"pred_case2_{model_name}"] = np.round(pred_case2, 2)

    metrics_df = pd.DataFrame(metric_rows).round(2)
    best_case1 = (
        metrics_df[metrics_df["case"] == "case1_users_4_7"]
        .sort_values(["mae", "mape"], ascending=[True, True])
        .iloc[0]
    )

    print("[7/8] Saving outputs...")
    metrics_by_case = {
        case: {
            row["model"]: {"mae": float(row["mae"]), "mape": float(row["mape"])}
            for _, row in metrics_df[metrics_df["case"] == case].iterrows()
        }
        for case in metrics_df["case"].unique()
    }

    print("\nEvaluation summary:")
    results = {
        "target": TARGET_COL,
        "feature_cols": FEATURE_COLS,
        "train_users": sorted(TRAIN_USERS),
        "test_users": sorted(TEST_USERS),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "models_tested": list(models.keys()),
        "metrics": metrics_by_case,
        "best_model_case1_by_mae": str(best_case1["model"]),
        "note": "Case 2 simulates user=15 by scaling traffic features because user-15 CSV is not loaded.",
    }

    # Round numeric columns in predictions output for readability.
    num_cols = pred_out.select_dtypes(include=[np.number]).columns
    pred_out[num_cols] = pred_out[num_cols].round(2)
    pred_out.to_csv(OUTPUT_DIR / "cpu_predictions_test.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)
    plot_model_comparison(metrics_df)

    # Save each trained model for later inference.
    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for model_name, model in trained_models.items():
        joblib.dump(model, model_dir / f"{model_name}.joblib")

    with open(OUTPUT_DIR / "cpu_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("[8/8] Done.")
    print("Training complete (simple regression mode).")
    print(metrics_df.to_string(index=False))
    print(f"Saved: { 'cpu_eval_summary.json'}")
    print(f"Saved: { 'cpu_predictions_test.csv'}")
    print(f"Saved: { 'model_comparison_metrics.csv'}")
    print(f"Saved models: {model_dir.resolve()}")
    return model_dir


def main() -> None:
    data = load_data(DATA_DIR)
    data = preprocess_and_feature_engineer(data)
    analyze_data(data)
    model_dir = train_and_evaluate(data)
    infer_saved_models_on_new_points(model_dir)


if __name__ == "__main__":
    main()
