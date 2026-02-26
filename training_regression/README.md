# Regression Pipeline (Phase 3 + Phase 4)

This implementation is **regression-only** (no deep learning yet), focused on your current step.

## What is implemented

### Phase 3 (current scope)
- Parametric conditioning on configuration variables:
  - `replicas`, `configured_replicas`, `cpu_limit`, `cpu_request`, `memory_limit`, `memory_request`, `experiment_users`
- Feature engineering:
  - lag features (1,2,3)
  - rolling statistics (mean/std over 3-step window)
  - Fourier periodicity features (daily and weekly sin/cos)
- Model input:
  - traffic features + configuration parameters (+ service categorical encoding)
- Model output (multi-target regression):
  - `avg_latency` (response time)
  - `throughput`
  - `cpu_usage`
- Models:
  - `LinearRegression`
  - `RandomForestRegressor`

### Phase 4 validation/evaluation
- Interpolation test:
  - evaluate on unseen traffic samples inside the central traffic distribution
- Extrapolation test:
  - train on lower traffic quantiles, test on higher-intensity traffic
- Configuration generalization:
  - train/test split by held-out configuration groups (`config_signature`)
- Metrics:
  - MAE
  - MAPE
  - R²
  - accuracy within 10% error under `low/medium/high` load scenarios

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r training_regression/requirements.txt
```

## Run
From repo root:
```bash
python3 training_regression/train_regression.py \
  --data-dir data_collection \
  --output-dir training_regression/output_regression
```

Optional split tuning:
```bash
python3 training_regression/train_regression.py \
  --data-dir data_collection \
  --output-dir training_regression/output_regression \
  --config-test-size 0.25 \
  --interp-test-size 0.20 \
  --extrap-quantile 0.80
```

## Outputs
- `run_summary.json`
- `phase4_metrics_summary.csv`
- `phase4_load_scenario_metrics.csv`
- `best_model.joblib`
- per-scenario predictions:
  - `predictions_configuration_generalization_<model>.csv`
  - `predictions_interpolation_<model>.csv`
  - `predictions_extrapolation_<model>.csv`
- `plots/`:
  - `target_distributions.png`
  - `correlation_heatmap.png`
  - `phase4_mae_by_scenario.png`
  - `phase4_mape_by_scenario.png`
  - `phase4_r2_by_scenario.png`
