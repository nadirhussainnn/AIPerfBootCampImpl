# AI-Driven Performance Modeling for Microservices Systems

End-to-end workflow for collecting Kubernetes + Istio service metrics under load, and training a regression model to predict CPU utilization.

## Project Structure
- `kubernetes/` Kubernetes manifests for app and Istio resources
- `locust/` load generation scripts, workloads and long running script for data collection
- `data_collection/` Prometheus-based metric collection scripts and CSV dataset
- `training_regression/` regression training + evaluation + model inference

## 1) Environment Setup

### Prerequisites
- Docker Desktop
- Kubernetes CLI (`kubectl`)
- Python 3.10+
- `curl`

### Enable Kubernetes in Docker Desktop
1. Open Docker Desktop
2. Go to `Settings -> Kubernetes`
3. Enable Kubernetes and wait until ready
4. Verify:
```bash
kubectl get nodes
```
Expected:
```text
NAME             STATUS   ROLES           AGE   VERSION
docker-desktop   Ready    control-plane   ...   ...
```

## 2) Install Istio and Enable Telemetry

### Install Istio CLI and control plane
```bash
curl -L https://istio.io/downloadIstio | sh -
cd istio-1.29.0
export PATH=$PWD/bin:$PATH
istioctl version

istioctl install --set profile=demo -y
kubectl label namespace default istio-injection=enabled
kubectl get pods -n istio-system
```

### Deploy Prometheus/Grafana/Jaeger addons
```bash
kubectl apply -f samples/addons/prometheus.yaml
kubectl apply -f samples/addons/grafana.yaml
kubectl apply -f samples/addons/jaeger.yaml
kubectl get pods -n istio-system
```

### Open Istio dashboards
```bash
istioctl dashboard prometheus
istioctl dashboard grafana
istioctl dashboard jaeger
```

## 3) Enable Kubernetes Resource Metrics

### metrics-server (for `kubectl top`)
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch deployment metrics-server -n kube-system --type='json' -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl top pods
```

### kube-state-metrics
```bash
kubectl apply -k "github.com/kubernetes/kube-state-metrics/examples/standard?ref=v2.18.0"
kubectl -n kube-system rollout status deploy/kube-state-metrics --timeout=180s
kubectl get pods -A | grep kube-state-metrics
kubectl get svc -A | grep kube-state-metrics
```

Verify scrape annotations:
```bash
kubectl -n kube-system get svc kube-state-metrics -o yaml | grep -E "prometheus.io/scrape|prometheus.io/port"
```

If missing, add:
```bash
kubectl -n kube-system annotate svc kube-state-metrics prometheus.io/scrape="true" --overwrite
kubectl -n kube-system annotate svc kube-state-metrics prometheus.io/port="8080" --overwrite
```

## 4) Add Istio Telemetry for `request_path`

Create `kubernetes/istio/new_istio.yaml` with:

```yaml
apiVersion: telemetry.istio.io/v1
kind: Telemetry
metadata:
  name: add-request-path
  namespace: istio-system
spec:
  metrics:
  - providers:
    - name: prometheus
    overrides:
    - match:
        metric: REQUEST_COUNT
      tagOverrides:
        request_path:
          operation: UPSERT
          value: request.path
```

Apply and verify:
```bash
kubectl apply -f kubernetes/istio/new_istio.yaml
kubectl get telemetry -A
```

## 5) Deploy Application

From repo root:
```bash
kubectl apply -f kubernetes/apps/acmeair-light.yaml
kubectl apply -f kubernetes/istio/
kubectl get pods -n default
```

## 6) Load Generation (Locust)

### Interactive Locust UI
```bash
cd locust
export LOCUST_THINK_TIME=0.42
locust -f locustfile.py
```

### Gradual workload script

- Run configurable script, that generates locust gradual ramp up, for configurable users, replicas.Producing csv in `TaskX_<user>_<replica>.csv` format
```bash
python3 locust/run_gradual_workload.py --users-start 1 --users-end 5
```

- If script stopped for some reason, you can continue for remaining users, for specific replica even.
```bash
python3 locust/run_gradual_workload.py --only-user 1 --only-replica 1
```

- Or script runs for 35 minutes, from 6 to 10 users (replicas: 1,2,3,4), generating single csv:
```bash
python3 locust/script.py
```

### Trace-based headless load
- Only if you need to capture brusty or sinosoidal loads
```bash
LOCUST_TIME_LIMIT=800 \
LOCUST_WORKLOAD_CSV="workloads/sin400.csv" \
LOCUST_AMPLITUDE=19 \
LOCUST_SHIFT=1 \
LOCUST_THINK_TIME=0.1 \
LOCUST_ENTRYPOINT_PATH="/" \
LOCUST_FIT_TRACE="false" \
locust -f locustfile_trace.py --headless --csv=logs/trace_experiment
```
---
# Model Training Pipeline

This folder contains a simple and explainable regression workflow to predict `cpu_usage` from workload history.
It currently evaluates multiple models and compares them side-by-side.

## 1) Data Collection

### Collection strategy used
- Runtime per experiment: `40 minutes` (default in `data_collection/script.py`)
- Services collected: `task1`, `task2`, `task3`
- Workload orchestration: `locust/run_gradual_workload.py` and `locust/script.py`
- Metric collection cadence: every `120 seconds`

### User/replica strategy
- Current dataset includes concurrent users: `1 to 10`
- Current dataset includes replicas: `1, 2, 3`
- You can extend runs to replica `4` in future campaigns if needed.

### Naming convention
- File format: `taskX_<users>_<replicas>.csv`
- Examples:
  - `task1_4_2.csv`
  - `task3_10_3.csv`

## 2) Dataset

- Total CSV files: `90`
- Total rows: `1770`
- Services: `task1`, `task2`, `task3`
- Users present: `1 to 10`
- Replicas present: `1,2,3`

### Raw schema (from collector)
- `timestamp, service, endpoint, replicas, cpu_limit, cpu_request, memory_limit, memory_request, request_rate, p50_latency, p95_latency, p99_latency, avg_latency, throughput, cpu_usage`

### Modeling target
- `cpu_usage`

### Features used for model
- `request_rate`
- `request_rate_lag_1`
- `request_rate_lag_2`
- `cpu_usage_lag_1`
- `cpu_usage_lag_2`

These were chosen to keep the model simple and explainable while capturing short-term temporal behavior.

## 3) Data Loading

Implemented in `train_regression.py`:
- Reads all `*.csv` from `../data_collection`
- Extracts `concurrent_users` from filename
- Adds `source_file` for per-file temporal processing
- Coerces `request_rate` and `cpu_usage` to numeric
- Parses `timestamp` to datetime

## 4) Preprocessing and Feature Engineering

Steps implemented:
- Drop fully empty rows
- Drop rows with invalid `timestamp` or target (`cpu_usage`)
- Detect suspicious all-zero metric rows
  - Columns checked: `cpu_limit, cpu_request, memory_limit, memory_request, request_rate, throughput, cpu_usage`
  - Strategy: convert these rows to NaN then impute via:
    1. per-file forward/backward fill
    2. global median fallback
- Sort by `source_file, timestamp`
- Create lag features
  - `lag_1`: value one 2-minute interval back
  - `lag_2`: value two 2-minute intervals back
- Keep only required modeling columns

## 5) Data Analysis

Generated automatically under `output/eda/`:
- `summary_stats.csv`
- `correlation_heatmap.png`
- `cpu_usage_distribution.png`
- `request_rate_vs_cpu_usage.png`

Purpose:
- Check feature-target relationships
- Inspect distribution and outliers
- Validate missingness and basic quality

## 6) Model Training

### Models currently included
- `LinearRegression`
- `RandomForestRegressor`

### Extensible model setup
- Models are defined in a single registry function in `train_regression.py`.
- To add another model, add one entry to that dictionary.

### Train/Test split rule
- Train users: `1,2,3,5,6,8,9,10`
- Test users: `4,7`

This simulates unseen user levels inside observed range.

## 7) Model Testing and Evaluation

Two evaluation cases are used:

### Case 1: In-range test
- Evaluate directly on test users `4` and `7`
- Expected behavior: relatively lower error

### Case 2: Extrapolation proxy
- Start from test rows but scale traffic features to mimic higher load (`user=15`-like intensity)
- Expected behavior: higher error than Case 1

### Metrics
1. MAE (Mean Absolute Error)
- Formula: `MAE = mean(|y_true - y_pred|)`
- Why used: robust, unit-preserving, easy to interpret in CPU units.

2. MAPE (Mean Absolute Percentage Error)
- Formula: `MAPE = mean(|(y_true - y_pred) / y_true|) * 100`
- Why used: relative error in percentage terms.
- Note: when true `cpu_usage` is very close to zero, MAPE can become very large.

## 8) Model Comparison

After training, the script compares all configured models across both test cases:
- Case 1 (users 4/7)
- Case 2 (simulated user-15 traffic scale)

Comparison metrics are produced for each model and case:
- MAE
- MAPE

Charts are generated under `output/charts/`:
- `model_comparison_mae.png`
- `model_comparison_mape.png`

## 9) Outputs

Under `training_regression/output/`:
- `cpu_eval_summary.json`
  - train/test users
  - row counts
  - per-case metrics for all models
  - best model for case1 (by MAE)
- `cpu_predictions_test.csv`
  - actual `cpu_usage`
  - per-model predictions for case1 and case2
- `model_comparison_metrics.csv`
  - tabular comparison: `model`, `case`, `mae`, `mape`
- `models/`
  - `linear_regression.joblib`
  - `random_forest.joblib`
- `new_point_predictions.csv`
  - predictions on synthetic unseen-user points (not users 4/7)
- `new_point_predictions.json`
  - same as above in JSON format

Under `training_regression/output/eda/`:
- EDA stats and plots listed above

Under `training_regression/output/charts/`:
- model comparison charts listed above

### Using saved models for new points
The script saves each model and immediately re-loads it for inference on new synthetic points:
- `new_user11_normal`
- `new_user12_medium`
- `new_user15_high`

This demonstrates how to test points that are not from test users `4` or `7`.

## 10) How To Run

From `training_regression/`:
```bash
python train_regression.py
```

## 11) Future Work
1. Design comprehensive data collection campaign, reflecting 

- Traffic intensity: Low, medium, high load scenarios, beyond 1 to 10 users.
- Traffic patterns: Uniform, bursty, gradual ramp - We only did gradual ramp.
- Endpoint distribution: Different API call mixes
- System configuration: CPU limits (100m, 250m, 500m, 1000m)


2. Train temporal networks like LSTM, TCN and Transformers for better predictions

3. Define more evaluation criterias like R^2, Prediction accuracy etc

4. Predict more interesting outcomes like
- Response Time (latency per endpoint)
- Throughput (requests/second)
 
 5. Make model parametric i.e we should be able to input `Incoming traffic patterns` and `System configuration` at run-time