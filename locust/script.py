import requests
import time
import csv
import os
import subprocess
from datetime import datetime

# CONFIGURATION
PROMETHEUS_URL = "http://localhost:9090"
SERVICES = ["task1", "task2", "task3"]
NAMESPACE = "default"

# AUTOMATION CONFIGURATION
USERS_RANGE = range(1, 6)      
REPLICAS_RANGE = range(3, 5)    
INTERVAL_SEC = 120              
DATA_POINTS_PER_RUN = 20     
COOLDOWN_SEC = 100              
LOCUST_HOST = "http://localhost:8080" 

# --- PATH CONFIGURATION ---
LOCUST_FILE_PATH = "locustfile.py"
# -----------------------------

def scale_deployments(replicas):
    print(f"\n[SYSTEM] Scaling all tasks to {replicas} replicas...")
    for service in SERVICES:
        subprocess.run(
            ["kubectl", "scale", f"deployment/{service}-deployment", f"--replicas={replicas}", f"-n={NAMESPACE}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    print(f"[SYSTEM] Waiting 45s for pods to stabilize...")
    time.sleep(45)

def start_locust(users):
    print(f"[SYSTEM] Starting Locust ({users} users) using: {LOCUST_FILE_PATH}")
    
    # FIX: Create a copy of the current system environment and add the mandatory variable
    env = os.environ.copy()
    env["LOCUST_THINK_TIME"] = "0.42" 
    
    process = subprocess.Popen(
        [
            "locust",
            "-f",
            LOCUST_FILE_PATH,
            "--headless",
            "-u",
            str(users),
            "-r",
            "2",
            "--host",
            LOCUST_HOST,
            "--run-time",
            f"{(INTERVAL_SEC * DATA_POINTS_PER_RUN)}s",
        ],
        env=env # Pass the environment variables here
    )
    return process

def stop_locust(process):
    print(f"[SYSTEM] Stopping Locust...")
    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=30)

def query_prometheus(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=10)
        res = response.json()
        if res["status"] == "success" and res["data"]["result"]:
            return round(float(res["data"]["result"][0]["value"][1]), 2)
        return 0.0
    except Exception:
        return 0.0


def get_endpoint_name(service):
    """Returns most active endpoint for a given service."""
    query = f'''
    topk(1,
      sum by (request_path) (
        rate(istio_requests_total{{
          reporter="destination",
          destination_service_name="{service}-svc",
          request_path!=""
        }}[{INTERVAL_SEC}s])
      )
    )
    '''

    try:
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )

        results = response.json()["data"]["result"]
        if results:
            return results[0]["metric"].get("request_path", "unknown")
        return "no_traffic"
    except Exception:
        return "error"


def collect_metrics(service):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_endpoint = get_endpoint_name(service)

    window = f"{INTERVAL_SEC}s"
    queries = {
        "replicas": f'count(up{{pod=~"{service}.*"}})',
        "cpu_limit": f'sum(kube_pod_container_resource_limits{{container="app", resource="cpu", pod=~"{service}.*"}})',
        "cpu_request": f'sum(kube_pod_container_resource_requests{{container="app", resource="cpu", pod=~"{service}-.*"}})',
        "memory_limit": f'sum(kube_pod_container_resource_limits{{container="app", resource="memory", pod=~"{service}-.*"}}) / 1024 / 1024',
        "memory_request": f'sum(kube_pod_container_resource_requests{{container="app", resource="memory", pod=~"{service}-.*"}}) / 1024 / 1024',
        "request_rate": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*"}}[{window}]))',
        "throughput": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*", response_code="200"}}[{window}]))',
        "p50_latency": f'histogram_quantile(0.50, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',
        "p95_latency": f'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',
        "p99_latency": f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',
        "avg_latency": f'sum(rate(istio_request_duration_milliseconds_sum{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) / sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*"}}[{window}]))',
        "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[{window}]))',
    }

    data = {"timestamp": timestamp, "service": service, "endpoint": current_endpoint}
    for key, query in queries.items():
        data[key] = query_prometheus(query)
    return data

def main():
    print(f"Starting Automated Scrape...")
    if not os.path.exists(LOCUST_FILE_PATH):
        print(f"[ERROR] Path does not exist: {LOCUST_FILE_PATH}")
        return

    fieldnames = ["timestamp", "replicas", "service", 
                  "cpu_limit", "cpu_request", "memory_limit", "memory_request", 
                  "request_rate", "throughput", "avg_latency", "p50_latency", "p95_latency", "p99_latency", "cpu_usage"]

    for users in USERS_RANGE:
        for service in SERVICES:
            filename = f"{service}_metrics.csv"
            exists = os.path.isfile(filename)
            with open(filename, 'a', newline='') as f:
                if exists: f.write('\n') 
                csv.DictWriter(f, fieldnames=fieldnames).writeheader()

        for replicas in REPLICAS_RANGE:
            scale_deployments(replicas)
            proc = None
            try:
                proc = start_locust(users)
                for step in range(DATA_POINTS_PER_RUN):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting {INTERVAL_SEC}s (Point {step+1}/{DATA_POINTS_PER_RUN})...")
                    time.sleep(INTERVAL_SEC)
                    for service in SERVICES:
                        row = collect_metrics(service)
                        with open(f"{service}_metrics.csv", 'a', newline='') as f:
                            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
                        print(f" -> {service} (U{users}, R{replicas}) | RPS: {row['request_rate']} | p99: {row['p99_latency']}ms")
            finally:
                stop_locust(proc)
            time.sleep(COOLDOWN_SEC)

if __name__ == "__main__":
    main()
