import requests
import time
import csv
import os
import argparse
from datetime import datetime

# CONFIGURATION
PROMETHEUS_URL = "http://localhost:9090"
SERVICES = ["task1", "task2", "task3"]
INTERVAL_SEC = 120 # 2 minute interval for data collection
 
def query_prometheus(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={'query': query})
        res = response.json()
        if res['status'] == 'success' and res['data']['result']:
            return round(float(res['data']['result'][0]['value'][1]), 2)
        return 0.0
    except Exception:
        return 0.0
    
def get_endpoint_name(service):
    """
    Returns most active endpoint for a given service.
    """

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

    except Exception as e:
        return f"error"
    
def collect_metrics(service):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    current_endpoint = get_endpoint_name(service)

    window = f"{INTERVAL_SEC}s"
    # These queries are mapped to the required Phase 2 Schema 
    queries = {
        # System Configuration
        "replicas": f'count(up{{pod=~"{service}.*"}})',

        # System Configuration - Use a specific container filter
        "cpu_limit": f'sum(kube_pod_container_resource_limits{{container="app", resource="cpu", pod=~"{service}.*"}})',
        "cpu_request": f'sum(kube_pod_container_resource_requests{{container="app", resource="cpu", pod=~"{service}-.*"}})',
        "memory_limit": f'sum(kube_pod_container_resource_limits{{container="app", resource="memory", pod=~"{service}-.*"}}) / 1024 / 1024',
        "memory_request": f'sum(kube_pod_container_resource_requests{{container="app", resource="memory", pod=~"{service}-.*"}}) / 1024 / 1024',

        # Traffic Patterns [cite: 34, 49]
        "request_rate": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*"}}[{window}]))',
        "throughput": f'sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*", response_code="200"}}[{window}]))',

        "p50_latency": f'histogram_quantile(0.50, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',
        "p95_latency": f'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',
        "p99_latency": f'histogram_quantile(0.99, sum(rate(istio_request_duration_milliseconds_bucket{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) by (le))',

        
        "avg_latency": f'sum(rate(istio_request_duration_milliseconds_sum{{reporter="destination", destination_workload=~"{service}.*"}}[{window}])) / sum(rate(istio_requests_total{{reporter="destination", destination_workload=~"{service}.*"}}[{window}]))',
        # Resource Utilization [cite: 35, 49]
        "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{service}.*"}}[{window}]))'
    }
    print(f"Collecting metrics for {service} at endpoint {current_endpoint}...")
    data = {"timestamp": timestamp, "service": service, "endpoint": current_endpoint}
    for key, q in queries.items():
        data[key] = query_prometheus(q)
    return data

def main():
    parser = argparse.ArgumentParser(description="Collect metrics for one experiment setup.")
    parser.add_argument("--users", type=int, required=True, help="User count for this experiment.")
    parser.add_argument("--replicas", type=int, required=True, help="Replica count for this experiment.")
    args = parser.parse_args()

    print(f"Starting data collection for {SERVICES} according to project schema...")
    while True:
        for service in SERVICES:
            row = collect_metrics(service)
            filename = f"{service}_{args.users}_{args.replicas}.csv"
            file_exists = os.path.isfile(filename)
            
            with open(filename, 'a', newline='') as f:
                # Ordering fields exactly like the proposal 
                fieldnames = ["timestamp", "service", "endpoint", "replicas", "cpu_limit", "cpu_request", 
                              "memory_limit", "memory_request", "request_rate", "p50_latency", "p95_latency", 
                              "p99_latency", "avg_latency", "throughput", "cpu_usage"]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
            print(f"[{row['timestamp']}] Logged {service} - Latency: {row['p95_latency']:.2f}ms")
        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    main()
