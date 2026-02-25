# Running locustfile.py 
- This allows us to open locust client in browser
```
locust -f locustfile.py
``` 

# Running gradual experiment

```

```
# Running locust from command line to use workloads
- It will run for 25 minutes, reading 
- If we set fit_trace false and limit 300, it will cover only rows 300 rows from csv. We want it to cover all rows.
- As required, we are keeping users between 1 to 20 at concurrent time. i.e no more than 20 concurrent users stress the system.
```
LOCUST_TIME_LIMIT=800 \
LOCUST_WORKLOAD_CSV="workloads/sin400.csv" \
LOCUST_AMPLITUDE=19 \
LOCUST_SHIFT=1 \
LOCUST_THINK_TIME=0.1 \
LOCUST_ENTRYPOINT_PATH="/" \
LOCUST_FIT_TRACE="false" \
locust -f locustfile_trace.py --headless --csv=logs/trace_experiment
```

- When we LOCUST_FIT_TRACE=false, it will loop over data again after reaching 800th row.

```
for i in $(seq 1 30); do
  echo "Starting run $i/30"
  LOCUST_TIME_LIMIT=800 \
  LOCUST_WORKLOAD_CSV="workloads/sin400.csv" \
  LOCUST_AMPLITUDE=19 \
  LOCUST_SHIFT=1 \
  LOCUST_THINK_TIME=0.1 \
  LOCUST_ENTRYPOINT_PATH="/" \
  LOCUST_FIT_TRACE="false" \
  locust -f locustfile_trace.py --headless --csv="logs/trace_experiment_run${i}"
done

```
# Running constan load
```
locust -f locustfile_constant.py \
  --headless \
  --users 1 \
  --spawn-rate 10 \
  --run-time 5m
```

# Running campaign with locust
```
python locust/run_campaigns.py --uniform-runtime 5m --ramp-runtime 5m --bursty-runtime 5m --pause-seconds 0
```