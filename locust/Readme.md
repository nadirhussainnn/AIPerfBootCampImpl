# Running gradual experiment

```
python3 locust/run_gradual_workload.py --users-start 1 --users-end 2
```

- For single step, we can run
```
python3 locust/run_gradual_workload.py --only-user 1 --only-replica 1
```

kubectl -n default scale deploy/task1-deployment deploy/task2-deployment deploy/task3-deployment --replicas=1
kubectl -n default rollout status deploy/task1-deployment --timeout=10m
kubectl -n default rollout status deploy/task2-deployment --timeout=10m
kubectl -n default rollout status deploy/task3-deployment --timeout=10m

python3 train_regression.py \
  --data-dir data_collection \
  --target avg_latency \
  --output-dir output

kubectl -n default get deploy task1-deployment task2-deployment task3-deployment -w

python3 locust/run_gradual_workload.py --only-user 3 --only-replica 3

# run after 30 min
( sleep 1800; python3 locust/run_gradual_workload.py --only-user 4 --only-replica 2 ) > locust/logs/gradual/schedule_u4_r2.log 2>&1 &

# run after 30+50 = 80 min
( sleep 4800; python3 locust/run_gradual_workload.py --only-user 5 --only-replica 2 ) > locust/logs/gradual/schedule_u5_r2.log 2>&1 &

# run after 30+50+50 = 130 min
( sleep 7800; python3 locust/run_gradual_workload.py --only-user 5 --only-replica 3 ) > locust/logs/gradual/schedule_u5_r3.log 2>&1 &


# 1) remove all currently scheduled background jobs
jobs -p | xargs kill

# 2) confirm none remain
jobs -l

# 3) reschedule with "kill running first" logic

nohup sh -c '
  sleep 30
  pkill -f "python3 locust/run_gradual_workload.py" || true
  pkill -f "python3 data_collection/script.py" || true
  pkill -f "python3 -m locust" || true
  python3 locust/run_gradual_workload.py --only-user 4 --only-replica 2

  sleep 3000
  pkill -f "python3 locust/run_gradual_workload.py" || true
  pkill -f "python3 data_collection/script.py" || true
  pkill -f "python3 -m locust" || true
  python3 locust/run_gradual_workload.py --only-user 5 --only-replica 2

  sleep 6200
  pkill -f "python3 locust/run_gradual_workload.py" || true
  pkill -f "python3 data_collection/script.py" || true
  pkill -f "python3 -m locust" || true
  python3 locust/run_gradual_workload.py --only-user 5 --only-replica 3
' > locust/logs/gradual/schedule_master.log 2>&1 < /dev/null &
disown

# Running data collector
```
python script.py --users 1 --replicas 1
```



# Running locustfile.py 
- This allows us to open locust client in browser
```
locust -f locustfile.py
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

# Why it was failing for replica 4

At replica=4, we are trying to run 12 app pods total (task1/2/3, each 4 replicas), and each app pod requests:
- cpu: 1
- memory: 1536Mi

So app requests alone are:
- CPU: 12 cores
- Memory: 18 GiB

Then Istio sidecars add extra requests per pod (~0.1 CPU, 128Mi) for another ~1.2 CPU and ~1.5 GiB.

Total requested is roughly:
- CPU: 13.2 (over your Docker CPU 12)
- Memory: 19.5 GiB (over your Docker memory 18 GiB)