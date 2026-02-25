import os
import sys
import pandas as pd
from locust import HttpUser, task, constant, LoadTestShape

# --- Helper function to read environment variables ---
def get_env_var(var_name, is_numeric=False, default=None):
    """
    Reads an environment variable. If it's mandatory (no default), exits if not set.
    """
    value = os.environ.get(var_name)
    if value is None:
        if default is not None:
            return default
        print(f"❌ ERROR: Mandatory environment variable '{var_name}' is not set.")
        sys.exit(1)
    
    if is_numeric:
        try:
            return float(value)
        except ValueError:
            print(f"❌ ERROR: Environment variable '{var_name}' must be a number, but got '{value}'.")
            sys.exit(1)
    return value

# --- TraceShape Class ---
class TraceShape(LoadTestShape):
    
    time_limit = int(get_env_var("LOCUST_TIME_LIMIT", is_numeric=True))
    trace_file = get_env_var("LOCUST_WORKLOAD_CSV")
    
    # CORRECTED: Read amplitude and shift from environment variables
    amplitude = int(get_env_var("LOCUST_AMPLITUDE", is_numeric=True, default=50))
    shift = int(get_env_var("LOCUST_SHIFT", is_numeric=True, default=10))
    
    fit_trace = get_env_var("LOCUST_FIT_TRACE", default="false").lower() == "true"
    
    def __init__(self):
        super().__init__()
        try:
            trace_data = pd.read_csv(self.trace_file).to_numpy().T[0]
            self.max_index = len(trace_data)
            max_val = max(trace_data)
            min_val = min(trace_data)
            self.scaled_trace = [
                int((v - min_val) / (max_val - min_val) * self.amplitude + self.shift)
                for v in trace_data
            ]
            print(f"✅ Workload trace '{self.trace_file}' loaded (Min Users: {self.shift}, Max Users: {self.amplitude + self.shift}).")
            if self.fit_trace:
                print("   -> 'Fit to Duration' mode ACTIVE.")
        except Exception as e:
            print(f"❌ ERROR: Could not read or process workload file '{self.trace_file}': {e}")
            sys.exit(1)

    # This runs evry second to determine the current user count based on the trace and elapsed time
    def tick(self):
        # Check if we've exceeded the time limit
        run_time = self.get_run_time()
        if run_time > self.time_limit:
            return None

        # fit_trace mean we want to stretch the trace to fit the entire duration, otherwise we loop through it 
        if self.fit_trace:
            scaling_ratio = self.max_index / self.time_limit
            trace_index = int(run_time * scaling_ratio)
            # Stretch the trace to fit the entire duration, but cap the index to avoid out-of-range errors
            trace_index = min(trace_index, self.max_index - 1)
        else:
            # repeat the trace if time exceeds its length
            trace_index = int(run_time) % self.max_index

        # target number of users at the instance of time.
        user_count = self.scaled_trace[trace_index]

        # 100 users/sec spawn_rate is how quickly we want to reach that user count. Setting it equal to user_count means we will try to reach the target user count within 1 second, which is aggressive but ensures we follow the trace closely.
        return (user_count, 100) # Using aggressive spawn rate, useful for brusts and spikes
    


# --- User Behavior ---
class MicroserviceUser(HttpUser):
    host = "http://localhost"
    # think_time,  defines the wait time between user actions. i.e user send requests in every .42s on average. 
    think_time = get_env_var("LOCUST_THINK_TIME", is_numeric=True)
    wait_time = constant(think_time)
    
    entrypoint_path = get_env_var("LOCUST_ENTRYPOINT_PATH", default="/")
    print(f"✅ User think time set to {think_time}s. Entrypoint path set to {entrypoint_path}.")

    # each user calls the entrypoint in a loop with the defined think time in between.
    @task
    def access_entrypoint(self):
        # The entrypoint is defined as an environment variable, for now we hit the / endpoint, 
        self.client.get(self.entrypoint_path)

