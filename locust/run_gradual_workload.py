import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from typing import List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Fixed campaign config.
HOST = "http://localhost"
LOCUSTFILE = os.path.join(SCRIPT_DIR, "locustfile_constant.py")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "logs", "gradual")
USERS_START = 1
USERS_END = 10
RUN_TIME = "40m"
RUN_TIME_SECONDS = 40 * 60
SPAWN_RATE = 100
NAMESPACE = "default"
PAUSE_SECONDS = 120
ROLLOUT_TIMEOUT = "10m"

DATA_COLLECTION_SCRIPT = os.path.join(REPO_ROOT, "data_collection", "script.py")
DATA_COLLECTION_OUTPUT_DIR = os.path.join(REPO_ROOT, "data_collection")

# Services from data_collection/script.py
DEPLOYMENTS = ["task1-deployment", "task2-deployment", "task3-deployment"]
SERVICES = ["task1", "task2", "task3"]

# One experiment per replica count.
REPLICA_COUNTS = [1, 2, 3, 4]
RESET_REPLICAS = 1


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def scale_deployments(namespace: str, deployments: List[str], replicas: int) -> None:
    for dep in deployments:
        run_cmd(
            [
                "kubectl",
                "-n",
                namespace,
                "scale",
                f"deployment/{dep}",
                f"--replicas={replicas}",
            ]
        )
    for dep in deployments:
        run_cmd(
            [
                "kubectl",
                "-n",
                namespace,
                "rollout",
                "status",
                f"deployment/{dep}",
                f"--timeout={ROLLOUT_TIMEOUT}",
            ]
        )


def run_one_experiment(
    host: str,
    locustfile: str,
    users: int,
    run_time: str,
    csv_prefix: str,
    spawn_rate: int,
) -> None:
    cmd = [
        "locust",
        "-f",
        locustfile,
        "--host",
        host,
        "--headless",
        "--users",
        str(users),
        "--spawn-rate",
        str(spawn_rate),
        "--run-time",
        run_time,
        "--csv",
        csv_prefix,
        "--csv-full-history",
    ]
    run_cmd(cmd)


def create_named_result_files(temp_prefix: str, users: int, replicas: int) -> None:
    """Create exactly-named csv files: taskX_<users>_<replicas>.csv."""
    history_file = f"{temp_prefix}_stats_history.csv"
    summary_file = f"{temp_prefix}_stats.csv"
    source_file = history_file if os.path.exists(history_file) else summary_file

    if not os.path.exists(source_file):
        raise RuntimeError(f"No Locust CSV output found for prefix: {temp_prefix}")

    for service in SERVICES:
        dst = os.path.join(OUTPUT_DIR, f"{service}_{users}_{replicas}.csv")
        shutil.copyfile(source_file, dst)


def cleanup_temp_locust_files(temp_prefix: str) -> None:
    for file_path in glob.glob(f"{temp_prefix}_*.csv"):
        os.remove(file_path)


def remove_old_experiment_outputs(users: int, replicas: int) -> None:
    for service in SERVICES:
        locust_file = os.path.join(OUTPUT_DIR, f"{service}_{users}_{replicas}.csv")
        data_file = os.path.join(DATA_COLLECTION_OUTPUT_DIR, f"{service}_{users}_{replicas}.csv")
        for path in (locust_file, data_file):
            if os.path.exists(path):
                os.remove(path)


def start_data_collection(users: int, replicas: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        DATA_COLLECTION_SCRIPT,
        "--users",
        str(users),
        "--replicas",
        str(replicas),
        "--duration-seconds",
        str(RUN_TIME_SECONDS),
        "--output-dir",
        DATA_COLLECTION_OUTPUT_DIR,
        "--overwrite",
    ]
    return subprocess.Popen(cmd)


def stop_data_collection(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gradual traffic experiments.")
    parser.add_argument("--users-start", type=int, default=USERS_START)
    parser.add_argument("--users-end", type=int, default=USERS_END)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.users_start > args.users_end:
        raise ValueError("--users-start must be <= --users-end")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_COLLECTION_OUTPUT_DIR, exist_ok=True)

    for users in range(args.users_start, args.users_end + 1):
        for replicas in REPLICA_COUNTS:
            run_id = f"u{users}_r{replicas}"
            temp_prefix = os.path.join(OUTPUT_DIR, f"tmp_{run_id}")
            collector_proc = None
            print(f"\n=== Experiment {run_id} | duration={RUN_TIME} ===")

            try:
                remove_old_experiment_outputs(users, replicas)
                cleanup_temp_locust_files(temp_prefix)

                scale_deployments(NAMESPACE, DEPLOYMENTS, replicas)
                collector_proc = start_data_collection(users, replicas)
                run_one_experiment(
                    host=HOST,
                    locustfile=LOCUSTFILE,
                    users=users,
                    run_time=RUN_TIME,
                    csv_prefix=temp_prefix,
                    spawn_rate=SPAWN_RATE,
                )

                if collector_proc.poll() is None:
                    collector_proc.wait(timeout=180)

                create_named_result_files(temp_prefix=temp_prefix, users=users, replicas=replicas)
                cleanup_temp_locust_files(temp_prefix)
                print(f"Experiment {run_id} completed successfully.")
            except subprocess.TimeoutExpired:
                print(f"Experiment {run_id} warning: data collector did not exit on time.")
            except subprocess.CalledProcessError as err:
                print(f"Experiment {run_id} failed: {' '.join(err.cmd)}")
                print("Continuing to reset and move to the next experiment.")
            except Exception as err:
                print(f"Experiment {run_id} failed with unexpected error: {err}")
                print("Continuing to reset and move to the next experiment.")
            finally:
                stop_data_collection(collector_proc)

                print(f"Resetting all deployments to {RESET_REPLICAS} replica(s)")
                try:
                    scale_deployments(NAMESPACE, DEPLOYMENTS, RESET_REPLICAS)
                except subprocess.CalledProcessError as reset_err:
                    print(f"Reset failed: {' '.join(reset_err.cmd)}")

                print(f"Waiting {PAUSE_SECONDS}s before next experiment...")
                time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
