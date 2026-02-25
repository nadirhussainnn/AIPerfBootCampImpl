import argparse
import glob
import os
import shutil
import subprocess
import time
from typing import List

# Fixed campaign config.
HOST = "http://localhost"
LOCUSTFILE = "locustfile_constant.py"
OUTPUT_DIR = "logs/gradual"
USERS_START = 1
USERS_END = 10
RUN_TIME = "40m"
SPAWN_RATE = 100
NAMESPACE = "default"
PAUSE_SECONDS = 120

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
                "--timeout=5m",
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
    # Prefer the detailed time-series file; fallback to summary if needed.
    history_file = f"{temp_prefix}_stats_history.csv"
    summary_file = f"{temp_prefix}_stats.csv"
    source_file = history_file if os.path.exists(history_file) else summary_file

    if not os.path.exists(source_file):
        raise RuntimeError(f"No Locust CSV output found for prefix: {temp_prefix}")

    for service in SERVICES:
        dst = os.path.join(OUTPUT_DIR, f"{service}_{users}_{replicas}.csv")
        shutil.copyfile(source_file, dst)

    # Keep output directory clean: remove temporary Locust multi-file artifacts.
    for file_path in glob.glob(f"{temp_prefix}_*.csv"):
        os.remove(file_path)


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

    # Required ordering: for each user count, run all replica variants.
    for users in range(args.users_start, args.users_end + 1):
        for replicas in REPLICA_COUNTS:
            run_id = f"u{users}_r{replicas}"
            temp_prefix = os.path.join(OUTPUT_DIR, f"tmp_{run_id}")
            print(f"\n=== Experiment {run_id} | duration={RUN_TIME} ===")

            scale_deployments(NAMESPACE, DEPLOYMENTS, replicas)
            try:
                run_one_experiment(
                    host=HOST,
                    locustfile=LOCUSTFILE,
                    users=users,
                    run_time=RUN_TIME,
                    csv_prefix=temp_prefix,
                    spawn_rate=SPAWN_RATE,
                )
                create_named_result_files(temp_prefix=temp_prefix, users=users, replicas=replicas)
            finally:
                # Reset to baseline replicas and give the system time to stabilize.
                print(f"Resetting all deployments to {RESET_REPLICAS} replica(s)")
                scale_deployments(NAMESPACE, DEPLOYMENTS, RESET_REPLICAS)
                print(f"Waiting {PAUSE_SECONDS}s before next experiment...")
                time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
