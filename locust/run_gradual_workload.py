import os
import subprocess
import time
from typing import List

# Fixed campaign config (no CLI args).
HOST = "http://localhost"
LOCUSTFILE = "locust/locustfile_constant.py"
OUTPUT_DIR = "locust/logs/gradual"
USERS_START = 1
USERS_END = 10
RUN_TIME = "40m"
SPAWN_RATE = 100
NAMESPACE = "default"
PAUSE_SECONDS = 120

# Services from data_collection/script.py -> task1, task2, task3.
DEPLOYMENTS = ["task1-deployment", "task2-deployment", "task3-deployment"]

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


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Required ordering: for each user count, run all replica variants.
    for users in range(USERS_START, USERS_END+1):
        for replicas in REPLICA_COUNTS:
            run_id = f"u{users}_r{replicas}"
            csv_prefix = os.path.join(OUTPUT_DIR, run_id)
            print(f"\n=== Experiment {run_id} | duration={RUN_TIME} ===")

            scale_deployments(NAMESPACE, DEPLOYMENTS, replicas)
            run_one_experiment(
                host=HOST,
                locustfile=LOCUSTFILE,
                users=users,
                run_time=RUN_TIME,
                csv_prefix=csv_prefix,
                spawn_rate=SPAWN_RATE,
            )

            # Reset to baseline replicas and give the system time to stabilize.
            print(f"Resetting all deployments to {RESET_REPLICAS} replica(s)")
            scale_deployments(NAMESPACE, DEPLOYMENTS, RESET_REPLICAS)
            print(f"Waiting {PAUSE_SECONDS}s before next experiment...")
            time.sleep(PAUSE_SECONDS)


if __name__ == "__main__":
    main()
