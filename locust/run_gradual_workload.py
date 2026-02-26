import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import List, TextIO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

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
DEFAULT_APP_MANIFEST = os.path.join(REPO_ROOT, "kubernetes", "apps", "acmeair-light.yaml")

DEPLOYMENTS = ["task1-deployment", "task2-deployment", "task3-deployment"]
SERVICES = ["task1", "task2", "task3"]
DEFAULT_REPLICA_COUNTS = [1, 2, 3, 4]
RESET_REPLICAS = 1


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def parse_int_list(value: str) -> List[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


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


def reset_to_defaults() -> None:
    """Restore default deployment specs (replicas/resources/env) from manifest."""
    run_cmd(["kubectl", "apply", "-f", DEFAULT_APP_MANIFEST])
    # Wait until defaults are fully rolled out before starting a new experiment.
    scale_deployments(NAMESPACE, DEPLOYMENTS, RESET_REPLICAS)


def run_one_experiment(users: int, csv_prefix: str) -> None:
    cmd = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        LOCUSTFILE,
        "--host",
        HOST,
        "--headless",
        "--users",
        str(users),
        "--spawn-rate",
        str(SPAWN_RATE),
        "--run-time",
        RUN_TIME,
        "--csv",
        csv_prefix,
        "--csv-full-history",
    ]
    run_cmd(cmd)


def create_named_result_files(temp_prefix: str, users: int, replicas: int) -> None:
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


def target_paths(users: int, replicas: int) -> List[str]:
    paths = []
    for service in SERVICES:
        paths.append(os.path.join(OUTPUT_DIR, f"{service}_{users}_{replicas}.csv"))
        paths.append(os.path.join(DATA_COLLECTION_OUTPUT_DIR, f"{service}_{users}_{replicas}.csv"))
    return paths


def data_collection_paths(users: int, replicas: int) -> List[str]:
    return [
        os.path.join(DATA_COLLECTION_OUTPUT_DIR, f"{service}_{users}_{replicas}.csv")
        for service in SERVICES
    ]


def all_files_present_and_nonempty(paths: List[str]) -> bool:
    return all(os.path.exists(path) and os.path.getsize(path) > 0 for path in paths)


def any_files_present(paths: List[str]) -> bool:
    return any(os.path.exists(path) for path in paths)


def delete_paths(paths: List[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def terminate_stale_collectors(users: int, replicas: int) -> None:
    pgrep = shutil.which("pgrep")
    if not pgrep:
        return

    pattern = f"data_collection/script.py --users {users} --replicas {replicas}"
    try:
        result = subprocess.run([pgrep, "-f", pattern], capture_output=True, text=True, check=False)
    except Exception:
        return

    pids = [pid.strip() for pid in result.stdout.splitlines() if pid.strip()]
    for pid in pids:
        if pid == str(os.getpid()):
            continue
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


def start_data_collection(users: int, replicas: int, overwrite: bool) -> tuple[subprocess.Popen, TextIO, str]:
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
    ]
    if overwrite:
        cmd.append("--overwrite")
    log_path = os.path.join(OUTPUT_DIR, f"collector_u{users}_r{replicas}.log")
    log_file = open(log_path, "a")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True)
    return proc, log_file, log_path


def stop_data_collection(proc: subprocess.Popen | None, log_file: TextIO | None) -> None:
    if proc is None or proc.poll() is not None:
        if log_file is not None and not log_file.closed:
            log_file.close()
        return

    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if log_file is not None and not log_file.closed:
            log_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gradual traffic experiments with safe resume.")
    parser.add_argument("--users-start", type=int, default=USERS_START)
    parser.add_argument("--users-end", type=int, default=USERS_END)
    parser.add_argument("--replicas", default=",".join(str(x) for x in DEFAULT_REPLICA_COUNTS))
    parser.add_argument("--only-user", type=int, help="Run a single step for this user.")
    parser.add_argument("--only-replica", type=int, help="Run a single step for this replica.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files for targeted steps.")
    parser.add_argument("--pause-seconds", type=int, default=PAUSE_SECONDS)
    return parser.parse_args()


def build_experiment_list(args: argparse.Namespace) -> List[tuple[int, int]]:
    if (args.only_user is None) ^ (args.only_replica is None):
        raise ValueError("--only-user and --only-replica must be provided together")

    replica_counts = parse_int_list(args.replicas)
    if not replica_counts:
        raise ValueError("--replicas must contain at least one integer")

    if args.only_user is not None:
        return [(args.only_user, args.only_replica)]

    if args.users_start > args.users_end:
        raise ValueError("--users-start must be <= --users-end")

    experiments: List[tuple[int, int]] = []
    for users in range(args.users_start, args.users_end + 1):
        for replicas in replica_counts:
            experiments.append((users, replicas))
    return experiments


def main() -> None:
    args = parse_args()
    experiments = build_experiment_list(args)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_COLLECTION_OUTPUT_DIR, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0

    try:
        for users, replicas in experiments:
            run_id = f"u{users}_r{replicas}"
            temp_prefix = os.path.join(OUTPUT_DIR, f"tmp_{run_id}")
            files = target_paths(users, replicas)
            collector_proc = None
            collector_log_file = None
            collector_log_path = ""

            print(f"\n=== Experiment {run_id} | duration={RUN_TIME} ===")

            if all_files_present_and_nonempty(files) and not args.force:
                print(f"Skipping {run_id}: files already complete.")
                skipped += 1
                continue

            if any_files_present(files) and not args.force:
                print(f"Skipping {run_id}: partial files exist. Use --force to rerun this step.")
                skipped += 1
                continue

            if args.force:
                delete_paths(files)
            cleanup_temp_locust_files(temp_prefix)

            try:
                print("Resetting deployments to default manifest configuration...")
                reset_to_defaults()
                scale_deployments(NAMESPACE, DEPLOYMENTS, replicas)
                terminate_stale_collectors(users, replicas)
                collector_proc, collector_log_file, collector_log_path = start_data_collection(
                    users, replicas, overwrite=args.force
                )

                run_one_experiment(users=users, csv_prefix=temp_prefix)
                # Persist final experiment files as soon as Locust finishes.
                create_named_result_files(temp_prefix=temp_prefix, users=users, replicas=replicas)
                cleanup_temp_locust_files(temp_prefix)

                if collector_proc.poll() is not None and collector_proc.returncode != 0:
                    raise RuntimeError(
                        f"Data collector exited early with code {collector_proc.returncode}. "
                        f"See log: {collector_log_path}"
                    )
                if collector_proc.poll() is None:
                    collector_proc.wait(timeout=180)
                if collector_proc.returncode not in (0, None):
                    raise RuntimeError(
                        f"Data collector failed with code {collector_proc.returncode}. "
                        f"See log: {collector_log_path}"
                    )
                if not all_files_present_and_nonempty(data_collection_paths(users, replicas)):
                    raise RuntimeError(
                        f"Missing data_collection CSV(s) for u{users}_r{replicas}. "
                        f"See log: {collector_log_path}"
                    )
                print(f"Experiment {run_id} completed successfully.")
                completed += 1
            except subprocess.TimeoutExpired:
                # Collector timeout after Locust completion should not drop data.
                print(f"Experiment {run_id} warning: data collector did not exit on time.")
                if all_files_present_and_nonempty(target_paths(users, replicas)):
                    completed += 1
                else:
                    failed += 1
            except FileNotFoundError as err:
                print(f"Experiment {run_id} failed: missing file/command -> {err.filename}")
                failed += 1
            except subprocess.CalledProcessError as err:
                print(f"Experiment {run_id} failed: {' '.join(err.cmd)}")
                failed += 1
            except Exception as err:
                print(f"Experiment {run_id} failed with unexpected error: {err}")
                failed += 1
            finally:
                stop_data_collection(collector_proc, collector_log_file)

                print(f"Resetting all deployments to {RESET_REPLICAS} replica(s)")
                try:
                    scale_deployments(NAMESPACE, DEPLOYMENTS, RESET_REPLICAS)
                except subprocess.CalledProcessError as reset_err:
                    print(f"Reset failed: {' '.join(reset_err.cmd)}")

                print(f"Waiting {args.pause_seconds}s before next experiment...")
                time.sleep(args.pause_seconds)
    except KeyboardInterrupt:
        print("\nRun interrupted by user. Exiting safely after reset.")

    print(
        f"\nRun summary: completed={completed}, skipped={skipped}, failed={failed}, total={len(experiments)}"
    )


if __name__ == "__main__":
    main()
