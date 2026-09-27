"""Execute a frozen standard-BCA lane once; preserve exits and stop on any failure."""
from pathlib import Path
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def status(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    os.replace(tmp, path)


@contextlib.contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_child(command, directory, timeout_seconds, *, cwd=ROOT, env=None):
    """Own exactly the child process group; never infer ownership from a stale PID."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    started = now()
    interrupted = None
    timed_out = False
    proc = None
    previous = {}

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = signum

    def stop_owned():
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupt)
        with (directory / "worker.log").open("x", encoding="utf8") as log:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            # Save start identity once for audit. Popen remains the ownership authority.
            try:
                proc_stat = Path(f"/proc/{proc.pid}/stat").read_text()
            except FileNotFoundError:
                proc_stat = None  # A harmless/failed child may exit immediately.
            save(directory / "dispatch.json", dict(command=command, started=started,
                 pid=proc.pid, process_group=proc.pid, proc_stat=proc_stat,
                 supervisor_pid=os.getpid(), slurm_job_id=os.getenv("SLURM_JOB_ID")))
            deadline = time.monotonic() + timeout_seconds
            while proc.poll() is None:
                if interrupted is not None or time.monotonic() >= deadline:
                    timed_out = interrupted is None
                    stop_owned()
                    break
                time.sleep(0.2)
            code = proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            stop_owned()
        for sig, old in previous.items():
            signal.signal(sig, old)
        save(directory / "actual_exit.json", dict(started=started, ended=now(),
             actual_returncode=None if proc is None else proc.returncode,
             timeout=timed_out, interruption_signal=interrupted))
    return code if code != 0 else (124 if timed_out else 128 + interrupted if interrupted else 0)


def run_lane(args):
    from runtime.config import resolve
    from runtime.provenance import source_files, sha
    manifest_path = Path(args.manifest).resolve()
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != args.sha256:
        raise ValueError("Frozen manifest identity mismatch")
    manifest = json.loads(payload)
    if manifest["calibration_weighting"] != "none":
        raise ValueError("This queue is standard/no-IW BCA only")
    if source_files() != manifest["source_sha256"]:
        raise ValueError("Frozen source mismatch")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output / "lane.lock"):
        # Refuse all resumes/retries, including interrupted supervisors.
        save(output / "controller_started.json", dict(time=now(), pid=os.getpid(),
             manifest_sha256=args.sha256, lane=args.lane, slurm_job_id=os.getenv("SLURM_JOB_ID")))
        completed = []
        code = 1
        try:
            for item in manifest["runs"]:
                if item["lane"] != args.lane:
                    continue
                run_id = item["row"]["run_id"]
                target = output / "runs" / run_id
                row = resolve(ROOT / "configs" / (item["algorithm"] + ".yaml"),
                              item["method"], item["seed"], target, item["dataset"])
                normalized = json.loads(json.dumps(row))
                if "options" in normalized:
                    normalized["options"]["output_dir"] = "{output_dir}"
                if normalized != item["row"]:
                    raise ValueError("Resolved scientific declaration changed: " + run_id)
                if source_files() != manifest["source_sha256"]:
                    raise ValueError("Source changed during the queue")
                status(output / "queue_status.json", dict(time=now(), status="running",
                       current=run_id, completed=completed, total=sum(x["lane"]==args.lane for x in manifest["runs"])))
                command = [sys.executable, str(ROOT / "train.py"), "--algorithm", item["algorithm"],
                           "--dataset", item["dataset"], "--method", item["method"],
                           "--seed", str(item["seed"]), "--output", str(target),
                           "--device", "cuda", "--lock", args.gpu_lock,
                           "--data-dir", args.data_dir]
                code = run_child(command, output / "attempts" / run_id, args.run_timeout)
                if code:
                    raise RuntimeError(f"Worker stopped: {run_id}, exit {code}; no retry")
                receipt = json.loads((target / "exit.json").read_text())
                if receipt.get("exit_code") != 0 or not receipt.get("completed"):
                    raise ValueError("Missing verified completion receipt")
                completed.append(dict(run_id=run_id, result_sha256=sha(target/"result.json")))
            code = 0
            status(output / "queue_status.json", dict(time=now(), status="complete", completed=completed))
        except BaseException as exc:
            code = code or 1
            status(output / "queue_status.json", dict(time=now(), status="stopped", completed=completed,
                   error=repr(exc), current=locals().get("run_id")))
            raise
        finally:
            save(output / "controller_exit.json", dict(time=now(), intended_exit_code=code,
                 completed_count=len(completed)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--lane", choices=("local", "cluster"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu-lock", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run-timeout", type=int, default=172800)
    args = p.parse_args()
    if args.run_timeout <= 0:
        p.error("Positive timeout required")
    run_lane(args)


if __name__ == "__main__":
    main()
