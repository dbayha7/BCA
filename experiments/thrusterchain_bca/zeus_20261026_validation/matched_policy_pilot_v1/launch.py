"""Run one approved, reserved six-run policy pilot; status/preflight are read-only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parents[1] / "titan_20261026"
EXPECTED_ENV = "/home/david/bca-operator-audit-20261026"
RUN_ORDER = ["seed0_native", "seed0_bca", "seed0_distance", "seed1_distance", "seed1_bca", "seed1_native"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_report(path):
    try:
        if not Path(path).is_file():
            return None, None
        result = read(path)
        if not isinstance(result, dict):
            raise ValueError("Report must be an object")
        return result, None
    except (OSError, ValueError) as exc:
        return None, type(exc).__name__ + ": " + str(exc)


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def validate_protocol(p):
    fixed = {"schema": "unifloral.matched_policy_pilot.v1", "critic_updates": 2000,
             "actor_updates": 1000, "auxiliary_updates": 2000, "refresh_at": 1000,
             "worker_timeout_seconds": 600, "evaluation_timeout_seconds": 600,
             "total_timeout_seconds": 4200, "batch_size": 256, "gamma": .98, "horizon": 100,
             "evaluation_policies": 7, "evaluation_episodes": 224, "simulator_transition_cap": 22400,
             "bca_kappa": 3., "actor_distance_coefficient": .1,
             "actor_distance_factor": 19.29619572569095, "posterior_draws": 1000,
             "alpha": .1, "credibility": .95, "full_training_hold": True,
             "automatic_restart": False, "head_ablation_deferred": True,
             "failed_one_sided_variant_excluded": True}
    for key, value in fixed.items():
        if p.get(key) != value or (type(value) is not bool and isinstance(p.get(key), bool)):
            raise ValueError("Approved fixed field differs: " + key)
    if p["run_order"] != RUN_ORDER or [(a["name"], a["kappa"]) for a in p["arms"]] != [("native", 0.), ("bca", 3.), ("distance", .1)]:
        raise ValueError("Approved arms/order differ")
    if p["streams"] != [{"index": 0, "native_seed": 2026102681, "aux_seed": 2026102683},
                        {"index": 1, "native_seed": 2026102682, "aux_seed": 2026102684}]:
        raise ValueError("Frozen paired streams differ")
    if p["evaluation_seeds"] != list(range(2026102800, 2026102832)):
        raise ValueError("Frozen evaluation seeds differ")
    return p


def verify_prepared(root=HERE):
    root = Path(root)
    manifest = read(root / "PREPARED.json")
    if manifest.get("scientific_execution_during_preparation") is not False:
        raise ValueError("Preparation may contain only synthetic fixtures")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha(root / name) != digest:
            raise ValueError("Prepared file changed: " + name)
    for name, digest in manifest["dependencies"].items():
        target = (root.parent / name).resolve()
        if not target.is_relative_to(root.parent.resolve()) or sha(target) != digest:
            raise ValueError("Pinned local dependency changed: " + name)
    p = validate_protocol(read(root / "protocol.json"))
    if read(root / "EVALUATION_SEEDS.json")["reset_seeds"] != p["evaluation_seeds"]:
        raise ValueError("Evaluation seed receipt differs")
    return p, manifest


def verify_inputs():
    restore = module("_matched_pilot_readonly_restore", HERE.parent / "bca_one_sided_v1/restore_logged.py")
    original_protocol = read(HANDOFF / "payload/unifloral_operator_alignment_v1/protocol.json")
    return restore.verify(original_protocol)  # Hashes/metadata only; no runtime/model initialization.


def reserve(root, token, started_monotonic=None):
    root = Path(root)
    if any((root / name).exists() for name in ("RUN_RESULT.json", "run")):
        raise ValueError("Existing pilot evidence; inspect status, never restart automatically")
    save_new(root / "ATTEMPT.json", {"token": token, "supervisor_pid": os.getpid(),
        "utc": datetime.now(timezone.utc).isoformat(), "status": "reserved_once",
        "started_monotonic": time.monotonic() if started_monotonic is None else started_monotonic,
        "no_automatic_restart": True})
    (root / "run").mkdir(exist_ok=False)


def scoped_process_check():
    allowed = {HERE, HERE.parent / "bca_one_sided_v1", HANDOFF,
               HANDOFF / "payload/unifloral_operator_alignment_v1"}
    targets = {(p / "launch.py").resolve() for p in allowed}
    targets.add((HANDOFF / "payload/unifloral_operator_alignment_v1/run_checks.py").resolve())
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid() or not (process / "comm").read_text().strip().startswith(("python", "pypy")):
                continue
            for raw in (process / "cmdline").read_bytes().split(b"\0")[1:]:
                arg = os.fsdecode(raw)
                if Path(arg).name not in ("launch.py", "run_checks.py"):
                    continue
                candidate = Path(arg)
                if not candidate.is_absolute():
                    candidate = (process / "cwd").resolve(strict=True) / candidate
                if candidate.resolve() in targets:
                    matches.append(int(process.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError("Scoped process inspection denied; no launch") from error
    if matches:
        raise RuntimeError("Existing scoped BCA process: " + str(matches))


def child_environment():
    env = dict(os.environ)
    env.update(BCA_TRANSFER_BUNDLE_ROOT=str(HANDOFF / "payload/transfer_bundle"),
        PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", JAX_PLATFORMS="cuda",
        JAX_DEFAULT_MATMUL_PRECISION="highest", JAX_ENABLE_X64="false",
        XLA_FLAGS="--xla_gpu_triton_gemm_any=True")
    return env


def terminate_owned(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        process.kill()  # Windows is used only by synthetic fixture children.
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=10)


def supervise(command, output, timeout_seconds, env=None, job_id="synthetic", announce=True, deadline=None):
    """One owned process group; progress is evidence, not a resume permission."""
    output = Path(output)
    started = time.monotonic()
    last_progress = None
    timed_out = interrupted = False
    error = None
    with (output / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=output, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"))
        try:
            while True:
                remaining = timeout_seconds - (time.monotonic() - started)
                if deadline is not None:
                    remaining = min(remaining, deadline - time.monotonic())
                if remaining <= 0:
                    timed_out = True
                    terminate_owned(process)
                    break
                try:
                    process.wait(timeout=min(5., remaining))
                    break
                except subprocess.TimeoutExpired:
                    progress_path = output / "progress.json"
                    if announce and progress_path.is_file():
                        try:
                            progress = read(progress_path)
                            compact = {k: progress[k] for k in ("status", "stage", "elapsed_seconds", "ledger", "counts",
                                "policy", "step", "attempted_transition_rows", "completed_transition_rows",
                                "completed_episodes", "inflight_transition_rows", "execution_counts_known") if k in progress}
                            serialized = json.dumps(compact, sort_keys=True)
                            if serialized != last_progress:
                                print(json.dumps({"run": job_id, "progress": compact}), flush=True)
                                last_progress = serialized
                        except (OSError, json.JSONDecodeError):
                            pass  # Atomic writer may have exited between the existence check and read.
        except BaseException as exc:
            interrupted = True
            error = type(exc).__name__ + ": " + str(exc)
            terminate_owned(process)
    return {"exit_code": process.returncode, "timed_out": timed_out, "interrupted": interrupted,
            "error": error, "elapsed_seconds": time.monotonic() - started}


def read_status(root=HERE):
    root = Path(root)
    if (root / "RUN_RESULT.json").is_file():
        return read(root / "RUN_RESULT.json")
    if not (root / "ATTEMPT.json").is_file():
        return {"status": "not_started", "scientific_execution_started": False}
    state = read(root / "PROGRESS.json") if (root / "PROGRESS.json").is_file() else {"status": "reserved"}
    job = state.get("current_job")
    if job and (root / "run" / job / "progress.json").is_file():
        state["worker_progress"] = read(root / "run" / job / "progress.json")
    state["note"] = "Without a terminal receipt, liveness and exact final counts are uncertain; never relaunch this reservation."
    return state


def worker_entry(token, job_id):
    p, _ = verify_prepared()
    attempt = read(HERE / "ATTEMPT.json")
    if sys.platform != "linux" or Path(sys.prefix).resolve() != Path(EXPECTED_ENV).resolve():
        raise RuntimeError("Scientific workers require the existing isolated Ubuntu environment")
    if token != attempt["token"] or os.getppid() != attempt["supervisor_pid"] or (HERE / "RUN_RESULT.json").exists():
        raise ValueError("Worker does not belong to the active reserved supervisor")
    if job_id not in RUN_ORDER + ["evaluation"]:
        raise ValueError("Unknown job")
    output = HERE / "run" / job_id
    if read(output / "DISPATCH.json")["token"] != token:
        raise ValueError("Dispatch token differs")
    save_new(output / "WORKER_ENTRY.json", {"job": job_id, "pid": os.getpid(), "protocol_sha256": sha(HERE / "protocol.json")})
    if job_id == "evaluation":
        worker = module("_matched_policy_evaluation", HERE / "evaluate_worker.py")
        reports = {name: HERE / "run" / name for name in RUN_ORDER}
        result = worker.run(p, output, reports)
        expected = "completed_bounded_policy_evaluation"
    else:
        worker = module("_matched_policy_training", HERE / "train_worker.py")
        stream, arm = job_id.split("_", 1)
        result = worker.run(p, output, arm, int(stream[4:]))
        expected = "completed_bounded_policy_training"
    if not (output / "report.json").exists():
        save_new(output / "report.json", result)
    return 0 if result.get("status") == expected else 1


def execute(p, manifest):
    if sys.platform != "linux" or Path(sys.prefix).resolve() != Path(EXPECTED_ENV).resolve():
        raise RuntimeError("Use only the existing isolated Ubuntu GPU Python")
    if "include-system-site-packages = false" not in (Path(sys.prefix) / "pyvenv.cfg").read_text():
        raise RuntimeError("Environment must not inherit external packages")
    scoped_process_check()
    started = time.monotonic()
    token = str(uuid.uuid4())
    reserve(HERE, token, started)
    jobs = []
    status = "running"
    error = None
    print("Reserved one policy pilot: six training runs, then one final evaluation; 70-minute total cap. No automatic restart.", flush=True)
    try:
        for job_id in RUN_ORDER + ["evaluation"]:
            remaining = p["total_timeout_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Total pilot wall cap exhausted")
            verify_prepared()
            output = HERE / "run" / job_id
            output.mkdir(exist_ok=False)
            cap = min(p["evaluation_timeout_seconds"] if job_id == "evaluation" else p["worker_timeout_seconds"],
                      p["total_timeout_seconds"] - (time.monotonic() - started))
            if cap <= 0:
                raise TimeoutError("Total pilot wall cap exhausted before dispatch")
            save_new(output / "DISPATCH.json", {"token": token, "job": job_id, "timeout_seconds": cap,
                "supervisor_pid": os.getpid(), "protocol_sha256": sha(HERE / "protocol.json")})
            atomic_json(HERE / "PROGRESS.json", {"status": "running", "current_job": job_id,
                "finished_jobs": len(jobs), "elapsed_seconds": time.monotonic() - started,
                "total_seconds_cap": p["total_timeout_seconds"], "no_automatic_restart": True})
            print("Starting " + job_id + "; worker cap " + str(round(cap, 2)) + " seconds.", flush=True)
            command = [sys.executable, "-B", "-u", str(HERE / "launch.py"), "--worker-token", token, "--job", job_id]
            outcome = supervise(command, output, cap, child_environment(), job_id,
                                deadline=started + p["total_timeout_seconds"])
            report, report_error = read_report(output / "report.json")
            expected = "completed_bounded_policy_evaluation" if job_id == "evaluation" else "completed_bounded_policy_training"
            complete = outcome["exit_code"] == 0 and not outcome["timed_out"] and not outcome["interrupted"] and report and report.get("status") == expected
            job = {"job": job_id, **outcome, "completed": bool(complete),
                "report_sha256": sha(output / "report.json") if report else None,
                "worker_status": report.get("status") if report else None,
                "report_read_error": report_error,
                "execution_counts_known": bool(report and report.get("execution_counts_known", bool(complete))),
                "ledger": report.get("ledger", report.get("counts")) if report else None}
            jobs.append(job)
            save_new(output / "SUPERVISOR_RESULT.json", job)
            print(json.dumps({"job_finished": job_id, "completed": bool(complete), "worker_status": job["worker_status"]}), flush=True)
            if not complete:
                status = "stopped_after_incomplete_job"
                break
        else:
            status = "completed_bounded_matched_policy_pilot"
    except BaseException as exc:
        status = "pilot_interrupted" if isinstance(exc, KeyboardInterrupt) else "pilot_failed"
        error = type(exc).__name__ + ": " + str(exc)
    try:
        verify_prepared()
        identity_unchanged = True
    except Exception as exc:
        identity_unchanged = False
        error = (error or "") + " Prepared identity check: " + str(exc)
        status = "prepared_source_changed"
    evaluation, evaluation_error = read_report(HERE / "run/evaluation/report.json")
    result = {"status": status, "jobs": jobs, "elapsed_seconds": time.monotonic() - started,
        "protocol_sha256": sha(HERE / "protocol.json"), "prepared_identity_unchanged": identity_unchanged,
        "execution_counts_known": len(jobs) == 7 and all(j["execution_counts_known"] for j in jobs),
        "decision": evaluation.get("decision") if evaluation else None, "error": error,
        "evaluation_report_read_error": evaluation_error,
        "full_training_hold": True, "no_automatic_restart": True,
        "interrupted_resume": "Not supported; inspect partial receipts, do not rerun this reservation."}
    save_new(HERE / "RUN_RESULT.json", result)
    print(json.dumps(result), flush=True)
    return 0 if status == "completed_bounded_matched_policy_pilot" and identity_unchanged else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--status", action="store_true")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_status()), flush=True)
        return 0
    if args.worker_token:
        return worker_entry(args.worker_token, args.job)
    p, manifest = verify_prepared()
    verify_inputs()
    if not args.execute:
        print(json.dumps({"status": "prepared_and_source_identities_verified_no_training",
            "protocol_sha256": sha(HERE / "protocol.json"), "training_runs": 6,
            "critic_updates_total": 12000, "actor_updates_total": 6000,
            "auxiliary_updates_total": 12000, "calibration_calls_total": 6,
            "simulation_transition_cap": 22400, "total_seconds_cap": 4200,
            "scientific_execution_started": False}), flush=True)
        return 0
    def stop(signum, frame):
        raise KeyboardInterrupt("Supervisor signal " + str(signum))
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)
    return execute(p, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
