"""One reserved auxiliary-only BCA validation; --preflight performs no fitting."""

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
import uuid

HERE = Path(__file__).resolve().parent
HANDOFF = HERE.parents[1] / "titan_20261026"
EXPECTED_ENV = "/home/david/bca-operator-audit-20261026"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_prepared(root=HERE):
    root = Path(root)
    manifest = json.loads((root / "PREPARED.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        path = root / name
        if Path(name).name != name or not path.is_file() or sha(path) != digest:
            raise ValueError("Prepared file changed or missing: " + name)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["auxiliary_updates"] != 128 or protocol["worker_timeout_seconds"] != 300:
        raise ValueError("Fixed budget changed")
    if manifest.get("real_fit_or_calibration_executed_during_preparation") is not False:
        raise ValueError("Preparation must not execute real fitting/calibration")
    for name in ("MATH_REVIEW.json", "data_provenance.json", "LAUNCH_TESTS.json", "IMPLEMENTATION_REVIEW.json"):
        if not (root / name).is_file():
            raise ValueError("Missing preparation evidence: " + name)
    return protocol, manifest


def reserve(root, token):
    root = Path(root)
    if (root / "RUN_RESULT.json").exists() or (root / "run").exists():
        raise ValueError("Prior result/run exists; inspect it, do not rerun")
    save_new(root / "ATTEMPT.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "token": token,
        "status": "reserved_once", "pid": os.getpid(),
    })
    (root / "run").mkdir(exist_ok=False)


def scoped_process_check():
    targets = {(HERE / "launch.py").resolve(), (HANDOFF / "launch.py").resolve(),
               (HANDOFF / "payload/unifloral_operator_alignment_v1/run_checks.py").resolve()}
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            if not (process / "comm").read_text().strip().startswith(("python", "pypy")):
                continue
            for raw in (process / "cmdline").read_bytes().split(b"\0")[1:]:
                name = os.fsdecode(raw)
                if Path(name).name not in ("launch.py", "run_checks.py"):
                    continue
                path = Path(name)
                if not path.is_absolute():
                    path = (process / "cwd").resolve(strict=True) / path
                if path.resolve() in targets:
                    matches.append(int(process.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise RuntimeError("Scoped process inspection denied; no execution") from error
    if matches:
        raise RuntimeError("An existing scoped BCA process is running: " + str(matches))


def child_environment():
    env = dict(os.environ)
    env.update(BCA_TRANSFER_BUNDLE_ROOT=str(HANDOFF / "payload/transfer_bundle"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", JAX_PLATFORMS="cuda",
               JAX_DEFAULT_MATMUL_PRECISION="highest", JAX_ENABLE_X64="false",
               XLA_FLAGS="--xla_gpu_triton_gemm_any=True")
    return env


def supervise(command, run_dir, timeout_seconds, env=None):
    """Bound one child. Fixture callers supply synthetic commands only."""
    run_dir = Path(run_dir)
    with (run_dir / "worker.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=run_dir, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=(os.name != "nt"))
        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "nt":
                process.kill()  # Synthetic Windows fixture child; real execution is Linux-only.
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        except BaseException as error:
            # The real worker owns a new session: Ctrl-C on the supervisor must
            # not leave it detached from its wall-time watchdog.
            if process.poll() is None:
                if os.name == "nt":
                    process.kill()
                else:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            process.wait(timeout=10)
            return {"exit_code": process.returncode, "timed_out": False,
                    "interrupted": True, "error": type(error).__name__ + ": " + str(error)}
    return {"exit_code": process.returncode, "timed_out": timed_out}


def worker_entry(token):
    protocol, _ = verify_prepared()
    attempt = json.loads((HERE / "ATTEMPT.json").read_text(encoding="utf-8"))
    if attempt["token"] != token or not (HERE / "run").is_dir() or (HERE / "RUN_RESULT.json").exists():
        raise ValueError("Worker does not match reserved attempt")
    worker = load_module("_bca_one_sided_worker", HERE / "worker.py")
    result = worker.run(protocol, HERE / "run")
    if not (HERE / "run/report.json").exists():
        save_new(HERE / "run/report.json", result)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_token:
        return worker_entry(args.worker_token)
    protocol, manifest = verify_prepared()
    if not args.execute:
        print(json.dumps({"status": "prepared_files_verified_no_fit", "protocol_sha256": sha(HERE / "protocol.json"),
                          "auxiliary_update_cap": 128, "worker_seconds_cap": 300, "real_fit_started": False,
                          "note": "GPU restoration and runtime checks occur before any update inside the manual worker."}))
        return 0
    if sys.platform != "linux" or Path(sys.prefix).resolve() != Path(EXPECTED_ENV).resolve():
        raise RuntimeError("Execute only with the existing isolated Ubuntu GPU environment")
    if "include-system-site-packages = false" not in (Path(sys.prefix) / "pyvenv.cfg").read_text():
        raise RuntimeError("Isolated environment may not inherit site packages")
    scoped_process_check()
    token = str(uuid.uuid4())
    reserve(HERE, token)
    print("Reserved one auxiliary-only validation: 128 updates, one calibration, 300-second worker cap.", flush=True)
    command = [sys.executable, "-B", "-u", str(HERE / "launch.py"), "--worker-token", token]
    try:
        outcome = supervise(command, HERE / "run", protocol["worker_timeout_seconds"], child_environment())
    except BaseException as error:
        outcome = {"exit_code": None, "timed_out": False, "supervisor_error": type(error).__name__ + ": " + str(error)}
    report_path = HERE / "run/report.json"
    report_read_error = None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
    except (OSError, json.JSONDecodeError) as error:
        report = None
        report_read_error = type(error).__name__ + ": " + str(error)
    identity_ok = all(sha(HERE / name) == digest for name, digest in manifest["files"].items())
    status = ("worker_interrupted" if outcome.get("interrupted") else
              "supervisor_failed" if outcome.get("supervisor_error") else
              "worker_timed_out" if outcome["timed_out"] else
              "worker_failed" if outcome["exit_code"] else "worker_finished")
    if not identity_ok:
        status = "prepared_source_changed_during_run"
    result = {"status": status, **outcome, "prepared_identity_unchanged": identity_ok,
              "protocol_sha256": sha(HERE / "protocol.json"), "report": str(report_path) if report else None,
              "report_sha256": sha(report_path) if report else None,
              "worker_status": report.get("status") if report else None,
              "worker_decision": report.get("decision") if report else None,
              "report_read_error": report_read_error,
              "execution_counts_known": bool(report and report.get("ledger")),
              "full_training_hold": True, "no_automatic_retry": True}
    save_new(HERE / "RUN_RESULT.json", result)
    print(json.dumps(result), flush=True)
    return 0 if outcome["exit_code"] == 0 and not outcome.get("interrupted") and not outcome["timed_out"] and identity_ok and report else 1


if __name__ == "__main__":
    raise SystemExit(main())
