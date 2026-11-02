"""Grouped saved-development prediction checks; default is metadata-only preflight."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PRIOR = BASE / "unifloral_development_mechanism_audit_v1p1"
PRIOR_MANIFEST_SHA256 = "310781618e4eecade8a2c3b1143a2eacb1f9e40c559aa62339d953eb570f941c"
PRIOR_READER_SHA256 = "099f928807618cc4dccd30265f70dc1175e8a6059ae9924d1ff1e15447c88278"
INPUT_ALLOWLIST_SHA256 = "d5b78438a0135a0ae0ec2b567ebf90f0321f85716d0a6c7434df17e4b73883e9"
PROTOCOL_SHA256 = "a2d8a2ac5de8b64f0e57a79be4c54a021a46e1437fd872cfc9b2f72d6ce0dd00"
FREEZE_NAME = "DEFINITIONS_FROZEN.json"
TIMEOUT_SECONDS = 60
SUCCESS = "completed_grouped_saved_development_prediction"
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, parse_constant=invalid)


def write(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def checked(base, relative):
    base = Path(base).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a safe relative authored component")
    path = base / relative
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Missing or redirected file: " + relative)
    for component in (path, *path.parents):
        if component == base:
            break
        if component.is_symlink():
            raise ValueError("Symlinked source component")
    return path.read_bytes()


def manifest_identity(base, expected=None):
    raw = checked(base, "prepared_files.json")
    if expected is not None and sha(raw) != expected:
        raise ValueError("Pinned prepared manifest changed")
    files = read_json(raw).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty prepared file map")
    if base == ROOT and not {"run_checks.py", "test_runner.py", "predictive.py", "protocol.json", FREEZE_NAME}.issubset(files):
        raise ValueError("Grouped-validation manifest omits a required component")
    for name, digest in files.items():
        if sha(checked(base, name)) != digest:
            raise ValueError("Prepared source changed: " + name)
    return {"manifest_sha256": sha(raw), "verified_files": len(files)}


def load_module(name, path, expected):
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError("Module changed before loading")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def assert_input_scope(allowlist):
    if not isinstance(allowlist, dict) or len(allowlist) != 10 or canonical(allowlist) != INPUT_ALLOWLIST_SHA256:
        raise ValueError("Reader inputs differ from the exact ten-file development-only allowlist")


def prior_reader():
    manifest_identity(PRIOR, PRIOR_MANIFEST_SHA256)
    module = load_module("_grouped_width_pinned_development_reader", PRIOR / "analyze_saved.py", PRIOR_READER_SHA256)
    assert_input_scope(module.INPUT_ALLOWLIST)
    return module


def frozen_protocol():
    raw = checked(ROOT, "protocol.json")
    if sha(raw) != PROTOCOL_SHA256:
        raise ValueError("Freeze the grouped protocol before loading numeric development detail")
    protocol = read_json(raw)
    freeze_raw = checked(ROOT, FREEZE_NAME)
    freeze = read_json(freeze_raw)
    if (freeze.get("schema") != "unifloral.grouped_width.freeze.v1"
            or freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("diagnostic_fits_started") is not False
            or freeze.get("detailed_development_inputs_loaded_for_this_stage") is not False
            or freeze.get("prior_development_outcomes_known") is not True
            or freeze.get("phase") != "post_hoc_exploratory_internal_validation"
            or type(freeze.get("confirmation_files_read")) is not int or freeze["confirmation_files_read"] != 0):
        raise ValueError("Grouped definition-freeze receipt does not match protocol")
    if (protocol.get("phase") != "exploratory_grouped_saved_development_only"
            or protocol.get("cpu_timeout_seconds") != TIMEOUT_SECONDS
            or protocol.get("previous_loader_directory") != PRIOR.name
            or protocol.get("previous_loader_sha256") != PRIOR_READER_SHA256
            or protocol.get("previous_manifest_sha256") != PRIOR_MANIFEST_SHA256):
        raise ValueError("Unsupported grouped saved-development scope or budget")
    return protocol, {"protocol_sha256": sha(raw), "freeze_receipt_sha256": sha(freeze_raw)}


def verify():
    """Read/hash/metadata checks only; never fit a model or inspect confirmation."""
    own = manifest_identity(ROOT)
    protocol, protocol_identity = frozen_protocol()
    reader = prior_reader()
    previous_identity, _ = reader.verify()
    assert_input_scope(previous_identity.get("input_hashes"))
    if previous_identity.get("prepared_manifest_sha256") != PRIOR_MANIFEST_SHA256:
        raise ValueError("Prior reader receipt differs from the pinned source manifest")
    return {"grouped_sources": own, "frozen_protocol": protocol_identity,
            "prior_reader_sha256": PRIOR_READER_SHA256, "prior_sources_and_inputs": previous_identity,
            "input_allowlist_sha256": INPUT_ALLOWLIST_SHA256}, protocol


def predictive_module():
    files = read_json(checked(ROOT, "prepared_files.json"))["files"]
    return load_module("_grouped_width_predictive", ROOT / "predictive.py", files["predictive.py"])


def validate_analysis(detail):
    required = {"status": "pass", "claim_scope": "exploratory_grouped_saved_development_prediction_only",
                "phase": "exploratory_grouped_saved_development_only", "new_policy_training": False,
                "new_simulator_calls": 0, "new_gpu_work": 0, "confirmation_files_read": 0,
                "full_training_hold": True, "automatic_training_approval": False,
                "method_or_strength_selected": False}
    if not isinstance(detail, dict) or any(type(detail.get(k)) is not type(v) or detail[k] != v for k, v in required.items()):
        raise ValueError("Predictive result exceeds the frozen grouped diagnostic scope")


def finalize(result):
    valid = (result.get("status") == SUCCESS and result.get("identity_before") is not None
             and result.get("identity_before") == result.get("identity_after"))
    result["diagnostic_result_valid"] = valid
    if not valid:
        if result.get("status") == SUCCESS:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = "Input/source/protocol identities are missing or changed"
        if "analysis" in result:
            result["unvalidated_analysis"] = result.pop("analysis")


def worker(output, expected_identity):
    result = {"started_utc": utc(), "status": "blocked_or_failed", "diagnostic_result_valid": False,
              "scope": "Grouped diagnostic regressions on saved development only; no new policy training or data",
              "new_work": {"policy_optimizer_updates": 0, "policy_forward_calls": 0, "calibrator_updates": 0,
                           "simulator_calls": 0, "gpu_work": 0, "candidate_actions_generated": 0,
                           "confirmation_files_read": 0}}
    with (output / "stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"utc": utc(), "stage": name}) + "\n")
            stages.flush()
        try:
            stage("frozen_protocol_source_and_input_preflight")
            result["identity_before"], protocol = verify()
            if canonical(result["identity_before"]) != expected_identity:
                raise ValueError("Worker identity differs from supervisor preflight")
            import numpy as np
            if platform.system() != "Windows" or platform.python_version() != "3.12.14" or np.__version__ != "2.3.5":
                raise ValueError("Expected existing Windows Python3.12.14 NumPy2.3.5 CPU analysis runtime")
            result["runtime"] = {"platform": platform.system(), "python": platform.python_version(),
                                 "numpy": np.__version__, "executable": sys.executable}
            stage("load_exact_prior_validated_development_batches")
            batches = prior_reader().load_batches()
            result["validated_inputs"] = {"batches": 2, "states": 128, "saved_candidate_rows": 1152,
                                          "loader": "Exact reviewed v1p1 development-only loader"}
            stage("fit_and_score_frozen_grouped_diagnostic_models")
            result["analysis"] = predictive_module().analyze(batches, protocol)
            validate_analysis(result["analysis"])
            result["status"] = SUCCESS
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            stage("source_protocol_and_saved_input_postchecks")
            try:
                result["identity_after"], _ = verify()
                if result.get("identity_before") != result["identity_after"]:
                    raise ValueError("Source/protocol/saved inputs changed during grouped validation")
            except Exception as error:
                result["status"] = "blocked_or_failed"
                result["postcheck_error"] = str(error)
            finalize(result)
            result["finished_utc"] = utc()
            write(output / "checks.json", result)
    return 0 if result["diagnostic_result_valid"] else 1


def run_child(command, output, env):
    """Only the newly created CPU child is bounded/killed; no process discovery."""
    result = {}
    start = time.monotonic()
    with (output / "checks.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            result["exit_code"] = process.wait(timeout=TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
            result["interrupted"] = isinstance(error, KeyboardInterrupt)
            process.kill()
            result["exit_code"] = process.wait()
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Frozen metadata/hash gates only; default")
    modes.add_argument("--execute", action="store_true", help="Run the frozen CPU grouped-development diagnostic")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-identity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        output = args.worker_output.resolve()
        if (args.execute or args.preflight or not args.expected_identity or output.parent != ROOT
                or not output.name.startswith("results_") or not output.is_dir()):
            parser.error("Worker requires an exclusive local supervisor output and matching identity")
        return worker(output, args.expected_identity)
    if args.expected_identity:
        parser.error("Worker identity requires worker-output")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"started_utc": utc(), "mode": "execute" if args.execute else "preflight", "status": "blocked_or_failed",
              "diagnostic_result_valid": False, "cpu_worker_timeout_seconds": TIMEOUT_SECONDS,
              "timeout_scope": "Child wall-clock wait capped at60seconds; supervisor hashing/postchecks outside that cap"}
    try:
        result["identity_before"], _ = verify()
        if not args.execute:
            result["status"] = "preflight_passed"
        else:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-output", str(output),
                       "--expected-identity", canonical(result["identity_before"])]
            result.update(run_child(command, output, env))
            if (output / "checks.json").is_file():
                raw = (output / "checks.json").read_bytes()
                child = read_json(raw)
                result["checks_sha256"] = sha(raw)
                for key in ("runtime", "validated_inputs", "new_work", "error", "postcheck_error"):
                    if key in child:
                        result[key] = child[key]
                if (result["exit_code"] == 0 and child.get("status") == SUCCESS and child.get("diagnostic_result_valid") is True
                        and child.get("identity_before") == result["identity_before"] == child.get("identity_after")
                        and not result.get("timed_out") and not result.get("interrupted")):
                    validate_analysis(child.get("analysis"))
                    result["status"] = SUCCESS
            else:
                result["worker_outcome"] = "No complete diagnostic receipt; preserve partial stage/log evidence without claiming a result"
    except BaseException as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            result["identity_after"], _ = verify()
            if result.get("identity_before") != result["identity_after"]:
                raise ValueError("Saved inputs or prepared source/protocol changed during supervision")
        except Exception as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = str(error)
        finalize(result)
        result["finished_utc"] = utc()
        write(output / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json")}))
    return 0 if result["status"] in ("preflight_passed", SUCCESS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
