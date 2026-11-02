"""Preflight by default; explicitly acknowledged, bounded frozen GPU audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PROTOCOL_SHA = "b30d1c5cb88004135070f12050bb97b3b1059f436017f25ca005bf52087f3aa2"
XLA = "--xla_gpu_triton_gemm_any=True"
TIMEOUT_SECONDS = 300
SUCCESS = "completed_frozen_development_operator_alignment"
sys.dont_write_bytecode = True


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checked(base, relative):
    base = Path(base).resolve()
    relative = Path(relative)
    path = base / relative
    if (relative.is_absolute() or ".." in relative.parts or ":" in str(relative)
            or path.is_symlink() or not path.resolve().is_relative_to(base) or not path.is_file()):
        raise ValueError("Unsafe or missing source: " + str(path))
    return path.read_bytes()


def module(name, path, expected):
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError("Source changed before module import: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    try:
        exec(compile(raw, str(path), "exec"), value.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return value


def verify():
    """Standard-library metadata/hash checks; no numerical outcomes or forwards."""
    manifest_raw = checked(ROOT, "prepared_files.json")
    manifest = read_json(manifest_raw)
    files = manifest.get("files")
    required = {"run_checks.py", "checkpoint_io.py", "gpu_execution.py", "operator_audit.py",
                "protocol.json", "DEFINITIONS_FROZEN.json"}
    if not isinstance(files, dict) or not required.issubset(files):
        raise ValueError("Prepared manifest lacks the required audit sources")
    for relative, wanted in files.items():
        if sha(checked(ROOT, relative)) != wanted:
            raise ValueError("Prepared source changed: " + relative)
    raw = checked(ROOT, "protocol.json")
    if sha(raw) != PROTOCOL_SHA:
        raise ValueError("Frozen operator protocol changed")
    protocol = read_json(raw)
    freeze = read_json(checked(ROOT, "DEFINITIONS_FROZEN.json"))
    if (freeze.get("schema") != "unifloral.operator_alignment.freeze.v1"
            or freeze.get("protocol_sha256") != PROTOCOL_SHA
            or freeze.get("new_evaluator_calls") != 0 or freeze.get("confirmation_files_read") != 0
            or protocol.get("worker_timeout_seconds") != TIMEOUT_SECONDS):
        raise ValueError("Prospective definition freeze or timeout differs")
    checkpoint = module("_operator_supervisor_checkpoint", ROOT / "checkpoint_io.py", files["checkpoint_io.py"])
    execution = module("_operator_supervisor_execution", ROOT / "gpu_execution.py", files["gpu_execution.py"])
    return {"prepared_manifest_sha256": sha(manifest_raw), "own_files": files,
            "protocol_sha256": PROTOCOL_SHA,
            "definition_freeze_sha256": sha(checked(ROOT, "DEFINITIONS_FROZEN.json")),
            "checkpoint": checkpoint.verify(protocol), "execution": execution.verify(protocol)}, protocol


def execution_environment(current):
    if current.get("XLA_FLAGS", "") not in ("", XLA):
        raise ValueError("Conflicting XLA_FLAGS; no override attempted")
    return {**current, "JAX_PLATFORMS": "cuda", "JAX_DEFAULT_MATMUL_PRECISION": "highest",
            "XLA_FLAGS": XLA, "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def runtime(identity):
    original = ROOT.parent / "unifloral_checkpoint_v1"
    # Original run.py is covered by checkpoint.verify's exact manifest closure.
    manifest = read_json(checked(original, "prepared_files.json"))
    helper = module("_operator_supervisor_original_runtime", original / "run.py", manifest["files"]["run.py"])
    return helper.runtime()


def validate_execution(execution):
    ledger = execution.get("ledger", {})
    exact = {"status": "completed_bounded_operator_execution", "transition_rows_attempted": 1152,
        "transition_rows_completed": 1152, "max_transition_rows": 1152, "full_continuations": 0,
        "current_actor_rows": 128, "current_critic_rows": 1408, "width_rows": 1408,
        "target_actor_rows": 1408, "target_critic_rows": 5632, "max_target_critic_rows": 5632,
        "native_smoothing_draws": 8, "max_native_smoothing_draws": 8,
        "fitting_transition_rows_read": 256, "new_diagnostic_fits": 0,
        "policy_optimizer_updates": 0, "calibrator_updates": 0, "new_candidate_actions": 0,
        "confirmation_files_read": 0, "checkpoint_rng_advanced": False,
        "native_smoothing_draws_are_new_not_original_refresh_replay": True}
    if any(type(ledger.get(k)) is not type(v) or ledger.get(k) != v for k, v in exact.items()):
        raise ValueError("Execution ledger did not satisfy the exact frozen work budget")
    before, after = execution.get("snapshot_before"), execution.get("snapshot_after")
    if (not isinstance(before, dict) or before != after
            or before.get("full_state_unchanged") is not True or before.get("inference_unchanged") is not True):
        raise ValueError("Frozen full-state checks did not pass")
    replays = execution.get("forward_replays")
    if (not isinstance(replays, list) or len(replays) != 3
            or any(row.get("passed") is not True or row.get("rtol") != 2e-5 or row.get("atol") != 2e-6 for row in replays)):
        raise ValueError("Required current actor/Q/U forward replays did not pass")
    return ledger


def validate_analysis(analysis, protocol):
    expected = {"status": "pass", "phase": protocol["phase"], "claim_scope": protocol["claim_scope"],
        "full_training_hold": True, "automatic_training_approval": False, "new_policy_training": False,
        "new_diagnostic_fits": 0, "policy_optimizer_updates": 0, "calibrator_updates": 0,
        "confirmation_files_read": 0, "method_or_strength_selected": False}
    if any(type(analysis.get(k)) is not type(v) or analysis.get(k) != v for k, v in expected.items()):
        raise ValueError("Analysis scope or completion flags differ")


def save_banks(output, development, fitting):
    """Lossless numeric NPZ plus JSON tree: no pickle or mutation of inputs."""
    import numpy as np
    arrays = {}
    def visit(value):
        if isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or not np.all(np.isfinite(value)):
                raise ValueError("Raw bank contains unsupported or nonfinite array")
            key = "array_" + str(len(arrays))
            arrays[key] = value
            return {"npz_key": key, "dtype": value.dtype.str, "shape": list(value.shape),
                    "array_bytes_sha256": sha(np.ascontiguousarray(value).tobytes())}
        if isinstance(value, np.generic):
            return visit(value.item())
        if isinstance(value, dict):
            return {str(k): visit(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(v) for v in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Unsupported raw bank metadata type: " + type(value).__name__)
    tree = {"schema": "unifloral.operator_alignment.raw_banks.v1",
            "development": visit(development), "fitting": visit(fitting)}
    encoded(tree)  # Validate all metadata before reserving output files.
    with (output / "raw_banks.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write(output / "raw_banks.json", tree)
    return {name: {"sha256": sha((output / name).read_bytes()), "bytes": (output / name).stat().st_size}
            for name in ("raw_banks.npz", "raw_banks.json")}


def validate_worker_request(output, manifest_sha):
    output = Path(output).resolve()
    if output.parent != ROOT.resolve() or not output.name.startswith("results_") or not output.is_dir() or output.is_symlink():
        raise ValueError("Worker output must be this suite's newly reserved results directory")
    request = read_json(checked(output, "REQUEST.json"))
    if (request.get("schema") != "unifloral.operator_alignment.request.v1"
            or request.get("protocol_sha256") != PROTOCOL_SHA
            or request.get("prepared_manifest_sha256") != manifest_sha
            or request.get("output_name") != output.name
            or request.get("allow_outside_upstream_constraints") is not True):
        raise ValueError("Worker reservation does not match the explicit supervisor request")
    return request


def worker(output, manifest_sha):
    output = Path(output).resolve()
    request = validate_worker_request(output, manifest_sha)
    write(output / "WORKER_STARTED.json", {"utc": datetime.now(timezone.utc).isoformat(), "request": request})
    result = {"status": "blocked_or_failed", "protocol_sha256": PROTOCOL_SHA,
              "execution_dispatched": False, "execution_counts_known": False,
              "scientific_claim_scope": "descriptive_native_residual_target_alignment_not_efficacy"}
    started = time.monotonic()
    with (output / "stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"stage": name, "elapsed_seconds": time.monotonic()-started}) + "\n")
            stages.flush()
        try:
            stage("source_precheck")
            identity, protocol = verify()
            result["identity_before"] = identity
            if identity["prepared_manifest_sha256"] != manifest_sha:
                raise ValueError("Prepared manifest changed after child dispatch")
            stage("runtime_gate")
            result["runtime"] = runtime(identity)
            result["runtime_verified"] = True
            execution_module = module("_operator_worker_execution", ROOT / "gpu_execution.py", identity["own_files"]["gpu_execution.py"])
            analysis_module = module("_operator_worker_analysis", ROOT / "operator_audit.py", identity["own_files"]["operator_audit.py"])
            stage("bounded_execution")
            result["execution_dispatched"] = True
            execution = execution_module.run(protocol)
            result["ledger"] = execution["ledger"]
            result["execution_counts_known"] = True
            validate_execution(execution)
            stage("preserve_raw_banks")
            result["raw_bank_artifacts"] = save_banks(output, execution["development"], execution["fitting"])
            execution_summary = {key: value for key, value in execution.items() if key not in ("development", "fitting")}
            write(output / "execution.json", execution_summary)
            stage("fixed_descriptive_analysis")
            analysis = analysis_module.analyze(execution["development"], execution["fitting"], protocol)
            validate_analysis(analysis, protocol)
            # Detect any change to the persisted raw bank files after analysis.
            for name, expected in result["raw_bank_artifacts"].items():
                if sha((output / name).read_bytes()) != expected["sha256"]:
                    raise ValueError("Saved raw bank artifact changed during analysis")
            write(output / "checks.json", {"status": "pass", "analysis": analysis,
                  "execution": execution_summary, "raw_bank_artifacts": result["raw_bank_artifacts"]})
            result.update(status=SUCCESS, analysis_passed=True, work_budget_verified=True,
                          frozen_state_verified=True, checks_sha256=sha((output / "checks.json").read_bytes()),
                          full_training_hold=True, automatic_training_approval=False)
            stage("completed")
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            if hasattr(error, "ledger"):
                result["ledger"] = error.ledger
                result["execution_counts_known"] = True
            stage("failed")
        try:
            identity_after, _ = verify()
            result["identity_after"] = identity_after
            if result.get("identity_before") != identity_after:
                raise ValueError("Source/checkpoint/input identity changed")
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = {"type": type(error).__name__, "message": str(error)}
        result["elapsed_seconds"] = time.monotonic()-started
        write(output / "worker.json", result)
    return 0 if result["status"] == SUCCESS else 1


def finalize_report(output, before, code, timed_out, elapsed, after=None, postcheck_error=None):
    path = output / "worker.json"
    receipt_valid = False
    report = {"status": "worker_did_not_complete"}
    try:
        if path.is_file():
            value = read_json(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("Worker receipt must be a JSON object")
            report, receipt_valid = value, True
    except Exception as error:
        # Termination can interrupt the child's exclusive write mid-JSON.
        report["receipt_parse_error"] = {"type": type(error).__name__, "message": str(error)}
    report.update(exit_code=code, timed_out=timed_out, supervisor_elapsed_seconds=elapsed,
                  supervisor_identity_before=before, supervisor_identity_after=after,
                  worker_timeout_seconds=TIMEOUT_SECONDS)
    if postcheck_error:
        report["supervisor_postcheck_error"] = postcheck_error
    if timed_out or code != 0 or after != before or report.get("status") != SUCCESS:
        report["status"] = "timed_out" if timed_out else "blocked_or_failed"
    if timed_out or not receipt_valid:
        # A killed child can have dispatched work without flushing its ledger.
        report["execution_counts_known"] = False
        report["actual_work_after_last_receipt"] = "unknown; do not infer zero from an absent/incomplete ledger"
        report["work_budget_verified"] = False
        report["frozen_state_verified"] = False
    if report["status"] == SUCCESS:
        if (report.get("identity_before") != before or report.get("identity_after") != before
                or report.get("runtime_verified") is not True or report.get("analysis_passed") is not True
                or report.get("work_budget_verified") is not True or report.get("frozen_state_verified") is not True
                or not (output / "checks.json").is_file()
                or sha((output / "checks.json").read_bytes()) != report.get("checks_sha256")):
            report["status"] = "blocked_or_failed"
            report["completion_gate_error"] = "Incomplete runtime/source/work/frozen/artifact evidence"
    report["full_training_hold"] = True
    report["automatic_training_approval"] = False
    return report


def supervise(identity, protocol):
    env = execution_environment(dict(os.environ))
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    write(output / "REQUEST.json", {"schema": "unifloral.operator_alignment.request.v1",
          "protocol_sha256": PROTOCOL_SHA, "prepared_manifest_sha256": identity["prepared_manifest_sha256"],
          "output_name": output.name, "allow_outside_upstream_constraints": True})
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--execute",
               "--allow-outside-upstream-constraints", "--worker-output", str(output),
               "--prepared-manifest-sha256", identity["prepared_manifest_sha256"]]
    print("Starting bounded frozen operator audit; output: " + str(output), flush=True)
    started = time.monotonic()
    code, timed_out, launch_error = None, False, None
    try:
        with (output / "stdout.log").open("x", encoding="utf-8") as stdout, (output / "stderr.log").open("x", encoding="utf-8") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            try:
                code = process.wait(timeout=TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()  # Only this supervisor's bounded child.
                code = process.wait(timeout=30)
            except BaseException:
                process.kill()
                code = process.wait(timeout=30)
                raise
    except BaseException as error:
        launch_error = {"type": type(error).__name__, "message": str(error)}
    after, postcheck_error = None, None
    try:
        after, _ = verify()
    except BaseException as error:
        postcheck_error = {"type": type(error).__name__, "message": str(error)}
    report = finalize_report(output, identity, code, timed_out, time.monotonic()-started, after, postcheck_error)
    if launch_error:
        report["launch_or_wait_error"] = launch_error
        report["status"] = "blocked_or_failed"
    write(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"),
                      "execution_counts_known": report.get("execution_counts_known", False)}), flush=True)
    return 0 if report["status"] == SUCCESS else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--prepared-manifest-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.execute and not args.allow_outside_upstream_constraints:
        parser.error("--execute requires explicit --allow-outside-upstream-constraints")
    if args.worker_output:
        if not args.execute or not args.prepared_manifest_sha256:
            parser.error("Hidden worker requires the supervisor's execute request and manifest identity")
        return worker(args.worker_output, args.prepared_manifest_sha256)
    identity, protocol = verify()
    if not args.execute:
        print(json.dumps({"status": "preflight_pass", "scientific_execution": False, "identity": identity}), flush=True)
        return 0
    return supervise(identity, protocol)


if __name__ == "__main__":
    raise SystemExit(main())
