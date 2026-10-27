"""Reserve one audit attempt, then invoke the unchanged frozen bounded launcher."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent / "titan_20261026"
AUDIT = HANDOFF / "payload/unifloral_operator_alignment_v1"
TARGET = Path("/home/david/bca-operator-audit-20261026")
READY_NAME = "readiness_20261026T220834_752076Z.json"
READY_SHA = "d7cecd21fd4e8106c7c385a2188fa5e37dfd47ee52feae7e4aa286cc038b7b08"
MANIFEST_SHA = "28df1b6bbffacda0a6659a203bf9aab2a2e82a259d3621642cdae4ad44c73adf"
SUCCESS = "completed_frozen_development_operator_alignment"
EXPECTED_PACKAGES = {"jax": "0.6.2", "jaxlib": "0.6.2", "jax-cuda12-plugin": "0.6.2",
    "jax-cuda12-pjrt": "0.6.2", "numpy": "2.5.4", "flax": "0.10.7",
    "optax": "0.2.8", "distrax": "0.1.5", "chex": "0.1.90",
    "tensorflow-probability": "0.25.0", "scipy": "1.18.1"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(name, path, digest=None):
    if digest is not None and sha(path) != digest:
        raise ValueError("Source changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def validate_readiness(ready, package, identity, python, model_path):
    if ready.get("status") != "ready_for_reserved_bounded_audit":
        raise ValueError("Successful readiness receipt required")
    if ready.get("package") != package or ready.get("identity") != identity:
        raise ValueError("Readiness source/checkpoint identity no longer matches")
    runtime, interpreter, model = (ready.get(k, {}) for k in ("runtime", "interpreter", "model"))
    if (runtime.get("executable") != str(python) or interpreter.get("executable") != str(python)
            or interpreter.get("prefix") != str(python.parent.parent)
            or interpreter.get("include_system_site_packages") is not False):
        raise ValueError("Readiness interpreter/prefix differs")
    if (runtime.get("packages") != EXPECTED_PACKAGES or runtime.get("backend") != "gpu"
            or not runtime.get("devices") or runtime.get("matmul_precision") != "highest"
            or runtime.get("jax_disable_jit") is not False or runtime.get("jax_enable_x64") is not False
            or runtime.get("environment", {}).get("XLA_FLAGS") != "--xla_gpu_triton_gemm_any=True"):
        raise ValueError("Readiness runtime did not pass the frozen gate")
    if (model.get("module") != "unifloral_pinned_td3bc_computational_slice"
            or Path(model.get("file", "")).resolve() != model_path.resolve()
            or model.get("sha256") != sha(model_path)
            or model.get("native_revision") != "f2dc1278eae18ed3e22c92255119c54885414d17"):
        raise ValueError("Readiness UniFloral model origin differs")


def existing_attempts(audit):
    return sorted(p for p in audit.glob("results_*") if p.is_dir())


def execute_once(root, audit, request, command, process_check, launch):
    """Pure filesystem reservation boundary; launch/process checks are injectable for fixtures."""
    for name in ("OPERATOR_ATTEMPT_RESULT.json", "operator_once.log"):
        prior = root / name
        if prior.exists() or prior.is_symlink():
            raise RuntimeError("Prior wrapper evidence requires inspection; no launch")
    if existing_attempts(audit) or process_check():
        raise RuntimeError("Existing audit attempt/process requires inspection; no launch")
    reservation = root / "OPERATOR_ATTEMPT.json"
    with reservation.open("x", encoding="utf-8") as stream:
        json.dump(request, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    result = {"status": "reserved_not_launched", "reservation": str(reservation),
        "launch_dispatched": False, "execution_counts_known": False}
    try:
        # Recheck after reservation to catch an already-starting legacy direct launch.
        if existing_attempts(audit) or process_check():
            raise RuntimeError("Audit appeared after reservation; reservation preserved, no launch")
        print("Reserved one audit attempt. Frozen worker limit: 300 seconds; maximum transitions: 1152.", flush=True)
        with (root / "operator_once.log").open("x", encoding="utf-8") as log:
            result["launch_dispatched"] = True
            result["launcher_exit_code"] = launch(command, log)
        attempts = existing_attempts(audit)
        if len(attempts) != 1 or not (attempts[0] / "report.json").is_file():
            raise RuntimeError("Expected one terminal frozen report; outcome requires inspection")
        report_path = attempts[0] / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result.update(audit_report=str(report_path), audit_report_sha256=sha(report_path),
            audit_status=report.get("status"), execution_counts_known=report.get("execution_counts_known", False),
            ledger=report.get("ledger"), worker_timeout_seconds=report.get("worker_timeout_seconds"))
        result["status"] = SUCCESS if result["launcher_exit_code"] == 0 and report.get("status") == SUCCESS else "audit_failed_preserved"
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="interrupted_outcome_requires_inspection" if isinstance(error, KeyboardInterrupt)
            else "failed_or_uncertain_preserved", error=type(error).__name__ + ": " + str(error))
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with (root / "OPERATOR_ATTEMPT_RESULT.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    # Never remove/reuse the reservation, including on failure, timeout or interruption.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Reserve and launch the already authorized bounded audit once")
    args = parser.parse_args()
    if not args.execute:
        print("Prepared for " + READY_NAME + "; use --execute for one reserved bounded audit")
        return 0
    python = TARGET / "bin/python"
    if sys.platform != "linux" or Path(sys.executable).absolute() != python or Path(sys.prefix).resolve() != TARGET.resolve():
        parser.error("Use the verified new Ubuntu environment")
    if (ROOT / "OPERATOR_ATTEMPT.json").exists() or (ROOT / "OPERATOR_ATTEMPT.json").is_symlink():
        parser.error("Attempt already reserved; inspect preserved evidence rather than relaunching")
    ready_path = ROOT / READY_NAME
    if sha(ready_path) != READY_SHA:
        raise ValueError("The reviewed readiness receipt changed")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    manifest_path = HANDOFF / "HANDOFF_MANIFEST.json"
    if sha(manifest_path) != MANIFEST_SHA:
        raise ValueError("Titan handoff manifest changed")
    files = {i["path"]: i["sha256"] for i in json.loads(manifest_path.read_text(encoding="utf-8"))["files"]}
    restorer = load_module("_once_restorer", HANDOFF / "restore_assets.py", files["restore_assets.py"])
    package = restorer.verify_package(HANDOFF)
    os.environ["BCA_TRANSFER_BUNDLE_ROOT"] = str(HANDOFF / "payload/transfer_bundle")
    runner = load_module("_once_frozen_runner", AUDIT / "run_checks.py", files["payload/unifloral_operator_alignment_v1/run_checks.py"])
    identity, protocol = runner.verify()
    runner.execution_environment(dict(os.environ))  # Preserve the existing conflicting-XLA-flags stop rule.
    model_path = HANDOFF / "payload/unifloral_td3bc_validation_v1/upstream/algorithms/td3_bc.py"
    validate_readiness(ready, package, identity, python, model_path)
    probe = load_module("_once_scoped_process_check", ROOT / "probe_runtime_identity.py")
    command = [str(python), "-B", "-u", str(HANDOFF / "launch.py"), "--execute", "--allow-outside-upstream-constraints"]
    request = {"schema": "bca.zeus.operator_once.v1", "utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "readiness": READY_NAME, "readiness_sha256": READY_SHA,
        "manifest_sha256": MANIFEST_SHA, "protocol_sha256": identity["protocol_sha256"],
        "worker_timeout_seconds": protocol["worker_timeout_seconds"], "max_transition_steps": protocol["max_transition_steps"],
        "command": command, "scope": "One frozen development audit; no training, fitting or confirmation"}
    def launch(argv, log):
        return subprocess.run(argv, cwd=HANDOFF, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    result = execute_once(ROOT, AUDIT, request, command, probe.audit_processes, launch)
    print(json.dumps({k: result.get(k) for k in ("status", "audit_status", "audit_report", "execution_counts_known", "error")}), flush=True)
    return 0 if result["status"] == SUCCESS else 1


if __name__ == "__main__":
    raise SystemExit(main())
