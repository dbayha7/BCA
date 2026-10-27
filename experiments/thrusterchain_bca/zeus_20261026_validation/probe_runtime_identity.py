"""Readiness only: exact GPU gate and UniFloral definition origins; no forwards or audit."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
HANDOFF = ROOT.parent / "titan_20261026"
AUDIT = HANDOFF / "payload/unifloral_operator_alignment_v1"
STAGE1 = HANDOFF / "payload/unifloral_td3bc_validation_v1"
TARGET = Path("/home/david/bca-operator-audit-20261026")
MANIFEST_SHA = "28df1b6bbffacda0a6659a203bf9aab2a2e82a259d3621642cdae4ad44c73adf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path, wanted):
    if sha(path) != wanted:
        raise ValueError("Source hash changed: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def audit_processes():
    targets = {(HANDOFF / "launch.py").resolve(), (AUDIT / "run_checks.py").resolve()}
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            comm = (process / "comm").read_text().strip()
            if not comm.startswith(("python", "pypy")):
                continue
            argv = (process / "cmdline").read_bytes().split(b"\0")
            for value in argv[1:]:
                arg = os.fsdecode(value)
                if Path(arg).name not in ("launch.py", "run_checks.py"):
                    continue
                path = Path(arg)
                if not path.is_absolute():
                    path = (process / "cwd").resolve(strict=True) / path
                if path.resolve() in targets:
                    matches.append(int(process.name))
                    break
        except (FileNotFoundError, ProcessLookupError):
            continue  # Process exited during the read-only scan.
        except PermissionError as error:
            raise RuntimeError("Scoped process check incomplete; permission denied") from error
    return matches


def main():
    if sys.platform != "linux" or Path(sys.prefix).resolve() != TARGET.resolve():
        raise RuntimeError("Run with the new isolated Ubuntu venv; no other interpreter accepted")
    output = ROOT / ("readiness_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + ".json")
    result = {"status": "blocked_or_failed", "scientific_execution": False,
        "evaluator_transitions": 0, "model_forward_rows": 0, "optimizer_updates": 0,
        "calibration_updates": 0, "confirmation_reads": 0, "audit_started": False}
    try:
        setup = json.loads((TARGET / "SETUP_RESULT.json").read_text(encoding="utf-8"))
        if setup.get("status") != "installed_pending_gpu_and_model_identity_checks":
            raise ValueError("A successful isolated setup receipt is required")
        if "include-system-site-packages = false" not in (TARGET / "pyvenv.cfg").read_text():
            raise ValueError("Environment must not inherit system site packages")
        attempts = sorted(p.name for p in AUDIT.glob("results_*") if p.is_dir())
        if attempts:
            raise RuntimeError("Prior audit attempts require inspection: " + ", ".join(attempts))
        pids = audit_processes()
        result["scoped_audit_process_check"] = {"matching_pids": pids,
            "scope": "Current user and current Ubuntu PID namespace; a snapshot, not an execution lock"}
        if pids:
            raise RuntimeError("Existing audit process detected; no probe attempted")
        manifest_path = HANDOFF / "HANDOFF_MANIFEST.json"
        if sha(manifest_path) != MANIFEST_SHA:
            raise ValueError("Titan handoff manifest changed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = {item["path"]: item["sha256"] for item in manifest["files"]}
        restorer = module("_readiness_restorer", HANDOFF / "restore_assets.py", files["restore_assets.py"])
        result["package"] = restorer.verify_package(HANDOFF)
        runner = module("_readiness_frozen_runner", AUDIT / "run_checks.py",
            files["payload/unifloral_operator_alignment_v1/run_checks.py"])
        os.environ["BCA_TRANSFER_BUNDLE_ROOT"] = str(HANDOFF / "payload/transfer_bundle")
        os.environ.update(runner.execution_environment(dict(os.environ)))
        identity, _ = runner.verify()
        result["identity"] = identity
        # Reuses the exact preserved gate, including package versions and GPU settings.
        result["runtime"] = runner.runtime(identity)
        loader = module("_readiness_native_loader", STAGE1 / "native_loader.py",
            files["payload/unifloral_td3bc_validation_v1/native_loader.py"])
        native = loader.load_native(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
        expected = (STAGE1 / "upstream/algorithms/td3_bc.py").resolve()
        if Path(native.__file__).resolve() != expected:
            raise ValueError("Loaded native module has the wrong origin")
        for name in ("DeterministicTanhActor", "SoftQNetwork", "DualQNetwork"):
            if getattr(native, name).__module__ != native.__name__:
                raise ValueError("Native class came from another module: " + name)
        for name in ("create_train_state", "make_train_step"):
            if Path(getattr(native, name).__code__.co_filename).resolve() != expected:
                raise ValueError("Native function came from another source: " + name)
        result.update(status="ready_for_reserved_bounded_audit", interpreter={
            "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "include_system_site_packages": False}, model={"module": native.__name__,
            "file": str(expected), "sha256": sha(expected), "native_revision": loader.REVISION,
            "definitions_only": True, "checkpoint_restoration_and_forward_replay": "Pending inside capped audit before its first evaluator transition"})
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "receipt": str(output),
        "gpu_backend": result.get("runtime", {}).get("backend"),
        "model_source": result.get("model", {}).get("file"),
        "evaluator_transitions": 0, "audit_started": False, "error": result.get("error")}), flush=True)
    return 0 if result["status"] == "ready_for_reserved_bounded_audit" else 1


if __name__ == "__main__":
    raise SystemExit(main())
