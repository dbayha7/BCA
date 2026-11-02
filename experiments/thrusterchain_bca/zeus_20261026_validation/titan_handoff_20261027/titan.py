"""Read-only Titan handoff checks; execution is permanently blocked in this entry."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SUPPORT = HERE.parent / "support_shift_pilot_v1"
HISTORICAL_TITAN_PYTHON = "/home/dbayha/bca-gpu/bin/python"
ORIGIN_HASHES = {
    "PREPARED.json": "eb303cb95ea641dd19af80a3ba46bcb50306e2987ff687213c18cf45ee2bf671",
    "protocol.json": "52abc37cd6e7a9c609d90369a7692b12db8542e7bd4c3940252389873b5e7b23",
}
BLOCK_REASON = (
    "The approved support-shift attempt already started on Zeus. Its terminal "
    "outcome is unresolved in this handoff. No duplicate, resume, or retry is "
    "authorized. This entry has no execution implementation or override."
)


def origin_status():
    path = HERE / "ATTEMPT_HOLD.json"
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, dict):
            raise ValueError("Hold snapshot must be a JSON object")
        error = None
    except (OSError, ValueError) as exc:
        snapshot = None
        error = type(exc).__name__ + ": " + str(exc)
    return {
        "status": "execution_held_origin_attempt_unresolved",
        "execution_allowed": False,
        "reason": BLOCK_REASON,
        "origin_attempt_snapshot": snapshot,
        "snapshot_error": error,
        "live_process_status_checked": False,
        "final_execution_counts_known": False,
        "note": "This is a saved origin snapshot, not a live Zeus process check.",
    }


def load_support():
    """Import only the unchanged stdlib launcher and its IO helpers."""
    source = SUPPORT / "launch.py"
    existing = sys.modules.get("common")
    if existing is not None and Path(existing.__file__).resolve() != (SUPPORT / "common.py").resolve():
        raise ValueError("Conflicting common module; use a fresh Python process")
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(SUPPORT))
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("_titan_readonly_support_launch", source)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode


def preflight():
    # Check the original manifests before importing their frozen helper code.
    for name, expected in ORIGIN_HASHES.items():
        actual = hashlib.sha256((SUPPORT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Original handoff identity changed: " + name)
    old_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        launcher = load_support()
        protocol, manifest = launcher.verify_prepared()
        launcher.verify_inputs()
    finally:
        sys.dont_write_bytecode = old_bytecode
    return {
        "status": "portable_inputs_verified_execution_held",
        "protocol_sha256": ORIGIN_HASHES["protocol.json"],
        "prepared_sha256": ORIGIN_HASHES["PREPARED.json"],
        "prepared_files_verified": len(manifest["files"]),
        "dependencies_verified": len(manifest["dependencies"]),
        "caps": {key: protocol[key] for key in (
            "total_critic_updates", "total_actor_updates", "total_auxiliary_updates",
            "total_calibrations", "evaluation_episodes", "simulator_transition_cap",
            "total_timeout_seconds")},
        "current_python": sys.executable,
        "historical_titan_python": HISTORICAL_TITAN_PYTHON,
        "historical_titan_runtime_currently_verified": False,
        "runtime_packages_or_cuda_checked": False,
        "models_loaded": False,
        "attempt_reserved": False,
        "scientific_execution_started": False,
        "origin": origin_status(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    # Intentionally unconditional: missing, modified, or deleted hold receipts
    # cannot turn this read-only handoff into a second experiment launcher.
    if args.execute:
        print(json.dumps(origin_status(), sort_keys=True))
        return 3
    if args.status:
        print(json.dumps(origin_status(), sort_keys=True))
        return 0
    try:
        result = preflight()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(json.dumps({"status": "preflight_failed", "execution_allowed": False,
                          "error": type(exc).__name__ + ": " + str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
