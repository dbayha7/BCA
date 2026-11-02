"""Versioned repair entrypoint: source verification, CPU fixtures and saved-data analysis.

No production execution option. No simulator, learner, dataset validation/replay or
calibration is dispatched. Missing historical research seals are not bypassed.
"""
from pathlib import Path
import argparse
import hashlib
import importlib
import json
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CANDIDATE = HERE.parent / "knee_s10_candidate_guard_repair_v1"
MANIFEST = HERE / "REPAIR_PACKAGE.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def importable_child_packages(directories):
    return [str(p) for d in directories for p in d.iterdir()
            if p.is_dir() and (p / "__init__.py").is_file()]


def verify_package():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest["purpose"] == "analysis_and_verification_only", "Wrong package purpose")
    snapshot = (REPO / manifest["source_snapshot"]).resolve()
    require(snapshot.is_relative_to(REPO) and snapshot.is_file()
            and sha(snapshot) == manifest["source_snapshot_sha256"], "Source snapshot lineage changed or missing")
    require(not importable_child_packages((HERE, CANDIDATE)), "Unlisted importable child package")
    listed = set()
    for relative, expected in manifest["files"].items():
        p = (REPO / relative).resolve()
        require(p.parent in (HERE, CANDIDATE), "Manifest path outside repaired packages")
        require(p.is_file() and sha(p) == expected, "Changed or missing repaired source: " + relative)
        listed.add(p)
    actual_python = {p.resolve() for d in (HERE, CANDIDATE) for p in d.glob("*.py")}
    require(actual_python == {p for p in listed if p.suffix == ".py"}, "Unlisted/missing Python source")
    return {"package": HERE.name, "manifest_sha256": sha(MANIFEST),
            "verified_files": len(listed), "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "scientific_execution_enabled": False}


def local_module(name, directory=HERE):
    for d in (str(HERE), str(CANDIDATE)):
        if d not in sys.path:
            sys.path.insert(0, d)
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == directory / (name + ".py"), "Wrong imported module: " + name)
    return module


def check():
    result = verify_package()
    result["loaded_modules"] = {}
    for name in ("core", "units", "stage1_worker", "stage2_worker", "amended_analysis"):
        module = local_module(name)
        result["loaded_modules"][name] = str(Path(module.__file__).resolve())
    result["loaded_modules"]["retained_support"] = str(Path(local_module("retained_support", CANDIDATE).__file__).resolve())
    return result


def analyse_known():
    verify_package()
    module = local_module("amended_analysis")
    result = module.analyse(REPO / module.correction_contract()["attempt_relative"])
    return {"purpose": "authenticated_historical_analysis", "attempt": result["attempt"],
            "accounting": result["R3_accounting"]["totals"],
            "original_decision": result["original"]["label"],
            "amended_decision": result["R1_v3_rules"]["label_under_v3"],
            "V0f_by_seed": result["R1_v3_rules"]["V0f_by_seed"], "sources": result["sources"]}


def support_bundle(path):
    """Read a fitting-only export. Hash checks do not independently attest its provenance."""
    verify_package()
    import numpy as np
    path = Path(path).resolve()
    metadata = json.loads(path.read_text(encoding="utf-8"))
    require(metadata["schema"] == "retained_support_fitting_export_v1", "Wrong support bundle schema")
    require(metadata["data_kind"] in ("synthetic_fixture", "development"), "Only fixture/development inputs supported")
    require(metadata["world"] in ("stall3", "nostall"), "Unknown world")
    file = (path.parent / metadata["arrays_file"]).resolve()
    require(file.parent == path.parent and file.suffix == ".npz", "Array bundle must be an adjacent NPZ")
    require(sha(file) == metadata["arrays_sha256"], "Support array hash mismatch")
    expected = {"obs", "actions", "episode_ids", "times", "deep_episode", "row_roles",
                "start_obs", "anchor_obs", "anchor_times"}
    with np.load(file, allow_pickle=False) as archive:
        require(set(archive.files) == expected, "Missing or unexpected support arrays")
        arrays = {k: archive[k] for k in expected}
    n = len(arrays["obs"])
    require(arrays["row_roles"].shape == (n,) and np.all(arrays["row_roles"] == 0), "Only fitting rows (role 0) allowed")
    require(arrays["start_obs"].shape == (128, 14) and arrays["anchor_obs"].shape == (48, 14), "Wrong fixed query counts")
    require(arrays["anchor_times"].shape == (48,), "Wrong anchor time shape")
    require({int(t): int(np.sum(arrays["anchor_times"] == t)) for t in np.unique(arrays["anchor_times"])}
            == {0: 16, 25: 16, 50: 16}, "Wrong fixed anchor time counts")
    require(arrays["actions"].shape == (n, 4), "Wrong action shape")
    region = local_module("knee_region")
    cutoff = ~region.past_knee_a0(arrays["actions"])[0]
    support = local_module("retained_support", CANDIDATE)
    result = support.retained_l15(arrays, cutoff, arrays["deep_episode"], arrays["start_obs"],
                                  arrays["anchor_obs"], arrays["anchor_times"])
    return {"purpose": "saved_fitting_array_support", "data_kind": metadata["data_kind"],
            "world": metadata["world"], "bundle_sha256": sha(path), "arrays_sha256": sha(file),
            "provenance_limit": "Byte integrity verified; row-role and collection provenance are producer declarations, not independent attestation",
            "formal_admission": False, "result": result}


REGRESSIONS = ["test_guard_fail_closed", "test_guard_accounting", "test_amended_analysis",
               "test_v3_repairs.R1V0f", "test_v3_repairs.R2Decomposition", "test_v3_repairs.R6Ledger",
               "test_v3_repairs.R5Terminal.test_exactly_one_terminal_write_with_the_counts",
               "test_v3_repairs.R5Terminal.test_watchdog_goes_through_the_owner",
               "test_v3_repairs.R5Terminal.test_a_failed_write_leaves_the_owner_open",
               "test_v3_repairs.R5Terminal.test_simultaneous_finish_writes_exactly_once",
               "test_stage1.Rules", "test_stage1.WorkerFlow", "test_stage2.Math", "test_step04.L2Math",
               "test_repaired_entrypoint"]


def tests():
    verify_package()
    groups = [(HERE, ["-m", "unittest", *REGRESSIONS, "-v"]),
              (CANDIDATE, ["test_retained_support.py", "-v"])]
    result = []
    for cwd, args in groups:
        command = [sys.executable, "-B", *args]
        proc = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=120)
        sys.stderr.write(proc.stdout + proc.stderr)
        result.append({"cwd": str(cwd), "exit_code": proc.returncode})
    require(all(x["exit_code"] == 0 for x in result), "CPU regression failure")
    return {"purpose": "CPU_fixture_tests_only", "groups": result, "scientific_updates": 0, "simulator_steps": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "test", "analyse-known-v2"):
        commands.add_parser(name)
    commands.add_parser("spend").add_argument("attempt", type=Path)
    commands.add_parser("support").add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    if args.command == "check":
        result = check()
    elif args.command == "test":
        result = tests()
    elif args.command == "analyse-known-v2":
        result = analyse_known()
    elif args.command == "support":
        result = support_bundle(args.bundle)
    else:
        verify_package()
        require(args.attempt.is_dir(), "Attempt directory missing")
        result = local_module("units").attempt_spend(args.attempt)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"status": "refused", "error": f"{type(exc).__name__}: {exc}",
                          "scientific_execution_started": False}), file=sys.stderr)
        raise SystemExit(1)
