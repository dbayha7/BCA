"""Read-only source/archive checks for a proposed Titan continuation; no execution."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ARCHIVE_NAME = "support_shift_retry2_interrupted_20261027T042116Z"
ARCHIVE_SHA = "8c5c11c41085e4201392c747c8e06f21df3c06fdcdc9b28a3cf84013c628fbf2"
PROTOCOL_SHA = "52abc37cd6e7a9c609d90369a7692b12db8542e7bd4c3940252389873b5e7b23"
CELLS = ["seed0_broad_count_matched", "seed0_restricted",
         "seed1_restricted", "seed1_broad_count_matched"]
COEFFICIENTS = {CELLS[0]: 0.07585831620304931, CELLS[1]: 0.0670392257429505}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def member(root, name):
    require(isinstance(name, str) and name and "\\" not in name and ":" not in name,
            "Unsafe archive member")
    parts = name.split("/")
    require(all(part not in ("", ".", "..") for part in parts), "Unsafe archive member")
    root = Path(root)
    target = root.joinpath(*parts)
    require(target.resolve().is_relative_to(root.resolve()), "Archive member escapes snapshot")
    current = target
    while current != root:
        require(not current.is_symlink(), "Symlinked archive member")
        current = current.parent
    require(not root.is_symlink(), "Symlinked snapshot root")
    return target


def verify_archive(path, expected_sha=ARCHIVE_SHA):
    path = Path(path)
    require(sha(path) == expected_sha, "Archive manifest identity changed")
    archive = read(path)
    require(archive["schema"] == "bca.interrupted_attempt_archive.v1"
            and archive["attempt_ordinal"] == 2, "Wrong interrupted attempt")
    snapshot = path.parent / "snapshot"
    names = [row["path"] for row in archive["files"]]
    require(len(names) == len(set(name.casefold() for name in names)), "Duplicate archive members")
    actual = set()
    for candidate in snapshot.rglob("*"):
        require(not candidate.is_symlink(), "Symlink in snapshot")
        if candidate.is_file():
            actual.add(candidate.relative_to(snapshot).as_posix())
    require(actual == set(names), "Archive snapshot membership changed")
    for row in archive["files"]:
        target = member(snapshot, row["path"])
        require(target.stat().st_size == row["bytes"] and sha(target) == row["sha256"],
                "Archive member changed: " + row["path"])
    return archive, snapshot


def validate_counts_and_match(report, matching, cell):
    expected = {"critic": 10000, "actor": 5000, "auxiliary": 10000,
                "actor_target": 5000, "critic_target": 5000, "outer_steps": 5000,
                "aux_fit_attempts": 10000, "refresh_attempts": 1, "refresh_accepted": 1}
    require(report["additional_updates"] == expected, "Warmup update accounting changed")
    require(matching["distance_coefficient"] == COEFFICIENTS[cell]
            and matching["probe_rows"] == 768 and matching["no_rematch"] is True
            and matching["optimizer_updates"] == matching["calibration_calls"] == 0,
            "Frozen training-only matching changed")


def verify_warmup(snapshot, cell):
    directory = Path(snapshot) / "run" / cell / "warmup"
    report, supervisor = read(directory / "report.json"), read(directory / "SUPERVISOR_RESULT.json")
    require(supervisor["completed"] is True and supervisor["exit_code"] == 0
            and supervisor["timed_out"] is False and supervisor["interrupted"] is False
            and supervisor["report_sha256"] == sha(directory / "report.json"), "Incomplete warmup seal")
    require(report["status"] == "completed_support_warmup" and report["cell"] == cell
            and report["seed_index"] == 0 and report["protocol_sha256"] == PROTOCOL_SHA
            and report["execution_counts_known"] is True, "Warmup identity changed")
    for name in ("actor_manifest", "state_manifest", "untrained_actor_manifest", "data_receipt", "matching_receipt"):
        require(sha(directory / (name + ".json")) == report[name + "_sha256"], "Warmup seal changed: " + name)
    actor, state = read(directory / "actor_manifest.json"), read(directory / "state_manifest.json")
    untrained, data = read(directory / "untrained_actor_manifest.json"), read(directory / "data_receipt.json")
    matching, refresh = read(directory / "matching_receipt.json"), read(directory / "refresh_receipt.json")
    validate_counts_and_match(report, matching, cell)
    require(state["complete_optimizer_target_rng_state"] is True and state["resume_authorized"] is False,
            "Full state unavailable or old authorization changed")
    require(state["artifact"] == actor["final_state"], "State/actor binding changed")
    for artifact in (state["artifact"], actor["actor_params"], untrained["actor_params"],
                     report["unique_replay_source_rows"], report["matching_gradients"]):
        path = member(directory, artifact["path"])
        require(path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"], "Export/provenance artifact changed")
    require(all(untrained["counts"][key] == 0 for key in ("critic", "actor", "auxiliary")), "Untrained control changed")
    require(data["training_rows"] == 57752 and data["calibration_rows"] == 128
            and data["protected_transition_rows_decoded"] == 0, "Data-role accounting changed")
    require(refresh["absolute_critic"] == 10000 and refresh["counts"]["reference_ready"] is True
            and refresh["refresh"]["flags"]["posterior_certified"] is True, "Warmup calibration changed")
    require(actor["normalization"] == state["normalization"] == untrained["normalization"], "Normalization binding changed")
    return {"cell": cell, "full_state": state["artifact"], "distance_coefficient": matching["distance_coefficient"],
            "initial_parameter_hashes": report["initial_parameter_hashes"], "additional_updates": report["additional_updates"],
            "untrained_actor": untrained["actor_params"], "metadata_seals_verified": True,
            "deserialization_or_runtime_restore_verified": False}


def verify_source_package(validation):
    package = Path(validation).parent / "titan_20261026"
    spec = importlib.util.spec_from_file_location("_titan_handoff_asset_verifier", package / "restore_assets.py")
    helper = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(helper)
    finally:
        sys.dont_write_bytecode = previous
    return helper.verify_package(root=package, require_assets=True)


def verify(validation=HERE.parent):
    validation = Path(validation)
    archive, snapshot = verify_archive(validation / "archived_attempts" / ARCHIVE_NAME / "ARCHIVE.json")
    require(sha(snapshot / "protocol.json") == PROTOCOL_SHA, "Scientific protocol changed")
    protocol = read(snapshot / "protocol.json")
    require(protocol["cells"] == CELLS and protocol["resume"] is False, "Original job coverage/guard changed")
    require(protocol["seeds"] == [dict(index=i, native_seed=2026103001+i, aux_seed=2026103011+i,
            branch_native_seed=2026103031+i, branch_aux_seed=2026103041+i) for i in (0, 1)], "Seeds changed")
    warmups = [verify_warmup(snapshot, cell) for cell in CELLS[:2]]
    require(warmups[0]["initial_parameter_hashes"] == warmups[1]["initial_parameter_hashes"], "Paired initializations differ")
    source = verify_source_package(validation)
    return {"status": "candidate_continuation_inputs_verified", "archive_sha256": ARCHIVE_SHA,
            "archive_files_verified": len(archive["files"]), "archive_bytes_verified": sum(x["bytes"] for x in archive["files"]),
            "scientific_protocol_sha256": PROTOCOL_SHA, "source_package": source, "warmups": warmups,
            "planned_cells": CELLS, "remaining_warmups": CELLS[2:],
            "candidate_remaining_work": {"critic_updates": 44000, "actor_updates": 22000, "auxiliary_updates": 44000,
                "calibrations": 14, "evaluation_episodes": 512, "simulator_transitions": 51200},
            "training_started": False, "gpu_runtime_verified": False, "runtime_restore_verified": False,
            "continuation_execution_implemented": False, "separate_continuation_review_required": True,
            "original_resume_supported": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Read-only checks (also the default)")
    parser.parse_args(argv)
    try:
        result = verify()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "handoff_verification_failed", "error": type(exc).__name__ + ": " + str(exc),
                          "training_started": False}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
