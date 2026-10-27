"""Retry lineage and read-only process clearance; no scientific imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGIN_NAME = "support_shift_pilot_v1"
ARCHIVE_NAME = "support_shift_pilot_v1_interrupted_20261027T033240Z"
ORIGIN_PREPARED_SHA = "eb303cb95ea641dd19af80a3ba46bcb50306e2987ff687213c18cf45ee2bf671"
PROTOCOL_SHA = "52abc37cd6e7a9c609d90369a7692b12db8542e7bd4c3940252389873b5e7b23"
ORIGIN_PIDS = (372, 486)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lineage(root=HERE):
    root = Path(root).resolve()
    origin = root.parent / ORIGIN_NAME
    archive = root.parent / "archived_attempts" / ARCHIVE_NAME / "ARCHIVE.json"
    retry = read(root / "RETRY.json")
    require(isinstance(retry, dict), "Retry lineage must be an object")
    require(retry.get("origin_prepared_sha256") == ORIGIN_PREPARED_SHA
            and sha(origin / "PREPARED.json") == ORIGIN_PREPARED_SHA, "Original prepared identity changed")
    require(retry.get("protocol_sha256") == PROTOCOL_SHA
            and sha(root / "protocol.json") == sha(origin / "protocol.json") == PROTOCOL_SHA,
            "Scientific protocol changed")
    require(retry.get("prior_attempt_counts_are_lower_bounds") is True,
            "Prior interrupted work must remain separately accounted")
    archive_sha = sha(archive)
    require(retry.get("archive_manifest_sha256") == archive_sha, "Archived attempt identity changed")
    archive_data = read(archive)
    snapshot = archive.parent / "snapshot"
    for record in archive_data["files"]:
        path = (snapshot / record["path"]).resolve()
        require(path.is_relative_to(snapshot.resolve()), "Archive member escapes snapshot")
        require(path.stat().st_size == record["bytes"] and sha(path) == record["sha256"],
                "Archive snapshot changed: " + record["path"])
    # Only supervision and human instructions differ; every original science,
    # data, evaluation and fixture file retains its original exact bytes.
    original = read(origin / "PREPARED.json")
    for name, expected in original["files"].items():
        require(sha(origin / name) == expected, "Original frozen source changed: " + name)
        if name not in ("launch.py", "README.md", "test_launch.py"):
            require(sha(root / name) == expected, "Original science/input changed: " + name)
    return {"archive_manifest_sha256": archive_sha,
            "origin_prepared_sha256": ORIGIN_PREPARED_SHA, "protocol_sha256": PROTOCOL_SHA,
            "retry_lineage_sha256": sha(root / "RETRY.json")}


def accounting(root=HERE):
    root = Path(root)
    identity = lineage(root)
    archive = read(root.parent / "archived_attempts" / ARCHIVE_NAME / "ARCHIVE.json")
    return {"attempt_ordinal": 2, **identity,
            "origin_classification": archive["classification"],
            "prior_attempt_counts_are_lower_bounds": True,
            "prior_attempt_final_counts_known": False,
            "prior_attempt_last_committed_ledger": archive["last_committed_ledger"],
            "new_attempt_counts_are_separate": True,
            "new_attempt_per_attempt_limits_unchanged": True}


def process_targets(root=HERE):
    root = Path(root).resolve()
    return {(root / "launch.py").resolve(),
            (root.parent / ORIGIN_NAME / "launch.py").resolve(),
            (root.parent / "matched_policy_pilot_v1/launch.py").resolve(),
            (root.parent / "bca_one_sided_v1/launch.py").resolve(),
            (root.parents[1] / "titan_20261026/launch.py").resolve()}


def inspect_processes(root=HERE, proc_root=Path("/proc"), own_pid=None, own_uid=None):
    """Inspect same-user Python launch paths; do not signal or attach to a process."""
    own_pid = os.getpid() if own_pid is None else own_pid
    own_uid = os.getuid() if own_uid is None else own_uid
    targets = process_targets(root)
    observations = {str(pid): {"pid": pid, "status": "absent"} for pid in ORIGIN_PIDS}
    matches = []
    for process in Path(proc_root).iterdir():
        if not process.name.isdigit():
            continue
        pid = int(process.name)
        if pid == own_pid:
            if pid in ORIGIN_PIDS:
                observations[str(pid)] = {"pid": pid, "status": "reused_by_current_inspecting_process"}
            continue
        try:
            if process.stat().st_uid != own_uid:
                if pid in ORIGIN_PIDS:
                    observations[str(pid)] = {"pid": pid, "status": "present_other_user"}
                continue
            comm = (process / "comm").read_text().strip()
            if not comm.startswith(("python", "pypy")):
                if pid in ORIGIN_PIDS:
                    observations[str(pid)] = {"pid": pid, "status": "present_non_python", "comm": comm}
                continue
            matched = None
            for raw in (process / "cmdline").read_bytes().split(b"\0")[1:]:
                path = Path(os.fsdecode(raw))
                if path.name != "launch.py":
                    continue
                if not path.is_absolute():
                    path = (process / "cwd").resolve(strict=True) / path
                if path.resolve() in targets:
                    matched = str(path.resolve())
                    break
            if matched:
                matches.append({"pid": pid, "launch_path": matched})
            if pid in ORIGIN_PIDS:
                observations[str(pid)] = {"pid": pid,
                    "status": "matching_launch_process" if matched else "present_no_matching_launch_path",
                    "matching_launch_path": matched}
        except (FileNotFoundError, ProcessLookupError):
            if pid in ORIGIN_PIDS:
                observations[str(pid)] = {"pid": pid, "status": "disappeared_during_inspection"}
        except PermissionError as exc:
            raise RuntimeError("Scoped process inspection denied; no clearance or launch") from exc
    return {"matching_processes": matches, "recorded_origin_pids": list(observations.values()),
            "scope": "Same-user Python processes at the exact old/retry/prior BCA launcher paths; recorded old PID observations"}


def verify_no_processes(root=HERE):
    require(sys.platform == "linux", "Process clearance must run manually in the existing Ubuntu environment")
    observed = inspect_processes(root)
    require(not observed["matching_processes"], "Existing scoped BCA process: " + str(observed["matching_processes"]))
    return observed


def check_origin_stopped(root=HERE):
    root = Path(root)
    identity = lineage(root)
    observed = verify_no_processes(root)
    receipt = {"schema": "unifloral.support_retry.origin_stopped.v1",
        "status": "no_matching_origin_or_retry_process_observed",
        "checked_utc": datetime.now(timezone.utc).isoformat(), **identity,
        "observations": observed, "attempt_reserved": False, "scientific_execution_started": False,
        "note": "Point-in-time process observation, not exact final work accounting; execute rechecks before reserving."}
    path = root / "ORIGIN_STOPPED.json"
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return receipt


def require_clearance(root=HERE):
    root = Path(root)
    identity = lineage(root)
    receipt = read(root / "ORIGIN_STOPPED.json")
    require(isinstance(receipt, dict), "Successful manual origin process clearance required")
    if receipt.get("schema") == "unifloral.support_retry.origin_stopped.v1":
        require(receipt.get("status") == "no_matching_origin_or_retry_process_observed"
                and isinstance(receipt.get("observations"), dict)
                and receipt["observations"].get("matching_processes") == [], "Invalid scoped clearance observations")
    elif receipt.get("schema") == "unifloral.support_retry.user_pid_check.v1":
        expected = [{"pid": pid, "status": "absent"} for pid in ORIGIN_PIDS]
        require(receipt.get("status") == "recorded_origin_pids_absent_user_report"
                and receipt.get("source") == "trusted_user_terminal_report"
                and receipt.get("recorded_origin_pids") == expected
                and all(type(row["pid"]) is int for row in receipt["recorded_origin_pids"])
                and receipt.get("global_scoped_process_absence_verified") is False,
                "Invalid user-reported recorded-PID clearance")
    else:
        raise ValueError("Successful manual origin process clearance required")
    require(all(receipt.get(key) == value for key, value in identity.items()), "Process clearance lineage changed")
    require(receipt.get("attempt_reserved") is False and receipt.get("scientific_execution_started") is False,
            "Invalid process clearance scope")
    require(isinstance(receipt.get("checked_utc"), str), "Missing clearance timestamp")
    datetime.fromisoformat(receipt["checked_utc"])
    return verify_no_processes(root)
