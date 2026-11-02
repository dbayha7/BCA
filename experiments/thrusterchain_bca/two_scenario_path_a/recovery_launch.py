"""One additive recovery; original source, smoke and interrupted outputs stay sealed."""
import argparse
from datetime import datetime
from decimal import Decimal
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch as L
import smoke_launch as S

OLD = HERE / "run_amendment_1"
OUT = HERE / "run_recovery_1"
REUSED = "honeypot_s0_warmup"
LOST = "honeypot_s0_native"
PIDS = (435, 1693)
KEYS = ("critic", "actor", "auxiliary", "calibrations")
JOB_CAPS = dict(zip(KEYS, (10000, 5000, 10000, 1)))
LOST_LOWER = dict(zip(KEYS, (4200, 2100, 4200, 0)))
LOST_UPPER = dict(JOB_CAPS)
SMOKE_COUNTS = dict(zip(KEYS, (14, 8, 14, 3)))
CUMULATIVE_CAPS = dict(zip(KEYS, (190014, 95008, 190014, 22)))
APPROVAL_TEXT = ("I approve one Path A recovery preserving the completed Honeypot seed-0 warmup and the passed "
    "runtime smoke. Restart only the interrupted native continuation from that warmup, then finish the unchanged "
    "remaining jobs and evaluation. I approve conservative cumulative limits of 190014 critic, 95008 actor, "
    "190014 auxiliary updates and 22 calibrations, including unknown interrupted work. Reserve 60 seconds for "
    "recovery preparation and at most 10660 seconds for the recovery process from the original remaining time. "
    "The original four-hour and 498400-transition ceilings remain. No smoke/warmup rerun, Path B execution, "
    "scientific changes or automatic retry.")


def allocation():
    return {"schema": "two_scenario.path_a.recovery.v1", "old_production_process_cap_seconds": 11490,
        "old_production_elapsed_seconds": 769.481604245, "remaining_before_recovery_seconds": 10720.518395755,
        "recovery_preparation_charge_seconds": 60, "recovery_process_cap_seconds": 10660,
        "internal_seconds": 10645, "external_timeout_seconds": 10655, "kill_grace_seconds": 5,
        "unallocated_seconds": .518395755, "cumulative_time_ceiling_seconds": 14400,
        "cumulative_simulator_transition_ceiling": 498400, "new_evaluation_transition_cap": 237600,
        "new_training_jobs": 17, "new_training_counts": {k: 17*v for k, v in JOB_CAPS.items()},
        "reused_warmup_counts": JOB_CAPS, "smoke_counts": SMOKE_COUNTS,
        "old_lost_work_lower": LOST_LOWER, "old_lost_work_upper": LOST_UPPER,
        "cumulative_optimizer_caps": CUMULATIVE_CAPS, "cumulative_execution_counts_known": False,
        "scientific_protocol_changed": False, "automatic_retry": False,
        "inactivity_evidence": {"reported_observation_est": "November 2, 2026 at 5:15 a.m. EST (minute precision)",
            "requested_pids": list(PIDS), "user_result": "ps returned its header with no entries",
            "meaning": "Only these recorded supervisor/worker PIDs were absent at that observation; not a global process guarantee"}}


def jobs():
    original = L.jobs()
    L.require(sum(j["id"] == REUSED for j in original) == 1, "Exactly one completed warmup must be reused")
    result = [dict(j) for j in original if j["id"] != REUSED]
    L.require(len(result) == 18 and sum(j["arm"] != "evaluation" for j in result) == 17,
              "Recovery plan must have 17 training jobs and one evaluation")
    return result


def inventory(root):
    root = Path(root).resolve()
    L.require(root.is_dir() and not root.is_symlink(), "Missing or linked lineage directory")
    files, directories = {}, []
    for path in sorted(root.rglob("*")):
        L.require(not path.is_symlink() and path.resolve().is_relative_to(root), "Linked lineage entry is not accepted")
        relative = path.relative_to(root).as_posix()
        if path.is_file():
            files[relative] = L.sha(path)
        elif path.is_dir():
            directories.append(relative)
        else:
            raise ValueError("Unexpected lineage entry: " + relative)
    return {"files": files, "directories": directories}


def verify_inventory(root, expected):
    L.require(inventory(root) == expected, "Original lineage contents or directory entries changed: " + str(root))


def verify_warmup(directory):
    directory = Path(directory)
    report = L.accepted_job(directory, 0)
    receipt = L.read(directory / "checkpoint.json")
    protocol = L.read(HERE / "protocol.json")
    binding_metadata = L.read(HERE / "DATA_RECEIPT.json")["conditions"]["honeypot"]["metadata"]
    # Same canonical JSON digest used by unchanged integration.py, without model imports.
    canonical = lambda value: __import__("hashlib").sha256(json.dumps(value, sort_keys=True,
                                  separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    L.require(report["condition"] == receipt["condition"] == "honeypot" and
              report["seed_index"] == receipt["seed_index"] == 0 and
              report["arm"] == receipt["arm"] == "warmup", "Wrong reusable condition/seed/boundary")
    L.require(report["checkpoint_receipt_sha256"] == L.sha(directory / "checkpoint.json") and
              report["checkpoint_sha256"] == receipt["checkpoint_sha256"] == L.sha(directory / "full_state.msgpack"),
              "Warmup checkpoint/report hash chain differs")
    L.require(receipt["protocol_sha256"] == canonical(protocol) and
              receipt["prepared_metadata_sha256"] == canonical(binding_metadata) and
              report["prepared_metadata"] == binding_metadata, "Warmup protocol/data/mask/calibration binding differs")
    L.require(receipt["dataset_sha256"] == binding_metadata["dataset_sha256"] and
              receipt["geometry_sha256"] == binding_metadata["geometry_sha256"] and
              receipt["normalization"] == report["normalization"] and
              receipt["normalization_sha256"] == canonical(receipt["normalization"]), "Warmup normalization/source binding differs")
    L.require(receipt["counts"] == report["final_cumulative_counts"] and
              receipt["counts"]["critic"] == 10000 and receipt["counts"]["refresh_accepted"] == 1 and
              receipt["counts"]["phase"] == 0 and receipt["counts"]["reference_ready"], "Warmup boundary incomplete")
    L.require(receipt["state_digest"] == report["final_exact_roundtrip"]["state_digest"], "Warmup exact-state receipt differs")
    return report


def original_evidence():
    S.check(approval=True)
    proof = S.smoke_identity()
    attempt, result = L.read(OLD / "ATTEMPT.json"), L.read(OLD / "RESULT.json")
    L.require(attempt["mode"] == result["mode"] == "production" and attempt["cap_seconds"] == 11490 and
              attempt["amendment_manifest_sha256"] == L.sha(HERE / "AMENDMENT_MANIFEST.json"), "Wrong original attempt")
    L.require(attempt["smoke_proof"] == result["smoke_proof"] == proof, "Original authorizing smoke proof differs")
    L.require(result["status"] == "incomplete" and result["execution_counts_known"] is False and
              result["error"] == "InterruptedError: Supervisor signal 2" and
              result["jobs"] == [{"exit_code": 0, "id": REUSED}], "Original outcome differs; review before recovery")
    L.require(Decimal(str(result["elapsed_seconds"])) == Decimal("769.481604245") and
              Decimal(11490) - Decimal(str(result["elapsed_seconds"])) == Decimal("10720.518395755"),
              "Remaining original process allowance differs")
    L.require(attempt["supervisor_pid"] == PIDS[0] and L.read(OLD / LOST / "PROCESS.json")["pid"] == PIDS[1],
              "Recorded old process identities differ")
    L.require(not any((OLD / LOST / name).exists() for name in
                     ("full_state.msgpack", "checkpoint.json", "report.json", "WORKER_EXIT.json")),
              "The interrupted job has newer terminal/checkpoint evidence; do not restart blindly")
    last = json.loads((OLD / LOST / "progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    L.require(last["local_step"] == 4200 and last["physical_counts"]["attempted"] == LOST_LOWER and
              last["physical_counts"]["completed"] == LOST_LOWER, "Known interrupted-work lower bound changed")
    verify_warmup(OLD / REUSED)
    return {"old_run": inventory(OLD), "smoke_tree": inventory(HERE / "smoke_attempt_1"),
            "smoke_proof": proof, "last_lost_progress_est": last["observed_est"],
            "last_lost_progress_counts": LOST_LOWER, "old_result_sha256": L.sha(OLD / "RESULT.json")}


def prepare():
    L.require(not (HERE / "RECOVERY_MANIFEST.json").exists() and not OUT.exists(), "Never reseal/reset a recovery")
    lineage = original_evidence()
    names = ("recovery_launch.py", "test_recovery_launch.py", "RECOVERY.md", "RECOVERY_VERIFICATION.json",
             "MANIFEST.json", "AMENDMENT_MANIFEST.json", "AMENDMENT_APPROVAL.json")
    L.save(HERE / "RECOVERY_MANIFEST.json", {"allocation": allocation(), "jobs": jobs(), "lineage": lineage,
        "files": {(HERE / name).relative_to(L.REPO).as_posix(): L.sha(HERE / name) for name in names},
        "approval_created": False, "execution_started": False})
    return {"status": "prepared_recovery_not_executed", "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json")}


def approval_expected():
    return {"approval": APPROVAL_TEXT, "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"),
            "recovery_process_cap_seconds": 10660, "cumulative_optimizer_caps": CUMULATIVE_CAPS,
            "reuse_only_completed_warmup": REUSED, "no_automatic_retry": True}


def check(approval=False):
    protocol = S.check(approval=True)
    manifest = L.read(HERE / "RECOVERY_MANIFEST.json")
    L.verify_pins(manifest["files"])
    L.require(manifest["allocation"] == allocation() and manifest["jobs"] == jobs(), "Recovery plan/budget changed")
    L.require(original_evidence() == manifest["lineage"], "Original outputs or smoke evidence changed")
    if approval:
        L.require(L.read(HERE / "RECOVERY_APPROVAL.json") == approval_expected(), "Explicit recovery approval absent or mismatched")
    return protocol


def old_processes_absent():
    L.require(os.name == "posix", "Recorded PID check is for the existing Linux runtime only")
    for pid in PIDS:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            raise RuntimeError("Cannot establish old PID inactivity: " + str(pid)) from exc
        raise RuntimeError("Recorded old PID is present (possibly reused): %d; inspect it, do not terminate it automatically" % pid)


def mirror_expected(manifest):
    prefix = REUSED + "/"
    old = manifest["lineage"]["old_run"]
    return {"files": {name[len(prefix):]: sha for name, sha in old["files"].items() if name.startswith(prefix)},
            "directories": [name[len(prefix):] for name in old["directories"] if name.startswith(prefix)]}


def reuse_receipt(manifest):
    return {"source": str((OLD / REUSED).relative_to(L.REPO)), "source_inventory": mirror_expected(manifest),
            "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"),
            "training_updates_in_copy": 0, "reused_prior_training_counts": JOB_CAPS}


def verify_mirror():
    manifest = L.read(HERE / "RECOVERY_MANIFEST.json")
    directory = OUT / REUSED
    expected = mirror_expected(manifest)
    actual = inventory(directory)
    L.require(set(actual["files"]) == set(expected["files"]) | {"REUSED_FROM.json"} and
              actual["directories"] == expected["directories"] and
              all(actual["files"][name] == value for name, value in expected["files"].items()),
              "Recovery warmup mirror is missing, altered or has unexpected entries")
    L.require(L.read(directory / "REUSED_FROM.json") == reuse_receipt(manifest), "Warmup reuse lineage changed")
    verify_warmup(directory)


def copy_warmup():
    manifest = L.read(HERE / "RECOVERY_MANIFEST.json")
    verify_inventory(OLD, manifest["lineage"]["old_run"])
    verify_warmup(OLD / REUSED)
    expected = mirror_expected(manifest)
    destination = OUT / REUSED
    destination.mkdir(exist_ok=False)
    for relative in expected["directories"]:
        (destination / relative).mkdir(exist_ok=False)
    for relative, digest in expected["files"].items():
        raw = (OLD / REUSED / relative).read_bytes()
        L.require(__import__("hashlib").sha256(raw).hexdigest() == digest, "Warmup changed while copying")
        with (destination / relative).open("xb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    L.save(destination / "REUSED_FROM.json", reuse_receipt(manifest))
    verify_mirror()


def worker_main(job_id):
    L.require(os.name == "posix", "Manual existing WSL runtime only")
    started = time.monotonic()
    protocol = check(approval=True)
    old_processes_absent()
    verify_mirror()
    expected = next((job for job in jobs() if job["id"] == job_id), None)
    L.require(expected is not None, "Unknown recovery job; reused warmup must not run")
    output = OUT / job_id
    job, attempt = L.read(output / "JOB.json"), L.read(OUT / "ATTEMPT.json")
    proof = S.smoke_identity()
    digest = L.sha(HERE / "RECOVERY_MANIFEST.json")
    L.require(all(job[k] == v for k, v in expected.items()) and
              job["recovery_manifest_sha256"] == attempt["recovery_manifest_sha256"] == digest and
              job["smoke_proof"] == attempt["smoke_proof"] == proof, "Recovery job/attempt/proof differs")
    end = started + min(float(job["remaining_seconds"]), 10645)
    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Recovery remaining deadline reached")
    final = {"status": "failed", "identity_unchanged": False, "error": None, "no_automatic_retry": True}
    try:
        deadline()
        binding, train, services, review = S._services(protocol)
        review["recovery_approval_sha256"] = L.sha(HERE / "RECOVERY_APPROVAL.json")
        if job["arm"] == "evaluation":
            worker = L.load("_path_a_worker", HERE / "worker.py")
            report = worker.evaluate_job(protocol, output, deadline, binding, train, services, review)
        else:
            warmup = None if job["arm"] == "warmup" else OUT / f"{job['condition']}_s{job['seed_index']}_warmup"
            if warmup is not None:
                L.accepted_job(warmup, 0)
            report = train.run_job(protocol, job["condition"], job["seed_index"], job["arm"], output,
                warmup_dir=warmup, deadline=deadline, services=services, training_review=review)
            L.require(report["status"] == "completed_training_job", "Recovery training job incomplete")
        deadline()
        check(approval=True)
        verify_mirror()
        final.update(status="completed", identity_unchanged=True, report_sha256=L.sha(output / "report.json"))
    except BaseException as exc:
        final.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    final.update(elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat())
    L.save(output / "WORKER_EXIT.json", final)
    print(json.dumps(final), flush=True)
    return final


def run_child(spec, job_dir, deadline, log):
    """Retain child ownership through receipt write, wait, timeout and cleanup."""
    child = None
    try:
        L.require(time.monotonic() < deadline, "Recovery deadline exhausted before child dispatch")
        child = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "recovery_launch.py"), "--worker", spec["id"]],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        L.save(job_dir / "PROCESS.json", {"pid": child.pid})
        return child.wait(timeout=max(.001, deadline-time.monotonic()))
    except BaseException:
        if child is not None:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            else:
                child.wait()
        raise


def evaluation_bounds(accepted, current, output):
    reports = [r for r in accepted if r.get("status") == "completed_frozen_evaluation"]
    if reports:
        exact = sum(reports[-1]["physical_counts"]["completed"].values())
        return {"lower": exact, "upper": exact, "execution_counts_known": True, "progress_read_error": None}
    if current is None or current["arm"] != "evaluation":
        return {"lower": 0, "upper": 0, "execution_counts_known": True, "progress_read_error": None}
    lower, error = 0, None
    caps = {"returns": 153600, "own_qpi": 57600, "common_reference": 21600, "prefix_parity": 4800}
    try:
        lines = (Path(output) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                counts = row["completed"]
                L.require(set(counts) == set(caps) and all(type(counts[k]) is int and 0 <= counts[k] <= caps[k] for k in caps),
                          "Invalid evaluation progress counts")
                lower = sum(counts.values())
                break
            except Exception as exc:
                error = "Ignored incomplete/invalid trailing progress: " + str(exc)
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    return {"lower": lower, "upper": 237600, "execution_counts_known": False, "progress_read_error": error,
            "note": "Saved completed calls are a lower bound only; zero lower bound does not mean zero actual work."}


def accounting(accepted, current=None, *, prior_simulator_steps, evaluation_output=None):
    completed = {k: sum(r["counts"]["completed"][k] for r in accepted if r.get("status") == "completed_training_job") for k in KEYS}
    upper = {k: completed[k] + (JOB_CAPS[k] if current is not None and current["arm"] != "evaluation" else 0) for k in KEYS}
    lower_total = {k: SMOKE_COUNTS[k] + JOB_CAPS[k] + LOST_LOWER[k] + completed[k] for k in KEYS}
    upper_total = {k: SMOKE_COUNTS[k] + JOB_CAPS[k] + LOST_UPPER[k] + upper[k] for k in KEYS}
    L.require(all(upper_total[k] <= CUMULATIVE_CAPS[k] for k in KEYS), "Recovery cumulative physical upper bound exceeds approval")
    evaluation = evaluation_bounds(accepted, current, evaluation_output)
    return {"new_verified_completed_training_counts": completed, "new_training_upper_bounds": upper,
        "reused_warmup_counts": JOB_CAPS, "smoke_counts": SMOKE_COUNTS,
        "prior_lost_work_lower": LOST_LOWER, "prior_lost_work_upper": LOST_UPPER,
        "cumulative_physical_lower": lower_total, "cumulative_physical_upper": upper_total,
        "cumulative_execution_counts_known": False, "new_evaluation_transition_bounds": evaluation,
        "cumulative_simulator_transition_lower": prior_simulator_steps + evaluation["lower"],
        "cumulative_simulator_transition_upper": prior_simulator_steps + evaluation["upper"],
        "note": "Reused warmup is not new work. Original interrupted native tail is bounded, never assumed zero. Pending undispatched jobs consume no work."}


def execute():
    L.require(os.name == "posix", "Manual existing WSL runtime only; no assistant launch")
    started = time.monotonic()
    check(approval=True)
    old_processes_absent()
    proof = S.smoke_identity()
    prior_simulator_steps = 189655 + sum(S.smoke_passed()["physical"]["counts"]["completed"].values())
    OUT.mkdir(exist_ok=False)
    deadline = started + 10645
    L.save(OUT / "ATTEMPT.json", {"started_est": datetime.now(L.EST).isoformat(), "supervisor_pid": os.getpid(),
        "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "smoke_proof": proof,
        "process_cap_seconds": 10660, "no_automatic_retry": True,
        "missing_terminal_receipt": "Incomplete: use persisted progress lower bounds and declared job upper bounds, never assume zero."})
    result = {"status": "incomplete_recovery", "jobs": [], "error": None, "no_automatic_retry": True,
              "training_hold_remains": True, "reused_job": REUSED, "smoke_rerun": False}
    accepted, current = [], None
    old_handlers = {}
    def stop(signum, frame):
        raise InterruptedError("Recovery supervisor signal " + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        old_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, stop)
    try:
        copy_warmup()
        for spec in jobs():
            check(approval=True)
            verify_mirror()
            L.require(S.smoke_identity() == proof, "Recovery smoke proof changed")
            remaining = deadline-time.monotonic()
            L.require(remaining > 0, "Recovery shared process deadline exhausted")
            job_dir = OUT / spec["id"]
            job_dir.mkdir(exist_ok=False)
            L.save(job_dir / "JOB.json", {**spec, "remaining_seconds": remaining,
                "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "smoke_proof": proof})
            current = spec
            print(json.dumps({"status": "starting_recovery_job", "id": spec["id"], "remaining_seconds": remaining}), flush=True)
            with (job_dir / "worker.log").open("x", encoding="utf-8") as log:
                rc = run_child(spec, job_dir, deadline, log)
            result["jobs"].append({"id": spec["id"], "exit_code": rc})
            report = L.accepted_job(job_dir, rc)
            accepted.append(report)
            current = None
            L.save(job_dir / "SUPERVISOR.json", {"status": "accepted", "exit_code": rc,
                                               "finished_est": datetime.now(L.EST).isoformat()})
            print(json.dumps({"status": "accepted_recovery_job", "id": spec["id"], "exit_code": rc}), flush=True)
        check(approval=True)
        verify_mirror()
        L.require(S.smoke_identity() == proof and time.monotonic() < deadline, "Final recovery proof/deadline failed")
        result.update(status="completed_path_a_recovery", new_execution_counts_known=True,
            completed_evaluation_transitions=sum(accepted[-1]["physical_counts"]["completed"].values()),
            evaluation_report=str((OUT / "evaluation/report.json").relative_to(L.REPO)))
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), new_execution_counts_known=False)
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    result.update(accounting(accepted, current, prior_simulator_steps=prior_simulator_steps, evaluation_output=OUT / "evaluation"),
                  all_declared_new_jobs_completed=result["status"] == "completed_path_a_recovery",
                  elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat())
    if result["status"] == "completed_path_a_recovery":
        result["cumulative_simulator_transitions"] = prior_simulator_steps + result["completed_evaluation_transitions"]
        L.require(result["cumulative_simulator_transitions"] <= 498400, "Original simulator ceiling exceeded")
    L.save(OUT / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.check:
        check()
        print(json.dumps({"status": "checked_recovery_not_executed", "approval_present": (HERE / "RECOVERY_APPROVAL.json").exists(),
                          "attempt_exists": OUT.exists(), "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json")}))
    elif args.worker:
        raise SystemExit(0 if worker_main(args.worker)["status"] == "completed" else 1)
    else:
        raise SystemExit(0 if execute()["status"] == "completed_path_a_recovery" else 1)
