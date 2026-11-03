"""Titan-only transport profile; all scientific recovery functions remain unchanged."""
import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_launch as R
L, S = R.L, R.S
RECOVERY_CHECK = R.check
SOURCE_HOLD_DIRECTORY = R.OUT
OUT = HERE / "run_titan_recovery_1"
VALIDATION = HERE / "titan_validation_1"
TARGET_REPO = "/mnt/c/Users/David Bayha/BCA-PathA"
TARGET_PYTHON = "/home/dbayha/bca-gpu/bin/python"
APPROVAL_TEXT = ("I approve migrating the existing Path A recovery to Titan only, preserving the completed "
    "Honeypot seed-0 warmup, passed Zeus runtime smoke, scientific protocol and cumulative work limits. "
    "The remaining 10660 seconds includes a 60-second preparation and unmeasured source-check overhead allowance, "
    "one zero-update Titan compatibility validation capped at 120 seconds and, only if it passes, "
    "the unchanged 17 training jobs plus evaluation capped at 10480 seconds. "
    "No additional optimizer, calibration or simulator work is authorized for validation. No further Zeus "
    "execution, automatic retry, Path B execution or increased cumulative budget is authorized.")


def allocation():
    return {"schema": "two_scenario.path_a.titan_migration.v1", "target_machine": "Titan",
        "target_repository": TARGET_REPO, "target_python": TARGET_PYTHON,
        "validation_cap_seconds": 120, "validation_internal_seconds": 105,
        "validation_external_timeout_seconds": 115, "kill_grace_seconds": 5,
        "preparation_and_unmeasured_source_check_allowance_seconds": 60,
        "preparation_allowance_is_measured_runtime": False,
        "production_cap_seconds": 10480, "production_internal_seconds": 10465,
        "production_external_timeout_seconds": 10475, "shared_remaining_seconds": 10660,
        "validation_new_work": dict(critic=0, actor=0, auxiliary=0, calibrations=0, simulator_steps=0, reset_calls=0),
        "cumulative_optimizer_caps": R.CUMULATIVE_CAPS, "new_evaluation_transition_cap": 237600,
        "cumulative_simulator_ceiling": 498400, "cumulative_time_ceiling_seconds": 14400,
        "prior_pid_handling": "Zeus435/1693 absence was user-observed before migration; never query those numbers in Titan's PID namespace",
        "source_hold": "Exclusive source run_recovery_1/HANDOFF_HOLD.json; archive omits this directory",
        "cross_machine_exclusion": "Operational handoff to Titan only; no distributed lock or guarantee against manual clones",
        "scientific_functions_changed": False, "automatic_retry": False}


def source_hold():
    proof = L.read(HERE / "TITAN_SOURCE_HOLD.json")
    L.require(proof.get("status") == "held_for_Titan_migration_not_a_run" and proof.get("execution_started") is False,
              "Missing explicit source migration hold")
    if SOURCE_HOLD_DIRECTORY.exists():
        contents = R.inventory(SOURCE_HOLD_DIRECTORY)
        L.require(contents["directories"] == [] and set(contents["files"]) == {"HANDOFF_HOLD.json"} and
                  L.read(SOURCE_HOLD_DIRECTORY / "HANDOFF_HOLD.json") == proof,
                  "A real/changed Zeus recovery reservation exists; do not migrate or duplicate it")
    return proof


def prepare():
    RECOVERY_CHECK(approval=True)
    source_hold()
    L.require(not any(p.exists() for p in (HERE / "TITAN_MANIFEST.json", VALIDATION, OUT)),
              "Never reseal or reset a Titan attempt")
    names = ("titan_launch.py", "test_titan_launch.py", "titan_compatibility.py", "test_titan_compatibility.py",
             "TITAN_HANDOFF.md", "TITAN_VERIFICATION.json", "TITAN_SOURCE_HOLD.json",
             "RECOVERY_MANIFEST.json", "RECOVERY_APPROVAL.json")
    L.save(HERE / "TITAN_MANIFEST.json", {"allocation": allocation(), "jobs": R.jobs(),
        "files": {(HERE / name).relative_to(L.REPO).as_posix(): L.sha(HERE / name) for name in names},
        "recovery_sources_and_original_outputs_verified_transitively": True,
        "source_hold_sha256": L.sha(HERE / "TITAN_SOURCE_HOLD.json"),
        "approval_created": False, "target_runtime_verified": False})
    return {"status": "prepared_titan_profile_not_executed", "titan_manifest_sha256": L.sha(HERE / "TITAN_MANIFEST.json")}


def approval_expected():
    return {"approval": APPROVAL_TEXT, "titan_manifest_sha256": L.sha(HERE / "TITAN_MANIFEST.json"),
            "target_machine": "Titan", "shared_remaining_seconds": 10660,
            "validation_then_science_only_on_pass": True, "no_further_Zeus_execution": True}


def check(approval=False):
    protocol = RECOVERY_CHECK(approval=True)
    source_hold()
    manifest = L.read(HERE / "TITAN_MANIFEST.json")
    L.verify_pins(manifest["files"])
    L.require(manifest["allocation"] == allocation() and manifest["jobs"] == R.jobs() and
              manifest["source_hold_sha256"] == L.sha(HERE / "TITAN_SOURCE_HOLD.json"), "Titan profile/lineage changed")
    if approval:
        L.require(L.read(HERE / "TITAN_APPROVAL.json") == approval_expected(), "Explicit manifest-bound Titan migration approval differs")
    return protocol


def target_identity():
    L.require(os.name == "posix" and str(L.REPO.resolve()) == TARGET_REPO and sys.executable == TARGET_PYTHON,
              "Use only the specified Titan checkout and interpreter; Zeus must not execute this profile")
    L.require(sys.version_info[:2] == (3, 12), "Titan Python major/minor differs")
    L.require(not SOURCE_HOLD_DIRECTORY.exists(), "Source hold/Zeus recovery directory must not be copied to Titan")
    return {"machine": "Titan", "repository": str(L.REPO.resolve()), "executable": sys.executable,
            "python": sys.version, "hostname": platform.node(), "pid_namespace": "Titan; no Zeus PID probes"}


def validation_proof():
    final = L.read(VALIDATION / "RESULT.json")
    report = L.read(VALIDATION / "report.json")
    attempt = L.read(VALIDATION / "ATTEMPT.json")
    identity = target_identity()
    L.require(final["status"] == "passed_titan_validation" and final["identity_unchanged"] and
              final["elapsed_seconds"] <= 115 and final["report_sha256"] == L.sha(VALIDATION / "report.json"),
              "Titan compatibility validation is missing, failed or changed")
    L.require(report["status"] == "passed_zero_update_titan_compatibility" and report["passed"] is True and
              report["new_work"] == allocation()["validation_new_work"], "Titan validation did not establish zero-update compatibility")
    L.require(attempt["titan_manifest_sha256"] == L.sha(HERE / "TITAN_MANIFEST.json") and
              attempt["titan_approval_sha256"] == L.sha(HERE / "TITAN_APPROVAL.json") and
              attempt["target_identity"] == final["target_identity"] == identity,
              "Validation belongs to another profile, approval or target runtime")
    return {name: L.sha(VALIDATION / name) for name in ("ATTEMPT.json", "RESULT.json", "report.json", "progress.jsonl")}


def validate():
    started = time.monotonic()
    protocol = check(approval=True)
    identity = target_identity()
    L.require(not OUT.exists(), "Scientific Titan reservation already exists")
    VALIDATION.mkdir(exist_ok=False)
    L.save(VALIDATION / "ATTEMPT.json", {"started_est": datetime.now(L.EST).isoformat(), "pid": os.getpid(),
        "target_identity": identity, "titan_manifest_sha256": L.sha(HERE / "TITAN_MANIFEST.json"),
        "titan_approval_sha256": L.sha(HERE / "TITAN_APPROVAL.json"), "process_cap_seconds": 120,
        "no_automatic_retry": True, "new_optimizer_calibration_simulator_work": 0})
    final = {"status": "failed_titan_validation", "identity_unchanged": False, "error": None,
             "target_identity": identity, "charged_seconds": 120, "remaining_science_seconds": 10480,
             "preparation_allowance_seconds": 60, "preparation_allowance_is_measured_runtime": False,
             "no_automatic_retry": True}
    end = started + 105
    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Titan validation internal deadline reached")
    def persist(event):
        with (VALIDATION / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(L.plain(event), sort_keys=True, allow_nan=False) + "\n"); stream.flush()
    def stop(signum, frame):
        raise InterruptedError("Titan validation signal " + str(signum))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGALRM)}
    for sig in handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, end-time.monotonic()))
    try:
        persist({"phase": "zero_update_validation_start", "new_work": allocation()["validation_new_work"]})
        deadline()
        binding, training, services, review = S._services(protocol)
        worker = L.load("_path_a_worker", HERE / "worker.py")
        compatibility = L.load("_path_a_titan_compatibility", HERE / "titan_compatibility.py")
        report = compatibility.check_compatibility(protocol, binding, training, services, review, worker, deadline, persist,
                                                    warmup_dir=R.OLD / R.REUSED)
        L.save(VALIDATION / "report.json", report)
        L.require(report["status"] == "passed_zero_update_titan_compatibility" and report["passed"] is True and
                  report["new_work"] == allocation()["validation_new_work"], "Titan zero-update compatibility did not pass")
        deadline()
        check(approval=True)
        L.require(target_identity() == identity, "Target identity changed during validation")
        final.update(status="passed_titan_validation", identity_unchanged=True, report_sha256=L.sha(VALIDATION / "report.json"))
    except BaseException as exc:
        final.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    final.update(elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat())
    if final["elapsed_seconds"] > 115:
        final.update(status="failed_titan_validation", error="Titan validation exceeded its external time allowance")
    L.save(VALIDATION / "RESULT.json", final)
    print(json.dumps(final), flush=True)
    return final


@contextmanager
def recovery_profile(proof):
    """Process-local orchestration bindings only; restore them on every exit."""
    old = (R.OUT, R.check, R.old_processes_absent)
    def guarded_check(approval=False):
        protocol = check(approval=approval)
        L.require(validation_proof() == proof, "Authorizing Titan validation proof changed")
        return protocol
    R.OUT, R.check, R.old_processes_absent = OUT, guarded_check, target_identity
    try:
        yield
    finally:
        R.OUT, R.check, R.old_processes_absent = old


def worker_main(job_id):
    check(approval=True)
    proof = validation_proof()
    job = L.read(OUT / job_id / "JOB.json")
    attempt = L.read(OUT / "ATTEMPT.json")
    L.require(job["titan_manifest_sha256"] == attempt["titan_manifest_sha256"] == L.sha(HERE / "TITAN_MANIFEST.json") and
              job["titan_validation_proof"] == attempt["titan_validation_proof"] == proof,
              "Titan child is not bound to the validated one-shot migration")
    with recovery_profile(proof):
        return R.worker_main(job_id)


def run_child(spec, job_dir, deadline, log):
    child = None
    try:
        L.require(time.monotonic() < deadline, "Titan science deadline exhausted before dispatch")
        child = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "titan_launch.py"), "--worker", spec["id"]],
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


def execute():
    started = time.monotonic()
    check(approval=True)
    identity, proof = target_identity(), validation_proof()
    smoke = S.smoke_identity()
    prior_steps = 189655 + sum(S.smoke_passed()["physical"]["counts"]["completed"].values())
    OUT.mkdir(exist_ok=False)
    end = started + 10465
    L.save(OUT / "ATTEMPT.json", {"started_est": datetime.now(L.EST).isoformat(), "supervisor_pid": os.getpid(),
        "target_identity": identity, "titan_manifest_sha256": L.sha(HERE / "TITAN_MANIFEST.json"),
        "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "titan_validation_proof": proof,
        "smoke_proof": smoke, "process_cap_seconds": 10480, "validation_charged_seconds": 120,
        "preparation_allowance_seconds": 60, "preparation_allowance_is_measured_runtime": False,
        "no_automatic_retry": True})
    result = {"status": "incomplete_titan_recovery", "jobs": [], "error": None, "no_automatic_retry": True,
              "training_hold_remains": True, "target_identity": identity, "validation_charged_seconds": 120,
              "preparation_allowance_seconds": 60, "preparation_allowance_is_measured_runtime": False,
              "scientific_functions_changed": False}
    accepted, current = [], None
    def stop(signum, frame):
        raise InterruptedError("Titan recovery supervisor signal " + str(signum))
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for sig in handlers:
        signal.signal(sig, stop)
    try:
        with recovery_profile(proof):
            R.copy_warmup()
            for job in R.jobs():
                R.check(approval=True)
                R.verify_mirror()
                L.require(S.smoke_identity() == smoke and target_identity() == identity, "Titan proof/identity changed")
                remaining = end-time.monotonic()
                L.require(remaining > 0, "Titan science shared deadline exhausted")
                directory = OUT / job["id"]
                directory.mkdir(exist_ok=False)
                L.save(directory / "JOB.json", {**job, "remaining_seconds": remaining,
                    "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "smoke_proof": smoke,
                    "titan_manifest_sha256": L.sha(HERE / "TITAN_MANIFEST.json"), "titan_validation_proof": proof})
                current = job
                print(json.dumps({"status": "starting_titan_job", "id": job["id"], "remaining_seconds": remaining}), flush=True)
                with (directory / "worker.log").open("x", encoding="utf-8") as log:
                    rc = run_child(job, directory, end, log)
                result["jobs"].append({"id": job["id"], "exit_code": rc})
                accepted.append(L.accepted_job(directory, rc))
                current = None
                L.save(directory / "SUPERVISOR.json", {"status": "accepted", "exit_code": rc, "finished_est": datetime.now(L.EST).isoformat()})
                print(json.dumps({"status": "accepted_titan_job", "id": job["id"]}), flush=True)
            R.check(approval=True)
            R.verify_mirror()
            L.require(time.monotonic() < end, "Titan final acceptance exceeded shared deadline")
            result.update(status="completed_titan_path_a_recovery", new_execution_counts_known=True,
                evaluation_report=str((OUT / "evaluation/report.json").relative_to(L.REPO)))
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), new_execution_counts_known=False)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    result.update(R.accounting(accepted, current, prior_simulator_steps=prior_steps, evaluation_output=OUT / "evaluation"),
                  elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat(),
                  all_declared_new_jobs_completed=result["status"] == "completed_titan_path_a_recovery")
    L.save(OUT / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--validate", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.check:
        check()
        print(json.dumps({"status": "source_checked_titan_not_executed", "approval_present": (HERE / "TITAN_APPROVAL.json").exists(),
            "validation_exists": VALIDATION.exists(), "attempt_exists": OUT.exists()}))
    elif args.validate:
        raise SystemExit(0 if validate()["status"] == "passed_titan_validation" else 1)
    elif args.worker:
        raise SystemExit(0 if worker_main(args.worker)["status"] == "completed" else 1)
    else:
        raise SystemExit(0 if execute()["status"] == "completed_titan_path_a_recovery" else 1)
