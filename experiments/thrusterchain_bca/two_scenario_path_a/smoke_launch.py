"""Additive runtime-smoke amendment; the original sealed Path A is unchanged."""
import argparse
from datetime import datetime
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

ORIGINAL_MANIFEST = "2f10a640684f6ea7040df9fe0388544207e1c4562a547a2ff24a04d31bf1a366"
LEARNING_CAPS = {"critic": 14, "actor": 8, "auxiliary": 14, "calibrations": 3}
PHYSICAL_CAPS = {"prefix_parity": 2688, "bank_edges": 288, "forced_qpi": 144, "returns": 200}
APPROVAL_TEXT = ("I approve the Path A runtime-smoke amendment: disposable native and original BCA checks "
    "with fixed BC, at most 14 critic, 8 actor, 14 auxiliary updates, 3 genuine calibrations, 3320 simulator "
    "transitions and 900 seconds. Only after the complete authenticated smoke passes, I approve the unchanged "
    "18-job Path A comparison and evaluation within 11490 further seconds. Cumulative optimizer limits are "
    "180014 critic, 90008 actor, 180014 auxiliary updates and 21 calibrations. The original four-hour and "
    "498400-transition ceilings remain; no Path B execution or automatic retry.")


def allocation():
    return {"schema": "two_scenario.path_a.smoke_amendment.v1", "original_manifest_sha256": ORIGINAL_MANIFEST,
        "prior_simulator_transitions": 189655, "prior_charged_seconds": 1950,
        "amendment_preparation_charge_seconds": 60, "smoke_process_seconds": 900,
        "production_process_seconds": 11490, "cumulative_time_ceiling_seconds": 14400,
        "smoke_optimizer_caps": LEARNING_CAPS, "smoke_transition_caps": PHYSICAL_CAPS,
        "smoke_total_transition_cap": 3320, "production_evaluation_transition_cap": 237600,
        "planned_cumulative_simulator_maximum": 430575, "cumulative_simulator_ceiling": 498400,
        "cumulative_optimizer_caps": {"critic": 180014, "actor": 90008, "auxiliary": 180014, "calibrations": 21},
        "smoke_disposable_condition": "bottleneck", "smoke_seed_index": 0,
        "warmup_schedule": [4], "branch_schedule": [4, 6], "branch_budget": 8,
        "duplicate_step": "serialize phase-1 state at critic7, run actor-bearing step8 twice with identical RNG",
        "coverage": "all six final banks, 16 anchors and 48 common edges each; at least one retained and one omitted edge per bank",
        "failure_policy": "stop, preserve evidence, no replacement anchors, changed thresholds or retry",
        "scientific_protocol_changed": False, "approval_created": False}


def check(approval=False):
    L.require(L.sha(HERE / "MANIFEST.json") == ORIGINAL_MANIFEST, "Original Path A seal changed")
    protocol = L.check(approval=False)
    L.require(not (HERE / "LAUNCH_APPROVAL.json").exists(), "Legacy launch approval must not bypass required smoke")
    L.require(not (HERE / "run_1").exists(), "Original Path A attempt exists; do not overlap or retry")
    manifest = L.read(HERE / "AMENDMENT_MANIFEST.json")
    L.require(manifest["allocation"] == allocation(), "Amendment budget changed")
    L.verify_pins(manifest["files"])
    if approval:
        expected = {"approval": APPROVAL_TEXT, "amendment_manifest_sha256": L.sha(HERE / "AMENDMENT_MANIFEST.json"),
                    "allow_disposable_smoke": True, "allow_training_only_after_passed_smoke": True}
        L.require(L.read(HERE / "AMENDMENT_APPROVAL.json") == expected, "New explicit smoke and conditional-run approval missing")
    return protocol


def prepare():
    L.require(L.sha(HERE / "MANIFEST.json") == ORIGINAL_MANIFEST, "Wrong original seal")
    L.check()
    L.require(not any((HERE / n).exists() for n in ("smoke_attempt_1", "run_amendment_1", "AMENDMENT_MANIFEST.json")),
              "Never reseal or reset an amendment attempt")
    names = ("smoke.py", "test_smoke.py", "smoke_evaluate.py", "test_smoke_evaluate.py", "smoke_launch.py",
             "test_smoke_launch.py", "AMENDMENT.md", "SMOKE_VERIFICATION.json", "MANIFEST.json")
    L.save(HERE / "AMENDMENT_MANIFEST.json", {"allocation": allocation(), "files": {
        (HERE / n).relative_to(L.REPO).as_posix(): L.sha(HERE / n) for n in names},
        "original_sources_verified_transitively": True, "approval_created": False, "runtime_executed": False})
    return {"status": "prepared_not_executed", "amendment_manifest_sha256": L.sha(HERE / "AMENDMENT_MANIFEST.json")}


def smoke_passed(directory=None, *, require_supervisor=True):
    directory = HERE / "smoke_attempt_1" if directory is None else Path(directory)
    final = L.read(directory / "WORKER_EXIT.json")
    report = L.read(directory / "report.json")
    L.require(final["status"] == "completed" and final["identity_unchanged"] and
              final["report_sha256"] == L.sha(directory / "report.json"), "Smoke worker/integrity/report incomplete")
    L.require(report["status"] == "passed_runtime_smoke" and report["passed"] is True and
              report["amendment_manifest_sha256"] == L.sha(HERE / "AMENDMENT_MANIFEST.json"), "Smoke did not pass this exact amendment")
    learning, physical = report["learning"], report["physical"]
    L.require(learning["status"] == "passed_learning_smoke" and learning["execution_counts_known"] and
              learning["counts"]["attempted"] == learning["counts"]["completed"] == LEARNING_CAPS and
              learning["restored_actor_next_step_exact"] == {"native": True, "bca": True}, "Learning/restore smoke incomplete")
    L.require(physical["status"] == "completed_technical_evaluation_smoke" and physical["passed"] and
              physical["completed_disposable_arms"] == 2 and len(physical["bank_coverage"]) == 6 and
              {(b["seed_index"], b["condition"]) for b in physical["bank_coverage"]} ==
                  {(s, c) for s in range(3) for c in ("honeypot", "bottleneck")} and
              all(b["anchors"] == 16 and b["common_edges"] == 48 and b["passed"] and
                  b["omitted_count"] >= 1 and b["retained_count"] >= 1 and b["omitted_count"] + b["retained_count"] == 48
                  for b in physical["bank_coverage"]), "Physical coverage/evaluator smoke incomplete")
    counts = physical["counts"]
    L.require(counts["execution_counts_known"] and counts["attempted"] == counts["completed"] and
              set(counts["completed"]) == set(PHYSICAL_CAPS) and
              counts["completed"]["prefix_parity"] == 2688 and counts["completed"]["bank_edges"] == 288 and
              all(0 <= counts["completed"][k] <= PHYSICAL_CAPS[k] for k in PHYSICAL_CAPS), "Smoke physical accounting differs")
    L.require(final["artifacts_sha256"] == L.sha(directory / "ARTIFACTS.json"), "Smoke artifact manifest changed")
    for relative, expected in L.read(directory / "ARTIFACTS.json").items():
        path = (directory / relative).resolve()
        L.require(path.is_relative_to(directory.resolve()) and L.sha(path) == expected, "Smoke raw artifact changed")
    if require_supervisor:
        terminal = L.read(directory / "RESULT.json")
        L.require(terminal["status"] == "passed_runtime_smoke" and terminal["mode"] == "smoke" and
                  terminal["execution_counts_known"] and terminal["elapsed_seconds"] <= 900,
                  "Smoke supervisor is not successfully terminal within its allowance")
    return report


def smoke_identity():
    smoke_passed()
    return {name: L.sha(HERE / "smoke_attempt_1" / name)
            for name in ("RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json")}


def _services(protocol):
    sys.path.insert(0, str(L.REPO))
    binding = L.load("_path_a_binding", HERE / "binding.py")
    train = L.load("_path_a_train", HERE / "train.py")
    services = train.load_runtime(binding)
    review = {"training_authorized": True, "protocol_sha256": binding.I.digest(protocol),
        "dataset_sha256": binding.DATASET_SHA256, "geometry_sha256": binding.GEOMETRY_SHA256,
        "amendment_approval_sha256": L.sha(HERE / "AMENDMENT_APPROVAL.json")}
    return binding, train, services, review


def run_smoke(protocol, output, deadline):
    binding, train, services, review = _services(protocol)
    review["scope"] = "disposable_path_a_smoke"
    # Both conditions must reproduce their exact prepared rows in the target
    # numerical runtime before even the disposable learning updates begin.
    for condition in protocol["conditions"]:
        prepared = binding.prepare(services.native, condition, protocol=protocol)
        L.require(prepared.metadata == services.expected_data_receipt["conditions"][condition]["metadata"],
                  "Target-runtime data binding differs: " + condition)
    learning_module = L.load("_path_a_learning_smoke", HERE / "smoke.py")
    physical_module = L.load("_path_a_physical_smoke", HERE / "smoke_evaluate.py")
    worker = L.load("_path_a_worker", HERE / "worker.py")
    learning_out, physical_out = output / "learning", output / "physical"
    learning_out.mkdir(exist_ok=False)
    physical_out.mkdir(exist_ok=False)
    def progress(value):
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(L.plain(value), sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
    policies, learning = learning_module.run_learning_smoke(protocol, services, review, deadline, learning_out, progress)
    L.require(learning["status"] == "passed_learning_smoke", "Disposable learning/restore/calibration smoke failed")
    callbacks = {arm: worker.callbacks(services.runtime, value["adapter"], value["state"], {"disposable": True, "arm": arm})
                 for arm, value in policies.items()}
    core = L.PAYLOAD / "transfer_bundle/core_v4"
    sys.path.insert(0, str(core))
    from thrusterchain import environment as canonical
    L.require(Path(canonical.__file__).resolve() == core / "thrusterchain/environment.py", "Wrong canonical plant")
    physics = canonical.Physics(stall=True, stall_slope=3.)
    original = L.read(L.V2 / "setup_attempt_1/RESULT.json")
    L.require(physics.as_dict() == original["physics"] and canonical.provenance() == original["runtime"]["canonical"], "Changed physical runtime")
    class Counter(worker.PhysicalCounter):
        CAPS = PHYSICAL_CAPS
        def __call__(self, state, action):
            L.require(sum(self.attempted.values()) < 3320, "Smoke transition cap exhausted")
            return super().__call__(state, action)
    counter = Counter(lambda x, a: canonical.dynamics(x, a, physics), deadline, progress)
    geometry = L.read(L.V2 / "setup_attempt_1/geometry.json")
    task = binding.D.T.Task(counter, geometry)
    A, B = canonical.matrices(physics)
    controller = binding.D.T.tracking_controller(A, B, protocol["geometry_recipe"])
    physical = physical_module.check_and_evaluate(task, controller, geometry, protocol, callbacks, counter,
                   lambda name, value: L.save(physical_out / (name + ".json"), value))
    for policy in callbacks.values():
        policy["assert_unchanged"]()
    L.require(physical["passed"], "Final frozen bank coverage or evaluator smoke failed; no replacements")
    return {"status": "passed_runtime_smoke", "passed": True, "learning": learning, "physical": physical,
            "amendment_manifest_sha256": L.sha(HERE / "AMENDMENT_MANIFEST.json"),
            "runtime": services.runtime.runtime_receipt(services.native), "both_condition_data_bindings_exact": True,
            "production_training_updates": 0, "all_learning_work_disposable_and_not_reused": True}


def worker_main(job_id):
    L.require(os.name == "posix", "Manual existing WSL runtime only")
    started = time.monotonic()
    protocol = check(approval=True)
    is_smoke = job_id == "smoke"
    output = HERE / "smoke_attempt_1" if is_smoke else HERE / "run_amendment_1" / job_id
    job = L.read(output / "JOB.json")
    L.require(job["id"] == job_id and job["amendment_manifest_sha256"] == L.sha(HERE / "AMENDMENT_MANIFEST.json"), "Wrong amended job")
    if not is_smoke:
        proof = smoke_identity()
        L.require(job["smoke_proof"] == proof == L.read(HERE / "run_amendment_1/ATTEMPT.json")["smoke_proof"],
                  "Scientific job's authorizing smoke proof changed")
        expected = next((j for j in L.jobs() if j["id"] == job_id), None)
        L.require(expected is not None and all(job[k] == v for k, v in expected.items()), "Wrong scientific job")
    end = started + min(job["remaining_seconds"], 885 if is_smoke else 11475)
    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Amended shared deadline reached")
    result = {"status": "failed", "identity_unchanged": False, "error": None, "no_automatic_retry": True}
    try:
        if is_smoke:
            report = run_smoke(protocol, output, deadline)
            L.save(output / "report.json", report)
            paths = [output / "report.json"]
            for folder in ("learning", "physical"):
                paths.extend(p for p in (output / folder).rglob("*") if p.is_file())
            L.save(output / "ARTIFACTS.json", {p.relative_to(output).as_posix(): L.sha(p) for p in paths})
            result["artifacts_sha256"] = L.sha(output / "ARTIFACTS.json")
        else:
            binding, train, services, review = _services(protocol)
            if job["arm"] == "evaluation":
                W = L.load("_path_a_worker", HERE / "worker.py")
                report = W.evaluate_job(protocol, output, deadline, binding, train, services, review)
            else:
                warmup = None if job["arm"] == "warmup" else output.parent / f"{job['condition']}_s{job['seed_index']}_warmup"
                if warmup is not None:
                    L.accepted_job(warmup, 0)
                report = train.run_job(protocol, job["condition"], job["seed_index"], job["arm"], output,
                    warmup_dir=warmup, deadline=deadline, services=services, training_review=review)
                L.require(report["status"] == "completed_training_job", "Scientific job incomplete")
        check(approval=True)
        result.update(status="completed", identity_unchanged=True, report_sha256=L.sha(output / "report.json"))
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    result.update(elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat())
    L.save(output / "WORKER_EXIT.json", result)
    print(json.dumps(result), flush=True)
    return result


def run_child(spec, job_dir, deadline, log):
    """Own the child through receipt persistence and every failure path."""
    child = None
    try:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "smoke_launch.py"), "--worker", spec["id"]],
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


def supervise(mode):
    L.require(os.name == "posix", "Manual existing WSL runtime only")
    started = time.monotonic()
    check(approval=True)
    is_smoke = mode == "smoke"
    # This gate runs before scientific reservation, child creation or initialization.
    proof = None if is_smoke else smoke_identity()
    out = HERE / ("smoke_attempt_1" if is_smoke else "run_amendment_1")
    out.mkdir(exist_ok=False)
    cap = 900 if is_smoke else 11490
    deadline = started + cap - 15
    L.save(out / "ATTEMPT.json", {"mode": mode, "started_est": datetime.now(L.EST).isoformat(),
         "supervisor_pid": os.getpid(), "cap_seconds": cap,
         "amendment_manifest_sha256": L.sha(HERE / "AMENDMENT_MANIFEST.json"), "smoke_proof": proof,
         "no_automatic_retry": True})
    result = {"status": "incomplete", "mode": mode, "jobs": [], "error": None,
              "no_automatic_retry": True, "training_hold_remains": True, "smoke_proof": proof}
    accepted = []
    def stop(signum, frame):
        raise InterruptedError("Supervisor signal " + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        for spec in ([{"id": "smoke", "arm": "smoke"}] if is_smoke else L.jobs()):
            check(approval=True)
            if not is_smoke:
                L.require(smoke_identity() == proof, "Authorizing smoke proof changed")
            remaining = deadline-time.monotonic()
            L.require(remaining > 0, "Shared process deadline exhausted")
            job_dir = out if is_smoke else out / spec["id"]
            if not is_smoke:
                job_dir.mkdir(exist_ok=False)
            L.save(job_dir / "JOB.json", {**spec, "remaining_seconds": remaining,
                   "amendment_manifest_sha256": L.sha(HERE / "AMENDMENT_MANIFEST.json"), "smoke_proof": proof})
            with (job_dir / "worker.log").open("x", encoding="utf-8") as log:
                rc = run_child(spec, job_dir, deadline, log)
            result["jobs"].append({"id": spec["id"], "exit_code": rc})
            L.require(rc == 0, "Amended worker failed")
            report = smoke_passed(out, require_supervisor=False) if is_smoke else L.accepted_job(job_dir, rc)
            accepted.append(report)
            L.save(job_dir / "SUPERVISOR.json", {"status": "accepted", "exit_code": rc, "finished_est": datetime.now(L.EST).isoformat()})
        check(approval=True)
        if not is_smoke:
            L.require(smoke_identity() == proof, "Authorizing smoke proof changed before final acceptance")
        result.update(status="passed_runtime_smoke" if is_smoke else "completed_path_a_study",
                      execution_counts_known=True)
        if is_smoke:
            result["smoke_optimizer_counts"] = accepted[0]["learning"]["counts"]
            result["smoke_physical_counts"] = accepted[0]["physical"]["counts"]
        else:
            result["scientific_training_counts"] = {k: sum(r["counts"]["completed"][k] for r in accepted[:-1]) for k in LEARNING_CAPS}
            result["evaluation_transitions"] = sum(accepted[-1]["physical_counts"]["completed"].values())
            result["cumulative_transitions"] = 189655 + sum(smoke_passed()["physical"]["counts"]["completed"].values()) + result["evaluation_transitions"]
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), execution_counts_known=False,
                      accounting_note="Preserve current dispatch/report lower bounds and declared per-phase/job upper bounds; never assume zero or retry.")
    result.update(elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(L.EST).isoformat())
    L.save(out / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.check:
        check()
        print(json.dumps({"status": "source_checked_not_executed", "approval_present": (HERE / "AMENDMENT_APPROVAL.json").exists(),
                          "smoke_attempt_exists": (HERE / "smoke_attempt_1").exists(), "scientific_attempt_exists": (HERE / "run_amendment_1").exists()}))
    elif args.worker:
        raise SystemExit(0 if worker_main(args.worker)["status"] == "completed" else 1)
    else:
        result = supervise("smoke" if args.smoke else "production")
        raise SystemExit(0 if result["status"] in ("passed_runtime_smoke", "completed_path_a_study") else 1)
