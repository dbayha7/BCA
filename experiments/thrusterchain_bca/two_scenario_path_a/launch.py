"""Thin, one-attempt Path A dispatcher. Preparation never imports a learner."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
V2 = HERE.parent / "two_scenario_v2"
REPAIR = REPO / "analysis/two_scenario_v2_repair_v1"
PAYLOAD = HERE.parent / "titan_20261026/payload"
RUNTIME = HERE.parent / "zeus_20261026_validation/central_quartile_training_v1_evaluation_rerun1"
EST = timezone(timedelta(hours=-5))
APPROVAL_TEXT = ("I approve Path A only: unchanged v2 task, reviewed separating slab for Bottleneck, "
                 "Honeypot danger exclusion, native UniFloral versus original BCA kappa 3 with fixed BC, "
                 "three paired seeds per condition, 18 jobs and 12 final policies, at most 180000 critic, "
                 "90000 actor and 180000 auxiliary updates, 18 calibrations and 237600 evaluation transitions. "
                 "I approve the final geometry, dataset and protocol, with a 12450-second remaining process cap; "
                 "the original four-hour and 498400-transition cumulative ceilings are not reset. No Path B execution.")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value.tolist() if hasattr(value, "tolist") else value


def save(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def load(name, path):
    if name in sys.modules:
        result = sys.modules[name]
        require(Path(result.__file__).resolve() == Path(path).resolve(), "Module path collision: " + name)
        return result
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configuration():
    p = read(V2 / "protocol.json")
    p.update(schema="two_scenario.path_a.v1", status="prepared_for_final_review",
             training_authorized=False, scientific_evaluation_authorized=False,
             condition_anchor_times={"honeypot": 29, "bottleneck": 25},
             condition_anchor_recipes={"honeypot": "controller_until25_then_opposite25_through28",
                                       "bottleneck": "controller_until25"},
             common_actions=["controller", "opposite_controller", "zero"],
             bottleneck_mask="reviewed_closed_projection_slab_0.2_to_0.8",
             prior_simulator_transitions=189655, prior_charged_seconds=1890,
             final_preparation_reserved_seconds=60, total_process_cap_seconds=12450,
             internal_process_seconds=12435, external_timeout_seconds=12445, kill_grace_seconds=5,
             cumulative_transition_ceiling=498400, cumulative_time_ceiling_seconds=14400,
             planned_cumulative_transition_maximum=427255,
             evaluation_tight_maximum=233376, unused_evaluation_reserve=4224,
             setup_transition_reads="hash/schema only; no dynamics replay",
             development_status="inspected v2 development construction; new frozen evaluation seeds, exploratory three-seed comparison",
             automatic_retry=False)
    return p


def jobs():
    result = []
    for condition in ("honeypot", "bottleneck"):
        for seed in range(3):
            for arm in ("warmup", "native", "bca"):
                result.append({"id": f"{condition}_s{seed}_{arm}", "condition": condition,
                               "seed_index": seed, "arm": arm})
    result.append({"id": "evaluation", "arm": "evaluation"})
    return result


def source_files():
    old = read(V2 / "MANIFEST.json")
    paths = {REPO / relative for relative in old["files"]}
    paths.update(V2 / n for n in ("MANIFEST.json", "SETUP_APPROVAL.json"))
    paths.update((V2 / "setup_attempt_1").glob("*.json"))
    paths.add(V2 / "setup_attempt_1/parent_data.npz")
    paths.update(REPAIR / n for n in ("connector_cut.py", "CUT_RULE.json", "CUT_REPORT.json",
                                     "danger_diagnostic.py", "test_danger_diagnostic.py", "DANGER_CONTRACT.json"))
    paths.update((REPAIR / "danger_attempt_1").glob("*.json"))
    paths.add(REPAIR / "danger_attempt_1/transitions.jsonl")
    paths.update(RUNTIME / n for n in ("runtime.py", "common.py", "state_exact.py"))
    native = PAYLOAD / "unifloral_td3bc_validation_v1"
    paths.update(native / n for n in ("native_loader.py", "candidate.py", "provenance/unifloral.json"))
    paths.update(p for p in (native / "upstream").rglob("*") if p.is_file())
    uncertainty = PAYLOAD / "unifloral_bca_pipeline_validation_v1"
    paths.update(uncertainty / n for n in ("bca_bridge.py", "uncertainty_provenance.json"))
    paths.update(p for p in (uncertainty / "uncertainty_source").rglob("*") if p.is_file())
    paths.add(PAYLOAD / "unifloral_checkpoint_v1/training.py")
    paths.update(p for p in HERE.iterdir() if p.suffix in (".py", ".md") and p.is_file())
    paths.update(HERE / n for n in ("protocol.json", "DATA_RECEIPT.json", "VERIFICATION.json"))
    return paths


def verify_pins(files):
    for relative, expected in files.items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO) and path.is_file() and sha(path) == expected,
                "Prepared identity changed: " + relative)


def verify_setup():
    diagnostic = load("_path_a_danger_diagnostic", REPAIR / "danger_diagnostic.py")
    diagnostic.check()
    result = read(REPAIR / "danger_attempt_1/RESULT.json")
    require(result["decision"] == "passed_development_diagnostic" and result["prepared_identity_unchanged"] and
            result["execution_counts_known"] and result["counts"]["plant_calls_completed"] == 613 and
            result["cumulative_transition_upper"] == 189655, "Diagnostic lineage/accounting differs")
    require(read(REPAIR / "CUT_REPORT.json")["decision"] == "saved_data_mask_checks_passed", "Missing cut evidence")
    require(not list((HERE.parent / "two_scenario_v3").glob("setup_attempt_*")), "Path B work requires budget reconciliation")


def prepare():
    require(not (HERE / "MANIFEST.json").exists() and not (HERE / "run_1").exists(), "Never reseal an existing attempt")
    verify_setup()
    require(read(HERE / "protocol.json") == configuration(), "Final protocol differs")
    require(read(HERE / "DATA_RECEIPT.json")["status"] == "verified_final_binding", "Missing final exact-data validation")
    save(HERE / "MANIFEST.json", {"schema": "two_scenario.path_a.seal.v1", "files": {
        p.relative_to(REPO).as_posix(): sha(p) for p in sorted(source_files())}, "jobs": jobs(),
        "training_approval_created": False, "target_runtime_end_to_end_verified": False})
    return {"status": "sealed_for_review_not_authorized", "manifest_sha256": sha(HERE / "MANIFEST.json")}


def approval_expected():
    return {"approval": APPROVAL_TEXT, "manifest_sha256": sha(HERE / "MANIFEST.json"),
            "path": "A", "process_cap_seconds": 12450, "training_authorized": True}


def check(approval=False):
    manifest = read(HERE / "MANIFEST.json")
    verify_pins(manifest["files"])
    require(manifest["jobs"] == jobs(), "Job plan changed")
    require(read(HERE / "protocol.json") == configuration(), "Protocol changed")
    verify_setup()
    if approval:
        require(read(HERE / "LAUNCH_APPROVAL.json") == approval_expected(), "Final Path A launch approval absent or mismatched")
    return read(HERE / "protocol.json")


def accepted_job(job_dir, exit_code):
    require(exit_code == 0, "Child exited unsuccessfully")
    final = read(job_dir / "WORKER_EXIT.json")
    require(final["status"] == "completed" and final["identity_unchanged"], "Worker/postcheck incomplete")
    report = read(job_dir / "report.json")
    require(sha(job_dir / "report.json") == final["report_sha256"], "Report differs from worker exit receipt")
    if job_dir.name == "evaluation":
        require(report["status"] == "completed_frozen_evaluation" and report["completed_policy_evaluations"] == 12 and
                report["completed_paired_cells"] == 6 and report["completed_return_episodes"] == 1536, "Incomplete evaluation")
        physical = report["physical_counts"]
        caps = {"returns": 153600, "own_qpi": 57600, "common_reference": 21600, "prefix_parity": 4800}
        require(physical["execution_counts_known"] and physical["attempted"] == physical["completed"] and
                set(physical["completed"]) == set(caps) and
                all(0 <= physical["completed"][k] <= caps[k] for k in caps), "Evaluation accounting differs")
        require(report["logical_completed_transitions_by_phase"] == physical["completed"], "Logical/physical counts disagree")
    else:
        caps = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "calibrations": 1}
        require(report["status"] == "completed_training_job" and report["execution_counts_known"] and
                report["counts"]["attempted"] == caps and report["counts"]["completed"] == caps and
                report["simulator_steps"] == 0, "Training accounting differs")
    return report


def execute():
    require(os.name == "posix", "Manual existing WSL runtime required; no assistant execution")
    started = time.monotonic()
    protocol = check(approval=True)
    deadline = started + protocol["internal_process_seconds"]
    out = HERE / "run_1"
    out.mkdir(exist_ok=False)
    save(out / "ATTEMPT.json", {"manifest_sha256": sha(HERE / "MANIFEST.json"),
         "started_est": datetime.now(EST).isoformat(), "supervisor_pid": os.getpid(),
         "cap_seconds": 12450, "no_automatic_retry": True,
         "missing_terminal_receipt": "Incomplete, never assume zero work or automatically retry."})
    result = {"status": "incomplete", "jobs": [], "training_hold_remains": True,
              "prior_simulator_transitions": 189655, "prior_charged_seconds": 1890,
              "final_preparation_charge_seconds": 60, "error": None, "no_automatic_retry": True}
    child = None
    completed_reports = []
    def stop(signum, frame):
        raise InterruptedError("Supervisor signal " + str(signum))
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, stop)
    try:
        for job in jobs():
            check(approval=True)
            remaining = deadline - time.monotonic()
            require(remaining > 0, "Global execution deadline reached")
            job_dir = out / job["id"]
            job_dir.mkdir(exist_ok=False)
            save(job_dir / "JOB.json", {**job, "remaining_seconds": remaining,
                 "manifest_sha256": sha(HERE / "MANIFEST.json")})
            with (job_dir / "worker.log").open("x", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "worker.py"), "--job", job["id"]],
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                save(job_dir / "PROCESS.json", {"pid": child.pid, "started_est": datetime.now(EST).isoformat()})
                try:
                    rc = child.wait(timeout=max(.001, deadline-time.monotonic()))
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
                finally:
                    child = None
            row = {"id": job["id"], "exit_code": rc, "finished_est": datetime.now(EST).isoformat()}
            result["jobs"].append(row)
            completed_reports.append(accepted_job(job_dir, rc))
            save(job_dir / "SUPERVISOR.json", row)
            print(json.dumps(row), flush=True)
        check(approval=True)
        result["status"] = "completed_path_a_study"
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
    # Never infer exact work from successful subprocess exits alone.
    result["all_declared_jobs_completed"] = result["status"] == "completed_path_a_study"
    result["execution_counts_known"] = result["all_declared_jobs_completed"]
    if result["all_declared_jobs_completed"]:
        result["completed_training_counts"] = {
            k: sum(r["counts"]["completed"][k] for r in completed_reports[:-1])
            for k in ("critic", "actor", "auxiliary", "calibrations")}
        result["completed_evaluation_transitions"] = sum(completed_reports[-1]["physical_counts"]["completed"].values())
        result["cumulative_simulator_transitions"] = 189655 + result["completed_evaluation_transitions"]
        result["evaluation_report"] = str((out / "evaluation/report.json").relative_to(REPO))
    else:
        result["accounting_note"] = "Inspect completed reports and current dispatch logs; unfinished jobs retain planned upper bounds, not zero."
        result["verified_completed_training_lower_bounds"] = {
            k: sum(r["counts"]["completed"][k] for r in completed_reports if r.get("status") == "completed_training_job")
            for k in ("critic", "actor", "auxiliary", "calibrations")}
        result["per_unfinished_training_job_upper_bounds"] = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "calibrations": 1}
        result["unfinished_evaluation_upper_bound"] = 237600 if (out / "evaluation/JOB.json").exists() else 0
    result.update(elapsed_seconds=time.monotonic()-started, finished_est=datetime.now(EST).isoformat())
    save(out / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare()))
    elif args.check:
        check()
        print(json.dumps({"status": "source_checked_not_executed", "manifest_sha256": sha(HERE / "MANIFEST.json"),
                          "approval_present": (HERE / "LAUNCH_APPROVAL.json").exists(), "attempt_exists": (HERE / "run_1").exists()}))
    else:
        raise SystemExit(0 if execute()["status"] == "completed_path_a_study" else 1)
