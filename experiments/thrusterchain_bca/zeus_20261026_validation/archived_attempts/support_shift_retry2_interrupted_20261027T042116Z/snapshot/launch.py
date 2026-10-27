"""Approved support pilot: exclusive manual launch, read-only status/preflight."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from common import HERE, PAYLOAD, read, sha, atomic_json, module, require

require(HERE.resolve() == Path(__file__).resolve().parent, "Conflicting common module; use a fresh retry process")
retry_gate = module("_support_retry_gate", HERE / "retry_gate.py")

EXPECTED_ENV = "/home/david/bca-operator-audit-20261026"
prior = module("_support_supervisor_primitives", HERE.parent / "matched_policy_pilot_v1/launch.py")
save_new, read_report, supervise = prior.save_new, prior.read_report, prior.supervise
reserve = prior.reserve


def validate_protocol(p):
    fixed = {"schema": "unifloral.support_shift_pilot.v1", "warmup_critic_updates": 10000,
        "warmup_actor_updates": 5000, "warmup_auxiliary_updates": 10000,
        "branch_critic_updates": 2000, "branch_actor_updates": 1000, "branch_auxiliary_updates": 2000,
        "calibration_boundaries": [10000,11000], "final_critic_step": 12000,
        "total_critic_updates": 64000, "total_actor_updates": 32000, "total_auxiliary_updates": 64000,
        "total_calibrations": 16, "total_posterior_draws": 16000, "total_replay_row_draws": 8192000,
        "training_rows_per_condition": 57752, "training_episodes": 768, "calibration_episodes": 128,
        "gradient_probe_rows_per_cell": 768, "support_cutoff": .5, "support_coordinate": 0,
        "bca_kappa": 3., "actor_distance_factor": 19.29619572569095, "gradient_norm_minimum": 1e-6,
        "initial_residual_scale": 1., "alpha": .1, "credibility": .95, "posterior_draws": 1000,
        "batch_size": 256, "gamma": .98, "horizon": 100, "progress_every": 20,
        "warmup_timeout_seconds": 1800, "branch_timeout_seconds": 300, "evaluation_timeout_seconds": 600,
        "total_timeout_seconds": 9000, "evaluation_policies": 16, "evaluation_episodes": 512,
        "simulator_transition_cap": 51200, "evaluation_bootstrap_draws": 2000,
        "evaluation_bootstrap_seed": 2026103091, "return_band_atol": 1e-6, "return_band_rtol": 1e-6,
        "automatic_restart": False, "resume": False, "full_training_hold": True,
        "head_ablation_deferred": True, "failed_one_sided_variant_excluded": True,
        "conditions": ["broad_count_matched", "restricted"], "arms": ["native","bca","distance"],
        "cells": ["seed0_broad_count_matched","seed0_restricted","seed1_restricted","seed1_broad_count_matched"]}
    for key, value in fixed.items():
        require(p.get(key) == value and not (type(value) is not bool and isinstance(p.get(key), bool)), "Approved protocol changed: " + key)
    require(p["seeds"] == [dict(index=i, native_seed=2026103001+i, aux_seed=2026103011+i,
        branch_native_seed=2026103031+i, branch_aux_seed=2026103041+i) for i in (0,1)], "Fresh paired seeds differ")
    require(p["evaluation_seeds"] == list(range(2026104000,2026104032)), "Evaluation seeds differ")
    return p


def verify_prepared(root=HERE):
    root = Path(root)
    retry_gate.lineage(root)
    m = read(root / "PREPARED.json")
    require(m["scientific_execution_during_preparation"] is False, "Preparation cannot contain scientific execution")
    for name, expected in m["files"].items():
        require(Path(name).name == name and sha(root / name) == expected, "Prepared file changed: " + name)
    for name, expected in m["dependencies"].items():
        path = (root.parent / name).resolve()
        require(path.is_relative_to(root.parent.resolve()) and sha(path) == expected, "Dependency changed: " + name)
    p = validate_protocol(read(root / "protocol.json"))
    require(sha(root / "DATA_PLAN.json") == p["data_plan_sha256"], "Data row plan changed")
    require(sha(root / "PROTOCOL.md") == p["protocol_text_sha256"], "Reviewed protocol text changed")
    return p, m


def verify_inputs():
    # Original helper verifies pinned sources/data by hashes only; it does not restore a model.
    return prior.verify_inputs()


def job_specs(p, root=HERE):
    root = Path(root)
    jobs = []
    for cell in p["cells"]:
        seed, condition = cell.split("_",1)
        jobs.append({"id": cell + "/warmup", "kind": "warmup", "cell": cell,
                     "seed_index": int(seed[-1]), "condition": condition})
    for cell in p["cells"]:
        seed, condition = cell.split("_",1)
        arms = p["arms"] if seed == "seed0" else list(reversed(p["arms"]))
        for arm in arms:
            jobs.append({"id": cell + "/" + arm, "kind": "branch", "cell": cell,
                "seed_index": int(seed[-1]), "condition": condition, "arm": arm,
                "warmup_path": str(root / "run" / cell / "warmup")})
    jobs.append({"id": "evaluation", "kind": "evaluation", "jobs_root": str(root / "run")})
    return jobs


def scoped_process_check():
    targets = {(HERE / "launch.py").resolve(), (HERE.parent / "support_shift_pilot_v1/launch.py").resolve(), (HERE.parent / "matched_policy_pilot_v1/launch.py").resolve(),
        (HERE.parent / "bca_one_sided_v1/launch.py").resolve(), (PAYLOAD.parent / "launch.py").resolve()}
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if process.stat().st_uid != os.getuid() or not (process / "comm").read_text().strip().startswith(("python","pypy")):
                continue
            for raw in (process / "cmdline").read_bytes().split(b"\0")[1:]:
                path = Path(os.fsdecode(raw))
                if path.name != "launch.py":
                    continue
                if not path.is_absolute():
                    path = (process / "cwd").resolve(strict=True) / path
                if path.resolve() in targets:
                    matches.append(int(process.name)); break
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise RuntimeError("Scoped process check denied; no launch") from exc
    require(not matches, "Existing scoped BCA process: " + str(matches))


def read_status(root=HERE):
    root = Path(root)
    if (root / "RUN_RESULT.json").is_file():
        return read(root / "RUN_RESULT.json")
    if not (root / "ATTEMPT.json").is_file():
        return {"status": "not_started", "scientific_execution_started": False,
                "retry_lineage": retry_gate.accounting(root),
                "manual_process_clearance_receipt_present": (root / "ORIGIN_STOPPED.json").is_file(),
                "launch_requires": "Successful --check-origin-stopped receipt and a fresh process check before reserve"}
    state = read(root / "PROGRESS.json") if (root / "PROGRESS.json").is_file() else {"status": "reserved"}
    job = state.get("current_job")
    if job:
        report, error = read_report(root / "run" / job / "progress.json")
        state.update(worker_progress=report, progress_read_error=error)
    state["retry_lineage"] = retry_gate.accounting(root)
    state["note"] = "Without a terminal result, liveness and final counts are uncertain. Do not relaunch."
    return state


def verify_warmup_pairs(p, root=HERE):
    """Check fresh numeric weights, not condition-specific normalization constants."""
    reports = {cell: read(Path(root) / "run" / cell / "warmup/report.json") for cell in p["cells"]}
    for index in (0,1):
        left, right = [reports[f"seed{index}_{condition}"] for condition in p["conditions"]]
        require(left["initial_parameter_hashes"] == right["initial_parameter_hashes"], "Initial networks differ across paired data conditions")
        require(all(value == 0 for value in left["initial_optimizer_steps"].values()) and
                all(value == 0 for value in right["initial_optimizer_steps"].values()), "Initial optimizer counts are not fresh")
    a, b = [reports[f"seed{index}_restricted"]["initial_parameter_hashes"] for index in (0,1)]
    require(a["actor"] != b["actor"] and a["critic"] != b["critic"], "Independent seed networks are identical")
    return {"paired_initial_parameters_equal": True, "different_seed_networks_distinct": True,
            "initial_optimizer_steps_zero": True, "normalization_excluded_from_weight_comparison": True}


def expected_status(job):
    return "completed_support_" + (job["kind"] if job["kind"] != "branch" else "branch")


def worker_entry(token, job_id):
    p, _ = verify_prepared()
    require(sys.platform == "linux" and Path(sys.prefix).resolve() == Path(EXPECTED_ENV).resolve(), "Use existing isolated Ubuntu Python")
    attempt = read(HERE / "ATTEMPT.json")
    require(token == attempt["token"] and os.getppid() == attempt["supervisor_pid"] and
            not (HERE / "RUN_RESULT.json").exists(), "Worker is not owned by active supervisor")
    jobs = {job["id"]: job for job in job_specs(p)}
    require(job_id in jobs, "Unknown job")
    output = HERE / "run" / job_id
    dispatch = read(output / "DISPATCH.json")
    require(dispatch["token"] == token, "Dispatch reservation differs")
    save_new(output / "WORKER_ENTRY.json", {"job": job_id, "pid": os.getpid(), "protocol_sha256": sha(HERE / "protocol.json")})
    job = {**jobs[job_id], "timeout_seconds": dispatch["timeout_seconds"],
           "deadline_monotonic": dispatch["deadline_monotonic"]}
    worker = module("_support_" + job["kind"] + "_worker", HERE / ("evaluate_worker.py" if job["kind"] == "evaluation" else "train_worker.py"))
    result = worker.run(p, output, job)
    if not (output / "report.json").exists():
        save_new(output / "report.json", result)
    return 0 if result.get("status") == expected_status(job) else 1


def execute(p, manifest):
    require(sys.platform == "linux" and Path(sys.prefix).resolve() == Path(EXPECTED_ENV).resolve(), "Use existing isolated Ubuntu GPU Python")
    require("include-system-site-packages = false" in (Path(sys.prefix) / "pyvenv.cfg").read_text(), "Environment must not inherit packages")
    retry_gate.require_clearance(HERE)
    retry_lineage = retry_gate.accounting(HERE)
    scoped_process_check()
    started, token = time.monotonic(), str(uuid.uuid4())
    reserve(HERE, token, started)
    completed_jobs, paired = [], None
    status, error = "running", None
    print("Reserved support pilot retry attempt 2: four fresh warmups, twelve branches, final evaluation; unchanged 150-minute per-attempt cap. Prior interrupted work remains separately recorded. No automatic restart.", flush=True)
    try:
        for specification in job_specs(p):
            if specification["kind"] == "branch" and paired is None:
                paired = verify_warmup_pairs(p)
                save_new(HERE / "run/PAIRED_INITIALIZATION.json", paired)
            remaining = p["total_timeout_seconds"] - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Global pilot deadline exhausted")
            verify_prepared()
            job_id = specification["id"]
            output = HERE / "run" / job_id
            output.mkdir(parents=True, exist_ok=False)
            cap = min(p[specification["kind"] + "_timeout_seconds"], remaining)
            dispatch = {"token": token, "job": specification, "timeout_seconds": cap,
                "supervisor_pid": os.getpid(), "deadline_monotonic": started + p["total_timeout_seconds"],
                "protocol_sha256": sha(HERE / "protocol.json")}
            save_new(output / "DISPATCH.json", dispatch)
            atomic_json(HERE / "PROGRESS.json", {"status": "running", "current_job": job_id,
                "finished_jobs": len(completed_jobs), "elapsed_seconds": time.monotonic()-started,
                "total_seconds_cap": p["total_timeout_seconds"], "no_automatic_restart": True})
            print("Starting " + job_id + "; worker cap " + str(round(cap,2)) + " seconds.", flush=True)
            command = [sys.executable,"-B","-u",str(HERE / "launch.py"),"--worker-token",token,"--job",job_id]
            outcome = supervise(command, output, cap, prior.child_environment(), job_id,
                                deadline=started+p["total_timeout_seconds"])
            report, report_error = read_report(output / "report.json")
            complete = bool(outcome["exit_code"] == 0 and not outcome["timed_out"] and not outcome["interrupted"]
                            and report and report.get("status") == expected_status(specification))
            result = {"job": job_id, **outcome, "completed": complete,
                "report_sha256": sha(output / "report.json") if report else None,
                "worker_status": report.get("status") if report else None, "report_read_error": report_error,
                "execution_counts_known": bool(report and report.get("execution_counts_known",False)),
                "ledger": report.get("ledger",report.get("counts")) if report else None}
            completed_jobs.append(result)
            save_new(output / "SUPERVISOR_RESULT.json", result)
            print({"job_finished": job_id, "completed": complete, "worker_status": result["worker_status"]}, flush=True)
            if not complete:
                status = "stopped_after_incomplete_job"
                break
        else:
            status = "completed_bounded_support_shift_pilot"
    except BaseException as exc:
        status = "pilot_interrupted" if isinstance(exc,KeyboardInterrupt) else "pilot_failed"
        error = type(exc).__name__ + ": " + str(exc)
    try:
        verify_prepared()
        verify_inputs()
        identity = True
    except Exception as exc:
        identity, status = False, "prepared_source_changed"
        error = (error or "") + " Prepared identity: " + str(exc)
    evaluation, evaluation_error = read_report(HERE / "run/evaluation/report.json")
    result = {"status": status, "jobs": completed_jobs, "elapsed_seconds": time.monotonic()-started,
        "retry_lineage": retry_lineage,
        "protocol_sha256": sha(HERE / "protocol.json"), "prepared_identity_unchanged": identity,
        "execution_counts_known": bool(completed_jobs) and all(job["execution_counts_known"] for job in completed_jobs),
        "all_declared_jobs_completed": len(completed_jobs)==17 and all(job["completed"] for job in completed_jobs),
        "paired_initialization": paired, "decision": evaluation.get("decision") if evaluation else None,
        "error": error, "evaluation_report_read_error": evaluation_error,
        "full_training_hold": True, "no_automatic_restart": True, "resume_supported": False}
    save_new(HERE / "RUN_RESULT.json", result)
    atomic_json(HERE / "PROGRESS.json", {"status": status, "finished_jobs": len(completed_jobs),
        "elapsed_seconds": result["elapsed_seconds"], "terminal_receipt": "RUN_RESULT.json", "no_automatic_restart": True})
    print(__import__("json").dumps(result), flush=True)
    return 0 if status == "completed_bounded_support_shift_pilot" and identity else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--check-origin-stopped", action="store_true")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        print(__import__("json").dumps(read_status()), flush=True); return 0
    if args.worker_token:
        return worker_entry(args.worker_token,args.job)
    p, manifest = verify_prepared()
    verify_inputs()
    if args.check_origin_stopped:
        print(__import__("json").dumps(retry_gate.check_origin_stopped(HERE)), flush=True)
        return 0
    if not args.execute:
        print(__import__("json").dumps({"status":"prepared_sources_verified_no_scientific_execution",
            "protocol_sha256":sha(HERE/"protocol.json"), "critic_updates_cap":64000,"actor_updates_cap":32000,
            "auxiliary_updates_cap":64000,"calibrations_cap":16,"evaluation_episodes_cap":512,
            "evaluation_transitions_cap":51200,"total_seconds_cap":9000,"attempt_reserved":False,
            "retry_lineage":retry_gate.accounting(HERE),
            "manual_process_clearance_receipt_present":(HERE/"ORIGIN_STOPPED.json").is_file(),
            "launch_requires":"Successful --check-origin-stopped receipt and a fresh process check before reserve"}), flush=True)
        return 0
    def stop(signum,frame):
        raise KeyboardInterrupt("Supervisor signal " + str(signum))
    for name in ("SIGINT","SIGTERM","SIGHUP"):
        if hasattr(signal,name): signal.signal(getattr(signal,name),stop)
    return execute(p,manifest)


if __name__ == "__main__": raise SystemExit(main())
