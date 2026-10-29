"""Prepared central-quartile study. Scientific execution is disabled pending review."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import uuid

from common import HERE, PAYLOAD, read, sha, atomic_json, module, require

EXPECTED_ENV = "/home/david/bca-operator-audit-20261026"
REPO = HERE.parents[3]
EXECUTION_ENABLED = False  # Changing protocol flags or worker tokens cannot bypass this hold.
prior = module("_quartile_supervisor_primitives", HERE.parent / "matched_policy_pilot_v1/launch.py")
save_new, read_report, supervise = prior.save_new, prior.read_report, prior.supervise
reserve = prior.reserve


def validate_protocol(p):
    fixed = {"schema": "unifloral.central_quartile_training.v1", "warmup_critic_updates": 10000,
        "warmup_actor_updates": 5000, "warmup_auxiliary_updates": 10000,
        "branch_critic_updates": 10000, "branch_actor_updates": 5000, "branch_auxiliary_updates": 10000,
        "calibration_boundaries": [10000,15000], "final_critic_step": 20000,
        "total_critic_updates": 360000, "total_actor_updates": 180000, "total_auxiliary_updates": 360000,
        "total_calibrations": 36, "total_posterior_draws": 36000, "total_replay_row_draws": 46080000,
        "training_episodes": 768, "calibration_episodes": 128,
        "gradient_probe_rows_per_cell": 768,
        "bca_kappa": 3., "actor_distance_factor": 19.29619572569095, "gradient_norm_minimum": 1e-6,
        "initial_residual_scale": 1., "alpha": .1, "credibility": .95, "posterior_draws": 1000,
        "batch_size": 256, "gamma": .98, "horizon": 100, "progress_every": 20,
        "warmup_timeout_seconds": 1800, "branch_timeout_seconds": 1800, "evaluation_timeout_seconds": 7200,
        "total_timeout_seconds": 43200, "checkpoint_every": 1000,
        "execution_authorized": False, "execution_status": "hold_pending_budget_review",
        "automatic_restart": False, "resume": False, "full_training_hold": True,
        "head_ablation_deferred": True, "failed_one_sided_variant_excluded": True,
        "conditions": ["full", "hole", "random"], "arms": ["native","bca","distance"]}
    for key, value in fixed.items():
        require(p.get(key) == value and not (type(value) is not bool and isinstance(p.get(key), bool)), "Approved protocol changed: " + key)
    require([s['index'] for s in p['seeds']] == [0,1,2], 'Three independent seed records required')
    for key in ('native_seed','aux_seed','branch_native_seed','branch_aux_seed'):
        values = [s[key] for s in p['seeds']]
        require(all(type(v) is int and v >= 0 for v in values) and len(set(values)) == 3,
                'Distinct integer seed stream required: ' + key)
    expected_cells = {f'seed{i}_{c}' for i in range(3) for c in p['conditions']}
    require(len(p['cells']) == 9 and set(p['cells']) == expected_cells, 'Complete 3x3 cell matrix required')
    return p


def require_execution_authorization(p):
    require(EXECUTION_ENABLED and p.get('execution_authorized') is True
            and p.get('execution_status') == 'approved_for_one_attempt',
            'Scientific execution is disabled pending explicit review of the new budget; no attempt reserved')


def verify_prepared(root=HERE):
    root = Path(root)
    m = read(root / "PREPARED.json")
    require(m["scientific_execution_during_preparation"] is False, "Preparation cannot contain scientific execution")
    for name, expected in m["files"].items():
        path = (root / name).resolve()
        require(Path(name).name == name and path.is_relative_to(root.resolve())
                and sha(path) == expected, "Prepared file changed or escaped: " + name)
    for name, expected in m["dependencies"].items():
        path = (root.parent / name).resolve()
        require(path.is_relative_to(root.parent.resolve()) and sha(path) == expected, "Dependency changed: " + name)
    # Repository-relative inputs can include the pinned Titan payload without
    # weakening containment of same-directory dependency records.
    for name, expected in m.get("inputs", {}).items():
        relative = Path(name)
        path = (REPO / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(REPO.resolve())
                and path.is_file() and sha(path) == expected, "Repository input changed or escaped: " + name)
    p = validate_protocol(read(root / "protocol.json"))
    if 'data_plan_sha256' in p:
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
        offset = int(seed[-1])
        arms = p["arms"][offset:] + p["arms"][:offset]
        for arm in arms:
            jobs.append({"id": cell + "/" + arm, "kind": "branch", "cell": cell,
                "seed_index": int(seed[-1]), "condition": condition, "arm": arm,
                "warmup_path": str(root / "run" / cell / "warmup")})
    jobs.append({"id": "evaluation", "kind": "evaluation", "jobs_root": str(root / "run")})
    return jobs


def scoped_process_check():
    targets = {(HERE / "launch.py").resolve(), (HERE.parent / "matched_policy_pilot_v1/launch.py").resolve(),
        (HERE.parent / "bca_one_sided_v1/launch.py").resolve(), (PAYLOAD.parent / "launch.py").resolve()}
    # Only known launch scripts in this validation workspace, not arbitrary
    # user processes or another project's command lines.
    targets.update(p.resolve() for p in HERE.parent.glob('*/launch.py'))
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
        return {"status": "not_started", "scientific_execution_started": False}
    state = read(root / "PROGRESS.json") if (root / "PROGRESS.json").is_file() else {"status": "reserved"}
    job = state.get("current_job")
    if job:
        report, error = read_report(root / "run" / job / "progress.json")
        state.update(worker_progress=report, progress_read_error=error)
    state["note"] = "Without a terminal result, liveness and final counts are uncertain. Do not relaunch."
    return state


def verify_warmup_pairs(p, root=HERE):
    """Authenticate paired fresh weights, optimizer state and common preprocessing."""
    reports, normalization, provenance = {}, {}, {}
    for cell in p['cells']:
        folder = Path(root)/'run'/cell/'warmup'
        supervisor, report = read(folder/'SUPERVISOR_RESULT.json'), read(folder/'report.json')
        require(supervisor.get('completed') and supervisor.get('report_sha256') == sha(folder/'report.json')
                and report.get('status') == 'completed_central_quartile_warmup'
                and report.get('execution_counts_known'), 'Warmup pairing receipt is not sealed: '+cell)
        seed_name, condition = cell.split('_', 1)
        require(report.get('cell') == cell and report.get('condition') == condition
                and report.get('seed_index') == int(seed_name[4:])
                and report.get('protocol_sha256') == sha(Path(root)/'protocol.json'),
                'Warmup pairing cell/protocol identity differs: '+cell)
        for name in ('actor_manifest','data_receipt'):
            require(report.get(name+'_sha256') == sha(folder/(name+'.json')), 'Warmup pairing artifact changed: '+name)
        reports[cell] = report
        actor, data = read(folder/'actor_manifest.json'), read(folder/'data_receipt.json')
        require(actor.get('cell') == cell and data.get('condition') == condition,
                'Warmup pairing artifact condition differs: '+cell)
        normalization[cell] = actor['normalization']
        provenance[cell] = data['common_normalization']
    for index in range(3):
        cells = [f"seed{index}_{condition}" for condition in p["conditions"]]
        group = [reports[cell] for cell in cells]
        require(all(x['initial_parameter_hashes'] == group[0]['initial_parameter_hashes'] for x in group),
                "Initial networks differ across paired data conditions")
        require(all(value == 0 for x in group for value in x['initial_optimizer_steps'].values()),
                "Initial optimizer counts are not fresh")
        require(all(x['initial_optimizer_hashes'] == group[0]['initial_optimizer_hashes'] for x in group),
                'Initial optimizer states differ across paired data conditions')
        require(all(normalization[cell] == normalization[cells[0]] for cell in cells),
                'Common normalization differs across paired data conditions')
        require(all(provenance[cell] == provenance[cells[0]] for cell in cells),
                'Common normalization row provenance differs across data conditions')
    for network in ('actor','critic'):
        require(len({reports[f'seed{i}_full']['initial_parameter_hashes'][network] for i in range(3)}) == 3,
                "Independent seed networks are identical")
    require(all(normalization[cell] == normalization[p['cells'][0]] for cell in p['cells'])
            and all(provenance[cell] == provenance[p['cells'][0]] for cell in p['cells']),
            'Common normalization must be identical across initialization seeds')
    return {"paired_initial_parameters_equal": True, "different_seed_networks_distinct": True,
            "initial_optimizer_steps_zero": True, "initial_optimizer_states_equal":True,
            "common_normalization_equal":True, "common_normalization_provenance_equal":True,
            "normalization_excluded_from_weight_comparison":False}


def expected_status(job):
    return "completed_central_quartile_" + job['kind']


def worker_entry(token, job_id):
    require(EXECUTION_ENABLED, 'Scientific workers are disabled pending explicit budget review')
    p, _ = verify_prepared()
    require_execution_authorization(p)
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
    worker = module("_quartile_" + job["kind"] + "_worker", HERE / ("evaluate_worker.py" if job["kind"] == "evaluation" else "train_worker.py"))
    result = worker.run(p, output, job)
    if not (output / "report.json").exists():
        save_new(output / "report.json", result)
    return 0 if result.get("status") == expected_status(job) else 1


def execute(p, manifest):
    require_execution_authorization(p)
    require(sys.platform == "linux" and Path(sys.prefix).resolve() == Path(EXPECTED_ENV).resolve(), "Use existing isolated Ubuntu GPU Python")
    require("include-system-site-packages = false" in (Path(sys.prefix) / "pyvenv.cfg").read_text(), "Environment must not inherit packages")
    require(shutil.disk_usage(HERE).free >= p['minimum_free_workspace_bytes'],
            'Insufficient workspace capacity for immutable checkpoints; nothing reserved')
    scoped_process_check()
    started, token = time.monotonic(), str(uuid.uuid4())
    reserve(HERE, token, started)
    completed_jobs, paired, active_job = [], None, None
    status, error = "running", None
    print("Reserved one central-quartile study: nine warmups, 27 branches, final evaluation; 12-hour global cap. No automatic restart.", flush=True)
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
            active_job = job_id
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
            last_progress, progress_error = read_report(output / 'progress.json')
            complete = bool(outcome["exit_code"] == 0 and not outcome["timed_out"] and not outcome["interrupted"]
                            and report and report.get("status") == expected_status(specification))
            result = {"job": job_id, **outcome, "completed": complete,
                "report_sha256": sha(output / "report.json") if report else None,
                "worker_status": report.get("status") if report else None, "report_read_error": report_error,
                "execution_counts_known": bool(report and report.get("execution_counts_known",False)
                    and not outcome['timed_out'] and not outcome['interrupted']),
                "ledger": report.get("ledger",report.get("counts")) if report else
                    last_progress.get('ledger',last_progress.get('counts')) if last_progress else None,
                "progress_read_error":progress_error,
                "ledger_source":'terminal_report' if report else 'last_progress_lower_bound' if last_progress else 'unavailable'}
            completed_jobs.append(result)
            save_new(output / "SUPERVISOR_RESULT.json", result)
            active_job = None
            print({"job_finished": job_id, "completed": complete, "worker_status": result["worker_status"]}, flush=True)
            if not complete:
                status = "stopped_after_incomplete_job"
                break
        else:
            status = "completed_bounded_central_quartile_training"
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
        "protocol_sha256": sha(HERE / "protocol.json"), "prepared_identity_unchanged": identity,
        "execution_counts_known": bool(completed_jobs) and active_job is None
            and all(job["execution_counts_known"] for job in completed_jobs),
        "active_job_at_exception":active_job,
        "all_declared_jobs_completed": len(completed_jobs)==len(job_specs(p)) and all(job["completed"] for job in completed_jobs),
        "paired_initialization": paired, "decision": evaluation.get("decision") if evaluation else None,
        "error": error, "evaluation_report_read_error": evaluation_error,
        "full_training_hold": True, "no_automatic_restart": True, "resume_supported": False}
    save_new(HERE / "RUN_RESULT.json", result)
    atomic_json(HERE / "PROGRESS.json", {"status": status, "finished_jobs": len(completed_jobs),
        "elapsed_seconds": result["elapsed_seconds"], "terminal_receipt": "RUN_RESULT.json", "no_automatic_restart": True})
    print(__import__("json").dumps(result), flush=True)
    return 0 if status == "completed_bounded_central_quartile_training" and identity else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--validate-runtime", action="store_true")
    parser.add_argument("--worker-token", help=argparse.SUPPRESS)
    parser.add_argument("--job", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.status:
        print(__import__("json").dumps(read_status()), flush=True); return 0
    if args.worker_token:
        require(not any((args.preflight,args.status,args.execute,args.validate_runtime)), 'Worker flags cannot combine with public modes')
        return worker_entry(args.worker_token,args.job)
    p, manifest = verify_prepared()
    verify_inputs()
    if args.validate_runtime:
        validator = HERE / 'validate_runtime.py'
        require(validator.is_file(), 'Bounded synthetic runtime validator is not prepared')
        return module('_quartile_runtime_validation', validator).main(p, manifest)
    if not args.execute:
        print(__import__("json").dumps({"status":"prepared_sources_verified_no_scientific_execution",
            "protocol_sha256":sha(HERE/"protocol.json"), "critic_updates_cap":p['total_critic_updates'],
            "actor_updates_cap":p['total_actor_updates'], "auxiliary_updates_cap":p['total_auxiliary_updates'],
            "calibrations_cap":p['total_calibrations'], "evaluation_episodes_cap":p['evaluation_episodes'],
            "evaluation_transitions_cap":p['simulator_transition_cap'],"total_seconds_cap":p['total_timeout_seconds'],
            "execution_authorized":False,"execution_status":p['execution_status'],"attempt_reserved":False}), flush=True)
        return 0
    def stop(signum,frame):
        raise KeyboardInterrupt("Supervisor signal " + str(signum))
    for name in ("SIGINT","SIGTERM","SIGHUP"):
        if hasattr(signal,name): signal.signal(getattr(signal,name),stop)
    return execute(p,manifest)


if __name__ == "__main__": raise SystemExit(main())
