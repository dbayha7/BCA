"""Titan recovery of the interrupted Path A production attempt (additive).

Nothing scientific executes on import. The original seal, the smoke amendment,
Zeus's passed smoke and the interrupted attempt run_amendment_1 are verified
and never modified. Zeus's completed honeypot_s0_warmup is reused byte for
byte; the interrupted honeypot_s0_native is discarded, charged at its upper
bound and re-run from that warmup. Every remaining job, seed, dataset, mask,
objective and evaluation is the unchanged sealed Path A code.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import launch as L
import smoke_launch as S

AMENDMENT_MANIFEST_SHA256 = "ea550df3c1805df38e03e101e007bfbb89df770ffc7a5287f138c1f12bd91549"
REUSED = "honeypot_s0_warmup"
INTERRUPTED = "honeypot_s0_native"
KEYS = ("critic", "actor", "auxiliary", "calibrations")
JOB_CAPS = {"critic": 10000, "actor": 5000, "auxiliary": 10000, "calibrations": 1}
TITAN_ENV = {"OPENBLAS_CORETYPE": "Haswell"}
SMOKE_SECONDS = 900
PRODUCTION_SECONDS = 28800
EVALUATION_RESERVE_SECONDS = 3600
JOB_MARGIN = 1.25
PRIOR_SIMULATOR_TRANSITIONS = 189655
EVALUATION_TRANSITION_CAP = 237600
SIMULATOR_CEILING = 498400
SAVED_CONTROLLER_ROWS = {"total": 613, "safe": 598}
# Bank files that depend only on the fixed controller, plant and seeds, never on a learner.
BANK_FILE = re.compile(r"^(smoke_evaluation_plan|smoke_own_policy_anchor|smoke_(honeypot|bottleneck)_seed[0-2]_.+)\.json$")
BANK_FILES = 2 + 6 * (1 + 16 + 48 + 1)
SOURCES = ("recovery_launch.py", "test_recovery_launch.py", "RECOVERY.md", "RECOVERY_VERIFICATION.json")


def remaining_jobs():
    """The unchanged sealed job order without the reused warmup."""
    return [job for job in L.jobs() if job["id"] != REUSED]


def production_caps():
    training = sum(job["arm"] != "evaluation" for job in remaining_jobs())
    return {key: training * JOB_CAPS[key] for key in KEYS}


def allocation():
    return {"schema": "two_scenario.path_a.titan_recovery.v1", "machine": "Titan",
        "original_manifest_sha256": S.ORIGINAL_MANIFEST, "amendment_manifest_sha256": AMENDMENT_MANIFEST_SHA256,
        "required_environment": dict(TITAN_ENV), "reused_completed_job": REUSED, "rerun_interrupted_job": INTERRUPTED,
        "remaining_jobs": [job["id"] for job in remaining_jobs()],
        "titan_smoke": {"optimizer_caps": dict(S.LEARNING_CAPS), "transition_caps": dict(S.PHYSICAL_CAPS),
            "transition_cap": sum(S.PHYSICAL_CAPS.values()), "process_seconds": SMOKE_SECONDS,
            "warmup_restore_model_initializations": 1, "warmup_restore_optimizer_updates": 0,
            "zeus_equivalence": {"controller_actions": SAVED_CONTROLLER_ROWS["safe"], "controller_simulator_transitions": 0,
                                 "bank_files": BANK_FILES, "bank_extra_simulator_transitions": 0}},
        "production": {"training_jobs": len(remaining_jobs()) - 1, "optimizer_caps": production_caps(),
            "evaluation_transition_cap": EVALUATION_TRANSITION_CAP, "return_episodes": 1536,
            "process_seconds": PRODUCTION_SECONDS, "evaluation_reserve_seconds": EVALUATION_RESERVE_SECONDS,
            "job_boundary_margin": JOB_MARGIN},
        "scientific_protocol_changed": False, "no_automatic_retry": True, "path_b_authorized": False}


def approval_text():
    a, p = allocation(), allocation()["production"]
    return ("I approve the Titan recovery of Path A attempt run_amendment_1: reuse Zeus's completed, authenticated "
        f"{REUSED} checkpoint byte for byte; discard and re-run the interrupted {INTERRUPTED} from it, charging the "
        "interrupted work at its upper bound. On Titan with OPENBLAS_CORETYPE=Haswell, first one disposable runtime smoke "
        f"(at most {S.LEARNING_CAPS['critic']} critic, {S.LEARNING_CAPS['actor']} actor, {S.LEARNING_CAPS['auxiliary']} "
        f"auxiliary updates, {S.LEARNING_CAPS['calibrations']} calibrations, {a['titan_smoke']['transition_cap']} simulator "
        f"transitions and {SMOKE_SECONDS} seconds) that must reproduce Zeus's controller actions and probe banks exactly and "
        f"restore the Zeus warmup exactly. Only after it passes, the remaining {p['training_jobs']} unchanged training jobs "
        f"and the frozen evaluation: at most {p['optimizer_caps']['critic']} critic, {p['optimizer_caps']['actor']} actor, "
        f"{p['optimizer_caps']['auxiliary']} auxiliary updates, {p['optimizer_caps']['calibrations']} calibrations, "
        f"{EVALUATION_TRANSITION_CAP} evaluation transitions and {PRODUCTION_SECONDS} seconds, stopping at a job boundary "
        "when the next job cannot finish. Protocol, data, seeds and evaluation are unchanged; the 498400-transition "
        "ceiling remains; no Path B execution and no automatic retry.")


def approval_expected():
    return {"approval": approval_text(), "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"),
            "machine": "Titan", "allow_titan_smoke": True, "allow_production_only_after_passed_titan_smoke": True}


def tree_hashes(directory):
    directory = Path(directory)
    L.require(directory.is_dir(), "Missing artifact directory: " + str(directory))
    result = {}
    for path in sorted(directory.rglob("*")):
        L.require(not path.is_symlink(), "Linked artifact refused: " + str(path))
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = L.sha(path)
    return result


def _progress_records(path):
    if not path.exists():
        return [], False
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    records, truncated = [], False
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Only a final record can be cut by a hard stop; anything earlier is corruption.
            L.require(index == len(lines) - 1, "Corrupted progress record before the final line: " + str(path))
            truncated = True
    return records, truncated


def interrupted_bounds(job_dir):
    """Lower bounds from persisted counts; the planned job caps remain the upper bound unless exactly known."""
    job_dir = Path(job_dir)
    records, truncated = _progress_records(job_dir / "progress.jsonl")
    lower, last = dict.fromkeys(KEYS, 0), None
    for record in records:
        completed = record.get("physical_counts", {}).get("completed", {})
        for key in KEYS:
            lower[key] = max(lower[key], int(completed.get(key, 0)))
        last = record.get("observed_est", last)
    upper, source = dict(JOB_CAPS), "progress_lower_bound_with_planned_job_cap"
    if (job_dir / "report.json").exists():
        report = L.read(job_dir / "report.json")
        L.require(report.get("status") != "completed_training_job", "Interrupted job reports completion; review required")
        counts = report.get("counts", {})
        for key in KEYS:
            lower[key] = max(lower[key], int(counts.get("completed", {}).get(key, 0)))
        if report.get("execution_counts_known") is True and counts.get("attempted") == counts.get("completed"):
            upper = {key: int(counts["completed"][key]) for key in KEYS}
            lower, source = dict(upper), "failed_job_report_exact"
    L.require(all(0 <= lower[k] <= upper[k] <= JOB_CAPS[k] for k in KEYS), "Inconsistent interrupted-job bounds")
    return {"lower": lower, "upper": upper, "source": source, "progress_records": len(records),
            "truncated_final_progress_line": truncated, "last_observed_est": last}


def prior_lineage(smoke_dir, run_dir, *, smoke_check, amendment_manifest_sha256):
    """Authenticate Zeus's smoke and interrupted attempt without changing a byte."""
    smoke_dir, run_dir = Path(smoke_dir), Path(run_dir)
    smoke = smoke_check(smoke_dir)
    proof = {name: L.sha(smoke_dir / name) for name in ("RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json")}
    attempt = L.read(run_dir / "ATTEMPT.json")
    L.require(attempt.get("mode") == "production" and attempt.get("smoke_proof") == proof and
              attempt.get("amendment_manifest_sha256") == amendment_manifest_sha256,
              "Interrupted attempt is not bound to this smoke and amendment")
    entries = sorted(p.name for p in run_dir.iterdir() if p.is_dir())
    L.require(entries == sorted((REUSED, INTERRUPTED)),
              "Prior attempt differs from one completed warmup and one interrupted native job: " + ", ".join(entries))

    def bound_job(directory, arm):
        job = L.read(directory / "JOB.json")
        expected = {"id": "honeypot_s0_" + arm, "condition": "honeypot", "seed_index": 0, "arm": arm,
                    "amendment_manifest_sha256": amendment_manifest_sha256, "smoke_proof": proof}
        L.require(all(job.get(k) == v for k, v in expected.items()), "Prior job reservation differs: " + directory.name)
        return job

    warm = run_dir / REUSED
    bound_job(warm, "warmup")
    report = L.accepted_job(warm, 0)
    supervisor = L.read(warm / "SUPERVISOR.json")
    L.require(supervisor.get("status") == "accepted" and supervisor.get("exit_code") == 0, "Warmup was not accepted by its supervisor")
    receipt = L.read(warm / "checkpoint.json")
    raw_sha = L.sha(warm / "full_state.msgpack")
    L.require(report.get("condition") == "honeypot" and report.get("seed_index") == 0 and report.get("arm") == "warmup" and
              report.get("checkpoint_sha256") == raw_sha == receipt.get("checkpoint_sha256") and
              report.get("checkpoint_receipt_sha256") == L.sha(warm / "checkpoint.json") and
              receipt.get("arm") == "warmup" and receipt.get("condition") == "honeypot" and receipt.get("seed_index") == 0 and
              receipt.get("counts", {}).get("critic") == 10000, "Warmup checkpoint chain differs")

    native = run_dir / INTERRUPTED
    bound_job(native, "native")
    L.require(not (native / "SUPERVISOR.json").exists() and not (native / "full_state.msgpack").exists() and
              not (native / "checkpoint.json").exists(), "Interrupted job has acceptance or a final checkpoint; review required")
    if (native / "WORKER_EXIT.json").exists():
        L.require(L.read(native / "WORKER_EXIT.json").get("status") != "completed", "Interrupted worker reports completion")
    terminal = L.read(run_dir / "RESULT.json") if (run_dir / "RESULT.json").exists() else None
    L.require(terminal is None or terminal.get("status") != "completed_path_a_study", "Prior attempt reports a completed study")
    smoke_terminal = L.read(smoke_dir / "RESULT.json")
    return {"zeus_smoke_proof": proof, "zeus_smoke_elapsed_seconds": smoke_terminal.get("elapsed_seconds"),
        "zeus_smoke_counts": {"learning": smoke["learning"]["counts"]["completed"],
                              "physical": smoke["physical"]["counts"]["completed"]},
        "zeus_runtime": smoke.get("runtime"),
        "attempt": {"started_est": attempt.get("started_est"), "terminal_receipt_present": terminal is not None,
                    "terminal_status": None if terminal is None else terminal.get("status"),
                    "terminal_error": None if terminal is None else terminal.get("error"),
                    "elapsed_seconds": None if terminal is None else terminal.get("elapsed_seconds")},
        "reused_warmup": {"job": REUSED, "report_sha256": L.sha(warm / "report.json"), "checkpoint_sha256": raw_sha,
                          "checkpoint_receipt_sha256": L.sha(warm / "checkpoint.json"),
                          "state_digest": receipt.get("state_digest"), "elapsed_seconds": report.get("elapsed_seconds"),
                          "counts": {k: report["counts"]["completed"][k] for k in KEYS}, "files": tree_hashes(warm)},
        "interrupted": {"job": INTERRUPTED, **interrupted_bounds(native), "files": tree_hashes(native)}}


def accounting(lineage):
    a = allocation()
    zeus, warm = lineage["zeus_smoke_counts"]["learning"], lineage["reused_warmup"]["counts"]
    low, high = lineage["interrupted"]["lower"], lineage["interrupted"]["upper"]
    new = {k: a["titan_smoke"]["optimizer_caps"][k] + a["production"]["optimizer_caps"][k] for k in KEYS}
    prior_low = {k: zeus[k] + warm[k] + low[k] for k in KEYS}
    prior_high = {k: zeus[k] + warm[k] + high[k] for k in KEYS}
    zeus_transitions = sum(lineage["zeus_smoke_counts"]["physical"].values())
    maximum = PRIOR_SIMULATOR_TRANSITIONS + zeus_transitions + a["titan_smoke"]["transition_cap"] + EVALUATION_TRANSITION_CAP
    L.require(maximum <= SIMULATOR_CEILING, "Recovery would exceed the cumulative simulator ceiling")
    return {"prior_optimizer_lower": prior_low, "prior_optimizer_upper": prior_high, "new_optimizer_caps": new,
        "cumulative_optimizer_upper": {k: prior_high[k] + new[k] for k in KEYS},
        "discarded_interrupted_work_upper": dict(high),
        "prior_simulator_transitions": PRIOR_SIMULATOR_TRANSITIONS, "zeus_smoke_transitions": zeus_transitions,
        "titan_smoke_transition_cap": a["titan_smoke"]["transition_cap"], "evaluation_transition_cap": EVALUATION_TRANSITION_CAP,
        "cumulative_simulator_maximum": maximum, "cumulative_simulator_ceiling": SIMULATOR_CEILING,
        "time": {"original_four_hour_allowance_exhausted": True, "zeus_smoke_elapsed_seconds": lineage["zeus_smoke_elapsed_seconds"],
                 "zeus_production_elapsed_seconds": lineage["attempt"]["elapsed_seconds"],
                 "new_titan_seconds": SMOKE_SECONDS + PRODUCTION_SECONDS}}


def controller_equivalence(control, rows):
    """Recompute Zeus's saved fixed-controller actions; no plant transition is executed."""
    import numpy as np
    safe = [row for row in rows if row.get("variant") == "safe"]
    L.require(len(rows) == SAVED_CONTROLLER_ROWS["total"] and len(safe) == SAVED_CONTROLLER_ROWS["safe"],
              "Saved Zeus controller trace count differs")
    mismatched = []
    for index, row in enumerate(safe):
        state = np.asarray(row["state"], np.float64)
        L.require(state.shape == (13,), "Saved controller state is not the complete 13-state vector")
        action = np.asarray(control(state.copy(), int(row["time"])))
        L.require(action.dtype == np.float32 and action.shape == (4,), "Controller must return one float32 action")
        if action.tobytes() != np.asarray(row["action"], np.float32).tobytes():
            mismatched.append(index)
    return {"compared_controller_actions": len(safe), "mismatched": len(mismatched),
            "first_mismatched_rows": mismatched[:10], "identical": not mismatched, "simulator_transitions": 0}


def bank_equivalence(zeus_physical, titan_physical):
    """Byte comparison of every controller/plant-only smoke file; learner-dependent files are excluded."""
    def names(directory):
        return sorted(p.name for p in Path(directory).iterdir() if p.is_file() and BANK_FILE.match(p.name))
    zeus, titan = names(zeus_physical), names(titan_physical)
    L.require(len(zeus) == BANK_FILES and zeus == titan, "Titan and Zeus bank file sets differ")
    differing = [n for n in zeus if (Path(zeus_physical) / n).read_bytes() != (Path(titan_physical) / n).read_bytes()]
    return {"files_compared": len(zeus), "differing": differing, "identical": not differing,
            "excluded": "learner-dependent forced-action, free-return and summary files"}


def install_reused_warmup(source, target, expected):
    """Byte-identical copy so the unchanged sealed evaluator finds every job as a sibling."""
    source, target = Path(source), Path(target)
    L.require(tree_hashes(source) == expected, "Zeus warmup bytes changed before reuse")
    target.mkdir(exist_ok=False)
    for relative in sorted(expected):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with (source / relative).open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
    L.require(tree_hashes(target) == expected, "Reused warmup copy differs from the authenticated original")
    return {"source": source.name, "target": target.name, "files": expected, "byte_identical": True}


def next_job_fits(arm, remaining, durations):
    """Stop between jobs instead of losing a job to the deadline; the evaluation reserve stays protected."""
    if arm == "evaluation":
        return remaining >= EVALUATION_RESERVE_SECONDS
    estimate = JOB_MARGIN * max(durations) if durations else 0.
    return remaining >= estimate + EVALUATION_RESERVE_SECONDS


def require_wsl():
    L.require(os.name == "posix", "Manual existing WSL runtime only")


def require_titan_environment():
    L.require(all(os.environ.get(k) == v for k, v in TITAN_ENV.items()),
              "Titan recovery requires OPENBLAS_CORETYPE=Haswell in the launching environment")


def _lineage_now():
    return prior_lineage(HERE / "smoke_attempt_1", HERE / "run_amendment_1", smoke_check=S.smoke_passed,
                         amendment_manifest_sha256=AMENDMENT_MANIFEST_SHA256)


def prepare():
    L.require(not any((HERE / n).exists() for n in ("RECOVERY_MANIFEST.json", "titan_smoke_1", "run_recovery_1")),
              "Never reseal or reset the recovery")
    L.require(L.sha(HERE / "AMENDMENT_MANIFEST.json") == AMENDMENT_MANIFEST_SHA256, "Amendment seal differs")
    S.check(approval=True)
    lineage = _lineage_now()
    files = {(HERE / n).relative_to(L.REPO).as_posix(): L.sha(HERE / n) for n in SOURCES}
    files[(HERE / "AMENDMENT_APPROVAL.json").relative_to(L.REPO).as_posix()] = L.sha(HERE / "AMENDMENT_APPROVAL.json")
    for folder in ("smoke_attempt_1", "run_amendment_1"):
        files.update({(HERE / folder / r).relative_to(L.REPO).as_posix(): h for r, h in tree_hashes(HERE / folder).items()})
    L.save(HERE / "RECOVERY_MANIFEST.json", {"allocation": allocation(), "prior": lineage, "accounting": accounting(lineage),
           "files": files, "approval_created": False, "runtime_executed": False})
    return {"status": "recovery_sealed_not_approved", "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json")}


def check(approval=False):
    protocol = S.check(approval=True)
    manifest = L.read(HERE / "RECOVERY_MANIFEST.json")
    L.require(manifest["allocation"] == allocation(), "Recovery budget changed")
    L.verify_pins(manifest["files"])
    if approval:
        L.require(L.read(HERE / "RECOVERY_APPROVAL.json") == approval_expected(), "Explicit Titan recovery approval missing or mismatched")
    return protocol


def titan_smoke_passed(directory=None, *, require_supervisor=True):
    """The sealed smoke's own acceptance rule, plus every Titan-specific equivalence check."""
    directory = HERE / "titan_smoke_1" if directory is None else Path(directory)
    report = S.smoke_passed(directory, require_supervisor=require_supervisor)
    checks = report.get("titan_checks", {})
    L.require(report.get("recovery_manifest_sha256") == L.sha(HERE / "RECOVERY_MANIFEST.json") and checks.get("passed") is True and
              checks.get("environment") == TITAN_ENV and checks["controller"]["identical"] is True and
              checks["banks"]["identical"] is True and checks["banks"]["files_compared"] == BANK_FILES and
              checks["warmup_restore"]["exact"] is True, "Titan-specific smoke checks incomplete")
    return report


def titan_smoke_identity():
    titan_smoke_passed()
    return {name: L.sha(HERE / "titan_smoke_1" / name) for name in ("RESULT.json", "WORKER_EXIT.json", "report.json", "ARTIFACTS.json")}


def _canonical(protocol):
    sys.path.insert(0, str(L.REPO))
    binding = L.load("_path_a_binding", HERE / "binding.py")
    core = L.PAYLOAD / "transfer_bundle/core_v4"
    sys.path.insert(0, str(core))
    from thrusterchain import environment as canonical
    L.require(Path(canonical.__file__).resolve() == core / "thrusterchain/environment.py", "Wrong canonical plant")
    physics = canonical.Physics(stall=True, stall_slope=3.)
    original = L.read(L.V2 / "setup_attempt_1/RESULT.json")
    L.require(physics.as_dict() == original["physics"] and canonical.provenance() == original["runtime"]["canonical"],
              "Changed physical runtime")
    A, B = canonical.matrices(physics)
    return binding.D.T.tracking_controller(A, B, protocol["geometry_recipe"])


def warmup_restore_check(protocol, warmup_dir):
    """Zero updates: Zeus's warmup must restore and fork exactly in Titan's runtime."""
    _, train, services, review = S._services(protocol)
    receipt = L.read(Path(warmup_dir) / "checkpoint.json")
    prepared, adapter, state = train._prepare(protocol, "honeypot", 0, services, review)
    adapter, state, lineage = train._branch_start(services, prepared, adapter, state, protocol, "honeypot", 0, "native", Path(warmup_dir))
    exact = lineage["warmup_state_digest"] == receipt["state_digest"] and lineage["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    L.require(exact, "Zeus warmup does not restore exactly in Titan's runtime")
    return {**lineage, "zeus_state_digest": receipt["state_digest"], "exact": True,
            "model_initializations": 1, "optimizer_updates": 0, "simulator_transitions": 0}


def run_titan_smoke(protocol, output, deadline):
    control = _canonical(protocol)
    rows = [json.loads(line) for line in (L.REPAIR / "danger_attempt_1/transitions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    controller = controller_equivalence(control, rows)
    L.save(output / "titan_controller_equivalence.json", controller)
    L.require(controller["identical"], "Titan controller differs from Zeus's saved actions; no learner was initialized")
    deadline()
    report = S.run_smoke(protocol, output, deadline)
    banks = bank_equivalence(HERE / "smoke_attempt_1/physical", output / "physical")
    L.save(output / "titan_bank_equivalence.json", banks)
    L.require(banks["identical"], "Titan probe banks differ from Zeus's: " + ", ".join(banks["differing"][:5]))
    deadline()
    restore = warmup_restore_check(protocol, HERE / "run_amendment_1" / REUSED)
    L.save(output / "titan_warmup_restore.json", restore)
    return {**report, "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"),
            "titan_checks": {"passed": True, "environment": {k: os.environ[k] for k in TITAN_ENV},
                             "hostname": socket.gethostname(), "controller": controller, "banks": banks, "warmup_restore": restore}}


def worker_main(job_id):
    require_wsl()
    require_titan_environment()
    started = time.monotonic()
    protocol = check(approval=True)
    is_smoke = job_id == "smoke"
    output = HERE / "titan_smoke_1" if is_smoke else HERE / "run_recovery_1" / job_id
    job = L.read(output / "JOB.json")
    L.require(job["id"] == job_id and job["recovery_manifest_sha256"] == L.sha(HERE / "RECOVERY_MANIFEST.json"), "Wrong recovery job")
    if not is_smoke:
        proof = titan_smoke_identity()
        L.require(job["titan_smoke_proof"] == proof == L.read(HERE / "run_recovery_1/ATTEMPT.json")["titan_smoke_proof"],
                  "Recovery job's authorizing Titan smoke changed")
        expected = next((j for j in remaining_jobs() if j["id"] == job_id), None)
        L.require(expected is not None and all(job[k] == v for k, v in expected.items()), "Wrong recovery job")
    end = started + min(job["remaining_seconds"], SMOKE_SECONDS - 15 if is_smoke else PRODUCTION_SECONDS - 15)

    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Recovery shared deadline reached")
    result = {"status": "failed", "identity_unchanged": False, "error": None, "no_automatic_retry": True}
    try:
        if is_smoke:
            report = run_titan_smoke(protocol, output, deadline)
            L.save(output / "report.json", report)
            paths = [output / "report.json"] + sorted(output.glob("titan_*.json"))
            for folder in ("learning", "physical"):
                paths.extend(p for p in (output / folder).rglob("*") if p.is_file())
            L.save(output / "ARTIFACTS.json", {p.relative_to(output).as_posix(): L.sha(p) for p in paths})
            result["artifacts_sha256"] = L.sha(output / "ARTIFACTS.json")
        else:
            binding, train, services, review = S._services(protocol)
            review["recovery_approval_sha256"] = L.sha(HERE / "RECOVERY_APPROVAL.json")
            smoke_runtime = titan_smoke_passed()["runtime"]
            runtime = services.runtime.runtime_receipt(services.native)
            L.require(runtime["device_kinds"] == smoke_runtime["device_kinds"] and runtime["packages"] == smoke_runtime["packages"],
                      "Production runtime differs from the passed Titan smoke")
            if job["arm"] == "evaluation":
                W = L.load("_path_a_worker", HERE / "worker.py")
                report = W.evaluate_job(protocol, output, deadline, binding, train, services, review)
            else:
                warmup = None if job["arm"] == "warmup" else output.parent / f"{job['condition']}_s{job['seed_index']}_warmup"
                if warmup is not None:
                    L.accepted_job(warmup, 0)
                report = train.run_job(protocol, job["condition"], job["seed_index"], job["arm"], output,
                                       warmup_dir=warmup, deadline=deadline, services=services, training_review=review)
                L.require(report["status"] == "completed_training_job", "Recovery training job incomplete")
        check(approval=True)
        result.update(status="completed", identity_unchanged=True, report_sha256=L.sha(output / "report.json"))
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), traceback=traceback.format_exc())
    result.update(elapsed_seconds=time.monotonic() - started, finished_est=datetime.now(L.EST).isoformat())
    L.save(output / "WORKER_EXIT.json", result)
    print(json.dumps(result), flush=True)
    return result


def run_child(spec, job_dir, deadline, log):
    """Own the child through receipt persistence and every failure path."""
    child = None
    try:
        child = subprocess.Popen([sys.executable, "-B", "-u", str(HERE / "recovery_launch.py"), "--worker", spec["id"]],
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        L.save(job_dir / "PROCESS.json", {"pid": child.pid, "started_est": datetime.now(L.EST).isoformat()})
        return child.wait(timeout=max(.001, deadline - time.monotonic()))
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


class StopAtBoundary(Exception):
    pass


def supervise(mode):
    require_wsl()
    require_titan_environment()
    started = time.monotonic()
    check(approval=True)
    is_smoke = mode == "smoke"
    # This gate runs before any recovery reservation, copy, child or learner.
    proof = None if is_smoke else titan_smoke_identity()
    out = HERE / ("titan_smoke_1" if is_smoke else "run_recovery_1")
    out.mkdir(exist_ok=False)
    cap = SMOKE_SECONDS if is_smoke else PRODUCTION_SECONDS
    deadline = started + cap - 15
    L.save(out / "ATTEMPT.json", {"mode": mode, "machine": "Titan", "hostname": socket.gethostname(),
           "started_est": datetime.now(L.EST).isoformat(), "supervisor_pid": os.getpid(), "cap_seconds": cap,
           "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "titan_smoke_proof": proof,
           "environment": {k: os.environ[k] for k in TITAN_ENV}, "no_automatic_retry": True})
    result = {"status": "incomplete", "mode": mode, "jobs": [], "error": None, "no_automatic_retry": True, "titan_smoke_proof": proof}
    accepted, durations = [], []

    def stop(signum, frame):
        raise InterruptedError("Supervisor signal " + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if not is_smoke:
            expected = L.read(HERE / "RECOVERY_MANIFEST.json")["prior"]["reused_warmup"]["files"]
            L.save(out / "REUSED_WARMUP.json", install_reused_warmup(HERE / "run_amendment_1" / REUSED, out / REUSED, expected))
        for spec in ([{"id": "smoke", "arm": "smoke"}] if is_smoke else remaining_jobs()):
            check(approval=True)
            if not is_smoke:
                L.require(titan_smoke_identity() == proof, "Authorizing Titan smoke changed")
            remaining = deadline - time.monotonic()
            L.require(remaining > 0, "Recovery process deadline exhausted")
            if not is_smoke and not next_job_fits(spec["arm"], remaining, durations):
                raise StopAtBoundary(f"{spec['id']} cannot finish in {remaining:.0f} remaining seconds")
            job_dir = out if is_smoke else out / spec["id"]
            if not is_smoke:
                job_dir.mkdir(exist_ok=False)
            L.save(job_dir / "JOB.json", {**spec, "remaining_seconds": remaining,
                   "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json"), "titan_smoke_proof": proof})
            begun = time.monotonic()
            with (job_dir / "worker.log").open("x", encoding="utf-8") as log:
                rc = run_child(spec, job_dir, deadline, log)
            result["jobs"].append({"id": spec["id"], "exit_code": rc, "seconds": time.monotonic() - begun})
            L.require(rc == 0, "Recovery worker failed: " + spec["id"])
            report = titan_smoke_passed(out, require_supervisor=False) if is_smoke else L.accepted_job(job_dir, rc)
            if not is_smoke and spec["arm"] != "evaluation":
                durations.append(time.monotonic() - begun)
            accepted.append(report)
            L.save(job_dir / "SUPERVISOR.json", {"status": "accepted", "exit_code": rc, "finished_est": datetime.now(L.EST).isoformat()})
        check(approval=True)
        if is_smoke:
            result.update(status="passed_runtime_smoke", execution_counts_known=True,
                          smoke_optimizer_counts=accepted[0]["learning"]["counts"], smoke_physical_counts=accepted[0]["physical"]["counts"])
        else:
            L.require(titan_smoke_identity() == proof, "Authorizing Titan smoke changed before final acceptance")
            result.update(status="completed_path_a_study", execution_counts_known=True,
                recovery_training_counts={k: sum(r["counts"]["completed"][k] for r in accepted[:-1]) for k in KEYS},
                evaluation_transitions=sum(accepted[-1]["physical_counts"]["completed"].values()),
                evaluation_report=str((out / "evaluation/report.json").relative_to(L.REPO)))
    except StopAtBoundary as exc:
        result.update(status="stopped_at_job_boundary", error=str(exc), execution_counts_known=True,
            accepted_jobs=[row["id"] for row in result["jobs"]],
            recovery_training_counts={k: sum(r["counts"]["completed"][k] for r in accepted) for k in KEYS},
            note="Every accepted job keeps its authenticated checkpoint; continuing needs a new explicit approval.")
    except BaseException as exc:
        result.update(error=type(exc).__name__ + ": " + str(exc), execution_counts_known=False,
                      accounting_note="Accepted jobs are exact; the unfinished job keeps its planned upper bound, never zero; no retry.")
    result.update(elapsed_seconds=time.monotonic() - started, finished_est=datetime.now(L.EST).isoformat())
    L.save(out / "RESULT.json", result)
    print(json.dumps(result), flush=True)
    return result


def approve():
    """Only the user runs this: it records their explicit approval of the sealed recovery."""
    check(approval=False)
    print(approval_text(), flush=True)
    L.save(HERE / "RECOVERY_APPROVAL.json", approval_expected())
    return {"status": "approval_recorded", "recovery_manifest_sha256": L.sha(HERE / "RECOVERY_MANIFEST.json")}


def status():
    state = {name: (HERE / name).exists() for name in ("RECOVERY_MANIFEST.json", "RECOVERY_APPROVAL.json", "titan_smoke_1", "run_recovery_1")}
    if not state["RECOVERY_MANIFEST.json"]:
        S.check(approval=True)
        lineage = _lineage_now()
        return {"status": "ready_to_prepare", **state, "interrupted_bounds": {k: lineage["interrupted"][k] for k in ("lower", "upper", "source")},
                "accounting": accounting(lineage)}
    check(approval=False)
    return {"status": "source_checked_not_executed", **state}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true", help="verify copied Zeus artifacts or the sealed recovery; no execution")
    modes.add_argument("--prepare", action="store_true", help="seal RECOVERY_MANIFEST.json once")
    modes.add_argument("--approve", action="store_true", help="user only: record explicit approval of the sealed recovery")
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--worker")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(status(), indent=2, default=str))
    elif args.prepare:
        print(json.dumps(prepare()))
    elif args.approve:
        print(json.dumps(approve()))
    elif args.worker:
        raise SystemExit(0 if worker_main(args.worker)["status"] == "completed" else 1)
    else:
        result = supervise("smoke" if args.smoke else "production")
        raise SystemExit(0 if result["status"] in ("passed_runtime_smoke", "completed_path_a_study") else 1)
