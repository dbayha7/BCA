"""Loader-only recovery adapter. Reuse the original runner; never reset its reservation."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "minimal_comparison"
REPO = HERE.parents[4]
PRODUCER = HERE.parent / "formal_g2"
DIAGNOSTIC = HERE.parent / "reset_lineage_diagnostic_v1/receipt_20261101T092712_649674_EST_424.json"
ORIGINAL_MANIFEST_SHA = "400490f6f2c3a5ab702e80733fcf28a34a1308f7d06df6df476bc8235e593c1d"
DIAGNOSTIC_SHA = "127554723540a58ab51ef587cff437c698662efb33f29bdcae02fca55c7271f4"
KNOWN_PRIOR_SECONDS = 26.408195991
CHARGED_PRIOR_SECONDS = 100
RECOVERY_PROCESS_CAP = 10700
DISPATCH_DEADLINE_SECONDS = 10690
APPROVAL_TEXT = "I approve the loader-only minimal comparison recovery, with unchanged native versus BCA kappa 3 science and a 10700-second process cap within the original three-hour allowance."
EST = timezone(timedelta(hours=-5))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def module(name, path):
    require(sha(ORIGINAL / "MANIFEST.json") == ORIGINAL_MANIFEST_SHA, "Original source manifest changed")
    expected = read(ORIGINAL / "MANIFEST.json")["files"][Path(path).relative_to(REPO).as_posix()]
    require(sha(path) == expected, "Original module changed before import")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def budget_record():
    return {"original_execution_allowance_seconds": 10800,
            "known_prior_seconds": KNOWN_PRIOR_SECONDS,
            "unmeasured_wrapper_allowance_seconds": CHARGED_PRIOR_SECONDS - KNOWN_PRIOR_SECONDS,
            "charged_prior_and_wrapper_allowance_seconds": CHARGED_PRIOR_SECONDS,
            "recovery_process_cap_seconds": RECOVERY_PROCESS_CAP,
            "wrapper_allowance_is_not_a_measurement": True,
            "actual_unmeasured_overhead_remains_unknown": True,
            "expected_new_simulator_steps": 170400,
            "historical_dataset_validation_rows_reused_not_executed": 1536000,
            "original_simulator_cap_preserved": 1706400,
            "critic_updates": 90000, "actor_updates": 45000,
            "auxiliary_updates": 90000, "calibrations": 9}


def prior_assessment():
    require(sha(DIAGNOSTIC) == DIAGNOSTIC_SHA, "Reset diagnostic evidence changed")
    prior = read(ORIGINAL / "run/RESULT.json")
    failed = read(ORIGINAL / "run/warmup_s0/report.json")
    supervisor = read(ORIGINAL / "run/warmup_s0/SUPERVISOR_RESULT.json")
    diagnostic = read(DIAGNOSTIC)
    for name, digest in diagnostic["prior_attempt"]["files"].items():
        require(sha(ORIGINAL / name) == digest, "Prior attempt changed since reset diagnostic: " + name)
    require(failed["job_sha256"] == sha(ORIGINAL / "run/warmup_s0/job.json"), "Failed worker job binding differs")
    require(prior["status"] == "incomplete" and set(prior["jobs"]) == {"preflight", "warmup_s0"},
            "Original attempt is not the reviewed terminal failure")
    require(failed["status"] == "failed" and failed["error"] ==
            "KneeDatasetError: Episode 16: initial state lineage differs", "Original failure differs")
    require(supervisor["exit_code"] == 1 and not supervisor.get("timed_out")
            and not supervisor.get("interrupted"), "Original worker termination differs")
    require("dataset = D.load(path)" in failed["traceback"]
            and 'dataset, plan = load_world(job, job["world"])' in failed["traceback"],
            "Missing evidence for the pre-initialization failure")
    require(failed["simulator_steps_exact"] == 0 and diagnostic["counts"]["simulator_steps"] == 0,
            "Prior simulator costs changed")
    total = prior["elapsed_seconds"] + diagnostic["elapsed_seconds"]
    require(abs(total - KNOWN_PRIOR_SECONDS) < 1e-9, "Prior elapsed accounting changed")
    return {"original_raw_accounting_unchanged": prior["accounting"],
            "source_inferred_prior_training_updates": 0, "source_inferred_prior_calibrations": 0,
            "basis": "Authenticated traceback failed in load_world before bind/state initialization/train/calibration; sealed source reviewed. This is a source-path inference, not replacement terminal counters.",
            "prior_simulator_steps": 0, "diagnostic_resets": 1,
            "known_prior_seconds": total}


def seal():
    require(not (HERE / "MANIFEST.json").exists(), "Recovery already sealed; do not reseal")
    require(sha(ORIGINAL / "MANIFEST.json") == ORIGINAL_MANIFEST_SHA, "Original seal differs")
    original = read(ORIGINAL / "MANIFEST.json")
    pins = dict(original["files"])
    producer = read(PRODUCER / "G2_FORMAL_MANIFEST.json")
    for section in ("files", "upstream"):
        for relative, digest in producer[section].items():
            path = (PRODUCER / relative).resolve()
            require(path.is_relative_to(REPO), "Producer path escaped repository")
            key = path.relative_to(REPO).as_posix()
            require(key not in pins or pins[key] == digest, "Producer and consumer source pins conflict: " + key)
            pins[key] = digest
    extra = [ORIGINAL / "MANIFEST.json", ORIGINAL / "LAUNCH_APPROVAL.json", DIAGNOSTIC,
             PRODUCER / "G2_FORMAL_MANIFEST.json", PRODUCER / "G2_FORMAL_APPROVAL.json",
             PRODUCER / "formal_g2_run/attempt_1/report.json",
             PRODUCER / "formal_g2_run/attempt_1/collection.json"]
    extra += [p for p in (ORIGINAL / "run").rglob("*") if p.is_file()]
    extra += [p for p in HERE.iterdir() if p.is_file() and p.suffix in {".py", ".md"}]
    for path in extra:
        pins[path.relative_to(REPO).as_posix()] = sha(path)
    for relative, digest in pins.items():
        require(sha(REPO / relative) == digest, "Source or evidence changed: " + relative)
    cfg = read(ORIGINAL / "config.json")
    require(sha(REPO / cfg["dataset_relative"]) == cfg["dataset_archive_sha256"], "Dataset changed")
    save_new(HERE / "MANIFEST.json", {"schema": "minimal_comparison.loader_recovery.v1",
             "original_manifest_sha256": ORIGINAL_MANIFEST_SHA,
             "scientific_config_sha256": sha(ORIGINAL / "config.json"),
             "files": pins, "budget": budget_record(), "prior_assessment": prior_assessment(),
             "change": "Immutable producer-validated dataset loading only; no tolerance changes or science changes",
             "scientific_execution_during_preparation": False})


def check():
    manifest = read(HERE / "MANIFEST.json")
    require(sha(ORIGINAL / "MANIFEST.json") == ORIGINAL_MANIFEST_SHA, "Original seal changed")
    require(manifest["budget"] == budget_record(), "Recovery budget changed")
    require(sha(ORIGINAL / "config.json") == manifest["scientific_config_sha256"], "Science configuration changed")
    for relative, digest in manifest["files"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO) and sha(path) == digest, "Pinned source/evidence changed: " + relative)
    require(manifest["prior_assessment"] == prior_assessment(), "Prior assessment changed")
    cfg = read(ORIGINAL / "config.json")
    require(sha(REPO / cfg["dataset_relative"]) == cfg["dataset_archive_sha256"], "Dataset changed")
    files = {str(REPO / p): h for p, h in manifest["files"].items()}
    files[str(HERE / "MANIFEST.json")] = sha(HERE / "MANIFEST.json")
    # Original worker checks its own package identity; both source closures are pinned.
    return cfg, {"package_dir": str(ORIGINAL), "files": files}


def bounded_supervisor(owned, deadline):
    def supervise(command, directory, cap, **kwargs):
        available = deadline - time.monotonic() - 10
        require(available > 0, "Recovery global execution budget exhausted")
        effective = min(float(cap), available)
        result = owned.supervise(command, directory, effective, deadline=deadline - 10, **kwargs)
        result["recovery_effective_supervisor_cap_seconds"] = effective
        return result
    return SimpleNamespace(child_environment=owned.child_environment, supervise=supervise)


def adapted_runner(deadline):
    old = module("_recovery_original_launch", ORIGINAL / "launch.py")
    old.HERE = HERE
    old.APPROVAL_TEXT = APPROVAL_TEXT
    original_load = old.load_module
    def load(name, path):
        if name == "_minimal_saved_metrics":
            return original_load(name, ORIGINAL / "metrics.py")
        value = original_load(name, path)
        return bounded_supervisor(value, deadline) if name == "_minimal_owned_supervisor" else value
    old.load_module = load
    def checked():
        require(time.monotonic() < deadline - 10, "Recovery global execution budget exhausted")
        return check()
    old.check = checked
    return old


def execute(approval):
    started = time.monotonic()
    require(os.name == "posix", "Production launch requires the user-run Linux runtime")
    def stop(signum, frame):
        if signum == signal.SIGALRM:
            raise TimeoutError("Recovery global deadline reached")
        raise KeyboardInterrupt("Recovery supervisor signal " + str(signum))
    handlers = {sig: signal.getsignal(sig) for sig in
                (signal.SIGALRM, signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    for sig in handlers:
        signal.signal(sig, stop)
    signal.setitimer(signal.ITIMER_REAL, DISPATCH_DEADLINE_SECONDS - 10)
    try:
        runner = adapted_runner(started + DISPATCH_DEADLINE_SECONDS)
        result = runner.execute(approval)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    elapsed = time.monotonic() - started
    recovery = {"status": result["status"], "result_sha256": sha(HERE / "run/RESULT.json"),
                "manifest_sha256": sha(HERE / "MANIFEST.json"), "budget": budget_record(),
                "elapsed_seconds_including_initial_checks": elapsed,
                "known_cumulative_active_seconds": KNOWN_PRIOR_SECONDS + elapsed,
                "charged_cumulative_seconds_including_wrapper_allowance": CHARGED_PRIOR_SECONDS + elapsed,
                "prior_assessment": prior_assessment(), "recovery_accounting": result["accounting"],
                "original_raw_cumulative_count_uncertainty_preserved": True,
                "no_automatic_retry": True, "finished_est": datetime.now(EST).isoformat()}
    save_new(HERE / "run/RECOVERY_RESULT.json", recovery)
    return recovery


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seal", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--execute", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    if args.seal:
        seal()
    elif args.check:
        cfg, integrity = check()
        print(json.dumps({"status": "identity_checked_no_scientific_execution", "pinned_files": len(integrity["files"]),
                          "budget": budget_record()}))
    else:
        if args.approval is None:
            parser.error("--execute requires the manifest-bound recovery approval")
        result = execute(args.approval)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "completed_exploratory_comparison" else 1)
