"""Prepare only seed 2 plus unchanged evaluation; expanded physical work needs approval."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
PREVIOUS = HERE.parent / "minimal_comparison_recovery_v1"
ORIGINAL = HERE.parent / "minimal_comparison"
PREVIOUS_MANIFEST_SHA = "fc20c6b89b04be6398be64e674dfa1c425ba832436e788e564431acc34c17fc8"
PREVIOUS_RESULT_SHA = "5e0b6ec2deeca5c4ec91ec568f6bc177073cdd72e18590bb4025a7193638ffd3"
PREVIOUS_RECOVERY_SHA = "5d708783dbf77d5427c864b972ba070e238957a458fc75ca6c764790d0a4694a"
PROCESS_CAP = 7500
CHARGED_PRIOR_SECONDS = 3300
APPROVAL_TEXT = "I approve this seed-2-only continuation and unchanged three-seed evaluation, with cumulative physical caps of 100000 critic, 50000 actor, 100000 auxiliary updates and 10 calibrations, and a 7500-second process cap within the original three-hour time allowance."
COMPLETED = tuple(f"{kind}_s{i}" for i in (0, 1) for kind in ("warmup", "native", "bca"))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def import_previous(name="_continuation_previous_launch"):
    require(sha(PREVIOUS / "MANIFEST.json") == PREVIOUS_MANIFEST_SHA, "Previous recovery manifest changed")
    pins = read(PREVIOUS / "MANIFEST.json")["files"]
    source = PREVIOUS / "launch.py"
    require(sha(source) == pins[source.relative_to(REPO).as_posix()], "Previous launcher changed before import")
    spec = importlib.util.spec_from_file_location(name, source)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_previous():
    previous = import_previous()
    previous.check()  # Pure hashes/JSON only; no production imports or reservation.
    run = PREVIOUS / "run"
    require(sha(run / "RESULT.json") == PREVIOUS_RESULT_SHA and
            sha(run / "RECOVERY_RESULT.json") == PREVIOUS_RECOVERY_SHA, "Interrupted attempt evidence changed")
    result, receipt = read(run / "RESULT.json"), read(run / "RECOVERY_RESULT.json")
    require(receipt["result_sha256"] == PREVIOUS_RESULT_SHA and result["status"] == "incomplete",
            "Wrong previous terminal result")
    supervisor = read(run / "warmup_s2/SUPERVISOR_RESULT.json")
    require(supervisor["error"] == "KeyboardInterrupt: Recovery supervisor signal 1" and
            supervisor["interrupted"] is True and supervisor["timed_out"] is False and
            supervisor["exit_code"] == -9, "Previous interruption differs")
    require(not list((run / "warmup_s2").glob("*.msgpack")), "An unfinished-warmup checkpoint now exists; review before restart")
    require(not list(run.glob("evaluate_s*")), "Prior evaluation artifacts now exist; avoid duplicate evaluation")
    cfg = read(ORIGINAL / "config.json")
    cfg_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    verified = {}
    for name in COMPLETED:
        directory = run / name
        report = read(directory / "report.json")
        outcome = result["jobs"][name]["supervisor"]
        require(sha(directory / "report.json") == result["jobs"][name]["report_sha256"], "Reused report hash differs")
        require(report["status"] == "completed" and outcome["exit_code"] == 0 and
                not outcome["timed_out"] and not outcome["interrupted"], "Reused job did not complete")
        require(report["job_sha256"] == sha(directory / "job.json") and report["config_sha256"] == cfg_digest,
                "Reused job/config binding differs")
        require(report["recovery_manifest_sha256"] == PREVIOUS_MANIFEST_SHA and
                report["integrity_after"] == {"changed": [], "verified_files": 198}, "Reused source identity differs")
        if not name.startswith("warmup"):
            require(report["warmup_report_sha256"] == sha(run / ("warmup_s" + name[-1]) / "report.json"),
                    "Warmup/continuation pairing differs")
        checkpoints = {}
        for tag, checkpoint in report["checkpoints"].items():
            for field in ("actor", "state"):
                entry = checkpoint[field]
                require(sha(directory / entry["path"]) == entry["sha256"], "Reused checkpoint differs")
                checkpoints[entry["path"]] = entry["sha256"]
        verified[name] = {"report_sha256": sha(directory / "report.json"), "checkpoints": checkpoints,
                          "new_critic_updates": 10000, "new_actor_updates": 5000,
                          "new_auxiliary_updates": 10000, "new_calibrations": 1}
    progress = read(run / "warmup_s2/progress.json")
    require(progress["step"] == 5000 and progress["counts"]["critic"] == 5000 and
            progress["counts"]["actor"] == 2500 and progress["counts"]["auxiliary"] == 5000 and
            progress["counts"]["refresh_accepted"] == 0, "Interrupted progress changed")
    return {"completed_jobs": verified, "previous_terminal_receipt": receipt,
            "interrupted_progress": progress, "signal_source": "unknown; only SIGHUP receipt established"}


def budget_record():
    receipt = read(PREVIOUS / "run/RECOVERY_RESULT.json")
    known = receipt["known_cumulative_active_seconds"]
    charged = receipt["charged_cumulative_seconds_including_wrapper_allowance"]
    require(abs(known - 3154.914181852) < 1e-8 and abs(charged - 3228.505985861) < 1e-8,
            "Prior time debit differs")
    return {"original_time_allowance_seconds": 10800, "known_prior_seconds": known,
            "previous_charged_seconds": charged, "previous_remaining_seconds": 10800 - charged,
            "additional_unmeasured_wrapper_allowance_seconds": CHARGED_PRIOR_SECONDS - charged,
            "all_wrapper_allowance_seconds": CHARGED_PRIOR_SECONDS - known,
            "charged_prior_seconds": CHARGED_PRIOR_SECONDS, "process_cap_seconds": PROCESS_CAP,
            "allowances_are_not_measured_overhead": True,
            "new_work": {"critic_updates": 30000, "actor_updates": 15000,
                         "auxiliary_updates": 30000, "calibrations": 3, "return_episodes": 768,
                         "expected_simulator_steps": 170400},
            "lost_warmup_bounds": {"critic_updates": [5000, 10000], "actor_updates": [2500, 5000],
                                   "auxiliary_updates": [5000, 10000], "calibrations": [0, 1]},
            "cumulative_physical_bounds": {"critic_updates": [95000, 100000], "actor_updates": [47500, 50000],
                                          "auxiliary_updates": [95000, 100000], "calibrations": [9, 10]},
            "proposed_cumulative_caps": {"critic_updates": 100000, "actor_updates": 50000,
                                         "auxiliary_updates": 100000, "calibrations": 10},
            "original_simulator_cap_unchanged": 1706400,
            "prior_raw_simulator_bounds_preserved": [0, 128000],
            "prior_raw_count_uncertainty_preserved": True,
            "first_failure_zero_updates_is_separate_source_path_assessment": True}


def jobs(root):
    root = Path(root)
    warm, native = root / "warmup_s2", root / "native_s2"
    yield {"id": "warmup_s2", "kind": "warmup", "seed_index": 2}
    yield {"id": "native_s2", "kind": "branch", "seed_index": 2, "warmup_dir": str(warm)}
    yield {"id": "bca_s2", "kind": "bca_branch", "seed_index": 2,
           "warmup_dir": str(warm), "native_branch_dir": str(native)}
    for i in range(3):
        source = PREVIOUS / "run" if i in (0, 1) else root
        yield {"id": f"evaluate_s{i}", "kind": "evaluate", "seed_index": i,
               "warmup_dir": str(source / f"warmup_s{i}"),
               "native_branch_dir": str(source / f"native_s{i}"), "bca_branch_dir": str(source / f"bca_s{i}")}


def seal():
    require(not (HERE / "MANIFEST.json").exists(), "Already sealed; preserve this version")
    evidence = verify_previous()
    pins = dict(read(PREVIOUS / "MANIFEST.json")["files"])
    extras = [PREVIOUS / "MANIFEST.json", PREVIOUS / "LAUNCH_APPROVAL.json"]
    extras += [p for p in (PREVIOUS / "run").rglob("*") if p.is_file()]
    extras += [p for p in HERE.iterdir() if p.is_file() and p.suffix in {".py", ".md"}]
    for path in extras:
        pins[path.relative_to(REPO).as_posix()] = sha(path)
    save_new(HERE / "MANIFEST.json", {"schema": "minimal_comparison.seed2_continuation.v1",
             "previous_manifest_sha256": PREVIOUS_MANIFEST_SHA, "files": pins,
             "budget": budget_record(), "reuse_evidence": evidence,
             "scientific_config_sha256": sha(ORIGINAL / "config.json"),
             "execution_approved_at_preparation": False, "no_automatic_retry": True})


def check():
    m = read(HERE / "MANIFEST.json")
    require(m["budget"] == budget_record(), "Budget changed")
    require(m["scientific_config_sha256"] == sha(ORIGINAL / "config.json"), "Science changed")
    for relative, digest in m["files"].items():
        path = (REPO / relative).resolve()
        require(path.is_relative_to(REPO) and sha(path) == digest, "Source/evidence changed: " + relative)
    require(m["reuse_evidence"] == verify_previous(), "Reuse evidence changed")
    files = {str(REPO / p): h for p, h in m["files"].items()}
    files[str(HERE / "MANIFEST.json")] = sha(HERE / "MANIFEST.json")
    return read(ORIGINAL / "config.json"), {"package_dir": str(ORIGINAL), "files": files}


def authorize(path):
    approval = read(path)
    require(approval.get("approval") == APPROVAL_TEXT and approval.get("launch_authorized") is True and
            approval.get("manifest_sha256") == sha(HERE / "MANIFEST.json") and
            approval.get("proposed_cumulative_caps") == budget_record()["proposed_cumulative_caps"],
            "Expanded cumulative-work approval has not been provided for this manifest")


def adapted_runner():
    previous = import_previous()
    previous.HERE = HERE
    previous.APPROVAL_TEXT = APPROVAL_TEXT
    previous.RECOVERY_PROCESS_CAP = PROCESS_CAP
    previous.DISPATCH_DEADLINE_SECONDS = PROCESS_CAP - 10
    previous.KNOWN_PRIOR_SECONDS = budget_record()["known_prior_seconds"]
    previous.CHARGED_PRIOR_SECONDS = CHARGED_PRIOR_SECONDS
    previous.budget_record = budget_record
    previous.prior_assessment = verify_previous
    previous.check = check
    adapted = previous.adapted_runner
    def runner(deadline):
        original = adapted(deadline)
        original.jobs = jobs
        return original
    previous.adapted_runner = runner
    return previous


def cumulative_work_bounds(new_accounting):
    """Keep the old raw receipt; strengthen only the displayed lower bound using saved progress."""
    prior = read(PREVIOUS / "run/RECOVERY_RESULT.json")["recovery_accounting"]["counts"]
    progress_lower = {"critic_updates": 65000, "actor_updates": 32500,
                      "auxiliary_updates": 65000, "calibrations": 6, "simulator_steps": 0}
    return {name: {"lower": max(old["lower"], progress_lower[name]) + new_accounting["counts"][name]["lower"],
                   "upper": old["upper"] + new_accounting["counts"][name]["upper"],
                   "known": old["known"] and new_accounting["counts"][name]["known"]}
            for name, old in prior.items()}


def execute(approval):
    check()
    authorize(approval)
    result = adapted_runner().execute(approval)
    summary = {"status": result["status"], "manifest_sha256": sha(HERE / "MANIFEST.json"),
               "recovery_result_sha256": sha(HERE / "run/RECOVERY_RESULT.json"),
               "cumulative_work_bounds": cumulative_work_bounds(result["recovery_accounting"]),
               "accounting_note": "Previous raw receipts unchanged; lower bounds include saved seed-2 progress. The first data-loading failure remains a separate source-inferred zero-update assessment.",
               "budget": budget_record(), "no_automatic_retry": True}
    save_new(HERE / "run/CONTINUATION_RESULT.json", summary)
    return summary


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
        cfg, record = check()
        print(json.dumps({"status": "prepared_pending_expanded_work_approval", "pinned_files": len(record["files"]),
                          "budget": budget_record(), "run_reserved": (HERE / "run").exists()}))
    else:
        if args.approval is None:
            parser.error("--execute requires a new explicit approval; preparation does not create one")
        result = execute(args.approval)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "completed_exploratory_comparison" else 1)
