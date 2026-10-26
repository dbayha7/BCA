"""Saved-development decision utility; default is metadata-only preflight."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import random
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
PRIOR = BASE / "unifloral_development_mechanism_audit_v1p1"
PRIOR_MANIFEST_SHA256 = "310781618e4eecade8a2c3b1143a2eacb1f9e40c559aa62339d953eb570f941c"
PRIOR_READER_SHA256 = "099f928807618cc4dccd30265f70dc1175e8a6059ae9924d1ff1e15447c88278"
INPUT_ALLOWLIST_SHA256 = "d5b78438a0135a0ae0ec2b567ebf90f0321f85716d0a6c7434df17e4b73883e9"
PROTOCOL_SHA256 = "ed8d8ef91bc32bad43ad6daee6c30672edf3e04428f01bad219acd6dd980b883"
GROUPED = BASE / "unifloral_grouped_width_validation_v1"
GROUPED_RESULTS = GROUPED / "results_20261026T204745_262478Z"
GROUPED_MANIFEST_SHA256 = "48e63c5d456ca1ffa02efdb54f13c25623f10f6d6245a4f93bde3e942736cfa9"
GROUPED_REPORT_SHA256 = "40514326cbfa8924e9cf9a5950e0824bfe7f49d5fa6a48a4f10120f1a04b784e"
GROUPED_CHECKS_SHA256 = "526833e7f3b112d8db0bb84a53a233719e6cec9faba42f7645745b7d68e03821"
PREDICTION_MODELS = {"base", "base_plus_residual_width", "base_plus_residual_permuted_width"}
FREEZE_NAME = "DEFINITIONS_FROZEN.json"
TIMEOUT_SECONDS = 60
SUCCESS = "completed_saved_development_decision_utility"
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_json(raw):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, parse_constant=invalid)


def write(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def checked(base, relative):
    base = Path(base).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a safe relative authored component")
    path = base / relative
    if path.resolve() != path.absolute() or not path.is_file():
        raise ValueError("Missing or redirected file: " + relative)
    for component in (path, *path.parents):
        if component == base:
            break
        if component.is_symlink():
            raise ValueError("Symlinked source component")
    return path.read_bytes()


def manifest_identity(base, expected=None):
    raw = checked(base, "prepared_files.json")
    if expected is not None and sha(raw) != expected:
        raise ValueError("Pinned prepared manifest changed")
    files = read_json(raw).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Empty prepared file map")
    if base == ROOT and not {"run_checks.py", "test_runner.py", "decision.py", "protocol.json", FREEZE_NAME}.issubset(files):
        raise ValueError("Decision-utility manifest omits a required component")
    for name, digest in files.items():
        if sha(checked(base, name)) != digest:
            raise ValueError("Prepared source changed: " + name)
    return {"manifest_sha256": sha(raw), "verified_files": len(files)}


def load_module(name, path, expected):
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError("Module changed before loading")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def assert_input_scope(allowlist):
    if not isinstance(allowlist, dict) or len(allowlist) != 10 or canonical(allowlist) != INPUT_ALLOWLIST_SHA256:
        raise ValueError("Reader inputs differ from the exact ten-file development-only allowlist")


def prior_reader():
    manifest_identity(PRIOR, PRIOR_MANIFEST_SHA256)
    module = load_module("_decision_utility_pinned_development_reader", PRIOR / "analyze_saved.py", PRIOR_READER_SHA256)
    assert_input_scope(module.INPUT_ALLOWLIST)
    return module


def frozen_protocol():
    raw = checked(ROOT, "protocol.json")
    if sha(raw) != PROTOCOL_SHA256:
        raise ValueError("Freeze the decision-utility protocol before loading numeric development detail")
    protocol = read_json(raw)
    freeze_raw = checked(ROOT, FREEZE_NAME)
    freeze = read_json(freeze_raw)
    if (freeze.get("schema") != "unifloral.development_decision_utility.freeze.v1"
            or freeze.get("protocol_sha256") != PROTOCOL_SHA256
            or freeze.get("decision_utility_computed") is not False
            or freeze.get("prior_development_outcomes_and_predictive_summaries_known") is not True
            or freeze.get("phase") != "post_hoc_exploratory_internal_validation"
            or type(freeze.get("confirmation_files_read")) is not int or freeze["confirmation_files_read"] != 0):
        raise ValueError("Decision-utility definition-freeze receipt does not match protocol")
    if (protocol.get("phase") != "exploratory_saved_development_decision_utility_only"
            or protocol.get("cpu_timeout_seconds") != TIMEOUT_SECONDS
            or protocol.get("previous_loader_directory") != PRIOR.name
            or protocol.get("previous_loader_sha256") != PRIOR_READER_SHA256
            or protocol.get("previous_manifest_sha256") != PRIOR_MANIFEST_SHA256
            or protocol.get("grouped_manifest_sha256") != GROUPED_MANIFEST_SHA256
            or protocol.get("prediction_report_sha256") != GROUPED_REPORT_SHA256
            or protocol.get("prediction_checks_sha256") != GROUPED_CHECKS_SHA256):
        raise ValueError("Unsupported decision-utility scope or budget")
    return protocol, {"protocol_sha256": sha(raw), "freeze_receipt_sha256": sha(freeze_raw)}


def verify():
    """Read/hash/metadata checks only; never fit a model or inspect confirmation."""
    own = manifest_identity(ROOT)
    protocol, protocol_identity = frozen_protocol()
    reader = prior_reader()
    previous_identity, _ = reader.verify()
    assert_input_scope(previous_identity.get("input_hashes"))
    if previous_identity.get("prepared_manifest_sha256") != PRIOR_MANIFEST_SHA256:
        raise ValueError("Prior reader receipt differs from the pinned source manifest")
    predictions = verify_prediction_stage(previous_identity)
    return {"decision_utility_sources": own, "frozen_protocol": protocol_identity,
            "prior_reader_sha256": PRIOR_READER_SHA256, "prior_sources_and_inputs": previous_identity,
            "input_allowlist_sha256": INPUT_ALLOWLIST_SHA256, "saved_prediction_stage": predictions}, protocol


def verify_prediction_stage(previous_identity):
    grouped = manifest_identity(GROUPED, GROUPED_MANIFEST_SHA256)
    grouped_identity = {"grouped_sources": grouped,
                        "frozen_protocol": {"protocol_sha256": sha(checked(GROUPED, "protocol.json")),
                                            "freeze_receipt_sha256": sha(checked(GROUPED, FREEZE_NAME))},
                        "prior_reader_sha256": PRIOR_READER_SHA256,
                        "prior_sources_and_inputs": previous_identity,
                        "input_allowlist_sha256": INPUT_ALLOWLIST_SHA256}
    raw_report, raw_checks = checked(GROUPED_RESULTS, "report.json"), checked(GROUPED_RESULTS, "checks.json")
    if sha(raw_report) != GROUPED_REPORT_SHA256 or sha(raw_checks) != GROUPED_CHECKS_SHA256:
        raise ValueError("Pinned grouped prediction results changed")
    report, checks = read_json(raw_report), read_json(raw_checks)
    for value in (report, checks):
        if (value.get("status") != "completed_grouped_saved_development_prediction"
                or value.get("diagnostic_result_valid") is not True
                or value.get("identity_before") != grouped_identity or value.get("identity_after") != grouped_identity):
            raise ValueError("Saved prediction stage did not complete under the exact current pinned source/input identity")
        work = value.get("new_work", {})
        for key in ("calibrator_updates", "candidate_actions_generated", "confirmation_files_read", "gpu_work",
                    "policy_forward_calls", "policy_optimizer_updates", "simulator_calls"):
            if type(work.get(key)) is not int or work[key] != 0:
                raise ValueError("Prediction-stage no-new-policy/data boundary failed")
    if report.get("exit_code") != 0 or report.get("checks_sha256") != GROUPED_CHECKS_SHA256:
        raise ValueError("Prediction supervisor/worker result linkage is invalid")
    analysis = checks.get("analysis", {})
    if (analysis.get("status") != "pass" or analysis.get("phase") != "exploratory_grouped_saved_development_only"
            or analysis.get("confirmation_files_read") != 0 or analysis.get("new_policy_training") is not False
            or analysis.get("diagnostic_regression_fits") != 30):
        raise ValueError("Saved predictive analysis has the wrong stage or fit budget")
    return {"manifest": grouped, "report_sha256": sha(raw_report), "checks_sha256": sha(raw_checks),
            "source_identity_sha256": canonical(grouped_identity), "same_development_inputs_and_checkpoint": True,
            "scope": "Byte-authenticated cached predictions; no predictive model is imported, refitted or evaluated here"}


def read_prediction_receipt():
    raw = checked(GROUPED_RESULTS, "checks.json")
    if sha(raw) != GROUPED_CHECKS_SHA256:
        raise ValueError("Cached predictions changed before numeric loading")
    return read_json(raw)["analysis"]


def expected_fold_assignment(records):
    """Reproduce only the fixed metadata shuffle; no features, targets or fits."""
    folds = [-1] * 128
    for stratum, (batch, t) in enumerate((batch, t) for batch in ("development_1", "development_2") for t in (0, 25, 50, 75)):
        ids = sorted((group for group, row in records.items() if row["batch"] == batch and row["time"] == t),
                     key=lambda group: records[group]["episode_seed"])
        if len(ids) != 16:
            raise ValueError("Expected16 complete episodes per development batch/time cell")
        random.Random(2026103021 + stratum).shuffle(ids)
        for position, group in enumerate(ids):
            folds[group] = position % 8
    return folds


def validate_predictions(batches, receipt):
    """Authenticate OOF/transfer exclusion and alignment without computing utility."""
    records = {}
    if [batch.get("name") for batch in batches] != ["development_1", "development_2"]:
        raise ValueError("Expected exactly the two pinned development batches")
    for batch in batches:
        if len(batch.get("queries", [])) != 64:
            raise ValueError("Development batch size is not64")
        for row in batch["queries"]:
            group = row.get("group_id")
            if type(group) is not int or group in records or row.get("batch") != batch["name"]:
                raise ValueError("Duplicated/misaligned episode group")
            records[group] = row
    if set(records) != set(range(128)):
        raise ValueError("Saved development groups are not the complete128 episode set")
    folds = receipt.get("fold_assignment")
    if (not isinstance(folds, list) or len(folds) != 128 or any(type(f) is not int for f in folds)
            or folds != expected_fold_assignment(records) or receipt.get("fold_assignment_sha256") != canonical(folds)):
        raise ValueError("Saved grouped-fold assignment differs from its fixed metadata-only construction")
    fits = receipt.get("fit_receipts", [])
    if not isinstance(fits, list) or len(fits) != 10 or len({fit.get("split") for fit in fits}) != 10:
        raise ValueError("Expected eight CV and two transfer fit receipts")
    fits = {fit["split"]: fit for fit in fits}

    def check_fit(name, heldout, training):
        fit = fits.get(name, {})
        for key, expected in (("heldout_group_ids", heldout), ("training_group_ids", training)):
            ids = fit.get(key)
            if (not isinstance(ids, list) or any(type(group) is not int for group in ids)
                    or len(ids) != len(set(ids)) or set(ids) != expected):
                raise ValueError("Saved fit split has missing, duplicate or leaked episode IDs: " + name)
        if set(fit["training_group_ids"]) & set(fit["heldout_group_ids"]) or set(fit.get("models", {})) != PREDICTION_MODELS:
            raise ValueError("Prediction model set or training/held-out exclusion failed")

    def check_episode(row, expected_fold):
        group = row.get("group_id")
        if type(group) is not int or group not in records:
            raise ValueError("Unknown prediction episode")
        expected = records[group]
        if (any(row.get(key) != expected[key] for key in ("batch", "time", "episode_seed"))
                or row.get("fold") != expected_fold):
            raise ValueError("Saved prediction metadata differs from its source episode")
        predictions = row.get("predictions", {})
        if set(predictions) != PREDICTION_MODELS:
            raise ValueError("Saved prediction model keys changed")
        for values in predictions.values():
            if (not isinstance(values, list) or len(values) != 9
                    or any(type(value) not in (int, float) or not math.isfinite(value) for value in values)):
                raise ValueError("Saved predictions must contain nine finite real values per model")
        return group

    episodes = receipt.get("cv", {}).get("episodes", [])
    if not isinstance(episodes, list) or len(episodes) != 128:
        raise ValueError("Missing complete OOF prediction rows")
    seen = set()
    for row in episodes:
        group = row.get("group_id")
        if type(group) is not int or not 0 <= group < 128:
            raise ValueError("Invalid OOF group ID")
        group = check_episode(row, folds[group])
        if group in seen:
            raise ValueError("Duplicate OOF prediction row")
        seen.add(group)
    for fold in range(8):
        heldout = {group for group in records if folds[group] == fold}
        if len(heldout) != 16:
            raise ValueError("Each grouped fold must hold16 episodes")
        check_fit("cv_fold_" + str(fold), heldout, set(records) - heldout)
    transfers = receipt.get("cross_batch_transfer", [])
    if not isinstance(transfers, list) or len(transfers) != 2:
        raise ValueError("Expected both saved cross-batch directions")
    directions = set()
    for transfer in transfers:
        train, held = transfer.get("training_batch"), transfer.get("heldout_batch")
        if (train, held) not in {("development_1", "development_2"), ("development_2", "development_1")} or (train, held) in directions:
            raise ValueError("Duplicate or invalid cross-batch prediction direction")
        directions.add((train, held))
        training = {g for g, row in records.items() if row["batch"] == train}
        heldout = set(records) - training
        check_fit("transfer_" + train + "_to_" + held, heldout, training)
        rows = transfer.get("episodes", [])
        if not isinstance(rows, list) or len(rows) != 64:
            raise ValueError("Cross-batch prediction row count changed")
        ids = [check_episode(row, None) for row in rows]
        if len(set(ids)) != 64 or set(ids) != heldout:
            raise ValueError("Cross-batch predictions leak or omit target episodes")
    return {"oof_episodes": 128, "folds": 8, "heldout_episodes_per_fold": 16, "training_episodes_per_fold": 112,
            "all_heldout_episode_ids_excluded_from_training": True, "prediction_models": sorted(PREDICTION_MODELS),
            "finite_values_per_model_per_episode": 9, "transfer_directions": 2, "transfer_episodes_per_direction": 64,
            "fold_assignment_sha256": canonical(folds), "prediction_checks_sha256": GROUPED_CHECKS_SHA256,
            "new_predictive_model_imports": 0, "new_diagnostic_fits": 0}


def decision_module():
    files = read_json(checked(ROOT, "prepared_files.json"))["files"]
    return load_module("_saved_decision_utility", ROOT / "decision.py", files["decision.py"])


def validate_analysis(detail):
    required = {"status": "pass", "claim_scope": "fixed_heldout_signed_error_correction_diagnostic_only",
                "phase": "exploratory_saved_development_decision_utility_only", "new_policy_training": False, "new_model_fitted": False,
                "new_diagnostic_fits": 0, "new_simulator_calls": 0, "new_gpu_work": 0, "confirmation_files_read": 0,
                "full_training_hold": True, "automatic_training_approval": False,
                "method_or_strength_selected": False}
    if not isinstance(detail, dict) or any(type(detail.get(k)) is not type(v) or detail[k] != v for k, v in required.items()):
        raise ValueError("Decision result exceeds the frozen no-fit diagnostic scope")


def finalize(result):
    valid = (result.get("status") == SUCCESS and result.get("identity_before") is not None
             and result.get("identity_before") == result.get("identity_after"))
    result["diagnostic_result_valid"] = valid
    if not valid:
        if result.get("status") == SUCCESS:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = "Input/source/protocol identities are missing or changed"
        if "analysis" in result:
            result["unvalidated_analysis"] = result.pop("analysis")


def worker(output, expected_identity):
    result = {"started_utc": utc(), "status": "blocked_or_failed", "diagnostic_result_valid": False,
              "scope": "Saved held-out prediction decision diagnostic only; no new models, policy training, or data",
              "new_work": {"policy_optimizer_updates": 0, "policy_forward_calls": 0, "calibrator_updates": 0,
                           "simulator_calls": 0, "gpu_work": 0, "candidate_actions_generated": 0, "diagnostic_model_fits": 0,
                           "confirmation_files_read": 0}}
    with (output / "stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"utc": utc(), "stage": name}) + "\n")
            stages.flush()
        try:
            stage("frozen_protocol_source_and_input_preflight")
            result["identity_before"], protocol = verify()
            if canonical(result["identity_before"]) != expected_identity:
                raise ValueError("Worker identity differs from supervisor preflight")
            import numpy as np
            if platform.system() != "Windows" or platform.python_version() != "3.12.14" or np.__version__ != "2.3.5":
                raise ValueError("Expected existing Windows Python3.12.14 NumPy2.3.5 CPU analysis runtime")
            result["runtime"] = {"platform": platform.system(), "python": platform.python_version(),
                                 "numpy": np.__version__, "executable": sys.executable}
            stage("load_exact_prior_validated_development_batches")
            batches = prior_reader().load_batches()
            result["validated_inputs"] = {"batches": 2, "states": 128, "saved_candidate_rows": 1152,
                                          "loader": "Exact reviewed v1p1 development-only loader"}
            stage("authenticate_saved_oof_and_transfer_predictions")
            prediction_receipt = read_prediction_receipt()
            result["validated_predictions"] = validate_predictions(batches, prediction_receipt)
            stage("evaluate_fixed_saved_prediction_decision_utility")
            result["analysis"] = decision_module().analyze({"batches": batches, "prediction_receipt": prediction_receipt}, protocol)
            validate_analysis(result["analysis"])
            result["status"] = SUCCESS
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            stage("source_protocol_and_saved_input_postchecks")
            try:
                result["identity_after"], _ = verify()
                if result.get("identity_before") != result["identity_after"]:
                    raise ValueError("Source/protocol/saved inputs changed during decision utility")
            except Exception as error:
                result["status"] = "blocked_or_failed"
                result["postcheck_error"] = str(error)
            finalize(result)
            result["finished_utc"] = utc()
            write(output / "checks.json", result)
    return 0 if result["diagnostic_result_valid"] else 1


def run_child(command, output, env):
    """Only the newly created CPU child is bounded/killed; no process discovery."""
    result = {}
    start = time.monotonic()
    with (output / "checks.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            result["exit_code"] = process.wait(timeout=TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
            result["interrupted"] = isinstance(error, KeyboardInterrupt)
            process.kill()
            result["exit_code"] = process.wait()
    result["elapsed_seconds"] = time.monotonic() - start
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Frozen metadata/hash gates only; default")
    modes.add_argument("--execute", action="store_true", help="Run only the frozen CPU saved-prediction decision diagnostic")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-identity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        output = args.worker_output.resolve()
        if (args.execute or args.preflight or not args.expected_identity or output.parent != ROOT
                or not output.name.startswith("results_") or not output.is_dir()):
            parser.error("Worker requires an exclusive local supervisor output and matching identity")
        return worker(output, args.expected_identity)
    if args.expected_identity:
        parser.error("Worker identity requires worker-output")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"started_utc": utc(), "mode": "execute" if args.execute else "preflight", "status": "blocked_or_failed",
              "diagnostic_result_valid": False, "cpu_worker_timeout_seconds": TIMEOUT_SECONDS,
              "timeout_scope": "Child wall-clock wait capped at60seconds; supervisor hashing/postchecks outside that cap"}
    try:
        result["identity_before"], _ = verify()
        if not args.execute:
            result["status"] = "preflight_passed"
        else:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-output", str(output),
                       "--expected-identity", canonical(result["identity_before"])]
            result.update(run_child(command, output, env))
            if (output / "checks.json").is_file():
                raw = (output / "checks.json").read_bytes()
                child = read_json(raw)
                result["checks_sha256"] = sha(raw)
                for key in ("runtime", "validated_inputs", "validated_predictions", "new_work", "error", "postcheck_error"):
                    if key in child:
                        result[key] = child[key]
                if (result["exit_code"] == 0 and child.get("status") == SUCCESS and child.get("diagnostic_result_valid") is True
                        and child.get("identity_before") == result["identity_before"] == child.get("identity_after")
                        and not result.get("timed_out") and not result.get("interrupted")):
                    validate_analysis(child.get("analysis"))
                    result["status"] = SUCCESS
            else:
                result["worker_outcome"] = "No complete diagnostic receipt; preserve partial stage/log evidence without claiming a result"
    except BaseException as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            result["identity_after"], _ = verify()
            if result.get("identity_before") != result["identity_after"]:
                raise ValueError("Saved inputs or prepared source/protocol changed during supervision")
        except Exception as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = str(error)
        finalize(result)
        result["finished_utc"] = utc()
        write(output / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json")}))
    return 0 if result["status"] in ("preflight_passed", SUCCESS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
