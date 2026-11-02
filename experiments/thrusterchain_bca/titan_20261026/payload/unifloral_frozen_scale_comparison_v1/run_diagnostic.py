"""New-start frozen scale comparison; default is a stdlib-only provenance preflight."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
BLOCKS = BASE / "unifloral_scale_gradient_blocks_v1"
BLOCK_RESULTS = BLOCKS / "results_20261026T163812_964247Z"
EVALUATOR = BASE / "unifloral_frozen_usefulness_v1_seedfix"
CHECKPOINT = BASE / "unifloral_export_recovery_v2/recovery_20261023T214750_256386Z/inference_export/eligible_checkpoint.json"
BLOCK_MANIFEST_SHA = "02df7804d2cf87b1d9de1ef717a065fbce8d56d0ea4decc8ae7482789beb244f"
BLOCK_RUNNER_SHA = "ada18eed08a0600961da431d027cae15f90172d96c33da9d64bbd575b06aeed7"
BLOCK_REPORT_SHA = "8db13c0b2ea477c9529dce2656ea3c65dfb0b5d93515dea4a43e82c681172a26"
BLOCK_CHECKS_SHA = "049b1a6b6cb83620fc7fe08dd8f78f7d038d5c38aa457dea046e34b5007b251a"
CONTRAST_SOURCE = "unifloral_scale_gradient_validation_v1/results_20261026T021136_510051Z/checks.json"
CONTRAST_SOURCE_SHA = "f634a043ea5158829b09aeae348a6489d00ef637daa982a272dff63c6fb9ba5c"
EVALUATOR_MANIFEST_SHA = "c21f3ade3dddcb06068168de0a3c509847f4f0d8f4eb6a499d5b3a8b7d851fc2"
EVALUATOR_RUNNER_SHA = "c11936ca9dc2bcb75f4c43bfeaeab4d4cf9d94d8bccc2db8d21b267f31946898"
INFERENCE_SHA = "b21f603622eb2e46c9eb02bd10ce210311af1273403173e0769764567bd05400"
ORIGINAL_SIMULATION_SHA = "0c78e20eb16884a1868c372e45dcfc2659723058b01d04655f689614ade7d824"
SIMULATION_SHA = "77707f12e1f207a585299f98753264708a443f6bc17c00b32ee1c5816780203f"
XLA = "--xla_gpu_triton_gemm_any=True"
TIMEOUT_SECONDS = 600
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return sha(raw)


def checked(base, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Expected a relative pinned component")
    path = base / relative
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()) or not path.is_file():
        raise ValueError("Unsafe or missing source component: " + relative)
    return path.read_bytes()


def manifest_identity(base, expected=None):
    raw = checked(base, "prepared_files.json")
    if expected is not None and sha(raw) != expected:
        raise ValueError("Pinned source manifest changed: " + base.name)
    files = json.loads(raw).get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing source file map: " + base.name)
    for name, digest in files.items():
        if sha(checked(base, name)) != digest:
            raise ValueError("Prepared source changed: " + base.name + "/" + name)
    if base == ROOT and not {"run_diagnostic.py", "simulation.py", "SIMULATION_DIFF.json",
                             "scoring.py", "score_contract.py", "protocol.json"}.issubset(files):
        raise ValueError("Comparison manifest omits a required authored component")
    return {"manifest_sha256": sha(raw), "verified_files": len(files)}


def load_module(name, path, expected):
    """Execute only the exact bytes just hashed; called at explicit entry points."""
    raw = Path(path).read_bytes()
    if sha(raw) != expected:
        raise ValueError("Module changed before loading: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def helpers():
    manifest_identity(BLOCKS, BLOCK_MANIFEST_SHA)
    manifest_identity(EVALUATOR, EVALUATOR_MANIFEST_SHA)
    blocks = load_module("_comparison_blocks_provenance", BLOCKS / "run_checks.py", BLOCK_RUNNER_SHA)
    evaluator = load_module("_comparison_original_evaluator", EVALUATOR / "run_diagnostic.py", EVALUATOR_RUNNER_SHA)
    return blocks, evaluator


def protocol_value():
    protocol = read(ROOT / "protocol.json")
    fixed = {"phase": "development", "query_count": 64, "query_times": [0, 25, 50, 75],
             "query_seed": 2026102700, "candidate_seed": 2026102807, "gamma": 0.98,
             "horizon": 100, "family": "recorded", "physics": {"stall": True, "stall_slope": 1.5},
             "observation_mode": "time_aware_14d", "task_semantics": "finite_horizon", "terminal_bootstrap": False,
             "max_transition_steps": 60000, "max_continuations": 576,
             "learner_and_calibrator_updates": 0, "timeout_seconds": TIMEOUT_SECONDS,
             "bootstrap_seed": 2026102343, "bootstrap_draws": 2000,
             "confirmation_seed_start_reserved_not_generated": 2026102500,
             "confirmation_count_reserved_not_generated": 64,
             "previous_development_seed_start": 2026102400, "previous_development_count": 64,
             "scientific_validation_status": "incomplete_design_review_required"}
    if any(type(protocol.get(key)) is not type(value) or protocol[key] != value for key, value in fixed.items()):
        raise ValueError("Only the frozen new-start development protocol is supported")
    previous = set(range(protocol["previous_development_seed_start"],
                         protocol["previous_development_seed_start"] + protocol["previous_development_count"]))
    current = set(range(protocol["query_seed"], protocol["query_seed"] + protocol["query_count"]))
    confirmation = set(range(protocol["confirmation_seed_start_reserved_not_generated"],
                             protocol["confirmation_seed_start_reserved_not_generated"] + protocol["confirmation_count_reserved_not_generated"]))
    if previous & current or previous & confirmation or current & confirmation:
        raise ValueError("Previous development, new development and confirmation starts must be disjoint")
    return protocol


def verify():
    """Read/hash gates only, including original stdlib provenance helpers."""
    own = manifest_identity(ROOT)
    blocks, evaluator = helpers()
    block_identity, _ = blocks.verify()
    core_identity = evaluator.identities()
    report_raw = checked(BLOCK_RESULTS, "report.json")
    checks_raw = checked(BLOCK_RESULTS, "checks.json")
    if sha(report_raw) != BLOCK_REPORT_SHA or sha(checks_raw) != BLOCK_CHECKS_SHA:
        raise ValueError("Passing fitting-block evidence changed")
    report, detail = json.loads(report_raw), json.loads(checks_raw)
    rows = detail.get("checks", [])
    if (report.get("status") != "pass" or report.get("exit_code") != 0
            or report.get("check_count") != 10 or report.get("checks_sha256") != BLOCK_CHECKS_SHA
            or report.get("identity_before") != block_identity or report.get("identity_after") != block_identity
            or report.get("runtime_verified") is not True or report.get("zero_work_verified") is not True
            or detail.get("status") != "pass" or detail.get("passed") is not True
            or len(rows) != 10 or any(row.get("status") != "pass" for row in rows)):
        raise ValueError("The fixed fitting-block audit did not pass with unchanged source identities")
    zero = blocks.check_zero_work(detail)
    if (report.get("zero_work") != zero or zero.get("gradient_blocks_completed") != 8
            or zero.get("candidate_banks_generated") != 0 or zero.get("scales_refitted") != 0
            or zero.get("development_labels_read") is not False or zero.get("confirmation_data_read") is not False):
        raise ValueError("Fitting-block evidence does not establish the declared preceding stage")
    original = checked(EVALUATOR, "simulation.py")
    updated = checked(ROOT, "simulation.py")
    if sha(original) != ORIGINAL_SIMULATION_SHA or sha(updated) != SIMULATION_SHA:
        raise ValueError("Simulation source identity changed")
    expected = original
    for old, new in ((b'"query_seed": 2026102400', b'"query_seed": 2026102700'),
                     (b'"candidate_seed": 2026102342', b'"candidate_seed": 2026102807')):
        if expected.count(old) != 1:
            raise ValueError("Original simulation seed anchor is ambiguous")
        expected = expected.replace(old, new)
    if updated != expected:
        raise ValueError("Simulation differs beyond the two prospective seed replacements")
    protocol = protocol_value()
    contrast_raw = checked(BASE, CONTRAST_SOURCE)
    if (sha(contrast_raw) != CONTRAST_SOURCE_SHA or protocol.get("contrast_source") != CONTRAST_SOURCE
            or protocol.get("contrast_source_sha256") != CONTRAST_SOURCE_SHA):
        raise ValueError("Fixed contrast provenance differs from the pinned fitting-only audit")
    fitting = json.loads(contrast_raw)
    if fitting.get("status") != "pass" or fitting.get("passed") is not True:
        raise ValueError("The source fitting audit did not pass")
    fitting_checks = {row["name"]: row["details"] for row in fitting["checks"]}
    statistics = fitting_checks["fixed_fitting_candidate_bank_and_contrast_scales"]["statistics"]
    contrasts = {key: statistics[source]["centered_rms"] for key, source in
                 (("q1", "q1"), ("u", "actual_u"), ("disagreement", "raw_abs_twin_disagreement"),
                  ("actor_distance", "actor_squared_distance"))}
    if any(type(value) is not float or not 0 < value < float("inf") for value in contrasts.values()):
        raise ValueError("Saved fitting contrasts must be positive finite scalars")
    factors = {"bca": fitting_checks["detached_fitting_statistics_live_action_route"]["factor"],
               "disagreement": contrasts["q1"] / contrasts["disagreement"],
               "actor_distance": contrasts["q1"] / contrasts["actor_distance"]}
    if (factors["bca"] != 31.372587203979492 or protocol.get("fitting_contrasts") != contrasts
            or protocol.get("contrast_factors") != factors):
        raise ValueError("Protocol factors or contrasts differ from exact saved fitting evidence")
    factor_identity = {"source": CONTRAST_SOURCE, "source_sha256": sha(contrast_raw),
                       "fitting_contrasts": contrasts, "contrast_factors": factors,
                       "bca_source": "detached_fitting_statistics_live_action_route.details.factor; exact saved float32 result",
                       "control_rule": "Saved action-centered fitting C_Q divided by the respective saved control contrast"}
    inference = load_module("_comparison_contract_preflight", EVALUATOR / "inference.py", INFERENCE_SHA)
    _, _, receipt = inference.read_contract(CHECKPOINT, protocol)
    if (receipt["manifest_sha256"] != block_identity["checkpoint_manifest_sha256"]
            or receipt["file_hashes"] != block_identity["checkpoint_role_hashes"]):
        raise ValueError("Checkpoint preflight differs from the fixed recovered export")
    return {"comparison": own, "blocks_provenance": block_identity, "evaluator_core_provenance": core_identity,
            "passing_blocks": {"report_sha256": sha(report_raw), "checks_sha256": sha(checks_raw), "verified_checks": 10},
            "protocol_sha256": sha(checked(ROOT, "protocol.json")), "simulation_sha256": sha(updated),
            "factor_provenance": factor_identity, "checkpoint": receipt}, protocol


def own_module(name):
    files = read(ROOT / "prepared_files.json")["files"]
    return load_module("_comparison_" + name, ROOT / (name + ".py"), files[name + ".py"])


def verify_scores(prelabel, protocol):
    canonical = lambda value: sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                              ensure_ascii=True, allow_nan=False).encode("utf-8"))
    expected = {"raw_q", "actor", "twin_min"}
    for family in ("bca", "contrast_bca", "contrast_disagreement", "contrast_actor_distance"):
        expected.update(family + "_kappa_" + format(kappa, "g") for kappa in (0, 0.03, 0.1, 0.3, 1, 3, 10))
    if set(prelabel.get("methods", {})) != expected:
        raise ValueError("Exactly the 31 frozen selectors are required, without prior RMS families")
    if (prelabel.get("selections_frozen_before_truevalues") is not True
            or prelabel.get("protocol_sha256") != canonical(protocol)
            or prelabel.get("prelabel_signature") != canonical({key: value for key, value in prelabel.items()
                                                                  if key != "prelabel_signature"})):
        raise ValueError("Prelabel signature, protocol identity or labelblind selection seal is invalid")


def worker(output, expected_identity):
    result = {"started_utc": utc(), "status": "blocked_or_failed", "runtime_verified": False,
              "frozen_state_verified": False, "scientific_validation_status": "incomplete_design_review_required",
              "claim_scope": "New-development frozen first-action comparison; no training efficacy, chosen kappa or production normalization approval"}
    policy, frozen_guard = None, {"forbidden_api_calls": 0}
    with (output / "stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"utc": utc(), "stage": name}) + "\n")
            stages.flush()

        try:
            stage("source_checkpoint_and_protocol_preflight")
            result["identity_before"], protocol = verify()
            if sha(encoded(result["identity_before"])) != expected_identity:
                raise ValueError("Worker identities differ from supervisor preflight")
            blocks, evaluator = helpers()
            stage("fixed_gpu_runtime")
            result["runtime"] = blocks.runtime()
            result["runtime_verified"] = True
            import numpy as np
            inference = load_module("_comparison_frozen_inference", EVALUATOR / "inference.py", INFERENCE_SHA)
            policy, scale_bank, receipt = inference.load(CHECKPOINT, protocol)
            for key in ("manifest_sha256", "file_hashes"):
                if receipt[key] != result["identity_before"]["checkpoint"][key]:
                    raise ValueError("Checkpoint changed between preflight and numeric loading")
            result["checkpoint"] = receipt
            result["frozen_state_sha256_before"] = policy.digest()
            sys.path.insert(0, str(evaluator.BUNDLE / "core_v4"))
            from thrusterchain import environment as core
            from thrusterchain.datasets import inference_evaluation_guard
            simulation, scoring = own_module("simulation"), own_module("scoring")
            stage("new_queries_candidates_and_labelblind_scores")
            with inference_evaluation_guard(frozen_guard):
                queries = simulation.build_queries(core, policy, protocol)
                result["partial_ledger"] = queries.ledger
                banks = simulation.candidate_bank(policy, queries, protocol)
                result["partial_ledger"] = {**queries.ledger, **banks.ledger}
                observations = [query["observation"] for query in queries]
                prelabel = scoring.score_candidates(policy, observations, banks, scale_bank, protocol)
                verify_scores(prelabel, protocol)
                policy.assert_unchanged()
            result["prelabel_sha256"] = write(output / "prelabel_scores_and_selections.json", prelabel)
            with (output / "queries_and_actions.npz").open("xb") as stream:
                arrays = {}
                for i, (query, bank) in enumerate(zip(queries, banks)):
                    for key in ("state", "observation"):
                        arrays[f"{key}_{i}"] = query[key]
                    arrays[f"actions_{i}"] = bank
                    arrays[f"time_{i}"] = np.asarray(query["t"], dtype=np.int32)
                    arrays[f"episode_seed_{i}"] = np.asarray(query["episode_seed"], dtype=np.int64)
                    arrays[f"prefix_actions_{i}"] = np.asarray(query["prefix_actions"], dtype=np.float32)
                np.savez_compressed(stream, **arrays)
            result["query_bank_sha256"] = sha((output / "queries_and_actions.npz").read_bytes())
            stage("prelabel_and_query_artifacts_sealed")
            with inference_evaluation_guard(frozen_guard):
                values, ledger = simulation.continuation_values(core, policy, queries, banks, protocol)
                policy.assert_unchanged()
            result["ledger"] = ledger
            if (frozen_guard["forbidden_api_calls"] != 0 or ledger["learner_optimizer_updates"] != 0
                    or ledger["calibrator_updates"] != 0 or ledger["confirmation_starts_generated"] != 0):
                raise ValueError("Frozen evaluation violated its zero-update or confirmation boundary")
            if (sha(encoded(prelabel)) != result["prelabel_sha256"]
                    or sha((output / "prelabel_scores_and_selections.json").read_bytes()) != result["prelabel_sha256"]
                    or sha((output / "queries_and_actions.npz").read_bytes()) != result["query_bank_sha256"]):
                raise ValueError("Prelabel decisions or query artifacts changed after labels were generated")
            result["conditional_values_sha256"] = write(output / "conditional_values.json",
                {"values": [np.asarray(value).tolist() for value in values], "ledger": ledger,
                 "conditional_monte_carlo_error": 0,
                 "reason": "Deterministic full-state continuation under the same frozen actor",
                 "numerical_error_bound": "Not certified; predeclared numerical tie band applies"})
            stage("analyze_all_frozen_selectors_without_selection")
            verify_scores(prelabel, protocol)
            result["analysis"] = scoring.analyze(prelabel, values, protocol)
            if result["analysis"].get("best_kappa_selected") is not False:
                raise ValueError("Development analysis must not automatically choose kappa")
            result["status"] = "completed_development_diagnostic"
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
            if hasattr(error, "ledger"):
                result["partial_ledger"] = error.ledger
        finally:
            stage("frozen_state_and_source_postchecks")
            result["frozen_guard"] = frozen_guard
            if policy is not None:
                try:
                    policy.assert_unchanged()
                    result["frozen_state_sha256_after"] = policy.digest()
                    result["frozen_state_verified"] = (result.get("frozen_state_sha256_before") == result["frozen_state_sha256_after"]
                                                        and frozen_guard["forbidden_api_calls"] == 0)
                    if not result["frozen_state_verified"]:
                        raise ValueError("Frozen policy/reference state changed")
                except Exception as error:
                    result["status"] = "blocked_or_failed"
                    result["frozen_state_error"] = str(error)
            try:
                result["identity_after"], _ = verify()
                if result.get("identity_before") != result["identity_after"]:
                    raise ValueError("Source, protocol or checkpoint identities changed during the run")
            except Exception as error:
                result["status"] = "blocked_or_failed"
                result["postcheck_error"] = str(error)
            result["finished_utc"] = utc()
            write(output / "diagnostic.json", result)
    return 0 if result["status"] == "completed_development_diagnostic" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--preflight", action="store_true", help="Stdlib provenance checks only; default")
    modes.add_argument("--execute", action="store_true", help="Run only the frozen new-development comparison")
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-identity-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        if not args.allow_outside_upstream_constraints or not args.expected_identity_sha256 or args.preflight:
            parser.error("Worker requires provisional-runtime acknowledgement and supervisor identity")
        output = args.worker_output.resolve()
        if output.parent != ROOT or not output.name.startswith("results_") or not output.is_dir():
            parser.error("Worker output must be the new local supervisor directory")
        return worker(output, args.expected_identity_sha256)
    if args.execute and not args.allow_outside_upstream_constraints:
        parser.error("Execution requires explicit acknowledgement of the provisional upstream-exceeding runtime")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"started_utc": utc(), "mode": "execute" if args.execute else "preflight", "status": "blocked_or_failed",
              "runtime_verified": False, "frozen_state_verified": False, "timeout_seconds": TIMEOUT_SECONDS,
              "scientific_validation_status": "incomplete_design_review_required",
              "claim_scope": "Development comparison only; no kappa choice, confirmation or production normalization approval"}
    try:
        result["identity_before"], _ = verify()
        if not args.execute:
            result.update(status="preflight_passed", scientific_execution=False)
        else:
            if os.environ.get("XLA_FLAGS", "") not in ("", XLA):
                raise ValueError("Conflicting XLA_FLAGS; no override attempted")
            env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest", XLA_FLAGS=XLA,
                       XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-output", str(output),
                       "--allow-outside-upstream-constraints", "--expected-identity-sha256", sha(encoded(result["identity_before"]))]
            print("Starting frozen new-development comparison; output: " + str(output), flush=True)
            start = time.monotonic()
            with (output / "diagnostic.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    result["exit_code"] = process.wait(timeout=TIMEOUT_SECONDS)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                    result["interrupted"] = isinstance(error, KeyboardInterrupt)
                    process.kill()  # Only the newly created bounded evaluator child.
                    result["exit_code"] = process.wait()
            result["elapsed_seconds"] = time.monotonic() - start
            if (output / "diagnostic.json").is_file():
                child_raw = (output / "diagnostic.json").read_bytes()
                child = json.loads(child_raw)
                result["diagnostic_sha256"] = sha(child_raw)
                result["worker_status"] = child.get("status")
                for key in ("runtime_verified", "frozen_state_verified", "runtime", "ledger", "frozen_guard",
                            "prelabel_sha256", "query_bank_sha256", "conditional_values_sha256", "error", "postcheck_error"):
                    if key in child:
                        result[key] = child[key]
                if (result["exit_code"] == 0 and child.get("status") == "completed_development_diagnostic"
                        and child.get("identity_before") == result["identity_before"] == child.get("identity_after")
                        and child.get("runtime_verified") is True and child.get("frozen_state_verified") is True
                        and not result.get("timed_out") and not result.get("interrupted")):
                    for key, filename in (("prelabel_sha256", "prelabel_scores_and_selections.json"),
                                          ("query_bank_sha256", "queries_and_actions.npz"),
                                          ("conditional_values_sha256", "conditional_values.json")):
                        if sha((output / filename).read_bytes()) != child[key]:
                            raise ValueError("Completed child artifact hash mismatch: " + filename)
                    result["status"] = "completed_development_diagnostic"
            else:
                result["worker_outcome"] = "No completed diagnostic receipt; inspect stage journal and log"
    except BaseException as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            result["identity_after"], _ = verify()
            if result.get("identity_before") != result["identity_after"]:
                raise ValueError("Source, protocol or checkpoint identities changed during supervision")
        except Exception as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = str(error)
        result["finished_utc"] = utc()
        write(output / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json"),
                      "runtime_verified": result["runtime_verified"], "frozen_state_verified": result["frozen_state_verified"]}), flush=True)
    return 0 if result["status"] in ("preflight_passed", "completed_development_diagnostic") else 1


if __name__ == "__main__":
    raise SystemExit(main())
