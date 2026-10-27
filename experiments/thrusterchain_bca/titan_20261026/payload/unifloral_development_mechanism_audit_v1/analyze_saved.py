"""Saved-development mechanism audit. No models, new states, or confirmation inputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
TIMEOUT_SECONDS = 60
PROTOCOL_SHA256 = "e2d3f63ad6f8e66cb1a0041ee4faa4955d6e551dc4cd6711478c66f79c8d200e"
AMENDMENT_SHA256 = "6f8cee93477cf02dc0fabbfaec952ef213aabaf271ef7e6b1fc2f7e1809a5aa6"
EFFECTIVE_PROTOCOL_SHA256 = "593da359683d7bb858142d7194a2be9c44f0ece5e9e62093fe7963deb38c55d2"
FROZEN_STATE_SHA256 = "72ec7a4f7b5568834c5f102f0862cf8e6031dcc32f328083821f2fc7c66ab26c"
CHECKPOINT_SHA256 = "a54856669e51fd86234c9e9624c62135cd111b0bd7fad07cc1c4fb8e252c899c"
CHECKPOINT_ROLES = {
    "parameters": "b8f2d15f54aab85e64583bc253c069aab9eed91988aec4ac064f87522736d90f",
    "scale_bank": "6e6b95b88ef97d09f38166dabbc32739d9a495e667df80d0d6f4f66be76496e8",
    "training_receipt": "30315a0d76a6a5b7fccaa7afb6d6dd827d8f5e75abfc17c9a3e43eb4ead19ddf",
    "normalization_receipt": "4f6be8817906c2ed99a9e2ef1355c43d9ce5640ec3984e104625b85d25ba5dc3",
    "calibration_receipt": "bbfc1826a6c4095b1948eacf5daec8298706f5d09b5afbe8642701104c607a3f",
}
BATCHES = (
    {"name": "development_1", "directory": "unifloral_frozen_usefulness_v1_seedfix/results_20261024T234619_729786Z",
     "query_seed": 2026102400, "candidate_seed": 2026102342,
     "protocol_canonical_sha256": "a482f397df5f9106465512ca59852088262bf623291f230edfea7b36b28d4982",
     "files": {
         "report.json": "6de2bdb6fb7e3365710771fe03990420824554e79cef83cbdd1da72a53a48979",
         "diagnostic.json": "9f738163d56926d128624f8ed1d2f6f31203a8b389bf2567d4cfbe2f2925cc57",
         "prelabel_scores_and_selections.json": "a337e2038634417cd33fb934cb5d1a97a8ef90dfaeb303016b200a596977da07",
         "conditional_values.json": "fb391bfd31e2e98fca081bd57a47c136f4f3eeb76e0b0b765aaa4b312e5ebd30",
         "queries_and_actions.npz": "9464e6c2209ede701f26bd7fda157b8d461eabc50a1800361f0c6b4f1234c929"}},
    {"name": "development_2", "directory": "unifloral_frozen_scale_comparison_v1/results_20261026T172507_062277Z",
     "query_seed": 2026102700, "candidate_seed": 2026102807,
     "protocol_canonical_sha256": "130aafcae9db21279c190b7be1ebb59ddb7ce754da2d0dcc2b49c14d2e67f9d5",
     "files": {
         "report.json": "8a19abdf5f74fcaa39a3c936cd91e29b1802309f09f1b35b6f7e681859c03bd5",
         "diagnostic.json": "62c671c9dc35ff35a5e8dad64c83f592f418fbb8e6a4c28a2dff18b89894a2b0",
         "prelabel_scores_and_selections.json": "16f84c403326d12b41b182968a1741601dc31ae8da31f145e43234d11082dd84",
         "conditional_values.json": "c7686e6e4c292a0644479e35eb2d607e0b43078a8bd5d1735e460704b7ac6955",
         "queries_and_actions.npz": "1a9940f20d2210c06987ec648d5696952edfcd1e26169c5301c5ce898ac0bccc"}},
)
INPUT_ALLOWLIST = {batch["directory"] + "/" + name: digest for batch in BATCHES for name, digest in batch["files"].items()}
sys.dont_write_bytecode = True


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def utc():
    return datetime.now(timezone.utc).isoformat()


def parse_json(raw):
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


def safe_bytes(base, relative):
    base = Path(base).resolve()
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Expected a safe relative file")
    target = base / relative
    if target.resolve() != target.absolute() or not target.is_file():
        raise ValueError("Missing or redirected source: " + relative)
    for component in (target, *target.parents):
        if component == base:
            break
        if component.is_symlink():
            raise ValueError("Symlinked input is forbidden")
    return target.read_bytes()


def input_bytes(relative):
    if relative not in INPUT_ALLOWLIST:
        raise ValueError("Input is outside the exact two-development-batch allowlist")
    raw = safe_bytes(BASE, relative)
    if sha(raw) != INPUT_ALLOWLIST[relative]:
        raise ValueError("Pinned saved development input changed: " + relative)
    return raw


def verify_ledger(ledger, spec):
    expected = {"phase": "development", "status": "complete", "query_count": 64,
                "query_seed": spec["query_seed"], "candidate_seed": spec["candidate_seed"],
                "query_times": [0, 25, 50, 75], "gamma": 0.98, "horizon": 100,
                "confirmation_starts_generated": 0, "learner_optimizer_updates": 0, "calibrator_updates": 0,
                "transition_steps_attempted": 38504, "transition_steps_completed": 38504,
                "prefix_steps_completed": 2400, "verification_steps_completed": 104,
                "continuation_steps_completed": 36000, "continuations_completed": 576,
                "monte_carlo_replications": 0, "deterministic_continuations_per_candidate": 1,
                "conditional_monte_carlo_standard_error": 0.0,
                "frozen_policy_unchanged": True, "candidate_selection_used_simulator_labels": False}
    if any(type(ledger.get(k)) is not type(v) or ledger[k] != v for k, v in expected.items()):
        raise ValueError("Saved development ledger violates its frozen phase, seeds, or accounting")


def verify_metadata(report, diagnostic, spec):
    for value in (report, diagnostic):
        if value.get("status") != "completed_development_diagnostic" or value.get("identity_before") != value.get("identity_after"):
            raise ValueError("Saved development did not complete with stable source identities")
    if report.get("exit_code") != 0 or not report.get("identity_before") or not diagnostic.get("identity_before"):
        raise ValueError("Missing completed source/runtime receipt")
    analysis = diagnostic.get("analysis", {})
    if analysis.get("phase") != "development" or analysis.get("confirmation_data_accessed") is not False:
        raise ValueError("Analysis was not explicitly isolated from confirmation")
    if diagnostic.get("frozen_guard", {}).get("forbidden_api_calls") != 0:
        raise ValueError("Saved frozen guard failed")
    receipt = diagnostic.get("checkpoint", {})
    if (receipt.get("manifest_sha256") != CHECKPOINT_SHA256 or receipt.get("file_hashes") != CHECKPOINT_ROLES
            or receipt.get("frozen_state_sha256") != FROZEN_STATE_SHA256):
        raise ValueError("Saved checkpoint/frozen identity differs from the fixed shared export")
    if spec["name"] == "development_2":
        if (diagnostic.get("runtime_verified") is not True or diagnostic.get("frozen_state_verified") is not True
                or diagnostic.get("frozen_state_sha256_before") != FROZEN_STATE_SHA256
                or diagnostic.get("frozen_state_sha256_after") != FROZEN_STATE_SHA256):
            raise ValueError("Second development runtime/frozen-state checks did not pass")
    elif diagnostic.get("optimizer_updates") != 0:
        raise ValueError("First development receipt did not certify zero updates")
    verify_ledger(diagnostic.get("ledger", {}), spec)
    if (diagnostic.get("prelabel_sha256") != spec["files"]["prelabel_scores_and_selections.json"]
            or diagnostic.get("query_bank_sha256") != spec["files"]["queries_and_actions.npz"]):
        raise ValueError("Saved diagnostic/prelabel/query hash links differ")
    return {"name": spec["name"], "phase": "development", "query_seed_start": spec["query_seed"],
            "query_count": 64, "candidate_seed": spec["candidate_seed"], "frozen_state_sha256": FROZEN_STATE_SHA256,
            "checkpoint_manifest_sha256": CHECKPOINT_SHA256, "confirmation_data_accessed": False,
            "source_identity_sha256": canonical(diagnostic["identity_before"])}


def verify():
    """Hash and metadata only; no numeric signal/outcome inspection or model imports."""
    manifest_raw = safe_bytes(ROOT, "prepared_files.json")
    manifest = parse_json(manifest_raw)
    required = {"analyze_saved.py", "test_inputs.py", "metrics.py", "protocol.json",
                "DEFINITIONS_FROZEN.json", "PROTOCOL_AMENDMENT_01.json", "AMENDMENT_FROZEN.json"}
    files = manifest.get("files", {})
    if not required.issubset(files):
        raise ValueError("Prepared manifest lacks required authored files")
    for name, digest in files.items():
        if sha(safe_bytes(ROOT, name)) != digest:
            raise ValueError("Prepared audit source changed: " + name)
    protocol_raw = safe_bytes(ROOT, "protocol.json")
    if sha(protocol_raw) != PROTOCOL_SHA256:
        raise ValueError("Protocol must be frozen before inspecting numeric development detail")
    base_protocol = parse_json(protocol_raw)
    definitions_raw = safe_bytes(ROOT, "DEFINITIONS_FROZEN.json")
    definitions = parse_json(definitions_raw)
    amendment_raw = safe_bytes(ROOT, "PROTOCOL_AMENDMENT_01.json")
    amendment = parse_json(amendment_raw)
    amendment_freeze_raw = safe_bytes(ROOT, "AMENDMENT_FROZEN.json")
    amendment_freeze = parse_json(amendment_freeze_raw)
    if (definitions.get("protocol_sha256") != PROTOCOL_SHA256
            or definitions.get("confirmation_outcomes_read_for_this_audit") is not False
            or definitions.get("new_simulator_or_gpu_work") is not False
            or sha(amendment_raw) != AMENDMENT_SHA256 or amendment.get("base_protocol_sha256") != PROTOCOL_SHA256
            or amendment.get("prior_definition_preserved") is not True
            or amendment.get("outcomes_inspected_since_initial_freeze") is not False
            or amendment_freeze.get("base_protocol_sha256") != PROTOCOL_SHA256
            or amendment_freeze.get("amendment_sha256") != AMENDMENT_SHA256
            or amendment_freeze.get("effective_protocol_sha256") != EFFECTIVE_PROTOCOL_SHA256
            or amendment_freeze.get("detailed_audit_inputs_loaded") is not False
            or amendment_freeze.get("confirmation_files_read") != 0):
        raise ValueError("Definition/amendment freeze receipts are inconsistent")
    if not isinstance(amendment.get("additions"), dict) or set(amendment["additions"]) & set(base_protocol):
        raise ValueError("The frozen amendment may only add definitions")
    protocol = {**base_protocol, **amendment["additions"]}
    if (canonical(protocol) != EFFECTIVE_PROTOCOL_SHA256
            or protocol.get("allowed_development_runs") != [spec["directory"] for spec in BATCHES]
            or protocol.get("phase") != "exploratory_saved_development_only"
            or protocol.get("cpu_timeout_seconds") != TIMEOUT_SECONDS):
        raise ValueError("Effective frozen mechanism protocol differs")
    identities = []
    for spec in BATCHES:
        raw = {name: input_bytes(spec["directory"] + "/" + name) for name in spec["files"]}
        identities.append(verify_metadata(parse_json(raw["report.json"]), parse_json(raw["diagnostic.json"]), spec))
    return {"prepared_manifest_sha256": sha(manifest_raw), "verified_source_files": len(files),
            "protocol_sha256": sha(protocol_raw), "effective_protocol_sha256": EFFECTIVE_PROTOCOL_SHA256,
            "freeze_hashes": {"DEFINITIONS_FROZEN.json": sha(definitions_raw),
                              "PROTOCOL_AMENDMENT_01.json": sha(amendment_raw),
                              "AMENDMENT_FROZEN.json": sha(amendment_freeze_raw)},
            "input_hashes": INPUT_ALLOWLIST,
            "development_metadata": identities,
            "scope": "Exactly two saved development result directories; no confirmation paths or model state loaded"}, protocol


def validate_batch(spec, prelabel, labels, arrays, *, group_offset):
    """Validate authenticated numeric inputs. Also accepts synthetic fixtures in tests."""
    import numpy as np
    if (prelabel.get("schema") != "unifloral_frozen_usefulness_scores.v1" or prelabel.get("phase") != "development"
            or prelabel.get("selections_frozen_before_truevalues") is not True
            or prelabel.get("protocol_sha256") != spec["protocol_canonical_sha256"]
            or prelabel.get("prelabel_signature") != canonical({k: v for k, v in prelabel.items() if k != "prelabel_signature"})):
        raise ValueError("Saved score/protocol seal or development-only phase is invalid")
    verify_ledger(labels.get("ledger", {}), spec)
    if labels.get("conditional_monte_carlo_error") != 0:
        raise ValueError("Expected saved deterministic continuation values")
    queries, values = prelabel.get("queries", []), labels.get("values", [])
    method = prelabel.get("methods", {}).get("raw_q", {})
    if (len(queries) != 64 or len(values) != 64 or method.get("available") is not True
            or method.get("family") != "raw_q" or method.get("kappa_penalty") is not None):
        raise ValueError("Missing fixed 64-state raw-Q score bank")
    for key in ("scores", "selected_indices", "selection_details"):
        if len(method.get(key, [])) != 64:
            raise ValueError("Raw-Q method rows are misaligned")
    expected_keys = {prefix + str(i) for i in range(64) for prefix in
                     ("state_", "observation_", "actions_", "time_", "episode_seed_", "prefix_actions_")}
    if set(arrays) != expected_keys:
        raise ValueError("Unexpected or missing saved query-bank array")

    def numeric(value, shape, name, dtype=None):
        data = np.asarray(value)
        if (data.shape != shape or data.dtype.kind not in "fiu" or not np.all(np.isfinite(data))
                or (dtype is not None and data.dtype != np.dtype(dtype))):
            raise ValueError("Invalid finite shape/dtype for " + name)
        return data

    records = []
    for i, query in enumerate(queries):
        t = (0, 25, 50, 75)[i % 4]
        if (type(query.get("index")) is not int or type(query.get("episode_group_id")) is not int
                or query["index"] != i or query["episode_group_id"] != i):
            raise ValueError("Query ordering/group identity changed")
        if query.get("width_usable") is not True or query.get("width_valid") is not True:
            raise ValueError("Invalid frozen uncertainty flags; do not replace uncertainty with zero")
        state = numeric(arrays["state_" + str(i)], (13,), "physical state", np.float64)
        observation = numeric(arrays["observation_" + str(i)], (14,), "observation", np.float32)
        actions = numeric(arrays["actions_" + str(i)], (9, 4), "actions", np.float32)
        prefix = numeric(arrays["prefix_actions_" + str(i)], (t, 4), "prefix", np.float32)
        saved_time = numeric(arrays["time_" + str(i)], (), "time", np.int32)
        seed = numeric(arrays["episode_seed_" + str(i)], (), "episode seed", np.int64)
        if int(saved_time) != t or int(seed) != spec["query_seed"] + i:
            raise ValueError("Saved seed/time differs from the allowed development batch")
        if not np.array_equal(observation, np.concatenate((state, [(100 - t) / 100.0])).astype(np.float32)):
            raise ValueError("Observation/state/time alignment differs")
        if (not np.array_equal(numeric(query["observation"], (14,), "JSON observation"), observation)
                or not np.array_equal(numeric(query["actions"], (9, 4), "JSON actions"), actions)
                or np.any(np.abs(actions) > 1.0) or np.any(np.abs(prefix) > 1.0)
                or any(np.array_equal(actions[j], actions[k]) for j in range(9) for k in range(j))):
            raise ValueError("Saved JSON/NPZ action bank differs or has invalid deployed actions")
        q = numeric(query["q"], (9, 2), "twin Q").astype(np.float64)
        u = numeric(query["u"], (9,), "BCA width").astype(np.float64)
        disagreement = numeric(query["disagreement"], (9,), "disagreement").astype(np.float64)
        distance = numeric(query["actor_distance"], (9,), "actor distance").astype(np.float64)
        returns = numeric(values[i], (9,), "saved continuation values").astype(np.float64)
        if np.any(u < 0) or np.any(disagreement < 0) or np.any(distance < 0):
            raise ValueError("Saved nonnegative uncertainty/control signal is negative")
        action64 = actions.astype(np.float64)
        if (not np.allclose(disagreement, np.abs(q[:, 0] - q[:, 1]), rtol=1e-6, atol=1e-7)
                or not np.allclose(distance, np.mean((action64 - action64[0]) ** 2, axis=1), rtol=1e-6, atol=1e-7)):
            raise ValueError("Saved disagreement/actor-distance arithmetic is inconsistent")
        scores = numeric(method["scores"][i], (9,), "raw Q scores").astype(np.float64)
        if not np.array_equal(scores, q[:, 0]):
            raise ValueError("Saved raw-Q score is not Q1")
        maximum = float(scores.max())
        near = np.flatnonzero(np.abs(scores - maximum) <= 1e-6 + 1e-6 * abs(maximum)).tolist()
        selected = method["selected_indices"][i]
        detail = method["selection_details"][i]
        if (type(selected) is not int or selected != near[0] or detail.get("index") != selected
                or detail.get("exact_argmax_index") != int(np.argmax(scores))
                or detail.get("near_best_indices") != near or detail.get("maximum_score") != maximum
                or detail.get("selected_score") != float(scores[selected])
                or detail.get("near_tie") is not (len(near) > 1)
                or detail.get("inconclusive_selection") is not (len(near) > 1)
                or detail.get("top_two_gap") != float(np.sort(scores)[-1] - np.sort(scores)[-2])):
            raise ValueError("Saved raw-Q selection does not follow its sealed first-near-best rule")
        records.append({"batch": spec["name"], "local_group_id": i, "group_id": group_offset + i,
                        "time": t, "episode_seed": int(seed), "actions": actions, "observation": observation,
                        "q1": q[:, 0], "q2": q[:, 1], "u": u, "disagreement": disagreement,
                        "actor_distance": distance, "values": returns, "raw_q_index": selected, "actor_index": 0,
                        "raw_q_near_tie": detail["near_tie"]})
    return {"name": spec["name"], "queries": records}


def load_batches():
    import numpy as np
    batches = []
    for index, spec in enumerate(BATCHES):
        raw = {name: input_bytes(spec["directory"] + "/" + name) for name in
               ("prelabel_scores_and_selections.json", "conditional_values.json", "queries_and_actions.npz")}
        with np.load(io.BytesIO(raw["queries_and_actions.npz"]), allow_pickle=False) as archive:
            if len(archive.files) != len(set(archive.files)):
                raise ValueError("Duplicate query-bank array names")
            arrays = {name: archive[name] for name in archive.files}
        batches.append(validate_batch(spec, parse_json(raw["prelabel_scores_and_selections.json"]),
                                      parse_json(raw["conditional_values.json"]), arrays, group_offset=64 * index))
    seeds = [query["episode_seed"] for batch in batches for query in batch["queries"]]
    groups = [query["group_id"] for batch in batches for query in batch["queries"]]
    if (len(set(seeds)) != 128 or groups != list(range(128)) or any(2026102500 <= seed <= 2026102563 for seed in seeds)):
        raise ValueError("Development batches overlap or contain a reserved confirmation seed")
    return batches


def metrics_module():
    manifest = parse_json(safe_bytes(ROOT, "prepared_files.json"))
    raw = safe_bytes(ROOT, "metrics.py")
    if sha(raw) != manifest["files"]["metrics.py"]:
        raise ValueError("Metric source changed before loading")
    spec = importlib.util.spec_from_file_location("_saved_development_mechanism_metrics", ROOT / "metrics.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(ROOT / "metrics.py"), "exec"), module.__dict__)
    return module


def worker(output, expected_identity):
    result = {"started_utc": utc(), "status": "blocked_or_failed", "scientific_outcome_valid": False,
              "scope": "Saved development mechanism description only; no new actions, values, selector, or efficacy claim",
              "new_work": {"simulator_steps": 0, "policy_forward_calls": 0, "optimizer_updates": 0,
                           "calibrator_updates": 0, "candidate_actions_generated": 0, "scale_refits": 0,
                           "confirmation_files_read": False, "strength_sweeps": 0}}
    with (output / "stages.jsonl").open("x", encoding="utf-8") as stages:
        def stage(name):
            stages.write(json.dumps({"utc": utc(), "stage": name}) + "\n")
            stages.flush()
        try:
            stage("verify_pinned_inputs_and_protocol")
            result["identity_before"], protocol = verify()
            if canonical(result["identity_before"]) != expected_identity:
                raise ValueError("Worker input/source identity differs from supervisor")
            import numpy as np
            if np.__version__ != "2.4.1" or platform.system() != "Windows":
                raise ValueError("This CPU saved-data audit requires the prepared Windows NumPy2.4.1 runtime")
            result["runtime"] = {"python": sys.version, "numpy": np.__version__, "platform": platform.system(), "executable": sys.executable}
            stage("validate_saved_arrays_seals_and_alignment")
            batches = load_batches()
            result["validated_inputs"] = {"batches": 2, "states": 128, "saved_candidate_rows": 1152,
                                          "groups": 128, "same_checkpoint": True, "disjoint_development_seeds": True}
            stage("compute_frozen_mechanism_descriptives")
            result["analysis"] = metrics_module().analyze(batches, protocol)
            required_scope = {"status": "pass", "claim_scope": "exploratory_saved_development_mechanism_only",
                              "full_training_hold": True, "automatic_training_approval": False,
                              "method_or_strength_selected": False, "new_model_fitted": False,
                              "bootstrap_performed": False, "confirmation_files_read": 0,
                              "new_simulator_calls": 0, "new_gpu_work": 0}
            if any(type(result["analysis"].get(key)) is not type(value) or result["analysis"][key] != value
                   for key, value in required_scope.items()):
                raise ValueError("Mechanism output exceeded its frozen saved-data descriptive scope")
            result["status"] = "completed_saved_development_mechanism_audit"
        except BaseException as error:
            result["status"] = "blocked_or_failed"
            result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            stage("input_source_protocol_postchecks")
            try:
                result["identity_after"], _ = verify()
                if result.get("identity_before") != result["identity_after"]:
                    raise ValueError("Saved inputs, analysis source, or protocol changed")
            except Exception as error:
                result["status"] = "blocked_or_failed"
                result["postcheck_error"] = str(error)
            result["scientific_outcome_valid"] = result["status"] == "completed_saved_development_mechanism_audit"
            if not result["scientific_outcome_valid"] and "analysis" in result:
                result["unvalidated_analysis"] = result.pop("analysis")
            result["finished_utc"] = utc()
            write(output / "audit.json", result)
    return 0 if result["scientific_outcome_valid"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="Hash/metadata checks only; default")
    mode.add_argument("--execute", action="store_true", help="Analyze only the two pinned saved development batches on CPU")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--expected-identity", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        output = args.worker_output.resolve()
        if (args.execute or args.preflight or not args.expected_identity or output.parent != ROOT
                or not output.name.startswith("results_") or not output.is_dir()):
            parser.error("Worker requires an exclusive local supervisor output and its identity")
        return worker(output, args.expected_identity)
    if args.expected_identity:
        parser.error("Worker identity without output is invalid")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"started_utc": utc(), "mode": "execute" if args.execute else "preflight", "status": "blocked_or_failed",
              "scientific_outcome_valid": False, "cpu_worker_timeout_seconds": TIMEOUT_SECONDS,
              "timeout_scope": "Wall-clock child wait capped at60seconds; supervisor input hashing and postchecks are outside that cap"}
    try:
        result["identity_before"], _ = verify()
        if not args.execute:
            result["status"] = "preflight_passed"
        else:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker-output", str(output),
                       "--expected-identity", canonical(result["identity_before"])]
            start = time.monotonic()
            with (output / "audit.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    result["exit_code"] = process.wait(timeout=TIMEOUT_SECONDS)
                except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                    result["timed_out"] = isinstance(error, subprocess.TimeoutExpired)
                    result["interrupted"] = isinstance(error, KeyboardInterrupt)
                    process.kill()  # Only this newly created bounded CPU analysis child.
                    result["exit_code"] = process.wait()
            result["elapsed_seconds"] = time.monotonic() - start
            if (output / "audit.json").is_file():
                raw = (output / "audit.json").read_bytes()
                child = parse_json(raw)
                result["audit_sha256"] = sha(raw)
                for key in ("runtime", "validated_inputs", "new_work", "error", "postcheck_error"):
                    if key in child:
                        result[key] = child[key]
                if (result["exit_code"] == 0 and child.get("status") == "completed_saved_development_mechanism_audit"
                        and child.get("scientific_outcome_valid") is True
                        and child.get("identity_before") == result["identity_before"] == child.get("identity_after")
                        and not result.get("timed_out") and not result.get("interrupted")):
                    result["status"] = child["status"]
            else:
                result["worker_outcome"] = "No complete audit receipt; preserve log and stages without reporting an analysis result"
    except BaseException as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        try:
            result["identity_after"], _ = verify()
            if result.get("identity_before") != result["identity_after"]:
                raise ValueError("Saved inputs, sources or protocol changed during supervision")
        except Exception as error:
            result["status"] = "blocked_or_failed"
            result["postcheck_error"] = str(error)
        result["scientific_outcome_valid"] = result["status"] == "completed_saved_development_mechanism_audit"
        result["finished_utc"] = utc()
        write(output / "report.json", result)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json")}))
    return 0 if result["status"] in ("preflight_passed", "completed_saved_development_mechanism_audit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
