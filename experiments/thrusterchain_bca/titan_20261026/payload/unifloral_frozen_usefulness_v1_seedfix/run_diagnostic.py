"""Bounded frozen-policy evaluator. Default is provenance preflight, not execution."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / "unifloral_td3bc_validation_v1"
STAGE2 = ROOT.parent / "unifloral_bca_pipeline_validation_v1"
BUNDLE_WINDOWS = Path(r"C:\Users\David Bayha\Downloads\titan_transfer_20261022_v1\titan_transfer_20261022_v1")
BUNDLE_LINUX = Path("/mnt/c/Users/David Bayha/Downloads/titan_transfer_20261022_v1/titan_transfer_20261022_v1")
BUNDLE = BUNDLE_WINDOWS if os.name == "nt" else BUNDLE_LINUX
BUNDLE_SHA = "3443280fd4340ba64aef4d9601136257014bf2a6c61ca8c890eba06afa76690a"
PINNED_MANIFESTS = {STAGE1: "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516",
                    STAGE2: "6eb91c5bcc6c2edddc1ab93c19424307e12a3c8363bf9063fa4bbe2664180478"}
XLA_FLAGS = "--xla_gpu_triton_gemm_any=True"
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def identities():
    checked = {}
    for root in (ROOT, STAGE1, STAGE2):
        manifest = root / "prepared_files.json"
        if root in PINNED_MANIFESTS and sha(manifest) != PINNED_MANIFESTS[root]:
            raise ValueError("Previously verified suite manifest changed: " + str(root))
        files = json.loads(manifest.read_text(encoding="utf-8"))["files"]
        for name, expected in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError("Prepared source changed: " + str(path))
        checked[root.name] = {"manifest_sha256": sha(manifest), "verified_files": len(files)}
    if sha(BUNDLE / "BUNDLE_MANIFEST.json") != BUNDLE_SHA:
        raise ValueError("Transfer bundle manifest changed")
    bundle = json.loads((BUNDLE / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    core_files = {name: expected for name, expected in bundle.items() if name.startswith("core_v4/")}
    if not core_files:
        raise ValueError("No pinned core source files")
    for name, expected in core_files.items():
        path = (BUNDLE / name).resolve()
        if not path.is_relative_to(BUNDLE) or sha(path) != expected:
            raise ValueError("Pinned core source changed: " + name)
    checked["core"] = {"bundle_manifest_sha256": BUNDLE_SHA, "verified_files": len(core_files)}
    return checked


def protocol_value():
    value = json.loads((ROOT / "protocol.json").read_text(encoding="utf-8"))
    if value["phase"] != "development" or value["query_count"] != 64 or value["max_transition_steps"] != 60000:
        raise ValueError("Only the fixed development budget is implemented")
    if value["gamma"] != 0.98 or value["horizon"] != 100 or value["learner_and_calibrator_updates"] != 0:
        raise ValueError("Matched finite task and zero-update contract required")
    return value


def worker(output, checkpoint):
    result = {"started_utc": utc(), "status": "running", "optimizer_updates": 0}
    try:
        result["identity_before"] = identities()
        protocol = protocol_value()
        result["protocol_sha256"] = sha(ROOT / "protocol.json")
        from inference import read_contract, load
        _, _, pre_import_receipt = read_contract(checkpoint, protocol)
        import jax
        import numpy as np
        result["runtime"] = {"python": sys.version, "executable": sys.executable,
            "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "precision": str(jax.config.jax_default_matmul_precision),
            "jax_disable_jit": bool(jax.config.jax_disable_jit), "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "packages": {name: importlib.metadata.version(name) for name in
                ("jax", "jaxlib", "numpy", "flax", "optax", "distrax", "scipy", "gymnasium")}}
        if (jax.default_backend() != "gpu" or str(jax.config.jax_default_matmul_precision) != "highest"
                or jax.config.jax_disable_jit or jax.config.jax_enable_x64 or os.environ.get("XLA_FLAGS") != XLA_FLAGS):
            raise RuntimeError("Runtime must match the validated GPU float32/highest-precision configuration")
        policy, scale_bank, checkpoint_receipt = load(checkpoint, protocol)
        if any(checkpoint_receipt[key] != pre_import_receipt[key]
               for key in ("manifest_sha256", "file_hashes")):
            raise RuntimeError("Checkpoint export changed between preflight and inference load")
        result["checkpoint"] = checkpoint_receipt
        # The scorer has no simulator reference; all simulator use starts here.
        sys.path.insert(0, str(BUNDLE / "core_v4"))
        from thrusterchain import environment as core
        from thrusterchain.datasets import inference_evaluation_guard
        from simulation import build_queries, candidate_bank, continuation_values
        from metrics import score_candidates, analyze
        frozen_guard = {"forbidden_api_calls": 0}
        with inference_evaluation_guard(frozen_guard):
            queries = build_queries(core, policy, protocol)
            result["partial_ledger"] = queries.ledger
            banks = candidate_bank(policy, queries, protocol)
            result["partial_ledger"] = {**queries.ledger, **banks.ledger}
            observations = [query["observation"] for query in queries]
            prelabel = score_candidates(policy, observations, banks, scale_bank, protocol)
            policy.assert_unchanged()
        # Persist decisions before any candidate-return labels are computed.
        write(output / "prelabel_scores_and_selections.json", prelabel)
        with (output / "queries_and_actions.npz").open("xb") as stream:
            arrays = {}
            for i, (query, bank) in enumerate(zip(queries, banks)):
                arrays[f"state_{i}"] = query["state"]
                arrays[f"observation_{i}"] = query["observation"]
                arrays[f"actions_{i}"] = bank
                arrays[f"time_{i}"] = np.asarray(query["t"], dtype=np.int32)
                arrays[f"episode_seed_{i}"] = np.asarray(query["episode_seed"], dtype=np.int64)
                arrays[f"prefix_actions_{i}"] = np.asarray(query["prefix_actions"], dtype=np.float32)
            np.savez_compressed(stream, **arrays)
        result["prelabel_sha256"] = sha(output / "prelabel_scores_and_selections.json")
        result["query_bank_sha256"] = sha(output / "queries_and_actions.npz")
        with inference_evaluation_guard(frozen_guard):
            values, ledger = continuation_values(core, policy, queries, banks, protocol)
            policy.assert_unchanged()
        if frozen_guard["forbidden_api_calls"] != 0:
            raise RuntimeError("A prohibited fitting/data-generation API was called")
        write(output / "conditional_values.json", {"values": [np.asarray(v).tolist() for v in values],
              "ledger": ledger, "conditional_monte_carlo_error": 0,
              "reason": "Deterministic source dynamics, exact full state/time, deterministic frozen actor",
              "numerical_error_bound": "Not certified; predeclared numerical tie band used"})
        result["analysis"] = analyze(prelabel, values, protocol)
        result["ledger"] = ledger
        result["frozen_guard"] = frozen_guard
        result["status"] = "completed_development_diagnostic"
        result["claim_scope"] = "Frozen first-action selection only; no policy-training efficacy, chosen kappa or Q1-error coverage guarantee"
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if hasattr(error, "ledger"):
            result["partial_ledger"] = error.ledger
    try:
        result["identity_after"] = identities()
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["postcheck_error"] = str(error)
    result["finished_utc"] = utc()
    write(output / "diagnostic.json", result)
    return 0 if result["status"] == "completed_development_diagnostic" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-manifest", type=Path, default=ROOT / "eligible_checkpoint.json")
    parser.add_argument("--preflight", action="store_true", help="Read/hash gates only; no JAX or simulator imports")
    parser.add_argument("--execute", action="store_true", help="Run the fixed development diagnostic only after all checkpoint gates pass")
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_output:
        if not args.allow_outside_upstream_constraints:
            parser.error("Worker requires the explicit provisional-runtime acknowledgement")
        output = args.worker_output.resolve()
        if output.parent != ROOT or not output.is_dir():
            parser.error("Worker output must be an existing local result folder")
        return worker(output, args.checkpoint_manifest.resolve())
    if args.execute and args.preflight:
        parser.error("Choose preflight or execute")
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    report = {"started_utc": utc(), "status": "pending", "optimizer_updates": 0,
              "execute_requested": args.execute, "meaningful_trained_result_ready": False,
              "independent_training_and_export_review_required": True}
    try:
        report["identity_before"] = identities()
        protocol = protocol_value()
        from inference import read_contract
        _, _, receipt = read_contract(args.checkpoint_manifest, protocol)
        report["checkpoint_preflight"] = receipt
        report["status"] = "preflight_passed_runtime_and_independent_receipt_review_still_required"
        if args.execute:
            if not args.allow_outside_upstream_constraints:
                raise ValueError("Explicit provisional-runtime acknowledgement required; JAX/NumPy exceed upstream bounds")
            if os.environ.get("XLA_FLAGS", "") not in ("", XLA_FLAGS):
                raise ValueError("Conflicting XLA_FLAGS; no override attempted")
            env = dict(os.environ, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                       XLA_FLAGS=XLA_FLAGS, XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONUNBUFFERED="1",
                       PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--worker-output", str(output), "--checkpoint-manifest", str(args.checkpoint_manifest.resolve()),
                       "--allow-outside-upstream-constraints"]
            start = time.monotonic()
            with (output / "diagnostic.log").open("x", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=600, check=False)
                    report["exit_code"] = completed.returncode
                except subprocess.TimeoutExpired:
                    report["exit_code"] = None
                    report["timed_out"] = True
            report["elapsed_seconds"] = time.monotonic() - start
            if report["exit_code"] == 0:
                child = json.loads((output / "diagnostic.json").read_text(encoding="utf-8"))
                if child["status"] != "completed_development_diagnostic" or any(
                    child["checkpoint"][key] != receipt[key] for key in ("manifest_sha256", "file_hashes")):
                    raise RuntimeError("Worker result or checkpoint identity differs from supervisor preflight")
            report["status"] = "completed_development_diagnostic" if report["exit_code"] == 0 else "blocked_or_failed"
            report["diagnostic_result_available"] = report["exit_code"] == 0
            report["interpretation"] = "Byte consistency is not independent attestation that the policy was trained or the export is truthful"
    except Exception as error:
        report["status"] = "blocked_or_failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        report["identity_after"] = identities()
    except Exception as error:
        report["status"] = "blocked_or_failed"
        report["postcheck_error"] = str(error)
    if report["status"] != "completed_development_diagnostic":
        report["meaningful_trained_result_ready"] = False
    report["finished_utc"] = utc()
    write(output / "report.json", report)
    print(json.dumps({"status": report["status"], "report": str(output / "report.json"),
                      "error": report.get("error"), "optimizer_updates": 0}), flush=True)
    return 0 if report["status"] in ("completed_development_diagnostic", "preflight_passed_runtime_and_independent_receipt_review_still_required") else 1


if __name__ == "__main__":
    raise SystemExit(main())
