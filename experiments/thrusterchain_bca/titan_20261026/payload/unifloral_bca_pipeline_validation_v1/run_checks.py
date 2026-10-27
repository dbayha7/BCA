"""Bounded actual-BCA pipeline diagnostics; no scientific imports in supervisor.

Uses the frozen stage-one UniFloral loader and candidate. This is a snapshot
composition test, not a production training loop or a new benchmark run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
STAGE1 = ROOT.parent / "unifloral_td3bc_validation_v1"
STAGE1_MANIFEST_SHA256 = "1e26f713822020275179845f30c8691467951745d6c7c829ff017b7045904516"
sys.dont_write_bytecode = True


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify_files(root, manifest_name):
    manifest = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    checked = {}
    for name, expected in manifest["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Manifest path escapes its root")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Hash mismatch: {target}")
        checked[name] = actual
    return {"passed": True, "files": checked}


def identities():
    manifest_sha = hashlib.sha256((STAGE1 / "prepared_files.json").read_bytes()).hexdigest()
    if manifest_sha != STAGE1_MANIFEST_SHA256:
        raise ValueError("Frozen stage-one manifest changed")
    return {"stage_one_manifest_sha256": manifest_sha,
            "stage_one_harness": verify_files(STAGE1, "prepared_files.json"),
            "stage_two": verify_files(ROOT, "prepared_files.json")}


def prior_runner():
    # Call only after identities(); imports stdlib, no native algorithm or JAX.
    spec = importlib.util.spec_from_file_location("stage_one_frozen_runner", STAGE1 / "run_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def versions(previous):
    result = previous.versions()
    try:
        result["scipy"] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        result["scipy"] = None
    return result


def worker(stage, output, override):
    result = {"stage": stage, "started_utc": utc(), "status": "running"}
    try:
        result["identity_before"] = identities()
        sys.path.insert(0, str(STAGE1))
        previous = prior_runner()
        if stage == "probe":
            return previous.worker("probe", output, override)
        from native_loader import load_native, verify_source
        result["native_source_before"] = verify_source(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
        result["packages"] = versions(previous)
        result["upstream_constraint_checks"] = previous.constraints(result["packages"])
        result["explicit_constraint_override"] = override
        if not all(c["satisfied"] for c in result["upstream_constraint_checks"]) and not override:
            raise RuntimeError("Explicit provisional-check override required for upstream constraint mismatch")
        import jax
        result["runtime"] = {
            "executable": sys.executable, "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "effective_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "environment": {key: os.environ.get(key) for key in (
                "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE")}}
        if jax.default_backend() != "gpu" or any(d.platform != "gpu" for d in jax.devices()):
            raise RuntimeError("GPU required; no CPU fallback")
        if str(jax.config.jax_default_matmul_precision) != "highest":
            raise RuntimeError("Highest precision is not effective")
        if bool(jax.config.jax_disable_jit) or bool(jax.config.jax_enable_x64):
            raise RuntimeError("Expected JIT enabled and float32 runtime, matching stage one")
        if os.environ.get("XLA_FLAGS") != previous.NATIVE_XLA_FLAGS:
            raise RuntimeError("Native XLA flags differ from declared setting")
        native = load_native(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
        result["native_slice"] = native._source_receipt
        from pipeline_checks import run
        result["pipeline"] = run(native, output, ROOT)
        if result["pipeline"].get("passed") is not True:
            raise RuntimeError("Actual BCA pipeline checks did not all pass")
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "fail"
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    try:
        result["identity_after"] = identities()
        from native_loader import verify_source
        result["native_source_after"] = verify_source(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
    except Exception as error:
        result["status"] = "fail"
        result["postcheck_error"] = {"type": type(error).__name__, "message": str(error)}
    result["finished_utc"] = utc()
    write_json(output / "pipeline.json", result)
    print(json.dumps({"stage": stage, "status": result["status"], "report": str(output / "pipeline.json")}))
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true")
    parser.add_argument("--worker", choices=("probe", "pipeline"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None or args.output.resolve().parent != ROOT or not args.output.is_dir():
            parser.error("Worker output must be an existing results directory within this suite")
        return worker(args.worker, args.output.resolve(), args.allow_outside_upstream_constraints)
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"schema_version": 1, "started_utc": utc(), "status": "pending",
              "scientific_validation_status": "incomplete_design_review_required",
              "scope": "Fixed synthetic replay; actual BCA pipeline on a frozen UniFloral snapshot",
              "limits": {"primary_critic_optimizer_updates": 0, "actor_optimizer_updates": 0,
                         "auxiliary_optimizer_attempts": 4, "refresh_calls": 3,
                         "environment_rollouts": 0, "real_dataset_generation": 0},
              "explicit_constraint_override": args.allow_outside_upstream_constraints,
              "design_decisions": "See DESIGN.md. Diagnostics do not authorize a production policy change.",
              "stages": []}
    report = output / "report.json"
    try:
        result["identity_before"] = identities()
        sys.path.insert(0, str(STAGE1))
        previous = prior_runner()
        from native_loader import verify_source
        result["native_source_before"] = verify_source(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
        result["packages"] = versions(previous)
        result["upstream_constraint_checks"] = previous.constraints(result["packages"])
        missing = [name for name, version in result["packages"].items() if version is None]
        if missing:
            raise RuntimeError("Missing installed packages: " + ", ".join(missing) + "; nothing installed")
        if not all(c["satisfied"] for c in result["upstream_constraint_checks"]) and not args.allow_outside_upstream_constraints:
            raise RuntimeError("Explicit provisional-check flag required for JAX/NumPy outside upstream bounds")
        flags = os.environ.get("XLA_FLAGS", "")
        result["inherited_xla_flags"] = flags
        if flags not in ("", previous.NATIVE_XLA_FLAGS):
            raise RuntimeError("Inherited XLA_FLAGS conflict; stopped without replacing them")
        env = dict(os.environ)
        env.update(JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                   XLA_FLAGS=previous.NATIVE_XLA_FLAGS, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        write_json(report, result)
        for stage, timeout in (("probe", 180), ("pipeline", 600)):
            print(f"Starting bounded {stage}; output: {output}", flush=True)
            start = time.monotonic()
            log_path = output / f"{stage}.log"
            command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                       "--worker", stage, "--output", str(output)]
            if args.allow_outside_upstream_constraints:
                command.append("--allow-outside-upstream-constraints")
            with log_path.open("x", encoding="utf-8") as log:
                try:
                    done = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                          stderr=subprocess.STDOUT, timeout=timeout, check=False)
                    receipt = {"stage": stage, "exit_code": done.returncode,
                               "elapsed_seconds": time.monotonic() - start, "log": str(log_path)}
                except subprocess.TimeoutExpired:
                    receipt = {"stage": stage, "exit_code": None, "timed_out": True,
                               "elapsed_seconds": time.monotonic() - start, "log": str(log_path)}
            result["stages"].append(receipt)
            write_json(report, result)
            print(json.dumps(receipt), flush=True)
            if receipt.get("exit_code") != 0:
                raise RuntimeError(f"{stage} failed or timed out; see {log_path}. No fallback was run")
        result["status"] = "pass"
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    try:
        result["identity_after"] = identities()
        from native_loader import verify_source
        result["native_source_after"] = verify_source(STAGE1 / "upstream", STAGE1 / "provenance/unifloral.json")
    except Exception as error:
        result["status"] = "blocked_or_failed"
        result["postcheck_error"] = {"type": type(error).__name__, "message": str(error)}
    result["finished_utc"] = utc()
    write_json(report, result)
    print(json.dumps({"status": result["status"], "report": str(report),
                      "scientific_validation_status": result["scientific_validation_status"]}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
