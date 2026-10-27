"""User-run bounded probe and parity harness. Imports no JAX in the supervisor."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
NATIVE_XLA_FLAGS = "--xla_gpu_triton_gemm_any=True"
PACKAGES = ("jax", "jaxlib", "numpy", "flax", "optax", "distrax", "chex",
            "tensorflow-probability", "jax-cuda12-plugin", "jax-cuda12-pjrt")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def suite_identity():
    prepared = json.loads((ROOT / "prepared_files.json").read_text(encoding="utf-8"))
    hashes = {}
    for name, expected in prepared["files"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("Prepared file path outside suite")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Prepared harness file changed: {name}")
        hashes[name] = actual
    return {"passed": True, "files": hashes}


def versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def constraints(package_versions):
    checks = []
    for name, bound in (("jax", (0, 6)), ("numpy", (2, 0))):
        version = package_versions.get(name)
        match = re.match(r"^(\d+)\.(\d+)", version or "")
        checks.append({"package": name, "installed": version,
                       "upstream_constraint": "<" + ".".join(map(str, bound)),
                       "satisfied": bool(match and tuple(map(int, match.groups())) < bound)})
    return checks


def worker(stage, output, allow_constraint_override):
    result = {"stage": stage, "started_utc": utc(), "status": "running"}
    try:
        result["harness_before"] = suite_identity()
        result["packages"] = versions()
        result["upstream_constraint_checks"] = constraints(result["packages"])
        result["explicit_constraint_override"] = allow_constraint_override
        if not all(c["satisfied"] for c in result["upstream_constraint_checks"]) and not allow_constraint_override:
            raise RuntimeError("Worker requires explicit override for installed versions outside upstream constraints")
        from native_loader import load_native, verify_source
        result["source_before"] = verify_source(ROOT / "upstream", ROOT / "provenance/unifloral.json")
        import jax
        import jax.numpy as jnp
        result["runtime"] = {
            "python": platform.python_version(), "executable": sys.executable,
            "packages": versions(), "backend": jax.default_backend(),
            "devices": [str(d) for d in jax.devices()],
            "device_kinds": [d.device_kind for d in jax.devices()],
            "effective_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "jax_disable_jit": bool(jax.config.jax_disable_jit),
            "environment": {key: os.environ.get(key) for key in (
                "JAX_PLATFORMS", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE")},
        }
        if jax.default_backend() != "gpu" or any(d.platform != "gpu" for d in jax.devices()):
            raise RuntimeError("GPU required; CPU fallback is forbidden for this probe")
        if str(jax.config.jax_default_matmul_precision) != "highest":
            raise RuntimeError("Highest matmul precision did not take effect")
        if bool(jax.config.jax_disable_jit):
            raise RuntimeError("JIT is disabled; this probe requires compiled native updates")
        if os.environ.get("XLA_FLAGS") != NATIVE_XLA_FLAGS:
            raise RuntimeError("Native XLA flags do not match the declared setting")
        native = load_native(ROOT / "upstream", ROOT / "provenance/unifloral.json")
        result["native_slice"] = native._source_receipt
        if stage == "probe":
            # Device compilation only. No policy, dataset generation or optimizer.
            x = jnp.arange(64, dtype=jnp.float32).reshape(8, 8) / 64
            value = jax.jit(lambda a: a @ a.T)(x)
            value.block_until_ready()
            if not bool(jnp.all(jnp.isfinite(value))):
                raise RuntimeError("Non-finite device matmul result")
            result["optimizer_updates"] = 0
            result["checks"] = ["GPU selected", "highest precision effective",
                                "real dependencies imported", "8x8 compiled matmul finite"]
        else:
            from parity_checks import run
            result["parity"] = run(native, output)
            if result["parity"].get("passed") is not True:
                raise RuntimeError("Deterministic parity checks did not pass")
        result["status"] = "pass"
    except Exception as exc:
        result["status"] = "fail"
        result["error"] = {"type": type(exc).__name__, "message": str(exc),
                           "traceback": traceback.format_exc()}
    try:
        from native_loader import verify_source
        result["source_after"] = verify_source(ROOT / "upstream", ROOT / "provenance/unifloral.json")
        result["harness_after"] = suite_identity()
    except Exception as exc:
        result["status"] = "fail"
        result["postcheck_error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["finished_utc"] = utc()
    write_json(output / f"{stage}.json", result)
    print(json.dumps({"stage": stage, "status": result["status"], "report": str(output / f"{stage}.json")}))
    return 0 if result["status"] == "pass" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-outside-upstream-constraints", action="store_true",
                        help="Explicitly allow provisional kernel checks with JAX/NumPy outside upstream bounds")
    parser.add_argument("--run-parity", action="store_true", help="After probe, run fixed tiny deterministic update checks")
    parser.add_argument("--worker", choices=("probe", "parity"), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None or args.output.resolve().parent != ROOT or not args.output.is_dir():
            parser.error("Worker output must be an existing result directory in this suite")
        return worker(args.worker, args.output.resolve(), args.allow_outside_upstream_constraints)
    output = ROOT / ("results_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    result = {"schema_version": 1, "started_utc": utc(), "status": "pending",
              "scope": "Provisional UniFloral native computational parity; no efficacy or environment certification",
              "scientific_validation_status": "incomplete_design_review_required",
              "packages": versions(), "python": platform.python_version(), "executable": sys.executable,
              "explicit_constraint_override": args.allow_outside_upstream_constraints,
              "limits": {"primary_critic_optimizer_updates": 31 if args.run_parity else 0,
                         "actor_optimizer_updates": 16 if args.run_parity else 0,
                         "environment_rollouts": 0, "real_dataset_generation": 0},
              "adaptations": ["Unchanged computational AST nodes only; upstream CLI/data/eval not imported",
                              "Native XLA_FLAGS value set before JAX import, rather than upstream's late assignment",
                              "Highest matmul precision and disabled preallocation declared for all checks",
                              "Fixed synthetic fixture, batch size 8, native gamma 0.99; no benchmark reproduction"],
              "stages": []}
    report = output / "report.json"
    try:
        result["harness_before"] = suite_identity()
        from native_loader import verify_source
        result["source_before"] = verify_source(ROOT / "upstream", ROOT / "provenance/unifloral.json")
        result["upstream_constraint_checks"] = constraints(result["packages"])
        missing = [name for name in ("jax", "jaxlib", "numpy", "flax", "optax", "distrax") if result["packages"][name] is None]
        if missing:
            raise RuntimeError("Missing required installed packages: " + ", ".join(missing) + "; no installation attempted")
        if not all(c["satisfied"] for c in result["upstream_constraint_checks"]) and not args.allow_outside_upstream_constraints:
            raise RuntimeError("Installed JAX/NumPy are outside pinned upstream requirements; explicit provisional-check flag required")
        inherited_xla = os.environ.get("XLA_FLAGS", "")
        result["inherited_xla_flags"] = inherited_xla
        if inherited_xla not in ("", NATIVE_XLA_FLAGS):
            raise RuntimeError("Existing XLA_FLAGS conflict with the native flag; stopped without replacing them")
        env = dict(os.environ)
        env.update(JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                   XLA_FLAGS=NATIVE_XLA_FLAGS, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        write_json(report, result)
        for stage in (("probe", "parity") if args.run_parity else ("probe",)):
            print(f"Starting bounded {stage}; output: {output}", flush=True)
            log_path = output / f"{stage}.log"
            start = time.monotonic()
            with log_path.open("x", encoding="utf-8") as log:
                try:
                    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                               "--worker", stage, "--output", str(output)]
                    if args.allow_outside_upstream_constraints:
                        command.append("--allow-outside-upstream-constraints")
                    completed = subprocess.run(command,
                                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=180 if stage == "probe" else 600, check=False)
                    receipt = {"stage": stage, "exit_code": completed.returncode,
                               "elapsed_seconds": time.monotonic() - start, "log": str(log_path)}
                except subprocess.TimeoutExpired:
                    receipt = {"stage": stage, "exit_code": None, "timed_out": True,
                               "elapsed_seconds": time.monotonic() - start, "log": str(log_path)}
            result["stages"].append(receipt)
            write_json(report, result)
            print(json.dumps(receipt), flush=True)
            if receipt.get("exit_code") != 0:
                raise RuntimeError(f"{stage} failed or timed out; see {log_path}. No fallback or further stage was run")
        result["status"] = "pass"
    except Exception as exc:
        result["status"] = "blocked_or_failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
    try:
        from native_loader import verify_source
        result["source_after"] = verify_source(ROOT / "upstream", ROOT / "provenance/unifloral.json")
        result["harness_after"] = suite_identity()
    except Exception as exc:
        result["status"] = "blocked_or_failed"
        result["postcheck_error"] = {"type": type(exc).__name__, "message": str(exc)}
    result["finished_utc"] = utc()
    write_json(report, result)
    print(json.dumps({"status": result["status"], "report": str(report),
                      "scientific_validation_status": result["scientific_validation_status"]}), flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
