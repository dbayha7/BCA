"""Bounded read-only Linux Python discovery; no installation or audit execution."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


REQUIRED = {
    "jax": "0.6.2", "jaxlib": "0.6.2", "jax-cuda12-plugin": "0.6.2",
    "jax-cuda12-pjrt": "0.6.2", "numpy": "2.5.4", "flax": "0.10.7",
    "optax": "0.2.8", "distrax": "0.1.5", "chex": "0.1.90",
    "tensorflow-probability": "0.25.0", "scipy": "1.18.1",
}
XLA = "--xla_gpu_triton_gemm_any=True"
MAX_CANDIDATES = 16
MAX_CHILDREN_PER_LOCATION = 12
MAX_SECONDS = 60
PACKAGES = """
import importlib.metadata as m, json, sys
versions = {}
for name in json.loads(sys.argv[1]):
    try:
        versions[name] = m.version(name)
    except m.PackageNotFoundError:
        versions[name] = None
print(json.dumps({'python': sys.version.split()[0], 'versions': versions}))
"""
DEVICES = """
import json, jax
print(json.dumps({'backend': jax.default_backend(),
    'devices': [d.device_kind for d in jax.devices()],
    'precision': str(jax.config.jax_default_matmul_precision),
    'x64': bool(jax.config.jax_enable_x64),
    'jit_disabled': bool(jax.config.jax_disable_jit)}))
"""


def candidates(extra):
    home = Path.home()
    project = Path(__file__).resolve().parents[3]
    roots = [project, home / "bca", home / "BCA", home / "dev/BCA"]
    names = []
    for value in extra:
        names.append(Path(value).expanduser())
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if os.environ.get(key):
            names.append(Path(os.environ[key]) / "bin/python")
    names.append(home / "bca-gpu/bin/python")
    for root in roots:
        names.extend(root / name / "bin/python" for name in (".venv", "venv"))
    for name in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        names.append(home / name / "bin/python")
    locations = [home / p for p in (
        ".venvs", "venvs", "envs", "miniconda3/envs", "anaconda3/envs",
        "miniforge3/envs", "mambaforge/envs", ".local/share/mamba/envs",
    )]
    for location in locations:
        try:
            # One directory level only, capped while reading; no recursive scan.
            with os.scandir(location) as entries:
                children = []
                for entry in entries:
                    children.append(entry.name)
                    if len(children) >= MAX_CHILDREN_PER_LOCATION:
                        print("SCAN_LIMIT: " + str(location), flush=True)
                        break
            names.extend(location / name / "bin/python" for name in sorted(children))
        except FileNotFoundError:
            pass
        except OSError as error:
            print("UNREADABLE: {} ({})".format(location, type(error).__name__), flush=True)
    # Always inspect the running system/PATH interpreter as well.
    found = [Path(sys.executable)]
    if shutil.which("python3"):
        found.append(Path(shutil.which("python3")))
    found.extend(names)
    unique = []
    seen = set()
    for path in found:
        # Keep venv paths distinct: their python symlinks often share one target.
        path = Path(os.path.abspath(str(path)))
        if str(path) in seen:
            continue
        seen.add(str(path))
        try:
            if path.is_file() and os.access(path, os.X_OK):
                unique.append(path)
        except OSError:
            continue
    return unique[:MAX_CANDIDATES], len(unique) > MAX_CANDIDATES


def probe(path, code, args, env, timeout):
    result = subprocess.run([str(path), "-B", "-c", code] + args,
        env=env, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode:
        # Avoid long native tracebacks or incidental environment contents.
        return None, "exit={}; stderr omitted ({} chars)".format(
            result.returncode, len(result.stderr))
    try:
        return json.loads(result.stdout), None
    except ValueError:
        return None, "unrecognized output; no compatibility claim"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", action="append", default=[],
        help="Explicit additional interpreter; repeat if needed")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run manually inside Ubuntu-22.04; this script never starts WSL")
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    paths, capped = candidates(args.python)
    print("Read-only discovery: <=16 interpreters, 60-second probe budget; no audit or installs.", flush=True)
    print("Frozen requirements: " + " ".join(k + "=" + v for k, v in REQUIRED.items()), flush=True)
    print("Evaluator also needs gymnasium (tested 1.0.0; not equality-gated).", flush=True)
    compatible = []
    examined = 0
    for path in paths:
        remaining = MAX_SECONDS - (time.monotonic() - started)
        if remaining < 1:
            print("TIME_LIMIT: remaining candidates not examined", flush=True)
            break
        print("PYTHON: " + str(path), flush=True)
        examined += 1
        try:
            package, error = probe(path, PACKAGES,
                [json.dumps(list(REQUIRED) + ["gymnasium"])], env, min(5, remaining))
            if error:
                print("  UNAVAILABLE: " + error, flush=True)
                continue
            actual = package["versions"]
            present = " ".join(k + "=" + v for k, v in actual.items() if v is not None)
            missing = ",".join(k for k, v in actual.items() if v is None)
            print("  Python " + package["python"] + "; " + (present or "no audit packages"), flush=True)
            if missing:
                print("  missing: " + missing, flush=True)
            mismatch = [k for k, v in REQUIRED.items() if actual[k] != v]
            if mismatch or actual["gymnasium"] is None:
                print("  FROZEN_GATE: incompatible; GPU probe skipped", flush=True)
                continue
            if env.get("XLA_FLAGS", "") not in ("", XLA):
                print("  GPU_PROBE: blocked by conflicting XLA_FLAGS; unchanged", flush=True)
                continue
            remaining = MAX_SECONDS - (time.monotonic() - started)
            if remaining < 1:
                print("  GPU_PROBE: time budget exhausted", flush=True)
                break
            gpu_env = dict(env, JAX_PLATFORMS="cuda", JAX_DEFAULT_MATMUL_PRECISION="highest",
                XLA_FLAGS=XLA, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                JAX_ENABLE_COMPILATION_CACHE="false")
            device, error = probe(path, DEVICES, [], gpu_env, min(15, remaining))
            if error:
                print("  GPU_PROBE: failed; " + error, flush=True)
                continue
            passed = (device["backend"] == "gpu" and bool(device["devices"]) and device["precision"] == "highest"
                and not device["x64"] and not device["jit_disabled"])
            print("  GPU_PROBE: " + json.dumps(device, separators=(",", ":")), flush=True)
            if passed:
                compatible.append(str(path))
                print("  CANDIDATE: package/device checks passed; audit still unexecuted", flush=True)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT: probe child stopped; no compatibility claim", flush=True)
        except (OSError, KeyError, TypeError) as error:
            print("  PROBE_ERROR: " + type(error).__name__, flush=True)
    print("SUMMARY: examined={}; capped={}; candidates={}".format(
        examined, capped, json.dumps(compatible)), flush=True)
    print("Scope is limited to the named locations; absence here is not global absence.", flush=True)
    print("No process-overlap check, checkpoint restore, model forward or evaluator call performed.", flush=True)


if __name__ == "__main__":
    main()
