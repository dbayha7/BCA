"""Resolve the frozen runtime without installing packages or creating an environment."""
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
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
EXPECTED = {
    "jax": "0.6.2", "jaxlib": "0.6.2", "jax-cuda12-plugin": "0.6.2",
    "jax-cuda12-pjrt": "0.6.2", "numpy": "2.5.4", "flax": "0.10.7",
    "optax": "0.2.8", "distrax": "0.1.5", "chex": "0.1.90",
    "tensorflow-probability": "0.25.0", "scipy": "1.18.1", "gymnasium": "1.0.0",
}


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def lock_report(report):
    installed = {}
    lines = []
    for item in report["install"]:
        meta = item["metadata"]
        name, version = normalize(meta["name"]), meta["version"]
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("Unexpected package name")
        if name in installed or item.get("is_yanked"):
            raise ValueError("Duplicate or yanked resolution entry: " + name)
        url = urlparse(item["download_info"]["url"])
        digest = item["download_info"]["archive_info"]["hashes"]["sha256"]
        if (url.scheme != "https" or url.hostname != "files.pythonhosted.org"
                or not url.path.endswith(".whl") or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            raise ValueError("Resolution must use hashed official PyPI wheels: " + name)
        if not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version):
            raise ValueError("Unexpected version syntax: " + name)
        installed[name] = version
        lines.append(name + "==" + version + " --hash=sha256:" + digest)
    for name, version in EXPECTED.items():
        if installed.get(name) != version:
            raise ValueError("Frozen version missing or changed: " + name)
    if not any(name.startswith("nvidia-") for name in installed):
        raise ValueError("CUDA extra did not resolve packaged NVIDIA libraries")
    return installed, "\n".join(sorted(lines)) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
        help="Read PyPI metadata, possibly download wheels, and save a resolution receipt; never install")
    args = parser.parse_args()
    if not args.dry_run:
        parser.print_help()
        return 0
    if (sys.platform != "linux" or sys.version_info[:2] != (3, 12)
            or platform.python_implementation() != "CPython"
            or platform.machine() not in ("x86_64", "AMD64")):
        parser.error("Use the existing Ubuntu corlverify CPython 3.12 interpreter on x86_64")
    try:
        pip_version = importlib.metadata.version("pip")
    except importlib.metadata.PackageNotFoundError:
        parser.error("This interpreter has no pip; stop without modifying it")
    if tuple(int(x) for x in pip_version.split(".")[:2]) < (22, 2):
        parser.error("pip >=22.2 is needed for dry-run/report; stop without upgrading this environment")
    requirements = ROOT / "runtime_requirements.txt"
    expected_lines = {k + "==" + v for k, v in EXPECTED.items()}
    expected_lines.remove("jax==0.6.2")
    expected_lines.add("jax[cuda12]==0.6.2")
    actual_lines = [l.strip() for l in requirements.read_text().splitlines()
        if l.strip() and not l.lstrip().startswith("#")]
    if len(actual_lines) != len(expected_lines) or set(actual_lines) != expected_lines:
        raise ValueError("Frozen requirements changed; no resolution attempted")
    output = ROOT / ("resolution_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    output.mkdir(exist_ok=False)
    temporary = output / "tmp"
    temporary.mkdir()
    env = {k: v for k, v in os.environ.items()
        if not k.startswith("PIP_") and k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temporary))
    command = [sys.executable, "-B", "-m", "pip", "--isolated", "--disable-pip-version-check",
        "install", "--dry-run", "--ignore-installed", "--only-binary=:all:",
        "--no-cache-dir", "--no-input", "--progress-bar", "off", "--timeout", "15",
        "--retries", "1", "--index-url", "https://pypi.org/simple", "--report",
        str(output / "pip-report.json"), "-r", str(requirements)]
    print("Resolving exact packages only; no venv or package installation. Receipt: " + str(output), flush=True)
    result = {"status": "failed", "python": sys.executable, "python_version": sys.version,
        "pip": pip_version, "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
        "environment_created": False, "packages_installed": False, "audit_started": False,
        "command": command}
    try:
        with (output / "resolver.log").open("x", encoding="utf-8") as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, timeout=300, check=False).returncode
        result["exit_code"] = code
        if code:
            raise RuntimeError("Resolver failed; see resolver.log. No substitution or retry performed")
        report_path = output / "pip-report.json"
        installed, lock = lock_report(json.loads(report_path.read_text()))
        lock_path = output / "resolved-requirements.txt"
        with lock_path.open("x", encoding="utf-8", newline="\n") as f:
            f.write(lock)
        result.update(status="resolved_for_review", package_count=len(installed), packages=installed,
            report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
            lock_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    with (output / "resolution.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: result.get(k) for k in ("status", "package_count", "error")}), flush=True)
    print("Review resolution.json and the hashed lock before authorizing a NEW isolated venv.", flush=True)
    return 0 if result["status"] == "resolved_for_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())
