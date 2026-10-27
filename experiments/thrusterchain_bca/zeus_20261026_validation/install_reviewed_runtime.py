"""Create one new isolated venv from the reviewed lock; never run models or the audit."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parent
RESOLUTION = ROOT / "resolution_20261026T215529_236287Z"
TARGET = Path("/home/david/bca-operator-audit-20261026")
BASE = Path("/home/david/miniconda3/envs/corlverify/bin/python")
HASHES = {
    "resolution.json": "5f744787ac5e1b50997422abc66f65b3868dddd09f37c63d93174b2a8bbf82a7",
    "pip-report.json": "b8bcb14a19c9a7489161815054e8c02b98ca7ba94c02212c390911f16adeca5d",
    "resolved-requirements.txt": "017b189f4786193db03c32fd7b3d691ae320c1274eff11e1e0c3766764655d14",
}
MANIFEST_SHA = "28df1b6bbffacda0a6659a203bf9aab2a2e82a259d3621642cdae4ad44c73adf"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verified_inputs():
    for name, expected in HASHES.items():
        if digest(RESOLUTION / name) != expected:
            raise ValueError("Reviewed resolution changed: " + name)
    receipt = json.loads((RESOLUTION / "resolution.json").read_text(encoding="utf-8"))
    if receipt["status"] != "resolved_for_review" or receipt["package_count"] != 58:
        raise ValueError("Resolution is not the reviewed 58-package result")
    handoff = ROOT.parent / "titan_20261026"
    manifest = handoff / "HANDOFF_MANIFEST.json"
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError("Frozen Titan handoff manifest changed")
    for item in json.loads(manifest.read_text(encoding="utf-8"))["files"]:
        path = handoff / item["path"]
        if (not path.resolve().is_relative_to(handoff.resolve()) or not path.is_file()
                or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]):
            raise ValueError("Frozen Titan payload changed: " + item["path"])
    return receipt


def run(command, env, log, timeout=1800):
    log.write("COMMAND: " + json.dumps(command) + "\n")
    log.flush()
    subprocess.run(command, env=env, cwd=TARGET, stdout=log,
        stderr=subprocess.STDOUT, timeout=timeout, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-reviewed", action="store_true",
        help="Only after explicit approval: create the new target and install the reviewed hashed wheels")
    args = parser.parse_args()
    receipt = verified_inputs()
    if not args.install_reviewed:
        print(json.dumps({"status": "plan_only", "target": str(TARGET), "packages": 58,
            "lock_sha256": HASHES["resolved-requirements.txt"], "payload_files_verified": 360,
            "existing_environments_modified": False, "audit_started": False}, indent=2))
        return 0
    if (sys.platform != "linux" or platform.machine() != "x86_64"
            or platform.python_implementation() != "CPython" or sys.version_info[:3] != (3, 12, 14)
            or Path(sys.executable).absolute() != BASE or os.geteuid() == 0):
        parser.error("Use Ubuntu's existing corlverify Python 3.12.14 as the regular user")
    if TARGET.parent.resolve() != Path.home().resolve():
        parser.error("Target must be directly under the current user's /home/david")
    if TARGET.exists() or TARGET.is_symlink():
        parser.error("Target already exists; inspect it before any further action. Nothing changed")
    # Atomic exclusive reservation: never reuse, clear or upgrade an existing venv.
    TARGET.mkdir(mode=0o700, exist_ok=False)
    result = {"status": "setup_started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "target": str(TARGET), "base_interpreter": str(BASE), "python_version": sys.version,
        "reviewed_hashes": HASHES, "disk_free_before_bytes": shutil.disk_usage(TARGET).free,
        "existing_environments_modified": False, "audit_started": False,
        "training_started": False, "runtime_gpu_verified": False, "model_modules_loaded": False}
    with (TARGET / "SETUP_REQUEST.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("Creating isolated environment: " + str(TARGET), flush=True)
    print("Installing 58 hash-pinned wheels; progress in " + str(TARGET / "installer.log"), flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PIP_")
        and k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")}
    env.update(PIP_CONFIG_FILE=os.devnull, PYTHONDONTWRITEBYTECODE="1")
    try:
        with (TARGET / "installer.log").open("x", encoding="utf-8") as log:
            # Uses bundled ensurepip in the NEW venv, never installs into corlverify.
            venv.EnvBuilder(with_pip=True, system_site_packages=False,
                clear=False, upgrade=False, symlinks=True, upgrade_deps=False).create(TARGET)
            python = str(TARGET / "bin/python")
            config = (TARGET / "pyvenv.cfg").read_text(encoding="utf-8")
            if "include-system-site-packages = false" not in config:
                raise ValueError("New venv unexpectedly inherits system packages")
            temporary = TARGET / "setup_tmp"
            temporary.mkdir()
            env["TMPDIR"] = str(temporary)
            run([python, "-I", "-B", "-m", "pip", "--isolated", "--disable-pip-version-check",
                "install", "--require-hashes", "--only-binary=:all:", "--no-cache-dir",
                "--no-input", "--progress-bar", "off", "--timeout", "30", "--retries", "1",
                "--index-url", "https://pypi.org/simple", "--report", str(TARGET / "install-report.json"),
                "-r", str(RESOLUTION / "resolved-requirements.txt")], env, log)
            run([python, "-I", "-B", "-m", "pip", "--isolated", "check"], env, log, timeout=60)
            probe = "import importlib.metadata as m,json,sys; print(json.dumps({'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'packages':{n:m.version(n) for n in json.loads(sys.argv[1])}}))"
            observed = json.loads(subprocess.check_output([python, "-I", "-B", "-c", probe,
                json.dumps(list(receipt["packages"]))], env=env, text=True, timeout=60))
            if Path(observed["prefix"]).resolve() != TARGET.resolve() or observed["packages"] != receipt["packages"]:
                raise ValueError("Installed prefix/package versions differ from the reviewed resolution")
            verified_inputs()
            result.update(status="installed_pending_gpu_and_model_identity_checks", installed=observed,
                include_system_site_packages=False, pip_check_passed=True,
                install_report_sha256=digest(TARGET / "install-report.json"))
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="setup_interrupted_preserved" if isinstance(error, KeyboardInterrupt)
            else "setup_failed_preserved", error=type(error).__name__ + ": " + str(error))
    result.update(finished_utc=datetime.now(timezone.utc).isoformat(),
        disk_free_after_bytes=shutil.disk_usage(TARGET).free)
    with (TARGET / "SETUP_RESULT.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({"status": result["status"], "receipt": str(TARGET / "SETUP_RESULT.json"),
        "audit_started": False, "error": result.get("error")}), flush=True)
    return 0 if result["status"] == "installed_pending_gpu_and_model_identity_checks" else 1


if __name__ == "__main__":
    raise SystemExit(main())
