"""CLOUD ONLY. Write requirements-lock.txt and env_record.json (with a .sha256 sidecar) from the cloud's Stage 2 virtual
environment, the one launch.sh uses (/tmp/claude-0/bcavenv). Run it with that environment's Python:

    /tmp/claude-0/bcavenv/bin/python -I outputs/shift/beyond-the-data/stage2_gpu/cloud/build_env_record.py

requirements-lock.txt: every installed distribution except the ten that need a system MuJoCo (envfacts.EXCLUDED_FROM_LOCK;
nothing on the Stage 2 path imports them), each pinned to its exact version and to the SHA-256 of the one wheel file
on the Python Package Index whose tags equal the installed wheel's tags (so the user's install gets the same files).
env_record.json: the cloud's Python build, every locked package's version, wheel tags and installed-file SHA-256, the
gate's recorded software block and dependency record, the processor and its OpenBLAS kernel. The self-test compares
the user's machine against it. Reads the network (pypi.org) and the environment; writes only the two files.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "lib"))
import envfacts as E  # noqa: E402

ROOT = Path("/home/user/bca")
GATE_PREREG = ROOT / "runs" / "shift" / "preregistration.json"


def canonical_json(obj):
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False)


def write_with_sidecar(path, text):
    path.write_text(text, encoding="utf8")
    digest = hashlib.sha256(text.encode("utf8")).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def expand_tags(filename):
    """The (python, abi, platform) tags a wheel file name declares, expanded ('a.b' means a or b)."""
    stem = filename[:-4]
    parts = stem.split("-")
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    return sorted(f"{p}-{a}-{q}" for p in py.split(".") for a in abi.split(".") for q in plat.split("."))


def pypi_files(name, version):
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    return [(u["filename"], u["digests"]["sha256"]) for u in data["urls"]]


def main():
    import importlib.metadata as metadata
    dists = {}
    for d in metadata.distributions():
        name = E.canonical_name(d.metadata["Name"])
        dists[name] = d.version
    locked, problems, packages = [], [], {}
    for name in sorted(dists):
        if name in E.EXCLUDED_FROM_LOCK:
            continue
        version = dists[name]
        tags = E.wheel_tags(name)
        match = None
        try:
            files = pypi_files(name, version)
        except Exception as error:  # noqa: BLE001
            problems.append(f"{name}=={version}: the package index could not be read ({error})")
            files = []
        for fn, sha in files:
            if fn.endswith(".whl") and expand_tags(fn) == tags:
                match = (fn, sha)
        if match is None:
            problems.append(f"{name}=={version}: no wheel on the package index has the installed tags {tags}")
        packages[name] = dict(version=version, wheel_tags=tags, wheel_file=None if match is None else match[0],
                              wheel_sha256=None if match is None else match[1],
                              installed_files=E.dist_content_sha256(name))
        locked.append((name, version, None if match is None else match[1]))
    if problems:
        sys.exit("refused:\n  " + "\n  ".join(problems))
    lines = ["# The cloud's Stage 2 environment (/tmp/claude-0/bcavenv), exactly, without the ten packages that need a",
             "# system MuJoCo and that nothing on the Stage 2 path imports (" + ", ".join(E.EXCLUDED_FROM_LOCK) + ").",
             "# Each line pins the exact version and the SHA-256 of the one wheel file the cloud installed.",
             "# Install (inside WSL, as root, after creating /root/bcavenv with Python 3.10.20):",
             "#   uv pip install --python /root/bcavenv/bin/python --no-deps --require-hashes --only-binary :all: \\",
             "#       -r /home/user/bca/outputs/shift/beyond-the-data/stage2_gpu/requirements-lock.txt",
             "# Written by cloud/build_env_record.py."]
    for name, version, sha in locked:
        lines.append(f"{name}=={version} \\")
        lines.append(f"    --hash=sha256:{sha}")
    lock_text = "\n".join(lines) + "\n"
    (PKG / "requirements-lock.txt").write_text(lock_text, encoding="utf8")
    gate = json.loads(GATE_PREREG.read_text(encoding="utf8"))
    sys.path.insert(0, str(ROOT))
    import experiments.shift as S  # noqa: E402  (appends the dependency directory; imports nothing else)
    try:
        uv = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip() or None
    except OSError:
        uv = None
    record = dict(
        kind="stage2-offcloud-env-record",
        about="the cloud's Stage 2 environment, the reference the self-test compares the user's machine with",
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        python=E.python_facts(), uv=uv,
        lock_sha256=hashlib.sha256(lock_text.encode("utf8")).hexdigest(),
        packages=packages, excluded_from_lock=list(E.EXCLUDED_FROM_LOCK),
        gate_software=gate["software"], gate_dependency_directory=gate["dependency_directory"],
        dependency_directory_now=S.dependency_record(),
        cpu=E.cpu_info(), openblas=E.openblas_cores(),
        launch_environment=dict(JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
                                PYTHONPATH=str(ROOT), cpu_pinning="taskset -c <one cpu>"),
    )
    digest = write_with_sidecar(PKG / "env_record.json", canonical_json(record))
    print(f"wrote {PKG / 'requirements-lock.txt'} ({len(locked)} packages) and {PKG / 'env_record.json'} ({digest})")


if __name__ == "__main__":
    main()
