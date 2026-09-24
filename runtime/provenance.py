"""Record code and dependency identity for each independent run."""

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_files():
    paths = [ROOT / "train.py", ROOT / "check.py"] + [
        p
        for folder in ("algorithms", "calibration", "runtime")
        for p in (ROOT / folder).glob("*.py")
    ]
    paths += list((ROOT / "configs").glob("*.csv")) + list(
        (ROOT / "configs").glob("*.PROVENANCE")
    )
    paths += list((ROOT / "configs").glob("*.yaml"))
    paths += [
        p
        for p in (
            ROOT / "requirements.txt",
            ROOT / "configs/sources.json",
            ROOT / "configs/unifloral.json",
            ROOT / "README.md",
            ROOT / "ALGORITHMS.md",
            ROOT / "LICENSE",
            ROOT / "NOTICE",
            ROOT / "INTEGRATION.md",
        )
        if p.is_file()
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}


def source_identity():
    result = source_files()
    result["runtime"] = {
        "python": platform.python_version(),
        "python_build": platform.python_build(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "execution_environment": {
            name: os.environ.get(name)
            for name in (
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "JAX_DEFAULT_MATMUL_PRECISION",
                "XLA_FLAGS",
                "XLA_PYTHON_CLIENT_PREALLOCATE",
                "CUDA_VISIBLE_DEVICES",
                "MUJOCO_GL",
                "MUJOCO_PY_FORCE_CPU",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "installed_distributions": dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in importlib.metadata.distributions()
                if dist.metadata.get("Name")
            )
        ),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "jax",
                "jaxlib",
                "flax",
                "optax",
                "chex",
                "distrax",
                "numpy",
                "scipy",
                "ml_dtypes",
                "tensorflow-probability",
                "orbax-checkpoint",
                "tensorstore",
                "gym",
                "d4rl",
                "mujoco-py",
                "h5py",
                "PyYAML",
                "dm-control",
            )
        },
    }
    return result


def write(path, value):
    with Path(path).open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
