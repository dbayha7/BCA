"""Facts about the Python environment and the processor, read only. Standard library only (numpy is imported only
inside openblas_cores). Used by the cloud to write env_record.json and by the self-test to compare against it.

Nothing here imports the study's code, reads its data or writes anything.
"""

import ctypes
import glob
import hashlib
import importlib.metadata as metadata
import os
import platform
import re
import sys

# Installed in the cloud but never imported on the Stage 2 path (they need a system MuJoCo 2.1 and a C compiler):
# left out of requirements-lock.txt.
EXCLUDED_FROM_LOCK = ("mujoco-py", "d4rl", "gym", "gym-notices", "dm-control", "mujoco", "pybullet", "glfw",
                      "pyopengl", "labmaze")
# The eight versions the gate's pre-registration records and checks in every phase.
GATE_PACKAGES = ("numpy", "jax", "jaxlib", "flax", "optax", "scipy", "h5py", "gymnasium")
# Packages whose files can change the numbers: a difference in their installed files fails the self-test.
NUMERIC_PACKAGES = ("numpy", "scipy", "jax", "jaxlib", "flax", "optax", "h5py", "ml-dtypes", "opt-einsum", "chex",
                    "msgpack", "orbax-checkpoint", "tensorstore", "typing-extensions", "absl-py", "etils", "toolz")
CPU_FLAGS = ("avx512f", "avx512bw", "avx512vl", "avx512dq", "avx512_bf16", "avx2", "fma")


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def read_lock(path):
    """[(name, version)] from requirements-lock.txt ('name==version' lines; hash lines and comments skipped)."""
    out = []
    with open(path, encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "--")):
                continue
            m = re.match(r"([A-Za-z0-9_.\-]+)==([^\s\\;]+)", line)
            if m:
                out.append((canonical_name(m.group(1)), m.group(2)))
    return out


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def dist_content_sha256(name):
    """SHA-256 over the installed files of one distribution, read from disk: sorted (path, SHA-256 of the bytes), leaving
    out its metadata folder (*.dist-info: the installer's name and the REQUESTED marker differ between installs),
    scripts outside the package (../bin: their first line names the environment's own path) and bytecode caches.
    Equal values mean the same package files, byte for byte. None when the distribution is not installed."""
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return None
    rows = []
    for f in dist.files or ():
        rel = str(f).replace(os.sep, "/")
        if rel.startswith("../") or ".dist-info/" in rel or "__pycache__" in rel or rel.endswith((".pyc", ".pyo")):
            continue
        p = dist.locate_file(f)
        try:
            with open(p, "rb") as handle:
                digest = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            digest = "missing"
        rows.append(f"{rel}\0{digest}")
    rows.sort()
    return dict(files=len(rows), sha256=hashlib.sha256("\n".join(rows).encode("utf8")).hexdigest())


def wheel_tags(name):
    """The Tag lines of the installed distribution's WHEEL file (which wheel was installed), or []."""
    try:
        text = metadata.distribution(name).read_text("WHEEL") or ""
    except metadata.PackageNotFoundError:
        return []
    return sorted(line.split(":", 1)[1].strip() for line in text.splitlines() if line.startswith("Tag:"))


def openblas_cores():
    """The matrix-kernel family OpenBLAS picked in numpy's and scipy's bundled copies (for example 'SkylakeX' or
    'Haswell'), read through the libraries' own functions. The gate's anchors depend on it."""
    out = {}
    try:
        import numpy
        base = os.path.dirname(os.path.dirname(numpy.__file__))
    except ImportError:
        return out
    for pkg, pattern, prefixes in (
            ("numpy", "numpy.libs/libscipy_openblas64_*.so", ("scipy_openblas_get_corename64_",
                                                               "openblas_get_corename64_")),
            ("scipy", "scipy.libs/libscipy_openblas-*.so", ("scipy_openblas_get_corename",
                                                             "openblas_get_corename"))):
        libs = sorted(glob.glob(os.path.join(base, pattern)))
        if not libs:
            out[pkg] = None
            continue
        try:
            lib = ctypes.CDLL(libs[0])
        except OSError as error:
            out[pkg] = f"not loadable: {error}"
            continue
        name = None
        for fn in prefixes:
            f = getattr(lib, fn, None)
            if f is not None:
                f.restype = ctypes.c_char_p
                name = f().decode("ascii", "replace")
                break
        out[pkg] = name
    return out


def cpu_info():
    """Processor model, the instruction-set flags that matter here, and how many logical processors exist and this
    process may use."""
    model, flags = None, set()
    try:
        with open("/proc/cpuinfo", encoding="utf8", errors="replace") as handle:
            for line in handle:
                if model is None and line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                if line.startswith("flags") and not flags:
                    flags = set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return dict(model=model, flags={f: (f in flags) for f in CPU_FLAGS}, logical_processors=os.cpu_count(),
                usable_by_this_process=len(os.sched_getaffinity(0)), machine=platform.machine())


def python_facts():
    return dict(version=sys.version.split()[0], full=sys.version, implementation=platform.python_implementation(),
                libc=" ".join(platform.libc_ver()), executable=sys.executable, prefix=sys.prefix)
