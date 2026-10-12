"""ThrusterChain: a real Gymnasium environment with a controlled, exactly measured distribution shift.

Five floats tied by springs sit in a channel; a steady current and waves push them downstream; four thrusters between
neighbouring floats push back. Past 80% throttle a thruster stalls and pushes less (the no-stall comparison world
keeps the push rising). A recording controller drives the floats and its log is the offline dataset; the D4RL rule
(remove the densest 15% of recorded commands from training, allow them again at evaluation) makes the shift.

Modules (all new; nothing outside experiments/shift is edited):
  thruster_chain  the environment, the one dynamics function (numpy and JAX), the 8 registered ids, the start laws
                  and the four evaluation banks
  recorder        the textbook gain, the recording controller and the reference controllers
  dataset         the recorded log (D4RL field names), HDF5 + JSON manifest, splits, calibration bank, replay check
  shift           the removal rule (gap), random removal, frozen row pools, the per-state shift card and m(s)
  truth           exact values by float64 simulation: Q, slopes, over-preference, returns, points, attribution
  stage0          the trigger bisection, the outcome-free marks T1-T9 and the forecasts

Dependencies. Gymnasium 1.0.0 lives in its own directory (install_deps.sh): BCA_SHIFT_DEPS, default
~/.cache/bca-shift/py. Importing this package appends that directory to the END of sys.path, so it can add gymnasium
and its two helpers (farama_notifications; cloudpickle and typing_extensions when the environment lacks them) but can
never replace a module of the repository or of the shared Python environment. The import refuses (ImportError) unless
gymnasium 1.0.0 imports from that directory, numpy is still the shared environment's 2.2.6, and the directory holds
nothing but the four packages install_deps.sh puts there. dependency_record() gives the directory's resolved path and
a SHA-256 over every file in it; the pre-registration records it and every later stage checks it. Nothing is ever
installed into the shared environment.

Because gymnasium is found only through this package, import experiments.shift (or any of its modules) before
gymnasium, from the repository root, e.g.  python -c "import experiments.shift.thruster_chain, gymnasium as gym; ...".
"""

import hashlib
import importlib
import os
import sys
from pathlib import Path

GYMNASIUM_VERSION = "1.0.0"
NUMPY_VERSION = "2.2.6"
DEPS_ENV = "BCA_SHIFT_DEPS"
DEPS_DEFAULT = Path.home() / ".cache" / "bca-shift" / "py"
ROOT = Path(__file__).resolve().parents[2]
# the only top-level entries the dependency directory may hold (install_deps.sh's four wheels, the installer's lock
# file and Python's bytecode cache); anything else is refused
DEPS_ALLOWED = ("gymnasium", "gymnasium-1.0.0.dist-info", "cloudpickle", "cloudpickle-3.1.2.dist-info",
                "farama_notifications", "farama_notifications-0.0.6.dist-info", "typing_extensions.py",
                "typing_extensions-4.16.0.dist-info", ".lock", "__pycache__")


def deps_dir():
    """The directory that holds gymnasium and its helpers (BCA_SHIFT_DEPS or ~/.cache/bca-shift/py)."""
    return Path(os.environ.get(DEPS_ENV) or DEPS_DEFAULT).expanduser().resolve()


def _inside(path, directory):
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return path == directory or path.startswith(directory + os.sep)


def unexpected_entries(directory=None):
    """Top-level entries of the dependency directory other than DEPS_ALLOWED (sorted)."""
    directory = deps_dir() if directory is None else Path(directory)
    return sorted(name for name in os.listdir(directory) if name not in DEPS_ALLOWED)


def dependency_record(directory=None):
    """The dependency directory as the pre-registration records it: its resolved path, the number of files and a
    SHA-256 over every file (relative path and bytes, in sorted order), leaving out Python's bytecode caches
    (__pycache__, *.pyc) and the installer's empty lock file, and the top-level entries it should not hold."""
    directory = deps_dir() if directory is None else Path(directory).expanduser().resolve()
    digest, files = hashlib.sha256(), 0
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        rel = path.relative_to(directory)
        if "__pycache__" in rel.parts or path.suffix in (".pyc", ".pyo") or rel.as_posix() == ".lock":
            continue
        digest.update(rel.as_posix().encode("utf8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
        files += 1
    return dict(path=str(directory), files=files, sha256=digest.hexdigest(),
                unexpected_entries=unexpected_entries(directory))


def _require_dependencies():
    """Append the dependency directory to the end of sys.path (it can add modules but never shadow the repository's
    or the shared environment's), then refuse unless the directory holds only DEPS_ALLOWED, gymnasium 1.0.0 comes from
    it and numpy is the shared environment's own 2.2.6. Returns the gymnasium module."""
    deps = str(deps_dir())
    hint = (f"Install with: BCA_SHIFT_PYTHON=/path/to/your/venv/bin/python sh "
            f"{ROOT / 'experiments/shift/install_deps.sh'} (installs into {deps}; set {DEPS_ENV} to choose another "
            "directory).")
    if not os.path.isdir(deps):
        raise ImportError(f"experiments.shift needs gymnasium {GYMNASIUM_VERSION} in {deps}, which does not exist. "
                          + hint)
    extra = unexpected_entries(deps)
    if extra:
        raise ImportError(f"experiments.shift refuses: the dependency directory {deps} holds entries other than the "
                          f"four packages of install_deps.sh: {extra}")
    if deps in sys.path:
        sys.path.remove(deps)
    sys.path.append(deps)
    import numpy

    if numpy.__version__ != NUMPY_VERSION or _inside(numpy.__file__, deps):
        raise ImportError(f"experiments.shift refuses: numpy is {numpy.__version__} from {numpy.__file__}, expected "
                          f"the shared environment's {NUMPY_VERSION} (never a copy in {deps})")
    try:
        gymnasium = importlib.import_module("gymnasium")
    except ImportError as error:
        raise ImportError(f"experiments.shift cannot import gymnasium from {deps} ({error}). " + hint) from error
    if gymnasium.__version__ != GYMNASIUM_VERSION or not _inside(gymnasium.__file__, deps):
        raise ImportError(f"experiments.shift refuses: gymnasium {gymnasium.__version__} imported from "
                          f"{gymnasium.__file__}, expected {GYMNASIUM_VERSION} from {deps}. " + hint)
    return gymnasium


def environment_packages():
    """Sorted 'name==version' of every distribution in the shared environment (sys.prefix), excluding the dependency
    directory: the list the build checks hash before and after, to show nothing was installed into the environment."""
    import importlib.metadata as metadata

    deps = str(deps_dir())
    names = set()
    for dist in metadata.distributions():
        location = str(getattr(dist, "_path", ""))
        if location and _inside(location, deps):
            continue
        names.add(f"{dist.metadata['Name']}=={dist.version}")
    return sorted(names)


_require_dependencies()
