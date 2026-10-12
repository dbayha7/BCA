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
~/.cache/bca-shift/py. Importing this package puts that directory first on sys.path and refuses (ImportError) unless
gymnasium 1.0.0 imports from it and numpy is still the shared environment's 2.2.6. Nothing is ever installed into
the shared environment.
"""

import importlib
import os
import sys
from pathlib import Path

GYMNASIUM_VERSION = "1.0.0"
NUMPY_VERSION = "2.2.6"
DEPS_ENV = "BCA_SHIFT_DEPS"
DEPS_DEFAULT = Path.home() / ".cache" / "bca-shift" / "py"
ROOT = Path(__file__).resolve().parents[2]


def deps_dir():
    """The directory that holds gymnasium and its helpers (BCA_SHIFT_DEPS or ~/.cache/bca-shift/py)."""
    return Path(os.environ.get(DEPS_ENV) or DEPS_DEFAULT).expanduser().resolve()


def _inside(path, directory):
    path, directory = os.path.realpath(path), os.path.realpath(directory)
    return path == directory or path.startswith(directory + os.sep)


def _require_dependencies():
    """Put the dependency directory first on sys.path, then refuse unless gymnasium 1.0.0 comes from it and numpy is
    the shared environment's own 2.2.6. Returns the gymnasium module."""
    deps = str(deps_dir())
    hint = (f"Install with: BCA_SHIFT_PYTHON={sys.executable} sh {ROOT / 'experiments/shift/install_deps.sh'} "
            f"(installs into {deps}; set {DEPS_ENV} to choose another directory).")
    if not os.path.isdir(deps):
        raise ImportError(f"experiments.shift needs gymnasium {GYMNASIUM_VERSION} in {deps}, which does not exist. "
                          + hint)
    if deps in sys.path:
        sys.path.remove(deps)
    sys.path.insert(0, deps)
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
