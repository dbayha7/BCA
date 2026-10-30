"""Paths and the authenticated canonical-core loader for knee_edge_ood_v1. Importing this module does no work beyond
turning off bytecode writing: sealed and hashed folders (rerun1, the gate, the core) must never gain __pycache__."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent                                   # zeus_20261026_validation
REPO = HERE.parents[3]                                     # the BCA repository root
PAYLOAD = HERE.parents[1] / "titan_20261026/payload"
CORE_ROOT = PAYLOAD / "transfer_bundle/core_v4"
RERUN1 = VALIDATION / "central_quartile_training_v1_evaluation_rerun1"
GATE = VALIDATION / "conditional_ood_gate_central_quartile_v1"
DESIGN = VALIDATION / "knee_edge_ood_design_v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    return sha_bytes(Path(path).read_bytes())


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def module(name, path):
    """Load a module from an exact file path once; a second load must come from the same file."""
    path = Path(path).resolve()
    if name in sys.modules:
        result = sys.modules[name]
        require(Path(result.__file__).resolve() == path, "Module path collision: " + name)
        return result
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    try:
        spec.loader.exec_module(result)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return result


def load_environment():
    """The canonical core's time-aware adapter (`thrusterchain.environment`), source hashes authenticated on load.

    Production (WSL) has gymnasium installed. The Windows test environment keeps it in the frozen shift
    dependency folder, which `experiments.shift` puts on sys.path; that package is imported read-only.
    """
    if str(CORE_ROOT) not in sys.path:
        sys.path.insert(0, str(CORE_ROOT))
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        import experiments.shift  # noqa: F401
    import thrusterchain.environment as environment
    environment.canonical  # authenticated by load_canonical() at import
    return environment


def load_datasets():
    load_environment()
    import thrusterchain.datasets as datasets
    return datasets
