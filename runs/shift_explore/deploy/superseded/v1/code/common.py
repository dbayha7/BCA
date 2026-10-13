"""EXPLORATORY (Stage 1 of outputs/shift/beyond-the-data/PLAN.md, section 4.3: the deployment-check channel, no
training). Outside the gate; changes no pre-registered rule. NO BCA: no band, calibrator, threshold, dose or BCA
update is fitted, read or called. Nothing from algorithms/td3_bc_bca.py or learners.bca_* is called; from calibration/
only calibration/bank.stratified_bank runs, because the gate's data loader (experiments/shift/dataset.py, through
acceptance.load_seed) calls it to regenerate the log's calibration-bank split (a NumPy row sampler).

Shared plumbing for the deploy component:
  - the room harness (runs/shift_explore/room/code/harness.py) and the fallback diagnostic
    (scratchpad/ood/fallback/fallback.py) are imported read-only by executing their exact verified bytes: the room
    harness's SHA-256 must equal the one the room's code manifest records (and the manifest must match its sidecar),
    fallback.py's must equal the one pinned here; on any mismatch the import refuses. Nothing is compiled to disk.
  - every output goes under runs/shift_explore/deploy (never runs/shift, never experiments/, never the room's folder),
    as strict sorted JSON with a '<name>.sha256' sidecar, labelled EXPLORATORY.
"""

import hashlib
import json
import math
import sys
import time
import types
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "shift_explore" / "deploy"
SPEC_PATH = OUT / "spec.json"
MANIFEST_NAME = "code_manifest.json"
CODE_FILES = ("common.py", "signals.py", "run_job.py", "summarise.py", "write_spec.py", "write_manifest.py",
              "launch.sh")

ROOM_OUT = ROOT / "runs" / "shift_explore" / "room"
ROOM_MANIFEST = ROOM_OUT / "code_manifest.json"
ROOM_SPEC = ROOM_OUT / "spec.json"
ROOM_HARNESS_COPY = ROOM_OUT / "code" / "harness.py"
ROOM_HARNESS_SOURCE = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/room/harness.py")
ROOM_POOLED = ROOM_OUT / "pooled.json"

FALLBACK_PY = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad/ood/fallback/fallback.py")
FALLBACK_PY_SHA256 = "fa09489fa0a8ec77b42d1a6753ddeb9aa6a807cf84bdc42868c7a1701e4129b3"
FALLBACK_CACHE = FALLBACK_PY.parent / "cache"     # its held-out distance scores; reused read-only (keyed by a data hash)
FALLBACK_OUT = ROOT / "runs" / "shift_explore" / "fallback"
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"

LABEL = ("EXPLORATORY: Stage 1 deployment-check channel (plan section 4.3), computed outside the gate on saved gate "
         "actors; no training; not a gated output; changes no pre-registered rule; no BCA band, calibrator, threshold, "
         "dose or BCA update was used (stand-ins and oracles only)")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def _read_verified(path):
    """A JSON with a '<name>.sha256' sidecar (the gate's format); refuses when missing or when the sidecar differs."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"refused: {path} is missing")
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != sha256_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {path} differs from its SHA-256 sidecar")
    return json.loads(text)


# ------------------------------------------------------------------------------------------------ pinned imports

_MODULES = {}


def _exec_module(name, path, data):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod          # before exec: dataclasses look their module up in sys.modules
    try:
        exec(compile(data, str(path), "exec"), mod.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def room_pins():
    """(room manifest record, its SHA-256, harness SHA-256 the manifest records)."""
    man = _read_verified(ROOM_MANIFEST)
    if man.get("kind") != "room-code-manifest" or not str(man.get("label", "")).startswith("EXPLORATORY"):
        raise SystemExit(f"refused: {ROOM_MANIFEST} is not a labelled room code manifest")
    want = man["files"].get("harness.py")
    if not want:
        raise SystemExit("refused: the room manifest does not list harness.py")
    return man, file_sha256(ROOM_MANIFEST), want


def load_room_harness():
    """The room harness, read-only: the exact bytes of its pinned copy (runs/shift_explore/room/code/harness.py),
    executed only when their SHA-256 equals the room manifest's (and the scratchpad source, when present, has the same
    SHA-256). Refuses on any mismatch."""
    if "room_harness" in _MODULES:
        return _MODULES["room_harness"]
    man, man_sha, want = room_pins()
    data = ROOM_HARNESS_COPY.read_bytes()
    got = sha256_bytes(data)
    if got != want:
        raise SystemExit(f"refused: the room harness copy has SHA-256 {got}, the room manifest records {want}")
    if ROOM_HARNESS_SOURCE.exists() and file_sha256(ROOM_HARNESS_SOURCE) != want:
        raise SystemExit("refused: the room harness source differs from the room manifest")
    if man.get("spec_sha256") != file_sha256(ROOM_SPEC):
        raise SystemExit("refused: the room manifest names another room spec than runs/shift_explore/room/spec.json")
    mod = _exec_module("room_harness", ROOM_HARNESS_COPY, data)
    _MODULES["room_harness"] = mod
    _MODULES["room_pins"] = dict(room_manifest_sha256=man_sha, room_harness_sha256=got,
                                 room_spec_sha256=man["spec_sha256"])
    return mod


def load_fallback():
    """The fallback diagnostic's module (its Reference, conformal_threshold, fallback_command and run_rule), read-only:
    its exact bytes, executed only when their SHA-256 equals FALLBACK_PY_SHA256. Its main() is never called and its
    output folder is never written."""
    if "fallback" in _MODULES:
        return _MODULES["fallback"]
    data = FALLBACK_PY.read_bytes()
    got = sha256_bytes(data)
    if got != FALLBACK_PY_SHA256:
        raise SystemExit(f"refused: fallback.py has SHA-256 {got}, pinned {FALLBACK_PY_SHA256}")
    load_room_harness()     # the gate modules are imported once, by the harness, first
    mod = _exec_module("deploy_fallback", FALLBACK_PY, data)
    _MODULES["fallback"] = mod
    return mod


def pins():
    load_room_harness()
    return dict(_MODULES["room_pins"], fallback_py_sha256=FALLBACK_PY_SHA256)


# ------------------------------------------------------------------------------------------------ outputs

def guard_out(path):
    """Refuse any output outside runs/shift_explore/deploy (never runs/shift, experiments/ or the room's folder)."""
    p = Path(path).resolve()
    if OUT.resolve() not in p.parents and p != OUT.resolve():
        raise SystemExit(f"refused: {p} is outside {OUT}")
    for protected in (ROOT / "runs" / "shift", ROOT / "experiments", ROOM_OUT):
        if p == protected.resolve() or protected.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}")
    return p


def canonical(payload):
    H = load_room_harness()
    return H.A.canonical(payload)


def write_json(path, payload):
    """acceptance.write_json's format (strict sorted JSON, '<name>.sha256' sidecar), restricted to OUT."""
    path = guard_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = sha256_bytes(text.encode("utf8"))
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path):
    """A JSON written by write_json; None when missing; SystemExit when its sidecar does not match."""
    if not Path(path).exists():
        return None
    return _read_verified(path)


def load_spec():
    spec = read_json(SPEC_PATH)
    if spec is None:
        raise SystemExit(f"refused: no spec at {SPEC_PATH}; it must be written before any run")
    if not str(spec.get("label", "")).startswith("EXPLORATORY") or spec.get("kind") != "deploy-spec":
        raise SystemExit("refused: the deploy spec is not labelled")
    return spec, file_sha256(SPEC_PATH)


def code_hashes(names=CODE_FILES):
    return {n: file_sha256(HERE / n) for n in names}


def check_code(spec_sha):
    """Refuse unless the deploy code manifest exists, matches its sidecar, names the current spec, and every file it
    lists (here and in its copy under OUT/code/) still has the SHA-256 it recorded; also re-check the pinned imports
    against the spec. Returns (manifest, its SHA-256)."""
    path = OUT / MANIFEST_NAME
    man = read_json(path)
    if man is None:
        raise SystemExit(f"refused: no code manifest at {path}; write it (write_manifest.py) before any run")
    if man.get("spec_sha256") != spec_sha:
        raise SystemExit("refused: the code manifest names another spec; write a new, labelled manifest")
    now = code_hashes(tuple(man["files"]))
    bad = [n for n, h in man["files"].items() if now[n] != h]
    copies = [n for n, h in man["files"].items()
              if not (OUT / "code" / n).exists() or file_sha256(OUT / "code" / n) != h]
    if bad or copies:
        raise SystemExit("refused: the code differs from the manifest (" + ", ".join(bad + [f"code/{n}" for n in copies])
                         + "); any change needs a new, labelled manifest")
    return man, file_sha256(path)


def check_pins(spec):
    """The pinned imports (room harness, room manifest, room spec, fallback.py) and the gate files are those the spec
    names."""
    H = load_room_harness()
    p = pins()
    for k in ("room_manifest_sha256", "room_harness_sha256", "room_spec_sha256", "fallback_py_sha256"):
        if spec["pinned"][k] != p[k]:
            raise SystemExit(f"refused: {k} is {p[k]}, the spec pins {spec['pinned'][k]}")
    if file_sha256(PLAN_PATH) != spec["plan"]["sha256"]:
        raise SystemExit("refused: the plan changed since the spec")
    now = {rel: file_sha256(ROOT / rel) for rel in H.GATE_FILES}
    bad = [k for k, v in spec["gate_files_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit("refused: gate files changed since the spec: " + ", ".join(bad))
    return p


# ------------------------------------------------------------------------------------------------ small statistics

def mean_se(values):
    """(mean, standard error over seeds with ddof 1, n, positive count) of the non-missing values."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return dict(mean=None, se=None, n=0, positive=0, values=list(values))
    m = sum(v) / len(v)
    se = (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))) if len(v) > 1 else None
    return dict(mean=m, se=se, n=len(v), positive=sum(1 for x in v if x > 0), values=list(values))
