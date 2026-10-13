"""EXPLORATORY (Stage 1 of outputs/shift/beyond-the-data/PLAN.md, section 4.4: the detection rehearsal, no BCA).
Outside the gate; changes no pre-registered rule. NO BCA: no band, calibrator, threshold, dose or BCA update is
fitted, read or called. Nothing from algorithms/td3_bc_bca.py or learners.bca_* is called; from calibration/ only
calibration/bank.stratified_bank runs, because the gate's data loader (experiments/shift/dataset.py, through
acceptance.load_seed) calls it to regenerate the log's calibration-bank split (a NumPy row sampler, not a BCA signal).

Shared plumbing of the detect component. Import-light on purpose: it holds paths, hashing, the pinned room-harness
import, the gate loaders' context and the output guard. It holds NO signal and NO label: the label script
(labels.py) and the signal script (signals.py) both import it, and neither can reach the other through it.

  - the room harness (runs/shift_explore/room/code/harness.py) is imported read-only by executing its exact bytes,
    only when their SHA-256 equals the one the room's code manifest records, the manifest matches its sidecar and the
    SHA-256 pinned here, the manifest names the pinned room spec, and the scratchpad source (when present) has the
    same SHA-256. Any mismatch refuses. Nothing is compiled to disk (sys.dont_write_bytecode).
  - outputs go under runs/shift_explore/detect (never runs/shift, never experiments/, never the room's folder);
    Stage 2 may write its BCA width scores and their measures under runs/shift_explore/stage2 through the declared
    hook. Strict sorted JSON with a '<name>.sha256' sidecar (the gate's format); arrays as .npz with a file sidecar
    and a content SHA-256 (learners.tree_sha256 of the arrays, which does not depend on the zip container).
"""

import hashlib
import io
import json
import sys
import time
import types
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

ROOT = Path("/home/user/bca")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "shift_explore" / "detect"
STAGE2_OUT = ROOT / "runs" / "shift_explore" / "stage2"
TEST_OUT = HERE / "_test_out"          # synthetic tests only (test_detect.py); never data
SPEC_PATH = OUT / "spec.json"
MANIFEST_NAME = "code_manifest.json"
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"
CODE_FILES = ("common.py", "pairs.py", "labels.py", "signals.py", "measures.py", "detect.py", "run_detect.py",
              "write_spec.py", "write_manifest.py", "launch.sh", "checks_seed0.py", "test_detect.py")

ROOM_OUT = ROOT / "runs" / "shift_explore" / "room"
ROOM_MANIFEST = ROOM_OUT / "code_manifest.json"
ROOM_SPEC = ROOM_OUT / "spec.json"
ROOM_HARNESS_COPY = ROOM_OUT / "code" / "harness.py"
ROOM_HARNESS_SOURCE = HERE.parent / "room" / "harness.py"
# The room test's code as pinned when this component was written (its manifest v1, spec v2). Any change refuses.
ROOM_PINNED = dict(manifest_sha256="469ada26b9af7eff33be17b8241867e83e44c45a30d2d137dffe7deca3f32391",
                   spec_sha256="56bbda24cdbaa10cc0de0261450de5d9034e08e188c88e9432be83d8cf136bf4",
                   harness_sha256="97f547b14d55c3afa392cbdf99508e45c1b862d8a285030808c5783a067ef2cf")

LABEL = ("EXPLORATORY: Stage 1 detection rehearsal (plan section 4.4), computed outside the gate on saved gate "
         "artifacts; not a gated output; changes no pre-registered rule; no BCA band, calibrator, threshold, dose or "
         "BCA update was used (data-only stand-ins and oracles only)")
LABEL_STAGE2 = ("EXPLORATORY/STAGE 2 hook output: BCA width scored by the locked detection script of plan section 4.4 "
                "under the Stage 2 pre-registration; written by Stage 2, not by the Stage 1 rehearsal")

SEEDS = (0, 1, 2, 3, 4)
WORLDS = ("stall", "no_stall")
LEARNER_SEED = 20261015


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def rng(rep, *tags):
    """np.random.default_rng([20261015, seed, *tags]) (learners.rng_np with the gate's learner seed)."""
    return np.random.default_rng([LEARNER_SEED, int(rep), *[int(t) for t in tags]])


def content_sha256(arrays):
    """SHA-256 over a dict of arrays in sorted key order: key, dtype, shape and bytes of each (learners.tree_sha256's
    rule with the key names added). Independent of the .npz container."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(str(k).encode() + b"|" + str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ pinned room harness

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


def _read_verified(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"refused: {path} is missing")
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != sha256_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {path} differs from its SHA-256 sidecar")
    return json.loads(text)


def load_room_harness():
    """The room harness, read-only, from the exact bytes its code manifest pins (module docstring). Refuses on any
    mismatch. Returns the module; room_pins() gives the hashes checked."""
    if "room_harness" in _MODULES:
        return _MODULES["room_harness"]
    man = _read_verified(ROOM_MANIFEST)
    man_sha = file_sha256(ROOM_MANIFEST)
    if man_sha != ROOM_PINNED["manifest_sha256"]:
        raise SystemExit("refused: the room's code manifest is not the one the detect component was pinned to")
    if man.get("kind") != "room-code-manifest" or not str(man.get("label", "")).startswith("EXPLORATORY"):
        raise SystemExit("refused: the room's code manifest is not labelled")
    if man.get("spec_sha256") != ROOM_PINNED["spec_sha256"] or file_sha256(ROOM_SPEC) != ROOM_PINNED["spec_sha256"]:
        raise SystemExit("refused: the room's spec is not the one the detect component was pinned to")
    want = man["files"].get("harness.py")
    if want != ROOM_PINNED["harness_sha256"]:
        raise SystemExit("refused: the room manifest's harness.py is not the pinned one")
    data = ROOM_HARNESS_COPY.read_bytes()
    if sha256_bytes(data) != want:
        raise SystemExit("refused: the room's pinned copy of harness.py differs from its manifest")
    if ROOM_HARNESS_SOURCE.exists() and file_sha256(ROOM_HARNESS_SOURCE) != want:
        raise SystemExit("refused: the room's source harness.py differs from its manifest")
    mod = _exec_module("room_harness", ROOM_HARNESS_COPY, data)
    _MODULES["room_harness"] = mod
    _MODULES["room_pins"] = dict(room_manifest_sha256=man_sha, room_spec_sha256=man["spec_sha256"],
                                 room_harness_sha256=want, room_harness_copy=str(ROOM_HARNESS_COPY))
    return mod


def room_pins():
    load_room_harness()
    return dict(_MODULES["room_pins"])


def gate():
    """(harness, acceptance, learners, recorder, thruster_chain, truth) through the pinned harness's imports."""
    H = load_room_harness()
    return H, H.A, H.L, H.RC, H.TC, H.TR


# ------------------------------------------------------------------------------------------------ outputs

ALLOWED_ROOTS = (OUT, STAGE2_OUT, TEST_OUT)
PROTECTED = (ROOT / "runs" / "shift", ROOT / "experiments", ROOM_OUT, ROOT / "algorithms", ROOT / "calibration")


def guard_out(path, roots=ALLOWED_ROOTS):
    """Refuse any output outside the allowed roots (runs/shift_explore/detect; Stage 2's folder for the BCA hook; the
    synthetic-test folder) or inside a protected one (runs/shift, experiments/, the room's folder, algorithms/,
    calibration/)."""
    p = Path(path).resolve()
    if not any(r.resolve() == p or r.resolve() in p.parents for r in roots):
        raise SystemExit(f"refused: {p} is outside {', '.join(str(r) for r in roots)}")
    for protected in PROTECTED:
        if p == protected.resolve() or protected.resolve() in p.parents:
            raise SystemExit(f"refused: {p} is inside {protected}")
    return p


def plain(x):
    """numpy -> Python, recursively; nonfinite floats -> None."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else None
    return x


def canonical(payload):
    return json.dumps(plain(payload), sort_keys=True, indent=1, allow_nan=False, ensure_ascii=False) + "\n"


def write_json(path, payload, roots=ALLOWED_ROOTS):
    """Strict sorted JSON with a '<name>.sha256' sidecar; returns the SHA-256 of the text."""
    path = guard_out(path, roots)
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


def save_npz(path, arrays, roots=ALLOWED_ROOTS):
    """Arrays as an uncompressed .npz (no pickle) with a '<name>.sha256' sidecar of the file bytes. Returns
    (file SHA-256, content SHA-256)."""
    path = guard_out(path, roots)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for k in sorted(arrays):
            a = np.ascontiguousarray(np.asarray(arrays[k]))
            if a.dtype == object:
                raise ValueError(f"{k}: object arrays are not allowed")
            info = zipfile.ZipInfo(f"{k}.npy", date_time=(2026, 1, 1, 0, 0, 0))   # fixed: the bytes are reproducible
            b = io.BytesIO()
            np.lib.format.write_array(b, a, allow_pickle=False)
            zf.writestr(info, b.getvalue())
    blob = buf.getvalue()
    path.write_bytes(blob)
    digest = sha256_bytes(blob)
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest, content_sha256(arrays)


def load_npz(path, file_sha256_expected=None, content_sha256_expected=None):
    """An .npz written by save_npz, verified against its sidecar (and the expected file and content SHA-256 when
    given). Returns a dict of arrays."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"refused: {path} is missing")
    blob = path.read_bytes()
    got = sha256_bytes(blob)
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != got:
        raise SystemExit(f"refused: {path} differs from its SHA-256 sidecar")
    if file_sha256_expected is not None and got != file_sha256_expected:
        raise SystemExit(f"refused: {path} is not the file its record names")
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    if content_sha256_expected is not None and content_sha256(arrays) != content_sha256_expected:
        raise SystemExit(f"refused: the arrays of {path} are not the ones its record names")
    return arrays


# ------------------------------------------------------------------------------------------------ spec and code

def load_spec():
    spec = read_json(SPEC_PATH)
    if spec is None:
        raise SystemExit(f"refused: no spec at {SPEC_PATH}; it must be written before any run")
    if spec.get("kind") != "detect-spec" or not str(spec.get("label", "")).startswith("EXPLORATORY"):
        raise SystemExit("refused: the detect spec is not labelled")
    return spec, file_sha256(SPEC_PATH)


def code_hashes(names=CODE_FILES):
    return {n: file_sha256(HERE / n) for n in names}


def check_code(spec_sha, out=OUT):
    """Refuse unless the code manifest exists, matches its sidecar, names the current spec, and every file it lists
    (here and in its copy under OUT/code/) still has the SHA-256 it records. Returns (manifest, its SHA-256)."""
    path = out / MANIFEST_NAME
    man = read_json(path)
    if man is None:
        raise SystemExit(f"refused: no code manifest at {path}; write it (write_manifest.py) before any phase")
    if man.get("spec_sha256") != spec_sha:
        raise SystemExit("refused: the code manifest names another spec; write a new, labelled manifest")
    now = code_hashes(tuple(man["files"]))
    bad = [n for n, h in man["files"].items() if now[n] != h]
    copies = [n for n, h in man["files"].items()
              if not (out / "code" / n).exists() or file_sha256(out / "code" / n) != h]
    if bad or copies:
        raise SystemExit("refused: the code differs from the manifest (" + ", ".join(bad + [f"code/{n}" for n in copies])
                         + "); any change needs a new, labelled spec and manifest")
    return man, file_sha256(path)


def check_room_pins(spec):
    pins = room_pins()
    for k in ("room_manifest_sha256", "room_spec_sha256", "room_harness_sha256"):
        if spec["room_pins"][k] != pins[k]:
            raise SystemExit(f"refused: {k} differs from the one the detect spec records")
    return pins


def gate_files_sha256(names):
    return {rel: file_sha256(ROOT / rel) for rel in names}


def check_gate_files(spec):
    now = gate_files_sha256(tuple(spec["gate_files_sha256"]))
    bad = [k for k, v in spec["gate_files_sha256"].items() if now.get(k) != v]
    if bad:
        raise SystemExit("refused: gate files changed since the detect spec: " + ", ".join(bad))


def begin(phase_needs_code=True):
    """Every phase: the spec, the gate files, the room pins and (unless a synthetic test) the code manifest."""
    spec, spec_sha = load_spec()
    check_gate_files(spec)
    pins = check_room_pins(spec)
    man, man_sha = check_code(spec_sha) if phase_needs_code else (None, None)
    return dict(spec=spec, spec_sha256=spec_sha, manifest_sha256=man_sha, room_pins=pins)


# ------------------------------------------------------------------------------------------------ gate artifacts

def context(rep, worlds=WORLDS):
    """The gate's artifacts of one seed through the room harness's SeedContext (the gate's verified loaders:
    acceptance.load_prereg(refuse_protected(RUNS_DEFAULT)), load_seed, get_critic(train=False), the saved cloned
    actor, check 2's saved actors), with none(world): check 2's saved gapped-critic actor of the world ('gapped' in
    the stalling world, 'no_stall_gapped' in the no-stall world). The same interface as Stage 2's GateContext and
    FreshContext (prepared, critic, clone, none, sd, rep)."""
    H = load_room_harness()

    class DetectContext(H.SeedContext):
        role = "exploratory"

        def none(self, world="stall"):
            if "_h2" not in self.__dict__:
                self._h2 = self.h2()
            acts, rec, _ = self._h2
            key = "gapped" if world == "stall" else "no_stall_gapped"
            if rec["failed"].get(key):
                raise SystemExit(f"refused: check 2's {key} actor of seed {self.rep} failed in the gate")
            return acts[key]

    if int(rep) not in SEEDS:
        raise SystemExit(f"refused: seed {rep} is not a gate seed (0-4); fresh seeds come through Stage 2's context")
    return DetectContext(rep, worlds=tuple(worlds))


def world_index(world):
    return WORLDS.index(world)


def seed_dir(rep, world, out=OUT):
    return out / f"seed{int(rep)}" / world
