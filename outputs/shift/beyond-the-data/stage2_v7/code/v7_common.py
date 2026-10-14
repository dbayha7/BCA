"""Stage 2, version 7: shared plumbing for the v7 code (V7_SPEC.md). Standard library only at import time.

What lives here:
  - the fixed paths (the spec's shorthands S2/, S2C/, V7/, R7/, OUT/, the scratchpad folder launch.sh runs from) and the
    pins v7 holds of version 6 (its pre-registration, code manifest, approval, import record and commits);
  - strict sorted JSON with a '<name>.sha256' sidecar, written once (the gate's and v6's format);
  - read_record(): a record read without ever stopping the caller: 'ok', 'missing' or 'partial' (a JSON whose sidecar
    is missing or wrong), as C5 asks of the runner;
  - V7Refusal, the one exception every v7 check raises before the hand-over; the wrapper turns it into exit 97;
  - load_v6_common(): v6's stage2_common.py executed from its pinned bytes under the name 'v7_v6_stage2_common' (never
    under its own name, so that a process with no re-make entry has not imported 'stage2_common' before the hand-over,
    V7_SPEC C7 "Tested by"). The approval rules v7 reuses verbatim (S2C/stage2_common.py:83, :483-521, :524-549) are
    called from that module, never copied.

Nothing here edits, moves or deletes a pinned file.
"""

import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path("/home/user/bca")
V7_REL = "outputs/shift/beyond-the-data/stage2_v7"
V7 = ROOT / V7_REL
CODE = V7 / "code"
R7 = ROOT / "runs" / "shift_explore" / "stage2_v7"
OUT = ROOT / "runs" / "shift_explore" / "stage2"
S2_REL = "outputs/shift/beyond-the-data/stage2"
S2 = ROOT / S2_REL
S2C = S2 / "code"
SCRATCH = Path("/tmp/claude-0/-home-user/9e341792-006d-5a11-a8e9-8fac16ac7f5b/scratchpad")
S_FOLDER = SCRATCH / "ood" / "stage2"          # S2C/launch.sh:10 (the folder the v6 code runs from)
FALLBACK_PY = SCRATCH / "ood" / "fallback" / "fallback.py"
PYTHON = "/tmp/claude-0/bcavenv/bin/python"     # S2C/launch.sh:19
GATE_RUNS = ROOT / "runs" / "shift"
GATE_PREREG_SHA256 = "68d9e8779e148b7bfd2fcc6dc32e0a1091e5bfbbcb83cef2e4b58e4f6ec60fab"
GATE_VARIANT_DIR = GATE_RUNS / "68d9e8779e148b7b"

LABEL = ("Stage 2, version 7: the remaining Stage 2 work on the user's laptop (operational changes only; the method, "
         "marks, analysis and claim are version 6's)")
PREREG_KIND = "stage2-v7-preregistration"
MANIFEST_KIND = "stage2-v7-code-manifest"
LOCAL_KIND = "stage2-v7-local-approval"
IMPORT_KIND = "stage2-v7-import-record"
LOCAL_MODE = "local-terminal"
PREFIX_LEN = 16
APPROVAL_NAME = "approval_v7_{prefix}.json"
IMPORT_NAME = "import_v7_{prefix}.json"

# Version 6, as approved (V7_SPEC section 1 and C10). Every value is checked against the files when v7 writes or reads.
V6 = dict(
    prereg="outputs/shift/beyond-the-data/stage2/prereg.json",
    prereg_sha256="9b38c7c48a303cfd67e5db6f137e3a162997bb533d86b3f7647e0467b91ca3b1",
    code_manifest="outputs/shift/beyond-the-data/stage2/code_manifest.json",
    code_manifest_sha256="4b34af991a56066c41fbb55cb02db4074ae0b7e85b1e50802a53983edda2a753",
    approval_file="outputs/shift/beyond-the-data/stage2/approval_9b38c7c48a303cfd.json",
    approval_file_sha256="f7c96f77bfa32ab298430f5bec0be699258347a6a83d668eb947296bdfb3ed5c",
    import_record="outputs/shift/beyond-the-data/stage2/import_9b38c7c48a303cfd.json",
    import_record_sha256="92c1c2be64af813b6ac9b0cf18702af71e5074f1b66937759721149d8f7e788e",
    approval_commit="7dab7d545c49e0310176fc4f00b6a41492b30ea6",
    import_commit="2eab3c4513cee5988554bb0f9ba890f22a61e236",
    runs_side_approval="runs/shift_explore/stage2/approval.json",
    runs_side_approval_sha256="b78b87726a2e7030dfb45a0ad45d86cbae19cc46fb9f4b8d9724975cfb6c2e72",
)
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"   # v6's approver (the import record above)
BRANCH = "claude/main-bca-connection-x4o46s"
CONSTANTS_REL = V7_REL + "/world_constants.json"
CONSTANTS_SHA256 = "6602f9d17e977a3e1f6eb017eb1f0b9ec80b018041d62146f507983f36a5e379"

# launch_v7.sh's exact environment (V7_SPEC C4): these eight and nothing else.
ENV_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
PHASE_ENV = {"PATH": ENV_PATH, "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONPATH": str(ROOT),
             "PYTHONDONTWRITEBYTECODE": "1", "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1"}

EXIT_V7 = 97                          # a v7 refusal (C1, C5, C7): launch_v7.sh and the runner halt the run on it
REMAKE_SEEDS = (2, 3, 4)
TOL_A, TOL_K = 1e-14, 1e-12           # C1 step 2
W0_TOL = 1e-5                         # check (b)'s tolerance (runs/shift_explore/room/code/harness.py:89)


class V7Refusal(Exception):
    """A v7 check failed. Before the hand-over the wrapper writes one ledger line and exits 97 (never a traceback)."""


# --------------------------------------------------------------------------------------------- JSON and hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(x):
    """numpy -> Python, recursively; non-finite floats -> None (v6's rule, S2C/stage2_common.py:196-223)."""
    np = sys.modules.get("numpy")
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if np is not None:
        if isinstance(x, np.ndarray):
            return plain(x.tolist())
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            x = float(x)
    if isinstance(x, bool) or isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def canonical(payload):
    return json.dumps(plain(payload), indent=1, sort_keys=True, allow_nan=False)


def block_sha256(block):
    """The SHA-256 of one block of a JSON payload, in canonical form (used to bind probe records to the draft's
    re-make block, C7 'Probe records')."""
    return sha256_bytes(canonical(block).encode("utf8"))


def write_once(path, data):
    """Create a file with these bytes; refuse (V7Refusal) when it exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise V7Refusal(f"{path} exists; it is written once") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path, payload):
    """Strict sorted JSON and its '<name>.sha256' sidecar, each created once. Returns the SHA-256 of the text."""
    path = Path(path)
    side = path.with_name(path.name + ".sha256")
    if path.exists() or side.exists():
        raise V7Refusal(f"{path} exists; it is written once")
    text = canonical(payload).encode("utf8")
    digest = sha256_bytes(text)
    write_once(path, text)
    write_once(side, f"{digest}  {path.name}\n".encode("utf8"))
    return digest


def sidecar_digest(path):
    side = Path(str(path) + ".sha256")
    if not side.exists():
        return None
    words = side.read_text(encoding="utf8", errors="replace").split()
    return words[0] if words else None


def read_json_verified(path, what=None):
    """A JSON whose sidecar matches; V7Refusal otherwise (missing file, missing or wrong sidecar, not JSON)."""
    path = Path(path)
    what = what or str(path)
    if not path.exists():
        raise V7Refusal(f"{what} is missing ({path})")
    raw = path.read_bytes()
    if sidecar_digest(path) != sha256_bytes(raw):
        raise V7Refusal(f"{what} differs from its SHA-256 sidecar ({path})")
    try:
        return json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise V7Refusal(f"{what} is not JSON ({path})") from None


def read_record(path):
    """('ok', payload) | ('missing', None) | ('partial', reason). Never raises and never exits: a torn sidecar or a
    damaged JSON is reported as 'partial' (C5: 'the runner never exits on a damaged record')."""
    path = Path(path)
    try:
        if not path.exists():
            return "missing", None
        raw = path.read_bytes()
        side = sidecar_digest(path)
        if side is None:
            return "partial", "no sidecar"
        if side != sha256_bytes(raw):
            return "partial", "sidecar differs"
        return "ok", json.loads(raw.decode("utf8"))
    except Exception as error:  # noqa: BLE001 - reported, never raised (C5)
        return "partial", f"{type(error).__name__}: {error}"


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_stamp():
    """A folder-safe UTC time with microseconds (R7/interrupted/<phase>/<UTC time>/)."""
    t = time.time()
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime(t)) + f".{int((t % 1) * 1e6):06d}Z"


def boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def prefix_of(sha256):
    return str(sha256)[:PREFIX_LEN]


def inside(path, folder):
    p, f = Path(os.path.abspath(path)), Path(os.path.abspath(folder))
    return p == f or f in p.parents


# --------------------------------------------------------------------------------------------- version 6, read-only

_V6_COMMON = {}


def v6_manifest(root=ROOT):
    """S2/code_manifest.json, checked against its sidecar and v6's pinned SHA-256 (V7Refusal otherwise)."""
    path = Path(root) / V6["code_manifest"]
    raw = path.read_bytes()
    if sha256_bytes(raw) != V6["code_manifest_sha256"] or sidecar_digest(path) != V6["code_manifest_sha256"]:
        raise V7Refusal(f"{path} is not v6's pinned code manifest {V6['code_manifest_sha256'][:16]}...")
    return json.loads(raw.decode("utf8"))


def load_v6_common(source=None):
    """v6's stage2_common.py (S2C, byte-identical to the copy launch.sh runs), executed from its pinned bytes as the
    module 'v7_v6_stage2_common'. Import-light: no JAX and no gate import (S2C/stage2_common.py:5-6)."""
    if "mod" in _V6_COMMON:
        return _V6_COMMON["mod"]
    source = Path(source or (S2C / "stage2_common.py"))
    pinned = v6_manifest()["files"]["stage2_common.py"]
    data = source.read_bytes()
    if sha256_bytes(data) != pinned:
        raise V7Refusal(f"{source} is not v6's pinned stage2_common.py")
    name = "v7_v6_stage2_common"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(source))
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(source)
    sys.modules[name] = mod
    exec(compile(data, str(source), "exec"), mod.__dict__)
    _V6_COMMON["mod"] = mod
    return mod
