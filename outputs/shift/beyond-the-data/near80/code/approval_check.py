"""Near-80 (version 8): the one approval check every phase calls, the git identity refusal, and the test fixture.

FINAL, awaiting the owner's approval. Nothing here trains, fits or reads a result. Standard library only, except that
the git rules are version 6's own (outputs/shift/beyond-the-data/stage2/code/stage2_common.py), executed from its
verified bytes under the module name 'near80_v6_stage2_common' and called, never copied.

What it checks (require_approval, in this order):
  1. prereg.json: matches its '.sha256' sidecar; kind "near80-preregistration"; status "final, awaiting approval";
     approvable; nothing missing; its 'constants' block equals near80_constants.py now.
  2. The code manifest prereg.json pins (code_manifest.json and its sidecar), and every file it lists, unchanged; no
     unlisted file in code/.
  3. Every repository file the manifest pins (repository_files) and every pin of prereg.json ('pins') unchanged.
  4. The approval chain (verify_chain): the owner's approval_near80_<first 16>.json (typed at the owner's own terminal
     with approve_local_near80.py) approves THIS prereg.json; the session's import record import_near80_<first 16>.json
     names it; and, with git (read-only queries): exactly one commit on origin's branch added the approval file, its
     author and committer are the owner (35092758+dbayha7@users.noreply.github.com) and neither is the session's own
     git identity nor the configured user.email of the machine running the check, and the import record is on
     origin's branch byte for byte.
Every refusal raises ApprovalRefused with a message that starts with "refused: ". Git does not authenticate authors:
the owner checks on GitHub that the commit the import record names is theirs.

The test fixture (write_test_fixture, fixture_config) is a folder outside the repository holding the marker file
NEAR80_TEST_FIXTURE and synthetic files only. In a fixture git and the repository pins are skipped; every sidecar, the
manifest and the body rules are still checked. A configuration that skips git or pins is refused unless it points at
such a fixture.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402

# Version 6's stage2_common.py, pinned (the git rules near-80 reuses by calling them).
V6_STAGE2_COMMON_REL = "outputs/shift/beyond-the-data/stage2/code/stage2_common.py"
V6_STAGE2_COMMON_SHA256 = "dead5e3eb206aa8aa94694a2f1a8a4b552f3d151771e852674e1b4873f8a2bdc"
V6_MODULE_NAME = "near80_v6_stage2_common"

# The pins prereg.json records (section 'pins'), repository-relative. Checked again by every phase.
PINS_REL = {
    "s1_results_sha256": "outputs/shift/beyond-the-data/restructure/step1/results/results.json",
    "s2_results_sha256": "outputs/shift/beyond-the-data/restructure/step2_results/results.json",
    "s1_prespec_sha256": "outputs/shift/beyond-the-data/restructure/step1/prespec.json",
    "s2_prespec_sha256": "outputs/shift/beyond-the-data/restructure/step2/prespec.json",
    "s1_code_manifest_sha256": "outputs/shift/beyond-the-data/restructure/step1/code_manifest.json",
    "s2_code_manifest_sha256": "outputs/shift/beyond-the-data/restructure/step2/code_manifest.json",
    "v6_prereg_sha256": "outputs/shift/beyond-the-data/stage2/prereg.json",
    "v6_stage2_common_sha256": V6_STAGE2_COMMON_REL,
    "gate_prereg_sha256": "runs/shift/preregistration.json",
}

STATEMENT = ("I approve the near-80 pre-registration ({version}) whose SHA-256 is {sha}, and only that exact file. "
             "Approving it accepts every default of its section 16 and allows the runs it names, under its rules.")

_AUTHOR_RE = re.compile(r"\s*(\S.*?)\s*<([^<>\s]+@[^<>\s]+)>\s*")


class ApprovalRefused(Exception):
    """Every refusal of the approval chain. The message starts with 'refused: '."""

    def __init__(self, message):
        message = str(message)
        if not message.startswith("refused: "):
            message = "refused: " + message
        super().__init__(message)


# --------------------------------------------------------------------------------------------- JSON and hashes

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plain(x):
    """numpy -> Python, recursively; non-finite floats -> None; paths -> strings (version 7's rule)."""
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
    """Strict sorted JSON (version 7's v7_common.canonical): indent 1, sorted keys, no NaN, ASCII only."""
    return json.dumps(plain(payload), indent=1, sort_keys=True, allow_nan=False)


def canonical_constants():
    """near80_constants as plain JSON values (the 'constants' block of prereg.json)."""
    return json.loads(canonical(K.as_dict()))


def sidecar_digest(path):
    """The digest in '<path>.sha256' (its first word), or None when the sidecar is missing or empty."""
    side = Path(str(path) + ".sha256")
    if not side.is_file():
        return None
    words = side.read_text(encoding="utf8", errors="replace").split()
    return words[0] if words else None


def write_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise ApprovalRefused(f"{path} exists; it is written once") from None
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_once(path, payload):
    """Strict sorted JSON and its '<name>.sha256' sidecar, each created once. Returns the SHA-256 of the text."""
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if path.exists() or side.exists():
        raise ApprovalRefused(f"{path} exists; it is written once")
    text = canonical(payload).encode("utf8")
    digest = sha256_bytes(text)
    write_once(path, text)
    write_once(side, f"{digest}  {path.name}\n".encode("utf8"))
    return digest


def read_json_verified(path, what=None):
    """A JSON file whose sidecar matches; ApprovalRefused on a missing file, a missing or wrong sidecar, or bad JSON."""
    path = Path(path)
    what = what or path.name
    if not path.is_file():
        raise ApprovalRefused(f"{what} is missing ({path})")
    raw = path.read_bytes()
    side = sidecar_digest(path)
    if side is None:
        raise ApprovalRefused(f"{what} has no '.sha256' sidecar ({path})")
    if side != sha256_bytes(raw):
        raise ApprovalRefused(f"{what} differs from its '.sha256' sidecar ({path}): it was changed")
    try:
        return json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise ApprovalRefused(f"{what} is not JSON ({path})") from None


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inside(path, folder):
    p, f = Path(os.path.abspath(path)), Path(os.path.abspath(folder))
    return p == f or f in p.parents


# --------------------------------------------------------------------------------------------- names

def approval_name(prefix):
    return K.APPROVAL_NAME.format(prefix=prefix)


def import_name(prefix):
    return K.IMPORT_NAME.format(prefix=prefix)


def approval_rel(prefix):
    """The owner's approval file, relative to the repository root."""
    return K.N80_REL + "/" + approval_name(prefix)


def import_rel(prefix):
    """The import record, relative to the repository root."""
    return K.N80_REL + "/" + import_name(prefix)


# --------------------------------------------------------------------------------------------- configuration

@dataclass
class ApprovalConfig:
    n80_root: Path = K.N80          # where prereg.json, the manifest, the approval and the import live
    repo: Path = K.ROOT             # the git repository (git -C) and the root of repository_files
    check_git: bool = True          # False only in a test fixture
    check_pins: bool = True         # repository_files and the pins; False only in a test fixture
    branch: str = K.BRANCH
    fixture: Optional[Path] = None  # set by fixture_config()

    def __post_init__(self):
        self.n80_root = Path(self.n80_root)
        self.repo = Path(self.repo)
        if self.fixture is not None:
            self.fixture = Path(self.fixture)


@dataclass
class Prereg:
    path: Path
    sha256: str
    payload: dict = field(repr=False)

    @property
    def prefix(self):
        return self.sha256[:K.PREFIX_LEN]


def _marker_ok(folder):
    marker = Path(folder) / K.TEST_MARKER
    try:
        return marker.is_file() and marker.read_text(encoding="utf8") == K.TEST_MARKER_TEXT
    except OSError:
        return False


def _require_sound_config(cfg):
    """A configuration that skips git or the pins must point at a test fixture outside the repository."""
    if cfg.check_git and cfg.check_pins and cfg.fixture is None:
        return
    fx = cfg.fixture
    if fx is None:
        raise ApprovalRefused("a configuration that skips git or the pins is allowed only for a test fixture")
    if inside(fx, K.ROOT):
        raise ApprovalRefused(f"a test fixture must lie outside {K.ROOT} ({fx})")
    if not _marker_ok(fx):
        raise ApprovalRefused(f"{fx} is not a near-80 test fixture (no {K.TEST_MARKER} marker)")
    if Path(os.path.abspath(cfg.n80_root)) != Path(os.path.abspath(fx)):
        raise ApprovalRefused("a test fixture's configuration must read its own folder")


# --------------------------------------------------------------------------------------------- the checks

def load_prereg(cfg=None):
    """prereg.json against its sidecar; kind, label, status FINAL, approvable, nothing missing."""
    cfg = cfg or ApprovalConfig()
    path = Path(cfg.n80_root) / K.PREREG_NAME
    if not path.is_file():
        raise ApprovalRefused(f"no near-80 pre-registration at {path} (it is written by freeze.py after integration)")
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    side = sidecar_digest(path)
    if side is None:
        raise ApprovalRefused(f"{path} has no '.sha256' sidecar")
    if side != digest:
        raise ApprovalRefused(f"{path} differs from its '.sha256' sidecar (the file was changed)")
    try:
        payload = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise ApprovalRefused(f"{path} is not JSON") from None
    if not isinstance(payload, dict) or payload.get("kind") != K.PREREG_KIND:
        raise ApprovalRefused(f"{path} is not the near-80 pre-registration (kind {K.PREREG_KIND!r})")
    if payload.get("label") != K.LABEL:
        raise ApprovalRefused(f"{path} names another study (its label differs)")
    if payload.get("status") != K.STATUS_FINAL:
        raise ApprovalRefused(f"{path} is not final (status {payload.get('status')!r}); a draft cannot be approved")
    if payload.get("approvable") is not True or payload.get("missing") != []:
        raise ApprovalRefused(f"{path} is not approvable (missing: {payload.get('missing')!r})")
    return Prereg(path, digest, payload)


def check_constants(prereg):
    """prereg.json's 'constants' block equals near80_constants.py now."""
    want = canonical_constants()
    got = prereg.payload.get("constants")
    if not isinstance(got, dict):
        raise ApprovalRefused("prereg.json records no constants")
    differ = sorted(k for k in set(want) | set(got) if canonical(want.get(k)) != canonical(got.get(k)))
    if differ:
        raise ApprovalRefused(f"near80_constants.py differs from prereg.json's constants: {differ}")
    return want


def check_code(prereg, cfg=None):
    """The code manifest prereg.json pins, its sidecar, and every near-80 file it lists, unchanged. Returns it."""
    cfg = cfg or ApprovalConfig()
    root = Path(cfg.n80_root)
    code = prereg.payload.get("code") or {}
    path = root / K.MANIFEST_NAME
    manifest = read_json_verified(path, K.MANIFEST_NAME)
    digest = file_sha256(path)
    if digest != code.get("code_manifest_sha256"):
        raise ApprovalRefused(f"{K.MANIFEST_NAME} is not the manifest prereg.json pins")
    if not isinstance(manifest, dict) or manifest.get("kind") != K.MANIFEST_KIND:
        raise ApprovalRefused(f"{K.MANIFEST_NAME} is not the near-80 code manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or files != code.get("files"):
        raise ApprovalRefused(f"the files {K.MANIFEST_NAME} lists differ from the ones prereg.json records")
    bad = []
    for rel, want in sorted(files.items()):
        p = root / rel
        if not p.is_file():
            bad.append(f"{rel} is missing")
        elif file_sha256(p) != want:
            bad.append(f"{rel} differs")
    code_dir = root / "code"
    for d in (code_dir / "__pycache__", root / "__pycache__"):
        if d.exists():
            # Python would import bytecode found there (when its recorded source time and size match) before any
            # hash check could see it, so verified source could be replaced at run time by unverified bytecode.
            bad.append(f"{d.relative_to(root)} exists; remove it (the phases run with PYTHONDONTWRITEBYTECODE=1)")
    if code_dir.is_dir():
        for p in sorted(code_dir.rglob("*.pyc")):
            bad.append(f"{p.relative_to(root)} is compiled bytecode; remove it")
        for p in sorted(code_dir.iterdir()):
            if p.name != "__pycache__" and f"code/{p.name}" not in files:
                bad.append(f"code/{p.name} is not in the manifest")
    if bad:
        raise ApprovalRefused("near-80 files differ from the pinned code manifest: " + "; ".join(bad))
    return manifest


def check_pins(manifest, cfg=None):
    """Every repository file the manifest pins (repository_files) unchanged. Returns how many were checked."""
    cfg = cfg or ApprovalConfig()
    repo = Path(cfg.repo)
    files = manifest.get("repository_files")
    if not isinstance(files, dict):
        raise ApprovalRefused(f"{K.MANIFEST_NAME} lists no repository files")
    absent = [rel for rel in K.PIN_FILES if rel not in files]
    if absent:
        raise ApprovalRefused(f"{K.MANIFEST_NAME} does not pin {absent}")
    bad = []
    for rel, want in sorted(files.items()):
        p = repo / rel
        if not p.is_file():
            bad.append(f"{rel} is missing")
        elif file_sha256(p) != want:
            bad.append(f"{rel} differs")
    if bad:
        raise ApprovalRefused("repository files differ from the pinned code manifest: " + "; ".join(bad))
    return len(files)


def current_pins(repo=K.ROOT):
    """The SHA-256 of each file of PINS_REL now (ApprovalRefused when one is missing)."""
    out = {}
    for key, rel in PINS_REL.items():
        p = Path(repo) / rel
        if not p.is_file():
            raise ApprovalRefused(f"the pinned file {rel} is missing")
        out[key] = file_sha256(p)
    return out


def check_prereg_pins(prereg, manifest, cfg=None):
    """prereg.json's 'pins' block equals the files now."""
    cfg = cfg or ApprovalConfig()
    recorded = prereg.payload.get("pins")
    if not isinstance(recorded, dict):
        raise ApprovalRefused("prereg.json records no pins")
    now = current_pins(cfg.repo)
    now["repository_files_count"] = len(manifest.get("repository_files") or {})
    differ = sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
    if differ:
        raise ApprovalRefused(f"pinned files changed since prereg.json was written: {differ}")
    if recorded.get("v6_stage2_common_sha256") != V6_STAGE2_COMMON_SHA256:
        raise ApprovalRefused("prereg.json pins another version 6 stage2_common.py")
    return now


def check_local_approval(raw, prereg):
    """The owner's approval file (approve_local_near80.py's output) approves THIS prereg.json. Returns its body."""
    try:
        body = json.loads(bytes(raw).decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise ApprovalRefused("the approval file is not UTF-8 JSON") from None
    if not isinstance(body, dict):
        raise ApprovalRefused("the approval file is not a JSON object")
    code = prereg.payload.get("code") or {}
    checks = [
        (body.get("kind") == K.LOCAL_KIND, f"it is not a near-80 local approval ({K.LOCAL_KIND})"),
        (body.get("mode") == K.LOCAL_MODE, f"its mode is not {K.LOCAL_MODE}"),
        (body.get("label") == K.LABEL, "it names another study"),
        (body.get("prereg_path") == K.N80_REL + "/" + K.PREREG_NAME, "it names another pre-registration file"),
        (body.get("prereg_sha256") == prereg.sha256, "it approves another version of prereg.json"),
        (body.get("typed") == prereg.prefix, "the typed characters are not exactly the first 16 of the fingerprint"),
        (body.get("stdin_isatty") is True, "it was not typed at a terminal"),
        (bool(body.get("approved_at_utc")), "it holds no time"),
        (body.get("code_manifest_sha256") == code.get("code_manifest_sha256"),
         "it was made against another code manifest"),
        (body.get("approve_local_sha256") == (code.get("files") or {}).get("approve_local_near80.py"),
         "it was made with another approve_local_near80.py"),
        (prereg.sha256 in str(body.get("statement", "")), "its statement does not name this fingerprint"),
    ]
    for ok, why in checks:
        if not ok:
            raise ApprovalRefused(f"the approval file does not approve this pre-registration: {why}")
    return body


def _email(who):
    m = _AUTHOR_RE.fullmatch(str(who or ""))
    return m.group(2).lower() if m else None


def check_approver(record):
    """The positive check: the commit's author AND committer are the owner (case-insensitive email)."""
    for role in ("author", "committer"):
        if _email(record.get(role)) != K.APPROVER_EMAIL.lower():
            raise ApprovalRefused(f"the commit's {role} is {record.get(role)!r}, not the owner "
                                  f"<{K.APPROVER_EMAIL}>")
    return record


def load_v6_common(repo=K.ROOT):
    """Version 6's stage2_common.py from its verified bytes (SHA-256 dead5e3e...), as module
    'near80_v6_stage2_common', with its ROOT set to `repo` for the git queries. Import-light: no JAX, no gate."""
    source = K.ROOT / V6_STAGE2_COMMON_REL
    data = source.read_bytes()
    if sha256_bytes(data) != V6_STAGE2_COMMON_SHA256:
        raise ApprovalRefused(f"{source} is not version 6's pinned stage2_common.py")
    repo = Path(repo)
    name = V6_MODULE_NAME if repo == K.ROOT else f"{V6_MODULE_NAME}_{sha256_bytes(str(repo).encode())[:12]}"
    spec = importlib.util.spec_from_loader(name, loader=None, origin=str(source))
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(source)
    sys.modules[name] = mod
    exec(compile(data, str(source), "exec"), mod.__dict__)
    mod.ROOT = repo
    return mod


def verify_chain(prereg, cfg=None, v6c=None):
    """The approval file, the import record and (with check_git) git, read-only. Returns the facts of section 4.2."""
    cfg = cfg or ApprovalConfig()
    _require_sound_config(cfg)
    if prereg.payload.get("approvable") is not True or prereg.payload.get("status") != K.STATUS_FINAL:
        raise ApprovalRefused("prereg.json is not final")
    section = prereg.payload.get("approval") or {}
    branch = section.get("branch")
    if branch != cfg.branch:
        raise ApprovalRefused(f"prereg.json names the branch {branch!r}, not {cfg.branch!r}")
    if section.get("local_terminal_allowed") is not True or section.get("approver_email") != K.APPROVER_EMAIL:
        raise ApprovalRefused("prereg.json does not accept approval from the owner's own terminal")
    prefix = prereg.prefix
    a_rel, i_rel = approval_rel(prefix), import_rel(prefix)
    root = Path(cfg.n80_root)
    local = root / approval_name(prefix)
    if not local.is_file():
        raise ApprovalRefused(f"no approval of this fingerprint at {local} (the owner approves at their own terminal "
                              f"with approve_local_near80.py)")
    raw = local.read_bytes()
    local_sha = sha256_bytes(raw)
    if sidecar_digest(local) != local_sha:
        raise ApprovalRefused(f"{local.name} differs from its '.sha256' sidecar")
    check_local_approval(raw, prereg)
    imp = root / import_name(prefix)
    if not imp.is_file():
        raise ApprovalRefused(f"no import record at {imp} (written by 'import_approval.py import', then committed "
                              f"and pushed)")
    raw_imp = imp.read_bytes()
    if sidecar_digest(imp) != sha256_bytes(raw_imp):
        raise ApprovalRefused(f"{imp.name} differs from its '.sha256' sidecar")
    try:
        body = json.loads(raw_imp.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise ApprovalRefused(f"{imp.name} is not JSON") from None
    if not isinstance(body, dict):
        raise ApprovalRefused(f"{imp.name} is not a JSON object")
    want = dict(kind=K.IMPORT_KIND, label=K.LABEL, prereg_sha256=prereg.sha256, approval_file=a_rel,
                approval_file_sha256=local_sha, branch=branch)
    bad = sorted(k for k, v in want.items() if body.get(k) != v)
    if bad:
        raise ApprovalRefused(f"the import record does not match this approval ({', '.join(bad)})")
    record = dict(commit=body.get("commit"), author=body.get("author"), committer=body.get("committer"),
                  committed_sha256=local_sha)
    v6c = v6c or load_v6_common(cfg.repo)
    try:
        if cfg.check_git:
            v6c.check_commit_record(record, local_sha, v6c.agent_emails(v6c.git_session_email()))
            check_approver(record)
            v6c.check_commit_in_git(record, a_rel, branch)
            ref = v6c.remote_ref(branch)
            adders = v6c.git_adding_commits(ref, a_rel)
            if adders != [record["commit"]]:
                raise ApprovalRefused(f"exactly one commit on {ref} must add {a_rel}: git found "
                                      f"{'nothing (git failed)' if adders is None else adders}")
            if v6c.git_blob(ref, i_rel) != raw_imp:
                raise ApprovalRefused(f"the import record {i_rel} is not on origin's branch {branch} as written: "
                                      f"commit and push it (with its .sha256) before any phase runs")
        else:
            v6c.check_commit_record(record, local_sha, v6c.AGENT_GIT_EMAILS)
            check_approver(record)
    except v6c.LocalApprovalProblem as problem:
        raise ApprovalRefused(str(problem)) from None
    code = prereg.payload.get("code") or {}
    return {"prereg_sha256": prereg.sha256, "prefix": prefix,
            "code_manifest_sha256": code.get("code_manifest_sha256"),
            "approval_file": a_rel, "approval_sha256": local_sha, "import_file": i_rel,
            "import_sha256": sha256_bytes(raw_imp), "commit": record["commit"], "author": record["author"],
            "fixture": cfg.fixture is not None}


def require_approval(cfg=None):
    """The one gate every near-80 phase passes before anything else: load_prereg, the constants, check_code,
    check_pins and the prereg pins (unless a fixture), verify_chain. Returns the facts every phase record carries."""
    cfg = cfg or ApprovalConfig()
    _require_sound_config(cfg)
    prereg = load_prereg(cfg)
    check_constants(prereg)
    manifest = check_code(prereg, cfg)
    if cfg.check_pins:
        check_pins(manifest, cfg)
        check_prereg_pins(prereg, manifest, cfg)
    return verify_chain(prereg, cfg)


# --------------------------------------------------------------------------------------------- the git identity

def git_identity_emails(repo=K.ROOT):
    """Every user.email git would use in `repo` (system, global and local, through git's own lookup)."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    try:
        done = subprocess.run(["git", "-C", str(repo), "config", "--get-all", "user.email"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError, ValueError):
        raise ApprovalRefused("git could not be asked for its user.email") from None
    text = done.stdout.decode("utf8", "replace")
    if done.returncode == 1 and not text.strip():
        return []
    if done.returncode != 0:
        raise ApprovalRefused(f"git could not read its configuration in {repo} (exit status {done.returncode})")
    return [line.strip() for line in text.splitlines() if line.strip()]


def refuse_git_identity(repo=K.ROOT):
    """Version 7's require_no_identity, in Python: refuses when git's user.email here is the owner's or an agent's.
    An unset identity passes."""
    banned = [K.APPROVER_EMAIL.lower()] + [e.lower() for e in K.AGENT_EMAILS]
    for email in git_identity_emails(repo):
        if any(b in email.lower() for b in banned):
            raise ApprovalRefused(f"git's user.email here is {email}. Remove it (git config --global --unset "
                                  f"user.email) and commit later with git -c user.name=... -c user.email=... commit")


# --------------------------------------------------------------------------------------------- the test fixture

def fixture_config(fixture_dir):
    """The configuration of a test fixture: the marker file is required; git and the pins are skipped."""
    p = Path(os.path.abspath(fixture_dir))
    if inside(p, K.ROOT):
        raise ApprovalRefused(f"a test fixture must lie outside {K.ROOT} ({p})")
    if not _marker_ok(p):
        raise ApprovalRefused(f"{p} is not a near-80 test fixture (no {K.TEST_MARKER} marker)")
    return ApprovalConfig(n80_root=p, repo=p, check_git=False, check_pins=False, branch=K.BRANCH, fixture=p)


def _load_claims_text(code_root):
    path = Path(code_root) / "claims_text.py"
    if not path.is_file():
        return None
    name = f"near80_fixture_claims_text_{sha256_bytes(str(path).encode())[:12]}"
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def approval_section(branch=K.BRANCH):
    """prereg.json's 'approval' block (also used by near80_prereg)."""
    return {
        "route": ("the owner's own terminal, as versions 6 and 7: approve_local_near80.py writes the approval file, "
                  "the owner commits and pushes it under their own GitHub identity, the session imports it with "
                  "code/import_approval.py import and pushes the import record"),
        "script": "approve_local_near80.py",
        "approval_file": K.APPROVAL_NAME.format(prefix="<first 16 characters>"),
        "import_record": K.IMPORT_NAME.format(prefix="<first 16 characters>"),
        "local_terminal_allowed": True,
        "approver_email": K.APPROVER_EMAIL,
        "branch": branch,
        "accepts_defaults": True,
        "import_guard": ("every near-80 phase refuses without the import record on origin's branch; the import refuses "
                         "if the approval commit's author or committer is the session's own git identity or the "
                         "configured user.email of the machine running it, or is not the owner; the desktop runner "
                         "refuses if a git identity is set there"),
        "never": "no agent writes the approval file or makes that commit; no message counts as approval",
    }


def write_test_fixture(fixture_dir, code_root=K.CODE):
    """A synthetic, final, approved and imported near-80 folder outside the repository, for stub runs and tests.
    It copies the CURRENT files of code_root (and its parent's top-level files) that exist, writes a matching
    code_manifest.json, prereg.json, approval file and import record (commit '0' * 40, author and committer the
    owner), and returns fixture_config(fixture_dir). Nothing here is a real approval."""
    d = Path(os.path.abspath(fixture_dir))
    if inside(d, K.ROOT):
        raise ApprovalRefused(f"a test fixture must lie outside {K.ROOT} ({d})")
    code_root = Path(code_root)
    d.mkdir(parents=True, exist_ok=True)
    if (d / K.PREREG_NAME).exists() or (d / K.MANIFEST_NAME).exists():
        raise ApprovalRefused(f"{d} already holds a fixture (it is written once)")
    write_once(d / K.TEST_MARKER, K.TEST_MARKER_TEXT.encode("utf8"))
    files = {}
    for name in K.TOP_FILES:
        src = code_root.parent / name
        if src.is_file():
            shutil.copyfile(src, d / name)
            files[name] = file_sha256(d / name)
    (d / "code").mkdir(exist_ok=True)
    for name in K.CODE_FILES:
        src = code_root / name
        if src.is_file():
            shutil.copyfile(src, d / "code" / name)
            files["code/" + name] = file_sha256(d / "code" / name)
    manifest = dict(kind=K.MANIFEST_KIND, label=K.LABEL, files=files, repository_files={}, history=[],
                    reason="test fixture: synthetic files only", fixture=True, written=utc_now(),
                    rule="test fixture: git and the repository pins are not checked")
    msha = write_json_once(d / K.MANIFEST_NAME, manifest)
    claims = None
    mod = _load_claims_text(code_root)
    doc = code_root.parent / "PREREGISTRATION.md"
    if mod is not None and doc.is_file():
        claims = mod.extract_section14(doc.read_text(encoding="utf8"))
    prereg = dict(kind=K.PREREG_KIND, version=K.VERSION, label=K.LABEL, status=K.STATUS_FINAL,
                  status_line=K.STATUS_LINE_FINAL, approvable=True, missing=[], owner_answer=K.OWNER_ANSWER,
                  constants=canonical_constants(), claims_word_for_word=claims,
                  code=dict(code_manifest=K.MANIFEST_NAME, code_manifest_sha256=msha, files=files),
                  approval=approval_section(), pins={}, fixture=True, test_marker=K.TEST_MARKER,
                  runner=dict(entry="run_near80.sh", lanes_default=K.DEFAULT_LANES,
                              outputs="runs/shift_explore/near80/", phases=list(K.PHASE_KINDS)))
    psha = write_json_once(d / K.PREREG_NAME, prereg)
    prefix = psha[:K.PREFIX_LEN]
    approval = dict(kind=K.LOCAL_KIND, label=K.LABEL, mode=K.LOCAL_MODE, prereg_path=K.N80_REL + "/" + K.PREREG_NAME,
                    prereg_sha256=psha, typed=prefix, stdin_isatty=True, tty=None, approved_at_utc=utc_now(),
                    hostname="test-fixture", local_user=None, git_head=None, git_branch=None,
                    python=platform.python_version(), code_manifest_sha256=msha,
                    approve_local_sha256=files.get("approve_local_near80.py"),
                    statement=STATEMENT.format(version=K.VERSION, sha=psha),
                    how="test fixture: synthetic files only; not an approval", fixture=True)
    asha = write_json_once(d / approval_name(prefix), approval)
    who = f"{K.APPROVER_NAME} <{K.APPROVER_EMAIL}>"
    record = dict(kind=K.IMPORT_KIND, label=K.LABEL, prereg_sha256=psha, prereg_version=K.VERSION,
                  approval_file=approval_rel(prefix), approval_file_sha256=asha, commit="0" * 40, author=who,
                  committer=who, parents=[], signature_status=None, branch=K.BRANCH, remote_tip_at_import=None,
                  session_git_email=None, approver_email=K.APPROVER_EMAIL, imported_at=utc_now(),
                  note="test fixture: synthetic files only; not an import", fixture=True)
    write_json_once(d / import_name(prefix), record)
    return fixture_config(d)
