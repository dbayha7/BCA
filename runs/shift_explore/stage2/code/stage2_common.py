"""Stage 2 (plan section 5, the flags claim): shared plumbing. EXPLORATORY on gate seeds 0-4, confirmatory on fresh
seeds 5-9, under runs/shift_explore/stage2/prereg.json. Outside the gate; changes no rule of the gate's
pre-registration (runs/shift/preregistration.json).

What lives here, and why it is import-light (no JAX, no gate import at module level): the approval check must be
usable by approve.py and by the refusal tests without loading any ThrusterChain data or any network.

  - paths: every output goes under runs/shift_explore/stage2 (never runs/shift, never experiments/, never the room
    test's folder runs/shift_explore/room); strict sorted JSON with a '<name>.sha256' sidecar (the gate's format).
  - the pre-registration (prereg.json): read and verified against its sidecar.
  - the approval (approval.json, written only by approve.py): verified against the CURRENT prereg.json's SHA-256 on
    every call. A mode counts only when the pre-registration accepts it (allowed_modes). Version 6 accepts only
    'local-terminal', the user's choice: typed by the user at a terminal on the user's own machine with
    approve_local.py (one file per fingerprint, approval_<first 16>.json), committed and pushed by the user, then
    imported here by 'approve.py import', which reads the commit facts with read-only git queries and writes an import
    record (import_<first 16>.json) that must be committed and pushed before any BCA run. The modes 'terminal' (typed
    in this cloud session) and 'chat' remain in the code but refuse unless a pre-registration accepts them.
    require_approval() is the one gate every BCA entry point calls first; it also checks that every Stage 2 script
    still has the SHA-256 the pre-registration pins (through the code manifest).
  - git: only read-only queries (git(), below), never a write.
  - pinned imports: the room test's harness (runs/shift_explore/room/code/harness.py) and the fallback diagnostic
    (scratchpad/ood/fallback/fallback.py) are executed from their exact verified bytes; any mismatch refuses.

Nothing here fits, reads or runs any BCA quantity.
"""

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path("/home/user/bca")
HERE = Path(__file__).resolve().parent
SCRATCH_OOD = HERE.parent

LABEL = ("BCA lower-edge consumer (new method version), exploratory on seeds 0-4, confirmatory on fresh seeds 5-9")
LABEL_LONG = ("EXPLORATORY on gate seeds 0-4 and CONFIRMATORY on fresh seeds 5-9: Stage 2 of "
              "outputs/shift/beyond-the-data/PLAN.md (the flags claim), " + LABEL + "; computed outside the gate under "
              "runs/shift_explore/stage2/prereg.json; changes no rule of the gate's pre-registration")
METHOD_ID = "bca-lower-edge-1"
EXPLORATORY_SEEDS = (0, 1, 2, 3, 4)
CONFIRMATORY_SEEDS = (5, 6, 7, 8, 9)

# Chat-mode approval (pre-registration section 9): the user's message must be exactly ONE fixed sentence and nothing
# else, "approve <first 16 characters of prereg.json's SHA-256>, accept chat mode", matched in full, ignoring case,
# spaces at either end or between words, and one optional final full stop. Any other text is refused: a negation, a
# question, a quotation inside a longer message, or an approval with an added condition. The words "accept chat mode"
# record that the user accepts chat mode's weaker evidence. Approving the fingerprint accepts every default the
# pre-registration lists; any other choice needs a new version and a new fingerprint.
CHAT_ACCEPT_PHRASE = "accept chat mode"
CHAT_SENTENCE = "approve {prefix}, accept chat mode"
_CHAT_RE = re.compile(r"\s*approve\s+([0-9a-f]{16})\s*,\s*accept\s+chat\s+mode\s*\.?\s*", re.IGNORECASE)
PREFIX_LEN = 16

# Local-terminal approval (pre-registration section 9; the only mode version 6 accepts). The user runs approve_local.py
# on the user's own machine from the repository root; it writes REPO_STAGE2_REL/approval_<first 16>.json (kind
# LOCAL_KIND, mode LOCAL_MODE) with a '.sha256' sidecar, one file per fingerprint; the user commits and pushes it.
# 'approve.py import' finds, with read-only git queries, the one commit on origin's branch that added it, verifies
# everything, writes OUT/approval.json (mode LOCAL_MODE) with the commit facts, and writes the import record
# REPO_STAGE2_REL/import_<first 16>.json, which the orchestrating agent commits and pushes. verify_approval re-checks
# all of it, with git, at every BCA entry point, and refuses until the import record is on origin's branch.
LOCAL_MODE = "local-terminal"
CLOUD_TERMINAL_MODE = "terminal"
CHAT_MODE = "chat"
LOCAL_KIND = "stage2-local-approval"
IMPORT_KIND = "stage2-import-record"
REPO_STAGE2_REL = "outputs/shift/beyond-the-data/stage2"
LOCAL_APPROVAL_NAME = "approval_{prefix}.json"
IMPORT_RECORD_NAME = "import_{prefix}.json"
LOCAL_APPROVAL_PATTERN = REPO_STAGE2_REL + "/" + LOCAL_APPROVAL_NAME.format(prefix="<first 16 characters>")
IMPORT_RECORD_PATTERN = REPO_STAGE2_REL + "/" + IMPORT_RECORD_NAME.format(prefix="<first 16 characters>")
# The cloud session's own git identity (~/.gitconfig here): a commit authored or committed by it is never taken as the
# user's. The session's configured user.email, read with git at import and at every check, is refused too.
AGENT_GIT_EMAILS = ("noreply@anthropic.com",)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}([0-9a-f]{24})?")
_AUTHOR_RE = re.compile(r"\s*(\S.*?)\s*<([^<>\s]+@[^<>\s]+)>\s*")
_PREFIX_RE = re.compile(r"[0-9a-f]{16}")


def _checked_prefix(prefix):
    if not _PREFIX_RE.fullmatch(str(prefix)):
        raise ValueError(f"not a fingerprint prefix (16 lower-case hexadecimal characters): {prefix!r}")
    return str(prefix)


def local_approval_rel(prefix):
    """The committed local approval of the pre-registration whose fingerprint starts with `prefix`, relative to the
    repository root (one file per fingerprint, so an earlier approval never blocks a later version's)."""
    return REPO_STAGE2_REL + "/" + LOCAL_APPROVAL_NAME.format(prefix=_checked_prefix(prefix))


def import_record_rel(prefix):
    """The import record of that approval, relative to the repository root (committed and pushed after the import)."""
    return REPO_STAGE2_REL + "/" + IMPORT_RECORD_NAME.format(prefix=_checked_prefix(prefix))

# --------------------------------------------------------------------------------------------- pinned outside code

ROOM_OUT = ROOT / "runs" / "shift_explore" / "room"
ROOM_PINNED = dict(
    manifest_sha256="469ada26b9af7eff33be17b8241867e83e44c45a30d2d137dffe7deca3f32391",
    spec_sha256="56bbda24cdbaa10cc0de0261450de5d9034e08e188c88e9432be83d8cf136bf4",
    harness_sha256="97f547b14d55c3afa392cbdf99508e45c1b862d8a285030808c5783a067ef2cf")
FALLBACK_PY = SCRATCH_OOD / "fallback" / "fallback.py"
FALLBACK_PY_SHA256 = "fa09489fa0a8ec77b42d1a6753ddeb9aa6a807cf84bdc42868c7a1701e4129b3"
STANDINS_OUT = ROOT / "runs" / "shift_explore" / "standins"
STANDINS_SRC = SCRATCH_OOD / "standins"
DEPLOY_OUT = ROOT / "runs" / "shift_explore" / "deploy"
DETECT_SRC = SCRATCH_OOD / "detect"
DETECT_OUT = ROOT / "runs" / "shift_explore" / "detect"
PLAN_PATH = ROOT / "outputs" / "shift" / "beyond-the-data" / "PLAN.md"

# The Stage 2 scripts the pre-registration pins (write_manifest.py hashes them and copies them to OUT/code/).
CODE_FILES = ("stage2_common.py", "approve.py", "approve_local.py", "band.py", "weights.py", "policy.py", "deploy_check.py", "host.py",
              "fresh.py", "detect_contract.py", "run_stage2.py", "write_manifest.py", "write_prereg.py",
              "prereg_payload.py", "test_stage2.py", "launch.sh", "PREREGISTRATION.md")

# The repository files whose code Stage 2 imports read-only (BCA's own code and the gate's), pinned by SHA-256.
BCA_FILES = ("algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "calibration/network.py", "calibration/reference.py",
             "calibration/wbcp.py", "calibration/dose.py", "calibration/bank.py", "ALGORITHMS.md")


class ApprovalError(PermissionError):
    """Raised by every Stage 2 BCA entry point without a valid approval of the current pre-registration."""


REFUSAL = ("refused: Stage 2 (BCA lower-edge consumer) is not approved. No BCA band fit, calibrator fit, threshold, "
           "dose or width readout, deployment check or BCA training arm may run on ThrusterChain data before the user "
           "approves runs/shift_explore/stage2/prereg.json by a route it accepts (version 6: only on the user's own "
           "machine with approve_local.py, committed and pushed by the user, imported with 'approve.py import', and "
           "the import record committed and pushed)")


@dataclass(frozen=True)
class Paths:
    """Where Stage 2 reads and writes. Tests build one in a temporary directory; the real one is PATHS."""

    out: Path
    code_src: Path = HERE
    repo: Path = ROOT / REPO_STAGE2_REL      # the committed folder (REPO_STAGE2_REL in git) the user's approval arrives in

    @property
    def prereg(self):
        return self.out / "prereg.json"

    @property
    def approval(self):
        return self.out / "approval.json"

    @property
    def manifest(self):
        return self.out / "code_manifest.json"

    @property
    def code_copies(self):
        return self.out / "code"

    def local_approval(self, prefix):
        """The user's committed local approval of the fingerprint starting with `prefix`."""
        return self.repo / LOCAL_APPROVAL_NAME.format(prefix=_checked_prefix(prefix))

    def import_record(self, prefix):
        """The import record of that approval (written by 'approve.py import', then committed and pushed)."""
        return self.repo / IMPORT_RECORD_NAME.format(prefix=_checked_prefix(prefix))


PATHS = Paths(ROOT / "runs" / "shift_explore" / "stage2")
OUT = PATHS.out

FORBIDDEN = (ROOT / "runs" / "shift", ROOT / "experiments", ROOT / "algorithms", ROOT / "calibration",
             ROOT / "runtime", ROOT / "configs", ROOM_OUT, STANDINS_SRC, DETECT_SRC, SCRATCH_OOD / "deploy")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- JSON and hashes

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def plain(x):
    """numpy -> Python, recursively; nonfinite floats -> None (acceptance.plain's rule, restated without importing it)."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
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
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    return x


def canonical(payload):
    return json.dumps(plain(payload), indent=1, sort_keys=True, allow_nan=False)


def guard_out(path, paths=None):
    """Refuse any output outside Stage 2's folder, and anything inside a protected folder."""
    paths = paths or PATHS
    p = Path(path).resolve()
    out = paths.out.resolve()
    if p != out and out not in p.parents:
        raise SystemExit(f"refused: {p} is outside {out}")
    for protected in FORBIDDEN:
        protected = protected.resolve()
        if p == protected or protected in p.parents:
            raise SystemExit(f"refused: {p} is inside the protected folder {protected}")
    return p


def write_json(path, payload, paths=None, overwrite=False):
    """Strict sorted JSON and a '<name>.sha256' sidecar, restricted to Stage 2's folder; refuses to overwrite unless
    asked. Returns the SHA-256 of the text."""
    path = guard_out(path, paths)
    if path.exists() and not overwrite:
        raise SystemExit(f"refused: {path} exists; Stage 2 outputs are written once")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = canonical(payload)
    path.write_text(text, encoding="utf8")
    digest = sha256_bytes(text.encode("utf8"))
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf8")
    return digest


def read_json(path, required=True):
    """A JSON with a matching '<name>.sha256' sidecar. Missing: None (or refuse when required). A sidecar that does
    not match always refuses."""
    path = Path(path)
    if not path.exists():
        if required:
            raise SystemExit(f"refused: {path} is missing")
        return None
    text = path.read_text(encoding="utf8")
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != sha256_bytes(text.encode("utf8")):
        raise SystemExit(f"refused: {path} differs from its SHA-256 sidecar")
    return json.loads(text)


def save_tree(path, tree, paths=None):
    """Flax msgpack of a pytree under Stage 2's folder; returns its SHA-256."""
    from flax import serialization
    path = guard_out(path, paths)
    if path.exists():
        raise SystemExit(f"refused: {path} exists; Stage 2 outputs are written once")
    blob = serialization.to_bytes(tree)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return sha256_bytes(blob)


def load_tree(path, template, sha256):
    from flax import serialization
    blob = Path(path).read_bytes()
    if sha256_bytes(blob) != sha256:
        raise SystemExit(f"refused: {path} differs from its recorded SHA-256")
    return serialization.from_bytes(template, blob)


# --------------------------------------------------------------------------------------------- the pre-registration

@dataclass(frozen=True)
class Prereg:
    path: str
    sha256: str
    payload: dict = field(compare=False, repr=False)

    @property
    def prefix(self):
        return self.sha256[:PREFIX_LEN]


def load_prereg(paths=None):
    """prereg.json, verified against its sidecar. Refuses (SystemExit) when missing or changed."""
    paths = paths or PATHS
    p = paths.prereg
    if not p.exists():
        raise SystemExit(f"refused: no pre-registration at {p}; write it with write_prereg.py")
    raw = p.read_bytes()
    digest = sha256_bytes(raw)
    side = p.with_name(p.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != digest:
        raise SystemExit(f"refused: {p} differs from its SHA-256 sidecar (the file was changed)")
    payload = json.loads(raw.decode("utf8"))
    if payload.get("kind") != "stage2-preregistration" or payload.get("label") != LABEL:
        raise SystemExit(f"refused: {p} is not the Stage 2 pre-registration")
    return Prereg(str(p), digest, payload)


# --------------------------------------------------------------------------------------------- the approval

@dataclass(frozen=True)
class Approval:
    """A verified approval: the approval file and its SHA-256, the pre-registration it approves, and the mode."""

    path: str
    file_sha256: str
    prereg_path: str
    prereg_sha256: str
    mode: str


def chat_sentence(prefix):
    """The one sentence chat mode accepts for this pre-registration."""
    return CHAT_SENTENCE.format(prefix=prefix)


def message_approves(message, prefix):
    """Chat mode: the verbatim message is exactly the approval sentence 'approve <prefix>, accept chat mode' and
    nothing else (case, spaces at either end or between words and one final full stop ignored). Everything else is
    refused, including text that merely contains the prefix and the phrase (a negation, a question, a quotation, a
    condition)."""
    m = _CHAT_RE.fullmatch(str(message))
    return bool(m) and len(str(prefix)) == PREFIX_LEN and m.group(1).lower() == str(prefix).lower()


def allowed_modes(payload):
    """The approval modes a pre-registration accepts, from its approval section's flags (a missing flag is false).
    Version 6 accepts only LOCAL_MODE, the user's choice; CLOUD_TERMINAL_MODE and CHAT_MODE refuse unless a
    pre-registration sets cloud_terminal_allowed or chat_mode_allowed."""
    section = payload.get("approval") if isinstance(payload, dict) else None
    section = section if isinstance(section, dict) else {}
    flags = ((LOCAL_MODE, "local_terminal_allowed"), (CLOUD_TERMINAL_MODE, "cloud_terminal_allowed"),
             (CHAT_MODE, "chat_mode_allowed"))
    return [mode for mode, flag in flags if section.get(flag) is True]


class LocalApprovalProblem(ValueError):
    """A local (user's own machine) approval that does not approve the current pre-registration, or whose git record
    does not hold; the message says why."""


def check_local_approval(raw, prereg):
    """The user's local approval file, as committed (its bytes `raw`), written by approve_local.py. It must approve
    THIS pre-registration: kind LOCAL_KIND, mode LOCAL_MODE, this label, the committed copy of prereg.json, this
    prereg.json's SHA-256, typed characters equal to its first 16 characters exactly, typed at a terminal, a time, the
    code manifest and approve_local.py this pre-registration pins, and a statement that names this SHA-256 (the
    approval covers this exact pre-registration only). Raises LocalApprovalProblem naming the first problem; returns
    the parsed body."""
    try:
        body = json.loads(bytes(raw).decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise LocalApprovalProblem("the local approval file is not UTF-8 JSON") from None
    if not isinstance(body, dict) or body.get("kind") != LOCAL_KIND:
        raise LocalApprovalProblem(f"the local approval file is not a Stage 2 local approval (kind {LOCAL_KIND!r})")
    if body.get("mode") != LOCAL_MODE:
        raise LocalApprovalProblem(f"the local approval's mode is {body.get('mode')!r}, not {LOCAL_MODE!r}")
    if body.get("label") != LABEL:
        raise LocalApprovalProblem("the local approval names another study (its label differs)")
    if body.get("prereg_path") != REPO_STAGE2_REL + "/prereg.json":
        raise LocalApprovalProblem(f"the local approval names another pre-registration file ({body.get('prereg_path')!r})")
    if body.get("prereg_sha256") != prereg.sha256:
        raise LocalApprovalProblem(f"the local approval is for another version of prereg.json: it holds "
                                   f"{str(body.get('prereg_sha256'))[:16]}..., the current file is {prereg.prefix}...")
    if body.get("typed") != prereg.prefix:
        raise LocalApprovalProblem("the characters typed in the local approval are not exactly the first 16 characters "
                                   "of the current prereg.json's SHA-256")
    if body.get("stdin_isatty") is not True:
        raise LocalApprovalProblem("the local approval was not typed at a terminal")
    if not body.get("approved_at_utc"):
        raise LocalApprovalProblem("the local approval holds no time")
    code = prereg.payload.get("code") or {}
    if body.get("code_manifest_sha256") != code.get("code_manifest_sha256"):
        raise LocalApprovalProblem("the local approval was made against another code manifest")
    if body.get("approve_local_sha256") != (code.get("files") or {}).get("approve_local.py"):
        raise LocalApprovalProblem("the local approval was made with another approve_local.py")
    if prereg.sha256 not in str(body.get("statement", "")):
        raise LocalApprovalProblem("the local approval's statement does not name this pre-registration's SHA-256")
    return body


# --------------------------------------------------------------------------------------------- read-only git

GIT_TIMEOUT_S = 60
_GIT_READ_ONLY = ("rev-parse", "cat-file", "log", "merge-base", "config")


def git(*args):
    """One read-only git query in the repository at ROOT: (exit code, standard output as bytes), or (None, b"") when git
    cannot be run. Only the subcommands in _GIT_READ_ONLY are allowed, and 'config' only with --get; nothing here
    writes to the repository (GIT_OPTIONAL_LOCKS=0 also stops git from refreshing the index). The synthetic tests
    replace this function with a stand-in; they never run git."""
    if not args or args[0] not in _GIT_READ_ONLY or (args[0] == "config" and "--get" not in args):
        raise ValueError(f"refused: not a read-only git query: {args!r}")
    cmd = ["git", "-C", str(ROOT), "-c", "log.showSignature=false", "-c", "log.follow=false",
           "-c", "core.quotePath=false"] + [str(a) for a in args]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", LC_ALL="C")
    try:
        done = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=GIT_TIMEOUT_S, env=env)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None, b""
    return done.returncode, done.stdout


def _git(*args):
    # Looked up through the module at call time, so that a test's stand-in for git() is used everywhere.
    return sys.modules[__name__].git(*args)


def remote_ref(branch):
    """origin's branch as last fetched or pushed here: the copy of the branch on GitHub."""
    if not branch or not re.fullmatch(r"[A-Za-z0-9._/-]+", str(branch)) or ".." in str(branch):
        raise LocalApprovalProblem(f"the pre-registration names no usable branch ({branch!r})")
    return "refs/remotes/origin/" + str(branch)


def git_ref_commit(ref):
    code, out = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    text = out.decode("utf8", "replace").strip() if code == 0 else ""
    return text if _COMMIT_RE.fullmatch(text) else None


def git_commit_facts(commit):
    """Author and committer ('Name <email>', as git log's %an <%ae> and %cn <%ce>) and parents of a commit that exists
    here; None otherwise."""
    if not _COMMIT_RE.fullmatch(str(commit)):
        return None
    code, _ = _git("cat-file", "-e", commit + "^{commit}")
    if code != 0:
        return None
    code, out = _git("log", "-1", "--format=%an%x00%ae%x00%cn%x00%ce%x00%P", commit)
    if code != 0:
        return None
    parts = out.decode("utf8", "replace").rstrip("\n").split("\0")
    if len(parts) != 5:
        return None
    an, ae, cn, ce, parents = parts
    return dict(author=f"{an} <{ae}>", committer=f"{cn} <{ce}>", parents=parents.split())


def git_blob(rev, rel):
    """The bytes of file `rel` (relative to the repository root) as stored in `rev`; None when it is not there."""
    code, out = _git("cat-file", "blob", f"{rev}:{rel}")
    return out if code == 0 else None


def git_adding_commits(ref, rel):
    """Every commit in `ref`'s history that added `rel` (rename detection off, so a file added under this name always
    counts); None when git fails."""
    code, out = _git("log", "--no-renames", "--diff-filter=A", "--format=%H", ref, "--", rel)
    return out.decode("utf8", "replace").split() if code == 0 else None


def git_is_ancestor(commit, ref):
    code, _ = _git("merge-base", "--is-ancestor", commit, ref)
    return code == 0


def git_session_email():
    """This session's configured git user.email (lower case), or None."""
    code, out = _git("config", "--get", "user.email")
    text = out.decode("utf8", "replace").strip().lower() if code == 0 else ""
    return text or None


def git_signature_status(commit):
    """Evidence only: git's signature status letter for the commit (G, N, E, ...), or None."""
    code, out = _git("log", "-1", "--format=%G?", commit)
    text = out.decode("utf8", "replace").strip() if code == 0 else ""
    return text or None


def agent_emails(session_email=None):
    return tuple(sorted({e.lower() for e in AGENT_GIT_EMAILS} | ({session_email.lower()} if session_email else set())))


def check_commit_record(record, local_sha256, refused_emails=AGENT_GIT_EMAILS):
    """The shape of a local approval's git record: the full hash of the commit that added the file, its author AND its
    committer as 'Name <email>', neither of them one of `refused_emails` (the cloud session's own git identity), and
    the SHA-256 of the file as stored in that commit, which must equal the imported file's. Raises
    LocalApprovalProblem. It records who committed; it cannot authenticate them (git's author and committer fields are
    set by whoever makes the commit)."""
    record = record if isinstance(record, dict) else {}
    if not _COMMIT_RE.fullmatch(str(record.get("commit", ""))):
        raise LocalApprovalProblem("the commit is not a full lowercase git hash (40 or 64 hexadecimal characters)")
    refused = {e.lower() for e in refused_emails}
    for role in ("author", "committer"):
        m = _AUTHOR_RE.fullmatch(str(record.get(role, "")))
        if not m:
            raise LocalApprovalProblem(f"the commit's {role} is not given as 'Name <email>'")
        if m.group(2).lower() in refused:
            raise LocalApprovalProblem(f"the commit that added the local approval has the cloud session's own git "
                                       f"identity ({m.group(2)}) as its {role}, not the user")
    if record.get("committed_sha256") != local_sha256:
        raise LocalApprovalProblem("the file stored in that commit is not the local approval file being imported (their "
                                   "SHA-256 differ)")
    return record


def check_commit_in_git(record, rel, branch):
    """The git record against git itself (read-only queries): the commit exists here with the recorded author and
    committer, neither is the cloud session's configured identity, the file `rel` as stored in it has the recorded
    SHA-256, the commit is on origin's branch (as last fetched or pushed), and that branch's tip still holds the same
    file. Raises LocalApprovalProblem."""
    commit = str(record.get("commit", ""))
    facts = git_commit_facts(commit)
    if facts is None:
        raise LocalApprovalProblem(f"commit {commit[:16]}... is not a commit in this repository")
    for role in ("author", "committer"):
        if facts[role] != record.get(role):
            raise LocalApprovalProblem(f"the recorded {role} of commit {commit[:16]}... is not the one git holds")
    check_commit_record(dict(record, author=facts["author"], committer=facts["committer"]),
                        record.get("committed_sha256"), agent_emails(git_session_email()))
    blob = git_blob(commit, rel)
    if blob is None or sha256_bytes(blob) != record.get("committed_sha256"):
        raise LocalApprovalProblem(f"commit {commit[:16]}... does not hold {rel} with the recorded SHA-256")
    ref = remote_ref(branch)
    if git_ref_commit(ref) is None:
        raise LocalApprovalProblem(f"{ref} does not exist here (fetch the branch from GitHub first)")
    if not git_is_ancestor(commit, ref):
        raise LocalApprovalProblem(f"commit {commit[:16]}... is not on {ref}, the branch as fetched from GitHub")
    tip = git_blob(ref, rel)
    if tip is None or sha256_bytes(tip) != record.get("committed_sha256"):
        raise LocalApprovalProblem(f"{ref} no longer holds {rel} unchanged")
    return facts


def check_import_record(paths, prereg, approval_digest, record, branch):
    """The import record: present in the committed folder with a matching sidecar, naming this pre-registration, the
    user's approval file and its SHA-256, the same commit, author and committer, and the SHA-256 of the runs-side
    approval.json; and byte-identical on origin's branch (so it has been committed and pushed, and the user can see
    on GitHub which commit was imported). Raises LocalApprovalProblem."""
    path = paths.import_record(prereg.prefix)
    rel = import_record_rel(prereg.prefix)
    if not path.exists():
        raise LocalApprovalProblem(f"the import record {rel} is missing (it is written by 'approve.py import')")
    raw = path.read_bytes()
    side = path.with_name(path.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[:1] != [sha256_bytes(raw)]:
        raise LocalApprovalProblem(f"the import record {rel} differs from its SHA-256 sidecar")
    try:
        body = json.loads(raw.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise LocalApprovalProblem(f"the import record {rel} is not JSON") from None
    want = dict(kind=IMPORT_KIND, label=LABEL, prereg_sha256=prereg.sha256,
                approval_file=local_approval_rel(prereg.prefix), approval_file_sha256=record.get("committed_sha256"),
                commit=record.get("commit"), author=record.get("author"), committer=record.get("committer"),
                runs_side_approval_sha256=approval_digest)
    bad = sorted(k for k, v in want.items() if not isinstance(body, dict) or body.get(k) != v)
    if bad:
        raise LocalApprovalProblem(f"the import record {rel} does not match this approval ({', '.join(bad)})")
    if git_blob(remote_ref(branch), rel) != raw:
        raise LocalApprovalProblem(f"the import record {rel} is not on origin's branch {branch} as written: commit and "
                                   f"push it (with its .sha256) before any BCA run")
    return body


def write_import_record(paths, prefix, payload):
    """Write the import record (strict sorted JSON and a '.sha256' sidecar) into the committed folder, once. This is
    the one file Stage 2 writes outside its runs folder; the orchestrating agent commits and pushes it."""
    path = paths.import_record(prefix)
    if path.parent.resolve() != paths.repo.resolve():
        raise SystemExit(f"refused: {path} is outside {paths.repo}")
    side = path.with_name(path.name + ".sha256")
    if path.exists() or side.exists():
        raise SystemExit(f"refused: {path} exists; an import record is written once")
    text = canonical(payload).encode("utf8")
    digest = sha256_bytes(text)
    with open(path, "xb") as handle:
        handle.write(text)
    with open(side, "xb") as handle:
        handle.write(f"{digest}  {path.name}\n".encode("utf8"))
    return digest


def verify_approval(paths=None, pushed=True):
    """Read approval.json and check, against the CURRENT prereg.json: the sidecar matches; it is a Stage 2 approval
    written by approve.py; it names this prereg.json's path and SHA-256; the pre-registration was approvable (no
    missing pin); the pre-registration accepts its mode (allowed_modes; version 6 accepts only 'local-terminal'); and
    for that mode:
      'local-terminal': the stored copy of the user's local approval file has its recorded SHA-256 and approves this
          pre-registration (check_local_approval); it names this fingerprint's file; the commit record is complete and
          neither its author nor its committer is the cloud session's identity (check_commit_record); the committed
          file (paths.local_approval(prefix)) is still there, unchanged; git confirms the commit, its author and
          committer, the file in it, and that it is on origin's branch whose tip still holds the file
          (check_commit_in_git); and, unless pushed=False (only import's own check of the file it has just written),
          the import record matches and is on origin's branch (check_import_record);
      'terminal' (refused unless allowed): the typed characters equal the prefix and standard input was a terminal;
      'chat' (refused unless allowed): the stored verbatim message is exactly the approval sentence.
    Raises ApprovalError otherwise. Reads no ThrusterChain data. No mode can prove that the user, rather than someone
    or something with access to the user's terminal or to the repository, produced the approval, and an agent can
    write approval.json by hand (pre-registration section 9)."""
    paths = paths or PATHS
    try:
        prereg = load_prereg(paths)
    except SystemExit as error:
        raise ApprovalError(f"{REFUSAL} ({error})") from None
    a = paths.approval
    if not a.exists():
        raise ApprovalError(f"{REFUSAL} (no approval file at {a})")
    text = a.read_text(encoding="utf8")
    digest = sha256_bytes(text.encode("utf8"))
    side = a.with_name(a.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[0] != digest:
        raise ApprovalError(f"{REFUSAL} (the approval file differs from its sidecar)")
    try:
        body = json.loads(text)
    except ValueError:
        raise ApprovalError(f"{REFUSAL} (the approval file is not JSON)") from None
    if body.get("kind") != "stage2-approval" or body.get("label") != LABEL:
        raise ApprovalError(f"{REFUSAL} (not a Stage 2 approval file)")
    if Path(str(body.get("prereg_path", ""))).resolve() != paths.prereg.resolve():
        raise ApprovalError(f"{REFUSAL} (the approval names another pre-registration file)")
    if body.get("prereg_sha256") != prereg.sha256:
        raise ApprovalError(f"{REFUSAL} (the approval is for another version of prereg.json: it holds "
                            f"{str(body.get('prereg_sha256'))[:16]}..., the current file is {prereg.prefix}...)")
    if not prereg.payload.get("approvable", False):
        raise ApprovalError(f"{REFUSAL} (the pre-registration is a draft with missing pins: "
                            f"{prereg.payload.get('missing_pins')})")
    mode = body.get("mode")
    allowed = allowed_modes(prereg.payload)
    if mode not in allowed:
        raise ApprovalError(f"{REFUSAL} (the pre-registration does not accept approval mode {mode!r}; it accepts "
                            f"{', '.join(allowed) or 'no mode'})")
    if mode == LOCAL_MODE:
        if body.get("typed") != prereg.prefix or not body.get("stdin_isatty"):
            raise ApprovalError(f"{REFUSAL} (the imported approval does not hold the typed prefix from a terminal)")
        local = body.get("local_approval") if isinstance(body.get("local_approval"), dict) else {}
        local_text = local.get("text")
        if not isinstance(local_text, str) or sha256_bytes(local_text.encode("utf8")) != local.get("sha256"):
            raise ApprovalError(f"{REFUSAL} (the stored local approval differs from its recorded SHA-256)")
        rel = local_approval_rel(prereg.prefix)
        if local.get("path") != rel:
            raise ApprovalError(f"{REFUSAL} (the stored local approval names another file than {rel})")
        branch = ((prereg.payload.get("approval") or {}).get("local_terminal") or {}).get("branch")
        try:
            check_local_approval(local_text.encode("utf8"), prereg)
            check_commit_record(body.get("git"), local["sha256"])
            committed = paths.local_approval(prereg.prefix)
            if not committed.exists() or file_sha256(committed) != local["sha256"]:
                raise LocalApprovalProblem(f"the committed local approval {committed} is missing or changed since it "
                                           f"was imported")
            check_commit_in_git(body["git"], rel, branch)
            if pushed:
                check_import_record(paths, prereg, digest, body["git"], branch)
        except LocalApprovalProblem as problem:
            raise ApprovalError(f"{REFUSAL} ({problem})") from None
    elif mode == CLOUD_TERMINAL_MODE:
        if body.get("typed") != prereg.prefix or not body.get("stdin_isatty"):
            raise ApprovalError(f"{REFUSAL} (the terminal approval does not hold the typed prefix from a terminal)")
    elif mode == CHAT_MODE:
        msg = body.get("message_verbatim")
        if msg is None or sha256_bytes(str(msg).encode("utf8")) != body.get("message_sha256"):
            raise ApprovalError(f"{REFUSAL} (the stored chat message differs from its recorded SHA-256)")
        if not message_approves(msg, prereg.prefix):
            raise ApprovalError(f"{REFUSAL} (the chat message is not exactly the approval sentence "
                                f"'{chat_sentence(prereg.prefix)}')")
    else:  # pragma: no cover - allowed_modes names no other mode
        raise ApprovalError(f"{REFUSAL} (unknown approval mode {mode!r})")
    return Approval(str(a), digest, str(paths.prereg), prereg.sha256, mode)


def check_code(prereg, paths=None):
    """Every Stage 2 script, here and in its copy under OUT/code/, has the SHA-256 the code manifest records; the
    manifest matches its sidecar and has the SHA-256 the pre-registration pins. Also BCA's and the gate's files that
    Stage 2 imports. Raises ApprovalError (a BCA entry point must not run on changed code)."""
    paths = paths or PATHS
    pins = prereg.payload.get("code", {})
    try:
        man = read_json(paths.manifest)
    except SystemExit as error:
        raise ApprovalError(f"{REFUSAL} ({error})") from None
    if file_sha256(paths.manifest) != pins.get("code_manifest_sha256"):
        raise ApprovalError(f"{REFUSAL} (the code manifest is not the one the pre-registration pins)")
    bad = []
    for name, h in man["files"].items():
        src = paths.code_src / name
        cp = paths.code_copies / name
        if not src.exists() or file_sha256(src) != h:
            bad.append(name)
        if not cp.exists() or file_sha256(cp) != h:
            bad.append("code/" + name)
    for rel, h in man.get("repository_files", {}).items():
        if not (ROOT / rel).exists() or file_sha256(ROOT / rel) != h:
            bad.append(rel)
    if bad:
        raise ApprovalError(f"{REFUSAL} (code differs from the pinned manifest: {', '.join(bad)})")
    return man


def require_approval(paths=None, check_files=True):
    """The one gate every Stage 2 BCA entry point calls first: a valid approval of the current prereg.json, and (by
    default) every pinned script unchanged. Returns the verified Approval."""
    paths = paths or PATHS
    approval = verify_approval(paths)
    if check_files:
        check_code(load_prereg(paths), paths)
    return approval


def recheck(approval, paths=None):
    """Re-verify an Approval held by a long-running phase; refuse if anything changed since."""
    again = require_approval(paths)
    if again != approval:
        raise ApprovalError(f"{REFUSAL} (the approval or the pre-registration changed since this phase started)")
    return approval


# --------------------------------------------------------------------------------------------- pinned imports

_MODULES = {}


def _exec_module(name, path, data):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(data, str(path), "exec"), mod.__dict__)
    _MODULES[name] = mod
    return mod


def load_room_harness():
    """The room test's harness, executed from the bytes its code manifest records (manifest checked against its
    sidecar and the SHA-256 pinned here; the spec it names checked too). Also registered as 'harness'."""
    if "room_harness" in _MODULES:
        return _MODULES["room_harness"]
    man_path = ROOM_OUT / "code_manifest.json"
    text = man_path.read_bytes()
    side = (ROOM_OUT / "code_manifest.json.sha256").read_text(encoding="utf8").split()[0]
    if sha256_bytes(text) != side or side != ROOM_PINNED["manifest_sha256"]:
        raise SystemExit("refused: the room test's code manifest is not the one Stage 2 pins")
    man = json.loads(text)
    if man["spec_sha256"] != ROOM_PINNED["spec_sha256"] or file_sha256(ROOM_OUT / "spec.json") != man["spec_sha256"]:
        raise SystemExit("refused: the room test's spec is not the one Stage 2 pins")
    if man["files"]["harness.py"] != ROOM_PINNED["harness_sha256"]:
        raise SystemExit("refused: the room manifest's harness.py is not the pinned one")
    copy = ROOM_OUT / "code" / "harness.py"
    data = copy.read_bytes()
    if sha256_bytes(data) != ROOM_PINNED["harness_sha256"]:
        raise SystemExit("refused: the room's pinned copy of harness.py differs from its manifest")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    mod = _exec_module("room_harness", copy, data)
    sys.modules["harness"] = mod
    return mod


def load_fallback():
    """The fallback diagnostic (clip99 range, the distance check, run_rule's arithmetic), from its pinned bytes."""
    if "stage2_fallback" in _MODULES:
        return _MODULES["stage2_fallback"]
    data = FALLBACK_PY.read_bytes()
    if sha256_bytes(data) != FALLBACK_PY_SHA256:
        raise SystemExit("refused: fallback.py differs from the SHA-256 Stage 2 pins")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    return _exec_module("stage2_fallback", FALLBACK_PY, data)


def verify_component(out_dir, manifest_sha256):
    """Another component's code (the stand-ins test, the detection script) as pinned by the pre-registration: its
    code_manifest.json matches its sidecar and has the pinned SHA-256, and every file it lists has that SHA-256 in
    its copy under <out_dir>/code/ (from where it is imported). Refuses otherwise. Returns the manifest."""
    out_dir = Path(out_dir)
    p = out_dir / "code_manifest.json"
    man = read_json(p)
    if manifest_sha256 is None or file_sha256(p) != manifest_sha256:
        raise SystemExit(f"refused: {p} is not the manifest the pre-registration pins")
    bad = [n for n, h in man["files"].items()
           if not (out_dir / "code" / n).exists() or file_sha256(out_dir / "code" / n) != h]
    if bad:
        raise SystemExit(f"refused: {out_dir}/code differs from its pinned manifest: {bad}")
    return man


def load_pinned_module(name, path, sha256):
    """Any other pinned script (the stand-ins module, the locked detection script), by the SHA-256 the
    pre-registration records for it."""
    if name in _MODULES:
        return _MODULES[name]
    path = Path(path)
    if sha256 is None:
        raise SystemExit(f"refused: the pre-registration pins no SHA-256 for {path}")
    data = path.read_bytes()
    if sha256_bytes(data) != sha256:
        raise SystemExit(f"refused: {path} differs from the SHA-256 the pre-registration pins")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    return _exec_module(name, path, data)


def repository_hashes(files=BCA_FILES):
    return {rel: file_sha256(ROOT / rel) for rel in files}
