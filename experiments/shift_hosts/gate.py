"""The approval gate: no run on real ThrusterChain data, the learner alone included, without the user's terminal
approval of the frozen pre-specification (plan section 6.1 "What a stop means", section 10 item 15, stop rule S1).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. Pure Python
(json, hashlib, read-only git), so it imports in both environments.

require_run_approval() passes only when all of these hold, checked from disk at every call:
  1. experiments/shift_hosts/prespec.json exists (the frozen pre-specification; the plan's PRESPEC.md beside it is
     covered by the code manifest).
  2. The fingerprint is the SHA-256 of a canonical JSON of {prespec.json's SHA-256, the code manifest}. The code
     manifest (code_manifest()) holds the SHA-256 of every regular file of this package, whatever its name or suffix,
     except prespec.json itself and the approval files at the package root (approval_<16 hex>.json and its .sha256
     sidecar, APPROVAL_FILE), and of the repository files it runs (EXTERNAL_CODE: Unifloral's four files and their
     manifest, calibration/, algorithms/, runtime/, experiments/shift/, experiments/shift_iw/, experiments/truth/,
     experiments/signal/, the oracle). So any changed, added or removed file changes the fingerprint (S1).
  3. No compiled bytecode exists where the manifest's code lives (no __pycache__ folder or *.pyc file anywhere in the
     package or in the folders of EXTERNAL_CODE's files; bytecode_files()): Python would run a timestamp-checked .pyc
     whose recorded size and time match its source without reading the source, so bytecode could run in place of the
     hashed code.
  4. experiments/shift_hosts/approval_<first 16 hex of the fingerprint>.json exists, matches its sidecar
     approval_<...>.json.sha256 (sha256sum format), is JSON of kind "shift-hosts-approval" and holds the full
     fingerprint, the first 16 characters the user typed, prespec.json's SHA-256 and the same code manifest.
  5. The approval file, its sidecar and prespec.json are committed in git HEAD, byte for byte as on disk, and the
     latest commit that touched the approval file has the author email 35092758+dbayha7@users.noreply.github.com (the
     user's GitHub address; plan section 10, item 15).
No agent writes the approval file (approve_hosts.py, run by the user at the terminal, does). Nothing here writes
anything; git is only read (rev-parse, cat-file, log).

Stop rule S0 (plan 6.1: "Checked before stage 1"): s0_status() reads the stage-0 suite's record,
runs/shift_hosts/stage0/SUMMARY.json (run_stage0.sh's summary module writes it, with a sha256sum sidecar), and passes
only when it verifies, says the suite passed (status 0, s0.passed true, made-up arrays only) and was made on exactly
the code on disk (its manifest.code_manifest_sha256 equals the canonical SHA-256 of code_manifest(), runner.py's
canonical_sha256 rule). require_s0() raises ApprovalError otherwise; the stage driver calls it before a real start.
"""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
APPROVAL_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVAL_KIND = "shift-hosts-approval"
FINGERPRINT_KIND = "shift-hosts-fingerprint-v1"
PRESPEC_NAME = "prespec.json"
APPROVAL_PREFIX = "approval_"
# the only files of the package the code manifest leaves out besides prespec.json: the approval files approve_hosts.py
# writes at the package root (approval_<first 16 hex of the fingerprint>.json and its sidecar)
APPROVAL_FILE = re.compile(r"^approval_[0-9a-f]{16}\.json(\.sha256)?$")
S0_SUMMARY = REPO / "runs" / "shift_hosts" / "stage0" / "SUMMARY.json"
S0_SCHEMA = "shift-hosts-stage0-summary-v1"
# repository files the package runs or its stage-0 tests import (relative to the repository root; globs allowed)
EXTERNAL_CODE = (
    "baselines/unifloral/algorithms/td3_bc.py", "baselines/unifloral/algorithms/cql.py",
    "baselines/unifloral/algorithms/iql.py", "baselines/unifloral/algorithms/rebrac.py",
    "baselines/unifloral/Dockerfile", "baselines/unifloral/requirements.txt", "configs/unifloral.json",
    "calibration/*.py", "algorithms/*.py", "runtime/*.py", "experiments/__init__.py",
    "experiments/shift/*.py", "experiments/shift_iw/*.py", "experiments/shift_iw/launch.sh",
    "experiments/truth/*.py", "outputs/shift/oracle/oracle.py",
    # experiments/shift/{thruster_chain,truth,learners}.py import these, so data preparation and scoring run them
    "experiments/signal/*.py",
)
REFUSAL = ("refused: runs on real ThrusterChain data (the learner alone included) wait for the user's terminal approval "
           "of the frozen pre-specification: approve_hosts.py writes experiments/shift_hosts/approval_<fp16>.json and "
           "the user commits it under 35092758+dbayha7@users.noreply.github.com (plan section 10, item 15). No agent "
           "writes it")


class ApprovalError(PermissionError):
    """Raised by every path that needs the user's approval (gate) or registered made-up arrays (guard)."""


@dataclass(frozen=True)
class Approval:
    """A verified approval: the file, the fingerprint it approves, prespec.json's SHA-256, the approval file's own
    SHA-256, and the commit (with its author) that last touched the approval file."""

    path: str
    fingerprint: str
    prespec_sha256: str
    file_sha256: str
    commit: str
    author_email: str
    author_name: str

    def describe(self):
        return asdict(self)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sidecar(path):
    return Path(str(path) + ".sha256")


def is_approval_file(rel):
    """Whether a package-relative path is an approval file approve_hosts.py writes (APPROVAL_FILE, at the package root
    only): approval_<16 hex>.json or its .sha256 sidecar."""
    rel = Path(rel)
    return len(rel.parts) == 1 and bool(APPROVAL_FILE.match(rel.name))


def package_files(package=PACKAGE):
    """Every regular file of the package that the code manifest hashes: all of them, whatever their suffix, except
    prespec.json and the approval files at the package root, and compiled bytecode (__pycache__ folders and *.pyc
    files, which require_run_approval refuses outright: bytecode_files)."""
    package = Path(package)
    out = []
    for path in sorted(package.rglob("*")):
        rel = path.relative_to(package)
        if not path.is_file() or "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if rel.as_posix() == PRESPEC_NAME or is_approval_file(rel):
            continue
        out.append(path)
    return out


def bytecode_files(package=PACKAGE, repo=REPO):
    """Compiled bytecode where the code manifest's code lives: every __pycache__ folder and *.pyc file anywhere under
    the package, and in each folder that holds a file of EXTERNAL_CODE (not below it). Sorted paths; empty is clean."""
    package, repo = Path(package).resolve(), Path(repo).resolve()
    found = {p for p in package.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc"}
    folders = set()
    for pattern in EXTERNAL_CODE:
        folders.update(p.parent for p in repo.glob(pattern) if p.is_file())
    for folder in folders:
        found.update(p for p in folder.iterdir() if p.name == "__pycache__" or p.suffix == ".pyc")
    return sorted(found)


def code_manifest(package=PACKAGE, repo=REPO):
    """{repository-relative path: SHA-256} of every package file (package_files: all but prespec.json, the approval
    files and bytecode) and of every file matching EXTERNAL_CODE, sorted by path."""
    package, repo = Path(package).resolve(), Path(repo).resolve()
    files = set(package_files(package))
    for pattern in EXTERNAL_CODE:
        files.update(p for p in repo.glob(pattern) if p.is_file() and "__pycache__" not in p.parts)
    out = {}
    for path in sorted(files):
        try:
            rel = path.resolve().relative_to(repo).as_posix()
        except ValueError:
            rel = path.resolve().as_posix()
        out[rel] = sha256_file(path)
    return dict(sorted(out.items()))


def fingerprint(package=PACKAGE, repo=REPO):
    """(fingerprint, prespec.json's SHA-256, code manifest). Refuses (ApprovalError) when prespec.json is missing."""
    prespec = Path(package) / PRESPEC_NAME
    if not prespec.is_file():
        raise ApprovalError(f"{REFUSAL} (no frozen pre-specification at {prespec})")
    prespec_sha = sha256_file(prespec)
    manifest = code_manifest(package, repo)
    body = json.dumps(dict(kind=FINGERPRINT_KIND, prespec_sha256=prespec_sha, code_manifest=manifest),
                      sort_keys=True, separators=(",", ":")).encode("utf8")
    return hashlib.sha256(body).hexdigest(), prespec_sha, manifest


def approval_path(fp, package=PACKAGE):
    return Path(package) / f"{APPROVAL_PREFIX}{fp[:16]}.json"


def _git(repo, *args):
    """Run a read-only git command in `repo`; returns stdout bytes or raises ApprovalError."""
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=60)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        detail = getattr(error, "stderr", b"") or b""
        raise ApprovalError(f"{REFUSAL} (git {' '.join(args)} failed: {detail.decode('utf8', 'replace').strip()})") \
            from error
    return out.stdout


def committed_in_head(path, repo=REPO):
    """Whether `path` is in git HEAD with exactly the bytes it has on disk."""
    repo = Path(repo).resolve()
    rel = Path(path).resolve().relative_to(repo).as_posix()
    try:
        blob = _git(repo, "cat-file", "blob", f"HEAD:{rel}")
    except ApprovalError:
        return False
    return blob == Path(path).read_bytes()


def last_commit(path, repo=REPO):
    """(commit, author email, author name) of the latest commit in HEAD's history that touched `path`, or None."""
    repo = Path(repo).resolve()
    rel = Path(path).resolve().relative_to(repo).as_posix()
    out = _git(repo, "log", "-1", "--format=%H%x00%ae%x00%an", "HEAD", "--", rel).decode("utf8").strip()
    if not out:
        return None
    commit, email, name = out.split("\x00")
    return commit, email, name


def require_run_approval(purpose="run", *, package=PACKAGE, repo=REPO):
    """Refuse (ApprovalError) unless the user's approval of the current pre-specification and code is on disk and
    committed in git HEAD under the user's address (module docstring). Returns the Approval. `purpose` names what asked
    (it appears in the refusal)."""
    package, repo = Path(package).resolve(), Path(repo).resolve()
    why = f" [{purpose}]"
    fp, prespec_sha, manifest = fingerprint(package, repo)
    stale = bytecode_files(package, repo)
    if stale:
        shown = ", ".join(p.relative_to(repo).as_posix() if repo in p.parents else str(p) for p in stale[:5])
        raise ApprovalError(f"{REFUSAL} (compiled bytecode exists beside the hashed code, so it could run in place of "
                            f"it: {shown}{' ...' if len(stale) > 5 else ''}; delete it and run with "
                            f"PYTHONDONTWRITEBYTECODE=1){why}")
    path = approval_path(fp, package)
    if not path.is_file():
        raise ApprovalError(f"{REFUSAL} (no approval file {path.name} for the current fingerprint {fp[:16]}; the "
                            f"pre-specification or the code changed since any approval){why}")
    digest = sha256_file(path)
    side = sidecar(path)
    words = side.read_text(encoding="utf8").split() if side.is_file() else []
    if not words or words[0] != digest:
        raise ApprovalError(f"{REFUSAL} (the approval file is missing its sidecar or differs from it){why}")
    try:
        body = json.loads(path.read_text(encoding="utf8"))
    except ValueError as error:
        raise ApprovalError(f"{REFUSAL} (the approval file is not JSON){why}") from error
    if not isinstance(body, dict) or body.get("kind") != APPROVAL_KIND:
        raise ApprovalError(f"{REFUSAL} (not a shift-hosts approval file){why}")
    if body.get("fingerprint") != fp or body.get("typed") != fp[:16]:
        raise ApprovalError(f"{REFUSAL} (the approval names another fingerprint){why}")
    if body.get("prespec_sha256") != prespec_sha:
        raise ApprovalError(f"{REFUSAL} (the approval names another prespec.json){why}")
    if body.get("code_manifest") != manifest:
        raise ApprovalError(f"{REFUSAL} (the approval's code manifest differs from the code on disk){why}")
    for needed in (path, side, package / PRESPEC_NAME):
        if not committed_in_head(needed, repo):
            raise ApprovalError(f"{REFUSAL} ({needed.name} is not committed in git HEAD as it is on disk){why}")
    last = last_commit(path, repo)
    if last is None or last[1] != APPROVAL_EMAIL:
        raise ApprovalError(f"{REFUSAL} (the approval's commit is not by {APPROVAL_EMAIL}: "
                            f"{None if last is None else last[1]}){why}")
    return Approval(str(path), fp, prespec_sha, digest, last[0], last[1], last[2])


def verify(approval, *, package=PACKAGE, repo=REPO):
    """Re-verify an Approval from disk (nothing may have changed since); returns it, or raises ApprovalError."""
    if not isinstance(approval, Approval):
        raise ApprovalError(REFUSAL)
    again = require_run_approval("re-check", package=package, repo=repo)
    if again != approval:
        raise ApprovalError(f"{REFUSAL} (the approval, the pre-specification or the code changed since it was "
                            "verified)")
    return approval


# ------------------------------------------------------------------------------------------------ stop rule S0

def manifest_sha256(manifest):
    """The canonical SHA-256 of a code manifest (runner.canonical_sha256's rule for a plain dict of strings: sorted keys,
    no spaces, UTF-8): the code_manifest_sha256 of run records and of the stage-0 summary."""
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def s0_status(*, package=PACKAGE, repo=REPO, summary=None):
    """Stop rule S0 for the code on disk (module docstring): dict(passed, reason, summary path, its SHA-256, suite run,
    the summary's and the current code manifest's SHA-256). Reads only."""
    package, repo = Path(package).resolve(), Path(repo).resolve()
    path = Path(summary) if summary is not None else repo / "runs" / "shift_hosts" / "stage0" / "SUMMARY.json"
    now = manifest_sha256(code_manifest(package, repo))
    out = dict(passed=False, reason=None, summary=str(path), summary_sha256=None, suite_run=None,
               summary_code_manifest_sha256=None, code_manifest_sha256=now)
    if not path.is_file():
        out["reason"] = f"no stage-0 summary at {path} (run run_stage0.sh on these exact files)"
        return out
    digest = sha256_file(path)
    out["summary_sha256"] = digest
    side = sidecar(path)
    words = side.read_text(encoding="utf8").split() if side.is_file() else []
    if not words or words[0] != digest:
        out["reason"] = f"{path.name} is missing its sidecar or differs from it"
        return out
    try:
        body = json.loads(path.read_text(encoding="utf8"))
    except ValueError:
        out["reason"] = f"{path.name} is not JSON"
        return out
    if not isinstance(body, dict) or body.get("schema") != S0_SCHEMA:
        out["reason"] = f"{path.name} is not a {S0_SCHEMA} file"
        return out
    out["suite_run"] = body.get("suite_run")
    made = (body.get("manifest") or {}).get("code_manifest_sha256")
    out["summary_code_manifest_sha256"] = made
    if body.get("status") != 0 or (body.get("s0") or {}).get("passed") is not True:
        out["reason"] = f"the stage-0 suite {body.get('suite_run')} did not pass (status {body.get('status')!r})"
    elif body.get("made_up_arrays_only") is not True:
        out["reason"] = "the stage-0 summary does not say it used made-up arrays only"
    elif made != now:
        out["reason"] = (f"the stage-0 suite {body.get('suite_run')} ran on other code (its code manifest "
                         f"{str(made)[:16]}, the code on disk {now[:16]}): run run_stage0.sh on these exact files")
    else:
        out.update(passed=True, reason="the stage-0 suite passed on exactly this code")
    return out


def require_s0(purpose="run", *, package=PACKAGE, repo=REPO):
    """Refuse (ApprovalError) unless stop rule S0 holds for the code on disk (s0_status). Returns the status."""
    st = s0_status(package=package, repo=repo)
    if not st["passed"]:
        raise ApprovalError(f"refused: stop rule S0 (plan 6.1, checked before stage 1): {st['reason']} [{purpose}]")
    return st
