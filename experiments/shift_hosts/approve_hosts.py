"""Your terminal approval of the frozen pre-specification (plan section 10, item 15).

Run it yourself, at your own terminal, in WSL:

    wsl -u root
    cd /root/bca-hosts
    PYTHONDONTWRITEBYTECODE=1 /opt/bca-venv/bin/python experiments/shift_hosts/approve_hosts.py

What it does:
  1. Computes the fingerprint (gate.fingerprint): the SHA-256 of prespec.json's SHA-256 together with the code manifest,
     which holds the SHA-256 of every file of this package and of the repository code it runs.
  2. Checks that prespec.json and every file in the code manifest are committed in git HEAD exactly as they are on disk,
     so what you approve is a commit anyone can check out and re-hash.
  3. Shows what you are approving, and whether stop rule S0 holds for this exact code (gate.s0_status: the stage-0
     suite's summary was made on it; the stage driver refuses to start without it), and asks you to type the
     fingerprint's first 16 characters.
  4. Writes two new files and nothing else:
       experiments/shift_hosts/approval_<first 16>.json          the approval, in the format gate.py checks
       experiments/shift_hosts/approval_<first 16>.json.sha256   its sha256sum line
  5. Prints the git commands that commit the two files under your GitHub address and send them on.

Runs on real data start only after that commit: gate.require_run_approval() checks the two files, the fingerprint and
the commit's author. This file refuses, writing nothing, when it is not run at a terminal, when anything it hashes is not
committed, when an approval for this fingerprint exists already, or when the typed characters differ. No agent runs it.
"""

import sys

sys.dont_write_bytecode = True

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    from . import gate
except ImportError:  # run as a script: python experiments/shift_hosts/approve_hosts.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gate  # noqa: E402

STATEMENT = ("I approve the frozen pre-specification (experiments/shift_hosts/PRESPEC.md and prespec.json) and the code "
             "named in code_manifest, for the runs they describe on real ThrusterChain data, within their caps.")
COMMIT_NAME = "David Bayha"
BRANCH = "claude/shift-hosts"


class Refused(Exception):
    """Nothing was written; the message says why."""


def _blob_id(path):
    """Git's SHA-1 object id of the file's bytes (what `git hash-object` prints in a SHA-1 repository)."""
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _head_blobs(repo):
    """{repository-relative path: object id} of every file in git HEAD (read-only: git ls-tree)."""
    out = gate._git(repo, "ls-tree", "-r", "-z", "HEAD").decode("utf8")
    blobs = {}
    for entry in out.split("\0"):
        if not entry:
            continue
        meta, rel = entry.split("\t", 1)
        mode, kind, oid = meta.split()
        if kind == "blob":
            blobs[rel] = oid
    return blobs


def not_committed(paths, repo):
    """The repository-relative paths whose bytes on disk differ from git HEAD, or that HEAD lacks."""
    repo = Path(repo).resolve()
    sha1 = gate._git(repo, "rev-parse", "--show-object-format").decode("utf8").strip() == "sha1"
    blobs = _head_blobs(repo) if sha1 else None
    bad = []
    for rel in paths:
        path = repo / rel
        if not path.is_file():
            bad.append(rel)
        elif sha1:
            if blobs.get(rel) != _blob_id(path):
                bad.append(rel)
        elif not gate.committed_in_head(path, repo):
            bad.append(rel)
    return bad


def prepare(package=gate.PACKAGE, repo=gate.REPO):
    """Everything the approval names, computed from disk; raises Refused when something is not committed."""
    package, repo = Path(package).resolve(), Path(repo).resolve()
    try:
        fp, prespec_sha, manifest = gate.fingerprint(package, repo)
    except gate.ApprovalError:
        raise Refused(f"there is no frozen pre-specification yet ({package / gate.PRESPEC_NAME} is missing).")
    prespec_rel = (package / gate.PRESPEC_NAME).relative_to(repo).as_posix()
    outside = [rel for rel in manifest if Path(rel).is_absolute()]
    if outside:
        raise Refused(f"the code manifest names files outside the repository: {outside[:5]}")
    bad = not_committed([prespec_rel, *manifest], repo)
    if bad:
        shown = "\n    ".join(bad[:20]) + ("\n    ..." if len(bad) > 20 else "")
        raise Refused(f"{len(bad)} file(s) differ from git HEAD or are not committed, so the approval would not name a "
                      f"commit:\n    {shown}")
    head = gate._git(repo, "rev-parse", "HEAD").decode("utf8").strip()
    branch = gate._git(repo, "rev-parse", "--abbrev-ref", "HEAD").decode("utf8").strip()
    prespec = json.loads((package / gate.PRESPEC_NAME).read_text(encoding="utf8"))
    package_rel = package.relative_to(repo).as_posix() + "/"
    return dict(fingerprint=fp, prespec_sha256=prespec_sha, code_manifest=manifest, head_commit=head, branch=branch,
                prespec=prespec if isinstance(prespec, dict) else {},
                n_package=sum(1 for rel in manifest if rel.startswith(package_rel)),
                path=gate.approval_path(fp, package), s0=gate.s0_status(package=package, repo=repo))


def describe(info):
    """The text shown before the question."""
    fp, prespec = info["fingerprint"], info["prespec"]
    n = len(info["code_manifest"])
    lines = ["", "Unifloral hosts on ThrusterChain: approval of the frozen pre-specification", ""]
    if prespec.get("title"):
        lines.append(f"  Title              {prespec['title']}")
    lines += [f"  Pre-specification  experiments/shift_hosts/PRESPEC.md and prespec.json",
              f"  prespec.json       sha256 {info['prespec_sha256']}",
              f"  Code manifest      {n} files ({info['n_package']} in this package, {n - info['n_package']} elsewhere "
              f"in the repository, Unifloral's four files included)",
              f"  Git                branch {info['branch']}, commit {info['head_commit'][:12]}"]
    s0 = info.get("s0") or {}
    if s0.get("passed"):
        lines.append(f"  Stop rule S0       passed: the stage-0 suite {s0.get('suite_run')} ran on exactly this code")
    else:
        lines.append(f"  Stop rule S0       NOT YET: {s0.get('reason')}. The stage driver refuses to start until a "
                     "stage-0 suite passes on these exact files (plan 6.1: checked before stage 1)")
    summary = prespec.get("approval_summary") or []
    if summary:
        lines += ["", "  What you approve:"] + [f"    - {line}" for line in summary]
    lines += ["", f"  Fingerprint        {fp}",
              f"  First 16           {' '.join(fp[i:i + 4] for i in range(0, 16, 4))}", ""]
    return "\n".join(lines)


def approve(typed, package=gate.PACKAGE, repo=gate.REPO, now=None):
    """Write the approval for the typed characters (spaces and case ignored); returns (approval path, sidecar path).
    Raises Refused, writing nothing, when the characters differ, an approval exists already, or anything changed."""
    info = prepare(package, repo)
    fp = info["fingerprint"]
    typed = "".join(str(typed).split()).lower()
    if typed != fp[:16]:
        raise Refused("the typed characters are not the fingerprint's first 16.")
    path, side = info["path"], gate.sidecar(info["path"])
    if path.exists() or side.exists():
        raise Refused(f"an approval for this fingerprint exists already ({path.name}); nothing was overwritten.")
    now = now or datetime.datetime.now(datetime.timezone.utc)
    body = dict(kind=gate.APPROVAL_KIND, fingerprint=fp, typed=typed, prespec_sha256=info["prespec_sha256"],
                code_manifest=info["code_manifest"], head_commit=info["head_commit"], branch=info["branch"],
                approved_at_utc=now.strftime("%Y-%m-%dT%H:%M:%SZ"), statement=STATEMENT)
    data = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode("utf8")
    with open(path, "xb") as handle:
        handle.write(data)
    try:
        with open(side, "x", encoding="utf8", newline="\n") as handle:
            handle.write(f"{hashlib.sha256(data).hexdigest()}  {path.name}\n")
    except BaseException:
        path.unlink()
        raise
    return path, side


def next_steps(path, side, repo=gate.REPO, branch=BRANCH):
    rel = Path(path).resolve().relative_to(Path(repo).resolve()).as_posix()
    rel_side = Path(side).resolve().relative_to(Path(repo).resolve()).as_posix()
    fp16 = Path(path).name[len(gate.APPROVAL_PREFIX):-len(".json")]
    return "\n".join([
        "", "Written (nothing else was changed):", f"  {rel}", f"  {rel_side}", "",
        "Now commit the two files under your GitHub address and send them on, one line at a time:",
        f"  git add {rel} {rel_side}",
        f"  git -c user.name=\"{COMMIT_NAME}\" -c user.email={gate.APPROVAL_EMAIL} commit -m "
        f"\"Unifloral hosts: approve the pre-specification {fp16}\"",
        f"  GIT_CONFIG_GLOBAL=/root/.bca-gitconfig git push origin {branch}",
        "  exit",
        "Then in PowerShell:",
        "  cd C:\\Users\\David\\dev\\BCA",
        f"  git push origin {branch}",
        "", "Until that commit exists, every run on real data still refuses.", ""])


def main(stdin=None, stdout=None, package=gate.PACKAGE, repo=gate.REPO):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    try:
        if not (stdin.isatty() and stdout.isatty()):
            raise Refused("run this yourself at a terminal (input and output must be a terminal).")
        info = prepare(package, repo)
        if info["path"].exists():
            print(f"\nAn approval for this fingerprint exists already: {info['path'].name}. Nothing to do.", file=stdout)
            return 0
        print(describe(info), file=stdout)
        stdout.write("Type the fingerprint's first 16 characters to approve, or press Enter to stop: ")
        stdout.flush()
        typed = stdin.readline().strip()
        if not typed:
            raise Refused("stopped; nothing was approved.")
        if "".join(typed.split()).lower() != info["fingerprint"][:16]:
            raise Refused("the typed characters are not the fingerprint's first 16.")
        again = prepare(package, repo)
        if again["fingerprint"] != info["fingerprint"]:
            raise Refused("the pre-specification or the code changed while you were reading.")
        path, side = approve(typed, package, repo)
    except (Refused, gate.ApprovalError) as why:
        print(f"\nNot approved, nothing written: {why}", file=stdout)
        return 1
    print(next_steps(path, side, repo, info["branch"]), file=stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
