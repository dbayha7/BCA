#!/usr/bin/env python3
"""Near-80 (version 8): approve the final pre-registration at a terminal on YOUR OWN machine (PREREGISTRATION.md
section 11).

Run it yourself, on the Framework Desktop, inside Ubuntu as root (wsl -d Ubuntu-24.04 -u root; the repository at
/home/user/bca with version 7's restored runs/, which the code manifest pins), typing:

    cd /home/user/bca
    git pull --ff-only
    python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py --check
    python3 -I outputs/shift/beyond-the-data/near80/approve_local_near80.py

Keep the -I (isolated mode): Python then loads nothing from this script's own folder (nor from PYTHON* settings), so
no stray file there (a json.py, say) can run before the checks below.

What it does, in order. If any step fails, it prints "Refused." with the reason, writes nothing and exits with 2.
  1. Checks the committed files in outputs/shift/beyond-the-data/near80/:
       - prereg.json matches its fingerprint file prereg.json.sha256, is the near-80 pre-registration, is FINAL
         ("final, awaiting approval"; a draft cannot be approved), misses nothing, and offers approval from your own
         terminal with your GitHub address;
       - code_manifest.json is the manifest prereg.json pins and matches its own fingerprint file; every file it lists
         (this script included) has the fingerprint it records; every repository file it pins is unchanged;
       - the frozen code's no-training checks on seeds 0-4 (pre_approval/no_training_checks_seeds0_4.json) are the
         ones prereg.json pins, and they passed;
       - the draft the final version was built from (prereg_history/prereg_draft_<first 16>.json) is the one pinned;
       - the folder and its code/ hold no other file: only the files the manifest lists, prereg.json,
         code_manifest.json (each with its .sha256) and this fingerprint's own approval and import records; no
         __pycache__ and no other folder holding Python files.
  2. Refuses if an approval of this fingerprint is already there (it never overwrites one).
  3. Refuses unless standard input is a terminal: you type the answer yourself; nothing may be piped in.
  4. Shows a summary: the fingerprint, your answer accepting the defaults, the defaults taken, the seeds, the three
     settings, the four claims, the stop rules, the no-training result, the compute forecast and cap.
  5. Asks you to type the fingerprint's first 16 characters.
  6. Writes approval_near80_<first 16 characters>.json and its .sha256 file in that folder, once, reads them back,
     and prints the git commands to commit and push them under your own name. Nothing runs until the session has
     imported your approval and pushed its import record (import_near80_<first 16 characters>.json).

Standard library only (Python 3.8 or later); it imports nothing from code/. It asks git, read-only, for the current
commit and branch.
Options:
  --check    run the checks of step 1 and print the summary; ask nothing and write nothing.
"""

import argparse
import datetime
import getpass
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import textwrap

sys.dont_write_bytecode = True

REPO = "/home/user/bca"
FOLDER_POSIX = "outputs/shift/beyond-the-data/near80"
PREREG_NAME = "prereg.json"
MANIFEST_NAME = "code_manifest.json"
PREFIX_LEN = 16
APPROVAL_NAME = "approval_near80_{}.json"
IMPORT_NAME = "import_near80_{}.json"
KIND = "near80-local-approval"
MODE = "local-terminal"
PREREG_KIND = "near80-preregistration"
STATUS_FINAL = "final, awaiting approval"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
BRANCH = "claude/main-bca-connection-x4o46s"
SCRIPT_NAME = "approve_local_near80.py"
FROZEN_NAME = "pre_approval/no_training_checks_seeds0_4.json"
FROZEN_TOP = (PREREG_NAME, PREREG_NAME + ".sha256", MANIFEST_NAME, MANIFEST_NAME + ".sha256")
STATEMENT = ("I approve the near-80 pre-registration ({version}) whose SHA-256 is {sha}, and only that exact file. "
             "Approving it accepts every default of its section 16 and allows the runs it names, under its rules.")
WIDTH = 100


class Refused(Exception):
    """A check failed: nothing is written."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_bytes(path):
    with open(path, "rb") as handle:
        return handle.read()


def file_sha256(path):
    return sha256_bytes(read_bytes(path))


def sidecar_digest(path):
    side = path + ".sha256"
    if not os.path.isfile(side):
        raise Refused("{} is missing.".format(side))
    words = read_bytes(side).decode("utf8", "replace").split()
    if not words:
        raise Refused("{} is empty.".format(side))
    return words[0]


def load_json(data, name):
    try:
        return json.loads(data.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise Refused("{} is not readable JSON.".format(name))


def unlisted_files(folder, files, prefix):
    """Step 1's last check: every entry of the folder and of its code/ that is neither a file the manifest lists, one
    of the four frozen files at the top (prereg.json, code_manifest.json, each with its .sha256), nor this
    fingerprint's own approval or import record (with .sha256); any __pycache__; any other folder holding a .py file.
    Returns the problems (empty when none)."""
    allowed = set(f for f in files if "/" not in f) | set(FROZEN_TOP)
    for name in (APPROVAL_NAME.format(prefix), IMPORT_NAME.format(prefix)):
        allowed.update((name, name + ".sha256"))
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name == "__pycache__":
            out.append("{} exists (compiled Python bytecode); remove it".format(name))
        elif os.path.isdir(path) and not os.path.islink(path):
            if name == "code":
                continue
            for root, dirs, names in os.walk(path):
                if any(n.endswith(".py") or n.endswith(".pyc") for n in names) or "__pycache__" in dirs:
                    out.append("{}/ holds Python files (only code/ may)".format(name))
                    break
        elif name not in allowed:
            out.append("{} is not a file of this pre-registration (the code manifest does not list it)".format(name))
    code = os.path.join(folder, "code")
    if os.path.isdir(code):
        for name in sorted(os.listdir(code)):
            if name == "__pycache__":
                out.append("code/__pycache__ exists (compiled Python bytecode); remove it")
            elif "code/" + name not in files or not os.path.isfile(os.path.join(code, name)):
                out.append("code/{} is not in the code manifest".format(name))
    return out


def check_folder(folder=None, script_path=None, repo=REPO):
    """Step 1. Returns what the rest needs; raises Refused on any mismatch."""
    folder = folder or os.path.join(repo, *FOLDER_POSIX.split("/"))
    prereg_path = os.path.join(folder, PREREG_NAME)
    if not os.path.isfile(prereg_path):
        raise Refused("No near-80 pre-registration at {}. Pull the branch first.".format(prereg_path))
    raw = read_bytes(prereg_path)
    digest = sha256_bytes(raw)
    if sidecar_digest(prereg_path) != digest:
        raise Refused("prereg.json does not match its fingerprint file (it was changed).")
    prereg = load_json(raw, PREREG_NAME)
    if not isinstance(prereg, dict) or prereg.get("kind") != PREREG_KIND:
        raise Refused("prereg.json is not the near-80 pre-registration.")
    if prereg.get("status") != STATUS_FINAL:
        raise Refused("prereg.json is not final (its status is {!r}), so it cannot be approved.".format(
            prereg.get("status")))
    if prereg.get("approvable") is not True or prereg.get("missing") != []:
        raise Refused("prereg.json is not approvable (missing: {}).".format(prereg.get("missing")))
    section = prereg.get("approval") or {}
    if section.get("local_terminal_allowed") is not True:
        raise Refused("This pre-registration does not offer approval from your own terminal.")
    if section.get("approver_email") != APPROVER_EMAIL:
        raise Refused("This pre-registration names another approver ({}).".format(section.get("approver_email")))
    code = prereg.get("code") or {}
    man_path = os.path.join(folder, MANIFEST_NAME)
    if not os.path.isfile(man_path):
        raise Refused("code_manifest.json is missing.")
    man_raw = read_bytes(man_path)
    man_sha = sha256_bytes(man_raw)
    if man_sha != code.get("code_manifest_sha256") or sidecar_digest(man_path) != man_sha:
        raise Refused("code_manifest.json is not the manifest prereg.json pins, or differs from its fingerprint file.")
    manifest = load_json(man_raw, MANIFEST_NAME)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files or files != code.get("files"):
        raise Refused("The manifest's list of files differs from the one prereg.json records.")
    bad = []
    for rel in sorted(files):
        path = os.path.join(folder, *rel.split("/"))
        if not os.path.isfile(path):
            bad.append("{} is missing".format(rel))
        elif file_sha256(path) != files[rel]:
            bad.append("{} differs".format(rel))
    script_path = script_path or os.path.abspath(__file__)
    if files.get(SCRIPT_NAME) is None or files.get(SCRIPT_NAME) != file_sha256(script_path):
        bad.append("the script you are running ({}) is not the pinned {}".format(script_path, SCRIPT_NAME))
    repo_files = manifest.get("repository_files")
    if not isinstance(repo_files, dict):
        bad.append("the manifest lists no repository files")
        repo_files = {}
    missing_runs = False
    for rel in sorted(repo_files):
        path = os.path.join(repo, *rel.split("/"))
        if not os.path.isfile(path):
            bad.append("{} is missing".format(rel))
            missing_runs = missing_runs or rel.startswith("runs/")
        elif file_sha256(path) != repo_files[rel]:
            bad.append("{} differs".format(rel))
    frozen = prereg.get("no_training_checks_frozen") or {}
    fpath = os.path.join(folder, *str(frozen.get("path") or FROZEN_NAME).split("/"))
    if frozen.get("path") != FROZEN_NAME:
        bad.append("prereg.json does not pin {}".format(FROZEN_NAME))
    elif not os.path.isfile(fpath):
        bad.append("{} is missing".format(FROZEN_NAME))
    else:
        fsha = file_sha256(fpath)
        try:
            side = sidecar_digest(fpath)
        except Refused:
            side = None
        if fsha != frozen.get("sha256") or side != fsha:
            bad.append("{} is not the one prereg.json pins, or differs from its fingerprint file".format(FROZEN_NAME))
        else:
            body = load_json(read_bytes(fpath), FROZEN_NAME)
            checks = body.get("no_training_checks") if isinstance(body, dict) else None
            if (frozen.get("passed") is not True or not isinstance(checks, dict) or checks.get("passed") is not True
                    or checks.get("not_run") != []):
                bad.append("the frozen code's no-training checks on seeds 0-4 did not all pass")
    draft = prereg.get("draft") or {}
    dsha = draft.get("sha256")
    drel = str(draft.get("path") or "")
    if not dsha or drel != "prereg_history/prereg_draft_{}.json".format(str(dsha)[:PREFIX_LEN]):
        bad.append("prereg.json does not pin the draft it was built from")
    else:
        dpath = os.path.join(folder, *drel.split("/"))
        if not os.path.isfile(dpath) or file_sha256(dpath) != dsha:
            bad.append("{} is missing or differs".format(drel))
    bad.extend(unlisted_files(folder, files, digest[:PREFIX_LEN]))
    if bad:
        hint = ("\n  (runs/ is not in git: run this on the machine where version 7's restore placed runs/, or restore "
                "it first.)" if missing_runs else "")
        raise Refused("These files do not match the near-80 pre-registration:\n  - " + "\n  - ".join(bad) + hint)
    return dict(prereg=prereg, sha256=digest, manifest_sha256=man_sha, n_files=len(files),
                n_repo_files=len(repo_files), script_sha256=files[SCRIPT_NAME], section=section, folder=folder)


def git(repo, *args):
    """A read-only git query in the repository; None when git is missing or fails."""
    try:
        done = subprocess.run(("git", "-C", repo) + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              stdin=subprocess.DEVNULL, timeout=20, universal_newlines=True)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    out = (done.stdout or "").strip()
    return out if done.returncode == 0 and out else None


def wrap(text, first="  ", rest=None):
    return textwrap.fill(" ".join(str(text).split()), width=WIDTH, initial_indent=first,
                         subsequent_indent=first if rest is None else rest)


def bullets(items):
    return "\n".join(wrap(item, "  - ", "    ") for item in items) if items else "  (none)"


def summary(info):
    """Step 4's text."""
    p = info["prereg"]
    s = p.get("summary") or {}
    answer = p.get("owner_answer") or {}
    seeds = p.get("seeds") or {}
    frozen = p.get("no_training_checks_frozen") or {}
    compute = p.get("compute") or {}
    choices = ["{}. {}: {}".format(c.get("number"), c.get("choice"), json.dumps(c.get("taken")))
               for c in p.get("choices") or []]
    settings = ["{}: {}".format(k, (v or {}).get("role")) for k, v in sorted((p.get("settings") or {}).items())]
    parts = ["", "Near-80 (version 8): hiding the commands near 80%", wrap(p.get("label", "")),
             "  File: {}/{}".format(FOLDER_POSIX, PREREG_NAME),
             "  Checked: it matches its fingerprint file; the code manifest, all {} pinned near-80 files and all {} "
             "pinned repository files match.".format(info["n_files"], info["n_repo_files"]), "",
             "Fingerprint (SHA-256) of prereg.json:", "", "    " + info["sha256"], "",
             "Your answer: \"{}\" ({}, {}): {}.".format(answer.get("text"), answer.get("utc"), answer.get("where"),
                                                       answer.get("choices")), "",
             "The defaults of section 16 that approving takes:", bullets(choices), "",
             "Seeds: {} exploratory (seen); {} fresh (the only ones that can carry a claim); {} refused always.".format(
                 seeds.get("exploratory"), seeds.get("confirmatory"), seeds.get("refused_always")), "",
             "The three settings of the data:", bullets(settings), "",
             "The four claims (read only on seeds 10-14):", bullets(s.get("claims")), "",
             "The stop rules on seeds 0-4:", bullets(s.get("stop_rules")), "",
             "The frozen code's no-training checks on seeds 0-4 (before approval, nothing trained): {} rules, "
             "{}.".format(frozen.get("rules"), "all passed" if frozen.get("passed") is True else "NOT ALL PASSED"), "",
             "Compute: {} CPU-hours forecast in all; the cap for the whole version is {} CPU-hours.".format(
                 compute.get("total"), compute.get("budget_cap")), "",
             "Read it first: {}/PREREGISTRATION.md".format(FOLDER_POSIX), ""]
    return "\n".join(parts)


def write_once(path, data):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe(fn):
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - evidence only
        return None
    return value or None


def approval_body(info, typed, stdin=None, repo=REPO):
    p = info["prereg"]
    digest = info["sha256"]
    try:
        tty = os.ttyname((stdin or sys.stdin).fileno())
    except (AttributeError, OSError, ValueError):
        tty = None
    return dict(kind=KIND, label=p.get("label"), mode=MODE, prereg_path=FOLDER_POSIX + "/" + PREREG_NAME,
                prereg_sha256=digest, typed=typed, stdin_isatty=True, tty=tty, approved_at_utc=utc_now(),
                hostname=safe(socket.gethostname) or safe(platform.node), local_user=safe(getpass.getuser),
                git_head=git(repo, "rev-parse", "HEAD"), git_branch=git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                python=platform.python_version(), code_manifest_sha256=info["manifest_sha256"],
                approve_local_sha256=info["script_sha256"],
                statement=STATEMENT.format(version=p.get("version"), sha=digest),
                how=("Typed at a terminal on the owner's own machine with approve_local_near80.py. It takes effect "
                     "only after the owner commits and pushes it and the session imports it "
                     "(code/import_approval.py import) and pushes the import record."))


def main(argv=None, folder=None, script_path=None, stdin=None, input_fn=input, repo=REPO):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check and print the summary; write nothing")
    args = parser.parse_args(argv)
    stdin = sys.stdin if stdin is None else stdin
    if sys.version_info < (3, 8):
        print("Refused. This needs Python 3.8 or later.\nNothing was written.", file=sys.stderr)
        return 2
    try:
        info = check_folder(folder, script_path, repo)
        prefix = info["sha256"][:PREFIX_LEN]
        name = APPROVAL_NAME.format(prefix)
        path = os.path.join(info["folder"], name)
        if args.check:
            print(summary(info))
            print("--check: every check passed. Nothing was asked and nothing was written.")
            return 0
        if os.path.lexists(path) or os.path.lexists(path + ".sha256"):
            raise Refused("{} already exists: this fingerprint is approved already. An approval is never "
                          "overwritten.".format(name))
        if not (hasattr(stdin, "isatty") and stdin.isatty()):
            raise Refused("Standard input is not a terminal. Run this yourself in a terminal window; do not pipe or "
                          "redirect anything into it.")
        print(summary(info))
        try:
            typed = input_fn("To approve, type the first {} characters of the fingerprint above (or press Enter to "
                             "stop): ".format(PREFIX_LEN))
        except (EOFError, KeyboardInterrupt):
            print()
            raise Refused("Not approved: stopped before an answer was typed.")
        typed = (typed or "").strip()
        if not typed:
            raise Refused("Not approved: nothing was typed.")
        if typed != prefix:
            raise Refused("Not approved: the characters typed do not match the first {} characters of the "
                          "fingerprint.".format(PREFIX_LEN))
        body = approval_body(info, typed, stdin, repo)
        text = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True).encode("utf8")
        digest = sha256_bytes(text)
        try:
            write_once(path, text)
        except FileExistsError:
            raise Refused("{} appeared while you were typing. Nothing was overwritten.".format(name))
        try:
            write_once(path + ".sha256", "{}  {}\n".format(digest, name).encode("utf8"))
        except BaseException:
            os.remove(path)
            raise
        try:
            ok = file_sha256(path) == digest and sidecar_digest(path) == digest
        except (OSError, Refused):
            ok = False
        if not ok:
            for p in (path, path + ".sha256"):
                if os.path.lexists(p):
                    os.remove(p)
            raise Refused("The approval file did not read back as written, so it was removed. Run this again.")
    except Refused as problem:
        print("\nRefused. " + str(problem) + "\nNothing was written.", file=sys.stderr)
        return 2
    branch = info["section"].get("branch") or BRANCH
    rel = "{}/{}".format(FOLDER_POSIX, name)
    print("\n".join([
        "", "Approval written:", "  " + rel, "  " + rel + ".sha256", "",
        "It covers the near-80 pre-registration with fingerprint {} only.".format(info["sha256"]), "",
        "Now commit and push it under your own name, from {}, with these exact commands:".format(repo), "",
        "    git add {0} {0}.sha256".format(rel),
        "    git -c user.name=\"{}\" -c user.email={} commit -m \"Near-80: approve version 8 {}\"".format(
            APPROVER_NAME, APPROVER_EMAIL, prefix),
        "    git push origin {}".format(branch), "",
        "If the push is rejected (\"fetch first\" or \"non-fast-forward\": the session pushed something meanwhile), type:",
        "", "    git -c user.name=\"{}\" -c user.email={} pull --rebase origin {}".format(
            APPROVER_NAME, APPROVER_EMAIL, branch),
        "    git push origin {}".format(branch), "",
        "Once pushed, do not delete, edit or re-add these files. Then tell Claude in the chat that you have pushed;",
        "nothing runs until the session has imported it and pushed {}.".format(IMPORT_NAME.format(prefix)),
        "When the session says it has pushed the import record, bring it here with:", "",
        "    git -C {} pull --ff-only".format(repo), "",
        "If you change your mind before pushing, delete the two files instead.", ""]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
