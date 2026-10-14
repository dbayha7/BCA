#!/usr/bin/env python3
"""Stage 2, version 7: approve the v7 pre-registration at a terminal on YOUR OWN machine (V7_SPEC.md C10, step 10).

Run it yourself, in a terminal, from anywhere, after 'git pull --ff-only' on the branch:

    python3 /home/user/bca/outputs/shift/beyond-the-data/stage2_v7/approve_local_v7.py

What it does, in order. If any step fails, it stops and writes nothing.
  1. Checks the committed files in outputs/shift/beyond-the-data/stage2_v7/:
       - prereg_v7.json matches its fingerprint file prereg_v7.json.sha256, is FINAL (the handoff fields are filled in;
         a draft cannot be approved) and offers this way of approving;
       - code_manifest_v7.json is the manifest prereg_v7.json pins, matches its own .sha256, and every file it lists
         (this script included, in code/ and next to prereg_v7.json) has the fingerprint it records;
       - version 6's committed pre-registration is the one v7 names (v6 stays the governing document).
  2. Refuses if an approval of this fingerprint is already there (it never overwrites one).
  3. Refuses unless standard input is a terminal: you type the answer yourself; nothing may be piped in.
  4. Shows v7's fingerprint, v6's fingerprint, the changes v7 makes, the failures v7 can cause, the handoff summary,
     the probe results, the check (b) forecast gaps and the table of references re-made on the laptop with the lock
     names they become.
  5. Asks you to type the fingerprint's first 16 characters.
  6. Writes approval_v7_<first 16 characters>.json and its .sha256 file in that folder, once, then prints the git
     commands to commit and push them under your own name. Nothing runs until the cloud session has imported it and
     pushed its import record (import_v7_<first 16>.json), which names your commit so you can check it on GitHub.

Standard library only (Python 3.8 or later). It asks git, read-only, for the current commit and branch.
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

REPO = "/home/user/bca"
FOLDER_POSIX = "outputs/shift/beyond-the-data/stage2_v7"
V6_PREREG_POSIX = "outputs/shift/beyond-the-data/stage2/prereg.json"
PREFIX_LEN = 16
APPROVAL_NAME = "approval_v7_{}.json"
KIND = "stage2-v7-local-approval"
MODE = "local-terminal"
PREREG_KIND = "stage2-v7-preregistration"
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


def check_folder(folder=None, script_path=None, repo=REPO):
    """Step 1. Returns what the rest needs; raises Refused on any mismatch."""
    folder = folder or os.path.join(repo, *FOLDER_POSIX.split("/"))
    prereg_path = os.path.join(folder, "prereg_v7.json")
    if not os.path.isfile(prereg_path):
        raise Refused("No v7 pre-registration at {}. Pull the branch first.".format(prereg_path))
    raw = read_bytes(prereg_path)
    digest = sha256_bytes(raw)
    if sidecar_digest(prereg_path) != digest:
        raise Refused("prereg_v7.json does not match its fingerprint file (it was changed).")
    prereg = load_json(raw, "prereg_v7.json")
    if not isinstance(prereg, dict) or prereg.get("kind") != PREREG_KIND:
        raise Refused("prereg_v7.json is not the Stage 2 v7 pre-registration.")
    if prereg.get("approvable") is not True or prereg.get("missing"):
        raise Refused("prereg_v7.json is a draft (missing: {}), so it cannot be approved yet. The final version is "
                      "written after the handoff.".format(prereg.get("missing")))
    section = prereg.get("approval") or {}
    if section.get("local_terminal_allowed") is not True:
        raise Refused("This version does not offer approval from your own terminal.")
    code = prereg.get("code") or {}
    man_path = os.path.join(folder, "code_manifest_v7.json")
    if not os.path.isfile(man_path):
        raise Refused("code_manifest_v7.json is missing.")
    man_raw = read_bytes(man_path)
    man_sha = sha256_bytes(man_raw)
    if man_sha != code.get("code_manifest_v7_sha256") or sidecar_digest(man_path) != man_sha:
        raise Refused("code_manifest_v7.json is not the manifest prereg_v7.json pins, or differs from its sidecar.")
    manifest = load_json(man_raw, "code_manifest_v7.json")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files or files != code.get("files"):
        raise Refused("The manifest's list of files differs from the one prereg_v7.json records.")
    bad = []
    for rel in sorted(files):
        path = os.path.join(folder, *rel.split("/"))
        if not os.path.isfile(path):
            bad.append("{} is missing".format(rel))
        elif file_sha256(path) != files[rel]:
            bad.append("{} differs".format(rel))
    script_path = script_path or os.path.abspath(__file__)
    if files.get("approve_local_v7.py") != file_sha256(script_path):
        bad.append("the script you are running ({}) is not the pinned approve_local_v7.py".format(script_path))
    v6 = os.path.join(repo, *V6_PREREG_POSIX.split("/"))
    pred = prereg.get("predecessor") or {}
    if not os.path.isfile(v6) or file_sha256(v6) != pred.get("prereg_sha256"):
        bad.append("version 6's prereg.json is not the one v7 names ({}...)".format(str(pred.get("prereg_sha256"))[:16]))
    if bad:
        raise Refused("These files do not match the v7 pre-registration:\n  - " + "\n  - ".join(bad))
    return dict(prereg=prereg, sha256=digest, manifest_sha256=man_sha, n_files=len(files),
                script_sha256=files["approve_local_v7.py"], section=section, folder=folder,
                v6_sha256=pred.get("prereg_sha256"))


def git(*args):
    """A read-only git query in the repository; None when git is missing or fails."""
    try:
        done = subprocess.run(("git", "-C", REPO) + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              timeout=20, universal_newlines=True)
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
    h = p.get("handoff") or {}
    parts = ["", "Stage 2, version 7 (finishing the run on the laptop)", wrap(p.get("label", "")),
             "  File: {}/prereg_v7.json".format(FOLDER_POSIX),
             "  Checked: it matches its fingerprint file; the code manifest and all {} pinned files match."
             .format(info["n_files"]), "",
             "Fingerprint (SHA-256) of prereg_v7.json:", "", "    " + info["sha256"], "",
             "Version 6, which stays the governing document for everything not listed below:",
             "    " + str(info["v6_sha256"]), "",
             "What v7 changes (operations only):", bullets(p.get("changes")), "",
             "Failures running on the laptop can cause, which the cloud would not have had:",
             bullets(p.get("failures_v7_can_cause")), "",
             "The handoff:", bullets(h.get("summary")), ""]
    parts.append("Probe results and check (b) forecast gaps (largest parameter differences; tolerance {}):"
                 .format((p.get("remake") or {}).get("forecast_tolerance")))
    rows = []
    for name, r in sorted((h.get("probe_records") or {}).items()):
        rows.append("{}: {}; differs {}; F_ref {}; F_fresh {}".format(
            name, r.get("result"), r.get("differs"), r.get("F_ref"), r.get("F_fresh")))
    parts += [bullets(rows), "", "References re-made on the laptop (each replaces only these actors of that seed and "
              "world; deployment always reads the gate's):"]
    rows = []
    for e in h.get("remake_table") or []:
        rows.append("{} seed {} ({} world): {}".format(e.get("phase"), e.get("seed"), e.get("world"), ", ".join(
            "{} -> lock name '{}'".format(k, v.get("lock_name")) for k, v in sorted((e.get("replace") or {}).items()))))
    parts += [bullets(rows or ["none: every seed 2-4 phase reads the gate's saved files"]), "",
              "Labels per seed and component:", bullets(["{}: {}".format(k, v) for k, v in
                                                          sorted((h.get("labels") or {}).items())]), "",
              "Read it first: {}/PREREGISTRATION_V7.md.".format(FOLDER_POSIX), ""]
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


def approval_body(info, typed):
    p = info["prereg"]
    digest = info["sha256"]
    try:
        tty = os.ttyname(sys.stdin.fileno())
    except (AttributeError, OSError, ValueError):
        tty = None
    statement = ("I approve the Stage 2 version 7 pre-registration whose SHA-256 is {} (operational changes to "
                 "version 6, whose pre-registration {} stays the governing document), and only that exact file. "
                 "Approving it allows the runs it names, under its rules and version 6's.".format(
                     digest, info["v6_sha256"]))
    return dict(kind=KIND, label=p.get("label"), mode=MODE, prereg_path=FOLDER_POSIX + "/prereg_v7.json",
                prereg_version=p.get("version"), prereg_sha256=digest, v6_prereg_sha256=info["v6_sha256"],
                typed=typed, stdin_isatty=True, tty=tty, approved_at_utc=utc_now(),
                hostname=safe(socket.gethostname) or safe(platform.node), local_user=safe(getpass.getuser),
                git_head=git("rev-parse", "HEAD"), git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
                python=platform.python_version(), code_manifest_sha256=info["manifest_sha256"],
                approve_local_sha256=info["script_sha256"], statement=statement,
                how=("Typed at a terminal on the approver's own machine with approve_local_v7.py. It takes effect only "
                     "after the approver commits and pushes it and the cloud session imports it (approve_v7.py import)."))


def main(argv=None, folder=None, script_path=None, stdin=None, input_fn=input, repo=REPO):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check and print the summary; write nothing")
    args = parser.parse_args(argv)
    stdin = sys.stdin if stdin is None else stdin
    if sys.version_info < (3, 8):
        print("Refused: this needs Python 3.8 or later.", file=sys.stderr)
        return 2
    try:
        info = check_folder(folder, script_path, repo)
        name = APPROVAL_NAME.format(info["sha256"][:PREFIX_LEN])
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
        prefix = info["sha256"][:PREFIX_LEN]
        try:
            typed = input_fn("To approve, type the first {} characters of the fingerprint above (or press Enter to "
                             "stop): ".format(PREFIX_LEN))
        except (EOFError, KeyboardInterrupt):
            print()
            raise Refused("Not approved: stopped before an answer was typed.")
        typed = typed.strip()
        if not typed:
            raise Refused("Not approved: nothing was typed.")
        if typed != prefix:
            raise Refused("Not approved: the characters typed do not match the first {} characters of the "
                          "fingerprint.".format(PREFIX_LEN))
        body = approval_body(info, typed)
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
        if file_sha256(path) != digest or sidecar_digest(path) != digest:
            os.remove(path)
            os.remove(path + ".sha256")
            raise Refused("The approval file did not read back as written, so it was removed. Run this again.")
    except Refused as problem:
        print("\nRefused. " + str(problem) + "\nNothing was written.", file=sys.stderr)
        return 2
    branch = info["section"].get("branch") or "<the branch you pulled>"
    approver = info["section"].get("approver_email") or "<your GitHub no-reply address>"
    print("\n".join([
        "", "Approval written:", "  {}/{}".format(FOLDER_POSIX, name), "  {}/{}.sha256".format(FOLDER_POSIX, name), "",
        "It covers the v7 pre-registration with fingerprint {} only.".format(info["sha256"]), "",
        "Now commit and push it under your own name, from {}, with these exact commands:".format(repo), "",
        "    git add {0}/{1} {0}/{1}.sha256".format(FOLDER_POSIX, name),
        "    git -c user.name=\"David Bayha\" -c user.email={} commit -m \"Stage 2: approve v7 {}\"".format(
            approver, prefix),
        "    git push origin {}".format(branch), "",
        "Once pushed, do not delete, edit or re-add this file. Then tell Claude in the chat that you have pushed.",
        "Nothing runs until the cloud session has checked your commit with git and pushed its import record",
        "({}/import_v7_{}.json), which names your commit so that you can check it on GitHub.".format(FOLDER_POSIX,
                                                                                                    prefix),
        "If you change your mind before pushing, delete the two files instead.", ""]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
