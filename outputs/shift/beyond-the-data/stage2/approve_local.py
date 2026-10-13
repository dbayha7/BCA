#!/usr/bin/env python3
"""Stage 2: approve the pre-registration at a terminal on YOUR OWN machine.

Run it yourself, in a terminal, from the root of your copy of the repository, after pulling the branch:

    python3 outputs/shift/beyond-the-data/stage2/approve_local.py

(On Windows: py outputs\\shift\\beyond-the-data\\stage2\\approve_local.py)

What it does, in order. If any step fails, it stops and writes nothing.
  1. Checks the committed files in outputs/shift/beyond-the-data/stage2/:
       - prereg.json matches its fingerprint file prereg.json.sha256;
       - it is final (no missing pin) and offers this way of approving;
       - code_manifest.json is the manifest prereg.json pins, and matches its own .sha256;
       - every file in code/, and PREREGISTRATION.md and approve_local.py next to it, and this script itself, have
         the fingerprint the manifest records.
  2. Refuses if an approval of this fingerprint is already there (it never overwrites one). Each fingerprint has its
     own file, approval_<first 16 characters>.json, so an approval of an earlier version never blocks a later one.
  3. Refuses unless standard input is a terminal: you type the answer yourself; nothing may be piped in.
  4. Prints the pre-registration's full fingerprint (its SHA-256), what approving allows, and where to read it.
  5. Asks you to type the fingerprint's first 16 characters. Spaces at either end are ignored; otherwise the
     characters must match exactly.
  6. Writes approval_<first 16 characters>.json and its .sha256 file in that folder, then prints the git commands to
     commit and push them. The cloud session imports the approval only after your push. Its import reads your commit
     with git, and its record of what it imported (import_<first 16 characters>.json) is pushed to the branch too, so
     you can check on GitHub that the commit it names is yours.

It uses only the Python standard library (Python 3.8 or later). It asks git, read-only, for the current commit,
branch and configured author; if git is not available those fields are left empty.

Options:
  --check    run the checks in step 1 and print the fingerprint; ask nothing and write nothing.
  --mode M   accepted for convenience; the only mode here is 'local-terminal' ('terminal' means the same).
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

FOLDER_PARTS = ("outputs", "shift", "beyond-the-data", "stage2")
FOLDER = os.path.join(*FOLDER_PARTS)
FOLDER_POSIX = "/".join(FOLDER_PARTS)
PREFIX_LEN = 16
APPROVAL_NAME = "approval_{}.json"       # one file per fingerprint: approval_<first 16 characters>.json
KIND = "stage2-local-approval"
MODE = "local-terminal"
PREREG_KIND = "stage2-preregistration"
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


def posix(path):
    return path.replace(os.sep, "/")


def line_ending_hint(data):
    if b"\r\n" in data:
        return (" The file holds Windows line endings (CR LF), so something converted it after checkout; check "
                "'git config core.autocrlf' and check the branch out again.")
    return ""


def sidecar_digest(path):
    side = path + ".sha256"
    if not os.path.isfile(side):
        raise Refused("{} is missing.".format(posix(side)))
    words = read_bytes(side).decode("utf8", "replace").split()
    if not words:
        raise Refused("{} is empty.".format(posix(side)))
    return words[0]


def load_json(data, name):
    try:
        return json.loads(data.decode("utf8"))
    except (UnicodeDecodeError, ValueError):
        raise Refused("{} is not readable JSON.".format(name))


def check_folder(folder=FOLDER, script_path=None):
    """Step 1. Returns what the rest needs; raises Refused on any mismatch."""
    prereg_path = os.path.join(folder, "prereg.json")
    if not os.path.isfile(prereg_path):
        raise Refused("No pre-registration at {}. Run this from the root of the repository, after "
                      "'git pull' on the branch.".format(posix(prereg_path)))
    raw = read_bytes(prereg_path)
    digest = sha256_bytes(raw)
    if sidecar_digest(prereg_path) != digest:
        raise Refused("prereg.json does not match its fingerprint file prereg.json.sha256 (it was changed)."
                      + line_ending_hint(raw))
    prereg = load_json(raw, "prereg.json")
    if not isinstance(prereg, dict) or prereg.get("kind") != PREREG_KIND:
        raise Refused("prereg.json is not the Stage 2 pre-registration.")
    if prereg.get("approvable") is not True or prereg.get("missing_pins"):
        raise Refused("prereg.json is a draft (some pin is missing), so it cannot be approved.")
    approval_section = prereg.get("approval") or {}
    if approval_section.get("local_terminal_allowed") is not True:
        raise Refused("This version of the pre-registration does not offer approval from your own terminal.")

    code = prereg.get("code") or {}
    manifest_path = os.path.join(folder, "code_manifest.json")
    if not os.path.isfile(manifest_path):
        raise Refused("{} is missing.".format(posix(manifest_path)))
    manifest_raw = read_bytes(manifest_path)
    manifest_sha = sha256_bytes(manifest_raw)
    if manifest_sha != code.get("code_manifest_sha256"):
        raise Refused("code_manifest.json is not the manifest prereg.json pins." + line_ending_hint(manifest_raw))
    if sidecar_digest(manifest_path) != manifest_sha:
        raise Refused("code_manifest.json does not match its fingerprint file.")
    manifest = load_json(manifest_raw, "code_manifest.json")
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files or files != code.get("files"):
        raise Refused("The manifest's list of files differs from the one prereg.json records.")

    bad = []
    code_dir = os.path.join(folder, "code")
    for name in sorted(files):
        path = os.path.join(code_dir, name)
        if not os.path.isfile(path):
            bad.append("code/{} is missing".format(name))
        elif file_sha256(path) != files[name]:
            bad.append("code/{} differs".format(name) + line_ending_hint(read_bytes(path)))
    if os.path.isdir(code_dir):
        for name in sorted(os.listdir(code_dir)):
            if name.startswith(".") or not os.path.isfile(os.path.join(code_dir, name)):
                continue
            if name not in files:
                bad.append("code/{} is not in the manifest".format(name))
    for name in ("PREREGISTRATION.md", "approve_local.py"):
        path = os.path.join(folder, name)
        if name not in files:
            bad.append("{} is not pinned by the manifest".format(name))
        elif not os.path.isfile(path):
            bad.append("{} is missing".format(name))
        elif file_sha256(path) != files[name]:
            bad.append("{} differs from the pinned copy".format(name))
    script_path = script_path or os.path.abspath(__file__)
    if files.get("approve_local.py") != file_sha256(script_path):
        bad.append("the script you are running ({}) is not the pinned approve_local.py".format(script_path))
    if bad:
        raise Refused("These files do not match the pre-registration's code manifest:\n  - " + "\n  - ".join(bad))
    return dict(prereg=prereg, sha256=digest, manifest_sha256=manifest_sha, n_files=len(files),
                script_sha256=files["approve_local.py"], approval_section=approval_section)


def git(*args):
    """A read-only git query from the current folder; None when git is missing or fails."""
    try:
        done = subprocess.run(("git",) + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=20,
                              universal_newlines=True)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    out = (done.stdout or "").strip()
    return out if done.returncode == 0 and out else None


def wrap(text, first="  ", rest=None):
    return textwrap.fill(" ".join(str(text).split()), width=WIDTH, initial_indent=first,
                         subsequent_indent=first if rest is None else rest)


def bullets(items):
    return "\n".join(wrap(item, "  - ", "    ") for item in items)


def show(info):
    prereg = info["prereg"]
    section = info["approval_section"]
    allows = section.get("what_approval_allows") or []
    not_allowed = section.get("what_approval_does_not_allow") or []
    parts = [
        "",
        "Stage 2 pre-registration, version {}".format(prereg.get("version")),
        wrap(prereg.get("label", "")),
        "  File: {}/prereg.json".format(FOLDER_POSIX),
        "  Checked: it matches its fingerprint file, and the code manifest and all {} pinned files match."
        .format(info["n_files"]),
        "",
        "Fingerprint (SHA-256) of prereg.json:",
        "",
        "    " + info["sha256"],
        "",
        "What approving allows (the BCA runs this pre-registration names):",
        bullets(allows) if allows else "  (see PREREGISTRATION.md)",
    ]
    if not_allowed:
        parts += ["", "What approving does not allow:", bullets(not_allowed)]
    parts += [
        "",
        "Approving accepts every default the pre-registration lists (PREREGISTRATION.md section 14). Any other",
        "choice needs a new version and a new fingerprint.",
        "",
        "Read it first: {}/PREREGISTRATION.md (section 9 is about this approval).".format(FOLDER_POSIX),
        "",
    ]
    print("\n".join(parts))


def write_once(path, data):
    """Create the file; refuse if it exists. Bytes are written as they are (no line-ending conversion)."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe(fn):
    try:
        value = fn()
    except Exception:  # noqa: BLE001 - evidence only; never stop the approval for it
        return None
    return value or None


def approval_body(info, typed):
    prereg = info["prereg"]
    digest = info["sha256"]
    version = prereg.get("version")
    try:
        tty = os.ttyname(sys.stdin.fileno())
    except (AttributeError, OSError, ValueError):
        tty = None
    statement = ("I approve the Stage 2 pre-registration whose SHA-256 is {} (version {}), and only that exact file: "
                 "no earlier or later version, no edit of it and no other stage. Approving it accepts every default it "
                 "lists and allows the BCA runs it names, under its rules.".format(digest, version))
    return dict(
        kind=KIND,
        label=prereg.get("label"),
        mode=MODE,
        prereg_path=FOLDER_POSIX + "/prereg.json",
        prereg_version=version,
        prereg_sha256=digest,
        typed=typed,
        stdin_isatty=True,
        tty=tty,
        approved_at_utc=utc_now(),
        hostname=safe(socket.gethostname) or safe(platform.node),
        local_user=safe(getpass.getuser),
        git_head=git("rev-parse", "HEAD"),
        git_branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        python=platform.python_version(),
        code_manifest_sha256=info["manifest_sha256"],
        approve_local_sha256=info["script_sha256"],
        statement=statement,
        how=("Typed at a terminal on the approver's own machine with approve_local.py. It takes effect only after the "
             "approver commits and pushes it and the cloud session imports it (approve.py import), which records the "
             "commit and its author."),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="check and print the fingerprint; write nothing")
    parser.add_argument("--mode", choices=("local-terminal", "terminal"), default="local-terminal",
                        help="accepted for convenience; there is only one mode here")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 8):
        print("Refused: this needs Python 3.8 or later.", file=sys.stderr)
        return 2
    try:
        info = check_folder()
        name = APPROVAL_NAME.format(info["sha256"][:PREFIX_LEN])
        approval_path = os.path.join(FOLDER, name)
        if args.check:
            show(info)
            print("--check: every check passed. Nothing was asked and nothing was written.")
            print("An approval would be written to {}/{}.".format(FOLDER_POSIX, name))
            return 0
        if os.path.lexists(approval_path) or os.path.lexists(approval_path + ".sha256"):
            raise Refused("{} already exists: this fingerprint is approved already. An approval is never "
                          "overwritten.".format(posix(approval_path)))
        if not (hasattr(sys.stdin, "isatty") and sys.stdin.isatty()):
            raise Refused("Standard input is not a terminal. Run this yourself in a terminal window; do not pipe or "
                          "redirect anything into it.")
        show(info)
        others = sorted(n for n in os.listdir(FOLDER) if n.startswith("approval_") and n.endswith(".json") and n != name)
        if others:
            print(wrap("Note: this folder also holds {}, an approval of another version. It stays as it is and does "
                       "not approve this version.".format(", ".join(others)), "", ""))
            print()
        prefix = info["sha256"][:PREFIX_LEN]
        try:
            typed = input("To approve, type the first {} characters of the fingerprint above (or press Enter to stop): "
                          .format(PREFIX_LEN))
        except (EOFError, KeyboardInterrupt):
            print()
            raise Refused("Not approved: stopped before an answer was typed.")
        typed = typed.strip()
        if not typed:
            raise Refused("Not approved: nothing was typed.")
        if typed != prefix:
            hint = " The fingerprint uses lower-case letters; type them as shown." if typed.lower() == prefix else ""
            raise Refused("Not approved: the characters typed do not match the first {} characters of the "
                          "fingerprint.{}".format(PREFIX_LEN, hint))
        body = approval_body(info, typed)
        text = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True).encode("utf8")
        digest = sha256_bytes(text)
        try:
            write_once(approval_path, text)
        except FileExistsError:
            raise Refused("{} appeared while you were typing. Nothing was overwritten.".format(posix(approval_path)))
        try:
            write_once(approval_path + ".sha256", "{}  {}\n".format(digest, name).encode("utf8"))
        except FileExistsError:
            os.remove(approval_path)
            raise Refused("{}.sha256 appeared while you were typing. Nothing was overwritten.".format(
                posix(approval_path)))
        except BaseException:
            os.remove(approval_path)
            raise
        if file_sha256(approval_path) != digest or sidecar_digest(approval_path) != digest:
            os.remove(approval_path)
            os.remove(approval_path + ".sha256")
            raise Refused("The approval file did not read back as written, so it was removed. Run this again.")
    except Refused as problem:
        print("\nRefused. " + str(problem) + "\nNothing was written.", file=sys.stderr)
        return 2

    branch = (info["approval_section"].get("local_terminal") or {}).get("branch") or "<the branch you pulled>"
    prefix = info["sha256"][:PREFIX_LEN]
    lines = [
        "",
        "Approval written:",
        "  {}/{}".format(FOLDER_POSIX, name),
        "  {}/{}.sha256".format(FOLDER_POSIX, name),
        "",
        "It covers the pre-registration with fingerprint {} only.".format(info["sha256"]),
        "",
    ]
    if body.get("git_branch") and body["git_branch"] != branch:
        lines += ["Note: you are on branch '{}'. The approval must reach branch '{}'.".format(
            body["git_branch"], branch), ""]
    author = git("config", "user.name"), git("config", "user.email")
    if all(author):
        lines += ["Git will record the commit's author as: {} <{}>".format(*author), ""]
    lines += [
        "Now commit and push it, from the repository root, with these exact commands:",
        "",
        "    git add {0}/{1} {0}/{1}.sha256".format(FOLDER_POSIX, name),
        "    git commit -m \"Stage 2: approve pre-registration {} (local terminal)\"".format(prefix),
        "    git push origin {}".format(branch),
        "",
        "Once pushed, do not delete, edit or re-add this file: the cloud session checks that exactly one commit",
        "added it. Then tell Claude in the chat that you have pushed. Nothing runs until the cloud session has",
        "pulled your commit, checked it again with git, and pushed its import record",
        "({}/import_{}.json), which names your commit so that you can check it on GitHub.".format(FOLDER_POSIX, prefix),
        "If you change your mind before pushing, delete the two files instead.",
        "",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
