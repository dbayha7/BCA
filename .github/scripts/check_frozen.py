#!/usr/bin/env python3
"""Fail when a change edits, moves or deletes a frozen file.

Frozen files are hashed into pre-registrations, approvals and pinned receipts, so changing their bytes
silently breaks a record that was meant to be fixed in advance. The frozen paths are listed in
.github/frozen-paths.txt. Adding new files under a frozen folder is allowed.

The list is read from the base revision as well as from the change itself, so one change cannot
remove an entry and edit the file it protects: unfreezing is its own reviewed change, merged first.

Usage:
  python3 .github/scripts/check_frozen.py                    # working tree vs. merge base with origin/main
  python3 .github/scripts/check_frozen.py --base REV         # working tree vs. REV
  python3 .github/scripts/check_frozen.py --base A --head B  # commit B vs. commit A (CI)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

LIST = ".github/frozen-paths.txt"
ROOT = Path(__file__).resolve().parents[2]


def git(*args, cwd=ROOT):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def entries(text):
    """Frozen paths in a list file: a folder ends with '/', anything else is one file."""
    return {line.strip() for line in (text or "").splitlines() if line.strip() and not line.lstrip().startswith("#")}


def read_list(rev, cwd):
    """The list at a revision, or in the working tree when rev is None; empty if absent."""
    if rev is None:
        file = Path(cwd) / LIST
        return file.read_text(encoding="utf8") if file.is_file() else ""
    result = subprocess.run(["git", "cat-file", "blob", f"{rev}:{LIST}"], cwd=cwd, capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ""


def is_frozen(path, frozen):
    return any(path.startswith(entry) if entry.endswith("/") else path == entry for entry in frozen)


def violations(base, head=None, cwd=ROOT):
    """(status, path) for every frozen file that the change modifies, deletes or retypes."""
    frozen = entries(read_list(base, cwd)) | entries(read_list(head, cwd))
    # --no-renames reports a move as delete + add, and deleting a frozen file is what we look for.
    out = git("diff", "--name-status", "--no-renames", "-z", base, *([head] if head else []), cwd=cwd)
    fields = out.split("\0")[:-1]
    changes = zip(fields[0::2], fields[1::2])
    return [(status, path) for status, path in changes if status != "A" and is_frozen(path, frozen)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="revision to compare against (default: merge base of HEAD and origin/main)")
    parser.add_argument("--head", help="revision to check (default: the working tree)")
    opt = parser.parse_args(argv)
    try:
        base = opt.base or git("merge-base", "HEAD", "origin/main").strip()
        found = violations(base, opt.head)
    except RuntimeError as error:
        print(f"ERROR: {error}\nPass --base with the revision this change starts from.", file=sys.stderr)
        return 2
    if not found:
        print(f"OK: no frozen file is edited, moved or deleted (compared with {base[:12]}).")
        return 0
    print(f"FAIL: this change edits, moves or deletes frozen files listed in {LIST}:", file=sys.stderr)
    for status, path in found:
        print(f"  {status}  {path}", file=sys.stderr)
        if os.environ.get("GITHUB_ACTIONS"):
            print(f"::error file={path},title=Frozen file changed::{path} is frozen ({LIST}).")
    print(
        "These files are hashed into pre-registrations, approvals or pinned receipts, so changing their "
        f"bytes breaks the record. Restore them (git checkout {base[:12]} -- <path>) and put new work in "
        f"a new file. Only the repository owner unfreezes a path, by removing its entry from {LIST} "
        "in a change of its own that is merged first.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
