#!/usr/bin/env python3
"""Near-80 (version 9, main setting): approve the final pre-registration at a terminal on YOUR OWN machine
(PREREGISTRATION.md section 11). DRAFT, not frozen: this script refuses until freeze.py has written prereg.json.

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
       - prereg.json matches its fingerprint file prereg.json.sha256, is the near-80 pre-registration of version 9,
         is FINAL ("final, awaiting approval"; a draft cannot be approved), misses nothing, and offers approval from
         your own terminal with your GitHub address;
       - code_manifest.json is the manifest prereg.json pins and matches its own fingerprint file; every file it lists
         (this script included) has the fingerprint it records; every repository file it pins is unchanged;
       - the frozen code's pre-approval facts (pre_approval/pooled_facts_seeds0_4_spent10_14.json: the pooled
         counts on seeds 0-4 and the spent-seed switch facts on 10-14; nothing trained) are the ones prereg.json pins;
       - the draft the final version was built from (prereg_history/prereg_draft_<first 16>.json) is the one pinned;
       - the folder and its code/ hold no other file: only the files the manifest lists, prereg.json,
         code_manifest.json (each with its .sha256), this fingerprint's own approval and import records, version
         8's history (its approval and import records, its amendment_1.json, each with its .sha256; its
         pre_approval/ file; its results folder; its superseded prereg.json in prereg_history/), the build notes and
         the decision/ folder (history; the only folder besides code/ allowed to hold Python files: the two
         calculation scripts it keeps); no __pycache__ and no other folder holding Python files.
  2. Refuses if an approval of this fingerprint is already there (it never overwrites one).
  3. Refuses unless standard input is a terminal: you type the answer yourself; nothing may be piped in.
  4. Shows a summary: the fingerprint, your answer accepting the defaults, the defaults taken, the seeds, the two
     claims, the rules, the frozen counts, the compute forecast and cap, the hosts.
  5. Asks you to type the fingerprint's first 16 characters.
  6. Writes approval_near80_<first 16 characters>.json and its .sha256 file in that folder, once, reads them back,
     and prints the git commands to commit and push them under your own name. Nothing runs until the session has
     imported your approval and pushed its import record (import_near80_<first 16 characters>.json).

An amendment (amendment_v9_1.json beside prereg.json) puts the script in amendment mode, as version 8's amendment 1
did: the files that differ from the approved code manifest must be exactly the ones it names, documents are only
appended to, no file is added or removed, code/near80_constants.py never changes, your approval of prereg.json and
its import record must be in place, and the desktop check (near80_amendment1/check_amendment1.sh) must have passed.

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
VERSION = "version 9, near-80 main"
STATUS_FINAL = "final, awaiting approval"
APPROVER_EMAIL = "35092758+dbayha7@users.noreply.github.com"
APPROVER_NAME = "David Bayha"
BRANCH = "claude/main-bca-connection-x4o46s"
SCRIPT_NAME = "approve_local_near80.py"
FROZEN_NAME = "pre_approval/pooled_facts_seeds0_4_spent10_14.json"
FROZEN_TOP = (PREREG_NAME, PREREG_NAME + ".sha256", MANIFEST_NAME, MANIFEST_NAME + ".sha256")
STATEMENT = ("I approve the near-80 pre-registration ({version}) whose SHA-256 is {sha}, and only that exact file. "
             "Approving it accepts every choice of its section 16 as it records them (the defaults; the university "
             "cluster as the run machine, the desktop as the fallback; the results copied from the cluster to the "
             "desktop and committed from there) and allows the runs it names, under its rules.")
WIDTH = 100
# Version 8's history (near80_constants.V8_HISTORY_FILES, V8_HISTORY_DIRS, V8_FROZEN_NAME; test_v9 checks they agree).
V8_HISTORY_FILES = ("approval_near80_51e677ab76c8d2f7.json", "import_near80_51e677ab76c8d2f7.json", "amendment_1.json",
                    "approval_near80_amendment1_52fe05b92900290b.json",
                    "import_near80_amendment1_52fe05b92900290b.json")
V8_HISTORY_DIRS = ("results-20261020T115246Z",)
V8_FROZEN = "pre_approval/no_training_checks_seeds0_4.json"
V9_NOTES = "V9_BUILD_NOTES.md"
V9_DECISION_DIR = "decision"          # the session's decision files, kept as history (near80_constants.V9_DECISION_DIR)
# Amendments of version 9 (approval_check.py's names, restated; test_v9 checks they agree).
AMENDMENT_NAME = "amendment_v9_1.json"
AMENDMENT_KIND = "near80-amendment"
AMEND_APPROVAL_NAME = "approval_near80_v9_amendment1_{}.json"
AMEND_IMPORT_NAME = "import_near80_v9_amendment1_{}.json"
AMEND_KIND = "near80-local-amendment-approval"
AMEND_STATEMENT = ("I approve amendment 1 (SHA-256 {sha}) to the near-80 pre-registration {parent}. It changes only "
                   "the files it lists; every rule, mark, threshold and sentence of that pre-registration is "
                   "unchanged.")
NEVER_AMENDED = ("code/near80_constants.py",)
CHECK_POSIX = "outputs/shift/beyond-the-data/near80_amendment1"     # the desktop check and its result files
CHECK_KIND = "near80-amendment1-desktop-check"
CHECK_TESTS = ("test_near80.py", "test_approval.py", "test_runner.py", "test_pooled.py", "test_v9.py")


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
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise Refused("{} is not readable JSON.".format(name))


def unlisted_files(folder, files, prefix, extra=()):
    """Step 1's last check: every entry of the folder and of its code/ that is neither a file the manifest lists, one
    of the four frozen files at the top, this fingerprint's own approval or import record (with .sha256), version 8's
    history, nor the build notes; any __pycache__; any other folder holding a .py file (version 8's results folder
    holds none; the decision/ folder, history like the build notes, holds the two calculation scripts and is allowed
    by name: nothing puts it on Python's path). Returns the problems (empty when none)."""
    allowed = set(f for f in files if "/" not in f) | set(FROZEN_TOP) | set(extra) | {V9_NOTES}
    for name in (APPROVAL_NAME.format(prefix), IMPORT_NAME.format(prefix)):
        allowed.update((name, name + ".sha256"))
    for name in V8_HISTORY_FILES:
        allowed.update((name, name + ".sha256"))
    out = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if name == "__pycache__":
            out.append("{} exists (compiled Python bytecode); remove it".format(name))
        elif os.path.isdir(path) and not os.path.islink(path):
            if name in ("code", V9_DECISION_DIR):
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


def delta_problems(parent_files, amendment, folder):
    """approval_check.delta_problems, restated (standard library only)."""
    files = amendment.get("files") or {}
    delta = amendment.get("delta") or {}
    changed, added = delta.get("changed") or {}, delta.get("added") or {}
    removed, appended = list(delta.get("removed") or []), delta.get("appended") or {}
    out = []
    if added or removed:
        out.append("an amendment adds and removes no file (near80_constants.CODE_FILES cannot change)")
    if set(files) != (set(parent_files) - set(removed)) | set(added):
        out.append("its file list is not the parent's with the delta's additions and removals")
    out += ["{} is removed but the parent does not list it".format(r) for r in removed if r not in parent_files]
    out += ["{} is added but the parent lists it, or its SHA-256 differs".format(r) for r, h in added.items()
            if r in parent_files or files.get(r) != h]
    differ = sorted(r for r in set(parent_files) & set(files) if parent_files[r] != files[r])
    if differ != sorted(changed):
        out.append("the files that differ from the parent {} are not the ones its delta names {}".format(
            differ, sorted(changed)))
    for r, pair in sorted(changed.items()):
        if not (isinstance(pair, list) and len(pair) == 2 and pair[0] == parent_files.get(r)
                and pair[1] == files.get(r)):
            out.append("{}: the delta's old and new SHA-256 are not the parent's and the amendment's".format(r))
    out += ["{} may not change in an amendment".format(r) for r in NEVER_AMENDED if r in changed or r in removed]
    for r in sorted(set(changed) | set(removed)):
        if not r.endswith(".md"):
            continue
        a = appended.get(r) or {}
        path = os.path.join(folder, *r.split("/"))
        data = read_bytes(path) if os.path.isfile(path) else b""
        n = a.get("parent_bytes")
        if not (r in changed and type(n) is int and 0 < n < len(data)
                and a.get("parent_sha256") == parent_files.get(r) and sha256_bytes(data[:n]) == parent_files.get(r)):
            out.append("{} may only be appended to (the parent's text byte for byte, then the amendment's)".format(r))
    return out


def _str_map(m, value_ok):
    return isinstance(m, dict) and all(isinstance(k, str) and value_ok(v) for k, v in m.items())


def amendment_shape_ok(body):
    if not (all(body.get(k) is None or (isinstance(body[k], list) and all(isinstance(x, str) for x in body[k]))
                for k in ("cleared_failures", "unchanged", "runs"))
            and all(body.get(k) is None or isinstance(body[k], str) for k in ("title", "why_no_result_changes"))
            and (body.get("desktop_check") is None
                 or _str_map(body["desktop_check"], lambda v: isinstance(v, str)))):
        return False
    delta = body.get("delta")
    if not (isinstance(body.get("parent"), dict) and _str_map(body.get("files"), lambda v: isinstance(v, str))
            and isinstance(delta, dict)):
        return False
    removed = delta.get("removed") or []
    return (_str_map(delta.get("changed") or {}, lambda v: isinstance(v, list) and len(v) == 2
                     and all(isinstance(x, str) for x in v))
            and _str_map(delta.get("added") or {}, lambda v: isinstance(v, str))
            and isinstance(removed, list) and all(isinstance(r, str) for r in removed)
            and _str_map(delta.get("appended") or {}, lambda v: isinstance(v, dict)))


def desktop_checks(folder, amendment, amendment_sha):
    want = amendment.get("desktop_check")
    d = os.path.join(os.path.dirname(os.path.abspath(folder)), CHECK_POSIX.rsplit("/", 1)[-1])
    good = {}
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        path = os.path.join(d, name)
        if not (name.startswith("desktop_check_") and name.endswith(".json") and os.path.isfile(path)):
            continue
        try:
            raw = read_bytes(path)
            body = json.loads(raw.decode("utf8"))
            ok = sidecar_digest(path) == sha256_bytes(raw)
        except (OSError, Refused, UnicodeDecodeError, ValueError, RecursionError):
            continue
        tests = body.get("unit_tests") if isinstance(body, dict) else None
        if (ok and isinstance(body, dict) and body.get("kind") == CHECK_KIND and body.get("mode") == "official"
                and body.get("passed") is True and body.get("amendment_sha256") == amendment_sha
                and isinstance(want, dict) and bool(want) and body.get("check_sha256") == want
                and isinstance(tests, dict) and sorted(tests) == sorted(CHECK_TESTS)
                and all(type(v) is int and v == 0 for v in tests.values())):
            good[name] = sha256_bytes(raw)
    return good


def load_amendment(folder, prereg_sha, manifest_sha, label):
    path = os.path.join(folder, AMENDMENT_NAME)
    if not os.path.isfile(path):
        raise Refused("{} is missing.".format(AMENDMENT_NAME))
    raw = read_bytes(path)
    digest = sha256_bytes(raw)
    if sidecar_digest(path) != digest:
        raise Refused("{} does not match its fingerprint file (it was changed).".format(AMENDMENT_NAME))
    body = load_json(raw, AMENDMENT_NAME)
    if not isinstance(body, dict) or body.get("kind") != AMENDMENT_KIND or body.get("number") != 1:
        raise Refused("{} is not near-80's amendment 1 of version 9.".format(AMENDMENT_NAME))
    if body.get("label") != label:
        raise Refused("{} names another study.".format(AMENDMENT_NAME))
    if not amendment_shape_ok(body):
        raise Refused("{} is malformed: its parent, files or delta are not of the expected form.".format(
            AMENDMENT_NAME))
    parent = body["parent"]
    if parent.get("prereg_sha256") != prereg_sha or parent.get("code_manifest_sha256") != manifest_sha:
        raise Refused("{} amends another pre-registration or code manifest.".format(AMENDMENT_NAME))
    if not isinstance(body.get("files"), dict) or not body["files"]:
        raise Refused("{} lists no files.".format(AMENDMENT_NAME))
    return body, digest


def check_folder(folder=None, script_path=None, repo=REPO, amend=False):
    """Step 1. Returns what the rest needs; raises Refused on any mismatch."""
    folder = folder or os.path.join(repo, *FOLDER_POSIX.split("/"))
    prereg_path = os.path.join(folder, PREREG_NAME)
    if not os.path.isfile(prereg_path):
        raise Refused("No near-80 pre-registration at {}. Pull the branch first (or the freeze has not run: this "
                      "version is a draft).".format(prereg_path))
    raw = read_bytes(prereg_path)
    digest = sha256_bytes(raw)
    if sidecar_digest(prereg_path) != digest:
        raise Refused("prereg.json does not match its fingerprint file (it was changed).")
    prereg = load_json(raw, PREREG_NAME)
    if not isinstance(prereg, dict) or prereg.get("kind") != PREREG_KIND:
        raise Refused("prereg.json is not the near-80 pre-registration.")
    if prereg.get("version") != VERSION:
        raise Refused("prereg.json is {!r}, not {!r}: this script approves version 9 only.".format(
            prereg.get("version"), VERSION))
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
    amendment, amendment_sha, extra = None, None, ()
    if amend:
        amendment, amendment_sha = load_amendment(folder, digest, man_sha, prereg.get("label"))
        bad.extend(delta_problems(files, amendment, folder))
        parent = amendment.get("parent") or {}
        p16, a16 = digest[:PREFIX_LEN], amendment_sha[:PREFIX_LEN]
        for name, want in ((APPROVAL_NAME.format(p16), parent.get("approval_sha256")),
                           (IMPORT_NAME.format(p16), parent.get("import_sha256"))):
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                bad.append("{} is missing: an amendment is approved only on top of your approval".format(name))
            elif file_sha256(path) != want or sidecar_digest(path) != want:
                bad.append("{} is not the one {} names, or differs from its fingerprint file".format(
                    name, AMENDMENT_NAME))
        files = amendment["files"]
        extra = [n + s for n in (AMENDMENT_NAME, APPROVAL_NAME.format(p16), IMPORT_NAME.format(p16),
                                 AMEND_APPROVAL_NAME.format(a16), AMEND_IMPORT_NAME.format(a16))
                 for s in ("", ".sha256")]
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
    frozen = prereg.get("pooled_group_counts_frozen") or {}
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
    draft = prereg.get("draft") or {}
    dsha = draft.get("sha256")
    drel = str(draft.get("path") or "")
    if not dsha or drel != "prereg_history/prereg_draft_{}.json".format(str(dsha)[:PREFIX_LEN]):
        bad.append("prereg.json does not pin the draft it was built from")
    else:
        dpath = os.path.join(folder, *drel.split("/"))
        if not os.path.isfile(dpath) or file_sha256(dpath) != dsha:
            bad.append("{} is missing or differs".format(drel))
    bad.extend(unlisted_files(folder, files, digest[:PREFIX_LEN], extra))
    if bad:
        hint = ("\n  (runs/ is not in git: run this on the machine where version 7's restore placed runs/, or restore "
                "it first.)" if missing_runs else "")
        raise Refused("These files do not match the near-80 pre-registration:\n  - " + "\n  - ".join(bad) + hint)
    return dict(prereg=prereg, sha256=digest, manifest_sha256=man_sha, n_files=len(files),
                n_repo_files=len(repo_files), script_sha256=files[SCRIPT_NAME], section=section, folder=folder,
                amendment=amendment, amendment_sha256=amendment_sha)


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
    frozen = p.get("pooled_group_counts_frozen") or {}
    compute = p.get("compute") or {}
    hosts = p.get("hosts") or {}
    choices = ["{}. {}: {}".format(c.get("number"), c.get("choice"), json.dumps(c.get("taken")))
               for c in p.get("choices") or []]
    parts = ["", "Near-80 (version 9, main setting): the main setting on fresh seeds", wrap(p.get("label", "")),
             "  File: {}/{}".format(FOLDER_POSIX, PREREG_NAME),
             "  Checked: it matches its fingerprint file; the code manifest, all {} pinned near-80 files and all {} "
             "pinned repository files match.".format(info["n_files"], info["n_repo_files"]), "",
             "Fingerprint (SHA-256) of prereg.json:", "", "    " + info["sha256"], "",
             "Your answer: \"{}\" ({}, {}): {}.".format(answer.get("text"), answer.get("utc"), answer.get("where"),
                                                       answer.get("choices")), "",
             "Your second answer, on choice {} (where results are committed from): \"{}\" ({}, {}): {}.".format(
                 ", ".join((answer.get("second") or {}).get("choices") or []), (answer.get("second") or {}).get("text"),
                 (answer.get("second") or {}).get("utc"), (answer.get("second") or {}).get("where"),
                 (answer.get("second") or {}).get("about")), "",
             "The defaults of section 16 that approving takes:", bullets(choices), "",
             "Seeds: {} fresh (the only ones that can carry a claim; {} seeds); seed {} for the plumbing only; every "
             "seed another study used is refused.".format(seeds.get("fresh"), seeds.get("fresh_count"),
                                                            seeds.get("plumbing")), "",
             "The three claims (read only on the fresh seeds):", bullets(s.get("claims")), "",
             "The rules:", bullets(s.get("rules")), "",
             "The frozen code's pre-approval facts (nothing trained): the pooled groups' counts on seeds 0-4: {}; "
             "the spent seeds 10-14 from version 8's recorded second critics: {}.".format(
                 json.dumps(frozen.get("summary")), json.dumps(frozen.get("spent_summary"))), "",
             "Compute: {} CPU-hours per fresh seed; {} for the plumbing; the cap is {}.".format(
                 compute.get("per_fresh_seed"), compute.get("plumbing"), compute.get("budget_cap")), "",
             "Hosts: {}.".format("; ".join("{}: {}".format(k, v.get("entry")) for k, v in sorted(hosts.items()))), "",
             "Read it first: {}/PREREGISTRATION.md".format(FOLDER_POSIX), ""]
    return "\n".join(parts)


def summary_amendment(info):
    a = info["amendment"]
    delta = a.get("delta") or {}
    check = a.get("desktop_check") or {}
    changed = ["{}: {} -> {}{}".format(rel, old[:16], new[:16], " (appended to; the first {} bytes are the approved "
                                                                 "text)".format((delta.get("appended") or {}).get(
                                                                     rel, {}).get("parent_bytes"))
               if rel in (delta.get("appended") or {}) else "")
               for rel, (old, new) in sorted((delta.get("changed") or {}).items())]
    parts = ["", "Near-80 (version 9): amendment 1", wrap(a.get("title", "")),
             "  Amends: prereg.json {} (your approval of {} and its import record are in place).".format(
                 info["sha256"], info["sha256"][:PREFIX_LEN]),
             "  Checked: prereg.json, code_manifest.json and all {} pinned repository files are unchanged; every other "
             "file is the parent's except the ones below.".format(info["n_repo_files"]), "",
             "What changes (SHA-256, first 16 characters, before -> after):", bullets(changed), "",
             "What does not change:", bullets(a.get("unchanged")), "",
             "Kept, and what runs after your approval is imported:", bullets(a.get("runs")), "",
             "Why it cannot change a result:", wrap(a.get("why_no_result_changes", "")), "",
             "The desktop check of this amendment ({}/check_amendment1.sh):".format(CHECK_POSIX),
             bullets(["its code, as the amendment names it: check_amendment1.py {}, check_amendment1.sh {}".format(
                 str(check.get("py"))[:PREFIX_LEN], str(check.get("sh"))[:PREFIX_LEN])]
                 + (["passed: {}".format(n) for n in sorted(info.get("desktop_checks") or {})]
                    or ["none has passed yet; run it before you approve"])), "",
             "Fingerprint (SHA-256) of {}:".format(AMENDMENT_NAME), "", "    " + info["amendment_sha256"], "",
             "Read it first: {}/PREREGISTRATION.md, its amendment section".format(FOLDER_POSIX), ""]
    return "\n".join(parts)


def amendment_body(info, typed, stdin=None, repo=REPO):
    body = approval_body(info, typed, stdin, repo)
    for k in ("prereg_path", "prereg_sha256", "code_manifest_sha256"):
        body.pop(k)
    body.update(kind=AMEND_KIND, amendment_path=FOLDER_POSIX + "/" + AMENDMENT_NAME, amendment_number=1,
                amendment_sha256=info["amendment_sha256"], parent_prereg_sha256=info["sha256"],
                desktop_checks={CHECK_POSIX + "/" + n: h
                                for n, h in sorted((info.get("desktop_checks") or {}).items())},
                statement=AMEND_STATEMENT.format(sha=info["amendment_sha256"], parent=info["sha256"]),
                how=("Typed at a terminal on the owner's own machine with approve_local_near80.py (amendment mode). It "
                     "takes effect only after the owner commits and pushes it and the session imports it "
                     "(code/import_approval.py import-amendment) and pushes the import record."))
    return body


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
        base = folder or os.path.join(repo, *FOLDER_POSIX.split("/"))
        amend = os.path.lexists(os.path.join(base, AMENDMENT_NAME))
        info = check_folder(folder, script_path, repo, amend=amend)
        prefix = (info["amendment_sha256"] if amend else info["sha256"])[:PREFIX_LEN]
        name = (AMEND_APPROVAL_NAME if amend else APPROVAL_NAME).format(prefix)
        path = os.path.join(info["folder"], name)
        show = summary_amendment if amend else summary
        if amend:
            info["desktop_checks"] = desktop_checks(info["folder"], info["amendment"], info["amendment_sha256"])
        if args.check:
            print(show(info))
            print("--check: every check passed. Nothing was asked and nothing was written.")
            return 0
        if os.path.lexists(path) or os.path.lexists(path + ".sha256"):
            raise Refused("{} already exists: this fingerprint is approved already. An approval is never "
                          "overwritten.".format(name))
        if amend and not info["desktop_checks"]:
            raise Refused("No desktop check of this amendment has passed. Run {}/check_amendment1.sh first: its "
                          "result file must match its .sha256 file, say it ran in official mode and passed, and name "
                          "{}'s fingerprint {}.".format(CHECK_POSIX, AMENDMENT_NAME,
                                                        info["amendment_sha256"][:PREFIX_LEN]))
        if not (hasattr(stdin, "isatty") and stdin.isatty()):
            raise Refused("Standard input is not a terminal. Run this yourself in a terminal window; do not pipe or "
                          "redirect anything into it.")
        print(show(info))
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
        body = (amendment_body if amend else approval_body)(info, typed, stdin, repo)
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
        ("It covers amendment 1 with fingerprint {} only, on top of the pre-registration {}.".format(
            info["amendment_sha256"], info["sha256"]) if amend else
         "It covers the near-80 pre-registration (version 9) with fingerprint {} only.".format(info["sha256"])), "",
        "Now commit and push it under your own name, from {}, with these exact commands:".format(repo), "",
        ("    git add {0} {0}.sha256 {1}/desktop_check_*".format(rel, CHECK_POSIX) if amend else
         "    git add {0} {0}.sha256".format(rel)),
        "    git -c user.name=\"{}\" -c user.email={} commit -m \"Near-80: approve {} {}\"".format(
            APPROVER_NAME, APPROVER_EMAIL, "version 9 amendment 1" if amend else "version 9", prefix),
        "    git push origin {}".format(branch), "",
        "If the push is rejected (\"fetch first\" or \"non-fast-forward\": the session pushed something meanwhile), type:",
        "", "    git -c user.name=\"{}\" -c user.email={} pull --rebase origin {}".format(
            APPROVER_NAME, APPROVER_EMAIL, branch),
        "    git push origin {}".format(branch), "",
        "Once pushed, do not delete, edit or re-add these files. Then tell Claude in the chat that you have pushed;",
        "nothing runs until the session has imported it and pushed {}.".format(
            (AMEND_IMPORT_NAME if amend else IMPORT_NAME).format(prefix)),
        "When the session says it has pushed the import record, bring it here with:", "",
        "    git -C {} pull --ff-only".format(repo), "",
        "If you change your mind before pushing, delete the two files instead.", ""]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
