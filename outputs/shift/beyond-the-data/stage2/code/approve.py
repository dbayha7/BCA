"""Stage 2: the user's approval of the pre-registration runs/shift_explore/stage2/prereg.json.

It writes runs/shift_explore/stage2/approval.json (with a '<name>.sha256' sidecar) only in a mode the pre-registration
accepts (stage2_common.allowed_modes). Version 6 accepts ONE mode, the user's choice:

  local terminal. The user, on the user's own machine, pulls the branch and runs, from the repository root,

        python3 outputs/shift/beyond-the-data/stage2/approve_local.py

      which checks the committed prereg.json, its code manifest and every pinned file, refuses unless standard input
      is a terminal, prints the fingerprint and asks for its first 16 characters; it writes
      outputs/shift/beyond-the-data/stage2/approval_<first 16>.json (+ .sha256), one file per fingerprint, which the
      user commits and pushes. After pulling it, the orchestrating agent runs

        cd /home/user/bca && /tmp/claude-0/bcavenv/bin/python <stage2>/approve.py import

      Import reads the commit facts itself with read-only git queries (stage2_common.git); nobody passes them in:
      the one commit on origin's branch (refs/remotes/origin/<branch>, as fetched) that added the file, its author and
      committer, and the file as stored in it. It refuses when: no commit or more than one added the file; the commit
      is not on origin's branch, or that branch's tip no longer holds the file unchanged; the local file differs from
      its sidecar or from the file in that commit; it is not a 'local-terminal' approval of the CURRENT prereg.json (its
      SHA-256, and typed characters that are exactly its first 16; an approval of another version is refused); the
      author or the committer is this cloud session's own git identity (AGENT_GIT_EMAILS or its configured
      user.email); the prereg.json, code manifest, PREREGISTRATION.md, approve_local.py or any code copy stored in that
      commit, or in the committed folder, differs from the current ones; a runs-side approval exists; or any check
      at approval fails (below). It then writes runs/shift_explore/stage2/approval.json (mode 'local-terminal', the
      verbatim local file and the git facts) and the import record outputs/shift/beyond-the-data/stage2/
      import_<first 16>.json (+ .sha256), which the orchestrating agent commits and pushes. Every BCA entry point
      (stage2_common.verify_approval) re-checks all of it with git, and refuses until the import record is on origin's
      branch, so the user can see on GitHub which commit was imported.

  'terminal' (typed in this cloud session) and 'chat' (approve.py chat --message-file F, the exact sentence
      "approve <first 16 characters>, accept chat mode") remain in the code, as the gate's mechanisms, but version 6
      does not accept them: both refuse, and so does verify_approval for an approval.json that names them.

WHAT THE ROUTE CANNOT PROVE (pre-registration section 9). approve_local.py checks only that standard input is a
terminal. Git's author and committer fields are set by whoever makes the commit, and git does not check them, so
anyone able to push to the repository, an agent with push access included, could make such a commit under the user's
name. An agent could also write runs/shift_explore/stage2/approval.json by hand (runs/ is not committed). Import and
every check read git themselves and need the import record on GitHub, so such a forgery would be visible there, but
the code cannot prevent it: the route rests on the orchestrating agent following the procedure and on the user
checking, on GitHub, that the commit the import record names is the user's own.

Approving a fingerprint accepts every default the pre-registration lists (its open choices); any other choice needs a
new version of prereg.json and a new fingerprint.

Every mode refuses when: prereg.json is missing or differs from its sidecar; it is a draft (approvable: false, some
pin missing); runs/shift_explore/stage2/approval.json already exists; the code manifest or any pinned script differs
(the approved hash must cover the code that will run); a pinned component of another job (room test, fallback
diagnostic, stand-ins, detection script) differs NOW from the pin prereg.json records (prereg_payload.check_pins); the
pre-registration does not accept the mode.

  python approve.py show       prints the pre-registration's path, label, SHA-256, its first 16 characters, the
                               accepted modes, and whether it can be approved NOW: it runs the code and pin checks
                               (exit status 2 when it cannot)

Every Stage 2 BCA entry point re-verifies approval.json against the CURRENT prereg.json (stage2_common.require_approval)
and refuses otherwise, so editing prereg.json after approval voids it.
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stage2_common as C  # noqa: E402
import prereg_payload as PP  # noqa: E402

CHAT_WEAKER = ("Chat mode rests on trusting the orchestrating agent: the message reaches this program through a file "
               "written by that agent, so the program cannot tell a genuine user turn from a forged one. The user "
               "accepted this in the same message ('accept chat mode'). Terminal mode is no stronger in a session "
               "where the user has no terminal.")
TERMINAL_CAVEAT = ("Terminal mode checks only that standard input is a terminal and that the typed characters match. Any "
                   "agent with a shell can satisfy both through a pseudo-terminal, so this approval also rests on "
                   "trusting the orchestrating agent. The terminal's name and the parent processes are recorded as "
                   "evidence only.")
LOCAL_CAVEAT = ("Local-terminal mode: the approval was typed at a terminal on the user's own machine by approve_local.py, "
                "committed and pushed by the user, and imported here; import read the commit, its author and committer "
                "and the file stored in it with read-only git queries, and wrote an import record that must be "
                "committed and pushed before any BCA run. Nothing in this session typed or wrote the user's file. It "
                "still cannot prove who typed it: approve_local.py checks only that standard input is a terminal, and "
                "git's author and committer fields are set by whoever makes the commit and are not authenticated, so "
                "anyone able to push to the repository, an agent with push access included, could make such a commit "
                "under the user's name; an agent could also write this runs-side file by hand. The added strength "
                "holds only if the user checks on GitHub that the commit the import record names is the user's own.")
DEFAULTS_ACCEPTED = ("Approving this fingerprint accepts every default the pre-registration lists in open_choices; any "
                     "other choice needs a new version of prereg.json and a new fingerprint.")


class ApproveError(RuntimeError):
    pass


def _common_checks(paths, pins_fn=None):
    prereg = C.load_prereg(paths)
    if not prereg.payload.get("approvable", False):
        raise ApproveError(f"refused: prereg.json is a draft and cannot be approved (missing pins: "
                           f"{prereg.payload.get('missing_pins')})")
    if paths.approval.exists():
        raise ApproveError(f"refused: {paths.approval} exists already")
    try:
        C.check_code(prereg, paths)
    except C.ApprovalError as error:
        raise ApproveError(f"refused: the code does not match the pre-registration's pins ({error})") from None
    try:
        PP.check_pins(prereg, pins_fn)
    except SystemExit as error:
        raise ApproveError(f"refused: a pinned component differs now from prereg.json ({error})") from None
    return prereg


def _require_mode(prereg, mode):
    allowed = C.allowed_modes(prereg.payload)
    if mode not in allowed:
        raise ApproveError(f"refused: this pre-registration does not accept approval mode {mode!r} (it accepts: "
                           f"{', '.join(allowed) or 'no mode'}); another mode needs a new version of prereg.json")


def _process_chain(limit=12):
    """Evidence only: (pid, command line) of this process and its parents, read from /proc."""
    out, pid = [], os.getpid()
    for _ in range(limit):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf8", "replace").strip()
            stat = Path(f"/proc/{pid}/stat").read_text()
            ppid = int(stat[stat.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            break
        out.append(dict(pid=pid, cmd=cmd[:300]))
        if ppid <= 1:
            break
        pid = ppid
    return out


def _unlink_approval(paths):
    for f in (paths.approval, paths.approval.with_name(paths.approval.name + ".sha256")):
        if f.exists():
            f.unlink()


def _write(paths, body, pushed=True):
    """Write approval.json and check it at once (pushed=False only for import's own check: its import record is
    written next and pushed afterwards). Returns the file's SHA-256."""
    digest = C.write_json(paths.approval, body, paths)
    try:
        C.verify_approval(paths, pushed=pushed)       # the file just written must verify
    except C.ApprovalError:
        _unlink_approval(paths)
        raise
    return digest


def approve_terminal(paths=None, stdin=None, input_fn=input, log=print, pins_fn=None):
    paths = paths or C.PATHS
    stdin = sys.stdin if stdin is None else stdin
    prereg = _common_checks(paths, pins_fn)
    _require_mode(prereg, C.CLOUD_TERMINAL_MODE)
    if not (hasattr(stdin, "isatty") and stdin.isatty()):
        raise ApproveError("refused: terminal approval must be typed by the user at a terminal (standard input is not "
                           "a terminal)")
    log(f"Pre-registration: {prereg.path}\nLabel: {prereg.payload['label']}\nSHA-256: {prereg.sha256}")
    typed = input_fn("To approve Stage 2, type the first 16 characters of the pre-registration's SHA-256: ").strip()
    if typed != prereg.prefix:
        raise ApproveError("not approved: the typed characters do not match")
    try:
        tty = os.ttyname(stdin.fileno())
    except (OSError, AttributeError, ValueError):
        tty = None
    body = dict(kind="stage2-approval", label=C.LABEL, mode=C.CLOUD_TERMINAL_MODE, prereg_path=str(paths.prereg),
                prereg_sha256=prereg.sha256, typed=typed, stdin_isatty=True, tty=tty,
                process_chain=_process_chain(), approved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                terminal_mode_caveat=TERMINAL_CAVEAT, defaults_accepted=DEFAULTS_ACCEPTED,
                statement="Approved at a terminal after reading the pre-registration with this SHA-256.")
    _write(paths, body)
    log(f"approval written: {paths.approval}")
    return body


def approve_chat(message_file, paths=None, log=print, pins_fn=None):
    paths = paths or C.PATHS
    prereg = _common_checks(paths, pins_fn)
    _require_mode(prereg, C.CHAT_MODE)
    mf = Path(message_file)
    if not mf.exists():
        raise ApproveError(f"refused: no message file at {mf}")
    raw = mf.read_bytes()
    try:
        message = raw.decode("utf8")
    except UnicodeDecodeError:
        raise ApproveError("refused: the message file is not UTF-8 text") from None
    if not C.message_approves(message, prereg.prefix):
        raise ApproveError(f"not approved: the message is not exactly the approval sentence "
                           f"'{C.chat_sentence(prereg.prefix)}' (and nothing else)")
    body = dict(kind="stage2-approval", label=C.LABEL, mode=C.CHAT_MODE, prereg_path=str(paths.prereg),
                prereg_sha256=prereg.sha256, message_verbatim=message,
                message_sha256=hashlib.sha256(message.encode("utf8")).hexdigest(),
                message_file=str(mf.resolve()), message_file_sha256=hashlib.sha256(raw).hexdigest(),
                stdin_isatty=bool(hasattr(sys.stdin, "isatty") and sys.stdin.isatty()),
                approved_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), chat_mode_caveat=CHAT_WEAKER,
                defaults_accepted=DEFAULTS_ACCEPTED, required_sentence=C.chat_sentence(prereg.prefix),
                statement=("Approved in chat: the message above, written verbatim to the message file by the "
                           "orchestrating agent from the user's own turn, is exactly the approval sentence for this "
                           "pre-registration."))
    _write(paths, body)
    log(f"approval written (chat mode): {paths.approval}")
    return body


def _committed_names(prereg):
    """The pinned files of the committed folder, relative to it, with the SHA-256 each must have: prereg.json itself,
    the code manifest prereg.json pins, PREREGISTRATION.md and approve_local.py next to them, and code/<name> for every
    pinned file."""
    code = prereg.payload.get("code", {})
    files = code.get("files") or {}
    want = {"prereg.json": prereg.sha256, "code_manifest.json": code.get("code_manifest_sha256")}
    for name in ("PREREGISTRATION.md", "approve_local.py"):
        want[name] = files.get(name)
    for name, h in files.items():
        want["code/" + name] = h
    return want


def _check_committed_folder(prereg, paths):
    """The committed folder in this checkout (paths.repo) holds exactly the current pre-registration: prereg.json
    byte-identical to the current one (and its sidecar), the code manifest prereg.json pins (and its sidecar), and
    every pinned file's committed copy under code/, plus PREREGISTRATION.md and approve_local.py next to them, with the
    manifest's SHA-256."""
    repo = paths.repo
    bad = []
    for name in ("prereg.json", "code_manifest.json"):
        side = repo / (name + ".sha256")
        if not side.exists() or not (repo / name).exists() or \
                side.read_text(encoding="utf8").split()[:1] != [C.file_sha256(repo / name)]:
            bad.append(f"{name} (missing or differs from its sidecar)")
    for name, h in sorted(_committed_names(prereg).items()):
        f = repo / name
        if h is None or not f.exists() or C.file_sha256(f) != h:
            bad.append(name)
    if bad:
        raise ApproveError(f"refused: the committed folder {repo} does not hold the current pre-registration: "
                           f"{', '.join(sorted(set(bad)))}")


def _check_commit_tree(prereg, commit):
    """The same files as stored in the user's commit: what the user's approve_local.py checked is what runs here."""
    bad = []
    for name, h in sorted(_committed_names(prereg).items()):
        blob = C.git_blob(commit, C.REPO_STAGE2_REL + "/" + name)
        if h is None or blob is None or C.sha256_bytes(blob) != h:
            bad.append(name)
    if bad:
        raise ApproveError(f"refused: commit {commit[:16]}... does not hold the current pre-registration's files: "
                           f"{', '.join(bad)}")


def import_local(paths=None, log=print, pins_fn=None):
    """The second half of the local-terminal route. Verify the user's committed approval of the CURRENT fingerprint
    (paths.local_approval(prefix)) against prereg.json, its sidecar, the committed folder, the user's commit and every
    cloud-side pin; read the commit facts with read-only git queries (stage2_common.git); write the runs-side
    approval.json that every BCA entry point verifies, then the import record in the committed folder, which must be
    committed and pushed before any BCA run. Refuses on any mismatch, and when a runs-side approval exists."""
    paths = paths or C.PATHS
    prereg = _common_checks(paths, pins_fn)
    _require_mode(prereg, C.LOCAL_MODE)
    _check_committed_folder(prereg, paths)
    branch = ((prereg.payload.get("approval") or {}).get("local_terminal") or {}).get("branch")
    rel = C.local_approval_rel(prereg.prefix)
    local = paths.local_approval(prereg.prefix)
    if not local.exists():
        raise ApproveError(f"refused: no local approval of this fingerprint at {local} (pull the user's commit first)")
    raw = local.read_bytes()
    local_sha = C.sha256_bytes(raw)
    side = local.with_name(local.name + ".sha256")
    if not side.exists() or side.read_text(encoding="utf8").split()[:1] != [local_sha]:
        raise ApproveError(f"refused: {local} differs from its SHA-256 sidecar")
    try:
        ref = C.remote_ref(branch)
        tip = C.git_ref_commit(ref)
        if tip is None:
            raise C.LocalApprovalProblem(f"{ref} does not exist here (fetch the branch from GitHub first)")
        adders = C.git_adding_commits(ref, rel)
        if adders is None or len(adders) != 1:
            raise C.LocalApprovalProblem(f"exactly one commit on {ref} must add {rel}; git found "
                                         f"{'none (git failed)' if adders is None else len(adders)}")
        commit = adders[0]
        facts = C.git_commit_facts(commit)
        if facts is None:
            raise C.LocalApprovalProblem(f"commit {commit[:16]}... cannot be read")
        blob = C.git_blob(commit, rel)
        session_email = C.git_session_email()
        record = dict(commit=commit, author=facts["author"], committer=facts["committer"], parents=facts["parents"],
                      committed_sha256=None if blob is None else C.sha256_bytes(blob),
                      signature_status=C.git_signature_status(commit), remote_ref=ref, remote_tip_at_import=tip,
                      cloud_session_git_email=session_email,
                      read_by=("approve.py import, with read-only git queries in the cloud session's checkout: the one "
                               "commit on origin's branch that added the file, its author and committer, and the file "
                               "as stored in that commit"))
        body_local = C.check_local_approval(raw, prereg)
        C.check_commit_record(record, local_sha, C.agent_emails(session_email))
        C.check_commit_in_git(record, rel, branch)
    except C.LocalApprovalProblem as problem:
        raise ApproveError(f"refused: {problem}") from None
    _check_commit_tree(prereg, commit)
    imported_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    body = dict(kind="stage2-approval", label=C.LABEL, mode=C.LOCAL_MODE, prereg_path=str(paths.prereg),
                prereg_sha256=prereg.sha256, typed=body_local["typed"], stdin_isatty=True,
                local_approval=dict(path=rel, sha256=local_sha, text=raw.decode("utf8"),
                                    approved_at_utc=body_local.get("approved_at_utc"),
                                    hostname=body_local.get("hostname"), local_user=body_local.get("local_user"),
                                    git_head=body_local.get("git_head")),
                git=record, import_record=C.import_record_rel(prereg.prefix), imported_at=imported_at,
                importer_process_chain=_process_chain(), local_mode_caveat=LOCAL_CAVEAT,
                defaults_accepted=DEFAULTS_ACCEPTED,
                statement=("Approved by the user at a terminal on the user's own machine (approve_local.py), committed "
                           "and pushed by the user, and imported here after every check: this approval covers the "
                           "pre-registration with SHA-256 " + prereg.sha256 + " only."))
    digest = _write(paths, body, pushed=False)
    note = ("Written by 'approve.py import' in the cloud session and committed and pushed by the orchestrating agent. "
            "It names the commit that added your approval file, as git holds it. Check on GitHub that this commit is "
            "yours: if it is not, say so, and no BCA run may go on. Every BCA entry point refuses until this record "
            "is on the branch on GitHub.")
    rec = dict(kind=C.IMPORT_KIND, label=C.LABEL, prereg_sha256=prereg.sha256, prereg_version=prereg.payload.get("version"),
               approval_file=rel, approval_file_sha256=local_sha, commit=commit, author=facts["author"],
               committer=facts["committer"], parents=facts["parents"], signature_status=record["signature_status"],
               branch=branch, remote_tip_at_import=tip, runs_side_approval="runs/shift_explore/stage2/approval.json",
               runs_side_approval_sha256=digest, imported_at=imported_at, note=note)
    try:
        C.write_import_record(paths, prereg.prefix, rec)
    except BaseException:
        _unlink_approval(paths)
        raise
    rec_rel = C.import_record_rel(prereg.prefix)
    log(f"approval imported (local-terminal mode): {paths.approval}\n"
        f"  from commit {commit} by {facts['author']} (committer {facts['committer']})\n"
        f"  import record: {paths.import_record(prereg.prefix)}\n"
        f"Every BCA entry point refuses until the import record is on origin's branch. Commit and push it:\n"
        f"  git add {rec_rel} {rec_rel}.sha256\n"
        f"  git commit -m \"Stage 2: import record of approval {prereg.prefix}\"\n"
        f"  git push origin {branch}\n"
        f"Then tell the user the commit and author above and where the record is, so they can check them on GitHub.")
    return body


def approvable_now(prereg, paths, pins_fn=None):
    """(True, why) when prereg.json can be approved NOW: it is final, the code matches its manifest and every other
    job's pin matches what it records; (False, the reason) otherwise."""
    if not prereg.payload.get("approvable", False):
        return False, f"it is a draft (missing pins: {prereg.payload.get('missing_pins')})"
    try:
        C.check_code(prereg, paths)
    except C.ApprovalError as error:
        return False, f"the code does not match the pinned manifest ({str(error)[len(C.REFUSAL):].strip(' ()')})"
    try:
        PP.check_pins(prereg, pins_fn)
    except SystemExit as error:
        return False, str(error)
    return True, "final; the code matches its manifest; every other job's pin matches what prereg.json records"


def show(paths=None, log=print, pins_fn=None):
    """Print the pre-registration's fingerprint and whether it can be approved NOW. Returns (prereg, approvable now)."""
    paths = paths or C.PATHS
    prereg = C.load_prereg(paths)
    ok, why = approvable_now(prereg, paths, pins_fn)
    log(f"Pre-registration: {prereg.path}\nLabel: {prereg.payload['label']}\nVersion: {prereg.payload.get('version')}\n"
        f"SHA-256: {prereg.sha256}\nFirst 16 characters: {prereg.prefix}\n"
        f"Accepted approval modes: {', '.join(C.allowed_modes(prereg.payload)) or 'none'}\n"
        f"Route: the user runs python3 {C.REPO_STAGE2_REL}/approve_local.py on the user's own machine, commits and "
        f"pushes {C.local_approval_rel(prereg.prefix)} (+ .sha256); then approve.py import; then commit and push "
        f"{C.import_record_rel(prereg.prefix)} (+ .sha256)\n"
        f"Approvable now: {'yes' if ok else 'NO'} ({why})\n"
        f"Runs-side approval present: {paths.approval.exists()}\n"
        f"User's approval of this fingerprint present: {paths.local_approval(prereg.prefix).exists()}\n"
        f"Import record present: {paths.import_record(prereg.prefix).exists()}")
    return prereg, ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("terminal", help="refused unless the pre-registration accepts it (version 6 does not)")
    c = sub.add_parser("chat", help="refused unless the pre-registration accepts it (version 6 does not)")
    c.add_argument("--message-file", required=True)
    sub.add_parser("show", help="print the fingerprint and whether it can be approved now (exit status 2 if not)")
    sub.add_parser("import", help="import the user's committed local-terminal approval (reads git itself)")
    a = ap.parse_args(argv)
    try:
        if a.mode == "terminal":
            approve_terminal()
        elif a.mode == "chat":
            approve_chat(a.message_file)
        elif a.mode == "import":
            import_local()
        else:
            return 0 if show()[1] else 2
    except (ApproveError, C.ApprovalError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
