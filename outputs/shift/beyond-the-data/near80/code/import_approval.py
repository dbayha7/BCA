"""Near-80 (version 8): show the final pre-registration's fingerprint, and import the owner's approval.

  import_approval.py show     prints prereg.json's path, its fingerprint and first 16 characters, whether it can be
                              approved now (final, the near-80 code matches its manifest, the pinned repository files
                              hold; exit status 2 if not), and whether the owner's approval and the import record are
                              present.
  import_approval.py import   the second half of the route, in the session, after the owner ran
                              approve_local_near80.py at their own terminal, committed approval_near80_<first 16>.json
                              (+ .sha256) under their own name and pushed it. With version 6's read-only git queries
                              (called from version 6's own stage2_common.py, never copied) it checks that exactly one
                              commit on origin's branch added the file; that its author and its committer are NEITHER
                              the session's own git identity (noreply@anthropic.com) NOR the session's configured
                              user.email, AND are both the owner (35092758+dbayha7@users.noreply.github.com); that the
                              file in that commit is the file here and approves THIS prereg.json; and that the commit's
                              tree holds prereg.json, code_manifest.json and every manifest file with its pinned
                              fingerprint. Then it writes the import record import_near80_<first 16>.json (+ .sha256)
                              once and prints the commands that commit and push it. Every near-80 phase refuses until
                              that record is on origin's branch.
This is not authentication: git does not authenticate authors. The owner checks on GitHub that the commit the import
record names is theirs. No approval file is ever written here.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import approval_check as AC  # noqa: E402
import near80_constants as K  # noqa: E402

NOTE = ("Written by 'import_approval.py import' in the session and committed and pushed by the session. It names the "
        "commit that added your near-80 approval file, as git holds it. Check on GitHub that this commit is yours: if "
        "it is not, say so, and no near-80 phase may run. Every near-80 phase refuses until this record is on the "
        "branch on GitHub.")


def approvable_now(cfg=None):
    """(prereg, manifest, ok, why): final; the code matches its manifest; the constants and the pins hold."""
    cfg = cfg or AC.ApprovalConfig()
    prereg = AC.load_prereg(cfg)
    try:
        AC.check_constants(prereg)
        manifest = AC.check_code(prereg, cfg)
        if cfg.check_pins:
            AC.check_pins(manifest, cfg)
            AC.check_prereg_pins(prereg, manifest, cfg)
    except AC.ApprovalRefused as error:
        return prereg, None, False, str(error)
    section = prereg.payload.get("approval") or {}
    if (section.get("local_terminal_allowed") is not True or section.get("approver_email") != K.APPROVER_EMAIL
            or section.get("branch") != cfg.branch):
        return prereg, manifest, False, "prereg.json does not accept this route (approval section)"
    return prereg, manifest, True, "final; the near-80 code matches its manifest; the pinned files hold"


def show(cfg=None, log=print):
    cfg = cfg or AC.ApprovalConfig()
    prereg, _, ok, why = approvable_now(cfg)
    root = Path(cfg.n80_root)
    log(f"Near-80 pre-registration: {prereg.path}\nVersion: {prereg.payload.get('version')}\n"
        f"SHA-256: {prereg.sha256}\nFirst 16 characters: {prereg.prefix}\n"
        f"Route: the owner runs python3 {K.N80_REL}/approve_local_near80.py at their own terminal, commits and pushes "
        f"{AC.approval_rel(prereg.prefix)} (+ .sha256) under their own name; then import_approval.py import; then "
        f"commit and push {AC.import_rel(prereg.prefix)} (+ .sha256)\n"
        f"Approvable now: {'yes' if ok else 'NO'} ({why})\n"
        f"Owner's approval of this fingerprint present: {(root / AC.approval_name(prereg.prefix)).exists()}\n"
        f"Import record present: {(root / AC.import_name(prereg.prefix)).exists()}")
    return prereg, ok


def _check_commit_tree(v6c, prereg, commit, manifest):
    """The pinned files as stored in the owner's commit: what approve_local_near80.py checked is what runs."""
    want = {K.PREREG_NAME: prereg.sha256,
            K.MANIFEST_NAME: (prereg.payload.get("code") or {}).get("code_manifest_sha256")}
    want.update(manifest.get("files") or {})
    bad = []
    for rel, h in sorted(want.items()):
        blob = v6c.git_blob(commit, K.N80_REL + "/" + rel)
        if h is None or blob is None or AC.sha256_bytes(blob) != h:
            bad.append(rel)
    if bad:
        raise AC.ApprovalRefused(f"commit {commit[:16]}... does not hold the pinned near-80 files: {bad}")


def import_record(cfg=None, log=print, v6c=None):
    """Steps 1-6 of the import (module docstring). Returns the record written."""
    cfg = cfg or AC.ApprovalConfig()
    if not cfg.check_git:
        raise AC.ApprovalRefused("an import needs git; a test fixture cannot be imported")
    prereg, manifest, ok, why = approvable_now(cfg)
    if not ok:
        raise AC.ApprovalRefused(f"prereg.json cannot be approved now ({why})")
    branch = (prereg.payload.get("approval") or {}).get("branch")
    prefix = prereg.prefix
    rel = AC.approval_rel(prefix)
    root = Path(cfg.n80_root)
    local = root / AC.approval_name(prefix)
    imp = root / AC.import_name(prefix)
    if imp.exists() or Path(str(imp) + ".sha256").exists():
        raise AC.ApprovalRefused(f"{imp} exists already (an import record is written once)")
    if not local.is_file():
        raise AC.ApprovalRefused(f"no approval of this fingerprint at {local} (pull the owner's commit first)")
    raw = local.read_bytes()
    local_sha = AC.sha256_bytes(raw)
    if AC.sidecar_digest(local) != local_sha:
        raise AC.ApprovalRefused(f"{local.name} differs from its '.sha256' sidecar")
    v6c = v6c or AC.load_v6_common(cfg.repo)
    try:
        ref = v6c.remote_ref(branch)
        tip = v6c.git_ref_commit(ref)
        if tip is None:
            raise AC.ApprovalRefused(f"{ref} does not exist here (fetch the branch from GitHub first)")
        adders = v6c.git_adding_commits(ref, rel)
        if adders is None or len(adders) != 1:
            raise AC.ApprovalRefused(f"exactly one commit on {ref} must add {rel}; git found "
                                     f"{'nothing (git failed)' if adders is None else len(adders)}")
        commit = adders[0]
        facts = v6c.git_commit_facts(commit)
        if facts is None:
            raise AC.ApprovalRefused(f"commit {commit[:16]}... cannot be read")
        blob = v6c.git_blob(commit, rel)
        session_email = v6c.git_session_email()
        record = dict(commit=commit, author=facts["author"], committer=facts["committer"], parents=facts["parents"],
                      committed_sha256=None if blob is None else AC.sha256_bytes(blob),
                      signature_status=v6c.git_signature_status(commit), remote_ref=ref, remote_tip_at_import=tip,
                      session_git_email=session_email)
        AC.check_local_approval(raw, prereg)
        v6c.check_commit_record(record, local_sha, v6c.agent_emails(session_email))
        AC.check_approver(record)
        v6c.check_commit_in_git(record, rel, branch)
        _check_commit_tree(v6c, prereg, commit, manifest)
    except v6c.LocalApprovalProblem as problem:
        raise AC.ApprovalRefused(str(problem)) from None
    body = dict(kind=K.IMPORT_KIND, label=K.LABEL, prereg_sha256=prereg.sha256,
                prereg_version=prereg.payload.get("version"), approval_file=rel, approval_file_sha256=local_sha,
                commit=commit, author=facts["author"], committer=facts["committer"], parents=facts["parents"],
                signature_status=record["signature_status"], branch=branch, remote_tip_at_import=tip,
                session_git_email=session_email, approver_email=K.APPROVER_EMAIL, imported_at=AC.utc_now(), note=NOTE)
    digest = AC.write_json_once(imp, body)
    irel = AC.import_rel(prefix)
    log(f"Near-80 approval imported: {imp} ({digest[:16]})\n  from commit {commit} by {facts['author']} "
        f"(committer {facts['committer']})\nCommit and push it (the session does this):\n"
        f"  git add {irel} {irel}.sha256\n"
        f"  git commit -m \"Near-80: import record of the owner's approval {prefix}\"\n"
        f"  git push origin {branch}\n"
        f"Every near-80 phase refuses until this record is on origin's branch.")
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("show")
    sub.add_parser("import")
    a = ap.parse_args(argv)
    try:
        if a.mode == "import":
            import_record()
            return 0
        return 0 if show()[1] else 2
    except AC.ApprovalRefused as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
