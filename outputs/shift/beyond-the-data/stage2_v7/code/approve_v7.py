"""Stage 2, version 7: show the v7 pre-registration's fingerprint, and import the user's approval (V7_SPEC C10).

  approve_v7.py show      prints prereg_v7.json's path, version, SHA-256 and first 16 characters, and whether it can be
                          approved NOW (final, the v7 code matches its manifest, v6's pins hold); exit status 2 if not.
  approve_v7.py import    the second half of the route, in the cloud, after the user ran approve_local_v7.py at their
                          own terminal, committed approval_v7_<first 16>.json (+ .sha256) under their own name and
                          pushed it. Import reads the commit facts itself with v6's read-only git queries: exactly one
                          commit on origin's branch added the file; the author and the committer are NEITHER the cloud
                          session's identity (noreply@anthropic.com) NOR its configured user.email (v6's rule,
                          S2C/stage2_common.py:83, :483-521, :524-549, called from v6's module) AND both are v6's
                          approver 35092758+dbayha7@users.noreply.github.com (the new, positive check); the file in that
                          commit is the file here and its sidecar matches; it approves THIS prereg_v7.json; the commit
                          and the committed folder hold the pinned v7 files. It writes the import record
                          V7/import_v7_<first 16>.json (+ .sha256) once, which the agent commits and pushes; every v7
                          phase refuses until that record is on origin's branch.
This is still not authentication: git does not authenticate authors (S2/prereg.json:47). The user checks on GitHub
that the commit the import record names is their own. No approval file is ever written here.
"""

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402
import v7_verify as VF  # noqa: E402


class ApproveError(RuntimeError):
    pass


def approvable_now(prereg, v7_root):
    if not prereg.final:
        return False, f"it is a draft (missing: {prereg.payload.get('missing')})"
    try:
        VF.check_v7_code(prereg, v7_root)
        if prereg.payload.get("predecessor") != K.V6:
            return False, "it does not name version 6's pins"
        VF.check_v6_pins()
    except K.V7Refusal as error:
        return False, str(error)
    return True, "final; the v7 code matches its manifest; version 6's pins hold"


def show(v7_root=None, log=print):
    v7_root = Path(v7_root or K.V7)
    prereg = VF.load_prereg_v7(v7_root / "prereg_v7.json")
    ok, why = approvable_now(prereg, v7_root)
    log(f"v7 pre-registration: {prereg.path}\nVersion: {prereg.payload.get('version')}\nSHA-256: {prereg.sha256}\n"
        f"First 16 characters: {prereg.prefix}\nVersion 6 (governing): {K.V6['prereg_sha256']}\n"
        f"Route: the user runs python3 {K.V7_REL}/approve_local_v7.py at their own terminal, commits and pushes "
        f"{VF.approval_rel(prereg.prefix)} (+ .sha256) under their own name; then approve_v7.py import; then commit "
        f"and push {VF.import_rel(prereg.prefix)} (+ .sha256)\n"
        f"Approvable now: {'yes' if ok else 'NO'} ({why})\n"
        f"User's approval of this fingerprint present: {(v7_root / K.APPROVAL_NAME.format(prefix=prereg.prefix)).exists()}\n"
        f"Import record present: {(v7_root / K.IMPORT_NAME.format(prefix=prereg.prefix)).exists()}")
    return prereg, ok


def _check_commit_tree(v6c, prereg, commit, manifest):
    """The pinned files as stored in the user's commit: what approve_local_v7.py checked is what runs."""
    want = {"prereg_v7.json": prereg.sha256,
            "code_manifest_v7.json": (prereg.payload.get("code") or {}).get("code_manifest_v7_sha256")}
    want.update(manifest.get("files") or {})
    bad = []
    for rel, h in sorted(want.items()):
        blob = v6c.git_blob(commit, K.V7_REL + "/" + rel)
        if h is None or blob is None or K.sha256_bytes(blob) != h:
            bad.append(rel)
    if bad:
        raise ApproveError(f"refused: commit {commit[:16]}... does not hold the pinned v7 files: {bad}")


def import_v7(v7_root=None, log=print, v6c=None):
    v7_root = Path(v7_root or K.V7)
    v6c = v6c or K.load_v6_common()
    try:
        prereg = VF.load_prereg_v7(v7_root / "prereg_v7.json")
        ok, why = approvable_now(prereg, v7_root)
        if not ok:
            raise ApproveError(f"refused: prereg_v7.json cannot be approved now ({why})")
        manifest = VF.check_v7_code(prereg, v7_root)
        section = prereg.payload.get("approval") or {}
        branch = section.get("branch")
        if section.get("local_terminal_allowed") is not True or section.get("approver_email") != K.APPROVER_EMAIL:
            raise ApproveError("refused: prereg_v7.json does not accept this route")
        prefix = prereg.prefix
        rel = VF.approval_rel(prefix)
        local = v7_root / K.APPROVAL_NAME.format(prefix=prefix)
        imp = v7_root / K.IMPORT_NAME.format(prefix=prefix)
        if imp.exists() or Path(str(imp) + ".sha256").exists():
            raise ApproveError(f"refused: {imp} exists already (an import record is written once)")
        if not local.is_file():
            raise ApproveError(f"refused: no approval of this fingerprint at {local} (pull the user's commit first)")
        raw = local.read_bytes()
        local_sha = K.sha256_bytes(raw)
        if K.sidecar_digest(local) != local_sha:
            raise ApproveError(f"refused: {local} differs from its SHA-256 sidecar")
        ref = v6c.remote_ref(branch)
        tip = v6c.git_ref_commit(ref)
        if tip is None:
            raise ApproveError(f"refused: {ref} does not exist here (fetch the branch from GitHub first)")
        adders = v6c.git_adding_commits(ref, rel)
        if adders is None or len(adders) != 1:
            raise ApproveError(f"refused: exactly one commit on {ref} must add {rel}; git found "
                               f"{'none (git failed)' if adders is None else len(adders)}")
        commit = adders[0]
        facts = v6c.git_commit_facts(commit)
        if facts is None:
            raise ApproveError(f"refused: commit {commit[:16]}... cannot be read")
        blob = v6c.git_blob(commit, rel)
        session_email = v6c.git_session_email()
        record = dict(commit=commit, author=facts["author"], committer=facts["committer"], parents=facts["parents"],
                      committed_sha256=None if blob is None else K.sha256_bytes(blob),
                      signature_status=v6c.git_signature_status(commit), remote_ref=ref, remote_tip_at_import=tip,
                      cloud_session_git_email=session_email)
        VF.check_local_approval_v7(raw, prereg)
        v6c.check_commit_record(record, local_sha, v6c.agent_emails(session_email))
        VF.check_approver(record)
        v6c.check_commit_in_git(record, rel, branch)
        _check_commit_tree(v6c, prereg, commit, manifest)
    except (K.V7Refusal, v6c.LocalApprovalProblem) as problem:
        raise ApproveError(f"refused: {problem}") from None
    imported_at = K.utc_now()
    body = dict(kind=K.IMPORT_KIND, label=K.LABEL, prereg_sha256=prereg.sha256,
                prereg_version=prereg.payload.get("version"), v6_prereg_sha256=K.V6["prereg_sha256"],
                approval_file=rel, approval_file_sha256=local_sha, commit=commit, author=facts["author"],
                committer=facts["committer"], parents=facts["parents"], signature_status=record["signature_status"],
                branch=branch, remote_tip_at_import=tip, cloud_session_git_email=session_email,
                approver_email=K.APPROVER_EMAIL, imported_at=imported_at,
                note=("Written by 'approve_v7.py import' in the cloud session and committed and pushed by the "
                      "orchestrating agent. It names the commit that added your v7 approval file, as git holds it. "
                      "Check on GitHub that this commit is yours: if it is not, say so, and no v7 phase may run. "
                      "Every v7 phase refuses until this record is on the branch on GitHub."))
    digest = K.write_json_once(imp, body)
    log(f"v7 approval imported: {imp} ({digest[:16]})\n  from commit {commit} by {facts['author']} "
        f"(committer {facts['committer']})\nCommit and push it:\n  git add {VF.import_rel(prefix)} "
        f"{VF.import_rel(prefix)}.sha256\n  git commit -m \"Stage 2: import record of v7 approval {prefix}\"\n"
        f"  git push origin {branch}")
    return body


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("show")
    sub.add_parser("import")
    a = ap.parse_args(argv)
    try:
        if a.mode == "import":
            import_v7()
            return 0
        return 0 if show()[1] else 2
    except (ApproveError, K.V7Refusal) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
