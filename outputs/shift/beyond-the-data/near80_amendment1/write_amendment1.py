#!/usr/bin/env python3
"""Near-80 amendment 1: writes ../near80/amendment_1.json (+ .sha256) once, from the near-80 files as they are now.

The session runs it in the cloud, last: after the amended code, its tests and PREREGISTRATION.md section 19 are final
and reviewed. If anything changes after it has run but before amendment_1.json is pushed, the session removes the two
unpushed files and runs it again. Once pushed, amendment_1.json is never rewritten.

Standard library only. It reads git (read-only) for the approved bytes of the appended document and for the two
commits of the parent's approval. It refuses unless the changed files are exactly the eight that amendment 1 names
(PREREGISTRATION.md section 19.4), PREREGISTRATION.md is only appended to with a section 19, and prereg.json and
code_manifest.json are the approved ones. Every field it writes is checked again by approval_check.check_code
(delta_problems) and by approve_local_near80.py before the owner approves; nothing here is trusted on its own.
'desktop_check' names the desktop check's code (the SHA-256 of check_amendment1.py and check_amendment1.sh, next to
this script): approve_local_near80.py and the session's import accept only a result file made by that code.
  python3 -I write_amendment1.py [--repo DIR]
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True

N80_REL = "outputs/shift/beyond-the-data/near80"
CHECK_REL = "outputs/shift/beyond-the-data/near80_amendment1"
PARENT_PREREG_SHA256 = "51e677ab76c8d2f7fa7779d21db7f53bd40d42ff5e6be278d4abfb34c3396d4a"
PARENT_MANIFEST_SHA256 = "4fc25bbad23eeee4971a575c69611068d7817cb63607a5407fc1cf74b51b8150"
CHANGED = ("PREREGISTRATION.md", "approve_local_near80.py", "code/approval_check.py", "code/import_approval.py",
           "code/settingdata.py", "code/stubs.py", "code/test_approval.py", "code/test_runner.py")
SECTION_19 = b"\n---\n\n## 19. Amendment 1 "
CLEARED = (["critic.main.stall.s1{}".format(s) for s in range(5)]
           + ["critic_full.x.stall.s1{}".format(s) for s in range(5)])
TEXT = dict(
    title="Amendment 1 to near-80 version 8: a fresh seed's data is rebuilt with the log sizes it was generated with",
    document="PREREGISTRATION.md section 19",
    written="2026-10-19",
    cleared_failures=CLEARED,
    halt=dict(utc="2026-10-16T19:03:39Z", kind="FAILED",
              what=("the runner's FAILED halt: a phase exited 1 without its final record, so no new phase started and "
                    "nothing was deleted; all 10 phases below ended that way, in the loading of their seed's data")),
    cause=("settingdata.fresh_seed_data stores the log sizes (500, 200, 300) as 'sizes'; phase_main.run_phase writes "
           "its run sizes {chunk, starts, steps, updates} over that key in every data_main record; FreshSeed passed "
           "the dictionary's keys to dataset.generate, which raised a TypeError before anything was built or trained"),
    unchanged=["prereg.json and code_manifest.json (and their fingerprints)", "the 133 pinned repository files",
               "code/near80_constants.py and code/phase_main.py",
               "every rule, mark, threshold, seed, setting, arm, sentence of section 14, stop rule, budget, memory "
               "and lane rule"],
    runs=["kept: every record under runs/shift_explore/near80/, the plumbing record, the exploratory look and the "
          "ledger (the failed attempts' seconds count toward the budget)",
          "runs after the import: the 10 failed phases (attempt 2), then every phase not yet started",
          "not rerun: the plumbing (nothing in it reads a written fresh record back) and the exploratory look"],
    why_no_result_changes=("the amended FreshSeed changes only where the log sizes come from: for the 10 records "
                           "version 8's wrapper overwrote it uses the gate's sizes (500, 200, 300), which "
                           "fresh_seed_data used to make them, and a record that holds three positive integers is used "
                           "as written; the recorder and the stall slope still come from the record, as before. Every "
                           "phase that rebuilds a fresh seed requires the rebuilt log's hashes and both removals' "
                           "hashes to equal the record's; every phase that trains on the training file also requires "
                           "its SHA-256, its leak check and the four prepared-row hashes to equal the record's (the "
                           "full-log critic trains on the whole log, which the first check covers). So each phase gets "
                           "the data its data record describes, bit for bit, or stops. The 10 failed attempts are not "
                           "stops and make no seed or learner fail; they run again as attempt 2, and no phase's work "
                           "depends on its attempt number"),
    if_not_approved=("the run stays halted; no seeds 10-14 phase after its data runs and no claim on seeds 10-14 is "
                     "read"),
)


def refuse(msg):
    sys.exit("refused: {}".format(msg))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def read(p):
    with open(p, "rb") as f:
        return f.read()


def side(p):
    return read(p + ".sha256").decode().split()[0]


def git(repo, *a):
    r = subprocess.run(("git", "-C", repo) + a, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       stdin=subprocess.DEVNULL, env=dict(os.environ, GIT_OPTIONAL_LOCKS="0"))
    if r.returncode != 0:
        refuse("git {} failed".format(" ".join(a)))
    return r.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))))
    a = ap.parse_args()
    n80 = os.path.join(a.repo, *N80_REL.split("/"))
    chk = os.path.join(a.repo, *CHECK_REL.split("/"))
    out = os.path.join(n80, "amendment_1.json")
    if os.path.lexists(out) or os.path.lexists(out + ".sha256"):
        refuse("{} exists (written once; remove it only if it was never pushed)".format(out))
    praw = read(os.path.join(n80, "prereg.json"))
    psha = sha(praw)
    if psha != PARENT_PREREG_SHA256 or side(os.path.join(n80, "prereg.json")) != psha:
        refuse("prereg.json is not the approved {} (or differs from its sidecar)".format(PARENT_PREREG_SHA256[:16]))
    prereg = json.loads(praw.decode("utf8"))
    mraw = read(os.path.join(n80, "code_manifest.json"))
    msha = sha(mraw)
    if not (msha == PARENT_MANIFEST_SHA256 == prereg["code"]["code_manifest_sha256"]
            and side(os.path.join(n80, "code_manifest.json")) == msha):
        refuse("code_manifest.json is not the manifest the approved prereg.json pins")
    parent_files = json.loads(mraw.decode("utf8"))["files"]
    p16 = psha[:16]
    aname, iname = "approval_near80_{}.json".format(p16), "import_near80_{}.json".format(p16)
    for name in (aname, iname):
        if side(os.path.join(n80, name)) != sha(read(os.path.join(n80, name))):
            refuse("{} differs from its sidecar".format(name))
    imp = json.loads(read(os.path.join(n80, iname)).decode("utf8"))
    approval_commit = imp["commit"]
    import_commits = git(a.repo, "log", "--diff-filter=A", "--format=%H", "--",
                         N80_REL + "/" + iname).decode().split()
    if len(import_commits) != 1:
        refuse("git finds {} commits that add {} (one expected)".format(len(import_commits), iname))
    files = {rel: sha(read(os.path.join(n80, *rel.split("/")))) for rel in parent_files}
    changed = {rel: [parent_files[rel], files[rel]] for rel in sorted(parent_files) if files[rel] != parent_files[rel]}
    if sorted(changed) != sorted(CHANGED):
        refuse("the changed files {} are not the eight amendment 1 names {}".format(sorted(changed), sorted(CHANGED)))
    appended = {}
    for rel in changed:
        if rel.endswith(".md"):
            old = git(a.repo, "show", "{}:{}/{}".format(approval_commit, N80_REL, rel))
            new = read(os.path.join(n80, *rel.split("/")))
            if not (sha(old) == parent_files[rel] and new.startswith(old) and len(new) > len(old)):
                refuse("{} is not the approved text with an amendment appended".format(rel))
            if rel == "PREREGISTRATION.md" and not new[len(old):].startswith(SECTION_19):
                refuse("PREREGISTRATION.md's appended text does not start with '---' and '## 19. Amendment 1'")
            appended[rel] = dict(parent_bytes=len(old), parent_sha256=parent_files[rel])
    body = dict(TEXT, kind="near80-amendment", number=1, label=prereg["label"],
                parent=dict(prereg_sha256=psha, code_manifest_sha256=msha, approval_file=N80_REL + "/" + aname,
                            approval_sha256=sha(read(os.path.join(n80, aname))), import_file=N80_REL + "/" + iname,
                            import_sha256=sha(read(os.path.join(n80, iname))), approval_commit=approval_commit,
                            import_commit=import_commits[0]),
                files=files, delta=dict(changed=changed, added={}, removed=[], appended=appended),
                desktop_check=dict(py=sha(read(os.path.join(chk, "check_amendment1.py"))),
                                   sh=sha(read(os.path.join(chk, "check_amendment1.sh")))))
    text = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False).encode("utf8")
    digest = sha(text)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(text)
    fd = os.open(out + ".sha256", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write("{}  amendment_1.json\n".format(digest).encode())
    print("written: {}\nSHA-256: {}\nfirst 16 characters: {}\nchanged: {}\ndesktop check: {}".format(
        out, digest, digest[:16], sorted(changed), body["desktop_check"]))


if __name__ == "__main__":
    main()
