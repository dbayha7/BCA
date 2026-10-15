"""CLOUD ONLY. The return-path check: after the user pushes outputs/shift/beyond-the-data/stage2_gpu/results/, verify
what is on GitHub's branch (as last fetched: run 'git fetch' first), before anything reads the results.

    /tmp/claude-0/bcavenv/bin/python -I outputs/shift/beyond-the-data/stage2_gpu/cloud/check_return.py [--ref REF]

Checks, all read only (git cat-file, git ls-tree, git log):
  1. results/BUNDLE.json matches its sidecar; the results folder holds exactly the files it lists, each with its
     SHA-256; every result JSON has its .sha256 sidecar in the bundle and it matches.
  2. Every result names this pre-registration (prereg_sha256 / stage2_prereg_sha256), the gate's (gate_prereg_sha256),
     this approval file (approval_sha256) and the pinned detection script (detect_code_manifest_sha256,
     detect_spec_sha256) wherever it records them; every record is for a seed in 5-9 or the confirmatory group.
  3. The bundle's environment: the code hashes equal the pinned manifest, launch.sh is the pinned one and no test
     launcher was used, the software measured equals the gate's software block (Python 3.10.20, the eight versions,
     JAX_PLATFORMS=cpu, OMP_NUM_THREADS=1, XLA_FLAGS unset, backend cpu), the dependency folder equals the recorded
     one, the gate's check accepted the anchors, the late files are the cloud's own, and every phase is marked cpu.
  4. Every commit on the branch that touched results/ was authored and committed by the approver's identity (the
     identity of the approval commit), never by the cloud session's own.
  5. Cross references: each width score names its band, each tau its band, each policy and host values file its
     lock; the confirmatory marks and pooled look use the version the cloud decided; every phase finished or failed on
     record; each phase log ends with 'done' or the phase wrote a recorded stop.
Exit status 0 only when everything passes.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path("/home/user/bca")
OUT = ROOT / "runs" / "shift_explore" / "stage2"
REL = "outputs/shift/beyond-the-data/stage2_gpu/results"
BRANCH = "claude/main-bca-connection-x4o46s"
AGENT_EMAILS = ("noreply@anthropic.com",)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git(*args, binary=False):
    done = subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, timeout=120,
                          env=dict(PATH="/usr/bin:/bin", HOME="/root", GIT_OPTIONAL_LOCKS="0"))
    if done.returncode != 0:
        return None
    return done.stdout if binary else done.stdout.decode("utf8", "replace")


class Check:
    def __init__(self):
        self.failed = 0

    def ok(self, cond, name, detail=""):
        print(f"{'PASS' if cond else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
        if not cond:
            self.failed += 1
        return cond


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=f"refs/remotes/origin/{BRANCH}")
    a = ap.parse_args(argv)
    ck = Check()
    blob = lambda rel: git("cat-file", "blob", f"{a.ref}:{rel}", binary=True)   # noqa: E731
    raw = blob(f"{REL}/BUNDLE.json")
    side = blob(f"{REL}/BUNDLE.json.sha256")
    if not ck.ok(raw is not None and side is not None and side.decode().split()[:1] == [sha(raw)],
                 "BUNDLE.json and its sidecar on " + a.ref):
        return 1
    b = json.loads(raw.decode("utf8"))
    ck.ok(b.get("kind") == "stage2-offcloud-results", "bundle kind")
    ck.ok(not b.get("partial"), "bundle is final", b.get("status", ""))
    listed = {f"{REL}/files/{r['name']}": r for r in b["files"]}
    tree = set((git("ls-tree", "-r", "--name-only", a.ref, "--", REL) or "").split())
    extra = tree - set(listed) - {f"{REL}/BUNDLE.json", f"{REL}/BUNDLE.json.sha256"}
    missing = set(listed) - tree
    ck.ok(not extra and not missing, "results folder holds exactly the listed files",
          f"extra {sorted(extra)[:3]}, missing {sorted(missing)[:3]}")
    content = {}
    bad_hash = []
    for path, r in listed.items():
        data = blob(path)
        if data is None or sha(data) != r["sha256"]:
            bad_hash.append(r["path"])
        else:
            content[r["path"]] = data
    ck.ok(not bad_hash, "every file has its listed SHA-256", str(bad_hash[:3]))

    # the cloud's own pins
    sys.path.insert(0, json.loads((OUT / "code_manifest.json").read_text())["source_dir"])
    import stage2_common as C
    import run_stage2 as RS
    prereg = C.load_prereg()
    fp = prereg.sha256
    pins = prereg.payload["pins"]
    man = json.loads((OUT / "code_manifest.json").read_text(encoding="utf8"))
    gate = json.loads((ROOT / "runs/shift/preregistration.json").read_text(encoding="utf8"))
    gate_sha = sha((ROOT / "runs/shift/preregistration.json").read_bytes())
    approval_sha = sha((OUT / "approval.json").read_bytes())
    detect_man = json.loads((ROOT / "runs/shift_explore/detect/code_manifest.json").read_text(encoding="utf8"))

    # 1-2: sidecars and recorded hashes
    problems = []
    results = {r["path"]: r for r in b["files"] if r["kind"] == "result"}
    for path, r in results.items():
        data = content.get(path)
        if data is None:
            continue
        s = content.get(path + ".sha256")
        if s is None or s.decode().split()[:1] != [sha(data)]:
            problems.append(f"{path}: sidecar missing or different")
            continue
        body = json.loads(data.decode("utf8"))
        want = dict(prereg_sha256=fp, stage2_prereg_sha256=fp, gate_prereg_sha256=gate_sha,
                    approval_sha256=approval_sha, detect_code_manifest_sha256=pins["detect_manifest_sha256"],
                    detect_spec_sha256=detect_man["spec_sha256"])
        for k, v in want.items():
            if k in body and body[k] != v:
                problems.append(f"{path}: {k} is {str(body[k])[:16]}..., not the pinned {v[:16]}...")
        label = str(body.get("label", ""))
        if label != C.LABEL_LONG and not label.startswith("EXPLORATORY/STAGE 2"):
            problems.append(f"{path}: not labelled as a Stage 2 output")
        rep = body.get("rep")
        if rep is not None and int(rep) not in C.CONFIRMATORY_SEEDS:
            problems.append(f"{path}: seed {rep} is not a fresh seed")
        if body.get("group") not in (None, "confirmatory"):
            problems.append(f"{path}: group {body.get('group')}")
        if r.get("backend") != "cpu":
            problems.append(f"{path}: backend {r.get('backend')}")
    ck.ok(not problems, "every result: sidecar, pre-registration, gate, approval and detection hashes, seed, backend",
          "; ".join(problems[:4]))

    # 3: environment
    ck.ok(b.get("prereg_sha256") == fp and b.get("gate_prereg_sha256") == gate_sha and
          b.get("approval_json_sha256") == approval_sha and
          b.get("code_manifest_sha256") == prereg.payload["code"]["code_manifest_sha256"],
          "bundle names this pre-registration, the gate's, this approval and the pinned code manifest")
    ck.ok(b.get("stage2_code") == man["files"], "Stage 2 code used equals the pinned manifest")
    L = b.get("launcher") or {}
    ck.ok(L.get("sha256") == man["files"]["launch.sh"] == b.get("launch_sh_sha256") and not L.get("test"),
          "every phase ran through the pinned launch.sh", f"lanes {L.get('lanes')}, OPENBLAS_CORETYPE "
                                                           f"{L.get('openblas_coretype')}")
    ck.ok(b.get("software_now") == gate["software"] and b.get("software_equal_to_gate") is True,
          "software equals the gate's record (Python, versions, flags, backend cpu)", str(b.get("software_now"))[:200])
    ck.ok(b.get("dependency_record_now") == gate["dependency_directory"], "dependency folder equals the record")
    ck.ok((b.get("anchors_now") or {}).get("gate_accepts") is True, "the gate's check accepted on that processor",
          str((b.get("anchors_now") or {}).get("openblas")))
    ck.ok(b.get("backend") == "cpu" and all(p.get("backend") == "cpu" for p in b.get("phases", {}).values()),
          "backend recorded cpu for every phase")
    late_ok = True
    for name, h in (b.get("late_files_sha256") or {}).items():
        p = OUT / ("detect/" + name if name.startswith("marks_") else name)
        late_ok &= p.exists() and sha(p.read_bytes()) == h
    ck.ok(late_ok and b.get("late_files_sha256"), "the late files used are the cloud's own")

    # 4: commits
    log = git("log", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce%x00%G?", a.ref, "--", REL) or ""
    approval = json.loads((OUT / "approval.json").read_text(encoding="utf8"))
    user_email = re.search(r"<([^<>]+)>", approval["git"]["author"]).group(1).lower()
    session = (git("config", "--get", "user.email") or "").strip().lower()
    refused = {e.lower() for e in AGENT_EMAILS} | ({session} if session else set())
    commits, foreign = 0, []
    for line in log.splitlines():
        h, an, ae, cn, ce, sig = line.split("\0")
        commits += 1
        if ae.lower() != user_email or ce.lower() != user_email or ae.lower() in refused or ce.lower() in refused:
            foreign.append(f"{h[:12]} by {an} <{ae}> / {cn} <{ce}>")
    ck.ok(commits > 0 and not foreign, "every commit touching results/ is the approver's own",
          f"{commits} commit(s) by {user_email}" if not foreign else "; ".join(foreign[:3]))

    # 5: cross references and completeness
    def body_of(rel):
        return json.loads(content[rel].decode("utf8")) if rel in content else None

    xref = []
    for path in results:
        d = path.rsplit("/", 1)[0]
        name = path.rsplit("/", 1)[1]
        body = body_of(path)
        if name in ("scores_record.json", "tau.json"):
            band = body_of(f"{d}/band.json")
            if band is None or band.get("band_file_sha256") != body.get("band_file_sha256"):
                xref.append(f"{path}: names another band")
        if name == "values.json" and ("/policy/" in path or "/host/" in path):
            lock = content.get(f"{d}/lock.json")
            if lock is None or sha(lock) != body.get("lock_sha256"):
                xref.append(f"{path}: names another lock")
    ck.ok(not xref, "width scores, tau, policy and host values name their own band or lock", "; ".join(xref[:3]))
    version = RS.chosen_version()
    pooled = body_of("runs/shift_explore/stage2/pooled_confirmatory.json")
    ck.ok(pooled is not None and pooled.get("group") == "confirmatory" and pooled.get("version") == version and
          pooled.get("seeds") == list(C.CONFIRMATORY_SEEDS), "confirmatory pooled look, with the cloud's version",
          f"version {version}; flags_claim {None if pooled is None else pooled.get('flags_claim')}")
    states = {k: v.get("state") for k, v in (b.get("phases") or {}).items()}
    ck.ok(states and all(s in ("done", "skipped") for s in states.values()),
          "every phase finished or was skipped on record",
          f"{sum(s == 'done' for s in states.values())} done, {sum(s == 'skipped' for s in states.values())} skipped")
    unfinished = []
    for r in b["files"]:
        if r["kind"] == "phase log":
            text = content.get(r["path"], b"").decode("utf8", "replace").strip().splitlines()
            last = text[-1] if text else ""
            if " done: " not in last and " FAILED: " in last:
                unfinished.append(r["path"])
    stops = [p for p in results if p.endswith("stop.json")]
    ck.ok(len(unfinished) <= len(stops), "phase logs end in 'done' except recorded stops",
          f"{len(unfinished)} logs end in FAILED, {len(stops)} recorded stops")
    print(f"RESULT: {'PASS' if ck.failed == 0 else f'FAIL ({ck.failed} checks)'}")
    return 0 if ck.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
