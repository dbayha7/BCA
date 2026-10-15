"""CLOUD ONLY. Stage 2, version 7: the return check (V7_SPEC.md C10 'The return check'), run after the user pushed
V7/results/ (step 13) and the orphan branch stage2-v7-run-files (step 14), before any result is read.

    /tmp/claude-0/bcavenv/bin/python -I outputs/shift/beyond-the-data/stage2_v7/cloud/check_return_v7.py [--fetch]

Everything is read from git as fetched from GitHub (git cat-file, ls-tree, log; --fetch fetches the two branches
first), from the cloud's own copy of OUT/ when it still exists, and from the ledger attempt records the laptop pushed
(R7/ledger/<phase>/<n>.json on the orphan branch). Checks:
  1. fingerprints: results/BUNDLE.json matches its sidecar; results/files/ holds exactly the files it lists, each with
     its SHA-256, every JSON with its matching sidecar; the bundle names the final prereg_v7.json on the branch, its
     code manifest, version 6's pins, the constants file and HANDOFF.json, and the import record of this fingerprint
     is on the branch;
  2. the ledger is complete: every attempt's write log has its record with a matching sidecar; no attempt ran in test
     mode; every phase of the plan that is not inherited is done, skipped on record, or listed as unfinished;
  3. the constants facts (in place of the gate's 'gate_accepts True', GPU/cloud/check_return.py:150-151): the pinned
     file, A within 1e-14, K within 1e-12, B bit-identical, the tripwire installed, wherever an attempt recorded them,
     and recorded by every attempt that computed anything (a phase attempt that reached the hand-over or is done, a
     probe attempt that wrote its record); an attempt interrupted before the constants step needs none;
  4. the inherited files are unchanged: BUNDLE.json's SHA-256 of every inherited file, as found on the laptop at the
     end, against the final snapshot's RUNFILES_SHA256SUMS;
  5. no phase has both an inherited and a new output (no attempt of an inherited phase; no write to an inherited file);
  6. every new file under OUT/ on the orphan branch is named, with its SHA-256, by a ledger attempt whose wrapper
     verified this v7 import record before the phase ran (its checks name the import record's SHA-256: a fact of
     order that needs no clock; the laptop's clock and the cloud's are never compared here; a test fixture, whose
     wrapper does not check the chain, compares the attempt's start with the import time instead);
  7. the cloud's own copy of OUT/ (outside logs/), if it still exists, still matches the final snapshot;
  8. every commit touching results/ and every commit of the orphan branch is the approver's (author and committer),
     never the cloud session's;
  9. the orphan branch's RUNFILES_SHA256SUMS matches its committed bytes and the ledger; every model file matches the
     SHA-256 its lock records; every reference file matches its pin in the final prereg_v7.json;
 10. each seed 2-4 lock made on the laptop records, in its actor_sha256 field only, the re-made actor where the table
     has an entry and the gate's actor where it has none.
Writes V7/return_check.json (+ .sha256). Exit status 0 only when everything passes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
V7_DIR = HERE.parent
LIB = V7_DIR / "laptop" / "lib"
CODE_DIR = V7_DIR / "code"
for _p in (str(HERE), str(LIB), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import runner_v7 as RN  # noqa: E402
import selftest_v7 as ST  # noqa: E402
import make_handoff as MH  # noqa: E402

ORPHAN = "stage2-v7-run-files"
AGENT_EMAILS = ("noreply@anthropic.com",)
LOCK_NAMES = {"policy_train": "policy/seed{s}/lock.json", "policy_no_stall_train": "policy/seed{s}/no_stall/lock.json",
              "host_train": "host/stall/seed{s}/lock.json"}


class Check:
    def __init__(self, out=print):
        self.rows, self.out = [], out

    def ok(self, cond, name, detail=""):
        self.rows.append(dict(name=name, passed=bool(cond), detail=str(detail)[:2000]))
        self.out(f"{'PASS' if cond else 'FAIL'} {name}" + (f": {detail}" if detail else ""))
        return bool(cond)

    @property
    def failed(self):
        return [r for r in self.rows if not r["passed"]]


class Source:
    """Read-only access to one git ref of the repository."""

    def __init__(self, cfg, ref):
        self.cfg, self.ref = cfg, ref

    def blob(self, path):
        code, data = ST.git(self.cfg, "cat-file", "blob", f"{self.ref}:{path}", binary=True)
        return data if code == 0 else None

    def json(self, path, verify=True):
        data = self.blob(path)
        if data is None:
            return None
        if verify:
            side = self.blob(path + ".sha256")
            if side is None or side.decode("utf8", "replace").split()[:1] != [K.sha256_bytes(data)]:
                return None
        try:
            return json.loads(data.decode("utf8"))
        except ValueError:
            return None

    def files(self, prefix=""):
        code, text = ST.git(self.cfg, "ls-tree", "-r", "--name-only", self.ref, "--", prefix) if prefix else \
            ST.git(self.cfg, "ls-tree", "-r", "--name-only", self.ref)
        return text.splitlines() if code == 0 and text else []


def commits_touching(cfg, ref, path=None):
    args = ["log", "--format=%H%x09%an <%ae>%x09%cn <%ce>", ref] + (["--", path] if path else [])
    code, text = ST.git(cfg, *args)
    rows = []
    for line in text.splitlines() if code == 0 else []:
        h, a, c = line.split("\t")
        rows.append(dict(commit=h, author=a, committer=c))
    return rows


def email(who):
    import re
    m = re.search(r"<([^<>]+)>", who or "")
    return m.group(1).lower() if m else None


def check_return(cfg, main_ref, orphan_ref, cloud_out=None, out=print):
    import v7_probes as PR
    ck = Check(out)
    main, orphan = Source(cfg, main_ref), Source(cfg, orphan_ref)
    rel = lambda p: Path(os.path.abspath(p)).relative_to(os.path.abspath(cfg.root)).as_posix()   # noqa: E731
    v7_rel, out_rel, r7_rel = rel(cfg.v7_root), rel(cfg.out), rel(cfg.r7)
    res_rel = v7_rel + "/results"

    # ---- 1. fingerprints
    raw = main.blob(f"{res_rel}/BUNDLE.json")
    side = main.blob(f"{res_rel}/BUNDLE.json.sha256")
    if not ck.ok(raw is not None and side is not None and side.decode().split()[:1] == [K.sha256_bytes(raw)],
                 "results/BUNDLE.json and its sidecar", main_ref):
        return ck
    b = json.loads(raw.decode("utf8"))
    ck.ok(b.get("kind") == "stage2-v7-results" and (not b.get("test") or cfg.test), "bundle kind (not a test bundle)")
    listed = {f"{res_rel}/files/{r['name']}": r for r in b.get("files") or []}
    tree = set(main.files(f"{res_rel}/files"))
    ck.ok(tree == set(listed), "results/files holds exactly the listed files",
          f"extra {sorted(tree - set(listed))[:3]}, missing {sorted(set(listed) - tree)[:3]}")
    bad = [p for p, r in listed.items() if p in tree and K.sha256_bytes(main.blob(p) or b"") != r["sha256"]]
    ck.ok(not bad, "every result file has its listed SHA-256", bad[:3])
    nosc = [r["path"] for r in listed.values() if r["path"].endswith(".json") and
            (r.get("sidecar_ok") is False or (r.get("kind") == "result" and not r.get("sidecar_ok")))]
    ck.ok(not nosc, "every JSON record's sidecar matches", nosc[:3])
    prereg_raw = main.blob(f"{v7_rel}/prereg_v7.json")
    prereg = None if prereg_raw is None else json.loads(prereg_raw.decode("utf8"))
    psha = None if prereg_raw is None else K.sha256_bytes(prereg_raw)
    fp = b.get("fingerprints") or {}
    ck.ok(prereg is not None and prereg.get("approvable") is True and not prereg.get("missing") and
          main.json(f"{v7_rel}/prereg_v7.json") is not None, "the final prereg_v7.json is on the branch")
    if prereg is None:
        return ck
    h = prereg.get("handoff") or {}
    ck.ok(fp.get("prereg_v7_sha256") == psha and fp.get("code_manifest_v7_sha256") ==
          (prereg.get("code") or {}).get("code_manifest_v7_sha256") and fp.get("v6_prereg_sha256") ==
          K.V6["prereg_sha256"] and fp.get("v6_approval_sha256") == K.V6["runs_side_approval_sha256"] and
          fp.get("constants_sha256") == K.CONSTANTS_SHA256 and fp.get("handoff_sha256") == h.get("handoff_sha256"),
          "the bundle names this pre-registration, its manifest, version 6, the constants and the handoff", fp)
    imp = main.json(f"{v7_rel}/{K.IMPORT_NAME.format(prefix=K.prefix_of(psha))}")
    ck.ok(imp is not None and imp.get("prereg_sha256") == psha and imp.get("kind") == K.IMPORT_KIND,
          "the import record of this fingerprint is on the branch")
    imported_at = (imp or {}).get("imported_at") or "9999"
    imp_raw = main.blob(f"{v7_rel}/{K.IMPORT_NAME.format(prefix=K.prefix_of(psha))}")
    imp_sha = None if imp_raw is None else K.sha256_bytes(imp_raw)
    handoff = main.json(f"{v7_rel}/handoff/HANDOFF.json") or {}
    ck.ok(handoff and K.sha256_bytes(main.blob(f"{v7_rel}/handoff/HANDOFF.json")) == h.get("handoff_sha256"),
          "HANDOFF.json on the branch is the pinned one")
    phases = handoff.get("phases") or {}
    inherited = {p for p, s in phases.items() if s == "inherited"}

    # ---- 2, 3. the ledger
    ledger, problems, test_mode, consts_bad = {}, [], [], []
    for path in orphan.files(f"{r7_rel}/ledger"):
        if not path.endswith(".writes"):
            continue
        pid, name = path.split("/")[-2], path.split("/")[-1]
        recp = path[:-len(".writes")] + ".json"
        rec = orphan.json(recp)
        if rec is None:
            problems.append(recp)
            continue
        ledger.setdefault(pid, []).append(rec)
        if rec.get("test_mode") and not cfg.test:
            test_mode.append(recp)
        kind = (rec.get("classification") or {}).get("kind")
        c = rec.get("constants")
        # The constants facts must hold wherever they were recorded, and must be recorded by every attempt that
        # computed anything: a phase attempt that reached the hand-over or is done, a probe attempt that wrote its
        # record. An attempt interrupted before the constants step (after 'checks', while JAX loads) computed nothing
        # and was rerun (C5); it needs none.
        reached = kind == "done" or (not pid.startswith("probe_") and bool(rec.get("handover")))
        if c is not None or reached:
            c = c or {}
            d = c.get("differences") or {}
            ok = (c.get("sha256") == K.CONSTANTS_SHA256 and d.get("B_bit_identical") is True and
                  d.get("max_abs_dA") is not None and d["max_abs_dA"] <= K.TOL_A and
                  d.get("max_abs_dK") is not None and d["max_abs_dK"] <= K.TOL_K and
                  bool((c.get("tripwire") or {}).get("installed")))
            if not ok and kind != "v7-refusal":
                consts_bad.append(recp)
    ck.ok(not problems, "every attempt's write log has its record (sidecar verified)", problems[:3])
    ck.ok(not test_mode, "no attempt ran in test mode", test_mode[:3])
    ck.ok(not consts_bad, "the constants facts hold in every attempt that passed the wrapper's checks", consts_bad[:3])
    states = b.get("phases") or {}
    unfinished = set(b.get("unfinished") or [])
    open_ = [p for p, s in states.items() if s == "not done" and p not in unfinished]
    ck.ok(not open_, "every phase is done, inherited, skipped on record or listed as unfinished", open_[:5])
    nodone = [p for p, s in states.items() if s in ("done", "failed on record") and p not in inherited and
              not any((r.get("classification") or {}).get("kind") == "done" for r in ledger.get(p, []))]
    ck.ok(not nodone, "every phase made on the laptop has a 'done' attempt in the ledger", nodone[:5])
    ck.ok(not unfinished, "no phase is unfinished", sorted(unfinished)[:5])

    # ---- 4. the inherited files
    snap = handoff.get("snapshot") or {}
    code, sums_bytes = ST.git(cfg, "show", f"{snap.get('commit')}:{ST.SUMS_NAME}", binary=True)
    sums = ST.parse_sums(sums_bytes) if code == 0 and K.sha256_bytes(sums_bytes) == snap.get("sums_sha256") else None
    if not ck.ok(sums is not None, "the final snapshot's sums are readable here", snap):
        return ck
    partial = set(handoff.get("partial_files") or [])
    want = {p: s for p, s in sums.items() if not p.startswith(ST.LOGS_REL + "/") and p not in partial}
    got = b.get("inherited_files") or {}
    diff = sorted(p for p in set(want) | set(got) if want.get(p) != got.get(p))
    ck.ok(not diff, "every inherited file is unchanged on the laptop (BUNDLE.json against the final snapshot)", diff[:5])

    # ---- 5. inherited phases have no new output
    both = sorted(p for p in inherited if p in ledger)
    root = os.path.abspath(cfg.root)
    writes = [(w, r) for recs in ledger.values() for r in recs for w in r.get("writes") or []]
    touched = sorted({rel(w["path"]) for w, _ in writes if w.get("path", "").startswith(root + "/") and
                      rel(w["path"]) in want})
    ck.ok(not both and not touched, "no inherited phase ran on the laptop and no inherited file was written",
          f"phases {both[:3]}, files {touched[:3]}")

    # ---- 6. new OUT files named by the ledger, by an attempt that verified the v7 import (an order, not a clock)
    def after_import(start, verified):
        if imp_sha is not None and verified == imp_sha:
            return True
        return bool(cfg.test) and start > imported_at

    named = {}
    for w, r in writes:
        if w.get("sha256") and w.get("path", "").startswith(root + "/"):
            verified = (((r.get("checks") or {}).get("v7_chain")) or {}).get("import_sha256")
            named.setdefault(rel(w["path"]), []).append((w["sha256"], r.get("start_utc") or "", verified))
    orphan_files = orphan.files()
    osums_raw = orphan.blob(ST.SUMS_NAME)
    osums = ST.parse_sums(osums_raw) if osums_raw is not None else {}
    new_out = [p for p in orphan_files if p.startswith(out_rel + "/") and p not in want]
    unnamed = [p for p in new_out if not any(sha == osums.get(p) and after_import(start, verified)
                                             for sha, start, verified in named.get(p, []))]
    ck.ok(not unnamed, "every new file under OUT/ is named, with its SHA-256, by an attempt whose wrapper verified the "
                       "v7 import record", unnamed[:5])
    outside = [p for p in orphan_files if p != ST.SUMS_NAME and not p.startswith(r7_rel + "/") and p not in new_out]
    ck.ok(not outside, "the orphan branch holds nothing outside R7/ and the new OUT/ files", outside[:5])

    # ---- 7. the cloud's own copy of OUT/
    cloud_out = Path(cloud_out) if cloud_out else None
    if cloud_out is not None and cloud_out.is_dir():
        base = cloud_out.parent.parent.parent                    # .../runs/shift_explore/stage2 -> repository root
        mism = [p for p in want if p.startswith(out_rel + "/") and
                (not (base / p).is_file() or K.file_sha256(base / p) != want[p])]
        extra = [rel(p) if K.inside(p, cfg.root) else str(p) for p in cloud_out.rglob("*")
                 if p.is_file() and not K.inside(p, cloud_out / "logs") and
                 Path(os.path.abspath(p)).relative_to(os.path.abspath(base)).as_posix() not in sums]
        ck.ok(not mism and not extra, "the cloud's own copy of OUT/ still matches the final snapshot",
              f"changed {mism[:3]}, written after the snapshot {extra[:3]}")
    else:
        ck.ok(True, "the cloud's own copy of OUT/", "no longer exists here; not compared")

    # ---- 8. identities
    rows = commits_touching(cfg, main_ref, res_rel) + commits_touching(cfg, orphan_ref)
    badid = [r for r in rows if email(r["author"]) != K.APPROVER_EMAIL.lower() or
             email(r["committer"]) != K.APPROVER_EMAIL.lower() or email(r["author"]) in AGENT_EMAILS]
    ck.ok(rows and not badid, "every commit touching results/ and the orphan branch is the approver's", badid[:3])

    # ---- 9. the orphan branch's sums, model files, reference files
    tsums = {p: s for p, s in MH.tree_sums(cfg, orphan_ref).items() if p != ST.SUMS_NAME}
    ck.ok(osums and tsums == osums, "the orphan branch's RUNFILES_SHA256SUMS matches its committed bytes",
          sorted(set(tsums) ^ set(osums))[:3])
    lbad = [p for p in new_out if p in named and osums.get(p) not in {s for s, _, _ in named[p]}]
    ck.ok(not lbad, "the orphan branch's OUT/ files match the ledger's SHA-256", lbad[:3])
    mbad = []
    for p in new_out:
        if p.endswith("/lock.json"):
            lock = orphan.json(p)
            if lock and lock.get("actors_file"):
                m = p.rsplit("/", 1)[0] + "/" + lock["actors_file"]
                if osums.get(m, want.get(m)) != lock.get("actors_file_sha256"):
                    mbad.append(m)
    ck.ok(not mbad, "every model file matches the SHA-256 its lock records", mbad[:3])
    rbad = []
    for pid, row in (h.get("probe_records") or {}).items():
        for f, sha in (row.get("file_pins") or {}).items():
            if osums.get(f"{r7_rel}/{f}") != sha:
                rbad.append(f)
    ck.ok(not rbad, "every reference file matches its pin in the final prereg_v7.json", rbad[:3])

    # ---- 10. seed 2-4 locks record the right actors
    entries = {(e["phase"], e["seed"]): e for e in h.get("remake_table") or []}
    probes = {pid: orphan.json(f"{r7_rel}/selftest/{pid}.json") for pid in PR.all_probe_ids()}
    abad = []
    for s in K.REMAKE_SEEDS:
        for phase, lock_rel in LOCK_NAMES.items():
            pid = f"{phase}_seed{s}" + ("_stall" if phase == "host_train" else "")
            p = f"{out_rel}/{lock_rel.format(s=s)}"
            if pid in inherited or p not in new_out:
                continue
            lock = orphan.json(p) or {}
            e = entries.get((phase, s))
            if phase == "host_train":
                rec = probes.get(f"p2_seed{s}") or {}
                expect = {"plain": e["replace"]["actor"]["actor_sha256"] if e else (rec.get("actor") or {})
                          .get("gate_sha256")}
            else:
                world = "stall" if phase == "policy_train" else "no_stall"
                rec = probes.get(f"p1_seed{s}_{world}") or {}
                acts = (rec.get("p1a") or {}).get("actors") or {}
                keys = {"gapped": "none", "full": "full_none"} if world == "stall" else {"no_stall_gapped": "none"}
                expect = {ln: (e["replace"][k]["actor_sha256"] if e else (acts.get(k) or {}).get("gate_sha256"))
                          for k, ln in keys.items()}
            for ln, sha in expect.items():
                if sha is None or (lock.get("actor_sha256") or {}).get(ln) != sha:
                    abad.append(f"{p}: {ln}")
    ck.ok(not abad, "each seed 2-4 lock made on the laptop records the re-made or the gate's actor as the table says",
          abad[:5])
    return ck


def main(argv=None, out=print):
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = None
    if argv[:1] == ["--v7-test"]:
        fixture, argv = argv[1], argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-ref", default=f"refs/remotes/origin/{K.BRANCH}")
    ap.add_argument("--orphan-ref", default=f"refs/remotes/origin/{ORPHAN}")
    ap.add_argument("--fetch", action="store_true")
    a = ap.parse_args(argv)
    try:
        cfg = RN.config(fixture)
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    if a.fetch:
        for br in (K.BRANCH, ORPHAN):
            ST.git(cfg, "fetch", "--no-tags", "origin", f"+refs/heads/{br}:refs/remotes/origin/{br}", timeout=7200)
    cloud_out = cfg.extra.get("cloud_out") if cfg.test else str(K.OUT)
    ck = check_return(cfg, a.main_ref, a.orphan_ref, cloud_out, out)
    passed = not ck.failed and len(ck.rows) > 1
    body = dict(kind="stage2-v7-return-check", utc=K.utc_now(), passed=passed, main_ref=a.main_ref,
                main_commit=ST.git(cfg, "rev-parse", a.main_ref)[1], orphan_ref=a.orphan_ref,
                orphan_commit=ST.git(cfg, "rev-parse", a.orphan_ref)[1], checks=ck.rows,
                failed=[r["name"] for r in ck.failed],
                rule="refuses on any failure, before any result is read (V7_SPEC.md C10)")
    ST.write_status_json(cfg.v7_root / "return_check.json", body)
    out(f"RETURN CHECK: {'PASSED' if passed else 'FAILED'} ({len(ck.rows)} checks, {len(ck.failed)} failed); "
        f"written to {cfg.v7_root / 'return_check.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
