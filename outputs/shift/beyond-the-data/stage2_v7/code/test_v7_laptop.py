"""Stage 2, version 7: tests of the laptop package and the cloud scripts (V7_SPEC.md C2, C5, C6, C8, C10, C11).

SYNTHETIC FIXTURES ONLY, under the scratchpad (outside /home/user/bca): small git repositories with a fake run-files
branch, stub phases started by a stub launcher that uses the real ledger (v7_ledger.py begin/end, the audit hook, the
real classification), and one phase through the real launch_v7.sh and wrapper with the core's --v7-test seam (a stub
run_stage2.py). No real training runs; nothing is written under runs/, under the gate's folders or any pinned folder.
The real files read (never written): v6's committed code, pre-registration and manifest, and fallback.py (copied into
fixtures for the placement), and V7/world_constants.json (through the wrapper).

What is tested:
  [C8]  the plan: dependencies, seeds 0-4 limited to the policy test, deployment and host, inherited phases skipped,
        failed-on-record seeds skipped, longest chains first, the plan from a synthetic HANDOFF.json;
  [C2]  memory admission (7 GiB unmeasured, 1.25 x peak, max(7 GiB, 2 x peak) for the policy kinds, MemTotal - 8 GiB,
        MemAvailable), at most 2 policy_train until one is measured, the lane cap; lanes are the first N of the 16
        processors the self-test checks (a synthetic sysfs topology), and --cpus outside them refuses (in test mode
        too); a stand-in error blocks only new policy training until --reviewed; typing --reviewed again is harmless,
        and a review typed in a second window is picked up by the running runner;
  [C5]  interruptions: machine-wide (SIGTERM) reruns uncounted, phase-specific (SIGKILL, a memory error) counted, files
        moved never deleted (a torn sidecar included), the alone attempt after the third, UNFINISHED after the
        fourth, independent phases continue, WAITING FOR RETRY; --retry refuses from a non-terminal or for a phase that
        is not unfinished; a request is picked up by a running runner and allows one attempt, run alone; a machine-wide
        interruption of the retry reruns; FAILED and REFUSED halt; files no laptop attempt wrote halt; a done phase
        whose attempt record is missing gets it at the next start; a done phase whose final record no ledgered attempt
        wrote halts (and its provenance source says so); a re-made reference missing from the lock halts; a damaged
        reference file halts before anything starts; the test seam refuses a fixture missing any path setting;
  [C6]  restore: the pinned commit and sums, partial files and logs left out, nothing overwritten, verification,
        placement, the main checkout untouched; refusals (wrong sums, commit not on the branch, a changed file, a
        partial list that differs from HANDOFF.json); the runner refuses without HANDOFF.json and with WITHDRAWN.json;
        make_handoff: the paste block (refused unless from a READY full self-test with P3 passed; the refusal is
        recorded), read byte for byte from a synthetic session transcript, the process check on a synthetic /proc
        (the driver alone refuses), 'ready' (the most recent full self-test pasted must be an accepted one run under
        the committed draft; any pasted P3 failure ends v7), Q7 fixed in the cloud once records are imported, the
        snapshot commit, HANDOFF.json (the stop rule's time in exact form and before the records, within the checked
        clock tolerance, and the cloud's first paste), the final prereg_v7.json from records alone; the way back needs
        the user's quoted words and refuses a ledger listing that names a phase;
  [C10] push_runfiles_v7.sh: refuses anything outside R7/ and the new OUT/ files, a file no ledger names, a changed
        inherited file, a non-terminal; retries a push the remote refused once (POSIX sh); pushes as the approver and
        rebuilds the same commits; check_return_v7 (an attempt interrupted before the constants step passes) on a
        synthetic return (passes; fails on a changed result, an agent commit, an unnamed OUT file, a changed cloud copy,
        a wrong constants fact; orders OUT files by the import record the wrapper verified, not by the clock);
        process-hours count a killed attempt up to its write log's last line;
  [C11] the self-test's verdicts with stub probes: READY (waiting) with the WAIT line naming what is missing (the
        handoff, or the approval chain's reason), NOT READY tags, the forecast, records bound to another draft, the
        short form's record check, ST9's speed line (the stub prints run_p3's own field list) and clock (no probe
        starts when it is off), ST8 leaving V7/results/ out; an interrupted probe attempt settled at the next
        self-test, one self-test at a time; --remake putting back a damaged probe record from the cloud's copy; the
        paste block round trip; run_v7.sh's filter keeps refusals and WAIT lines, and its commands are typeable.

Run (one processor, low priority; the v6 run keeps processors 2 and 3, another agent uses 0):
  cd /home/user/bca && env -i PATH=/usr/bin:/bin HOME=/root PYTHONPATH=/home/user/bca PYTHONDONTWRITEBYTECODE=1 \\
      JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 taskset -c 1 nice -n 10 /tmp/claude-0/bcavenv/bin/python \\
      outputs/shift/beyond-the-data/stage2_v7/code/test_v7_laptop.py
"""

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parent
V7 = CODE.parent
LIB = V7 / "laptop" / "lib"
CLOUD = V7 / "cloud"
for _p in (str(CODE), str(LIB), str(CLOUD)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if "/home/user/bca" not in sys.path:
    sys.path.insert(1, "/home/user/bca")

import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402
import v7_probes as PR  # noqa: E402
import v7_verify as VF  # noqa: E402
import write_prereg_v7 as WP  # noqa: E402
import write_manifest_v7 as WM  # noqa: E402
import runner_v7 as RN  # noqa: E402
import selftest_v7 as ST  # noqa: E402
import make_results_v7 as MR  # noqa: E402
import make_handoff as MH  # noqa: E402
import check_return_v7 as CR  # noqa: E402

TEST_ROOT = K.SCRATCH / "v7_tests_laptop"
PY = K.PYTHON
GIB = RN.GIB
OUT_REL, R7_REL = "runs/shift_explore/stage2", "runs/shift_explore/stage2_v7"
APPROVER = ("David Bayha", K.APPROVER_EMAIL)
TESTER = ("Tester", "tester@example.invalid")
AGENT = ("Claude", "noreply@anthropic.com")


def sh(cmd, cwd=None, env=None, check=True, input=None, ident=None):
    full = dict(RN.BASE_ENV, GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_NOSYSTEM="1", **(env or {}))
    if ident:
        full.update(GIT_AUTHOR_NAME=ident[0], GIT_AUTHOR_EMAIL=ident[1], GIT_COMMITTER_NAME=ident[0],
                    GIT_COMMITTER_EMAIL=ident[1])
    done = subprocess.run([str(c) for c in cmd], cwd=None if cwd is None else str(cwd), env=full, capture_output=True,
                          text=True, input=input)
    if check and done.returncode != 0:
        raise AssertionError(f"{cmd} failed ({done.returncode}):\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return done


def utc_in(hours=0):
    """A UTC time this many hours from now, in v7's form (tests compare the laptop's and the cloud's clocks)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + hours * 3600))


def write_json_sidecar(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = K.canonical(payload).encode("utf8")
    path.write_bytes(text)
    Path(str(path) + ".sha256").write_text(f"{hashlib.sha256(text).hexdigest()}  {path.name}\n")
    return hashlib.sha256(text).hexdigest()


def hx(text):
    return hashlib.sha256(text.encode()).hexdigest()


# ============================================================================================ synthetic probe records

# The re-make block of the synthetic prereg_v7.json of final_prereg() and the self-test fixtures: records bind to it.
DRAFT_BLOCK = K.block_sha256(dict(q7_if_forecast_above_tolerance="stop"))


def p1_rec(seed, world, differs=False, reproduces=False, f_ref=0.0, f_fresh=0.0, block=DRAFT_BLOCK, ref_sha="0" * 64,
           written="2026-10-14T10:00:00Z", inputs=None):
    keys = [n for n, _ in PR.P1A_ARMS[world]]
    actors = {k: dict(laptop_sha256=hx(("L" if differs else "G") + k + str(seed)), gate_sha256=hx("G" + k + str(seed)),
                      max_abs_diff=1e-3 if differs else 0.0, differs=differs, laptop_failed=False, gate_failed=False)
              for k in keys}
    p1b = dict(reproduces_gate=reproduces, max_abs_diff_vs_gate=f_ref, max_abs_diff_vs_p1a=f_ref,
               max_abs_diff_vs_p1c=f_fresh)
    return dict(kind="stage2-v7-probe-p1", probe=PR.probe_id("p1", seed, world), seed=seed, world=world,
                inputs=dict(inputs or {}, remake_block_sha256=block),
                p1a=dict(names_written=sorted(keys), actors=actors,
                         reference_file=dict(path=f"refs/seed{seed}/check2_{world}.msgpack", sha256=ref_sha)),
                p1b=p1b, forecast=PR.forecast_of(actors, p1b, None, world), written_utc=written)


def p2_rec(seed, differs=False, block=DRAFT_BLOCK, ref_sha="1" * 64, written="2026-10-14T10:00:00Z", inputs=None):
    actor = dict(laptop_sha256=hx(("LA" if differs else "GA") + str(seed)), gate_sha256=hx("GA" + str(seed)),
                 max_abs_diff=0.1 if differs else 0.0, differs=differs, laptop_failed=False, gate_failed=False)
    snaps = {n: dict(path=f"refs/seed{seed}/host_plain/{n}", sha256=ref_sha) for n in ("snapshot_050.msgpack",
                                                                                        "snapshot_100.msgpack")}
    return dict(kind="stage2-v7-probe-p2", probe=PR.probe_id("p2", seed), seed=seed,
                inputs=dict(inputs or {}, remake_block_sha256=block), actor=actor, critic=dict(identical=not differs),
                reference_file=snaps["snapshot_100.msgpack"], snapshots=snaps, written_utc=written)


def write_probe_records(r7, block=DRAFT_BLOCK, inputs=None, differs=(), written="2026-10-14T10:00:00Z", over=None):
    """Nine records with real reference files under r7 (each record's file SHA-256 matches)."""
    r7 = Path(r7)
    for s in PR.PROBE_SEEDS:
        for w in PR.WORLDS:
            pid = PR.probe_id("p1", s, w)
            ref = r7 / f"refs/seed{s}/check2_{w}.msgpack"
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes(pid.encode())
            rec = p1_rec(s, w, differs=pid in differs, block=block, ref_sha=K.file_sha256(ref), written=written,
                         inputs=inputs)
            write_json_sidecar(PR.record_path(r7, pid), (over or {}).get(pid, rec))
        pid = PR.probe_id("p2", s)
        d = r7 / f"refs/seed{s}/host_plain"
        d.mkdir(parents=True, exist_ok=True)
        for n in ("snapshot_050.msgpack", "snapshot_100.msgpack"):
            (d / n).write_bytes((pid + n).encode())
        rec = p2_rec(s, differs=pid in differs, block=block, ref_sha=K.file_sha256(d / "snapshot_100.msgpack"),
                     written=written, inputs=inputs)
        rec["snapshots"]["snapshot_050.msgpack"]["sha256"] = K.file_sha256(d / "snapshot_050.msgpack")
        write_json_sidecar(PR.record_path(r7, pid), (over or {}).get(pid, rec))


def probe_pins(r7):
    out = {}
    for pid in PR.all_probe_ids():
        rec = json.loads(PR.record_path(r7, pid).read_text())
        files = [rec["p1a"]["reference_file"]] if rec["kind"] == "stage2-v7-probe-p1" else \
            list(rec["snapshots"].values())
        out[pid] = dict(sha256=K.file_sha256(PR.record_path(r7, pid)), file_pins={f["path"]: f["sha256"] for f in files})
    return out


# ============================================================================================ stub launcher and phase

STUB_LAUNCH = r'''#!/bin/sh
# Test stub of launch_v7.sh: the same ledger calls (begin, end) and phase lock, no wrapper, a stub phase.
F=__F__
CODE=__CODE__
PY=__PY__
R7=__R7__
OUTDIR=__OUT__
CPU=$1; shift
PID=$($PY "$CODE/v7_ledger.py" phase-id -- "$@") || exit 97
mkdir -p "$R7/locks" "$R7/logs" "$R7/ledger"
exec 8>"$R7/locks/$PID.lock"
flock -n 8 || { echo "refused: phase $PID is already running" >&2; exit 97; }
LEDGER=$($PY "$CODE/v7_ledger.py" begin --r7 "$R7" --out "$OUTDIR" --cpu "$CPU" --slot 1 --test -- "$@") || exit 97
NAME=$(echo "$*" | sed 's/--//g; s/  */_/g')
LOG="$R7/logs/$NAME.log"
echo "$(date -u +%FT%TZ) start: $* (cpu $CPU, slot 1, ledger $LEDGER)" >> "$LOG"
$PY "$F/stub_phase.py" "$LEDGER" "$CPU" "$@" >> "$LOG" 2>&1
STATUS=$?
$PY "$CODE/v7_ledger.py" end --ledger "$LEDGER" --status "$STATUS" --log "$LOG" --out "$OUTDIR" --r7 "$R7" >> "$LOG" 2>&1
exit "$STATUS"
'''

STUB_PHASE = r'''"""Test stub of one phase: writes the phase's table row as v6 would (JSON with sidecars), or fails as told."""
import json, os, signal, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, __CODE__)
import v7_common as K
import v7_ledger as LG
F, OUT, R7 = Path(__F__), Path(__OUT__), Path(__R7__)
ledger, cpu, argv = sys.argv[1], sys.argv[2], sys.argv[3:]
key = LG.phase_key(LG.parse_phase_args(argv))
pid = LG.phase_id(key)
n = int(Path(ledger).name.split(".")[0])
beh = json.loads((F / "behaviour.json").read_text()) if (F / "behaviour.json").exists() else {}
seq = beh.get(pid, [])
b = seq[n - 1] if n - 1 < len(seq) else beh.get("*", "ok")
parts = b.split(":")
kind, opts = parts[0], dict(p.split("=") for p in parts[1:])
L = LG.Ledger(ledger)
checks = json.loads((F / "checks.json").read_text()) if (F / "checks.json").exists() else dict(stub=True)
L.event("checks", result=checks, fingerprints=dict(stub=True))
L.event("constants", facts=dict(sha256=K.CONSTANTS_SHA256, differences=dict(
    max_abs_dA=float(opts.get("dA", 0.0)), max_abs_dK=float(opts.get("dK", 0.0)), B_bit_identical=True,
    A_bit_identical=True, K_bit_identical=True), tripwire=dict(installed=["x"], calls={})))
L.install_audit_hook()
with open(F / "trace.jsonl", "a") as h:
    h.write(json.dumps(dict(pid=pid, attempt=n, ev="start", t=time.time(), cpu=cpu, b=b)) + "\n")
time.sleep(float(opts.get("sleep", 0.05)))


def write(rel, payload=None, torn=False):
    p = OUT / rel.replace("*", "x")
    p.parent.mkdir(parents=True, exist_ok=True)
    if rel.endswith(".json"):
        text = K.canonical(payload if payload is not None else dict(kind="stub", pid=pid)).encode()
        p.write_bytes(text)
        side = "0" * 64 if torn else K.sha256_bytes(text)
        Path(str(p) + ".sha256").write_text(f"{side}  {p.name}\n")
    else:
        p.write_bytes(f"{pid} {n}".encode())


def probe(name):
    p = R7 / "selftest" / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def lock_payload():
    s, p = key.get("seed"), key["phase"]
    sha = {}
    if p == "policy_train":
        rec = probe(f"p1_seed{s}_stall")
        if rec:
            acts = rec["p1a"]["actors"]
            sha = dict(none=acts["gapped"]["gate_sha256"], full_none=acts["full"]["gate_sha256"])
    elif p == "policy_no_stall_train":
        rec = probe(f"p1_seed{s}_no_stall")
        if rec:
            sha = dict(none=rec["p1a"]["actors"]["no_stall_gapped"]["gate_sha256"])
    elif p == "host_train" and key.get("world") == "stall":
        rec = probe(f"p2_seed{s}")
        if rec:
            sha = dict(plain=rec["actor"]["gate_sha256"])
    standin = {}
    if kind == "standin_error":
        standin = dict(error="RuntimeError('stub')")
    elif kind == "standin_memory":
        standin = dict(error="MemoryError()")
    m = OUT / LG.row(key)["files"][0]
    return dict(kind="stub-lock", pid=pid, actor_sha256=sha, standin=standin, actors_file="actors.msgpack",
                actors_file_sha256=K.file_sha256(m) if m.exists() else None)


row = LG.row(key)
failed = {f["file"] for f in row["failed"]}
first_final = set(row["final"][0])
others = {f for alt in row["final"][1:] for f in alt}
todo = [f for f in row["files"] if f not in failed and f not in (others - first_final)]
code = 0
if kind in ("kill", "term", "memory", "torn"):
    write(todo[0], torn=kind == "torn")
    if kind == "torn" and len(todo) > 1:
        write(todo[1], lock_payload() if todo[1].endswith("lock.json") else None, torn=True)
    if kind == "memory":
        print("Traceback (most recent call last):\nMemoryError: out of memory", flush=True)
        code = 1
    else:
        os.kill(os.getpid(), signal.SIGKILL if kind in ("kill", "torn") else signal.SIGTERM)
elif kind == "fail":
    print("a stub failure (not memory)", flush=True)
    code = 1
elif kind == "refuse":
    L.event("v7-exit", code=97, reason="stub refusal")
    print("REFUSED (v7, exit 97): stub", flush=True)
    code = 97
elif kind == "stop":
    write([f for f in row["files"] if f.endswith("stop.json")][0], dict(kind="stub-stop", pid=pid))
    code = 2
elif kind == "data_failed":
    write(row["files"][0], dict(kind="stub-data", passed=False))
else:
    for f in todo:
        write(f, lock_payload() if f.endswith("lock.json") else (dict(kind="stub-data", passed=True)
                                                                    if f.endswith("data.json") else None))
L.event("py-exit", code=code, vmhwm_kb=int(opts.get("mem", 400000)), cpu_seconds=0.1)
with open(F / "trace.jsonl", "a") as h:
    h.write(json.dumps(dict(pid=pid, attempt=n, ev="end", t=time.time(), cpu=cpu, b=b)) + "\n")
sys.exit(code)
'''


def final_prereg(v7, handoff_sha, extra=None):
    """A synthetic final prereg_v7.json with a code manifest (one dummy file) and the handoff pin."""
    v7 = Path(v7)
    (v7 / "code").mkdir(parents=True, exist_ok=True)
    (v7 / "code" / "dummy.txt").write_text("synthetic\n")
    msha = write_json_sidecar(v7 / "code_manifest_v7.json",
                              dict(files={"code/dummy.txt": K.file_sha256(v7 / "code" / "dummy.txt")}))
    payload = dict(kind=K.PREREG_KIND, label=K.LABEL, version=7, approvable=True, missing=[],
                   predecessor=dict(K.V6), constants=dict(file=K.CONSTANTS_REL, sha256=K.CONSTANTS_SHA256),
                   code=dict(code_manifest_v7_sha256=msha, files={}),
                   remake=dict(q7_if_forecast_above_tolerance="stop"),
                   handoff=dict(dict(handoff_sha256=handoff_sha, remake_table=[]), **(extra or {})))
    return write_json_sidecar(v7 / "prereg_v7.json", payload)


class Fixture:
    """A runner fixture: the marker, runner.json, the stub launcher and phase, a final prereg_v7.json and HANDOFF.json.
    With git=True the layout is a git repository (repo/runs/..., repo/v7) with a fake run-files snapshot."""

    def __init__(self, root, phases=None, seeds=(2,), mem=(256, 250), lanes_cpus=None, settings=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / RN.TEST_MARKER).write_text(RN.TEST_MARKER_TEXT)
        self.repo = self.root / "repo"
        self.r7, self.out, self.v7 = self.repo / R7_REL, self.repo / OUT_REL, self.repo / "v7"
        for d in (self.r7, self.out, self.v7 / "handoff"):
            d.mkdir(parents=True, exist_ok=True)
        self.meminfo(*mem)
        subs = dict(__F__=repr(str(self.root)), __CODE__=repr(str(CODE)), __OUT__=repr(str(self.out)),
                    __R7__=repr(str(self.r7)))
        text = STUB_PHASE
        for k, v in subs.items():
            text = text.replace(k, v)
        (self.root / "stub_phase.py").write_text(text)
        text = STUB_LAUNCH.replace("__F__", str(self.root)).replace("__CODE__", str(CODE)).replace("__PY__", PY)
        text = text.replace("__R7__", str(self.r7)).replace("__OUT__", str(self.out))
        (self.root / "stub_launch.sh").write_text(text)
        self.settings = dict(root=str(self.repo), r7=str(self.r7), out=str(self.out), v7_root=str(self.v7),
                             prereg=str(self.v7 / "prereg_v7.json"), launcher=str(self.root / "stub_launch.sh"),
                             meminfo=str(self.root / "meminfo"), version="weighted", training=True,
                             cpus=lanes_cpus or list(range(101, 117)), seeds=list(seeds), pooled=False, poll=0.05,
                             retry_poll=0.2, stop_grace=2.0, s_folder=str(self.root / "s"),
                             fallback_target=str(self.root / "fb" / "fallback.py"), runfiles_branch="runfiles",
                             orphan_branch="stage2-v7-run-files", gate_variant_dir=str(self.root / "gate"),
                             approval_commit=None)
        self.settings.update(settings or {})
        self.save()
        if phases is not None:
            self.handoff(phases)

    def save(self):
        (self.root / "runner.json").write_text(json.dumps(self.settings))

    def meminfo(self, total_gib, avail_gib):
        (self.root / "meminfo").write_text(f"MemTotal: {int(total_gib * 1024 * 1024)} kB\n"
                                           f"MemFree: 1 kB\nMemAvailable: {int(avail_gib * 1024 * 1024)} kB\n")

    def handoff(self, phases, partial=(), snapshot=None, extra=None):
        body = dict(kind="stage2-v7-handoff", phases=phases, partial_files=list(partial),
                    snapshot=snapshot or dict(commit="c" * 40, sums_sha256="d" * 64))
        p = self.v7 / "handoff" / "HANDOFF.json"
        for q in (p, Path(str(p) + ".sha256")):
            if q.exists():
                q.unlink()
        sha = write_json_sidecar(p, body)
        final_prereg(self.v7, sha, extra)
        return body

    def behaviour(self, **b):
        (self.root / "behaviour.json").write_text(json.dumps(b))

    def cfg(self):
        return RN.Cfg.from_fixture(self.root)

    def run(self, *args, until_waiting=True, lanes=4):
        lines = []
        argv = ["--v7-test", str(self.root), "run", "--lanes", str(lanes)] + list(args)
        if until_waiting:
            argv.append("--until-waiting")
        code = RN.main(argv, out=lines.append)
        return code, lines

    def trace(self):
        p = self.root / "trace.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def intervals(self):
        starts, rows = {}, []
        for e in self.trace():
            k = (e["pid"], e["attempt"])
            if e["ev"] == "start":
                starts[k] = e["t"]
            elif k in starts:
                rows.append((e["pid"], e["attempt"], starts.pop(k), e["t"]))
        for k, t in starts.items():
            rows.append((k[0], k[1], t, t + 0.01))
        return rows

    def attempts(self, pid):
        out = []
        for n, w in RN.attempt_files(self.r7, pid):
            out.append(RN.attempt_record(w))
        return out


def max_overlap(rows, select=lambda pid: True):
    ev = sorted([(a, 1) for p, _, a, b in rows if select(p)] + [(b, -1) for p, _, a, b in rows if select(p)])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


class Tmp(unittest.TestCase):
    def setUp(self):
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


# ============================================================================================ C8: the plan

class PlanTests(Tmp):
    def test_plan_dependencies_and_seeds(self):
        tasks = RN.build_plan("weighted", True)
        by = {t.pid: t for t in tasks}
        self.assertEqual(len(by), len(tasks))
        kinds04 = {t.kind for t in tasks if t.seed in LG.EXPLORATORY}
        self.assertEqual(kinds04, {"policy_train", "policy_eval", "policy_no_stall_train", "policy_no_stall_eval",
                                   "deploy", "host_train", "host_eval"})
        for t in tasks:                                       # every task is one v7 runs, and its arguments re-parse
            self.assertEqual(LG.phase_id(LG.phase_key(LG.parse_phase_args(t.args))), t.pid)
            LG.check_phase_allowed(t.key)
            for d in t.deps:
                self.assertIn(d, by)
        self.assertEqual(by["deploy_seed2_stall"].deps, ["policy_no_stall_eval_seed2"])
        self.assertEqual(by["host_train_seed6_no_stall"].deps, ["host_eval_seed6_stall"])
        self.assertIn("detect_fit_weighted_seed6_stall_fixed", by)
        self.assertEqual(by["detect_score_seed6_stall_fixed_weighted"].deps, ["detect_fit_weighted_seed6_stall_fixed"])
        self.assertEqual(by["detect_score_seed6_stall_host_uniform"].deps, ["detect_fit_seed6_stall_host"])
        self.assertEqual(set(by["pooled_exploratory"].deps), {t.pid for t in tasks if t.group == "exploratory"} -
                         {"pooled_exploratory"})
        self.assertEqual(by["pooled_confirmatory"].deps, ["detect_marks_confirmatory_weighted"])
        self.assertEqual(len(by["detect_marks_confirmatory_weighted"].deps), 5 * 26)   # 26 phases per fresh seed
        self.assertGreater(by["fresh_data_seed6"].prio, by["policy_train_seed2"].prio)     # longest chains first
        uni = {t.pid for t in RN.build_plan("uniform", False)}
        self.assertFalse(any(p.startswith(("policy_", "host_", "detect_fit_weighted")) for p in uni))
        self.assertIn("deploy_seed2_stall", uni)

    def test_plan_only_from_handoff_and_skips(self):
        fx = Fixture(self.tmp / "fx", seeds=(2, 6))
        phases = {"policy_train_seed2": "inherited", "policy_eval_seed2": "inherited",
                  "deploy_seed2_stall": "partial"}
        fx.handoff(phases)
        write_json_sidecar(fx.out / "fresh/stall/seed6/data.json", dict(passed=False))
        code, lines = fx.run("--plan-only", until_waiting=False)
        self.assertEqual(code, 0, lines)
        text = "\n".join(lines)
        self.assertIn("inherited policy_train_seed2", " ".join(text.split()))
        self.assertRegex(text, r"done\s+fresh_data_seed6")            # failed on record counts as done
        self.assertRegex(text, r"skipped\s+fresh_learners_seed6.*failed its data checks")
        self.assertRegex(text, r"pending\s+deploy_seed2_stall")
        self.assertFalse((fx.r7 / "ledger" / "policy_train_seed2").exists())


# ============================================================================================ C2: memory, lanes

class MemoryUnitTests(unittest.TestCase):
    def test_reservations_and_admission(self):
        peaks = {"deploy": 1 * GIB, "policy_train": 2 * GIB, "policy_no_stall_train": 5 * GIB}
        self.assertEqual(RN.reservation("host_train", peaks), 7 * GIB)            # not measured: 7 GiB
        self.assertEqual(RN.reservation("deploy", peaks), int(1.25 * GIB))
        self.assertEqual(RN.reservation("policy_train", peaks), 7 * GIB)          # max(7 GiB, 2 x 2 GiB)
        self.assertEqual(RN.reservation("policy_no_stall_train", peaks), 10 * GIB)
        ok, _ = RN.admit("deploy", 7 * GIB, [("x", 20 * GIB)], 40 * GIB, 30 * GIB, peaks)
        self.assertTrue(ok)                                   # 27 <= 40 - 8 and 30 >= 7 + 8
        ok, why = RN.admit("deploy", 7 * GIB, [("x", 26 * GIB)], 40 * GIB, 30 * GIB, peaks)
        self.assertFalse(ok)
        self.assertIn("MemTotal", why)
        ok, why = RN.admit("deploy", 7 * GIB, [], 40 * GIB, 14 * GIB, peaks)
        self.assertFalse(ok)
        self.assertIn("MemAvailable", why)
        run2 = [("policy_train", 7 * GIB), ("policy_train", 7 * GIB)]
        self.assertFalse(RN.admit("policy_train", 7 * GIB, run2, 400 * GIB, 400 * GIB, {})[0])
        self.assertTrue(RN.admit("policy_train", 7 * GIB, run2, 400 * GIB, 400 * GIB, {"policy_train": GIB})[0])
        self.assertTrue(RN.admit("policy_eval", 7 * GIB, run2, 400 * GIB, 400 * GIB, {})[0])


class LaneTests(Tmp):
    def fake_sysfs(self, name, n, group):
        root = self.tmp / name
        for i in range(n):
            d = root / f"cpu{i}" / "topology"
            d.mkdir(parents=True)
            (d / "thread_siblings_list").write_text(group(i) + "\n")
        return root

    def test_lanes_are_the_first_n_of_the_sixteen_checked(self):
        """Fourth review: lanes are pick_cpus(16)[:n] (the processors ST4 and ST7 check), never pick_cpus(n)."""
        aff = set(range(24))
        ht = self.fake_sysfs("ht", 24, lambda i: f"{2 * (i // 2)}-{2 * (i // 2) + 1}")   # 12 cores, 2 threads each
        self.assertEqual(RN.pick_cpus(4, ht, aff), [0, 2, 4, 6])                          # one per core
        self.assertEqual(RN.pick_cpus(16, ht, aff), list(range(16)))                      # 12 cores < 16
        for n in range(1, 17):
            self.assertEqual(RN.lane_cpus(n, ht, aff), RN.pick_cpus(16, ht, aff)[:n])
        self.assertEqual(RN.lane_cpus(4, ht, aff), [0, 1, 2, 3])
        flat = self.fake_sysfs("flat", 24, str)                                          # 24 cores, no siblings
        self.assertEqual(RN.lane_cpus(12, flat, aff), list(range(12)))
        self.assertEqual(RN.pick_cpus(16, flat, {1, 2, 3}), [1, 2, 3])                    # only the allowed ones

    def test_cpus_outside_the_checked_ones_are_refused(self):
        """The --cpus refusal runs in test mode too (the fixture's own 16 processors are the checked ones)."""
        fx = Fixture(self.tmp / "fx", phases={})
        code, lines = fx.run("--plan-only", "--cpus", "1,2", until_waiting=False, lanes=2)
        self.assertEqual(code, 1, lines)
        self.assertIn("are not among the 16 the self-test checks", lines[-1])
        code, lines = fx.run("--plan-only", "--cpus", "101,103", until_waiting=False, lanes=2)
        self.assertEqual(code, 0, lines)
        self.assertIn("on processors [101, 103]", lines[0])
        code, lines = fx.run("--plan-only", until_waiting=False, lanes=3)
        self.assertIn("on processors [101, 102, 103]", lines[0])
        self.assertEqual(RN.checked_cpus(fx.cfg()), list(range(101, 117)))


class ProcessHoursTests(unittest.TestCase):
    def test_killed_attempts_count_to_their_write_logs_last_line(self):
        """Fourth review: an attempt without an end time (its launch shell died) counts up to its write log's last line,
        as killed_or_interrupted and in estimated_end_attempts, not in unknown_attempts."""
        def rec(kind, start, end=None, est=None):
            return dict(classification=dict(kind=kind), start_utc=start, end_utc=end, end_utc_estimated=est)
        rows = {"host_train_seed6_stall": [
                    dict(attempt=1, record=rec("done", "2026-10-14T10:00:00Z", "2026-10-14T12:00:00Z")),
                    dict(attempt=2, record=rec("machine-wide", "2026-10-14T13:00:00Z", None, "2026-10-14T13:30:00Z")),
                    dict(attempt=3, record=None)],
                "deploy_seed6_stall": [dict(attempt=1, record=rec("failed", "2026-10-14T10:00:00Z",
                                                                  "2026-10-14T10:15:00Z"))],
                "probe_p1_seed2_stall": [dict(attempt=1, record=rec("done", "2026-10-14T09:00:00Z", None,
                                                                    "2026-10-14T09:06:00Z"))]}
        h = MR.process_hours(None, rows)
        self.assertEqual((h["phases_done"], h["killed_or_interrupted"], h["failed_or_refused"], h["probes"]),
                         (2.0, 0.5, 0.25, 0.1))
        self.assertEqual((h["estimated_end_attempts"], h["unknown_attempts"]), (2, 1))
        self.assertEqual(h["total"], 2.85)
        self.assertEqual(MR._hours(rec("phase-specific", "2026-10-14T10:00:00Z")), (None, False))


class RunnerTests(Tmp):
    def test_runs_the_plan_skipping_inherited_within_lanes(self):
        fx = Fixture(self.tmp / "fx", phases={"policy_train_seed2": "inherited", "policy_eval_seed2": "inherited"})
        fx.behaviour(**{"*": "ok:sleep=0.3"})
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertFalse((fx.r7 / "ledger" / "policy_train_seed2").exists())        # inherited: never started
        ran = {e["pid"] for e in fx.trace()}
        self.assertEqual(len(ran), 8)
        self.assertLessEqual(max_overlap(fx.intervals()), 2)
        st = json.loads((fx.r7 / "status.json").read_text())
        self.assertEqual(st["counts"]["done"], 8)
        self.assertEqual(st["counts"]["inherited"], 2)
        rec = fx.attempts("host_eval_seed2_no_stall")[0]
        self.assertEqual(rec["classification"]["kind"], "done")
        out = []
        self.assertEqual(RN.main(["--v7-test", str(fx.root), "status"], out=out.append), 0)
        self.assertIn("runner running: no", out[0])
        self.assertIn("done 8", out[1])

    def test_policy_cap_and_memory_admission(self):
        fx = Fixture(self.tmp / "fx", phases={}, seeds=(0, 1, 2, 3, 4), mem=(1000, 1000))
        fx.behaviour(**{"*": "ok:sleep=0.4:mem=1000000"})
        code, lines = fx.run(lanes=8)
        self.assertEqual(code, 0, "\n".join(lines))
        rows = fx.intervals()
        pt = sorted((a, b) for p, _, a, b in rows if p.startswith("policy_train"))
        first_end = min(b for _, b in pt)
        before = [r for r in rows if r[0].startswith("policy_train") and r[2] < first_end - 0.05]
        self.assertLessEqual(len(before), 2)                   # at most 2 until one has been measured
        fx2 = Fixture(self.tmp / "fx2", phases={}, seeds=(0, 1, 2, 3, 4), mem=(30, 30))
        fx2.behaviour(**{"*": "ok:sleep=0.3:mem=9000000"})       # measured peaks ~8.6 GiB: reservations grow
        code, lines = fx2.run(lanes=8)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertLessEqual(max_overlap(fx2.intervals()), 3)   # 3 x 7 GiB <= 30 - 8
        self.assertTrue(any("waiting: memory" in ln for ln in lines))

    def test_standin_error_blocks_only_new_policy_training(self):
        fx = Fixture(self.tmp / "fx", phases={}, seeds=(2, 3))
        fx.behaviour(policy_train_seed2=["standin_memory"], **{"*": "ok"})
        code, lines = fx.run(lanes=1)
        text = "\n".join(lines)
        self.assertEqual(code, 5, text)
        self.assertIn("REVIEW: stand-in error in policy/seed2/lock.json", text)
        self.assertIn("run_v7.sh --lanes 1 --reviewed policy_train_seed2 (a second Ubuntu window is fine", text)
        self.assertTrue(fx.attempts("policy_eval_seed2"))                 # other phases continue
        self.assertFalse((fx.r7 / "ledger" / "policy_train_seed3").exists())
        self.assertFalse((fx.r7 / "ledger" / "policy_no_stall_train_seed2").exists())
        code, lines = fx.run("--reviewed", "policy_train_seed3", lanes=1)
        self.assertEqual(code, 1)                                      # no error to review there
        code, lines = fx.run("--reviewed", "policy_train_seed2", lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue((fx.r7 / "reviewed" / "policy_train_seed2.json").exists())
        code, lines = fx.run("--reviewed", "policy_train_seed2", lanes=2)      # typed again (shell history): harmless
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue(any("already written" in ln for ln in lines), lines)
        a = type("A", (), dict(phase="policy_train_seed2", argv="run_v7.sh --reviewed policy_train_seed2"))()
        say = []
        self.assertEqual(RN.cmd_review(fx.cfg(), a, say.append, isatty=False), 1)
        self.assertIn("terminal", say[-1])
        self.assertEqual(RN.cmd_review(fx.cfg(), a, say.append, isatty=True), 0)
        self.assertIn("was already recorded", say[-1])
        rows = RN.standin_errors(fx.cfg(), {})
        self.assertTrue(rows and rows[0]["memory"] and rows[0]["reviewed"])
        prov = json.loads(MR.write_provenance(fx.cfg(), "exploratory").read_text())
        labels = prov["seeds"]["seed2"]["policy stall"]["labels"]
        self.assertEqual(labels[0]["label"], "machine-caused: memory")


    def test_review_from_a_second_window_is_picked_up_by_the_running_runner(self):
        """Fourth review: 'run_v7.sh --reviewed <phase>' while the runner runs records the review (exit 10), and the
        running runner lifts the block within its retry poll; no restart, so no 'another runner is running'."""
        fx = Fixture(self.tmp / "fx", phases={}, seeds=(2, 3))
        fx.behaviour(policy_train_seed2=["standin_error"], policy_train_seed3=["ok:sleep=4"], **{"*": "ok:sleep=0.2"})
        proc = subprocess.Popen([PY, str(LIB / "runner_v7.py"), "--v7-test", str(fx.root), "run", "--lanes", "3"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(RN.BASE_ENV))
        seen = []

        def reader():
            for line in proc.stdout:
                seen.append(line)
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        t0 = time.time()
        while not any("REVIEW: stand-in error" in x for x in seen) and time.time() - t0 < 60:
            time.sleep(0.05)
        a = type("A", (), dict(phase="policy_train_seed2", argv="run_v7.sh --reviewed policy_train_seed2"))()
        say = []
        self.assertEqual(RN.cmd_review(fx.cfg(), a, say.append, isatty=True), 10, say + seen)
        self.assertIn("picks the review up", say[-1])
        self.assertEqual(proc.wait(timeout=300), 0, "".join(seen))
        th.join(timeout=5)
        proc.stdout.close()
        text = "".join(seen)
        self.assertIn("reviewed: every stand-in error has its review now", text)
        self.assertNotIn("WAITING FOR REVIEW", text)
        self.assertTrue(fx.attempts("policy_no_stall_train_seed2"))           # policy training started again
        self.assertEqual(json.loads((fx.r7 / "status.json").read_text())["counts"]["done"], 20)


class FixtureConfigTests(Tmp):
    def test_test_mode_needs_every_path_inside_the_fixture(self):
        """The runner's, self-test's, results' and push's test seam never falls back to a production path."""
        fx = Fixture(self.tmp / "fx", phases={})
        good = dict(fx.settings)
        for k in RN.Cfg.FIXTURE_PATHS:
            with self.subTest(missing=k):
                fx.settings = {x: v for x, v in good.items() if x != k}
                fx.save()
                with self.assertRaisesRegex(K.V7Refusal, "explicitly"):
                    fx.cfg()
                code, lines = fx.run(lanes=1)
                self.assertEqual((code, lines[-1][:8]), (1, "refused:"))
        for k, v, msg in (("r7", str(self.tmp / "elsewhere"), "inside the fixture"),
                          ("probe_script", "/bin/true", "inside the fixture"),
                          ("launcher", "/bin/true", "launcher must lie inside"),
                          ("launcher", str(CODE / "launch_v7.sh"), "launcher must lie inside"),
                          ("version", None, "run conditions")):
            with self.subTest(setting=k, value=v):
                fx.settings = dict(good, **{k: v})
                if k == "version":
                    fx.settings.pop("training")
                fx.save()
                with self.assertRaisesRegex(K.V7Refusal, msg):
                    fx.cfg()
        fx.settings = dict(good, launcher=str(CODE / "launch_v7.sh"), launcher_args=["--v7-test", str(fx.root)])
        fx.save()
        self.assertEqual(fx.cfg().launcher, CODE / "launch_v7.sh")          # the real launcher, given this fixture
        fx.settings = dict(good, launcher=str(CODE / "launch_v7.sh"), launcher_args=["--v7-test", str(self.tmp)])
        fx.save()
        with self.assertRaisesRegex(K.V7Refusal, "launcher must lie inside"):
            fx.cfg()


class DoneEvidenceTests(Tmp):
    def test_done_phase_without_record_is_settled_and_unexplained_files_halt(self):
        fx = Fixture(self.tmp / "fx", phases={})
        fx.behaviour(**{"*": "ok"})
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        # the launch shell of a done phase did not survive: its attempt record is missing
        w = RN.attempt_files(fx.r7, "host_eval_seed2_no_stall")[-1][1]
        rec = w.with_name(w.name.replace(".writes", ".json"))
        rec.unlink()
        Path(str(rec) + ".sha256").unlink()
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertEqual(RN.attempt_record(w)["classification"]["kind"], "done")
        self.assertTrue(any("wrote 1 missing attempt records" in ln for ln in lines), lines)
        self.assertEqual(MR.ledger_rows(fx.cfg())["host_eval_seed2_no_stall"][-1]["record"]["kind"],
                         "stage2-v7-ledger-attempt")
        # a done phase whose final record no laptop attempt wrote (sidecar matching): the run halts, nothing moves
        vals = fx.out / "host/no_stall/seed2/values.json"
        vals.unlink()
        Path(str(vals) + ".sha256").unlink()
        write_json_sidecar(vals, dict(kind="from elsewhere"))
        code, lines = fx.run(lanes=2)
        text = "\n".join(lines)
        self.assertEqual(code, 1, text)
        self.assertIn("HALT: host_eval_seed2_no_stall is done on file, but", text)
        self.assertTrue(vals.exists())
        prov = json.loads(MR.write_provenance(fx.cfg(), "exploratory").read_text())
        src = prov["seeds"]["seed2"]["host no_stall"]["phases"]["host_eval_seed2_no_stall"]["source"]
        self.assertTrue(src.startswith("UNEXPLAINED"), src)
        self.assertEqual(prov["seeds"]["seed2"]["host stall"]["phases"]["host_eval_seed2_stall"]["source"],
                         "v7-laptop")

    def test_remade_reference_not_in_the_lock_halts(self):
        entry = dict(phase="policy_train", seed=2, world="stall", reference_file="refs/seed2/check2_stall.msgpack",
                     file_sha256="0" * 64, names_in_file=["full", "gapped", "random"],
                     replace={"gapped": dict(actor_sha256="a" * 64, lock_name="none"),
                              "full": dict(actor_sha256="b" * 64, lock_name="full_none")},
                     probe_record="p1_seed2_stall", probe_record_sha256="c" * 64)
        fx = Fixture(self.tmp / "fx")
        fx.handoff({}, extra=dict(remake_table=[entry]))
        fx.behaviour(**{"*": "ok"})
        code, lines = fx.run(lanes=1)
        text = "\n".join(lines)
        self.assertEqual(code, 1, text)
        self.assertIn("HALT: reference: the lock of policy_train_seed2 does not record the re-made actors", text)
        self.assertEqual({e["pid"] for e in fx.trace()}, {"policy_train_seed2"})
        code, lines = fx.run(lanes=1)                          # at the next start too, from the files
        self.assertEqual(code, 1)
        self.assertIn("HALT: reference: the lock of policy_train_seed2", "\n".join(lines))

    def test_reference_files_are_checked_against_their_pins_before_the_run(self):
        fx = Fixture(self.tmp / "fx")
        write_probe_records(fx.r7)
        fx.handoff({}, extra=dict(probe_records=probe_pins(fx.r7)))
        (fx.r7 / "refs/seed3/check2_no_stall.msgpack").write_bytes(b"damaged")
        fx.behaviour(**{"*": "ok"})
        code, lines = fx.run(lanes=1)
        self.assertEqual(code, 1)
        self.assertIn("HALT: reference refs/seed3/check2_no_stall.msgpack differs from its pin", lines[-1])
        self.assertFalse(fx.trace())                            # nothing started


# ============================================================================================ C5: interruptions

class InterruptionTests(Tmp):
    def test_classes_alone_unfinished_and_retry(self):
        fx = Fixture(self.tmp / "fx", phases={"policy_train_seed2": "inherited", "policy_eval_seed2": "inherited",
                                              "policy_no_stall_train_seed2": "inherited",
                                              "policy_no_stall_eval_seed2": "inherited"})
        fx.behaviour(host_train_seed2_stall=["term", "kill", "kill:sleep=0.2", "memory", "memory:sleep=0.3",
                                             "term:sleep=0.8", "ok:sleep=0.5"],
                     **{"*": "ok:sleep=0.3"})
        code, lines = fx.run(lanes=3)
        text = "\n".join(lines)
        self.assertEqual(code, 4, text)
        self.assertIn("interrupted: host_train_seed2_stall (machine-wide", text)
        self.assertIn("interrupted: host_train_seed2_stall (phase-specific 1 of 3", text)
        self.assertIn("interrupted: host_train_seed2_stall (phase-specific 3 of 3", text)
        self.assertIn("waiting to run alone: host_train_seed2_stall", text)
        self.assertIn("UNFINISHED: host_train_seed2_stall", text)
        self.assertIn("WAITING FOR RETRY: host_train_seed2_stall", text)
        recs = fx.attempts("host_train_seed2_stall")
        self.assertEqual([r["classification"]["kind"] for r in recs],
                         ["machine-wide", "phase-specific", "phase-specific", "phase-specific", "phase-specific"])
        moved = sorted((fx.r7 / "interrupted" / "host_train_seed2_stall").glob("*.json"))
        self.assertEqual(len(moved), 5)
        mv = json.loads(moved[0].read_text())
        self.assertEqual(list(mv["files"]), ["host/stall/seed2/actors.msgpack"])
        self.assertFalse((fx.out / "host/stall/seed2/actors.msgpack").exists())
        rows = {(p, a): (s, e) for p, a, s, e in fx.intervals()}
        s5, e5 = rows[("host_train_seed2_stall", 5)]
        others = [(p, s, e) for (p, a), (s, e) in rows.items() if p != "host_train_seed2_stall"]
        self.assertFalse([o for o in others if o[1] < e5 and o[2] > s5])  # the attempt after the third ran alone
        self.assertTrue(any(p == "deploy_seed2_no_stall" for p, _, _ in others))   # independent phases ran
        self.assertFalse(fx.attempts("host_eval_seed2_stall"))           # dependent phases did not
        cfg = fx.cfg()
        say = []
        a = type("A", (), dict(phase="host_train_seed2_stall", argv=["run_v7.sh", "--retry", "x"]))()
        self.assertEqual(RN.cmd_retry(cfg, a, say.append, isatty=False), 1)
        self.assertIn("terminal", say[-1])
        say2 = []                                          # through the command line, as run_v7.sh calls it
        self.assertEqual(RN.main(["--v7-test", str(fx.root), "retry", "host_train_seed2_stall", "--argv",
                                  "run_v7.sh --retry host_train_seed2_stall"], out=say2.append), 1)
        self.assertIn("terminal", say2[-1])
        b = type("A", (), dict(phase="deploy_seed2_stall", argv=None))()
        self.assertEqual(RN.cmd_retry(cfg, b, say.append, isatty=True), 1)
        self.assertIn("done", say[-1])
        # a request written while the runner runs is picked up within its retry poll and runs alone
        proc = subprocess.Popen([PY, str(LIB / "runner_v7.py"), "--v7-test", str(fx.root), "run", "--lanes", "3"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                env=dict(RN.BASE_ENV))
        seen = []

        def reader():
            for line in proc.stdout:
                seen.append(line)
        th = threading.Thread(target=reader, daemon=True)
        th.start()
        t0 = time.time()
        while not any("WAITING FOR RETRY" in s for s in seen) and time.time() - t0 < 60:
            time.sleep(0.1)
        self.assertTrue(any("WAITING FOR RETRY" in s for s in seen), "".join(seen))
        self.assertEqual(RN.cmd_retry(cfg, a, say.append, isatty=True), 10, say)
        self.assertEqual(proc.wait(timeout=120), 0, "".join(seen))
        th.join(timeout=5)
        proc.stdout.close()
        text = "".join(seen)
        self.assertIn("retry request 1 for host_train_seed2_stall picked up", text)
        recs = fx.attempts("host_train_seed2_stall")
        self.assertEqual([r["classification"]["kind"] for r in recs[5:]], ["machine-wide", "done"])
        self.assertEqual(recs[5]["retry_request"]["request"], 1)
        req = json.loads((fx.r7 / "retry" / "host_train_seed2_stall" / "1.json").read_text())
        self.assertEqual(req["command_line"], ["run_v7.sh", "--retry", "x"])
        self.assertTrue((fx.r7 / "retry" / "host_train_seed2_stall" / "1.attempt_6.json").exists())
        self.assertTrue((fx.r7 / "retry" / "host_train_seed2_stall" / "1.attempt_7.json").exists())
        self.assertTrue(fx.attempts("host_eval_seed2_no_stall"))
        self.assertEqual(RN.cmd_retry(cfg, a, say.append, isatty=True), 1)      # done now: nothing to retry

    def test_torn_sidecar_is_moved_and_never_stops_the_runner(self):
        fx = Fixture(self.tmp / "fx", phases={})
        fx.behaviour(policy_train_seed2=["torn"], **{"*": "ok"})
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        moved = list((fx.r7 / "interrupted" / "policy_train_seed2").rglob("lock.json*"))
        self.assertEqual(sorted(p.name for p in moved), ["lock.json", "lock.json.sha256"])

    def test_failed_refused_and_unknown_files_halt(self):
        for beh, word, code_want in (("fail", "FAILED: policy_train_seed2 exit 1", 1),
                                     ("refuse", "REFUSED (v7, exit 97): policy_train_seed2", 1)):
            with self.subTest(beh=beh):
                fx = Fixture(self.tmp / beh, phases={})
                fx.behaviour(policy_train_seed2=[beh], **{"*": "ok"})
                code, lines = fx.run(lanes=1)
                self.assertEqual(code, code_want)
                self.assertIn(word, "\n".join(lines))
                self.assertEqual({e["pid"] for e in fx.trace()}, {"policy_train_seed2"})   # nothing new started
        fx = Fixture(self.tmp / "unknown", phases={})
        (fx.out / "policy/seed2").mkdir(parents=True)
        (fx.out / "policy/seed2/actors.msgpack").write_bytes(b"from nowhere")
        code, lines = fx.run(lanes=1)
        self.assertEqual(code, 1)
        self.assertIn("HALT: policy_train_seed2 has files", "\n".join(lines))
        self.assertTrue((fx.out / "policy/seed2/actors.msgpack").exists())             # never moved

    def test_failed_on_record_seed_skips_later_phases_and_stop(self):
        fx = Fixture(self.tmp / "fx", phases={}, seeds=(6,))
        fx.behaviour(fresh_data_seed6=["data_failed"])
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertEqual({e["pid"] for e in fx.trace()}, {"fresh_data_seed6"})
        st = json.loads((fx.r7 / "status.json").read_text())
        self.assertEqual(st["counts"]["skipped"], len(st["tasks"]) - 1)
        fx = Fixture(self.tmp / "fx2", phases={})
        fx.behaviour(policy_train_seed2=["stop"], **{"*": "ok"})
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 0, "\n".join(lines))
        st = json.loads((fx.r7 / "status.json").read_text())["tasks"]
        self.assertEqual(st["policy_eval_seed2"]["state"], "skipped")
        self.assertEqual(st["policy_no_stall_train_seed2"]["state"], "skipped")
        self.assertEqual(st["deploy_seed2_stall"]["state"], "done")

    def test_handoff_missing_and_withdrawn_refuse(self):
        fx = Fixture(self.tmp / "fx", phases={})
        (fx.v7 / "handoff" / "WITHDRAWN.json").write_text("{}")
        code, lines = fx.run(lanes=1)
        self.assertEqual(code, 1)
        self.assertIn("withdrawn", "\n".join(lines))
        (fx.v7 / "handoff" / "WITHDRAWN.json").unlink()
        (fx.v7 / "handoff" / "HANDOFF.json").unlink()
        code, lines = fx.run(lanes=1)
        self.assertEqual(code, 1)
        self.assertIn("handoff missing", "\n".join(lines))
        self.assertFalse(fx.trace())


# ============================================================================================ the real launch_v7.sh

class RealLauncherTests(unittest.TestCase):
    def test_runner_through_the_real_launch_v7(self):
        import test_v7 as T7
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(dir=TEST_ROOT))
        try:
            fx = T7.RedirectFixture.build(tmp / "fx")
            root = Path(fx["root"])
            v7 = Path(fx["v7"])
            tasks = RN.build_plan("weighted", True, seeds=(2,), pooled=False)
            phases = {t.pid: "inherited" for t in tasks if t.pid != "host_eval_seed2_stall"}
            p = v7 / "handoff" / "HANDOFF.json"
            p.unlink()
            Path(str(p) + ".sha256").unlink()
            hsha = write_json_sidecar(p, dict(kind="stage2-v7-handoff", phases=phases))
            payload = json.loads((v7 / "prereg_v7.json").read_text())
            payload["handoff"]["handoff_sha256"] = hsha
            write_json_sidecar(v7 / "prereg_v7.json", payload)
            (root / "meminfo").write_text("MemTotal: 268435456 kB\nMemAvailable: 268435456 kB\n")
            (root / "runner.json").write_text(json.dumps(dict(
                root=str(root), r7=fx["r7"], out=str(root / "out"), v7_root=str(v7), prereg=str(v7 / "prereg_v7.json"),
                launcher=str(CODE / "launch_v7.sh"), launcher_args=["--v7-test", str(root)],
                meminfo=str(root / "meminfo"), version="weighted", training=True, cpus=[1], seeds=[2], pooled=False,
                poll=0.2, retry_poll=1.0, s_folder=fx["s"])))
            lines = []
            code = RN.main(["--v7-test", str(root), "run", "--lanes", "1"], out=lines.append)
            self.assertEqual(code, 0, "\n".join(lines))
            vals = json.loads((root / "out" / "host" / "stall" / "seed2" / "values.json").read_text())
            self.assertEqual(vals["affinity"], [1])
            rec = RN.attempt_record(RN.attempt_files(Path(fx["r7"]), "host_eval_seed2_stall")[-1][1])
            self.assertEqual(rec["classification"]["kind"], "done")
            self.assertEqual(rec["checks"]["environment"]["variables"], sorted(K.PHASE_ENV))
            self.assertTrue(rec["constants"]["differences"]["B_bit_identical"])
            self.assertEqual(rec["cpu"], 1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================================ git fixtures (C6, C10)

V6_FILES = [K.V6["prereg"], K.V6["prereg"] + ".sha256", K.V6["code_manifest"], K.V6["code_manifest"] + ".sha256",
            ST.FALLBACK_SRC_REL]


def git(repo, *args, ident=TESTER, check=True, input=None):
    return sh(["git", "-C", repo] + list(args), ident=ident, check=check, input=input)


class GitFixture(Fixture):
    """A repository with a main branch (one tracked file), a bare origin, a fake run-files branch 'runfiles' holding
    a snapshot (a done phase of seed 0, a partial phase of seed 2, logs, files of no phase) with RUNFILES_SHA256SUMS,
    the v6 files place() needs (copied, untracked), and a cloud copy of the snapshot's OUT/."""

    def __init__(self, root, **kw):
        super().__init__(root, **kw)
        repo = self.repo
        sh(["git", "init", "-q", "-b", "main", repo])
        git(repo, "config", "user.email", TESTER[1])
        git(repo, "config", "user.name", TESTER[0])
        (repo / "README").write_text("tracked\n")
        (repo / ".gitignore").write_text("runs/\n")
        git(repo, "add", "README", ".gitignore")
        git(repo, "commit", "-q", "-m", "main", ident=AGENT)
        self.origin = self.root / "origin.git"
        sh(["git", "init", "-q", "--bare", "-b", "main", self.origin])
        git(repo, "remote", "add", "origin", str(self.origin))
        for rel in V6_FILES + [f"{K.S2_REL}/code/{n}" for n in json.loads(
                (K.ROOT / K.V6["code_manifest"]).read_text())["files"]]:
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(K.ROOT / rel, dst)
        self.snapshot_files = {
            f"{OUT_REL}/policy/seed0/actors.msgpack": b"actors0",
            f"{OUT_REL}/policy/seed2/actors.msgpack": b"partial actors of seed 2",
            f"{OUT_REL}/logs/explore_policy.log": b"2026-10-14T01:00:00Z start: x\nlast line\n",
            f"{OUT_REL}/logs/phase_policy_train_seed_0.log": b"2026-10-13T10:00:00Z start: --phase policy_train --seed "
                                                             b"0 (cpu 2)\n2026-10-13T11:00:00Z done: --phase\n",
            f"{OUT_REL}/prereg.json": b'{"v6": true}',
            "runs/shift/preregistration.json": b'{"gate": true}',
        }
        self.snapshot_json = {f"{OUT_REL}/policy/seed0/lock.json": dict(kind="stub-lock", pid="policy_train_seed0"),
                              f"{OUT_REL}/policy/seed0/values.json": dict(kind="stub", pid="policy_eval_seed0")}
        self.commit, self.sums_sha = self.make_snapshot()
        self.cloud_out = self.root / "cloud" / OUT_REL
        for path, data in self.all_snapshot_bytes().items():
            if path.startswith(OUT_REL + "/"):
                p = self.root / "cloud" / path
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        self.settings.update(pre_handoff=dict(commit=self.commit, sums_sha256=self.sums_sha),
                             cloud_out=str(self.cloud_out))
        self.save()

    def all_snapshot_bytes(self):
        out = dict(self.snapshot_files)
        for path, payload in self.snapshot_json.items():
            text = K.canonical(payload).encode()
            out[path] = text
            out[path + ".sha256"] = f"{hashlib.sha256(text).hexdigest()}  {Path(path).name}\n".encode()
        return out

    def make_snapshot(self, branch="runfiles"):
        work = self.root / "snapwork"
        sh(["git", "init", "-q", "-b", branch, work])
        files = self.all_snapshot_bytes()
        for path, data in files.items():
            p = work / path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        sums = "".join(f"{hashlib.sha256(files[p]).hexdigest()}  {p}\n" for p in sorted(files))
        (work / ST.SUMS_NAME).write_text(sums)
        git(work, "add", "-f", ".")
        git(work, "commit", "-q", "-m", "snapshot", ident=AGENT)
        git(work, "push", "-q", str(self.origin), f"{branch}:refs/heads/{branch}")
        git(self.repo, "fetch", "-q", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}")
        commit = git(work, "rev-parse", "HEAD").stdout.strip()
        return commit, hashlib.sha256(sums.encode()).hexdigest()


class RestoreTests(Tmp):
    def test_restore_extracts_verifies_places_and_leaves_partial_out(self):
        fx = GitFixture(self.tmp / "fx")
        cfg = fx.cfg()
        rec = ST.restore(cfg, fetch=False, out=lambda *a: None)
        self.assertTrue(rec["ok"])
        self.assertEqual(rec["partial_left_out"], [f"{OUT_REL}/policy/seed2/actors.msgpack"])
        self.assertTrue((fx.out / "policy/seed0/lock.json").exists())
        self.assertFalse((fx.out / "policy/seed2/actors.msgpack").exists())          # partial: not extracted
        self.assertFalse((fx.out / "logs").exists())                                 # the cloud's logs: not extracted
        self.assertEqual(rec["phase_states"]["policy_train_seed0"], "inherited")
        self.assertEqual(rec["phase_states"]["policy_train_seed2"], "partial")
        self.assertEqual(rec["phase_states"]["policy_train_seed3"], "not started")
        self.assertTrue((Path(fx.settings["s_folder"]) / "run_stage2.py").exists())  # v6 code placed
        self.assertTrue(Path(fx.settings["fallback_target"]).exists())
        self.assertEqual(git(fx.repo, "status", "--porcelain", "--untracked-files=no").stdout, "")
        again = ST.restore(cfg, fetch=False, out=lambda *a: None)                    # safe to run again
        self.assertEqual(again["files_verified"], rec["files_verified"])

    def test_restore_refusals(self):
        fx = GitFixture(self.tmp / "fx")
        cfg = fx.cfg()
        cfg.extra["pre_handoff"] = dict(commit=fx.commit, sums_sha256="e" * 64)
        with self.assertRaisesRegex(K.V7Refusal, "pinned SHA-256"):
            ST.restore(cfg, fetch=False)
        cfg.extra["pre_handoff"] = dict(commit=git(fx.repo, "rev-parse", "main").stdout.strip(),
                                        sums_sha256=fx.sums_sha)
        with self.assertRaisesRegex(K.V7Refusal, "is not on"):
            ST.restore(cfg, fetch=False)
        cfg = fx.cfg()
        (fx.out / "policy/seed0").mkdir(parents=True)
        (fx.out / "policy/seed0/actors.msgpack").write_bytes(b"something else")
        with self.assertRaisesRegex(K.V7Refusal, "differ from the pinned sums"):
            ST.restore(cfg, fetch=False)
        self.assertEqual((fx.out / "policy/seed0/actors.msgpack").read_bytes(), b"something else")   # not overwritten
        (fx.out / "policy/seed0/actors.msgpack").unlink()
        fx.handoff({}, partial=[], snapshot=dict(commit=fx.commit, sums_sha256=fx.sums_sha))
        with self.assertRaisesRegex(K.V7Refusal, "partial files differ from HANDOFF"):
            ST.restore(fx.cfg(), fetch=False)
        fx.handoff({}, partial=[f"{OUT_REL}/policy/seed2/actors.msgpack"],
                   snapshot=dict(commit=fx.commit, sums_sha256=fx.sums_sha))
        self.assertTrue(ST.restore(fx.cfg(), fetch=False)["ok"])


# ============================================================================================ C11: the self-test

STUB_PROBE_SCRIPT = r'''import json, sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, __CODE__)
import v7_common as K, v7_ledger as LG
args = sys.argv[1:]
cmd = args[0]
opt = {args[i]: args[i + 1] for i in range(1, len(args) - 1, 2)}
r7 = Path(opt["--r7"])
L = LG.Ledger(opt["--ledger"])
src = json.loads(Path(__SRC__).read_text())
if cmd in ("p1", "p2"):
    pid = f"{cmd}_seed{opt['--seed']}" + (f"_{opt['--world']}" if cmd == "p1" else "")
    if src.get("fail") == pid:
        sys.exit(3)
    rec = src["records"][pid]
    K.write_json_once(r7 / "selftest" / f"{pid}.json", rec)
elif cmd == "r1":
    K.write_json_once(r7 / "info" / f"r1_seed{opt['--seed']}_{opt['--world']}.json", dict(kind="stage2-v7-r1"))
elif cmd == "p3":
    import v7_probes as PR                             # the real run_p3's fields, from the list run_p3 itself checks
    vals = dict(seed=2, steps=1000, tree_sha256="t", bytes_sha256=src.get("p3_by_cpu", {}).get(opt["--cpu"], "b"),
                failed=False, cpu=[int(opt["--cpu"])], seconds=1.5)
    print(json.dumps({k: vals[k] for k in PR.P3_FIELDS}))
L.event("py-exit", code=0)
'''


class SelfTestTests(Tmp):
    def stubs(self, **over):
        good = dict(software=dict(python=dict(version="3.10.20", full="x"), software_equal=True, software_differs={},
                                  dependency_equal=True, dependency_record={}, wrong_version=[], lock_packages=67,
                                  files_differ_numeric=[], files_differ_other=[], cpu=dict(model="stub"),
                                  openblas={}, glibc="glibc 2.39"),
                    constants=dict(summary=dict(max_abs_dA=1e-18, max_abs_dK=1e-14), why=None),
                    gate=dict(gate_accepts=True, cpus=1), gate_raw=dict(gate_accepts=False, anchors_differing=44),
                    load_seed=dict(seed=0))
        good.update(over)
        return good

    def setup_fx(self, records, **stub_over):
        fx = GitFixture(self.tmp / "fx", settings=dict(cpus=[1, 2, 3]))
        ST.restore(fx.cfg(), fetch=False, out=lambda *a: None)
        git(fx.repo, "update-ref", "refs/remotes/origin/" + K.BRANCH, "main")
        fx.settings["approval_commit"] = git(fx.repo, "rev-parse", "main").stdout.strip()
        src = self.tmp / "probe_src.json"
        src.write_text(json.dumps(records))
        script = fx.root / "stub_probe.py"
        script.write_text(STUB_PROBE_SCRIPT.replace("__CODE__", repr(str(CODE))).replace("__SRC__", repr(str(src))))
        fx.settings.update(stub_probes=self.stubs(**stub_over), probe_script=str(script))
        fx.save()
        final_prereg(fx.v7, "f" * 64)              # then made a draft (no handoff yet): READY can only be 'waiting'
        payload = json.loads((fx.v7 / "prereg_v7.json").read_text())
        payload.update(approvable=False, missing=["handoff"], handoff=None)
        write_json_sidecar(fx.v7 / "prereg_v7.json", payload)
        return fx

    def records(self, **over):
        recs = {PR.probe_id("p1", s, w): p1_rec(s, w) for s in PR.PROBE_SEEDS for w in PR.WORLDS}
        recs.update({PR.probe_id("p2", s): p2_rec(s) for s in PR.PROBE_SEEDS})
        recs.update(over)
        return dict(records=recs)

    def test_verdicts(self):
        fx = self.setup_fx(self.records())
        cfg = fx.cfg()
        lines = []
        # reference files for the stub records are written by the probes' stub only as records; make them real
        write_probe_records(fx.r7)
        verdict, report = ST.self_test(cfg, full=False, out=lines.append)
        text = "\n".join(lines)
        self.assertEqual(verdict, "READY (waiting for handoff/approval)", text)
        self.assertIn("WAIT handoff: prereg_v7.json here is still the draft", text)        # the reason, above it
        self.assertLess(text.index("WAIT handoff"), text.index("READY (waiting"))
        self.assertIn("WARN ST9 clock: the laptop's clock could not be compared", text)
        self.assertIn("p1_seed2_stall: bit-identical", text)
        summary, recs = ST.decode_block(text)
        self.assertEqual(sorted(recs), sorted(PR.all_probe_ids()))
        self.assertEqual(recs["p2_seed3"], PR.record_path(fx.r7, "p2_seed3").read_bytes())
        self.assertEqual(summary["verdict"], verdict)
        for name, over, tag in (("gate", dict(gate=dict(gate_accepts=False, cpus=1, gate_error="anchors")), "gate"),
                                ("one processor", dict(gate=dict(gate_accepts=True, cpus=2)), "one processor"),
                                ("versions", dict(software=dict(self.stubs()["software"],
                                                                python=dict(version="3.12.3", full="x"))), "versions"),
                                ("constants", dict(constants=dict(why="K differs by 1e-9")), "constants"),
                                ("load_seed", dict(load_seed=dict(error="hash differs")), "load_seed")):
            with self.subTest(case=name):
                fx.settings["stub_probes"] = self.stubs(**over)
                fx.save()
                verdict, _ = ST.self_test(fx.cfg(), full=False, out=lambda *a: None)
                self.assertTrue(verdict.startswith(f"NOT READY ({tag}"), verdict)
        fx.settings["stub_probes"] = self.stubs()
        fx.save()
        rec = json.loads(PR.record_path(fx.r7, "p1_seed4_no_stall").read_text())
        rec["forecast"].update(F_fresh=3e-5, above_tolerance=["F_fresh"])
        (PR.record_path(fx.r7, "p1_seed4_no_stall")).unlink()
        Path(str(PR.record_path(fx.r7, "p1_seed4_no_stall")) + ".sha256").unlink()
        write_json_sidecar(PR.record_path(fx.r7, "p1_seed4_no_stall"), rec)
        verdict, _ = ST.self_test(fx.cfg(), full=False, out=lambda *a: None)
        self.assertEqual(verdict, "NOT READY (P1b: check (b) forecast above 1e-5)")
        # a record written under another draft (Q7 or the probe script changed afterwards): NOT READY, with the way back
        p = PR.record_path(fx.r7, "p2_seed2")
        rec = json.loads(p.read_text())
        rec["inputs"].update(remake_block_sha256="the block of an earlier draft", prereg_v7_sha256_at_run="9" * 64)
        p.unlink()
        Path(str(p) + ".sha256").unlink()
        write_json_sidecar(p, rec)
        lines = []
        verdict, _ = ST.self_test(fx.cfg(), full=False, out=lines.append)
        self.assertEqual(verdict, "NOT READY (P1/P2: records bind to another draft)")
        self.assertIn("write the draft again exactly as the one the records were made under (prereg_v7 9999",
                      "\n".join(lines))
        shutil.rmtree(fx.r7 / "selftest")
        verdict, _ = ST.self_test(fx.cfg(), full=False, out=lambda *a: None)
        self.assertTrue(verdict.startswith("NOT READY (P1/P2: records missing"), verdict)

    def test_full_form_runs_probes_once_and_p3(self):
        fx = self.setup_fx({})
        src_r7 = self.tmp / "src_r7"                    # what the probes would write: records and reference files
        write_probe_records(src_r7)
        shutil.copytree(src_r7 / "refs", fx.r7 / "refs")
        recs = dict(records={pid: json.loads(PR.record_path(src_r7, pid).read_text()) for pid in PR.all_probe_ids()})
        (self.tmp / "probe_src.json").write_text(json.dumps(recs))
        lines = []
        verdict, report = ST.self_test(fx.cfg(), full=True, out=lines.append)
        text = "\n".join(lines)
        self.assertEqual(verdict, "READY (waiting for handoff/approval)", text)
        self.assertTrue(report["data"]["p3"]["ok"])
        self.assertEqual(len(report["data"]["p3"]["runs"]), 6)
        self.assertIn("INFO ST9 speed: one 1,000-step actor-only training run took 1.5-1.5 s", text)
        self.assertTrue(all(set(PR.P3_FIELDS) <= set(x) for x in report["data"]["p3"]["runs"]))
        self.assertEqual(PR.record_path(fx.r7, "p2_seed4").read_bytes(), PR.record_path(src_r7, "p2_seed4").read_bytes())
        rec = RN.attempt_record(RN.attempt_files(fx.r7, "probe_p1_seed2_stall")[0][1])
        self.assertEqual(rec["classification"]["kind"], "done")                  # the probe is ledgered like a phase
        rec = RN.attempt_record(RN.attempt_files(fx.r7, "probe_p3_seed2")[0][1])
        self.assertEqual(rec["classification"]["kind"], "done")
        self.assertEqual(len(list((fx.r7 / "selftest" / "p3").glob("attempt_*.json"))), 6)
        n = len(RN.attempt_files(fx.r7, "probe_p1_seed2_stall"))
        (self.tmp / "probe_src.json").write_text(json.dumps(dict(recs, p3_by_cpu={"2": "other"})))
        verdict, _ = ST.self_test(fx.cfg(), full=True, out=lambda *a: None)
        self.assertEqual(len(RN.attempt_files(fx.r7, "probe_p1_seed2_stall")), n)   # a written record is never rerun
        self.assertEqual(verdict, "NOT READY (P3: results depend on lane or load)")
        (self.tmp / "probe_src.json").write_text(json.dumps(dict(recs, p3_by_cpu={"2": "crash"})))
        script = Path(fx.settings["probe_script"])                          # a P3 run that does not finish
        script.write_text(script.read_text().replace('elif cmd == "p3":\n',
                                                     'elif cmd == "p3":\n    if src.get("p3_by_cpu", {}).get(opt["--cpu"]) '
                                                     '== "crash":\n        sys.exit(9)\n'))
        verdict, report = ST.self_test(fx.cfg(), full=True, out=lambda *a: None)
        self.assertEqual(verdict, "NOT READY (P3: a run did not finish)")
        self.assertEqual((report["p3"]["differ"], report["p3"]["errors"]), (False, 3))
        self.assertFalse(MH.p3_differs(report["p3"]))
        self.assertTrue(MH.p3_differs(dict(ok=False, distinct=2)))             # an older summary: any failure counts

    def test_an_interrupted_probe_attempt_is_settled_at_the_next_self_test(self):
        fx = self.setup_fx(self.records())
        write_probe_records(fx.r7)
        n, w = LG.allocate_attempt(fx.r7, "probe_p1_seed3_stall")                # a self-test killed mid-probe
        LG.append_event(w, "launch-start", probe="p1_seed3_stall", attempt=n, cpu=1, boot_id=K.boot_id(),
                        r7=str(fx.r7), out_root=str(fx.out))
        LG.append_event(w, "wrapper-start", probe="p1_seed3_stall")
        self.assertIsNone(RN.attempt_record(w))
        lines = []
        code = ST.main(["--v7-test", str(fx.root), "short"], out=lines.append)
        self.assertEqual(code, 0, "\n".join(lines))
        rec = RN.attempt_record(w)
        self.assertIsNotNone(rec)                                                  # the record is written now
        self.assertEqual(rec["classification"]["kind"], "done")                   # its probe record exists
        self.assertTrue(any("wrote 1 missing attempt records" in ln for ln in lines))
        with ST.SelfTestLock(fx.cfg()):                                            # one self-test at a time
            self.assertEqual(ST.main(["--v7-test", str(fx.root), "short"], out=lines.append), 1)
        self.assertIn("another self-test is running", lines[-1])

    def test_wait_line_names_what_is_missing(self):
        """Fourth review (major): after steps 10-11 the reason the chain does not verify reaches a WAIT line."""
        fx = self.setup_fx(self.records())
        write_probe_records(fx.r7)
        final_prereg(fx.v7, "f" * 64, dict(probe_records=probe_pins(fx.r7)))      # final, but no HANDOFF.json here
        lines = []
        verdict, report = ST.self_test(fx.cfg(), full=False, out=lines.append)
        self.assertEqual(verdict, "READY (waiting for handoff/approval)", lines)
        wait = [ln for ln in lines if ln.startswith("WAIT ")]
        self.assertEqual(len(wait), 1, lines)
        self.assertIn("handoff missing", wait[0])
        self.assertIn("git -C /home/user/bca pull --ff-only", wait[0])
        self.assertEqual("WAIT " + report["waiting_for"], wait[0])
        from unittest import mock
        cfg = fx.cfg()
        cfg.check_chain = True
        prereg = VF.load_prereg_v7(cfg.prereg)
        with mock.patch.object(VF, "check_handoff", lambda *a, **k: {}), \
                mock.patch.object(VF, "verify_v7_chain", side_effect=VF.ChainProblem("no import record at X")):
            ok, why = ST.ready_after_handoff(cfg, prereg)
        self.assertFalse(ok)
        self.assertTrue(why.startswith("approval: no import record at X. If you have done steps 10 and 11"), why)
        self.assertIn("pull --ff-only", why)

    def test_clock_off_by_more_than_the_tolerance_runs_no_probe(self):
        """Fourth review: the probe records' laptop times enter the handoff; a clock more than 2 minutes off stops the
        full self-test before any probe runs (the short form only warns)."""
        fx = self.setup_fx({})
        fx.settings["clock_offset"] = 900.0
        fx.save()
        lines = []
        verdict, report = ST.self_test(fx.cfg(), full=True, out=lines.append)
        self.assertTrue(verdict.startswith("NOT READY (clock: the laptop's clock is off"), verdict)
        self.assertIn("wsl --shutdown", "\n".join(lines))
        self.assertFalse((fx.r7 / "ledger" / "probe_p1_seed2_stall").exists())        # no probe was started
        self.assertFalse(any(PR.record_path(fx.r7, pid).exists() for pid in PR.all_probe_ids()))
        self.assertEqual(report["clock"]["offset_seconds"], 900.0)
        self.assertNotIn("records missing", verdict)
        write_probe_records(fx.r7)
        verdict, _ = ST.self_test(fx.cfg(), full=False, out=lines.append)              # the short form: a warning
        self.assertEqual(verdict, "READY (waiting for handoff/approval)")
        self.assertTrue(any(ln.startswith("WARN ST9 clock: the laptop's clock is +900 s off") for ln in lines))
        fx.settings["clock_offset"] = -30.0
        fx.save()
        lines = []
        ST.self_test(fx.cfg(), full=False, out=lines.append)
        self.assertTrue(any(ln.startswith("INFO ST9 clock: the laptop's clock agrees") for ln in lines), lines)

    def test_damaged_probe_record_is_put_back_from_the_clouds_copy(self):
        """Fourth review: --remake also repairs a damaged probe record, from V7/handoff/probes/ when it equals the
        final pin (the damaged file is moved aside, never deleted); otherwise it says it cannot be repaired."""
        fx = self.setup_fx(self.records())
        write_probe_records(fx.r7)
        final_prereg(fx.v7, "f" * 64, dict(probe_records=probe_pins(fx.r7)))
        cloud = fx.v7 / "handoff" / "probes" / "selftest"
        cloud.mkdir(parents=True)
        for pid in PR.all_probe_ids():
            shutil.copyfile(PR.record_path(fx.r7, pid), cloud / f"{pid}.json")
        rec = PR.record_path(fx.r7, "p1_seed3_stall")
        good = rec.read_bytes()
        rec.write_bytes(good[:-5])                                                   # torn on the laptop's disk
        with self.assertRaisesRegex(K.V7Refusal, "puts back a damaged probe record"):
            RN.check_probe_pins(fx.cfg(), VF.load_prereg_v7(fx.v7 / "prereg_v7.json"))
        lines = []
        code = ST.main(["--v7-test", str(fx.root), "remake", "--probe", "p1_seed3_stall"], out=lines.append)
        self.assertEqual(code, 0, lines)
        self.assertEqual(rec.read_bytes(), good)
        self.assertEqual(K.sidecar_digest(rec), K.sha256_bytes(good))
        moved = list((fx.r7 / "interrupted" / "records").rglob("p1_seed3_stall.json"))
        self.assertEqual([m.read_bytes() for m in moved], [good[:-5]])
        self.assertTrue(any("was put back" in ln for ln in lines), lines)
        self.assertIn("nothing to remake", lines[-1])
        RN.check_probe_pins(fx.cfg(), VF.load_prereg_v7(fx.v7 / "prereg_v7.json"))   # passes again
        rec.write_bytes(b"{}")
        (cloud / "p1_seed3_stall.json").write_bytes(b"{}")
        code = ST.main(["--v7-test", str(fx.root), "remake", "--probe", "p1_seed3_stall"], out=lines.append)
        self.assertEqual(code, 1)
        self.assertIn("cannot be repaired under v7", lines[-1])
        self.assertEqual(rec.read_bytes(), b"{}")                                    # nothing moved

    def test_st8_leaves_results_out_and_names_changed_files(self):
        """Fourth review: committed interim results rewritten by 'run_v7.sh results' do not stop a restart."""
        fx = self.setup_fx({})
        res = fx.v7 / "results" / "BUNDLE.json"
        res.parent.mkdir(parents=True, exist_ok=True)
        res.write_text("interim\n")
        git(fx.repo, "add", "-f", "v7/results/BUNDLE.json")
        git(fx.repo, "commit", "-q", "-m", "user: interim results", ident=APPROVER)
        res.write_text("gathered again\n")
        r = ST.Report(out=lambda *a: None)
        ST.git_checks(fx.cfg(), r)
        self.assertEqual(r.failed, [], r.lines)
        (fx.repo / "README").write_text("changed\n")
        r = ST.Report(out=lambda *a: None)
        ST.git_checks(fx.cfg(), r)
        self.assertEqual(len(r.failed), 1, r.lines)
        self.assertIn("tracked files are changed in this clone (README;", r.lines[0])

    def test_r1_information_only_once(self):
        fx = self.setup_fx({})
        say = []
        self.assertEqual(ST.r1_cmd(fx.cfg(), say.append), 0)
        self.assertEqual(len(list((fx.r7 / "info").glob("r1_*.json"))), 6)
        self.assertIn("6 written", say[-1])
        self.assertEqual(ST.r1_cmd(fx.cfg(), say.append), 0)
        self.assertIn("0 written, 6 already there", say[-1])
        rec = RN.attempt_record(RN.attempt_files(fx.r7, "probe_r1_seed2_stall")[0][1])
        self.assertEqual(rec["classification"]["kind"], "done")

    def test_block_damage_refused(self):
        lines = ST.encode_block(dict(verdict="READY"), {"p1_seed2_stall": b'{"a": 1}'})
        summary, recs = ST.decode_block("junk\n" + "\n".join("  " + ln for ln in lines) + "\nmore junk")
        self.assertEqual(recs["p1_seed2_stall"], b'{"a": 1}')
        bad = [ln for ln in lines if not ln.startswith("V7DATA p1_seed2_stall 0")]
        with self.assertRaisesRegex(K.V7Refusal, "missing lines"):
            ST.decode_block("\n".join(bad))
        swapped = [ln.replace("V7DATA p1_seed2_stall 0 H4sI", "V7DATA p1_seed2_stall 0 H4sJ") for ln in lines]
        with self.assertRaises(K.V7Refusal):
            ST.decode_block("\n".join(swapped))


# ============================================================================================ the cloud: handoff

def complete_v7_tree(v7):
    import test_v7 as T7
    T7.complete_v7_tree(v7)


class HandoffTests(Tmp):
    def test_import_snapshot_write_and_final_from_records(self):
        fx = GitFixture(self.tmp / "fx")
        v7 = fx.v7
        shutil.rmtree(v7)
        complete_v7_tree(v7)
        man = WM.main([], v7_root=v7, log=lambda *a: None)
        body, dsha = WP.write_draft(v7, fx.r7, log=lambda *a: None)
        self.assertTrue(body["remake"]["q7_status"].startswith("open"))
        block = K.block_sha256(body["remake"])
        inputs = dict(constants_sha256=K.CONSTANTS_SHA256, gate_prereg_sha256=K.GATE_PREREG_SHA256,
                      v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                      probe_script_sha256=man["files"]["code/v7_probes.py"])
        laptop = self.tmp / "laptop_r7"
        write_probe_records(laptop, block=block, inputs=inputs, differs=("p1_seed2_stall", "p2_seed3"),
                            written=utc_in(hours=2))                       # the laptop's clock (ahead of the cloud's)
        records = {pid: PR.record_path(laptop, pid).read_bytes() for pid in PR.all_probe_ids()}
        cfg = fx.cfg()
        say = []
        ready = dict(verdict="READY (waiting for handoff/approval)", form="full", probe_lines=[], utc=utc_in(hours=2),
                     p3=dict(ok=True, distinct=1), prereg_v7_sha256=dsha, clock=dict(offset_seconds=3.0))
        for name, summ in (("short form", dict(ready, form="short")), ("NOT READY", dict(ready, verdict="NOT READY (x)")),
                           ("no P3", dict(ready, p3=None))):
            with self.subTest(case=name):
                bad = self.tmp / "notready.txt"
                bad.write_text("\n".join(ST.encode_block(summ, records)))
                with self.assertRaisesRegex(K.V7Refusal, "not from a READY full self-test"):
                    MH.import_selftest(cfg, bad, say.append)
                self.assertFalse((v7 / "handoff/probes/selftest").exists())          # nothing stored
        self.assertEqual(len(list((v7 / "handoff/probes").glob("selftest_refused_*.json"))), 3)   # but recorded
        # the accepted block, read byte for byte from the session's transcript (nothing retyped)
        block = "\n".join(ST.encode_block(ready, records))
        transcript = self.tmp / "session.jsonl"
        rows = [dict(type="user", message=dict(role="user", content="please take it from here"), uuid="u1"),
                dict(type="user", uuid="u2", timestamp="2026-10-14T12:00:00.000Z",
                     message=dict(role="user", content=[dict(type="text", text="$ sh selftest_v7.sh\n...\n" + block)])),
                dict(type="user", uuid="u3", message=dict(role="user", content=[dict(
                    type="tool_result", content="\n".join(ST.encode_block(dict(ready, verdict="NOT READY (y)"),
                                                                          records)))])),
                dict(type="assistant", uuid="u4", message=dict(role="assistant", content=[dict(type="text",
                                                                                         text="not this " + block)]))]
        transcript.write_text("\n".join(json.dumps(x) for x in rows) + "\n")
        text, source = MH.transcript_block(transcript)
        self.assertEqual((source["message_uuid"], source["message_timestamp"]), ("u2", "2026-10-14T12:00:00.000Z"))
        self.assertEqual(MH.main(["--v7-test", str(fx.root), "import-selftest", "--from-transcript", str(transcript)],
                                 out=say.append), 0, say[-4:])
        self.assertIn("probe records: 9 new", "\n".join(say[-3:]))
        imp = sorted((v7 / "handoff/probes").glob("selftest_import_*.json"))[-1]
        self.assertEqual(json.loads(imp.read_text())["source"]["message_uuid"], "u2")
        paste = self.tmp / "paste.txt"
        paste.write_text("$ sh selftest_v7.sh\n...\n" + block)
        self.assertEqual(MH.import_selftest(cfg, paste, say.append)[2], sorted(PR.all_probe_ids()))  # identical: ok
        stored = v7 / "handoff/probes/selftest/p1_seed2_stall.json"
        self.assertEqual(stored.read_bytes(), records["p1_seed2_stall"])
        # a record bound to another draft is refused
        other = dict(json.loads(records["p2_seed4"]), inputs=dict(inputs, remake_block_sha256="x"))
        bad = self.tmp / "bad.txt"
        bad.write_text("\n".join(ST.encode_block(ready, {"p2_seed4": K.canonical(other).encode()})))
        with self.assertRaisesRegex(K.V7Refusal, "does not bind") as cm:
            MH.import_selftest(cfg, bad, say.append)
        self.assertIn("write the draft again exactly as", str(cm.exception))
        # Q7 can no longer change in the cloud, which has only the imported records (its R7/ is empty)
        cloud_r7 = self.tmp / "cloud_r7"
        with self.assertRaisesRegex(K.V7Refusal, "can no longer change"):
            WP.write_draft(v7, cloud_r7, q7="proceed", q7_answered=True, q7_quote="proceed anyway",
                           q7_utc="2026-10-14T09:00:00Z", log=lambda *a: None)
        self.assertEqual(K.file_sha256(v7 / "prereg_v7.json"), dsha)
        # the handoff rests on the most recent full self-test pasted, run under the committed draft
        self.assertIn("ready: the most recent full self-test", MH.ready(cfg, say.append) and say[-1])
        late = self.tmp / "late.txt"
        late.write_text("\n".join(ST.encode_block(dict(ready, verdict="NOT READY (git: x)"), records)))
        with self.assertRaises(K.V7Refusal):
            MH.import_selftest(cfg, late, say.append)
        with self.assertRaisesRegex(K.V7Refusal, "most recent full self-test pasted .* was refused"):
            MH.ready(cfg, say.append)
        other = self.tmp / "other.txt"
        other.write_text("\n".join(ST.encode_block(dict(ready, prereg_v7_sha256="e" * 64), records)))
        MH.import_selftest(cfg, other, say.append)
        self.assertTrue(any("ran under prereg_v7 eeeeeeeeeeeeeeee" in x for x in say))
        with self.assertRaisesRegex(K.V7Refusal, "ran under prereg_v7 eeee"):
            MH.ready(cfg, say.append)
        MH.import_selftest(cfg, paste, say.append)                           # step 8 again, under the draft
        MH.ready(cfg, say.append)
        # the part 4 snapshot on the fake run-files branch
        (fx.out / "policy/seed2").mkdir(parents=True, exist_ok=True)
        ST.restore(cfg, fetch=False, out=lambda *a: None)
        for path, data in fx.all_snapshot_bytes().items():                  # the cloud holds the partial file too
            p = fx.repo / path
            if not p.exists():
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
        write_json_sidecar(fx.out / "policy/seed2/lock.json", dict(kind="stub-lock"))      # seed 2 finished since
        (fx.out / "logs/explore_policy.log").write_bytes(b"2026-10-14T01:00:00Z start: x\nlast line\nstopped\n")
        procs = fx.root / "proc"                                       # a synthetic /proc: the driver still runs
        for num, cmd in (("233", b"sh\0/tmp/x/ood/stage2_run/explore_policy.sh\0"),
                         ("400", b"python3\0/x/make_handoff.py\0snapshot\0")):
            (procs / num).mkdir(parents=True)
            (procs / num / "cmdline").write_bytes(cmd)
        fx.settings["proc_root"] = str(procs)
        fx.save()
        cfg = fx.cfg()
        with self.assertRaisesRegex(K.V7Refusal, "the driver explore_policy.sh"):
            MH.snapshot(cfg, push=True, out=say.append)
        shutil.rmtree(procs / "233")
        snap = MH.snapshot(cfg, push=True, trailer="Co-Authored-By: test", out=say.append)
        tip = git(fx.repo, "ls-remote", str(fx.origin), "refs/heads/runfiles").stdout.split()[0]
        self.assertEqual(tip, snap["commit"])
        (fx.out / "policy/seed0/lock.json").write_text("changed")
        with self.assertRaisesRegex(K.V7Refusal, "changed"):
            MH.snapshot(cfg, out=say.append)
        args = type("A", (), dict(snapshot=snap["commit"], stop_rule="stop at once", stop_rule_chosen="default",
                                  go_ahead="2026-10-14T13:00:00Z", stop="2026-10-14T13:05:00Z",
                                  routines=str(self.tmp / "routines.json"), processes=str(self.tmp / "procs.json")))()
        (self.tmp / "routines.json").write_text(json.dumps([dict(id=MH.WATCHDOG, name="watchdog", final_state="active",
                                                                 starts_phases=True)]))
        (self.tmp / "procs.json").write_text(json.dumps([dict(pid=275, cmd="run_stage2.py", signal=15)]))
        with self.assertRaisesRegex(K.V7Refusal, "not deleted"):
            MH.write_handoff(cfg, args, say.append)
        (self.tmp / "routines.json").write_text(json.dumps([dict(id=MH.WATCHDOG, name="watchdog", final_state="deleted",
                                                                 starts_phases=True)]))
        args.stop_rule = "let them finish"
        for chosen, msg in ((utc_in(hours=3), "not before the first P1/P2 record"),     # after the records
                            (utc_in(hours=1), "not before the first P1/P2 record"),     # after the cloud's import
                            (utc_in(hours=-24).replace("T", " ")[:16], "exactly the form")):  # not a UTC time
            with self.subTest(chosen=chosen):
                args.stop_rule_chosen = chosen
                with self.assertRaisesRegex(K.V7Refusal, msg):
                    MH.write_handoff(cfg, args, say.append)
        args.stop_rule_chosen = utc_in(hours=-24)
        hand = MH.write_handoff(cfg, args, say.append)
        self.assertEqual(hand["selftest"]["verdict"], ready["verdict"])
        self.assertTrue(hand["selftest"]["p3"]["ok"])
        self.assertEqual(hand["draft_prereg_v7_sha256"], dsha)
        self.assertEqual(sorted(hand["process_check"]), ["checked_utc", "explore_policy_sh", "note", "run_stage2_py"])
        self.assertEqual((hand["process_check"]["note"], hand["process_check"]["explore_policy_sh"]),
                         ("test fixture: synthetic processes", []))
        self.assertEqual(hand["selftest"]["prereg_v7_sha256_at_selftest"], dsha)
        self.assertEqual([r["kind"] for r in hand["pasted_selftests"]],
                         ["refused"] * 3 + ["accepted", "accepted", "refused", "accepted", "accepted"])
        self.assertEqual(hand["laptop_clock"]["tolerance_seconds"], ST.CLOCK_TOLERANCE_S)
        self.assertEqual(hand["phases"]["policy_train_seed2"], "inherited")
        self.assertEqual(hand["phases"]["policy_train_seed0"], "inherited")
        self.assertEqual(hand["partial_files"], [])
        self.assertEqual(hand["cloud_process_hours"]["total_completed_hours"], 1.0)
        self.assertEqual(hand["last_log_lines"][-1], "stopped")
        with self.assertRaisesRegex(K.V7Refusal, "written once"):
            MH.write_handoff(cfg, args, say.append)
        with self.assertRaisesRegex(K.V7Refusal, "can no longer change"):     # after the handoff, still fixed
            WP.write_draft(v7, cloud_r7, q7="proceed", q7_answered=True, q7_quote="proceed anyway",
                           q7_utc="2026-10-14T09:00:00Z", log=lambda *a: None)
        self.assertEqual(WP.write_draft(v7, cloud_r7, log=lambda *a: None)[1], dsha)   # its own draft: allowed
        final, fsha = WP.write_final(v7 / "handoff/HANDOFF.json", v7, v7 / "handoff/probes", log=lambda *a: None,
                                     require_files=False)
        self.assertTrue(final["approvable"])
        self.assertEqual(final["handoff"]["selftest"]["import_record_sha256"], hand["selftest"]["import_record_sha256"])
        kinds = sorted((e["phase"], e["seed"]) for e in final["handoff"]["remake_table"])
        self.assertEqual(kinds, [("host_train", 3)])                         # policy_train seed 2 is inherited
        self.assertEqual(final["handoff"]["labels"]["seed2 policy stall"], "inherited (cloud, v6)")
        pins = final["handoff"]["probe_records"]["p1_seed2_stall"]["file_pins"]
        self.assertEqual(pins, {"refs/seed2/check2_stall.msgpack": K.file_sha256(laptop / "refs/seed2/check2_stall.msgpack")})
        with self.assertRaisesRegex(K.V7Refusal, "HALT: reference"):          # present but different: refused
            (v7 / "handoff/probes/refs/seed2").mkdir(parents=True)
            (v7 / "handoff/probes/refs/seed2/check2_stall.msgpack").write_bytes(b"x")
            WP.build_final(json.loads(K.canonical(body)), dsha, v7 / "handoff/HANDOFF.json", v7 / "handoff/probes",
                           require_files=False)
        say = []
        base = ["--v7-test", str(fx.root), "withdraw", "--reason", "test"]
        quote = ["--user-quote", "no phase has started on my laptop", "--user-utc", "2026-10-14T13:30:00Z"]
        listing = self.tmp / "ls.txt"
        listing.write_text("root@laptop:~# ls -1 /home/user/bca/runs/shift_explore/stage2_v7/ledger\n"
                           "probe_p1_seed2_stall\npolicy_train_seed2\n")
        for args_ in (base, base + ["--no-laptop-phase-started"],
                      base + ["--no-laptop-phase-started", "--user-quote", "x", "--user-utc", "14 Oct 13:30"],
                      base + ["--no-laptop-phase-started"] + quote + ["--ledger-listing", str(listing)]):
            with self.subTest(args=args_[3:]):
                self.assertEqual(MH.main(args_, out=say.append), 1, say[-1:])
                self.assertFalse((v7 / "handoff/WITHDRAWN.json").exists())
        self.assertIn("phase folders ['policy_train_seed2']", say[-1])
        listing.write_text("root@laptop:~# ls -1 /home/user/bca/runs/shift_explore/stage2_v7/ledger\n"
                           "probe_p1_seed2_stall\nprobe_p3_seed2\n")
        self.assertEqual(MH.main(base + ["--no-laptop-phase-started"] + quote + ["--ledger-listing", str(listing)],
                                 out=say.append), 0, say[-1:])
        w = json.loads((v7 / "handoff/WITHDRAWN.json").read_text())
        self.assertEqual(w["user_confirmation"], dict(quote="no phase has started on my laptop",
                                                      utc="2026-10-14T13:30:00Z"))
        self.assertIn("probe_p3_seed2", w["ledger_listing"])


class HandoffUnitTests(Tmp):
    def test_process_check_sees_the_driver_and_the_phases(self):
        """Fourth review: the driver check of 'snapshot' and 'write' (C6 step 1), on a synthetic /proc."""
        procs = self.tmp / "proc"
        rows = {"233": b"sh\0/tmp/x/scratchpad/ood/stage2_run/explore_policy.sh\0",
                "26518": b"/tmp/claude-0/bcavenv/bin/python\0/x/run_stage2.py\0--phase\0host_train\0",
                "400": b"python3\0/x/make_handoff.py\0write\0", "12": b"bash\0", "self": b"ignored"}
        for num, cmd in rows.items():
            (procs / num).mkdir(parents=True)
            (procs / num / "cmdline").write_bytes(cmd)
        got = MH.cloud_processes(procs)
        self.assertEqual([r["pid"] for r in got["explore_policy_sh"]], [233])
        self.assertEqual([r["pid"] for r in got["run_stage2_py"]], [26518])
        with self.assertRaisesRegex(K.V7Refusal, "the driver explore_policy.sh"):
            MH.stopped_check(got)
        shutil.rmtree(procs / "26518")
        with self.assertRaisesRegex(K.V7Refusal, "explore_policy.sh: \\[\\{'pid': 233"):
            MH.stopped_check(MH.cloud_processes(procs))                     # the driver alone still refuses
        shutil.rmtree(procs / "233")
        ok = MH.stopped_check(MH.cloud_processes(procs))                    # only make_handoff and a shell left
        self.assertEqual((ok["run_stage2_py"], ok["explore_policy_sh"]), ([], []))

    def test_stop_rule_time_allows_the_checked_clock_tolerance_only(self):
        from datetime import datetime, timedelta
        rec, paste = datetime(2026, 10, 14, 10, 0, 0), datetime(2026, 10, 14, 11, 0, 0)
        self.assertTrue(MH.chosen_in_time(rec - timedelta(seconds=1), rec, paste))
        self.assertTrue(MH.chosen_in_time(rec + timedelta(seconds=60), rec, paste))     # within 120 s: the laptop's
        self.assertFalse(MH.chosen_in_time(rec + timedelta(seconds=300), rec, paste))   # clock may be that far behind
        self.assertFalse(MH.chosen_in_time(rec + timedelta(seconds=60), rec, rec + timedelta(seconds=30)))

    def test_a_p3_failure_in_any_pasted_full_selftest_ends_v7(self):
        """Fourth review: an earlier READY import cannot carry a handoff once a full self-test with P3 failed was
        pasted (the refusal is recorded); pasting the earlier block again does not help."""
        fx = GitFixture(self.tmp / "fx")
        v7 = fx.v7
        shutil.rmtree(v7)
        complete_v7_tree(v7)
        man = WM.main([], v7_root=v7, log=lambda *a: None)
        body, dsha = WP.write_draft(v7, fx.r7, log=lambda *a: None)
        inputs = dict(constants_sha256=K.CONSTANTS_SHA256, gate_prereg_sha256=K.GATE_PREREG_SHA256,
                      v6_code_manifest_sha256=K.V6["code_manifest_sha256"],
                      probe_script_sha256=man["files"]["code/v7_probes.py"])
        laptop = self.tmp / "laptop_r7"
        write_probe_records(laptop, block=K.block_sha256(body["remake"]), inputs=inputs)
        records = {pid: PR.record_path(laptop, pid).read_bytes() for pid in PR.all_probe_ids()}
        cfg, say = fx.cfg(), []
        ready = dict(verdict="READY (waiting for handoff/approval)", form="full", probe_lines=[], utc=utc_in(),
                     p3=dict(ok=True, distinct=1), prereg_v7_sha256=dsha)
        paste = self.tmp / "ready.txt"
        paste.write_text("\n".join(ST.encode_block(ready, records)))
        MH.import_selftest(cfg, paste, say.append)
        MH.ready(cfg, say.append)
        failed = self.tmp / "p3.txt"
        failed.write_text("\n".join(ST.encode_block(dict(ready, verdict="NOT READY (P3: results depend on lane or "
                                                                         "load)", p3=dict(ok=False, distinct=2)),
                                                  records)))
        with self.assertRaisesRegex(K.V7Refusal, "not from a READY full self-test"):
            MH.import_selftest(cfg, failed, say.append)
        MH.import_selftest(cfg, paste, say.append)                           # the earlier READY block again
        with self.assertRaisesRegex(K.V7Refusal, "failed the repeatability test P3: v7 ends"):
            MH.ready(cfg, say.append)
        args = type("A", (), dict(snapshot="0" * 40, stop_rule="stop at once", stop_rule_chosen="default",
                                  go_ahead="2026-10-14T13:00:00Z", stop="2026-10-14T13:05:00Z",
                                  routines=str(self.tmp / "none.json"), processes=str(self.tmp / "none.json")))()
        with self.assertRaisesRegex(K.V7Refusal, "failed the repeatability test P3"):
            MH.write_handoff(cfg, args, say.append)
        self.assertFalse((v7 / "handoff" / "HANDOFF.json").exists())


class ShellScriptTests(unittest.TestCase):
    def test_scripts_parse_and_run_v7_shows_refusals_and_typeable_commands(self):
        """Fourth review: run_v7.sh keeps the self-test's 'refused:', WAIT and HALT lines (a lock refusal, the reason
        it waits) and prints commands in full."""
        for f in sorted((V7 / "laptop").glob("*.sh")) + [LIB / "common_v7.sh"]:
            self.assertEqual(sh(["sh", "-n", f], check=False).returncode, 0, f)
        text = (V7 / "laptop" / "run_v7.sh").read_text()
        pattern = re.search(r"grep -E '(\^\([^']*)'", text).group(1)
        sample = ("INFO ST9 clock: fine\nrefused: another self-test is running (R7/selftest/selftest.lock); wait\n"
                  "WAIT approval: no import record at x\nV7DATA _summary 0 H4sI\nREADY (waiting for handoff/approval)\n")
        kept = sh(["grep", "-E", pattern], input=sample, check=False).stdout.splitlines()
        self.assertEqual(kept, ["refused: another self-test is running (R7/selftest/selftest.lock); wait",
                                "WAIT approval: no import record at x", "READY (waiting for handoff/approval)"])
        self.assertNotIn("run_v7.sh --retry <phase> continues", text)
        self.assertIn("type sh $PKG/run_v7.sh --retry <phase>", text)
        self.assertIn("sh $PKG/run_v7.sh --lanes $LANES --reviewed <phase>", text)
        self.assertIn('review "$PH"', text)                              # --reviewed is recorded before any restart


# ============================================================================================ C10: run files, return

class ReturnTests(Tmp):
    def build(self, import_first=False, verified=True):
        """A whole synthetic return: restore, a laptop run of seed 2 (stub phases), results, the user's commits.
        import_first: the import record exists before the run, with a time after every attempt (a skewed clock);
        verified: the stub wrapper's checks name it (as the real wrapper's verify_v7_chain does)."""
        fx = GitFixture(self.tmp / "fx")
        cfg = fx.cfg()
        tasks = RN.build_plan("weighted", True, seeds=(2,), pooled=False)
        states, partial = ST.snapshot_states(cfg, fx.commit, ST.parse_sums(git(fx.repo, "show", f"{fx.commit}:"
                                                                                  f"{ST.SUMS_NAME}").stdout.encode()))
        fx.handoff(states, partial=partial, snapshot=dict(commit=fx.commit, sums_sha256=fx.sums_sha))
        write_probe_records(fx.r7)
        pins = probe_pins(fx.r7)
        fx.handoff(states, partial=partial, snapshot=dict(commit=fx.commit, sums_sha256=fx.sums_sha),
                   extra=dict(probe_records=pins))
        prefix = K.prefix_of(K.file_sha256(fx.v7 / "prereg_v7.json"))
        if import_first:
            isha = write_json_sidecar(fx.v7 / f"import_v7_{prefix}.json",
                                      dict(kind=K.IMPORT_KIND, prereg_sha256=K.file_sha256(fx.v7 / "prereg_v7.json"),
                                           imported_at="2999-01-01T00:00:00Z"))
            if verified:
                (fx.root / "checks.json").write_text(json.dumps(dict(stub=True, v7_chain=dict(import_sha256=isha))))
        ST.restore(fx.cfg(), fetch=False, out=lambda *a: None)
        fx.behaviour(**{"*": "ok"})
        code, lines = fx.run(lanes=3)
        self.assertEqual(code, 0, "\n".join(lines))
        self.assertTrue(all(t.pid in states for t in tasks))
        MR.make_results(fx.cfg(), out=lambda *a: None)
        if not import_first:
            write_json_sidecar(fx.v7 / f"import_v7_{prefix}.json",
                               dict(kind=K.IMPORT_KIND, prereg_sha256=K.file_sha256(fx.v7 / "prereg_v7.json"),
                                    imported_at="2000-01-01T00:00:00Z"))
        git(fx.repo, "add", "-f", "v7/prereg_v7.json", "v7/prereg_v7.json.sha256", "v7/handoff",
            f"v7/import_v7_{prefix}.json", f"v7/import_v7_{prefix}.json.sha256", "v7/code_manifest_v7.json",
            "v7/code_manifest_v7.json.sha256", "v7/code")
        git(fx.repo, "commit", "-q", "-m", "agent: prereg, handoff, import", ident=AGENT)
        git(fx.repo, "add", "-f", "v7/results")
        git(fx.repo, "commit", "-q", "-m", "user: results", ident=APPROVER)
        return fx

    def push(self, fx, stdin=None):
        return sh(["sh", V7 / "laptop" / "push_runfiles_v7.sh", "--v7-test", fx.root], check=False, input=stdin)

    def check(self, fx):
        lines = []
        code = CR.main(["--v7-test", str(fx.root), "--main-ref", "main", "--orphan-ref",
                        "refs/remotes/origin/stage2-v7-run-files"], out=lines.append)
        return code, "\n".join(lines)

    def test_push_and_return_check(self):
        fx = self.build()
        hook = fx.origin / "hooks" / "pre-receive"                    # the remote refuses the first push only
        marker = self.tmp / "refused_once"
        hook.write_text(f"#!/bin/sh\nif [ ! -e '{marker}' ]; then touch '{marker}'; echo 'refused once' >&2; exit 1; "
                        f"fi\nexit 0\n")
        hook.chmod(0o755)
        done = self.push(fx)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("the push failed (try 1 of 5); trying again in 1 seconds", done.stdout)
        self.assertTrue(marker.exists())
        git(fx.repo, "fetch", "-q", "origin", "+refs/heads/stage2-v7-run-files:refs/remotes/origin/stage2-v7-run-files")
        files = git(fx.repo, "ls-tree", "-r", "--name-only", "refs/remotes/origin/stage2-v7-run-files").stdout.split()
        self.assertIn(f"{OUT_REL}/policy/seed2/actors.msgpack", files)
        self.assertIn(f"{R7_REL}/refs/seed2/check2_stall.msgpack", files)
        self.assertNotIn(f"{OUT_REL}/policy/seed0/lock.json", files)                  # inherited: not repeated
        self.assertFalse([f for f in files if f.endswith(".lock")])
        log = git(fx.repo, "log", "--format=%an <%ae>|%cn <%ce>", "refs/remotes/origin/stage2-v7-run-files").stdout
        who = f"{APPROVER[0]} <{APPROVER[1]}>"
        self.assertEqual(log.splitlines(), [f"{who}|{who}"] * 2)
        again = self.push(fx)
        self.assertIn("already pushed", again.stdout, again.stdout + again.stderr)   # the same commits again
        code, text = self.check(fx)
        self.assertEqual(code, 0, text)
        self.assertTrue(json.loads((fx.v7 / "return_check.json").read_text())["passed"])
        # tampering: a result changed in the repository's results commit
        lock = fx.v7 / "results" / "files" / ("runs__shift_explore__stage2__policy__seed2__lock.json")
        lock.write_text("changed")
        git(fx.repo, "commit", "-q", "-am", "user: change", ident=APPROVER)
        code, text = self.check(fx)
        self.assertEqual(code, 1)
        self.assertIn("FAIL every result file has its listed SHA-256", text)
        git(fx.repo, "reset", "-q", "--hard", "HEAD~1")
        # an agent commit touching results/
        (fx.v7 / "results" / "note.txt").write_text("x")
        git(fx.repo, "add", "-f", "v7/results/note.txt")
        git(fx.repo, "commit", "-q", "-m", "agent touches results", ident=AGENT)
        code, text = self.check(fx)
        self.assertIn("FAIL every commit touching results/", text)
        git(fx.repo, "reset", "-q", "--hard", "HEAD~1")
        # the cloud's own copy changed after the snapshot
        (fx.cloud_out / "policy/seed9").mkdir(parents=True)
        (fx.cloud_out / "policy/seed9/lock.json").write_text("{}")
        code, text = self.check(fx)
        self.assertIn("FAIL the cloud's own copy of OUT/", text)
        shutil.rmtree(fx.cloud_out / "policy/seed9")
        code, text = self.check(fx)
        self.assertEqual(code, 0, text)

    def test_return_check_orders_by_the_verified_import_not_the_clock(self):
        """Fourth review: the laptop's clock may lag the cloud's; an attempt whose wrapper verified the import record
        passes whatever the times say, and one that did not (a test fixture compares times) fails."""
        fx = self.build(import_first=True)
        self.assertEqual(self.push(fx).returncode, 0)
        git(fx.repo, "fetch", "-q", "origin", "+refs/heads/stage2-v7-run-files:refs/remotes/origin/stage2-v7-run-files")
        code, text = self.check(fx)
        self.assertEqual(code, 0, text)
        self.assertIn("by an attempt whose wrapper verified the v7 import record", text)
        shutil.rmtree(self.tmp / "fx")
        fx = self.build(import_first=True, verified=False)
        self.assertEqual(self.push(fx).returncode, 0)
        git(fx.repo, "fetch", "-q", "origin", "+refs/heads/stage2-v7-run-files:refs/remotes/origin/stage2-v7-run-files")
        code, text = self.check(fx)
        self.assertEqual(code, 1)
        self.assertIn("FAIL every new file under OUT/ is named", text)

    def test_bundle_and_provenance_name_an_unfinished_phase(self):
        fx = GitFixture(self.tmp / "fx")
        cfg = fx.cfg()
        states, partial = ST.snapshot_states(cfg, fx.commit, ST.parse_sums(git(fx.repo, "show", f"{fx.commit}:"
                                                                                  f"{ST.SUMS_NAME}").stdout.encode()))
        fx.handoff(states, partial=partial, snapshot=dict(commit=fx.commit, sums_sha256=fx.sums_sha))
        ST.restore(fx.cfg(), fetch=False, out=lambda *a: None)
        fx.behaviour(host_train_seed2_no_stall=["kill"] * 4, **{"*": "ok"})
        code, lines = fx.run(lanes=2)
        self.assertEqual(code, 4, "\n".join(lines))
        self.assertIn("unfinished: host_train_seed2_no_stall", lines[-1])               # the runner's last line
        with self.assertRaisesRegex(K.V7Refusal, "neither done nor unfinished"):
            MR.make_results(fx.cfg(), out=lambda *a: None)              # its dependents are not done: --partial
        b = MR.make_results(fx.cfg(), partial=True, out=lambda *a: None)
        self.assertEqual(b["unfinished"], ["host_train_seed2_no_stall"])
        self.assertTrue(b["partial"])
        self.assertIn("unfinished: host_train_seed2_no_stall", b["status"])
        prov = json.loads((fx.r7 / "provenance_exploratory.json").read_text())
        self.assertEqual(prov["unfinished"], ["host_train_seed2_no_stall"])
        self.assertEqual(b["phases"]["host_eval_seed2_no_stall"], "not done")

    def test_push_refusals(self):
        fx = self.build()
        done = sh(["sh", V7 / "laptop" / "push_runfiles_v7.sh"], check=False, input="")
        self.assertIn("refused: run this yourself at a terminal", done.stdout)
        stray = fx.out / "policy" / "seed2" / "stray.json"
        stray.write_text("{}")
        done = self.push(fx)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("no v7 ledger attempt wrote", done.stdout)
        stray.unlink()
        (fx.out / "policy/seed0/lock.json").write_text("changed")
        done = self.push(fx)
        self.assertIn("inherited files changed", done.stdout)
        self.assertEqual(git(fx.repo, "ls-remote", str(fx.origin), "refs/heads/stage2-v7-run-files").stdout, "")
        cfg = fx.cfg()
        plan = self.tmp / "plan"
        (fx.out / "policy/seed0/lock.json").write_bytes(fx.all_snapshot_bytes()[f"{OUT_REL}/policy/seed0/lock.json"])
        MR.runfiles_plan(cfg, plan)
        env = dict(GIT_INDEX_FILE=str(self.tmp / "idx"))
        sh(["git", "-C", fx.repo, "read-tree", "--empty"], env=env)
        sh(["git", "-C", fx.repo, "update-index", "--add", "README"], env=env)        # a path outside the plan
        tree = sh(["git", "-C", fx.repo, "write-tree"], env=env).stdout.strip()
        with self.assertRaisesRegex(K.V7Refusal, "outside the plan"):
            MR.runfiles_check(cfg, tree, plan)

    def test_return_check_catches_unnamed_out_file_and_bad_constants(self):
        fx = self.build()
        # an extra attempt killed by a restart after the wrapper's checks, while JAX loaded (no constants facts)
        n, w = LG.allocate_attempt(fx.r7, "deploy_seed2_stall")
        LG.append_event(w, "launch-start", key=dict(phase="deploy", seed=2, world="stall"), boot_id="old boot",
                        r7=str(fx.r7), out_root=str(fx.out), argv=["--phase", "deploy", "--seed", "2"])
        LG.append_event(w, "wrapper-start", argv=[])
        LG.append_event(w, "checks", result=dict(stub=True), fingerprints=dict(stub=True))
        rec = LG.finish_attempt(w, out_root=self.tmp / "not_done_then", r7=fx.r7, now_boot="new boot")
        self.assertEqual((rec["classification"]["kind"], rec["constants"], rec["handover"]), ("machine-wide", None, False))
        MR.make_results(fx.cfg(), out=lambda *a: None)
        git(fx.repo, "add", "-f", "v7/results")
        git(fx.repo, "commit", "-q", "-m", "user: results again", ident=APPROVER)
        self.assertEqual(self.push(fx).returncode, 0)
        git(fx.repo, "fetch", "-q", "origin", "+refs/heads/stage2-v7-run-files:refs/remotes/origin/stage2-v7-run-files")
        self.assertEqual(self.check(fx)[0], 0)
        env = dict(GIT_INDEX_FILE=str(self.tmp / "idx2"))
        orphan = "refs/remotes/origin/stage2-v7-run-files"
        sh(["git", "-C", fx.repo, "read-tree", orphan], env=env)
        blob = sh(["git", "-C", fx.repo, "hash-object", "-w", "--stdin"], input="sneaked").stdout.strip()
        sh(["git", "-C", fx.repo, "update-index", "--add", "--cacheinfo",
            f"100644,{blob},{OUT_REL}/policy/seed2/extra.json"], env=env)
        tree = sh(["git", "-C", fx.repo, "write-tree"], env=env).stdout.strip()
        c = sh(["git", "-C", fx.repo, "commit-tree", tree, "-p", orphan, "-m", "x"], ident=APPROVER).stdout.strip()
        git(fx.repo, "update-ref", orphan, c)
        code, text = self.check(fx)
        self.assertEqual(code, 1)
        self.assertIn("FAIL every new file under OUT/ is named", text)
        shutil.rmtree(self.tmp / "fx")
        fx = self.build()
        rec_path = sorted((fx.r7 / "ledger" / "deploy_seed2_stall").glob("*.json"))[0]
        rec = json.loads(rec_path.read_text())
        rec["constants"]["differences"]["max_abs_dK"] = 1e-9
        rec_path.write_text(K.canonical(rec))
        Path(str(rec_path) + ".sha256").write_text(f"{K.sha256_bytes(K.canonical(rec).encode())}  {rec_path.name}\n")
        self.assertEqual(self.push(fx).returncode, 0)
        git(fx.repo, "fetch", "-q", "origin", "+refs/heads/stage2-v7-run-files:refs/remotes/origin/stage2-v7-run-files")
        code, text = self.check(fx)
        self.assertEqual(code, 1)
        self.assertIn("FAIL the constants facts hold", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
