"""Stage 2, version 7: the lane scheduler that runs every remaining Stage 2 phase on the laptop (V7_SPEC.md C2, C5,
C6, C7, C8). run_v7.sh starts it after the placement of the v6 code and the short self-test; never run it on its own.

What it runs (the plan; C7, C8). Each phase is one process started through V7/code/launch_v7.sh (one processor,
the exact eight-variable environment, the v7 wrapper, then the pinned v6 run_stage2.py unchanged):
  seeds 0-4   policy_train, policy_eval, policy_no_stall_train, policy_no_stall_eval (when a version passed on seeds
              0-4 and the room test passed), then deploy (both worlds), and the full host (stalling world, then the
              no-stall world); then the exploratory pooled look, once every seed 0-4 phase is done or skipped;
  seeds 5-9   the plan of GPU/lib/runner.py:89-197: fresh_data, fresh_learners, the detection steps, the policy test,
              deploy, the full host; then the confirmatory detection marks and the confirmatory pooled look.
Phases that HANDOFF.json lists as inherited from the cloud never run here. A phase is done when its final record or a
failed-on-record file of v7's phase table (v7_ledger.row) is present and matches its sidecar, whatever its exit code.
A phase that is done on file and not inherited must have been made here: at start-up and whenever a phase is found or
ends done, an attempt classified 'done' in the ledger must name each file that makes it done with its current SHA-256,
and, for a phase with an entry in the re-make table, its lock must record the re-made actors; otherwise the run halts.
Before that, every attempt in R7/ledger/ without a record (its launch shell did not survive), done phases included, has
its record written once no process of it runs (settle_all).
Before anything starts, preflight() checks the final prereg_v7.json, the v7 code, the launcher, HANDOFF.json, both
approval chains, the last restore, and the nine probe records and every reference file against their pins in the
final prereg_v7.json (C7 'Probe records'; a mismatch is a HALT).
Records are read through v7_common.read_record, so a torn sidecar never stops the runner (it is reported as partial).
A seed that failed on record is never replaced: the phases the pinned code would refuse for it are skipped, by the
rules of fresh.recorded_failure and run_stage2.recorded_seed_failure, restated here on the same files.
Seeds 0-4 and 5-9 are scheduled together, longest remaining chain first.

Lanes and memory (C2). --lanes 1-16 (default 12), each lane one processor: the first N of the 16 processors the
self-test checks with the gate (ST4, ST7), so no lane runs on an unchecked processor. A phase starts only if (i) the reserved peaks
of the running phases plus its own are at most MemTotal minus 8 GiB, and (ii) MemAvailable is at least its own
reservation plus 8 GiB. A phase kind not yet measured on the laptop reserves 7 GiB; a measured kind 1.25 times its
largest VmHWM (from the ledger); policy_train and policy_no_stall_train the larger of 7 GiB and twice their peak. At most
2 policy_train run at once until one has been measured. When a new policy lock records standin.error or
standin.no_stall_error, no new policy_train or policy_no_stall_train starts until the user's review is recorded
('run_v7.sh --reviewed <phase>': from a second window while the runner runs, which picks it up within its retry poll,
or as a restart); every other phase continues. Typing the same review again only says it exists.

Interruptions (C5). Each attempt's ledger record (v7_ledger.finish_attempt) classifies how it ended. Machine-wide
interruptions (a changed boot id, a signal the runner sent, SIGHUP/SIGINT/SIGTERM, no exit status) rerun without
limit and are never counted. After the first and second phase-specific interruption (SIGKILL, SIGSEGV, SIGBUS or
SIGABRT on the same boot, or a log ending in a memory error) the phase reruns as usual; after the third it waits until
no other runnable phase remains and then runs alone; if that attempt is also interrupted phase-specifically the phase
is UNFINISHED: every phase that does not need it continues, and it waits for the user's 'run_v7.sh --retry <phase>'
(each request allows exactly one attempt, run alone; a machine-wide interruption of it reruns automatically). Before
a rerun every existing file of the phase's table row is moved, never deleted, to R7/interrupted/<phase>/<UTC time>/
(v7_ledger.move_interrupted). Exit 97 (a v7 refusal) halts the run; any other non-zero exit, or exit 0 without the
phase being done, halts new phases and names the log (as GPU/ does).

Ctrl-C once: no new phase starts and the running ones finish. A second Ctrl-C stops the running phases (each signal is
first written to the attempt's ledger, so the interruption is machine-wide and the phase reruns on the next start).

Files: R7/runner.lock (one runner per machine), R7/status.json, R7/runner.log, R7/retry/<phase>/<n>.json (requests,
written by 'retry') and <n>.attempt_<m>.json (which attempt used a request), R7/reviewed/<phase>.json ('review' or
run --reviewed).

Command line (run_v7.sh passes these):
  runner_v7.py run [--lanes N] [--cpus A,B,...] [--reviewed PHASE ...] [--plan-only]   (--reviewed: typed at a terminal)
  runner_v7.py retry PHASE        (standard input must be a terminal; the phase must be unfinished)
  runner_v7.py review PHASE       (standard input must be a terminal; records R7/reviewed/PHASE.json once)
  runner_v7.py status
'--v7-test <fixture>' before the command is for test_v7_laptop.py only (a synthetic fixture outside /home/user/bca).
"""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
LIB = Path(__file__).resolve().parent
LAPTOP = LIB.parent
V7_DIR = LAPTOP.parent
CODE_DIR = V7_DIR / "code"
for _p in (str(LIB), str(CODE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import v7_common as K  # noqa: E402
import v7_ledger as LG  # noqa: E402

GIB = 1024 ** 3
UNMEASURED_GIB = 7
HEADROOM_GIB = 8
MEASURED_FACTOR = 1.25
POLICY_KINDS = ("policy_train", "policy_no_stall_train")
POLICY_TRAIN_CAP = 2
MAX_LANES, DEFAULT_LANES = 16, 12
WORLDS = ("stall", "no_stall")
EXPLORATORY, CONFIRMATORY = LG.EXPLORATORY, LG.CONFIRMATORY
FINAL_STATES = ("done", "inherited", "skipped")
INTERRUPTED = ("machine-wide", "phase-specific")
BASE_ENV = {"PATH": K.ENV_PATH, "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
TEST_MARKER, TEST_MARKER_TEXT = "V7_TEST_FIXTURE", "v7 test fixture: synthetic files only\n"
# Rough minutes per phase at the cloud's per-core speed (V7_SPEC.md section 2 and C2); used only to order the queue.
EST_MINUTES = dict(fresh_data=8, fresh_learners=40, detect_pairs=5, detect_labels=5, detect_standins=5, detect_fit=6,
                   detect_fit_weighted=8, detect_score=3, detect_marks=2, policy_train=52, policy_eval=3,
                   policy_no_stall_train=10, policy_no_stall_eval=3, deploy=4, host_train=75, host_eval=1, pooled=2)


# --------------------------------------------------------------------------------------------- configuration

class Cfg:
    """Where the runner, the self-test and the results step read and write. Production values are fixed; a test
    fixture (a folder outside /home/user/bca holding the marker file V7_TEST_FIXTURE and runner.json) may move them."""

    def __init__(self, **kw):
        self.test = bool(kw.pop("test", False))
        self.fixture = kw.pop("fixture", None)
        self.root = Path(kw.pop("root", K.ROOT))
        self.r7 = Path(kw.pop("r7", K.R7))
        self.out = Path(kw.pop("out", K.OUT))
        self.v7_root = Path(kw.pop("v7_root", K.V7))
        self.prereg = Path(kw.pop("prereg", self.v7_root / "prereg_v7.json"))
        self.s_folder = Path(kw.pop("s_folder", K.S_FOLDER))
        self.launcher = Path(kw.pop("launcher", CODE_DIR / "launch_v7.sh"))
        self.launcher_args = [str(a) for a in kw.pop("launcher_args", [])]
        self.meminfo = Path(kw.pop("meminfo", "/proc/meminfo"))
        self.version = kw.pop("version", None)
        self.training = kw.pop("training", None)
        self.check_chain = bool(kw.pop("check_chain", True))
        self.poll = float(kw.pop("poll", 2.0))
        self.retry_poll = float(kw.pop("retry_poll", 30.0))
        self.cpus = kw.pop("cpus", None)
        self.seeds = tuple(kw.pop("seeds", EXPLORATORY + CONFIRMATORY))
        self.pooled = bool(kw.pop("pooled", True))
        self.stop_grace = float(kw.pop("stop_grace", 60.0))
        self.extra = kw

    @property
    def handoff(self):
        return self.v7_root / "handoff" / "HANDOFF.json"

    # Test mode: every one of these must be given and must resolve inside the fixture (nothing falls back to a
    # production path); the optional path settings, when given, too.
    FIXTURE_PATHS = ("root", "r7", "out", "s_folder", "v7_root", "prereg")
    FIXTURE_OPTIONAL_PATHS = ("meminfo", "probe_script", "fallback_target", "gate_variant_dir", "cloud_out",
                              "proc_root")

    @classmethod
    def from_fixture(cls, fixture):
        fixture = Path(os.path.realpath(fixture))
        marker = fixture / TEST_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf8") != TEST_MARKER_TEXT:
            raise K.V7Refusal(f"{fixture} is not a v7 test fixture (no {TEST_MARKER} marker)")
        for protected in (K.ROOT, K.R7, K.OUT, K.S_FOLDER, K.SCRATCH / "ood"):
            prot = Path(os.path.realpath(protected))
            if K.inside(fixture, prot) or K.inside(prot, fixture):
                raise K.V7Refusal(f"a test fixture may not lie inside or around {protected}")
        body = json.loads((fixture / "runner.json").read_text(encoding="utf8"))
        missing = [k for k in cls.FIXTURE_PATHS if not body.get(k)]
        if missing:
            raise K.V7Refusal(f"test mode needs every path setting given explicitly; missing {missing}")
        for k in cls.FIXTURE_PATHS + cls.FIXTURE_OPTIONAL_PATHS:
            if k in body and body[k] is not None and not K.inside(os.path.realpath(body[k]), fixture):
                raise K.V7Refusal(f"test setting {k} must lie inside the fixture {fixture}")
        launcher = body.get("launcher")
        if launcher is None:
            raise K.V7Refusal("test mode needs the launcher given explicitly")
        if not K.inside(os.path.realpath(launcher), fixture):
            real = os.path.realpath(launcher) == os.path.realpath(CODE_DIR / "launch_v7.sh")
            args = [str(a) for a in body.get("launcher_args") or []]
            if not (real and args[:1] == ["--v7-test"] and len(args) >= 2 and
                    Path(os.path.realpath(args[1])) == fixture):
                raise K.V7Refusal("test mode: the launcher must lie inside the fixture, or be the real launch_v7.sh "
                                  "with '--v7-test <this fixture>'")
        if body.get("version") is None and "training" not in body:
            raise K.V7Refusal("test mode gives the run conditions (version, training) directly; it never imports the "
                              "pinned v6 code")
        return cls(test=True, fixture=fixture, check_chain=False, **body)


def config(fixture=None):
    return Cfg.from_fixture(fixture) if fixture else Cfg()


# --------------------------------------------------------------------------------------------- small helpers

def utc():
    return K.utc_now()


def say_line(line):
    print(line, flush=True)


def log_name(args):
    """The log file launch_v7.sh writes for these arguments (its rule, as launch.sh's: drop '--', spaces to '_')."""
    return "_".join(" ".join(args).replace("--", "").split()) + ".log"


def lock_free(r7, pid):
    """True when no process holds the phase lock R7/locks/<phase id>.lock (launch_v7.sh and its phase hold it)."""
    p = Path(r7) / "locks" / f"{pid}.lock"
    if not p.exists():
        return True
    try:
        fd = os.open(str(p), os.O_RDONLY)
    except OSError:
        return True
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def held_phase_locks(r7):
    """The phases whose lock some process holds now (a phase from an earlier run still running)."""
    d = Path(r7) / "locks"
    return sorted(p.stem for p in d.glob("*.lock") if not lock_free(r7, p.stem)) if d.is_dir() else []


def runner_running(r7):
    p = Path(r7) / "runner.lock"
    if not p.exists():
        return False
    try:
        fd = os.open(str(p), os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def lane_cpus(n, sysfs="/sys/devices/system/cpu", affinity=None):
    """The processors of n lanes: always the first n of pick_cpus(16), the 16 processors the self-test checks with the
    gate on every start (ST4, ST7), so that no lane runs on a processor the self-test did not check."""
    return pick_cpus(MAX_LANES, sysfs, affinity)[:n]


def pick_cpus(n, sysfs="/sys/devices/system/cpu", affinity=None):
    """n processors on different physical cores when the topology says so, else the first n (as GPU/lib/runner.py).
    sysfs and affinity (a set of processor numbers) are for the tests; by default the machine's own."""
    allowed = set(os.sched_getaffinity(0) if affinity is None else affinity)
    groups, seen = [], set()
    for d in sorted(Path(sysfs).glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cpu = int(d.name[3:])
        if cpu not in allowed:
            continue
        sib = d / "topology" / "thread_siblings_list"
        key = sib.read_text().strip() if sib.exists() else str(cpu)
        if key not in seen:
            seen.add(key)
            groups.append(cpu)
    return groups[:n] if len(groups) >= n else sorted(allowed)[:n]


def checked_cpus(cfg):
    """The processors the self-test checks (ST4, ST7) and lanes may use: pick_cpus(16) on the machine; a test
    fixture's own list (runner.json 'cpus')."""
    return list(cfg.cpus) if cfg.cpus else pick_cpus(MAX_LANES)


def lanes_for(cfg, lanes, given=None):
    """(processors, refusal): the first 'lanes' of checked_cpus(cfg), or the given ones (--cpus), which must all be
    among them (C2: no lane on a processor the self-test did not check)."""
    allowed = checked_cpus(cfg)
    cpus = [int(c) for c in given] if given else allowed[:lanes]
    if len(cpus) < lanes or len(set(cpus)) != len(cpus):
        return None, f"give {lanes} different processor numbers"
    outside = sorted(set(cpus[:lanes]) - set(allowed))
    if outside:
        return None, (f"processors {outside} are not among the {len(allowed)} the self-test checks ({allowed}); "
                      f"lanes use only those")
    return cpus[:lanes], None


def meminfo(path="/proc/meminfo"):
    """(MemTotal, MemAvailable) in bytes."""
    vals = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
            vals[parts[0][:-1]] = int(parts[1]) * 1024
    return vals.get("MemTotal", 0), vals.get("MemAvailable", 0)


# --------------------------------------------------------------------------------------------- the handoff

def read_handoff(cfg):
    """HANDOFF.json's payload (verified sidecar) or None."""
    state, body = K.read_record(cfg.handoff)
    return body if state == "ok" else None


def handoff_phases(cfg):
    body = read_handoff(cfg)
    return dict((body or {}).get("phases") or {})


# --------------------------------------------------------------------------------------------- the plan

class Task:
    def __init__(self, args, deps, group):
        self.args = [str(a) for a in args]
        self.key = LG.check_phase_allowed(LG.phase_key(LG.parse_phase_args(self.args)))
        self.pid = LG.phase_id(self.key)
        self.kind = self.key["phase"]
        self.seed = self.key.get("seed")
        self.deps, self.group = list(deps), group
        est = EST_MINUTES[self.kind]
        if self.kind == "host_train" and self.key.get("world") == "no_stall":
            est = 30
        self.est, self.prio = est, est
        self.state, self.note = "pending", None
        self.cpu = self.proc = self.started = self.finished = self.code = None
        self.before = 0
        self.alone, self.retry_n, self.tagged, self.reserved = False, None, False, 0

    def row(self):
        return dict(pid=self.pid, phase=self.kind, args=self.args, deps=self.deps, state=self.state, note=self.note,
                    cpu=self.cpu, started=self.started, finished=self.finished, exit_code=self.code,
                    alone=self.alone, retry_request=self.retry_n, reserved_gib=round(self.reserved / GIB, 2),
                    log=log_name(self.args))


def build_plan(version, training, seeds=EXPLORATORY + CONFIRMATORY, pooled=True):
    """The task list (C7, C8). version: the frozen version ('uniform', 'weighted' or None); training: whether the policy
    test and the full host run (a version passed and the room test passed)."""
    Vd = version or "uniform"
    tasks = []

    def add(args, deps, group):
        t = Task(args, deps, group)
        tasks.append(t)
        return t.pid

    for s in [x for x in EXPLORATORY if x in seeds]:
        chain = []
        if training:
            t1 = add(["--phase", "policy_train", "--seed", s], [], "exploratory")
            t2 = add(["--phase", "policy_eval", "--seed", s], [t1], "exploratory")
            t3 = add(["--phase", "policy_no_stall_train", "--seed", s], [t2], "exploratory")
            t4 = add(["--phase", "policy_no_stall_eval", "--seed", s], [t3], "exploratory")
            chain = [t4]
        for W in WORLDS:
            add(["--phase", "deploy", "--seed", s, "--world", W], chain, "exploratory")
        if training:
            h1 = add(["--phase", "host_train", "--seed", s, "--world", "stall"], chain, "exploratory")
            h2 = add(["--phase", "host_eval", "--seed", s, "--world", "stall"], [h1], "exploratory")
            h3 = add(["--phase", "host_train", "--seed", s, "--world", "no_stall"], [h2], "exploratory")
            add(["--phase", "host_eval", "--seed", s, "--world", "no_stall"], [h3], "exploratory")
    explo = [t.pid for t in tasks]
    if pooled and explo:
        add(["--phase", "pooled", "--group", "exploratory"], explo, "exploratory")
    first_conf = len(tasks)
    for s in [x for x in CONFIRMATORY if x in seeds]:
        t_data = add(["--phase", "fresh_data", "--seed", s], [], "confirmatory")
        t_learn = add(["--phase", "fresh_learners", "--seed", s], [t_data], "confirmatory")
        no_bca = []
        for W in WORLDS:
            t_p = add(["--phase", "detect_pairs", "--seed", s, "--world", W], [t_learn], "confirmatory")
            t_l = add(["--phase", "detect_labels", "--seed", s, "--world", W], [t_p], "confirmatory")
            no_bca.append(add(["--phase", "detect_standins", "--seed", s, "--world", W], [t_l], "confirmatory"))
        fit = {}
        for W, C in (("stall", "fixed"), ("stall", "host"), ("no_stall", "fixed")):
            fit[(W, C, "uniform")] = add(["--phase", "detect_fit", "--seed", s, "--world", W, "--critic", C], no_bca,
                                         "confirmatory")
        if Vd == "weighted":
            for W in WORLDS:
                fit[(W, "fixed", "weighted")] = add(["--phase", "detect_fit_weighted", "--seed", s, "--world", W,
                                                     "--critic", "fixed"], [fit[(W, "fixed", "uniform")]],
                                                    "confirmatory")
        scores = []
        for W, C, v in (("stall", "fixed", Vd), ("stall", "host", "uniform"), ("no_stall", "fixed", Vd)):
            scores.append(add(["--phase", "detect_score", "--seed", s, "--world", W, "--critic", C, "--version", v],
                              [fit[(W, C, v)]], "confirmatory"))
        chain = list(scores)
        if training:
            t1 = add(["--phase", "policy_train", "--seed", s], chain, "confirmatory")
            t2 = add(["--phase", "policy_eval", "--seed", s], [t1], "confirmatory")
            t3 = add(["--phase", "policy_no_stall_train", "--seed", s], [t2], "confirmatory")
            t4 = add(["--phase", "policy_no_stall_eval", "--seed", s], [t3], "confirmatory")
            chain = [t4]
        for W in WORLDS:
            add(["--phase", "deploy", "--seed", s, "--world", W], list(chain), "confirmatory")
        if training:
            h1 = add(["--phase", "host_train", "--seed", s, "--world", "stall"], list(chain), "confirmatory")
            h2 = add(["--phase", "host_eval", "--seed", s, "--world", "stall"], [h1], "confirmatory")
            h3 = add(["--phase", "host_train", "--seed", s, "--world", "no_stall"], [h2], "confirmatory")
            add(["--phase", "host_eval", "--seed", s, "--world", "no_stall"], [h3], "confirmatory")
    conf = [t.pid for t in tasks[first_conf:]]
    if pooled and conf:
        m = add(["--phase", "detect_marks", "--group", "confirmatory", "--version", Vd], conf, "confirmatory")
        add(["--phase", "pooled", "--group", "confirmatory"], [m], "confirmatory")
    by = {t.pid: t for t in tasks}
    if len(by) != len(tasks):
        raise K.V7Refusal("the plan holds a phase twice")
    children = {t.pid: [] for t in tasks}
    for t in tasks:
        for d in t.deps:
            children[d].append(t.pid)
    for t in reversed(tasks):                       # dependents always come later in the list
        t.prio = t.est + max((by[c].prio for c in children[t.pid]), default=0)
    return tasks


def run_conditions(cfg):
    """(version, training) by the pinned v6 code's own rules (run_stage2.chosen_version and room_condition), read
    only, from the folder launch.sh runs from; a test fixture gives them directly."""
    if cfg.test:
        if cfg.version is None and cfg.training is None:
            raise K.V7Refusal("test mode gives the run conditions directly; it never imports the pinned v6 code")
        return cfg.version, bool(cfg.training)
    folder = str(cfg.s_folder)
    if folder not in sys.path:
        sys.path.insert(0, folder)
    try:
        import run_stage2 as RS
        version = RS.chosen_version()
        room = RS.room_condition()
    except SystemExit as error:
        raise K.V7Refusal(f"the pinned v6 code cannot decide the run conditions: {error}") from None
    return version, bool(version is not None and room.get("passed"))


# --------------------------------------------------------------------------------------------- records on file

def _rec(path):
    state, body = K.read_record(path)
    return body if state == "ok" else None


def fresh_failed(cfg, s):
    """fresh.recorded_failure restated (S2C/fresh.py:268-281): a fresh seed whose data checks failed in either world,
    or whose learners failed. Missing or partial records are not failures. None for seeds 0-4."""
    if s not in CONFIRMATORY:
        return None
    for w in WORLDS:
        d = _rec(cfg.out / "fresh" / w / f"seed{s}" / "data.json")
        if d is not None and not d.get("passed"):
            return f"fresh seed {s} failed its data checks in the {w} world (never replaced)"
    lr = _rec(cfg.out / "fresh" / "stall" / f"seed{s}" / "learners.json")
    if lr is not None and lr.get("failed"):
        return lr.get("reason") or f"a fresh learner of seed {s} failed (never replaced)"
    return None


def band_failed(cfg, s, world, critic, version):
    for v in (version, "uniform") if version == "weighted" else (version,):
        b = _rec(cfg.out / "detect" / v / world / critic / f"seed{s}" / "band.json")
        if b is not None and b.get("failed"):
            return f"the {v} band fit of seed {s} ({world}, {critic} critic) failed: {b.get('reason')}"
    return None


def seed_world_failed(cfg, s, world, critic, version):
    """run_stage2.recorded_seed_failure restated (S2C/run_stage2.py:389-401)."""
    return fresh_failed(cfg, s) or band_failed(cfg, s, world, critic, version)


def stopped(cfg, s, world="stall"):
    return _rec(cfg.out / "policy" / f"seed{s}" / ("no_stall/" if world == "no_stall" else "") / "stop.json") is not None


def skip_reason(cfg, t, version):
    Vd = version or "uniform"
    k, s = t.kind, t.seed
    if k in ("detect_marks", "pooled") or s is None:
        return None
    why = fresh_failed(cfg, s) if k != "fresh_data" else None
    if why:
        return why
    if k == "detect_fit_weighted":
        return band_failed(cfg, s, t.key["world"], "fixed", "uniform")
    if k == "detect_score":
        return seed_world_failed(cfg, s, t.key["world"], t.key["critic"], t.key["version"])
    if k == "deploy":
        return seed_world_failed(cfg, s, t.key["world"], "fixed", Vd)
    if k == "policy_eval" and stopped(cfg, s):
        return "the policy test stopped this seed (its stop.json is on record); an evaluation would be refused"
    if k in ("policy_no_stall_train", "policy_no_stall_eval") and stopped(cfg, s):
        return "the stalling world's policy test stopped this seed (its stop.json); no no-stall arm runs"
    if k == "policy_no_stall_eval" and stopped(cfg, s, "no_stall"):
        return "the no-stall policy test stopped this seed (its stop.json); an evaluation would be refused"
    return None


# --------------------------------------------------------------------------------------------- attempts (C5)

def attempt_files(r7, pid):
    d = Path(r7) / "ledger" / pid
    out = []
    if d.is_dir():
        for p in d.glob("*.writes"):
            stem = p.name[:-len(".writes")]
            if stem.isdigit():
                out.append((int(stem), p))
    return sorted(out)


def attempt_record(writes):
    return _rec(Path(writes).with_name(Path(writes).name.replace(".writes", ".json")))


def settle(cfg, pid, log_path=None, write=True):
    """Write the record of every attempt whose launch shell did not survive (no record yet), once no process of the
    phase runs (write=False: read only). Returns {attempt number: record or None}."""
    out = {}
    free = write and lock_free(cfg.r7, pid)
    for n, w in attempt_files(cfg.r7, pid):
        rec = attempt_record(w)
        if rec is None and free:
            rec = LG.finish_attempt(w, out_root=cfg.out, log_path=log_path, r7=cfg.r7)
        out[n] = rec
    return out


def log_for_attempt(cfg, writes):
    """The per-phase log an attempt appended to: R7/logs/<log name of its arguments> for a phase (launch_v7.sh's rule),
    R7/selftest/logs/probe_<probe>.log for a probe."""
    events, _ = LG.read_events(writes)
    start = next((e for e in events if e.get("ev") == "launch-start"), {}) or {}
    if start.get("probe"):
        return cfg.r7 / "selftest" / "logs" / f"probe_{start['probe']}.log"
    return cfg.r7 / "logs" / log_name(start["argv"]) if start.get("argv") else None


def probe_done_record(cfg, probe, n):
    """The file whose presence (with a matching sidecar) means a probe attempt finished (as the self-test passes it to
    v7_ledger.py end)."""
    if re.fullmatch(r"p[12]_seed\d(_stall|_no_stall)?", probe or ""):
        return cfg.r7 / "selftest" / f"{probe}.json"
    if re.fullmatch(r"p3_seed\d", probe or ""):
        return cfg.r7 / "selftest" / "p3" / f"attempt_{n}.json"
    if re.fullmatch(r"r1_seed\d_(stall|no_stall)", probe or ""):
        return cfg.r7 / "info" / f"{probe}.json"
    return None


def settle_all(cfg, probes=False, phases=True):
    """Write the attempt record of every attempt in R7/ledger/ that has none (or a torn one), done phases included,
    once no process of it runs: a phase folder only when its phase lock is free; a probe folder (probes=True) only
    when the caller holds the self-test lock. Returns the list of (phase id, attempt) written (C5, C10)."""
    d = cfg.r7 / "ledger"
    written = []
    if not d.is_dir():
        return written
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        is_probe = sub.name.startswith("probe_")
        if (is_probe and not probes) or (not is_probe and not phases):
            continue
        if not is_probe and not lock_free(cfg.r7, sub.name):
            continue
        for n, w in attempt_files(cfg.r7, sub.name):
            if attempt_record(w) is not None:
                continue
            log = log_for_attempt(cfg, w)
            rec = None
            if is_probe:
                record = probe_done_record(cfg, sub.name[len("probe_"):], n)
                rec = LG.finish_attempt(w, out_root=cfg.out, log_path=log, r7=cfg.r7, probe_record=record)
            else:
                rec = LG.finish_attempt(w, out_root=cfg.out, log_path=log, r7=cfg.r7)
            written.append((sub.name, n, (rec.get("classification") or {}).get("kind")))
    return written


def done_files(key, out):
    """The files that make a phase done (its final record alternative or its failed-on-record file), relative to
    OUT/; [] when the phase is not done."""
    done, how, detail = LG.phase_done(key, out)
    if not done:
        return []
    return list(detail) if how == "final" else [detail]


def ledger_evidence(cfg, key, records):
    """(ok, why): a phase that is done on file and not inherited must have been made by a ledgered laptop attempt: an
    attempt classified 'done' whose writes name each file that makes it done, with its current SHA-256 (C5, C10)."""
    files = done_files(key, cfg.out)
    if not files:
        return False, "not done"
    named = {}
    for rec in records.values():
        if rec and (rec.get("classification") or {}).get("kind") == "done":
            for w in rec.get("writes") or []:
                if w.get("sha256"):
                    named.setdefault(os.path.realpath(w["path"]), set()).add(w["sha256"])
    missing = [f for f in files if K.file_sha256(cfg.out / f) not in named.get(os.path.realpath(cfg.out / f), set())]
    if missing:
        return False, (f"{missing} {'is' if len(missing) == 1 else 'are'} on file, but no attempt on this laptop "
                       f"classified 'done' wrote {'it' if len(missing) == 1 else 'them'} with this SHA-256")
    return True, None


def phase_specific_count(records):
    return sum(1 for r in records.values() if r and (r.get("classification") or {}).get("kind") == "phase-specific")


RETRY_RE = re.compile(r"(\d+)\.json$")
TAG_RE = re.compile(r"(\d+)\.attempt_(\d+)\.json$")


def retry_requests(r7, pid):
    d = Path(r7) / "retry" / pid
    out = {}
    if d.is_dir():
        for p in d.iterdir():
            m = RETRY_RE.fullmatch(p.name)
            if m:
                rec = _rec(p)
                if rec is not None:
                    out[int(m.group(1))] = rec
    return out


def retry_tags(r7, pid):
    d = Path(r7) / "retry" / pid
    out = {}
    if d.is_dir():
        for p in d.iterdir():
            m = TAG_RE.fullmatch(p.name)
            if m and _rec(p) is not None:
                out.setdefault(int(m.group(1)), []).append(int(m.group(2)))
    return out


def active_retry(r7, pid, records):
    """The first retry request not used yet: none of its attempts ended other than machine-wide (C5: a retry
    interrupted machine-wide reruns automatically; any other ending uses it)."""
    tags = retry_tags(r7, pid)
    for n in sorted(retry_requests(r7, pid)):
        used = any((records.get(m) or {}).get("classification", {}).get("kind") not in (None, "machine-wide")
                   for m in tags.get(n, []) if records.get(m) is not None)
        if not used:
            return n
    return None


def need(cfg, pid, records=None, write=True):
    """('normal' | 'alone' | 'unfinished', retry request or None, phase-specific count) from the phase's attempt
    records (C5)."""
    records = settle(cfg, pid, write=write) if records is None else records
    n_ps = phase_specific_count(records)
    if n_ps < 3:
        return "normal", None, n_ps
    if n_ps == 3:
        return "alone", None, n_ps
    n = active_retry(cfg.r7, pid, records)
    return ("alone", n, n_ps) if n is not None else ("unfinished", None, n_ps)


def write_retry_request(cfg, pid, argv, tty):
    d = cfg.r7 / "retry" / pid
    n = max(retry_requests(cfg.r7, pid), default=0) + 1
    nxt = max((n_ for n_, _ in attempt_files(cfg.r7, pid)), default=0) + 1
    body = dict(kind="stage2-v7-retry-request", phase_id=pid, request=n, next_attempt=nxt, utc=utc(),
                command_line=list(argv), tty=tty, boot_id=K.boot_id(),
                rule=("one attempt, run alone; whether the phase is retried is not a choice (C5): the request sets "
                      "only when the next attempt starts"))
    K.write_json_once(d / f"{n}.json", body)
    return n, body


# --------------------------------------------------------------------------------------------- memory (C2)

def measured_peaks(cfg):
    """{phase kind: largest VmHWM in bytes} over finished (done) laptop attempts in the ledger."""
    peaks = {}
    d = cfg.r7 / "ledger"
    if not d.is_dir():
        return peaks
    for sub in d.iterdir():
        if not sub.is_dir() or sub.name.startswith("probe_"):
            continue
        for p in sub.glob("*.json"):
            rec = _rec(p)
            if not rec or (rec.get("test_mode") and not cfg.test):
                continue
            if (rec.get("classification") or {}).get("kind") != "done" or not rec.get("vmhwm_kb"):
                continue
            kind = (rec.get("key") or {}).get("phase")
            if kind:
                peaks[kind] = max(peaks.get(kind, 0), int(rec["vmhwm_kb"]) * 1024)
    return peaks


def reservation(kind, peaks):
    peak = peaks.get(kind)
    if peak is None:
        return UNMEASURED_GIB * GIB
    if kind in POLICY_KINDS:
        return max(UNMEASURED_GIB * GIB, 2 * peak)
    return int(MEASURED_FACTOR * peak)


def admit(kind, own, running, total, available, peaks):
    """(ok, why). running: [(kind, reserved bytes)]."""
    if kind == "policy_train" and "policy_train" not in peaks and \
            sum(1 for k, _ in running if k == "policy_train") >= POLICY_TRAIN_CAP:
        return False, f"at most {POLICY_TRAIN_CAP} policy_train at once until one has been measured"
    if sum(r for _, r in running) + own > total - HEADROOM_GIB * GIB:
        return False, (f"reserved {sum(r for _, r in running) / GIB:.1f} GiB + {own / GIB:.1f} GiB would exceed "
                       f"MemTotal - {HEADROOM_GIB} GiB = {(total - HEADROOM_GIB * GIB) / GIB:.1f} GiB")
    if available < own + HEADROOM_GIB * GIB:
        return False, f"MemAvailable {available / GIB:.1f} GiB < {own / GIB:.1f} + {HEADROOM_GIB} GiB"
    return True, None


# --------------------------------------------------------------------------------------------- stand-in errors (C2)

def reviewed(cfg, pid):
    return _rec(cfg.r7 / "reviewed" / f"{pid}.json") is not None


def standin_errors(cfg, phases_state):
    """Every stand-in error in a policy lock the laptop wrote (not inherited): dict rows with the phase whose
    --reviewed lifts it ('error' copied into a no-stall lock is the stalling lock's, S2C/policy.py:549)."""
    rows = []
    for s in EXPLORATORY + CONFIRMATORY:
        for phase, rel in (("policy_train", f"policy/seed{s}/lock.json"),
                           ("policy_no_stall_train", f"policy/seed{s}/no_stall/lock.json")):
            pid = f"{phase}_seed{s}"
            if phases_state.get(pid) == "inherited":
                continue
            lock = _rec(cfg.out / rel)
            if lock is None:
                continue
            st = lock.get("standin") or {}
            for field in ("error", "no_stall_error"):
                if st.get(field) is None:
                    continue
                by = f"policy_train_seed{s}" if (phase == "policy_no_stall_train" and field == "error") else pid
                text = str(st[field])
                rows.append(dict(pid=pid, lock=rel, field=field, error=text, reviewed_by=by,
                                 reviewed=reviewed(cfg, by), memory=any(p in text for p in LG.MEMORY_PATTERNS)))
    return rows


def write_review(cfg, pid, rows, argv, tty):
    """R7/reviewed/<phase>.json, once. Typing the same --reviewed again is not an error: the record that exists is
    returned with 'already' set (the lock is write-once, so the errors it reviewed cannot have changed)."""
    mine = [r for r in rows if r["reviewed_by"] == pid]
    if not mine:
        raise K.V7Refusal(f"--reviewed {pid}: no policy lock written on this laptop records a stand-in error for it")
    path = cfg.r7 / "reviewed" / f"{pid}.json"
    if path.exists() or Path(str(path) + ".sha256").exists():
        state, body = K.read_record(path)
        if state != "ok":
            raise K.V7Refusal(f"{path} is damaged ({body}); paste this to the cloud")
        return dict(body, already=True)
    body = dict(kind="stage2-v7-standin-review", phase_id=pid, utc=utc(), errors=mine, command_line=list(argv), tty=tty,
                rule="new policy training starts again; the lock stays as written (write-once); C2")
    K.write_json_once(path, body)
    return body


# --------------------------------------------------------------------------------------------- the runner

class Orphan:
    """A phase whose launch shell ended while the phase's own process still holds its lock: it counts as running
    until the lock is free (its attempt record is then written by settle())."""

    def __init__(self, r7, pid):
        self.r7, self.pid_name, self.pid, self.returncode = r7, pid, 0, None

    def poll(self):
        return None if not lock_free(self.r7, self.pid_name) else 0


RUN_V7 = f"sh {LAPTOP}/run_v7.sh"


class Runner:
    def __init__(self, cfg, tasks, cpus, version, training, phases_state, out=print, until_waiting=False,
                 entries=None):
        self.cfg, self.tasks, self.cpus = cfg, tasks, list(cpus)
        self.by = {t.pid: t for t in tasks}
        self.version, self.training, self.phases_state = version, training, phases_state
        # the final prereg_v7.json's re-make table, keyed by phase id (C7)
        self.entries = {LG.phase_id(dict(phase=e["phase"], seed=e["seed"], world=e["world"])): e
                        for e in (entries or [])}
        self.out, self.until_waiting = out, until_waiting
        self.stop_requested, self.kill_requested, self.halted, self.review_wait = False, False, None, False
        self.notes, self.peaks = {}, measured_peaks(cfg)
        self.block = []
        self.last_retry_check = 0.0
        self.launcher_sha = K.file_sha256(cfg.launcher)
        self.logfile = cfg.r7 / "runner.log"

    # ---- output
    def say(self, msg):
        line = f"{utc()} {msg}"
        try:
            self.out(line)
        except OSError:
            pass
        try:
            with open(self.logfile, "a", encoding="utf8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass

    def once(self, key, msg):
        if self.notes.get(key) != msg:
            self.notes[key] = msg
            self.say(msg)

    # ---- state
    def log_path(self, t):
        return self.cfg.r7 / "logs" / log_name(t.args)

    def ready(self, t):
        return all(self.by[d].state in FINAL_STATES for d in t.deps if d in self.by)

    def resolve(self, t):
        """inherited | done | skipped | pending, from the files (never from memory)."""
        if self.phases_state.get(t.pid) == "inherited":
            return "inherited", "inherited from the cloud (HANDOFF.json)"
        done, how, detail = LG.phase_done(t.key, self.cfg.out)
        if done:
            return "done", None if how == "final" else "failed on record (never replaced)"
        if (detail or {}).get("partial"):
            self.once(f"partial {t.pid}", f"note: {t.pid} has a partial record {detail['partial']} (its sidecar is "
                                          f"missing or differs); it counts as not done")
        why = skip_reason(self.cfg, t, self.version)
        if why:
            return "skipped", why
        return "pending", None

    def running(self):
        return [t for t in self.tasks if t.proc is not None]

    # ---- start-up: settle earlier attempts (C5)
    def startup(self):
        written = settle_all(self.cfg)
        if written:
            self.say(f"wrote {len(written)} missing attempt records of earlier attempts (their launch shell did not "
                     f"survive): {', '.join(f'{p} {n} ({k})' for p, n, k in written[:8])}")
        for t in self.tasks:
            state, why = self.resolve(t)
            if state == "done" and not self.verify_done(t):
                continue
            if state != "pending":
                t.state, t.note = state, why
                continue
            records = settle(self.cfg, t.pid, self.log_path(t))
            self.after_records(t, records, at_start=True)
        self.refresh_block()

    def verify_done(self, t, records=None):
        """A phase done on file and not inherited: a ledgered laptop attempt made it (ledger_evidence), and, where the
        re-make table has an entry, its lock records the re-made actors (C7 Labels). Halts the run otherwise."""
        records = settle(self.cfg, t.pid, self.log_path(t)) if records is None else records
        ok, why = ledger_evidence(self.cfg, t.key, records)
        if not ok:
            t.state = "halted"
            self.halt(f"HALT: {t.pid} is done on file, but {why}; nothing was moved. Paste this to the cloud and do "
                      f"not restart until the cloud has looked")
            return False
        entry = self.entries.get(t.pid)
        if entry:
            import v7_refs as RF
            check = LG.check_lock_records_reference(RF.ledger_entry(entry), self.cfg.out)
            if check.get("ok") is False:
                t.state = "halted"
                self.halt(f"HALT: reference: the lock of {t.pid} does not record the re-made actors "
                          f"({check.get('details')}); paste this to the cloud and do not restart until it has looked")
                return False
        return True

    def after_records(self, t, records, at_start=False):
        """Move the files of an interrupted attempt aside; halt on files no laptop attempt wrote."""
        files = LG.row_existing_files(t.key, self.cfg.out)
        last = records[max(records)] if records else None
        kind = (last or {}).get("classification", {}).get("kind")
        if files and not records:
            self.halt(f"HALT: {t.pid} has files under {self.cfg.out} that no attempt on this laptop wrote: "
                      f"{files[:5]}; nothing was moved. Paste this to the cloud and do not restart until it has "
                      f"looked.")
            t.state = "halted"
            return
        if files and kind in INTERRUPTED:
            moved = LG.move_interrupted(t.key, self.cfg.out, self.cfg.r7, kind, last["classification"]["why"])
            self.say(f"moved {len(moved['files'])} files of the interrupted {t.pid} to "
                     f"{self.cfg.r7 / 'interrupted' / t.pid} (never deleted)")
        if at_start and kind in ("failed", "v7-refusal"):
            self.say(f"note: the last attempt of {t.pid} ended '{kind}' ({last['classification']['why']}); "
                     f"it is started again (as GPU/ does); its files were left in place")

    def refresh_block(self):
        self.block = standin_errors(self.cfg, self.phases_state)
        for r in self.block:
            if not r["reviewed"]:
                self.once(f"review {r['lock']} {r['field']}",
                          f"REVIEW: stand-in error in {r['lock']} ({r['field']}: {r['error'][:200]}); no new policy "
                          f"training starts; every other phase continues. Paste this to the cloud; once it has looked, "
                          f"type: {RUN_V7} --lanes {len(self.cpus)} --reviewed {r['reviewed_by']} (a second Ubuntu "
                          f"window is fine while this runner runs: it picks the review up within a minute; if the run "
                          f"has stopped, type it inside tmux)")

    def blocked(self):
        return any(not r["reviewed"] for r in self.block)

    # ---- starting and ending
    def start(self, t, cpu, alone=False, retry_n=None):
        if K.file_sha256(self.cfg.launcher) != self.launcher_sha:
            raise K.V7Refusal(f"{self.cfg.launcher} changed while the runner runs")
        t.before = max((n for n, _ in attempt_files(self.cfg.r7, t.pid)), default=0)
        t.cpu, t.started, t.state, t.alone, t.retry_n, t.tagged = cpu, utc(), "running", alone, retry_n, False
        t.reserved = 0 if alone else reservation(t.kind, self.peaks)
        errf = open(self.cfg.r7 / "launch_stderr.txt", "a", encoding="utf8")
        t.proc = subprocess.Popen(["sh", str(self.cfg.launcher)] + self.cfg.launcher_args + [str(cpu)] + t.args,
                                  cwd=str(self.cfg.root if self.cfg.test else K.ROOT), env=dict(BASE_ENV),
                                  stdin=subprocess.DEVNULL, stdout=errf, stderr=errf, start_new_session=True)
        errf.close()
        how = " ALONE (all the memory, nothing else starts)" if alone else ""
        if retry_n:
            how += f", retry request {retry_n}"
        self.say(f"start  {t.pid} on processor {cpu} (attempt {t.before + 1}{how}; reserves "
                 f"{t.reserved / GIB:.1f} GiB)")

    def new_attempts(self, t):
        return [(n, w) for n, w in attempt_files(self.cfg.r7, t.pid) if n > t.before]

    def tag_retry(self, t):
        if t.retry_n is None or t.tagged:
            return
        new = self.new_attempts(t)
        if not new:
            return
        n, w = new[-1]
        try:
            K.write_json_once(self.cfg.r7 / "retry" / t.pid / f"{t.retry_n}.attempt_{n}.json",
                              dict(kind="stage2-v7-retry-attempt", phase_id=t.pid, request=t.retry_n, attempt=n,
                                   utc=utc()))
        except K.V7Refusal:
            pass
        try:
            LG.append_event(w, "retry-request", request=t.retry_n, phase_id=t.pid)
        except OSError:
            pass
        t.tagged = True

    def finish(self, t):
        code = t.proc.returncode
        self.tag_retry(t)
        deadline = time.time() + 30
        while not lock_free(self.cfg.r7, t.pid) and time.time() < deadline:
            time.sleep(0.2)                                    # the phase's own process may outlive its shell
        if not lock_free(self.cfg.r7, t.pid):
            if not isinstance(t.proc, Orphan):
                self.say(f"note: the launch shell of {t.pid} ended (exit {code}) but its phase still runs; waiting")
            t.proc = Orphan(self.cfg.r7, t.pid)
            return
        t.code, t.finished, t.proc = code, utc(), None
        records = settle(self.cfg, t.pid, self.log_path(t))
        new = {n: r for n, r in records.items() if n > t.before}
        if not new:
            what = "REFUSED (v7, exit 97)" if code == K.EXIT_V7 else f"FAILED: {t.pid} exit {code}"
            t.state, t.note = "halted", f"launch_v7.sh ended (exit {code}) before the phase started"
            self.halt(f"{what}: {t.pid}: launch_v7.sh ended before the phase started; see "
                      f"{self.cfg.r7 / 'launch_stderr.txt'}")
            return
        rec = new[max(new)] or {}
        kind = (rec.get("classification") or {}).get("kind")
        why = (rec.get("classification") or {}).get("why")
        if kind == "done":
            ref = rec.get("reference") or {}
            check = rec.get("reference_lock_check")
            if ref.get("entry") and check is not None and check.get("ok") is False:
                t.state = "halted"
                self.halt(f"HALT: reference: the lock of {t.pid} does not record the re-made actors "
                          f"({check.get('details')}); paste this to the cloud and do not restart until it has looked")
                return
            if not self.verify_done(t, records):
                return
            state, note = self.resolve(t)
            t.state, t.note = ("done", note) if state in ("done", "inherited") else (state, note)
            self.say(f"done   {t.pid} (exit {code})" + (f": {note}" if note else ""))
            self.peaks = measured_peaks(self.cfg)
            if t.kind in POLICY_KINDS:
                self.refresh_block()
            if t.kind == "pooled":
                self.provenance(t.key["group"])
            return
        if kind in INTERRUPTED:
            self.after_records(t, records)
            mode, _, n_ps = need(self.cfg, t.pid, records)
            if kind == "machine-wide":
                self.say(f"interrupted: {t.pid} (machine-wide: {why}); it reruns")
            elif n_ps <= 3:
                self.say(f"interrupted: {t.pid} (phase-specific {n_ps} of 3: {why})")
            else:
                self.say(f"interrupted: {t.pid} (phase-specific, when run alone: {why})")
            t.state = "pending"
            if mode == "alone" and n_ps == 3 and kind == "phase-specific":
                self.once(f"alone {t.pid}", f"waiting to run alone: {t.pid}")
            if mode == "unfinished":
                t.state, t.note = "unfinished", f"{n_ps} phase-specific interruptions; waiting for --retry"
                self.say(f"UNFINISHED: {t.pid}. It must be retried until done: paste the output of "
                         f"'tail -n 30 {self.log_path(t)}' to the cloud, then type: {RUN_V7} --retry {t.pid}")
            return
        t.state = "halted"
        if kind == "v7-refusal":
            self.halt(f"REFUSED (v7, exit 97): {t.pid}: {why}; log {self.log_path(t)}. Paste the output and the output "
                      f"of 'tail -n 30 {self.log_path(t)}' to the cloud")
        else:
            self.halt(f"FAILED: {t.pid} exit {code}: {why}; log {self.log_path(t)}. No new phase starts; paste the "
                      f"output of 'tail -n 30 {self.log_path(t)}' to the cloud and do not restart until the cloud has "
                      f"looked")

    def halt(self, msg):
        if self.halted is None:
            self.halted = msg
        self.say(msg)

    def provenance(self, group):
        try:
            import make_results_v7 as MR
            path = MR.write_provenance(self.cfg, group)
            self.say(f"provenance: {path}")
        except Exception as error:  # noqa: BLE001 - bookkeeping only; reported, never fatal
            self.say(f"WARN the provenance file for {group} could not be written: {type(error).__name__}: {error}")

    # ---- the second Ctrl-C (C5: every signal the runner sends is in the ledger first)
    def stop_running(self, sig=signal.SIGTERM, why="second Ctrl-C"):
        for t in self.running():
            for n, w in self.new_attempts(t):
                try:
                    LG.append_event(w, "runner-signal", signal=int(sig), why=why)
                except OSError:
                    pass
            try:
                if not isinstance(t.proc, Orphan):
                    os.killpg(t.proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            self.say(f"stopped {t.pid} (signal {int(sig)}, recorded in its ledger: a machine-wide interruption; it "
                     f"reruns on the next start)")

    # ---- the loop
    def check_retries(self, force=False):
        if not force and time.time() - self.last_retry_check < self.cfg.retry_poll:
            return
        self.last_retry_check = time.time()
        if self.block:                                     # a review written by 'run_v7.sh --reviewed' meanwhile
            before = self.blocked()
            self.refresh_block()
            if before and not self.blocked():
                self.review_wait = False
                self.say("reviewed: every stand-in error has its review now; new policy training may start again")
        for t in self.tasks:
            if t.state == "unfinished":
                mode, n, _ = need(self.cfg, t.pid)
                if mode == "alone":
                    t.state, t.note = "pending", f"retry request {n} picked up"
                    self.say(f"retry request {n} for {t.pid} picked up: it runs alone when nothing else is runnable")

    def schedule(self):
        if self.halted or self.stop_requested:
            return
        run = self.running()
        if any(t.alone for t in run):
            return
        total, avail = meminfo(self.cfg.meminfo)
        cands = []
        for t in sorted(self.tasks, key=lambda x: -x.prio):
            if t.state != "pending" or not self.ready(t):
                continue
            if not lock_free(self.cfg.r7, t.pid):          # started outside this runner; it is not started twice
                self.once(f"elsewhere {t.pid}", f"waiting: {t.pid} is running outside this runner (its phase lock "
                                                 f"is held)")
                continue
            state, why = self.resolve(t)
            if state == "done" and not self.verify_done(t):
                return
            if state != "pending":
                t.state, t.note = state, why
                self.say({"inherited": "inherited", "done": "found ", "skipped": "skip  "}[state] + f" {t.pid}" +
                         (f": {why}" if why and state == "skipped" else " (finished earlier)" if state == "done"
                          else ""))
                continue
            mode, n, n_ps = need(self.cfg, t.pid)
            if mode == "unfinished":
                t.state, t.note = "unfinished", f"{n_ps} phase-specific interruptions; waiting for --retry"
                self.once(f"unfinished {t.pid}", f"UNFINISHED: {t.pid}. Paste the output of 'tail -n 30 "
                                                  f"{self.log_path(t)}' to the cloud, then type: {RUN_V7} --retry "
                                                  f"{t.pid}")
                continue
            if t.kind in POLICY_KINDS and self.blocked():
                self.once(f"blocked {t.pid}", f"waiting: {t.pid} (no new policy training until the stand-in error "
                                              f"is reviewed)")
                continue
            cands.append((t, mode, n))
        normal = [c for c in cands if c[1] == "normal"]
        alone = [c for c in cands if c[1] == "alone"]
        for t, _, n in alone:
            self.once(f"alone {t.pid}", f"waiting to run alone: {t.pid}" + (f" (retry request {n})" if n else ""))
        busy = {t.cpu for t in run}
        for t, _, _ in normal:
            free = [c for c in self.cpus if c not in busy]
            if not free:
                self.once("slot", f"waiting: slot (all {len(self.cpus)} lanes busy)")
                break
            own = reservation(t.kind, self.peaks)
            ok, why = admit(t.kind, own, [(r.kind, r.reserved) for r in self.running()], total, avail, self.peaks)
            if not ok:
                self.once(f"memory {t.pid}", f"waiting: memory ({t.pid}: {why})")
                continue
            self.notes.pop(f"memory {t.pid}", None)
            self.notes.pop("slot", None)
            self.start(t, free[0])
            busy.add(free[0])
            avail -= own
        if not normal and alone and not self.running():
            t, _, n = alone[0]
            self.start(t, self.cpus[0], alone=True, retry_n=n)

    def write_status(self):
        total, avail = meminfo(self.cfg.meminfo)
        body = dict(kind="stage2-v7-runner-status", updated=utc(), test=self.cfg.test, lanes=len(self.cpus),
                    cpus=self.cpus, version=self.version, training=self.training,
                    memory=dict(total_gib=round(total / GIB, 2), available_gib=round(avail / GIB, 2),
                                reserved_gib=round(sum(t.reserved for t in self.running()) / GIB, 2),
                                measured_peaks_gib={k: round(v / GIB, 3) for k, v in sorted(self.peaks.items())}),
                    running={t.pid: dict(pid=t.proc.pid, cpu=t.cpu, started=t.started, alone=t.alone)
                             for t in self.running()},
                    unfinished=sorted(t.pid for t in self.tasks if t.state == "unfinished"),
                    standin_errors=self.block, halted=self.halted, stop_requested=self.stop_requested,
                    counts={s: sum(1 for t in self.tasks if t.state == s) for s in
                            ("pending", "running", "done", "inherited", "skipped", "unfinished", "halted")},
                    tasks={t.pid: t.row() for t in self.tasks})
        tmp = self.cfg.r7 / "status.json.tmp"
        tmp.write_text(json.dumps(body, indent=1, sort_keys=True, default=str), encoding="utf8")
        os.replace(tmp, self.cfg.r7 / "status.json")

    def finished_all(self):
        return all(t.state in FINAL_STATES for t in self.tasks)

    def run(self):
        killed_at = None
        while True:
            for t in self.running():
                self.tag_retry(t)
                if t.proc.poll() is not None:
                    self.finish(t)
            if self.kill_requested and killed_at is None:
                self.stop_running()
                killed_at = time.time()
            elif killed_at is not None and time.time() - killed_at > self.cfg.stop_grace and self.running():
                self.stop_running(signal.SIGKILL, "second Ctrl-C; still running after the grace period")
                killed_at = time.time() + 10 ** 6
            self.check_retries()
            self.schedule()
            self.write_status()
            if not self.running():
                if self.halted or self.stop_requested or self.finished_all():
                    break
                waiting = sorted(t.pid for t in self.tasks if t.state == "unfinished")
                ready = [t for t in self.tasks if t.state == "pending" and self.ready(t)]
                startable = [t for t in ready if not (t.kind in POLICY_KINDS and self.blocked())]
                if not startable and waiting:
                    self.once("waiting", f"WAITING FOR RETRY: {', '.join(waiting)}. For each, type: {RUN_V7} "
                                         f"--retry <phase> (a second Ubuntu window is fine while this runner runs)")
                    if self.until_waiting:
                        break
                elif not startable and ready:
                    self.review_wait = True
                    self.say(f"WAITING FOR REVIEW: only policy training is left ({', '.join(t.pid for t in ready)}) "
                             f"and it waits for the stand-in review. Paste the REVIEW lines to the cloud; once it has "
                             f"looked, type inside tmux: {RUN_V7} --lanes {len(self.cpus)} --reviewed "
                             f"<the phase the REVIEW line names>")
                    break
            time.sleep(self.cfg.poll)
        self.write_status()
        return self.verdict()

    def verdict(self):
        unfinished = sorted(t.pid for t in self.tasks if t.state == "unfinished")
        c = {s: sum(1 for t in self.tasks if t.state == s) for s in FINAL_STATES}
        tail = f"unfinished: {', '.join(unfinished)}" if unfinished else "unfinished: none"
        if self.halted:
            self.say(f"runner ends (halted): {c['done']} done, {c['inherited']} inherited, {c['skipped']} skipped; "
                     f"{tail}")
            return 1
        if self.finished_all():
            self.say(f"runner ends: every phase done, inherited or skipped on record ({c['done']} done, "
                     f"{c['inherited']} inherited, {c['skipped']} skipped); {tail}")
            return 0
        if self.review_wait:
            self.say(f"runner ends while WAITING FOR REVIEW (a stand-in error); {tail}")
            return 5
        if unfinished and not self.stop_requested:
            self.say(f"runner ends while WAITING FOR RETRY: {tail}")
            return 4
        self.say(f"runner ends (stopped by you; nothing failed): {c['done']} done, "
                 f"{sum(1 for t in self.tasks if t.state == 'pending')} not started; {tail}")
        return 3


# --------------------------------------------------------------------------------------------- preflight

def check_probe_pins(cfg, prereg):
    """C7 'Probe records': the nine probe records and every reference file match their pins in the final
    prereg_v7.json (a mismatch means R7/ was damaged: HALT). Returns the number of files checked."""
    pins = (prereg.payload.get("handoff") or {}).get("probe_records") or {}
    if not pins:
        if cfg.test:
            return 0
        raise K.V7Refusal("HALT: probe record: the final prereg_v7.json pins no probe records")
    bad, n = [], 0
    for pid, row in sorted(pins.items()):
        rec = cfg.r7 / "selftest" / f"{pid}.json"
        n += 1
        if not rec.is_file() or K.file_sha256(rec) != row.get("sha256") or K.sidecar_digest(rec) != row.get("sha256"):
            bad.append(f"HALT: probe record {pid} differs from its pin in the final prereg_v7.json")
        for rel, sha in sorted((row.get("file_pins") or {}).items()):
            n += 1
            f = cfg.r7 / rel
            if not f.is_file() or K.file_sha256(f) != sha:
                bad.append(f"HALT: reference {rel} differs from its pin in the final prereg_v7.json")
    if bad:
        raise K.V7Refusal("; ".join(bad[:4]) + f". Paste this to the cloud; it will guide 'sh {LAPTOP}/selftest_v7.sh "
                          f"--remake <probe>', which puts back a damaged probe record from the cloud's copy in "
                          f"V7/handoff/probes/ (only when its SHA-256 equals the pin) and remakes a damaged reference "
                          f"file (only when the same call reproduces the pin)")
    return n


def preflight(cfg):
    """Refuses (V7Refusal) unless the final prereg_v7.json, the v7 code, HANDOFF.json (no WITHDRAWN.json), the v7 and
    v6 approval chains, the restore, and the probe records and reference files (against their pins) all verify.
    Returns (prereg, handoff)."""
    import v7_verify as VF
    prereg = VF.load_prereg_v7(cfg.prereg)
    VF.require_final(prereg)
    if not cfg.test:
        if prereg.payload.get("predecessor") != K.V6:
            raise K.V7Refusal("prereg_v7.json does not name version 6's pins")
        VF.check_v7_code(prereg, cfg.v7_root)
        if K.file_sha256(cfg.launcher) != (prereg.payload["code"]["files"] or {}).get("code/launch_v7.sh"):
            raise K.V7Refusal("launch_v7.sh is not the pinned one")
    handoff = VF.check_handoff(prereg, cfg.v7_root)
    if cfg.check_chain:
        try:
            VF.verify_v7_chain(prereg, cfg.v7_root)
        except K.V7Refusal as error:
            raise K.V7Refusal(f"refused: approval: {error} (steps 10-11 are not complete)") from None
        VF.verify_v6_chain()
    if not cfg.test:
        rec = _rec(cfg.r7 / "restore_last.json")
        snap = (handoff.get("snapshot") or {})
        if rec is None or not rec.get("ok") or rec.get("commit") != snap.get("commit") or \
                rec.get("sums_sha256") != snap.get("sums_sha256"):
            raise K.V7Refusal("refused: restore: the last restore is not the handoff's final snapshot (README step 9: "
                              "rerun restore_v7.sh)")
    check_probe_pins(cfg, prereg)
    return prereg, handoff


# --------------------------------------------------------------------------------------------- commands

def cmd_run(cfg, a, out=say_line):
    cfg.r7.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "ledger", "locks", "slots", "retry", "reviewed"):
        (cfg.r7 / sub).mkdir(parents=True, exist_ok=True)
    with open(cfg.r7 / "runner.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            out("refused: another runner is running on this machine (R7/runner.lock); 'run_v7.sh status' shows it")
            return 1
        return _run_locked(cfg, a, out)


def _run_locked(cfg, a, out):
    held = held_phase_locks(cfg.r7)
    if held:
        out(f"refused: phases from an earlier run are still running ({', '.join(held)}); wait for them to end")
        return 1
    if a.until_waiting and not cfg.test:
        out("refused: --until-waiting is for the package's own tests only")
        return 1
    try:
        prereg, handoff = preflight(cfg)
        version, training = run_conditions(cfg)
    except K.V7Refusal as error:
        text = str(error)
        out(text if text.startswith(("refused", "HALT")) else f"refused: {text}")
        return 1
    entries = (prereg.payload.get("handoff") or {}).get("remake_table") or []
    phases_state = dict(handoff.get("phases") or {})
    tasks = build_plan(version, training, cfg.seeds, cfg.pooled)
    lanes = a.lanes
    if not 1 <= lanes <= MAX_LANES:
        out(f"refused: --lanes must be 1-{MAX_LANES}")
        return 1
    cpus, why = lanes_for(cfg, lanes, a.cpus.split(",") if a.cpus else None)
    if why:
        out(f"refused: {why}")
        return 1
    runner = Runner(cfg, tasks, cpus, version, training, phases_state, out=out, until_waiting=a.until_waiting,
                    entries=entries)
    if a.plan_only:
        out(f"plan: {len(tasks)} phases; version {version}; policy test and full host "
            f"{'run' if training else 'do not run'}; lanes {lanes} on processors {cpus}")
        for t in sorted(tasks, key=lambda x: -x.prio):
            state, why = runner.resolve(t)
            mode, n, n_ps = need(cfg, t.pid, write=False)
            extra = f" [{n_ps} phase-specific; {mode}]" if n_ps else ""
            out(f"  {state:<9} {t.pid:<52} chain {t.prio:>4} min{extra}" + (f"  ({why})" if why else ""))
        return 0
    if a.reviewed and not cfg.test and not sys.stdin.isatty():
        out("refused: --reviewed must be typed at a terminal (standard input is not one)")
        return 1
    for pid in a.reviewed or []:
        try:
            body = write_review(cfg, pid, standin_errors(cfg, phases_state), sys.argv, _tty())
            runner.say(f"reviewed: {pid} (R7/reviewed/{pid}.json" + (f", already written {body.get('utc')}"
                                                                      if body.get("already") else "") +
                       "); new policy training may start again")
        except K.V7Refusal as error:
            out(f"refused: {error}")
            return 1
    runner.say(f"plan: {len(tasks)} phases; version {version}; policy test and full host "
               f"{'run' if training else 'do not run'}; lanes {lanes} on processors {cpus}; "
               f"{sum(1 for v in phases_state.values() if v == 'inherited')} phases inherited from the cloud")

    def on_signal(signum, frame):
        if not runner.stop_requested:
            runner.stop_requested = True
            runner.say("stop requested: no new phase starts; the running ones finish first (this can take over an "
                       "hour). Press Ctrl-C again to stop them now (they rerun on the next start)")
        elif not runner.kill_requested:
            runner.kill_requested = True
            runner.say("second Ctrl-C: stopping the running phases")

    old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        runner.startup()
        return runner.run()
    except K.V7Refusal as error:
        runner.say(f"refused: {error}")
        return 1
    finally:
        for s, h in old.items():
            signal.signal(s, h)


def _tty():
    try:
        return os.ttyname(sys.stdin.fileno())
    except (OSError, AttributeError, ValueError):
        return None


def cmd_retry(cfg, a, out=say_line, isatty=None):
    isatty = sys.stdin.isatty() if isatty is None else isatty
    if not isatty:
        out("refused: --retry must be typed at a terminal (standard input is not one)")
        return 1
    try:
        _, handoff = preflight(cfg) if not cfg.test else (None, read_handoff(cfg) or {})
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    pid = a.phase
    tasks = {t.pid: t for t in build_plan(*run_conditions(cfg), cfg.seeds, cfg.pooled)}
    if pid not in tasks:
        out(f"refused: {pid} is not a phase of the plan (use the name printed after UNFINISHED:)")
        return 1
    t = tasks[pid]
    if (handoff.get("phases") or {}).get(pid) == "inherited" or LG.phase_done(t.key, cfg.out)[0]:
        out(f"refused: {pid} is done; there is nothing to retry")
        return 1
    if not lock_free(cfg.r7, pid):
        out(f"refused: {pid} is running now")
        return 1
    mode, n, n_ps = need(cfg, pid, write=False)
    if mode != "unfinished":
        out(f"refused: {pid} is not unfinished ({n_ps} phase-specific interruptions" +
            (f"; retry request {n} is still waiting to be used)" if n else ")"))
        return 1
    argv = sys.argv if a.argv is None else (a.argv.split() if isinstance(a.argv, str) else list(a.argv))
    n, body = write_retry_request(cfg, pid, argv, _tty())
    if runner_running(cfg.r7):
        out(f"retry request {n} for {pid} written (R7/retry/{pid}/{n}.json); the running runner picks it up within a "
            f"minute and runs it alone when nothing else is runnable")
        return 10
    out(f"retry request {n} for {pid} written (R7/retry/{pid}/{n}.json); starting the runner")
    return 0


def cmd_review(cfg, a, out=say_line, isatty=None):
    """'run_v7.sh --reviewed <phase>' (C2): records the user's review of a stand-in error, typed at a terminal, once
    (typing it again only says so). Exit 10 when a runner is running (it picks the review up within a minute), 0 when
    none runs (run_v7.sh then starts one), 1 on a refusal."""
    isatty = sys.stdin.isatty() if isatty is None else isatty
    if not isatty:
        out("refused: --reviewed must be typed at a terminal (standard input is not one)")
        return 1
    try:
        _, handoff = preflight(cfg) if not cfg.test else (None, read_handoff(cfg) or {})
    except K.V7Refusal as error:
        text = str(error)
        out(text if text.startswith(("refused", "HALT")) else f"refused: {text}")
        return 1
    argv = sys.argv if a.argv is None else (a.argv.split() if isinstance(a.argv, str) else list(a.argv))
    try:
        body = write_review(cfg, a.phase, standin_errors(cfg, dict(handoff.get("phases") or {})), argv, _tty())
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    if body.get("already"):
        out(f"reviewed: {a.phase} was already recorded ({body.get('utc')}, R7/reviewed/{a.phase}.json); nothing new to "
            f"record")
    else:
        out(f"reviewed: {a.phase} recorded (R7/reviewed/{a.phase}.json); new policy training may start again")
    if runner_running(cfg.r7):
        out("the running runner picks the review up within a minute; nothing else to type")
        return 10
    return 0


def cmd_status(cfg, out=say_line):
    alive = runner_running(cfg.r7)
    p = cfg.r7 / "status.json"
    if not p.exists():
        out(f"no status yet ({p}); runner running: {'yes' if alive else 'no'}")
        return 0
    body = json.loads(p.read_text(encoding="utf8"))
    out(f"runner running: {'yes' if alive else 'no'}; last update {body.get('updated')}; lanes {body.get('lanes')} "
        f"on processors {body.get('cpus')}")
    out("counts: " + ", ".join(f"{k} {v}" for k, v in sorted((body.get("counts") or {}).items())))
    m = body.get("memory") or {}
    out(f"memory: MemTotal {m.get('total_gib')} GiB, MemAvailable {m.get('available_gib')} GiB, reserved by running "
        f"phases {m.get('reserved_gib')} GiB")
    for pid, r in sorted((body.get("running") or {}).items()):
        out(f"  running {pid} on processor {r.get('cpu')} since {r.get('started')}" + (" ALONE" if r.get("alone") else ""))
    if body.get("unfinished"):
        out(f"UNFINISHED: {', '.join(body['unfinished'])}")
    for r in body.get("standin_errors") or []:
        if not r.get("reviewed"):
            out(f"REVIEW: stand-in error in {r.get('lock')}")
    if body.get("halted"):
        out(f"halted: {body['halted']}")
    return 0


def main(argv=None, out=say_line):
    argv = list(sys.argv[1:] if argv is None else argv)
    fixture = None
    if argv[:1] == ["--v7-test"]:
        fixture, argv = argv[1], argv[2:]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--lanes", type=int, default=DEFAULT_LANES)
    r.add_argument("--cpus")
    r.add_argument("--reviewed", action="append")
    r.add_argument("--plan-only", action="store_true")
    r.add_argument("--until-waiting", action="store_true", help=argparse.SUPPRESS)
    t = sub.add_parser("retry")
    t.add_argument("phase")
    t.add_argument("--argv", help=argparse.SUPPRESS)          # the user's command line, as one string
    v = sub.add_parser("review")
    v.add_argument("phase")
    v.add_argument("--argv", help=argparse.SUPPRESS)
    sub.add_parser("status")
    a = ap.parse_args(argv)
    try:
        cfg = config(fixture)
    except K.V7Refusal as error:
        out(f"refused: {error}")
        return 1
    if a.cmd == "run":
        return cmd_run(cfg, a, out)
    if a.cmd == "retry":
        return cmd_retry(cfg, a, out)
    if a.cmd == "review":
        return cmd_review(cfg, a, out)
    return cmd_status(cfg, out)


if __name__ == "__main__":
    sys.exit(main())
