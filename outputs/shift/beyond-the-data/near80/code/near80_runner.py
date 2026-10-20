"""Near-80 (version 9, main setting): the lane scheduler. DRAFT, not frozen. run_near80.sh (the desktop) and
near80_slurm_job.sh (the cluster) start it after their own checks; the commands are

  near80_runner.py run [--lanes N] [--seeds a,b] [--group-only] [--tag T]
                                              the official run (default 10 lanes, 1-16); continues after any restart.
                                              --seeds restricts this runner to the phases of the given fresh seeds
                                              (one Slurm job per seed); --group-only to the group's lock and look
                                              (the last job); --tag names this runner's lock and status file.
  near80_runner.py plumbing [--lanes N]       the unit tests, a 40-step smoke of every phase kind on seed 0, the
                                              fresh-seed code path on seed 0 and the reproduction check (scratch only)
  near80_runner.py status                     what is running, done, waiting, stopped and skipped (reads only)
  near80_runner.py selftest [--for CMD]       the short self-test (prints READY or NOT READY)
  near80_runner.py plan                       the phase plan and each phase's state (runs nothing)
  near80_runner.py retry PHASE_ID             one more attempt of an UNFINISHED phase (typed at a terminal)
  near80_runner.py budget-continue STAGE      lifts a budget pause by one forecast of that stage (typed at a terminal)
  near80_runner.py report                     the halt report (pooled.halt_report), when the run stopped early
  near80_runner.py results                    copies every record (no model file) into near80/results-<utc>/
  near80_runner.py runfiles-plan --dir DIR    the list of run files for push_runfiles_near80.sh
  near80_runner.py stub --fixture DIR [--lanes N] [--seeds a,b] [--refuse a,b]   stub mode on a test fixture
                                              outside the repository (--refuse: seeds whose synthetic data make BCA
                                              refuse, for the refusal rule)

Version 7's runner (stage2_v7/laptop/lib/runner_v7.py) restated with near-80 names; it is never imported here.
  One runner per output root and tag: OUT/runner<tag>.lock. Several runners may share one output root when their
    seeds do not overlap (the cluster's one-job-per-seed entry); each phase holds its own lock, so no phase runs
    twice. Version 8's version-7 gating is gone: version 7's run finished on 16 October 2026.
  Plumbing gate: 'run' refuses unless OUT/plumbing/plumbing.json exists, matches its sidecar, passed, and carries the
    current prereg_sha256. A reproduction difference writes OUT/plumbing/reproduction_differs.json (write-once): then
    'run' and 'plumbing' refuse for good (section 8.2). Any other plumbing failure writes
    OUT/plumbing/failed_<utc>.json and the plumbing may be run again.
  Approval: run, plumbing and retry call approval_check.require_approval() first; every phase process checks again.
  Lanes: 1-16 (default 10); processors from version 7's pick_cpus(16)[:lanes] (one per physical core, within the
    processors this process may use: a Slurm job's allocation); each phase:
    env -i <PHASE_ENV> nice -n 5 taskset -c <cpu> <python> phase_main.py --phase-id <id> --out <root> ...
    with standard output and error appended to OUT/logs/<phase_id>.log. (The unit tests on Windows spawn the same
    command without env -i, nice and taskset; nothing official runs there.)
  Memory rule (version 7's): a phase starts only if the running phases' reservations plus its own are at most
    MemTotal - 8 GiB and MemAvailable >= own + 8 GiB. Reservation: an unmeasured kind 7 GiB; a measured kind 1.25 times
    its largest peak; a policy phase the larger of 7 GiB and twice its peak; at most 2 policy phases at once until
    one is measured. On the cluster the job's memory (NEAR80_MEM_TOTAL_KB) stands for MemTotal, and the free memory
    is that total minus the running phases' reservations (the node's meminfo says nothing about the job's share).
  Budget guard: CPU-hours per stage from the ledger; a stage beyond PAUSE_FACTOR times its forecast starts no new
    phase; beyond CAP_CPU_HOURS nothing starts. Each typed 'budget-continue STAGE' raises both by one forecast.
  Safe stop: one Ctrl-C (or Slurm's SIGTERM), no new phase starts and the running ones finish (exit 3); a second
    sends SIGTERM to the running phases (recorded first in their ledger: machine-wide, they rerun on the next start).
  Interruptions (ledger.classify_attempt): machine-wide reruns without limit; phase-specific reruns twice as usual,
    the third time alone, then the phase is UNFINISHED and waits for 'retry'. Before any rerun every existing file of
    the phase is moved, never deleted, to OUT/interrupted/. A recorded stop (exit 2 with its stop file) is never
    rerun. Exit 97 from a phase halts the run; any other failure stops new phases and names the log.
  Status: OUT/status<tag>.json (+ sidecar), rewritten each poll through os.replace.
Exit codes: 0 finished; 3 stopped safely; 4 waiting for a retry; 5 paused by the budget guard; 97 refused; 1 a phase
or a check failed.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402

EXIT_FINISHED, EXIT_STOPPED, EXIT_RETRY, EXIT_FAILED = 0, 3, 4, 1
EXIT_REFUSED, EXIT_PAUSED = K.EXIT_REFUSED, K.EXIT_PAUSED
RUN_N80 = (f"sh {K.N80}/run_near80.sh" if K.HOST == "desktop" else f"sh {K.N80}/run_near80_cluster.sh")
FINAL_STATES = PH.FINAL
INTERRUPTED = LG.INTERRUPTED
SYSFS = "/sys/devices/system/cpu"
PLUMBING_SCRATCH = Path(K.HOME) / "near80_v9_plumbing"
PAUSE_TEXT = "no result is read differently because of a pause"
STATUS_NAME = {"pending": "waiting"}           # the status file's word for a phase that has not started yet
STUB_MEMINFO = "MemTotal:       25165824 kB\nMemAvailable:   24117248 kB\n"   # a stub fixture's: 24 GiB, as WSL
MODEL_SUFFIXES = (".msgpack", ".npz", ".hdf5", ".h5")   # model, array and training files: never in results-*/
POSIX = os.name == "posix"


# --------------------------------------------------------------------------------------------- small helpers

def say_line(line):
    print(line, flush=True)


lock_free = LG.lock_free


def phase_lock_free(out_root, pid):
    return lock_free(Path(out_root) / "locks" / f"{pid}.lock")


def runner_lock_name(tag=None):
    return "runner.lock" if not tag else f"runner_{tag}.lock"


def status_name(tag=None):
    return K.LAYOUT["status"] if not tag else f"status_{tag}.json"


def runner_running(root, tag=None):
    """A runner holds its lock under root (any tag when tag is None)."""
    root = Path(root)
    if tag is not None:
        return not lock_free(root / runner_lock_name(tag))
    return any(not lock_free(p) for p in root.glob("runner*.lock")) if root.is_dir() else False


def held_phase_locks(root):
    """version 7's held_phase_locks: the phases whose lock some process holds now."""
    d = Path(root) / "locks"
    return sorted(p.stem for p in d.glob("*.lock") if not lock_free(p)) if d.is_dir() else []


def pick_cpus(n, sysfs=SYSFS, affinity=None):
    """version 7's pick_cpus: n processors on different physical cores when the topology says so, else the first n;
    always within the processors this process may use (a Slurm job's allocation)."""
    allowed = set(LG.cpu_affinity() if affinity is None else affinity)
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


def lane_cpus(lanes, sysfs=SYSFS, affinity=None, given=None):
    """(processors, refusal): the first `lanes` of pick_cpus(16) (or the given ones, which must be among them)."""
    if not 1 <= int(lanes) <= K.MAX_LANES:
        return None, f"--lanes must be 1-{K.MAX_LANES}"
    allowed = pick_cpus(K.MAX_LANES, sysfs, affinity) if given is None else list(given)
    cpus = allowed[:int(lanes)]
    if len(cpus) < int(lanes) or len(set(cpus)) != len(cpus):
        return None, f"{lanes} lanes need {lanes} different processors; this machine offers {allowed}"
    return cpus, None


def meminfo(path="/proc/meminfo", reserved=0):
    """(MemTotal, MemAvailable) in bytes. With NEAR80_MEM_TOTAL_KB set (a Slurm job), the job's memory is the total
    and the free memory is that total minus the running phases' reservations."""
    env_total = os.environ.get("NEAR80_MEM_TOTAL_KB")
    if env_total:
        total = int(env_total) * 1024
        return total, max(0, total - int(reserved))
    vals = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("MemTotal:", "MemAvailable:"):
            vals[parts[0][:-1]] = int(parts[1]) * 1024
    return vals.get("MemTotal", 0), vals.get("MemAvailable", 0)


def reservation(kind, peaks):
    """version 7's reservation with near-80's policy kind."""
    peak = peaks.get(kind)
    if peak is None:
        return K.UNMEASURED_GIB * K.GIB
    if kind in K.POLICY_KINDS:
        return max(K.UNMEASURED_GIB * K.GIB, 2 * peak)
    return int(K.MEASURED_FACTOR * peak)


def admit(kind, own, running, total, available, peaks):
    """version 7's admit: (ok, why). running: [(kind, reserved bytes)]."""
    if kind in K.POLICY_KINDS and kind not in peaks and \
            sum(1 for k, _ in running if k in K.POLICY_KINDS) >= K.POLICY_CAP_UNMEASURED:
        return False, f"at most {K.POLICY_CAP_UNMEASURED} policy phases at once until one has been measured"
    if sum(r for _, r in running) + own > total - K.HEADROOM_GIB * K.GIB:
        return False, (f"reserved {sum(r for _, r in running) / K.GIB:.1f} GiB + {own / K.GIB:.1f} GiB would exceed "
                       f"MemTotal - {K.HEADROOM_GIB} GiB = {(total - K.HEADROOM_GIB * K.GIB) / K.GIB:.1f} GiB")
    if available < own + K.HEADROOM_GIB * K.GIB:
        return False, f"MemAvailable {available / K.GIB:.1f} GiB < {own / K.GIB:.1f} + {K.HEADROOM_GIB} GiB"
    return True, None


def budget(hours, forecast=K.FORECAST_CPU_HOURS, factor=K.PAUSE_FACTOR, cap=K.CAP_CPU_HOURS, continues=None):
    """(paused stages {stage: why}, whole run over the cap: why or None). Only the stages of the official plan pause
    (K.BUDGET_STAGES; the plumbing has no phase there). continues: {stage: number of typed budget-continue records};
    each raises that stage's pause point and the cap by one forecast of the stage."""
    continues = dict(continues or {})
    paused = {}
    for stage in K.BUDGET_STAGES:
        f = float(forecast.get(stage, 0.0))
        n = int(continues.get(stage, 0))
        h = float(hours.get(stage, 0.0))
        limit = (factor + n) * f
        if h > limit:
            paused[stage] = (f"paused: the {stage.replace('_', ' ')} stage has used {h:.1f} CPU-hours, more than "
                             f"{limit:g} ({factor:g} times its forecast of {f:g}" +
                             (f", plus {n} typed continuation{'s' if n != 1 else ''}" if n else "") +
                             f"); {PAUSE_TEXT}. To allow one more forecast, type: {RUN_N80} budget-continue {stage}")
    cap = float(cap) + sum(int(continues.get(st, 0)) * float(forecast.get(st, 0.0)) for st in K.BUDGET_STAGES)
    total = sum(float(v) for v in hours.values())
    over = None if total <= cap else (f"paused: the run has used {total:.1f} CPU-hours, beyond the cap of {cap:g}; "
                                      f"no new phase starts; {PAUSE_TEXT}. To allow one more forecast of the stage "
                                      f"that waits, type: {RUN_N80} budget-continue <stage>")
    return paused, over


def budget_continues(out_root):
    """{stage: number of verified budget-continue records under OUT/budget/<stage>/}."""
    out = {}
    for stage in K.BUDGET_STAGES:
        d = Path(out_root) / "budget" / stage
        if d.is_dir():
            out[stage] = sum(1 for q in d.iterdir() if RETRY_RE.match(q.name) and LG.ok_record(q) is not None)
    return out


def tty_name():
    try:
        return os.ttyname(sys.stdin.fileno())
    except (OSError, AttributeError, ValueError):
        return None


# --------------------------------------------------------------------------------------------- configuration

class Cfg:
    """Where and how the runner works. Official values are fixed; stub mode and the tests may move the paths (a
    stub fixture's runner.json may give meminfo, cpus and poll; every path stays inside the fixture)."""

    def __init__(self, **kw):
        self.mode = kw.pop("mode", "official")
        self.out = Path(kw.pop("out", K.OUT))
        self.fixture = kw.pop("fixture", None)
        self.plan = kw.pop("plan", "smoke" if self.mode == "scratch" else "full")
        self.lanes = int(kw.pop("lanes", K.DEFAULT_LANES))
        self.cpus = kw.pop("cpus", None)
        self.sysfs = kw.pop("sysfs", SYSFS)
        self.meminfo = Path(kw.pop("meminfo", "/proc/meminfo"))
        self.poll = float(kw.pop("poll", 2.0))
        self.stop_grace = float(kw.pop("stop_grace", 120.0))
        self.python = kw.pop("python", K.PYTHON)
        self.phase_main = Path(kw.pop("phase_main", K.CODE / "phase_main.py"))
        self.sizes = dict(kw.pop("sizes", {}) or {})
        self.approval_fn = kw.pop("approval_fn", None)
        self.extra_pythonpath = kw.pop("extra_pythonpath", None)
        self.use_taskset = bool(kw.pop("use_taskset", POSIX))
        self.forecast = dict(kw.pop("forecast", K.FORECAST_CPU_HOURS))
        self.cap = float(kw.pop("cap", K.CAP_CPU_HOURS))
        self.fresh_path = bool(kw.pop("fresh_path", False))
        self.seeds = kw.pop("seeds", None)              # this runner starts the phases of these seeds only
        self.group_only = bool(kw.pop("group_only", False))   # ... or the group phases (lock, look) only
        self.tag = kw.pop("tag", None)                  # this runner's lock and status file name
        self.out_extra = kw
        if self.seeds is not None:
            self.seeds = sorted({int(s) for s in self.seeds})
            for s in self.seeds:
                K.group_of(s)
        if self.mode == "official" and K.HOST == "test":
            raise PH.Refused("refused: the test host never runs an official run")
        if self.mode == "official" and Path(os.path.realpath(str(self.out))) != Path(os.path.realpath(str(K.OUT))):
            raise PH.Refused(f"refused: the official run writes to {K.OUT} only")
        if self.mode in ("stub", "scratch") and LG.inside(self.out, K.ROOT):
            raise PH.Refused(f"refused: {self.mode} outputs must lie outside {K.ROOT}")
        if self.mode == "stub":
            if self.fixture is None or not LG.inside(self.out, self.fixture):
                raise PH.Refused("refused: stub outputs go inside the fixture")
        if self.tag is not None and not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", str(self.tag)):
            raise PH.Refused("refused: --tag takes letters, digits, '_', '.' and '-' only (at most 40)")
        PH.check_plan_spec(self.plan, self.mode)

    def phase_env(self):
        env = dict(K.PHASE_ENV)
        if self.mode == "stub":
            # synthetic fixtures only: keep Python's usual bytecode cache (the prefix makes every phase compile JAX and
            # the gate's code from source, about 3 s each); official and scratch phases keep the prefix
            env.pop("PYTHONPYCACHEPREFIX", None)
        if self.mode == "stub" and self.extra_pythonpath:
            extra = [p for p in str(self.extra_pythonpath).split(os.pathsep) if p]
            if any(LG.inside(p, K.ROOT) for p in extra):
                raise PH.Refused("refused: extra stub import folders must lie outside the repository")
            env["PYTHONPATH"] = os.pathsep.join(extra + [env["PYTHONPATH"]])
        if self.mode != "official":
            for name in ("NEAR80_HOST", "NEAR80_ROOT", "NEAR80_PYTHON", "BCA_SHIFT_DEPS", "SYSTEMROOT", "TEMP", "TMP"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
        elif K.HOST == "cluster":
            for name in ("NEAR80_HOST", "NEAR80_ROOT", "NEAR80_PYTHON", "BCA_SHIFT_DEPS", "NEAR80_MEM_TOTAL_KB"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
        return env

    def approval(self):
        if self.approval_fn is not None:
            return self.approval_fn()
        import approval_check as AC
        cfg = AC.fixture_config(self.fixture) if self.mode == "stub" else None
        return AC.require_approval(cfg)

    def mine(self, key):
        """Does this runner start this phase? Every phase without --seeds and --group-only; with --seeds, the phases
        of those seeds; with --group-only, the group phases (lock, look)."""
        if self.group_only:
            return key.seed is None
        if self.seeds is not None:
            return key.seed is not None and int(key.seed) in self.seeds
        return True


def stub_fixture_memory(fx):
    """A fixture without runner.json gets its own meminfo (24 GiB, as WSL on the desktop) and runner.json, so that a
    stub run never depends on this machine's free memory."""
    fx = Path(fx)
    if not (fx / "runner.json").is_file():
        (fx / "meminfo").write_text(STUB_MEMINFO, encoding="utf8")
        (fx / "runner.json").write_text(json.dumps(dict(meminfo=str(fx / "meminfo"), poll=0.2)), encoding="utf8")


def stub_cfg(fixture, lanes, seeds=None, sizes=None):
    """The configuration of a stub run on a fixture (runner.json inside it may give meminfo, cpus and poll)."""
    fx = Path(os.path.realpath(str(fixture)))
    body = {}
    if (fx / "runner.json").is_file():
        body = json.loads((fx / "runner.json").read_text(encoding="utf8"))
    for k in ("meminfo",):
        if body.get(k) and not LG.inside(os.path.realpath(body[k]), fx):
            raise PH.Refused(f"refused: the stub setting {k} must lie inside the fixture")
    plan = "full" if not seeds else "seeds:" + ",".join(str(int(s)) for s in seeds)
    sz = dict(steps=K.SMOKE_STEPS, chunk=K.SMOKE_CHUNK, updates=K.SMOKE_UPDATES, starts=5)
    sz.update(sizes or {})
    return Cfg(mode="stub", out=fx / "out", fixture=fx, plan=plan, lanes=lanes, cpus=body.get("cpus"),
               meminfo=body.get("meminfo", "/proc/meminfo"), poll=float(body.get("poll", 0.2)), sizes=sz,
               extra_pythonpath=os.environ.get("NEAR80_STUB_PYTHONPATH"))


# --------------------------------------------------------------------------------------------- tasks

class Task:
    def __init__(self, key, deps, prio):
        self.key, self.pid, self.kind = key, PH.phase_id(key), key.kind
        self.deps, self.prio = list(deps), prio
        self.state, self.note = "pending", None
        self.cpu = self.lane = self.proc = self.n = self.started = self.code = None
        self.reserved, self.alone, self.retry_n = 0, False, None
        self.mine = True

    def row(self):
        return dict(state=STATUS_NAME.get(self.state, self.state), note=self.note, plain=PH.plain_name(self.key),
                    cpu=self.cpu, lane=self.lane,
                    attempt=self.n, started=self.started, exit_code=self.code, alone=self.alone,
                    retry_request=self.retry_n, reserved_gib=round(self.reserved / K.GIB, 2), stage=PH.stage(self.key),
                    mine=self.mine)

    def key_fields(self):
        k = self.key
        return dict(phase_kind=k.kind, setting=k.setting, world=k.world, seed=k.seed, group=k.group)


def build_tasks(keys):
    pr = PH.priorities(keys)
    inside = set(keys)
    return [Task(k, [PH.phase_id(d) for d in PH.needs(k, keys) if d in inside], pr[k]) for k in keys]


# --------------------------------------------------------------------------------------------- retries

RETRY_RE = re.compile(r"^(\d+)\.json$")


def retry_dir(out_root, pid):
    return Path(out_root) / "retry" / pid


def retry_requests(out_root, pid):
    d = retry_dir(out_root, pid)
    out = {}
    if d.is_dir():
        for p in d.iterdir():
            m = RETRY_RE.match(p.name)
            if m and LG.ok_record(p) is not None:
                out[int(m.group(1))] = LG.ok_record(p)
    return out


def active_retry(out_root, pid):
    """The first retry request not used yet: no attempt that used it ended other than machine-wide."""
    starts = {n: LG.start_record(out_root, pid, n) or {} for n in LG.attempt_numbers(out_root, pid)}
    for r in sorted(retry_requests(out_root, pid)):
        used = False
        for n, st in starts.items():
            if st.get("retry_request") != r:
                continue
            rec = LG.attempt_record(out_root, pid, n)
            if rec is None or LG.classification(rec) != "machine-wide":
                used = True
        if not used:
            return r
    return None


def need(out_root, pid):
    """('normal' | 'alone' | 'unfinished', retry request or None, phase-specific count)."""
    records = LG.attempt_records(out_root, pid)
    n_ps = LG.phase_specific_count(records)
    if n_ps < 3:
        return "normal", None, n_ps
    if n_ps == 3:
        return "alone", None, n_ps
    r = active_retry(out_root, pid)
    return ("alone", r, n_ps) if r is not None else ("unfinished", None, n_ps)


def write_retry_request(out_root, pid, argv, tty):
    n = max(retry_requests(out_root, pid), default=0) + 1
    body = dict(kind="near80-retry-request", phase_id=pid, request=n, utc=LG.utc(), command_line=list(argv), tty=tty,
                boot_id=LG.boot_id(), rule="one attempt, run alone; the request sets only when the next attempt starts")
    LG.write_json_once(retry_dir(out_root, pid) / f"{n}.json", body)
    return n


# --------------------------------------------------------------------------------------------- the runner

class Runner:
    def __init__(self, cfg, keys, cpus, approval, out=say_line, until_waiting=False):
        self.cfg, self.cpus, self.approval = cfg, list(cpus), approval
        self.keys = list(keys)
        self.tasks = build_tasks(self.keys)
        for t in self.tasks:
            t.mine = cfg.mine(t.key)
        self.by = {t.pid: t for t in self.tasks}
        self.out = out
        self.until_waiting = until_waiting
        self.stop_requested = self.kill_requested = False
        self.halted, self.halt_code = None, None
        self.notes = {}
        self.plumbing_peaks, self.plumbing_seconds = plumbing_facts(cfg.out)
        self.peaks = LG.measured_peaks(cfg.out, self.plumbing_peaks)
        self.hours = LG.cpu_hours(cfg.out, self.plumbing_seconds)
        self.continues = budget_continues(cfg.out)
        self.logfile = cfg.out / ("runner.log" if not cfg.tag else f"runner_{cfg.tag}.log")
        self.paused_pids, self.paused = set(), False

    # ---- output
    def say(self, msg):
        line = f"{LG.utc()} {msg}"
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

    def halt(self, msg, code=EXIT_FAILED):
        if self.halted is None:
            self.halted, self.halt_code = msg, code
        self.say(msg)

    # ---- state
    def log_path(self, t):
        return self.cfg.out / "logs" / f"{t.pid}.log"

    def dep_state(self, pid):
        t = self.by.get(pid)
        return t.state if t is not None else None

    def ready(self, t):
        return all(self.dep_state(d) in FINAL_STATES for d in t.deps)

    def resolve(self, t):
        """(state, why) from the files: done, stopped, skipped (a record, or a rule), or pending."""
        st = PH.done(t.key, self.cfg.out)
        if st in FINAL_STATES:
            rec = LG.ok_record(self.cfg.out / PH.skipped_rel(t.key)) if st == "skipped" else None
            return st, (rec or {}).get("reason")
        states = {self.by[d].key: self.by[d].state for d in t.deps if d in self.by}
        why = PH.rule_skip(t.key, self.cfg.out, self.keys, states=states)
        if why:
            return "skipped", why
        return "pending", None

    def running(self):
        return [t for t in self.tasks if t.proc is not None]

    def refresh_others(self):
        """The phases another runner owns (--seeds, --group-only): their states are read from the files each poll,
        so that this runner's phases can follow them."""
        for t in self.tasks:
            if t.mine or t.state in FINAL_STATES:
                continue
            state, why = self.resolve(t)
            if state != "pending":
                t.state, t.note = state, why

    # ---- start-up: settle attempts whose runner did not survive
    def settle(self, t):
        for n in LG.attempt_numbers(self.cfg.out, t.pid):
            if LG.attempt_record(self.cfg.out, t.pid, n) is not None:
                continue
            if not phase_lock_free(self.cfg.out, t.pid):
                continue
            st = PH.done(t.key, self.cfg.out)
            LG.finish_attempt(self.cfg.out, t.key_fields(), t.pid, n, done=st in ("done", "skipped"),
                              stopped=st == "stopped", exit_code=None, log_path=self.log_path(t))

    def startup(self):
        for t in self.tasks:
            if t.mine:
                self.settle(t)
        for t in self.tasks:
            state, why = self.resolve(t)
            if state != "pending":
                t.state, t.note = state, why
                continue
            if t.mine:
                self.after_records(t)

    def after_records(self, t):
        """Move the files of an interrupted attempt aside (never deleted) before the phase reruns."""
        recs = LG.attempt_records(self.cfg.out, t.pid)
        last = recs[max(recs)] if recs else None
        kind = LG.classification(last)
        files = [r for r in PH.outputs(t.key) if (self.cfg.out / r).is_file()]
        if files and kind in INTERRUPTED + ("failed", "refused"):
            moved = LG.move_interrupted(t.pid, self.cfg.out, f"{kind}: {(last.get('classification') or {}).get('why')}",
                                        PH.outputs(t.key))
            self.say(f"moved {len(moved)} files of the interrupted {t.pid} to {self.cfg.out / 'interrupted' / t.pid} "
                     f"(never deleted)")
        elif files and not recs:
            self.halt(f"HALT: {t.pid} has files that no attempt of this runner wrote: {files[:4]}; nothing was moved")
            t.state = "halted"

    # ---- starting and ending
    def command(self, t, n, cpu):
        cmd = []
        if POSIX:
            cmd = ["env", "-i"] + [f"{k}={v}" for k, v in sorted(self.cfg.phase_env().items())]
            cmd += ["nice", "-n", "5"]
            if self.cfg.use_taskset:
                cmd += ["taskset", "-c", str(cpu)]
        cmd += [self.cfg.python, str(self.cfg.phase_main), "--phase-id", t.pid, "--out", str(self.cfg.out),
                "--mode", self.cfg.mode, "--plan", self.cfg.plan, "--attempt", str(n)]
        if self.cfg.fixture is not None:
            cmd += ["--fixture", str(self.cfg.fixture)]
        for k in ("steps", "chunk", "updates", "starts"):
            if self.cfg.sizes.get(k) is not None:
                cmd += [f"--{k}", str(int(self.cfg.sizes[k]))]
        if self.cfg.fresh_path:
            cmd += ["--fresh-path"]
        return cmd

    def start(self, t, cpu, alone=False, retry_n=None):
        lane = self.cpus.index(cpu) if cpu in self.cpus else None
        n = LG.allocate_attempt(self.cfg.out, t.pid, cpu=cpu, lane=lane, mode=self.cfg.mode, retry_request=retry_n,
                                alone=alone, **t.key_fields())
        t.n, t.cpu, t.lane, t.started, t.state, t.alone, t.retry_n = n, cpu, lane, LG.utc(), "running", alone, retry_n
        t.reserved = 0 if alone else reservation(t.kind, self.peaks)
        logp = self.log_path(t)
        logp.parent.mkdir(parents=True, exist_ok=True)
        with open(logp, "a", encoding="utf8") as handle:
            handle.write(LG.start_line(t.pid, n, cpu) + "\n")
        handle = open(logp, "a", encoding="utf8")
        kw = dict(start_new_session=True) if POSIX else dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                                             env=self.cfg.phase_env())
        t.proc = subprocess.Popen(self.command(t, n, cpu), cwd=LG.work_dir(), stdin=subprocess.DEVNULL, stdout=handle,
                                  stderr=subprocess.STDOUT, **kw)
        handle.close()
        how = " ALONE (nothing else starts)" if alone else ""
        if retry_n:
            how += f", retry request {retry_n}"
        self.say(f"start  {t.pid} on processor {cpu} (attempt {n}{how}; reserves {t.reserved / K.GIB:.1f} GiB)")

    def finish(self, t):
        code = t.proc.returncode
        deadline = time.time() + 30
        while not phase_lock_free(self.cfg.out, t.pid) and time.time() < deadline:
            time.sleep(0.1)
        t.proc, t.code = None, code
        st = PH.done(t.key, self.cfg.out)
        rec = LG.finish_attempt(self.cfg.out, t.key_fields(), t.pid, t.n, done=st in ("done", "skipped"),
                                stopped=st == "stopped", exit_code=code, log_path=self.log_path(t))
        kind, why = LG.classification(rec), (rec.get("classification") or {}).get("why")
        self.peaks = LG.measured_peaks(self.cfg.out, self.plumbing_peaks)
        self.hours = LG.cpu_hours(self.cfg.out, self.plumbing_seconds)
        if kind == "done":
            t.state = st
            if st == "skipped":
                t.note = (LG.ok_record(self.cfg.out / PH.skipped_rel(t.key)) or {}).get("reason")
            self.say(f"{'done  ' if st == 'done' else 'skip  '} {t.pid} (exit {code})" + (f": {t.note}" if t.note
                                                                                         else ""))
            return
        if kind == "stopped":
            t.state = "stopped"
            stop = LG.ok_record(self.cfg.out / PH.stop_rel(t.key)) or {}
            t.note = stop.get("reason")
            self.say(f"STOPPED {t.pid}: {t.note} (written once; never rerun)")
            return
        if kind in INTERRUPTED:
            self.after_records(t)
            mode, _, n_ps = need(self.cfg.out, t.pid)
            t.state = "pending"
            self.say(f"interrupted: {t.pid} ({kind}: {why})" + (f"; phase-specific {n_ps} of 3" if kind ==
                                                                 "phase-specific" else "; it reruns"))
            if mode == "unfinished":
                t.state, t.note = "unfinished", f"{n_ps} phase-specific interruptions; waiting for retry"
                self.say(f"UNFINISHED: {t.pid}. Look at 'tail -n 30 {self.log_path(t)}', then type: {RUN_N80} "
                         f"retry {t.pid}")
            return
        t.state = "halted"
        if kind == "refused":
            self.halt(f"REFUSED (exit 97): {t.pid}: {why}; log {self.log_path(t)}", EXIT_REFUSED)
        else:
            self.halt(f"FAILED: {t.pid} exit {code}: {why}; log {self.log_path(t)}. No new phase starts; nothing was "
                      f"deleted")

    # ---- the second Ctrl-C
    def stop_running(self, sig=signal.SIGTERM, why="second Ctrl-C"):
        for t in self.running():
            try:
                LG.append_signal(self.cfg.out, t.pid, t.n, int(sig), why)
            except OSError:
                pass
            try:
                if POSIX:
                    os.killpg(t.proc.pid, sig)
                else:
                    t.proc.terminate()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            self.say(f"stopped {t.pid} (signal {int(sig)}, recorded in its ledger: a machine-wide interruption; it "
                     f"reruns on the next start)")

    # ---- the loop
    def check_retries(self):
        for t in self.tasks:
            if t.state == "unfinished":
                mode, n, _ = need(self.cfg.out, t.pid)
                if mode == "alone":
                    t.state, t.note = "pending", f"retry request {n} picked up"
                    self.say(f"retry request {n} for {t.pid} picked up: it runs alone when nothing else is runnable")

    def schedule(self):
        if self.halted or self.stop_requested:
            return
        run = self.running()
        if any(t.alone for t in run):
            return
        self.continues = budget_continues(self.cfg.out)
        paused, over = budget(self.hours, self.cfg.forecast, K.PAUSE_FACTOR, self.cfg.cap, self.continues)
        self.paused_pids = set()
        if over:
            self.once("over", over)
            for t in self.tasks:
                if t.state == "pending" and t.mine:
                    t.note = over
                    self.paused_pids.add(t.pid)
            return
        total, avail = meminfo(self.cfg.meminfo, sum(r.reserved for r in run))
        cands = []
        for t in sorted(self.tasks, key=lambda x: -x.prio):
            if t.state != "pending" or not t.mine or not self.ready(t):
                continue
            if not phase_lock_free(self.cfg.out, t.pid):
                self.once(f"elsewhere {t.pid}", f"waiting: {t.pid} is running outside this runner (its lock is held)")
                continue
            state, why = self.resolve(t)
            if state != "pending":
                t.state, t.note = state, why
                self.say(f"{'found ' if state == 'done' else 'skip  '} {t.pid}" + (f": {why}" if why else ""))
                continue
            ok, why = PH.admit(t.key, self.approval, None, self.cfg.mode)
            if not ok:
                t.state = "halted"
                self.halt(f"REFUSED: {t.pid}: {why}", EXIT_REFUSED)
                return
            stage = PH.stage(t.key)
            if stage in paused:
                t.note = paused[stage]
                self.paused_pids.add(t.pid)
                self.once(f"paused {stage}", paused[stage])
                continue
            mode, n, n_ps = need(self.cfg.out, t.pid)
            if mode == "unfinished":
                t.state, t.note = "unfinished", f"{n_ps} phase-specific interruptions; waiting for retry"
                self.once(f"unfinished {t.pid}", f"UNFINISHED: {t.pid}; type: {RUN_N80} retry {t.pid}")
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
                break
            own = reservation(t.kind, self.peaks)
            if own > total - K.HEADROOM_GIB * K.GIB:
                t.state = "halted"
                self.halt(f"HALT: {t.pid} reserves {own / K.GIB:.1f} GiB, more than MemTotal - {K.HEADROOM_GIB} GiB = "
                          f"{(total - K.HEADROOM_GIB * K.GIB) / K.GIB:.1f} GiB; it can never start (give the machine "
                          f"more memory). Nothing was deleted")
                return
            ok, why = admit(t.kind, own, [(r.kind, r.reserved) for r in self.running()], total, avail, self.peaks)
            if not ok:
                self.once(f"memory {t.pid}", f"waiting: memory ({t.pid}: {why})")
                continue
            self.start(t, free[0])
            busy.add(free[0])
            avail -= own
        if not normal and alone and not self.running():
            t, _, n = alone[0]
            self.start(t, self.cpus[0], alone=True, retry_n=n)

    def status_body(self):
        total, avail = meminfo(self.cfg.meminfo, sum(r.reserved for r in self.running()))
        paused, over = budget(self.hours, self.cfg.forecast, K.PAUSE_FACTOR, self.cfg.cap, self.continues)
        counts = {}
        for t in self.tasks:
            name = STATUS_NAME.get(t.state, t.state)
            counts[name] = counts.get(name, 0) + 1
        return dict(kind="near80-runner-status", updated=LG.utc(), mode=self.cfg.mode, plan=self.cfg.plan,
                    lanes=len(self.cpus), cpus=self.cpus, host=K.HOST, tag=self.cfg.tag, seeds=self.cfg.seeds,
                    group_only=self.cfg.group_only,
                    memory=dict(total_gib=round(total / K.GIB, 2), available_gib=round(avail / K.GIB, 2),
                                reserved_gib=round(sum(t.reserved for t in self.running()) / K.GIB, 2),
                                measured_peaks_gib={k: round(v / K.GIB, 3) for k, v in sorted(self.peaks.items())}),
                    cpu_hours={k: round(v, 3) for k, v in sorted(self.hours.items())}, paused=paused, over_cap=over,
                    budget_continues=dict(self.continues), paused_run=bool(self.paused),
                    halted=self.halted, stop_requested=self.stop_requested, counts=counts,
                    skipped={t.pid: t.note for t in self.tasks if t.state == "skipped"},
                    phases={t.pid: t.row() for t in self.tasks})

    def write_status(self):
        try:
            LG.write_json_replace(self.cfg.out / status_name(self.cfg.tag), self.status_body())
        except OSError:
            pass

    def mine_all(self):
        return [t for t in self.tasks if t.mine]

    def finished_all(self):
        return all(t.state in FINAL_STATES for t in self.mine_all())

    def run(self):
        killed_at = None
        while True:
            for t in self.running():
                if t.proc.poll() is not None:
                    self.finish(t)
            if self.kill_requested and killed_at is None:
                self.stop_running()
                killed_at = time.time()
            elif killed_at is not None and time.time() - killed_at > self.cfg.stop_grace and self.running():
                self.stop_running(signal.SIGKILL if POSIX else signal.SIGTERM,
                                  "second Ctrl-C; still running after the grace period")
                killed_at = time.time() + 10 ** 6
            self.refresh_others()
            self.check_retries()
            self.schedule()
            self.write_status()
            if not self.running():
                if self.halted or self.stop_requested or self.finished_all():
                    break
                waiting = [t for t in self.mine_all() if t.state == "unfinished"]
                startable = [t for t in self.mine_all() if t.state == "pending" and self.ready(t)]
                others_pending = [t for t in self.tasks if not t.mine and t.state not in FINAL_STATES]
                if not startable:
                    if waiting:
                        self.once("waiting", f"WAITING FOR RETRY: {', '.join(t.pid for t in waiting)}. For each, "
                                             f"type: {RUN_N80} retry <phase>")
                        break
                    if others_pending:
                        self.once("others", f"waiting: {len(others_pending)} phases of other runners are not finished "
                                            f"({others_pending[0].pid})")
                        time.sleep(self.cfg.poll)
                        continue
                    stuck = [t.pid for t in self.mine_all() if t.state not in FINAL_STATES]
                    self.halt(f"HALT: no phase can start and the plan is not finished ({stuck[:5]})")
                    break
                if all(t.pid in self.paused_pids for t in startable):
                    self.say("PAUSED (budget): " + "; ".join(sorted({t.note for t in startable if t.note})))
                    self.paused = True
                    break
                if self.until_waiting and not any(t.state == "running" for t in self.tasks):
                    self.once("stuck", f"waiting: {len(startable)} phases cannot start now ({startable[0].pid})")
            time.sleep(self.cfg.poll)
        self.write_status()
        return self.verdict()

    def verdict(self):
        mine = self.mine_all()
        c = {s: sum(1 for t in mine if t.state == s) for s in FINAL_STATES}
        unfinished = sorted(t.pid for t in mine if t.state == "unfinished")
        tail = f"unfinished: {', '.join(unfinished)}" if unfinished else "unfinished: none"
        if self.halted:
            self.say(f"runner ends (halted): {c['done']} done, {c['stopped']} stopped, {c['skipped']} skipped; {tail}")
            return self.halt_code or EXIT_FAILED
        if self.finished_all():
            self.say(f"runner ends: every phase done, stopped or skipped ({c['done']} done, {c['stopped']} stopped, "
                     f"{c['skipped']} skipped)")
            return EXIT_FINISHED
        if self.paused and not self.stop_requested:
            self.say(f"runner ends PAUSED by the budget guard: {c['done']} done, "
                     f"{sum(1 for t in mine if t.state == 'pending')} not started; {tail}")
            return EXIT_PAUSED
        if unfinished and not self.stop_requested:
            self.say(f"runner ends while WAITING FOR RETRY: {tail}")
            return EXIT_RETRY
        self.say(f"runner ends (stopped safely; nothing failed): {c['done']} done, "
                 f"{sum(1 for t in mine if t.state == 'pending')} not started; {tail}")
        return EXIT_STOPPED


def plumbing_facts(out_root):
    """(first peaks {kind: vmhwm_kb}, CPU seconds) from OUT/plumbing/plumbing.json, when it exists and verifies."""
    rec = LG.ok_record(Path(out_root) / K.LAYOUT["plumbing"])
    if not rec:
        return {}, None
    return dict(rec.get("peaks") or {}), rec.get("cpu_seconds", rec.get("seconds"))


def plumbing_peaks(smoke_peaks, repro_kb):
    """The reservations the plumbing seeds (PREREGISTRATION.md section 13): the reproduction check runs the critic,
    second critic, switch, clone, reference-actor and policy code at full length in one process, so its peak serves
    those kinds (and the full-log critic); every other kind's 40-step smoke peak is raised to at least
    SMOKE_FLOOR_GIB / 1.25, so that it reserves at least SMOKE_FLOOR_GIB. Either counts only until an official
    attempt of the kind has been measured (ledger.measured_peaks). {kind: vmhwm_kb}."""
    floor_kb = int(math.ceil(K.SMOKE_FLOOR_GIB * K.GIB / K.MEASURED_FACTOR / 1024))
    out = {kind: max(int(kb), floor_kb) for kind, kb in (smoke_peaks or {}).items()
           if kb and kind not in K.REPRO_PEAK_KINDS}
    if repro_kb:
        out.update({kind: int(repro_kb) for kind in K.REPRO_PEAK_KINDS})
    return out


def run_locked(cfg, keys, approval, out=say_line, until_waiting=False, signals=True):
    """Hold OUT/runner<tag>.lock and run the plan. Returns the exit code."""
    cfg.out.mkdir(parents=True, exist_ok=True)
    for sub in ("logs", "ledger", "locks", "retry"):
        (cfg.out / sub).mkdir(parents=True, exist_ok=True)
    with open(cfg.out / runner_lock_name(cfg.tag), "a+") as lock:
        try:
            LG.lock_exclusive(lock)
        except LG.LockHeld:
            out(f"refused: another near-80 runner with this tag is running on this output root "
                f"({runner_lock_name(cfg.tag)}); type {RUN_N80} status to see it")
            return EXIT_REFUSED
        if cfg.seeds is None and not cfg.group_only and runner_running(cfg.out):
            others = [p.name for p in cfg.out.glob("runner*.lock") if not lock_free(p) and p.name !=
                      runner_lock_name(cfg.tag)]
            if others:
                out(f"refused: other runners hold this output root ({others}); a whole-plan runner runs alone")
                return EXIT_REFUSED
        held = [p for p in held_phase_locks(cfg.out) if cfg.mine(PH.parse_id(p))]
        if held:
            out(f"refused: phases of this runner's share are still running ({', '.join(held[:5])}); wait for them to "
                f"end")
            return EXIT_REFUSED
        cpus, why = lane_cpus(cfg.lanes, cfg.sysfs, given=cfg.cpus)
        if why:
            out(f"refused: {why}")
            return EXIT_REFUSED
        runner = Runner(cfg, keys, cpus, approval, out=out, until_waiting=until_waiting)
        share = ("" if cfg.seeds is None and not cfg.group_only else
                 f"; this runner: {'the group phases' if cfg.group_only else 'seeds ' + ','.join(map(str, cfg.seeds))}")
        runner.say(f"plan: {len(keys)} phases ({cfg.plan}, {cfg.mode}); lanes {len(cpus)} on processors {cpus}{share}")
        old = {}

        def on_signal(signum, frame):
            if not runner.stop_requested:
                runner.stop_requested = True
                runner.say("stop requested: no new phase starts; the running ones finish first. Press Ctrl-C again "
                           "to stop them now (they rerun on the next start)")
            elif not runner.kill_requested:
                runner.kill_requested = True
                runner.say("second Ctrl-C: stopping the running phases")

        if signals:
            names = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
            old = {s: signal.getsignal(s) for s in names}
            signal.signal(signal.SIGINT, on_signal)
            signal.signal(signal.SIGTERM, on_signal)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            runner.startup()
            return runner.run()
        except (PH.Refused, LG.RecordError) as error:
            runner.say(str(error) if str(error).startswith("refused") else f"refused: {error}")
            return EXIT_REFUSED
        finally:
            for s, h in old.items():
                signal.signal(s, h)


# --------------------------------------------------------------------------------------------- checks of a start

def refuse_reproduction_differs(out_root):
    p = Path(out_root) / K.LAYOUT["reproduction_differs"]
    if p.exists():
        raise PH.Refused(f"refused: {p} exists: the new code did not reproduce the saved files, so nothing official "
                         f"starts in this version (section 8.2). Type: {RUN_N80} report")


def require_plumbing(out_root, approval):
    """'run' refuses unless the plumbing record exists, verifies, passed and carries the current fingerprint, and
    no reproduction difference was recorded."""
    refuse_reproduction_differs(out_root)
    path = Path(out_root) / K.LAYOUT["plumbing"]
    try:
        rec = LG.read_json_verified(path)
    except LG.RecordError as error:
        raise PH.Refused(f"refused: the plumbing has not passed yet ({error}); type: {RUN_N80} plumbing "
                         f"--lanes 10") from None
    if rec.get("kind") != "near80-plumbing" or rec.get("passed") is not True:
        raise PH.Refused("refused: the plumbing did not pass; nothing official starts")
    if rec.get("prereg_sha256") != approval.get("prereg_sha256"):
        raise PH.Refused("refused: the plumbing record was made under another pre-registration")
    return rec


def cmd_budget_continue(cfg, stage, out=say_line, isatty=None, argv=None):
    """One typed continuation of a budget pause: OUT/budget/<stage>/<n>.json, write-once. Each raises that stage's
    pause point and the cap by one forecast of the stage; no result is read differently because of it."""
    isatty = sys.stdin.isatty() if isatty is None else isatty
    if not isatty:
        out("refused: budget-continue must be typed at a terminal (standard input is not one)")
        return EXIT_REFUSED
    if stage not in K.BUDGET_STAGES:
        out(f"refused: the stage must be one of {', '.join(K.BUDGET_STAGES)}")
        return EXIT_REFUSED
    d = cfg.out / "budget" / stage
    n = budget_continues(cfg.out).get(stage, 0) + 1
    while (d / f"{n}.json").exists():
        n += 1
    hours = LG.cpu_hours(cfg.out, plumbing_facts(cfg.out)[1])
    LG.write_json_once(d / f"{n}.json", dict(kind="near80-budget-continue", stage=stage, n=n, utc=LG.utc(),
                                             tty=tty_name(), command_line=list(argv if argv is not None else sys.argv),
                                             cpu_hours=hours, forecast=K.FORECAST_CPU_HOURS.get(stage),
                                             rule="one more forecast of this stage; no result is read differently"))
    out(f"budget-continue {n} for the {stage.replace('_', ' ')} stage recorded ({d / f'{n}.json'}): it allows one more "
        f"forecast ({K.FORECAST_CPU_HOURS.get(stage):g} CPU-hours) of this stage and of the cap. " +
        ("The running runner picks it up." if runner_running(cfg.out) else
         f"Start the run again: {RUN_N80} run --lanes 10"))
    return EXIT_FINISHED


def refuse_identity(repo=K.ROOT):
    import approval_check as AC
    try:
        AC.refuse_git_identity(repo)
    except AC.ApprovalRefused as error:
        raise PH.Refused(str(error) if str(error).startswith("refused") else f"refused: {error}") from None


def refuse_root_user():
    """The cluster entry: never root. The desktop entry runs as root inside WSL (version 7's rule); the test host
    never runs an official phase."""
    if K.HOST == "cluster" and hasattr(os, "geteuid") and os.geteuid() == 0:
        raise PH.Refused("refused: the cluster entry never runs as root")
    if K.HOST == "cluster" and os.path.realpath(str(K.ROOT)) == os.path.realpath(K.DESKTOP_ROOT):
        raise PH.Refused(f"refused: the cluster entry runs from a clone other than {K.DESKTOP_ROOT}")
    if K.HOST == "cluster" and not LG.inside(K.ROOT, K.CLUSTER_SCRATCH):
        raise PH.Refused(f"refused: on the cluster the clone must lie under {K.CLUSTER_SCRATCH} (got {K.ROOT})")


def approval_or_refuse(cfg):
    try:
        return cfg.approval()
    except PH.Refused:
        raise
    except Exception as error:  # noqa: BLE001 - approval_check.ApprovalRefused and its kin
        text = str(error)
        raise PH.Refused(text if text.startswith("refused") else f"refused: approval: {text}") from None


# --------------------------------------------------------------------------------------------- the self-test

def selftest(cfg, for_cmd=None, out=say_line, checks=None):
    """The short self-test. Returns True when READY. checks: {name: callable} replaces the defaults (tests)."""
    fails = []
    checks = checks if checks is not None else default_checks(cfg, for_cmd)
    for name, fn in checks.items():
        try:
            detail = fn()
            out(f"PASS {name}" + (f": {detail}" if detail else ""))
        except Exception as error:  # noqa: BLE001 - every check reports, none stops the others
            fails.append(name)
            out(f"FAIL {name}: {error}")
    out("READY" if not fails else f"NOT READY ({', '.join(fails)})")
    return not fails


def _run(cmd, env=None, timeout=300):
    r = subprocess.run(cmd, env=env, cwd=LG.work_dir(), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])}... exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout.strip()


def _clean(env, cmd):
    """A command under exactly `env` (env -i where it exists)."""
    return ["env", "-i"] + [f"{k}={v}" for k, v in env.items()] + cmd if POSIX else cmd


def default_checks(cfg, for_cmd):
    env = cfg.phase_env()
    cpus, why = lane_cpus(cfg.lanes, cfg.sysfs, given=cfg.cpus)
    out = {}

    def python():
        if cfg.python != K.PYTHON:
            raise RuntimeError(f"the phases' Python is {cfg.python}, not {K.PYTHON}")
        v = _run(_clean(env, [cfg.python, "-c", "import numpy; print(numpy.__version__)"]), env=None if POSIX else env)
        if v != "2.2.6":
            raise RuntimeError(f"numpy is {v}, not 2.2.6")
        return f"{cfg.python}, numpy {v}"

    def jax_cpu():
        v = _run(_clean(env, [cfg.python, "-c", "import jax; print(jax.devices()[0].platform)"]),
                 env=None if POSIX else env)
        if v != "cpu":
            raise RuntimeError(f"JAX runs on {v}")
        return "JAX on the processor"

    out["host"] = lambda: (refuse_root_user() or f"{K.HOST}: {K.ROOT}")
    out["versions"] = python
    out["jax"] = jax_cpu
    if for_cmd in ("run", "plumbing", "retry"):
        out["approval"] = lambda: (approval_or_refuse(cfg) or {}).get("prefix")
    else:
        out["code manifest and pins"] = manifest_and_pins

    def placement():
        if K.HOST != "desktop":
            return "not the desktop: version 6's code is read from the clone's own files (no placement)"
        _run(["env", "-i", "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "HOME=/root",
              "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "/usr/bin/python3", "-I", str(K.V7_PLACE), "place"])
        return "version 6's code and the stand-ins are placed (version 7's placement)"

    def one_cpu():
        if why:
            raise RuntimeError(why)
        if not POSIX:
            return "no taskset on this platform (never an official run)"
        for c in cpus:
            v = _run(["taskset", "-c", str(c), sys.executable, "-c",
                      "import os; print(len(os.sched_getaffinity(0)))"])
            if v != "1":
                raise RuntimeError(f"processor {c}: {v} processors visible through taskset")
        return f"one processor per lane on {cpus}"

    def writable():
        cfg.out.mkdir(parents=True, exist_ok=True)
        p = cfg.out / f".selftest_{os.getpid()}"
        p.write_text("ok")
        p.unlink()
        return str(cfg.out)

    def gate_record():
        p = K.ROOT / "runs" / "shift" / "pre" / "data" / "base" / "stall" / "seed0" / "record.json"
        if not p.is_file():
            raise RuntimeError(f"{p} is missing")
        return "the gate's data record of seed 0 exists"

    def room_code():
        p = K.ROOT / "runs" / "shift_explore" / "room" / "code" / "harness.py"
        if not p.is_file():
            raise RuntimeError(f"{p} is missing (the room harness lives under runs/, which git does not hold)")
        return "the room harness is in place"

    def step2_saved():
        d = K.S2_OUT / "policy"
        missing = [f"seed{s}/{w}/{f}" for s in K.SMOKE_SEEDS for w in K.WORLDS
                   for f in ("lock.json", "lock.json.sha256", "actors.msgpack", "values.json")
                   if not (d / f"seed{s}" / w / f).is_file()]
        if missing:
            raise RuntimeError(f"step 2's saved actors of seed 0 are missing under {d}: {missing[:6]}")
        if not K.S2_RESULTS.is_file() or not K.S1_RESULTS.is_file():
            raise RuntimeError(f"step 1's or step 2's committed results are missing ({K.S1_RESULTS}, {K.S2_RESULTS})")
        return "step 2's saved actors of seed 0 (both worlds) and step 1's and step 2's results"

    def gate_inputs():
        """The gate's saved files the reproduction check reads on seed 0: the critics (gap removal, both worlds; the
        full log, stalling world), the cloned actors and check 2's actors."""
        from experiments.shift import acceptance as A
        p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        need = [A.critic_dir(p, K.VARIANT, w, "gap", K.CONTROLLER, False, 0) / f for w in K.WORLDS
                for f in ("record.json", "critic.msgpack")]
        need += [A.critic_dir(p, K.VARIANT, "stall", "none", K.CONTROLLER, False, 0) / f
                 for f in ("record.json", "critic.msgpack")]
        need += [A._variant_root(p, "clone", K.VARIANT) / w / "seed0" / f for w in K.WORLDS
                 for f in ("record.json", "actor.msgpack")]
        need += [A._variant_root(p, "harm_return", K.VARIANT) / f"alpha{K.ALPHA:g}" / "seed0" / f
                 for f in ("values.json", "actors.msgpack")]
        missing = [str(q) for q in need if not Path(q).is_file()]
        if missing:
            raise RuntimeError(f"the gate's saved files the reproduction check reads are missing: {missing[:4]}")
        return f"the gate's saved critics, cloned actors and check 2's actors of seed 0 ({len(need)} files)"

    out["placement"] = placement
    out["one processor per lane"] = one_cpu
    out["git identity"] = refuse_identity
    out["output root writable"] = writable
    out["gate data"] = gate_record
    out["room harness"] = room_code
    if for_cmd in ("run", "plumbing"):
        out["step 2's saved actors"] = step2_saved
    if for_cmd == "plumbing":
        out["the gate's saved files for the reproduction check"] = gate_inputs
    return out


def manifest_and_pins():
    import approval_check as AC
    cfg = AC.ApprovalConfig()
    pr = AC.load_prereg(cfg)
    man = AC.check_code(pr, cfg)
    n = AC.check_pins(man, cfg)
    return f"prereg.json {pr.prefix}, {n} pinned repository files unchanged"


# --------------------------------------------------------------------------------------------- commands

def cmd_run(cfg, out=say_line):
    try:
        approval = approval_or_refuse(cfg)
        if cfg.mode == "official":
            refuse_root_user()
            refuse_identity()
            require_plumbing(cfg.out, approval)
    except PH.Refused as error:
        out(str(error))
        return EXIT_REFUSED
    return run_locked(cfg, PH.plan(spec=cfg.plan), approval, out)


def cmd_stub(fixture, lanes, seeds=None, out=say_line, sizes=None, refuse=None):
    fx = Path(os.path.realpath(str(fixture)))
    if LG.inside(fx, K.ROOT):
        out(f"refused: a test fixture must lie outside {K.ROOT}")
        return EXIT_REFUSED
    try:
        if not (fx / K.TEST_MARKER).is_file():
            if fx.exists() and any(fx.iterdir()):
                out(f"refused: {fx} is neither empty nor a near-80 test fixture")
                return EXIT_REFUSED
            import approval_check as AC
            AC.write_test_fixture(fx)
        if refuse:
            import stubs
            p = fx / stubs.STUB_FILE
            if not p.exists():
                p.write_text(json.dumps(dict(refuse_seeds=[int(s) for s in refuse])), encoding="utf8")
        stub_fixture_memory(fx)
        cfg = stub_cfg(fx, lanes, seeds, sizes)
        env = cfg.phase_env()
        if not (fx / "step2" / "results.json").exists():
            cpus, why = lane_cpus(1, cfg.sysfs, given=cfg.cpus)
            if why:
                out(f"refused: {why}")
                return EXIT_REFUSED
            cmd = [cfg.python, str(K.CODE / "stubs.py"), "step2-fixture", str(fx)]
            if POSIX:
                cmd = ["env", "-i"] + [f"{k}={v}" for k, v in env.items()] + ["taskset", "-c", str(cpus[0])] + cmd
            _run(cmd, env=None if POSIX else env, timeout=1800)
        approval = approval_or_refuse(cfg)
    except PH.Refused as error:
        out(str(error))
        return EXIT_REFUSED
    except RuntimeError as error:
        out(f"refused: the stub fixture could not be prepared: {error}")
        return EXIT_REFUSED
    return run_locked(cfg, PH.plan(spec=cfg.plan), approval, out)


def run_tests(env, out=say_line):
    res = {}
    for name in K.TEST_FILES:
        t0 = time.perf_counter()
        cmd = [K.PYTHON, str(K.CODE / name)]
        r = subprocess.run(_clean(env, cmd), env=None if POSIX else env, cwd=LG.work_dir(), stdin=subprocess.DEVNULL,
                           capture_output=True, text=True)
        res[name] = dict(exit_code=r.returncode, seconds=time.perf_counter() - t0, tail=(r.stderr or "")[-2000:])
        out(f"{'PASS' if r.returncode == 0 else 'FAIL'} {name} ({res[name]['seconds']:.0f}s)")
    return res


def cmd_plumbing(cfg, out=say_line, tests_fn=None, smoke_fn=None, repro_fn=None, scratch=None, signals=True):
    """Scratch only (section 8.2): the unit tests, then (if they passed) the smoke of every phase kind on seed 0 (40
    steps) and the fresh-seed code path on seed 0, then (if those passed) the reproduction check (full length).
      passed                          OUT/plumbing/plumbing.json (write-once): 'run' may start.
      the reproduction check differed  OUT/plumbing/reproduction_differs.json (write-once): nothing official ever starts
                                       in this version; 'plumbing' and 'run' refuse for good.
      any other failure               OUT/plumbing/failed_<utc>.json (kept): the cause and the logs; once the cause is
                                       fixed, 'plumbing' may be run again (in a new scratch folder)."""
    t0 = time.perf_counter()
    try:
        approval = approval_or_refuse(cfg)
        refuse_reproduction_differs(cfg.out)
        if cfg.mode == "official":
            refuse_root_user()
            refuse_identity()
    except PH.Refused as error:
        out(str(error))
        return EXIT_REFUSED
    path = cfg.out / K.LAYOUT["plumbing"]
    if path.exists():
        out(f"refused: {path} exists: the plumbing has passed; nothing to do. Start the run: {RUN_N80} run --lanes 10")
        return EXIT_REFUSED
    scratch = Path(scratch) if scratch else PLUMBING_SCRATCH / LG.utc_stamp()
    if LG.inside(scratch, K.ROOT):
        out("refused: the plumbing's scratch folder must lie outside the repository")
        return EXIT_REFUSED
    if scratch.exists() and any(scratch.iterdir()):
        out(f"refused: the plumbing's scratch folder {scratch} is not empty")
        return EXIT_REFUSED
    state = dict(stop=0)
    old = {}

    def on_signal(signum, frame):
        state["stop"] += 1
        out("stop requested: the plumbing stops after the running part (nothing after it starts); it is recorded as "
            "interrupted and may be run again")

    with open(_ensure(cfg.out) / runner_lock_name(cfg.tag), "a+") as lock:
        try:
            LG.lock_exclusive(lock)
        except LG.LockHeld:
            out("refused: another near-80 runner is running on this output root")
            return EXIT_REFUSED
        if signals:
            names = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGHUP] if hasattr(signal, "SIGHUP") else [])
            old = {sg: signal.getsignal(sg) for sg in names}
            signal.signal(signal.SIGINT, on_signal)
            signal.signal(signal.SIGTERM, on_signal)
            if hasattr(signal, "SIGHUP"):
                signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            env = cfg.phase_env()
            tests = (tests_fn or run_tests)(env, out)
            tests_ok = bool(tests) and all(v.get("exit_code") == 0 for v in tests.values())
            sm = dict(smoke={}, peaks={}, fresh={}, codes=[], cpu_seconds=0.0)
            if tests_ok and not state["stop"]:
                sm.update((smoke_fn or run_smoke)(cfg, scratch, approval, out, signals=signals))
            smoke_ok = bool(sm["smoke"]) and all(sm["smoke"].values()) and bool(sm["fresh"]) and \
                all(sm["fresh"].values())
            if EXIT_STOPPED in sm["codes"]:
                state["stop"] += 1
            repro, rrec = {}, {}
            if smoke_ok and not state["stop"]:
                repro, rrec = (repro_fn or run_reproduction)(cfg, scratch / "reproduction", out)
        finally:
            for sg, h in old.items():
                signal.signal(sg, h)
    differs = [(w, name, v) for w, items in (repro or {}).items() for name, v in (items or {}).items()
               if not (v or {}).get("passed")]
    passed = bool(tests_ok and smoke_ok and repro and not differs and not state["stop"])
    cpu = sum(float(v.get("seconds") or 0.0) for v in tests.values()) + float(sm.get("cpu_seconds") or 0.0) + \
        float((rrec or {}).get("seconds") or 0.0)
    rec = dict(tests=tests, smoke=sm["smoke"], fresh_path=sm["fresh"], reproduction=repro,
               smoke_peaks=sm["peaks"], reproduction_vmhwm_kb=(rrec or {}).get("vmhwm_kb"),
               peaks=plumbing_peaks(sm["peaks"], (rrec or {}).get("vmhwm_kb")),
               prereg_sha256=approval.get("prereg_sha256"), scratch=str(scratch), seconds=time.perf_counter() - t0,
               cpu_seconds=cpu, host=K.HOST, written=LG.utc())
    if passed:
        LG.write_json_once(path, dict(rec, kind="near80-plumbing", passed=True))
        out(f"plumbing PASSED: {path}. Start the run: {RUN_N80} run --lanes 10")
        return EXIT_FINISHED
    if tests_ok and smoke_ok and repro and differs and not state["stop"]:
        dpath = cfg.out / K.LAYOUT["reproduction_differs"]
        LG.write_json_once(dpath, dict(rec, kind="near80-reproduction-differs", passed=False))
        for w, name, v in differs:
            out(f"  reproduction differs: {(v or {}).get('plain') or name} ({K.WORLD_PLAIN.get(w, w)})")
        out(f"plumbing FAILED: the reproduction check differed ({dpath}). By section 8.2 nothing official starts in "
            f"this version. Type: {RUN_N80} report")
        return EXIT_FAILED
    if state["stop"]:
        cause = "interrupted (Ctrl-C)"
    elif not tests_ok:
        cause = "a unit test failed: " + ", ".join(k for k, v in tests.items() if v.get("exit_code") != 0)
    elif not smoke_ok:
        bad = [k for k, ok in sm["smoke"].items() if not ok] + [f"fresh path {w}" for w, ok in sm["fresh"].items()
                                                                 if not ok]
        cause = "the smoke did not finish every phase: " + ", ".join(bad[:8]) + (" ..." if len(bad) > 8 else "")
    else:
        cause = "the reproduction check did not finish (no verdict); see its log"
    fpath = cfg.out / K.LAYOUT["plumbing_failed"].format(utc=LG.utc_stamp())
    k = 1
    while fpath.exists():
        fpath = cfg.out / K.LAYOUT["plumbing_failed"].format(utc=f"{LG.utc_stamp()}_{k}")
        k += 1
    LG.write_json_once(fpath, dict(rec, kind="near80-plumbing-failed", passed=False, cause=cause,
                                   logs=dict(smoke=str(scratch / "smoke" / "logs"),
                                             fresh_path=str(scratch / "fresh_path" / "logs"),
                                             reproduction=str(scratch / "reproduction" / "reproduction.log")),
                                   rule="not a reproduction difference: the plumbing may be run again once the cause "
                                        "is fixed (section 8.2)"))
    out(f"plumbing FAILED: {cause}. Recorded in {fpath}; logs under {scratch}. Nothing official was written. Once the "
        f"cause is fixed, run it again: {RUN_N80} plumbing --lanes 10")
    return EXIT_STOPPED if state["stop"] else EXIT_FAILED


def _ensure(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def _attempt_seconds(out_root):
    return sum(float(r.get("seconds") or 0.0) for r in LG.all_attempt_records(out_root))


def run_smoke(cfg, scratch, approval, out=say_line, signals=True):
    """The smoke (scratch, 40 steps, seed 0, every kind) and the fresh-seed path. Returns {smoke: {phase id: ok},
    peaks: {kind: vmhwm_kb}, fresh: {world: ok}, codes: [the two runners' exit codes], cpu_seconds}."""
    sizes = dict(steps=K.SMOKE_STEPS, chunk=K.SMOKE_CHUNK, updates=K.SMOKE_UPDATES, starts=K.SMOKE_STARTS)
    scfg = Cfg(mode="scratch", out=scratch / "smoke", plan="smoke", lanes=cfg.lanes, cpus=cfg.cpus,
               meminfo=cfg.meminfo, poll=cfg.poll, sizes=sizes, approval_fn=lambda: approval)
    keys = PH.plan(spec="smoke")
    codes = [run_locked(scfg, keys, approval, out, until_waiting=True, signals=signals)]
    smoke = {PH.phase_id(k): PH.done(k, scfg.out) == "done" for k in keys}
    peaks = {}
    for rec in LG.all_attempt_records(scfg.out):
        if LG.classification(rec) == "done" and rec.get("vmhwm_kb"):
            peaks[rec["phase_kind"]] = max(peaks.get(rec["phase_kind"], 0), int(rec["vmhwm_kb"]))
    fresh = {}
    fcfg = Cfg(mode="scratch", out=scratch / "fresh_path", plan="smoke", lanes=min(2, cfg.lanes), cpus=cfg.cpus,
               meminfo=cfg.meminfo, poll=cfg.poll, sizes=sizes, approval_fn=lambda: approval, fresh_path=True)
    if codes[0] != EXIT_STOPPED:
        fkeys = [PH.make_key("data_main", "main", w, 0) for w in K.WORLDS]
        codes.append(run_locked(fcfg, fkeys, approval, out, until_waiting=True, signals=signals))
        for k in fkeys:
            rec = LG.ok_record(fcfg.out / PH.record_rel(k)) or {}
            fresh[k.world] = bool((rec.get("fresh_path_reproduces_gate") or {}).get("passed"))
    cpu = _attempt_seconds(scfg.out) + (_attempt_seconds(fcfg.out) if fcfg.out.exists() else 0.0)
    return dict(smoke=smoke, peaks=peaks, fresh=fresh, codes=codes, cpu_seconds=cpu)


def run_reproduction(cfg, folder, out=say_line):
    """The reproduction check in its own process (phase_main.py --reproduction). Returns (its items, its record), or
    ({}, {}) when it did not finish (not a reproduction difference)."""
    cpus, why = lane_cpus(1, cfg.sysfs, given=cfg.cpus)
    if why:
        out(f"FAIL the reproduction check could not start: {why}")
        return {}, {}
    cmd = [cfg.python, str(cfg.phase_main), "--reproduction", "--out", str(folder), "--mode", "scratch"]
    if POSIX:
        cmd = ["env", "-i"] + [f"{k}={v}" for k, v in cfg.phase_env().items()] + ["nice", "-n", "5", "taskset", "-c",
                                                                                  str(cpus[0])] + cmd
    _ensure(folder)
    with open(folder / "reproduction.log", "a", encoding="utf8") as handle:
        r = subprocess.run(cmd, cwd=LG.work_dir(), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                           env=None if POSIX else cfg.phase_env())
    rec = LG.ok_record(folder / "reproduction.json")
    if r.returncode != 0 or rec is None or not rec.get("items"):
        out(f"FAIL the reproduction check did not finish (exit {r.returncode}); log {folder / 'reproduction.log'}")
        return {}, {}
    vm = rec.get("vmhwm_mb")
    return rec["items"], dict(seconds=rec.get("seconds"), vmhwm_kb=None if vm is None else int(float(vm) * 1024))


def cmd_status(out_root, out=say_line):
    root = Path(out_root)
    alive = runner_running(root)
    files = sorted(root.glob("status*.json")) if root.is_dir() else []
    if not files:
        out(f"no status yet ({root}); runner running: {'yes' if alive else 'no'}")
        return 0
    for p in files:
        body = json.loads(p.read_text(encoding="utf8"))
        out(f"[{p.name}] runner running: {'yes' if alive else 'no'}; last update {body.get('updated')}; lanes "
            f"{body.get('lanes')} on processors {body.get('cpus')}" +
            (f"; seeds {body.get('seeds')}" if body.get("seeds") else "") +
            ("; the group phases" if body.get("group_only") else ""))
        out("counts: " + ", ".join(f"{k} {v}" for k, v in sorted((body.get("counts") or {}).items())))
        m = body.get("memory") or {}
        out(f"memory: MemTotal {m.get('total_gib')} GiB, MemAvailable {m.get('available_gib')} GiB, reserved "
            f"{m.get('reserved_gib')} GiB")
        out("CPU-hours: " + ", ".join(f"{k} {v}" for k, v in sorted((body.get("cpu_hours") or {}).items())))
        for pid, r in sorted((body.get("phases") or {}).items()):
            if r.get("state") == "running":
                out(f"  running {pid} on processor {r.get('cpu')} since {r.get('started')}")
        for pid, r in sorted((body.get("phases") or {}).items()):
            if r.get("state") in ("stopped", "unfinished", "halted"):
                out(f"  {r['state']} {pid}: {r.get('note')}")
        for why in (body.get("paused") or {}).values():
            out(why)
        if body.get("over_cap"):
            out(body["over_cap"])
        if body.get("budget_continues"):
            out("budget-continue records: " + ", ".join(f"{k} {v}" for k, v in sorted(body["budget_continues"].items())))
        if body.get("halted"):
            out(f"halted: {body['halted']}")
    return 0


def cmd_plan(cfg, out=say_line):
    keys = PH.plan(spec=cfg.plan)
    tasks = build_tasks(keys)
    for t in sorted(tasks, key=lambda x: -x.prio):
        st = PH.done(t.key, cfg.out)
        why = None
        if st == "missing":
            why = PH.rule_skip(t.key, cfg.out, keys)
            st = "skipped" if why else "waiting"
        out(f"  {st:<9} {t.pid:<44} chain {t.prio:>4} min  {PH.plain_name(t.key)}" + (f"  ({why})" if why else ""))
    out(f"plan: {len(keys)} phases")
    return 0


def cmd_retry(cfg, pid, out=say_line, isatty=None, argv=None):
    isatty = sys.stdin.isatty() if isatty is None else isatty
    if not isatty:
        out("refused: retry must be typed at a terminal (standard input is not one)")
        return EXIT_REFUSED
    try:
        approval_or_refuse(cfg)
        key = PH.parse_id(pid)
        PH.check_key(key, cfg.mode, cfg.plan)
    except PH.Refused as error:
        out(str(error))
        return EXIT_REFUSED
    if PH.done(key, cfg.out) != "missing":
        out(f"refused: {pid} is {PH.done(key, cfg.out)}; there is nothing to retry")
        return EXIT_REFUSED
    if not phase_lock_free(cfg.out, pid):
        out(f"refused: {pid} is running now")
        return EXIT_REFUSED
    mode, n, n_ps = need(cfg.out, pid)
    if mode != "unfinished":
        out(f"refused: {pid} is not unfinished ({n_ps} phase-specific interruptions"
            + (f"; retry request {n} is still waiting to be used)" if n else ")"))
        return EXIT_REFUSED
    n = write_retry_request(cfg.out, pid, argv if argv is not None else sys.argv, tty_name())
    out(f"retry request {n} for {pid} written; " + ("the running runner picks it up" if runner_running(cfg.out) else
                                                    f"start the run again: {RUN_N80} run --lanes 10"))
    return EXIT_FINISHED


def _result_files(out_root):
    """[(source, path inside results-*/files/)] of every record of the output root: JSON records with their sidecars,
    the looks' .md, the logs (as .txt, because .gitignore ignores *.log); no model, array or training file, no lock and
    no temporary file."""
    src = Path(out_root)
    files = []
    for q in sorted(src.rglob("*")):
        if not q.is_file() or q.is_symlink():
            continue
        rel = q.relative_to(src)
        name = q.name
        if rel.parts[0] == "locks" or name.endswith(MODEL_SUFFIXES) or name.endswith(".lock") or ".tmp" in name:
            continue
        if q.suffix in (".json", ".md") or name.endswith(".sha256"):
            files.append((q, rel))
        elif q.suffix == ".log":
            files.append((q, rel.with_suffix(".txt")))
    return files


def make_results(out_root=K.OUT, dest_root=K.N80, out=say_line, stamp=None):
    """Copies every record of the output root (_result_files) into <dest_root>/results-<utc>/files/ with SHA256SUMS
    (every copied file's SHA-256) and RESULTS.json (+ .sha256), write-once. Reads OUT only; never recomputes a number.
    Returns the folder."""
    src = Path(out_root)
    if not src.is_dir():
        raise PH.Refused(f"refused: {src} does not exist; there is nothing to gather")
    stamp = stamp or LG.utc_stamp()
    dest = Path(dest_root) / f"results-{stamp}"
    if dest.exists():
        raise PH.Refused(f"refused: {dest} exists (write-once)")
    sums = []
    for q, rel in _result_files(src):
        target = dest / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(q, target)
        sums.append(f"{LG.file_sha256(target)}  files/{rel.as_posix()}")
    (dest / "SHA256SUMS").write_text("\n".join(sorted(sums, key=lambda x: x[66:])) + "\n", encoding="utf8")
    states = {}
    for p in sorted(src.glob("status*.json")):
        status = LG.ok_record(p) or {}
        for pid, r in (status.get("phases") or {}).items():
            if r.get("mine", True) or pid not in states:
                states[pid] = r.get("state")
    plumb = LG.ok_record(src / K.LAYOUT["plumbing"]) or {}
    LG.write_json_once(dest / "RESULTS.json", dict(
        kind="near80-results", version=K.VERSION, created=LG.utc(), source=str(src), files=len(sums),
        sha256sums_sha256=LG.file_sha256(dest / "SHA256SUMS"), prereg_sha256=plumb.get("prereg_sha256"),
        finished=bool(states) and all(v in FINAL_STATES for v in states.values()),
        counts={s: sum(1 for v in states.values() if v == s) for s in sorted(set(states.values()))} or None,
        note="records only: no model (.msgpack), array (.npz) or training (.hdf5) file; those go to the run-files "
             "branch (push_runfiles_near80.sh). Logs are copied as .txt."))
    rel = dest.relative_to(K.ROOT) if LG.inside(dest, K.ROOT) else dest
    out(f"results gathered: {dest} ({len(sums)} files). Commit and push them under your own name:")
    out(f"  cd {K.ROOT}")
    out(f"  git add {rel}")
    out(f"  git -c user.name=\"{K.APPROVER_NAME}\" -c user.email={K.APPROVER_EMAIL} commit -m \"Near-80 v9: results "
        f"{stamp}\"")
    out(f"  git push origin {K.BRANCH}")
    out(f"If the push is rejected: git -c user.name=\"{K.APPROVER_NAME}\" -c user.email={K.APPROVER_EMAIL} pull "
        f"--rebase origin {K.BRANCH}, then git push origin {K.BRANCH} again.")
    return dest


def cmd_results(cfg, out=say_line):
    try:
        make_results(cfg.out, K.N80, out)
    except (PH.Refused, OSError, LG.RecordError) as error:
        out(str(error) if str(error).startswith("refused") else f"refused: {error}")
        return EXIT_REFUSED
    return EXIT_FINISHED


def runfiles_plan(out_root, plan_dir, repo=K.ROOT):
    """The run files for push_runfiles_near80.sh: every file under the output root (model files included) except
    locks, temporary files and the training files (.hdf5, which the data records' fingerprints and the logs
    regenerate). Writes plan_dir/files.list (NUL-separated paths relative to the repository), RUNFILES_SHA256SUMS
    and env.sh. Returns the number of files."""
    src, repo, d = Path(out_root), Path(repo), Path(plan_dir)
    if not LG.inside(src, repo):
        raise PH.Refused(f"refused: {src} is not inside {repo}")
    files, sums, newest = [], [], 0.0
    for q in sorted(src.rglob("*")):
        if not q.is_file() or q.is_symlink():
            continue
        name = q.name
        if name.endswith(".lock") or ".tmp" in name or name.endswith((".hdf5", ".h5")):
            continue
        rel = q.relative_to(repo).as_posix()
        files.append(rel)
        sums.append(f"{LG.file_sha256(q)}  {rel}")
        newest = max(newest, q.stat().st_mtime)
    if not files:
        raise PH.Refused(f"refused: there are no run files under {src}")
    d.mkdir(parents=True, exist_ok=True)
    (d / "files.list").write_bytes(b"".join(f.encode("utf8") + b"\0" for f in files))
    (d / "RUNFILES_SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf8")
    date = f"@{int(newest)} +0000"            # git's own date format: the newest file's time (a rerun rebuilds it)
    (d / "env.sh").write_text("\n".join([f"REPO_DIR='{repo}'", f"AUTHOR_NAME='{K.APPROVER_NAME}'",
                                          f"AUTHOR_EMAIL='{K.APPROVER_EMAIL}'", f"COMMIT_DATE='{date}'",
                                          f"RUNFILES_BRANCH='{K.RUNFILES_BRANCH}'", f"COUNT='{len(files)}'",
                                          f"HOST='{K.HOST}'"]) + "\n", encoding="utf8")
    return len(files)


def cmd_report(out_root, out=say_line):
    pooled = K.load_local("pooled")
    try:
        rep = pooled.halt_report(Path(out_root))
    except pooled.LookRefused as error:
        text = str(error)
        out(text if text.startswith("refused") else f"refused: {text}")
        return EXIT_REFUSED
    sentences = (rep or {}).get("sentences") if isinstance(rep, dict) else None
    if sentences:
        for s in sentences:
            out(s.get("text") if isinstance(s, dict) else str(s))
    else:
        out(json.dumps(rep, indent=1, sort_keys=True, default=str))
    return 0


def _seeds_arg(text):
    return [int(s) for s in str(text).split(",") if s.strip()] if text else None


def main(argv=None, out=say_line):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plumbing"):
        p = sub.add_parser(name)
        p.add_argument("--lanes", type=int, default=K.DEFAULT_LANES)
        p.add_argument("--seeds", default=None, help="this runner's seeds, e.g. 40,41 (the cluster's per-seed jobs)")
        p.add_argument("--group-only", action="store_true", help="the group phases only (the cluster's last job)")
        p.add_argument("--tag", default=None, help="this runner's lock and status file name")
    sub.add_parser("status")
    p = sub.add_parser("selftest")
    p.add_argument("--for", dest="for_cmd", choices=("run", "plumbing", "retry", "selftest"), default="selftest")
    p.add_argument("--lanes", type=int, default=K.DEFAULT_LANES)
    sub.add_parser("plan")
    p = sub.add_parser("retry")
    p.add_argument("phase_id")
    p = sub.add_parser("budget-continue")
    p.add_argument("stage")
    sub.add_parser("report")
    sub.add_parser("results")
    p = sub.add_parser("runfiles-plan")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("stub")
    p.add_argument("--fixture", required=True)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--seeds", default=None, help="a stub run on fewer seeds, e.g. 0,40")
    p.add_argument("--refuse", default=None, help="seeds whose synthetic data make BCA refuse, e.g. 41")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "stub":
            if not 1 <= a.lanes <= K.MAX_LANES:
                out(f"refused: --lanes must be 1-{K.MAX_LANES}")
                return EXIT_REFUSED
            seeds = _seeds_arg(a.seeds)
            for s in seeds or []:
                K.group_of(s)
            return cmd_stub(a.fixture, a.lanes, seeds, out, refuse=_seeds_arg(a.refuse))
        lanes = getattr(a, "lanes", K.DEFAULT_LANES)
        if not 1 <= lanes <= K.MAX_LANES:
            out(f"refused: --lanes must be 1-{K.MAX_LANES}")
            return EXIT_REFUSED
        extra = {}
        if a.cmd in ("run", "plumbing"):
            extra = dict(seeds=_seeds_arg(a.seeds), group_only=bool(a.group_only), tag=a.tag)
            if a.cmd == "plumbing" and (extra["seeds"] or extra["group_only"]):
                out("refused: the plumbing runs as one job (no --seeds or --group-only)")
                return EXIT_REFUSED
        cfg = Cfg(lanes=lanes, **extra)
        if a.cmd == "run":
            return cmd_run(cfg, out)
        if a.cmd == "plumbing":
            return cmd_plumbing(cfg, out)
        if a.cmd == "status":
            return cmd_status(cfg.out, out)
        if a.cmd == "selftest":
            return 0 if selftest(cfg, a.for_cmd, out) else 1
        if a.cmd == "plan":
            return cmd_plan(cfg, out)
        if a.cmd == "retry":
            return cmd_retry(cfg, a.phase_id, out)
        if a.cmd == "budget-continue":
            return cmd_budget_continue(cfg, a.stage, out)
        if a.cmd == "report":
            return cmd_report(cfg.out, out)
        if a.cmd == "results":
            return cmd_results(cfg, out)
        if a.cmd == "runfiles-plan":
            n = runfiles_plan(cfg.out, a.dir)
            out(f"run files planned: {n}")
            return EXIT_FINISHED
    except (PH.Refused, ValueError) as error:
        text = str(error)
        out(text if text.startswith("refused") else f"refused: {text}")
        return EXIT_REFUSED
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
