"""Runs the confirmatory phases of Stage 2 (fresh seeds 5-9) in the pre-registered order, each through the pinned
launch.sh, unchanged: one phase per process, pinned to one processor with taskset, JAX on the processor, at most two
processes at once. Called by run_confirmatory.sh after restore.sh; never run it on its own.

Order, per seed (each step waits for the one before it; seeds run side by side when two lanes are given):
  fresh_data, fresh_learners                                   new logs and learners, no BCA
  detect_pairs, detect_labels, detect_standins  (stall, then no_stall)   no BCA
  detect_fit (stall fixed, stall host, no_stall fixed)          BCA's band; tau
  detect_fit_weighted (stall fixed, no_stall fixed)             only when the frozen version is the weighted one
  detect_score (stall fixed, stall host [uniform], no_stall fixed)
  policy_train, policy_eval, policy_no_stall_train, policy_no_stall_eval    only when a version passed on seeds 0-4
  deploy (stall, no_stall)                                       after the policy lock, so BCA's k=1 actor is checked
  host_train / host_eval (stall, then no_stall)                  only when a version passed on seeds 0-4
then, once every seed is finished or failed on record: detect_marks --group confirmatory, then pooled.

Resumable: a phase counts as finished when the outputs it writes exist and match their SHA-256 sidecars (every phase
writes once and refuses to overwrite), so a rerun starts only what is missing. A seed that failed on record (its data
checks, a learner, a band fit, or the policy test's stop) is never replaced: the phases the pinned code would refuse
for it are skipped, exactly as the pinned code decides (fresh.recorded_failure, run_stage2.recorded_seed_failure).
Any other phase that ends without its outputs stops the run: nothing is deleted, nothing is retried, and the log is
named for the cloud to read.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
LIB = Path(__file__).resolve().parent
PKG = LIB.parent
ROOT = Path("/home/user/bca")
OUT = ROOT / "runs" / "shift_explore" / "stage2"
OFFCLOUD = ROOT / "runs" / "shift_explore" / "stage2_offcloud"
SEEDS = (5, 6, 7, 8, 9)
WORLDS = ("stall", "no_stall")
MAX_LANES = 2              # the pre-registration's approval: at most 2 processes at once
BASE_ENV = dict(PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", HOME="/root", LANG="C.UTF-8",
                LC_ALL="C.UTF-8")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(msg, log=None):
    line = f"{now()} {msg}"
    try:
        print(line, flush=True)
    except OSError:
        pass
    if log is not None:
        with open(log, "a", encoding="utf8") as handle:
            handle.write(line + "\n")


def load_pinned():
    """The pinned Stage 2 modules, imported read only from their restored place (the decisions the runner needs are
    the pinned code's own: the frozen version, the room condition, recorded failures)."""
    man = json.loads((OUT / "code_manifest.json").read_text(encoding="utf8"))
    code_dir = man["source_dir"]
    for p in (code_dir, str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import detect_contract as DC
    import fresh as F
    import run_stage2 as RS
    import stage2_common as C
    return C, RS, F, DC, Path(code_dir), man


class Task:
    def __init__(self, name, args, deps, done, skip=None, seed=None):
        self.name, self.args, self.deps, self.done_fn, self.skip_fn, self.seed = name, args, deps, done, skip, seed
        self.state, self.cpu, self.proc, self.started, self.finished, self.code, self.note = \
            "pending", None, None, None, None, None, None


def plan(C, RS, F, DC, version, training):
    """The task list (in launch order) for seeds 5-9 and the confirmatory group."""
    Vd = version or "uniform"
    tasks = []

    def verified(*paths):
        def fn():
            for p in paths:
                if not Path(p).exists():
                    return False
                C.read_json(p)                       # refuses (SystemExit) when the sidecar does not match
            return True
        return fn

    def any_of(*paths):
        def fn():
            return any(verified(p)() for p in paths)
        return fn

    def add(name, args, deps, done, skip=None, seed=None):
        tasks.append(Task(name, [str(a) for a in args], deps, done, skip, seed))
        return name

    for K in SEEDS:
        def seed_failed(K=K):
            return F.recorded_failure(K)

        fdir = lambda w, K=K: OUT / "fresh" / w / f"seed{K}"          # noqa: E731
        ddir = lambda w, K=K: OUT / "detect_fresh" / f"seed{K}" / w     # noqa: E731
        bdir = lambda v, w, c, K=K: OUT / "detect" / v / w / c / f"seed{K}"   # noqa: E731
        pdir = OUT / "policy" / f"seed{K}"
        t_data = add(f"fresh_data/seed{K}", ["--phase", "fresh_data", "--seed", K], [],
                     verified(fdir("stall") / "data.json", fdir("no_stall") / "data.json"), seed=K)
        t_learn = add(f"fresh_learners/seed{K}", ["--phase", "fresh_learners", "--seed", K], [t_data],
                      verified(fdir("stall") / "learners.json"), seed_failed, K)
        no_bca = []
        for W in WORLDS:
            t_p = add(f"detect_pairs/seed{K}/{W}", ["--phase", "detect_pairs", "--seed", K, "--world", W], [t_learn],
                      verified(ddir(W) / "pairs.json"), seed_failed, K)
            t_l = add(f"detect_labels/seed{K}/{W}", ["--phase", "detect_labels", "--seed", K, "--world", W], [t_p],
                      verified(ddir(W) / "labels.json"), seed_failed, K)
            t_s = add(f"detect_standins/seed{K}/{W}", ["--phase", "detect_standins", "--seed", K, "--world", W], [t_l],
                      verified(*[ddir(W) / "scores" / f"{n}.json" for n in DC.STANDIN_NAMES]), seed_failed, K)
            no_bca.append(t_s)
        fit = {}
        for W, Cr in (("stall", "fixed"), ("stall", "host"), ("no_stall", "fixed")):
            d = bdir("uniform", W, Cr)

            def fit_done(d=d):
                if not verified(d / "band.json")():
                    return False
                return bool(C.read_json(d / "band.json").get("failed")) or verified(d / "tau.json")()
            fit[(W, Cr, "uniform")] = add(f"detect_fit/seed{K}/{W}/{Cr}",
                                          ["--phase", "detect_fit", "--seed", K, "--world", W, "--critic", Cr],
                                          list(no_bca), fit_done, seed_failed, K)
        if Vd == "weighted":
            for W in WORLDS:
                def w_skip(W=W, K=K):
                    return F.recorded_failure(K) or RS.recorded_seed_failure(K, W, "fixed", "uniform")
                fit[(W, "fixed", "weighted")] = add(
                    f"detect_fit_weighted/seed{K}/{W}/fixed",
                    ["--phase", "detect_fit_weighted", "--seed", K, "--world", W, "--critic", "fixed"],
                    [fit[(W, "fixed", "uniform")]], verified(bdir("weighted", W, "fixed") / "band.json",
                                                             bdir("weighted", W, "fixed") / "tau.json"), w_skip, K)
        scores = []
        for W, Cr, v in (("stall", "fixed", Vd), ("stall", "host", "uniform"), ("no_stall", "fixed", Vd)):
            def s_skip(W=W, Cr=Cr, v=v, K=K):
                return RS.recorded_seed_failure(K, W, Cr, v)
            scores.append(add(f"detect_score/seed{K}/{W}/{Cr}/{v}",
                              ["--phase", "detect_score", "--seed", K, "--world", W, "--critic", Cr, "--version", v],
                              [fit[(W, Cr, v)]], verified(bdir(v, W, Cr) / "scores_record.json"), s_skip, K))
        chain = list(scores)
        if training:
            def ns_skip(K=K):
                why = F.recorded_failure(K)
                if why:
                    return why
                if (OUT / "policy" / f"seed{K}" / "stop.json").exists():
                    return "the stalling world's policy test stopped this seed (its stop.json); no no-stall arm runs"
                return None
            t1 = add(f"policy_train/seed{K}", ["--phase", "policy_train", "--seed", K], chain,
                     any_of(pdir / "lock.json", pdir / "stop.json"), seed_failed, K)
            t2 = add(f"policy_eval/seed{K}", ["--phase", "policy_eval", "--seed", K], [t1],
                     any_of(pdir / "values.json", pdir / "stop.json"), seed_failed, K)
            t3 = add(f"policy_no_stall_train/seed{K}", ["--phase", "policy_no_stall_train", "--seed", K], [t2],
                     any_of(pdir / "no_stall" / "lock.json", pdir / "no_stall" / "stop.json"), ns_skip, K)
            t4 = add(f"policy_no_stall_eval/seed{K}", ["--phase", "policy_no_stall_eval", "--seed", K], [t3],
                     any_of(pdir / "no_stall" / "values.json", pdir / "no_stall" / "stop.json"), ns_skip, K)
            chain = [t4]
        for W in WORLDS:
            def d_skip(W=W, K=K):
                return RS.recorded_seed_failure(K, W, "fixed", Vd)
            add(f"deploy/seed{K}/{W}", ["--phase", "deploy", "--seed", K, "--world", W], list(chain),
                verified(OUT / "deploy" / W / f"seed{K}" / "values.json"), d_skip, K)
        if training:
            hdir = lambda w, K=K: OUT / "host" / w / f"seed{K}"            # noqa: E731
            h1 = add(f"host_train/seed{K}/stall", ["--phase", "host_train", "--seed", K, "--world", "stall"],
                     list(chain), verified(hdir("stall") / "lock.json"), seed_failed, K)
            h2 = add(f"host_eval/seed{K}/stall", ["--phase", "host_eval", "--seed", K, "--world", "stall"], [h1],
                     verified(hdir("stall") / "values.json"), seed_failed, K)
            h3 = add(f"host_train/seed{K}/no_stall", ["--phase", "host_train", "--seed", K, "--world", "no_stall"],
                     [h2], verified(hdir("no_stall") / "lock.json"), seed_failed, K)
            add(f"host_eval/seed{K}/no_stall", ["--phase", "host_eval", "--seed", K, "--world", "no_stall"], [h3],
                verified(hdir("no_stall") / "values.json"), seed_failed, K)
    every_seed = [t.name for t in tasks]
    m = add("detect_marks/confirmatory", ["--phase", "detect_marks", "--group", "confirmatory", "--version", Vd],
            every_seed, verified(OUT / "detect" / f"marks_confirmatory_{Vd}.json"))
    add("pooled/confirmatory", ["--phase", "pooled", "--group", "confirmatory"], [m],
        verified(OUT / "pooled_confirmatory.json"))
    return tasks


def log_name(args):
    """The log file launch.sh writes for these arguments (its own rule: drop '--', spaces to '_')."""
    text = " ".join(args).replace("--", "")
    return "_".join(text.split()) + ".log"


def pick_cpus(n):
    """n processors on different physical cores when the topology says so (two threads of one core would slow each
    other), else the first n."""
    groups, seen = [], set()
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        cpu = int(d.name[3:])
        if cpu not in os.sched_getaffinity(0):
            continue
        sib = d / "topology" / "thread_siblings_list"
        key = sib.read_text().strip() if sib.exists() else str(cpu)
        if key not in seen:
            seen.add(key)
            groups.append(cpu)
    cpus = groups[:n] if len(groups) >= n else sorted(os.sched_getaffinity(0))[:n]
    return cpus


class Runner:
    def __init__(self, tasks, cpus, launch, env, log, status_path, launcher_note):
        self.tasks = tasks
        self.by_name = {t.name: t for t in tasks}
        self.cpus, self.launch, self.env, self.log, self.status_path = cpus, launch, env, log, status_path
        self.launcher_note = launcher_note
        self.stop_requested, self.failed = False, []

    def write_status(self):
        rows = {t.name: dict(state=t.state, cpu=t.cpu, started=t.started, finished=t.finished, exit_code=t.code,
                             note=t.note, log=str(OUT / "logs" / log_name(t.args)), args=t.args)
                for t in self.tasks}
        payload = dict(kind="stage2-offcloud-runner-status", updated=now(), launcher=self.launcher_note,
                       cpus=self.cpus, tasks=rows,
                       running={t.name: t.proc.pid for t in self.tasks if t.proc is not None})
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf8")
        os.replace(tmp, self.status_path)

    def resolve(self, t):
        """done | skipped | pending, from the files (never from memory)."""
        if t.done_fn():
            return "done", None
        if t.skip_fn is not None:
            why = t.skip_fn()
            if why:
                return "skipped", str(why)
        return "pending", None

    def ready(self, t):
        return all(self.by_name[d].state in ("done", "skipped") for d in t.deps)

    def start(self, t, cpu):
        if sha_file(self.launch) != self.launcher_note["sha256"]:
            raise SystemExit(f"refused: {self.launch} changed while running")
        t.cpu, t.started, t.state = cpu, now(), "running"
        errf = open(OFFCLOUD / "launch_stderr.txt", "a", encoding="utf8")
        t.proc = subprocess.Popen(["sh", str(self.launch), str(cpu)] + t.args, cwd=str(ROOT), env=self.env,
                                  stdin=subprocess.DEVNULL, stdout=errf, stderr=errf, start_new_session=True)
        errf.close()
        say(f"start  {t.name} on processor {cpu} (pid {t.proc.pid})", self.log)

    def finish(self, t):
        t.code, t.finished, t.proc = t.proc.returncode, now(), None
        state, why = self.resolve(t)
        if state == "done":
            t.state, t.note = "done", None if t.code == 0 else f"exit {t.code} with its outputs written (a recorded stop)"
            say(f"done   {t.name} (exit {t.code})", self.log)
        elif state == "skipped":
            t.state, t.note = "skipped", why
            say(f"skip   {t.name}: {why}", self.log)
        else:
            t.state = "failed"
            t.note = f"exit {t.code} without its outputs; log {OUT / 'logs' / log_name(t.args)}"
            self.failed.append(t.name)
            say(f"FAILED {t.name}: {t.note}. Nothing was deleted. No new phase starts; paste the last 30 lines of "
                f"that log to the cloud.", self.log)

    def run(self):
        while True:
            for t in self.tasks:
                if t.proc is not None and t.proc.poll() is not None:
                    self.finish(t)
            busy = {t.cpu for t in self.tasks if t.proc is not None}
            running = len(busy)
            if not self.stop_requested and not self.failed:
                for t in self.tasks:
                    if t.state != "pending" or not self.ready(t):
                        continue
                    state, why = self.resolve(t)
                    if state != "pending":
                        t.state, t.note = state, why
                        say(f"{'found ' if state == 'done' else 'skip  '} {t.name}" + (f": {why}" if why else
                                                                                         " (finished earlier)"),
                            self.log)
                        continue
                    free = [c for c in self.cpus if c not in busy]
                    if not free:
                        break
                    self.start(t, free[0])
                    busy.add(free[0])
                    running += 1
            self.write_status()
            if running == 0:
                progressed = any(t.state == "pending" and self.ready(t) for t in self.tasks)
                if not progressed or self.stop_requested or self.failed:
                    break
            time.sleep(1)
        self.write_status()
        if self.failed:
            return 1
        if all(t.state in ("done", "skipped") for t in self.tasks):
            return 0
        return 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lanes", type=int, required=True, choices=(1, 2),
                    help="processes at once (the pre-registration allows at most 2 in all, cloud included)")
    ap.add_argument("--cpus", help="processor numbers for the lanes, e.g. 2,4 (default: separate physical cores)")
    ap.add_argument("--plan-only", action="store_true", help="print the plan and the state of every phase, run nothing")
    ap.add_argument("--openblas-coretype", choices=("SkylakeX",),
                    help="only if the self-test showed the anchors need it on this AVX-512 processor and the cloud "
                         "agreed")
    ap.add_argument("--launcher", help=argparse.SUPPRESS)       # tests only (needs BCA_OFFCLOUD_TEST=1)
    a = ap.parse_args(argv)
    OFFCLOUD.mkdir(parents=True, exist_ok=True)
    log = OFFCLOUD / "runner.log"
    lock = open(OFFCLOUD / "runner.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("refused: another run_confirmatory.sh is running")
    old = OFFCLOUD / "status.json"
    if old.exists():
        for name, pid in (json.loads(old.read_text(encoding="utf8")).get("running") or {}).items():
            try:
                cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            except OSError:
                continue
            if b"launch.sh" in cmd or b"run_stage2.py" in cmd:
                raise SystemExit(f"refused: phase {name} from an earlier run is still running (pid {pid}); wait for it")
    C, RS, F, DC, code_dir, man = load_pinned()
    version = RS.chosen_version()                 # refuses without the late files
    room = RS.room_condition()
    training = version is not None and bool(room["passed"])
    launch = code_dir / "launch.sh"
    test = False
    if a.launcher:
        if os.environ.get("BCA_OFFCLOUD_TEST") != "1":
            raise SystemExit("refused: --launcher is for the package's own tests only")
        launch, test = Path(a.launcher), True
    elif sha_file(launch) != man["files"]["launch.sh"]:
        raise SystemExit("refused: launch.sh is not the pinned one")
    tasks = plan(C, RS, F, DC, version, training)
    cpus = [int(c) for c in a.cpus.split(",")] if a.cpus else pick_cpus(a.lanes)
    if len(cpus) < a.lanes or len(set(cpus)) != len(cpus):
        raise SystemExit(f"refused: give {a.lanes} different processor numbers")
    cpus = cpus[:a.lanes]
    env = dict(BASE_ENV)
    if a.openblas_coretype:
        env["OPENBLAS_CORETYPE"] = a.openblas_coretype
    if test:
        env["BCA_OFFCLOUD_TEST"] = "1"
    ident = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "user.email"], capture_output=True, text=True,
                           env=dict(BASE_ENV, GIT_OPTIONAL_LOCKS="0")).stdout.strip() or None
    note = dict(path=str(launch), sha256=sha_file(launch), pinned=man["files"]["launch.sh"], test=test,
                openblas_coretype=a.openblas_coretype, lanes=a.lanes, git_user_email_at_start=ident)
    runner = Runner(tasks, cpus, launch, env, log, OFFCLOUD / "status.json", note)
    shown = version or "none passed (uniform band, no training arm)"
    head = (f"plan: {len(tasks)} phases for seeds 5-9; version {shown}; policy test and full host "
            f"{'run' if training else 'do not run'}; lanes {a.lanes} on processors {cpus}")
    if a.plan_only:
        print(head)
        for t in tasks:
            if runner.ready(t):
                state, why = runner.resolve(t)
                t.state = state
                shown = "next" if state == "pending" else state
            else:
                shown, why = ("done", None) if t.done_fn() else ("waiting", None)
            print(f"  {shown:<8} {t.name}" + (f"  ({why})" if why else ""))
        return 0
    say(head, log)
    if not test or os.environ.get("BCA_OFFCLOUD_TEST_PREFLIGHT") == "1":
        sys.path.insert(0, str(LIB))
        import selftest_checks as ST
        if not ST.preflight(cpus, {k: v for k, v in env.items() if k == "OPENBLAS_CORETYPE"}):
            say("refused: a check before the first phase failed (lines above); no phase was started", log)
            return 1

    def stop(signum, frame):
        if not runner.stop_requested:
            say("stop requested: no new phase starts; the running ones finish first (do not close this window)", log)
        runner.stop_requested = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    code = runner.run()
    done = sum(t.state == "done" for t in tasks)
    skipped = sum(t.state == "skipped" for t in tasks)
    say(f"runner ends: {done} done, {skipped} skipped on record, {len(runner.failed)} failed, "
        f"{sum(t.state == 'pending' for t in tasks)} not started", log)
    return code


if __name__ == "__main__":
    sys.exit(main())
