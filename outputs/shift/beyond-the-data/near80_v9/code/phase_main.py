"""Near-80 (version 9, DRAFT): the one entry every phase process runs. The runner starts it as
  env -i <PHASE_ENV> nice -n 5 taskset -c <cpu> <python> phase_main.py --phase-id ID --out ROOT
      [--mode official|scratch|stub] [--fixture DIR] [--plan full|smoke|seeds:a,b] [--attempt N]
      [--steps N --chunk N --updates N --starts N]
and the plumbing also as  phase_main.py --reproduction --out SCRATCH --mode scratch  (the reproduction check) and
phase_main.py --phase-id data_main.main.<world>.s0 --fresh-path ... (the fresh-seed code path on seed 0).

In order, before any heavy import (build spec, section 5.4):
  1. the id is parsed and checked against the table (every seed another study holds is refused in every mode; scratch
     mode is the plumbing smoke on seed 0 only, so the fresh seeds and the confirmatory group are refused there); the
     phase lock
     OUT/locks/<phase_id>.lock is taken with flock (refused while another process holds it);
  2. the environment: exactly near80_constants.PHASE_ENV (official), and exactly one processor visible;
  3. the output root: official mode only near80_constants.OUT; scratch and stub only outside /home/user/bca (stub
     only inside its fixture); never a pinned folder; write-once (a phase whose files exist is refused: the runner
     moves an interrupted attempt's files aside first);
  4. the approval: approval_check.require_approval() (official and scratch), require_approval(fixture_config(DIR))
     (stub); prereg.json's constants must equal near80_constants;
  5. phases.admit (exit 97 when refused; exit 0 with a skipped record when a need was stopped or skipped);
  6. the training-file check for every kind that reads a setting's training rows (a mismatch is a recorded stop);
  7. the phase itself; its record is written once with the approval facts, the processor, the peak memory, the time
     and the run's sizes (under 'run_sizes': version 8's wrapper wrote them over a data record's own 'sizes', the cause
     of its amendment 1). A recorded stop writes stops/<phase_id>.json (and the policy's stop.json) and exits 2: a
     stopped phase is written once and never rerun.
  8. at exit, OUT/ledger/<phase_id>/<n>.usage.json {vmhwm_kb, seconds, exit_code}; the runner writes the attempt record.
Exit codes: 0 done (or skipped by rule), 2 a recorded stop, 97 refused, anything else a failure.
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import fcntl  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402

# step2lib.PINNED (test_runner checks that they agree) plus step 2's own output folder: never written.
PINNED_REL = ("experiments", "runs/shift", "runs/shift_explore/stage2", "runs/shift_explore/stage2_v7",
              "runs/shift_explore/room", "runs/shift_explore/standins", "runs/shift_explore/detect",
              "runs/shift_explore/deploy", "runs/shift_explore/fallback", "runs/shift_explore/trace",
              "runs/shift_explore/anatomy", "runs/shift_explore/empirical", "runs/shift_explore/principled",
              "runs/shift_explore/restructure/step1", "outputs/shift/beyond-the-data/stage2",
              "outputs/shift/beyond-the-data/stage2_v7", "outputs/shift/beyond-the-data/stage2_gpu",
              "outputs/shift/beyond-the-data/stage2_v7_design", "outputs/shift/beyond-the-data/restructure/step1",
              "outputs/shift/beyond-the-data/restructure/analysis", "calibration", "algorithms", "runtime")
EXTRA_PINNED_REL = ("runs/shift_explore/restructure", "runs/shift_explore/near80", "outputs/shift/beyond-the-data/near80")
LOOK_KINDS = ("look_exploratory", "look_confirmatory")


class Refusal(Exception):
    pass


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# --------------------------------------------------------------------------------------------- checks

def check_environment(mode, environ=None, affinity=None):
    """Official: exactly PHASE_ENV. Every mode: exactly one processor visible."""
    environ = dict(os.environ if environ is None else environ)
    if mode == "official" and environ != dict(K.PHASE_ENV):
        extra = sorted(set(environ) - set(K.PHASE_ENV))
        wrong = sorted(k for k in K.PHASE_ENV if environ.get(k) != K.PHASE_ENV[k])
        raise Refusal(f"refused: the environment must be exactly the phase environment (extra {extra}, wrong {wrong})")
    cpus = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    if len(cpus) != 1:
        raise Refusal(f"refused: a phase runs on exactly one processor (taskset -c <cpu>); this process may use {cpus}")
    return cpus


def check_out_root(out, mode, fixture=None):
    """The output root of a mode (module docstring); returns it resolved."""
    p = Path(os.path.realpath(str(out)))
    for rel in PINNED_REL + EXTRA_PINNED_REL:
        if LG.inside(p, ROOT / rel):
            raise Refusal(f"refused: {p} is inside the pinned folder {ROOT / rel}")
    if mode == "official":
        if p != Path(os.path.realpath(str(K.OUT))):
            raise Refusal(f"refused: official outputs go to {K.OUT} only (got {p})")
        return p
    if LG.inside(p, ROOT):
        raise Refusal(f"refused: {mode} outputs must lie outside {ROOT} (got {p})")
    if mode == "stub":
        if fixture is None:
            raise Refusal("refused: stub mode needs --fixture")
        fx = Path(os.path.realpath(str(fixture)))
        if LG.inside(fx, ROOT):
            raise Refusal("refused: a test fixture must lie outside the repository")
        if not LG.inside(p, fx):
            raise Refusal(f"refused: stub outputs go inside the fixture {fx} (got {p})")
        if not (fx / K.TEST_MARKER).is_file():
            raise Refusal(f"refused: {fx} is not a near-80 version 9 test fixture (no {K.TEST_MARKER})")
    elif mode != "scratch":
        raise Refusal(f"refused: unknown mode {mode!r}")
    return p


def check_sizes(a, mode):
    given = [v for v in (a.steps, a.chunk, a.updates, a.starts) if v is not None]
    if mode == "official" and given:
        raise Refusal("refused: --steps, --chunk, --updates and --starts are for scratch and stub runs only")
    for v in given:
        if int(v) <= 0:
            raise Refusal("refused: sizes must be positive")
    if a.steps is not None and a.chunk is not None and int(a.steps) % int(a.chunk):
        raise Refusal("refused: --steps must be a multiple of --chunk")


def take_lock(out_root, pid):
    """The phase lock OUT/locks/<phase_id>.lock (flock); refused while another process holds it. The returned file
    stays open for the life of the process."""
    d = Path(out_root) / "locks"
    d.mkdir(parents=True, exist_ok=True)
    handle = open(d / f"{pid}.lock", "a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise Refusal(f"refused: phase {pid} is already running (its lock is held)") from None
    return handle


def check_write_once(key, out_root):
    st = PH.done(key, out_root)
    if st != "missing":
        raise Refusal(f"refused: {PH.phase_id(key)} is already {st} (write-once)")
    present = [r for r in PH.outputs(key) if (Path(out_root) / r).exists()]
    if present:
        raise Refusal(f"refused: files of an earlier attempt exist ({present[:4]}); the runner moves them aside "
                      f"before a rerun (never deleted)")


def approval_facts(mode, fixture):
    """(approval facts, prereg payload) through approval_check; Refusal when it refuses or the constants differ."""
    import approval_check as AC
    try:
        cfg = AC.fixture_config(fixture) if mode == "stub" else None
        facts = AC.require_approval(cfg)
        prereg = AC.load_prereg(cfg if cfg is not None else AC.ApprovalConfig()).payload
    except AC.ApprovalRefused as error:
        text = str(error)
        raise Refusal(text if text.startswith("refused") else f"refused: {text}") from None
    if prereg.get("constants") != json.loads(LG.canonical(K.as_dict())):
        raise Refusal("refused: prereg.json's constants differ from near80_constants (the code is not the frozen code)")
    forget_approval_modules()
    return facts, prereg


def forget_approval_modules():
    """approval_check loads version 6's stage2_common.py from verified bytes under its own module name and leaves it in
    sys.modules; step 2's machinery.check_imports refuses any module from version 6's folder that it did not load
    itself. Once the approval is checked, that module is no longer needed here, so it is dropped from sys.modules
    (its bytes were checked against their pin when it was loaded)."""
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("near80_v6_") and f and getattr(mod, "__step2_sha256__", None) is None and \
                LG.inside(f, K.V6_CODE):
            del sys.modules[name]


def write_skip(key, out_root, why, mode, prereg_sha256=None):
    """The skipped record; prereg_sha256 is the approved fingerprint in official mode, None in scratch and stub."""
    rel = PH.skipped_rel(key)
    LG.write_json_once(Path(out_root) / rel, dict(kind="near80v9-phase-skipped", phase_id=PH.phase_id(key),
                                                  phase_kind=key.kind, setting=key.setting, world=key.world,
                                                  seed=key.seed, group=key.group, reason=why, mode=mode,
                                                  prereg_sha256=prereg_sha256, written=LG.utc()))
    say(f"skipped {PH.phase_id(key)}: {why}")


def write_stop(run, stop, facts):
    k = run.key
    body = dict(kind="near80v9-phase-stop", phase_id=run.phase_id, phase_kind=k.kind, setting=k.setting, world=k.world,
                seed=k.seed, group=k.group, reason=stop.reason, detail=stop.detail, checks=stop.checks, mode=run.mode,
                prereg_sha256=run.prereg_sha256, approval=facts, written=LG.utc())
    if k.kind == "policy":
        import policy_phase as PP
        LG.write_json_once(Path(run.out_root) / PH.policy_stop_rel(k),
                           dict(PP.policy_stop_payload(run, stop), prereg_sha256=run.prereg_sha256, written=LG.utc()))
    LG.write_json_once(Path(run.out_root) / PH.stop_rel(k), body)
    say(f"STOP {run.phase_id}: {stop.reason}")


# --------------------------------------------------------------------------------------------- dispatch

def needs_training_file(key):
    if key.kind not in PH.TRAINING_ROW_KINDS:
        return False
    return key.setting == "main" and key.seed in K.CONFIRMATORY


def dispatch(run, fresh_path=False):
    """The phase's record (the look phases write their own files and return the look)."""
    import settingdata as SD
    k = run.key
    if k.kind == "data_main":
        return SD.fresh_path_check(run) if fresh_path else SD.data_main(run)
    if k.kind in LOOK_KINDS:
        pooled = K.load_local("pooled")
        return pooled.run_look(k.group, run.out_root, run.mode, run.prereg)
    if k.kind == "lock":
        import policy_phase as PP
        return PP.stage_lock(run)
    cache = {}
    tf = None
    ctx_seed = k.seed
    if needs_training_file(k):
        ctx0 = SD.context(run, k.setting, ctx_seed)
        passed, detail, prep = SD.check_training_file(run, k.setting, ctx0.seed(k.world), ctx0.learn)
        tf = dict(passed=passed, detail=detail)
        if not passed:
            raise PH.PhaseStop(PH.STOP_REASONS["training_file"], detail)
        cache[k.world] = prep
        ctx = SD.context(run, k.setting, ctx_seed, prepared_cache=cache)
        ctx._sd.update(ctx0._sd)
    else:
        setting = "main" if k.kind == "critic_full" else k.setting
        ctx = SD.context(run, setting, ctx_seed)
    if k.kind in ("critic", "critic_full", "critic_second", "switch", "clone", "reference_actors"):
        import learner_phases as LP
        rec = LP.DISPATCH[k.kind](run, ctx)
    elif k.kind == "policy":
        import policy_phase as PP
        rec = PP.run_policy(run, ctx)
    elif k.kind == "evaluate":
        import evaluate_phase as EP
        rec = EP.evaluate(run, ctx)
    else:
        raise Refusal(f"refused: {k.kind!r} has no phase function")
    if tf is not None:
        rec["training_file_check"] = tf
    return rec


def run_phase(a, key, out_root, n, t0):
    pid = PH.phase_id(key)
    cpus = check_environment(a.mode)
    check_write_once(key, out_root)
    facts, prereg = approval_facts(a.mode, a.fixture)
    look = PH.read_look(out_root)
    skip_sha = facts.get("prereg_sha256") if a.mode == "official" else None
    ok, why = PH.admit(key, facts, look, a.mode)
    if not ok:
        if PH.is_skip(why):
            write_skip(key, out_root, why, a.mode, skip_sha)
            return 0
        raise Refusal(why)
    keys = PH.plan(spec=a.plan)
    state, why = PH.needs_state(key, out_root, keys)
    if state == "skip":
        write_skip(key, out_root, why, a.mode, skip_sha)
        return 0
    if state == "wait":
        raise Refusal(f"refused: {pid} cannot start yet: {why}")
    run = PH.Run(key, out_root, mode=a.mode, fixture=a.fixture, prereg=prereg, approval=facts, steps=a.steps,
                 chunk=a.chunk, updates=a.updates, starts=a.starts, plan=a.plan, attempt=n, log=say)
    if a.mode == "stub":
        import stubs
        if run.starts is None:
            run.starts = stubs.STUB_STARTS
        stubs.install(run)
    say(f"{pid}: {a.mode} -> {out_root}; processor {cpus}; attempt {n}")
    try:
        rec = dispatch(run, fresh_path=a.fresh_path)
    except PH.PhaseStop as stop:
        write_stop(run, stop, facts)
        return K.EXIT_STOP
    if key.kind in LOOK_KINDS:
        got = PH.read_look(out_root, key.group)
        if got is None:
            raise Refusal(f"refused: the {key.group} look did not write its record")
        say(f"{pid}: look written")
        return 0
    rec = dict(rec, mode=a.mode, prereg_sha256=run.prereg_sha256, phase_id=pid, phase_kind=key.kind,
               approval=facts, cpus=cpus, attempt=n, plan=a.plan,
               run_sizes=dict(steps=a.steps, chunk=a.chunk, updates=a.updates, starts=run.starts),
               vmhwm_mb=(LG.vmhwm_kb() or 0) / 1024.0, phase_seconds=time.perf_counter() - t0, written=LG.utc())
    rec.setdefault("seconds", rec["phase_seconds"])
    LG.write_json_once(Path(out_root) / PH.record_rel(key), rec)
    say(f"{pid}: done ({time.perf_counter() - t0:.0f}s)")
    return 0


def run_reproduction(a, out_root, t0):
    check_environment(a.mode)
    if a.mode != "scratch":
        raise Refusal("refused: the reproduction check runs in scratch mode only")
    facts, prereg = approval_facts(a.mode, None)
    key = PH.make_key("evaluate", "main", "stall", 0)
    run = PH.Run(key, out_root, mode="scratch", prereg=prereg, approval=facts, log=say)
    import evaluate_phase as EP
    items = EP.reproduction(run)
    passed = all(v["passed"] for w in items.values() for v in w.values())
    LG.write_json_once(Path(out_root) / "reproduction.json",
                       dict(kind="near80v9-reproduction", items=items, passed=bool(passed), approval=facts,
                            seconds=time.perf_counter() - t0, vmhwm_mb=(LG.vmhwm_kb() or 0) / 1024.0,
                            written=LG.utc()))
    say(f"reproduction check: {'passed' if passed else 'FAILED'}")
    return 0


def parser():
    ap = argparse.ArgumentParser(description="one near-80 phase (started by near80_runner.py)")
    ap.add_argument("--phase-id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="official", choices=("official", "scratch", "stub"))
    ap.add_argument("--fixture")
    ap.add_argument("--plan", default="full")
    ap.add_argument("--attempt", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--chunk", type=int)
    ap.add_argument("--updates", type=int)
    ap.add_argument("--starts", type=int)
    ap.add_argument("--fresh-path", action="store_true", help="scratch only: the fresh-seed path on seed 0")
    ap.add_argument("--reproduction", action="store_true", help="scratch only: the reproduction check")
    return ap


def main(argv=None):
    t0 = time.perf_counter()
    a = parser().parse_args(argv)
    try:
        check_sizes(a, a.mode)
        out_root = check_out_root(a.out, a.mode, a.fixture)
        if a.reproduction:
            if a.phase_id or a.steps is not None:
                raise Refusal("refused: --reproduction takes no phase id and runs at full length")
            out_root.mkdir(parents=True, exist_ok=True)
            handle = take_lock(out_root, "reproduction")
            try:
                return run_reproduction(a, out_root, t0)
            finally:
                handle.close()
        if not a.phase_id:
            raise Refusal("refused: --phase-id is required")
        key = PH.parse_id(a.phase_id)
        PH.check_key(key, a.mode, a.plan)
        if a.fresh_path and (a.mode != "scratch" or key.kind != "data_main" or key.seed != 0):
            raise Refusal("refused: --fresh-path is a scratch check of data_main on seed 0 only")
    except (Refusal, PH.Refused, ValueError) as error:
        text = str(error)
        print(text if text.startswith("refused") else f"refused: {text}", file=sys.stderr, flush=True)
        return K.EXIT_REFUSED
    pid = PH.phase_id(key)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        handle = take_lock(out_root, pid)
    except Refusal as error:
        print(str(error), file=sys.stderr, flush=True)
        return K.EXIT_REFUSED
    n = a.attempt if a.attempt is not None else LG.allocate_attempt(out_root, pid, mode=a.mode, cpu=None, lane=None,
                                                                    manual=True)
    code = 1
    try:
        code = run_phase(a, key, out_root, n, t0)
    except (Refusal, PH.Refused) as error:
        text = str(error)
        print(text if text.startswith("refused") else f"refused: {text}", file=sys.stderr, flush=True)
        code = K.EXIT_REFUSED
    except LG.RecordError as error:
        print(str(error), file=sys.stderr, flush=True)
        code = K.EXIT_REFUSED
    except SystemExit as error:
        text = str(error.code) if error.code is not None else ""
        if isinstance(error.code, int):
            code = error.code
        else:
            print(text, file=sys.stderr, flush=True)
            code = K.EXIT_REFUSED if text.startswith("refused") else 1
    except BaseException:  # noqa: BLE001 - every failure is reported with its traceback and an exit code
        traceback.print_exc()
        sys.stderr.flush()
        code = 1
    finally:
        extra = {}
        if a.mode == "stub":
            try:
                import stubs
                extra = stubs.report()
            except Exception:  # noqa: BLE001
                extra = {}
        try:
            LG.write_usage(out_root, pid, n, time.perf_counter() - t0, code, **extra)
        except (LG.RecordError, OSError) as error:
            print(f"note: the usage record was not written: {error}", file=sys.stderr, flush=True)
        handle.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
