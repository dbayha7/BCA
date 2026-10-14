"""Stage 2, version 7: the ledger (V7_SPEC C10 "The ledger"), the phase table (C5) and the attempt record.

Standard library only, so that launch_v7.sh can call it quickly before and after every phase.

The write log. Every attempt of a phase (or probe) has one append-only file R7/ledger/<phase id>/<n>.writes, one JSON
object per line, created once by 'begin' (launch_v7.sh) or by allocate_attempt() (a probe). Lines, in order:
  launch-start   launch_v7.sh, before the process starts: UTC time, boot id, processor, slot, arguments
  wrapper-start  the Python process (run_stage2_v7.py or v7_probes.py)
  open-w, rename, remove, ...   sys.addaudithook, AS THEY HAPPEN: every open in a write mode (which covers
                 write_text, write_bytes and np.savez), os.rename and os.replace (both raise the 'os.rename' audit
                 event), os.remove/unlink, os.rmdir, shutil.rmtree, os.link, os.symlink, os.truncate; a killed attempt
                 still leaves its list
  checks, constants, reference, facts, handover   what the wrapper verified and installed (C1, C3, C4, C7, C10)
  runner-signal  the runner, before it sends a signal (so that the interruption is classed machine-wide, C5)
  v7-exit        a v7 refusal or tripwire, written immediately before os._exit(97)
  py-exit        the Python process's own end (its exit code, VmHWM and CPU time)
  launch-end     launch_v7.sh, after the process ended: its real exit status (128 + N for signal N) and the boot id
The attempt record R7/ledger/<phase id>/<n>.json (+ sidecar) is written once, after the attempt, by finish_attempt()
(launch_v7.sh calls it through 'end'; the runner calls 'finish' for an attempt whose launch shell did not survive). It
records the arguments and processor, start and end times, exit status and its classification (done, machine-wide or
phase-specific interruption, v7 refusal, failed), the boot id at start and end, every file written with its SHA-256,
the writes outside the phase's table row and outside R7/ (flagged), VmHWM and CPU time, the machine facts, the
constants facts, the reference used (C7) with the lock check, any retry request, and the v7 and v6 fingerprints.
C5 finds files through the phase table, never through the ledger; the ledger is a cross-check.

The phase table (C5; pinned in prereg_v7.json by write_prereg_v7.py): each phase's files under OUT/, in the order
written, its final record and its failed-on-record file. A phase is DONE when its final record or a failed-on-record
file is present and verified (sidecar), whatever its exit code.

CLI (used by launch_v7.sh; every refusal exits 97):
  v7_ledger.py phase-id -- <run_stage2.py arguments>
  v7_ledger.py begin --r7 DIR --out DIR --cpu N --slot N [--test] -- <run_stage2.py arguments>
  v7_ledger.py begin-probe --r7 DIR --probe ID --cpu N      (the self-test, before a probe process: prints the path)
  v7_ledger.py event --ledger FILE --kind KIND [--json OBJECT]
  v7_ledger.py end --ledger FILE --status N [--log FILE] [--out DIR] [--r7 DIR] [--probe-record FILE]
  v7_ledger.py finish --ledger FILE [--log FILE] [--out DIR] [--r7 DIR] [--probe-record FILE]
"""

import argparse
import fnmatch
import json
import os
import re
import resource
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import v7_common as K  # noqa: E402

# --------------------------------------------------------------------------------------------- phase arguments

# run_stage2.py's own parser, restated argument for argument (S2C/run_stage2.py:648-656; test_v7 compares the two by
# their syntax trees), so that v7 reads --phase, --seed and --world exactly as run_stage2.py will.
PHASE_CHOICES = ("fresh_data", "fresh_learners", "select_standin", "detect_pairs", "detect_labels", "detect_standins",
                 "detect_fit", "detect_fit_weighted", "detect_score", "detect_marks", "policy_train", "policy_eval",
                 "policy_no_stall_train", "policy_no_stall_eval", "deploy", "host_train", "host_eval", "pooled")
GROUP_CHOICES = ("exploratory", "confirmatory")


def phase_parser():
    ap = argparse.ArgumentParser(prog="run_stage2.py")
    ap.add_argument("--phase", required=True, choices=PHASE_CHOICES)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--world", choices=("stall", "no_stall"), default="stall")
    ap.add_argument("--critic", choices=("fixed", "host"), default="fixed")
    ap.add_argument("--version", choices=("uniform", "weighted"), default="uniform")
    ap.add_argument("--group", choices=GROUP_CHOICES)
    return ap


def parse_phase_args(argv):
    """The namespace run_stage2.py's parser gives for these arguments; V7Refusal where it would stop."""
    ap = phase_parser()
    try:
        ns = ap.parse_args(list(argv))
    except SystemExit:
        raise K.V7Refusal(f"run_stage2.py would refuse these arguments: {list(argv)}") from None
    return ns


EXPLORATORY, CONFIRMATORY = (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)
SEED_WORLD = ("detect_pairs", "detect_labels", "detect_standins", "deploy", "host_train", "host_eval")


def effective_world(phase, world):
    """The world a phase works in: run_stage2.py ignores --world for the policy phases (S2C/run_stage2.py:685-686),
    whose world is fixed by the name; every other phase takes --world, default 'stall' (S2C/run_stage2.py:655)."""
    if phase in ("policy_train", "policy_eval"):
        return "stall"
    if phase in ("policy_no_stall_train", "policy_no_stall_eval"):
        return "no_stall"
    return world


def phase_key(ns):
    """The arguments that matter for this phase (the others run_stage2.py ignores), as a dict."""
    p = ns.phase
    if p == "select_standin":
        return dict(phase=p)
    if p in ("detect_marks", "pooled"):
        if ns.group is None:
            raise K.V7Refusal("--group is required (run_stage2.py would refuse)")
        return dict(phase=p, group=ns.group, version=ns.version) if p == "detect_marks" else dict(phase=p, group=ns.group)
    if ns.seed is None or ns.seed not in EXPLORATORY + CONFIRMATORY:
        raise K.V7Refusal("--seed must be one of 0-9 (run_stage2.py would refuse)")
    key = dict(phase=p, seed=int(ns.seed), world=effective_world(p, ns.world))
    if p in ("detect_fit", "detect_fit_weighted"):
        key["critic"] = ns.critic
    if p == "detect_score":
        key.update(critic=ns.critic, version=ns.version)
    if p in ("fresh_data", "fresh_learners"):
        key.pop("world")
    return key


def phase_id(key):
    """A canonical name for the phase (ledger and lock folders): e.g. policy_train_seed2, host_train_seed3_stall,
    detect_score_seed5_stall_fixed_weighted, pooled_exploratory."""
    p = key["phase"]
    parts = [p]
    if "seed" in key:
        parts.append(f"seed{key['seed']}")
        if p in SEED_WORLD or p.startswith("detect_"):
            parts.append(key["world"])
    for k in ("critic", "group", "version"):
        if k in key:
            parts.append(key[k])
    return "_".join(parts)


def check_phase_allowed(key):
    """What v7 runs (C7, C8): on seeds 0-4 only the policy test, the deployment check and the full host (no detection
    phase and no select_standin runs again); on seeds 5-9 every phase; the confirmatory detection marks; both pooled
    looks. V7Refusal otherwise."""
    p = key["phase"]
    if p == "select_standin":
        raise K.V7Refusal("select_standin does not run again under v7 (its choice is inherited, C7)")
    if p == "detect_marks" and key["group"] != "confirmatory":
        raise K.V7Refusal("the exploratory detection marks are inherited; v7 does not run them again (C7)")
    seed = key.get("seed")
    if seed in EXPLORATORY and not (p.startswith("policy_") or p in ("deploy", "host_train", "host_eval")):
        raise K.V7Refusal(f"{p} on gate seed {seed} does not run again under v7 (C7)")
    return key


# --------------------------------------------------------------------------------------------- the phase table

STANDIN_NAMES = ("distance", "density", "classifier_ratio", "two_head", "residual_evidence")  # S2C/detect_contract.py:88
FRESH_FILES = (("critic_gap", "stall"), ("critic_full", "stall"), ("critic_gap_no_stall", "no_stall"),
               ("clone", "stall"), ("clone_no_stall", "no_stall"), ("none", "stall"), ("full_none", "stall"),
               ("none_no_stall", "no_stall"), ("host", "stall"))                            # S2C/fresh.py:180-190


def row(key):
    """The phase's table row: files (patterns relative to OUT/, in the order written; every JSON's .sha256 sidecar
    belongs to the row too), final (alternatives, each a list of JSON files that must all be present and verified),
    failed (failed-on-record files: present and verified, with field == equals, or present at all when field is
    None)."""
    p, s = key["phase"], key.get("seed")
    w = key.get("world")
    if p == "fresh_data":
        a, b = f"fresh/stall/seed{s}/data.json", f"fresh/no_stall/seed{s}/data.json"
        return dict(files=[a, b], final=[[b]], failed=[dict(file=a, field="passed", equals=False)])
    if p == "fresh_learners":
        rec = f"fresh/stall/seed{s}/learners.json"
        files = [f"fresh/{wd}/seed{s}/{n}.msgpack" for n, wd in FRESH_FILES] + [rec]
        return dict(files=files, final=[[rec]], failed=[dict(file=rec, field="failed", equals=True)])
    if p in ("detect_pairs", "detect_labels"):
        d, n = f"detect_fresh/seed{s}/{w}", p.split("_")[1]
        return dict(files=[f"{d}/{n}.npz", f"{d}/{n}.json"], final=[[f"{d}/{n}.json"]], failed=[])
    if p == "detect_standins":
        d = f"detect_fresh/seed{s}/{w}/scores"
        files = [f for n in STANDIN_NAMES for f in (f"{d}/{n}.npz", f"{d}/{n}.json")]
        return dict(files=files, final=[[f"{d}/{n}.json" for n in STANDIN_NAMES]], failed=[])
    if p == "detect_fit":
        d = f"detect/uniform/{w}/{key['critic']}/seed{s}"
        return dict(files=[f"{d}/bank_scores.npz", f"{d}/band.msgpack", f"{d}/band.json", f"{d}/tau.json"],
                    final=[[f"{d}/tau.json"]], failed=[dict(file=f"{d}/band.json", field="failed", equals=True)])
    if p == "detect_fit_weighted":
        d = f"detect/weighted/{w}/{key['critic']}/seed{s}"
        return dict(files=[f"{d}/band.msgpack", f"{d}/band.json", f"{d}/tau.json"], final=[[f"{d}/tau.json"]],
                    failed=[])
    if p == "detect_score":
        d = f"detect/{key['version']}/{w}/{key['critic']}/seed{s}"
        return dict(files=[f"{d}/scores/*", f"{d}/scores_record.json"], final=[[f"{d}/scores_record.json"]], failed=[])
    if p == "detect_marks":
        f = f"detect/marks_{key['group']}_{key['version']}.json"
        return dict(files=[f], final=[[f]], failed=[])
    if p in ("policy_train", "policy_no_stall_train"):
        d = f"policy/seed{s}" + ("/no_stall" if p == "policy_no_stall_train" else "")
        return dict(files=[f"{d}/actors.msgpack", f"{d}/lock.json", f"{d}/stop.json"],
                    final=[[f"{d}/lock.json"], [f"{d}/stop.json"]], failed=[dict(file=f"{d}/stop.json", field=None)])
    if p in ("policy_eval", "policy_no_stall_eval"):
        f = f"policy/seed{s}" + ("/no_stall" if p == "policy_no_stall_eval" else "") + "/values.json"
        return dict(files=[f], final=[[f]], failed=[])
    if p == "deploy":
        f = f"deploy/{w}/seed{s}/values.json"
        return dict(files=[f"cache/scores_{w}_seed{s}_*.npz", f], final=[[f]], failed=[])
    if p == "host_train":
        d = f"host/{w}/seed{s}"
        return dict(files=[f"{d}/actors.msgpack", f"{d}/lock.json"], final=[[f"{d}/lock.json"]], failed=[])
    if p == "host_eval":
        f = f"host/{w}/seed{s}/values.json"
        return dict(files=[f], final=[[f]], failed=[])
    if p == "pooled":
        f = f"pooled_{key['group']}.json"
        return dict(files=[f], final=[[f]], failed=[])
    raise K.V7Refusal(f"{p} has no row in v7's phase table")


def phase_table_text():
    """The table as pinned in prereg_v7.json: each phase with patterns written with placeholders."""
    out = {}
    for p in PHASE_CHOICES:
        if p == "select_standin":
            continue
        key = dict(phase=p, seed="{K}", world="{W}", critic="{C}", version="{V}", group="{G}")
        out[p] = row(key)
    return out


def _patterns(key):
    r = row(key)
    pats = []
    for f in r["files"]:
        pats.append(f)
        if f.endswith(".json"):
            pats.append(f + ".sha256")
    return pats


def in_row(rel, key):
    rel = str(rel)
    return any(fnmatch.fnmatchcase(rel, p) for p in _patterns(key))


def row_existing_files(key, out_root):
    """Every existing file of the row under out_root (sidecars and glob matches included), relative paths sorted."""
    out_root = Path(out_root)
    found = set()
    for pat in _patterns(key):
        if any(c in pat for c in "*?["):
            base = out_root / Path(pat).parent
            if base.is_dir():
                for p in base.rglob("*"):
                    if p.is_file() and fnmatch.fnmatchcase(p.relative_to(out_root).as_posix(), pat):
                        found.add(p.relative_to(out_root).as_posix())
        elif (out_root / pat).is_file():
            found.add(pat)
    return sorted(found)


def phase_done(key, out_root):
    """(True, 'final' | 'failed-on-record', detail) or (False, None, detail). Reads through read_record: never raises
    on a damaged record (it is reported as partial)."""
    out_root = Path(out_root)
    r = row(key)
    partial = []
    for f in r["failed"]:
        state, data = K.read_record(out_root / f["file"])
        if state == "ok" and (f.get("field") is None or data.get(f["field"]) == f["equals"]):
            return True, "failed-on-record", f["file"]
        if state == "partial":
            partial.append(f["file"])
    for alternative in r["final"]:
        states = [K.read_record(out_root / f) for f in alternative]
        if all(s == "ok" for s, _ in states):
            return True, "final", alternative
        partial += [f for f, (s, _) in zip(alternative, states) if s == "partial"]
    return False, None, dict(partial=sorted(set(partial)))


# --------------------------------------------------------------------------------------------- the write log

_CURRENT = {}
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
LOGGED_EVENTS = {"os.rename": "rename", "os.remove": "remove", "os.rmdir": "rmdir", "shutil.rmtree": "rmtree",
                 "os.link": "link", "os.symlink": "symlink", "os.truncate": "truncate"}


def usage():
    """VmHWM (kB, from /proc/self/status) and CPU seconds of this process so far."""
    hwm = None
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                hwm = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return dict(vmhwm_kb=hwm, cpu_seconds=round(ru.ru_utime + ru.ru_stime, 3), maxrss_kb=int(ru.ru_maxrss))


def _path_text(p):
    if isinstance(p, int):
        return f"<fd {p}>"
    try:
        return os.path.abspath(os.fsdecode(p))
    except Exception:  # noqa: BLE001 - never stop the phase for the log's sake
        return repr(p)


class Ledger:
    """One attempt's write log, opened once (O_APPEND) before the audit hook is installed, written with os.write so
    that logging never raises an audit event of its own."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_file():
            raise K.V7Refusal(f"the ledger file {self.path} does not exist (it is created by 'begin')")
        self.fd = os.open(str(self.path), os.O_WRONLY | os.O_APPEND)
        self.hook = False
        _CURRENT["ledger"] = self

    def event(self, ev, /, **fields):
        line = json.dumps(dict(fields, ev=ev, utc=K.utc_now(), t=round(time.time(), 6), pid=os.getpid()),
                          sort_keys=True, default=str, allow_nan=True)
        os.write(self.fd, (line + "\n").encode("utf8", "replace"))

    def install_audit_hook(self):
        if self.hook:
            return
        event = self.event

        def hook(name, args):
            try:
                if name == "open":
                    path, mode, flags = args
                    if mode is None:
                        if not (int(flags or 0) & WRITE_FLAGS):
                            return
                        how = f"flags={int(flags)}"
                    else:
                        mode = str(mode)
                        if not any(c in mode for c in "wax+"):
                            return
                        how = mode
                    event("open-w", path=_path_text(path), mode=how)
                elif name in LOGGED_EVENTS:
                    fields = dict(path=_path_text(args[0]))
                    if name in ("os.rename", "os.link", "os.symlink") and len(args) > 1:
                        fields["dst"] = _path_text(args[1])
                    event(LOGGED_EVENTS[name], **fields)
            except Exception:  # noqa: BLE001 - an audit hook must never raise into the phase
                pass

        sys.addaudithook(hook)
        self.hook = True

    def hard_exit(self, reason, code=K.EXIT_V7):
        hard_exit(reason, code, self)


def current():
    return _CURRENT.get("ledger")


def hard_exit(reason, code=K.EXIT_V7, ledger=None):
    """One ledger line, one line on standard error, then os._exit(code): never an exception, because pinned v6 code
    catches any Exception around the stand-in rival (S2C/policy.py:419, :558)."""
    ledger = ledger or current()
    try:
        if ledger is not None:
            ledger.event("v7-exit", code=int(code), reason=str(reason), **usage())
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.stderr.write(f"REFUSED (v7, exit {code}): {reason}\n")
        sys.stderr.flush()
        sys.stdout.flush()
    except Exception:  # noqa: BLE001
        pass
    os._exit(int(code))


def allocate_attempt(r7, pid):
    """Create R7/ledger/<pid>/<n>.writes exclusively, n one more than the largest existing attempt. Returns (n, path)."""
    d = Path(r7) / "ledger" / pid
    d.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split(".")[0]) for p in d.glob("*.writes") if p.name.split(".")[0].isdigit()]
    n = max(nums, default=0) + 1
    while True:
        path = d / f"{n}.writes"
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(fd)
            return n, path
        except FileExistsError:
            n += 1


def append_event(writes_path, ev, /, **fields):
    """Append one line to an attempt's write log from another process (launch_v7.sh, the runner)."""
    line = json.dumps(dict(fields, ev=ev, utc=K.utc_now(), t=round(time.time(), 6), pid=os.getpid()),
                      sort_keys=True, default=str)
    fd = os.open(str(writes_path), os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, (line + "\n").encode("utf8"))
    finally:
        os.close(fd)


def read_events(path):
    """Every complete line of a write log (a torn last line is skipped and counted)."""
    events, torn = [], 0
    try:
        text = Path(path).read_text(encoding="utf8", errors="replace")
    except OSError:
        return events, torn
    for line in text.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            torn += 1
    return events, torn


# --------------------------------------------------------------------------------------------- classification (C5)

MEMORY_PATTERNS = ("MemoryError", "RESOURCE_EXHAUSTED", "out of memory")
MACHINE_SIGNALS = {1: "SIGHUP", 2: "SIGINT", 15: "SIGTERM"}
PHASE_SIGNALS = {9: "SIGKILL", 11: "SIGSEGV", 7: "SIGBUS", 6: "SIGABRT"}


def signal_of(status):
    """The signal behind an exit status as the shell records it (128 + N), or a negative return code (-N)."""
    if status is None:
        return None
    status = int(status)
    if status < 0:
        return -status
    if 128 < status < 128 + 65:
        return status - 128
    return None


def log_ends_in_memory_error(log_tail):
    lines = [ln for ln in (log_tail or "").splitlines() if ln.strip()][-40:]
    return any(p in ln for ln in lines for p in MEMORY_PATTERNS)


def classify_attempt(done, status, boot_start, boot_end, runner_signals=(), log_tail=""):
    """C5's rule, declared before any laptop result: (kind, why). kind is one of
      'done'            its final record or a failed-on-record file is present and verified (any exit code);
      'machine-wide'    the boot id changed; or the runner stopped it; or SIGHUP, SIGINT or SIGTERM; or no exit status
                        was recorded (a closed terminal or tmux session) -- never counted, rerun without limit;
      'phase-specific'  on the same boot: SIGKILL, SIGSEGV, SIGBUS or SIGABRT the runner did not send, or a log that
                        ends in a memory error (MemoryError, RESOURCE_EXHAUSTED, 'out of memory') -- counted;
      'v7-refusal'      exit 97: halts the run;
      'failed'          any other non-zero exit, or exit 0 without the phase being done: halts new phases."""
    if done:
        return "done", "its final record or failed-on-record file is present and verified"
    if boot_start and boot_end and boot_start != boot_end:
        return "machine-wide", "the machine or WSL restarted while it ran (the boot id changed)"
    if runner_signals:
        return "machine-wide", f"the runner stopped it (signals {sorted(set(int(s) for s in runner_signals))})"
    if status is None:
        return "machine-wide", "no exit status was recorded (its launch shell ended with it)"
    sig = signal_of(status)
    if sig in MACHINE_SIGNALS:
        return "machine-wide", f"ended by {MACHINE_SIGNALS[sig]}"
    if sig in PHASE_SIGNALS:
        return "phase-specific", f"killed by {PHASE_SIGNALS[sig]} on the same boot, not sent by the runner"
    if log_ends_in_memory_error(log_tail):
        return "phase-specific", "its log ends in a memory error"
    if int(status) == K.EXIT_V7:
        return "v7-refusal", "exit 97 (a v7 refusal)"
    if sig is not None:
        return "failed", f"ended by signal {sig}"
    return "failed", f"exit {status} without the phase being done"


# --------------------------------------------------------------------------------------------- the attempt record

def _tail(path, limit=20000):
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf8", "replace")
    except OSError:
        return ""


def _first(events, ev):
    return next((e for e in events if e.get("ev") == ev), None)


def _last(events, ev):
    return next((e for e in reversed(events) if e.get("ev") == ev), None)


def finish_attempt(writes_path, out_root=None, log_path=None, r7=None, now_boot=None, retry_request=None,
                   probe_record=None):
    """Write R7/ledger/<phase id>/<n>.json (+ sidecar) once, from the write log, the phase table and the files as they
    are now. Returns the record (or the existing one when it was written already)."""
    writes_path = Path(writes_path)
    rec_path = writes_path.with_name(writes_path.name.replace(".writes", ".json"))
    state, existing = K.read_record(rec_path)
    if state == "ok":
        return existing
    events, torn = read_events(writes_path)
    start = _first(events, "launch-start") or _first(events, "wrapper-start") or {}
    end = _last(events, "launch-end")
    r7 = Path(r7 or start.get("r7") or writes_path.parent.parent.parent)
    out_root = Path(out_root or start.get("out_root") or K.OUT)
    status = None if end is None else end.get("status")
    boot_start = start.get("boot_id")
    boot_end = (end or {}).get("boot_id") or (now_boot if now_boot is not None else K.boot_id())
    key = start.get("key")
    probe = start.get("probe")
    done, how = False, None
    if key:
        done, how, _ = phase_done(key, out_root)
    elif probe and probe_record:
        done = K.read_record(probe_record)[0] == "ok"
        how = "probe record" if done else None
    runner_signals = [e.get("signal") for e in events if e.get("ev") == "runner-signal"]
    kind, why = classify_attempt(done, status, boot_start, boot_end, runner_signals, _tail(log_path) if log_path else "")
    writes = {}
    for e in events:
        if e.get("ev") in ("open-w", "rename", "remove", "rmdir", "rmtree", "link", "symlink", "truncate"):
            target = e.get("dst") if e.get("ev") == "rename" else e.get("path")
            if not target or target.startswith("<fd"):
                continue
            w = writes.setdefault(target, dict(path=target, events=[]))
            w["events"].append(e.get("ev"))
            if e.get("ev") == "rename" and e.get("path"):
                writes.setdefault(e["path"], dict(path=e["path"], events=[]))["events"].append("renamed-away")
    for path, w in writes.items():
        p = Path(path)
        w["sha256"] = K.file_sha256(p) if p.is_file() else None
        w["in_r7"] = K.inside(p, r7)
        w["in_row"] = bool(key) and K.inside(p, out_root) and in_row(Path(os.path.abspath(p)).relative_to(
            os.path.abspath(out_root)).as_posix(), key)
        w["flagged"] = not (w["in_r7"] or w["in_row"])
    py_exit = _last(events, "py-exit") or {}
    v7_exit = _last(events, "v7-exit") or {}
    reference = _last(events, "reference")
    ref_check = None
    if reference and reference.get("entry") and key:
        ref_check = check_lock_records_reference(reference["entry"], out_root)
    m = re.match(r"(\d+)\.writes$", writes_path.name)
    record = dict(
        kind="stage2-v7-ledger-attempt", phase_id=writes_path.parent.name, attempt=int(m.group(1)) if m else None,
        key=key, probe=probe, argv=start.get("argv"), cpu=start.get("cpu"), slot=start.get("slot"),
        test_mode=bool(start.get("test")), start_utc=start.get("utc"), end_utc=(end or {}).get("utc"),
        exit_status=status, signal=signal_of(status), boot_id_start=boot_start, boot_id_end=boot_end,
        classification=dict(kind=kind, why=why), done=done, done_how=how,
        writes=sorted(writes.values(), key=lambda w: w["path"]),
        flagged_writes=sorted(w["path"] for w in writes.values() if w["flagged"]),
        vmhwm_kb=(py_exit or v7_exit).get("vmhwm_kb"), cpu_seconds=(py_exit or v7_exit).get("cpu_seconds"),
        py_exit=py_exit or None, v7_exit=v7_exit or None, runner_signals=runner_signals,
        checks=(_last(events, "checks") or {}).get("result"), constants=(_last(events, "constants") or {}).get("facts"),
        facts=(_last(events, "facts") or {}).get("facts"), reference=reference, reference_lock_check=ref_check,
        retry_request=retry_request, fingerprints=(_last(events, "checks") or {}).get("fingerprints"),
        torn_lines=torn, write_log=str(writes_path), log=str(log_path) if log_path else None)
    try:
        K.write_json_once(rec_path, record)
    except K.V7Refusal:
        state, existing = K.read_record(rec_path)
        return existing if state == "ok" else record
    return record


def check_lock_records_reference(entry, out_root):
    """After a phase with a re-made reference (C7, Labels): the lock must record, under the lock name of each replaced
    actor, the re-made actor's SHA-256. Returns dict(ok, lock, details); ok None when no lock was written."""
    lock_rel = entry.get("lock")
    state, lock = K.read_record(Path(out_root) / lock_rel) if lock_rel else ("missing", None)
    if state != "ok":
        return dict(ok=None, lock=lock_rel, details=f"lock {state}")
    rows, ok = {}, True
    for name, spec in (entry.get("replace") or {}).items():
        got = (lock.get("actor_sha256") or {}).get(spec["lock_name"])
        rows[spec["lock_name"]] = dict(expected=spec["actor_sha256"], recorded=got, same=got == spec["actor_sha256"])
        ok = ok and got == spec["actor_sha256"]
    return dict(ok=bool(ok), lock=lock_rel, details=rows)


# --------------------------------------------------------------------------------------------- moving files aside

def move_files_aside(rel_paths, base, dest, record_path, **fields):
    """Move (never delete) each existing file base/<rel> to dest/<rel> (one rename each, same file system) and write
    record_path (+ sidecar) with their SHA-256 and the given fields. C5 (interrupted phases), C7 (an interrupted probe;
    a damaged reference). Returns the record."""
    base, dest = Path(base), Path(dest)
    moved = {}
    for rel in rel_paths:
        src = base / rel
        if not src.is_file():
            continue
        target = dest / rel
        if target.exists():
            raise K.V7Refusal(f"{target} exists; nothing is overwritten")
        target.parent.mkdir(parents=True, exist_ok=True)
        moved[rel] = K.file_sha256(src)
        os.rename(src, target)
    record = dict(fields, kind="stage2-v7-moved-aside", base=str(base), dest=str(dest), files=moved,
                  moved_utc=K.utc_now())
    K.write_json_once(record_path, record)
    return record


def move_interrupted(key, out_root, r7, kind, why):
    """C5: every existing file of the phase's row goes to R7/interrupted/<phase id>/<UTC time>/ at the same relative
    paths; R7/interrupted/<phase id>/<UTC time>.json records their SHA-256 and the kind of interruption."""
    pid = phase_id(key)
    stamp = K.utc_stamp()
    d = Path(r7) / "interrupted" / pid
    return move_files_aside(row_existing_files(key, out_root), out_root, d / stamp, d / f"{stamp}.json",
                            phase_id=pid, interruption=kind, why=why)


# --------------------------------------------------------------------------------------------- command line

def _split(argv):
    argv = list(argv)
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    return argv, []


def main(argv=None):
    own, rest = _split(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("phase-id")
    b = sub.add_parser("begin")
    b.add_argument("--r7", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--cpu", required=True)
    b.add_argument("--slot", required=True)
    b.add_argument("--test", action="store_true")
    bp = sub.add_parser("begin-probe")
    bp.add_argument("--r7", required=True)
    bp.add_argument("--probe", required=True)
    bp.add_argument("--cpu", required=True)
    e = sub.add_parser("event")
    e.add_argument("--ledger", required=True)
    e.add_argument("--kind", required=True)
    e.add_argument("--json", default="{}")
    for name in ("end", "finish"):
        f = sub.add_parser(name)
        f.add_argument("--ledger", required=True)
        if name == "end":
            f.add_argument("--status", required=True, type=int)
        f.add_argument("--log")
        f.add_argument("--out")
        f.add_argument("--r7")
        f.add_argument("--probe-record")
    a = ap.parse_args(own)
    try:
        if a.cmd == "phase-id":
            key = check_phase_allowed(phase_key(parse_phase_args(rest)))
            print(phase_id(key))
        elif a.cmd == "begin":
            key = check_phase_allowed(phase_key(parse_phase_args(rest)))
            pid = phase_id(key)
            n, path = allocate_attempt(a.r7, pid)
            append_event(path, "launch-start", key=key, phase_id=pid, attempt=n, argv=rest, cpu=int(a.cpu),
                         slot=int(a.slot), boot_id=K.boot_id(), r7=str(Path(a.r7).resolve()),
                         out_root=str(Path(a.out).resolve()), test=bool(a.test), shell_pid=os.getppid())
            print(path)
        elif a.cmd == "begin-probe":
            if not re.fullmatch(r"(p1_seed[234]_(stall|no_stall)|p2_seed[234]|p3_seed\d|r1_seed[234]_(stall|no_stall)|"
                                r"remake_p[12]_seed[234](_stall|_no_stall)?)", a.probe):
                raise K.V7Refusal(f"unknown probe {a.probe!r}")
            n, path = allocate_attempt(a.r7, f"probe_{a.probe}")
            append_event(path, "launch-start", probe=a.probe, attempt=n, cpu=int(a.cpu), boot_id=K.boot_id(),
                         r7=str(Path(a.r7).resolve()), out_root=str(K.OUT), shell_pid=os.getppid())
            print(path)
        elif a.cmd == "event":
            append_event(a.ledger, a.kind, **json.loads(a.json))
        else:
            if a.cmd == "end":
                append_event(a.ledger, "launch-end", status=a.status, boot_id=K.boot_id())
            rec = finish_attempt(a.ledger, out_root=a.out, log_path=a.log, r7=a.r7, probe_record=a.probe_record)
            print(f"{rec['phase_id']} attempt {rec['attempt']}: {rec['classification']['kind']} "
                  f"({rec['classification']['why']})")
    except K.V7Refusal as error:
        print(f"refused: {error}", file=sys.stderr)
        return K.EXIT_V7
    return 0


if __name__ == "__main__":
    sys.exit(main())
