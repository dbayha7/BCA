"""Near-80 (version 9, main setting): strict JSON records, the attempt ledger, peak memory, interruption classes and
the moving aside of an interrupted phase's files (version 7's rules, restated with near-80 names). DRAFT, not frozen.

Standard library only, so that the runner can read the ledger without numpy or JAX. The file locks use fcntl where
it exists (the desktop, the cluster) and msvcrt otherwise (the unit tests on Windows); official runs need fcntl.

Records. Every JSON any near-80 code writes is json.dumps(plain(payload), indent=1, sort_keys=True,
allow_nan=False) encoded as UTF-8 (non-finite floats become null), written once (os.O_CREAT | os.O_EXCL) with a
sidecar '<name>.sha256' holding '<hex>  <name>\\n'. read_json_verified refuses a file whose sidecar is missing or
differs; read_record never raises and reports 'ok', 'missing' or 'partial'.

The ledger. Each attempt of a phase has, under OUT/ledger/<phase_id>/:
  <n>.start.json   written once by the runner when it launches attempt n (time, boot id, processor, lane, mode,
                   retry request);
  <n>.signals      append-only lines, one per signal the runner sends to the attempt (so that the interruption is
                   classed machine-wide even when the runner itself dies afterwards);
  <n>.usage.json   written once by the phase process at its exit: {vmhwm_kb, seconds, exit_code};
  <n>.json         the attempt record, written once by the runner at the end of the attempt (or, for an attempt whose
                   runner did not survive, at the next start once nothing of the phase runs).
The attempt record (kind 'near80-attempt') is the schema the looks may read for the resource table.

Interruption classes (classify_attempt, version 7's rule): machine-wide (a changed boot id, a signal the runner sent,
SIGHUP, SIGINT or SIGTERM, no exit status) reruns without limit and is never counted; phase-specific (SIGKILL, SIGSEGV,
SIGBUS or SIGABRT the runner did not send, or a log that ends in a memory error) reruns twice as usual, the third time
alone, then the phase is unfinished and waits for 'retry'. Exit 97 is a refusal and halts the run; any other
non-zero exit, or exit 0 without the phase's final record, is a failure and stops new phases. A phase that exits 2
with its stop file is stopped and never rerun.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

try:
    import fcntl  # noqa: E402
except ImportError:                                   # Windows (the unit tests only)
    fcntl = None
    import msvcrt  # noqa: E402

ATTEMPT_KIND = "near80-attempt"
HAS_FCNTL = fcntl is not None


# --------------------------------------------------------------------------------------------- portable pieces

class LockHeld(OSError):
    """Another process holds the lock."""


def lock_exclusive(handle):
    """A non-blocking exclusive lock on an open file; LockHeld when another process holds it."""
    try:
        if HAS_FCNTL:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as error:
        raise LockHeld(str(error)) from None


def unlock(handle):
    try:
        if HAS_FCNTL:
            fcntl.flock(handle, fcntl.LOCK_UN)
        else:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def lock_free(path):
    """True when no process holds the lock on path (a missing file is free)."""
    p = Path(path)
    if not p.exists():
        return True
    try:
        handle = open(p, "a+")
    except OSError:
        return True
    try:
        lock_exclusive(handle)
        unlock(handle)
        return True
    except LockHeld:
        return False
    finally:
        handle.close()


def cpu_affinity():
    """The processors this process may use: sched_getaffinity where it exists, else every processor."""
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return list(range(os.cpu_count() or 1))


def work_dir():
    """The folder phase processes start in: /tmp where it exists, else the system's temporary folder."""
    return "/tmp" if os.path.isdir("/tmp") else tempfile.gettempdir()
MOVED_KIND = "near80-moved-aside"
EXIT_REFUSED, EXIT_STOP = 97, 2
MEMORY_PATTERNS = ("MemoryError", "RESOURCE_EXHAUSTED", "out of memory")
MACHINE_SIGNALS = {1: "SIGHUP", 2: "SIGINT", 15: "SIGTERM"}
PHASE_SIGNALS = {9: "SIGKILL", 11: "SIGSEGV", 7: "SIGBUS", 6: "SIGABRT"}
CLASSES = ("done", "stopped", "refused", "failed", "machine-wide", "phase-specific")
INTERRUPTED = ("machine-wide", "phase-specific")


class RecordError(Exception):
    """A record is missing, has no sidecar, or differs from its sidecar; or a write-once file exists."""


# --------------------------------------------------------------------------------------------- JSON and hashes

def plain(x):
    """Plain JSON types, recursively: numpy arrays and scalars through tolist(), non-finite floats to None, paths to
    strings, tuples to lists."""
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist") and not isinstance(x, (bytes, bytearray)):
        return plain(x.tolist())
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    return x


def canonical(payload):
    return json.dumps(plain(payload), indent=1, sort_keys=True, allow_nan=False)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sidecar_path(path):
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def sidecar_digest(path):
    """The digest a '<name>.sha256' sidecar holds, or None when it is missing or unreadable."""
    s = sidecar_path(path)
    try:
        return s.read_text(encoding="utf8").split()[0]
    except (OSError, IndexError):
        return None


def _write_excl(path, data):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
    finally:
        os.close(fd)


def write_json_once(path, payload):
    """Strict JSON (canonical) and its sidecar, both created exclusively. Refuses (RecordError) when either exists.
    Returns the SHA-256 of the text."""
    p = Path(path)
    side = sidecar_path(p)
    if p.exists() or side.exists():
        raise RecordError(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(payload).encode("utf8")
    digest = sha256_bytes(data)
    try:
        _write_excl(p, data)
    except FileExistsError:
        raise RecordError(f"refused: {p} exists (write-once)") from None
    _write_excl(side, f"{digest}  {p.name}\n".encode("utf8"))
    return digest


def write_bytes_once(path, blob):
    """A binary file created exclusively; returns its SHA-256."""
    p = Path(path)
    if p.exists():
        raise RecordError(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_excl(p, bytes(blob))
    except FileExistsError:
        raise RecordError(f"refused: {p} exists (write-once)") from None
    return sha256_bytes(bytes(blob))


def write_json_replace(path, payload):
    """The one rewritable file (status.json): text through a temporary file and os.replace, then its sidecar the
    same way."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(payload).encode("utf8")
    digest = sha256_bytes(data)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    stmp = p.with_name(p.name + f".sha256.tmp{os.getpid()}")
    stmp.write_text(f"{digest}  {p.name}\n", encoding="utf8")
    os.replace(stmp, sidecar_path(p))
    return digest


def read_json_verified(path):
    """A JSON with a matching sidecar; RecordError on a missing file, a missing sidecar or a mismatch."""
    p = Path(path)
    if not p.is_file():
        raise RecordError(f"refused: {p} is missing")
    data = p.read_bytes()
    want = sidecar_digest(p)
    if want is None:
        raise RecordError(f"refused: {p} has no .sha256 sidecar")
    if sha256_bytes(data) != want:
        raise RecordError(f"refused: {p} differs from its .sha256 sidecar")
    return json.loads(data.decode("utf8"))


def read_record(path):
    """('ok', body) | ('missing', None) | ('partial', why): never raises."""
    p = Path(path)
    if not p.exists() and not sidecar_path(p).exists():
        return "missing", None
    try:
        return "ok", read_json_verified(p)
    except (RecordError, ValueError, OSError) as error:
        return "partial", str(error)


def ok_record(path):
    state, body = read_record(path)
    return body if state == "ok" else None


# --------------------------------------------------------------------------------------------- time and machine

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_stamp():
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def boot_id(path="/proc/sys/kernel/random/boot_id"):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def vmhwm_kb():
    """Peak resident memory of this process (VmHWM, kB) or None."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def inside(path, root):
    p, r = os.path.realpath(str(path)), os.path.realpath(str(root))
    return p == r or p.startswith(r.rstrip("/") + "/")


# --------------------------------------------------------------------------------------------- attempts

ATTEMPT_RE = re.compile(r"^(\d+)\.start\.json$")


def ledger_dir(out_root, phase_id):
    return Path(out_root) / "ledger" / phase_id


def attempt_numbers(out_root, phase_id):
    d = ledger_dir(out_root, phase_id)
    if not d.is_dir():
        return []
    return sorted(int(m.group(1)) for m in (ATTEMPT_RE.match(p.name) for p in d.iterdir()) if m)


def allocate_attempt(out_root, phase_id, **fields):
    """Create OUT/ledger/<phase_id>/<n>.start.json exclusively, n one more than the largest existing attempt (never
    reused). Returns n."""
    d = ledger_dir(out_root, phase_id)
    d.mkdir(parents=True, exist_ok=True)
    n = max(attempt_numbers(out_root, phase_id), default=0) + 1
    while True:
        try:
            write_json_once(d / f"{n}.start.json", dict(fields, kind="near80-attempt-start", phase_id=phase_id, n=n,
                                                        started_utc=utc(), started_epoch=time.time(),
                                                        boot_id_start=boot_id(), pid=os.getpid()))
            return n
        except RecordError:
            n += 1


def append_signal(out_root, phase_id, n, sig, why):
    """One line per signal the runner sends to attempt n, written BEFORE the signal is sent."""
    d = ledger_dir(out_root, phase_id)
    d.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(signal=int(sig), why=str(why), utc=utc()), sort_keys=True)
    fd = os.open(str(d / f"{n}.signals"), os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        os.write(fd, (line + "\n").encode("utf8"))
    finally:
        os.close(fd)


def runner_signals(out_root, phase_id, n):
    p = ledger_dir(out_root, phase_id) / f"{n}.signals"
    out = []
    try:
        for line in p.read_text(encoding="utf8").splitlines():
            try:
                out.append(int(json.loads(line)["signal"]))
            except (ValueError, KeyError, TypeError):
                continue
    except OSError:
        pass
    return out


def write_usage(out_root, phase_id, n, seconds, exit_code, **extra):
    """The phase process's own record of its end (written once, at exit)."""
    return write_json_once(ledger_dir(out_root, phase_id) / f"{n}.usage.json",
                           dict(extra, kind="near80-attempt-usage", phase_id=phase_id, n=n, vmhwm_kb=vmhwm_kb(),
                                seconds=float(seconds), exit_code=int(exit_code), ended_utc=utc()))


def usage(out_root, phase_id, n):
    return ok_record(ledger_dir(out_root, phase_id) / f"{n}.usage.json")


def start_record(out_root, phase_id, n):
    return ok_record(ledger_dir(out_root, phase_id) / f"{n}.start.json")


def attempt_record(out_root, phase_id, n):
    return ok_record(ledger_dir(out_root, phase_id) / f"{n}.json")


def attempt_records(out_root, phase_id):
    """{n: attempt record or None (not written yet)} for every allocated attempt."""
    return {n: attempt_record(out_root, phase_id, n) for n in attempt_numbers(out_root, phase_id)}


def signal_of(status):
    """The signal behind an exit status: a negative return code (-N) or the shell's 128 + N."""
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


def classify_attempt(done, status, boot_start, boot_end, runner_signals=(), log_tail="", stopped=False):
    """Version 7's rule (v7_ledger.classify_attempt) with near-80's two extra endings: (kind, why), kind one of
      'done'            the phase's final record is present and verified (any exit code);
      'stopped'         its stop file is present and verified (a recorded stop, exit 2): never rerun;
      'machine-wide'    the boot id changed, the runner sent a signal, SIGHUP/SIGINT/SIGTERM, or no exit status;
      'phase-specific'  SIGKILL, SIGSEGV, SIGBUS or SIGABRT on the same boot not sent by the runner, or a log that
                        ends in a memory error;
      'refused'         exit 97 (tested before the memory patterns: a refusal is never an interruption);
      'failed'          any other non-zero exit, or exit 0 without the phase's final record.
    log_tail must be THIS attempt's own output."""
    if done:
        return "done", "its final record is present and verified"
    if stopped:
        return "stopped", "its stop file is present and verified (a recorded stop; never rerun)"
    if boot_start and boot_end and boot_start != boot_end:
        return "machine-wide", "the machine restarted while it ran (the boot id changed)"
    if runner_signals:
        return "machine-wide", f"the runner stopped it (signals {sorted(set(int(s) for s in runner_signals))})"
    if status is None:
        return "machine-wide", "no exit status was recorded (the process that waited for it ended with it)"
    sig = signal_of(status)
    if sig in MACHINE_SIGNALS:
        return "machine-wide", f"ended by {MACHINE_SIGNALS[sig]}"
    if sig in PHASE_SIGNALS:
        return "phase-specific", f"killed by {PHASE_SIGNALS[sig]} on the same boot, not sent by the runner"
    if int(status) == EXIT_REFUSED:
        return "refused", "exit 97 (a refusal)"
    if log_ends_in_memory_error(log_tail):
        return "phase-specific", "its log ends in a memory error"
    if sig is not None:
        return "failed", f"ended by signal {sig}"
    return "failed", f"exit {status} without the phase's final record"


START_LINE_RE = re.compile(r"\bstart: attempt (\d+) of (\S+)\b")


def start_line(phase_id, n, cpu):
    """The line the runner appends to the phase's log before attempt n (attempt_log_text finds it)."""
    return f"{utc()} start: attempt {int(n)} of {phase_id} on processor {cpu}"


def attempt_log_text(log_path, phase_id, n):
    """THIS attempt's own lines of the per-phase log (which every attempt appends to): from its start line up to the
    next start line. Empty when no start line names it."""
    try:
        lines = Path(log_path).read_bytes().decode("utf8", "replace").splitlines()
    except OSError:
        return ""
    starts = [(i, int(m.group(1)), m.group(2)) for i, m in ((i, START_LINE_RE.search(ln)) for i, ln in
                                                          enumerate(lines)) if m]
    mine = [i for i, k, pid in starts if k == int(n) and pid == phase_id]
    if not mine:
        return ""
    first = mine[-1]
    later = [i for i, _, _ in starts if i > first]
    return "\n".join(lines[first + 1:later[0] if later else len(lines)])


def finish_attempt(out_root, key_fields, phase_id, n, *, done, stopped, exit_code, log_path=None, now_boot=None,
                   cpu=None, lane=None, mode=None, retry_request=None):
    """Write OUT/ledger/<phase_id>/<n>.json once (or return the existing record). key_fields: the phase key's
    fields (phase_kind, setting, world, seed, group). exit_code: the process's return code, or None when no
    process exit was observed (then the phase's own usage record, if any, gives it)."""
    existing = attempt_record(out_root, phase_id, n)
    if existing is not None:
        return existing
    start = start_record(out_root, phase_id, n) or {}
    use = usage(out_root, phase_id, n) or {}
    status = exit_code if exit_code is not None else use.get("exit_code")
    boot_start = start.get("boot_id_start")
    boot_end = boot_id() if now_boot is None else now_boot
    signals = runner_signals(out_root, phase_id, n)
    tail = attempt_log_text(log_path, phase_id, n)[-20000:] if log_path else ""
    kind, why = classify_attempt(done, status, boot_start, boot_end, signals, tail, stopped)
    seconds = use.get("seconds")
    if seconds is None and start.get("started_epoch") is not None:
        seconds = max(0.0, time.time() - float(start["started_epoch"]))
    record = dict(kind=ATTEMPT_KIND, phase_id=phase_id, n=int(n), cpu=start.get("cpu", cpu),
                  lane=start.get("lane", lane), started_utc=start.get("started_utc"), ended_utc=use.get("ended_utc")
                  or utc(), seconds=seconds, vmhwm_kb=use.get("vmhwm_kb"), exit_code=status,
                  signal=signal_of(status), classification=dict(kind=kind, why=why), boot_id_start=boot_start,
                  boot_id_end=boot_end, retry_request=start.get("retry_request", retry_request),
                  mode=start.get("mode", mode), runner_signals=signals, log=str(log_path) if log_path else None,
                  **{k: key_fields.get(k) for k in ("phase_kind", "setting", "world", "seed", "group")})
    try:
        write_json_once(ledger_dir(out_root, phase_id) / f"{n}.json", record)
    except RecordError:
        got = attempt_record(out_root, phase_id, n)
        return got if got is not None else record
    return record


def classification(record):
    return ((record or {}).get("classification") or {}).get("kind")


def phase_specific_count(records):
    return sum(1 for r in records.values() if classification(r) == "phase-specific")


# --------------------------------------------------------------------------------------------- the ledger read

def all_attempt_records(out_root):
    """Every attempt record under OUT/ledger/ (verified sidecars only)."""
    d = Path(out_root) / "ledger"
    out = []
    if not d.is_dir():
        return out
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        for n in attempt_numbers(out_root, sub.name):
            rec = attempt_record(out_root, sub.name, n)
            if rec is not None:
                out.append(rec)
    return out


def measured_peaks(out_root, extra=None):
    """{phase kind: largest peak in bytes} over attempts classified 'done' (and stopped: a stop is a full
    measurement too). The plumbing record's first peaks (extra: {kind: vmhwm_kb}) count only for a kind that no such
    attempt has measured yet: once one has, its own measurements alone set the reservation (version 7's rule: peaks
    from full-size phases)."""
    peaks = {}
    for rec in all_attempt_records(out_root):
        if classification(rec) not in ("done", "stopped") or not rec.get("vmhwm_kb"):
            continue
        kind = rec.get("phase_kind")
        if kind:
            peaks[kind] = max(peaks.get(kind, 0), int(rec["vmhwm_kb"]) * 1024)
    for kind, kb in (extra or {}).items():
        if kb and kind not in peaks:
            peaks[kind] = int(kb) * 1024
    return peaks


def stage_of(phase_kind, setting, group):
    """The budget stage of a phase: 'fresh' (the fresh seeds' phases and the fresh group's lock and look, the only
    stage of the official plan) or 'plumbing' (the smoke seed's phases and its group: scratch and stub runs)."""
    return "fresh" if group == "fresh" else "plumbing"


def cpu_hours(out_root, plumbing_seconds=None):
    """{stage: hours} summed over every attempt's seconds in the ledger (every class counts: compute was spent),
    plus the plumbing record's seconds under 'plumbing'."""
    out = {}
    for rec in all_attempt_records(out_root):
        if rec.get("mode") not in (None, "official", "stub", "scratch"):
            continue
        s = rec.get("seconds") or 0.0
        st = stage_of(rec.get("phase_kind"), rec.get("setting"), rec.get("group"))
        out[st] = out.get(st, 0.0) + float(s) / 3600.0
    if plumbing_seconds:
        out["plumbing"] = out.get("plumbing", 0.0) + float(plumbing_seconds) / 3600.0
    return out


# --------------------------------------------------------------------------------------------- moving aside

def move_files_aside(rel_paths, base, dest, record_path, **fields):
    """Move (never delete) each existing base/<rel> to dest/<rel> (one rename each) and write record_path (+ sidecar)
    with their SHA-256. Returns the list of moved relative paths."""
    base, dest = Path(base), Path(dest)
    moved = {}
    for rel in rel_paths:
        src = base / rel
        if not src.is_file():
            continue
        target = dest / rel
        if target.exists():
            raise RecordError(f"refused: {target} exists; nothing is overwritten")
        target.parent.mkdir(parents=True, exist_ok=True)
        moved[rel] = file_sha256(src)
        os.rename(src, target)
    write_json_once(record_path, dict(fields, kind=MOVED_KIND, base=str(base), dest=str(dest), files=moved,
                                      moved_utc=utc()))
    return sorted(moved)


def move_interrupted(phase_id, out_root, why, rel_paths, stamp=None):
    """Every existing output file of the phase (rel_paths, relative to OUT/: phases.outputs(key)) goes to
    OUT/interrupted/<phase_id>/<utc>/ at the same relative paths; OUT/interrupted/<phase_id>/<utc>.json records their
    SHA-256 and why. Returns the moved paths (empty when there was nothing to move: no record is written then)."""
    out_root = Path(out_root)
    existing = [r for r in rel_paths if (out_root / r).is_file()]
    if not existing:
        return []
    stamp = stamp or utc_stamp()
    d = out_root / "interrupted" / phase_id
    k = 0
    while (d / (stamp if k == 0 else f"{stamp}_{k}")).exists() or \
            (d / f"{stamp if k == 0 else f'{stamp}_{k}'}.json").exists():
        k += 1
    name = stamp if k == 0 else f"{stamp}_{k}"
    return move_files_aside(existing, out_root, d / name, d / f"{name}.json", phase_id=phase_id, why=why)
