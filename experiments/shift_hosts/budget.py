"""Compute accounting and the budget's stop rules for the stage driver (plan sections 6.1, stop rules S6 and S6b,
7.1-7.4; approval items 9-10). Pure Python (it imports in both environments).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods.

Definitions (plan section 6.1, S6b; words list):
  card-hour       one hour of wall-clock time with at least one card run active. The ledger keeps each card run's
                  interval on the driver's monotonic clock (time.monotonic, which does not jump when WSL's wall clock
                  does; the ReBRAC stage-0 report saw a 31-minute jump), per driver session (the boot id and the
                  driver's process: monotonic values of different sessions are not comparable). Card-hours are the
                  length of the union of a session's intervals, summed over sessions, so two runs at once never count
                  twice.
  processor-hour  one hour of measured processor time (user plus system), summed over all processes: every child the
                  driver waits for (os.wait4's rusage, which includes that child's own waited-for children) and the
                  driver itself (its own getrusage, added at every ledger save under kind "driver"). A card run's busy
                  waiting counts as it really is.
Caps (approval item 10, plan 7.3): medium-expert 14 card-hours and 21 processor-hours for stages 0-2a (stage 0's
compute is entered once, from the operator's numbers, when the driver first starts); medium 12 and 16 for stage 2b.

Crash safety. Every launched child gets an open ledger entry at launch (its card interval and its processor seconds),
which the driver's heartbeat moves forward while the child runs (the card interval's end; the child's processor time
read from /proc/<pid>/stat), and which the exact numbers replace when the child is reaped. So a child that was running
when the driver died is still counted up to the last heartbeat. A run that finished while no driver watched it (a
resumed driver finds its record and skips it) gets a "recovered" entry from its own record (wall and processor seconds,
less what its open entries already count). ledger.json is written with a kept previous version (ledger.json.prev and
its sidecar); when the current file does not verify (a crash between the file and its sidecar), the previous version is
read and the driver logs it. The ledger records its mode ("dry-run" or "real") and refuses to open in the other.

S6 (after stage 1): stage 1's measured speeds project stages 1-2 (plan 7.2's model, with stage 1's own numbers: start-up
and compile, steady milliseconds per outer step, refresh and table seconds, the processor share of a card run's wall
time, and the scoring lanes' measured cost); a projection above a cap stops; so does a learner alone whose stage-1
time per outer step is more than twice its random-array reference (TD3+BC 0.6 ms, CQL 2.1, IQL 0.33, ReBRAC 1.6). Both
sides of that comparison are one run at a time: stage 1 runs each learner alone by itself, so its time is compared with
twice the reference unscaled. During stage 2 an alone run more than twice its stage-1 time per outer step at the chosen
concurrency (the stage-1 time times the measured slowdown of a run sharing the card, when two run at once) stops too
("stop and re-cost").
S6b (before each new seed, and before the sham-alone and room-reference blocks, which are whole units too): the running
totals of the quality plus that unit's projected cost (its card runs and the scoring lanes they will need) would pass
the cap: stop between whole units, never inside one.
Two at once (plan 7.4; S6): stage 2 runs two card runs at once only if stage 1's measured aggregate throughput of two
runs at once is at least 1.15 times one run's, and stop rule S8 held (TWO_AT_ONCE_RULE).
"""

import json
import math
import os
import resource
import shutil
import time
from pathlib import Path

from experiments.shift_hosts import handoff as H

SCHEMA = "shift-hosts-ledger-v1"
MODES = ("dry-run", "real")
CAPS = {
    "medium-expert": dict(card_hours=14.0, processor_hours=21.0, covers="stages 0, 1 and 2a (approval item 10)"),
    "medium": dict(card_hours=12.0, processor_hours=16.0, covers="stage 2b (approval item 10, a separate yes)"),
}
REFERENCE_MS = dict(td3_bc=0.6, cql=2.1, iql=0.33, rebrac=1.6)     # S6 (plan 6.1; plan 7.1: one run at a time)
TWO_AT_ONCE_GAIN = 1.15
TWO_AT_ONCE_RULE = ("Stage 2 runs two card runs at once only if, in stage 1, the aggregate throughput of two runs "
                    "at once (the IW setup's weights-of-1 and repeat runs together, outer steps per wall-clock "
                    "second) is at least 1.15 times one run's (the IW run alone) for the four learners together, and "
                    "stop rule S8 held for every learner; otherwise one run at a time.")
# processor-s per lane until a lane of that kind has run (then its measured time is used). The budget re-cost's high
# values (PRESPEC.md section 13): score from test_scoring's measured 59.6 s for 5 actor readouts on the 7 full banks;
# coverage, S7 and viewer are allowances. The first defaults (120 / 180 / 300 / 90) added about 6 processor-hours of
# projection that no measurement supports, enough for S6 to stop a study that fits its caps.
DEFAULT_LANE_SECONDS = dict(score=45.0, coverage=90.0, s7=120.0, viewer=70.0)
SLOWDOWN_FACTOR = 2.0                                                               # S6's "more than twice"


def session_id():
    """This boot and driver process: monotonic clock values are comparable within one session only."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        boot = "unknown-boot"
    return f"{boot}/{os.getpid()}/{time.time_ns()}"


def union_seconds(intervals):
    """Length of the union of [start, end] intervals (one session's monotonic clock)."""
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted((float(a), float(b)) for a, b in intervals):
        if e < s:
            raise ValueError(f"an interval ends before it starts: {s} > {e}")
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def kept_path(path):
    """Where write_json_kept keeps the previous version of a file: <name>.prev (with its own .sha256 sidecar)."""
    path = Path(path)
    return path.with_name(path.name + ".prev")


def write_json_kept(path, obj):
    """handoff.write_json, after keeping the current file (when it verifies) as <name>.prev with its sidecar, so a
    crash between a file and its sidecar never loses everything (read_json_kept)."""
    path = Path(path)
    if path.exists():
        try:
            H.verify_sidecar(path)
            prev = kept_path(path)
            shutil.copyfile(path, prev)
            shutil.copyfile(H.sidecar(path), H.sidecar(prev))
        except (H.HandoffError, OSError):
            pass                                # keep the older .prev: the current file is the one that failed
    return H.write_json(path, obj)


def read_json_kept(path):
    """(body, note): the file after its sidecar check, or, when it does not verify, its kept previous version (note says
    so). Raises handoff.HandoffError when neither verifies."""
    path = Path(path)
    try:
        return H.read_json(path), None
    except H.HandoffError as error:
        prev = kept_path(path)
        if not prev.exists():
            raise
        body = H.read_json(prev)
        return body, f"{path.name} did not verify ({error}); its kept previous version {prev.name} was read"


def child_processor_seconds(pid):
    """A running child's processor time so far (user + system, its waited-for children included) from
    /proc/<pid>/stat, or None when it cannot be read."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        fields = text[text.rindex(")") + 2:].split()
        ticks = sum(int(fields[i]) for i in (11, 12, 13, 14))     # utime, stime, cutime, cstime (stat fields 14-17)
        return ticks / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def own_processor_seconds():
    """This process's own processor time (user + system)."""
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Ledger:
    """The running compute of the study, one JSON file with a sidecar (write_json_kept), rewritten after every change.
    Entries: card intervals {quality, label, session, start, end} and processor amounts {quality, label, seconds,
    kind}; an entry with open true belongs to a child still running (its numbers are the last heartbeat's); plus stage
    0's entry and the mode."""

    def __init__(self, path, mode=None):
        self.path = Path(path)
        self.note = None
        self.body = dict(schema=SCHEMA, mode=mode, card=[], processor=[], stage0=None, created_utc=None)
        if self.path.exists() or kept_path(self.path).exists():
            self.body, self.note = read_json_kept(self.path)
            if self.body.get("schema") != SCHEMA:
                raise ValueError(f"{self.path} is not a {SCHEMA} ledger")
            if mode is not None:
                have = self.body.get("mode")
                if have is None and mode == "real":
                    raise ValueError(f"{self.path} records no mode, so it is not a real study's ledger (a dry run's?)")
                if have is not None and have != mode:
                    raise ValueError(f"{self.path} is a {have} ledger; a {mode} driver never uses it")
                self.body["mode"] = mode

    def save(self):
        if self.body.get("created_utc") is None:
            self.body["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        write_json_kept(self.path, self.body)

    def set_stage0(self, card_hours, processor_hours, source):
        """Stage 0's compute (medium-expert's cap covers stages 0-2a), entered once."""
        if self.body.get("stage0") is not None:
            return self.body["stage0"]
        self.body["stage0"] = dict(card_hours=float(card_hours), processor_hours=float(processor_hours),
                                   source=str(source))
        self.save()
        return self.body["stage0"]

    def add_card(self, quality, label, session, start, end, **extra):
        e = dict(quality=quality, label=label, session=session, start=float(start), end=float(end), **extra)
        self.body["card"].append(e)
        return e

    def add_processor(self, quality, label, seconds, kind, **extra):
        e = dict(quality=quality, label=label, seconds=float(seconds), kind=kind, **extra)
        self.body["processor"].append(e)
        return e

    def add_to(self, quality, label, seconds, kind):
        """Add seconds to the one processor entry of (quality, label, kind), made on first use (the driver's own time,
        one entry per session and quality)."""
        for e in self.body["processor"]:
            if (e["quality"], e["label"], e["kind"]) == (quality, label, kind):
                e["seconds"] = float(e["seconds"]) + float(seconds)
                return e
        return self.add_processor(quality, label, seconds, kind)

    def orphan_open_entries(self, session):
        """Mark the open entries of other (ended) sessions: their children were running when that driver stopped; the
        last heartbeat's numbers stay counted. Returns how many."""
        n = 0
        for e in self.body["card"] + self.body["processor"]:
            if e.get("open") is True and e.get("session") != session:
                e["open"] = "orphaned: its driver session ended while the child ran; counted to the last heartbeat"
                n += 1
        return n

    def counted(self, label):
        """(card seconds, processor seconds of kind "card", whether a closed card interval exists) this ledger already
        counts for a card run label."""
        card = [e for e in self.body["card"] if e["label"] == label]
        closed = any(not e.get("open") for e in card)
        secs = sum(float(e["end"]) - float(e["start"]) for e in card)
        proc = sum(float(e["seconds"]) for e in self.body["processor"] if e["label"] == label and e["kind"] == "card")
        return secs, proc, closed

    def recover(self, quality, label, wall_seconds, processor_seconds):
        """A card run that finished while no driver watched it (module docstring): add what its record says it used,
        less what this ledger already counts for it, once. Returns the added (card seconds, processor seconds), or None
        when a closed card interval already counts it."""
        secs, proc, closed = self.counted(label)
        if closed:
            return None
        card = max(0.0, float(wall_seconds or 0.0) - secs)
        cpu = max(0.0, float(processor_seconds or 0.0) - proc)
        self.add_card(quality, label, f"recovered/{label}", 0.0, card, recovered=True)
        self.add_processor(quality, label, cpu, "card", recovered=True)
        return card, cpu

    def card_hours(self, quality):
        by_session = {}
        for e in self.body["card"]:
            if e["quality"] == quality:
                by_session.setdefault(e["session"], []).append((e["start"], e["end"]))
        hours = sum(union_seconds(v) for v in by_session.values()) / 3600.0
        if quality == "medium-expert" and self.body.get("stage0"):
            hours += self.body["stage0"]["card_hours"]
        return hours

    def processor_hours(self, quality):
        hours = sum(e["seconds"] for e in self.body["processor"] if e["quality"] == quality) / 3600.0
        if quality == "medium-expert" and self.body.get("stage0"):
            hours += self.body["stage0"]["processor_hours"]
        return hours

    def lane_seconds(self, kind):
        """The mean measured processor seconds of one scoring-lane job of a kind (None until one finished)."""
        v = [e["seconds"] for e in self.body["processor"] if e["kind"] == f"lane:{kind}"]
        return sum(v) / len(v) if v else None

    def totals(self, quality):
        return dict(card_hours=self.card_hours(quality), processor_hours=self.processor_hours(quality))


# ------------------------------------------------------------------------------------------------ the speed model

def run_speed(record, process_wall=None, process_cpu=None):
    """One stage-1 run's measured numbers (runner.py's record): steady milliseconds per outer step, the start-up and
    compile seconds (the driver's process wall time minus the run's chunk time at the steady speed, so Python start,
    imports and compiling are all in it), refresh and table seconds per refresh, readout seconds, and the processor
    share of the wall time (the driver's measured user + system seconds of the process when given, else the run's
    own)."""
    t = record["timings"]
    ms = t.get("steady_ms_per_outer_step")
    steps = int(t.get("outer_steps") or 0)
    if not ms or steps <= 0:
        raise ValueError(f"{record.get('run_id')}: no steady speed recorded")
    wall = float(process_wall if process_wall is not None else t["wall_seconds"])
    cpu = float(process_cpu if process_cpu is not None else t["processor_seconds"])
    stepping = steps * ms / 1000.0
    refresh = list(t.get("refresh_seconds") or [])
    tables = list(t.get("table_seconds") or [])
    overhead = wall - stepping - sum(refresh) - sum(tables) - sum(t.get("readout_seconds") or [])
    return dict(ms_per_step=float(ms), startup_seconds=max(0.0, overhead), steps=steps,
                refresh_seconds=(sum(refresh) / len(refresh)) if refresh else 0.0,
                table_seconds=(sum(tables) / len(tables)) if tables else 0.0,
                busy=min(1.5, cpu / wall) if wall > 0 else 1.0, wall=wall)


def project_run(speed, *, steps=100_000, refreshes=18, tables=0, readout_tables=0, concurrency=1, gain=1.0):
    """(card seconds, processor seconds) of one run from a measured speed (plan 7.2's model with measured numbers):
    start-up and compile + steps x ms + refreshes x refresh seconds + (IW) tables x table seconds (+ the last
    refresh's readout tables). At two at once the run's wall time stretches by 2 / gain while the card-hours it adds are
    its wall time divided by 2 (two runs share the card)."""
    wall = (speed["startup_seconds"] + steps * speed["ms_per_step"] / 1000.0 + refreshes * speed["refresh_seconds"]
            + (tables + readout_tables) * speed["table_seconds"])
    if concurrency == 2:
        stretched = wall * 2.0 / max(gain, 1e-9)
        return stretched / 2.0, stretched * speed["busy"]
    return wall, wall * speed["busy"]


def lane_cost(ledger, kind):
    v = ledger.lane_seconds(kind) if ledger is not None else None
    return float(v if v is not None else DEFAULT_LANE_SECONDS[kind])


def project_unit(units, speeds, *, concurrency=1, gain=1.0, ledger=None):
    """(card hours, processor hours) of a block of runs: units is a list of dicts {learner, setup, steps, refreshes,
    lanes: {kind: count}}; speeds[(learner, kind)] with kind "alone" (alone, room), "standard" (standard, sham) or
    "iw" (the IW setups). Scoring lanes add processor time only (they run beside the card queue)."""
    card = proc = 0.0
    for u in units:
        kind = "alone" if u["setup"] in ("alone", "room") else ("iw" if u["setup"].startswith("iw") else "standard")
        sp = speeds[(u["learner"], kind)]
        refreshes = u.get("refreshes", 0 if kind == "alone" else 18)
        tables = refreshes if kind == "iw" else 0
        c, p = project_run(sp, steps=u.get("steps", 100_000), refreshes=refreshes, tables=tables,
                           readout_tables=2 if kind == "iw" else 0, concurrency=concurrency, gain=gain)
        card += c
        proc += p
        for lk, n in (u.get("lanes") or {}).items():
            proc += n * lane_cost(ledger, lk)
    return card / 3600.0, proc / 3600.0


def s6b_check(ledger, quality, projected, committed_processor_hours=0.0):
    """S6b before a new unit: running totals + committed lane work + the unit's projection against the quality's cap.
    Returns (ok, detail)."""
    cap = CAPS[quality]
    tot = ledger.totals(quality)
    card = tot["card_hours"] + projected[0]
    proc = tot["processor_hours"] + committed_processor_hours + projected[1]
    ok = card <= cap["card_hours"] and proc <= cap["processor_hours"]
    return ok, dict(quality=quality, cap=cap, totals=tot, committed_processor_hours=committed_processor_hours,
                    projected=dict(card_hours=projected[0], processor_hours=projected[1]),
                    after=dict(card_hours=card, processor_hours=proc), passed=ok)


def two_at_once(single_seconds, pair_seconds, steps_each, s8_ok):
    """The two-at-once rule (TWO_AT_ONCE_RULE): single_seconds = {learner: one IW run's wall seconds alone};
    pair_seconds = {learner: the wall seconds from the start of the first to the end of the last of the two runs at
    once}; steps_each = outer steps of each run. Gain = (2 x steps / pair) / (steps / single), over the four learners
    together (sums of seconds). Returns (concurrency 1 or 2, detail)."""
    per = {}
    for l in single_seconds:
        per[l] = 2.0 * single_seconds[l] / pair_seconds[l] if pair_seconds.get(l) else None
    tot_single = sum(single_seconds.values())
    tot_pair = sum(pair_seconds.values())
    gain = 2.0 * tot_single / tot_pair if tot_pair > 0 else 0.0
    choose = 2 if (gain >= TWO_AT_ONCE_GAIN and bool(s8_ok)) else 1
    return choose, dict(rule=TWO_AT_ONCE_RULE, gain=gain, per_learner=per, threshold=TWO_AT_ONCE_GAIN,
                        s8_held=bool(s8_ok), steps_each=int(steps_each), concurrency=choose)


def s6_alone_check(ms_alone, reference=REFERENCE_MS):
    """S6's second half: each learner alone's stage-1 milliseconds per outer step at most twice its random-array
    reference. Both are one run at a time (stage 1 runs each learner alone by itself; the reference was measured so),
    so the limit is never scaled by the two-at-once slowdown (module docstring). Returns (ok, detail)."""
    detail, ok = {}, True
    for l, ms in ms_alone.items():
        limit = SLOWDOWN_FACTOR * reference[l]
        detail[l] = dict(ms=ms, reference_ms=reference[l], limit_ms=limit, passed=ms <= limit)
        ok = ok and ms <= limit
    return ok, detail


def describe(ledger):
    """A JSON-ready summary of the ledger for the driver's status file."""
    return {q: dict(ledger.totals(q), cap=CAPS[q]) for q in CAPS}


def to_json(x):
    return json.loads(json.dumps(H.plain(x)))
