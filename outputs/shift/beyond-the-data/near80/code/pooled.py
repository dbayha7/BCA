"""Near-80 (version 9, main setting): the look. DRAFT, not frozen. One look per group: the fresh group (the official
run: the validity conditions, then C1 and C2, the pooled-group count and the refusals) and the smoke group (the
plumbing's seed 0, in scratch and stub runs, every sentence prefixed "not a claim"). Each fills in the sentences of
section 14 of PREREGISTRATION.md exactly as its section 14.5 says. Also the halt report.

Inputs (written by the runner's phases, each checked against its .sha256 sidecar; in official mode every record must
be official and carry the approved fingerprint of prereg.json): the data records (data_main: the gate's checks, the
outcome-free count of the pooled groups), BCA's switch records (the certificate on the pooled groups; refused: true
when BCA's height was not finite), the policy locks (the arm tables with the refused arms, the strengths), the values
of every world and seed, the stop and skip records, the stage lock (its plan names the seeds), the runners' status
files, the ledger's attempt records (the resource table) and the plumbing record (the halt report).

The numbers (section 14's bracket table; every one pooled, that is the mean over the group's seeds of each seed's
paired difference, stalling world, strong current, main setting, unless stated):
  S, N          seeds whose switch is on in the stalling world, off in the no-stall world (whole numbers)
  X, H, W       BCA's switched ramp minus no caution; the full-log actor minus no caution; no-stall: no caution minus BCA
  NET, D, R     step 2's net for BCA; NET minus the net of the best rule that cannot switch itself off; that rule's
                name (ties: the first in step 2's order). Step 2's own functions (step2lib.net, paired, go_no_go).
  E, L          BCA minus the matched even pull; BCA's loss to no caution in recorded conditions, big waves, big kicks
  D1, K1 ...    C2: BCA's net minus the split-conformal switch's net, pooled, and the seeds where it was positive;
                D2, K2 the mean-difference switch with the ramp; D3, K3 the mean-difference switch with the range limit
  P             no caution's share of strong-current steps past 80% (written as a percentage)
  V             the noise floor: |BCA's switched ramp minus the same ramp trained alone|, pooled
  NSEEDS, SEEDS the number of fresh seeds and their range
  NMIN, NMAX, NREACH, NSW, N8MIN, N8MAX   the pooled full-throttle group's smallest and largest size over the
                seed-worlds, how many reached 29 rows, how many seed-worlds, and version 8's groups' sizes for scale

Failed seeds (section 14.5). A seed that failed its data (the gate's checks) or had a phase stop fails every mark of
C1 and C2: its numbers are set to null before any mark is computed, and its own clause ("seed [s] failed [check]
...") tells it. A seed where BCA REFUSED (its certificate on with a height that is not finite, in either world) fails
every mark that needs BCA, C1's and C2's, with its own clause ("on seed [s] BCA's certificate did not certify ...");
the other arms' numbers stay and are reported. A seed with no failure, stop or refusal whose evaluation lacks a
number C1 or C2 reads fails the same way with [check] "its evaluation (numbers missing)"; one whose matched rivals
could not be matched fails C1's marks only. The validity conditions are computed over the seeds that did not fail
(a refused seed's no-caution and full-log numbers count: they do not depend on BCA).

Outputs (write-once): looks/<group>.md (the sentences first, then the tables) and looks/<group>.json (+ .sha256).

Command line:  pooled.py report --out-root DIR      the halt report (reads only)
               pooled.py fresh|smoke --out-root DIR --mode scratch|stub [--fixture DIR]
The official look is written by the runner's look phase only (phase_main.py calls run_look).
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import claims_text as CT  # noqa: E402

MODES = ("official", "scratch", "stub")
SCORED = K.SCORED
MEASURES = ("points", "held70_points", "capped80_points", "unseen_gain", "overshoot_loss", "past80_share",
            "zone_shares", "lower_tail_points")
CLAIMS = ("c1", "c2")
SMOKE_STATUS = "plumbing check, not a claim"
LOOK_KIND = {"fresh": "near80-v9-look-fresh", "smoke": "near80-v9-look-smoke"}
HALT_KIND = "near80-v9-halt-report"
GO_RULE = dict(g1_margin=K.MARKS["net_lead_margin"], g2_floor=K.MARKS["keeps_floor"],
               g3_margin=K.MARKS["even_pull_margin"])
SEEDS_WORDS = "seed [seeds]"
EVALUATION_MISSING_CHECK = "its evaluation (numbers missing)"
RIVAL_MISSING_CHECK = "its matching (a rival could not be matched or scored)"
TWIN_INDEX = {t: i + 1 for i, t in enumerate(K.TWIN_ARMS)}      # D1/K1 the split-conformal switch, ...
MATCHED_RIVALS = ("standin", "even_pull", "random_1", "random_2")
REPRO_ORDER = ("removal", "critic", "clone", "none", "w0_check", "obca_k1", "range_hinge")
REPRO_PLAIN = {"removal": "the main test's removal", "critic": "the main test's critic",
               "clone": "the main test's cloned actor", "none": "the main test's no-caution actor",
               "w0_check": "step 2's plumbing-check actor", "obca_k1": "step 2's switched-ramp actor",
               "critic_second": "step 2's switched-ramp actor", "switch": "step 2's switched-ramp actor",
               "range_hinge": "step 2's hand-set range-limit actor"}
SETTING_PLAIN = {"main": "main setting"}


def _go_needs(setting="main"):
    out = [("points", setting, w, a, SCORED) for w in K.WORLDS for a in (K.PRIMARY, "none") + K.WORLD_BLIND]
    out += [("points", setting, "stall", a, sit) for sit in K.KEEPS_SITUATIONS for a in (K.PRIMARY, "none")]
    return tuple(out)


RIVAL_NEEDS = tuple(("points", "main", w, a, SCORED) for w in K.WORLDS for a in MATCHED_RIVALS)
# What C1 and C2 and the validity conditions read, as (measure, setting, world, logical id, situation), without the
# matched rivals (a seed with no refusal whose rivals could not be matched fails C1's marks only) and without the
# BCA-dependent arms of a refused seed (its own clause tells it).
EVALUATION_NEEDS = tuple(x for x in _go_needs("main") if x not in RIVAL_NEEDS) + (
    ("points", "main", "stall", "full_none", SCORED), ("past80_share", "main", "stall", "none", SCORED)) + tuple(
    ("points", "main", w, t, SCORED) for w in K.WORLDS for t in K.TWIN_ARMS)
BCA_NEEDS = tuple(x for x in EVALUATION_NEEDS if x[3] in K.BCA_DEPENDENT)


class LookRefused(Exception):
    """A look or a report cannot be made from these files; the message starts with 'refused: '."""


# --------------------------------------------------------------------------------------------- JSON

def plain(x):
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
    return sha256_bytes(Path(path).read_bytes())


def sidecar_path(path):
    p = Path(path)
    return p.with_name(p.name + ".sha256")


def sidecar_digest(path):
    try:
        return sidecar_path(path).read_text(encoding="utf8").split()[0]
    except (OSError, IndexError):
        return None


def read_json_verified(path):
    """(payload, SHA-256) of a JSON whose sidecar matches; LookRefused on a missing file, a missing sidecar or a
    mismatch."""
    p = Path(path)
    if not p.is_file():
        raise LookRefused(f"refused: {p} is missing")
    data = p.read_bytes()
    want = sidecar_digest(p)
    if want is None:
        raise LookRefused(f"refused: {p} has no .sha256 sidecar")
    digest = sha256_bytes(data)
    if digest != want:
        raise LookRefused(f"refused: {p} differs from its .sha256 sidecar")
    try:
        return json.loads(data.decode("utf8")), digest
    except ValueError as error:
        raise LookRefused(f"refused: {p} is not JSON ({error})") from None


def _write_excl(path, data):
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def write_json_once(path, payload):
    """Canonical JSON and its sidecar, both created exclusively (write-once). Returns the SHA-256."""
    p = Path(path)
    if p.exists() or sidecar_path(p).exists():
        raise LookRefused(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(payload).encode("utf8")
    digest = sha256_bytes(data)
    _write_excl(p, data)
    _write_excl(sidecar_path(p), f"{digest}  {p.name}\n".encode("utf8"))
    return digest


def write_text_once(path, text):
    p = Path(path)
    if p.exists():
        raise LookRefused(f"refused: {p} exists (write-once)")
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_excl(p, text.encode("utf8"))


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def inside(path, root):
    p, r = os.path.realpath(str(path)), os.path.realpath(str(root))
    return p == r or p.startswith(r.rstrip("/\\") + os.sep)


def canonical_constants():
    return json.loads(canonical(K.as_dict()))


# --------------------------------------------------------------------------------------------- step 2's functions

_STEP2 = {}


def step2():
    """Step 2's step2lib.py (net, paired, go_no_go, mean_or_none, pts_of), from bytes checked against step 2's code
    manifest, under its own module name so that it never stands in for the runner's own import of it."""
    if "mod" in _STEP2:
        return _STEP2["mod"]
    path = K.S2_DIR / "step2lib.py"
    mod = sys.modules.get("step2lib")
    if mod is not None and os.path.realpath(str(getattr(mod, "__file__", "") or "")) == os.path.realpath(str(path)):
        _STEP2["mod"] = mod
        return mod
    data = path.read_bytes()
    try:
        want = json.loads((K.S2_DIR / "code_manifest.json").read_text(encoding="utf8"))["files"]["step2lib.py"]
    except (OSError, ValueError, KeyError) as error:
        raise LookRefused(f"refused: step 2's code manifest cannot be read ({error})") from None
    if sha256_bytes(data) != want:
        raise LookRefused("refused: step 2's step2lib.py differs from step 2's code manifest")
    mod = types.ModuleType("near80_pooled_step2lib")
    mod.__file__ = str(path)
    mod.__near80_sha256__ = want
    exec(compile(data, str(path), "exec"), mod.__dict__)
    mod.ROOT = Path(K.ROOT)
    sys.modules["near80_pooled_step2lib"] = mod
    _STEP2["mod"] = mod
    return mod


# --------------------------------------------------------------------------------------------- reading the results

class Reader:
    """Verified reads under an output root; records every input's SHA-256 (relative path -> digest)."""

    def __init__(self, root, mode, prereg_sha256):
        self.root, self.mode, self.prereg_sha256 = Path(root), mode, prereg_sha256
        self.inputs = {}

    def read(self, rel, kinds=None, strict=True, required=False, keep=True):
        p = self.root / rel
        if not p.exists() and not sidecar_path(p).exists():
            if required:
                raise LookRefused(f"refused: {rel} is missing")
            return None
        body, digest = read_json_verified(p)
        if kinds is not None and body.get("kind") not in kinds:
            raise LookRefused(f"refused: {rel} is not a {' or '.join(kinds)} record (kind {body.get('kind')!r})")
        self.check(rel, body, strict)
        if keep:
            self.inputs[str(rel)] = digest
        return body

    def keep(self, rel):
        self.inputs[str(rel)] = file_sha256(self.root / rel)

    def check(self, rel, body, strict):
        has_mode = "mode" in body
        if self.mode == "official":
            if (strict or has_mode) and body.get("mode") != "official":
                raise LookRefused(f"refused: {rel} is not an official record (mode {body.get('mode')!r})")
            if (strict or "prereg_sha256" in body) and body.get("prereg_sha256") != self.prereg_sha256:
                raise LookRefused(f"refused: {rel} was written under another pre-registration "
                                  f"({body.get('prereg_sha256')!r})")
        elif has_mode and body.get("mode") != self.mode:
            raise LookRefused(f"refused: {rel} is a {body.get('mode')!r} record, not {self.mode!r}")


class Results:
    """What a look reads. m[measure]['main'][world][seed][logical id][situation] holds the numbers with failed and
    refused seeds' BCA numbers set to null (for the marks); raw[...] the same without that rule (for the validity
    conditions and the reported tables)."""

    def __init__(self, group, seeds, mode, prereg_sha256, out_root):
        self.group, self.seeds, self.mode = group, [int(s) for s in seeds], mode
        self.prereg_sha256, self.out_root = prereg_sha256, Path(out_root)
        self.raw = {m: {st: {w: {s: {} for s in self.seeds} for w in K.WORLDS} for st in K.SETTINGS}
                    for m in MEASURES}
        self.m = None
        self.values, self.switches, self.locks, self.data_main = {}, {}, {}, {}
        self.failed = {}                    # seed -> reason (fails C1 and C2)
        self.failed_checks = {}             # seed -> [check]
        self.c1_only = {}                   # seed -> reason (a matched rival missing: C1's marks only)
        self.c1_only_checks = {}
        self.refused = {}                   # seed -> {world: n_full_throttle}: BCA refused (fails the BCA marks)
        self.counts = {}                    # (seed, world) -> the data record's pooled_groups block
        self.phase_stops, self.skipped, self.runner_skipped = [], {}, {}
        self.plumbing, self.attempts, self.notes = None, [], []
        self.plan, self.inputs = None, {}

    @property
    def pts(self):
        return self.m["points"]

    def fail(self, seed, reason, check):
        if seed not in self.failed:
            self.failed[seed] = reason
            self.failed_checks[seed] = check

    def fail_c1_only(self, seed, reason, check):
        if seed not in self.c1_only and seed not in self.failed:
            self.c1_only[seed] = reason
            self.c1_only_checks[seed] = check

    def apply_failures(self):
        """m: raw with a failed seed's numbers set to null, and a refused seed's BCA-dependent arms set to null."""
        self.m = copy.deepcopy(self.raw)
        for s in self.failed:
            if s not in self.seeds:
                continue
            for measure in MEASURES:
                for w in K.WORLDS:
                    self.m[measure]["main"][w][s] = {a: None for a in K.ARMS["main"]}
        for s in self.refused:
            if s not in self.seeds:
                continue
            for measure in MEASURES:
                for w in K.WORLDS:
                    for a in K.BCA_DEPENDENT:
                        self.m[measure]["main"][w][s][a] = None

    @property
    def ran(self):
        return dict(main=True)


def group_seeds(group):
    return list(K.SMOKE_SEEDS if group == "smoke" else K.FRESH)


def resolve_seeds(group, seeds, plan, mode):
    full = group_seeds(group)
    chosen = None
    if seeds is not None:
        chosen = [int(s) for s in seeds]
    elif plan and str(plan).startswith("seeds:"):
        chosen = [int(v) for v in str(plan)[6:].split(",") if v.strip()]
    elif plan == "smoke":
        chosen = list(K.SMOKE_SEEDS)
    if chosen is None:
        return full
    out = []
    for s in chosen:
        try:
            g = K.group_of(s)
        except ValueError as error:
            raise LookRefused(str(error) if str(error).startswith("refused") else f"refused: {error}") from None
        if g == group and s not in out:
            out.append(s)
    if mode == "official" and sorted(out) != sorted(full):
        raise LookRefused(f"refused: the official {group} look pools every seed of its group ({full})")
    if not out:
        raise LookRefused(f"refused: no seed of the {group} group in {chosen}")
    return sorted(out)


def official_prereg():
    """(payload, SHA-256) of the official prereg.json, verified against its sidecar."""
    return read_json_verified(K.N80 / K.PREREG_NAME)


def data_main_check(rec):
    """[check] for a seed whose main data failed the gate's checks."""
    if rec.get("ball_leak_checks_passed") is False:
        return "the ball's leak checks"
    marks = rec.get("gate_world_marks") or {}
    if any(v is False for v in marks.values()):
        return "the gate's five world checks on the ball alone"
    return "the gate's data checks"


def strip_the(text):
    return text[4:] if text.startswith("the ") else text


def phase_text(kind, setting):
    """[phase]: the phase's plain name in the runner's phase table, with its setting when it has one."""
    base = K.PHASE_PLAIN.get(kind, str(kind))
    return base + (f" ({SETTING_PLAIN[setting]})" if setting in SETTING_PLAIN else "")


def stopped_phase_check(stop):
    """[check] for a seed failed by a stopped phase: "its [phase] phase" (with its setting and world)."""
    where = [SETTING_PLAIN[stop["setting"]]] if stop.get("setting") in SETTING_PLAIN else []
    if stop.get("world") in K.WORLD_PLAIN:
        where.append(K.WORLD_PLAIN[stop["world"]])
    base = strip_the(K.PHASE_PLAIN.get(stop.get("phase_kind"), str(stop.get("phase_kind"))))
    return f"its {base} phase" + (f" ({', '.join(where)})" if where else "")


def _add_stop(R, st, rel):
    seed = st.get("seed")
    summary = dict(phase_id=st.get("phase_id"), phase_kind=st.get("phase_kind"), setting=st.get("setting"),
                   world=st.get("world"), seed=seed, group=st.get("group"), reason=st.get("reason"), file=str(rel))
    if any(x.get("phase_id") == summary["phase_id"] for x in R.phase_stops):
        return
    R.phase_stops.append(summary)
    why = f"{K.PHASE_PLAIN.get(summary['phase_kind'], summary['phase_kind'])} stopped ({summary['reason']})"
    if seed is None:
        if summary.get("phase_kind") == "lock" and summary.get("group") == R.group:
            for s in R.seeds:
                R.fail(s, why, stopped_phase_check(summary))
        return
    R.fail(int(seed), why, stopped_phase_check(summary))


def _phase_order(x):
    kind = x.get("phase_kind")
    return (-1 if x.get("seed") is None else int(x["seed"]),
            K.PHASE_KINDS.index(kind) if kind in K.PHASE_KINDS else 99, str(x.get("setting")),
            K.WORLDS.index(x["world"]) if x.get("world") in K.WORLDS else 9)


def load_results(out_root, group, mode="official", prereg_sha256=None, seeds=None):
    """Every input of a look (module docstring), checked. Official mode: out_root must be the official output folder,
    every record official and of the approved pre-registration (prereg_sha256; default the SHA-256 of prereg.json)."""
    if group not in K.GROUPS:
        raise LookRefused(f"refused: unknown group {group!r}")
    if mode not in MODES:
        raise LookRefused(f"refused: unknown mode {mode!r}")
    root = Path(out_root)
    if mode == "official":
        if os.path.realpath(str(root)) != os.path.realpath(str(K.OUT)):
            raise LookRefused(f"refused: the official look reads {K.OUT} only (got {root})")
        if not prereg_sha256:
            prereg_sha256 = official_prereg()[1]
    rd = Reader(root, mode, prereg_sha256)
    stage = rd.read(K.LAYOUT["stage_lock"].format(group=group), kinds=("near80-stage-lock",))
    plan = (stage or {}).get("plan")
    if mode == "official" and plan not in (None, "full"):
        raise LookRefused(f"refused: the official {group} lock names the plan {plan!r}, not the full plan")
    R = Results(group, resolve_seeds(group, seeds, plan, mode), mode, prereg_sha256, root)
    R.plan = plan
    stage_locks = (stage or {}).get("locks") or {}

    for s in R.seeds:
        for w in K.WORLDS:
            dm = rd.read(K.LAYOUT["data_main"].format(world=w, seed=s), kinds=("near80-data-main",))
            R.data_main[(s, w)] = dm
            if dm is None:
                continue
            if dm.get("seed_failed_reason"):
                R.fail(s, f"the main data failed in the {K.WORLD_PLAIN[w]}: {dm['seed_failed_reason']}",
                       data_main_check(dm))
            R.counts[(s, w)] = dm.get("pooled_groups")
    for w in K.WORLDS:
        for s in R.seeds:
            kw = dict(setting="main", world=w, seed=s)
            sw = rd.read(K.LAYOUT["switch"].format(**kw), kinds=("near80-switch",))
            if sw is not None:
                R.switches[("main", w, s)] = sw
                if sw.get("refused"):
                    R.refused.setdefault(s, {})[w] = (sw.get("groups") or {}).get("n_full_throttle")
            lock = rd.read(K.LAYOUT["policy_lock"].format(**kw), kinds=("near80-policy-lock",))
            if lock is not None:
                R.locks[("main", w, s)] = lock
            rel = K.LAYOUT["values"].format(**kw)
            v = rd.read(rel, kinds=("near80-values",))
            if v is None:
                continue
            if (v.get("setting"), v.get("world"), v.get("seed")) != ("main", w, s):
                raise LookRefused(f"refused: {rel} holds the values of another setting, world or seed")
            pid = K.phase_id("policy", "main", w, s)
            want = (stage_locks.get(pid) or {}).get("lock_sha256")
            if stage is not None and want != v.get("lock_sha256"):
                raise LookRefused(f"refused: {rel} is not the evaluation of the lock the {group} lock lists")
            R.values[("main", w, s)] = v
            ids = v.get("arm_ids") or {}
            sits = v.get("situations") or {}
            for logical in K.ARMS["main"]:
                name = ids.get(logical)
                for measure in MEASURES:
                    R.raw[measure]["main"][w][s][logical] = None if name is None else {
                        sit: ((sits.get(sit) or {}).get(measure) or {}).get(name) for sit in K.SITUATIONS}

    stops_dir = root / "stops"
    if stops_dir.is_dir():
        for p in sorted(stops_dir.glob("*.json")):
            rel = p.relative_to(root)
            st = rd.read(rel, kinds=("near80-phase-stop",), keep=False)
            if st.get("group") == group or (st.get("seed") is not None and int(st["seed"]) in R.seeds):
                rd.keep(rel)
                _add_stop(R, st, rel)
    for w in K.WORLDS:
        for s in R.seeds:
            rel = K.LAYOUT["policy_stop"].format(setting="main", world=w, seed=s)
            ps = rd.read(rel, kinds=("near80-policy-stop",))
            if ps is not None:
                _add_stop(R, dict(ps, phase_id=K.phase_id("policy", "main", w, s), phase_kind="policy", group=group),
                          rel)
    R.phase_stops.sort(key=_phase_order)

    skipped_dir = root / "skipped"
    if skipped_dir.is_dir():
        for p in sorted(skipped_dir.glob("*.json")):
            rel = p.relative_to(root)
            sk = rd.read(rel, kinds=("near80-phase-skipped",), strict=False, keep=False)
            if sk.get("group") == group:
                rd.keep(rel)
                R.skipped[sk.get("phase_id")] = sk.get("reason")
    for p in sorted(root.glob("status*.json")):
        try:
            status, _ = read_json_verified(p)
        except LookRefused as error:
            R.notes.append(f"the runner's status was not read ({error}); it is reported only")
            continue
        for pid, note in (status.get("skipped") or {}).items():
            last = str(pid).rsplit(".", 1)[-1]
            seed = int(last[1:]) if last.startswith("s") and last[1:].isdigit() else None
            if (seed is not None and seed in R.seeds) or last == group:
                R.runner_skipped[pid] = note
    R.plumbing = rd.read(K.LAYOUT["plumbing"], kinds=("near80-plumbing",), strict=False)

    ledger = root / "ledger"
    if ledger.is_dir():
        for d in sorted(x for x in ledger.iterdir() if x.is_dir()):
            for p in sorted(d.glob("*.json")):
                if not re.fullmatch(r"\d+\.json", p.name):
                    continue
                try:
                    a, _ = read_json_verified(p)
                except LookRefused as error:
                    R.notes.append(f"an attempt record was not read ({error})")
                    continue
                if a.get("kind") == "near80-attempt":
                    R.attempts.append(a)
    R.inputs = dict(rd.inputs)
    for s, gone in evaluation_gaps(R, "all").items():
        R.fail(s, "numbers missing from its evaluation, with no failure, stop or refusal record: " + ", ".join(gone),
               EVALUATION_MISSING_CHECK)
    for s, gone in evaluation_gaps(R, "rivals").items():
        R.fail_c1_only(s, "a matched rival could not be matched (its numbers are missing), with no failure or stop "
                          "record: " + ", ".join(gone), RIVAL_MISSING_CHECK)
    R.apply_failures()
    return R


def evaluation_needs(R, which):
    if which == "rivals":
        return RIVAL_NEEDS
    return EVALUATION_NEEDS


def evaluation_gaps(R, which="all"):
    """{seed: [what is missing]} for the group's seeds that have no failure record yet lack a number C1 or C2 reads
    (section 14.5): such a seed's learner failed. A refused seed's BCA-dependent numbers are not required (its own
    clause tells it); for 'rivals', a seed whose matched rivals could not be matched fails C1's marks only."""
    needs = evaluation_needs(R, which)
    out = {}
    for s in R.seeds:
        if s in R.failed or (which == "rivals" and s in R.c1_only):
            continue
        gone = []
        for measure, st, w, arm, sit in needs:
            if s in R.refused and arm in K.BCA_DEPENDENT:
                continue
            if s in R.refused and which == "rivals":
                continue
            row = R.raw[measure][st][w][s].get(arm)
            if row is None or row.get(sit) is None:
                gone.append(f"{arm} {measure} ({SETTING_PLAIN[st]}, {K.WORLD_PLAIN[w]}, {K.SITUATION_PLAIN[sit]})")
        if gone:
            out[s] = gone
    return out


# --------------------------------------------------------------------------------------------- the numbers

def _per(m, setting, world, arm, seeds, sit=SCORED):
    S = step2()
    return [S.pts_of(m[setting], world, s, arm, sit) for s in seeds]


def pooled_arm(m, setting, world, arm, seeds, sit=SCORED):
    per = _per(m, setting, world, arm, seeds, sit)
    return dict(per_seed=per, mean=step2().mean_or_none(per))


def _diff(a, b):
    return None if a is None or b is None else float(a) - float(b)


def arm_diff(m, a, b, seeds, world="stall", sit=SCORED):
    """Per seed: arm a's points minus arm b's; pooled mean; seeds positive."""
    per = [_diff(x, y) for x, y in zip(_per(m, "main", world, a, seeds, sit), _per(m, "main", world, b, seeds, sit))]
    return dict(per_seed=per, mean=step2().mean_or_none(per), positive=sum(1 for v in per if v is not None and v > 0))


def net_per_seed(m, arm, seeds):
    """Step 2's net across worlds per seed (step2lib.net), None where a number is missing."""
    return step2().net(m["main"], arm, seeds)["per_seed"]


def twin_difference(m, twin, seeds):
    """C2's number for one twin: BCA's net across worlds minus the twin's, per seed; pooled mean; seeds positive."""
    a, b = net_per_seed(m, K.PRIMARY, seeds), net_per_seed(m, twin, seeds)
    per = [_diff(x, y) for x, y in zip(a, b)]
    return dict(per_seed=per, mean=step2().mean_or_none(per), positive=sum(1 for v in per if v is not None and v > 0),
                net_bca=a, net_twin=b)


def _count_switch(R, world, want_on):
    n = 0
    for s in R.seeds:
        sw = R.switches.get(("main", world, s))
        on = ((sw or {}).get("certificate") or {}).get("on")
        if sw is not None and on is not None and bool(on) == want_on:
            n += 1
    return n


def claim1_points(R):
    """The points C1's go marks read: R.m's points with the seeds that failed C1's marks only (a matched rival could
    not be matched) set to null, so such a seed fails every mark of C1 and is told by its own clause."""
    pts = R.m["points"]
    gone = [s for s in R.c1_only if s in R.seeds]
    if not gone:
        return pts
    pts = copy.deepcopy(pts)
    for s in gone:
        for w in K.WORLDS:
            pts["main"][w][s] = {a: None for a in K.ARMS["main"]}
    return pts


def _claim1_numbers(m, seeds):
    S = step2()
    g = S.go_no_go(m["main"], seeds, GO_RULE)
    d1 = g["G1_detail"]
    net = d1["net_primary"]["net"]
    best_net = d1["best_world_blind_net"]
    lead = None if net is None or best_net is None else net - best_net
    keeps = {sit: g["G2_detail"][sit] for sit in K.KEEPS_SITUATIONS}
    loss = {sit: None if keeps[sit]["mean"] is None else -keeps[sit]["mean"] for sit in K.KEEPS_SITUATIONS}
    return g, net, best_net, lead, keeps, loss


def count_summary(R):
    """The outcome-free count over the group's seed-worlds (section 8.2): NMIN, NMAX, NREACH, NSW, N8MIN, N8MAX."""
    rows = []
    for s in R.seeds:
        for w in K.WORLDS:
            c = R.counts.get((s, w)) or {}
            g = c.get("groups") or {}
            v8 = c.get("version8_groups") or {}
            rows.append(dict(seed=s, world=w, defined=c.get("defined"), n_full_throttle=g.get("n_full_throttle"),
                             n_rest=g.get("n_rest"), n_survivors=g.get("n_survivors"),
                             reaches_min_rows=(c.get("certification") or {}).get("reaches_min_rows"),
                             version8_n_full_throttle=v8.get("n_full_throttle")))
    sizes = [r["n_full_throttle"] for r in rows if r["n_full_throttle"] is not None]
    v8s = [r["version8_n_full_throttle"] for r in rows if r["version8_n_full_throttle"] is not None]
    return dict(rows=rows, NMIN=min(sizes) if sizes else None, NMAX=max(sizes) if sizes else None,
                NREACH=sum(1 for r in rows if r["reaches_min_rows"]), NSW=len(rows),
                N8MIN=min(v8s) if v8s else None, N8MAX=max(v8s) if v8s else None, min_rows=K.CERT_MIN_ROWS)


def seeds_range(seeds):
    seeds = sorted(int(s) for s in seeds)
    if not seeds:
        return ""
    if len(seeds) == 1:
        return str(seeds[0])
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return CT.join_items([str(s) for s in seeds])


def brackets(results):
    """Every bracket of section 14's table (numbers; R and twin names as plain names), and per-seed lists."""
    R = results
    S = step2()
    m, seeds = R.m, R.seeds
    pts1 = claim1_points(R)
    x = S.paired(pts1["main"], "stall", K.PRIMARY, "none", seeds, SCORED)
    h = S.paired(m["points"]["main"], "stall", "full_none", "none", seeds, SCORED)
    w = S.paired(pts1["main"], "no_stall", "none", K.PRIMARY, seeds, SCORED)
    g, net, best_net, lead, keeps, loss = _claim1_numbers(pts1, seeds)
    best = g["G1_detail"]["best_world_blind"]
    ev = g["G3_detail"]
    losses = list(loss.values())
    twins = {t: twin_difference(m["points"], t, seeds) for t in K.TWIN_ARMS}
    p = pooled_arm(m["past80_share"], "main", "stall", "none", seeds)
    nf = noise_floor(R)
    cs = count_summary(R)
    br = dict(
        S=_count_switch(R, "stall", True), N=_count_switch(R, "no_stall", False),
        X=x["mean"], H=h["mean"], W=w["mean"], NET=net, D=lead, R=None if best is None else K.ARM_PLAIN[best],
        R_id=best, R_net=best_net, E=ev["mean"],
        L=None if any(v_ is None for v_ in losses) else max(losses), L_by_situation=loss,
        P=p["mean"], V=nf["V"], NSEEDS=len(seeds), SEEDS=seeds_range(seeds),
        NMIN=cs["NMIN"], NMAX=cs["NMAX"], NREACH=cs["NREACH"], NSW=cs["NSW"], N8MIN=cs["N8MIN"], N8MAX=cs["N8MAX"])
    for t, i in TWIN_INDEX.items():
        br[f"D{i}"] = twins[t]["mean"]
        br[f"K{i}"] = twins[t]["positive"]
    per_seed = dict(X=x["per_seed"], H=h["per_seed"], W=w["per_seed"], NET=g["G1_detail"]["net_primary"]["per_seed"],
                    E=ev["per_seed"], P=p["per_seed"], seeds=list(seeds),
                    L={sit: [None if v_ is None else -v_ for v_ in keeps[sit]["per_seed"]] for sit in keeps},
                    twins={t: dict(per_seed=v["per_seed"], net_bca=v["net_bca"], net_twin=v["net_twin"])
                           for t, v in twins.items()},
                    noise_floor=nf["per_seed"])
    return dict(br, per_seed=per_seed)


def seeds_not_failed(R):
    """The group's seeds that did not fail (a failed seed is told by its own clause)."""
    return [s for s in R.seeds if s not in R.failed]


def noise_floor(R):
    """Section 8.5's noise floor on the group's seeds that neither failed nor refused (raw numbers): the pooled
    |BCA's switched ramp minus the same ramp trained alone|, stalling world, strong current, at most 5. When no such
    seed is left, or a number is missing on one, it cannot be read (unmeasured: true, passed None)."""
    seeds = [s for s in R.seeds if s not in R.failed and s not in R.refused]
    d = arm_diff(R.raw["points"], K.PRIMARY, K.NOISE_ARM, seeds) if seeds else dict(per_seed=[], mean=None, positive=0)
    gone = [s for s, x in zip(seeds, d["per_seed"]) if x is None]
    unmeasured = bool(not seeds or gone or d["mean"] is None)
    v = None if unmeasured else abs(float(d["mean"]))
    passed = None if unmeasured else bool(v <= K.MARKS["noise_floor_max"])
    return dict(run=True, passed=passed, V=v, per_seed=d["per_seed"], seeds=seeds, missing_seeds=gone,
                unmeasured=unmeasured, max_allowed=K.MARKS["noise_floor_max"],
                arm=K.ARM_PLAIN[K.PRIMARY], copy=K.ARM_PLAIN[K.NOISE_ARM])


def conditions(R, seeds):
    """The harm and the past-80 share, pooled over `seeds` (raw numbers): each with value, per_seed, its minimum and
    passed (a missing number fails it)."""
    S = step2()
    raw = R.raw["points"]["main"]
    h = S.paired(raw, "stall", "full_none", "none", seeds, SCORED)
    p = pooled_arm(R.raw["past80_share"], "main", "stall", "none", seeds)
    harm = dict(value=h["mean"], per_seed=h["per_seed"], minimum=K.MARKS["harm_min"],
                passed=bool(h["mean"] is not None and h["mean"] >= K.MARKS["harm_min"]))
    past = dict(value=p["mean"], per_seed=p["per_seed"], minimum=K.MARKS["past80_min"],
                passed=bool(p["mean"] is not None and p["mean"] >= K.MARKS["past80_min"]))
    return dict(harm=harm, past80=past)


def validity(results):
    """Section 9.1, on the group's seeds that did not fail: a failed seed fails every mark and is told by its own
    clause, so it does not make the validity conditions fail. When every seed failed, the conditions cannot be read
    and do not stop the claims from being written as failed."""
    R = results
    ok = seeds_not_failed(R)
    c = conditions(R, ok)
    harm, past = c["harm"], c["past80"]
    if not ok:
        for x in (harm, past):
            x.update(passed=True, note="every seed failed; each seed's own clause tells the outcome")
    for x in (harm, past):
        x["seeds"] = list(ok)
    return dict(claims=dict(harm=harm, past80=past, passed=bool(harm["passed"] and past["passed"]), seeds=list(ok),
                            failed_seeds=sorted(R.failed)),
                noise_floor=noise_floor(R))


def marks_fixed():
    """C2's marks as near80_constants holds them, or None while DECISION.md has not fixed them."""
    c = K.MARKS_C2
    if any(c.get(k) is None for k in ("net_margin", "positive_seeds_min", "multiplicity")):
        return None
    return dict(net_margin=float(c["net_margin"]), positive_seeds_min=int(c["positive_seeds_min"]),
                multiplicity=str(c["multiplicity"]))


def marks(results):
    """C1 (step 2's three go marks on the main setting) and C2 (BCA's switched ramp against each plain twin, paired by
    seed on the net across worlds)."""
    R = results
    m = R.m
    g, net, best_net, lead, keeps, loss = _claim1_numbers(claim1_points(R), R.seeds)
    best = g["G1_detail"]["best_world_blind"]
    nets = {a: g["G1_detail"]["net_world_blind"][a]["net"] for a in K.WORLD_BLIND}
    c1 = dict(
        net_lead=dict(net=net, best_rival=best, best_rival_plain=None if best is None else K.ARM_PLAIN[best],
                      best_rival_net=best_net, lead=lead, margin=K.MARKS["net_lead_margin"], rival_nets=nets,
                      per_seed=g["G1_detail"]["net_primary"]["per_seed"], passed=bool(g["G1"])),
        keeps={sit: dict(difference=keeps[sit]["mean"], loss=loss[sit], per_seed=keeps[sit]["per_seed"],
                         floor=K.MARKS["keeps_floor"],
                         passed=bool(keeps[sit]["mean"] is not None and keeps[sit]["mean"] >= K.MARKS["keeps_floor"]))
               for sit in K.KEEPS_SITUATIONS},
        even_pull=dict(difference=g["G3_detail"]["mean"], per_seed=g["G3_detail"]["per_seed"],
                       margin=K.MARKS["even_pull_margin"], passed=bool(g["G3"])),
        passed=bool(g["G1"] and g["G2"] and g["G3"]))
    fixed = marks_fixed()
    twins = {}
    for t in K.TWIN_ARMS:
        d = twin_difference(m["points"], t, R.seeds)
        passed = None
        if fixed is not None:
            passed = bool(d["mean"] is not None and d["mean"] >= fixed["net_margin"]
                          and d["positive"] >= fixed["positive_seeds_min"])
        twins[t] = dict(difference=d["mean"], positive=d["positive"], per_seed=d["per_seed"], passed=passed,
                        plain=K.ARM_PLAIN[t], index=TWIN_INDEX[t])
    all_pass = None if fixed is None else bool(all(v["passed"] for v in twins.values()))
    c2 = dict(twins=twins, fixed=fixed, margins=dict(K.MARKS_C2), passed=all_pass,
              placeholders=sorted(K.placeholders_present()))
    return {"c1": c1, "c2": c2}


# --------------------------------------------------------------------------------------------- outcomes and sentences

def outcomes(look):
    """{c1: status, c2: status, c2_each: {twin: status}} and why."""
    val, mk = look["validity"], look["marks"]
    failed = bool(look.get("failed_checks"))
    out, why = {}, {}
    valid = val["claims"]["passed"]
    out["c1"] = "invalid" if not valid else ("pass" if mk["c1"]["passed"] else "fail")
    fixed = mk["c2"]["fixed"]
    nf = val["noise_floor"]
    each = {}
    if not valid:
        out["c2"], why["c2"] = "invalid", "the validity conditions"
    elif fixed is None:
        out["c2"], why["c2"] = "not_fixed", "DECISION.md has not fixed C2's margins"
    elif nf["passed"] is False:
        out["c2"], why["c2"] = "dropped", "the noise floor"
    elif failed:
        out["c2"], why["c2"] = "fail_failed_seed", "a seed failed"
    elif fixed["multiplicity"] == "each_separately":
        each = {t: ("pass" if v["passed"] else "fail") for t, v in mk["c2"]["twins"].items()}
        out["c2"] = "each"
    else:
        out["c2"] = "pass" if mk["c2"]["passed"] else "fail"
    out["c2_each"] = each
    return out, why


def _template_getter(templates, group):
    T = templates["sentences"]

    def t(sid):
        text = T[sid]["text"]
        if group == "smoke":
            text = text.replace("fresh seeds [SEEDS]", "the plumbing's seed [SEEDS]").replace("fresh seeds", "seeds")
        return text
    return t


def refused_clause_values(R_refused, seed):
    worlds = R_refused.get(seed) or {}
    ns = [v for v in worlds.values() if v is not None]
    return dict(s=CT.fmt(seed, "count"), n=CT.fmt(min(ns), "count") if ns else CT.NOT_AVAILABLE,
                world=CT.join_items([K.WORLD_PLAIN[w] for w in K.WORLDS if w in worlds]) or "both worlds")


def _c1_clauses(look, t):
    """[one clause per missed mark] in section 14.5's order; a missed mark whose number is not available because a
    seed failed or refused is told by that seed's own clause."""
    mk = look["marks"]["c1"]
    failed = {int(s): c for s, c in (look.get("failed_checks") or {}).items()}
    for s, c in (look.get("c1_only_checks") or {}).items():
        failed.setdefault(int(s), c)
    refused = {int(s): v for s, v in (look.get("refused") or {}).items()}
    covered = bool(failed or refused)
    out = []
    nl = mk["net_lead"]
    if not nl["passed"] and not (covered and (nl["net"] is None or nl["lead"] is None)):
        out.append(CT.fill(t("c1.clause.net"), dict(NET=nl["net"], D=nl["lead"],
                                                    R=nl["best_rival_plain"] or CT.NOT_AVAILABLE)))
    for sit in K.KEEPS_SITUATIONS:
        k = mk["keeps"][sit]
        if not k["passed"] and not (covered and k["loss"] is None):
            out.append(CT.fill(t("c1.clause.keeps"), dict(L=k["loss"], situation=K.SITUATION_PLAIN[sit])))
    ep = mk["even_pull"]
    if not ep["passed"] and not (covered and ep["difference"] is None):
        out.append(CT.fill(t("c1.clause.even_pull"), dict(E=ep["difference"])))
    for s in sorted(failed):
        out.append(CT.fill(t("c1.clause.failed_seed"), dict(s=CT.fmt(s, "count"), check=failed[s])))
    for s in sorted(refused):
        if s in failed:
            continue
        out.append(CT.fill(t("c1.clause.refused_seed"), refused_clause_values(refused, s)))
    if not out:
        raise ValueError("c1 failed but no mark was missed")
    return out


def _c2_clauses(look, t):
    mk = look["marks"]["c2"]
    refused = {int(s): v for s, v in (look.get("refused") or {}).items()}
    out = []
    for twin, v in mk["twins"].items():
        if v["passed"] is False and not (refused and v["difference"] is None):
            out.append(CT.fill(t("c2.clause.twin"), dict(twin=v["plain"], DT=v["difference"],
                                                         KT=CT.fmt(v["positive"], "count"),
                                                         NSEEDS=CT.fmt(len(look["seeds"]), "count"))))
    for s in sorted(refused):
        out.append(CT.fill(t("c2.clause.refused_seed"), refused_clause_values(refused, s)))
    if not out:
        raise ValueError("c2 failed but no twin was missed")
    return out


def phase_stop_sentence(template, stop, group=None):
    seed = stop.get("seed")
    if seed is None:
        seeds = group_seeds(stop.get("group") or group or "fresh")
        s = seeds_range(seeds)
    else:
        s = CT.fmt(int(seed), "count")
    world = K.WORLD_PLAIN.get(stop.get("world"), "both worlds")
    return CT.fill(template, dict(s=s, world=world, phase=phase_text(stop.get("phase_kind"), stop.get("setting")),
                                  reason=str(stop.get("reason") or "no reason recorded")))


def refused_sentence(template, seed, world, n):
    return CT.fill(template, dict(s=CT.fmt(int(seed), "count"), world=K.WORLD_PLAIN.get(world, str(world)),
                                  n=CT.fmt(n, "count") if n is not None else CT.NOT_AVAILABLE))


def reproduction_sentence(template, names):
    m = re.search(r"\[[^\[\]]* / [^\[\]]*\]", template)
    if m is None:
        raise ValueError("the reproduction stop has no list bracket")
    return CT.fill(template, {}, specials={m.group(0): CT.join_items(names)})


def seeds_words(seeds):
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return ("seed " if len(items) == 1 else "seeds ") + CT.join_items(items)


def fill_seeds(template, seeds, values=None):
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return CT.fill(template, dict(values or {}, seeds=CT.join_items(items)),
                   specials={SEEDS_WORDS: seeds_words(seeds)})


def failed_seeds_of(look):
    return sorted(int(s) for s in (look.get("failed_checks") or {}))


def sentences(look, templates, group):
    """The look's sentences, in order, as [{id, text}] (section 14 and its section 14.5)."""
    CT.check_templates(templates)
    t = _template_getter(templates, group)
    br = look["brackets"]
    val = look["validity"]
    out_c = look.get("outcomes") or outcomes(look)[0]
    prefix = templates["smoke_prefix"]
    n_seeds = CT.fmt(len(look["seeds"]), "count")
    rows = []

    def add(sid, text, claim=True):
        if group == "smoke" and claim:
            text = f"{prefix} {text}"
        rows.append(dict(id=sid, text=text))

    if group == "fresh":
        add("always_attached", CT.fill(templates["always_attached"], dict(NSEEDS=n_seeds, SEEDS=br["SEEDS"])),
            claim=False)
    c1 = out_c["c1"]
    common = dict(SEEDS=br["SEEDS"], NSEEDS=n_seeds)
    if c1 == "pass":
        add("c1.pass", CT.fill(t("c1.pass"), dict(common, S=CT.fmt(br["S"], "count"), N=CT.fmt(br["N"], "count"),
                                                  X=br["X"], H=br["H"], W=br["W"], NET=br["NET"], D=br["D"],
                                                  R=br["R"] or CT.NOT_AVAILABLE, E=br["E"])))
    elif c1 == "fail":
        add("c1.fail", CT.fill(t("c1.fail"), common,
                               specials={"[one clause per missed mark]": "; ".join(_c1_clauses(look, t))}))
    else:
        v = val["claims"]
        reasons = []
        if not v["harm"]["passed"]:
            reasons.append(CT.fill(t("c1.reason.harm"), dict(H=v["harm"]["value"])))
        if not v["past80"]["passed"]:
            reasons.append(CT.fill(t("c1.reason.past80"), dict(P=CT.fmt(v["past80"]["value"], "percent"))))
        add("c1.invalid", CT.fill(t("c1.invalid"), dict(common, reason="; ".join(reasons))))
    c2 = out_c["c2"]
    tw = look["marks"]["c2"]["twins"]
    tvals = dict(common)
    for twin, v in tw.items():
        tvals[f"D{v['index']}"] = v["difference"]
        tvals[f"K{v['index']}"] = CT.fmt(v["positive"], "count")
    if c2 == "not_fixed":
        add("c2.not_fixed", CT.fill(t("c2.not_fixed"), common))
    elif c2 == "dropped":
        add("c2.dropped", CT.fill(t("c2.dropped"), dict(common, V=br["V"])))
    elif c2 == "fail_failed_seed":
        failed = failed_seeds_of(look)
        every = set(int(s) for s in look["seeds"]) <= set(failed)
        sid = "c2.every_seed_failed" if every else "c2.failed_seed"
        add(sid, fill_seeds(t(sid), failed, common))
    elif c2 == "pass":
        add("c2.holds", CT.fill(t("c2.holds"), tvals))
    elif c2 == "fail":
        add("c2.fails", CT.fill(t("c2.fails"), common,
                                specials={"[one clause per twin not beaten]": "; ".join(_c2_clauses(look, t))}))
    elif c2 == "each":
        for twin, v in tw.items():
            sid = "c2.each.holds" if out_c["c2_each"].get(twin) == "pass" else "c2.each.fails"
            add(sid, CT.fill(t(sid), dict(common, twin=v["plain"], DT=v["difference"],
                                          KT=CT.fmt(v["positive"], "count"))))
    cs = look["counts"]
    add("count.reported", CT.fill(t("count.reported"), dict(common, NMIN=CT.fmt(cs["NMIN"], "count"),
                                                            NMAX=CT.fmt(cs["NMAX"], "count"),
                                                            NREACH=CT.fmt(cs["NREACH"], "count"),
                                                            NSW=CT.fmt(cs["NSW"], "count"),
                                                            N8MIN=CT.fmt(cs["N8MIN"], "count"),
                                                            N8MAX=CT.fmt(cs["N8MAX"], "count"))), claim=False)
    for s, worlds in sorted((int(a), b) for a, b in (look.get("refused") or {}).items()):
        for w in K.WORLDS:
            if w in worlds:
                add("stop.refused", refused_sentence(t("stop.refused"), s, w, worlds[w]), claim=False)
    for st in look.get("phase_stops") or []:
        add("stop.phase", phase_stop_sentence(t("stop.phase"), st, group), claim=False)
    return rows


def _claim_of(sid):
    if sid.startswith("c1"):
        return "c1"
    if sid.startswith("c2"):
        return "c2"
    return None


# --------------------------------------------------------------------------------------------- the reported tables

def _table(raw, measure, setting, world, arm, seeds):
    return {sit: pooled_arm(raw[measure], setting, world, arm, seeds, sit) for sit in K.SITUATIONS}


def _zone_means(raw, setting, world, arm, seeds):
    out = {}
    for sit in K.SITUATIONS:
        per = _per(raw["zone_shares"], setting, world, arm, seeds, sit)
        if not per or any(p is None for p in per):
            out[sit] = None
        else:
            out[sit] = [sum(float(p[i]) for p in per) / len(per) for i in range(3)]
    return out


def _num(x):
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return x


def reported(R):
    """The reported tables (no mark reads them): points and the other measures per world, arm and situation, nets,
    zone shares, switches and heights (with the refusals), arm tables, strengths and matches, the pooled-group
    counts, skipped and stopped phases, resources."""
    raw, seeds = R.raw, R.seeds
    S = step2()
    tables = {measure: {"main": {w: {a: _table(raw, measure, "main", w, a, seeds) for a in K.ARMS["main"]
                                     if not (w != "stall" and a in K.STALL_ONLY_ARMS)} for w in K.WORLDS}}
              for measure in MEASURES if measure != "zone_shares"}
    zones = {"main": {w: {a: _zone_means(raw, "main", w, a, seeds) for a in K.ARMS["main"]} for w in K.WORLDS}}
    nets = {"main": {a: S.net(raw["points"]["main"], a, seeds) for a in K.ARMS["main"] if a not in ("none", "full_none")}}
    twins = {t: twin_difference(raw["points"], t, seeds) for t in K.TWIN_ARMS}
    switches = {}
    for (st, w, s), sw in sorted(R.switches.items()):
        c, g = sw.get("certificate") or {}, sw.get("groups") or {}
        switches.setdefault(w, {})[str(s)] = dict(
            defined=sw.get("defined"), on=c.get("on"), height=_num(c.get("height")), q05=_num(c.get("q05")),
            finite_share_tail=c.get("finite_share_tail"), t_lo=_num(sw.get("t_lo")), t_up=_num(sw.get("t_up")),
            n_full_throttle=g.get("n_full_throttle"), n_rest=g.get("n_rest"), n_survivors=sw.get("n_survivors"),
            refused=sw.get("refused"), refusal=sw.get("refusal"))
    locks = {}
    for (st, w, s), lock in sorted(R.locks.items()):
        ids = lock.get("arm_ids") or {}
        strengths = lock.get("strengths") or {}
        locks.setdefault(w, {})[str(s)] = dict(
            arm_table=(lock.get("arm_table") or {}).get("arms"), t_lo=(lock.get("arm_table") or {}).get("t_lo"),
            t_up=(lock.get("arm_table") or {}).get("t_up"), arm_ids=ids, refused_arms=lock.get("refused_arms"),
            strengths={a: strengths.get(n) for a, n in ids.items() if n is not None and n in strengths},
            matched_names=lock.get("matched_names"),
            standin={k: (lock.get("standin") or {}).get(k) for k in ("arm", "coefficient", "error", "reason")},
            target_strength=lock.get("target_strength"), checks=lock.get("checks"),
            meandiff_limit_kind=lock.get("meandiff_limit_kind"))
    resources = {}
    for a in R.attempts:
        key = f"{a.get('group') or 'x'}/{a.get('phase_kind')}"
        r = resources.setdefault(key, dict(attempts=0, seconds=0.0, max_vmhwm_kb=None, classes={}))
        r["attempts"] += 1
        r["seconds"] += float(a.get("seconds") or 0.0)
        if a.get("vmhwm_kb") is not None:
            r["max_vmhwm_kb"] = max(r["max_vmhwm_kb"] or 0, int(a["vmhwm_kb"]))
        cls = (a.get("classification") or {}).get("kind")
        r["classes"][str(cls)] = r["classes"].get(str(cls), 0) + 1
    for r in resources.values():
        r["cpu_hours"] = r["seconds"] / 3600.0
    total_hours = sum(r["seconds"] for r in resources.values()) / 3600.0
    data = dict(main_failed={str(s): (rec or {}).get("seed_failed_reason") for (s, w), rec in R.data_main.items()
                             if (rec or {}).get("seed_failed_reason")},
                pooled_counts={f"{w}/seed{s}": c for (s, w), c in sorted(R.counts.items())})
    return dict(measures=tables, zone_shares=zones, nets=nets, twins=twins, switches=switches, locks=locks,
                noise_floor=noise_floor(R), skipped=dict(R.skipped), runner_skipped=dict(R.runner_skipped),
                phase_stops=list(R.phase_stops), resources=dict(by_phase_kind=resources, cpu_hours_total=total_hours),
                data=data, notes=list(R.notes))


# --------------------------------------------------------------------------------------------- the look

def _base_look(R, group):
    br = brackets(R)
    per_seed = br.pop("per_seed")
    return dict(
        kind=LOOK_KIND[group], mode=R.mode, prereg_sha256=R.prereg_sha256 if R.mode == "official" else None,
        phase_id=K.phase_id("look", group=group), group=group, seeds=list(R.seeds), label=K.LABEL, version=K.VERSION,
        plan=R.plan, inputs=dict(R.inputs),
        failed_seeds={str(s): r for s, r in sorted(R.failed.items())},
        failed_checks={str(s): c for s, c in sorted(R.failed_checks.items())},
        c1_only={str(s): r for s, r in sorted(R.c1_only.items())},
        c1_only_checks={str(s): c for s, c in sorted(R.c1_only_checks.items())},
        refused={str(s): dict(v) for s, v in sorted(R.refused.items())},
        brackets=br, per_seed=per_seed, validity=validity(R), marks=marks(R), counts=count_summary(R), ran=R.ran,
        phase_stops=list(R.phase_stops))


def _finish(look, templates, group):
    look["outcomes"], look["outcome_reasons"] = outcomes(look)
    rows = sentences(look, templates, group)
    look["sentences"] = rows
    claims = {}
    for c in CLAIMS:
        ids = [r["id"] for r in rows if _claim_of(r["id"]) == c]
        if group == "smoke":
            claims[c] = dict(status=SMOKE_STATUS, claim="", smoke_outcome=look["outcomes"][c], sentence_ids=ids)
        else:
            claims[c] = dict(status=look["outcomes"][c], sentence_ids=ids)
            if c == "c2" and look["outcomes"].get("c2_each"):
                claims[c]["twins"] = dict(look["outcomes"]["c2_each"])
    look["claims"] = claims
    return look


def make_look(results, templates):
    R = results
    look = _base_look(R, R.group)
    look = _finish(look, templates, R.group)
    look["reported"] = reported(R)
    look["written"] = utc()
    return look


def default_templates(mode, prereg=None):
    """prereg.json's claims block in official mode; otherwise section 14 of the current PREREGISTRATION.md."""
    if mode == "official":
        block = (prereg or {}).get("claims_word_for_word")
        if not block:
            raise LookRefused("refused: prereg.json has no claims_word_for_word block")
        return CT.check_templates(block)
    return CT.extract_section14((K.N80 / "PREREGISTRATION.md").read_text(encoding="utf8"))


def run_look(group, out_root, mode, prereg, prereg_sha256=None, seeds=None, templates=None):
    """The look phase (phase_main calls this): checks, reads, computes, writes looks/<group>.md then
    looks/<group>.json (+ .sha256), write-once. Returns the look."""
    if group not in K.GROUPS or mode not in MODES:
        raise LookRefused(f"refused: unknown group {group!r} or mode {mode!r}")
    root = Path(out_root)
    if (prereg or {}).get("constants") != canonical_constants():
        raise LookRefused("refused: prereg.json's constants differ from near80_constants (not the frozen code)")
    if mode == "official":
        if os.path.realpath(str(root)) != os.path.realpath(str(K.OUT)):
            raise LookRefused(f"refused: the official look is written to {K.OUT} only")
        body, digest = official_prereg()
        if json.loads(canonical(body)) != json.loads(canonical(prereg)):
            raise LookRefused("refused: the pre-registration given is not prereg.json")
        if prereg_sha256 is not None and prereg_sha256 != digest:
            raise LookRefused("refused: the approved fingerprint is not prereg.json's")
        prereg_sha256 = digest
    else:
        if inside(root, K.ROOT):
            raise LookRefused(f"refused: {mode} looks are written outside {K.ROOT} only")
        prereg_sha256 = None
    if templates is None:
        templates = default_templates(mode, prereg)
    CT.check_templates(templates)
    json_path = root / K.LAYOUT["look"].format(group=group)
    md_path = root / K.LAYOUT["look_md"].format(group=group)
    for p in (json_path, sidecar_path(json_path), md_path):
        if p.exists():
            raise LookRefused(f"refused: {p} exists (write-once)")
    R = load_results(root, group, mode, prereg_sha256=prereg_sha256, seeds=seeds)
    look = make_look(R, templates)
    digest = sha256_bytes(canonical(look).encode("utf8"))
    try:
        md = look_md(look, digest)
    except Exception as error:  # noqa: BLE001 - the report must never cost the look (the JSON is the record)
        md = (f"# Near-80 v9 {group} look\n\nThe full report could not be formatted ({error!r}); read {json_path.name} "
              f"(SHA-256 {digest}).\n")
    write_text_once(md_path, md)
    got = write_json_once(json_path, look)
    if got != digest:
        raise LookRefused("refused: the look's JSON did not hash as computed")
    return look


# --------------------------------------------------------------------------------------------- the halt report

def _infer_mode(root):
    if os.path.realpath(str(root)) == os.path.realpath(str(K.OUT)):
        return "official"
    for rel in [K.LAYOUT["status"]] + [K.LAYOUT["stage_lock"].format(group=g) for g in K.GROUPS]:
        try:
            body, _ = read_json_verified(Path(root) / rel)
            if body.get("mode") in ("scratch", "stub"):
                return body["mode"]
        except LookRefused:
            continue
    for p in sorted(Path(root).glob("data/main/*/seed*/data.json")):
        try:
            body, _ = read_json_verified(p)
            if body.get("mode") in ("scratch", "stub"):
                return body["mode"]
        except LookRefused:
            continue
    return "scratch"


def halt_report(out_root, templates=None, mode=None, write=False):
    """Whichever stop applies now (the runner's 'report'): the reproduction stop when the plumbing differed, then one
    sentence per refused seed-world and per stopped phase. Reads only, unless write=True (looks/halt_report.json,
    once)."""
    root = Path(out_root)
    mode = mode or _infer_mode(root)
    prereg, prereg_sha = None, None
    if mode == "official":
        prereg, prereg_sha = official_prereg()
    if templates is None:
        templates = default_templates(mode, prereg)
    t = _template_getter(templates, "fresh")
    rd = Reader(root, mode, prereg_sha)
    rows, stop = [], None
    rp = rd.read(K.LAYOUT["reproduction_differs"], kinds=("near80-reproduction-differs",), strict=False)
    pl = rp or rd.read(K.LAYOUT["plumbing"], kinds=("near80-plumbing",), strict=False)
    names = reproduction_failures((pl or {}).get("reproduction"))
    if names:
        stop = "stop.reproduction"
        rows.append(dict(id=stop, text=reproduction_sentence(t(stop), names)))
    refused = []
    for p in sorted(root.glob("learners/main/*/seed*/switch.json")):
        try:
            sw = rd.read(p.relative_to(root), kinds=("near80-switch",))
        except LookRefused:
            continue
        if sw and sw.get("refused"):
            refused.append((int(sw.get("seed")), sw.get("world"), (sw.get("groups") or {}).get("n_full_throttle")))
    for s, w, n in sorted(refused, key=lambda x: (x[0], K.WORLDS.index(x[1]) if x[1] in K.WORLDS else 9)):
        rows.append(dict(id="stop.refused", text=refused_sentence(t("stop.refused"), s, w, n)))
    stops = []
    sd = root / "stops"
    if sd.is_dir():
        for p in sorted(sd.glob("*.json")):
            st = rd.read(p.relative_to(root), kinds=("near80-phase-stop",))
            stops.append(dict(phase_id=st.get("phase_id"), phase_kind=st.get("phase_kind"), setting=st.get("setting"),
                              world=st.get("world"), seed=st.get("seed"), group=st.get("group"),
                              reason=st.get("reason")))
    stops.sort(key=_phase_order)
    for st in stops:
        rows.append(dict(id="stop.phase", text=phase_stop_sentence(t("stop.phase"), st)))
    rep = dict(kind=HALT_KIND, mode=mode, prereg_sha256=prereg_sha, stop=stop, sentences=rows, phase_stops=stops,
               refused=refused, inputs=dict(rd.inputs), note=None if rows else "No stop applies now.", written=utc())
    if write:
        write_json_once(root / K.LAYOUT["halt_report"], rep)
    return rep


def reproduction_failures(repro):
    """The plain names of the reproduction items that differed, in its order, each once."""
    if not repro:
        return []
    items = []
    flat = all(isinstance(v, dict) and "passed" in v for v in repro.values())
    groups = [repro] if flat else [v for v in repro.values() if isinstance(v, dict)]
    for g in groups:
        for name, v in g.items():
            if isinstance(v, dict) and v.get("passed") is False:
                items.append((REPRO_ORDER.index(name) if name in REPRO_ORDER else 99,
                              v.get("plain") or REPRO_PLAIN.get(name, str(name))))
    out = []
    for _, plain_name in sorted(items, key=lambda x: x[0]):
        if plain_name not in out:
            out.append(plain_name)
    return out


# --------------------------------------------------------------------------------------------- the report text

def _f(x, kind="points"):
    return "-" if x is None else CT.fmt(x, kind)


def _pct(x):
    return "-" if x is None else CT.fmt(x, "percent") + "%"


def look_md(look, digest):
    """looks/<group>.md: the sentences first, then the tables."""
    g = look["group"]
    br, val, mk = look["brackets"], look["validity"], look["marks"]
    head = "Fresh seeds" if g == "fresh" else "Plumbing, seed 0"
    L = [f"# Near-80 (version 9, main setting): {head} look", "", look["label"], "",
         f"Mode: {look['mode']}. Seeds {', '.join(str(s) for s in look['seeds'])}. Pre-registration "
         f"{look['prereg_sha256'] or 'not recorded (a scratch or stub run)'}. This look's JSON SHA-256 {digest}.", ""]
    if g == "smoke":
        L += ["The plumbing's seed 0 never carries a claim: every claim below is a check of the code, not a claim.", ""]
    L += ["## The sentences (section 14 of the pre-registration)", ""]
    L += [f"- {r['text']}" for r in look["sentences"]] + [""]
    nf = val["noise_floor"]
    L += ["## Validity (section 9.1) and the noise floor (section 8.5)", "",
          f"- The harm (full-log actor minus no caution, stalling world, strong current): "
          f"{_f(val['claims']['harm']['value'])} (at least 30): {'holds' if val['claims']['harm']['passed'] else 'fails'}",
          f"- No caution's share of strong-current steps past 80%: {_pct(val['claims']['past80']['value'])} "
          f"(at least 25%): {'holds' if val['claims']['past80']['passed'] else 'fails'}",
          "- The noise floor (BCA's switched ramp minus the same ramp trained alone, pooled): " + (
              "could not be read (" + (f"a number is missing on {seeds_words(nf['missing_seeds'])}" if
                                       nf["missing_seeds"] else "no seed without a failure or a refusal") + ")"
              if nf["unmeasured"] else f"{_f(nf['V'])} (at most 5): {'passed' if nf['passed'] else 'failed'}")]
    left_out = val["claims"].get("failed_seeds") or []
    if left_out:
        L.append("- Left out of the conditions above (failed; each fails every mark): seed "
                 + CT.join_items([str(s) for s in left_out]))
    L.append("")
    c1, c2 = mk["c1"], mk["c2"]
    L += ["## Marks", "", "| claim | mark | value | needed | result |", "|---|---|---|---|---|"]
    nl = c1["net_lead"]
    L.append(f"| C1 | net lead over {nl['best_rival_plain'] or 'not available'} | {_f(nl['lead'])} | 10 | "
             f"{'pass' if nl['passed'] else 'fail'} |")
    for sit in K.KEEPS_SITUATIONS:
        k = c1["keeps"][sit]
        L.append(f"| C1 | keeps the gain, {K.SITUATION_PLAIN[sit]} | {_f(k['difference'])} | -3 | "
                 f"{'pass' if k['passed'] else 'fail'} |")
    L.append(f"| C1 | beats the matched even pull | {_f(c1['even_pull']['difference'])} | 15 | "
             f"{'pass' if c1['even_pull']['passed'] else 'fail'} |")
    fixed = c2["fixed"]
    need = ("not fixed (DECISION.md)" if fixed is None else
            f"{fixed['net_margin']:g}, and {fixed['positive_seeds_min']} seeds")
    for twin, v in c2["twins"].items():
        res = "not fixed" if v["passed"] is None else ("pass" if v["passed"] else "fail")
        L.append(f"| C2 | net across worlds, BCA minus {v['plain']} (positive in {v['positive']} seeds) | "
                 f"{_f(v['difference'])} | {need} | {res} |")
    L += [""]
    L += ["## Claims", ""] + [f"- {c}: {v['status']}" + (f" ({v['smoke_outcome']})" if "smoke_outcome" in v else "")
                              + (f" (per twin: {v['twins']})" if v.get("twins") else "")
                              for c, v in look["claims"].items()] + [""]
    if c2["placeholders"]:
        L += ["Open items DECISION.md must fix (the freeze refuses while any remains): "
              + ", ".join(c2["placeholders"]), ""]
    if look["failed_seeds"] or look["c1_only"] or look["refused"]:
        L += ["## Failed and refused seeds", ""]
        L += [f"- failed, seed {s}: {r}" for s, r in look["failed_seeds"].items()]
        L += [f"- C1's marks only, seed {s}: {r}" for s, r in look["c1_only"].items()]
        L += [f"- BCA refused, seed {s}: {v}" for s, v in look["refused"].items()] + [""]
    cs = look["counts"]
    L += ["## The pooled groups (outcome-free, before any training)", "",
          "| seed | world | full-throttle rows | rest | survivors | reaches 29 | version 8's group |",
          "|---|---|---|---|---|---|---|"]
    for r in cs["rows"]:
        L.append(f"| {r['seed']} | {K.WORLD_PLAIN.get(r['world'], r['world'])} | {_f(r['n_full_throttle'], 'count')} | "
                 f"{_f(r['n_rest'], 'count')} | {_f(r['n_survivors'], 'count')} | "
                 f"{'yes' if r['reaches_min_rows'] else ('no' if r['reaches_min_rows'] is False else '-')} | "
                 f"{_f(r['version8_n_full_throttle'], 'count')} |")
    L.append("")
    L += ["## Brackets", "", "| bracket | value |", "|---|---|"]
    for name in ("S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "D1", "K1", "D2", "K2", "D3", "K3", "P", "V",
                 "NSEEDS", "SEEDS", "NMIN", "NMAX", "NREACH", "NSW", "N8MIN", "N8MAX"):
        v = br.get(name)
        if name == "P":
            L.append(f"| P | {_pct(v)} |")
            continue
        kind = "count" if name in ("S", "N", "K1", "K2", "K3", "NSEEDS", "NMIN", "NMAX", "NREACH", "NSW", "N8MIN",
                                   "N8MAX") else "points"
        L.append(f"| {name} | {v if isinstance(v, str) else _f(v, kind)} |")
    rep = look.get("reported") or {}
    pts = (rep.get("measures") or {}).get("points") or {}
    if "main" in pts:
        L += ["", "## Points in strong current, main setting (mean over seeds)", "",
              "| arm | stalling world | no-stall world |", "|---|---|---|"]
        for a in K.ARMS["main"]:
            s_ = ((pts["main"].get("stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
            n_ = ((pts["main"].get("no_stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
            L.append(f"| {K.ARM_PLAIN.get(a, a)} ({a}) | {_f(s_)} | {_f(n_)} |")
    sw = rep.get("switches") or {}
    if sw:
        L += ["", "## BCA's switch on the pooled groups", "",
              "| world | seed | on | height | full-throttle rows | rest | finite share | refused |", "|---|---|---|---|---|---|---|---|"]
        for w in K.WORLDS:
            for s, r in sorted((sw.get(w) or {}).items(), key=lambda x: int(x[0])):
                L.append(f"| {K.WORLD_PLAIN[w]} | {s} | {r.get('on')} | {_f(r.get('height'), 'value')} | "
                         f"{_f(r.get('n_full_throttle'), 'count')} | {_f(r.get('n_rest'), 'count')} | "
                         f"{_f(r.get('finite_share_tail'), 'value')} | {'yes' if r.get('refused') else 'no'} |")
    stops = look.get("phase_stops") or []
    if stops:
        L += ["", "## Stopped phases", ""] + [f"- {s['phase_id']}: {s['reason']}" for s in stops]
    res = (rep.get("resources") or {})
    if res.get("by_phase_kind"):
        L += ["", "## Resources (from the ledger)", "", "| group/phase kind | attempts | CPU-hours | peak memory (GiB) |",
              "|---|---|---|---|"]
        for k, r in sorted(res["by_phase_kind"].items()):
            peak = None if r["max_vmhwm_kb"] is None else r["max_vmhwm_kb"] / 1024 ** 2
            L.append(f"| {k} | {r['attempts']} | {r['cpu_hours']:.2f} | {'-' if peak is None else f'{peak:.2f}'} |")
        L.append(f"\nTotal CPU-hours so far: {res['cpu_hours_total']:.2f}.")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- command line

def main(argv=None):
    ap = argparse.ArgumentParser(description="near-80 v9 looks: pooled.py fresh|smoke|report")
    ap.add_argument("what", choices=("fresh", "smoke", "report"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--mode", choices=MODES, default=None)
    ap.add_argument("--fixture", default=None, help="stub mode: the test fixture whose prereg.json is used")
    ap.add_argument("--write", action="store_true", help="report only: also write looks/halt_report.json (once)")
    a = ap.parse_args(argv)
    try:
        if a.what == "report":
            rep = halt_report(Path(a.out_root), mode=a.mode, write=a.write)
            for r in rep["sentences"]:
                print(r["text"])
            if not rep["sentences"]:
                print(rep["note"])
            return 0
        mode = a.mode or "scratch"
        if mode == "official":
            raise LookRefused("refused: the official look is written by the runner's look phase only")
        if mode == "stub":
            if not a.fixture:
                raise LookRefused("refused: stub mode needs --fixture")
            prereg, _ = read_json_verified(Path(a.fixture) / K.PREREG_NAME)
        else:
            prereg, _ = official_prereg()
        look = run_look(a.what, Path(a.out_root), mode, prereg)
        for r in look["sentences"]:
            print(r["text"])
        return 0
    except LookRefused as error:
        print(str(error), file=sys.stderr)
        return K.EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
