"""Near-80 (version 9, DRAFT): the looks. The confirmatory look on version 9's fresh seeds (C1, then C2, then the
decision), the plumbing look on seed 0 (every number computed and labeled "plumbing, not a claim"), and the halt
report. Each fills in the sentences of section 14 of PREREGISTRATION.md exactly as its section 14.6 says.

Inputs (written by the runner's phases, each checked against its .sha256 sidecar; in official mode every record must
be official and carry the approved fingerprint of prereg.json): the data records (with the outcome-free count of BCA's
two pooled groups), BCA's switch records, the values of every seed and world, the policy locks (the arm tables, the
refusals, the reported strengths), the stop and skip records, the stage lock (its plan names the seeds), the runner's
status, the ledger's attempt records (the resource table) and the plumbing record (the halt report).

The claims block (near80_constants.CLAIMS, copied into prereg.json's constants):
  C1  version 8's claim 4, marks unchanged: step 2's three go marks in the main setting (step2lib.go_no_go): BCA's
      net across worlds at least the best world-blind rival's plus 10; BCA minus no caution at least -3 in recorded
      conditions, big waves and big kicks (stalling world); BCA minus the matched even pull at least 15.
  C2  BCA's switched ramp beats each plain twin (the split-conformal switch; the mean-difference switch with the ramp;
      the mean-difference switch with the hand-set range limit), paired by seed, on the claims block's measure ('net':
      step 2's per-seed net across worlds; 'stall_points': strong-current points in the stalling world), with the
      claims block's margins, statistic ('mean': the pooled paired difference is at least the margin; 'paired_t': a
      one-sided paired t-test that the mean difference exceeds the margin), optional minimum of seeds where BCA is
      ahead, multiplicity rule ('all': every comparison at its own level, C2 holds only if every one passes; 'holm',
      'bonferroni': family-wise at alpha) and optional noise floor (C2 is reported only when a paired difference's
      standard error exceeds it). While any of these is the TODO marker (a draft, a stub or a scratch run), C2's
      numbers are computed and its outcome is 'open': no claim sentence is written.
  The decision (section 9.4): CLAIMS['decision'] says which outcomes count as shown ('C2' or 'C1_and_C2'); 'open'
      while it is TODO.

Failed seeds (section 9.3; every seed is needed). A seed whose data failed the gate's checks, or that has a stopped
phase, fails every mark of C1 and C2. A seed where BCA refused (its switch on and its height not finite, in either
world: the refusal rule) fails every mark of C1 and C2 (they need BCA), with [check] "BCA's height (...)"; its arms
that need no BCA are trained and scored, and the reported tables show them. A seed with no failure record whose
evaluation lacks a number that C1 or C2 reads fails that claim with [check] "its evaluation (numbers missing)"; one
whose matched rivals alone are missing fails C1 only, with [check] "its matching (a rival could not be matched or
scored)". A failed seed's numbers are set to null for the claim it fails before any mark is computed, so the pooled
numbers of its marks are "not available" and its own clause tells why; C2 is then written with its "If a seed failed"
sentence. A failed seed is never replaced.

Outputs (write-once): looks/<group>.md (the sentences first, then the tables) and looks/<group>.json (+ .sha256).

Command line:  pooled.py report --out-root DIR      the halt report (reads only)
               pooled.py exploratory|confirmatory --out-root DIR --mode scratch|stub [--fixture DIR]
The official looks are written by the runner's look phases only (phase_main.py calls run_look).
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
SETTING = "main"
MEASURES = ("points", "held70_points", "capped80_points", "unseen_gain", "overshoot_loss", "past80_share",
            "zone_shares", "lower_tail_points")
CLAIM_IDS = ("C1", "C2")
EXPLORATORY_STATUS = "plumbing, not a claim"
LOOK_KIND = {"exploratory": "near80v9-look-exploratory", "confirmatory": "near80v9-look-confirmatory"}
HALT_KIND = "near80v9-halt-report"
SEEDS_WORDS = "seed [seeds]"                    # written "seed 2", "seeds 2 and 3", "seeds 1, 2 and 3" (section 14.6)
EVALUATION_MISSING_CHECK = "its evaluation (numbers missing)"
RIVAL_MISSING_CHECK = "its matching (a rival could not be matched or scored)"
MATCHED_RIVALS = ("standin", "even_pull", "random_1", "random_2")
REPRO_ORDER = ("removal", "critic", "clone", "none", "w0_check", "obca_k1", "range_hinge")
REPRO_PLAIN = {"removal": "the main test's removal", "critic": "the main test's critic",
               "clone": "the main test's cloned actor", "none": "the main test's no-caution actor",
               "w0_check": "step 2's plumbing-check actor", "obca_k1": "step 2's switched-ramp actor",
               "critic_second": "step 2's switched-ramp actor", "switch": "step 2's switched-ramp actor",
               "range_hinge": "step 2's hand-set range-limit actor"}
MULT_WORDS = {"all": "every comparison at its own level; C2 holds only if every one passes",
              "holm": "Holm's step-down rule over the comparisons", "bonferroni": "Bonferroni over the comparisons"}
DECISION_WORDS = {"C2": "C2 holds", "C1_and_C2": "C1 and C2 both hold"}


def go_rule():
    m = K.CLAIMS["C1"]["marks"]
    return dict(g1_margin=m["net_lead_margin"], g2_floor=m["keeps_floor"], g3_margin=m["even_pull_margin"])


def _go_needs():
    """The numbers step 2's three go marks read (C1): BCA's switched ramp, no caution and the five world-blind rivals
    in strong current in both worlds; BCA's switched ramp and no caution in the stalling world's other three
    situations. (measure, world, logical id, situation)."""
    out = [("points", w, a, SCORED) for w in K.WORLDS for a in (K.PRIMARY, "none") + K.WORLD_BLIND]
    out += [("points", "stall", a, sit) for sit in K.KEEPS_SITUATIONS for a in (K.PRIMARY, "none")]
    return tuple(out)


RIVAL_NEEDS = tuple(("points", w, a, SCORED) for w in K.WORLDS for a in MATCHED_RIVALS)
VALIDITY_NEEDS = (("points", "stall", "full_none", SCORED), ("points", "stall", "none", SCORED),
                  ("past80_share", "stall", "none", SCORED))
C1_NEEDS = tuple(x for x in _go_needs() if x not in RIVAL_NEEDS) + tuple(
    x for x in VALIDITY_NEEDS if x not in _go_needs())


def c2_needs():
    """The numbers C2 reads: BCA's switched ramp, no caution and each plain twin in strong current, in both worlds
    ('net') or in the stalling world ('stall_points')."""
    c2 = K.CLAIMS["C2"]
    worlds = K.WORLDS if c2.get("measure", "net") == "net" else ("stall",)
    arms = (c2["primary"], "none") + tuple(c2["comparators"])
    return tuple(("points", w, a, SCORED) for w in worlds for a in arms)


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
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
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
    return p == r or p.startswith(r.rstrip("/") + "/")


def canonical_constants():
    return json.loads(canonical(K.as_dict()))


def is_todo(x):
    return isinstance(x, str) and x.startswith(K.TODO)


# --------------------------------------------------------------------------------------------- step 2's functions

_STEP2 = {}


def step2():
    """Step 2's step2lib.py (net, paired, go_no_go, mean_or_none, pts_of), from bytes checked against step 2's code
    manifest, under its own module name so that it never stands in for the runner's own import of it. When the
    process already imported step 2's file as 'step2lib', that module is used."""
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
    mod = types.ModuleType("near80v9_pooled_step2lib")
    mod.__file__ = str(path)
    mod.__near80_sha256__ = want
    exec(compile(data, str(path), "exec"), mod.__dict__)
    sys.modules["near80v9_pooled_step2lib"] = mod
    _STEP2["mod"] = mod
    return mod


# --------------------------------------------------------------------------------------------- reading the results

class Reader:
    """Verified reads under an output root; records every input's SHA-256 (relative path -> digest)."""

    def __init__(self, root, mode, prereg_sha256):
        self.root, self.mode, self.prereg_sha256 = Path(root), mode, prereg_sha256
        self.inputs = {}

    def read(self, rel, kinds=None, strict=True, required=False, keep=True):
        """The record at root/rel, or None when neither it nor its sidecar exists. strict: the record must carry the
        mode (and in official mode the approved prereg_sha256); otherwise each is checked when present."""
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
    """What a look reads. raw[measure][world][seed][logical id][situation] holds the numbers of the main setting; m["C1"]
    and m["C2"] the same with the seeds that failed that claim set to null (for its marks)."""

    def __init__(self, group, seeds, mode, prereg_sha256, out_root):
        self.group, self.seeds, self.mode = group, [int(s) for s in seeds], mode
        self.prereg_sha256, self.out_root = prereg_sha256, Path(out_root)
        self.raw = {m: {w: {s: {} for s in self.seeds} for w in K.WORLDS} for m in MEASURES}
        self.m = None
        self.values, self.switches, self.locks, self.data_main, self.counts = {}, {}, {}, {}, {}
        self.failed = {c: {} for c in CLAIM_IDS}
        self.failed_checks = {c: {} for c in CLAIM_IDS}
        self.refused = {}
        self.phase_stops, self.skipped, self.runner_skipped = [], {}, {}
        self.plumbing, self.attempts, self.notes = None, [], []
        self.plan, self.inputs = None, {}

    def fail(self, seed, claims, reason, check):
        for c in claims:
            if seed not in self.failed[c]:
                self.failed[c][seed] = reason
                self.failed_checks[c][seed] = check

    def apply_failures(self):
        self.m = {}
        for c in CLAIM_IDS:
            m = copy.deepcopy(self.raw)
            for s in self.failed[c]:
                if s not in self.seeds:
                    continue
                for measure in MEASURES:
                    for w in K.WORLDS:
                        m[measure][w][s] = {a: None for a in K.ARMS[SETTING]}
            self.m[c] = m


def group_seeds(group):
    return list(K.EXPLORATORY if group == "exploratory" else K.CONFIRMATORY)


def resolve_seeds(group, seeds, plan, mode):
    full = group_seeds(group)
    chosen = None
    if seeds is not None:
        chosen = [int(s) for s in seeds]
    elif plan and str(plan).startswith("seeds:"):
        chosen = [int(v) for v in str(plan)[6:].split(",") if v.strip()]
    elif plan == "smoke":
        chosen = [0]                    # the plumbing smoke runs every phase kind on seed 0 only
    if chosen is None:
        return full if group == "confirmatory" else [0]
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
    """[check] for a seed whose data failed the gate's checks."""
    if rec.get("ball_leak_checks_passed") is False:
        return "the ball's leak checks"
    marks = rec.get("gate_world_marks") or {}
    if any(v is False for v in marks.values()):
        return "the gate's five world checks on the ball alone"
    return "the gate's data checks"


def strip_the(text):
    return text[4:] if text.startswith("the ") else text


def phase_text(kind):
    """[phase]: the phase's plain name in the runner's phase table."""
    return K.PHASE_PLAIN.get(kind, str(kind))


def stopped_phase_check(stop):
    """[check] for a seed failed by a stopped phase: "its [phase] phase" (with its world)."""
    where = [K.WORLD_PLAIN[stop["world"]]] if stop.get("world") in K.WORLD_PLAIN else []
    base = strip_the(K.PHASE_PLAIN.get(stop.get("phase_kind"), str(stop.get("phase_kind"))))
    return f"its {base} phase" + (f" ({', '.join(where)})" if where else "")


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def refused_cert(cert):
    """BCA refused: its switch is on but its height is not finite (the refusal rule)."""
    return bool((cert or {}).get("on") and not math.isfinite(num((cert or {}).get("height"))))


def refused_check(n_tail, world):
    return (f"BCA's height (BCA refused: its full-throttle group had {CT.fmt(n_tail, 'count')} rows in the "
            f"{K.WORLD_PLAIN.get(world, world)}, and BCA's certifiable bar is {K.CERTIFIABLE_MIN})")


def _add_stop(R, st, rel):
    seed = st.get("seed")
    summary = dict(phase_id=st.get("phase_id"), phase_kind=st.get("phase_kind"), setting=st.get("setting"),
                   world=st.get("world"), seed=seed, group=st.get("group"), reason=st.get("reason"), file=str(rel))
    if any(x.get("phase_id") == summary["phase_id"] for x in R.phase_stops):
        return
    R.phase_stops.append(summary)
    why = f"{K.PHASE_PLAIN.get(summary['phase_kind'], summary['phase_kind'])} stopped ({summary['reason']})"
    if seed is None:
        # the group's lock stopped: no evaluation of the group can run, so every seed of the group fails every mark
        if summary.get("phase_kind") == "lock" and summary.get("group") == R.group:
            for s in R.seeds:
                R.fail(s, CLAIM_IDS, why, stopped_phase_check(summary))
        return
    R.fail(int(seed), CLAIM_IDS, why, stopped_phase_check(summary))


def _phase_order(x):
    kind = x.get("phase_kind")
    return (-1 if x.get("seed") is None else int(x["seed"]),
            K.PHASE_KINDS.index(kind) if kind in K.PHASE_KINDS else 99,
            K.WORLDS.index(x["world"]) if x.get("world") in K.WORLDS else 9)


def load_results(out_root, group, mode="official", prereg_sha256=None, seeds=None):
    """Every input of a look (module docstring), checked. Official mode: out_root must be the official output folder,
    every record official and of the approved pre-registration (prereg_sha256; default the SHA-256 of prereg.json)."""
    if group not in K.GROUPS:
        raise LookRefused(f"refused: unknown group {group!r}")
    if mode not in MODES:
        raise LookRefused(f"refused: unknown mode {mode!r}")
    if mode == "official" and group != "confirmatory":
        raise LookRefused("refused: the official run has the confirmatory look only")
    root = Path(out_root)
    if mode == "official":
        if os.path.realpath(str(root)) != os.path.realpath(str(K.OUT)):
            raise LookRefused(f"refused: the official looks read {K.OUT} only (got {root})")
        if not prereg_sha256:
            prereg_sha256 = official_prereg()[1]
    rd = Reader(root, mode, prereg_sha256)
    stage = rd.read(K.LAYOUT["stage_lock"].format(group=group), kinds=("near80v9-stage-lock",))
    plan = (stage or {}).get("plan")
    if mode == "official" and plan not in (None, "full"):
        raise LookRefused(f"refused: the official {group} lock names the plan {plan!r}, not the full plan")
    R = Results(group, resolve_seeds(group, seeds, plan, mode), mode, prereg_sha256, root)
    R.plan = plan
    stage_locks = (stage or {}).get("locks") or {}

    for s in R.seeds:
        for w in K.WORLDS:
            dm = rd.read(K.LAYOUT["data_main"].format(world=w, seed=s), kinds=("near80v9-data-main",))
            R.data_main[(s, w)] = dm
            R.counts[(s, w)] = (dm or {}).get("group_count")
            if dm and dm.get("seed_failed_reason"):
                R.fail(s, CLAIM_IDS, f"the main data failed in the {K.WORLD_PLAIN[w]}: {dm['seed_failed_reason']}",
                       data_main_check(dm))
    for w in K.WORLDS:
        for s in R.seeds:
            kw = dict(setting=SETTING, world=w, seed=s)
            sw = rd.read(K.LAYOUT["switch"].format(**kw), kinds=("near80v9-switch",))
            if sw is not None:
                R.switches[(w, s)] = sw
                if refused_cert(sw.get("certificate")):
                    n_tail = (sw.get("groups") or {}).get("n_full_throttle")
                    R.refused.setdefault(s, {})[w] = dict(n_tail=n_tail, q05=(sw.get("certificate") or {}).get("q05"))
            lock = rd.read(K.LAYOUT["policy_lock"].format(**kw), kinds=("near80v9-policy-lock",))
            if lock is not None:
                R.locks[(w, s)] = lock
            rel = K.LAYOUT["values"].format(**kw)
            v = rd.read(rel, kinds=("near80v9-values",))
            if v is None:
                continue
            if (v.get("setting"), v.get("world"), v.get("seed")) != (SETTING, w, s):
                raise LookRefused(f"refused: {rel} holds the values of another setting, world or seed")
            pid = K.phase_id("policy", SETTING, w, s)
            want = (stage_locks.get(pid) or {}).get("lock_sha256")
            if stage is not None and want != v.get("lock_sha256"):
                raise LookRefused(f"refused: {rel} is not the evaluation of the lock the {group} lock lists")
            R.values[(w, s)] = v
            ids = v.get("arm_ids") or {}
            sits = v.get("situations") or {}
            for logical in K.ARMS[SETTING]:
                name = ids.get(logical)
                for measure in MEASURES:
                    R.raw[measure][w][s][logical] = None if name is None else {
                        sit: ((sits.get(sit) or {}).get(measure) or {}).get(name) for sit in K.SITUATIONS}
    for s in sorted(R.refused):
        w = next(x for x in K.WORLDS if x in R.refused[s])
        R.fail(s, CLAIM_IDS, f"BCA refused in the {K.WORLD_PLAIN[w]} (its full-throttle group had "
                             f"{R.refused[s][w].get('n_tail')} rows): the refusal rule counts it as BCA failing "
                             f"this seed", refused_check(R.refused[s][w].get("n_tail"), w))

    stops_dir = root / "stops"
    if stops_dir.is_dir():
        for p in sorted(stops_dir.glob("*.json")):
            rel = p.relative_to(root)
            st = rd.read(rel, kinds=("near80v9-phase-stop",), keep=False)
            if st.get("group") == group or (st.get("seed") is not None and int(st["seed"]) in R.seeds):
                rd.keep(rel)
                _add_stop(R, st, rel)
    for w in K.WORLDS:
        for s in R.seeds:
            rel = K.LAYOUT["policy_stop"].format(setting=SETTING, world=w, seed=s)
            ps = rd.read(rel, kinds=("near80v9-policy-stop",))
            if ps is not None:
                _add_stop(R, dict(ps, phase_id=K.phase_id("policy", SETTING, w, s), phase_kind="policy",
                                  group=group), rel)
    R.phase_stops.sort(key=_phase_order)

    skipped_dir = root / "skipped"
    if skipped_dir.is_dir():
        for p in sorted(skipped_dir.glob("*.json")):
            rel = p.relative_to(root)
            sk = rd.read(rel, kinds=("near80v9-phase-skipped",), strict=False, keep=False)
            if sk.get("group") == group:
                rd.keep(rel)
                R.skipped[sk.get("phase_id")] = sk.get("reason")
    try:
        status, _ = read_json_verified(root / K.LAYOUT["status"])
        for pid, note in (status.get("skipped") or {}).items():
            last = str(pid).rsplit(".", 1)[-1]
            seed = int(last[1:]) if last.startswith("s") and last[1:].isdigit() else None
            if (seed is not None and seed in R.seeds) or last == group:
                R.runner_skipped[pid] = note
    except LookRefused as error:
        R.notes.append(f"the runner's status was not read ({error}); it is reported only")
    R.plumbing = rd.read(K.LAYOUT["plumbing"], kinds=("near80v9-plumbing",), strict=False)

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
                if a.get("kind") == "near80v9-attempt":
                    R.attempts.append(a)
    R.inputs = dict(rd.inputs)
    for s, gone in evaluation_gaps(R, C1_NEEDS, ("C1",)).items():
        R.fail(s, ("C1",), "numbers missing from its evaluation, with no failure or stop record: " + ", ".join(gone),
               EVALUATION_MISSING_CHECK)
    for s, gone in evaluation_gaps(R, RIVAL_NEEDS, ("C1",)).items():
        R.fail(s, ("C1",), "a matched rival could not be matched (its numbers are missing), with no failure or stop "
                           "record: " + ", ".join(gone), RIVAL_MISSING_CHECK)
    for s, gone in evaluation_gaps(R, c2_needs(), ("C2",)).items():
        R.fail(s, ("C2",), "numbers missing from its evaluation, with no failure or stop record: " + ", ".join(gone),
               EVALUATION_MISSING_CHECK)
    R.apply_failures()
    return R


def evaluation_gaps(R, needs, books):
    """{seed: [what is missing]} for the seeds that have no failure record in `books` yet lack a number of `needs`."""
    out = {}
    for s in R.seeds:
        if any(s in R.failed[b] for b in books):
            continue
        gone = []
        for measure, w, arm, sit in needs:
            row = R.raw[measure][w][s].get(arm)
            if row is None or row.get(sit) is None:
                gone.append(f"{arm} {measure} ({K.WORLD_PLAIN[w]}, {K.SITUATION_PLAIN[sit]})")
        if gone:
            out[s] = gone
    return out


# --------------------------------------------------------------------------------------------- validity

def validity(R):
    """Section 9.0: the harm (the full-log actor minus no caution) and no caution's share of steps past 80%, pooled
    over the seeds that did not fail (C1's numbers), stalling world, strong current. A number that cannot be computed
    (every seed failed) never makes validity fail: the failed seeds' own sentences tell why."""
    v = K.CLAIMS["validity"]
    pts, past = R.m["C1"]["points"]["stall"], R.m["C1"]["past80_share"]["stall"]
    harms, shares = [], []
    for s in R.seeds:
        f, n = (pts[s].get("full_none") or {}).get(SCORED), (pts[s].get("none") or {}).get(SCORED)
        if f is not None and n is not None:
            harms.append(float(f) - float(n))
        sh = (past[s].get("none") or {}).get(SCORED)
        if sh is not None:
            shares.append(float(sh))
    H = sum(harms) / len(harms) if harms else None
    P = sum(shares) / len(shares) if shares else None
    harm_ok = H is None or H >= float(v["harm_min"])
    share_ok = P is None or P >= float(v["past80_min"])
    return dict(H=H, P=P, harm_min=v["harm_min"], past80_min=v["past80_min"], harm_passed=bool(harm_ok),
                share_passed=bool(share_ok), passed=bool(harm_ok and share_ok), seeds_read=len(harms))


# --------------------------------------------------------------------------------------------- C1

def c1_marks(R):
    """C1: step 2's three go marks on the main setting (step2lib.go_no_go), with C1's failed seeds left out."""
    S = step2()
    pts = R.m["C1"]["points"]
    g = S.go_no_go(pts, R.seeds, go_rule())
    d1 = g["G1_detail"]
    net, best_net = d1["net_primary"]["net"], d1["best_world_blind_net"]
    lead = None if net is None or best_net is None else net - best_net
    best = d1["best_world_blind"]
    marks = K.CLAIMS["C1"]["marks"]
    keeps = {sit: g["G2_detail"][sit] for sit in K.KEEPS_SITUATIONS}
    return dict(
        net_lead=dict(net=net, best_rival=best, best_rival_plain=None if best is None else K.ARM_PLAIN[best],
                      best_rival_net=best_net, lead=lead, margin=marks["net_lead_margin"],
                      rival_nets={a: d1["net_world_blind"][a]["net"] for a in K.WORLD_BLIND},
                      per_seed=d1["net_primary"]["per_seed"], passed=bool(g["G1"])),
        keeps={sit: dict(difference=keeps[sit]["mean"], loss=None if keeps[sit]["mean"] is None else -keeps[sit]["mean"],
                         per_seed=keeps[sit]["per_seed"], floor=marks["keeps_floor"],
                         passed=bool(keeps[sit]["mean"] is not None and keeps[sit]["mean"] >= marks["keeps_floor"]))
               for sit in K.KEEPS_SITUATIONS},
        even_pull=dict(difference=g["G3_detail"]["mean"], per_seed=g["G3_detail"]["per_seed"],
                       margin=marks["even_pull_margin"], passed=bool(g["G3"])),
        passed=bool(g["G1"] and g["G2"] and g["G3"]))


# --------------------------------------------------------------------------------------------- C2

def per_seed_measure(pts, arm, seeds, measure):
    """Per seed: the measure of `arm` (step 2's per-seed net across worlds, or stalling-world strong-current points)."""
    S = step2()
    if measure == "net":
        return S.net(pts, arm, seeds)["per_seed"]
    if measure == "stall_points":
        return [S.pts_of(pts, "stall", s, arm, SCORED) for s in seeds]
    raise ValueError(f"unknown C2 measure {measure!r} (one of {K.C2_MEASURES})")


def _mean(xs):
    return None if not xs or any(x is None for x in xs) else sum(xs) / len(xs)


def _sd(xs):
    if not xs or any(x is None for x in xs) or len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def t_pvalue(diffs, margin):
    """One-sided paired t-test of H0: mean(diffs) <= margin against mean(diffs) > margin; None when it cannot be
    computed (a missing value, fewer than 2 seeds, or no spread with the mean at the margin)."""
    if not diffs or any(x is None for x in diffs) or len(diffs) < 2:
        return None
    n, m, sd = len(diffs), _mean(diffs), _sd(diffs)
    if sd == 0:
        return 0.0 if m > margin else 1.0
    from scipy import stats
    t = (m - float(margin)) / (sd / math.sqrt(n))
    return float(stats.t.sf(t, df=n - 1))


def t_interval(diffs, level):
    """The two-sided t-interval of the mean paired difference at error `level` (e.g. 0.05 for 95%): (low, high), or
    None when it cannot be computed (a missing value or fewer than 2 seeds)."""
    if not diffs or any(x is None for x in diffs) or len(diffs) < 2:
        return None
    from scipy import stats
    n, m, sd = len(diffs), _mean(diffs), _sd(diffs)
    half = float(stats.t.ppf(1.0 - float(level) / 2.0, df=n - 1)) * sd / math.sqrt(n)
    return (m - half, m + half)


def c2_config():
    """C2's block, and whether it is open (any TODO marker left)."""
    c2 = K.CLAIMS["C2"]
    open_ = bool(K.todo_paths(c2, "CLAIMS/C2"))
    return c2, open_


def c2_marks(R):
    """C2 (module docstring): one comparison per plain twin, then the multiplicity rule, the noise floor and the
    outcome. Every number is computed whatever the block holds; while the block is open (TODO) no comparison is
    decided."""
    c2, open_ = c2_config()
    pts = R.m["C2"]["points"]
    measure = c2.get("measure", "net")
    primary = per_seed_measure(pts, c2["primary"], R.seeds, measure)
    comps = {}
    for twin in c2["comparators"]:
        other = per_seed_measure(pts, twin, R.seeds, measure)
        d = [None if a is None or b is None else float(a) - float(b) for a, b in zip(primary, other)]
        margin = c2["margins"].get(twin)
        mean = _mean(d)
        sd = _sd(d)
        comps[twin] = dict(twin=twin, plain=K.ARM_PLAIN[twin], per_seed=d, mean=mean, sd=sd,
                           se=None if sd is None else sd / math.sqrt(len(d)),
                           positive=sum(1 for v in d if v is not None and v > 0), n=len(d),
                           margin=None if is_todo(margin) else margin,
                           p=None if is_todo(margin) or margin is None else t_pvalue(d, margin))
    out = dict(measure=measure, comparisons=comps, open=open_, statistic=c2.get("statistic"),
               multiplicity=c2.get("multiplicity"), alpha=c2.get("alpha"),
               min_positive_seeds=c2.get("min_positive_seeds"), noise_floor_max=c2.get("noise_floor_max"))
    if open_:
        out.update(passed=None, dropped=None, note="the claims block holds TODO markers: C2 is not decided")
        return out
    stat, mult, alpha = c2["statistic"], c2["multiplicity"], c2.get("alpha")
    kmin, nf = c2.get("min_positive_seeds"), c2.get("noise_floor_max")
    if stat not in K.C2_STATISTICS or mult not in K.C2_MULTIPLICITY:
        raise LookRefused(f"refused: C2's statistic {stat!r} or multiplicity {mult!r} is not one of the pre-set ones")
    m = len(comps)
    levels = {}
    if stat == "interval":
        if mult == "holm":
            raise LookRefused("refused: C2's interval statistic takes multiplicity 'all' or 'bonferroni', not holm")
        level = float(alpha) / (m if mult == "bonferroni" else 1)
        for t, c in comps.items():
            levels[t] = level
            c.update(interval_level=1.0 - level, interval=t_interval(c["per_seed"], level))
    if stat == "paired_t":
        ps = sorted((c["p"] if c["p"] is not None else 2.0, t) for t, c in comps.items())
        if mult in ("all",):
            levels = {t: float(alpha) for t in comps}
        elif mult == "bonferroni":
            levels = {t: float(alpha) / m for t in comps}
        else:                                                        # Holm's step-down levels, in order of p
            for i, (_, t) in enumerate(ps):
                levels[t] = float(alpha) / (m - i)
    stop_holm = False
    for i, (t, c) in enumerate(sorted(comps.items(), key=lambda kv: (kv[1]["p"] if kv[1]["p"] is not None else 2.0))
                               if stat == "paired_t" and mult == "holm" else comps.items()):
        if c["mean"] is None:
            ok = False
        elif stat == "mean":
            ok = bool(c["mean"] >= float(c["margin"]))
        elif stat == "interval":
            lo, hi = c["interval"] or (None, None)
            ok = bool(lo is not None and c["mean"] >= float(c["margin"]) and lo > 0)
            c["interval_result"] = ("shown" if ok else "excluded" if hi is not None and hi < float(c["margin"])
                                    else "unresolved")
        else:
            ok = bool(c["p"] is not None and c["p"] < levels[t] and not stop_holm)
            if mult == "holm" and not ok:
                stop_holm = True                                     # Holm stops at the first comparison not rejected
        if kmin is not None:
            ok = bool(ok and c["positive"] >= int(kmin))
        c.update(passed=ok, level=levels.get(t))
    se_max = max((c["se"] for c in comps.values() if c["se"] is not None), default=None)
    dropped = bool(nf is not None and se_max is not None and se_max > float(nf))
    worst = None
    if dropped:
        worst = max((c for c in comps.values() if c["se"] is not None), key=lambda c: c["se"])["twin"]
    out.update(passed=bool(all(c["passed"] for c in comps.values())), dropped=dropped, se_max=se_max,
               se_max_twin=worst)
    return out


def decision(outcomes):
    """The decision rule (section 9.4): 'open' while CLAIMS['decision'] is TODO; otherwise shown / not shown."""
    rule = K.CLAIMS.get("decision")
    if rule is None or is_todo(rule):
        return dict(rule=rule, status="open")
    if "invalid" in outcomes.values():
        return dict(rule=rule, status="no conclusion", words=DECISION_WORDS.get(rule))
    if rule not in DECISION_WORDS:
        raise LookRefused(f"refused: the decision rule {rule!r} is not one of {sorted(DECISION_WORDS)}")
    if rule == "C2":
        shown = outcomes["C2"] == "pass"
    else:
        shown = outcomes["C1"] == "pass" and outcomes["C2"] == "pass"
    return dict(rule=rule, status="shown" if shown else "not shown", words=DECISION_WORDS[rule])


# --------------------------------------------------------------------------------------------- brackets

def seed_range_words(seeds):
    seeds = sorted(int(s) for s in seeds)
    if not seeds:
        return CT.NOT_AVAILABLE
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    return CT.join_items([str(s) for s in seeds])


def _count_switch(R, world, want_on):
    n = 0
    for s in R.seeds:
        sw = R.switches.get((world, s))
        on = ((sw or {}).get("certificate") or {}).get("on")
        if sw is not None and on is not None and bool(on) == want_on:
            n += 1
    return n


def count_numbers(R):
    """The outcome-free count over the group's seed-worlds (data records): smallest and largest pooled full-throttle
    group, and how many seed-worlds were below BCA's certifiable bar (36)."""
    ns = [c.get("n_tail") for c in R.counts.values() if c and c.get("defined")]
    below = sum(1 for c in R.counts.values() if c and c.get("defined") and c.get("below_bar"))
    return dict(GMIN=min(ns) if ns else None, GMAX=max(ns) if ns else None, NSW=len(ns), GB=below)


def brackets(R, c1, c2):
    """Every bracket of section 14's table (numbers; names as plain words), and per-seed lists."""
    S = step2()
    seeds = R.seeds
    pts1 = R.m["C1"]["points"]
    x = S.paired(pts1, "stall", K.PRIMARY, "none", seeds, SCORED)
    h = S.paired(pts1, "stall", "full_none", "none", seeds, SCORED)
    w = S.paired(pts1, "no_stall", "none", K.PRIMARY, seeds, SCORED)
    raw = R.raw["points"]
    nets = {a: S.net(raw, a, seeds)["net"] for a in (K.PRIMARY, "conformal_k1", "meandiff_k1", "meandiff_limit",
                                                     "range_hinge", "obca_k2")}
    losses = [c1["keeps"][sit]["loss"] for sit in K.KEEPS_SITUATIONS]
    cn = count_numbers(R)
    mult = c2.get("multiplicity")
    br = dict(SEEDS=seed_range_words(seeds), NS=len(seeds),
              S=_count_switch(R, "stall", True), N=_count_switch(R, "no_stall", False),
              X=x["mean"], H=h["mean"], W=w["mean"], NET=c1["net_lead"]["net"], D=c1["net_lead"]["lead"],
              R=c1["net_lead"]["best_rival_plain"], E=c1["even_pull"]["difference"],
              L=None if any(v is None for v in losses) else max(losses),
              N1=nets["conformal_k1"], N2=nets["meandiff_k1"], N3=nets["meandiff_limit"], NZ=nets["range_hinge"],
              NK2=nets["obca_k2"], NETRAW=nets[K.PRIMARY], RF=len(R.refused),
              MULT=MULT_WORDS.get(mult, CT.NOT_AVAILABLE) if not is_todo(mult) else CT.NOT_AVAILABLE,
              V=c2.get("se_max"), NF=c2.get("noise_floor_max") if not is_todo(c2.get("noise_floor_max")) else None,
              **cn)
    per_seed = dict(X=x["per_seed"], H=h["per_seed"], W=w["per_seed"], NET=c1["net_lead"]["per_seed"],
                    E=c1["even_pull"]["per_seed"], seeds=list(seeds),
                    C2={t: c["per_seed"] for t, c in c2["comparisons"].items()})
    return dict(br, per_seed=per_seed)


# --------------------------------------------------------------------------------------------- outcomes and sentences

def outcomes(look):
    """{C1: pass | fail | invalid, C2: pass | fail | fail_failed_seed | dropped | open | invalid}. When validity
    fails, neither claim is read either way ('invalid')."""
    mk = look["marks"]
    if not (mk.get("validity") or {}).get("passed", True):
        return {"C1": "invalid", "C2": "invalid"}
    out = {"C1": "pass" if mk["C1"]["passed"] else "fail"}
    c2 = mk["C2"]
    if c2.get("open"):
        out["C2"] = "open"
    elif c2.get("dropped"):
        out["C2"] = "dropped"
    elif look["failed_checks"].get("C2"):
        out["C2"] = "fail_failed_seed"
    else:
        out["C2"] = "pass" if c2["passed"] else "fail"
    return out


def _template_getter(templates, group):
    T = templates["sentences"]

    def t(sid):
        text = T[sid]["text"]
        if group == "exploratory":
            text = text.replace("fresh seeds", "seeds")
        return text
    return t


def seeds_words(seeds):
    """"seed [seeds]" as section 14.6 writes it: "seed 2", "seeds 2 and 3", "seeds 1, 2 and 3"."""
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return ("seed " if len(items) == 1 else "seeds ") + CT.join_items(items)


def fill_seeds(template, seeds, values=None):
    """A template whose "seed [seeds]" lists `seeds` (singular or plural by count); other brackets from `values`."""
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return CT.fill(template, dict(values or {}, seeds=CT.join_items(items)),
                   specials={SEEDS_WORDS: seeds_words(seeds)})


def c1_clauses(look, t):
    """[clauses] of C1's "Fails" sentence, in section 14.6's order. A missed mark whose number is not available because
    a seed failed is told by that seed's own clause."""
    mk = look["marks"]["C1"]
    failed = {int(s): c for s, c in (look["failed_checks"].get("C1") or {}).items()}
    covered = bool(failed)
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
    if not out:
        raise ValueError("C1 failed but no mark was missed")
    return out


def c2_comparison_texts(look, t):
    """[comparisons] of C2's sentence: one clause per plain twin, in the claims block's order."""
    c2 = look["marks"]["C2"]
    sid = {"paired_t": "c2.comparison.t", "interval": "c2.comparison.interval"}.get(c2["statistic"],
                                                                                   "c2.comparison.mean")
    out = []
    for twin, c in c2["comparisons"].items():
        vals = dict(T=c["plain"], DIFF=c["mean"], MARGIN=c["margin"], KPOS=CT.fmt(c["positive"], "count"),
                    NS=CT.fmt(c["n"], "count"), RESULT="met" if c.get("passed") else "missed")
        if sid == "c2.comparison.t":
            vals.update(PV=CT.fmt(c["p"], "value"), LEVEL=CT.fmt(c.get("level"), "value"))
        if sid == "c2.comparison.interval":
            lo, hi = c.get("interval") or (None, None)
            vals.update(LO=lo, HI=hi, CONF=CT.fmt(c.get("interval_level"), "percent"),
                        RESULT=c.get("interval_result") or "unresolved")
        out.append(CT.fill(t(sid), vals))
    return out


COUNT_BRACKETS = ("NS", "S", "N", "RF", "NSW", "GB", "GMIN", "GMAX")


def bracket_values(br):
    """The bracket table's values as the sentences write them: counts as whole numbers, the rest as they are (fill
    writes a number with one decimal); the per-seed lists are left out."""
    out = {}
    for k, v in br.items():
        if k == "per_seed":
            continue
        out[k] = CT.fmt(v, "count") if k in COUNT_BRACKETS else (CT.NOT_AVAILABLE if v is None else v)
    return out


def sentences(look, templates, group):
    """The look's sentences, in order, as [{id, text}] (section 14 and its section 14.6)."""
    CT.check_templates(templates)
    t = _template_getter(templates, group)
    br = look["brackets"]
    bv = bracket_values(br)
    oc = look.get("outcomes") or outcomes(look)
    prefix = templates["exploratory_prefix"]
    rows = []

    def add(sid, text, claim=True):
        if group == "exploratory" and claim:
            text = f"{prefix} {text}"
        rows.append(dict(id=sid, text=text))

    if group == "confirmatory":
        add("always_attached", CT.fill(templates["always_attached"], dict(NS=CT.fmt(br["NS"], "count"),
                                                                          SEEDS=br["SEEDS"])), claim=False)
    add("count.always", CT.fill(t("count.always"), dict(GMIN=CT.fmt(br["GMIN"], "count"),
                                                         GMAX=CT.fmt(br["GMAX"], "count"),
                                                         NSW=CT.fmt(br["NSW"], "count"),
                                                         GB=CT.fmt(br["GB"], "count"))))
    refused = sorted(int(s) for s in look.get("refused") or {})
    if refused:
        add("refusal.some", fill_seeds(t("refusal.some"), refused, dict(RF=CT.fmt(len(refused), "count"),
                                                                        NS=CT.fmt(br["NS"], "count"))))
    else:
        add("refusal.none", CT.fill(t("refusal.none"), dict(NS=CT.fmt(br["NS"], "count"))))
    if oc["C1"] == "invalid":
        val = look["marks"]["validity"]
        reasons = []
        if not val["harm_passed"]:
            reasons.append(CT.fill(t("validity.reason.harm"), dict(H=val["H"])))
        if not val["share_passed"]:
            reasons.append(CT.fill(t("validity.reason.share"), dict(P=CT.fmt(val["P"], "percent"))))
        add("validity.fails", CT.fill(t("validity.fails"), bv, specials={"[reason]": "; ".join(reasons)}))
    elif oc["C1"] == "pass":
        add("c1.holds", CT.fill(t("c1.holds"), bv))
    else:
        add("c1.fails", CT.fill(t("c1.fails"), bv, specials={
            "[clauses]": "; ".join(c1_clauses(look, t))}))
    c2 = oc["C2"]
    if c2 in ("pass", "fail"):
        sid = "c2.holds" if c2 == "pass" else "c2.fails"
        add(sid, CT.fill(t(sid), bv, specials={
            "[comparisons]": "; ".join(c2_comparison_texts(look, t))}))
    elif c2 == "fail_failed_seed":
        failed = sorted(int(s) for s in look["failed_checks"].get("C2") or {})
        every = set(int(s) for s in look.get("seeds") or []) <= set(failed)
        sid = "c2.every_seed_failed" if every else "c2.failed_seed"
        add(sid, fill_seeds(t(sid), failed))
    elif c2 == "dropped":
        mk = look["marks"]["C2"]
        add("c2.dropped", CT.fill(t("c2.dropped"), dict(T=K.ARM_PLAIN.get(mk.get("se_max_twin"), CT.NOT_AVAILABLE),
                                                        V=br["V"], NF=br["NF"])))
    add("c2.scale", CT.fill(t("c2.scale"), bv))
    dec = look.get("decision") or {}
    if dec.get("status") in ("shown", "not shown") and group == "confirmatory":
        sid = "decision.shown" if dec["status"] == "shown" else "decision.not_shown"
        add(sid, CT.fill(t(sid), dict(RULE=dec["words"])))
    for st in look.get("phase_stops") or []:
        add("stop.phase", phase_stop_sentence(t("stop.phase"), st, group), claim=False)
    return rows


def phase_stop_sentence(template, stop, group=None):
    seed = stop.get("seed")
    if seed is None:
        s = seed_range_words(group_seeds(stop.get("group") or group or "confirmatory"))
    else:
        s = CT.fmt(int(seed), "count")
    world = K.WORLD_PLAIN.get(stop.get("world"), "both worlds")
    return CT.fill(template, dict(s=s, world=world, phase=phase_text(stop.get("phase_kind")),
                                  reason=str(stop.get("reason") or "no reason recorded")))


def no_training_sentence(template, rule):
    v = rule.get("value")
    value = ("passed" if v else "failed") if isinstance(v, bool) else (CT.NOT_AVAILABLE if v is None else str(v))
    return CT.fill(template, dict(check=str(rule.get("rule") or "a no-training check").replace(":", ",", 1),
                                  value=value, threshold=str(rule.get("threshold") or "")))


def reproduction_sentence(template, names):
    m = re.search(r"\[[^\[\]]* / [^\[\]]*\]", template)
    if m is None:
        raise ValueError("the reproduction stop has no list bracket")
    return CT.fill(template, {}, specials={m.group(0): CT.join_items(names)})


# --------------------------------------------------------------------------------------------- the reported tables

def _per(raw_measure, world, arm, seeds, sit=SCORED):
    S = step2()
    return [S.pts_of(raw_measure, world, s, arm, sit) for s in seeds]


def pooled_arm(raw_measure, world, arm, seeds, sit=SCORED):
    per = _per(raw_measure, world, arm, seeds, sit)
    return dict(per_seed=per, mean=step2().mean_or_none(per))


def reported(R):
    """The reported tables (no mark reads them): every measure per world, arm and situation (every number that
    exists, on every seed, refused seeds included), nets, the switches and their groups, the outcome-free counts, the
    arm tables and refusals, strengths and matches, skipped and stopped phases, resources."""
    raw, seeds = R.raw, R.seeds
    S = step2()
    tables = {measure: {w: {a: {sit: pooled_arm(raw[measure], w, a, seeds, sit) for sit in K.SITUATIONS}
                            for a in K.ARMS[SETTING] if not (w != "stall" and a in K.STALL_ONLY_ARMS)}
                        for w in K.WORLDS}
              for measure in MEASURES if measure != "zone_shares"}
    nets = {a: S.net(raw["points"], a, seeds) for a in K.ARMS[SETTING] if a not in ("none", "full_none")}
    switches = {}
    for (w, s), sw in sorted(R.switches.items()):
        c, g = sw.get("certificate") or {}, sw.get("groups") or {}
        ro = sw.get("reported_only") or {}
        switches.setdefault(w, {})[str(s)] = dict(
            defined=sw.get("defined"), on=c.get("on"), height=num(c.get("height")), q05=num(c.get("q05")),
            refused=refused_cert(c), t_lo=num(sw.get("t_lo")), t_up=num(sw.get("t_up")),
            n_full_throttle=g.get("n_full_throttle"), n_rest=g.get("n_rest"), n_survivors=sw.get("n_survivors"),
            sources=sw.get("group_sources"), one_row_per_episode=(ro.get("one_row_per_episode") or {}).get(
                "certificate"), bank_only=(ro.get("bank_only") or {}).get("certificate"))
    counts = {f"{w}/seed{s}": c for (s, w), c in sorted(R.counts.items())}
    locks = {}
    for (w, s), lock in sorted(R.locks.items()):
        ids = lock.get("arm_ids") or {}
        strengths = lock.get("strengths") or {}
        locks.setdefault(w, {})[str(s)] = dict(
            arm_table=(lock.get("arm_table") or {}).get("arms"), t_lo=(lock.get("arm_table") or {}).get("t_lo"),
            t_up=(lock.get("arm_table") or {}).get("t_up"), arm_ids=ids, refusal=lock.get("refusal"),
            plain_twins=lock.get("plain_twins"),
            strengths={a: strengths.get(n) for a, n in ids.items() if n is not None and n in strengths},
            matched_names=lock.get("matched_names"),
            standin={k: (lock.get("standin") or {}).get(k) for k in ("arm", "coefficient", "error", "reason")},
            target_strength=lock.get("target_strength"), checks=lock.get("checks"))
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
    return dict(measures=tables, nets=nets, switches=switches, outcome_free_counts=counts, locks=locks,
                skipped=dict(R.skipped), runner_skipped=dict(R.runner_skipped), phase_stops=list(R.phase_stops),
                resources=dict(by_phase_kind=resources, cpu_hours_total=total_hours),
                main_failed={f"{w}/seed{s}": (rec or {}).get("seed_failed_reason") for (s, w), rec in
                             R.data_main.items() if (rec or {}).get("seed_failed_reason")},
                notes=list(R.notes))


# --------------------------------------------------------------------------------------------- the looks

def make_look(R, templates, group):
    c1 = c1_marks(R)
    c2 = c2_marks(R)
    val = validity(R)
    br = brackets(R, c1, c2)
    br["P"] = val["P"]
    per_seed = br.pop("per_seed")
    look = dict(
        kind=LOOK_KIND[group], mode=R.mode, prereg_sha256=R.prereg_sha256 if R.mode == "official" else None,
        phase_id=K.phase_id(f"look_{group}", group=group), group=group, seeds=list(R.seeds), label=K.LABEL,
        plan=R.plan, inputs=dict(R.inputs), claims_block=json.loads(canonical(K.CLAIMS)),
        failed_seeds={c: {str(s): r for s, r in sorted(R.failed[c].items())} for c in CLAIM_IDS},
        failed_checks={c: {str(s): x for s, x in sorted(R.failed_checks[c].items())} for c in CLAIM_IDS},
        refused={str(s): v for s, v in sorted(R.refused.items())},
        brackets=br, per_seed=per_seed, marks=dict(C1=c1, C2=c2, validity=val), phase_stops=list(R.phase_stops))
    look["outcomes"] = outcomes(look)
    look["decision"] = decision(look["outcomes"])
    rows = sentences(look, templates, group)
    look["sentences"] = rows
    claims = {}
    for c in CLAIM_IDS:
        ids = [r["id"] for r in rows if r["id"].lower().startswith(c.lower() + ".")]
        if group == "exploratory":
            claims[c] = dict(status=EXPLORATORY_STATUS, claim="", plumbing_outcome=look["outcomes"][c],
                             sentence_ids=ids)
        else:
            claims[c] = dict(status=look["outcomes"][c], sentence_ids=ids)
    look["claims"] = claims
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
            raise LookRefused(f"refused: official looks are written to {K.OUT} only")
        body, digest = official_prereg()
        if json.loads(canonical(body)) != json.loads(canonical(prereg)):
            raise LookRefused("refused: the pre-registration given is not prereg.json")
        if prereg_sha256 is not None and prereg_sha256 != digest:
            raise LookRefused("refused: the approved fingerprint is not prereg.json's")
        if K.todo_paths():
            raise LookRefused(f"refused: the claims block still holds TODO markers: {K.todo_paths()}")
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
    look = make_look(R, templates, group)
    digest = sha256_bytes(canonical(look).encode("utf8"))
    try:
        md = look_md(look, digest)
    except Exception as error:  # noqa: BLE001 - the report must never cost the look (the JSON is the record)
        md = (f"# Near-80 version 9 {group} look\n\nThe full report could not be formatted ({error!r}); read "
              f"{json_path.name} (SHA-256 {digest}).\n")
    write_text_once(md_path, md)
    got = write_json_once(json_path, look)
    if got != digest:
        raise LookRefused("refused: the look's JSON did not hash as computed")
    return look


# --------------------------------------------------------------------------------------------- the halt report

def halt_sentence_no_training(rules, templates=None):
    """Section 14.5's no-training stop for the first failed rule (freeze.py, step 7). Raises ValueError when no rule
    failed."""
    rule = next((r for r in rules or [] if not r.get("passed")), None)
    if rule is None:
        raise ValueError("no no-training rule failed")
    templates = templates or default_templates("scratch")
    return no_training_sentence(templates["sentences"]["stop.no_training"]["text"], rule)


def _infer_mode(root):
    if os.path.realpath(str(root)) == os.path.realpath(str(K.OUT)):
        return "official"
    for rel in (K.LAYOUT["status"],):
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
    """Whichever stop applies now (the runner's 'report'): the reproduction stop when the reproduction check
    differed, then one sentence per stopped phase. Reads only, unless write=True (looks/halt_report.json, once)."""
    root = Path(out_root)
    mode = mode or _infer_mode(root)
    prereg, prereg_sha = None, None
    if mode == "official":
        prereg, prereg_sha = official_prereg()
    if templates is None:
        templates = default_templates(mode, prereg)
    t = _template_getter(templates, "confirmatory")
    rd = Reader(root, mode, prereg_sha)
    rows, stop = [], None
    rp = rd.read(K.LAYOUT["reproduction_differs"], kinds=("near80v9-reproduction-differs",), strict=False)
    pl = rp or rd.read(K.LAYOUT["plumbing"], kinds=("near80v9-plumbing",), strict=False)
    names = reproduction_failures((pl or {}).get("reproduction"))
    if names:
        stop = "stop.reproduction"
        rows.append(dict(id=stop, text=reproduction_sentence(t(stop), names)))
    stops = []
    sd = root / "stops"
    if sd.is_dir():
        for p in sorted(sd.glob("*.json")):
            st = rd.read(p.relative_to(root), kinds=("near80v9-phase-stop",))
            stops.append(dict(phase_id=st.get("phase_id"), phase_kind=st.get("phase_kind"), setting=st.get("setting"),
                              world=st.get("world"), seed=st.get("seed"), group=st.get("group"),
                              reason=st.get("reason")))
    stops.sort(key=_phase_order)
    for st in stops:
        rows.append(dict(id="stop.phase", text=phase_stop_sentence(t("stop.phase"), st)))
    rep = dict(kind=HALT_KIND, mode=mode, prereg_sha256=prereg_sha, stop=stop, sentences=rows, phase_stops=stops,
               inputs=dict(rd.inputs), note=None if rows else "No stop applies now.", written=utc())
    if write:
        write_json_once(root / K.LAYOUT["halt_report"], rep)
    return rep


def reproduction_failures(repro):
    """The plain names of the reproduction items that differed, in their order, each once. Accepts {item: {passed,
    plain}} and {world: {item: {...}}}."""
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


def look_md(look, digest):
    """looks/<group>.md: the sentences first, then the tables."""
    g = look["group"]
    br, mk = look["brackets"], look["marks"]
    head = "Plumbing look, seed 0" if g == "exploratory" else f"Confirmatory look, fresh seeds {br['SEEDS']}"
    L = [f"# Near-80 (version 9): {head}", "", look["label"], "",
         f"Mode: {look['mode']}. Seeds {', '.join(str(s) for s in look['seeds'])}. Pre-registration "
         f"{look['prereg_sha256'] or 'not recorded (a scratch or stub run)'}. This look's JSON SHA-256 {digest}.", ""]
    if g == "exploratory":
        L += ["Seed 0 never carries a claim: every number below is plumbing, not a claim.", ""]
    L += ["## The sentences (section 14 of the pre-registration)", ""]
    L += [f"- {r['text']}" for r in look["sentences"]] + [""]
    c1, c2 = mk["C1"], mk["C2"]
    L += ["## C1 (version 8's claim 4, main setting)", "", "| mark | value | needed | result |", "|---|---|---|---|"]
    nl = c1["net_lead"]
    L.append(f"| net lead over {nl['best_rival_plain'] or 'not available'} | {_f(nl['lead'])} | "
             f"{_f(nl['margin'])} | {'pass' if nl['passed'] else 'fail'} |")
    for sit in K.KEEPS_SITUATIONS:
        k = c1["keeps"][sit]
        L.append(f"| keeps the gain, {K.SITUATION_PLAIN[sit]} | {_f(k['difference'])} | {_f(k['floor'])} | "
                 f"{'pass' if k['passed'] else 'fail'} |")
    L.append(f"| beats the matched even pull | {_f(c1['even_pull']['difference'])} | {_f(c1['even_pull']['margin'])} "
             f"| {'pass' if c1['even_pull']['passed'] else 'fail'} |")
    L += ["", f"## C2 (BCA's switched ramp against each plain twin; measure {c2['measure']})", "",
          "| plain twin | mean paired difference | margin | sd | p (paired t) | BCA ahead | result |",
          "|---|---|---|---|---|---|---|"]
    for twin, c in c2["comparisons"].items():
        res = "open" if c2.get("open") else ("pass" if c.get("passed") else "fail")
        L.append(f"| {c['plain']} | {_f(c['mean'])} | {'TODO' if c['margin'] is None else _f(c['margin'])} | "
                 f"{_f(c['sd'])} | {_f(c['p'], 'value')} | {c['positive']} of {c['n']} | {res} |")
    L += ["", "## Claims", ""] + [f"- {c}: {v['status']}" + (f" ({v['plumbing_outcome']})" if
                                                             'plumbing_outcome' in v else "")
                                  for c, v in look["claims"].items()]
    L += [f"- decision: {look['decision'].get('status')} (rule {look['decision'].get('rule')})", ""]
    fs = look["failed_seeds"]
    if fs["C1"] or fs["C2"]:
        L += ["## Failed seeds", ""] + [f"- {c}, seed {s}: {r}" for c in CLAIM_IDS for s, r in fs[c].items()] + [""]
    L += ["## Brackets", "", "| bracket | value |", "|---|---|"]
    for name in ("SEEDS", "NS", "S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "N1", "N2", "N3", "NZ", "NK2",
                 "RF", "GMIN", "GMAX", "NSW", "GB", "MULT", "V", "NF"):
        v = br.get(name)
        kind = "count" if name in ("NS", "S", "N", "RF", "GMIN", "GMAX", "NSW", "GB") else "points"
        L.append(f"| {name} | {v if isinstance(v, str) else _f(v, kind)} |")
    rep = look.get("reported") or {}
    pts = (rep.get("measures") or {}).get("points") or {}
    L += ["", "## Points in strong current, main setting (mean over seeds)", "",
          "| arm | stalling world | no-stall world |", "|---|---|---|"]
    for a in K.ARMS[SETTING]:
        s_ = ((pts.get("stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
        n_ = ((pts.get("no_stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
        L.append(f"| {K.ARM_PLAIN.get(a, a)} ({a}) | {_f(s_)} | {_f(n_)} |")
    sw = rep.get("switches") or {}
    if sw:
        L += ["", "## BCA's switch on the pooled groups", "",
              "| world | seed | full-throttle rows | rest rows | on | height | refused |", "|---|---|---|---|---|---|---|"]
        for w, by in sw.items():
            for s, r in by.items():
                L.append(f"| {K.WORLD_PLAIN[w]} | {s} | {r['n_full_throttle']} | {r['n_rest']} | {r['on']} | "
                         f"{_f(r['height'])} | {r['refused']} |")
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
    ap = argparse.ArgumentParser(description="version 9 looks: pooled.py exploratory|confirmatory|report")
    ap.add_argument("what", choices=("exploratory", "confirmatory", "report"))
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
            raise LookRefused("refused: the official looks are written by the runner's look phases only")
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
