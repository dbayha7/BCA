"""Near-80 (version 9, main setting): the look. DRAFT, not frozen. One look per group: the fresh group (the official
run: the validity conditions, then C1, then C2 and its mirror C3, the pooled-group count and the refusals) and the
smoke group (the plumbing's seed 0, in scratch and stub runs, every sentence prefixed "not a claim"). Each fills in
the sentences of section 14 of PREREGISTRATION.md exactly as its section 14.5 says. Also the halt report.

Inputs (written by the runner's phases, each checked against its .sha256 sidecar; in official mode every record must
be official and carry the approved fingerprint of prereg.json): the data records (data_main: the gate's checks, the
outcome-free count of the pooled groups), BCA's switch records (the certificate on the pooled groups; refused: true
when BCA's height was not finite; the sorted full-throttle scores for the ranks), the policy locks (the arm tables
with the refused arms, the heights, the strengths), the values of every world and seed, the stop and skip records,
the stage lock (its plan names the seeds), the runners' status files, the ledger's attempt records (the resource
table) and the plumbing record (the halt report).

The numbers (section 14's bracket table; every one pooled, that is the mean over the group's seeds of each seed's
paired difference, stalling world, strong current, main setting, unless stated):
  S, N          seeds whose switch is on in the stalling world, off in the no-stall world (whole numbers)
  X, H, W       BCA's switched ramp minus no caution; the full-log actor minus no caution; no-stall: no caution minus BCA
  NET, D, R     step 2's net for BCA; NET minus the net of the best rule that cannot switch itself off; that rule's
                name. The rivals are K.LEAD_RIVALS (step 2's five world-blind rivals plus the hand-set range limit;
                ties to the first); step 2's own lead over its own five (step2lib.go_no_go, pinned) is reported beside.
  E, L          BCA minus the matched even pull; BCA's loss to no caution in recorded conditions, big waves, big kicks
  K, F, NC      the certified seeds (BCA's height finite in both worlds; neither failed nor refused), the refused
                seeds, and the number of certified seeds C2 and C3 read
  DC, RC, LC, EC   C1's lead, best rival, largest loss and even-pull margin on the certified seeds alone (description)
  D1, K1, LB1 ...  C2 and C3: BCA's switched ramp trained alone (obca_k1_alone) minus the twin, net across worlds,
                over the certified seeds; the seeds where it was positive; the one-sided 95% lower bound of the mean
                (Student t). Twin 1 the mean-difference switch with the plain 90% gap, 2 with twice it, 3 with the
                hand-set range limit (K.TWIN_ARMS). KMIN = ceil(0.8 NC).
  A             seed-worlds (of 20) where the mean-difference switch made BCA's on/off call
  P             no caution's share of strong-current steps past 80% (written as a percentage)
  V             the noise floor: |BCA's switched ramp minus the same ramp trained alone|, pooled over the certified seeds
  NSEEDS, SEEDS the number of fresh seeds and their range
  NMIN, NMAX, NREACH, NSW, N8MIN, N8MAX   the pooled full-throttle group's smallest and largest size over the
                seed-worlds, how many reached 29 rows, how many seed-worlds, and version 8's groups' sizes for scale

Failed, stopped and refused seeds (section 14.5). A seed that failed its data (the gate's checks) or had a phase
stop fails every mark of C1, C2 and C3: its numbers are set to null before any mark is computed, and its own clause
("seed [s] failed [check] ..." or "seed [s]'s [phase] stopped ...") tells it; when every missed mark is told by such
a clause, C1's last sentence says that a stop and not a return decided it. A seed where BCA REFUSED (its certificate
on with a height that is not finite, in either world) is read per arm: in C1 the refused arms (K.BCA_DEPENDENT) and
the controls matched to BCA (unmatched there) take the no-caution actor's numbers in that seed-world, so no mark
fails by a missing number and the refusal's cost enters the pooled means; its own clause names it; C2 and C3 leave
the seed out (they read the certified seeds only, at least K.MIN_CERTIFIED of them). A seed with no failure, stop or
refusal whose evaluation lacks a number C1, C2 or C3 reads fails the same way with [check] "its evaluation (numbers
missing)"; one whose matched rivals could not be matched (and BCA did not refuse) fails C1's marks only. The validity
conditions are computed over the seeds that did not fail (a refused seed's no-caution and full-log numbers count:
they do not depend on BCA).

C2 and C3 (section 9.3). Per twin, d = NET(obca_k1_alone) - NET(twin) over the certified seeds: "ahead" when the mean
is at least MARKS_C2.net_margin and d is positive in at least ceil(0.8 n) seeds; "behind" (C3 holds for the twin)
when the mean is at most MARKS_C3.net_margin and d is positive in at most MARKS_C3.positive_seeds_max seeds; "tie"
otherwise. C2 holds when BCA is ahead of all three (multiplicity all_three). A failed seed fails C2 and C3 by the
pre-set rule before anything else is said; then "Not read" below MIN_CERTIFIED certified seeds takes precedence over
"dropped" by the noise floor; both over holds / fails.

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
CLAIMS = ("c1", "c2", "c3")
SMOKE_STATUS = "plumbing check, not a claim"
LOOK_KIND = {"fresh": "near80-v9-look-fresh", "smoke": "near80-v9-look-smoke"}
HALT_KIND = "near80-v9-halt-report"
GO_RULE = dict(g1_margin=K.MARKS["net_lead_margin"], g2_floor=K.MARKS["keeps_floor"],
               g3_margin=K.MARKS["even_pull_margin"])
SEEDS_WORDS = "seed [seeds]"
EVALUATION_MISSING_CHECK = "its evaluation (numbers missing)"
RIVAL_MISSING_CHECK = "its matching (a rival could not be matched or scored)"
TWIN_INDEX = {t: i + 1 for i, t in enumerate(K.TWIN_ARMS)}      # D1/K1/LB1 the plain 90% gap, 2 twice it, 3 the limit
MATCHED_RIVALS = ("standin", "even_pull", "random_1", "random_2")
# A refused seed-world's arms that take the no-caution actor's numbers in C1 (section 5.4): BCA's own arms and the
# controls matched to its strength (unmatched when it refused).
SUBSTITUTED = tuple(K.BCA_DEPENDENT) + MATCHED_RIVALS
READINGS = ("ahead", "tie", "behind")
# The dose ladder (reported): the stalling world's strong-current points by height, from the mean gap to the limit.
LADDER = ("meangap_k1", "fixed_gated", "meandiff_q90x1", "obca_k1", "meandiff_q90x2", "obca_k2", "range_hinge")
# Student t, one-sided 95% (0.95 quantile) by degrees of freedom 1-30, for the reported lower bound (the pinned
# environment has no scipy in the look's process); beyond 30 the df-30 value is used (a little conservative).
T_95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
        11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725,
        21: 1.721, 22: 1.717, 23: 1.714, 24: 1.711, 25: 1.708, 26: 1.706, 27: 1.703, 28: 1.701, 29: 1.699, 30: 1.697}
REPRO_ORDER = ("removal", "critic", "clone", "none", "w0_check", "obca_k1", "range_hinge")
REPRO_PLAIN = {"removal": "the main test's removal", "critic": "the main test's critic",
               "clone": "the main test's cloned actor", "none": "the main test's no-caution actor",
               "w0_check": "step 2's plumbing-check actor", "obca_k1": "step 2's switched-ramp actor",
               "critic_second": "step 2's switched-ramp actor", "switch": "step 2's switched-ramp actor",
               "range_hinge": "step 2's hand-set range-limit actor"}
SETTING_PLAIN = {"main": "main setting"}


def _go_needs(setting="main"):
    out = [("points", setting, w, a, SCORED) for w in K.WORLDS for a in (K.PRIMARY, "none") + K.LEAD_RIVALS]
    out += [("points", setting, "stall", a, sit) for sit in K.KEEPS_SITUATIONS for a in (K.PRIMARY, "none")]
    return tuple(out)


RIVAL_NEEDS = tuple(("points", "main", w, a, SCORED) for w in K.WORLDS for a in MATCHED_RIVALS)
# What C1, C2, C3 and the validity conditions read, as (measure, setting, world, logical id, situation), without the
# matched rivals (a seed with no refusal whose rivals could not be matched fails C1's marks only). In a refused seed
# the BCA-dependent arms and the twins' values are not required (C1 substitutes the no-caution actor for the former;
# C2 and C3 leave the seed out), only in the certified seeds.
EVALUATION_NEEDS = tuple(x for x in _go_needs("main") if x not in RIVAL_NEEDS) + (
    ("points", "main", "stall", "full_none", SCORED), ("past80_share", "main", "stall", "none", SCORED)) + tuple(
    ("points", "main", w, t, SCORED) for w in K.WORLDS for t in K.TWIN_ARMS + (K.NOISE_ARM,))
BCA_NEEDS = tuple(x for x in EVALUATION_NEEDS if x[3] in K.BCA_DEPENDENT)
CERTIFIED_ONLY_NEEDS = tuple(x for x in EVALUATION_NEEDS if x[3] in K.TWIN_ARMS + tuple(K.BCA_DEPENDENT))


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
    """What a look reads. m[measure]['main'][world][seed][logical id][situation] holds the numbers with a failed
    seed's numbers set to null and a refused seed-world's BCA-dependent arms and unmatched controls replaced by the
    no-caution actor's (for the marks); raw[...] the same without those rules (for the validity conditions and the
    reported tables)."""

    def __init__(self, group, seeds, mode, prereg_sha256, out_root):
        self.group, self.seeds, self.mode = group, [int(s) for s in seeds], mode
        self.prereg_sha256, self.out_root = prereg_sha256, Path(out_root)
        self.raw = {m: {st: {w: {s: {} for s in self.seeds} for w in K.WORLDS} for st in K.SETTINGS}
                    for m in MEASURES}
        self.m = None
        self.values, self.switches, self.locks, self.data_main = {}, {}, {}, {}
        self.failed = {}                    # seed -> reason (fails C1, C2 and C3)
        self.failed_checks = {}             # seed -> [check]
        self.stopped = {}                   # seed -> [stop summaries] (a seed failed by a stopped phase)
        self.c1_only = {}                   # seed -> reason (a matched rival missing: C1's marks only)
        self.c1_only_checks = {}
        self.refused = {}                   # seed -> {world: n_full_throttle}: BCA refused (read per arm)
        self.substituted = {}               # seed -> {world: [arms that took the no-caution actor's numbers]}
        self.counts = {}                    # (seed, world) -> the data record's pooled_groups block
        self.phase_stops, self.skipped, self.runner_skipped = [], {}, {}
        self.plumbing, self.attempts, self.notes = None, [], []
        self.plan, self.inputs = None, {}

    @property
    def pts(self):
        return self.m["points"]

    @property
    def certified(self):
        """The seeds neither failed nor refused: the seeds C2, C3 and the noise floor read."""
        return [s for s in self.seeds if s not in self.failed and s not in self.refused]

    def fail(self, seed, reason, check):
        if seed not in self.failed:
            self.failed[seed] = reason
            self.failed_checks[seed] = check

    def fail_c1_only(self, seed, reason, check):
        if seed not in self.c1_only and seed not in self.failed:
            self.c1_only[seed] = reason
            self.c1_only_checks[seed] = check

    def apply_failures(self):
        """m: raw with a failed seed's numbers set to null, and, in a refused seed, every BCA-dependent arm and every
        control matched to BCA whose values are null in a seed-world (refused, unmatched) given a copy of the
        no-caution actor's numbers there (section 5.4: the refused arm's controller is the one trained without
        caution, and a control matched to zero strength is the same actor). raw is untouched."""
        self.m = copy.deepcopy(self.raw)
        for s in self.failed:
            if s not in self.seeds:
                continue
            for measure in MEASURES:
                for w in K.WORLDS:
                    self.m[measure]["main"][w][s] = {a: None for a in K.ARMS["main"]}
        for s in self.refused:
            if s not in self.seeds or s in self.failed:
                continue
            for w in K.WORLDS:
                taken = []
                for a in SUBSTITUTED:
                    if self.raw["points"]["main"][w][s].get(a) is None:
                        taken.append(a)
                        for measure in MEASURES:
                            none = self.m[measure]["main"][w][s].get("none")
                            self.m[measure]["main"][w][s][a] = None if none is None else copy.deepcopy(none)
                self.substituted.setdefault(s, {})[w] = taken

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
                R.stopped.setdefault(s, []).append(summary)
        return
    R.fail(int(seed), why, stopped_phase_check(summary))
    R.stopped.setdefault(int(seed), []).append(summary)


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
    """{seed: [what is missing]} for the group's seeds that have no failure record yet lack a number C1, C2 or C3
    reads (section 14.5): such a seed's learner failed. A refused seed's BCA-dependent numbers (C1 substitutes the
    no-caution actor) and its twins' numbers (C2 and C3 leave the seed out) are not required; for 'rivals', a seed
    whose matched rivals could not be matched fails C1's marks only (a refused seed's unmatched rivals are
    substituted, not missing)."""
    needs = evaluation_needs(R, which)
    out = {}
    for s in R.seeds:
        if s in R.failed or (which == "rivals" and s in R.c1_only):
            continue
        gone = []
        for measure, st, w, arm, sit in needs:
            if s in R.refused and (measure, st, w, arm, sit) in CERTIFIED_ONLY_NEEDS:
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


def t_one_sided_95(df):
    """Student t's 0.95 quantile for df degrees of freedom (T_95; the df-30 value beyond 30); None below 1."""
    df = int(df)
    if df < 1:
        return None
    return T_95[min(df, 30)]


def lower_bound_95(per):
    """The one-sided 95% lower bound of the mean of the paired differences per (None where fewer than two numbers):
    mean - t(0.95, n - 1) * sd / sqrt(n), sd the sample standard deviation (n - 1). Reported, not used."""
    xs = [float(v) for v in per if v is not None]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean - t_one_sided_95(n - 1) * math.sqrt(var) / math.sqrt(n)


def twin_difference(m, twin, seeds, bca=K.NOISE_ARM):
    """C2's and C3's number for one twin: the net across worlds of BCA's side (obca_k1_alone, the copy trained
    alone like the twins; or obca_k1 for the reported reading) minus the twin's, per seed over `seeds` (the certified
    seeds); the pooled mean; the seeds positive; n (the seeds with a number); the one-sided 95% lower bound."""
    a, b = net_per_seed(m, bca, seeds), net_per_seed(m, twin, seeds)
    per = [_diff(x, y) for x, y in zip(a, b)]
    return dict(per_seed=per, mean=step2().mean_or_none(per), positive=sum(1 for v in per if v is not None and v > 0),
                n=sum(1 for v in per if v is not None), lower_bound=lower_bound_95(per), net_bca=a, net_twin=b,
                seeds=list(seeds), bca=bca)


def twin_reading(d, n_certified):
    """'ahead' (C2's part: the mean at least MARKS_C2.net_margin and positive in at least ceil(0.8 n) seeds),
    'behind' (C3: the mean at most MARKS_C3.net_margin and positive in at most MARKS_C3.positive_seeds_max seeds) or
    'tie'; None when the mean is missing. Also the sign thresholds used."""
    n = int(n_certified)
    kmin = K.c2_positive_needed(n)
    kmax = int(K.MARKS_C3["positive_seeds_max"])
    if d["mean"] is None or n <= 0:
        return None, dict(kmin=kmin, kmax=kmax, n=n)
    if d["mean"] >= float(K.MARKS_C2["net_margin"]) and d["positive"] >= kmin:
        return "ahead", dict(kmin=kmin, kmax=kmax, n=n)
    if d["mean"] <= float(K.MARKS_C3["net_margin"]) and d["positive"] <= kmax:
        return "behind", dict(kmin=kmin, kmax=kmax, n=n)
    return "tie", dict(kmin=kmin, kmax=kmax, n=n)


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
    """C1's numbers over `seeds` (points with the failure rules applied): the net lead computed here over
    K.LEAD_RIVALS (step 2's five world-blind rivals plus the hand-set range limit; ties to the first), with step 2's
    own go_no_go (pinned step2lib) for the keeps and even-pull marks and for its own lead over its own five rivals
    (reported beside). Returns (g, net, best_net, lead, keeps, loss, extra) with extra = dict(best, rival_nets,
    passed, step2_lead)."""
    S = step2()
    g = S.go_no_go(m["main"], seeds, GO_RULE)
    d1 = g["G1_detail"]
    net = d1["net_primary"]["net"]
    rival = {a: S.net(m["main"], a, seeds) for a in K.LEAD_RIVALS}
    vals = [rival[a]["net"] for a in K.LEAD_RIVALS]
    best_net = None if any(v is None for v in vals) else max(vals)
    best = None if best_net is None else [a for a in K.LEAD_RIVALS if rival[a]["net"] == best_net][0]
    lead = None if net is None or best_net is None else net - best_net
    passed = bool(lead is not None and lead >= float(K.MARKS["net_lead_margin"]))
    keeps = {sit: g["G2_detail"][sit] for sit in K.KEEPS_SITUATIONS}
    loss = {sit: None if keeps[sit]["mean"] is None else -keeps[sit]["mean"] for sit in K.KEEPS_SITUATIONS}
    s2_best = d1["best_world_blind"]
    s2_lead = None if net is None or d1["best_world_blind_net"] is None else net - d1["best_world_blind_net"]
    extra = dict(best=best, best_net=best_net, rival_nets={a: rival[a]["net"] for a in K.LEAD_RIVALS}, passed=passed,
                 rivals=list(K.LEAD_RIVALS),
                 step2_lead=dict(best_rival=s2_best, best_rival_plain=None if s2_best is None else K.ARM_PLAIN[s2_best],
                                 best_rival_net=d1["best_world_blind_net"], lead=s2_lead, passed=bool(g["G1"]),
                                 rivals=list(K.WORLD_BLIND),
                                 note="step 2's own lead over its own five world-blind rivals (step2lib.go_no_go, "
                                      "pinned); reported beside C1's lead over the enlarged set"))
    return g, net, best_net, lead, keeps, loss, extra


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


def switch_calls(R):
    """Per seed-world, from the switch records: the four switches' calls (BCA's certificate, the mean difference
    > 0.10, the plain 90% gap > 0.10, the rank test p < 0.05) and whether each made BCA's call; A = the seed-worlds
    where the mean-difference switch did."""
    rows, agree = [], dict(meandiff=0, conformal=0, rank=0)
    n = 0
    for s in R.seeds:
        for w in K.WORLDS:
            sw = R.switches.get(("main", w, s))
            if sw is None or not sw.get("defined", True):
                rows.append(dict(seed=s, world=w, defined=False if sw is not None else None))
                continue
            c, g, rt = sw.get("certificate") or {}, sw.get("groups") or {}, sw.get("rank_test") or {}
            bca = c.get("on")
            md = _num(g.get("mean_full_throttle")), _num(g.get("mean_rest"))
            q9 = _num(g.get("q90_full_throttle")), _num(g.get("q90_rest"))
            p = _num(rt.get("p"))
            mean_gap = None if None in md else md[0] - md[1]
            plain_gap = None if None in q9 else q9[0] - q9[1]
            row = dict(seed=s, world=w, defined=True, bca=bca, height=_num(c.get("height")),
                       mean_gap=mean_gap, meandiff=None if mean_gap is None else bool(mean_gap > K.MEANDIFF_MARGIN),
                       plain_gap=plain_gap,
                       conformal=None if plain_gap is None else bool(plain_gap > K.ARM_RULE["conformal_margin"]),
                       rank_p=p, rank=None if p is None else bool(p < K.ARM_RULE["rank_p"]), refused=sw.get("refused"))
            if bca is not None:
                n += 1
                for k in ("meandiff", "conformal", "rank"):
                    if row[k] is not None and bool(row[k]) == bool(bca):
                        agree[k] += 1
            rows.append(row)
    return dict(rows=rows, seed_worlds=n, agree=agree, A=agree["meandiff"])


def brackets(results):
    """Every bracket of section 14's table (numbers; R and twin names as plain names), and per-seed lists."""
    R = results
    S = step2()
    m, seeds = R.m, R.seeds
    cert = R.certified
    pts1 = claim1_points(R)
    x = S.paired(pts1["main"], "stall", K.PRIMARY, "none", seeds, SCORED)
    h = S.paired(m["points"]["main"], "stall", "full_none", "none", seeds, SCORED)
    w = S.paired(pts1["main"], "no_stall", "none", K.PRIMARY, seeds, SCORED)
    g, net, best_net, lead, keeps, loss, x1 = _claim1_numbers(pts1, seeds)
    best = x1["best"]
    ev = g["G3_detail"]
    losses = list(loss.values())
    twins = {t: twin_difference(m["points"], t, cert) for t in K.TWIN_ARMS}
    p = pooled_arm(m["past80_share"], "main", "stall", "none", seeds)
    nf = noise_floor(R)
    cs = count_summary(R)
    calls = switch_calls(R)
    # the three marks on the certified seeds alone (description)
    if cert:
        gc, netc, bestc_net, leadc, keepsc, lossc, xc = _claim1_numbers(pts1, cert)
        lossesc = list(lossc.values())
        cert_marks = dict(lead=leadc, best=xc["best"], best_plain=None if xc["best"] is None else K.ARM_PLAIN[xc["best"]],
                          loss=None if any(v_ is None for v_ in lossesc) else max(lossesc),
                          even_pull=gc["G3_detail"]["mean"],
                          passed=bool(xc["passed"] and gc["G2"] and gc["G3"]), net=netc, n=len(cert))
    else:
        cert_marks = dict(lead=None, best=None, best_plain=None, loss=None, even_pull=None, passed=None, net=None, n=0)
    br = dict(
        S=_count_switch(R, "stall", True), N=_count_switch(R, "no_stall", False),
        X=x["mean"], H=h["mean"], W=w["mean"], NET=net, D=lead, R=None if best is None else K.ARM_PLAIN[best],
        R_id=best, R_net=best_net, E=ev["mean"],
        L=None if any(v_ is None for v_ in losses) else max(losses), L_by_situation=loss,
        K=len(cert), F=len([s for s in R.refused if s in seeds]), NC=len(cert),
        DC=cert_marks["lead"], RC=cert_marks["best_plain"], RC_id=cert_marks["best"], LC=cert_marks["loss"],
        EC=cert_marks["even_pull"], certified_marks_pass=cert_marks["passed"],
        KMIN=K.c2_positive_needed(len(cert)), A=calls["A"],
        P=p["mean"], V=nf["V"], NSEEDS=len(seeds), SEEDS=seeds_range(seeds),
        NMIN=cs["NMIN"], NMAX=cs["NMAX"], NREACH=cs["NREACH"], NSW=cs["NSW"], N8MIN=cs["N8MIN"], N8MAX=cs["N8MAX"])
    for t, i in TWIN_INDEX.items():
        br[f"D{i}"] = twins[t]["mean"]
        br[f"K{i}"] = twins[t]["positive"]
        br[f"LB{i}"] = twins[t]["lower_bound"]
    per_seed = dict(X=x["per_seed"], H=h["per_seed"], W=w["per_seed"], NET=g["G1_detail"]["net_primary"]["per_seed"],
                    E=ev["per_seed"], P=p["per_seed"], seeds=list(seeds), certified=list(cert),
                    L={sit: [None if v_ is None else -v_ for v_ in keeps[sit]["per_seed"]] for sit in keeps},
                    twins={t: dict(per_seed=v["per_seed"], net_bca=v["net_bca"], net_twin=v["net_twin"],
                                   seeds=v["seeds"]) for t, v in twins.items()},
                    noise_floor=nf["per_seed"], rival_nets=x1["rival_nets"], step2_lead=x1["step2_lead"],
                    certified_marks=cert_marks)
    return dict(br, per_seed=per_seed)


def seeds_not_failed(R):
    """The group's seeds that did not fail (a failed seed is told by its own clause)."""
    return [s for s in R.seeds if s not in R.failed]


def noise_floor(R):
    """Section 9.1's noise floor on the certified seeds (neither failed nor refused; raw numbers): the pooled
    |BCA's switched ramp minus the same ramp trained alone|, stalling world, strong current, at most 5. When no such
    seed is left, or a number is missing on one, it cannot be read (unmeasured: true, passed None)."""
    seeds = R.certified
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
    """C2's and C3's marks as near80_constants holds them, or None while any is not fixed (cannot occur in the
    frozen code; kept for the extractor's 'Margins not fixed' sentence)."""
    c2, c3 = K.MARKS_C2, K.MARKS_C3
    if any(c2.get(k) is None for k in ("net_margin", "positive_seeds_min", "positive_share", "multiplicity")):
        return None
    if any(c3.get(k) is None for k in ("net_margin", "positive_seeds_max")) or K.MIN_CERTIFIED is None:
        return None
    return dict(net_margin=float(c2["net_margin"]), positive_seeds_min=int(c2["positive_seeds_min"]),
                positive_share=float(c2["positive_share"]), multiplicity=str(c2["multiplicity"]),
                c3_net_margin=float(c3["net_margin"]), c3_positive_seeds_max=int(c3["positive_seeds_max"]),
                min_certified=int(K.MIN_CERTIFIED))


def marks(results):
    """C1 (step 2's three go marks on the main setting, the lead over K.LEAD_RIVALS), C2 (BCA's switched ramp trained
    alone against each plain twin, paired by certified seed on the net across worlds: ahead of all three) and C3 (the
    mirror: a twin ahead of BCA, one claim per twin)."""
    R = results
    m = R.m
    cert = R.certified
    g, net, best_net, lead, keeps, loss, x1 = _claim1_numbers(claim1_points(R), R.seeds)
    best = x1["best"]
    c1 = dict(
        net_lead=dict(net=net, best_rival=best, best_rival_plain=None if best is None else K.ARM_PLAIN[best],
                      best_rival_net=best_net, lead=lead, margin=K.MARKS["net_lead_margin"], rival_nets=x1["rival_nets"],
                      rivals=x1["rivals"], per_seed=g["G1_detail"]["net_primary"]["per_seed"], passed=x1["passed"],
                      step2_lead=x1["step2_lead"]),
        keeps={sit: dict(difference=keeps[sit]["mean"], loss=loss[sit], per_seed=keeps[sit]["per_seed"],
                         floor=K.MARKS["keeps_floor"],
                         passed=bool(keeps[sit]["mean"] is not None and keeps[sit]["mean"] >= K.MARKS["keeps_floor"]))
               for sit in K.KEEPS_SITUATIONS},
        even_pull=dict(difference=g["G3_detail"]["mean"], per_seed=g["G3_detail"]["per_seed"],
                       margin=K.MARKS["even_pull_margin"], passed=bool(g["G3"])),
        passed=bool(x1["passed"] and g["G2"] and g["G3"]))
    fixed = marks_fixed()
    n = len(cert)
    not_read = bool(fixed is not None and n < fixed["min_certified"])
    twins = {}
    for t in K.TWIN_ARMS:
        d = twin_difference(m["points"], t, cert)
        reading, thr = (None, dict(kmin=K.c2_positive_needed(n), kmax=int(K.MARKS_C3["positive_seeds_max"]), n=n))
        if fixed is not None and not not_read:
            reading, thr = twin_reading(d, n)
        twins[t] = dict(difference=d["mean"], positive=d["positive"], n=d["n"], lower_bound=d["lower_bound"],
                        per_seed=d["per_seed"], seeds=d["seeds"], reading=reading,
                        passed=None if reading is None else bool(reading == "ahead"),
                        behind=None if reading is None else bool(reading == "behind"),
                        positive_needed=thr["kmin"], positive_at_most=thr["kmax"], plain=K.ARM_PLAIN[t],
                        index=TWIN_INDEX[t], bca=K.NOISE_ARM)
    readable = fixed is not None and not not_read and all(v["reading"] is not None for v in twins.values())
    all_ahead = None if not readable else bool(all(v["reading"] == "ahead" for v in twins.values()))
    c2 = dict(twins=twins, fixed=fixed, margins=dict(K.MARKS_C2), passed=all_ahead, n_certified=n,
              certified=list(cert), min_certified=K.MIN_CERTIFIED, not_read=not_read,
              positive_needed=K.c2_positive_needed(n), placeholders=sorted(K.placeholders_present()))
    c3 = dict(twins={t: (None if not readable else bool(v["reading"] == "behind")) for t, v in twins.items()},
              margins=dict(K.MARKS_C3), passed=None if not readable else bool(any(v["reading"] == "behind"
                                                                                 for v in twins.values())),
              n_certified=n, not_read=not_read)
    return {"c1": c1, "c2": c2, "c3": c3}


# --------------------------------------------------------------------------------------------- outcomes and sentences

def outcomes(look):
    """{c1: status, c1_by_stop: bool, c2: status, c2_each: {twin: reading}, c3: status, c3_each: {twin: status}} and
    why. C2's statuses in order of precedence: invalid, not_fixed, fail_failed_seed (a failed seed fails every mark by
    the pre-set rule), not_read, dropped, pass, fail (section 14.5); C3 follows C2's status, and reads 'holds' /
    'fails' when C2 was read."""
    val, mk = look["validity"], look["marks"]
    failed = bool(look.get("failed_checks"))
    out, why = {}, {}
    valid = val["claims"]["passed"]
    out["c1"] = "invalid" if not valid else ("pass" if mk["c1"]["passed"] else "fail")
    out["c1_by_stop"] = bool(out["c1"] == "fail" and _c1_numbers_missed(mk["c1"]) == [] and
                             (look.get("failed_checks") or look.get("c1_only_checks")))
    fixed = mk["c2"]["fixed"]
    nf = val["noise_floor"]
    each = {}
    if not valid:
        out["c2"], why["c2"] = "invalid", "the validity conditions"
    elif fixed is None:
        out["c2"], why["c2"] = "not_fixed", "the margins are not fixed"
    elif failed:
        out["c2"], why["c2"] = "fail_failed_seed", "a seed failed (every mark fails by the pre-set rule)"
    elif mk["c2"]["not_read"]:
        out["c2"], why["c2"] = "not_read", f"fewer than {fixed['min_certified']} certified seeds"
    elif nf["passed"] is False:
        out["c2"], why["c2"] = "dropped", "the noise floor"
    else:
        each = {t: v["reading"] for t, v in mk["c2"]["twins"].items()}
        out["c2"] = "pass" if mk["c2"]["passed"] else "fail"
    out["c2_each"] = each
    if out["c2"] in ("pass", "fail"):
        out["c3_each"] = {t: ("holds" if v == "behind" else "fails") for t, v in each.items()}
        out["c3"] = "holds" if any(v == "holds" for v in out["c3_each"].values()) else "fails"
    else:
        out["c3_each"] = {}
        out["c3"] = out["c2"]
        why["c3"] = why.get("c2")
    return out, why


def _c1_numbers_missed(c1):
    """The marks of C1 missed on their numbers (not by a failed or stopped seed): the net lead, the keeps (per
    situation), the even pull, as clause names."""
    out = []
    if not c1["net_lead"]["passed"] and c1["net_lead"]["lead"] is not None:
        out.append("net")
    for sit in K.KEEPS_SITUATIONS:
        k = c1["keeps"][sit]
        if not k["passed"] and k["loss"] is not None:
            out.append(f"keeps:{sit}")
    if not c1["even_pull"]["passed"] and c1["even_pull"]["difference"] is not None:
        out.append("even_pull")
    return out


def _template_getter(templates, group):
    T = templates["sentences"]

    def t(sid):
        text = T[sid]["text"]
        if group == "smoke":
            text = text.replace("fresh seeds [SEEDS]", "the plumbing's seed [SEEDS]").replace("fresh seeds", "seeds")
        return text
    return t


def certify_percent(n):
    """[p]: the exact chance of a finite height for a group of n rows, as a percentage number (None when n is not
    known)."""
    if n is None:
        return None
    return 100.0 * K.certify_chance(int(n))


def refused_clause_values(R_refused, seed):
    worlds = R_refused.get(seed) or {}
    ns = [v for v in worlds.values() if v is not None]
    n = min(ns) if ns else None
    return dict(s=CT.fmt(seed, "count"), n=CT.fmt(n, "count") if n is not None else CT.NOT_AVAILABLE,
                p=certify_percent(n),
                world=CT.join_items([K.WORLD_PLAIN[w] for w in K.WORLDS if w in worlds]) or "both worlds")


def _c1_clauses(look, t):
    """([one clause per missed mark] in section 14.5's order, any clause about a number): the net lead, the keeps,
    the even pull; then per failed seed, in seed order, the stopped-phase clause (one per stopped phase) for a seed
    failed by a stop, else the failed-seed clause; then per refused seed. A missed mark whose number is not available
    because a seed failed is told by that seed's own clause only."""
    mk = look["marks"]["c1"]
    failed = {int(s): c for s, c in (look.get("failed_checks") or {}).items()}
    for s, c in (look.get("c1_only_checks") or {}).items():
        failed.setdefault(int(s), c)
    stopped = {}
    for st in look.get("phase_stops") or []:
        if st.get("seed") is not None:
            stopped.setdefault(int(st["seed"]), []).append(st)
        elif st.get("phase_kind") == "lock":
            for s in failed:
                stopped.setdefault(s, []).append(st)
    refused = {int(s): v for s, v in (look.get("refused") or {}).items()}
    covered = bool(failed)
    out, numbers = [], False
    nl = mk["net_lead"]
    if not nl["passed"] and not (covered and (nl["net"] is None or nl["lead"] is None)):
        out.append(CT.fill(t("c1.clause.net"), dict(NET=nl["net"], D=nl["lead"],
                                                    R=nl["best_rival_plain"] or CT.NOT_AVAILABLE)))
        numbers = True
    for sit in K.KEEPS_SITUATIONS:
        k = mk["keeps"][sit]
        if not k["passed"] and not (covered and k["loss"] is None):
            out.append(CT.fill(t("c1.clause.keeps"), dict(L=k["loss"], situation=K.SITUATION_PLAIN[sit])))
            numbers = True
    ep = mk["even_pull"]
    if not ep["passed"] and not (covered and ep["difference"] is None):
        out.append(CT.fill(t("c1.clause.even_pull"), dict(E=ep["difference"])))
        numbers = True
    for s in sorted(failed):
        if s in stopped and str(failed[s]).startswith("its ") and " phase" in str(failed[s]):
            for st in stopped[s]:
                out.append(CT.fill(t("c1.clause.stopped_seed"),
                                   dict(s=CT.fmt(s, "count"), phase=phase_text(st.get("phase_kind"), st.get("setting")),
                                        reason=str(st.get("reason") or "no reason recorded"))))
        else:
            out.append(CT.fill(t("c1.clause.failed_seed"), dict(s=CT.fmt(s, "count"), check=failed[s])))
    for s in sorted(refused):
        if s in failed:
            continue
        out.append(CT.fill(t("c1.clause.refused_seed"), refused_clause_values(refused, s)))
    if not out:
        raise ValueError("c1 failed but no mark was missed")
    return out, numbers


def _c2_clauses(look, t):
    """([one clause per twin] in the twins' fixed order, each by its reading, then the refused seeds' clauses in
    seed order; the last sentence's id by the four cases)."""
    mk = look["marks"]["c2"]
    refused = {int(s): v for s, v in (look.get("refused") or {}).items()}
    nc = CT.fmt(mk["n_certified"], "count")
    out, readings = [], {}
    for twin, v in mk["twins"].items():
        r = v["reading"]
        readings[twin] = r
        vals = dict(twin=v["plain"], DT=v["difference"], LBT=v["lower_bound"], KT=CT.fmt(v["positive"], "count"), NC=nc)
        sid = {"ahead": "c2.clause.ahead", "tie": "c2.clause.tie", "behind": "c2.clause.behind"}.get(r)
        if sid is None:
            raise ValueError(f"c2 was read but {twin} has no reading")
        out.append(CT.fill(t(sid), vals))
    for s in sorted(refused):
        out.append(CT.fill(t("c2.clause.refused_seed"), refused_clause_values(refused, s)))
    order = list(K.TWIN_ARMS)
    behind = [tw for tw in order if readings[tw] == "behind"]
    if behind:
        last = ("c2.last.twin", dict(twin=mk["twins"][behind[0]]["plain"]))
    elif readings[order[0]] == "ahead" and all(readings[tw] == "tie" for tw in order[1:]):
        last = ("c2.last.dose", {})
    elif all(readings[tw] == "tie" for tw in order):
        last = ("c2.last.none", {})
    else:
        last = ("c2.last.other", {})
    return out, last


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
                                  n=CT.fmt(n, "count") if n is not None else CT.NOT_AVAILABLE, p=certify_percent(n)))


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
                                                  R=br["R"] or CT.NOT_AVAILABLE, E=br["E"],
                                                  K=CT.fmt(br["K"], "count"), F=CT.fmt(br["F"], "count"))))
    elif c1 == "fail":
        clauses, numbers = _c1_clauses(look, t)
        last = t("c1.last.numbers") if numbers else t("c1.last.stop")
        add("c1.fail", CT.fill(t("c1.fail"), common,
                               specials={"[one clause per missed mark]": "; ".join(clauses), "[last sentence]": last}))
    else:
        v = val["claims"]
        reasons = []
        if not v["harm"]["passed"]:
            reasons.append(CT.fill(t("c1.reason.harm"), dict(H=v["harm"]["value"])))
        if not v["past80"]["passed"]:
            reasons.append(CT.fill(t("c1.reason.past80"), dict(P=CT.fmt(v["past80"]["value"], "percent"))))
        add("c1.invalid", CT.fill(t("c1.invalid"), dict(common, reason="; ".join(reasons))))
    if group == "fresh":
        cm = br.get("certified_marks_pass")
        reading = CT.NOT_AVAILABLE if cm is None else ("pass" if cm else "fail")
        add("c1.certified", CT.fill(t("c1.certified"), dict(NC=CT.fmt(br["NC"], "count"), reading=reading,
                                                            DC=br["DC"], RC=br["RC"] or CT.NOT_AVAILABLE, LC=br["LC"],
                                                            EC=br["EC"])), claim=False)
    c2 = out_c["c2"]
    tw = look["marks"]["c2"]["twins"]
    nc = CT.fmt(br["NC"], "count")
    tvals = dict(common, NC=nc, KMIN=CT.fmt(br["KMIN"], "count"), A=CT.fmt(br["A"], "count"))
    for twin, v in tw.items():
        tvals[f"D{v['index']}"] = v["difference"]
        tvals[f"K{v['index']}"] = CT.fmt(v["positive"], "count")
        tvals[f"LB{v['index']}"] = v["lower_bound"]
    if c2 == "not_fixed":
        add("c2.not_fixed", CT.fill(t("c2.not_fixed"), tvals))
    elif c2 == "not_read":
        add("c2.not_read", CT.fill(t("c2.not_read"), tvals))
    elif c2 == "dropped":
        add("c2.dropped", CT.fill(t("c2.dropped"), dict(tvals, V=br["V"])))
    elif c2 == "fail_failed_seed":
        failed = failed_seeds_of(look)
        every = set(int(s) for s in look["seeds"]) <= set(failed)
        sid = "c2.every_seed_failed" if every else "c2.failed_seed"
        add(sid, fill_seeds(t(sid), failed, common))
    elif c2 == "pass":
        add("c2.holds", CT.fill(t("c2.holds"), tvals))
    elif c2 == "fail":
        clauses, (last_id, last_vals) = _c2_clauses(look, t)
        add("c2.fails", CT.fill(t("c2.fails"), tvals,
                                specials={"[one clause per twin]": "; ".join(clauses),
                                          "[last sentence]": CT.fill(t(last_id), last_vals)}))
    if c2 in ("pass", "fail"):
        behind = [twin for twin, v in tw.items() if v["reading"] == "behind"]
        for twin in behind:
            v = tw[twin]
            add("c3.holds", CT.fill(t("c3.holds"), dict(tvals, twin=v["plain"], DT=v["difference"],
                                                        KT=CT.fmt(v["positive"], "count"))))
        if not behind:
            add("c3.fails", CT.fill(t("c3.fails"), tvals))
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
    if sid.startswith("c3"):
        return "c3"
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
    cert = R.certified
    twins = {t: dict(against_alone=twin_difference(raw["points"], t, cert, bca=K.NOISE_ARM),
                     against_batch=twin_difference(raw["points"], t, cert, bca=K.PRIMARY),
                     over_every_seed=twin_difference(raw["points"], t, seeds, bca=K.NOISE_ARM)) for t in K.TWIN_ARMS}
    switches = {}
    for (st, w, s), sw in sorted(R.switches.items()):
        c, g = sw.get("certificate") or {}, sw.get("groups") or {}
        sorted_tail = (sw.get("pooled") or {}).get("tail_scores_sorted") or []
        r_rest, q90_rest = _num(c.get("R_rest")), _num(g.get("q90_rest"))
        heights = dict(bca=_num(c.get("height")), plain_gap=None if None in (_num(g.get("q90_full_throttle")), q90_rest)
                       else _num(g.get("q90_full_throttle")) - q90_rest)
        ranks = {}
        if sorted_tail:
            ranks = dict(bca_threshold=_rank_from_top(sorted_tail, _num(c.get("R_tail"))),
                         plain_q90=_rank_from_top(sorted_tail, _num(g.get("q90_full_throttle"))),
                         twice_plain=None if heights["plain_gap"] is None or q90_rest is None else
                         _rank_from_top(sorted_tail, q90_rest + 2.0 * heights["plain_gap"]),
                         twice_bca=None if heights["bca"] is None or r_rest is None else
                         _rank_from_top(sorted_tail, r_rest + 2.0 * heights["bca"]), n=len(sorted_tail))
        switches.setdefault(w, {})[str(s)] = dict(
            defined=sw.get("defined"), on=c.get("on"), height=_num(c.get("height")), q05=_num(c.get("q05")),
            finite_share_tail=c.get("finite_share_tail"), t_lo=_num(sw.get("t_lo")), t_up=_num(sw.get("t_up")),
            n_full_throttle=g.get("n_full_throttle"), n_rest=g.get("n_rest"), n_survivors=sw.get("n_survivors"),
            refused=sw.get("refused"), refusal=sw.get("refusal"), heights=heights, ranks=ranks,
            certify_chance=None if g.get("n_full_throttle") is None else K.certify_chance(int(g["n_full_throttle"])))
    calls = switch_calls(R)
    ladder = {}
    for s in seeds:
        lock = R.locks.get(("main", "stall", s)) or {}
        table = ((lock.get("arm_table") or {}).get("arms") or {})
        ladder[str(s)] = [dict(arm=a, plain=K.ARM_PLAIN.get(a, a), height=(table.get(a) or {}).get("h"),
                               k=(table.get(a) or {}).get("k"),
                               points=S.pts_of(raw["points"]["main"], "stall", s, a, SCORED)) for a in LADDER]
    composites = _composites(R)
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
    return dict(measures=tables, zone_shares=zones, nets=nets, twins=twins, switches=switches, switch_calls=calls,
                dose_ladder=ladder, composites=composites, locks=locks, certified=list(cert),
                substituted={str(s): v for s, v in sorted(R.substituted.items())},
                noise_floor=noise_floor(R), skipped=dict(R.skipped), runner_skipped=dict(R.runner_skipped),
                phase_stops=list(R.phase_stops), resources=dict(by_phase_kind=resources, cpu_hours_total=total_hours),
                data=data, notes=list(R.notes))


def _rank_from_top(sorted_tail, threshold):
    """The rank from the top of a threshold among the sorted full-throttle scores (1 = at or above the largest);
    None for a missing or non-finite threshold."""
    if threshold is None or not math.isfinite(float(threshold)):
        return None
    t = float(threshold)
    return int(sum(1 for v in sorted_tail if float(v) > t) + 1)


def _composites(R):
    """The two reported composites, assembled from trained arms per seed (raw points; no training): BCA's switch
    turning on the hand-set range limit (the range limit where BCA's switch was on, no caution where it was off, in
    each world); and "refusal means the range limit" (BCA's switched ramp where it set a height, the range limit
    where it refused). Each with its per-seed net across worlds and the pooled mean."""
    S = step2()
    raw = R.raw["points"]["main"]
    out = {}
    for name in ("bca_switch_range_limit", "refusal_means_range_limit"):
        per = []
        for s in R.seeds:
            vals = {}
            for w in K.WORLDS:
                sw = R.switches.get(("main", w, s)) or {}
                on = ((sw.get("certificate") or {}).get("on"))
                if name == "bca_switch_range_limit":
                    arm = K.HAND_SET_LIMIT if on else "none"
                else:
                    arm = K.PRIMARY if S.pts_of(raw, w, s, K.PRIMARY, SCORED) is not None else K.HAND_SET_LIMIT
                vals[w] = (S.pts_of(raw, w, s, arm, SCORED), S.pts_of(raw, w, s, "none", SCORED))
            gain = _diff(vals["stall"][0], vals["stall"][1])
            loss = _diff(vals["no_stall"][1], vals["no_stall"][0])
            per.append(None if gain is None or loss is None else gain - loss)
        out[name] = dict(per_seed=per, net=S.mean_or_none(per), seeds=list(R.seeds),
                         note="a composite of trained arms, assembled after the fact; description only")
    return out


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
        if c == "c3" and look["outcomes"].get("c3_each"):
            claims[c]["twins"] = dict(look["outcomes"]["c3_each"])
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
    c1, c2, c3 = mk["c1"], mk["c2"], mk["c3"]
    L += ["## Marks", "", "| claim | mark | value | needed | result |", "|---|---|---|---|---|"]
    nl = c1["net_lead"]
    L.append(f"| C1 | net lead over {nl['best_rival_plain'] or 'not available'} (the six world-blind rivals) | "
             f"{_f(nl['lead'])} | 10 | {'pass' if nl['passed'] else 'fail'} |")
    s2 = nl.get("step2_lead") or {}
    L.append(f"| (reported) | step 2's own lead over its own five rivals ({s2.get('best_rival_plain') or 'not available'}) | "
             f"{_f(s2.get('lead'))} | 10 | {'pass' if s2.get('passed') else 'fail'} |")
    for sit in K.KEEPS_SITUATIONS:
        k = c1["keeps"][sit]
        L.append(f"| C1 | keeps the gain, {K.SITUATION_PLAIN[sit]} | {_f(k['difference'])} | -3 | "
                 f"{'pass' if k['passed'] else 'fail'} |")
    L.append(f"| C1 | beats the matched even pull | {_f(c1['even_pull']['difference'])} | 15 | "
             f"{'pass' if c1['even_pull']['passed'] else 'fail'} |")
    cm = (look.get("per_seed") or {}).get("certified_marks") or {}
    L.append(f"| (description) | C1's three marks on the {cm.get('n', 0)} certified seeds alone (lead {_f(cm.get('lead'))} "
             f"over {cm.get('best_plain') or 'not available'}, largest loss {_f(cm.get('loss'))}, even pull "
             f"{_f(cm.get('even_pull'))}) | | | "
             f"{'not available' if cm.get('passed') is None else ('pass' if cm.get('passed') else 'fail')} |")
    fixed = c2["fixed"]
    nc = c2["n_certified"]
    need = ("not fixed" if fixed is None else
            f"ahead: mean >= {fixed['net_margin']:g} and positive in >= {c2['positive_needed']} of {nc}; behind: mean <= "
            f"{fixed['c3_net_margin']:g} and positive in <= {fixed['c3_positive_seeds_max']} of {nc}")
    for twin, v in c2["twins"].items():
        res = v["reading"] or ("not read" if c2["not_read"] else "not fixed")
        L.append(f"| C2 / C3 | net across worlds, BCA's switched ramp trained alone minus {v['plain']} (positive in "
                 f"{v['positive']} of {nc} certified seeds; one-sided 95% lower bound {_f(v['lower_bound'])}) | "
                 f"{_f(v['difference'])} | {need} | {res} |")
    L += [""]
    L += ["## Claims", ""] + [f"- {c}: {v['status']}" + (f" ({v['smoke_outcome']})" if "smoke_outcome" in v else "")
                              + (f" (per twin: {v['twins']})" if v.get("twins") else "")
                              for c, v in look["claims"].items()] + [""]
    if c2["placeholders"]:
        L += ["Open items (the freeze refuses while any remains): " + ", ".join(c2["placeholders"]), ""]
    if look["failed_seeds"] or look["c1_only"] or look["refused"]:
        L += ["## Failed and refused seeds", ""]
        L += [f"- failed, seed {s}: {r}" for s, r in look["failed_seeds"].items()]
        L += [f"- C1's marks only, seed {s}: {r}" for s, r in look["c1_only"].items()]
        L += [f"- BCA refused, seed {s}: {v} (in C1 its arms and the controls matched to it take the no-caution "
              f"actor's numbers; left out of C2 and C3)" for s, v in look["refused"].items()] + [""]
    L += [f"Certified seeds (C2, C3 and the noise floor): {', '.join(str(s) for s in c2['certified']) or 'none'} "
          f"({nc}; at least {c2['min_certified']} needed).", ""]
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
    counts = ("S", "N", "K", "F", "NC", "K1", "K2", "K3", "KMIN", "A", "NSEEDS", "NMIN", "NMAX", "NREACH", "NSW",
              "N8MIN", "N8MAX")
    for name in ("S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "K", "F", "NC", "DC", "RC", "LC", "EC", "D1", "K1",
                 "LB1", "D2", "K2", "LB2", "D3", "K3", "LB3", "KMIN", "A", "P", "V", "NSEEDS", "SEEDS", "NMIN", "NMAX",
                 "NREACH", "NSW", "N8MIN", "N8MAX"):
        v = br.get(name)
        if name == "P":
            L.append(f"| P | {_pct(v)} |")
            continue
        kind = "count" if name in counts else "points"
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
    tw = rep.get("twins") or {}
    if tw:
        L += ["", "## C2 and C3's differences per twin, on the certified seeds (net across worlds; description "
                  "beyond the readings)", "",
              "| twin | against BCA trained alone (mean, positive, lower bound) | against the batch-trained obca_k1 | "
              "over every seed |", "|---|---|---|---|"]
        for t_, v in tw.items():
            a_, b_, e_ = v["against_alone"], v["against_batch"], v["over_every_seed"]
            L.append(f"| {K.ARM_PLAIN.get(t_, t_)} | {_f(a_['mean'])}, {a_['positive']} of {a_['n']}, {_f(a_['lower_bound'])} "
                     f"| {_f(b_['mean'])}, {b_['positive']} of {b_['n']}, {_f(b_['lower_bound'])} "
                     f"| {_f(e_['mean'])}, {e_['positive']} of {e_['n']} |")
    comp = rep.get("composites") or {}
    if comp:
        L += ["", "## Reported composites (assembled from trained arms; description only)", ""]
        L += [f"- {name.replace('_', ' ')}: net across worlds {_f(v['net'])} (per seed "
              f"{', '.join(_f(x) for x in v['per_seed'])})" for name, v in comp.items()]
    sw = rep.get("switches") or {}
    if sw:
        L += ["", "## BCA's switch on the pooled groups", "",
              "| world | seed | on | height | plain 90% gap | full-throttle rows | rest | finite share | chance of a "
              "finite height | refused | rank of BCA's threshold / of the plain 90% point (from the top, of n) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for w in K.WORLDS:
            for s, r in sorted((sw.get(w) or {}).items(), key=lambda x: int(x[0])):
                rk = r.get("ranks") or {}
                L.append(f"| {K.WORLD_PLAIN[w]} | {s} | {r.get('on')} | {_f(r.get('height'), 'value')} | "
                         f"{_f((r.get('heights') or {}).get('plain_gap'), 'value')} | "
                         f"{_f(r.get('n_full_throttle'), 'count')} | {_f(r.get('n_rest'), 'count')} | "
                         f"{_f(r.get('finite_share_tail'), 'value')} | {_pct(r.get('certify_chance'))} | "
                         f"{'yes' if r.get('refused') else 'no'} | "
                         f"{_f(rk.get('bca_threshold'), 'count')} / {_f(rk.get('plain_q90'), 'count')} of "
                         f"{_f(rk.get('n'), 'count')} |")
    calls = rep.get("switch_calls") or {}
    if calls.get("rows"):
        ag = calls["agree"]
        L += ["", f"## The switch calls ({calls['seed_worlds']} seed-worlds; made BCA's call: the mean difference "
                  f"{ag['meandiff']}, the split conformal {ag['conformal']}, the rank test {ag['rank']})", "",
              "| seed | world | BCA | mean difference (gap) | split conformal (gap) | rank test (p) |",
              "|---|---|---|---|---|---|"]
        for r in calls["rows"]:
            if not r.get("defined"):
                L.append(f"| {r['seed']} | {K.WORLD_PLAIN.get(r['world'], r['world'])} | the gap rule is not defined | | | |")
                continue
            L.append(f"| {r['seed']} | {K.WORLD_PLAIN.get(r['world'], r['world'])} | {r['bca']} | {r['meandiff']} "
                     f"({_f(r['mean_gap'], 'value')}) | {r['conformal']} ({_f(r['plain_gap'], 'value')}) | {r['rank']} "
                     f"({_f(r['rank_p'], 'value')}) |")
    ladder = rep.get("dose_ladder") or {}
    if ladder:
        L += ["", "## The dose ladder (stalling world, strong current, points by height)", "",
              "| seed | " + " | ".join(f"{a}" for a in LADDER) + " |", "|---|" + "---|" * len(LADDER)]
        for s, rows in ladder.items():
            L.append(f"| {s} | " + " | ".join(f"{_f(r['points'])} (h {_f(r['height'], 'value')})" for r in rows) + " |")
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
