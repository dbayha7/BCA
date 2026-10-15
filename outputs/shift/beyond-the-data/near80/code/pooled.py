"""Near-80 (version 8): the looks. The exploratory look on seeds 0-4 (the stop rules of section 8.3, every mark
computed and labeled "exploratory, not a claim"), the confirmatory look on seeds 10-14 (the validity conditions, then
claims 1 to 4), and the halt report. Each fills in the sentences of section 14 of PREREGISTRATION.md exactly as its
section 14.6 says.

Inputs (written by the runner's phases, each checked against its .sha256 sidecar; in official mode every record must
be official and carry the approved fingerprint of prereg.json): the data records (data_main, data_band, the fixed
checks), BCA's switch records, the values of every setting, world and seed, the readouts, the policy locks (for the
reported strengths and arm tables), the stop and skip records, the stage lock (its plan names the seeds), the runner's
status, the ledger's attempt records (the resource table), the plumbing record (the halt report) and, for the
confirmatory look, the exploratory look.

The numbers (section 14's bracket table; every one pooled, that is the mean over the group's seeds of each seed's
paired difference, stalling world, strong current, near-80 setting, unless stated):
  S, N     seeds whose near-80 switch is on in the stalling world, off in the no-stall world (whole numbers)
  X, H, W  BCA's switched ramp minus no caution; the full-log actor minus no caution; no-stall: no caution minus BCA
  NET, D, R  step 2's net for BCA; NET minus the net of the best rule that cannot switch itself off; that rule's name
             (ties: the first in step 2's order). Step 2's own functions (step2lib.net, paired, go_no_go) compute them.
  E, L     BCA minus the matched even pull; BCA's loss to no caution in recorded conditions, big waves, big kicks
  G, O     BCA's unseen-throttle gain (as trained minus held to 70%) and overshoot loss (capped at 80% minus as trained)
  G0/O0, GZ/OZ, GV/OV  the same for no caution, the hand-set range limit, version 6's BCA
  Q        the mean over seeds of the textbook rule clipped to the near-80 recorded range (strong current, points)
  A, B, C, K  band costs (random rows minus near-80) of BCA and the hand-set range limit; pooled B - A; seeds with
              B - A > 0
  M        the group's seeds where check 10 did not hold
  P        no caution's share of strong-current steps past 80% (written as a percentage)
  V, arm   the larger of |random rows - main| for the hand-set range limit and BCA (seeds 0-4) and that arm's name
  dB, dR, d0  near-80 minus main for BCA, the hand-set range limit and no caution, on the same seeds

Failed seeds. A seed that failed its data (data_main: every claim; data_band checks 1-8: claims 1-3) or had a phase
stop (near-80, random rows: claims 1-3; main: claim 4; the main data and the full-log critic: every claim) fails every
mark that needs it: its evaluation numbers are set to null for those settings before any mark is computed, so the
pooled numbers of the marks are "not available", and the seed's own clause ("seed [s] failed [check] ...") tells it.
A seed with no failure or stop record whose evaluation lacks a number that the validity conditions or claims 2 and 3
read from an actor that is always trained (EVALUATION_NEEDS; its learner failed, section 10) fails for claims 1-3 in
the same way, with [check] "its evaluation (numbers missing)". The validity conditions (section 9.1) are computed over
the group's seeds that did not fail: a failed seed never makes claim 1 "invalid". When a seed failed, claims 2 and 3
are written with their "If a seed failed" sentences (section 14.6), never with pooled numbers that the failed seed made
"not available". Check 10's count M leaves out a failed seed that has no check-10 record. The exploratory stop rules
(section 8.3) need every seed 0-4: when a number the harm or the rival rule needs is missing on any seed, the near-80
and random-rows settings stop before the fresh seeds with section 14.5's sentence "A stop rule cannot be read"; when a
number the noise floor needs is missing, the noise floor cannot be read and claim 3 is dropped with the "not measured"
form of its dropped sentence.

Outputs (write-once): looks/<group>.md (the sentences first, then the tables) and looks/<group>.json (+ .sha256);
the runner reads its kind, prereg_sha256, group and, for the exploratory look, fresh_near80_allowed.

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
MEASURES = ("points", "held70_points", "capped80_points", "unseen_gain", "overshoot_loss", "past80_share",
            "zone_shares", "lower_tail_points")
SHARED_ARMS = tuple(a for a in K.ARMS["random_rows"] if a in K.ARMS["near80"])
NOISE_ARMS = ("range_hinge", K.PRIMARY)          # section 8.3's order: the hand-set range limit, BCA's switched ramp
CLAIMS = ("claim1", "claim2", "claim3", "claim4")
EXPLORATORY_STATUS = "exploratory, not a claim"
LOOK_KIND = {"exploratory": "near80-look-exploratory", "confirmatory": "near80-look-confirmatory"}
HALT_KIND = "near80-halt-report"
GO_RULE = dict(g1_margin=K.MARKS["net_lead_margin"], g2_floor=K.MARKS["keeps_floor"],
               g3_margin=K.MARKS["even_pull_margin"])
FIRST_HARM_FORM = "lost only [H]"               # where the harm stop's first form starts (section 14.5)
SEEDS_WORDS = "seed [seeds]"                    # written "seed 2", "seeds 2 and 3", "seeds 1, 2 and 3" (section 14.6)
EVALUATION_MISSING_CHECK = "its evaluation (numbers missing)"
# The numbers the validity conditions (harm, past-80 share, the range limit's band cost) and claims 2 and 3 (BCA's
# gain, overshoot and band cost) read from the four actors that are always trained: (measure, setting, arm), stalling
# world, strong current. "band" is the band cost's reference setting: random rows, or main without that control.
EVALUATION_NEEDS = (("points", "near80", "full_none"), ("points", "near80", "none"), ("past80_share", "near80", "none"),
                    ("points", "near80", "range_hinge"), ("points", "band", "range_hinge"),
                    ("points", "near80", "obca_k1"), ("points", "band", "obca_k1"),
                    ("unseen_gain", "near80", "obca_k1"), ("overshoot_loss", "near80", "obca_k1"))
REPRO_ORDER = ("removal", "critic", "clone", "none", "w0_check", "obca_k1", "range_hinge")
REPRO_PLAIN = {"removal": "the main test's removal", "critic": "the main test's critic",
               "clone": "the main test's cloned actor", "none": "the main test's no-caution actor",
               "w0_check": "step 2's plumbing-check actor", "obca_k1": "step 2's switched-ramp actor",
               "critic_second": "step 2's switched-ramp actor", "switch": "step 2's switched-ramp actor",
               "range_hinge": "step 2's hand-set range-limit actor"}
SETTING_PLAIN = {"main": "main setting", "near80": "near-80 setting", "random_rows": "random-rows setting",
                 "band": "near-80 and random-rows settings"}

# Section 8.1's checks, in its order: (the rule name's start in near80lib.no_training_checks, [check] as the sentence
# writes it, [threshold] as section 8.1 writes it, how [value] is written).
NO_TRAINING_CHECKS = (
    ("check 1: the band is really missing", "check 1, the leak check of the band", "all pass", "flag"),
    ("check 1: the random-rows control is matched",
     "check 1, the random-rows control's match in total and in each kind of water", "all pass", "flag"),
    ("check 2:", "check 2, the band rows left in the evaluation split", "at least 300", "number"),
    ("check 3:", "check 3, the extra rows as a share of the ball-kept rows", "1% to 6%", "percent"),
    ("check 4: the gate's world checks", "check 4, the gate's five world checks on the ball alone", "all pass",
     "marks"),
    ("check 4: full-throttle commands removed", "check 4, the share of full-throttle commands removed in the near-80 "
     "setting", "the gate's mark", "number"),
    ("check 4: kept rows past 80%", "check 4, the share of kept rows past 80% in the near-80 setting",
     "the gate's mark", "number"),
    ("check 4: identical-rule share", "check 4, the same-behavior share in the near-80 setting", "the gate's mark",
     "number"),
    ("check 4: hard-water rows kept", "check 4, the hard-water rows kept in the near-80 setting", "at least 35%",
     "percent"),
    ("check 5:", "check 5, the band rows among the ball-kept hard-water rows", "at least 10%", "percent"),
    ("check 6: BCA's gap rule is defined, lower end", "check 6, BCA's gap rule in the near-80 setting",
     "defined, lower end in [0.695, 0.70]", "gap"),
    ("check 6: BCA's gap rule is defined in the random-rows", "check 6, BCA's gap rule in the random-rows setting",
     "defined", "defined"),
    ("check 7:", "check 7, the calibration bank rows kept", "at least 500", "number"),
    ("check 8:", "check 8, the empty band reproducing the gate's removal", "hashes equal", "hashes"),
    ("check 9: the stall-aware controller", "check 9, the stall-aware controller's share of strong-current steps with "
     "thruster 1 in the band", "at least 50% of steps", "percent"),
    ("check 9: holding the textbook rule", "check 9, the points the textbook rule loses when held to 70%",
     "at least 15 points", "number"),
    ("check 10:", "check 10, the points the rule clipped to the recorded range loses", "at least 10 points", "number"),
)


class LookRefused(Exception):
    """A look or a report cannot be made from these files; the message starts with 'refused: '."""


# --------------------------------------------------------------------------------------------- JSON (section 3)

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
    mod = types.ModuleType("near80_pooled_step2lib")
    mod.__file__ = str(path)
    mod.__near80_sha256__ = want
    exec(compile(data, str(path), "exec"), mod.__dict__)
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
    """What a look reads (build spec, section 6.1). m[measure][setting][world][seed][logical id][situation] holds the
    numbers with failed seeds set to null (for the marks); raw[...] the same without that rule (for the validity
    conditions, the stop rules and the reported tables)."""

    def __init__(self, group, seeds, mode, prereg_sha256, out_root, defaults=None):
        self.group, self.seeds, self.mode = group, [int(s) for s in seeds], mode
        self.prereg_sha256, self.out_root = prereg_sha256, Path(out_root)
        self.defaults = dict(K.DEFAULTS, **(defaults or {}))
        self.raw = {m: {st: {w: {s: {} for s in self.seeds} for w in K.WORLDS} for st in K.SETTINGS}
                    for m in MEASURES}
        self.m = None
        self.values, self.switches, self.locks, self.readouts = {}, {}, {}, {}
        self.data_main, self.data_band, self.fixed = {}, {}, None
        self.failed_seeds_claims_1_3, self.failed_seeds_claim4 = {}, {}
        self.failed_checks = {"claims_1_3": {}, "claim_4": {}}
        self.check10, self.q_points = {}, {}
        self.phase_stops, self.skipped, self.runner_skipped = [], {}, {}
        self.exploratory, self.exploratory_sha256 = None, None
        self.plumbing, self.attempts, self.notes = None, [], []
        self.plan, self.inputs = None, {}

    @property
    def pts(self):
        return self.m["points"]

    @property
    def unseen_gain(self):
        return self.m["unseen_gain"]

    @property
    def overshoot_loss(self):
        return self.m["overshoot_loss"]

    @property
    def past80_share(self):
        return self.m["past80_share"]

    def fail(self, seed, which, reason, check):
        book = self.failed_seeds_claims_1_3 if which == "claims_1_3" else self.failed_seeds_claim4
        if seed not in book:
            book[seed] = reason
            self.failed_checks[which][seed] = check

    def apply_failures(self):
        self.m = copy.deepcopy(self.raw)
        for which, settings in (("claims_1_3", ("near80", "random_rows")), ("claim_4", ("main",))):
            book = self.failed_seeds_claims_1_3 if which == "claims_1_3" else self.failed_seeds_claim4
            for s in book:
                if s not in self.seeds:
                    continue
                for measure in MEASURES:
                    for st in settings:
                        for w in K.WORLDS:
                            self.m[measure][st][w][s] = {a: None for a in K.ARMS[st]}

    @property
    def ran(self):
        near = True
        if self.group == "confirmatory":
            near = bool((self.exploratory or {}).get("fresh_near80_allowed") is True)
        main = True if self.group == "exploratory" else bool(self.defaults.get("main_on_fresh_seeds", True))
        return dict(near80=near, main=main, both=bool(near and main))


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


def check_index(rule_name):
    """(position in section 8.1's order, entry) of a rule of near80lib.no_training_checks."""
    name = str(rule_name or "")
    for i, entry in enumerate(NO_TRAINING_CHECKS):
        if name.startswith(entry[0]):
            return i, entry
    m = re.match(r"check (\d+)", name)
    return (100 + int(m.group(1)) if m else 1000), None


def check_number(rule_name):
    m = re.match(r"check (\d+)", str(rule_name or ""))
    return int(m.group(1)) if m else None


def check_name(rule_name):
    """[check]: the plain name of a rule of section 8.1."""
    _, entry = check_index(rule_name)
    if entry is not None:
        return entry[1]
    name = str(rule_name or "a no-training check")
    return name.replace(":", ",", 1) if re.match(r"check \d+:", name) else name


def _threshold_words(text):
    t = str(text or "").strip()
    if t.startswith(">="):
        return "at least " + t[2:].strip()
    if t.startswith("<="):
        return "at most " + t[2:].strip()
    return t


def check_threshold(rule):
    """[threshold] as section 8.1 writes it (the gate's marks also name their number)."""
    _, entry = check_index(rule.get("rule"))
    if entry is None:
        return _threshold_words(rule.get("threshold"))
    if entry[2] == "the gate's mark":
        return f"the gate's mark, {_threshold_words(rule.get('threshold'))}"
    return entry[2]


def check_value(rule):
    """[value] with four significant digits (a share as a percentage where section 8.1 writes its threshold as one)."""
    _, entry = check_index(rule.get("rule"))
    how = entry[3] if entry is not None else "number"
    v = rule.get("value")
    if how == "marks" and isinstance(v, dict):
        bad = [k for k, ok in v.items() if ok is False]
        return (CT.join_items(bad) + " failed") if bad else "all passed"
    if how == "gap":
        return "not defined" if v is None or isinstance(v, bool) else CT.fmt(v, "value")
    if isinstance(v, bool) or how in ("flag", "defined", "hashes"):
        if how == "defined":
            return "defined" if v else "not defined"
        if how == "hashes":
            return "hashes equal" if v else "hashes differ"
        return "passed" if v else "failed"
    if v is None or isinstance(v, (dict, list, str)):
        return CT.NOT_AVAILABLE if v is None else str(v)
    if how == "percent":
        s = CT.fmt(float(v) * 100.0, "value")
        return s if s == CT.NOT_AVAILABLE else s + "%"
    return CT.fmt(v, "value")


def first_failed_rule(rules, numbers=None):
    """The first failed rule in section 8.1's order (ties: the order given), or None. numbers: only these checks."""
    failed = [(check_index(r.get("rule"))[0], i, r) for i, r in enumerate(rules or [])
              if not r.get("passed") and (numbers is None or check_number(r.get("rule")) in numbers)]
    return min(failed, key=lambda x: (x[0], x[1]))[2] if failed else None


def data_band_check(rec):
    """[check] for a seed whose near-80 data failed a check numbered 1-8."""
    first = first_failed_rule(rec.get("rules"), numbers=range(1, 9)) or rec.get("first_failed_check")
    return check_name((first or {}).get("rule")) if first else "a no-training check"


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


def _stop_affects(stop):
    """(fails claims 1-3, fails claim 4) for a seed-level stop."""
    kind, setting = stop.get("phase_kind"), stop.get("setting")
    if kind == "data_main" or kind == "critic_full" or setting == "x":
        return True, True
    if setting in ("near80", "random_rows", "band"):
        return True, False
    if setting == "main":
        return False, True
    return True, True


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
        # (each is told by its own clause), rather than the validity conditions failing on numbers never measured
        if summary.get("phase_kind") == "lock" and summary.get("group") == R.group:
            for s in R.seeds:
                R.fail(s, "claims_1_3", why, stopped_phase_check(summary))
                R.fail(s, "claim_4", why, stopped_phase_check(summary))
        return
    seed = int(seed)
    a13, a4 = _stop_affects(summary)
    if a13:
        R.fail(seed, "claims_1_3", why, stopped_phase_check(summary))
    if a4:
        R.fail(seed, "claim_4", why, stopped_phase_check(summary))


def _phase_order(x):
    kind = x.get("phase_kind")
    return (-1 if x.get("seed") is None else int(x["seed"]),
            K.PHASE_KINDS.index(kind) if kind in K.PHASE_KINDS else 99, str(x.get("setting")),
            K.WORLDS.index(x["world"]) if x.get("world") in K.WORLDS else 9)


def load_results(out_root, group, mode="official", prereg_sha256=None, seeds=None, defaults=None):
    """Every input of a look (module docstring), checked. Official mode: out_root must be the official output folder,
    every record official and of the approved pre-registration (prereg_sha256; default the SHA-256 of prereg.json)."""
    if group not in K.GROUPS:
        raise LookRefused(f"refused: unknown group {group!r}")
    if mode not in MODES:
        raise LookRefused(f"refused: unknown mode {mode!r}")
    root = Path(out_root)
    if mode == "official":
        if os.path.realpath(str(root)) != os.path.realpath(str(K.OUT)):
            raise LookRefused(f"refused: the official looks read {K.OUT} only (got {root})")
        if not prereg_sha256:
            prereg_sha256 = official_prereg()[1]
    rd = Reader(root, mode, prereg_sha256)
    stage = rd.read(K.LAYOUT["stage_lock"].format(group=group), kinds=("near80-stage-lock",))
    plan = (stage or {}).get("plan")
    if mode == "official" and plan not in (None, "full"):
        raise LookRefused(f"refused: the official {group} lock names the plan {plan!r}, not the full plan")
    R = Results(group, resolve_seeds(group, seeds, plan, mode), mode, prereg_sha256, root, defaults)
    R.plan = plan
    stage_locks = (stage or {}).get("locks") or {}

    for s in R.seeds:
        for w in K.WORLDS:
            dm = rd.read(K.LAYOUT["data_main"].format(world=w, seed=s), kinds=("near80-data-main",))
            R.data_main[(s, w)] = dm
            if dm and dm.get("seed_failed_reason"):
                why = f"the main data failed in the {K.WORLD_PLAIN[w]}: {dm['seed_failed_reason']}"
                R.fail(s, "claims_1_3", why, data_main_check(dm))
                R.fail(s, "claim_4", why, data_main_check(dm))
        for w in K.WORLDS:
            db = rd.read(K.LAYOUT["data_band"].format(world=w, seed=s), kinds=("near80-data-band",))
            R.data_band[(s, w)] = db
            if db is None:
                continue
            if db.get("seed_failed_reason"):
                R.fail(s, "claims_1_3", f"the near-80 data failed in the {K.WORLD_PLAIN[w]}: "
                                        f"{db['seed_failed_reason']}", data_band_check(db))
            if w == "stall":
                R.check10[s] = db.get("check10")
                pts = ((db.get("data_edge_points") or {}).get("near80") or {})
                R.q_points[s] = pts.get(SCORED) if isinstance(pts, dict) else None
    if group == "exploratory":
        R.fixed = rd.read(K.LAYOUT["fixed"], kinds=("near80-fixed-checks",))

    for setting in K.SETTINGS:
        for w in K.WORLDS:
            for s in R.seeds:
                kw = dict(setting=setting, world=w, seed=s)
                sw = rd.read(K.LAYOUT["switch"].format(**kw), kinds=("near80-switch",))
                if sw is not None:
                    R.switches[(setting, w, s)] = sw
                lock = rd.read(K.LAYOUT["policy_lock"].format(**kw), kinds=("near80-policy-lock",))
                if lock is not None:
                    R.locks[(setting, w, s)] = lock
                if setting != "main":
                    ro = rd.read(K.LAYOUT["readouts"].format(**kw), kinds=("near80-readouts",))
                    if ro is not None:
                        R.readouts[(setting, w, s)] = ro
                rel = K.LAYOUT["values"].format(**kw)
                v = rd.read(rel, kinds=("near80-values",))
                if v is None:
                    continue
                if (v.get("setting"), v.get("world"), v.get("seed")) != (setting, w, s):
                    raise LookRefused(f"refused: {rel} holds the values of another setting, world or seed")
                pid = K.phase_id("policy", setting, w, s)
                want = (stage_locks.get(pid) or {}).get("lock_sha256")
                if v.get("source") == "trained" and stage is not None and want != v.get("lock_sha256"):
                    raise LookRefused(f"refused: {rel} is not the evaluation of the lock the {group} lock lists")
                if v.get("source") == "step2_saved" and stage is not None:
                    saved = ((stage.get("step2_saved") or {}).get(f"{w}/seed{s}") or {}).get("lock_sha256")
                    if saved is not None and saved != v.get("lock_sha256"):
                        raise LookRefused(f"refused: {rel} is not the evaluation of step 2's saved lock")
                R.values[(setting, w, s)] = v
                ids = v.get("arm_ids") or {}
                sits = v.get("situations") or {}
                for logical in K.ARMS[setting]:
                    name = ids.get(logical)
                    for measure in MEASURES:
                        R.raw[measure][setting][w][s][logical] = None if name is None else {
                            sit: ((sits.get(sit) or {}).get(measure) or {}).get(name) for sit in K.SITUATIONS}

    stops_dir = root / "stops"
    if stops_dir.is_dir():
        for p in sorted(stops_dir.glob("*.json")):
            rel = p.relative_to(root)
            st = rd.read(rel, kinds=("near80-phase-stop",), keep=False)
            if st.get("group") == group or (st.get("seed") is not None and int(st["seed"]) in R.seeds):
                rd.keep(rel)
                _add_stop(R, st, rel)
    for setting in K.SETTINGS:
        for w in K.WORLDS:
            for s in R.seeds:
                rel = K.LAYOUT["policy_stop"].format(setting=setting, world=w, seed=s)
                ps = rd.read(rel, kinds=("near80-policy-stop",))
                if ps is not None:
                    _add_stop(R, dict(ps, phase_id=K.phase_id("policy", setting, w, s), phase_kind="policy",
                                      group=group), rel)
    R.phase_stops.sort(key=_phase_order)

    skipped_dir = root / "skipped"
    if skipped_dir.is_dir():
        for p in sorted(skipped_dir.glob("*.json")):
            rel = p.relative_to(root)
            sk = rd.read(rel, kinds=("near80-phase-skipped",), strict=False, keep=False)
            if sk.get("group") == group:
                rd.keep(rel)
                R.skipped[sk.get("phase_id")] = sk.get("reason")
    try:
        status, _ = read_json_verified(root / K.LAYOUT["status"])
        tail = "exploratory" if group == "exploratory" else "confirmatory"
        for pid, note in (status.get("skipped") or {}).items():
            last = str(pid).rsplit(".", 1)[-1]
            seed = int(last[1:]) if last.startswith("s") and last[1:].isdigit() else None
            if (seed is not None and seed in R.seeds) or last == tail:
                R.runner_skipped[pid] = note
    except LookRefused as error:
        R.notes.append(f"the runner's status was not read ({error}); it is reported only")

    if group == "confirmatory":
        rel = K.LAYOUT["look"].format(group="exploratory")
        R.exploratory = rd.read(rel, kinds=(LOOK_KIND["exploratory"],), required=True)
        R.exploratory_sha256 = rd.inputs[str(rel)]
    pl = rd.read(K.LAYOUT["plumbing"], kinds=("near80-plumbing",), strict=False)
    R.plumbing = pl

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
    for s, gone in evaluation_gaps(R).items():
        R.fail(s, "claims_1_3", "numbers missing from its evaluation, with no failure or stop record: "
                                + ", ".join(gone), EVALUATION_MISSING_CHECK)
    R.apply_failures()
    return R


def evaluation_gaps(R):
    """{seed: [what is missing]} for the group's seeds that have no failure record for claims 1-3 yet lack a number of
    EVALUATION_NEEDS (section 14.6): such a seed's learner failed, so it fails every mark of claims 1-3 (section 10).
    Empty when the near-80 setting did not run on the group's seeds."""
    if not R.ran["near80"]:
        return {}
    band = "random_rows" if R.defaults.get("random_rows_control", True) else "main"
    out = {}
    for s in R.seeds:
        if s in R.failed_seeds_claims_1_3:
            continue
        gone = []
        for measure, setting, arm in EVALUATION_NEEDS:
            st = band if setting == "band" else setting
            row = R.raw[measure][st]["stall"][s].get(arm)
            if row is None or row.get(SCORED) is None:
                gone.append(f"{arm} {measure} ({st})")
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


def setting_diff(m, a_setting, b_setting, world, arm, seeds, sit=SCORED):
    """Per seed: arm's points in setting a minus its points in setting b; pooled mean; seeds positive."""
    per = [_diff(x, y) for x, y in zip(_per(m, a_setting, world, arm, seeds, sit),
                                       _per(m, b_setting, world, arm, seeds, sit))]
    return dict(per_seed=per, mean=step2().mean_or_none(per), positive=sum(1 for v in per if v is not None and v > 0))


def band_cost(m, arm, seeds, world="stall", sit=SCORED, defaults=None):
    """Per seed: random rows minus near-80 (without the random-rows control: main minus near-80)."""
    control = bool((defaults or K.DEFAULTS).get("random_rows_control", True))
    return setting_diff(m, "random_rows" if control else "main", "near80", world, arm, seeds, sit)


def _count_switch(R, setting, world, want_on):
    n = 0
    for s in R.seeds:
        sw = R.switches.get((setting, world, s))
        on = ((sw or {}).get("certificate") or {}).get("on")
        if sw is not None and on is not None and bool(on) == want_on:
            n += 1
    return n


def _claim1_numbers(m, setting, seeds):
    S = step2()
    g = S.go_no_go(m[setting], seeds, GO_RULE)
    d1 = g["G1_detail"]
    net = d1["net_primary"]["net"]
    best_net = d1["best_world_blind_net"]
    lead = None if net is None or best_net is None else net - best_net
    keeps = {sit: g["G2_detail"][sit] for sit in K.KEEPS_SITUATIONS}
    loss = {sit: None if keeps[sit]["mean"] is None else -keeps[sit]["mean"] for sit in K.KEEPS_SITUATIONS}
    return g, net, best_net, lead, keeps, loss


def brackets(results, group=None, exploratory=None):
    """Every bracket of section 14's table (numbers; R and arm as plain names), and per-seed lists. Marks' numbers use
    the failed-seed rule; H, P and B here are the claim values (the validity and stop values are in validity())."""
    R = results
    group = group or R.group
    S = step2()
    m, seeds = R.m, R.seeds
    near = m["points"]["near80"]
    x = S.paired(near, "stall", K.PRIMARY, "none", seeds, SCORED)
    h = S.paired(near, "stall", "full_none", "none", seeds, SCORED)
    w = S.paired(near, "no_stall", "none", K.PRIMARY, seeds, SCORED)
    g, net, best_net, lead, keeps, loss = _claim1_numbers(m["points"], "near80", seeds)
    best = g["G1_detail"]["best_world_blind"]
    ev = g["G3_detail"]
    losses = list(loss.values())
    gain = {a: pooled_arm(m["unseen_gain"], "near80", "stall", a, seeds) for a in
            (K.PRIMARY, "none", "range_hinge", "version6_bca")}
    over = {a: pooled_arm(m["overshoot_loss"], "near80", "stall", a, seeds) for a in
            (K.PRIMARY, "none", "range_hinge", "version6_bca")}
    q = [R.q_points.get(s) for s in seeds]
    a_cost = band_cost(m["points"], K.PRIMARY, seeds, defaults=R.defaults)
    b_cost = band_cost(m["points"], "range_hinge", seeds, defaults=R.defaults)
    c_per = [_diff(b, a) for a, b in zip(a_cost["per_seed"], b_cost["per_seed"])]
    p = pooled_arm(m["past80_share"], "near80", "stall", "none", seeds)
    d = {a: setting_diff(m["points"], "near80", "main", "stall", a, seeds) for a in (K.PRIMARY, "range_hinge", "none")}
    v, arm = None, None
    if group == "exploratory":
        nf = noise_floor(R)
        v, arm = nf["V"], nf["arm"]
    elif exploratory is not None:
        v = (exploratory.get("brackets") or {}).get("V")
        arm = (exploratory.get("brackets") or {}).get("arm")
    c10 = check10_counts(R)
    br = dict(
        S=_count_switch(R, "near80", "stall", True), N=_count_switch(R, "near80", "no_stall", False),
        X=x["mean"], H=h["mean"], W=w["mean"], NET=net, D=lead, R=None if best is None else K.ARM_PLAIN[best],
        R_id=best, R_net=best_net, E=ev["mean"],
        L=None if any(v_ is None for v_ in losses) else max(losses), L_by_situation=loss,
        G=gain[K.PRIMARY]["mean"], O=over[K.PRIMARY]["mean"], G0=gain["none"]["mean"], O0=over["none"]["mean"],
        GZ=gain["range_hinge"]["mean"], OZ=over["range_hinge"]["mean"], GV=gain["version6_bca"]["mean"],
        OV=over["version6_bca"]["mean"], Q=S.mean_or_none(q), A=a_cost["mean"], B=b_cost["mean"],
        C=S.mean_or_none(c_per), K=sum(1 for c in c_per if c is not None and c > 0), M=c10["M"], P=p["mean"],
        V=v, arm=arm, dB=d[K.PRIMARY]["mean"], dR=d["range_hinge"]["mean"], d0=d["none"]["mean"])
    per_seed = dict(X=x["per_seed"], H=h["per_seed"], W=w["per_seed"], NET=g["G1_detail"]["net_primary"]["per_seed"],
                    E=ev["per_seed"], G=gain[K.PRIMARY]["per_seed"], O=over[K.PRIMARY]["per_seed"],
                    A=a_cost["per_seed"], B=b_cost["per_seed"], C=c_per, Q=q, P=p["per_seed"],
                    dB=d[K.PRIMARY]["per_seed"], dR=d["range_hinge"]["per_seed"], d0=d["none"]["per_seed"],
                    L={sit: [None if v_ is None else -v_ for v_ in keeps[sit]["per_seed"]] for sit in keeps},
                    check10={str(s): R.check10.get(s) for s in seeds}, seeds=list(seeds))
    return dict(br, per_seed=per_seed)


def seeds_not_failed(R):
    """The group's seeds that did not fail for claims 1-3 (a failed seed is told by its own clause)."""
    return [s for s in R.seeds if s not in R.failed_seeds_claims_1_3]


def check10_counts(R):
    """Check 10 per seed (section 9.1). M counts the seeds whose check-10 record exists and failed, plus any seed that
    did not fail for claims 1-3 yet has no check-10 record (it cannot be shown to hold). A seed that failed for claims
    1-3 and has no check-10 record (its near-80 data were never built or stopped) is not counted: by the pre-set rule
    it fails every mark, and its own clause says so. The condition holds when M is at most the number of the group's
    seeds minus 4 (at most 1 of 5)."""
    held = [s for s in R.seeds if (R.check10.get(s) or {}).get("passed") is True]
    failed = [s for s in R.seeds if (R.check10.get(s) or {}).get("passed") is False]
    unknown = [s for s in R.seeds if s not in held and s not in failed and s not in R.failed_seeds_claims_1_3]
    return dict(passed=len(held), M=len(failed) + len(unknown), failed_seeds=failed, unknown_seeds=unknown,
                per_seed={str(s): R.check10.get(s) for s in R.seeds})


def noise_floor(R):
    """Section 8.3's noise floor on the group's seeds (raw numbers): |pooled random rows - main| for the hand-set
    range limit and BCA's switched ramp, both at most 5; V the larger and its arm (ties: the first); no caution's
    difference reported only. Not run without the random-rows control. When a number it needs is missing on a seed,
    it cannot be read: unmeasured is true, passed is None, V is None and missing_seeds lists those seeds (claim 3 is
    then dropped with the "not measured" sentence, section 14.6)."""
    control = bool(R.defaults.get("random_rows_control", True))
    diffs = {a: setting_diff(R.raw["points"], "random_rows", "main", "stall", a, R.seeds)
             for a in NOISE_ARMS + ("none",)}
    if not control:
        return dict(run=False, passed=None, V=None, arm=None, differences=diffs, unmeasured=False, missing_seeds=[],
                    max_allowed=K.MARKS["noise_floor_max"])
    sizes = [(None if diffs[a]["mean"] is None else abs(diffs[a]["mean"]), a) for a in NOISE_ARMS]
    missing = [a for v, a in sizes if v is None]
    gone = sorted({s for a in NOISE_ARMS for s, x in zip(R.seeds, diffs[a]["per_seed"]) if x is None})
    if missing:
        v, arm_id, passed = None, missing[0], None
    else:
        v = max(x for x, _ in sizes)
        arm_id = next(a for x, a in sizes if x == v)
        passed = bool(all(x <= K.MARKS["noise_floor_max"] for x, _ in sizes))
    return dict(run=True, passed=passed, V=v, arm_id=arm_id, arm=K.ARM_PLAIN[arm_id], differences=diffs,
                unmeasured=bool(missing), missing_seeds=gone if missing else [],
                max_allowed=K.MARKS["noise_floor_max"])


def conditions(R, seeds):
    """The harm, the past-80 share and the hand-set range limit's band cost, pooled over `seeds` (raw numbers):
    {harm, past80, rival}, each with value, per_seed, its minimum and passed (a missing number fails it)."""
    S = step2()
    raw = R.raw["points"]["near80"]
    h = S.paired(raw, "stall", "full_none", "none", seeds, SCORED)
    p = pooled_arm(R.raw["past80_share"], "near80", "stall", "none", seeds)
    b = band_cost(R.raw["points"], "range_hinge", seeds, defaults=R.defaults)
    harm = dict(value=h["mean"], per_seed=h["per_seed"], minimum=K.MARKS["harm_min"],
                passed=bool(h["mean"] is not None and h["mean"] >= K.MARKS["harm_min"]))
    past = dict(value=p["mean"], per_seed=p["per_seed"], minimum=K.MARKS["past80_min"],
                passed=bool(p["mean"] is not None and p["mean"] >= K.MARKS["past80_min"]))
    rival = dict(value=b["mean"], per_seed=b["per_seed"], minimum=K.MARKS["rival_band_cost_min"],
                 passed=bool(b["mean"] is not None and b["mean"] >= K.MARKS["rival_band_cost_min"]))
    return dict(harm=harm, past80=past, rival=rival)


def validity(results):
    """Section 9.1 (and section 14.6), on the group's seeds that did not fail for claims 1-3: a failed seed fails every
    mark of claims 1-3 and is told by its own clause, so it does not make the validity conditions fail. A condition
    fails when its number is below its minimum, or missing for a seed that did not fail. When every seed failed, the
    conditions cannot be read and do not stop the claim from being written as failed (each seed's clause says why)."""
    R = results
    ok = seeds_not_failed(R)
    c = conditions(R, ok)
    harm, past, rival = c["harm"], c["past80"], c["rival"]
    if not ok:
        for x in (harm, past, rival):
            x.update(passed=True, note="every seed failed for claims 1-3; each seed's own clause tells the outcome")
    for x in (harm, past, rival):
        x["seeds"] = list(ok)
    c10 = check10_counts(R)
    need = K.MARKS["check10_seeds_min"]
    return dict(
        claims_1_3=dict(harm=harm, past80=past, passed=bool(harm["passed"] and past["passed"]), seeds=list(ok),
                        failed_seeds=sorted(R.failed_seeds_claims_1_3)),
        claims_2_3=dict(check10=dict(c10, needed=need), passed=bool(c10["M"] <= len(R.seeds) - need)),
        claim3=dict(rival_band_cost=rival, passed=rival["passed"]),
        claim4=dict(passed=True, note="nothing from the near-80 setting (section 9.5)"))


def stop_rules(results):
    """Section 8.3's stop rules (exploratory group, seeds 0-4): the harm is there, the band changes the rival's problem
    (the hand-set range limit's band cost; without the random-rows control, the main setting minus the near-80
    setting), the noise floor. fresh_near80_allowed = harm and rival; BCA's own marks never stop the fresh seeds."""
    R = results
    if R.group != "exploratory":
        raise LookRefused("refused: the stop rules are read on the exploratory seeds 0-4 only")
    v = conditions(R, R.seeds)              # every seed is needed (section 8.3): no seed is left out here
    hv, pv = v["harm"], v["past80"]
    harm = dict(H=hv["value"], P=pv["value"], per_seed=dict(H=hv["per_seed"], P=pv["per_seed"]),
                harm_passed=hv["passed"], share_passed=pv["passed"], passed=bool(hv["passed"] and pv["passed"]),
                minimum_points=K.MARKS["harm_min"], minimum_share=K.MARKS["past80_min"])
    control = bool(R.defaults.get("random_rows_control", True))
    if control:
        b = v["rival"]
        rival = dict(B=b["value"], per_seed=b["per_seed"], uses="the hand-set range limit's band cost (random rows "
                     "minus near-80)", minimum=K.MARKS["rival_band_cost_min"], passed=b["passed"])
    else:
        d = setting_diff(R.raw["points"], "main", "near80", "stall", "range_hinge", R.seeds)
        rival = dict(B=d["mean"], per_seed=d["per_seed"], uses="the hand-set range limit, main setting minus near-80 "
                     "setting (no random-rows control)", minimum=K.MARKS["rival_band_cost_min"],
                     passed=bool(d["mean"] is not None and d["mean"] >= K.MARKS["rival_band_cost_min"]))
    # A stop rule whose number is missing on some seed (a phase stopped, or an actor's numbers are missing) cannot be
    # read: every seed is needed, so the near-80 and random-rows settings stop before the fresh seeds, with their own
    # sentence (section 14.5, "A stop rule cannot be read") instead of a "too little" sentence about a number that was
    # never measured.
    gone_h = [s for s, a, b in zip(R.seeds, hv["per_seed"], pv["per_seed"]) if a is None or b is None]
    gone_b = [s for s, x in zip(R.seeds, rival["per_seed"]) if x is None]
    what = (["the harm"] if gone_h else []) + (["the band's cost to the hand-set range limit"] if gone_b else [])
    unmeasured = None if not what else dict(what=what, seeds=sorted(set(gone_h) | set(gone_b)),
                                            harm_seeds=gone_h, rival_seeds=gone_b)
    nf = noise_floor(R)
    allowed = bool(unmeasured is None and harm["passed"] and rival["passed"])
    return dict(harm=harm, rival_unaffected=rival, unmeasured=unmeasured, noise_floor=nf,
                fresh_near80_allowed=allowed, claim3_dropped=bool(nf["run"] and nf["passed"] is not True),
                note="BCA's own result on seeds 0-4 never stops the fresh seeds (choice 9)")


def marks(results, setting="near80"):
    """Claims 1, 2 and 3 on the near-80 setting, or claim 4 (step 2's three go marks) on the main setting."""
    R = results
    m = R.m
    g, net, best_net, lead, keeps, loss = _claim1_numbers(m["points"], setting, R.seeds)
    best = g["G1_detail"]["best_world_blind"]
    nets = {a: g["G1_detail"]["net_world_blind"][a]["net"] for a in K.WORLD_BLIND}
    go = dict(
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
    if setting == "main":
        return {"claim4": go}
    gain = pooled_arm(m["unseen_gain"], "near80", "stall", K.PRIMARY, R.seeds)
    pos = sum(1 for v in gain["per_seed"] if v is not None and v > 0)
    over = pooled_arm(m["overshoot_loss"], "near80", "stall", K.PRIMARY, R.seeds)
    c2 = dict(gain=dict(value=gain["mean"], per_seed=gain["per_seed"], positive=pos, minimum=K.MARKS["gain_min"],
                        positive_needed=K.MARKS["gain_positive_seeds_min"],
                        passed=bool(gain["mean"] is not None and gain["mean"] >= K.MARKS["gain_min"]
                                    and pos >= K.MARKS["gain_positive_seeds_min"])),
              overshoot=dict(value=over["mean"], per_seed=over["per_seed"], maximum=K.MARKS["overshoot_max"],
                             passed=bool(over["mean"] is not None and over["mean"] <= K.MARKS["overshoot_max"])))
    a_cost = band_cost(m["points"], K.PRIMARY, R.seeds, defaults=R.defaults)
    b_cost = band_cost(m["points"], "range_hinge", R.seeds, defaults=R.defaults)
    c_per = [_diff(b, a) for a, b in zip(a_cost["per_seed"], b_cost["per_seed"])]
    c = step2().mean_or_none(c_per)
    k = sum(1 for v in c_per if v is not None and v > 0)
    c3 = dict(A=a_cost["mean"], B=b_cost["mean"], C=c, K=k,
              per_seed=dict(A=a_cost["per_seed"], B=b_cost["per_seed"], C=c_per),
              minimum=K.MARKS["band_cost_diff_min"], positive_needed=K.MARKS["band_cost_positive_seeds_min"],
              band_cost="random rows minus near-80" if R.defaults.get("random_rows_control", True)
              else "main minus near-80 (no random-rows control)",
              passed=bool(c is not None and c >= K.MARKS["band_cost_diff_min"]
                          and k >= K.MARKS["band_cost_positive_seeds_min"]))
    return {"claim1": go, "claim2": c2, "claim3": c3}


# --------------------------------------------------------------------------------------------- outcomes and sentences

def outcomes(look):
    """{claim: status, ...} and why claim 3 is not written ('outcome_reasons')."""
    ran, val, mk, group = look["ran"], look["validity"], look["marks"], look["group"]
    dropped = bool(look.get("claim3_dropped"))
    failed = bool((look.get("failed_checks") or {}).get("claims_1_3"))
    out, why = {}, {}
    if not ran["near80"]:
        out.update(claim1="not run", claim2="not run", claim3="not run")
        why["claim3"] = "the near-80 setting did not run on these seeds"
    else:
        valid = val["claims_1_3"]["passed"]
        c10 = val["claims_2_3"]["passed"]
        out["claim1"] = "invalid" if not valid else ("pass" if mk["claim1"]["passed"] else "fail")
        if not valid or not c10:
            out["claim2"] = "invalid"
        elif failed:
            out["claim2"] = "fail_failed_seed"
        else:
            out["claim2"] = "pass" if (mk["claim2"]["gain"]["passed"] and mk["claim2"]["overshoot"]["passed"]) \
                else "fail"
        if group == "exploratory" and dropped:
            out["claim3"], why["claim3"] = "dropped", "the noise floor on seeds 0-4"
        elif not valid:
            out["claim3"], why["claim3"] = "invalid", "claim 1's validity conditions"
        elif not c10:
            out["claim3"], why["claim3"] = "invalid", "check 10"
        elif dropped:
            out["claim3"], why["claim3"] = "dropped", "the noise floor on seeds 0-4"
        elif not val["claim3"]["passed"]:
            out["claim3"], why["claim3"] = "invalid", "the hand-set range limit's band cost"
        elif failed:
            out["claim3"] = "fail_failed_seed"
        else:
            out["claim3"] = "pass" if mk["claim3"]["passed"] else "fail"
    out["claim4"] = "not run" if not ran["main"] else ("pass" if mk["claim4"]["passed"] else "fail")
    return out, why


def _template_getter(templates, group):
    T = templates["sentences"]

    def t(sid):
        text = T[sid]["text"]
        if group == "exploratory" and sid.startswith("claim") and "seeds 0-4" not in text:
            text = text.replace("fresh seeds 10-14", "seeds 0-4").replace("fresh seeds", "seeds")
        return text
    return t


def _clauses(look, t, claim, which, covered=None):
    """[one clause per missed mark] (claim 1) or [clauses as in claim 1] (claim 4), in section 14.6's order. A missed
    mark whose number is not available because a seed failed is told by that seed's own clause."""
    mk = look["marks"][claim]
    failed = {int(s): c for s, c in (look["failed_checks"][which] or {}).items()}
    covered = bool(failed) if covered is None else covered
    out = []
    nl = mk["net_lead"]
    if not nl["passed"] and not (covered and (nl["net"] is None or nl["lead"] is None)):
        out.append(CT.fill(t("claim1.clause.net"), dict(NET=nl["net"], D=nl["lead"],
                                                        R=nl["best_rival_plain"] or CT.NOT_AVAILABLE)))
    for sit in K.KEEPS_SITUATIONS:
        k = mk["keeps"][sit]
        if not k["passed"] and not (covered and k["loss"] is None):
            out.append(CT.fill(t("claim1.clause.keeps"), dict(L=k["loss"], situation=K.SITUATION_PLAIN[sit])))
    ep = mk["even_pull"]
    if not ep["passed"] and not (covered and ep["difference"] is None):
        out.append(CT.fill(t("claim1.clause.even_pull"), dict(E=ep["difference"])))
    for s in sorted(failed):
        out.append(CT.fill(t("claim1.clause.failed_seed"), dict(s=CT.fmt(s, "count"), check=failed[s])))
    if not out and covered:
        return _clauses(look, t, claim, which, covered=False)
    if not out:
        raise ValueError(f"{claim} failed but no mark was missed")
    return out


def harm_stop_sentence(template, H, P):
    """Section 14.5's harm stop with its "[or: ...]" written as section 14.6 says."""
    harm_ok = H is not None and H >= K.MARKS["harm_min"]
    share_ok = P is not None and P >= K.MARKS["past80_min"]
    if harm_ok and share_ok:
        raise ValueError("the harm stop does not apply (both the harm and the share pass)")
    i = template.index("[or: ")
    depth, j = 0, None
    for k in range(i, len(template)):
        if template[k] == "[":
            depth += 1
        elif template[k] == "]":
            depth -= 1
            if depth == 0:
                j = k
                break
    if j is None:
        raise ValueError("the harm stop's [or: ...] is not closed")
    second = template[i + len("[or: "):j]
    k = template.index(FIRST_HARM_FORM)
    first = template[k:i].rstrip()
    head, tail = template[:k], template[j + 1:]
    if not harm_ok and share_ok:
        body = first
    elif harm_ok and not share_ok:
        body = second
    else:
        body = first + " and " + second
    return CT.fill(head + body + tail, dict(H=H, P=CT.fmt(P, "percent")))


def phase_stop_sentence(template, stop, group=None):
    seed = stop.get("seed")
    if seed is None:
        seeds = group_seeds(stop.get("group") or group or "exploratory")
        s = f"{seeds[0]}-{seeds[-1]}"
    else:
        s = CT.fmt(int(seed), "count")
    world = K.WORLD_PLAIN.get(stop.get("world"), "both worlds")
    return CT.fill(template, dict(s=s, world=world, phase=phase_text(stop.get("phase_kind"), stop.get("setting")),
                                  reason=str(stop.get("reason") or "no reason recorded")))


def no_training_sentence(template, rule):
    return CT.fill(template, dict(check=check_name(rule.get("rule")), value=check_value(rule),
                                  threshold=check_threshold(rule)))


def reproduction_sentence(template, names):
    m = re.search(r"\[[^\[\]]* / [^\[\]]*\]", template)
    if m is None:
        raise ValueError("the reproduction stop has no list bracket")
    return CT.fill(template, {}, specials={m.group(0): CT.join_items(names)})


def _stop_after_look(stops):
    """The stop sentence id that the exploratory stop rules call for, or None (section 14.6's order: a stop rule that
    cannot be read, too little harm, the rival unaffected)."""
    if not stops:
        return None
    if stops.get("unmeasured"):
        return "stop.unmeasured"
    if not stops["harm"]["passed"]:
        return "stop.harm"
    if not stops["rival_unaffected"]["passed"]:
        return "stop.rival"
    return None


def seeds_words(seeds):
    """"seed [seeds]" as section 14.6 writes it: "seed 2", "seeds 2 and 3", "seeds 1, 2 and 3"."""
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return ("seed " if len(items) == 1 else "seeds ") + CT.join_items(items)


def fill_seeds(template, seeds, values=None):
    """A template whose "seed [seeds]" lists `seeds` (singular or plural by count); other brackets from `values`."""
    items = [CT.fmt(int(s), "count") for s in sorted(int(x) for x in seeds)]
    return CT.fill(template, dict(values or {}, seeds=CT.join_items(items)),
                   specials={SEEDS_WORDS: seeds_words(seeds)})


def unmeasured_sentence(template, unmeasured):
    """Section 14.5's "A stop rule cannot be read": [what] lists what could not be measured ("the harm", "the band's
    cost to the hand-set range limit", or both joined by " and "), "seed [seeds]" the seeds where a number is
    missing."""
    return fill_seeds(template, unmeasured["seeds"], dict(what=CT.join_items(unmeasured["what"])))


def failed_seeds_of(look):
    """The seeds that failed for claims 1-3, in order (claim 1's sentence tells why each failed)."""
    return sorted(int(s) for s in (look.get("failed_checks") or {}).get("claims_1_3") or {})


def noise_floor_of(look, group):
    """The noise floor record of the exploratory stop rules (this look's, or the confirmatory look's copy)."""
    st = look.get("stops") if group == "exploratory" else (look.get("exploratory_stops") or {}).get("stops")
    return (st or {}).get("noise_floor") or {}


def look_stop_rows(t, stops, choice6=True):
    """[{id, text}] of the stop the exploratory stop rules call for (and the claim-4 note after it), or []."""
    sid = _stop_after_look(stops)
    if sid is None:
        return []
    if sid == "stop.unmeasured":
        text = unmeasured_sentence(t(sid), stops["unmeasured"])
    elif sid == "stop.harm":
        text = harm_stop_sentence(t(sid), stops["harm"]["H"], stops["harm"]["P"])
    else:
        text = CT.fill(t(sid), dict(B=stops["rival_unaffected"]["B"]))
    rows = [dict(id=sid, text=text)]
    if choice6:
        rows.append(dict(id="stop.claim4_note", text=t("stop.claim4_note")))
    return rows


def sentences(look, templates, group):
    """The look's sentences, in order, as [{id, text}] (section 14 and its section 14.6)."""
    CT.check_templates(templates)
    t = _template_getter(templates, group)
    br, val = look["brackets"], look["validity"]
    out_c = look.get("outcomes") or outcomes(look)[0]
    why = look.get("outcome_reasons") or outcomes(look)[1]
    prefix = templates["exploratory_prefix"]
    rows = []

    def add(sid, text, claim=True):
        if group == "exploratory" and claim:
            text = f"{prefix} {text}"
        rows.append(dict(id=sid, text=text))

    if group == "confirmatory":
        add("always_attached", templates["always_attached"], claim=False)
    c1 = out_c["claim1"]
    if c1 == "pass":
        add("claim1.pass", CT.fill(t("claim1.pass"), dict(
            S=CT.fmt(br["S"], "count"), N=CT.fmt(br["N"], "count"), X=br["X"], H=br["H"], W=br["W"], NET=br["NET"],
            D=br["D"], R=br["R"] or CT.NOT_AVAILABLE, E=br["E"])))
    elif c1 == "fail":
        clauses = _clauses(look, t, "claim1", "claims_1_3")
        last = {"pass": "claim1.last.claim4_passed", "fail": "claim1.last.claim4_failed",
                "not run": "claim1.last.claim4_not_run"}[out_c["claim4"]]
        add("claim1.fail", CT.fill(t("claim1.fail"), {}, specials={
            "[one clause per missed mark]": "; ".join(clauses), "[last sentence]": t(last)}))
    elif c1 == "invalid":
        v13 = val["claims_1_3"]
        reasons = []
        if not v13["harm"]["passed"]:
            reasons.append(CT.fill(t("claim1.reason.harm"), dict(H=v13["harm"]["value"])))
        if not v13["past80"]["passed"]:
            reasons.append(CT.fill(t("claim1.reason.past80"), dict(P=CT.fmt(v13["past80"]["value"], "percent"))))
        add("claim1.invalid", CT.fill(t("claim1.invalid"), dict(reason="; ".join(reasons))))
    if c1 in ("pass", "fail"):
        if not val["claims_2_3"]["passed"]:
            add("claim2.check10", CT.fill(t("claim2.check10"), dict(M=CT.fmt(br["M"], "count"))))
        elif out_c["claim2"] == "fail_failed_seed":
            add("claim2.failed_seed", fill_seeds(t("claim2.failed_seed"), failed_seeds_of(look)))
        else:
            c2 = look["marks"]["claim2"]
            sid = "claim2.{}_{}".format("pass" if c2["gain"]["passed"] else "fail",
                                        "pass" if c2["overshoot"]["passed"] else "fail")
            add(sid, CT.fill(t(sid), dict(G=br["G"], O=br["O"])))
    if c1 != "not run":
        add("claim2.scale", CT.fill(t("claim2.scale"), dict(G0=br["G0"], O0=br["O0"], GZ=br["GZ"], OZ=br["OZ"],
                                                           GV=br["GV"], OV=br["OV"], Q=br["Q"])))
    c3 = out_c["claim3"]
    nf = noise_floor_of(look, group)
    if c3 == "dropped" and nf.get("unmeasured"):
        add("claim3.dropped_unmeasured", fill_seeds(t("claim3.dropped_unmeasured"), nf.get("missing_seeds") or []))
    elif c3 == "dropped":
        add("claim3.dropped", CT.fill(t("claim3.dropped"), dict(arm=br["arm"] or CT.NOT_AVAILABLE, V=br["V"])))
    elif c3 == "fail_failed_seed":
        add("claim3.failed_seed", fill_seeds(t("claim3.failed_seed"), failed_seeds_of(look)))
    elif c3 == "invalid" and why.get("claim3") == "the hand-set range limit's band cost":
        add("claim3.invalid", CT.fill(t("claim3.invalid"), dict(B=val["claim3"]["rival_band_cost"]["value"])))
    elif c3 in ("pass", "fail"):
        sid = "claim3.holds" if c3 == "pass" else "claim3.fails"
        add(sid, CT.fill(t(sid), dict(A=br["A"], B=br["B"], C=br["C"], K=CT.fmt(br["K"], "count"))))
    c4 = out_c["claim4"]
    if c4 == "pass":
        n4 = look["marks"]["claim4"]["net_lead"]
        add("claim4.holds", CT.fill(t("claim4.holds"), dict(NET=n4["net"], D=n4["lead"],
                                                           R=n4["best_rival_plain"] or CT.NOT_AVAILABLE)))
    elif c4 == "fail":
        add("claim4.fails", CT.fill(t("claim4.fails"), {}, specials={
            "[clauses as in claim 1]": "; ".join(_clauses(look, t, "claim4", "claim_4"))}))
    if c4 != "not run" and look["ran"]["both"]:
        add("claim4.both_settings", CT.fill(t("claim4.both_settings"), dict(dB=br["dB"], dR=br["dR"], d0=br["d0"])))
    stops = look.get("stops") if group == "exploratory" else (look.get("exploratory_stops") or {}).get("stops")
    if group == "exploratory" or not look["ran"]["near80"]:
        for r in look_stop_rows(t, stops, look.get("choice6_default", True)):
            add(r["id"], r["text"], claim=False)
    for st in look.get("phase_stops") or []:
        add("stop.phase", phase_stop_sentence(t("stop.phase"), st, group), claim=False)
    return rows


def _claim_of(sid):
    if sid.startswith("claim"):
        return sid.split(".")[0]
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
    """The reported tables (no mark reads them): points and the other measures per setting, world, arm and situation
    (every number that exists), band costs of the shared arms, nets, zone shares, switches and heights, arm tables,
    strengths and matches, readouts, the noise floor's differences, skipped and stopped phases, resources."""
    raw, seeds = R.raw, R.seeds
    S = step2()
    tables = {measure: {st: {w: {a: _table(raw, measure, st, w, a, seeds) for a in K.ARMS[st]
                                 if not (w != "stall" and a in K.STALL_ONLY_ARMS)}
                             for w in K.WORLDS} for st in K.SETTINGS}
              for measure in MEASURES if measure != "zone_shares"}
    zones = {st: {w: {a: _zone_means(raw, st, w, a, seeds) for a in K.ARMS[st]} for w in K.WORLDS}
             for st in K.SETTINGS}
    costs = {a: {w: {sit: band_cost(raw["points"], a, seeds, w, sit, R.defaults) for sit in K.SITUATIONS}
                 for w in K.WORLDS} for a in SHARED_ARMS}
    nets = {st: {a: S.net(raw["points"][st], a, seeds) for a in K.ARMS[st] if a not in ("none", "full_none")}
            for st in K.SETTINGS}
    switches = {}
    for (st, w, s), sw in sorted(R.switches.items()):
        c, g = sw.get("certificate") or {}, sw.get("groups") or {}
        switches.setdefault(st, {}).setdefault(w, {})[str(s)] = dict(
            defined=sw.get("defined"), on=c.get("on"), height=_num(c.get("height")), q05=_num(c.get("q05")),
            t_lo=_num(sw.get("t_lo")), t_up=_num(sw.get("t_up")), n_full_throttle=g.get("n_full_throttle"),
            n_rest=g.get("n_rest"), n_survivors=sw.get("n_survivors"))
    locks = {}
    for (st, w, s), lock in sorted(R.locks.items()):
        ids = lock.get("arm_ids") or {}
        strengths = lock.get("strengths") or {}
        locks.setdefault(st, {}).setdefault(w, {})[str(s)] = dict(
            arm_table=(lock.get("arm_table") or {}).get("arms"), t_lo=(lock.get("arm_table") or {}).get("t_lo"),
            t_up=(lock.get("arm_table") or {}).get("t_up"), arm_ids=ids,
            strengths={a: strengths.get(n) for a, n in ids.items() if n is not None and n in strengths},
            matched_names=lock.get("matched_names"),
            standin={k: (lock.get("standin") or {}).get(k) for k in ("arm", "coefficient", "error")},
            target_strength=lock.get("target_strength"), checks=lock.get("checks"))
    readouts = {}
    for (st, w, s), ro in sorted(R.readouts.items()):
        readouts.setdefault(st, {}).setdefault(w, {})[str(s)] = dict(
            band_rows=ro.get("band_rows"), band_rows_past_80=ro.get("band_rows_past_80"),
            critic_error=ro.get("critic_error"), version6_width=ro.get("version6_width"))
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
    data = dict(check10={str(s): R.check10.get(s) for s in seeds}, q_points={str(s): R.q_points.get(s) for s in seeds},
                main_failed={str(s): (rec or {}).get("seed_failed_reason") for (s, w), rec in R.data_main.items()
                             if (rec or {}).get("seed_failed_reason")},
                band_failed={f"{w}/seed{s}": (rec or {}).get("seed_failed_reason")
                             for (s, w), rec in R.data_band.items() if (rec or {}).get("seed_failed_reason")},
                fixed_checks=None if R.fixed is None else dict(passed=R.fixed.get("passed"), rules=R.fixed.get("rules")))
    return dict(measures=tables, zone_shares=zones, band_costs=costs, nets=nets, switches=switches, locks=locks,
                readouts=readouts, noise_floor_differences=noise_floor(R)["differences"],
                skipped=dict(R.skipped), runner_skipped=dict(R.runner_skipped), phase_stops=list(R.phase_stops),
                resources=dict(by_phase_kind=resources, cpu_hours_total=total_hours), data=data,
                notes=list(R.notes))


# --------------------------------------------------------------------------------------------- the looks

def _base_look(R, group):
    br = brackets(R, group, exploratory=R.exploratory)
    per_seed = br.pop("per_seed")
    mk = dict(marks(R, "near80"))
    mk.update(marks(R, "main"))
    return dict(
        kind=LOOK_KIND[group], mode=R.mode, prereg_sha256=R.prereg_sha256 if R.mode == "official" else None,
        phase_id=K.phase_id(f"look_{group}", group=group), group=group, seeds=list(R.seeds), label=K.LABEL,
        plan=R.plan, inputs=dict(R.inputs),
        failed_seeds=dict(claims_1_3={str(s): r for s, r in sorted(R.failed_seeds_claims_1_3.items())},
                          claim_4={str(s): r for s, r in sorted(R.failed_seeds_claim4.items())}),
        failed_checks={w: {str(s): c for s, c in sorted(d.items())} for w, d in R.failed_checks.items()},
        brackets=br, per_seed=per_seed, validity=validity(R), marks=mk, ran=R.ran,
        choice6_default=bool(R.defaults.get("main_on_fresh_seeds", True)),
        random_rows_control=bool(R.defaults.get("random_rows_control", True)),
        phase_stops=list(R.phase_stops))


def _finish(look, templates, group):
    look["outcomes"], look["outcome_reasons"] = outcomes(look)
    rows = sentences(look, templates, group)
    look["sentences"] = rows
    claims = {}
    for c in CLAIMS:
        ids = [r["id"] for r in rows if _claim_of(r["id"]) == c]
        if group == "exploratory":
            claims[c] = dict(status=EXPLORATORY_STATUS, claim="", exploratory_outcome=look["outcomes"][c],
                             sentence_ids=ids)
        else:
            claims[c] = dict(status=look["outcomes"][c], sentence_ids=ids)
    look["claims"] = claims
    return look


def exploratory_look(results, templates):
    """The exploratory look on seeds 0-4: the stop rules, every mark labeled "exploratory, not a claim", the
    sentences with the exploratory prefix."""
    R = results
    if R.group != "exploratory":
        raise LookRefused("refused: the exploratory look reads the exploratory group")
    look = _base_look(R, "exploratory")
    st = stop_rules(R)
    look.update(stops=st, fresh_near80_allowed=bool(st["fresh_near80_allowed"]),
                claim3_dropped=bool(st["claim3_dropped"]))
    look = _finish(look, templates, "exploratory")
    look["reported"] = reported(R)
    look["written"] = utc()
    return look


def confirmatory_look(results, exploratory, templates):
    """The confirmatory look on seeds 10-14: the validity conditions, then claims 1 to 4."""
    R = results
    if R.group != "confirmatory":
        raise LookRefused("refused: the confirmatory look reads the confirmatory group")
    ex = exploratory or {}
    if ex.get("kind") != LOOK_KIND["exploratory"]:
        raise LookRefused("refused: the confirmatory look needs the exploratory look")
    R.exploratory = ex
    look = _base_look(R, "confirmatory")
    look.update(exploratory_stops=dict(fresh_near80_allowed=ex.get("fresh_near80_allowed"),
                                       claim3_dropped=ex.get("claim3_dropped"), stops=ex.get("stops"),
                                       V=(ex.get("brackets") or {}).get("V"),
                                       arm=(ex.get("brackets") or {}).get("arm")),
                claim3_dropped=bool(ex.get("claim3_dropped")), exploratory_look_sha256=R.exploratory_sha256)
    look = _finish(look, templates, "confirmatory")
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


def run_look(group, out_root, mode, prereg, prereg_sha256=None, seeds=None, templates=None, defaults=None):
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
    R = load_results(root, group, mode, prereg_sha256=prereg_sha256, seeds=seeds, defaults=defaults)
    look = exploratory_look(R, templates) if group == "exploratory" else \
        confirmatory_look(R, R.exploratory, templates)
    digest = sha256_bytes(canonical(look).encode("utf8"))
    try:
        md = look_md(look, digest)
    except Exception as error:  # noqa: BLE001 - the report must never cost the look (the JSON is the record)
        md = (f"# Near-80 {group} look\n\nThe full report could not be formatted ({error!r}); read {json_path.name} "
              f"(SHA-256 {digest}).\n")
    write_text_once(md_path, md)
    got = write_json_once(json_path, look)
    if got != digest:
        raise LookRefused("refused: the look's JSON did not hash as computed")
    return look


# --------------------------------------------------------------------------------------------- the halt report

def halt_sentence_no_training(rules, templates=None):
    """Section 14.5's no-training stop for the first failed rule in section 8.1's order (freeze.py, step 7). Raises
    ValueError when no rule failed."""
    rule = first_failed_rule(rules)
    if rule is None:
        raise ValueError("no no-training rule failed")
    templates = templates or default_templates("scratch")
    return no_training_sentence(templates["sentences"]["stop.no_training"]["text"], rule)


def _infer_mode(root):
    if os.path.realpath(str(root)) == os.path.realpath(str(K.OUT)):
        return "official"
    for rel in (K.LAYOUT["status"], K.LAYOUT["stage_lock"].format(group="exploratory")):
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
    """Whichever stop applies now (the runner's 'report'): exactly one stop sentence, the first that applies of a
    no-training check on seeds 0-4, reproduction, a stop rule that cannot be read, too little harm, the rival
    unaffected (each of the last three with the claim-4 note), then one sentence per stopped phase. Reads only, unless
    write=True (looks/halt_report.json, once)."""
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

    failed = []
    for s in K.EXPLORATORY:
        for w in K.WORLDS:
            db = rd.read(K.LAYOUT["data_band"].format(world=w, seed=s), kinds=("near80-data-band",))
            if db is not None and db.get("halts_run"):
                rules = [r for r in db.get("rules") or [] if not r.get("passed")]
                failed += rules or ([db["first_failed_check"]] if db.get("first_failed_check") else [])
    fx = rd.read(K.LAYOUT["fixed"], kinds=("near80-fixed-checks",))
    if fx is not None and (fx.get("halts_run") or (fx.get("passed") is False and "halts_run" not in fx)):
        failed += [r for r in fx.get("rules") or [] if not r.get("passed")]
    first = first_failed_rule([dict(r, passed=False) for r in failed])
    if first is not None:
        stop = "stop.no_training"
        rows.append(dict(id=stop, text=no_training_sentence(t(stop), first)))
    if stop is None:
        # the plumbing writes plumbing/reproduction_differs.json (write-once) when the reproduction check differs
        rp = rd.read(K.LAYOUT["reproduction_differs"], kinds=("near80-reproduction-differs",), strict=False)
        pl = rp or rd.read(K.LAYOUT["plumbing"], kinds=("near80-plumbing",), strict=False)
        names = reproduction_failures((pl or {}).get("reproduction"))
        if names:
            stop = "stop.reproduction"
            rows.append(dict(id=stop, text=reproduction_sentence(t(stop), names)))
    if stop is None:
        ex = rd.read(K.LAYOUT["look"].format(group="exploratory"), kinds=(LOOK_KIND["exploratory"],))
        found = look_stop_rows(t, (ex or {}).get("stops"), bool(K.DEFAULTS.get("main_on_fresh_seeds", True)))
        if found:
            stop = found[0]["id"]
            rows += found
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
               inputs=dict(rd.inputs), note=None if rows else "No stop applies now.", written=utc())
    if write:
        write_json_once(root / K.LAYOUT["halt_report"], rep)
    return rep


def reproduction_failures(repro):
    """The plain names of the reproduction items that differed (section 5.8), in its order, each once. Accepts
    {item: {passed, plain}} and {world: {item: {...}}}."""
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
    head = "Exploratory look, seeds 0-4" if g == "exploratory" else "Confirmatory look, seeds 10-14"
    L = [f"# Near-80 (version 8): {head}", "", look["label"], "",
         f"Mode: {look['mode']}. Seeds {', '.join(str(s) for s in look['seeds'])}. Pre-registration "
         f"{look['prereg_sha256'] or 'not recorded (a scratch or stub run)'}. This look's JSON SHA-256 {digest}.", ""]
    if g == "exploratory":
        L += ["Seeds 0-4 never carry a claim: every claim below is exploratory, not a claim.", ""]
    L += ["## The sentences (section 14 of the pre-registration)", ""]
    L += [f"- {r['text']}" for r in look["sentences"]] + [""]
    L += ["## Validity (section 9.1)", "",
          f"- The harm (full-log actor minus no caution, near-80, stalling world, strong current): "
          f"{_f(val['claims_1_3']['harm']['value'])} (at least 30): "
          f"{'holds' if val['claims_1_3']['harm']['passed'] else 'fails'}",
          f"- No caution's share of strong-current steps past 80%: {_pct(val['claims_1_3']['past80']['value'])} "
          f"(at least 25%): {'holds' if val['claims_1_3']['past80']['passed'] else 'fails'}",
          f"- Check 10 held in {val['claims_2_3']['check10']['passed']} of {len(look['seeds'])} seeds; it did not "
          f"hold in {val['claims_2_3']['check10']['M']} (at most 1): "
          f"{'holds' if val['claims_2_3']['passed'] else 'fails'}",
          f"- The hand-set range limit's band cost: {_f(val['claim3']['rival_band_cost']['value'])} (at least 5): "
          f"{'holds' if val['claim3']['passed'] else 'fails'}"]
    left_out = val["claims_1_3"].get("failed_seeds") or []
    if left_out:
        L.append("- Left out of the conditions above (failed for claims 1-3; each fails every mark): seed "
                 + CT.join_items([str(s) for s in left_out]))
    L.append("")
    if g == "exploratory":
        st = look["stops"]
        un = st.get("unmeasured")
        L += ["## Stop rules (section 8.3)", "",
              "- Every stop rule can be read on every seed: " + (
                  "yes" if not un else f"no ({CT.join_items(un['what'])}: seed "
                                       f"{CT.join_items([str(x) for x in un['seeds']])})"),
              f"- The harm is there: {'passed' if st['harm']['passed'] else 'failed'}",
              f"- The band changes the rival's problem: {_f(st['rival_unaffected']['B'])} (at least 5): "
              f"{'passed' if st['rival_unaffected']['passed'] else 'failed'}",
              "- The noise floor: " + ("not run" if not st["noise_floor"]["run"] else
                                       "could not be measured (a number is missing on "
                                       + seeds_words(st["noise_floor"].get("missing_seeds") or []) + ")"
                                       if st["noise_floor"].get("unmeasured") else
                                       f"largest difference {_f(st['noise_floor']['V'])} ({st['noise_floor']['arm']}), "
                                       f"at most 5: {'passed' if st['noise_floor']['passed'] else 'failed'}"),
              f"- The near-80 settings on seeds 10-14: {'allowed' if look['fresh_near80_allowed'] else 'stopped'}; "
              f"claim 3 {'dropped' if look['claim3_dropped'] else 'kept'}", ""]
    c1, c4 = mk["claim1"], mk["claim4"]
    L += ["## Marks", "", "| claim | mark | value | needed | result |", "|---|---|---|---|---|"]
    for name, c in (("1 (near-80)", c1), ("4 (main)", c4)):
        nl = c["net_lead"]
        L.append(f"| {name} | net lead over {nl['best_rival_plain'] or 'not available'} | {_f(nl['lead'])} | 10 | "
                 f"{'pass' if nl['passed'] else 'fail'} |")
        for sit in K.KEEPS_SITUATIONS:
            k = c["keeps"][sit]
            L.append(f"| {name} | keeps the gain, {K.SITUATION_PLAIN[sit]} | {_f(k['difference'])} | -3 | "
                     f"{'pass' if k['passed'] else 'fail'} |")
        L.append(f"| {name} | beats the matched even pull | {_f(c['even_pull']['difference'])} | 15 | "
                 f"{'pass' if c['even_pull']['passed'] else 'fail'} |")
    c2, c3 = mk["claim2"], mk["claim3"]
    L += [f"| 2 | unseen-throttle gain (positive in {c2['gain']['positive']} seeds) | {_f(c2['gain']['value'])} | "
          f"5, and 4 seeds | {'pass' if c2['gain']['passed'] else 'fail'} |",
          f"| 2 | overshoot loss | {_f(c2['overshoot']['value'])} | at most 5 | "
          f"{'pass' if c2['overshoot']['passed'] else 'fail'} |",
          f"| 3 | band cost, range limit minus BCA (positive in {c3['K']} seeds) | {_f(c3['C'])} | 5, and 4 seeds | "
          f"{'pass' if c3['passed'] else 'fail'} |", ""]
    L += ["## Claims", ""] + [f"- {c}: {v['status']}" + (f" ({v['exploratory_outcome']})" if
                                                          'exploratory_outcome' in v else "")
                              for c, v in look["claims"].items()] + [""]
    fs = look["failed_seeds"]
    if fs["claims_1_3"] or fs["claim_4"]:
        L += ["## Failed seeds", ""] + [f"- claims 1-3, seed {s}: {r}" for s, r in fs["claims_1_3"].items()] + \
             [f"- claim 4, seed {s}: {r}" for s, r in fs["claim_4"].items()] + [""]
    L += ["## Brackets", "", "| bracket | value |", "|---|---|"]
    for name in ("S", "N", "X", "H", "W", "NET", "D", "R", "E", "L", "G", "O", "G0", "O0", "GZ", "OZ", "GV", "OV",
                 "Q", "A", "B", "C", "K", "M", "P", "V", "arm", "dB", "dR", "d0"):
        v = br.get(name)
        if name == "P":
            L.append(f"| P | {_pct(v)} |")
            continue
        kind = "count" if name in ("S", "N", "K", "M") else "points"
        L.append(f"| {name} | {v if isinstance(v, str) else _f(v, kind)} |")
    rep = look.get("reported") or {}
    pts = (rep.get("measures") or {}).get("points") or {}
    for st in K.SETTINGS:
        if st not in pts:
            continue
        L += ["", f"## Points in strong current, {SETTING_PLAIN[st]} (mean over seeds)", "",
              "| arm | stalling world | no-stall world |", "|---|---|---|"]
        for a in K.ARMS[st]:
            s_ = ((pts[st].get("stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
            n_ = ((pts[st].get("no_stall") or {}).get(a) or {}).get(SCORED, {}).get("mean")
            L.append(f"| {K.ARM_PLAIN.get(a, a)} ({a}) | {_f(s_)} | {_f(n_)} |")
    costs = rep.get("band_costs") or {}
    if costs:
        L += ["", "## Band costs, stalling world, strong current", "", "| arm | mean | per seed |", "|---|---|---|"]
        for a, by_w in costs.items():
            c = by_w["stall"][SCORED]
            L.append(f"| {K.ARM_PLAIN.get(a, a)} | {_f(c['mean'])} | {', '.join(_f(v) for v in c['per_seed'])} |")
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
    ap = argparse.ArgumentParser(description="near-80 looks: pooled.py exploratory|confirmatory|report")
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
