"""Near-80 (version 8): the phase table. Every phase the runner may start, its key and id, what it needs, when it is
admitted, when it is done, stopped or skipped, the files it writes, and its plain name.

Standard library only (with near80_constants and ledger), so the runner, the phase process and the tests read the same
table without loading numpy or JAX.

A phase key is PhaseKey(kind, setting, world, seed, group); setting and world may be 'x', seed may be None. The id
is near80_constants.phase_id(...), for example 'policy.near80.stall.s3', 'lock.x.x.exploratory'. The data phase of
the two band settings uses the setting token 'band' (it builds the near-80 and the random-rows data together).

The table (PREREGISTRATION.md sections 8 and 10; the build spec, section 5.2):
  fixed_checks       no seed (exploratory group)        approval only
  data_main          main, seeds 0-4 and 10-14           10-14: the exploratory look
  data_band          band, seeds 0-4 and 10-14           data_main of the same seed and world; 10-14: the look must
                                                         allow the near-80 settings
  critic, critic_second, switch, clone, reference_actors, policy
                     near80 and random_rows on 0-4; main, near80 and random_rows on 10-14
  critic_full        stalling world, seeds 10-14 (seeds 0-4 reuse the gate's full-log critic, checked)
  v6_band, readouts  near80 and random_rows, every seed
  evaluate           main, near80 and random_rows, every seed (main on 0-4: step 2's saved actors, retrained never)
  lock               per group, after every policy of the group
  look_exploratory, look_confirmatory   after every evaluation and readout of their group
Every need must be done; a stopped or skipped need makes the phase skipped (the lock and the looks run after their
needs are done, stopped or skipped). A seed whose main data failed the gate's checks is failed for every setting;
a seed whose band data failed checks 1-8 is failed for the near-80 and random-rows settings only. A failed seed is
never replaced.
"""
import sys

sys.dont_write_bytecode = True

from collections import namedtuple  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402
import ledger as LG  # noqa: E402

PhaseKey = namedtuple("PhaseKey", "kind setting world seed group")

SKIP_LOOK = "stopped before the fresh seeds by the exploratory stop rules"
BAND_SETTINGS = ("near80", "random_rows")
LEARNER_KINDS = ("critic", "critic_second", "switch", "clone", "reference_actors", "policy")
TRAINING_ROW_KINDS = ("critic", "critic_second", "switch", "clone", "reference_actors", "v6_band", "policy")
LENIENT_KINDS = ("lock", "look_exploratory", "look_confirmatory")     # run after needs done, stopped or skipped
GROUP_KINDS = ("fixed_checks", "lock", "look_exploratory", "look_confirmatory")
FINAL = ("done", "stopped", "skipped")
SKIPPED_REL = "skipped/{phase_id}.json"            # this module's own record of a skip decided by rule
PLANS = ("full", "smoke")                           # or 'seeds:<a>,<b>,...' (a stub run on fewer seeds)
SMOKE_SEED = 0


class Refused(Exception):
    """A key outside the table, a reserved seed, or a phase that may not start; the message starts with 'refused: '."""


class PhaseStop(Exception):
    """A recorded stop (PREREGISTRATION.md section 8.3, plumbing stops): a learner that fails, a built-in check (the
    height-0 check, a non-finite network), a saved input whose fingerprint does not match, or a training file that
    fails its check. phase_main writes stops/<phase_id>.json once and exits 2; a stopped phase is never rerun."""

    def __init__(self, reason, detail=None, checks=None):
        super().__init__(reason)
        self.reason, self.detail, self.checks = str(reason), detail, checks


class Run:
    """What one phase process knows: its key, the output root, the mode ('official', 'scratch' or 'stub'), the
    fixture (stub), the pre-registration's payload and the approval facts, the shortened sizes of a smoke or stub
    run (None: full length) and the plan it belongs to."""

    def __init__(self, key, out_root, mode="official", fixture=None, prereg=None, approval=None, steps=None,
                 chunk=None, updates=None, starts=None, plan="full", attempt=None, log=None):
        self.key, self.out_root, self.mode, self.fixture = key, Path(out_root), mode, fixture
        self.prereg, self.approval = prereg or {}, approval or {}
        self.steps, self.chunk, self.updates, self.starts = steps, chunk, updates, starts
        self.plan, self.attempt = plan, attempt
        self.log = log or (lambda *a: print(*a, flush=True))

    @property
    def phase_id(self):
        return phase_id(self.key)

    @property
    def official(self):
        return self.mode == "official"

    @property
    def stub(self):
        return self.mode == "stub"

    @property
    def full_length(self):
        return self.steps is None

    @property
    def prereg_sha256(self):
        """The approved fingerprint in official mode; None in scratch and stub records."""
        return self.approval.get("prereg_sha256") if self.official else None

    def pin(self, name):
        """A pin of prereg.json ('pins' block), or None when the payload has none (a test fixture)."""
        return (self.prereg.get("pins") or {}).get(name)


STOP_REASONS = dict(
    training_file="a training file that fails its check",
    fingerprint="a saved input whose fingerprint does not match",
    learner="a learner that fails (a non-finite network)",
    height0="the height-0 check failed (a ramp arm with height 0 differs from the no-caution actor)",
    check="a built-in check failed",
)


# --------------------------------------------------------------------------------------------- keys and ids

def make_key(kind, setting="x", world="x", seed=None, group=None):
    if seed is not None:
        seed = int(seed)
        group = K.group_of(seed)
    return PhaseKey(kind, setting, world, seed, group)


def phase_id(key):
    return K.phase_id(key.kind, key.setting, key.world, key.seed, None if key.seed is not None else key.group)


def parse_id(pid):
    """The key of an id; Refused when it is not of the form '<kind>.<setting>.<world>.<s{seed} | group | x>'."""
    parts = str(pid).split(".")
    if len(parts) != 4 or not all(parts):
        raise Refused(f"refused: {pid!r} is not a phase id ('<kind>.<setting>.<world>.<s<seed> | group>')")
    kind, setting, world, tail = parts
    seed, group = None, None
    if tail.startswith("s") and tail[1:].isdigit():
        seed = int(tail[1:])
        if seed in K.RESERVED:
            raise Refused(f"refused: seed {seed} belongs to version 6/7 (5-9); no near-80 phase touches it")
        try:
            group = K.group_of(seed)
        except ValueError as error:
            raise Refused(str(error)) from None
    elif tail in K.GROUPS:
        group = tail
    elif tail != "x":
        raise Refused(f"refused: {pid!r}: the last part must be s<seed>, a group or 'x'")
    return PhaseKey(kind, setting, world, seed, group)


def settings_for(kind, seed):
    s = int(seed)
    if kind in LEARNER_KINDS:
        return BAND_SETTINGS if s in K.EXPLORATORY else ("main",) + BAND_SETTINGS
    if kind in ("v6_band", "readouts"):
        return BAND_SETTINGS
    if kind == "evaluate":
        return ("main",) + BAND_SETTINGS
    if kind == "data_main":
        return ("main",)
    if kind == "data_band":
        return ("band",)
    if kind == "critic_full":
        return ("x",)
    return ()


def table_keys(seeds=None, groups=K.GROUPS):
    """Every key of the full table, in the order of section 5.2 (kind by kind; within a kind: seed, setting,
    world)."""
    seeds = list(K.EXPLORATORY + K.CONFIRMATORY) if seeds is None else [int(s) for s in seeds]
    seeds = [s for s in seeds if K.group_of(s) in groups]
    out = []
    for kind in K.PHASE_KINDS:
        if kind == "fixed_checks":
            if "exploratory" in groups:
                out.append(make_key(kind, group="exploratory"))
            continue
        if kind == "lock":
            out += [make_key(kind, group=g) for g in K.GROUPS if g in groups]
            continue
        if kind in ("look_exploratory", "look_confirmatory"):
            g = kind.split("_")[1]
            if g in groups:
                out.append(make_key(kind, group=g))
            continue
        for s in seeds:
            if kind == "critic_full":
                if s in K.CONFIRMATORY:
                    out.append(make_key(kind, "x", "stall", s))
                continue
            for setting in settings_for(kind, s):
                for world in K.WORLDS:
                    out.append(make_key(kind, setting, world, s))
    return out


def smoke_keys():
    """The plumbing smoke (scratch only): every kind on seed 0, both worlds, plus the kinds and settings that the
    official table runs on seeds 10-14 only (the full-log critic and the main setting's learners and actor
    training), so that every code path runs once at 40 steps. Seeds 10-14 are never touched."""
    s = SMOKE_SEED
    out = list(table_keys(seeds=[s], groups=("exploratory",)))
    extra = [make_key("critic_full", "x", "stall", s)]
    for kind in LEARNER_KINDS:
        for world in K.WORLDS:
            extra.append(make_key(kind, "main", world, s))
    by_kind = {}
    for key in out + extra:
        by_kind.setdefault(key.kind, []).append(key)
    ordered = []
    for kind in K.PHASE_KINDS:
        if kind == "look_confirmatory":
            continue
        ordered += by_kind.get(kind, [])
    return ordered


def plan(groups=K.GROUPS, spec="full"):
    """The phase list of a plan: 'full' (the official table), 'smoke' (plumbing) or 'seeds:<a>,<b>' (a stub run on
    fewer seeds; never official)."""
    if spec == "full":
        return table_keys(groups=groups)
    if spec == "smoke":
        return smoke_keys()
    if str(spec).startswith("seeds:"):
        seeds = [int(v) for v in str(spec)[6:].split(",") if v.strip()]
        mine = {K.group_of(s) for s in seeds}
        return table_keys(seeds=seeds, groups=tuple(g for g in groups if g in mine))
    raise Refused(f"refused: unknown plan {spec!r}")


SCRATCH_FRESH = "refused: seeds 10-14 run only in the official run (scratch is the plumbing on seed 0)"


def check_plan_spec(spec, mode):
    """The official run uses the full plan only; a scratch run is the plumbing smoke on seed 0 only (PREREGISTRATION.md
    section 8.2), so it never reaches seeds 10-14; a stub run may use any plan."""
    if mode == "official" and spec != "full":
        raise Refused("refused: the official run uses the full plan only")
    if mode == "scratch" and spec != "smoke":
        raise Refused("refused: a scratch run is the plumbing smoke on seed 0 only (--plan smoke)")
    plan(spec=spec)
    return spec


def check_key(key, mode="official", spec="full"):
    """Refuses (Refused) a reserved seed (5-9, in every mode), a seed outside 0-4 and 10-14, a phase of seeds 10-14 or
    of the confirmatory group in scratch mode, and every key that is not in the plan's table (for example
    critic.main.stall.s0 in the official table). Returns the key."""
    if key.seed is not None:
        if int(key.seed) in K.RESERVED:
            raise Refused(f"refused: seed {key.seed} belongs to version 6/7 (5-9); no near-80 phase touches it")
        if int(key.seed) not in K.EXPLORATORY + K.CONFIRMATORY:
            raise Refused(f"refused: seed {key.seed} is not one of this version's seeds (0-4, 10-14)")
    if mode == "scratch":
        if key.seed is not None and int(key.seed) in K.CONFIRMATORY:
            raise Refused(SCRATCH_FRESH)
        if key.seed is None and key.group == "confirmatory":
            raise Refused("refused: the confirmatory group runs only in the official run (scratch is the plumbing on "
                          "seed 0)")
    if key.kind not in K.PHASE_KINDS:
        raise Refused(f"refused: {key.kind!r} is not a phase kind of the table")
    if key.setting not in K.SETTINGS + ("band", "x") or key.world not in K.WORLDS + ("x",):
        raise Refused(f"refused: setting {key.setting!r} or world {key.world!r} is outside the table")
    if key.setting == "band" and key.kind != "data_band":
        raise Refused("refused: the setting token 'band' belongs to data_band only")
    check_plan_spec(spec, mode)
    if key not in plan(spec=spec):
        raise Refused(f"refused: {phase_id(key)} is not in the table" + ("" if spec == "full" else f" ({spec})"))
    return key


# --------------------------------------------------------------------------------------------- needs

def data_key(setting, world, seed):
    return make_key("data_main", "main", world, seed) if setting == "main" else make_key("data_band", "band", world,
                                                                                       seed)


def look_key(group):
    return make_key(f"look_{group}", group=group)


def needs(key, keys=None):
    """The keys this phase needs (all of them must be done; for the lock and the looks: done, stopped or skipped).
    keys: the plan (for the group phases); default the full table."""
    keys = table_keys() if keys is None else keys
    k, st, w, s = key.kind, key.setting, key.world, key.seed
    if k == "fixed_checks":
        return []
    if k == "data_main":
        return [] if s in K.EXPLORATORY else [look_key("exploratory")]
    if k == "data_band":
        return [make_key("data_main", "main", w, s)]
    if k in ("critic", "critic_second", "clone"):
        return [data_key(st, w, s)]
    if k == "critic_full":
        return [make_key("data_main", "main", "stall", s)]
    if k == "switch":
        return [make_key("critic_second", st, w, s)]
    if k == "reference_actors":
        out = [make_key("critic", st, w, s), make_key("clone", st, w, s)]
        if w == "stall" and st in ("near80", "main") and s in K.CONFIRMATORY:
            out.append(make_key("critic_full", "x", "stall", s))
        return out
    if k == "v6_band":
        return [make_key("critic", st, w, s)]
    if k == "policy":
        out = [make_key("switch", st, "stall", s), make_key("switch", st, "no_stall", s), make_key("critic", st, w, s),
               make_key("clone", st, w, s), make_key("reference_actors", st, w, s)]
        if st == "near80":
            out.append(make_key("v6_band", st, w, s))
        if w == "no_stall":
            out.append(make_key("policy", st, "stall", s))
        return out
    if k == "lock":
        return [x for x in keys if x.kind == "policy" and x.group == key.group]
    if k == "evaluate":
        out = [make_key("lock", group=key.group)]
        if not (st == "main" and s in K.EXPLORATORY):
            out.append(make_key("policy", st, w, s))
        else:
            out.append(make_key("data_main", "main", w, s))
        return out
    if k == "readouts":
        return [make_key("lock", group=key.group), make_key("v6_band", st, w, s)]
    if k in ("look_exploratory", "look_confirmatory"):
        g = key.group
        out = [x for x in keys if x.group == g and x.kind in ("evaluate", "readouts", "data_main", "data_band")]
        if g == "exploratory":
            out += [x for x in keys if x.kind == "fixed_checks"]
        else:
            out.append(look_key("exploratory"))
        return out
    raise Refused(f"refused: {k!r} has no row in the table")


def children(keys):
    """{key: [keys that need it]} within a plan (only needs that are in the plan)."""
    inside = set(keys)
    out = {k: [] for k in keys}
    for k in keys:
        for d in needs(k, keys):
            if d in inside:
                out[d].append(k)
    return out


# --------------------------------------------------------------------------------------------- time (queue order)

def est_minutes(key):
    """Rough minutes per phase (queue order only: longest remaining chain first, as version 7)."""
    k = key.kind
    if k == "policy":
        if key.setting == "random_rows":
            return 4
        return 40 if key.world == "stall" else 12
    if k == "data_main":
        return 1 if key.seed in K.EXPLORATORY else 3
    return dict(critic=8, critic_second=8, critic_full=8, clone=2, reference_actors=4, v6_band=5, evaluate=2,
                readouts=2, data_band=2, fixed_checks=2, switch=1, lock=1, look_exploratory=1,
                look_confirmatory=1)[k]


def priorities(keys):
    """{key: estimated minutes of the longest chain that starts with it}."""
    ch = children(keys)
    memo = {}

    def prio(k, depth=0):
        if k in memo:
            return memo[k]
        if depth > len(keys):
            raise Refused("refused: the plan's needs form a cycle")
        memo[k] = est_minutes(k) + max((prio(c, depth + 1) for c in ch[k]), default=0)
        return memo[k]

    return {k: prio(k) for k in keys}


def stage(key):
    return LG.stage_of(key.kind, key.setting, key.group)


# --------------------------------------------------------------------------------------------- files

def record_rel(key):
    """The phase's final record, relative to the output root."""
    k, st, w, s = key.kind, key.setting, key.world, key.seed
    if k == "fixed_checks":
        return K.LAYOUT["fixed"]
    if k == "data_main":
        return K.LAYOUT["data_main"].format(world=w, seed=s)
    if k == "data_band":
        return K.LAYOUT["data_band"].format(world=w, seed=s)
    if k == "critic_full":
        return K.LAYOUT["critic_full"].format(seed=s)
    if k in ("critic", "critic_second", "switch", "clone", "reference_actors", "v6_band"):
        return K.LAYOUT[k].format(setting=st, world=w, seed=s)
    if k == "policy":
        return K.LAYOUT["policy_lock"].format(setting=st, world=w, seed=s)
    if k == "lock":
        return K.LAYOUT["stage_lock"].format(group=key.group)
    if k == "evaluate":
        return K.LAYOUT["values"].format(setting=st, world=w, seed=s)
    if k == "readouts":
        return K.LAYOUT["readouts"].format(setting=st, world=w, seed=s)
    if k in ("look_exploratory", "look_confirmatory"):
        return K.LAYOUT["look"].format(group=key.group)
    raise Refused(f"refused: {k!r} has no record")


def stop_rel(key):
    return K.LAYOUT["phase_stop"].format(phase_id=phase_id(key))


def policy_stop_rel(key):
    return K.LAYOUT["policy_stop"].format(setting=key.setting, world=key.world, seed=key.seed)


def skipped_rel(key):
    return SKIPPED_REL.format(phase_id=phase_id(key))


def _with_sidecars(rels):
    out = []
    for r in rels:
        out.append(r)
        if r.endswith(".json"):
            out.append(r + ".sha256")
    return out


def outputs(key):
    """Every file the phase may write under the output root (relative paths, JSON sidecars included): what is
    checked for write-once before it starts and moved aside, never deleted, before a rerun."""
    k, st, w, s = key.kind, key.setting, key.world, key.seed
    rec = record_rel(key)
    d = str(Path(rec).parent)
    files = [rec]
    if k == "data_main" and s in K.CONFIRMATORY:
        t = K.LAYOUT["train_file"].format(setting="main", world=w, seed=s)
        files += [t, t + ".manifest.json"]
    if k == "data_band":
        for setting in BAND_SETTINGS:
            t = K.LAYOUT["train_file"].format(setting=setting, world=w, seed=s)
            files += [t, t + ".manifest.json"]
    if k in ("critic", "critic_full"):
        files.append(f"{d}/critic.msgpack")
    if k == "critic_second":
        files.append(f"{d}/critic_second.msgpack")
    if k == "clone":
        files.append(f"{d}/clone.msgpack")
    if k == "reference_actors":
        files.append(f"{d}/reference_actors.msgpack")
    if k == "v6_band":
        files += [f"{d}/v6_band.msgpack", f"{d}/v6_bank_scores.npz"]
    if k == "policy":
        files += [f"{d}/actors.msgpack", policy_stop_rel(key)]
    if k in ("look_exploratory", "look_confirmatory"):
        files.append(K.LAYOUT["look_md"].format(group=key.group))
    files.append(stop_rel(key))
    return _with_sidecars(files)


# --------------------------------------------------------------------------------------------- states

def done(key, out_root):
    """'done' when the final record exists and matches its sidecar; 'stopped' when its stop file (or the policy
    stop.json) does; 'skipped' when a skip was recorded; otherwise 'missing'."""
    root = Path(out_root)
    if LG.read_record(root / record_rel(key))[0] == "ok":
        return "done"
    if LG.read_record(root / stop_rel(key))[0] == "ok":
        return "stopped"
    if key.kind == "policy" and LG.read_record(root / policy_stop_rel(key))[0] == "ok":
        return "stopped"
    if LG.read_record(root / skipped_rel(key))[0] == "ok":
        return "skipped"
    return "missing"


def read_look(out_root, group="exploratory"):
    """The look record of a group with a verified sidecar, or None."""
    return LG.ok_record(Path(out_root) / K.LAYOUT["look"].format(group=group))


def seed_failures(out_root, seed):
    """(reason the seed failed for every setting or None, reason it failed for the near-80 settings or None), from
    its data records: a data_main record of either world with seed_failed_reason fails every setting; a data_band
    record of either world with seed_failed_reason fails the near-80 and random-rows settings."""
    root = Path(out_root)
    main, band = None, None
    for w in K.WORLDS:
        r = LG.ok_record(root / K.LAYOUT["data_main"].format(world=w, seed=seed))
        if r and r.get("seed_failed_reason") and main is None:
            main = f"seed {seed} failed the gate's checks in the {K.WORLD_PLAIN[w]}: {r['seed_failed_reason']}"
        b = LG.ok_record(root / K.LAYOUT["data_band"].format(world=w, seed=seed))
        if b and b.get("seed_failed_reason") and band is None:
            band = f"seed {seed} failed a no-training check in the {K.WORLD_PLAIN[w]}: {b['seed_failed_reason']}"
    return main, band


def rule_skip(key, out_root, keys=None, states=None, look=None, _memo=None):
    """Why this phase is skipped by rule (or None): a failed seed, a stopped or skipped need (followed through
    chains of needs that are themselves skipped by rule), or a look that stops the near-80 settings before the fresh
    seeds. states: {key: state} already known (the runner's), else read from the files."""
    memo = {} if _memo is None else _memo
    if key.seed is not None and key.kind not in ("data_main",):
        main, band = seed_failures(out_root, key.seed)
        if main:
            return main
        if band and key.setting in BAND_SETTINGS and key.kind != "data_band":
            return band
    if key.seed in K.CONFIRMATORY and (key.setting in BAND_SETTINGS or key.kind == "data_band"):
        look = read_look(out_root) if look is None else look
        if look is not None and look.get("fresh_near80_allowed") is not True:
            return SKIP_LOOK
    if key.kind in LENIENT_KINDS:
        return None
    for d in needs(key, keys):
        st = (states or {}).get(d) or effective_state(d, out_root, keys, look, memo)
        if st in ("stopped", "skipped"):
            return f"{K.PHASE_PLAIN[d.kind]} ({phase_id(d)}) was {st}"
    return None


def effective_state(key, out_root, keys=None, look=None, memo=None):
    """done(key) when final on file; 'skipped' when a rule skips it; otherwise 'missing'."""
    memo = {} if memo is None else memo
    if key in memo:
        return memo[key]
    memo[key] = "missing"                       # a guard against cycles (the plan has none)
    st = done(key, out_root)
    if st == "missing" and rule_skip(key, out_root, keys, look=look, _memo=memo):
        st = "skipped"
    memo[key] = st
    return st


def needs_state(key, out_root, keys=None):
    """('ready', None) | ('skip', why) | ('wait', why): whether the needs allow this phase to start now."""
    memo = {}
    why = rule_skip(key, out_root, keys, _memo=memo)
    if why:
        return "skip", why
    for d in needs(key, keys):
        st = effective_state(d, out_root, keys, memo=memo)
        if key.kind in LENIENT_KINDS:
            if st not in FINAL:
                return "wait", f"{phase_id(d)} is not finished"
        elif st != "done":
            return "wait", f"{phase_id(d)} is not done"
    return "ready", None


# --------------------------------------------------------------------------------------------- admission

def admit(key, approval, look, mode="official"):
    """(ok, why). Every phase needs the approval facts (approval_check.require_approval's dict). A phase of seeds
    10-14 (or of the confirmatory group) also needs the exploratory look record: kind 'near80-look-exploratory',
    the same prereg_sha256 (records of stub runs carry none, so there, and only there, the look's may also be null; a
    scratch run never reaches seeds 10-14). A near-80 or random-rows phase on seeds 10-14 (and data_band there) needs
    look['fresh_near80_allowed'] is True; otherwise (False, SKIP_LOOK): the phase is skipped and nothing of it is
    generated."""
    if not approval:
        return False, "refused: no verified approval (approval_check.require_approval)"
    if key.group == "confirmatory" and key.kind != "look_exploratory":
        if mode == "scratch":
            return False, SCRATCH_FRESH
        if look is None:
            return False, "refused: the exploratory look does not exist yet (seeds 10-14 wait for it)"
        if look.get("kind") != "near80-look-exploratory":
            return False, "refused: the exploratory look record is not a near-80 exploratory look"
        allowed = {approval.get("prereg_sha256")} if mode != "stub" else {approval.get("prereg_sha256"), None}
        if look.get("prereg_sha256") not in allowed or (mode == "official" and look.get("mode") != "official"):
            return False, "refused: the exploratory look was made under another pre-registration"
        if key.seed is not None and (key.setting in BAND_SETTINGS or key.kind == "data_band"):
            if look.get("fresh_near80_allowed") is not True:
                return False, SKIP_LOOK
    return True, None


def is_skip(why):
    return why == SKIP_LOOK


def plain_name(key):
    base = K.PHASE_PLAIN[key.kind]
    parts = []
    if key.setting not in ("x", "band"):
        parts.append({"main": "main setting", "near80": "near-80 setting", "random_rows": "random-rows setting"}[
            key.setting])
    if key.setting == "band":
        parts.append("near-80 and random-rows settings")
    if key.world != "x":
        parts.append(K.WORLD_PLAIN[key.world])
    if key.seed is not None:
        parts.append(f"seed {key.seed}")
    elif key.group:
        parts.append(f"{key.group} group")
    return base + (f" ({', '.join(parts)})" if parts else "")
