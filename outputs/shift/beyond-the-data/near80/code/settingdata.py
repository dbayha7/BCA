"""Near-80 (version 9, main setting): the main setting's data, the fresh seeds' data, the training files, the
outcome-free count of the pooled groups, and the setting contexts every learner, actor-training and evaluation phase
reads. DRAFT, not frozen.

The one setting is 'main': the main test's removal (the ball; shift.make_removal(log, 'gap', gap)) with the gate's
normalization (the ball-kept training rows' states), as step 1 did for its second critic.

Seed 0 (the plumbing's seed, scratch and stub runs only) comes from the gate (acceptance.load_seed regenerates the
log and checks it against the gate's data record). The fresh seeds are new: fresh_seed_data restates version 6's
fresh.make_fresh_data (dataset.generate with the gate's recorder settings, sizes and stall slope, the gate's ball, the
ball's leak checks, the five world checks in the stalling world); FreshSeed regenerates a fresh seed and checks it
against its data record (with amendment 1's reading of the record's sizes). Every seed any other study used is
refused everywhere (near80_constants.group_of).

Every seed-world, the plumbing's seed included, gets its own training file and its own critics and actors under the
output root: version 8's reuse of the gate's saved artifacts on seeds 0-4 is gone with the exploratory stage.

Nothing here trains a learner. The data record (kind near80-data-main) carries the outcome-free count of the pooled
groups (near80lib.pooled_count_record), computed before any training (section 8.2).
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402

ROOT = Path(K.ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import ledger as LG  # noqa: E402
import phases as PH  # noqa: E402
import near80lib as N  # noqa: E402
import facts_near80 as F  # noqa: E402
from experiments.shift import acceptance as A  # noqa: E402
from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import learners as L  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

PREPARED_KEYS = ("rows", "train", "bank", "normalization")
_GATE = {}


# --------------------------------------------------------------------------------------------- the gate

def gate_prereg():
    """The gate's pre-registration through its own loader and guard; its variant must be the base world."""
    if "prereg" not in _GATE:
        p = A.load_prereg(A.refuse_protected(A.RUNS_DEFAULT))
        A.guard(p)
        variant = A.gate_outcome(p)["variant"]
        if variant != K.VARIANT:
            raise PH.Refused(f"refused: the gate's variant is {variant}, near-80 runs on {K.VARIANT}")
        _GATE["prereg"] = p
    return _GATE["prereg"]


def check_gate_pin(run):
    """The gate's pre-registration must be the one prereg.json pins (official and scratch)."""
    want = run.pin("gate_prereg_sha256")
    if run.stub:
        return None
    p = gate_prereg()
    if want is None and run.official:
        raise PH.Refused("refused: prereg.json pins no gate pre-registration")
    if want is not None and p.digest != want:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the gate's pre-registration", want=want,
                                                                got=p.digest))
    return p.digest


def learn_settings(run):
    """The gate's learner settings (stub mode: learners.LearnSettings(), the same values without the gate)."""
    if run.stub:
        return L.LearnSettings()
    return gate_prereg().config.learn


def eval_cfg(run):
    """What acceptance._evaluate reads from the gate's configuration: the bank size (--starts in a smoke or stub)."""
    if run.starts is not None:
        return types.SimpleNamespace(bank_size=int(run.starts))
    return gate_prereg().config


def anchors(run, world):
    """{situation: {'recording', 'top', ...}} of the gate (acceptance.anchors_of); stub mode replaces it."""
    return A.anchors_of(gate_prereg(), K.VARIANT, world)


# --------------------------------------------------------------------------------------------- seeds

def gate_seed(seed, world):
    """The plumbing's seed 0: acceptance.load_seed(prereg.runs, cfg, 'base', world, seed), which regenerates the log
    and checks it, its ball and both gate removals against the gate's data record."""
    if int(seed) not in K.SMOKE_SEEDS:
        raise PH.Refused(f"refused: seed {seed} is not the plumbing's seed ({K.SMOKE_SEEDS})")
    p = gate_prereg()
    return A.load_seed(p.runs, p.config, K.VARIANT, world, int(seed))


def gate_record_sha256(seed, world):
    path = A.data_dir(gate_prereg().runs, K.VARIANT, world, int(seed)) / "record.json"
    return LG.file_sha256(path) if path.is_file() else None


def fresh_settings():
    """(recorder spec, sizes, stall slope) of the gate's base world: runs/shift/pre/stage0_base.json's chosen
    recorder, the gate's sizes and stall slope."""
    p = gate_prereg()
    return A.stage0_spec(p.runs, K.VARIANT), tuple(p.config.sizes), float(p.config.slope(K.VARIANT))


def fresh_log(seed, world, spec, sizes, slope):
    """dataset.generate of a fresh seed (never a seed another study used; the stub replaces it with a synthetic
    log)."""
    K.group_of(seed)
    return DS.generate(int(seed), world, spec, tuple(sizes), slope)


def fresh_gap(lg):
    return SH.define_gap(lg)


class SeedView:
    """acceptance.SeedData's interface for a fresh seed (or a stub's synthetic seed): the log, its ball, the three
    gate removals ('none', 'gap', 'random'), the gate's normalization and prepared data on demand."""

    def __init__(self, rep, world, record, log, gap, spec, physics):
        self.rep, self.world, self.variant, self.record = int(rep), world, K.VARIANT, record
        self.log, self.gap, self.spec, self.physics = log, gap, spec, physics
        self.removals = {k: SH.make_removal(log, k, gap) for k in SH.REMOVALS}
        self.arrays = {}
        self._prepared = {}

    def norm(self):
        return L.normalization(self.log, self.removals["gap"])

    def prepared(self, kind, learn):
        if kind not in self._prepared:
            self._prepared[kind] = L.prepare(self.log, self.removals[kind], self.norm(), learn)
        return self._prepared[kind]


def _gap_of(g):
    return SH.Gap(centre=tuple(g["centre"]), rho=g["rho"], mass=g["mass"], rule=g["rule"], q=g["q"],
                  competing_ratio=math.inf if g["competing_ratio"] is None else g["competing_ratio"],
                  record_sha256=g["record_sha256"])


def _ball_checks(lg, gap, rg, rr):
    train, bank = lg.split_rows("train"), lg.bank_rows
    checks = dict(gap_rows_in_gapped_training=SH.leak_count(rg.kept_train_rows(lg), lg, gap),
                  gap_rows_in_gapped_bank=SH.leak_count(rg.bank_rows(lg), lg, gap),
                  random_same_count=bool(rr.count == rg.count and rr.bank_removed.sum() == rg.bank_removed.sum()),
                  leak_counter_finds_every_training_gap_row=bool(SH.leak_count(train, lg, gap) == rg.count),
                  leak_counter_finds_every_bank_gap_row=bool(SH.leak_count(bank, lg, gap) == int(rg.bank_removed.sum())),
                  evaluation_split_untouched=bool(not rg.removed[lg.split_rows("evaluation")].any()))
    return checks, ball_checks_passed(checks)


def ball_checks_passed(checks):
    return bool(checks.get("gap_rows_in_gapped_training") == 0 and checks.get("gap_rows_in_gapped_bank") == 0
                and all(v for v in checks.values() if isinstance(v, bool)))


def fresh_seed_data(seed, world, out_root=None):
    """Version 6's fresh.make_fresh_data, restated for the fresh seeds (and seed 0 in the plumbing's fresh-path
    check): the log (dataset.generate with the gate's base-world recorder, sizes and stall slope), the gate's ball
    (shift.define_gap), the removals 'gap' and 'random', the ball's leak checks and, in the stalling world, the gate's
    five world checks (acceptance.world_marks, required). Returns (data fields, SeedView or None when no ball)."""
    s = int(seed)
    K.group_of(s)
    spec, sizes, slope = fresh_settings()
    physics = TC.world_physics(world, slope)
    lg = fresh_log(s, world, spec, sizes, slope)
    gap, grec = fresh_gap(lg)
    rec = dict(source="fresh", seed=s, world=world, variant=K.VARIANT, log_sizes=list(sizes), stall_slope=slope,
               recorder=spec.as_dict(), physics=physics.as_dict(), log_sha256=lg.array_hashes(),
               bank_rows=int(len(lg.bank_rows)), gap_record_sha256=SH.GAP.canonical_sha256(grec))
    if gap is None:
        rec.update(defined=False, gap=None, removals=None, ball_leak_checks=None, ball_leak_checks_passed=False,
                   gate_world_marks=None, passed_gate_checks=False,
                   seed_failed_reason=f"the gate's rule found no ball ({grec.get('reason')})")
        return rec, None
    rg, rr = SH.make_removal(lg, "gap", gap), SH.make_removal(lg, "random", gap)
    checks, ok = _ball_checks(lg, gap, rg, rr)
    rec.update(defined=True, gap=gap.as_dict(), removals={k: r.summary(lg) for k, r in (("gap", rg), ("random", rr))},
               ball_leak_checks=checks, ball_leak_checks_passed=ok, gate_world_marks=None)
    passed = ok
    if world == "stall":
        card, marks = A.world_marks(lg, gap, spec, physics)
        rec.update(gate_world_card=card, gate_world_marks={k: bool(v) for k, v in marks.items()})
        passed = passed and all(marks.values())
    failed = None
    if not passed:
        bad = [k for k, v in (rec.get("gate_world_marks") or {}).items() if not v]
        failed = ("the gate's checks failed: " + (f"world checks {', '.join(bad)}" if bad else "") +
                  ("" if ok else ("; " if bad else "") + "the ball's leak checks"))
    rec.update(passed_gate_checks=bool(passed), seed_failed_reason=failed)
    return rec, SeedView(s, world, rec, lg, gap, spec, physics)


def fresh_record_sizes(record):
    """The log sizes (training, calibration and evaluation episodes) a fresh seed's data record was generated with:
    'log_sizes', three positive integers (version 9 keeps them under their own name, apart from the phase wrapper's
    run sizes; amendment 1's lesson). Anything else is refused."""
    s = record.get("log_sizes")
    if isinstance(s, list) and len(s) == 3 and all(type(v) is int and v > 0 for v in s):
        return tuple(s)
    raise PH.Refused(f"refused: the data record's log_sizes {s!r} are not three positive integers")


class FreshSeed(SeedView):
    """A fresh seed regenerated and checked against its data record (version 6's FreshSeed, restated): the log's
    array hashes and both removals must equal the record's."""

    def __init__(self, seed, world, record):
        if not record.get("passed_gate_checks"):
            raise PH.Refused(f"refused: fresh seed {seed} ({world}) failed the gate's checks; it is never replaced")
        spec = RC.RecorderSpec(**record["recorder"])
        lg = fresh_log(seed, world, spec, fresh_record_sizes(record), record["stall_slope"])
        if lg.array_hashes() != record["log_sha256"]:
            raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the regenerated log of seed {seed} "
                                                                          f"({world})"))
        super().__init__(seed, world, record, lg, _gap_of(record["gap"]), spec, lg.physics)
        for k in ("gap", "random"):
            s = self.removals[k].summary(lg)
            want = record["removals"][k]
            if s["kept_rows_sha256"] != want["kept_rows_sha256"] or s["bank_rows_sha256"] != want["bank_rows_sha256"]:
                raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the {k} removal of seed {seed} "
                                                                              f"({world})"))


def data_main_record(out_root, seed, world):
    path = Path(out_root) / K.LAYOUT["data_main"].format(world=world, seed=seed)
    try:
        return LG.read_json_verified(path)
    except LG.RecordError as error:
        raise PH.Refused(f"refused: {error}") from None


def seed_data(seed, world, out_root, run=None):
    """The verified seed data behind a data_main record: the gate's (seed 0, its log hashes equal to the record's) or
    a regenerated fresh seed."""
    rec = data_main_record(out_root, seed, world)
    if rec.get("seed_failed_reason"):
        raise PH.Refused(f"refused: seed {seed} failed its data checks; it is never replaced")
    if int(seed) in K.SMOKE_SEEDS:
        sd = gate_seed(seed, world)
        if sd.record.get("log_sha256") != rec.get("log_sha256"):
            raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the gate's log of seed {seed} ({world})"))
        return sd
    return FreshSeed(seed, world, rec)


# --------------------------------------------------------------------------------------------- the setting

def removal(setting, sd):
    """The setting's shift.Removal of one seed's log (the main setting: the ball)."""
    if setting == "main":
        return sd.removals["gap"]
    raise PH.Refused(f"refused: {setting!r} is not a setting of this version")


def prepared(setting, sd, learn, rm=None):
    """learners.prepare(log, removal, the gate's normalization, learn)."""
    return L.prepare(sd.log, removal(setting, sd) if rm is None else rm, sd.norm(), learn)


def prepared_hashes(prep):
    return {k: prep.sha256[k] for k in PREPARED_KEYS}


def read_training_arrays(path):
    import h5py
    with h5py.File(path, "r") as f:
        return {k: f[k][()] for k in DS.D4RL_KEYS}


def write_training_file(setting, sd, out_root, rm=None):
    """dataset.write_d4rl_training (the six D4RL arrays of the kept training rows), then the file is read back and
    checked against the kept rows' arrays (near80lib.training_file_check). Returns the record's entry."""
    rel = K.LAYOUT["train_file"].format(setting=setting, world=sd.world, seed=sd.rep)
    path = Path(out_root) / rel
    man_path = Path(str(path) + ".manifest.json")
    if path.exists() or man_path.exists():
        raise PH.Refused(f"refused: {path} exists (write-once)")
    rm = removal(setting, sd) if rm is None else rm
    manifest = DS.write_d4rl_training(sd.log, rm, path)
    check = N.training_file_check(read_training_arrays(path), sd.log, rm)
    return dict(path=rel, removal=rm.kind, file_sha256=manifest["file_sha256"], rows=manifest["rows"],
                rows_sha256=manifest["rows_sha256"], arrays={k: v["sha256"] for k, v in manifest["arrays"].items()},
                manifest_sha256=LG.file_sha256(man_path), training_file_check=check)


def check_training_file(run, setting, sd, learn):
    """Step 6 of every phase that reads the setting's training rows: the written train.hdf5 re-read and checked
    against the regenerated log and the rebuilt removal (near80lib.training_file_check); the rebuilt removal's
    kept-row and bank-row hashes equal the data record's; learners.prepare's rows, train, bank and normalization
    hashes equal the data record's prepared_sha256; the file's SHA-256 equals the record's. Returns (passed, detail,
    prepared)."""
    key = run.key
    out = run.out_root
    rec = data_main_record(out, key.seed, key.world)
    entry = rec.get("train_file") or {}
    want_rm = (rec.get("removals") or {}).get("gap") or {}
    want_prep = rec.get("prepared_sha256") or {}
    rel = K.LAYOUT["train_file"].format(setting=setting, world=key.world, seed=key.seed)
    path = out / rel
    detail = dict(path=rel)
    if not path.is_file():
        detail["missing"] = True
        return False, detail, None
    detail["file_sha256_equal"] = bool(LG.file_sha256(path) == entry.get("file_sha256"))
    rm = removal(setting, sd)
    c = N.training_file_check(read_training_arrays(path), sd.log, rm)
    detail["training_file_check"] = c
    s = rm.summary(sd.log)
    detail["removal_equal"] = bool(s["kept_rows_sha256"] == want_rm.get("kept_rows_sha256")
                                   and s["bank_rows_sha256"] == want_rm.get("bank_rows_sha256"))
    prep = prepared(setting, sd, learn, rm)
    got = prepared_hashes(prep)
    detail["prepared_equal"] = {k: bool(got[k] == want_prep.get(k)) for k in PREPARED_KEYS}
    passed = bool(detail["file_sha256_equal"] and c["passed"] and detail["removal_equal"]
                  and all(detail["prepared_equal"].values()))
    return passed, detail, prep


# --------------------------------------------------------------------------------------------- the data phase

def gate_seed_fields(sd, world):
    """The data record's fields of the gate's seed 0, from the gate's own record (its checks are the gate's)."""
    g = sd.record
    checks = g.get("checks") or {}
    ok = ball_checks_passed(checks)
    marks = g.get("t1_t5") if world == "stall" else None
    passed = bool(g.get("passed")) and ok and (all((marks or {}).values()) if world == "stall" else True)
    return dict(source="gate", log_sha256=g.get("log_sha256"), gap=g.get("gap"), removals=g.get("removals"),
                ball_leak_checks=checks, ball_leak_checks_passed=ok,
                gate_world_marks=None if marks is None else {k: bool(v) for k, v in marks.items()},
                passed_gate_checks=passed,
                seed_failed_reason=None if passed else "the gate's data checks failed on this seed")


def data_main(run):
    """The main setting's data record (kind near80-data-main): the gate's seed 0 (scratch and stub) or a fresh seed,
    with the training file, the prepared hashes and, before any training, the outcome-free count of the pooled
    groups (section 8.2)."""
    key = run.key
    seed, world = key.seed, key.world
    rec = dict(kind="near80-data-main", seed=seed, world=world, group=key.group, train_file=None,
               prepared_sha256=None, pooled_groups=None)
    if seed in K.SMOKE_SEEDS:
        sd = gate_seed(seed, world)
        rec.update(gate_seed_fields(sd, world),
                   gate_record_sha256=None if run.stub else gate_record_sha256(seed, world))
    else:
        fields, sd = fresh_seed_data(seed, world, run.out_root)
        rec.update(fields)
        if run.stub and sd is not None:
            import stubs
            rec.update(stub_note=stubs.STUB_NOTE, stub_would_fail=dict(seed_failed_reason=rec["seed_failed_reason"],
                                                                       passed_gate_checks=rec["passed_gate_checks"]),
                       seed_failed_reason=None, passed_gate_checks=True)
    if sd is not None and rec["passed_gate_checks"]:
        t0 = time.perf_counter()
        rec["pooled_groups"] = N.pooled_count_record(sd.log, sd.gap, removal("main", sd), N.load_step1lib())
        rec["pooled_groups"]["seconds"] = time.perf_counter() - t0
        learn = learn_settings(run)
        rec["train_file"] = write_training_file("main", sd, run.out_root)
        if not rec["train_file"]["training_file_check"]["passed"]:
            raise PH.PhaseStop(PH.STOP_REASONS["training_file"], rec["train_file"])
        rec["prepared_sha256"] = prepared_hashes(prepared("main", sd, learn))
    return rec


def fresh_path_check(run):
    """Plumbing only (scratch, seed 0): the fresh-seed code path must reproduce the gate's seed-0 data record (its
    log hashes, ball, and both removals' kept-row and bank-row hashes); the record is then read back as a fresh
    record is (fresh_record_sizes), amendment 1's lesson."""
    key = run.key
    if run.mode != "scratch" or key.seed != 0:
        raise PH.Refused("refused: the fresh-path check is a scratch check on seed 0 only")
    fields, sd = fresh_seed_data(0, key.world)
    g = gate_seed(0, key.world).record
    same = dict(log_sha256=bool(fields["log_sha256"] == g["log_sha256"]),
                gap=bool(_as_json(fields.get("gap")) == _as_json(g.get("gap"))),
                sizes_read_back=bool(fresh_record_sizes(_as_json(fields)) == tuple(fields["log_sizes"])))
    for k in ("gap", "random"):
        a, b = (fields.get("removals") or {}).get(k) or {}, (g.get("removals") or {}).get(k) or {}
        same[f"{k}_removal"] = bool(a.get("kept_rows_sha256") == b.get("kept_rows_sha256")
                                    and a.get("bank_rows_sha256") == b.get("bank_rows_sha256"))
    rec = dict(fields)
    rec.update(kind="near80-data-main", seed=0, world=key.world, group=key.group, train_file=None,
               prepared_sha256=None, pooled_groups=None,
               fresh_path_reproduces_gate=dict(same, passed=bool(all(same.values()))))
    return rec


def _as_json(x):
    """x as the strict JSON of a record holds it (tuples as lists, non-finite numbers as null)."""
    import json
    return json.loads(LG.canonical(x))


# --------------------------------------------------------------------------------------------- setting contexts

def learner_dir(out_root, setting, world, seed):
    return Path(out_root) / Path(K.LAYOUT["critic"].format(setting=setting, world=world, seed=seed)).parent


def load_record(path):
    try:
        return LG.read_json_verified(path)
    except LG.RecordError as error:
        raise PH.Refused(f"refused: {error}") from None


def load_tree_checked(path, template, sha256):
    try:
        return L.load_tree(path, template, sha256)
    except ValueError as error:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=str(path), error=str(error))) from None


class SettingContext:
    """What a learner, actor-training or evaluation phase reads, for the main setting and one seed (both worlds on
    demand): prepared(world), critic(world), full_critic(), clone(world), none(world), full_none(), anchors(world),
    switch_record(world). Also the attributes the stand-ins module and acceptance._evaluate read: sd[world] (log and
    ball), rep, learn, cfg, prereg (the gate's)."""

    def __init__(self, run, setting, seed, worlds=K.WORLDS, prepared_cache=None):
        self.run, self.setting, self.rep = run, setting, int(seed)
        self.out_root = Path(run.out_root)
        self.learn = learn_settings(run)
        self.cfg = eval_cfg(run)
        self.prereg = None if run.stub else gate_prereg()
        self.worlds = tuple(worlds)
        self._sd = {}
        self._prepared = dict(prepared_cache or {})
        self._prepared_checked = set()
        self.sd = _LazySeeds(self)

    def seed(self, world):
        if world not in self._sd:
            self._sd[world] = seed_data(self.rep, world, self.out_root, self.run)
        return self._sd[world]

    def expected_prepared(self, world):
        """The prepared hashes (rows, train, bank, normalization) the data record holds for this seed and world."""
        return data_main_record(self.out_root, self.rep, world).get("prepared_sha256") or {}

    def check_prepared(self, world, prep):
        """Every phase that reads the setting's prepared training and bank rows through this context compares them
        with the data record's prepared_sha256 once: a difference is a recorded stop (a training file that fails its
        check)."""
        want = self.expected_prepared(world)
        got = prepared_hashes(prep)
        equal = {k: bool(got[k] == want.get(k)) for k in PREPARED_KEYS}
        if not all(equal.values()):
            raise PH.PhaseStop(PH.STOP_REASONS["training_file"], dict(
                what=f"the {self.setting} setting's prepared rows of seed {self.rep} ({world}) differ from the data "
                     f"record's prepared_sha256", prepared_equal=equal))

    def prepared(self, world="stall"):
        if world not in self._prepared:
            self._prepared[world] = prepared(self.setting, self.seed(world), self.learn)
        if world not in self._prepared_checked:
            self.check_prepared(world, self._prepared[world])
            self._prepared_checked.add(world)
        return self._prepared[world]

    # ---- learners
    def critic(self, world="stall"):
        d = learner_dir(self.out_root, self.setting, world, self.rep)
        rec = load_record(d / "critic.json")
        return load_tree_checked(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])

    def full_critic(self):
        """The full-log critic of the stalling world (critic_full's)."""
        d = Path(self.out_root) / Path(K.LAYOUT["critic_full"].format(seed=self.rep)).parent
        rec = load_record(d / "critic.json")
        return load_tree_checked(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])

    def clone(self, world="stall"):
        d = learner_dir(self.out_root, self.setting, world, self.rep)
        rec = load_record(d / "clone.json")
        return load_tree_checked(d / "clone.msgpack", L.actor_template(), rec["file_sha256"])

    def _reference(self, world, name):
        d = learner_dir(self.out_root, self.setting, world, self.rep)
        rec = load_record(d / "reference_actors.json")
        acts = load_tree_checked(d / "reference_actors.msgpack", {k: L.actor_template() for k in rec["names"]},
                                 rec["file_sha256"])
        if name not in acts or L.tree_sha256(acts[name]) != rec["actor_sha256"][name]:
            raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"reference actor {name} ({world})"))
        return acts[name]

    def none(self, world="stall"):
        return self._reference(world, "none")

    def full_none(self):
        return self._reference("stall", "full_none")

    def anchors(self, world="stall"):
        return anchors(self.run, world)

    def switch_record(self, world):
        return load_record(learner_dir(self.out_root, self.setting, world, self.rep) / "switch.json")


class _LazySeeds(dict):
    """ctx.sd[world] for the stand-ins module (it reads .log and .gap)."""

    def __init__(self, ctx):
        super().__init__()
        self._ctx = ctx

    def __missing__(self, world):
        sd = self._ctx.seed(world)
        self[world] = sd
        return sd


def gate_critic(sd, removal):
    """A gate seed's fixed-controller critic (the stall-aware controller) through the gate's loader, which checks the
    saved file's SHA-256: removal 'gap' (the main test's critic) or 'none' (the full-log critic). The reproduction
    check reads them on seed 0."""
    rec, params = A.get_critic(gate_prereg(), sd, removal, K.CONTROLLER, train=False)
    if params is None:
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what=f"the gate's {removal} critic of seed {sd.rep} "
                                                                 f"({sd.world}) failed in the gate"))
    return params


def gate_clone(seed, world):
    """A gate seed's saved cloned actor (the gate's read path, its SHA-256 checked); the reproduction check."""
    p = gate_prereg()
    d = A._variant_root(p, "clone", K.VARIANT) / world / f"seed{int(seed)}"
    rec = A.finished(d / "record.json", p)
    if rec is None:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the gate's cloned actor at {d}"))
    return load_tree_checked(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])


def gate_check2_actor(seed, name):
    """Check 2's saved actor of a gate seed (the gate's harm_return values and actors, with their SHA-256), alpha 5;
    the reproduction check."""
    p = gate_prereg()
    d = A._variant_root(p, "harm_return", K.VARIANT) / f"alpha{K.ALPHA:g}" / f"seed{int(seed)}"
    rec = A.finished(d / "values.json", p)
    if rec is None:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"check 2's values at {d}"))
    if rec["failed"].get(name):
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what=f"check 2's {name} actor failed in the gate"))
    acts = load_tree_checked(d / "actors.msgpack", {k: L.actor_template() for k in rec["actor_names"]},
                             rec["actors_sha256"])
    return acts[name]


def context(run, setting, seed, worlds=K.WORLDS, prepared_cache=None):
    return SettingContext(run, setting, seed, worlds, prepared_cache)
