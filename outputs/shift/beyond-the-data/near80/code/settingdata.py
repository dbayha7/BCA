"""Near-80 (version 8): the data of the three settings, the fresh data of seeds 10-14, the training files, and the
setting contexts every learner, actor-training and evaluation phase reads.

Settings (PREREGISTRATION.md section 3): 'main' is the main test's removal (the ball; shift.make_removal(log, 'gap',
gap)); 'near80' is near80lib.band_removal (the ball, then every training and bank row with a command from 70% to 90%
on any thruster); 'random_rows' is near80lib.random_rows_removal (the ball, then as many random rows as the band
takes, in each kind of water). Every setting uses the gate's normalization (the ball-kept training rows' states),
as step 1 did for its second critic.

Seeds 0-4 come from the gate (acceptance.load_seed regenerates each log and checks it against the gate's data
record). Seeds 10-14 are new: fresh_seed_data restates version 6's fresh.make_fresh_data (dataset.generate with the
gate's recorder settings, sizes and stall slope, the gate's ball, the ball's leak checks, the five world checks in the
stalling world); FreshSeed regenerates a fresh seed and checks it against its data record. Seeds 5-9 are refused
everywhere (near80_constants.group_of).

Nothing here trains a learner. The records written here are the data records of section 5.5 of the build spec:
near80-data-main, near80-data-band and near80-fixed-checks.
"""
import sys

sys.dont_write_bytecode = True

import math  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/user/bca")
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import near80_constants as K  # noqa: E402
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


def facts_anchors(run, world):
    """{situation: (J of the recording controller, J of the world's top controller)} as the facts script computes
    them (facts_near80.anchors_for), on the full banks (on --starts starts in a smoke or stub)."""
    return F.anchors_for(world, bank_n=run.starts)


def no_training_thresholds(run):
    return N.NO_TRAINING


# --------------------------------------------------------------------------------------------- seeds

def gate_seed(seed, world):
    """Seeds 0-4: acceptance.load_seed(prereg.runs, cfg, 'base', world, seed), which regenerates the log and checks
    it, its ball and both gate removals against the gate's data record."""
    if int(seed) not in K.EXPLORATORY:
        raise PH.Refused(f"refused: seed {seed} is not a gate seed (0-4)")
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
    """dataset.generate of a fresh seed (never for seeds 5-9; the stub replaces it with a synthetic log)."""
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
    """Version 6's fresh.make_fresh_data, restated for seeds 10-14 (and seed 0 in the plumbing's fresh-path check):
    the log (dataset.generate with the gate's base-world recorder, sizes and stall slope), the gate's ball
    (shift.define_gap), the removals 'gap' and 'random', the ball's leak checks and, in the stalling world, the gate's
    five world checks (acceptance.world_marks, required). Returns (data fields, SeedView or None when no ball)."""
    s = int(seed)
    K.group_of(s)
    spec, sizes, slope = fresh_settings()
    physics = TC.world_physics(world, slope)
    lg = fresh_log(s, world, spec, sizes, slope)
    gap, grec = fresh_gap(lg)
    rec = dict(source="fresh", seed=s, world=world, variant=K.VARIANT, sizes=list(sizes), stall_slope=slope,
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


RUN_SIZE_KEYS = ("chunk", "starts", "steps", "updates")


def fresh_record_sizes(record):
    """Amendment 1. The log sizes (training, calibration and evaluation episodes) a fresh seed's data record was
    generated with. fresh_seed_data stores them as 'sizes', three integers. Version 8's phase wrapper
    (phase_main.run_phase) then wrote its own run sizes ({chunk, starts, steps, updates}) over that key in every
    data_main record it wrote (each None, or a positive integer in scratch and stub runs); such a record's log was
    generated with the gate's sizes (fresh_settings(), which fresh_seed_data used), and FreshSeed's log-hash and
    removal checks confirm them. Exactly these two shapes are read; anything else is refused."""
    s = record.get("sizes")
    if isinstance(s, list) and len(s) == 3 and all(type(v) is int and v > 0 for v in s):
        return tuple(s)
    if isinstance(s, dict) and tuple(sorted(s)) == RUN_SIZE_KEYS and all(
            v is None or (type(v) is int and v > 0) for v in s.values()):
        return tuple(int(v) for v in fresh_settings()[1])
    raise PH.Refused(f"refused: the data record's sizes {s!r} are neither three positive integers nor the run sizes "
                     f"that version 8's phase wrapper wrote over them")


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


def data_band_record(out_root, seed, world):
    path = Path(out_root) / K.LAYOUT["data_band"].format(world=world, seed=seed)
    try:
        return LG.read_json_verified(path)
    except LG.RecordError as error:
        raise PH.Refused(f"refused: {error}") from None


def seed_data(seed, world, out_root, run=None):
    """The verified seed data behind a data_main record: the gate's (0-4, its log hashes equal to the record's) or a
    regenerated fresh seed (10-14)."""
    rec = data_main_record(out_root, seed, world)
    if rec.get("seed_failed_reason"):
        raise PH.Refused(f"refused: seed {seed} failed its data checks; it is never replaced")
    if int(seed) in K.EXPLORATORY:
        sd = gate_seed(seed, world)
        if sd.record.get("log_sha256") != rec.get("log_sha256"):
            raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the gate's log of seed {seed} ({world})"))
        return sd
    return FreshSeed(seed, world, rec)


# --------------------------------------------------------------------------------------------- settings

def removal(setting, sd):
    """The setting's shift.Removal of one seed's log."""
    if setting == "main":
        return sd.removals["gap"]
    near = N.band_removal(sd.log, sd.gap)
    if setting == "near80":
        return near
    if setting == "random_rows":
        return N.random_rows_removal(sd.log, sd.gap, near, sd.spec, sd.physics)
    raise PH.Refused(f"refused: {setting!r} is not a setting")


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
    checked against the leak-checked arrays (near80lib.training_file_check). Returns the record's entry."""
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
    """Step 6 of every phase that reads a setting's training rows (build spec 5.4): the written train.hdf5 re-read
    and checked against the regenerated log and the rebuilt removal (near80lib.training_file_check); the rebuilt
    removal's kept-row and bank-row hashes equal the data record's; learners.prepare's rows, train, bank and
    normalization hashes equal the data record's prepared_sha256; the file's SHA-256 equals the record's. Returns
    (passed, detail, prepared)."""
    key = run.key
    out = run.out_root
    if setting == "main":
        rec = data_main_record(out, key.seed, key.world)
        entry = rec.get("train_file") or {}
        want_rm = (rec.get("removals") or {}).get("gap") or {}
        want_prep = rec.get("prepared_sha256") or {}
    else:
        rec = data_band_record(out, key.seed, key.world)
        entry = (rec.get("train_files") or {}).get(setting) or {}
        want_rm = (rec.get("removal_sha256") or {}).get(setting) or {}
        want_prep = (rec.get("prepared_sha256") or {}).get(setting) or {}
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


# --------------------------------------------------------------------------------------------- the data phases

def data_main(run):
    """The main setting's data record (kind near80-data-main): the gate's seed (0-4) or a fresh seed (10-14), with
    the main setting's training file and prepared hashes on 10-14."""
    key = run.key
    seed, world = key.seed, key.world
    rec = dict(kind="near80-data-main", seed=seed, world=world, group=key.group, train_file=None,
               prepared_sha256=None)
    if seed in K.EXPLORATORY:
        sd = gate_seed(seed, world)
        g = sd.record
        checks = g.get("checks") or {}
        ok = ball_checks_passed(checks)
        marks = g.get("t1_t5") if world == "stall" else None
        passed = bool(g.get("passed")) and ok and (all((marks or {}).values()) if world == "stall" else True)
        rec.update(source="gate", log_sha256=g.get("log_sha256"), gap=g.get("gap"), removals=g.get("removals"),
                   ball_leak_checks=checks, ball_leak_checks_passed=ok,
                   gate_world_marks=None if marks is None else {k: bool(v) for k, v in marks.items()},
                   passed_gate_checks=passed, gate_record_sha256=None if run.stub else gate_record_sha256(seed, world),
                   seed_failed_reason=None if passed else "the gate's data checks failed on this seed")
        return rec
    fields, sd = fresh_seed_data(seed, world, run.out_root)
    rec.update(fields)
    if run.stub and sd is not None:
        import stubs
        rec.update(stub_note=stubs.STUB_NOTE, stub_would_fail=dict(seed_failed_reason=rec["seed_failed_reason"],
                                                                   passed_gate_checks=rec["passed_gate_checks"]),
                   seed_failed_reason=None, passed_gate_checks=True)
    if sd is not None and rec["passed_gate_checks"]:
        learn = learn_settings(run)
        rec["train_file"] = write_training_file("main", sd, run.out_root)
        if not rec["train_file"]["training_file_check"]["passed"]:
            raise PH.PhaseStop(PH.STOP_REASONS["training_file"], rec["train_file"])
        rec["prepared_sha256"] = prepared_hashes(prepared("main", sd, learn))
    return rec


def fresh_path_check(run):
    """Plumbing only (scratch, seed 0): the fresh-seed code path must reproduce the gate's seed-0 data record (its
    log hashes, ball, and both removals' kept-row and bank-row hashes)."""
    key = run.key
    if run.mode != "scratch" or key.seed != 0:
        raise PH.Refused("refused: the fresh-path check is a scratch check on seed 0 only")
    fields, sd = fresh_seed_data(0, key.world)
    g = gate_seed(0, key.world).record
    same = dict(log_sha256=bool(fields["log_sha256"] == g["log_sha256"]),
                gap=bool(_as_json(fields.get("gap")) == _as_json(g.get("gap"))))
    for k in ("gap", "random"):
        a, b = (fields.get("removals") or {}).get(k) or {}, (g.get("removals") or {}).get(k) or {}
        same[f"{k}_removal"] = bool(a.get("kept_rows_sha256") == b.get("kept_rows_sha256")
                                    and a.get("bank_rows_sha256") == b.get("bank_rows_sha256"))
    rec = dict(fields)
    rec.update(kind="near80-data-main", seed=0, world=key.world, group=key.group, train_file=None,
               prepared_sha256=None, fresh_path_reproduces_gate=dict(same, passed=bool(all(same.values()))))
    return rec


def _as_json(x):
    """x as the strict JSON of a record holds it (tuples as lists, non-finite numbers as null)."""
    import json
    return json.loads(LG.canonical(x))


def _check_number(rule):
    try:
        return int(rule["rule"].split(":")[0].split()[1])
    except (IndexError, ValueError):
        return None


def data_band(run):
    """The near-80 and random-rows settings' data of one seed-world (kind near80-data-band): the no-training facts
    (facts_near80.seed_world_facts, with the data-edge rollouts in the stalling world), the rules of
    near80lib.no_training_checks for this seed-world, both training files (each read back and checked), the removals'
    hashes and the prepared hashes the learner phases check against."""
    key = run.key
    seed, world = key.seed, key.world
    sd = seed_data(seed, world, run.out_root, run)
    main = data_main_record(run.out_root, seed, world)
    step1 = N.load_step1lib()
    an = facts_anchors(run, "stall") if world == "stall" else None
    rec_for_facts = sd.record if seed in K.EXPLORATORY else dict(removals=main["removals"])
    t0 = time.perf_counter()
    facts = F.seed_world_facts(sd.log, sd.gap, world, sd.physics, sd.spec, step1, rec=rec_for_facts, anchors=an,
                               bank_n=run.starts)
    nt = N.no_training_checks([facts], fixed=None, t=no_training_thresholds(run))
    rules = nt["rules"]
    passed_1_8 = all(r["passed"] for r in rules if (_check_number(r) or 0) <= 8)
    failed = [r for r in rules if not r["passed"]]
    first = None if not failed else dict(rule=failed[0]["rule"], value=failed[0]["value"],
                                         threshold=failed[0]["threshold"], scope=failed[0]["scope"])
    c10 = [r for r in rules if _check_number(r) == 10]
    learn = learn_settings(run)
    train_files, prep_hashes, rm_hashes = {}, {}, {}
    for setting in PH.BAND_SETTINGS:
        rm = removal(setting, sd)
        s = rm.summary(sd.log)
        rm_hashes[setting] = dict(kept_rows_sha256=s["kept_rows_sha256"], bank_rows_sha256=s["bank_rows_sha256"],
                                  kind=rm.kind)
        train_files[setting] = write_training_file(setting, sd, run.out_root, rm)
        prep_hashes[setting] = prepared_hashes(prepared(setting, sd, learn, rm))
    bad_files = [s for s, e in train_files.items() if not e["training_file_check"]["passed"]]
    pts = facts.get("data_edge_points")
    rec = dict(kind="near80-data-band", seed=seed, world=world, group=key.group, facts=facts, rules=rules,
               passed_checks_1_8=bool(passed_1_8), first_failed_check=first,
               check10=None if world != "stall" or not c10 else dict(value=c10[0]["value"],
                                                                     passed=bool(c10[0]["passed"])),
               data_edge_points=None if pts is None else dict(main=pts["ball_alone"], near80=pts["near80"],
                                                              random_rows=pts["random_rows"]),
               train_files=train_files, prepared_sha256=prep_hashes, removal_sha256=rm_hashes,
               halts_run=bool(seed in K.EXPLORATORY and first is not None),
               seed_failed_reason=None if passed_1_8 else (
                   f"no-training check failed: {first['rule']} (value {first['value']}, threshold "
                   f"{first['threshold']})" if first else "a no-training check numbered 1-8 failed"),
               facts_seconds=time.perf_counter() - t0)
    if bad_files:
        raise PH.PhaseStop(PH.STOP_REASONS["training_file"], dict(settings=bad_files, train_files=train_files))
    return stub_neutral(run, rec)


def stub_neutral(run, rec):
    """Stub mode only: synthetic logs are not built to pass the no-training checks, so their rules are recorded but
    fail no seed and halt nothing."""
    if run.stub:
        import stubs
        rec.update(seed_failed_reason=None, halts_run=False, stub_note=stubs.STUB_NOTE,
                   stub_would_fail=dict(seed_failed_reason=rec.get("seed_failed_reason"),
                                        halts_run=rec.get("halts_run")))
    return rec


def fixed_checks(run):
    """Check 9 on the fixed test start sets (kind near80-fixed-checks): facts_near80.fixed_bank_facts and the two
    check-9 rules of near80lib.no_training_checks."""
    an = facts_anchors(run, "stall")
    facts = F.fixed_bank_facts(an, bank_n=run.starts)
    rules = N.no_training_checks([], fixed=facts, t=no_training_thresholds(run))["rules"]
    return stub_neutral(run, dict(kind="near80-fixed-checks", facts=facts, rules=rules,
                                  passed=bool(all(r["passed"] for r in rules)),
                                  halts_run=bool(not all(r["passed"] for r in rules))))


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
    """What a learner, actor-training or evaluation phase reads, for one setting and seed (both worlds on demand):
    prepared(world), critic(world), full_critic(), clone(world), none(world), full_none(), anchors(world),
    switch_row(world). Also the attributes the stand-ins module and acceptance._evaluate read: sd[world] (log and
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
        """The prepared hashes (rows, train, bank, normalization) the setting's data record holds for this seed and
        world: data_band's for the near-80 and random-rows settings, data_main's for the main setting on seeds
        10-14. None for the main setting on seeds 0-4 (the gate's own loader checks the gate's data there)."""
        if self.setting == "main":
            if self.rep in K.EXPLORATORY:
                return None
            return data_main_record(self.out_root, self.rep, world).get("prepared_sha256") or {}
        rec = data_band_record(self.out_root, self.rep, world)
        return (rec.get("prepared_sha256") or {}).get(self.setting) or {}

    def check_prepared(self, world, prep):
        """Every phase that reads a setting's prepared training and bank rows through this context (the critics,
        BCA's switch and height, the actors, the evaluations, the readouts) compares them with the data record's
        prepared_sha256 once: a difference is a recorded stop (a training file that fails its check)."""
        want = self.expected_prepared(world)
        if want is None:
            return
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
    def _gate_seed_artifacts(self):
        return self.setting == "main" and self.rep in K.EXPLORATORY

    def critic(self, world="stall"):
        if self._gate_seed_artifacts():
            return gate_critic(self.seed(world), "gap")
        d = learner_dir(self.out_root, self.setting, world, self.rep)
        rec = load_record(d / "critic.json")
        return load_tree_checked(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])

    def full_critic(self):
        """The full-log critic of the stalling world: the gate's on seeds 0-4 (read through its loader, which checks
        the file's SHA-256), critic_full's on seeds 10-14."""
        if self.rep in K.EXPLORATORY:
            return gate_critic(self.seed("stall"), "none")
        d = Path(self.out_root) / Path(K.LAYOUT["critic_full"].format(seed=self.rep)).parent
        rec = load_record(d / "critic.json")
        return load_tree_checked(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])

    def clone(self, world="stall"):
        if self._gate_seed_artifacts():
            return gate_clone(self.rep, world)
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
        if self._gate_seed_artifacts():
            return gate_check2_actor(self.rep, "gapped" if world == "stall" else "no_stall_gapped")
        return self._reference(world, "none")

    def full_none(self):
        if self._gate_seed_artifacts():
            return gate_check2_actor(self.rep, "full")
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
    saved file's SHA-256: removal 'gap' (the main test's critic) or 'none' (the full-log critic)."""
    rec, params = A.get_critic(gate_prereg(), sd, removal, K.CONTROLLER, train=False)
    if params is None:
        raise PH.PhaseStop(PH.STOP_REASONS["learner"], dict(what=f"the gate's {removal} critic of seed {sd.rep} "
                                                                 f"({sd.world}) failed in the gate"))
    return params


def gate_clone(seed, world):
    """A gate seed's saved cloned actor (the gate's read path, its SHA-256 checked)."""
    p = gate_prereg()
    d = A._variant_root(p, "clone", K.VARIANT) / world / f"seed{int(seed)}"
    rec = A.finished(d / "record.json", p)
    if rec is None:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what=f"the gate's cloned actor at {d}"))
    return load_tree_checked(d / "actor.msgpack", L.actor_template(), rec["result"]["file_sha256"])


def gate_check2_actor(seed, name):
    """Check 2's saved actor of a gate seed (the gate's harm_return values and actors, with their SHA-256), alpha 5."""
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
