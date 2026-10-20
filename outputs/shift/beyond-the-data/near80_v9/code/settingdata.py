"""Near-80 (version 9): the main setting's data, the fresh data of version 9's seeds, the training files, the
outcome-free count of BCA's two pooled groups with the held-out top-up, and the setting contexts every learner,
actor-training and evaluation phase reads.

The one setting (PREREGISTRATION.md section 3): 'main' is the main test's removal (the ball; shift.make_removal(log,
'gap', gap)), with the gate's normalization (the ball-kept training rows' states), as step 1 did for its second
critic. BCA's two groups are pooled from every held-out recorded row outside the ball (near80lib.pooled_groups); the
data phase counts them from the commands alone before any learner is trained (near80lib.outcome_free_count), tops the
full-throttle group up to 36 rows with held-out blocks of new episodes when it is smaller (topup.py; blocks of 100
episodes, at most 10, each world its own number), and records the count by source, the blocks' array SHA-256, the
held-out rows' and transitions' fingerprints and the new data checks (section 8.1), which the switch phase checks. A
fresh seed that fails a data check is a failed seed (not BCA's; never replaced).

Seed 0 (the plumbing only) comes from the gate (acceptance.load_seed regenerates its log and checks it against the
gate's data record). The confirmatory seeds are new: fresh_seed_data restates version 6's fresh.make_fresh_data
(dataset.generate with the gate's recorder settings, sizes and stall slope, the gate's ball, the ball's leak checks,
the five world checks in the stalling world); FreshSeed regenerates a fresh seed and checks it against its data
record. Every other seed is refused everywhere (near80_constants.group_of).

Nothing here trains a learner. The record written here is the data record of section 5.5 of the build spec:
near80v9-data-main.
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


def topup_block(seed, world, spec, slope, k):
    """Top-up block k of a seed-world (topup.make_block: the gate's environment, streams [20261012, seed, 61 / 62, k];
    the stub replaces it with a synthetic block)."""
    import topup as TU
    return TU.make_block(seed, world, spec, slope, k)


def ball_checks_passed(checks):
    return bool(checks.get("gap_rows_in_gapped_training") == 0 and checks.get("gap_rows_in_gapped_bank") == 0
                and all(v for v in checks.values() if isinstance(v, bool)))


def fresh_seed_data(seed, world, out_root=None):
    """Version 6's fresh.make_fresh_data, restated for the fresh seeds (and seed 0 in the plumbing's fresh-path check):
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


def seed_data(seed, world, out_root, run=None):
    """The verified seed data behind a data_main record: the gate's (seed 0, the plumbing; its log hashes equal to the
    record's) or a regenerated fresh seed."""
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
    """The setting's shift.Removal of one seed's log (the main setting only: the gate's 'gap' removal)."""
    if setting == "main":
        return sd.removals["gap"]
    raise PH.Refused(f"refused: {setting!r} is not a setting of version 9 (the main setting only)")


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
    if setting != "main":
        raise PH.Refused(f"refused: {setting!r} is not a setting of version 9 (the main setting only)")
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


# --------------------------------------------------------------------------------------------- the data phases

def topup_plan(run, sd):
    """(make, target, cap) of the top-up (section 3.2). The plumbing's smoke (scratch, seed 0, --smoke-force) forces
    exactly one block whatever the count, so that every top-up code path runs once (section 8.2)."""
    make = lambda k: topup_block(sd.rep, sd.world, sd.spec, float(sd.physics.stall_slope), k)
    if getattr(run, "smoke_force", False):
        return make, 10 ** 9, 1
    return make, K.CERTIFIABLE_MIN, int(K.TOPUP["cap"])


def heldout_fingerprints(sd, lg, rows):
    """(rows SHA-256, transitions SHA-256) of the held-out rows H of the groups' view: the rows, and their transitions
    as learners._transition builds them with the gate's normalization (what the switch phase scores)."""
    mean, std = (np.asarray(v, np.float32) for v in sd.norm())
    tr = L._transition(lg, rows, mean, std)
    return N.rows_sha256(rows), L.tree_sha256(tr)


def group_count(run, sd, rec):
    """The outcome-free count of BCA's two pooled groups with the top-up (section 3.2, reported; commands only, before
    any learner is trained), the held-out rows' and transitions' fingerprints the switch phase checks, and the new
    data checks of section 8.1. Fills rec['group_count'], rec['topup'], rec['heldout_rows_sha256'],
    rec['heldout_transitions_sha256'] and rec['data_checks_v9']; returns the failed-seed reason or None. A failed group
    construction check is a coding error: a recorded stop."""
    step1 = N.load_step1lib()
    kept = sd.removals["gap"].kept_train_rows(sd.log)
    count, blocks = N.outcome_free_count(sd.log, sd.gap, kept, step1, K.GAP_RULE, topup=topup_plan(run, sd))
    lg = N.view(sd.log, blocks)
    h = N.heldout_rows(lg, sd.gap, K.HELDOUT_SPLITS)
    rec["group_count"] = count
    rec["topup"] = dict(count.get("topup") or {}, tags=dict(master=K.TOPUP["master"], starts_and_noise=K.TOPUP[
        "tag_starts"], coin=K.TOPUP["tag_coin"]), forced_in_smoke=bool(getattr(run, "smoke_force", False)))
    rec["heldout_rows_sha256"], rec["heldout_transitions_sha256"] = heldout_fingerprints(sd, lg, h)
    if count["defined"] and not count["group_checks"]["passed"]:
        raise PH.PhaseStop(PH.STOP_REASONS["check"], dict(what="BCA's two pooled groups hold a row they must not "
                                                               "hold", group_checks=count["group_checks"]))
    checks = dict(heldout=count.get("heldout_checks"))
    if count["defined"]:
        checks["count_unchanged_when_outcomes_scrambled"] = N.count_scramble_check(
            sd.log, sd.gap, kept, step1, K.GAP_RULE, blocks, count, [*K.COUNT_SCRAMBLE_TAG, sd.rep, K.WORLD_TAG[
                sd.world]])
        again = [topup_block(sd.rep, sd.world, sd.spec, float(sd.physics.stall_slope), b["k"]) for b in blocks]
        import topup as TU
        bad = TU.check_blocks(again, TU.summary(blocks))
        checks["topup_blocks_reproduce"] = dict(passed=not bad, problems=bad, blocks=len(blocks))
    rec["data_checks_v9"] = checks
    failed = [k for k, v in checks.items() if isinstance(v, dict) and v.get("passed") is False]
    return ("the new data checks failed: " + ", ".join(failed)) if failed else None


def topup_blocks(sd, rec):
    """The data record's top-up blocks of a seed-world, regenerated and checked against the record's array SHA-256
    (a difference is a recorded stop: a saved input whose fingerprint does not match)."""
    import topup as TU
    want = (rec.get("topup") or {}).get("blocks") or []
    blocks = [topup_block(sd.rep, sd.world, sd.spec, float(sd.physics.stall_slope), int(b["k"])) for b in want]
    bad = TU.check_blocks(blocks, want)
    if bad:
        raise PH.PhaseStop(PH.STOP_REASONS["fingerprint"], dict(what="the top-up blocks", problems=bad))
    return blocks


def data_main(run):
    """The main setting's data record (kind near80v9-data-main): the gate's seed (seed 0, the plumbing only) or a
    fresh seed, with the main setting's training file and prepared hashes on the fresh seeds, and on every seed the
    outcome-free count of BCA's two pooled groups (group_count) and the held-out rows' fingerprint."""
    key = run.key
    seed, world = key.seed, key.world
    rec = dict(kind="near80v9-data-main", seed=seed, world=world, group=key.group, train_file=None,
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
        why = group_count(run, sd, rec)
        if why and not rec["seed_failed_reason"]:
            rec["seed_failed_reason"] = why
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
        why = group_count(run, sd, rec)          # before any learner is trained: commands only
        if why:
            rec.update(passed_gate_checks=False, seed_failed_reason=why)
            return rec
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
    rec.update(kind="near80v9-data-main", seed=0, world=key.world, group=key.group, train_file=None,
               prepared_sha256=None, fresh_path_reproduces_gate=dict(same, passed=bool(all(same.values()))))
    if sd is not None:
        why = group_count(run, sd, rec)          # the fresh path's count and top-up, read back by read_back_fresh
        rec["fresh_path_data_checks"] = why
    return rec


def read_back_fresh(out_root, seed, world):
    """Plumbing check (c), section 8.2: a fresh-path data record, written through the phase wrapper, read back the way
    every later phase reads a fresh seed's record (FreshSeed: the regenerated log's hashes, the ball and both
    removals; the record's own sizes, never the wrapper's), with its top-up blocks regenerated and checked and its
    held-out fingerprints recomputed. Returns a dict with 'passed'."""
    rec = data_main_record(out_root, seed, world)
    out = dict(seed=int(seed), world=world)
    try:
        sd = FreshSeed(seed, world, rec)
        blocks = topup_blocks(sd, rec)
        lg = N.view(sd.log, blocks)
        h = N.heldout_rows(lg, sd.gap, K.HELDOUT_SPLITS)
        rows_sha, tr_sha = heldout_fingerprints(sd, lg, h)
        out.update(log_and_removals=True, topup_blocks=len(blocks),
                   heldout_rows_equal=bool(rows_sha == rec.get("heldout_rows_sha256")),
                   heldout_transitions_equal=bool(tr_sha == rec.get("heldout_transitions_sha256")),
                   sizes=list(fresh_record_sizes(rec)), wrapper_keys_kept_apart=bool("run_sizes" in rec))
    except (PH.Refused, PH.PhaseStop) as error:
        out.update(log_and_removals=False, error=str(error))
    out["passed"] = bool(out.get("log_and_removals") and out.get("heldout_rows_equal")
                         and out.get("heldout_transitions_equal") and out.get("wrapper_keys_kept_apart"))
    return out


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
        """The prepared hashes (rows, train, bank, normalization) the main setting's data record holds for this seed
        and world (the fresh seeds). None on seed 0 (the gate's own loader checks the gate's data there)."""
        if self.setting != "main":
            raise PH.Refused(f"refused: {self.setting!r} is not a setting of version 9")
        if self.rep in K.EXPLORATORY:
            return None
        return data_main_record(self.out_root, self.rep, world).get("prepared_sha256") or {}

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
        """Version 9 trains every learner of the main setting itself, also on seed 0 in the plumbing's smoke; the gate's
        saved critics and actors are read by the reproduction check only (evaluate_phase.reproduction)."""
        return False

    def critic(self, world="stall"):
        if self._gate_seed_artifacts():
            return gate_critic(self.seed(world), "gap")
        d = learner_dir(self.out_root, self.setting, world, self.rep)
        rec = load_record(d / "critic.json")
        return load_tree_checked(d / "critic.msgpack", L.critic_template(), rec["file_sha256"])

    def full_critic(self):
        """The full-log critic of the stalling world (critic_full's, on every seed of the plan)."""
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

    def heldout_view(self, world):
        """(the groups' view of this seed-world: the log and its top-up blocks, regenerated and checked against the
        data record; the data record). The switch phase reads it; no scoring phase does."""
        rec = data_main_record(self.out_root, self.rep, world)
        sd = self.seed(world)
        return N.view(sd.log, topup_blocks(sd, rec)), rec


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
