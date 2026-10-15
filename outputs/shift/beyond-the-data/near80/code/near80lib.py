"""Near-80 version (DRAFT, not approved): the data change, its control, the leak check and the no-training checks.

DRAFT. Nothing here trains a learner, fits a BCA band or reads an outcome of a learner. Everything is a pure NumPy
function on a log (or on facts computed from one), tested on synthetic logs in test_near80.py. See
../PREREGISTRATION.md (sections 3, 4 and 8) and ../prereg_draft.json.

The band (the new removal), exactly:
  A recorded row is IN THE BAND when, for at least one thruster k (scope "any"; scope "thruster1" reads k = 1 only),
      0.70 <= |a_k| <= 0.90,
  where a_k is the stored float32 command compared at its exact float64 value (the convention of shift.Gap.contains)
  and the edges are the float64 numbers 0.70 and 0.90, both included. Both signs count. So float32(0.7) =
  0.699999988... lies just OUTSIDE the band (a command "held to 70%" is on the seen side), float32(0.9) =
  0.899999976... lies inside it.

The near-80 removal ("ball_band"): the main test's removal first, bit for bit (shift.make_removal(log, "gap", gap):
the densest-15% ball), then every training row and every calibration-bank row in the band as well. The evaluation
split and the calibration rows outside the bank are never touched. Built as a shift.Removal; shift.py is imported,
never edited.

The control ("ball_random_rows"), drawn separately in calm and in hard water: the ball, then, in each kind of water,
exactly as many extra training rows as the band removes there on that log, drawn uniformly from the ball-kept training
rows of that kind with default_rng([20261012, rep, 99, 80, 2 + s]) (s = 0 calm, s = 1 hard water); the bank loses, in
each kind of water, exactly as many extra rows as the band takes from it there, drawn uniformly from the ball-kept bank
rows of that kind with default_rng([20261012, rep, 99, 80, 4 + s]). A row is hard water when the recording
controller's own rule says so on its state (RC.Recorder(spec, physics).in_region). So the control keeps exactly the
near-80 removal's share of hard-water rows and differs from it only in which commands it hides. (No tag list ends in
0: numpy drops a trailing 0.)
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]                       # .../bca (outputs/shift/beyond-the-data/near80/code -> repository root)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import stage0 as S0  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

STATUS = "DRAFT for the owner's approval. Nothing has been trained."

# ------------------------------------------------------------------------------------------------ fixed settings

BAND_LO = 0.70
BAND_HI = 0.90
SCOPES = ("any", "thruster1")
SCOPE = "any"
HOLD = BAND_LO                       # the "held to the band's lower edge" evaluation clip (unseen-throttle gain)
CAP_80 = TC.KNEE                     # the "capped at 80%" evaluation clip (overshoot loss; shift card measure 8)
KIND_BAND = "ball_band"
KIND_RANDOM = "ball_random_rows"
TAG_NEAR80 = 80                      # appended to shift's random-removal tags: [DATA_SEED, rep, 99, 80, ...]
STRATA = ("calm", "hard water")      # s = 0, 1: the recording controller's own rule on the row's state
TRAIN_TAG_BASE = 2                   # training draw of stratum s: [DATA_SEED, rep, 99, 80, 2 + s]
BANK_TAG_BASE = 4                    # bank draw of stratum s:     [DATA_SEED, rep, 99, 80, 4 + s]

EXPLORATORY_SEEDS = (0, 1, 2, 3, 4)  # seen; every result on them is exploratory
CONFIRMATORY_SEEDS = (10, 11, 12, 13, 14)  # fresh; refused by this draft code (nothing is approved)
RESERVED_SEEDS = (5, 6, 7, 8, 9)     # version 6/7's confirmatory seeds; always refused

STEP1_DIR = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step1"

# The no-training checks (kill rules before any training). "seen": the seeds 0-4 values known when each was set.
NO_TRAINING = dict(
    eval_band_rows_min=300,                       # seen 775-1,072 (any thruster)
    extra_share_min=0.01, extra_share_max=0.06,   # band-only rows / ball-kept training rows; seen 2.5-3.6%
    full_throttle_removed_min=S0.MARKS["T1_full_throttle_removed_min"],   # 0.99, the gate's T1; seen 99.1-99.5%
    kept_past80_share_max=S0.MARKS["T2_kept_past80_share_max"],           # 0.002, the gate's T2; seen 0.08-0.14%
    identical_rule_min=S0.MARKS["T4_identical_rule_min"],                 # 0.998, the gate's T4; seen 99.86-99.92%
    region_kept_min=0.35,          # SET AFTER SEEING seeds 0-4 (38.5-40.6% after the band; the gate's 40% fails 2)
    region_band_share_min=0.10,    # band rows among the ball-kept region rows; seen 12.5-14.3%
    # step 1's gap rule, lower end, near-80 setting: in [BAND_LO - 0.005, BAND_LO]; seen 0.6999-0.7000
    gap_lo_min=round(BAND_LO - 0.005, 6), gap_lo_max=BAND_LO,
    bank_min=500,                                 # seen 805-832 (BCA's own minimum is 36)
    stall_aware_band_share_min=0.50,              # test start sets, strong current, thruster 1 in the band; seen 81.1%
    hold_loss_min=15.0,                           # test start sets, stall-aware minus textbook held to HOLD; seen 29.3
    data_edge_loss_min=10.0,                      # per seed, textbook clipped to the recorded range; seen 17.5-19.0
)


class Refused(Exception):
    """A guard refused: nothing was computed or written."""


class LeakError(AssertionError):
    """A leak check failed: a removed command is still visible to training, or a held-out row was touched."""


# ------------------------------------------------------------------------------------------------ seeds and paths

def check_seeds(seeds):
    """This draft reads seeds 0-4 only. Seeds 5-9 belong to version 6/7; seeds 10-14 are this version's fresh
    confirmatory seeds and stay untouched until the owner's approval is imported (no code for that exists yet)."""
    seeds = [int(s) for s in seeds]
    for s in seeds:
        if s in RESERVED_SEEDS:
            raise Refused(f"refused: seed {s} belongs to version 6/7's confirmation (5-9)")
        if s in CONFIRMATORY_SEEDS:
            raise Refused(f"refused: seed {s} is a fresh confirmatory seed of this version (10-14); this draft code "
                          "never generates, loads or reads it")
        if s not in EXPLORATORY_SEEDS:
            raise Refused(f"refused: seed {s} is not one of the seen seeds 0-4")
    return seeds


def guard_out(path):
    """Outputs go outside the repository (scratch) or under runs/shift_explore/near80/ only. Never into a pinned
    folder, never into this code folder, never over an existing file."""
    p = Path(path).resolve()
    official = ROOT / "runs" / "shift_explore" / "near80"
    if ROOT == p or ROOT in p.parents:
        if not (official == p.parent or official in p.parents):
            raise Refused(f"refused: {p} is inside the repository but not under {official}")
    if p.exists():
        raise Refused(f"refused: {p} exists (outputs are write-once)")
    return p


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_step1lib():
    """Restructured BCA's step-1 library, executed from bytes verified against step 1's code manifest (never edited).
    It holds the throttle index and the gap rule that place BCA's ramp."""
    path = STEP1_DIR / "step1lib.py"
    data = path.read_bytes()
    want = json.loads((STEP1_DIR / "code_manifest.json").read_text())["files"]["step1lib.py"]
    if sha256_bytes(data) != want:
        raise Refused("refused: step1lib.py does not match step 1's code manifest")
    mod = types.ModuleType("near80_step1lib")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------------------ the band

def throttle_abs(actions):
    """|a| at the exact float64 value of the stored float32 command, shape (n, 4)."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != TC.DA:
        raise ValueError(f"commands must have shape (n, {TC.DA}), got {a.shape}")
    return np.abs(np.asarray(a, np.float32).astype(np.float64))


def in_band(actions, lo=BAND_LO, hi=BAND_HI, scope=SCOPE):
    """Per row: is the command in the band [lo, hi] (closed, float64 edges) on any thruster (scope 'any') or on
    thruster 1 (scope 'thruster1')? Both signs count."""
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if not (0.0 < float(lo) < float(hi) <= 1.0):
        raise ValueError(f"band edges must satisfy 0 < lo < hi <= 1, got [{lo}, {hi}]")
    m = throttle_abs(actions)
    per = (m >= float(lo)) & (m <= float(hi))
    return per.any(axis=1) if scope == "any" else per[:, 0]


def band_removal(log, gap, lo=BAND_LO, hi=BAND_HI, scope=SCOPE, empty=False):
    """The near-80 removal of a log: shift.make_removal(log, 'gap', gap) (the main test's ball, bit for bit), then
    every training row and bank row in the band. empty=True adds nothing (the reproduction check: it must equal the
    gate's own gap removal). Returns a shift.Removal of kind 'ball_band'."""
    base = SH.make_removal(log, "gap", gap)
    removed = base.removed.copy()
    bank_removed = base.bank_removed.copy()
    if not empty:
        band = in_band(log["actions"], lo, hi, scope)
        train = log.split_rows("train")
        removed[train] |= band[train]
        bank_removed |= band[log.bank_rows]
    if removed[log.split_rows("evaluation")].any() or removed[log.split_rows("calibration")].any():
        raise LeakError("the band removal must touch training rows only (the bank is handled separately)")
    return SH.Removal(KIND_BAND, removed, bank_removed, gap)


def removed_mask(log, gap=None, lo=BAND_LO, hi=BAND_HI, scope=SCOPE):
    """The removed mask (bool per log row, True only on training rows) of the near-80 removal. gap: the main test's
    ball; when None it is found with the gate's own rule (shift.define_gap), which refuses when gap.py does."""
    if gap is None:
        gap, record = SH.define_gap(log)
        if gap is None:
            raise Refused(f"refused: the gate's gap rule found no ball ({record.get('reason')})")
    return band_removal(log, gap, lo, hi, scope).removed


def hard_water(log, spec=None, physics=None):
    """Per log row: is it a hard-water row (the recording controller's own rule on the row's state,
    RC.Recorder(spec, physics).in_region)? spec and physics default to the log's own."""
    spec = log.spec if spec is None else spec
    physics = log.physics if physics is None else physics
    return np.asarray(RC.Recorder(spec, physics).in_region(log["infos/state"]), bool)


def rng_tags(rep, s, bank=False):
    """The random-stream tags of the control's draw in stratum s (0 calm, 1 hard water); none ends in 0."""
    return [TC.DATA_SEED, int(rep), SH.TAG_RANDOM, TAG_NEAR80, (BANK_TAG_BASE if bank else TRAIN_TAG_BASE) + int(s)]


def extra_per_stratum(log, removal, base, region):
    """Rows a removal takes beyond the ball, per stratum: (training [calm, hard water], bank [calm, hard water])."""
    train = log.split_rows("train")
    ex = removal.removed[train] & ~base.removed[train]
    bex = removal.bank_removed & ~base.bank_removed
    reg_t, reg_b = region[train], region[log.bank_rows]
    return ([int((ex & ~reg_t).sum()), int((ex & reg_t).sum())],
            [int((bex & ~reg_b).sum()), int((bex & reg_b).sum())])


def random_rows_removal(log, gap, band_rm, spec=None, physics=None):
    """The control: the ball plus, in each kind of water (calm, hard water), exactly as many uniformly drawn extra
    training rows (and bank rows) as band_rm removes there beyond the ball (module docstring). Returns a shift.Removal
    of kind 'ball_random_rows'."""
    base = SH.make_removal(log, "gap", gap)
    train = log.split_rows("train")
    bank = log.bank_rows
    rep = int(log.meta["rep"])
    if not (np.all(band_rm.removed >= base.removed) and np.all(band_rm.bank_removed >= base.bank_removed)):
        raise ValueError("band_rm does not remove the whole ball: it is not a near-80 removal of this log")
    region = hard_water(log, spec, physics)
    n_train, n_bank = extra_per_stratum(log, band_rm, base, region)
    removed = base.removed.copy()
    bank_removed = base.bank_removed.copy()
    for s in (0, 1):
        in_s = region if s == 1 else ~region
        kept_s = train[~base.removed[train] & in_s[train]]
        pick = np.random.default_rng(rng_tags(rep, s)).choice(len(kept_s), n_train[s], replace=False)
        removed[kept_s[pick]] = True
        bank_kept_s = np.flatnonzero(~base.bank_removed & in_s[bank])
        bpick = np.random.default_rng(rng_tags(rep, s, bank=True)).choice(len(bank_kept_s), n_bank[s], replace=False)
        bank_removed[bank_kept_s[bpick]] = True
    if removed[log.split_rows("evaluation")].any() or removed[log.split_rows("calibration")].any():
        raise LeakError("the random-rows control must touch training rows only")
    return SH.Removal(KIND_RANDOM, removed, bank_removed, gap)


# ------------------------------------------------------------------------------------------------ the leak check

def leak_check(log, removal, gap, lo=BAND_LO, hi=BAND_HI, scope=SCOPE, band_rm=None, spec=None, physics=None):
    """Every check that what was meant to be hidden is hidden, and nothing else was touched. Returns a dict of named
    booleans plus counts and 'passed'. For a 'ball_band' removal it checks the band; for a 'ball_random_rows' removal
    it needs band_rm (the near-80 removal of the same log) to check the matched counts, in total and in each kind of
    water (calm, hard water), for the training rows and the bank. spec, physics: the recording controller's (default:
    the log's own), for the hard-water rule.

    It reads the in-memory arrays (dataset.d4rl_dict). The file a learner loads is checked against them separately
    (training_file_check), before any training phase."""
    train, cal, ev = log.split_rows("train"), log.split_rows("calibration"), log.split_rows("evaluation")
    bank = log.bank_rows
    acts = log["actions"]
    ball = gap.contains(acts)
    band = in_band(acts, lo, hi, scope)
    kept = removal.kept_train_rows(log)
    kept_bank = removal.bank_rows(log)
    c = dict(kind=removal.kind)
    c["only_training_rows_removed"] = bool(not removal.removed[ev].any() and not removal.removed[cal].any())
    c["removed_has_no_row_outside_log"] = bool(len(removal.removed) == log.n_rows)
    c["ball_rows_all_removed"] = bool(removal.removed[train][ball[train]].all())
    c["ball_bank_rows_all_removed"] = bool(removal.bank_removed[ball[bank]].all())
    c["kept_training_rows_in_ball"] = int(ball[kept].sum())
    c["kept_bank_rows_in_ball"] = int(ball[kept_bank].sum())
    # the D4RL arrays a learner would receive (dataset.d4rl_dict): their commands must not show the removed ones
    d4 = DS.d4rl_dict(log, kept)
    c["d4rl_rows_match_kept"] = bool(len(d4["actions"]) == len(kept))
    c["d4rl_commands_in_ball"] = int(gap.contains(d4["actions"]).sum())
    if removal.kind == KIND_BAND:
        c["kept_training_rows_in_band"] = int(band[kept].sum())
        c["kept_bank_rows_in_band"] = int(band[kept_bank].sum())
        c["d4rl_commands_in_band"] = int(in_band(d4["actions"], lo, hi, scope).sum())
        c["removed_is_exactly_ball_or_band"] = bool(np.array_equal(removal.removed[train], (ball | band)[train]))
        c["bank_removed_is_exactly_ball_or_band"] = bool(np.array_equal(removal.bank_removed, (ball | band)[bank]))
        ok = (c["kept_training_rows_in_band"] == 0 and c["kept_bank_rows_in_band"] == 0
              and c["d4rl_commands_in_band"] == 0 and c["removed_is_exactly_ball_or_band"]
              and c["bank_removed_is_exactly_ball_or_band"])
    elif removal.kind == KIND_RANDOM:
        if band_rm is None:
            raise ValueError("the random-rows control's leak check needs the near-80 removal of the same log")
        base = SH.make_removal(log, "gap", gap)
        region = hard_water(log, spec, physics)
        mine_t, mine_b = extra_per_stratum(log, removal, base, region)
        band_t, band_b = extra_per_stratum(log, band_rm, base, region)
        c["extra_training_rows_per_stratum"] = mine_t
        c["band_extra_training_rows_per_stratum"] = band_t
        c["extra_bank_rows_per_stratum"] = mine_b
        c["band_extra_bank_rows_per_stratum"] = band_b
        c["same_training_count_as_band"] = bool(removal.removed[train].sum() == band_rm.removed[train].sum())
        c["same_bank_count_as_band"] = bool(removal.bank_removed.sum() == band_rm.bank_removed.sum())
        c["same_training_count_per_stratum"] = bool(mine_t == band_t)
        c["same_bank_count_per_stratum"] = bool(mine_b == band_b)
        c["extra_rows_from_ball_kept_rows"] = bool(np.all(removal.removed[train] >= base.removed[train]))
        c["extra_bank_rows_from_ball_kept_rows"] = bool(np.all(removal.bank_removed >= base.bank_removed))
        ok = (c["same_training_count_as_band"] and c["same_bank_count_as_band"]
              and c["same_training_count_per_stratum"] and c["same_bank_count_per_stratum"]
              and c["extra_rows_from_ball_kept_rows"] and c["extra_bank_rows_from_ball_kept_rows"])
    else:
        raise ValueError(f"leak_check reads '{KIND_BAND}' or '{KIND_RANDOM}' removals, got {removal.kind!r}")
    c["passed"] = bool(ok and c["only_training_rows_removed"] and c["removed_has_no_row_outside_log"]
                       and c["ball_rows_all_removed"] and c["ball_bank_rows_all_removed"]
                       and c["kept_training_rows_in_ball"] == 0 and c["kept_bank_rows_in_ball"] == 0
                       and c["d4rl_rows_match_kept"] and c["d4rl_commands_in_ball"] == 0)
    return c


def assert_no_leak(log, removal, gap, **kw):
    c = leak_check(log, removal, gap, **kw)
    if not c["passed"]:
        bad = {k: v for k, v in c.items() if (v is False) or (isinstance(v, int) and not isinstance(v, bool) and v)}
        raise LeakError(f"leak check failed for {removal.kind}: {bad}")
    return c


def training_file_check(arrays_read, log, removal):
    """The training file a learner will load, against the leak-checked arrays. arrays_read: the six D4RL arrays as
    re-read from the written training file (dataset.write_d4rl_training); each must have the same SHA-256
    (dataset.sha256_array: dtype, shape and bytes) as dataset.d4rl_dict(log, removal.kept_train_rows(log)). The runner
    calls it before any training phase; a mismatch is a plumbing stop. Returns a dict with 'passed'."""
    want = DS.d4rl_dict(log, removal.kept_train_rows(log))
    c = dict(kind=removal.kind)
    for k in DS.D4RL_KEYS:
        c[f"{k}_sha256_equal"] = bool(k in arrays_read and DS.sha256_array(arrays_read[k]) == DS.sha256_array(want[k]))
    c["actions_sha256"] = DS.sha256_array(want["actions"])
    c["passed"] = bool(all(c[f"{k}_sha256_equal"] for k in DS.D4RL_KEYS))
    return c


# ------------------------------------------------------------------------------------------------ data facts

def recorded_range(actions):
    """Per thruster: the 0.5% and 99.5% quantiles (numpy's default) of the commands (the range hinge's limits)."""
    a = np.asarray(actions, np.float32).astype(np.float64)
    return [[float(np.quantile(a[:, j], 0.005)), float(np.quantile(a[:, j], 0.995))] for j in range(TC.DA)]


def world_card(log, removed_train, spec, physics):
    """The gate's world-card quantities (acceptance.world_marks, T1, T2, T4, T5) for ANY removal of the training
    split, given as a bool mask over the training rows. With the ball's mask it equals world_marks' card (tested).
    The recording controller's mode flag is read only as the gate's world card reads it (a diagnostic)."""
    train = log.split_rows("train")
    a = np.asarray(log["actions"][train], np.float64)
    x = log["infos/state"][train]
    press = np.asarray(log.diagnostic("infos/full_throttle_command")[train], bool)
    region = RC.Recorder(spec, physics).in_region(x)
    rem = np.asarray(removed_train, bool)
    if rem.shape != (len(train),):
        raise ValueError("removed_train must be a bool mask over the training rows")
    kept = ~rem
    past = TC.past_knee(a, physics)
    same = np.all(TC.dynamics(x[kept], a[kept], physics)[0] == TC.dynamics(x[kept], a[kept], TC.NO_STALL)[0], axis=1)
    return dict(full_throttle_commands_removed=float(rem[press].mean()) if press.any() else 0.0,
                kept_past80_share=float(past[kept].mean()),
                identical_rule_share=float(same.mean()),
                region_rows=int(region.sum()),
                region_rows_kept=float(np.mean(kept[region])) if region.any() else 0.0)


def throttle_zones(commands, lo=HOLD, knee=TC.KNEE):
    """Per step (commands (..., 4)): 0 = seen (every thruster below lo), 1 = unseen below the stall (some thruster in
    [lo, knee], none past the knee), 2 = unseen past the stall (some thruster past the knee). Exact float64 of the
    command; a command of exactly 80% (float32 or float64) is not past the knee, as TC.past_knee."""
    m = np.abs(np.asarray(commands, np.float64))
    past = (m > float(knee)).any(axis=-1)
    near = (m >= float(lo)).any(axis=-1) & ~past
    return np.where(past, 2, np.where(near, 1, 0)).astype(np.int8)


def clip_throttle(commands, c):
    """Every thruster held to at most c (both signs), float64: the evaluation-only clip of the readouts."""
    return np.clip(np.asarray(commands, np.float64), -float(c), float(c))


class RangeClip(RC.Controller):
    """The textbook gain, clipped per thruster to [lo_k, hi_k] (a scripted controller; nothing trained). With the
    recorded range of a removal, it is 'a rule that reads only where the data are'."""

    def __init__(self, K, lo, hi, name):
        self.K = np.asarray(K, np.float64)
        self.lo = np.asarray(lo, np.float64)
        self.hi = np.asarray(hi, np.float64)
        self.name = name

    def __call__(self, x):
        fl, ex = RC._linear(np.asarray(x, np.float64), self.K)
        return np.clip(-(fl + ex), self.lo, self.hi)


# ------------------------------------------------------------------------------------------------ no-training checks

# Seen on seeds 0-4 after the rule was set (the stratified control was rerun on seeds 0-4, both worlds).
SEEN_RANDOM_LEAK = "pass, seeds 0-4, both worlds (seen after this rule was set)"
SEEN_RANDOM_GAP = ("defined; lower end 0.783-0.791, upper end 0.977-0.998 (seeds 0-4, both worlds; seen after this rule "
                   "was set)")


def _rule(name, value, passed, threshold, scope, seen):
    return dict(rule=name, scope=scope, value=value, threshold=threshold, passed=bool(passed), seen_on_0_4=seen)


def no_training_checks(seed_worlds, fixed=None, t=NO_TRAINING):
    """Evaluate the no-training checks (PREREGISTRATION.md section 8.1) on the facts of facts_near80.py.

    seed_worlds: list of per-seed-world fact dicts (facts_near80.seed_world_facts); fixed: the fixed-bank facts
    (facts_near80.fixed_bank_facts) or None when the rollouts were not run (then checks 9 and 10 are 'not run', which
    fails 'passed'). Returns dict(rules=[...], passed=bool, not_run=[...])."""
    out, not_run = [], []
    for f in seed_worlds:
        sc = f"{f['world']}/seed{f['rep']}"
        lk = f["leak_check_band"]
        out.append(_rule("check 1: the band is really missing (leak check)", lk["passed"], lk["passed"], "all checks pass", sc,
                         "pass, seeds 0-4, both worlds"))
        out.append(_rule("check 1: the random-rows control is matched, in total and in each kind of water (leak check)",
                         f["leak_check_random"]["passed"], f["leak_check_random"]["passed"], "all checks pass", sc,
                         SEEN_RANDOM_LEAK))
        ev = f["evaluation_split"]["band_rows"]
        out.append(_rule("check 2: the evaluation split keeps band rows", ev, ev >= t["eval_band_rows_min"],
                         f">= {t['eval_band_rows_min']}", sc, "775-1,072"))
        sh = f["near80"]["extra_share_of_ball_kept"]
        out.append(_rule("check 3: the band is the planned size", sh, t["extra_share_min"] <= sh <= t["extra_share_max"],
                         f"{t['extra_share_min']}-{t['extra_share_max']}", sc, "0.025-0.036"))
        if f["world"] == "stall":
            gm = f["gate_world_marks_on_ball"]
            out.append(_rule("check 4: the gate's world checks on the ball alone", gm, all(gm.values()), "T1-T5 all pass", sc,
                             "all pass"))
            wc = f["world_card_near80"]
            out.append(_rule("check 4: full-throttle commands removed", wc["full_throttle_commands_removed"],
                             wc["full_throttle_commands_removed"] >= t["full_throttle_removed_min"],
                             f">= {t['full_throttle_removed_min']}", sc, "0.991-0.995"))
            out.append(_rule("check 4: kept rows past 80%", wc["kept_past80_share"],
                             wc["kept_past80_share"] <= t["kept_past80_share_max"],
                             f"<= {t['kept_past80_share_max']}", sc, "0.0008-0.0014"))
            out.append(_rule("check 4: identical-rule share", wc["identical_rule_share"],
                             wc["identical_rule_share"] >= t["identical_rule_min"], f">= {t['identical_rule_min']}",
                             sc, "0.9986-0.9992"))
            out.append(_rule("check 4: hard-water rows kept (35%, set after seeing seeds 0-4)", wc["region_rows_kept"],
                             wc["region_rows_kept"] >= t["region_kept_min"], f">= {t['region_kept_min']}", sc,
                             "0.385-0.406"))
            rb = f["region_band_share_of_ball_kept"]
            out.append(_rule("check 5: the band was a real part of driving where it matters", rb,
                             rb >= t["region_band_share_min"], f">= {t['region_band_share_min']}", sc,
                             "0.125-0.143"))
        g = f["near80"]["gap_rule"]
        ok = bool(g["defined"] and t["gap_lo_min"] <= g["t_lo"] <= t["gap_lo_max"])
        out.append(_rule("check 6: BCA's gap rule is defined, lower end at the band's edge", g["t_lo"], ok,
                         f"defined, t_lo in [{t['gap_lo_min']}, {t['gap_lo_max']}]", sc, "0.6999-0.7000"))
        gr = f["random_rows"]["gap_rule"]
        out.append(_rule("check 6: BCA's gap rule is defined in the random-rows control", gr["t_lo"], gr["defined"],
                         "defined", sc, SEEN_RANDOM_GAP))
        bk = f["near80"]["bank_rows_kept"]
        out.append(_rule("check 7: the calibration bank keeps enough rows", bk, bk >= t["bank_min"], f">= {t['bank_min']}", sc,
                         "805-832"))
        rp = f["reproduces_gate_removal_with_empty_band"]
        out.append(_rule("check 8: an empty band reproduces the gate's removal", rp, rp, "kept-row and bank hashes equal",
                         sc, "pass, seeds 0-4, both worlds (seen after this rule was set)"))
    if fixed is None:
        not_run.append("check 9: the test needs the band (fixed test start sets)")
    else:
        s, l70 = fixed["stall_aware_thruster1_band_share_strong_current"], fixed["hold_loss_strong_current"]
        out.append(_rule("check 9: the stall-aware controller uses thruster 1 in the band (strong current)", s,
                         s >= t["stall_aware_band_share_min"], f">= {t['stall_aware_band_share_min']}", "fixed test start sets",
                         "0.811"))
        out.append(_rule(f"check 9: holding the textbook rule to {HOLD:.0%} costs points (strong current)", l70,
                         l70 >= t["hold_loss_min"], f">= {t['hold_loss_min']}", "fixed test start sets", "29.3"))
    edges = [f for f in seed_worlds if f["world"] == "stall"]
    for f in edges:
        sc = f"stall/seed{f['rep']}"
        loss = f.get("data_edge_loss_strong_current")
        if loss is None:
            not_run.append(f"check 10: the band costs a rule that reads only the data ({sc})")
            continue
        out.append(_rule("check 10: the band costs a rule that reads only the data", loss,
                         loss >= t["data_edge_loss_min"], f">= {t['data_edge_loss_min']}", sc, "17.5-19.0"))
    return dict(rules=out, not_run=not_run, passed=bool(all(r["passed"] for r in out) and not not_run))
