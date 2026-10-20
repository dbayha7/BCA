"""Near-80 (version 9, main setting): the pooled groups (the refusal fix), the outcome-free count, the training-file
check and the small readout helpers. DRAFT, not frozen.

Nothing here trains a learner, fits a BCA band or reads an outcome of a learner. Everything is a pure NumPy function
on a log, tested on synthetic logs in test_near80.py. See ../PREREGISTRATION.md (sections 3, 4 and 8).

The pooled groups, exactly (section 3):
  The removal is the main test's ball (shift.make_removal(log, 'gap', gap)), unchanged. Step 1's gap rule on the
  ball-kept training commands gives t_lo (the lower end of the gap). Then
    the held-out rows  = every row of the calibration split and of the evaluation split whose command is outside the
                         ball (the ball's rows are hidden everywhere, as in version 8's bank);
    the full-throttle group = the training-split survivors (ball-kept training rows with throttle t > t_lo, which step
                         1 already scored and holds out from BCA's second critic) together with the held-out rows with
                         t > t_lo;
    the rest           = the held-out rows with t <= t_lo.
  Version 8's groups were the survivors plus the calibration bank (about 800 rows); the bank is a subset of the
  calibration split, so version 8's groups are a subset of these. The training rows, the test start sets and every
  rival's own rule are unchanged; no scored test reads the pooled rows (the four fixed test start sets are rollouts).
  Every switch arm reads the same groups (policy_phase.arm_row_from reads one switch record per world).

The outcome-free count (section 8.2): per seed and world, before any training, the sizes of the two pooled groups
and whether the full-throttle group reaches CERT_MIN_ROWS (29 rows: 0.9^29 < 0.05, the smallest size at which BCA's
1,000 posterior draws can certify the group). Reported only; it reads commands and split labels, never a reward or a
critic.
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import near80_constants as K  # noqa: E402

ROOT = Path(K.ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import recorder as RC  # noqa: E402
from experiments.shift import shift as SH  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

STATUS = K.STATUS_LINE_DRAFT
HOLD = K.HOLD
CAP_80 = TC.KNEE
STEP1_DIR = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step1"
SMOKE_SEEDS = K.SMOKE_SEEDS
FRESH_SEEDS = K.FRESH
POOLED_SPLITS = K.POOLED_SPLITS


class Refused(Exception):
    """A guard refused: nothing was computed or written."""


# ------------------------------------------------------------------------------------------------ seeds and paths

def check_seeds(seeds):
    """The facts script reads the gate's seeds 0-4 only (their logs are regenerated from the gate's records). The
    fresh seeds are never generated, loaded or read by it."""
    seeds = [int(s) for s in seeds]
    for s in seeds:
        if s in K.FRESH:
            raise Refused(f"refused: seed {s} is a fresh seed of this version; the facts script never generates, "
                          "loads or reads it")
        if s not in range(0, 5):
            raise Refused(f"refused: seed {s} is not one of the gate's seen seeds 0-4")
    return seeds


def guard_out(path):
    """Outputs go outside the repository (scratch) or under the official output folder only. Never into a pinned
    folder, never into this code folder, never over an existing file."""
    import os
    p = Path(path).resolve()

    def inside(child, parent):
        """Lexically or through links (a junction on Windows resolves to its target)."""
        for c, r in ((Path(os.path.abspath(str(child))), Path(os.path.abspath(str(parent)))),
                     (Path(os.path.realpath(str(child))), Path(os.path.realpath(str(parent))))):
            if c == r or r in c.parents:
                return True
        return False

    if inside(path, ROOT) and not inside(path, K.OUT):
        raise Refused(f"refused: {p} is inside the repository but not under {Path(K.OUT)}")
    if p.exists():
        raise Refused(f"refused: {p} exists (outputs are write-once)")
    return p


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_step1lib():
    """Restructured BCA's step-1 library, executed from bytes verified against step 1's code manifest (never edited).
    It holds the throttle index, the gap rule that places BCA's ramp, the overestimate, the certificate and the rank
    test. Its hard-coded ROOT is set to this host's repository after the exec (its functions read it at call time)."""
    path = STEP1_DIR / "step1lib.py"
    data = path.read_bytes()
    want = json.loads((STEP1_DIR / "code_manifest.json").read_text())["files"]["step1lib.py"]
    if sha256_bytes(data) != want:
        raise Refused("refused: step1lib.py does not match step 1's code manifest")
    mod = types.ModuleType("near80_step1lib")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    mod.ROOT = ROOT
    return mod


# ------------------------------------------------------------------------------------------------ the pooled groups

def throttle_index(actions):
    """Step 1's throttle index t(a) = max_k |a_k| over the four thrusters, float64 of the stored float32 command."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != TC.DA:
        raise ValueError(f"commands must have shape (n, {TC.DA}), got {a.shape}")
    return np.abs(np.asarray(a, np.float32)).max(axis=1).astype(np.float64)


def heldout_rows(log, gap, splits=POOLED_SPLITS):
    """The held-out recorded rows outside the removal ball: every row of the named splits (calibration, evaluation)
    whose command is not inside the ball, ascending log indices."""
    rows = np.concatenate([log.split_rows(s) for s in splits])
    rows = np.sort(np.asarray(rows, np.int64))
    inside = gap.contains(log["actions"][rows])
    return rows[~inside]


def pooled_groups(log, gap, removal, t_lo, step1=None):
    """The pooled groups of one log (module docstring). removal: the main test's removal (its kept training rows give
    the survivors). t_lo: the gap rule's lower end on the ball-kept training commands. Returns a dict with the rows of
    each part (log indices), their counts and SHA-256s, outcome-free."""
    t = throttle_index(log["actions"])
    kept = np.asarray(removal.kept_train_rows(log), np.int64)
    surv = kept[t[kept] > float(t_lo)]
    held = heldout_rows(log, gap)
    ft = t[held] > float(t_lo)
    tail_held, rest_held = held[ft], held[~ft]
    if gap.contains(log["actions"][held]).any():
        raise AssertionError("a held-out row inside the ball reached the pooled groups")
    if np.intersect1d(surv, held).size:
        raise AssertionError("a training row counted among the held-out rows")
    return dict(t_lo=float(t_lo), splits=list(POOLED_SPLITS), survivors=surv, heldout_tail=tail_held,
                heldout_rest=rest_held, n_survivors=int(len(surv)), n_heldout_tail=int(len(tail_held)),
                n_heldout_rest=int(len(rest_held)), n_full_throttle=int(len(surv) + len(tail_held)),
                n_rest=int(len(rest_held)), n_heldout=int(len(held)),
                n_heldout_in_ball=int(gap.contains(log["actions"][np.concatenate(
                    [log.split_rows(s) for s in POOLED_SPLITS])]).sum()),
                survivors_sha256=DS.sha256_array(surv), heldout_tail_sha256=DS.sha256_array(tail_held),
                heldout_rest_sha256=DS.sha256_array(rest_held))


def certification_expectation(n_tail, cfg=K.CERT, min_rows=K.CERT_MIN_ROWS):
    """What the group size alone says about BCA's certificate (reported, outcome-free): a posterior draw is finite
    with probability 1 - (1 - alpha)^n, the group is certified when at least `credibility` of the draws are finite,
    so the expected finite share and whether n reaches the smallest certifiable size."""
    n = int(n_tail)
    alpha = float(cfg["alpha"])
    expected = 1.0 - (1.0 - alpha) ** n
    return dict(n_full_throttle=n, expected_finite_share=expected, needed_finite_share=float(cfg["credibility"]),
                min_rows=int(min_rows), reaches_min_rows=bool(n >= min_rows),
                note="reported only: the certificate itself draws; a group below min_rows cannot be certified, one "
                     "at or above it usually is (certain only from about 33-36 rows at 1,000 draws)")


def pooled_count_record(log, gap, removal, step1):
    """The outcome-free count of one seed-world (section 8.2): the gap rule on the ball-kept training commands (step
    1's own function, verified bytes), the pooled groups' sizes and the certification expectation. No reward, no
    critic, no learner."""
    kept = removal.kept_train_rows(log)
    gr = step1.gap_rule(step1.throttle(log["actions"][kept]), **K.GAP_RULE)
    out = dict(gap_rule=dict(defined=bool(gr["defined"]), t_lo=None if gr["t_lo"] is None else float(gr["t_lo"]),
                             t_up=None if gr.get("t_up") is None else float(gr["t_up"]), n_tail=int(gr["n_tail"]),
                             gap=float(gr["gap"]), reason=gr.get("reason")),
               defined=bool(gr["defined"]), groups=None, certification=None,
               version8_groups=None)
    if not gr["defined"]:
        out["note"] = "the gap rule is not defined: BCA's switch is off and its groups are empty"
        return out
    g = pooled_groups(log, gap, removal, gr["t_lo"], step1)
    out["groups"] = {k: v for k, v in g.items() if not isinstance(v, np.ndarray)}
    out["certification"] = certification_expectation(g["n_full_throttle"])
    bank = np.asarray(removal.bank_rows(log), np.int64)
    t = throttle_index(log["actions"])
    out["version8_groups"] = dict(n_full_throttle=int(g["n_survivors"] + int((t[bank] > gr["t_lo"]).sum())),
                                  n_rest=int((t[bank] <= gr["t_lo"]).sum()),
                                  note="version 8's groups (the survivors and the calibration bank), for scale")
    return out


# ------------------------------------------------------------------------------------------------ the training file

def training_file_check(arrays_read, log, removal):
    """The training file a learner will load, against the kept rows' arrays. arrays_read: the six D4RL arrays as
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


# ------------------------------------------------------------------------------------------------ readout helpers

def recorded_range(actions):
    """Per thruster: the 0.5% and 99.5% quantiles (numpy's default) of the commands (the range hinge's limits)."""
    a = np.asarray(actions, np.float32).astype(np.float64)
    return [[float(np.quantile(a[:, j], 0.005)), float(np.quantile(a[:, j], 0.995))] for j in range(TC.DA)]


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


def hard_water(log, spec=None, physics=None):
    """Per log row: is it a hard-water row (the recording controller's own rule on the row's state)? Reported."""
    spec = log.spec if spec is None else spec
    physics = log.physics if physics is None else physics
    return np.asarray(RC.Recorder(spec, physics).in_region(log["infos/state"]), bool)


def finite_share_bound(n, alpha=K.CERT["alpha"]):
    """1 - (1 - alpha)^n: the expected share of finite posterior draws for a group of n rows."""
    return 1.0 - (1.0 - float(alpha)) ** int(n)


def min_rows_for(credibility=K.CERT["credibility"], alpha=K.CERT["alpha"]):
    """The smallest n with (1 - alpha)^n < 1 - credibility (29 at alpha 0.1, credibility 0.95)."""
    n = 1
    while (1.0 - float(alpha)) ** n >= 1.0 - float(credibility):
        n += 1
    return n


assert min_rows_for() == K.CERT_MIN_ROWS, "CERT_MIN_ROWS must be the smallest certifiable group size"
assert math.isclose(finite_share_bound(29), 1 - 0.9 ** 29)
