"""Near-80 (version 9, DRAFT): BCA's two pooled held-out groups, the outcome-free count, the training-file check and
the no-training checks.

DRAFT, not frozen, awaiting the owner. Nothing here trains a learner or reads an outcome of a learner. Everything is a
pure NumPy function on a log (or on facts computed from one), tested on synthetic logs in test_near80.py and
test_v9.py. See ../PREREGISTRATION.md (sections 3 and 8).

BCA's two groups (the refusal fix, decide/refusal.md section 8), exactly:
  H = every row of the calibration split and the evaluation split whose command lies OUTSIDE the main test's ball
      (shift.Gap.contains on the stored float32 command at its exact float64 value; the same rule that empties the
      calibration bank of its ball rows, shift.make_removal). The bank is part of the calibration split.
  survivors = the main setting's kept training rows past t_lo (step 1's gap rule on those rows' commands; they are
      held out from BCA's second critic, which never trains on them).
  full-throttle group (tail) = the survivors, then the rows of H with throttle t(a) = max_k |a_k| > t_lo;
  rest = the rows of H with t(a) <= t_lo.
Every switch arm reads these same two groups (the switch phase scores them once). No scored number reads them: the
scores come from simulator rollouts on the four fixed test start sets, which are not recorded rows. The groups read
commands only, so their sizes are known before any learner is trained (outcome_free_count).
"""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]                       # .../bca (outputs/shift/beyond-the-data/near80_v9/code -> repository root)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

STATUS = "DRAFT, not frozen, awaiting the owner. Nothing has been trained."

# ------------------------------------------------------------------------------------------------ fixed settings

HELDOUT_SPLITS = ("calibration", "evaluation")
HOLD = 0.70                          # the "held to 70%" evaluation clip (a reported measure)
CAP_80 = TC.KNEE                     # the "capped at 80%" evaluation clip (a reported measure)
SOURCES = ("survivors", "bank", "calibration_other", "evaluation")
CERTIFIABLE_MIN = 36                 # calibration.reference.certifiable's bar at 0.1 / 0.95 / 1,000 draws
CERT_SETTINGS = dict(alpha=0.1, credibility=0.95, draws=1000)

EXPLORATORY_SEEDS = (0, 1, 2, 3, 4)  # seen; the facts script reads them (commands only)
SPENT_SEEDS = (10, 11, 12, 13, 14)   # version 8's fresh seeds, spent; the facts script may read them (commands only)
RESERVED_SEEDS = (5, 6, 7, 8, 9)     # version 6/7's confirmatory seeds; always refused

STEP1_DIR = ROOT / "outputs" / "shift" / "beyond-the-data" / "restructure" / "step1"

# The no-training checks (section 8.1): the gate's data checks, step 1's gap rule, the groups' construction, and on
# seeds 0-4 the bank-only groups against step 1's recorded group sizes.
NO_TRAINING = dict(gap_rule_defined=True, groups_heldout_only=True, bank_only_reproduces_step1=True)


class Refused(Exception):
    """A guard refused: nothing was computed or written."""


class LeakError(AssertionError):
    """A group check failed: a group holds a row it must not hold."""


# ------------------------------------------------------------------------------------------------ seeds and paths

def check_seeds(seeds, allow_spent=False):
    """The facts script reads seeds 0-4 (and, with allow_spent, version 8's spent seeds 10-14) only, commands only.
    Version 9's fresh seeds are never generated, loaded or read by it."""
    seeds = [int(s) for s in seeds]
    for s in seeds:
        if s in RESERVED_SEEDS:
            raise Refused(f"refused: seed {s} belongs to version 6/7's confirmation (5-9)")
        if s in SPENT_SEEDS and allow_spent:
            continue
        if s not in EXPLORATORY_SEEDS:
            raise Refused(f"refused: seed {s} is not one of the seen seeds 0-4" +
                          ("" if allow_spent else " (version 8's spent seeds 10-14 need --spent)"))
    return seeds


def guard_out(path):
    """Outputs go outside the repository (scratch) or under runs/shift_explore/near80_v9/ only. Never into a pinned
    folder, never into this code folder, never over an existing file."""
    p = Path(path).resolve()
    official = ROOT / "runs" / "shift_explore" / "near80_v9"
    if ROOT == p or ROOT in p.parents:
        if not (official == p.parent or official in p.parents):
            raise Refused(f"refused: {p} is inside the repository but not under {official}")
    if p.exists():
        raise Refused(f"refused: {p} exists (outputs are write-once)")
    return p


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def rows_sha256(rows):
    return DS.sha256_array(np.asarray(rows, np.int64))


def load_step1lib():
    """Restructured BCA's step-1 library, executed from bytes verified against step 1's code manifest (never edited).
    It holds the throttle index, the gap rule, the scores and BCA's certificate."""
    path = STEP1_DIR / "step1lib.py"
    data = path.read_bytes()
    want = json.loads((STEP1_DIR / "code_manifest.json").read_text())["files"]["step1lib.py"]
    if sha256_bytes(data) != want:
        raise Refused("refused: step1lib.py does not match step 1's code manifest")
    mod = types.ModuleType("near80v9_step1lib")
    mod.__file__ = str(path)
    exec(compile(data, str(path), "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------------------ commands

def throttle_abs(actions):
    """|a| at the exact float64 value of the stored float32 command, shape (n, 4)."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != TC.DA:
        raise ValueError(f"commands must have shape (n, {TC.DA}), got {a.shape}")
    return np.abs(np.asarray(a, np.float32).astype(np.float64))


def throttle(actions):
    """Step 1's throttle index t(a) = max_k |a_k|, float64 of the stored float32 command (step1lib.throttle)."""
    return throttle_abs(actions).max(axis=1)


def throttle_zones(commands, lo=HOLD, knee=TC.KNEE):
    """Per step (commands (..., 4)): 0 = every thruster below lo, 1 = some thruster in [lo, knee], none past the knee,
    2 = some thruster past the knee. Exact float64 of the command; a command of exactly 80% is not past the knee."""
    m = np.abs(np.asarray(commands, np.float64))
    past = (m > float(knee)).any(axis=-1)
    near = (m >= float(lo)).any(axis=-1) & ~past
    return np.where(past, 2, np.where(near, 1, 0)).astype(np.int8)


def clip_throttle(commands, c):
    """Every thruster held to at most c (both signs), float64."""
    return np.clip(np.asarray(commands, np.float64), -float(c), float(c))


def recorded_range(actions):
    """Per thruster: the 0.5% and 99.5% quantiles (numpy's default) of the commands (the range hinge's limits)."""
    a = np.asarray(actions, np.float32).astype(np.float64)
    return [[float(np.quantile(a[:, j], 0.005)), float(np.quantile(a[:, j], 0.995))] for j in range(TC.DA)]


# ------------------------------------------------------------------------------------------------ the pooled groups

def heldout_rows(log, gap, splits=HELDOUT_SPLITS):
    """H: every row of the held-out splits whose command lies outside the ball, ascending (module docstring)."""
    rows = np.sort(np.concatenate([log.split_rows(s) for s in splits])).astype(np.int64)
    inside = np.asarray(gap.contains(log["actions"][rows]), bool)
    return rows[~inside]


def gap_rule_of_rows(log, rows, step1, gap_rule):
    """Step 1's gap rule on the commands of `rows` (the main setting's kept training rows)."""
    return step1.gap_rule(step1.throttle(np.asarray(log["actions"][np.asarray(rows, np.int64)])), **gap_rule)


def survivors_of(kept_rows, gr):
    """The survivors as log rows: the kept training rows the gap rule puts in the full-throttle group."""
    return np.asarray(np.asarray(kept_rows, np.int64)[np.asarray(gr["tail"], bool)], np.int64)


def pooled_groups(log, gap, survivors, t_lo, splits=HELDOUT_SPLITS):
    """(tail rows, rest rows): the survivors then the rows of H past t_lo; the rows of H at or below t_lo (module
    docstring). Log rows, int64; the survivors first in their given order, H's rows ascending."""
    h = heldout_rows(log, gap, splits)
    past = throttle(log["actions"][h]) > float(t_lo)
    surv = np.asarray(survivors, np.int64)
    return np.concatenate([surv, h[past]]).astype(np.int64), h[~past].astype(np.int64)


def bank_only_groups(log, gap, survivors, t_lo):
    """The as-built groups of version 8 and step 1 (the survivors and the kept bank rows past t_lo; the kept bank rows
    at or below t_lo): used only to tie this code to step 1's recorded group sizes (no-training check 4)."""
    bank = np.asarray(log.bank_rows, np.int64)
    kept = bank[~np.asarray(gap.contains(log["actions"][bank]), bool)]
    past = throttle(log["actions"][kept]) > float(t_lo)
    return (np.concatenate([np.asarray(survivors, np.int64), kept[past]]).astype(np.int64),
            kept[~past].astype(np.int64))


def source_of(log, rows, survivors):
    """Per row: 'survivors' (a survivor), 'bank' (a calibration-bank row), 'calibration_other' (another calibration
    row) or 'evaluation'; anything else is 'other' (never in a group)."""
    rows = np.asarray(rows, np.int64)
    surv = np.isin(rows, np.asarray(survivors, np.int64))
    bank = np.isin(rows, np.asarray(log.bank_rows, np.int64))
    split = np.asarray(log["infos/split"])[rows]
    out = np.full(len(rows), "other", dtype=object)
    out[(split == DS.SPLITS.index("evaluation")) & ~surv] = "evaluation"
    out[(split == DS.SPLITS.index("calibration")) & ~bank & ~surv] = "calibration_other"
    out[bank & ~surv] = "bank"
    out[surv] = "survivors"
    return out


def group_sources(log, rows, survivors):
    """{n, by_source {survivors, bank, calibration_other, evaluation}, episodes (distinct), rows_sha256}."""
    rows = np.asarray(rows, np.int64)
    src = source_of(log, rows, survivors)
    eps = np.asarray(log["infos/episode"])[rows]
    return dict(n=int(len(rows)), by_source={s: int(np.sum(src == s)) for s in SOURCES},
                other=int(np.sum(src == "other")), episodes=int(len(np.unique(eps))), rows_sha256=rows_sha256(rows))


def one_per_episode(log, rows):
    """The positions (into rows) of the first row of each episode, in rows' order: the reported one-row-per-episode
    variant of a group (outcome-free: the lowest log row of each episode in the group)."""
    rows = np.asarray(rows, np.int64)
    eps = np.asarray(log["infos/episode"])[rows]
    keep, seen = [], set()
    for i in np.argsort(rows, kind="stable"):
        e = int(eps[i])
        if e not in seen:
            seen.add(e)
            keep.append(int(i))
    return np.sort(np.asarray(keep, np.int64))


def group_checks(log, gap, kept_train_rows, survivors, tail, rest, t_lo, splits=HELDOUT_SPLITS):
    """Every check that the two groups hold exactly what they must (no-training check 3): no row inside the ball; no
    training row but the survivors; the survivors are exactly the kept training rows past t_lo; every other row is in
    a held-out split; the groups are disjoint and together are exactly H plus the survivors; each row of H sits in the
    group its throttle says. Returns a dict of named booleans and 'passed'."""
    tail, rest = np.asarray(tail, np.int64), np.asarray(rest, np.int64)
    surv = np.asarray(survivors, np.int64)
    kept = np.asarray(kept_train_rows, np.int64)
    h = heldout_rows(log, gap, splits)
    train = log.split_rows("train")
    both = np.concatenate([tail, rest])
    held = np.isin(both, np.concatenate([log.split_rows(s) for s in splits]))
    t_all = throttle(log["actions"][both]) if len(both) else np.zeros(0)
    c = dict(
        no_row_inside_the_ball=bool(not np.asarray(gap.contains(log["actions"][both]), bool).any()) if len(both)
        else True,
        training_rows_are_the_survivors_only=bool(np.array_equal(np.sort(both[np.isin(both, train)]), np.sort(surv))),
        survivors_are_kept_rows_past_t_lo=bool(np.array_equal(np.sort(surv),
                                                              np.sort(kept[throttle(log["actions"][kept]) > t_lo]))),
        survivors_first_in_the_tail=bool(np.array_equal(tail[:len(surv)], surv)),
        every_other_row_held_out=bool(np.all(held | np.isin(both, surv))),
        groups_disjoint=bool(len(np.intersect1d(tail, rest)) == 0 and len(np.unique(both)) == len(both)),
        groups_are_heldout_plus_survivors=bool(np.array_equal(np.sort(both), np.sort(np.concatenate([h, surv])))),
        tail_past_t_lo=bool(np.all(t_all[:len(tail)] > t_lo)),
        rest_at_or_below_t_lo=bool(np.all(t_all[len(tail):] <= t_lo)),
    )
    c["passed"] = bool(all(c.values()))
    return c


def certifiable(n):
    """BCA's own rule (calibration.reference.certifiable at 0.1 / 0.95 / 1,000): a uniform-weight group of n scores
    fails to certify with probability at most 1e-6."""
    from calibration import reference
    return bool(reference.certifiable(int(n), types.SimpleNamespace(**CERT_SETTINGS)))


def outcome_free_count(log, gap, kept_train_rows, step1, gap_rule):
    """The pre-training count of BCA's two groups (section 8.1, reported): commands only, before any learner is
    trained. Returns {defined, t_lo, t_up, n_tail, n_rest, tail {...}, rest {...}, certifiable, below_bar, ...}; when
    the gap rule is not defined the switch will be off and nothing is counted."""
    kept = np.asarray(kept_train_rows, np.int64)
    gr = gap_rule_of_rows(log, kept, step1, gap_rule)
    base = dict(defined=bool(gr["defined"]), t_lo=float(gr["t_lo"]) if gr.get("t_lo") is not None else None,
                t_up=None if gr.get("t_up") is None else float(gr["t_up"]), certifiable_bar=CERTIFIABLE_MIN,
                reads="commands only (no reward, no next state, no learner)")
    if not gr["defined"]:
        return dict(base, n_tail=0, n_rest=0, reason=gr.get("reason"), certifiable=False, below_bar=None)
    surv = survivors_of(kept, gr)
    tail, rest = pooled_groups(log, gap, surv, gr["t_lo"])
    checks = group_checks(log, gap, kept, surv, tail, rest, gr["t_lo"])
    ope = one_per_episode(log, tail)
    bt, br = bank_only_groups(log, gap, surv, gr["t_lo"])
    return dict(base, n_survivors=int(len(surv)), n_tail=int(len(tail)), n_rest=int(len(rest)),
                tail=group_sources(log, tail, surv), rest=group_sources(log, rest, surv),
                tail_one_per_episode=int(len(ope)), certifiable=certifiable(len(tail)),
                below_bar=bool(len(tail) < CERTIFIABLE_MIN), group_checks=checks,
                bank_only=dict(n_tail=int(len(bt)), n_rest=int(len(br)),
                               note="the as-built groups of version 8 and step 1, for comparison only"))


# ------------------------------------------------------------------------------------------------ training files

def training_file_check(arrays_read, log, removal):
    """The training file a learner will load, against the in-memory arrays. arrays_read: the six D4RL arrays as
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


# ------------------------------------------------------------------------------------------------ no-training checks

def _rule(name, value, passed, threshold, scope, seen):
    return dict(rule=name, scope=scope, value=value, threshold=threshold, passed=bool(passed), seen_on_0_4=seen)


def no_training_checks(seed_worlds):
    """The no-training checks (PREREGISTRATION.md section 8.1) on the facts of facts_v9.seed_world_facts. Returns
    dict(rules=[...], passed=bool, not_run=[...]). The outcome-free count is reported beside them, never a rule."""
    out, not_run = [], []
    for f in seed_worlds:
        sc = f"{f['world']}/seed{f['rep']}"
        g = f["gate_checks"]
        out.append(_rule("check 1: the gate's data checks (the ball's leak checks; the five world checks in the "
                         "stalling world)", g["passed"], g["passed"], "all pass", sc, "pass, seeds 0-4, both worlds"))
        c = f["count"]
        out.append(_rule("check 2: BCA's gap rule is defined on the main setting's kept commands", c["defined"],
                         c["defined"], "defined", sc, "defined, seeds 0-4, both worlds"))
        gc = (c.get("group_checks") or {}).get("passed")
        out.append(_rule("check 3: BCA's two groups hold held-out rows and the survivors only", gc, gc is True,
                         "all pass", sc, "to be seen when the frozen code runs"))
        rp = f.get("bank_only_reproduces_step1")
        if rp is None:
            not_run.append(f"check 4: the bank-only groups reproduce step 1's group sizes ({sc})")
        else:
            out.append(_rule("check 4: the bank-only groups reproduce step 1's recorded group sizes", rp, rp is True,
                             "equal", sc, "to be seen when the frozen code runs"))
    return dict(rules=out, not_run=not_run, passed=bool(all(r["passed"] for r in out) and not not_run))
