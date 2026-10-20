"""Near-80 (version 9): BCA's two pooled held-out groups, the outcome-free count with the held-out top-up, the
textbook split-conformal quantiles, the exact Wilks twin, the one-row-per-episode variant, the training-file check and
the no-training checks.

Nothing here trains a learner or reads an outcome of a learner. Everything is a pure NumPy function on a log (or on
facts computed from one), tested on synthetic logs in test_near80.py and test_v9.py. See ../PREREGISTRATION.md
(sections 3 and 8).

BCA's two groups (DECISION.md section 2.2; PREREGISTRATION.md section 3.1), exactly:
  H = every row of the calibration split and the evaluation split whose command lies OUTSIDE the main test's ball
      (shift.Gap.contains on the stored float32 command at its exact float64 value; the same rule that empties the
      calibration bank of its ball rows, shift.make_removal), plus every top-up row (topup.py) outside the ball. The
      bank is part of the calibration split.
  survivors = the main setting's kept training rows past t_lo (step 1's gap rule on those rows' commands; they are
      held out from BCA's second critic, which never trains on them).
  full-throttle group (tail) = the survivors, then the rows of H with throttle t(a) = max_k |a_k| > t_lo;
  rest = the rows of H with t(a) <= t_lo.
Top-up rows are addressed through GroupView: a log's rows keep their indices, block k's rows follow the log's
(index n_rows + (k - 1) * 10,000 + i) and its episodes follow the log's episodes. Every switch reads these same two
groups (the switch phase scores them once). No scored number reads them: the scores come from simulator rollouts on
the four fixed test start sets, which are not recorded rows. The groups read commands only, so their sizes are known
before any learner is trained (outcome_free_count).
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
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.shift import dataset as DS  # noqa: E402
from experiments.shift import thruster_chain as TC  # noqa: E402

STATUS = "DRAFT. Not frozen. Nothing has been trained."          # PREREGISTRATION.md line 3 while a draft

# ------------------------------------------------------------------------------------------------ fixed settings

HELDOUT_SPLITS = ("calibration", "evaluation")
HOLD = 0.70                          # the "held to 70%" evaluation clip (a reported measure)
CAP_80 = TC.KNEE                     # the "capped at 80%" evaluation clip (a reported measure)
SOURCES = ("survivors", "bank", "calibration_other", "evaluation", "topup")
CERTIFIABLE_MIN = 36                 # calibration.reference.certifiable's bar at 0.1 / 0.95 / 1,000 draws
EXACT_FLOOR = 29                     # the smallest group that can certify at all (reported)
CERT_SETTINGS = dict(alpha=0.1, credibility=0.95, draws=1000)
TOPUP_SPLIT_CODE = 3                 # infos/split of a top-up row in GroupView (the log's splits are 0-2)
VIEW_KEYS = ("observations", "actions", "rewards", "next_observations", "terminals", "infos/state", "infos/next_state",
             "infos/episode", "infos/split")

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


# ------------------------------------------------------------------------------------------------ the groups' view

class GroupView:
    """A log with its top-up blocks (topup.block_record dicts, in block order) behind it: what the groups read.
    view[key] for VIEW_KEYS is the log's array followed by the blocks' (block k's rows at n_log + (k - 1) * rows per
    block + i; its episodes after the log's; infos/split TOPUP_SPLIT_CODE); split_rows, bank_rows, meta and the
    diagnostic counter are the log's. With no block the view is the log itself (no copy)."""

    def __init__(self, log, blocks=()):
        self.log, self.blocks = log, list(blocks or [])
        self.n_log = int(log.n_rows)
        self.meta = log.meta
        self._cache = {}

    @property
    def n_rows(self):
        return self.n_log + sum(int(b["rows"]) for b in self.blocks)

    @property
    def bank_rows(self):
        return self.log.bank_rows

    @property
    def diagnostic_reads(self):
        return self.log.diagnostic_reads

    def split_rows(self, split):
        return self.log.split_rows(split)

    def topup_rows(self):
        return np.arange(self.n_log, self.n_rows, dtype=np.int64)

    def __getitem__(self, key):
        if not self.blocks:
            return self.log[key]
        if key not in VIEW_KEYS:
            raise KeyError(f"{key} is not an array of the groups' view")
        if key not in self._cache:
            base = np.asarray(self.log[key])
            parts = [base]
            if key == "infos/episode":
                first = int(base.max()) + 1 if len(base) else 0
                for i, b in enumerate(self.blocks):
                    per = int(b["rows"]) // int(b["episodes"])
                    parts.append((first + i * int(b["episodes"]) + np.arange(int(b["rows"])) // per).astype(base.dtype))
            elif key == "infos/split":
                parts += [np.full(int(b["rows"]), TOPUP_SPLIT_CODE, base.dtype) for b in self.blocks]
            else:
                parts += [np.asarray(b["arrays"][key], base.dtype) for b in self.blocks]
            arr = np.concatenate(parts)
            arr.setflags(write=False)
            self._cache[key] = arr
        return self._cache[key]


def view(log, blocks=()):
    return log if not blocks else GroupView(log, blocks)


def topup_rows_of(lg):
    return lg.topup_rows() if isinstance(lg, GroupView) else np.zeros(0, np.int64)


def n_log_rows(lg):
    return lg.n_log if isinstance(lg, GroupView) else int(lg.n_rows)


# ------------------------------------------------------------------------------------------------ the pooled groups

def heldout_rows(log, gap, splits=HELDOUT_SPLITS):
    """H: every row of the held-out splits, and every top-up row of a GroupView, whose command lies outside the ball,
    ascending (module docstring)."""
    rows = np.concatenate([log.split_rows(s) for s in splits] + [topup_rows_of(log)])
    rows = np.sort(rows).astype(np.int64)
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
    docstring). Rows of the log (or of a GroupView), int64; the survivors first in their given order, H ascending."""
    h = heldout_rows(log, gap, splits)
    past = throttle(log["actions"][h]) > float(t_lo)
    surv = np.asarray(survivors, np.int64)
    return np.concatenate([surv, h[past]]).astype(np.int64), h[~past].astype(np.int64)


def tail_count(log, gap, survivors, t_lo, splits=HELDOUT_SPLITS):
    """The size of the pooled full-throttle group (commands only)."""
    h = heldout_rows(log, gap, splits)
    return int(len(survivors) + int(np.sum(throttle(log["actions"][h]) > float(t_lo))))


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
    row), 'evaluation' or 'topup' (a top-up row); anything else is 'other' (never in a group)."""
    rows = np.asarray(rows, np.int64)
    n_log = n_log_rows(log)
    in_log = rows < n_log
    surv = np.isin(rows, np.asarray(survivors, np.int64))
    bank = np.isin(rows, np.asarray(log.bank_rows, np.int64))
    split = np.full(len(rows), -1, np.int64)
    split[in_log] = np.asarray(log["infos/split"])[rows[in_log]]
    out = np.full(len(rows), "other", dtype=object)
    out[(split == DS.SPLITS.index("evaluation")) & ~surv] = "evaluation"
    out[(split == DS.SPLITS.index("calibration")) & ~bank & ~surv] = "calibration_other"
    out[bank & ~surv] = "bank"
    out[~in_log & (rows < log.n_rows)] = "topup"
    out[surv] = "survivors"
    return out


def group_sources(log, rows, survivors):
    """{n, by_source {survivors, bank, calibration_other, evaluation, topup}, episodes (distinct), distinct_share,
    rows_sha256}."""
    rows = np.asarray(rows, np.int64)
    src = source_of(log, rows, survivors)
    eps = np.asarray(log["infos/episode"])[rows]
    n_eps = int(len(np.unique(eps)))
    return dict(n=int(len(rows)), by_source={s: int(np.sum(src == s)) for s in SOURCES},
                other=int(np.sum(src == "other")), episodes=n_eps,
                distinct_share=float(n_eps / len(rows)) if len(rows) else None, rows_sha256=rows_sha256(rows))


def one_per_episode(log, rows, tags):
    """The positions (into rows, ascending) of one row per episode, chosen uniformly within each episode with
    numpy.random.default_rng(tags), episodes in ascending order: the reported one-row-per-episode variant of a group
    (the pinned preview's rule; tags = [20261020, 9901, seed, WORLD_TAG[world], 1] for the full-throttle group, 2 for
    the rest). Outcome-free: it reads the rows' episodes only."""
    rows = np.asarray(rows, np.int64)
    eps = np.asarray(log["infos/episode"])[rows]
    r = np.random.default_rng(list(tags))
    keep = []
    for e in np.unique(eps):
        idx = np.flatnonzero(eps == e)
        keep.append(int(idx[r.integers(len(idx))]))
    return np.sort(np.asarray(keep, np.int64))


def group_checks(log, gap, kept_train_rows, survivors, tail, rest, t_lo, splits=HELDOUT_SPLITS):
    """Every check that the two groups hold exactly what they must (no-training check 3): no row inside the ball; no
    training row but the survivors; the survivors are exactly the kept training rows past t_lo; every other row is in
    a held-out split or the top-up; the groups are disjoint and together are exactly H plus the survivors; each row of
    H sits in the group its throttle says. Returns a dict of named booleans and 'passed'."""
    tail, rest = np.asarray(tail, np.int64), np.asarray(rest, np.int64)
    surv = np.asarray(survivors, np.int64)
    kept = np.asarray(kept_train_rows, np.int64)
    h = heldout_rows(log, gap, splits)
    train = log.split_rows("train")
    both = np.concatenate([tail, rest])
    held_rows = np.concatenate([log.split_rows(s) for s in splits] + [topup_rows_of(log)])
    held = np.isin(both, held_rows)
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


def heldout_checks(log, gap, kept_train_rows, second_rows=None, splits=HELDOUT_SPLITS):
    """The new data checks of section 8.1 (a failure is a failed seed): every held-out and top-up row the groups read
    lies outside the ball; no held-out or top-up row is a training row (the kept training rows, and the second
    critic's when given). Returns a dict of named booleans and 'passed'."""
    h = heldout_rows(log, gap, splits)
    kept = np.asarray(kept_train_rows, np.int64)
    c = dict(heldout_rows_outside_ball=bool(not np.asarray(gap.contains(log["actions"][h]), bool).any())
             if len(h) else True,
             no_heldout_or_topup_row_in_training=bool(len(np.intersect1d(h, kept)) == 0
                                                      and np.all(np.isin(kept, log.split_rows("train")))))
    if second_rows is not None:
        c["no_heldout_or_topup_row_in_the_second_critic"] = bool(
            len(np.intersect1d(h, np.asarray(second_rows, np.int64))) == 0)
    c["passed"] = bool(all(c.values()))
    return c


def certifiable(n):
    """BCA's own rule (calibration.reference.certifiable at 0.1 / 0.95 / 1,000): a uniform-weight group of n scores
    fails to certify with probability at most 1e-6."""
    from calibration import reference
    return bool(reference.certifiable(int(n), types.SimpleNamespace(**CERT_SETTINGS)))


def outcome_free_count(log, gap, kept_train_rows, step1, gap_rule, topup=None):
    """The pre-training count of BCA's two groups (section 3.2, reported): commands only, before any learner is
    trained. topup: None (count the log only) or (make, target, cap): make(k) -> top-up block k
    (topup.block_record); blocks are added while the full-throttle count is below target, at most cap
    (topup.top_up). Returns ({defined, t_lo, t_up, n_survivors, n_tail, n_rest, tail {...}, rest {...},
    tail_one_per_episode, certifiable, below_bar, recorded {n_tail, n_rest}, topup {...}, group_checks,
    heldout_checks, bank_only, ...}, blocks); when the gap rule is not defined nothing is counted (n_tail 0,
    below_bar True: the switch will be off) and no block is made."""
    kept = np.asarray(kept_train_rows, np.int64)
    gr = gap_rule_of_rows(log, kept, step1, gap_rule)
    base = dict(defined=bool(gr["defined"]), t_lo=float(gr["t_lo"]) if gr.get("t_lo") is not None else None,
                t_up=None if gr.get("t_up") is None else float(gr["t_up"]), certifiable_bar=CERTIFIABLE_MIN,
                exact_floor=EXACT_FLOOR, reads="commands only (no reward, no next state, no learner)")
    if not gr["defined"]:
        return dict(base, n_tail=0, n_rest=0, reason=gr.get("reason"), certifiable=False, below_bar=True,
                    topup=dict(blocks=[], n_blocks=0, cap_reached=False,
                               note="no top-up: the gap rule is not defined (no t_lo)")), []
    surv = survivors_of(kept, gr)
    n0 = tail_count(log, gap, surv, gr["t_lo"])
    blocks, cap_reached = [], False
    if topup is not None:
        import topup as TU
        make, target, cap = topup
        blocks, _, cap_reached = TU.top_up(lambda bl: tail_count(view(log, bl), gap, surv, gr["t_lo"]), make, n0,
                                           target, cap)
    lg = view(log, blocks)
    tail, rest = pooled_groups(lg, gap, surv, gr["t_lo"])
    checks = group_checks(lg, gap, kept, surv, tail, rest, gr["t_lo"])
    hchecks = heldout_checks(lg, gap, kept)
    bt, br = bank_only_groups(log, gap, surv, gr["t_lo"])
    rec_tail, rec_rest = pooled_groups(log, gap, surv, gr["t_lo"])
    import topup as TU
    out = dict(base, n_survivors=int(len(surv)), n_tail=int(len(tail)), n_rest=int(len(rest)),
               tail=group_sources(lg, tail, surv), rest=group_sources(lg, rest, surv),
               certifiable=certifiable(len(tail)), below_bar=bool(len(tail) < CERTIFIABLE_MIN), group_checks=checks,
               heldout_checks=hchecks, recorded=dict(n_tail=int(len(rec_tail)), n_rest=int(len(rec_rest)),
                                                     note="the recorded rows alone, before any top-up"),
               topup=dict(blocks=TU.summary(blocks), n_blocks=len(blocks), cap_reached=bool(cap_reached),
                          blocks_sha256=TU.blocks_sha256(blocks) if blocks else None,
                          rows=int(sum(b["rows"] for b in blocks)), target=CERTIFIABLE_MIN if topup else None),
               bank_only=dict(n_tail=int(len(bt)), n_rest=int(len(br)),
                              note="the as-built groups of version 8 and step 1, for comparison only"))
    return out, blocks


def count_scramble_check(log, gap, kept_train_rows, step1, gap_rule, blocks, count, tags):
    """The data-time check of section 8.1 that the count reads no outcome: the count of the same rows with their
    rewards, next observations and next states permuted (numpy.random.default_rng(tags)) equals the count. Returns a
    dict with 'passed'."""
    rng = np.random.default_rng(list(tags))
    arrays = {k: np.asarray(log._arrays[k]) for k in log._arrays}
    for k in ("rewards", "next_observations", "infos/next_state"):
        arrays[k] = arrays[k][rng.permutation(len(arrays[k]))]
    scrambled = DS.Log(arrays, dict(log.meta))
    sb = []
    for b in blocks:
        a = dict(b["arrays"])
        for k in ("rewards", "next_observations", "infos/next_state"):
            a[k] = np.asarray(a[k])[rng.permutation(len(a[k]))]
        sb.append(dict(b, arrays=a))
    lg = view(scrambled, sb)
    kept = np.asarray(kept_train_rows, np.int64)
    gr = gap_rule_of_rows(lg, kept, step1, gap_rule)
    if not gr["defined"]:
        same = not count.get("defined")
        return dict(passed=bool(same), n_tail=0, n_rest=0)
    surv = survivors_of(kept, gr)
    tail, rest = pooled_groups(lg, gap, surv, gr["t_lo"])
    ok = bool(int(len(tail)) == count.get("n_tail") and int(len(rest)) == count.get("n_rest")
              and rows_sha256(tail) == (count.get("tail") or {}).get("rows_sha256")
              and rows_sha256(rest) == (count.get("rest") or {}).get("rows_sha256"))
    return dict(passed=ok, n_tail=int(len(tail)), n_rest=int(len(rest)), tags=list(tags),
                scrambled=["rewards", "next_observations", "infos/next_state"])


# ------------------------------------------------------------------------------------------------ the plain switches

def split_conformal_rank(n, level=0.90):
    """The textbook split-conformal rank: ceil((n + 1) * level) (exact integer arithmetic for level 0.90)."""
    num = (int(n) + 1) * int(round(float(level) * 100))
    return -(-num // 100)


def split_conformal_quantile(scores, level=0.90):
    """The ceil((n + 1) * level)-th smallest score (+infinity when that rank exceeds n; nan for no score)."""
    s = np.sort(np.asarray(scores, np.float64))
    if len(s) == 0:
        return float("nan")
    r = split_conformal_rank(len(s), level)
    return float(s[r - 1]) if r <= len(s) else float("inf")


def split_conformal(tail, rest, margin=0.10, level=0.90):
    """The textbook split-conformal switch on the two groups (DECISION.md section 2.4): q_tail - q_rest > margin; height
    that gap. Also step 2's uncorrected gap (numpy's 'inverted_cdf' 90% points: the ceil(n * level)-th smallest),
    reported only."""
    qt, qr = split_conformal_quantile(tail, level), split_conformal_quantile(rest, level)
    gap = qt - qr if np.isfinite(qt) and np.isfinite(qr) else (float("inf") if qt == float("inf") else float("nan"))
    t, r = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    s2t = float(np.quantile(t, level, method="inverted_cdf")) if len(t) else float("nan")
    s2r = float(np.quantile(r, level, method="inverted_cdf")) if len(r) else float("nan")
    return dict(rank_tail=split_conformal_rank(len(t), level) if len(t) else None,
                rank_rest=split_conformal_rank(len(r), level) if len(r) else None,
                q_tail=qt, q_rest=qr, gap=gap, on=bool(np.isfinite(gap) and gap > margin) or bool(gap == float("inf")),
                height=gap if (np.isfinite(gap) and gap > margin) or gap == float("inf") else 0.0, margin=margin,
                step2_uncorrected=dict(q_tail=s2t, q_rest=s2r, gap=s2t - s2r,
                                       rank_tail=int(np.ceil(len(t) * level)) if len(t) else None,
                                       note="step 2's rank (no (n + 1) correction): reported only, no actor"))


def wilks_ranks(n, level=0.90, confidence=0.95):
    """(r, s): the 1-based ranks of Wilks' distribution-free confidence limits for the level-quantile of n scores:
    X_(r) is a lower limit at `confidence` (the largest r with P(Binomial(n, level) >= r) >= confidence) and X_(s) an
    upper limit (the smallest s with P(Binomial(n, level) <= s - 1) >= confidence). None when no such rank exists."""
    from scipy import stats
    n = int(n)
    r = None
    for k in range(n, 0, -1):
        if float(stats.binom.sf(k - 1, n, level)) >= confidence:
            r = k
            break
    s = None
    for k in range(1, n + 1):
        if float(stats.binom.cdf(k - 1, n, level)) >= confidence:
            s = k
            break
    return r, s


def wilks_twin(tail, rest, margin=0.10, level=0.90, confidence=0.95):
    """The exact Wilks twin of BCA (DECISION.md section 2.4, reported only, no actor): switch on when Wilks' lower
    limit of the full-throttle group's level-point minus the rest's level-point (inverted_cdf) exceeds margin; height
    Wilks' upper limit of the full-throttle group's level-point minus the rest's upper limit (as BCA's height is
    R_tail - R_rest). A limit with no rank is -inf (lower) or +inf (upper)."""
    t, r = np.sort(np.asarray(tail, np.float64)), np.sort(np.asarray(rest, np.float64))
    if len(t) == 0 or len(r) == 0:
        return dict(on=False, height=float("nan"), note="an empty group")
    rt, st = wilks_ranks(len(t), level, confidence)
    rr, sr = wilks_ranks(len(r), level, confidence)
    lo_t = float(t[rt - 1]) if rt else float("-inf")
    up_t = float(t[st - 1]) if st else float("inf")
    up_r = float(r[sr - 1]) if sr else float("inf")
    q_r = float(np.quantile(r, level, method="inverted_cdf"))
    stat = lo_t - q_r
    return dict(lower_tail=lo_t, upper_tail=up_t, upper_rest=up_r, q_rest=q_r, ranks=dict(
        lower_tail=rt, upper_tail=st, upper_rest=sr), statistic=stat, on=bool(stat > margin),
        height=up_t - up_r if bool(stat > margin) else 0.0, upper_gap=up_t - up_r, margin=margin)


def mean_difference(tail, rest, margin=0.10):
    """The plain switch: mean(tail) - mean(rest) > margin."""
    t, r = np.asarray(tail, np.float64), np.asarray(rest, np.float64)
    gap = float(t.mean() - r.mean()) if len(t) and len(r) else float("nan")
    return dict(gap=gap, on=bool(np.isfinite(gap) and gap > margin), margin=margin)


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
        hc = (c.get("heldout_checks") or {}).get("passed")
        out.append(_rule("check 3b: no held-out or top-up row lies in the ball or among the training rows", hc,
                         hc is True, "all pass", sc, "to be seen when the frozen code runs"))
        sc_ = (f.get("count_scramble_check") or {}).get("passed")
        if sc_ is None:
            not_run.append(f"check 3c: the count is unchanged when rewards and next states are scrambled ({sc})")
        else:
            out.append(_rule("check 3c: the count is unchanged when rewards and next states are scrambled", sc_,
                             sc_ is True, "equal", sc, "to be seen when the frozen code runs"))
        rp = f.get("bank_only_reproduces_step1")
        if rp is None:
            not_run.append(f"check 4: the bank-only groups reproduce step 1's group sizes ({sc})")
        else:
            out.append(_rule("check 4: the bank-only groups reproduce step 1's recorded group sizes", rp, rp is True,
                             "equal", sc, "to be seen when the frozen code runs"))
    return dict(rules=out, not_run=not_run, passed=bool(all(r["passed"] for r in out) and not not_run))
