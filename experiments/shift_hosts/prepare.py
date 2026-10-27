"""The prepared-data file of each quality and data seed (INTERFACES.md section 6; plan sections 2.1-2.2, 3.2 item 4,
3.8, 4.1-4.2 and 4.6): the weighting test's logs, read from the verified copies with experiments.shift_iw.data, cut
by the same ball, with ReBRAC's next commands, exact and estimated weights, the bank, the test-law steps with exact
next states, and in-ball flags. No learning and no BCA code: nothing here trains, fits a band or computes a return.

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods. It runs in
the data environment (/opt/bca-shift-venv, numpy 2.2.6), where experiments/shift and experiments/shift_iw import; the
training environment reads the written file with handoff.read, which checks every hash.

Per quality and seed, runs/shift_hosts/data/<quality>/seed<k>/prepared.npz (+ .npz.sha256, prepared.json, the
manifest, + .json.sha256), written with handoff.write. Row sets, each in stored log order (episode, then t):
  train        the gapped log's kept training rows: episodes 0-499 minus the ball (shift.make_removal "gap" through
               data.SeedData.removal, data.py:708-710; shift.py:164-192), as the weighting test cut them
  full         every training row (50,000), no removal: the room references only (plan section 5.1)
  bank         the kept bank rows: data.calibration_bank's 5 rows per calibration episode (data.py:353-361), cut by the
               same ball (shift.py:145-146, 176-179)
  test_strong, test_mild   every step of the 300-episode test logs (30,000 each), never cut (data.py:409-419)
Fields per set (n rows): obs, action, reward, next_obs, done (float32; done 0 everywhere, choice 2), next_action
(float32, ReBRAC's a'), next_action_true (bool), episode, t (int32), row (int64, the row in its source log), state,
next_state (float64, exact), reward64 (float64), current (float64, c = state[:, 10]), in_ball (bool), and on train, bank
and the test sets weight_strong (float64, exact w(c) = world.ratio(c, "strong"), world.py:128-135) and
weight_strong_estimated (float64, the logistic classifier's estimate, classifier.py:69-108, fitted as the weighting test
fitted it, run.py:560-563, and used as run.py:584-608 uses it).

ReBRAC's next command (plan section 3.8, choice 4): Unifloral's jnp.roll(actions, -1, axis=0) (rebrac.py:335) over the
set in stored order. train and full: the roll over the set itself (equal to Unifloral's roll bit for bit, test (e));
bank: the roll over the calibration episodes' kept rows (episodes 500-699 cut by the ball) in stored order, then the
bank rows (plan section 3.8, "Bank rows get a' by the same rule"); test sets: the roll over the test log.
next_action_true says whether a' is the same episode's command at t + 1, that is, whether the roll's source is the
next log row; plan section 3.8 counts the rows where it is not. A substituted a' is the next kept row's command:
later in the same episode when the next row was removed, the next episode's first kept command at t = 99, or the
set's first row for its last row (roll wraps). No removed command leaks in: every a' of train and bank is a kept
row's command (checked).

Stop rule S11 (plan section 6.1): the kept training and bank rows must equal the weighting test's records
(runs/shift_iw_v5/<quality>/host/gap/seed<k>/record.json, result.data: counts, and the hashes of the rows, the
normalized Transition, the bank, the normalization and the evaluation rows, reproduced by the weighting test's own
data.prepare, data.py:748-754); likewise the full log against host/full. Any difference stops before anything is
written.
The estimated-weight classifier must reproduce the band record's (run.py:762-766 refuses otherwise), and the ball is
recomputed from the same seed's expert log (data.expert_ball, data.py:422-425) and must equal world.json's.

Run (data environment, from the repository root; one process per group of seeds):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 20 \\
      /opt/bca-shift-venv/bin/python -m experiments.shift_hosts.prepare --seeds 0 1 2
  ... then --summary, which reads the ten manifests and writes runs/shift_hosts/data/summary.json.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import experiments.shift  # noqa: F401  (gymnasium path and version checks; must come first)
from experiments.shift import learners as LR
from experiments.shift_iw import classifier as CL
from experiments.shift_iw import data as D
from experiments.shift_iw import world as W
from experiments.shift_hosts import data_copy as DC
from experiments.shift_hosts import handoff as H

SCHEMA = "shift-hosts-prepared-v1"
PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
OUT = REPO / "runs" / "shift_hosts" / "data"
QUALITIES = DC.QUALITIES
SEEDS = DC.SEEDS
TILT = "strong"
SETS = ("train", "full", "bank", "test_strong", "test_mild")
WEIGHTED_SETS = ("train", "bank", "test_strong", "test_mild")
LENGTH = W.LENGTH                                    # 100 steps per episode
HIGH_CURRENT_SHARE = 0.1                             # the highest-current tenth (run.py:610-613)
ALPHA = 2.5                                          # learners_iw.ALPHA (learners_iw.py:91, used at 138-143)
# plan section 2.2, read from the same records (a sanity range; S11 compares each seed with its own record)
PLAN_ROWS = {"medium-expert": dict(train=(44_902, 45_372), bank=(869, 924)),
             "medium": dict(train=(47_439, 47_548), bank=(933, 964))}
# the exact weight w(c) = (M(-2) / M(2)) e^(4c) = e^(4c - 2.4) runs from e^-2 = 0.1353 to e^2 = 7.389 on [0.1, 1.1]
W_RANGE = (math.exp(-2.0), math.exp(2.0))
W_RTOL = 1e-12
FIELDS = dict(obs=("<f4", (13,)), action=("<f4", (4,)), reward=("<f4", ()), next_obs=("<f4", (13,)),
              done=("<f4", ()), next_action=("<f4", (4,)), next_action_true=("|b1", ()), episode=("<i4", ()),
              t=("<i4", ()), row=("<i8", ()), state=("<f8", (13,)), next_state=("<f8", (13,)), reward64=("<f8", ()),
              current=("<f8", ()), in_ball=("|b1", ()), weight_strong=("<f8", ()),
              weight_strong_estimated=("<f8", ()))
WEIGHT_FIELDS = ("weight_strong", "weight_strong_estimated")
# every repository file the data path imports (the weighting test's data, world, classifier and removal code, and what
# they import: experiments/shift/learners.py imports algorithms/, calibration/ and runtime/ at the top, and
# thruster_chain.py and learners.py import experiments/signal/); code_hashes() also adds any other imported one
REPOSITORY_CODE = (
    "algorithms/__init__.py", "algorithms/td3_bc.py", "algorithms/td3_bc_bca.py", "calibration/__init__.py",
    "calibration/bank.py", "calibration/dose.py", "calibration/network.py", "calibration/reference.py",
    "calibration/wbcp.py", "experiments/shift/__init__.py", "experiments/shift/dataset.py",
    "experiments/shift/learners.py", "experiments/shift/recorder.py", "experiments/shift/shift.py",
    "experiments/shift/thruster_chain.py", "experiments/shift/truth.py", "experiments/shift_iw/__init__.py",
    "experiments/shift_iw/classifier.py", "experiments/shift_iw/data.py", "experiments/shift_iw/world.py",
    "experiments/signal/__init__.py", "experiments/signal/gapped_lq.py", "experiments/signal/glq_match.py",
    "experiments/signal/lq_harness.py", "experiments/truth/__init__.py", "experiments/truth/gap.py",
    "experiments/truth/labeller.py", "experiments/truth/pools.py", "experiments/truth/state_restore.py",
    "runtime/__init__.py", "runtime/networks.py", "runtime/published.py", "runtime/rewards.py")
PACKAGE_CODE = ("prepare.py", "data_copy.py", "handoff.py")


class PrepareError(RuntimeError):
    """A check of this module's own rules failed: nothing is written for that quality and seed."""


class S11Error(PrepareError):
    """Stop rule S11 (plan section 6.1): a copied file or the prepared rows differ from the weighting test's records."""


# ------------------------------------------------------------------------------------------------ paths

def path_for(quality, seed):
    """runs/shift_hosts/data/<quality>/seed<k>/prepared.npz (INTERFACES.md section 6)."""
    return OUT / quality / f"seed{int(seed)}" / "prepared.npz"


def fields_of(set_name):
    """The field names of a set, in FIELDS order (the weights only on train, bank and the test sets)."""
    return [f for f in FIELDS if f not in WEIGHT_FIELDS or set_name in WEIGHTED_SETS]


# ------------------------------------------------------------------------------------------------ helpers

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def take(log, key, rows, dtype):
    """log[key][rows] as a C-contiguous copy, refusing a source dtype other than the expected one (no silent cast)."""
    x = np.asarray(log[key])
    if x.dtype != np.dtype(dtype):
        raise PrepareError(f"{key} is {x.dtype}, expected {np.dtype(dtype)}")
    return np.ascontiguousarray(x[rows])


def same_bits(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and \
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()


def require(ok, message, error=None):
    if not ok:
        raise (error or PrepareError)(message)


def roll_rule(log, rows):
    """Unifloral's a' over a row set in stored order, np.roll(actions[rows], -1, axis=0) (rebrac.py:335), the log row
    each a' comes from, and whether that is the next log row (the same episode's command at t + 1)."""
    rows = np.asarray(rows, np.int64)
    nxt = np.roll(take(log, "actions", rows, np.float32), -1, axis=0)
    src = np.roll(rows, -1)
    t = np.asarray(log["infos/t"])[rows]
    true = (src == rows + 1) & (t < LENGTH - 1)
    return nxt, true, src


def substitution(true, rows, t, in_ball_all, high=None):
    """Counts and shares of substituted a' (overall, and in the highest-current tenth when `high` is given), with
    the reason: the episode's last step (t = 99), or the next log row removed by the ball (it lies in the ball);
    "other" (expected 0) would be anything else."""
    sub = ~true
    reason_end = sub & (t == LENGTH - 1)
    reason_removed = sub & (t < LENGTH - 1) & in_ball_all[np.minimum(rows + 1, len(in_ball_all) - 1)]
    out = dict(rows=int(len(true)), substituted=int(sub.sum()), share=float(sub.mean()) if len(sub) else None,
               reasons=dict(episode_end=int(reason_end.sum()), next_row_removed=int(reason_removed.sum()),
                            other=int((sub & ~reason_end & ~reason_removed).sum())))
    if high is not None:
        out["highest_current_tenth"] = dict(rows=int(high.sum()), substituted=int(sub[high].sum()),
                                            share=float(sub[high].mean()))
    return out


def high_current(current, share=HIGH_CURRENT_SHARE):
    """Rows in the top `share` of current: c >= the (1 - share) quantile over the set (restated from
    experiments/shift_iw/run.py:610-613, high_current_rows; plan section 3.8's "highest-current tenth" is this tenth
    of the kept training rows)."""
    current = np.asarray(current, np.float64)
    return current >= np.quantile(current, 1.0 - share)


def kish(w):
    """Kish's effective size (sum w)^2 / sum w^2."""
    w = np.asarray(w, np.float64)
    return float(w.sum() ** 2 / (w ** 2).sum())


def describe(x):
    x = np.asarray(x, np.float64)
    return dict(min=float(x.min()), max=float(x.max()), mean=float(x.mean())) if len(x) else None


def json_same(a, b):
    return json.dumps(H.plain(a), sort_keys=True) == json.dumps(H.plain(b), sort_keys=True)


def learn_settings():
    """The weighting test's learner settings: learners.LearnSettings with the master seed 20261016 and alpha 2.5
    (restated from experiments/shift_iw/learners_iw.py:91, 138-143, which is not imported: it is BCA code). Only the
    evaluation rows' draw (learners.py:259-260) depends on it here."""
    return LR.LearnSettings(seed=W.MASTER_SEED, alpha=ALPHA)


def environment():
    import h5py
    import jax
    import scipy

    return dict(python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__, h5py=h5py.__version__,
                jax=jax.__version__, executable=sys.executable, host=platform.node())


def imported_repository_code():
    """Repository files (outside this package) imported by this process, as paths relative to the repository."""
    repo, own = REPO.resolve(), PACKAGE.resolve()
    out = set()
    for module in list(sys.modules.values()):
        f = getattr(module, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if repo in p.parents and own not in p.parents:
            out.add(p.relative_to(repo).as_posix())
    return out


def code_hashes():
    """SHA-256 of this package's data files and of every repository file the data path imports (REPOSITORY_CODE and
    anything else imported)."""
    out = {f"experiments/shift_hosts/{n}": sha256_file(PACKAGE / n) for n in PACKAGE_CODE}
    out.update({rel: sha256_file(REPO / rel) for rel in sorted(set(REPOSITORY_CODE) | imported_repository_code())})
    return out


# ------------------------------------------------------------------------------------------------ against records

def record_data(rel):
    """result.data of a weighting-test host record (a verified copy, read after its sidecar check). Only the data
    part is read: counts, row hashes and normalization."""
    rec = H.read_json(DC.copied(rel))
    return rec["result"]["data"]


def compare_prepared(prep, recorded, rows, bank_rows):
    """The weighting test's own data.prepare against its record: counts, the six hashes, the normalization, and that
    this module's row sets are its rows."""
    hashes = {k: dict(recorded=recorded["sha256"].get(k), reproduced=v, equal=recorded["sha256"].get(k) == v)
              for k, v in sorted(prep.sha256.items())}
    out = dict(recorded=dict(train=int(recorded["rows"]), bank=int(recorded["bank_rows"])),
               prepared=dict(train=int(len(rows)), bank=int(len(bank_rows))),
               counts_equal=int(recorded["rows"]) == len(rows) and int(recorded["bank_rows"]) == len(bank_rows),
               hashes=hashes, hashes_equal=all(h["equal"] for h in hashes.values()) and
               set(hashes) == set(recorded["sha256"]),
               normalization_equal=(H.plain(prep.obs_mean) == recorded["obs_mean"] and
                                    H.plain(prep.obs_std) == recorded["obs_std"]),
               rows_are_prepare_rows=bool(np.array_equal(prep.rows, rows)),
               bank_rows_are_prepare_rows=bool(np.array_equal(prep.bank_rows, bank_rows)),
               removal=recorded.get("removal"), rep=recorded.get("rep"))
    out["ok"] = bool(out["counts_equal"] and out["hashes_equal"] and out["normalization_equal"] and
                     out["rows_are_prepare_rows"] and out["bank_rows_are_prepare_rows"])
    return out


def ball_check(seed):
    """The ball recomputed from the same seed's expert log: data.expert_ball (v1's densest-15% rule,
    experiments/truth/gap.find_gap through shift.define_gap; data.py:422-425). Simulates the expert's recording
    controller (no learning); about half a minute per seed."""
    started = time.perf_counter()
    gap, record = D.expert_ball(int(seed))
    return dict(gap=None if gap is None else H.plain(gap.as_dict()), defined=gap is not None,
                seconds=round(time.perf_counter() - started, 1))


# ------------------------------------------------------------------------------------------------ one seed

def prepare_seed(quality, seed, *, copy_manifest, ball=None):
    """(arrays, meta, times) of one quality and seed. Raises before anything is written when a check fails: S11Error
    for stop rule S11, PrepareError for this module's other checks. `ball`: ball_check(seed), or None to skip."""
    t0, times = time.perf_counter(), {}
    tick = lambda name, s: times.__setitem__(name, round(time.perf_counter() - s, 2))
    s = time.perf_counter()
    try:
        source = DC.require_verified(copy_manifest, DC.seed_rels(quality, seed))
        data = D.load_seed(seed, D.data_dir(DC.v5_root(), quality))     # every file and array against its manifest
    except (DC.CopyError, ValueError, OSError) as error:
        raise S11Error(f"S11: {quality} seed {seed}: a copied file failed its check: {error}") from error
    tick("load", s)
    log_ = data.train_log
    gap = data.gap
    n_log = log_.n_rows
    require(n_log == 1000 * LENGTH and data.world["quality"] == quality and int(data.world["seed"]) == int(seed),
            f"{quality} seed {seed}: not the expected training-law log")
    for lg in (log_, data.tests["strong"], data.tests["mild"]):    # rows are episode * 100 + t (data.py:320-329)
        idx = np.arange(lg.n_rows)
        require(np.array_equal(lg["infos/episode"], idx // LENGTH) and np.array_equal(lg["infos/t"], idx % LENGTH),
                f"{quality} seed {seed}: a log's rows are not in episode, then step order")
        require(not np.asarray(lg["terminals"]).any(), "a recorded terminal is true (choice 2 expects none)")

    # the ball and the removal, as the weighting test cut them
    in_ball_all = np.asarray(log_["truth/in_ball"], bool)
    require(same_bits(gap.contains(log_["actions"]), in_ball_all), "truth/in_ball differs from the ball")
    removal = data.removal("gap")
    rows = {"train": removal.kept_train_rows(log_), "bank": removal.bank_rows(log_),
            "full": data.removal("none").kept_train_rows(log_)}
    require(np.array_equal(rows["full"], log_.split_rows("train")) and len(rows["full"]) == 500 * LENGTH,
            "the full log is not the 50,000 training rows")
    require(not in_ball_all[rows["train"]].any() and not in_ball_all[rows["bank"]].any(),
            "a kept training or bank row lies in the ball")
    require(len(log_.bank_rows) == 1000 and np.array_equal(D.calibration_bank(seed, 200, 500), log_.bank_rows),
            "the log's bank rows are not data.calibration_bank's")

    # stop rule S11: counts and hashes against the weighting test's records, through its own data.prepare
    s = time.perf_counter()
    settings = learn_settings()
    rel_gap, rel_full = DC.host_record_rel(quality, "gap", seed), DC.host_record_rel(quality, "full", seed)
    s11_gap = compare_prepared(D.prepare(data, "gap", settings), record_data(rel_gap), rows["train"], rows["bank"])
    full_bank = data.removal("none").bank_rows(log_)
    s11_full = compare_prepared(D.prepare(data, "none", settings), record_data(rel_full), rows["full"], full_bank)
    removal_rec = data.world["removal"]
    world_counts = dict(kept_training_rows=int(removal_rec["kept_training_rows"]),
                        kept_bank_rows=int(removal_rec["kept_bank_rows"]))
    lo_t, hi_t = PLAN_ROWS[quality]["train"]
    lo_b, hi_b = PLAN_ROWS[quality]["bank"]
    s11 = dict(rule="stop rule S11 (plan section 6.1): the prepared row counts must equal the weighting test's "
                    "records; here also the recorded hashes of the rows, the normalized Transition, the bank, the "
                    "normalization and the evaluation rows, reproduced by the weighting test's own data.prepare",
               gap=dict(record=rel_gap, record_sha256=source[rel_gap], **s11_gap),
               full=dict(record=rel_full, record_sha256=source[rel_full], **s11_full),
               world_removal=dict(**world_counts, equal=(world_counts["kept_training_rows"] == len(rows["train"]) and
                                                         world_counts["kept_bank_rows"] == len(rows["bank"]))),
               plan_range=dict(train=[lo_t, hi_t], bank=[lo_b, hi_b],
                               inside=lo_t <= len(rows["train"]) <= hi_t and lo_b <= len(rows["bank"]) <= hi_b))
    s11["ok"] = bool(s11_gap["ok"] and s11_full["ok"] and s11["world_removal"]["equal"] and s11["plan_range"]["inside"])
    tick("s11", s)
    if not s11["ok"]:
        raise S11Error(f"S11: {quality} seed {seed}: the prepared rows differ from the weighting test's records: "
                       f"{json.dumps(H.plain(s11))[:2000]}")

    # the ball: world.json's, recomputed from the expert's log (when given), the same for every quality
    recomputed = None
    if ball is not None:
        recomputed = dict(ball, equal=bool(ball["defined"] and json_same(ball["gap"], gap.as_dict())))
        require(recomputed["equal"], f"{quality} seed {seed}: the recomputed expert ball differs from world.json's")
    ball_meta = dict(gap=H.plain(gap.as_dict()), source="world.json gap (data.load_seed, data.py:731)",
                     rule="v1's densest-15% ball of the same seed's expert training commands (data.expert_ball, "
                          "data.py:422-425; experiments/truth/gap.find_gap through shift.define_gap, "
                          "shift.py:111-124)",
                     in_ball_column_equals_ball=True, recomputed=recomputed)

    # the classifier, fitted as the weighting test fitted it, and checked against its band record
    s = time.perf_counter()
    clf = CL.fit_samples(data.samples, TILT)
    band = H.read_json(DC.copied(DC.band_record_rel(quality, seed)))["classifiers"][TILT]
    bank_obs = np.asarray(log_["observations"][rows["bank"]])
    c_bank = log_.current[rows["bank"]]
    est_arm = CL.estimated_arm(clf, bank_obs, data.samples[TILT]["observations"], data.samples[TILT]["actions"], gap)
    compared = CL.compare(clf, bank_obs, c_bank, TILT)
    clf_rec = dict(
        fit="classifier.fit_samples(samples, 'strong'): the training-law sample (class 0) against the strong-law "
            "sample (class 1), 300 episodes each, observations only (run.py:560-563)",
        describe=H.plain(clf.describe()), band_record=DC.band_record_rel(quality, seed),
        matches_band_record=json_same(clf.describe(), band["classifier"]),
        compare=H.plain(compared), compare_matches_band_record=json_same(compared, band["compare"]),
        test_mass=float(est_arm["test_mass"]), test_mass_matches=float(est_arm["test_mass"]) == band["test_mass"],
        test_sample_in_ball_share=float(est_arm["test_sample_in_ball_share"]))
    require(clf_rec["matches_band_record"] and clf_rec["compare_matches_band_record"] and clf_rec["test_mass_matches"],
            f"{quality} seed {seed}: the classifier differs from the band record's (run.py:762-766 refuses this)")
    tick("classifier", s)

    # the sets
    s = time.perf_counter()
    logs = {"train": log_, "full": log_, "bank": log_, "test_strong": data.tests["strong"],
            "test_mild": data.tests["mild"]}
    rows["test_strong"] = np.arange(data.tests["strong"].n_rows, dtype=np.int64)
    rows["test_mild"] = np.arange(data.tests["mild"].n_rows, dtype=np.int64)
    w_exact = {}
    for name in ("train", "test_strong", "test_mild"):          # closed form on the whole log, as data.py:345-350
        lg = logs[name]
        w_all = W.ratio(lg.current, TILT)
        require(same_bits(w_all, np.asarray(lg["truth/weight_strong"])),
                f"{name}: world.ratio(c, 'strong') differs from the log's truth/weight_strong")
        w_exact[name] = w_all
    w_exact["bank"] = w_exact["train"]                           # the bank rows are rows of the training-law log
    nxt = {}
    for name in ("train", "full", "test_strong", "test_mild"):
        nxt[name] = roll_rule(logs[name], rows[name])
    cal_rows = log_.split_rows("calibration")
    cut = cal_rows[~in_ball_all[cal_rows]]                       # the calibration episodes, cut by the same ball
    cut_next, cut_true, cut_src = roll_rule(log_, cut)
    pos = np.searchsorted(cut, rows["bank"])
    require(np.array_equal(cut[pos], rows["bank"]), "a kept bank row is not a kept calibration row")
    nxt["bank"] = (cut_next[pos], cut_true[pos], cut_src[pos])
    for name in ("train", "bank"):                               # no removed command leaks in
        require(not in_ball_all[nxt[name][2]].any() and not gap.contains(nxt[name][0]).any(),
                f"{name}: a next command comes from a removed row")
    arrays = {}
    for name in SETS:
        lg, r = logs[name], rows[name]
        a = dict(obs=take(lg, "observations", r, np.float32), action=take(lg, "actions", r, np.float32),
                 reward=take(lg, "rewards", r, np.float32), next_obs=take(lg, "next_observations", r, np.float32),
                 done=np.zeros(len(r), np.float32), next_action=np.ascontiguousarray(nxt[name][0]),
                 next_action_true=np.ascontiguousarray(nxt[name][1]),
                 episode=take(lg, "infos/episode", r, np.int32), t=take(lg, "infos/t", r, np.int32),
                 row=np.ascontiguousarray(r, np.int64), state=take(lg, "infos/state", r, np.float64),
                 next_state=take(lg, "infos/next_state", r, np.float64),
                 reward64=take(lg, "infos/reward64", r, np.float64),
                 current=np.ascontiguousarray(lg.current[r], np.float64),
                 in_ball=np.ascontiguousarray(np.asarray(lg["truth/in_ball"], bool)[r]))
        if name in WEIGHTED_SETS:
            a["weight_strong"] = np.ascontiguousarray(w_exact[name][r])
            # the classifier's weight at the float64 observation, as run.py:584-608 computes it (one call per set)
            a["weight_strong_estimated"] = np.ascontiguousarray(clf.weight(np.asarray(a["obs"], np.float64)),
                                                                np.float64)
        require(list(a) == fields_of(name), f"{name}: fields {list(a)}")
        for f, x in a.items():
            dtype, tail = FIELDS[f]
            require(x.dtype.str == dtype and x.shape == (len(r),) + tail, f"{name}/{f}: {x.dtype.str} {x.shape}")
            arrays[f"{name}/{f}"] = x
    tick("sets", s)

    # checks on the arrays themselves (no learning)
    s = time.perf_counter()
    for name in SETS:
        g = H.group(arrays, name)
        for f, x in g.items():
            if x.dtype.kind == "f":
                require(np.isfinite(x).all(), f"{name}/{f} holds a non-finite number")
        require((np.abs(g["action"]) <= 1.0).all(), f"{name}: a command lies outside [-1, 1]")
        require(same_bits(g["in_ball"], gap.contains(g["action"])), f"{name}: in_ball differs from the ball")
        if name in WEIGHTED_SETS:
            w = g["weight_strong"]
            require(((w >= W_RANGE[0] * (1 - W_RTOL)) & (w <= W_RANGE[1] * (1 + W_RTOL))).all(),
                    f"{name}: an exact weight lies outside [e^-2, e^2]")
            require((g["weight_strong_estimated"] > 0).all(), f"{name}: an estimated weight is not positive")
    test_e = roll_test_e(arrays, log_, cut, pos)
    require(all(v["equal"] for v in test_e.values()), f"test (e) failed on the real arrays: {test_e}")
    tick("checks", s)

    # readouts
    def sub_of(name, with_high=True):
        lg = logs[name]
        g = H.group(arrays, name)
        high = high_current(g["current"]) if with_high else None
        return substitution(g["next_action_true"], g["row"], g["t"], np.asarray(lg["truth/in_ball"], bool), high)

    substituted = {name: sub_of(name) for name in SETS}
    weights = {}
    for name in WEIGHTED_SETS:
        g = H.group(arrays, name)
        weights[name] = dict(exact=describe(g["weight_strong"]), estimated=describe(g["weight_strong_estimated"]),
                             literal_range_0_135_to_7_389=bool(((g["weight_strong"] >= 0.135) &
                                                                (g["weight_strong"] <= 7.389)).all()))
    bank_w = arrays["bank/weight_strong"]
    bank_w_est = arrays["bank/weight_strong_estimated"]
    in_ball = {name: dict(rows=int(len(rows[name])), in_ball=int(arrays[f"{name}/in_ball"].sum()),
                          share=float(arrays[f"{name}/in_ball"].mean())) for name in SETS}
    world_steps = data.world["test_in_ball_steps"]
    require(in_ball["test_strong"]["in_ball"] == int(world_steps["strong"]) and
            in_ball["test_mild"]["in_ball"] == int(world_steps["mild"]),
            "the test logs' in-ball counts differ from world.json's")
    files = {DC.copy_rel(rel): digest for rel, digest in sorted(source.items())}
    originals = {DC.copy_rel(rel): f"{DC.SOURCE}/{rel}" for rel in sorted(source)}
    meta = dict(
        schema=SCHEMA, quality=quality, seed=int(seed), tilt=TILT,
        about="ThrusterChain weighting-test data for the shift_hosts study (PLAN_UNIFLORAL.md); prepare.py's module "
              "docstring defines every set and field; no learning",
        rows={name: int(len(rows[name])) for name in SETS},
        sets=dict(
            train="kept training rows: episodes 0-499 minus the ball (removal 'gap'), stored order",
            full="every training row (episodes 0-499), no removal: room references only",
            bank="kept bank rows: 5 per calibration episode 500-699 (data.calibration_bank), cut by the ball",
            test_strong="every step of the 300-episode strong-law test log, never cut",
            test_mild="every step of the 300-episode mild-law test log, never cut"),
        fields={name: fields_of(name) for name in SETS},
        recorded_rows=dict(gap=s11_gap["recorded"], full=s11_full["recorded"]),
        s11=s11, source=dict(
            files=files, originals=originals,
            world_json_sha256=source[DC.seed_rel(quality, seed, "world.json")],
            rule="every file read is a verified copy (data_copy.py; copy manifest runs/shift_hosts/source/"
                 "copy_manifest.json), checked again by SHA-256 before reading; files: the copy's path relative to "
                 "the repository -> its SHA-256 (hdf5 files, their .manifest.json, world.json, the host and band "
                 "records); originals: where it was copied from (read only)"),
        ball=ball_meta,
        next_action=dict(
            rule="Unifloral's jnp.roll(actions, -1, axis=0) (rebrac.py:335) over the set in stored order; bank: the "
                 "roll over the calibration episodes' kept rows (500-699 cut by the ball), then the bank rows; "
                 "next_action_true: the roll's source is the next log row (the same episode's command at t + 1)",
            substituted=substituted, test_e=test_e,
            calibration_rows_cut=dict(rows=int(len(cut)), of=int(len(cal_rows)))),
        weights=dict(
            exact_rule="world.ratio(c, 'strong') = (M(-2) / M(2)) exp(4 c) (world.py:128-135), equal bit for bit to "
                       "the log's truth/weight_strong",
            constant=W.ratio_constant(TILT), slope=W.ratio_slope(TILT), exact_range=list(W_RANGE),
            estimated_rule="classifier.Classifier.weight(float64 observation) of the strong-law classifier",
            per_set=weights,
            bank_kish=dict(exact=kish(bank_w), estimated=kish(bank_w_est), rows=int(len(bank_w)),
                           rule="(sum w)^2 / sum w^2 over the kept bank rows")),
        classifier=clf_rec, in_ball=in_ball,
        done="0.0 on every row: the recorded terminals are all false and time-limit rows are kept (choice 2)",
        normalization="not stored: the training block computes it from train/obs (INTERFACES.md section 5)",
        code=code_hashes(), environment=environment())
    times["total"] = round(time.perf_counter() - t0, 2)
    return arrays, meta, times


def roll_test_e(arrays, log_, cut, pos):
    """Test (e) on the real arrays (plan section 4.5; compares arrays only): every set's next_action equals JAX's
    jnp.roll(actions, -1, axis=0) bit for bit (Unifloral's own expression, rebrac.py:335); the bank's equals jnp.roll
    over the cut calibration episodes at the bank rows."""
    import jax.numpy as jnp

    out = {}
    for name in ("train", "full", "test_strong", "test_mild"):
        j = np.asarray(jnp.roll(jnp.asarray(arrays[f"{name}/action"]), -1, axis=0))
        out[name] = dict(equal=same_bits(j, arrays[f"{name}/next_action"]),
                         equal_numpy_roll=same_bits(np.roll(arrays[f"{name}/action"], -1, axis=0),
                                                    arrays[f"{name}/next_action"]))
    j = np.asarray(jnp.roll(jnp.asarray(take(log_, "actions", cut, np.float32)), -1, axis=0))[pos]
    out["bank"] = dict(equal=same_bits(j, arrays["bank/next_action"]),
                       equal_numpy_roll=same_bits(np.roll(take(log_, "actions", cut, np.float32), -1, axis=0)[pos],
                                                  arrays["bank/next_action"]))
    for v in out.values():
        v["equal"] = bool(v["equal"] and v["equal_numpy_roll"])
    return out


# ------------------------------------------------------------------------------------------------ files

def write_seed(quality, seed, arrays, meta, times):
    path = path_for(quality, seed)
    manifest = H.write(path, arrays, meta)
    H.write_json(path.parent / "prepare_times.json", dict(quality=quality, seed=int(seed), seconds=times,
                                                          finished_utc=datetime.now(timezone.utc).isoformat(
                                                              timespec="seconds"),
                                                          npz_sha256=manifest["npz_sha256"],
                                                          affinity=sorted(os.sched_getaffinity(0))))
    return manifest


def summary(qualities=QUALITIES, seeds=SEEDS, log=print):
    """runs/shift_hosts/data/summary.json from the written manifests (each read after its sidecar check)."""
    table = []
    for q in qualities:
        for k in seeds:
            path = path_for(q, k)
            man = H.read_manifest(path)
            H.verify_sidecar(path)
            m = man["meta"]
            sub = m["next_action"]["substituted"]
            table.append(dict(
                quality=q, seed=int(k), npz_sha256=man["npz_sha256"],
                manifest_sha256=H.sha256_file(H.manifest_path(path)),
                rows=m["rows"], recorded=m["recorded_rows"], s11_ok=m["s11"]["ok"],
                substituted_train=sub["train"]["share"],
                substituted_train_count=sub["train"]["substituted"],
                substituted_train_high=sub["train"]["highest_current_tenth"]["share"],
                substituted_train_high_count=sub["train"]["highest_current_tenth"]["substituted"],
                substituted_train_high_rows=sub["train"]["highest_current_tenth"]["rows"],
                substituted_bank=sub["bank"]["share"], substituted_bank_count=sub["bank"]["substituted"],
                substituted_full=sub["full"]["share"], reasons_train=sub["train"]["reasons"],
                bank_kish=m["weights"]["bank_kish"]["exact"],
                bank_kish_estimated=m["weights"]["bank_kish"]["estimated"],
                weights=m["weights"]["per_set"], in_ball=m["in_ball"],
                classifier_slope_on_c=m["classifier"]["describe"]["slope_on_c"],
                classifier_matches=m["classifier"]["matches_band_record"],
                ball_recomputed_equal=(m["ball"]["recomputed"] or {}).get("equal"),
                test_e=all(v["equal"] for v in m["next_action"]["test_e"].values())))
    H.write_json(OUT / "summary.json", dict(schema=SCHEMA + "-summary", rows=table))
    for r in table:
        log(f"{r['quality']:>13} seed{r['seed']}: train {r['rows']['train']} (record {r['recorded']['gap']['train']}), "
            f"bank {r['rows']['bank']} (record {r['recorded']['gap']['bank']}), substituted a' "
            f"{100 * r['substituted_train']:.2f}% (top-current tenth {100 * r['substituted_train_high']:.2f}%), bank "
            f"{100 * r['substituted_bank']:.2f}%, bank Kish {r['bank_kish']:.1f}, S11 {r['s11_ok']}")
    return table


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare the shift_hosts data files (no learning, no BCA).")
    p.add_argument("--qualities", nargs="+", default=list(QUALITIES), choices=QUALITIES)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--no-ball-recompute", action="store_true",
                   help="skip recomputing the expert's ball (about 30 s per seed)")
    p.add_argument("--summary", action="store_true", help="only write summary.json from the written files")
    a = p.parse_args(argv)
    if a.summary:
        summary(tuple(a.qualities), tuple(a.seeds))
        return 0
    try:
        copy_manifest = DC.read_manifest()
    except DC.CopyError as error:
        print(f"STOP (S11): {error}", flush=True)
        return 3
    for seed in a.seeds:
        ball = None if a.no_ball_recompute else ball_check(seed)
        if ball is not None:
            print(f"seed {seed}: expert ball recomputed in {ball['seconds']:.0f} s", flush=True)
        for quality in a.qualities:
            stop = OUT / quality / f"seed{int(seed)}" / "STOP.json"
            try:
                arrays, meta, times = prepare_seed(quality, seed, copy_manifest=copy_manifest, ball=ball)
            except PrepareError as error:
                rule = "S11" if isinstance(error, S11Error) else "a prepare.py check"
                H.write_json(stop, dict(quality=quality, seed=int(seed), rule=rule, stop=str(error)))
                print(f"STOP ({rule}) {quality} seed {seed}: {error}", flush=True)
                return 3
            manifest = write_seed(quality, seed, arrays, meta, times)
            for old in (stop, H.sidecar(stop)):          # an earlier attempt's stop no longer applies
                if old.exists():
                    old.unlink()
            sub = meta["next_action"]["substituted"]
            print(f"{quality} seed {seed}: train {meta['rows']['train']}, bank {meta['rows']['bank']} (S11 passed), "
                  f"a' substituted {100 * sub['train']['share']:.2f}% (top-current tenth "
                  f"{100 * sub['train']['highest_current_tenth']['share']:.2f}%), bank Kish "
                  f"{meta['weights']['bank_kish']['exact']:.1f}, {times['total']:.0f} s -> {path_for(quality, seed)} "
                  f"({manifest['npz_sha256'][:16]})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
