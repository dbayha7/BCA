"""Semi-synthetic WBCP benchmark on real D4RL transitions with exact ground truth.

Reads a frozen score artifact (<dir>/frozen.npz and frozen.json, schema
wbcp-frozen-scores-v1, written by experiments/wbcp/freeze_scores.py): a frozen model
gives every transition of a population pool one fixed nonconformity score. Test laws
tilt the pool, P_test(row i) proportional to a_i = exp(gamma z(x_i)), with z
standardized over the pool and a function of the transition's (s, a) only:

  policy   z = -||action - policy_action||: rows the frozen actor would take gain mass
  density  z = log distance to the k-th nearest neighbour in standardized (obs, action)
           space, among a fixed seeded reference subset of the pool (reference points
           at distance zero are skipped, so a row never counts itself): sparse rows gain
           mass. The raw distance is heavy-tailed on D4RL (9-17 SD), so exp(gamma z)
           would collapse the test law; --density-transform raw keeps it anyway
  state    z = one observation coordinate, or ||obs|| (--state-feature). ||obs|| is
           right-skewed on D4RL, so gamma >= 1 collapses n_eff (2.8% of the hopper pool
           at gamma 1) and WBCP then abstains, which is the correct response

Resampling rows never changes a row's own reward, next state or score, so the score
given (s, a) keeps its law: a pure covariate shift. Calibration rows are iid uniform
draws from the pool, exactly exchangeable with a uniform test row, and all ground
truth is exact rather than simulated:

  oracle ratio      w*(x) = a(x) / mean_pool(a)
  oracle test mass  E_test[w*] = mean(a^2) / mean(a)^2
  realized risk     R(lambda) = sum_i a_i 1{s_i > lambda} / sum_i a_i over the pool
  shortest valid    lambda* = the smallest pool score with R <= alpha

A threshold violates iff R(threshold) > alpha, equivalently iff it lies below lambda*.
Each arm also reports the 95th and 99th percentiles of realized risk and the mean excess
over alpha among failing trials (how far, not only how often). --shuffle-tilt SEED is a
control that reassigns each tilt's weights to pool rows by a fixed permutation.
An infinite threshold (abstention) has risk 0. Failure frequencies count abstentions
as valid; fail_certified drops them from the denominator, as reproduce_table1.py does.

The tilt never looks at the residual, score, reward or next state. Tilting on any of
them changes the law of the response given (s, a): the shift becomes conditional,
dP_test/dP_cal depends on the label, no covariate ratio w(x) exists, and the oracle
arm would weight calibration rows by their own labels. That would test a claim WBCP
does not make. tilt_feature() therefore never receives those arrays.

Arms: BQ-CP (uniform weights, lambda_hpd; the uniform BCA threshold, shift-blind);
RCPS (Hoeffding-Bentkus, shift-blind); W-CRC and WBCP with estimated weights; WBCP
with the oracle w* and the exact oracle test mass. Estimated weights come from
weight-fit samples Cw (uniform) and Tw (tilted) and unlabeled tilted rows Ts, all
independent of the scored calibration rows: a ridge-stabilized Newton logistic
discriminator with class-prior correction, on the tilt feature z (--discriminator
feature; well specified, since log w* is affine in z; unpenalized by default) or on
standardized (obs, action, action - policy_action) (--discriminator raw; misspecified
for nonlinear tilts, and unable to represent the symmetric policy tilt at all, so
policy x raw measures a missing feature rather than a degree of misspecification),
and wbar = mean of w-hat over Ts. --score raw replaces |y - q| / sigma by
|y - q|, to test whether the learned scale makes the score closer to pivotal.

Each trial draws one scored calibration sample per n and reuses it for every
(tilt, gamma) block and both scores (common random numbers, so comparisons are
paired; the shift-blind thresholds are computed once per trial). Every random stream
is keyed by (--seed, stage, n, trial, tilt, gamma, score), so a block's numbers do
not depend on the rest of the sweep or on --workers. --blocks draws whole episodes,
uniformly with replacement, until at least n rows: a dependence stress test, under
which exchangeability holds between episodes only. --per-episode K draws ceil(n/K)
episodes with probability proportional to length and K rows inside each, then keeps a
uniformly random n of those rows (a thinned bank); --spacing stratified takes one row
from each of K equal segments of the episode, so the rows sit about L/K steps apart;
--spacing reservation runs BCA's own reservation sampler (calibration/bank.py: distinct
episodes with inclusion probability exactly proportional to length, stratified rows, the
surplus over n removed at random). Every mode keeps each pool row's expected count equal,
so the modes differ only in within-bank dependence. Each result records this sampler
revision as bank_trim (calibration.bank.REMAINDER_TRIM); a result without it was drawn
before the remainder trim (2026-10-01), when per-episode banks lost the last draw's final
rows ('random', 'stratified') or kept the surplus ('reservation').

python experiments/wbcp/d4rl_benchmark.py --frozen runs/wbcp_frozen/hopper_medium \
    --tilt policy density state --gamma 0 0.5 1 --n 1103 --trials 1000 --score both \
    --workers 8 --output runs/wbcp_bench/hopper_medium.json
"""

import argparse
import json
import math
import multiprocessing
import os
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import special, stats
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from calibration.bank import REMAINDER_TRIM, stratified_bank  # noqa: E402
from calibration.wbcp import calibrate  # noqa: E402
from experiments.wbcp.reproduce_table1 import rcps_index, weighted_crc  # noqa: E402

SCHEMA = "wbcp-frozen-scores-v1"
ARMS = ("BQ-CP", "RCPS", "W-CRC", "WBCP", "WBCP (oracle w)")
TILTS = ("policy", "density", "state")
SCORES = ("normalized", "raw")
FIELDS = {"score": 1, "residual": 1, "sigma": 1, "obs": 2, "action": 2, "policy_action": 2,
          "episode": 1, "timestep": 1, "row": 1, "in_training": 1}


@dataclass(frozen=True)
class Pool:
    scores: dict  # score name -> float64 [N]: normalized |y - q| / sigma, raw |y - q|
    obs: np.ndarray  # float64 [N, ds]
    action: np.ndarray  # float64 [N, da]
    policy_action: np.ndarray  # float64 [N, da]
    episode: np.ndarray
    timestep: np.ndarray
    row: np.ndarray
    metadata: dict  # frozen.json
    population: str
    artifact_rows: int  # rows in the artifact before the population rule

    @property
    def size(self):
        return self.scores["normalized"].size


def load_pool(directory, population="heldout"):
    """Read and check a frozen score artifact; keep the rows of the population rule."""
    with open(os.path.join(directory, "frozen.json")) as handle:
        metadata = json.load(handle)
    if metadata.get("schema") != SCHEMA:
        raise ValueError(f"frozen.json schema must be {SCHEMA!r}, got {metadata.get('schema')!r}")
    with np.load(os.path.join(directory, "frozen.npz"), allow_pickle=False) as archive:
        missing = sorted(set(FIELDS) - set(archive.files))
        if missing:
            raise ValueError(f"frozen.npz lacks {missing}")
        data = {name: archive[name] for name in FIELDS}
    rows = len(data["score"])
    for name, ndim in FIELDS.items():
        if data[name].ndim != ndim or len(data[name]) != rows:
            raise ValueError(f"{name} must have {ndim} dimension(s) and {rows} rows")
    if data["policy_action"].shape != data["action"].shape:
        raise ValueError("policy_action must match action in shape")
    if data["in_training"].dtype != np.bool_:
        raise ValueError("in_training must be boolean")
    score, residual, sigma = (data[name].astype(np.float64) for name in ("score", "residual", "sigma"))
    arrays = [score, residual, sigma] + [data[name] for name in ("obs", "action", "policy_action")]
    if rows == 0 or not all(np.all(np.isfinite(values)) for values in arrays):
        raise ValueError("frozen arrays must be nonempty and finite")
    if not (np.all(sigma > 0) and np.all(score >= 0)):
        raise ValueError("sigma must be positive and scores nonnegative")
    if not np.allclose(score, np.abs(residual) / sigma, rtol=1e-5, atol=0.0):
        raise ValueError("score must equal |residual| / sigma")
    if population == "heldout":
        keep = ~data["in_training"]
    elif population == "all":
        keep = np.ones(rows, bool)
    else:
        raise ValueError("population must be 'heldout' or 'all'")
    if not keep.any():
        raise ValueError(f"the {population} population is empty")
    return Pool(
        scores={"normalized": score[keep], "raw": np.abs(residual[keep])},
        obs=data["obs"][keep].astype(np.float64), action=data["action"][keep].astype(np.float64),
        policy_action=data["policy_action"][keep].astype(np.float64),
        episode=data["episode"][keep].astype(np.int64), timestep=data["timestep"][keep].astype(np.int64),
        row=data["row"][keep].astype(np.int64), metadata=metadata, population=population, artifact_rows=rows,
    )


def standardize(values):
    """Center and scale over the pool (axis 0); constant columns are only centered."""
    values = np.asarray(values, np.float64)
    spread = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(spread > 0, spread, 1.0)


def knn_distance(points, reference, k):
    """Distance from each point to its k-th nearest reference point at positive distance."""
    tree = cKDTree(reference)
    out = np.empty(len(points))
    pending, extra = np.arange(len(points)), 1
    while pending.size:
        count = min(k + extra, len(reference))
        distance = tree.query(points[pending], k=count, workers=-1)[0].reshape(pending.size, count)
        zeros = np.sum(distance == 0, axis=1)
        done = zeros <= count - k  # the k-th positive distance is within the queried neighbours
        if not done.any() and count == len(reference):
            raise ValueError("fewer than k reference points at positive distance")
        out[pending[done]] = distance[done, k - 1 + zeros[done]]
        pending, extra = pending[~done], 2 * extra
    return out


def tilt_feature(kind, obs, action, policy_action, *, state_feature="norm", knn_k=10,
                 knn_reference=10_000, knn_seed=0, density_transform="log"):
    """Unstandardized tilt feature: a function of (s, a) only; scores never enter."""
    if kind == "policy":
        return -np.linalg.norm(action - policy_action, axis=1)
    if kind == "state":
        if state_feature == "norm":
            return np.linalg.norm(obs, axis=1)
        column = int(state_feature)
        if not 0 <= column < obs.shape[1]:
            raise ValueError(f"--state-feature {column} is outside the {obs.shape[1]} observation coordinates")
        return np.asarray(obs[:, column], np.float64)
    if kind == "density":
        space = standardize(np.hstack([obs, action]))
        chosen = np.random.default_rng(knn_seed).choice(len(space), min(knn_reference, len(space)), replace=False)
        distance = knn_distance(space, space[np.sort(chosen)], knn_k)
        if density_transform == "log":
            return np.log(distance)
        if density_transform == "raw":
            return distance
        raise ValueError(f"unknown density transform {density_transform!r}")
    raise ValueError(f"unknown tilt {kind!r}")


@dataclass(frozen=True)
class Tilt:
    name: str
    gamma: float
    z: np.ndarray  # standardized tilt feature over the pool
    log_mass: np.ndarray  # log(a / max a)
    mass: np.ndarray  # a / max a; every ratio below is scale free
    cumulative: np.ndarray
    total: float  # sum a (with max a = 1)
    square_total: float  # sum a^2
    last: int  # last row with positive mass

    @property
    def n_eff(self):
        return self.total ** 2 / self.square_total

    @property
    def oracle_wbar(self):
        """E_test[w*] = mean(a^2) / mean(a)^2 for w* = a / mean(a)."""
        return self.mass.size * self.square_total / self.total ** 2

    def oracle_ratio(self, rows):
        return self.mass[rows] * (self.mass.size / self.total)

    def sample(self, rng, size):
        """iid rows from the tilted law, by inversion of the cumulative mass."""
        found = np.searchsorted(self.cumulative, rng.random(size) * self.total, side="right")
        return np.minimum(found, self.last)


def make_tilt(name, z, gamma):
    exponent = float(gamma) * np.asarray(z, np.float64)
    log_mass = exponent - exponent.max()
    mass = np.exp(log_mass)
    cumulative = np.cumsum(mass)
    return Tilt(name, float(gamma) + 0.0, np.asarray(z, np.float64), log_mass, mass, cumulative,
                float(cumulative[-1]), float(np.dot(mass, mass)), int(np.flatnonzero(mass > 0)[-1]))


class ExactRisk:
    """R(lambda) = sum_i a_i 1{s_i > lambda} / sum_i a_i over the pool, by binary search."""

    def __init__(self, scores, mass, order=None):
        order = np.argsort(scores, kind="stable") if order is None else order
        self.sorted_scores = np.asarray(scores, np.float64)[order]
        tail = np.cumsum(np.asarray(mass, np.float64)[order][::-1])[::-1]  # tail[j] = sum_{i >= j}
        self.tail = np.append(tail, 0.0)

    def __call__(self, thresholds):
        thresholds = np.asarray(thresholds, np.float64)
        if np.isnan(thresholds).any():
            raise ValueError("a NaN threshold has no realized risk")
        above = np.searchsorted(self.sorted_scores, thresholds, side="right")
        return self.tail[above] / self.tail[0]

    def lambda_star(self, alpha):
        """The smallest pool score with R <= alpha: the shortest valid threshold."""
        return float(self.sorted_scores[np.argmax(self(self.sorted_scores) <= alpha)])


def fit_logistic(design, labels, ridge, iterations=100, tolerance=1e-10):
    """Newton logistic regression with an L2 penalty ridge/2 ||theta||^2 on the mean
    log-likelihood; column 0 of the design is the unpenalized intercept."""
    penalty = np.full(design.shape[1], float(ridge))
    penalty[0] = 0.0

    def objective(theta):
        eta = design @ theta
        return float(np.mean(labels * eta - np.logaddexp(0.0, eta)) - 0.5 * np.dot(penalty * theta, theta))

    theta = np.zeros(design.shape[1])
    value = objective(theta)
    for _ in range(iterations):
        prob = special.expit(design @ theta)
        gradient = design.T @ (labels - prob) / len(labels) - penalty * theta
        hessian = (design.T * (prob * (1.0 - prob))) @ design / len(labels) + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        scale = 1.0
        while objective(theta + scale * step) < value and scale > 1e-10:  # backtracking
            scale *= 0.5
        candidate = theta + scale * step
        candidate_value = objective(candidate)
        if candidate_value < value:
            break
        theta, value = candidate, candidate_value
        if np.max(np.abs(scale * step)) < tolerance:
            break
    return theta


def logistic_log_ratio(cal_features, test_features, ridge):
    """log dP_test/dP_cal from a logistic discriminator, with class-prior correction."""
    features = np.vstack([cal_features, test_features])
    design = np.column_stack([np.ones(len(features)), features])
    labels = np.concatenate([np.zeros(len(cal_features)), np.ones(len(test_features))])
    theta = fit_logistic(design, labels, ridge)
    offset = theta[0] + math.log(len(cal_features) / len(test_features))
    return lambda values: offset + np.asarray(values, np.float64) @ theta[1:]


def common_scale(log_weights, log_test_mass):
    """Exponentiate weights and test mass after removing their common maximum; it cancels."""
    shift = max(float(np.max(log_weights)), float(log_test_mass))
    return np.exp(log_weights - shift), math.exp(log_test_mass - shift)


def episode_groups(episode, timestep):
    """Pool rows grouped by episode: (row order, group starts, group lengths)."""
    order = np.lexsort((timestep, episode))
    ids = episode[order]
    starts = np.flatnonzero(np.concatenate([[True], ids[1:] != ids[:-1]]))
    return order, starts, np.diff(np.append(starts, ids.size))


def draw_calibration(rng, n, size, groups=None, per_episode=None, spacing="random"):
    """Scored calibration rows: n iid uniform pool rows; whole episodes (uniform, with
    replacement) until at least n rows; or, with per_episode=K, ceil(n/K) episode draws with
    probability proportional to length and K rows inside each. The K rows are uniform over
    the episode (spacing 'random') or one uniform row in each of K equal segments (spacing
    'stratified', so rows sit about L/K steps apart). When K does not divide n, a uniformly
    random n of the K ceil(n/K) drawn rows are kept, in draw order (one more draw from rng;
    none when K divides n). Cutting the last draw's final slots instead would under-sample
    late timesteps. Spacing 'reservation' is BCA's sampler (calibration/bank.py): ceil(n/K)
    distinct episodes with inclusion probability exactly proportional to length, one row per
    K segments, the surplus over n removed uniformly at random. In every mode each pool row
    has the same expected count (in 'reservation', only when no reserved episode is shorter
    than K), so the modes differ only in how strongly the calibration rows depend on each
    other."""
    if groups is None:
        return rng.integers(size, size=n)
    order, starts, lengths = groups
    if per_episode is not None and spacing == "reservation":
        episodes, offsets = stratified_bank(lengths, n, per_episode, rng)
        return order[np.concatenate([starts[e] + o for e, o in zip(episodes, offsets)])]
    if per_episode is not None:
        draws = -(-n // per_episode)
        episodes = rng.choice(len(starts), size=draws, p=lengths / lengths.sum())
        position = rng.random((draws, per_episode))
        if spacing == "stratified":
            position = (np.arange(per_episode) + position) / per_episode
        elif spacing != "random":
            raise ValueError(f"unknown spacing {spacing!r}")
        offsets = np.minimum((position * lengths[episodes, None]).astype(np.int64), lengths[episodes, None] - 1)
        rows = order[starts[episodes, None] + offsets].ravel()
        return rows if rows.size == n else rows[np.sort(rng.choice(rows.size, n, replace=False))]
    chosen, total = [], 0
    while total < n:
        episode = int(rng.integers(len(starts)))
        chosen.append(order[starts[episode]:starts[episode] + lengths[episode]])
        total += int(lengths[episode])
    return np.concatenate(chosen)


def _words(part):
    if isinstance(part, str):
        return [zlib.crc32(part.encode())]
    if isinstance(part, float):
        bits = struct.unpack("<Q", struct.pack("<d", part + 0.0))[0]
        return [bits & 0xFFFFFFFF, bits >> 32]
    if isinstance(part, (int, np.integer)) and 0 <= part < 2 ** 32:
        return [int(part)]
    raise ValueError(f"cannot key a random stream by {part!r}")


def stream(seed, *key):
    """An independent generator for one stage of one trial, fixed by the seed and the key."""
    words = tuple(word for part in key for word in _words(part))
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=words))


@lru_cache(maxsize=None)
def _rcps(n, alpha, delta):
    return rcps_index(n, alpha, delta)


@dataclass
class Setup:
    args: argparse.Namespace
    pool: Pool
    scores: tuple
    features: dict  # tilt name -> standardized z over the pool
    raw_features: dict  # tilt name -> unstandardized feature
    tilts: list  # one Tilt per (tilt, gamma) block, sweep order
    raw_center: np.ndarray  # pool statistics for --discriminator raw
    raw_scale: np.ndarray
    groups: tuple  # episode grouping for --blocks / --per-episode, else None
    risks: dict  # (block, score) -> ExactRisk


def build_setup(args):
    pool = load_pool(args.frozen, args.population)
    raw = {name: tilt_feature(name, pool.obs, pool.action, pool.policy_action,
                              state_feature=args.state_feature, knn_k=args.knn_k,
                              knn_reference=args.knn_reference, knn_seed=args.knn_seed,
                              density_transform=args.density_transform) for name in args.tilt}
    features = {name: standardize(values) for name, values in raw.items()}
    if getattr(args, "shuffle_tilt", None) is not None:  # control: same weights, no link to (s, a) or the score
        permutation = np.random.default_rng(args.shuffle_tilt).permutation(pool.size)
        features = {name: values[permutation] for name, values in features.items()}
    tilts = [make_tilt(name, features[name], gamma) for name in args.tilt for gamma in args.gamma]
    scores = SCORES if args.score == "both" else (args.score,)
    orders = {name: np.argsort(pool.scores[name], kind="stable") for name in scores}
    risks = {(b, name): ExactRisk(pool.scores[name], tilt.mass, orders[name])
             for b, tilt in enumerate(tilts) for name in scores}
    pieces = (pool.obs, pool.action, pool.action - pool.policy_action)
    center = np.concatenate([piece.mean(axis=0) for piece in pieces])
    spread = np.concatenate([piece.std(axis=0) for piece in pieces])
    groups = episode_groups(pool.episode, pool.timestep) if args.blocks or args.per_episode else None
    return Setup(args, pool, scores, features, raw, tilts, center, np.where(spread > 0, spread, 1.0), groups, risks)


def discriminator_features(setup, tilt, rows):
    if setup.args.discriminator == "feature":
        return setup.features[tilt.name][rows][:, None]
    pool = setup.pool
    stacked = np.hstack([pool.obs[rows], pool.action[rows], pool.action[rows] - pool.policy_action[rows]])
    return (stacked - setup.raw_center) / setup.raw_scale


def run_trial(setup, n, trial):
    """One trial: thresholds [block, score, arm] and weight diagnostics per block."""
    args, pool = setup.args, setup.pool
    seed, options = args.seed, dict(alpha=args.alpha, beta=args.beta, draws=args.draws)
    rows = draw_calibration(stream(seed, "cal", n, trial), n, pool.size, setup.groups, args.per_episode,
                            args.spacing)
    rcps_j = _rcps(rows.size, args.alpha, 1.0 - args.beta)
    thresholds = np.empty((len(setup.tilts), len(setup.scores), len(ARMS)))
    diagnostics = np.empty((len(setup.tilts), 3))  # estimated wbar, Kish n_eff of w-hat and of w*
    cal = {}
    for s, name in enumerate(setup.scores):
        scores = pool.scores[name][rows]
        order = np.argsort(scores, kind="stable")
        cal[name] = scores, order
        thresholds[:, s, 0] = calibrate(scores, stream(seed, "bq", n, trial, name), **options).lambda_hpd
        thresholds[:, s, 1] = math.inf if rcps_j is None else scores[order][rcps_j]
    for b, tilt in enumerate(setup.tilts):
        fit = stream(seed, "fit", n, trial, tilt.name, tilt.gamma)
        cw = fit.integers(pool.size, size=args.weight_fit)
        tw = tilt.sample(fit, args.weight_fit)
        ts = tilt.sample(fit, args.test_mass_size)
        log_ratio = logistic_log_ratio(discriminator_features(setup, tilt, cw),
                                       discriminator_features(setup, tilt, tw), args.ridge)
        log_wbar = float(special.logsumexp(log_ratio(discriminator_features(setup, tilt, ts)))) - math.log(ts.size)
        estimated = common_scale(log_ratio(discriminator_features(setup, tilt, rows)), log_wbar)
        oracle = common_scale(tilt.log_mass[rows], math.log(tilt.square_total / tilt.total))
        diagnostics[b] = [math.exp(min(log_wbar, 700.0)), _kish(estimated[0]), _kish(oracle[0])]
        for s, name in enumerate(setup.scores):
            scores, order = cal[name]
            weights, wbar = estimated
            thresholds[b, s, 2] = (weighted_crc(scores[order], weights[order], wbar, args.alpha)
                                   if weights.max() > 0 else math.inf)
            thresholds[b, s, 3] = _wbcp(scores, stream(seed, "wbcp", n, trial, tilt.name, tilt.gamma, name),
                                        estimated, options)
            thresholds[b, s, 4] = _wbcp(scores, stream(seed, "oracle", n, trial, tilt.name, tilt.gamma, name),
                                        oracle, options)
    return rows.size, thresholds, diagnostics


def _kish(weights):
    # scale-free, so divide by the largest weight first: a bank whose weights are all tiny next to
    # the test mass (common_scale) would otherwise square them to zero
    top = float(weights.max()) if weights.size else 0.0
    if not top > 0:
        return 0.0
    scaled = weights / top
    return float(scaled.sum()) ** 2 / float(np.dot(scaled, scaled))


def _wbcp(scores, rng, scaled, options):
    weights, wbar = scaled
    if not weights.max() > 0:  # every calibration weight underflowed against the test mass
        return math.inf
    return calibrate(scores, rng, weights, max(wbar, np.finfo(np.float64).tiny), **options).threshold


_SETUP = None  # inherited by forked workers


def _trial_chunk(task):
    n, start, stop = task
    return [run_trial(_SETUP, n, trial) for trial in range(start, stop)]


def run_trials(setup, n, trials, workers=1):
    chunk = max(1, math.ceil(trials / (4 * workers)))
    tasks = [(n, start, min(start + chunk, trials)) for start in range(0, trials, chunk)]
    global _SETUP
    _SETUP = setup
    try:
        if workers == 1:
            parts = [_trial_chunk(task) for task in tasks]
        else:
            with multiprocessing.get_context("fork").Pool(workers) as executor:
                parts = executor.map(_trial_chunk, tasks, chunksize=1)
    finally:
        _SETUP = None
    records = [record for part in parts for record in part]
    return (np.array([r[0] for r in records]), np.stack([r[1] for r in records]),
            np.stack([r[2] for r in records]))


def summarize_arm(thresholds, risk, alpha):
    trials = thresholds.size
    certified = np.isfinite(thresholds)
    failures = int(np.sum(risk > alpha))
    low, high = stats.binomtest(failures, trials).proportion_ci(method="exact")
    return dict(
        fail=failures / trials, ci=[float(low), float(high)], failures=failures, trials=trials,
        fail_certified=failures / int(certified.sum()) if certified.any() else None,
        risk=float(np.mean(risk)),  # abstentions count as risk 0
        risk_certified=float(np.mean(risk[certified])) if certified.any() else None,  # reproduce_table1's convention
        threshold=float(np.mean(thresholds[certified])) if certified.any() else None,
        abstain=1.0 - float(np.mean(certified)),
        # how far banks exceed alpha, not only how often (abstentions count as risk 0)
        risk_q95=float(np.quantile(risk, 0.95)), risk_q99=float(np.quantile(risk, 0.99)),
        excess_given_fail=float(np.mean(risk[risk > alpha] - alpha)) if failures else None,
    )


def summarize(setup, n, sizes, thresholds, diagnostics):
    alpha, blocks = setup.args.alpha, []
    for b, tilt in enumerate(setup.tilts):
        wbar_hat = diagnostics[:, b, 0]
        for s, name in enumerate(setup.scores):
            risk = setup.risks[b, name]
            realized = risk(thresholds[:, b, s, :].ravel()).reshape(len(sizes), len(ARMS))
            blocks.append(dict(
                tilt=tilt.name, gamma=tilt.gamma, n=n, score=name,
                lambda_star=risk.lambda_star(alpha), tilt_n_eff=tilt.n_eff, oracle_wbar=tilt.oracle_wbar,
                calibration_size=dict(mean=float(np.mean(sizes)), min=int(sizes.min()), max=int(sizes.max())),
                estimated_wbar=dict(mean=float(np.mean(wbar_hat)), median=float(np.median(wbar_hat)),
                                    q05=float(np.quantile(wbar_hat, 0.05)), q95=float(np.quantile(wbar_hat, 0.95))),
                calibration_n_eff=dict(estimated=float(np.mean(diagnostics[:, b, 1])),
                                       oracle=float(np.mean(diagnostics[:, b, 2]))),
                arms={arm: summarize_arm(thresholds[:, b, s, a], realized[:, a], alpha) for a, arm in enumerate(ARMS)},
            ))
    return blocks


def describe_tilts(setup):
    pool, out = setup.pool, {}
    for name, raw in setup.raw_features.items():
        correlation = {score: float(stats.spearmanr(setup.features[name], pool.scores[score])[0])
                       if np.ptp(setup.features[name]) > 0 else 0.0 for score in setup.scores}
        out[name] = dict(
            feature=dict(mean=float(raw.mean()), std=float(raw.std()), min=float(raw.min()), max=float(raw.max())),
            spearman_with_score=correlation,
            gammas={repr(t.gamma): dict(n_eff=t.n_eff, n_eff_fraction=t.n_eff / pool.size, oracle_wbar=t.oracle_wbar,
                                        max_oracle_w=float(t.mass.size / t.total))
                    for t in setup.tilts if t.name == name},
        )
    return out


def run(args):
    """Run the sweep described by parsed arguments and return the JSON-ready result."""
    started = time.perf_counter()
    setup = build_setup(args)
    setup_seconds = time.perf_counter() - started
    uniform = {name: ExactRisk(setup.pool.scores[name], np.ones(setup.pool.size)).lambda_star(args.alpha)
               for name in setup.scores}
    blocks, timings = [], dict(setup_seconds=setup_seconds, trials={})
    for n in args.n:
        begun = time.perf_counter()
        sizes, thresholds, diagnostics = run_trials(setup, n, args.trials, args.workers)
        seconds = time.perf_counter() - begun
        timings["trials"][str(n)] = dict(seconds=seconds, seconds_per_trial=seconds / args.trials)
        blocks.extend(summarize(setup, n, sizes, thresholds, diagnostics))
    pool = setup.pool
    return dict(
        settings=vars(args), bank_trim=REMAINDER_TRIM, frozen=pool.metadata,
        population=dict(rule=pool.population, rows=pool.size, artifact_rows=pool.artifact_rows,
                        episodes=int(np.unique(pool.episode).size)),
        lambda_star_uniform=uniform, tilts=describe_tilts(setup), blocks=blocks, timings=timings,
    )


def print_table(result):
    settings = result["settings"]
    print(f"{result['frozen'].get('dataset', '?')}  population={result['population']['rule']} "
          f"({result['population']['rows']} rows)  trials={settings['trials']}  draws={settings['draws']}  "
          f"alpha={settings['alpha']}  beta={settings['beta']}  discriminator={settings['discriminator']}"
          f"{'  blocks' if settings['blocks'] else ''}"
          f"{'  per-episode=' + str(settings['per_episode']) if settings.get('per_episode') else ''}"
          f"{' (' + settings['spacing'] + ')' if settings.get('per_episode') else ''}")
    for block in result["blocks"]:
        size = block["calibration_size"]
        print(f"\n[{block['tilt']} gamma={block['gamma']:g} n={block['n']} score={block['score']}]  "
              f"lambda*={block['lambda_star']:.4g}  tilt n_eff={block['tilt_n_eff']:.0f}  "
              f"wbar*={block['oracle_wbar']:.3f}  wbar-hat median={block['estimated_wbar']['median']:.3f}  "
              f"cal n_eff est/oracle={block['calibration_n_eff']['estimated']:.0f}/"
              f"{block['calibration_n_eff']['oracle']:.0f}  cal size={size['mean']:.0f}")
        print(f"  {'rule':<17}{'fail':>8}{'95% CI':>18}{'mean risk':>11}{'mean thr':>10}{'abstain':>9}")
        for arm, row in block["arms"].items():
            ci = f"[{row['ci'][0]:.1%}, {row['ci'][1]:.1%}]"
            thr = "-" if row["threshold"] is None else f"{row['threshold']:.4g}"
            print(f"  {arm:<17}{row['fail']:>8.1%}{ci:>18}{row['risk']:>11.2%}{thr:>10}{row['abstain']:>9.1%}")
    for n, timing in result["timings"]["trials"].items():
        print(f"\nn={n}: {timing['seconds']:.1f} s for trials, {timing['seconds_per_trial']:.3f} s per trial "
              f"(setup {result['timings']['setup_seconds']:.1f} s)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--frozen", required=True, help="directory holding frozen.npz and frozen.json")
    parser.add_argument("--population", choices=("heldout", "all"), default="heldout",
                        help="heldout: rows outside the model's training pool")
    parser.add_argument("--tilt", nargs="+", choices=TILTS, default=["policy"])
    parser.add_argument("--gamma", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--n", nargs="+", type=int, default=[1103], help="scored calibration sizes")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--score", choices=SCORES + ("both",), default="normalized")
    parser.add_argument("--discriminator", choices=("feature", "raw"), default="feature")
    parser.add_argument("--weight-fit", type=int, default=1000, help="|Cw| = |Tw|")
    parser.add_argument("--test-mass-size", type=int, default=1000, help="|Ts|, unlabeled tilted rows")
    parser.add_argument("--ridge", type=float, default=None,
                        help="discriminator L2 penalty on the mean log-likelihood, intercept excluded "
                             "(default 0 for --discriminator feature, which a ridge would bias, 1e-3 for raw)")
    parser.add_argument("--state-feature", default="norm", help="'norm' for ||obs||, or an observation index")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-reference", type=int, default=10_000, help="reference subset size for density")
    parser.add_argument("--knn-seed", type=int, default=20260930, help="fixes the density reference subset")
    parser.add_argument("--shuffle-tilt", type=int, default=None, metavar="SEED",
                        help="control: permute each standardized tilt feature over the pool rows (same "
                             "weight distribution, no relation to (s, a) or the score)")
    parser.add_argument("--density-transform", choices=("log", "raw"), default="log",
                        help="density tilt feature: log k-NN distance (default) or the raw distance")
    parser.add_argument("--blocks", action="store_true", help="draw whole episodes (dependence stress test)")
    parser.add_argument("--per-episode", type=int, default=None, metavar="K",
                        help="thinned bank: ceil(n/K) length-proportional episode draws, K rows each, "
                             "a uniformly random n of them kept")
    parser.add_argument("--spacing", choices=("random", "stratified", "reservation"), default="random",
                        help="with --per-episode: rows uniform in the episode, one per K equal segments, "
                             "or BCA's reservation sampler (distinct episodes, stratified rows)")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=2026093001)
    parser.add_argument("--workers", type=int, default=1, help="forked worker processes (Linux)")
    parser.add_argument("--output", help="optional JSON path; must not exist")
    args = parser.parse_args(argv)
    if args.output and os.path.exists(args.output):
        parser.error("--output already exists")
    for name in ("tilt", "gamma", "n"):
        if len(set(getattr(args, name))) != len(getattr(args, name)):
            parser.error(f"--{name} has repeated values")
    if args.per_episode is not None and (args.per_episode < 1 or args.blocks):
        parser.error("--per-episode needs a positive K and excludes --blocks")
    if min(args.n) < 1 or args.trials < 1 or args.draws < 1 or args.workers < 1:
        parser.error("--n, --trials, --draws and --workers must be positive")
    if not (0 < args.alpha < 1 and 0 < args.beta < 1):
        parser.error("--alpha and --beta must lie strictly between zero and one")
    if args.ridge is None:
        args.ridge = 0.0 if args.discriminator == "feature" else 1e-3
    if args.weight_fit < 1 or args.test_mass_size < 1 or args.ridge < 0:
        parser.error("--weight-fit and --test-mass-size must be positive and --ridge nonnegative")
    if args.knn_k < 1 or args.knn_reference <= args.knn_k:
        parser.error("--knn-reference must exceed --knn-k >= 1")
    if args.state_feature != "norm" and not args.state_feature.isdigit():
        parser.error("--state-feature must be 'norm' or a nonnegative integer")
    if not 0 <= args.seed < 2 ** 63:
        parser.error("--seed must be a nonnegative integer")
    return args


def main(argv=None):
    args = parse_args(argv)
    result = run(args)
    print_table(result)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "x") as handle:
            json.dump(result, handle, indent=1, allow_nan=False)
    return result


BLAS_THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

if __name__ == "__main__":
    # One BLAS thread per process, or forked workers oversubscribe the CPU. NumPy may already
    # be loaded by a site hook, so the limit only takes effect in a fresh interpreter.
    if any(os.environ.get(name) != "1" for name in BLAS_THREADS):
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ, **dict.fromkeys(BLAS_THREADS, "1")))
    main()
