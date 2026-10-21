"""Coverage of any band on any rows, with and without weights, outside and inside the gap.

Pure array functions: nothing here touches a network, a log or BCA's state. Part B gives them the band's numbers (the
bank scores, the target y, the critic's q, the width U); fitting that band on ThrusterChain data waits for the user's
go and is gated there.

Terms (SPEC.md):
  score      |y - q| / (unit x scale) on a calibration-bank row (calibration/reference.freeze_reference's rule)
  threshold  R = wbcp.calibrate(scores, rng, weights, test_mass, alpha 0.1, beta 0.95, 1,000 draws).threshold;
             R = +inf means "cannot certify"
  width      U = R x unit x max(scale, 1e-6) (calibration/dose.frozen_level_dose's rule)
  covered    q - U <= y <= q + U (calibration/network.hard_coverage's rule; ties are covered)
  refused    a step an arm declines to certify: a weighted arm refuses every test step whose recorded command lies
             inside the ball (its exact density ratio is +inf), and any arm refuses every step when R = +inf.
             A refused step never counts as covered.
Arms: 'unweighted' (no weights; never refuses inside the ball), 'exact' (w(c) = p_test / p_train at the bank rows),
'estimated' (the classifier's weights). All arms of one band share the bank scores and one random stream, so only R
differs: calibrate builds a fresh generator from the same seed for each call.

Per-step thresholds (version 4, the owner's decision). Weighted conformal prediction (Tibshirani, Barber, Candes and
Ramdas 2019) gives every test step x its own threshold R(w(x)): the bank weights w_j with the step's own weight as the
test mass. R(t) is non-decreasing in t when every t uses the same random stream (the posterior draws are the same; a
larger test atom only moves each draw's crossing up), and it becomes +inf (cannot certify) beyond some mass. Calling
calibrate once per step is too slow (1,000 draws each), so R is tabulated on a log-spaced grid of test masses
(mass_grid: ratio**k, k integer, covering every mass that is looked up; it holds 1.0 exactly whenever the masses reach
1, and it is [1.0] when every mass is 1). A step uses the smallest grid mass at least its own (lookup), which is
conservative because R is non-decreasing. threshold_table checks that the raw R never falls along the grid and applies
a running maximum if it ever did, recording both. The single-threshold version (one test mass for every step, the mean
weight over test-law steps outside the ball, as in version 3) is kept as a reported comparison ('<arm>_single').

Outcome per step: 1 covered, 0 not covered, -1 refused. coverage = covered / all steps of a group (refused count
against it); the refused share and the coverage among certified steps are reported beside it.
"""

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from calibration import wbcp
from experiments.shift_iw import world as W

ALPHA, BETA, DRAWS = 0.1, 0.95, 1000
COVERED, MISSED, REFUSED = 1, 0, -1
WEIGHTED_ARMS = ("exact", "estimated")
BOOTSTRAP = 2000


# ------------------------------------------------------------------------------------------------ thresholds

def generator(stream):
    """The posterior's random generator from a seed: np.random.default_rng of its uint32 words, as
    calibration/reference.freeze_reference builds it from a JAX key (pass np.asarray(key) of
    fold_in(refresh_key, 1347375956) to reproduce BCA's own threshold)."""
    return np.random.default_rng(np.asarray(stream, np.uint32).ravel())


def calibrate(scores, stream, weights=None, test_mass=None, *, alpha=ALPHA, beta=BETA, draws=DRAWS):
    """wbcp.calibrate with a fresh generator from `stream`. weights None: the unweighted (exchangeable) case."""
    return wbcp.calibrate(np.asarray(scores, np.float64), generator(stream), weights, test_mass, alpha=alpha,
                          beta=beta, draws=draws)


def arm_thresholds(scores, stream, arms, **kw):
    """One threshold per arm on the same scores and the same random stream. arms: {name: None (unweighted) or
    (bank weights, test mass)}. Returns {name: wbcp.Calibration}."""
    out = {}
    for name, spec in arms.items():
        if spec is None:
            out[name] = calibrate(scores, stream, **kw)
        else:
            weights, test_mass = spec
            out[name] = calibrate(scores, stream, np.asarray(weights, np.float64), float(test_mass), **kw)
    return out


def unit_check(scores, stream, **kw):
    """Mark W0: weights of 1 and a test mass of 1 reproduce the unweighted threshold bit for bit (threshold,
    lambda_hat, lambda_hpd and every posterior crossing). Returns (passes, the two calibrations)."""
    a = calibrate(scores, stream, **kw)
    b = calibrate(scores, stream, np.ones(len(scores)), 1.0, **kw)
    same = (np.array_equal(np.float64(a.threshold), np.float64(b.threshold))
            and a.lambda_hat == b.lambda_hat and a.lambda_hpd == b.lambda_hpd
            and np.array_equal(a.crossings, b.crossings))
    return bool(same), a, b


def exact_bank_weights(c, tilt):
    """Exact weights at bank rows: w(c) = p_tilt(c) / p_train(c) (the removal constant Z cancels)."""
    return W.ratio(c, tilt)


def kish(weights):
    """Kish effective size (sum w)^2 / sum w^2."""
    w = np.asarray(weights, np.float64)
    return float(w.sum() ** 2 / np.dot(w, w))


def width(threshold, unit, scale):
    """U = R x (max(scale, 1e-6) x unit) in float32, the arithmetic of calibration/dose.frozen_level_dose; +inf
    when R = +inf. `threshold` is one number or one per row (the per-step thresholds)."""
    s = np.maximum(np.asarray(scale, np.float32), np.float32(1e-6)) * np.float32(unit)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.asarray(threshold, np.float32) * s


def closed_form_dose(width_, unit, blend=0.5, refused_dose=None):
    """The dose 1 + blend U / (U + unit) in float64 (calibration/dose.level_critic_dose's formula without its float32
    rescaling). An infinite width (R = +inf) gets `refused_dose` when given (the per-row BCA arms: 1 + blend, the most
    caution), else 1 (BCA's own rule: an uncertified row gets the native multiplier)."""
    U = np.asarray(width_, np.float64)
    unit = float(unit)
    finite = np.isfinite(U)
    with np.errstate(invalid="ignore"):
        d = np.where(finite, 1.0 + blend * U / (U + unit), 1.0)
    if refused_dose is not None:
        d = np.where(finite, d, float(refused_dose))
    return d


# ------------------------------------------------------------------------------------------------ per-step thresholds

@dataclass(frozen=True)
class Table:
    """R(t) tabulated on a grid of test masses (threshold_table). grid: ascending test masses; raw: calibrate's
    threshold at each; thresholds: the running maximum of raw (what lookup uses; equal to raw when monotone);
    lambda_hat and n_eff do not depend on t."""

    grid: np.ndarray
    raw: np.ndarray
    thresholds: np.ndarray
    lambda_hat: float
    n_eff: float
    monotone: bool
    running_max_applied: bool

    def summary(self, unweighted=None):
        """A JSON-ready description: the grid and both threshold columns, the monotone flag and whether the running
        maximum changed anything, and (given the unweighted R) the ratios at the grid's ends."""
        th = lambda v: [float(x) if math.isfinite(x) else "inf" for x in np.asarray(v, np.float64)]
        out = dict(grid=[float(x) for x in self.grid], raw=th(self.raw), thresholds=th(self.thresholds),
                   lambda_hat=float(self.lambda_hat), n_eff=float(self.n_eff), points=int(len(self.grid)),
                   monotone=bool(self.monotone), running_max_applied=bool(self.running_max_applied),
                   infinite_from=(float(self.grid[np.argmax(np.isinf(self.thresholds))])
                                  if np.isinf(self.thresholds).any() else None))
        if unweighted is not None and math.isfinite(unweighted) and unweighted > 0:
            ends = (self.thresholds[0], self.thresholds[-1])
            out["ratio_to_unweighted_at_ends"] = [float(r / unweighted) if math.isfinite(r) else "inf" for r in ends]
        return out


def mass_grid(masses, ratio):
    """The log-spaced grid of test masses ratio**k (k integer) from the largest grid mass at most min(masses) to the
    smallest at least max(masses). It holds 1.0 exactly when the masses reach 1, and it is [1.0] when every mass is 1.
    masses: every (finite, positive) mass that will be looked up."""
    m = np.asarray(masses, np.float64).ravel()
    ratio = float(ratio)
    if not m.size or not np.all(np.isfinite(m) & (m > 0)):
        raise ValueError("test masses must be a nonempty vector of finite positive numbers")
    if not (math.isfinite(ratio) and ratio > 1.0):
        raise ValueError("the grid ratio must be a finite number above 1")
    step = math.log(ratio)
    lo = math.floor(math.log(float(m.min())) / step)
    hi = math.ceil(math.log(float(m.max())) / step)
    while math.exp(lo * step) > m.min():      # rounding guards: the ends must bracket every mass
        lo -= 1
    while math.exp(hi * step) < m.max():
        hi += 1
    return np.exp(np.arange(lo, hi + 1, dtype=np.float64) * step)


def threshold_table(scores, stream, weights, grid, *, alpha=ALPHA, beta=BETA, draws=DRAWS):
    """R(t) = calibrate(scores, stream, weights, t).threshold at every grid mass t, each call with a fresh generator
    from the same stream (so every t sees the same posterior draws). Checks that R never falls along the grid
    (monotone) and takes the running maximum if it ever does (running_max_applied). Returns a Table."""
    grid = np.asarray(grid, np.float64)
    if grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid) & (grid > 0)) or np.any(grid[1:] <= grid[:-1]):
        raise ValueError("the grid must be an ascending vector of finite positive test masses")
    weights = np.asarray(weights, np.float64)
    cals = [calibrate(scores, stream, weights, float(t), alpha=alpha, beta=beta, draws=draws) for t in grid]
    raw = np.array([c.threshold for c in cals], np.float64)
    monotone = bool(np.all(raw[1:] >= raw[:-1]))
    used = np.maximum.accumulate(raw)
    return Table(grid, raw, used, float(cals[0].lambda_hat), float(cals[0].n_eff), monotone,
                 bool(np.any(used != raw)))


def table_from_summary(summary):
    """A Table back from Table.summary() (a band record; 'inf' strings to +inf)."""
    f = lambda v: np.array([float(x) for x in v], np.float64)
    return Table(f(summary["grid"]), f(summary["raw"]), f(summary["thresholds"]), float(summary["lambda_hat"]),
                 float(summary["n_eff"]), bool(summary["monotone"]), bool(summary["running_max_applied"]))


def lookup(table, masses):
    """Per step: the threshold at the smallest grid mass at least the step's own test mass (conservative, since R is
    non-decreasing). Refuses a mass beyond the grid. Returns (R per step, grid index per step)."""
    m = np.asarray(masses, np.float64)
    if not np.all(np.isfinite(m) & (m > 0)):
        raise ValueError("test masses must be finite and positive")
    idx = np.searchsorted(table.grid, m, side="left")
    if np.any(idx >= len(table.grid)):
        raise ValueError("a test mass lies beyond the grid; build the grid on every mass that is looked up")
    return table.thresholds[idx], idx


def lookup_gaps(table, masses, idx=None):
    """How far the grid lookup can sit above a step's exact threshold: the largest relative mass gap
    grid[k] / mass - 1, and the largest relative threshold step R(t_k) / R(t_(k-1)) - 1 over the grid points used
    (R(t_(k-1)) <= R(mass) <= R(t_k)).

    possibly_rounded_into_refusal is an upper bound on the steps refused only because the lookup rounded their mass up:
    it counts every step whose looked-up R is +inf while the grid point below is finite. That includes steps whose own
    exact R is already +inf (a mass on the first infinite grid point, or between the true onset of +inf and that
    point), which the table cannot tell apart; the exact count is at most this number."""
    m = np.asarray(masses, np.float64)
    if idx is None:
        idx = lookup(table, m)[1]
    idx = np.asarray(idx)
    if not m.size:
        return dict(steps=0, max_mass_gap=None, max_threshold_gap=None, possibly_rounded_into_refusal=0)
    R = table.thresholds
    prev = np.where(idx > 0, R[np.maximum(idx - 1, 0)], R[idx])
    cur = R[idx]
    finite = np.isfinite(cur) & np.isfinite(prev) & (prev > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.where(finite, cur / np.where(finite, prev, 1.0) - 1.0, 0.0)
    return dict(steps=int(m.size), max_mass_gap=float(np.max(table.grid[idx] / m - 1.0)),
                max_threshold_gap=float(step.max()),
                possibly_rounded_into_refusal=int(np.sum(np.isinf(cur) & np.isfinite(prev) & (idx > 0))))


def per_step_summary(R_steps, unweighted=None):
    """The distribution of per-step thresholds over a group of steps: share refused (R = +inf) and, over the
    certified steps, the smallest, mean and largest R (and their ratio to the unweighted R when given)."""
    R = np.asarray(R_steps, np.float64)
    fin = np.isfinite(R)
    out = dict(steps=int(R.size), refused_share=float(np.mean(~fin)) if R.size else None)
    if fin.any():
        v = R[fin]
        out.update(min=float(v.min()), mean=float(v.mean()), max=float(v.max()))
        if unweighted is not None and math.isfinite(unweighted) and unweighted > 0:
            out.update(ratio_min=float(v.min() / unweighted), ratio_mean=float(v.mean() / unweighted),
                       ratio_max=float(v.max() / unweighted))
    return out


# ------------------------------------------------------------------------------------------------ outcomes

def refusals(in_ball, arm):
    """Steps an arm refuses before any threshold: a weighted arm refuses every step whose command is inside the ball
    (exact density ratio +inf); the unweighted arm refuses none."""
    in_ball = np.asarray(in_ball, bool)
    if arm in WEIGHTED_ARMS:
        return in_ball.copy()
    if arm == "unweighted":
        return np.zeros(in_ball.shape, bool)
    raise ValueError(f"arm must be 'unweighted' or one of {WEIGHTED_ARMS}")


def outcomes(y, q, U, refused=None):
    """Per step: 1 covered (q - U <= y <= q + U), 0 not covered, -1 refused (given refusals, or a non-finite width:
    R = +inf, cannot certify). A refused step is never covered."""
    y, q, U = (np.asarray(v, np.float64) for v in (y, q, U))
    covered = (y >= q - U) & (y <= q + U)
    out = np.where(covered, COVERED, MISSED).astype(np.int8)
    refuse = ~np.isfinite(U)
    if refused is not None:
        refuse = refuse | np.asarray(refused, bool)
    out[refuse] = REFUSED
    return out


def episode_bootstrap(values, episodes, boot=BOOTSTRAP, seed=(W.MASTER_SEED, 77), level=0.95, counts=None):
    """Percentile interval of a mean (or ratio sum(values) / sum(counts)) by resampling whole episodes."""
    values = np.asarray(values, np.float64)
    counts = np.ones_like(values) if counts is None else np.asarray(counts, np.float64)
    ids, inv = np.unique(np.asarray(episodes), return_inverse=True)
    sums = np.bincount(inv, values, len(ids))
    ns = np.bincount(inv, counts, len(ids))
    rng = np.random.default_rng(list(seed))
    idx = rng.integers(len(ids), size=(int(boot), len(ids)))
    with np.errstate(invalid="ignore", divide="ignore"):
        reps = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.nanquantile(reps, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def summarize(outcome, episodes, mask=None, boot=BOOTSTRAP, seed=(W.MASTER_SEED, 77)):
    """Coverage of a group of steps (mask; all by default): steps, coverage = covered / steps (refused count
    against it) with an episode-bootstrap 95% interval, refused share, and coverage among the certified steps."""
    o = np.asarray(outcome)
    ep = np.asarray(episodes)
    if mask is not None:
        o, ep = o[np.asarray(mask, bool)], ep[np.asarray(mask, bool)]
    n = int(len(o))
    if n == 0:
        return dict(steps=0, coverage=None, interval=None, refused_share=None, certified_coverage=None)
    cov = (o == COVERED).astype(np.float64)
    ref = o == REFUSED
    certified = int((~ref).sum())
    return dict(steps=n, episodes=int(len(np.unique(ep))), coverage=float(cov.mean()),
                interval=episode_bootstrap(cov, ep, boot, seed), refused_share=float(ref.mean()),
                certified_steps=certified,
                certified_coverage=float(cov[~ref].mean()) if certified else None)


def groups(outcome, episodes, in_ball, **kw):
    """summarize() outside the ball, inside the ball and overall."""
    in_ball = np.asarray(in_ball, bool)
    return dict(outside=summarize(outcome, episodes, ~in_ball, **kw), inside=summarize(outcome, episodes, in_ball, **kw),
                all=summarize(outcome, episodes, None, **kw))


# ------------------------------------------------------------------------------------------------ across seeds

def across_seeds(values, level=0.95):
    """Pooled mean of per-seed values, its two-sided t interval and one-sided lower bound (n - 1 degrees of
    freedom; 4 for five seeds)."""
    v = np.asarray(values, np.float64)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return dict(n=n, mean=m, sd=None, interval=None, lower_one_sided=None)
    se = float(v.std(ddof=1) / math.sqrt(n))
    t2 = stats.t.ppf((1 + level) / 2, n - 1)
    t1 = stats.t.ppf(level, n - 1)
    return dict(n=n, mean=m, sd=float(v.std(ddof=1)), interval=(m - t2 * se, m + t2 * se), lower_one_sided=m - t1 * se)


# ------------------------------------------------------------------------------------------------ forecast and checks

def forecast(c, covered, tilt):
    """Unweighted coverage under a tilt, forecast from no-tilt held-out steps outside the ball: given c, steps outside
    the ball are identical under every law, so the forecast is the w(c)-weighted mean of their covered indicators."""
    w = W.ratio(c, tilt)
    cov = np.asarray(covered, np.float64)
    return float(np.sum(w * cov) / np.sum(w))


def by_current(c, values, edges=None):
    """Mean of values in current bins (default 10 equal bins over [0.1, 1.1]): (edges, means, counts)."""
    c = np.asarray(c, np.float64)
    v = np.asarray(values, np.float64)
    edges = np.linspace(W.C_LO, W.C_HI, 11) if edges is None else np.asarray(edges, np.float64)
    b = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, len(edges) - 2)
    counts = np.bincount(b, minlength=len(edges) - 1)
    sums = np.bincount(b, v, minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return edges, sums / counts, counts


def decile_spearman(width, error, c, deciles=10):
    """Mark M2's statistic: Spearman's rank correlation of the width with the exact critic error inside each current
    decile (tenth of the rows by c), averaged over the deciles; also the pooled correlation and the per-decile
    values."""
    width, error, c = (np.asarray(v, np.float64) for v in (width, error, c))
    order = np.argsort(c, kind="stable")
    parts = np.array_split(order, int(deciles))
    per = [float(stats.spearmanr(width[p], error[p])[0]) for p in parts]
    return dict(within_deciles=float(np.mean(per)), per_decile=per, pooled=float(stats.spearmanr(width, error)[0]))


def dose_order(width, dose, unit=None):
    """Mark M3's statistics on rows of finite width, sorted by width (ties by dose): the largest fall of the dose as
    the width rises (M3 allows 1e-6, spec.json), the raw count of falls, and, given the unit, the largest gap between
    the dose and its closed form 1 + 0.5 U / (U + unit) in float64.

    BCA computes the dose in float32 under jax.jit, so between neighbouring distinct widths it can come out one float32
    step (about 1.2e-7 near 1.3) lower: a few such falls per 10,000 rows are rounding, not a fault."""
    width, dose = np.asarray(width, np.float64), np.asarray(dose, np.float64)
    keep = np.isfinite(width)
    w, d = width[keep], dose[keep]
    step = np.diff(d[np.lexsort((d, w))])
    out = dict(rows=int(keep.sum()), inversions=int(np.sum(step < 0)),
               max_drop=float(max(0.0, -float(step.min()))) if len(step) else 0.0)
    if unit is not None:
        u = float(np.float32(unit))
        out["max_closed_form_gap"] = float(np.abs(d - (1.0 + 0.5 * w / (w + u))).max()) if len(w) else 0.0
    return out
