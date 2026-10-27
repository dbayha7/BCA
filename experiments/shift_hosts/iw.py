"""IW (importance-weighted) BCA's arithmetic for the four restated Unifloral steps (plan sections 4.2 and 4.6; choices 16
and 20): the threshold table R(t) on a grid of test masses, the lookup that gives every training row its own threshold,
the per-row dose (TD3+BC, ReBRAC, CQL) and IQL's per-row actor weights, each learner's refusal rule, the threshold
reproduction of stop rule S3, the IW step after each standard refresh, and the last refresh's two readouts (the
estimated-weight table and the shuffled-weights placebo). Training environment; it also imports in /opt/bca-shift-venv
(it needs numpy, JAX, calibration/ and this package's bca_core and guard only).

An experiment-local attachment (plan section 3.1): it changes none of the repository's declared methods.

Why restated. The weighting test's own code cannot be imported in /opt/bca-venv: experiments/shift_iw imports
experiments.shift (experiments/shift_iw/__init__.py:24), which refuses numpy other than 2.2.6
(experiments/shift/__init__.py:101-103), and /opt/bca-venv has numpy 1.26.4. Test (c3), tests/test_iw_compare.py, runs the
originals in /opt/bca-shift-venv on made-up inputs and compares their saved outputs with this module bit for bit.

Terms (plan section 4.2; experiments/shift_iw/coverage.py:7-31):
  score       |y - q| / (u x max(eta, 1e-6)) on a bank row, exactly as the refresh computed it (bca_core.bank_scores)
  stream      the uint32 words of the refresh's posterior key. Every calibrate call builds a fresh numpy generator from
              it, so every test mass sees the same posterior draws (bca_core.posterior_stream)
  test mass   a training row's (or a test step's) own weight; the bank rows carry their own weights
  R(t)        wbcp.calibrate(scores, stream, bank weights, test mass t).threshold, with miscoverage 0.1, credibility 0.95
              and 1,000 draws (bca_core.SETTINGS, the same numbers as coverage.py:44). R(t) does not fall as t grows,
              and it is +inf ("cannot certify": the row is refused) beyond some mass
  grid        t = 1.01^k, k integer, from the largest grid mass at most the smallest mass looked up to the smallest grid
              mass at least the largest (about 400 points for the exact weights 0.135-7.389); [1.0] when every mass is 1
  lookup      a row's threshold is R at the smallest grid mass at least its own mass: the mass is rounded up, which is
              conservative because R(t) does not fall

Restated, line for line in substance (the names and the settings' source differ; nothing else):
  generator, calibrate     coverage.py:52-62 (the settings come from bca_core.SETTINGS)
  kish                     coverage.py:94-97
  Table, mass_grid, threshold_table, table_from_summary, lookup
                           coverage.py:124-205
  lookup_gaps, per_step_summary
                           coverage.py:208-246 (readouts for the refresh record)
  reproduces_calibration   learners_iw.py:555-561. reproduces(reference, scores, stream) applies it to the unweighted
                           calibrate of the scores, as learners_iw.py:936-938 does (stop rule S3)
  table_float32            learners_iw.py:589-597. row_thresholds indexes it by each row's grid index, which is the
                           threshold the weighting test's per-row step reads (table[row_index[idx]], learners_iw.py:831)
  threshold_tables         learners_iw.py:582-586, with this package's guard in place of the weighting test's gate
  per_row_dose             learners_iw.py:655-683, with two declared changes. (1) The scale network's prediction eta
                           is passed in: the restated steps compute it with the frozen reference's parameters, as line
                           664 does. (2) The refusal rule is the carried boolean most_caution: True gives a refused row
                           the dose 1 + blend (line 682, the original's only rule); False gives it standard BCA's dose 1
                           (calibration/dose.py:65-95). With most_caution True it is the original.
  dose_change              learners_iw.py:719-729
New (declared; plan section 4.2 and choice 16; plan section 4.6 and choice 20):
  per_row_iql_weights      IQL's per-row actor weights: the IQL port's own rule (calibration/iql_reference.py:157-181,
                           calibration/advantage.py:32-70) with each row's own threshold; Unifloral's own weight
                           (iql.py:245) before the first refresh and on rows without a positive advantage; and the two
                           refusal rules
  REFUSAL, most_caution, refusal_name, refusal_record
                           each learner's refusal rule (choice 16)
  iw_refresh               the IW step after each standard refresh (learners_iw.py:935-952, for a bca_core.BCAState):
                           S3, the table, and every training row's threshold
  placebo_permutation, last_refresh_arms
                           the last refresh's readouts: the estimated-weight table (choice 20) and the shuffled-weights
                           placebo (plan section 4.6). The placebo's permutation stream, default_rng([20261016, rep, 95]),
                           is new: the plan names none, and tag 95 is used nowhere else in the repository

Permission. The arithmetic here (calibrate, the table, the lookup, the dose and the weights) checks nothing: most of it
runs inside the restated steps' jitted code, and the BCA entry points that call it check (INTERFACES.md section 8). The
two callers of the tables defined here, iw_refresh and threshold_tables, are entry points and call
guard.require_bca(permission, bank) first, as the weighting test's threshold_tables calls its gate (learners_iw.py:585).
"""

import hashlib
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from calibration import wbcp
from experiments.shift_hosts import bca_core as BC
from experiments.shift_hosts import guard
from experiments.shift_hosts.bca_core import (LevelActorWeights, level_critic_dose, positive_scale,
                                              postcap_level_actor_weights, reference_valid, tree_finite)

GRID_RATIO = 1.01           # experiments/iw/spec.json:367 (band.per_step.grid_ratio); plan section 4.2
TAG_PLACEBO = 95            # the placebo's permutation stream default_rng([20261016, rep, 95]) (new; module docstring)
SETTINGS = BC.SETTINGS


# ------------------------------------------------------------------------------------------------ thresholds

def generator(stream):
    """The posterior's random generator from the refresh's stream (coverage.py:52-56): np.random.default_rng of its
    uint32 words, as calibration/reference.freeze_reference builds it (bca_core.posterior_stream)."""
    return np.random.default_rng(BC.posterior_stream(stream))


def calibrate(scores, stream, weights=None, test_mass=None, *, settings=SETTINGS):
    """wbcp.calibrate with a fresh generator from `stream` (coverage.py:59-62); weights None is the unweighted
    (exchangeable) case. Miscoverage, credibility and draws from bca_core's settings (0.1, 0.95, 1,000)."""
    return wbcp.calibrate(np.asarray(scores, np.float64), generator(stream), weights, test_mass,
                          alpha=settings.miscoverage, beta=settings.credibility, draws=settings.draws)


def kish(weights):
    """Kish effective size (sum w)^2 / sum w^2 (coverage.py:94-97)."""
    w = np.asarray(weights, np.float64)
    return float(w.sum() ** 2 / np.dot(w, w))


@dataclass(frozen=True)
class Table:
    """R(t) tabulated on a grid of test masses (threshold_table; coverage.py:124-150). grid: ascending test masses;
    raw: calibrate's threshold at each; thresholds: the running maximum of raw (what lookup uses; equal to raw when
    monotone); lambda_hat and n_eff do not depend on t."""

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


def mass_grid(masses, ratio=GRID_RATIO):
    """The log-spaced grid of test masses ratio**k (k integer) from the largest grid mass at most min(masses) to the
    smallest at least max(masses) (coverage.py:153-170). It holds 1.0 exactly when the masses reach 1, and it is [1.0]
    when every mass is 1. masses: every (finite, positive) mass that will be looked up. A grid point depends only on k,
    so a grid built on more masses has the same points where the two overlap."""
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


def threshold_table(scores, stream, weights, grid, *, settings=SETTINGS):
    """R(t) = calibrate(scores, stream, weights, t).threshold at every grid mass t, each call with a fresh generator
    from the same stream, so every t sees the same posterior draws (coverage.py:173-186). Checks that R never falls
    along the grid (monotone) and takes the running maximum if it ever does (running_max_applied). Returns a Table."""
    grid = np.asarray(grid, np.float64)
    if grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid) & (grid > 0)) or np.any(grid[1:] <= grid[:-1]):
        raise ValueError("the grid must be an ascending vector of finite positive test masses")
    weights = np.asarray(weights, np.float64)
    cals = [calibrate(scores, stream, weights, float(t), settings=settings) for t in grid]
    raw = np.array([c.threshold for c in cals], np.float64)
    monotone = bool(np.all(raw[1:] >= raw[:-1]))
    used = np.maximum.accumulate(raw)
    return Table(grid, raw, used, float(cals[0].lambda_hat), float(cals[0].n_eff), monotone,
                 bool(np.any(used != raw)))


def table_from_summary(summary):
    """A Table back from Table.summary() ('inf' strings to +inf; coverage.py:189-193)."""
    f = lambda v: np.array([float(x) for x in v], np.float64)
    return Table(f(summary["grid"]), f(summary["raw"]), f(summary["thresholds"]), float(summary["lambda_hat"]),
                 float(summary["n_eff"]), bool(summary["monotone"]), bool(summary["running_max_applied"]))


def lookup(table, masses):
    """Per row: the threshold at the smallest grid mass at least the row's own test mass (conservative, since R does
    not fall; coverage.py:196-205). Refuses a mass beyond the grid. Returns (R per row float64, grid index per row)."""
    m = np.asarray(masses, np.float64)
    if not np.all(np.isfinite(m) & (m > 0)):
        raise ValueError("test masses must be finite and positive")
    idx = np.searchsorted(table.grid, m, side="left")
    if np.any(idx >= len(table.grid)):
        raise ValueError("a test mass lies beyond the grid; build the grid on every mass that is looked up")
    return table.thresholds[idx], idx


def lookup_gaps(table, masses, idx=None):
    """How far the grid lookup can sit above a row's exact threshold (coverage.py:208-231): the largest relative mass
    gap grid[k] / mass - 1, the largest relative threshold step R(t_k) / R(t_(k-1)) - 1 over the grid points used, and
    possibly_rounded_into_refusal, an upper bound on the rows refused only because the lookup rounded their mass up
    (every row whose looked-up R is +inf while the grid point below is finite)."""
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
    """The distribution of per-row thresholds (coverage.py:234-246): share refused (R = +inf) and, over the certified
    rows, the smallest, mean and largest R (and their ratio to the unweighted R when given)."""
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


def reproduces_calibration(reference, calibration):
    """Whether a calibration stored in float32 equals a frozen reference's threshold, lambda_hat, lambda_hpd and n_eff
    bit for bit (learners_iw.py:555-561)."""
    with np.errstate(over="ignore"):
        pairs = [(calibration.threshold, reference.threshold), (calibration.lambda_hat, reference.lambda_hat),
                 (calibration.lambda_hpd, reference.lambda_hpd), (calibration.n_eff, reference.n_eff)]
        return all(np.float32(a).tobytes() == np.asarray(b, np.float32).tobytes() for a, b in pairs)


def reproduces(reference, scores, stream, *, settings=SETTINGS):
    """Stop rule S3: whether an unweighted calibrate on the bank scores and the refresh's stream reproduces the stored
    standard threshold (and lambda_hat, lambda_hpd, n_eff) bit for bit (learners_iw.py:936-938)."""
    return reproduces_calibration(reference, calibrate(scores, stream, settings=settings))


def table_float32(table):
    """A table's thresholds as BCA stores a threshold (float32); a finite threshold that overflows float32 is refused
    (freeze_reference's guard), so +inf in float32 always means 'cannot certify' (learners_iw.py:589-597)."""
    R = np.asarray(table.thresholds, np.float64)
    with np.errstate(over="ignore"):
        stored = R.astype(np.float32)
    if np.any(np.isfinite(R) & ~np.isfinite(stored)):
        raise ValueError("a finite threshold must stay finite when stored in float32")
    return stored


def row_thresholds(table, masses):
    """Every row's own threshold as BCA stores it: the table in float32 (table_float32) at each row's grid index
    (lookup), the value the weighting test's per-row step reads, table[row_index[idx]] (learners_iw.py:831, 947).
    Returns float32 [n]; the runner puts it into BCAState.thresholds of an IW setup."""
    return table_float32(table)[lookup(table, masses)[1]]


def threshold_tables(scores, stream, arms, *, permission, bank, settings=SETTINGS):
    """Several tables of R on the same scores and stream (learners_iw.py:582-586): arms {name: (bank weights, grid of
    test masses)}. An entry point: guard.require_bca(permission, bank) first (the weighting test calls its gate there).
    Returns {name: Table}."""
    guard.require_bca(permission, bank)
    return {name: threshold_table(scores, stream, w, grid, settings=settings) for name, (w, grid) in arms.items()}


# ------------------------------------------------------------------------------------------------ per-row dose and weights

def _flag(most_caution):
    flag = jnp.asarray(most_caution)
    if flag.ndim != 0 or flag.dtype != jnp.bool_:
        raise TypeError("most_caution must be a boolean scalar (BCAState.most_caution)")
    return flag


def per_row_dose(reference, eta, R_rows, blend, most_caution):
    """BCA's frozen-width dose with each row's own threshold (learners_iw.py:655-683): width_i = R_i x max(eta_i, 1e-6)
    x u with frozen_level_dose's validity checks taken row by row, then dose_i = 1 + blend U_i / (U_i + u) by
    calibration.dose.level_critic_dose (BCA's own arithmetic) for a finite R_i. Before the first refresh (reference not
    ready) every width is 0 and every dose exactly 1. A refused row (reference ready, R_i = +inf) gets, by the carried
    refusal rule most_caution (a traced boolean, so standard and IW BCA compile one program):
      False   dose 1, standard BCA's rule (an infinite width is unsupported; calibration/dose.py:65-95)
      True    dose 1 + blend = 1.5, the most caution: the weighting test's own rule for TD3+BC (lines 681-682), and the
              new rule for CQL and ReBRAC (choice 16)
    With every R_i equal to the reference's threshold this is frozen_level_dose (and the ports' bc_readout, which adds
    reference_valid to inputs_valid) bit for bit. eta: the scale network's prediction with the frozen reference's
    parameters at the rows (learners_iw.py:664 computes it inside). Traceable. Returns (CQLLevelDose, refused rows)."""
    ref = reference
    predictions = jnp.asarray(eta)                       # declared change (1): eta is passed in (line 664)
    R = jnp.asarray(R_rows, jnp.float32)
    if predictions.ndim != 1 or R.shape != predictions.shape:
        raise ValueError("per-row doses need one threshold per row")
    caution = _flag(most_caution)                        # declared change (2): the carried refusal rule
    unit = ref.residual_scale
    scale = jnp.maximum(predictions, 1e-06) * unit
    width = R * scale
    valid = (tree_finite(ref.cal_params)
             & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
             & jnp.isfinite(unit) & (unit > 0)
             & jnp.all(jnp.isfinite(scale) & (scale > 0))
             & jnp.all(~jnp.isnan(R) & (R >= 0))
             & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0)))))
    width = jnp.where(ref.ready, width, jnp.zeros_like(width))
    valid = jnp.where(ref.ready, valid, True)
    unit = jnp.where(ref.ready, unit, 1.0)
    base = level_critic_dose(width, jnp.ones_like(width, dtype=bool), unit, blend, valid)
    refused = ref.ready & jnp.isposinf(R)
    dose = jnp.where(refused & caution, jnp.asarray(1.0 + blend, base.dose.dtype), base.dose)   # (2): `& caution`
    return base._replace(dose=dose, inputs_valid=base.inputs_valid & reference_valid(ref)), refused


def per_row_iql_weights(reference, eta, R_rows, adv, beta, cap, most_caution):
    """IQL's BCA actor weights with each row's own threshold (new, plan sections 4.2-4.3). The band's width and support
    are the IQL port's own (calibration/iql_reference.py:157-181, posterior_width and weights_at_reference) with R_i in
    place of the one threshold, and its validity checks taken row by row as in per_row_dose:
      width_i = R_i x max(eta_i, 1e-6) x u; a row is supported when its width is finite and the inputs are valid.
    Weights (beta = Unifloral's args.beta, cap = args.exp_adv_clip):
      before the first refresh       Unifloral's own weight jnp.exp(adv * beta).clip(max=cap) (iql.py:245) on every
                                     row, bit for bit, so a band that is not ready leaves Unifloral's step unchanged
      supported, advantage > 0       BCA's post-cap shrinkage 1 + A / (A + U) x (min(e^(beta A), cap) - 1)
                                     (calibration/advantage.py:32-70, postcap_level_actor_weights; decision gain 1)
      supported, advantage <= 0      Unifloral's own weight (equal to the port's capped weight there)
      refused (ready, R_i = +inf)    most_caution False: weight 0 and unsupported, standard BCA's rule (the port's
                                     unsupported row; when no row is supported, has_support is False and the step skips
                                     its actor update, calibration/iql_state.py:45-79). most_caution True: weight 1
                                     where A > 0 (plain cloning, the limit of the shrinkage as U grows without bound)
                                     and Unifloral's own weight elsewhere, and the row is supported (a new rule, choice
                                     16)
    With every R_i equal to the reference's threshold and most_caution False this equals the port's
    weights_at_reference bit for bit. used_advantage is the port's, except on refused rows under most_caution (0 where
    A > 0, A elsewhere). Traceable. Returns (calibration.advantage.LevelActorWeights, refused rows)."""
    ref = reference
    predictions = jnp.asarray(eta)
    R = jnp.asarray(R_rows, jnp.float32)
    adv = jnp.asarray(adv)
    if predictions.ndim != 1 or R.shape != predictions.shape or adv.shape != predictions.shape:
        raise ValueError("per-row actor weights need one threshold and one advantage per row")
    caution = _flag(most_caution)
    unit = ref.residual_scale
    scale = positive_scale(predictions, unit)            # iql_reference.posterior_width, lines 159-171, row by row
    valid = (reference_valid(ref)
             & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
             & jnp.all(jnp.isfinite(scale) & (scale > 0))
             & jnp.all(~jnp.isnan(R) & (R >= 0)))
    width = R * scale
    valid = valid & jnp.all(jnp.isposinf(R) | (jnp.isfinite(width) & ((R == 0) | (width > 0))))
    usable = jnp.isfinite(width) & valid
    width = jnp.where(ref.ready, width, 0.0)              # iql_reference.weights_at_reference, lines 174-181
    usable = jnp.where(ref.ready, usable, True)
    valid = jnp.where(ref.ready, valid, True)
    base = postcap_level_actor_weights(adv, width, usable, beta, cap, valid)
    plain = jnp.exp(adv * beta).clip(max=cap)             # Unifloral's own weight, iql.py:245
    positive = adv > 0.0
    refused = ref.ready & jnp.isposinf(R)
    most = refused & caution & base.inputs_valid          # refused rows under IW's rule
    certified = jnp.where(positive, base.weights, jnp.where(base.support_mask, plain, 0.0))
    weights = jnp.where(ref.ready, jnp.where(most, jnp.where(positive, 1.0, plain), certified), plain)
    support = base.support_mask | most
    used = jnp.where(most, jnp.where(positive, 0.0, adv), base.used_advantage)
    out = LevelActorWeights(weights.astype(base.weights.dtype), used, support, base.inputs_valid, jnp.any(support))
    return jax.tree_util.tree_map(jax.lax.stop_gradient, out), refused


def dose_change(dose, base, high=None):
    """The dose statistics of one arm against another arm's doses on the same rows (learners_iw.py:719-729): the
    mean, the mean change, the mean absolute change, the largest absolute change and (given a mask of high-current rows)
    the mean change there. Plan section 5.4's "IW's movement" at the last refresh (IQL: the actor weights)."""
    d = np.asarray(dose, np.float64) - np.asarray(base, np.float64)
    out = dict(mean=float(np.mean(dose)), change_vs_unweighted=float(d.mean()), mean_abs_change=float(np.abs(d).mean()),
               max_abs_change=float(np.abs(d).max()))
    if high is not None:
        high = np.asarray(high, bool)
        out.update(high_current_rows=int(high.sum()),
                   high_current_change=float(d[high].mean()) if high.any() else None)
    return out


# ------------------------------------------------------------------------------------------------ refusal rules

REFUSAL = {
    # plan section 4.2 and choice 16: a refused row (R = +inf) gets the limit of the learner's own formula as the width
    # grows without bound under IW BCA ("refusal means most caution"); standard BCA keeps its own rule
    "td3_bc": dict(
        standard="dose 1", standard_source="calibration/dose.py:65-95 (an infinite width is unsupported: dose 1)",
        most_caution="dose 1 + blend (1.5)",
        most_caution_source="experiments/shift_iw/learners_iw.py:681-682 (the weighting test's own code)",
        new_rule=False),
    "cql": dict(
        standard="dose 1", standard_source="calibration/dose.py:65-95 (an infinite width is unsupported: dose 1)",
        most_caution="dose 1 + blend (1.5)",
        most_caution_source="new rule (choice 16): the limit of 1 + 0.5 U / (U + u) as U grows; the weighting test's "
                            "principle, not its code",
        new_rule=True),
    "rebrac": dict(
        standard="dose 1", standard_source="calibration/dose.py:65-95 (an infinite width is unsupported: dose 1)",
        most_caution="dose 1 + blend (1.5)",
        most_caution_source="new rule (choice 16): the limit of 1 + 0.5 U / (U + u) as U grows; the weighting test's "
                            "principle, not its code",
        new_rule=True),
    "iql": dict(
        standard="weight 0 (unsupported); no actor update when no row of the batch is supported",
        standard_source="calibration/iql_reference.py:174-181, calibration/advantage.py:32-70, "
                        "calibration/iql_state.py:45-79 (ALGORITHMS.md procedure D)",
        most_caution="weight 1 where the advantage is positive (plain cloning), Unifloral's weight "
                     "min(e^(beta A), cap) elsewhere",
        most_caution_source="new rule (choice 16): the limit of 1 + A / (A + U) (min(e^(beta A), cap) - 1) as U grows; "
                            "the weighting test's principle, not its code",
        new_rule=True),
}

# The carried refusal rule of each setup (INTERFACES.md section 3): standard BCA's own rule, or the most caution for the
# IW setups (including stage 1's weights-of-1 and repeat runs). The sham alone never refreshes, so its rule never acts.
SETUP_MOST_CAUTION = dict(standard=False, sham_alone=False, iw=True, iw_ones=True, iw_repeat=True)


def most_caution(setup):
    """BCAState.most_caution for a setup name (SETUP_MOST_CAUTION)."""
    if setup not in SETUP_MOST_CAUTION:
        raise ValueError(f"no refusal rule for setup {setup!r}; BCA setups are {sorted(SETUP_MOST_CAUTION)}")
    return SETUP_MOST_CAUTION[setup]


def refusal_name(most_caution_):
    """The run record's settings.refusal: "most_caution" or "standard"."""
    return "most_caution" if bool(most_caution_) else "standard"


def refusal_record(learner, most_caution_, blend=SETTINGS.blend):
    """A JSON-ready description of a learner's refusal rule under one setting (for the run record)."""
    if learner not in REFUSAL:
        raise ValueError(f"learner must be one of {sorted(REFUSAL)}")
    rule, key = REFUSAL[learner], refusal_name(most_caution_)
    out = dict(learner=learner, rule=key, refused_rows_get=rule[key], source=rule[key + "_source"],
               new_rule=bool(bool(most_caution_) and rule["new_rule"]))
    if learner != "iql":
        out["refused_dose"] = float(1.0 + blend) if bool(most_caution_) else 1.0
    return out


# ------------------------------------------------------------------------------------------------ the IW refresh step

class ReproductionError(AssertionError):
    """Stop rule S3: an unweighted calibrate on the bank scores does not reproduce the stored standard threshold bit for
    bit (learners_iw.py:937-938 raises AssertionError there)."""


class IWRefreshed(NamedTuple):
    state: Any          # the BCAState after the IW step: thresholds = each training row's R(t_i) in float32
    record: dict        # JSON-ready readout (iw_refresh)
    table: Any          # the Table, or None when the standard refresh was not valid
    R_rows: Any         # float64 [n_train]: each training row's looked-up threshold, or None


def _sha256(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def iw_refresh(refreshed, bank_weights, train_masses, grid, *, permission, bank, scores=None, settings=SETTINGS):
    """The IW step after a standard refresh (plan section 4.2; the weighting test's per-row refresh,
    learners_iw.py:935-952, for this package's BCAState). refreshed: bca_core.Refreshed from the learner's refresh
    (the state already holds the new standard reference and every row at the standard threshold). In order:
      1. guard.require_bca(permission, bank): bank is the array set the refresh ran on (an entry point).
      2. An invalid standard refresh changes nothing: the previous state (with the previous per-row thresholds) is kept,
         as the weighting test keeps its previous table (line 935).
      3. Stop rule S3: an unweighted calibrate on the bank scores (by default the scores the refresh returned,
         bca_core.calibrate_bank; or `scores` recomputed by the caller with the refresh's own key) must reproduce the
         stored standard threshold bit for bit, else ReproductionError.
      4. The table R(t) with the bank weights on `grid`, with the same stream (threshold_table).
      5. Every training row's threshold, its own mass rounded up to the grid (row_thresholds), replaces
         state.thresholds; the reference (BCA's own, with the standard threshold) is kept, as in the weighting test.
    bank_weights: one weight per bank row (prepared bank/weight_strong); train_masses: one test mass per training row
    (train/weight_strong); grid: mass_grid on every mass that will be looked up (build it once per seed; on the
    training masses and, so the last refresh's table also serves the coverage readout, the test steps' masses and the
    weight's two ends w(0.1) and w(1.1)). The training rows' thresholds do not depend on how far the grid extends when
    the raw table R(t) never falls along the grid (the record's iw.table.monotone true); threshold_table's running
    maximum starts at the grid's lowest point, so if R(t) falls somewhere below the smallest training mass, the running
    maximum can raise the training rows' thresholds above what a grid of the training masses alone (the weighting
    test's construction, shift_iw/run.py:906-911) gives. The record's iw.table.running_max_applied says whether the
    running maximum changed any grid point. With every weight and mass 1 the grid is
    [1.0], R(1) is the standard threshold and the state is the standard state bit for bit (stop rule S2).
    Returns IWRefreshed(state, record, table, R_rows)."""
    guard.require_bca(permission, bank)
    state = refreshed.state
    if not bool(np.asarray(state.most_caution)):
        raise ValueError("iw_refresh serves the IW setups (BCAState.most_caution True); standard BCA keeps one threshold")
    n_train = int(np.shape(state.thresholds)[0])
    weights = np.asarray(bank_weights, np.float64)
    masses = np.asarray(train_masses, np.float64)
    scores = np.asarray(refreshed.scores if scores is None else scores, np.float64)
    if weights.ndim != 1 or weights.shape != scores.shape:
        raise ValueError("the IW table needs one weight per bank row")
    if masses.shape != (n_train,):
        raise ValueError("the IW table needs one test mass per training row")
    if not refreshed.record.get("valid"):
        return IWRefreshed(state, dict(applied=False, s3_reproduced=None, refused_rows=None, iw=None,
                                       note="the standard refresh was not valid; the previous per-row thresholds are "
                                            "kept"), None, None)
    unweighted = calibrate(scores, refreshed.stream, settings=settings)
    if not reproduces_calibration(state.reference, unweighted):
        raise ReproductionError("stop rule S3: the bank scores do not reproduce the stored standard threshold bit for "
                                f"bit (recomputed {unweighted.threshold!r}, stored "
                                f"{float(np.asarray(state.reference.threshold))!r})")
    table = threshold_table(scores, refreshed.stream, weights, grid, settings=settings)
    R, idx = lookup(table, masses)
    stored = table_float32(table)[idx]
    R_u = float(unweighted.threshold)
    gaps = lookup_gaps(table, masses, idx)
    new = state._replace(thresholds=jnp.asarray(stored))
    record = dict(
        applied=True, s3_reproduced=True, refused_rows=int(np.sum(np.isinf(R))),
        iw=dict(table=table.summary(R_u), grid_points=int(len(table.grid)), monotone=bool(table.monotone),
                running_max_applied=bool(table.running_max_applied), max_mass_gap=gaps["max_mass_gap"],
                max_threshold_gap=gaps["max_threshold_gap"],
                possibly_rounded_into_refusal=gaps["possibly_rounded_into_refusal"],
                rows=per_step_summary(R, R_u), unweighted_threshold=R_u if math.isfinite(R_u) else "inf",
                kish=kish(weights), bank_rows=int(scores.size), train_rows=n_train,
                weights_sha256=_sha256(weights), masses_sha256=_sha256(masses), grid_sha256=_sha256(table.grid),
                thresholds_sha256=_sha256(stored)))
    return IWRefreshed(new, record, table, R)


# ------------------------------------------------------------------------------------------------ last-refresh readouts

def placebo_permutation(rep, n):
    """The shuffled-weights placebo's permutation of the n bank rows for data seed rep (plan section 4.6):
    np.random.default_rng([20261016, rep, 95]).permutation(n). One permutation per seed, shared by every learner."""
    return np.random.default_rng([BC.MASTER_SEED, int(rep), TAG_PLACEBO]).permutation(int(n))


def last_refresh_arms(rep, exact, estimated, *, ratio=GRID_RATIO):
    """The readout arms at an IW setup's last refresh (plan section 4.6), for threshold_tables on that refresh's bank
    scores and stream. exact, estimated: (bank weights, masses), the masses being every test mass that will be looked up
    (the kept training rows and the strong-law test steps outside the ball; exact w(c) or the classifier's weights).
      exact       the exact bank weights on the grid of the exact masses (the IW setup's own table)
      estimated   the classifier's bank weights on the grid of the estimated masses (choice 20); rows and steps are
                  looked up at their estimated masses
      shuffled    the placebo: the exact bank weights permuted across bank rows (placebo_permutation), on the exact grid;
                  rows and steps are looked up at their exact masses, so only the weights' link to the bank rows'
                  currents is broken
    Returns ({name: (bank weights, grid)}, info)."""
    w_exact, m_exact = (np.asarray(v, np.float64) for v in exact)
    w_est, m_est = (np.asarray(v, np.float64) for v in estimated)
    if w_exact.ndim != 1 or w_est.shape != w_exact.shape:
        raise ValueError("the exact and estimated arms need one weight per bank row each")
    perm = placebo_permutation(rep, len(w_exact))
    grid_exact = mass_grid(m_exact, ratio)
    arms = dict(exact=(w_exact, grid_exact), estimated=(w_est, mass_grid(m_est, ratio)),
                shuffled=(w_exact[perm], grid_exact))
    info = dict(placebo=dict(stream=[BC.MASTER_SEED, int(rep), TAG_PLACEBO], rows=int(len(perm)),
                             moved_rows=int(np.sum(perm != np.arange(len(perm)))),
                             permutation_sha256=_sha256(perm.astype(np.int64))),
                kish=dict(exact=kish(w_exact), estimated=kish(w_est), shuffled=kish(w_exact[perm])),
                grid_points=dict(exact=int(len(grid_exact)), estimated=int(len(arms["estimated"][1]))))
    return arms, info
