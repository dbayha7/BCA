"""One-sided BCA primitives; no data loading, fitting loop or calibration calls.

The retained scale objective is Bayesian-bootstrap-weighted soft coverage plus
width cost. Its event is y >= Q1 - unit*eta. This is not MSE error correction.
Default beta=20 and width_penalty=.005 match the pinned original source.
The caller owns data separation, frozen unit, reference acceptance and budgets.
"""

import math
from numbers import Real
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


ETA_OFFSET = .01
ETA_FLOOR = 1e-6
DEFAULT_BETA = 20.0
DEFAULT_WIDTH_PENALTY = .005


class CalibrationRefusal(ValueError):
    """A reference cannot be consumed as a finite actor penalty."""


class RadiusResult(NamedTuple):
    radius: object
    usable: object
    valid: object


def _coefficient(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(name + " must be a finite real scalar")
    if value < 0 or (positive and value <= 0):
        raise ValueError(name + " has invalid sign")


def _vector(value, name):
    result = np.asarray(value)
    if result.ndim != 1 or result.size == 0 or result.dtype.kind not in "fiu":
        raise ValueError(name + " must be a nonempty real vector; broadcasting is forbidden")
    result = result.astype(np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(name + " must be finite")
    return result


def _unit(value):
    result = np.asarray(value)
    if result.shape != () or result.dtype.kind not in "fiu":
        raise ValueError("unit must be a real scalar")
    result = float(result)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("unit must be finite and strictly positive")
    return result


def _jax_vectors(eta, q1, target, unit):
    eta = jnp.asarray(eta)
    if eta.ndim != 1 or eta.size == 0 or eta.dtype.kind != "f":
        raise ValueError("eta must be a nonempty floating vector")
    q1, target, unit = (jnp.asarray(value, dtype=eta.dtype) for value in (q1, target, unit))
    if q1.shape != eta.shape or target.shape != eta.shape or unit.shape != ():
        raise ValueError("q1/target must match eta exactly and unit must be scalar")
    return eta, jax.lax.stop_gradient(q1), jax.lax.stop_gradient(target), jax.lax.stop_gradient(unit)


def positive_eta(log_eta):
    """The original scale-network output transform, softplus(log_eta)+.01."""
    return jax.nn.softplus(jnp.asarray(log_eta)) + ETA_OFFSET


def soft_coverage(eta, q1, target, unit, beta=DEFAULT_BETA):
    """Lower-side coverage only; targets, Q1 and residual unit are detached.

    The original scale architecture guarantees eta>.01. The public width floor
    is therefore inactive during its fit. No upper-side sigmoid is retained.
    """
    _coefficient(beta, "beta", positive=True)
    eta, q1, target, unit = _jax_vectors(eta, q1, target, unit)
    return jax.nn.sigmoid(beta * (target / unit - q1 / unit + eta))


def scale_loss(eta, q1, target, unit, weights, alpha=.1, beta=DEFAULT_BETA,
               width_penalty=DEFAULT_WIDTH_PENALTY):
    """Return scalar coverage-plus-width loss; only eta receives gradients.

    Weights must already be normalized nonnegative Bayesian bootstrap masses.
    Eager invalid numeric inputs raise ValueError. Under jit, invalid numeric
    inputs return a nonfinite loss that MUST reject the optimizer update; static
    shape errors always raise. The runner must check finite loss and gradients.
    """
    _coefficient(alpha, "alpha", positive=True)
    if alpha >= 1:
        raise ValueError("alpha must be strictly below one")
    _coefficient(beta, "beta", positive=True)
    _coefficient(width_penalty, "width_penalty")
    eta, q1, target, unit = _jax_vectors(eta, q1, target, unit)
    weights = jax.lax.stop_gradient(jnp.asarray(weights, dtype=eta.dtype))
    if weights.shape != eta.shape:
        raise ValueError("weights must match eta exactly")
    valid = (
        jnp.all(jnp.isfinite(eta) & (eta > 0))
        & jnp.all(jnp.isfinite(q1)) & jnp.all(jnp.isfinite(target))
        & jnp.isfinite(unit) & (unit > 0)
        & jnp.all(jnp.isfinite(weights) & (weights >= 0))
        & jnp.isclose(weights.sum(), 1., rtol=1e-5, atol=1e-6)
    )
    if not isinstance(valid, jax.core.Tracer) and not bool(np.asarray(valid)):
        raise ValueError("Invalid scale-loss numeric domain")
    coverage = soft_coverage(eta, q1, target, unit, beta)
    loss = (jnp.sum(weights * coverage) - (1. - alpha)) ** 2
    loss = loss + width_penalty * jnp.sum(weights * eta ** 2)
    return jnp.where(valid, loss, jnp.asarray(jnp.nan, dtype=eta.dtype))


def directional_scores(q1, target, eta, unit):
    """NumPy nonconformity max(Q1-y,0)/(unit*max(eta,1e-6))."""
    q1, target, eta = (_vector(value, name) for value, name in ((q1, "q1"), (target, "target"), (eta, "eta")))
    if q1.shape != target.shape or q1.shape != eta.shape or np.any(eta <= 0):
        raise ValueError("q1/target/positive eta must have identical vector shapes")
    unit = _unit(unit)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scale = unit * np.maximum(eta, ETA_FLOOR)
        score = np.maximum(q1 - target, 0.) / scale
    if not np.all(np.isfinite(scale) & (scale > 0)) or not np.all(np.isfinite(score)):
        raise ValueError("Nonfinite directional score or residual scale")
    return score


def hard_coverage(q1, target, eta, unit):
    """Inclusive unit-radius event, including equality and zero excess."""
    return directional_scores(q1, target, eta, unit) <= 1.


def radius(eta, unit, threshold):
    """NumPy calibrated radius; zero is valid, infinite/invalid is a refusal."""
    eta = _vector(eta, "eta")
    unit = _unit(unit)
    raw_threshold = np.asarray(threshold)
    if raw_threshold.shape != () or raw_threshold.dtype.kind not in "fiu":
        raise CalibrationRefusal("threshold must be a real scalar")
    threshold = float(raw_threshold)
    if not math.isfinite(threshold) or threshold < 0 or np.any(eta <= 0):
        raise CalibrationRefusal("Only finite nonnegative thresholds and positive eta are usable")
    with np.errstate(over="ignore", invalid="ignore"):
        scale = unit * np.maximum(eta, ETA_FLOOR)
        result = threshold * scale
    if not np.all(np.isfinite(scale) & (scale > 0)) or not np.all(np.isfinite(result)):
        raise CalibrationRefusal("Calibrated radius overflow")
    return result


def frozen_scale(apply_fn, params, observations, actions):
    """Freeze reference parameter leaves while retaining live action gradients."""
    frozen_params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
    return apply_fn(frozen_params, observations, actions)


def frozen_radius(apply_fn, params, observations, actions, unit, threshold):
    """JIT-safe frozen reference radius with explicit refusal flags.

    Invalid/infinite references yield finite zero PLACEHOLDERS with both flags
    False. The caller MUST refuse any actor update unless both flags are True;
    placeholders are not authorization for an unpenalized fallback. Multiplying
    by an infinite threshold is never executed. Valid threshold zero is usable.
    """
    eta = jnp.asarray(frozen_scale(apply_fn, params, observations, actions))
    if eta.ndim != 1 or eta.size == 0 or eta.dtype.kind != "f":
        raise ValueError("Scale network must emit one floating scalar per row")
    unit = jax.lax.stop_gradient(jnp.asarray(unit, dtype=eta.dtype))
    threshold = jax.lax.stop_gradient(jnp.asarray(threshold, dtype=eta.dtype))
    if unit.shape != () or threshold.shape != ():
        raise ValueError("unit and threshold must be scalars")
    valid = (jnp.isfinite(unit) & (unit > 0) & jnp.isfinite(threshold) & (threshold >= 0)
             & jnp.all(jnp.isfinite(eta) & (eta > 0)))
    zeros = jnp.zeros_like(eta)

    def compute(_):
        scale = unit * jnp.maximum(eta, ETA_FLOOR)
        finite_scale = jnp.all(jnp.isfinite(scale) & (scale > 0))

        def multiply(_):
            result = threshold * scale
            finite = jnp.all(jnp.isfinite(result))
            return RadiusResult(jnp.where(finite, result, zeros), finite, finite)

        return jax.lax.cond(finite_scale, multiply, lambda _: RadiusResult(zeros, jnp.array(False), jnp.array(False)), operand=None)

    return jax.lax.cond(valid, compute, lambda _: RadiusResult(zeros, jnp.array(False), jnp.array(False)), operand=None)
