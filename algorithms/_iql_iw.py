"""BCA iql iw.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class IWLogWeights(NamedTuple):
    log_weights: jax.Array
    support_mask: jax.Array
    inputs_valid: jax.Array


class IWWeights(NamedTuple):
    log_weights: jax.Array
    mean_one_weights: jax.Array
    probabilities: jax.Array
    tau: jax.Array
    raw_ess_fraction: jax.Array
    ess_fraction: jax.Array
    ess_feasible: jax.Array
    support_mask: jax.Array
    supported_count: jax.Array
    positive_count: jax.Array
    inputs_valid: jax.Array


class BootstrapIW(NamedTuple):
    weights: jax.Array
    prior_ess_fraction: jax.Array
    iw_ess_fraction: jax.Array
    product_ess_fraction: jax.Array
    row_valid: jax.Array
    inputs_valid: jax.Array


def _detached(tree):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, tree)


def _vector(value, name):
    value = jnp.asarray(value)
    value = value.astype(jnp.result_type(value, jnp.float32))
    if value.ndim != 1 or value.size == 0:
        raise ValueError(name + " must be a nonempty one-dimensional array")
    return value


def _scalar(value, dtype, name):
    value = jnp.asarray(value, dtype=dtype)
    if value.ndim != 0:
        raise ValueError(name + " must be scalar")
    return value


def capped_awr_log_weights(advantage, beta, cap, mixing=1.0):
    a = _vector(advantage, "advantage")
    beta = _scalar(beta, a.dtype, "beta")
    cap = _scalar(cap, a.dtype, "cap")
    mixing = _scalar(mixing, a.dtype, "mixing")
    config_valid = (
        jnp.isfinite(beta)
        & (beta > 0)
        & jnp.isfinite(cap)
        & (cap >= 1)
        & jnp.isfinite(mixing)
        & (mixing >= 0)
        & (mixing <= 1)
    )
    b = jnp.where(jnp.isfinite(beta) & (beta > 0), beta, 1.0)
    c = jnp.where(jnp.isfinite(cap) & (cap >= 1), cap, 1.0)
    mix = jnp.where(jnp.isfinite(mixing) & (mixing >= 0) & (mixing <= 1), mixing, 0.0)
    safe_a = jnp.where(jnp.isfinite(a), a, 0.0)
    log_cap = jnp.log(c)
    scaled = b * safe_a
    raw_log = jnp.minimum(scaled, log_cap)
    raw_valid = jnp.all(jnp.isfinite(raw_log) | (mix < 1) & jnp.isneginf(raw_log))
    safe_log = jnp.where(jnp.isfinite(raw_log) | jnp.isneginf(raw_log), raw_log, 0.0)
    blended = jnp.logaddexp(jnp.log1p(-mix), jnp.log(mix) + safe_log)
    result = jnp.where(mix == 0, jnp.zeros_like(a), jnp.where(mix == 1, safe_log, blended))
    valid = (
        config_valid
        & jnp.all(jnp.isfinite(a))
        & ((mix == 0) | raw_valid)
        & jnp.all(jnp.isfinite(result))
    )
    return _detached(
        IWLogWeights(
            jnp.where(valid, result, -jnp.inf), jnp.full(a.shape, valid, dtype=bool), valid
        )
    )


def stabilize_log_weights(
    log_weights, *, support=None, ess_floor=0.25, tau_min=0.05, iterations=32
):
    lw = _vector(log_weights, "log_weights")
    if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 1:
        raise ValueError("iterations must be a positive static integer")
    if support is None:
        mask = jnp.ones(lw.shape, bool)
    else:
        mask = jnp.asarray(support)
        if mask.shape != lw.shape or mask.dtype != jnp.bool_:
            raise ValueError("support must be a boolean mask matching log_weights")
    floor = _scalar(ess_floor, lw.dtype, "ess_floor")
    minimum = _scalar(tau_min, lw.dtype, "tau_min")
    config_valid = (
        jnp.isfinite(floor)
        & (floor > 0)
        & (floor <= 1)
        & jnp.isfinite(minimum)
        & (minimum >= 0)
        & (minimum <= 1)
    )
    safe_min = jnp.where(jnp.isfinite(minimum) & (minimum >= 0) & (minimum <= 1), minimum, 0.0)
    supported = mask & jnp.isfinite(lw)
    valid = config_valid & jnp.all(jnp.isfinite(lw) | jnp.isneginf(lw)) & jnp.any(supported)
    maximum = jnp.max(jnp.where(supported, lw, -jnp.inf))
    maximum = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    centered = jnp.where(supported, lw - maximum, 0.0)
    valid = valid & jnp.all(jnp.isfinite(centered))
    centered = jnp.where(jnp.isfinite(centered), centered, 0.0)
    n = lw.size

    def distribution(tau):
        masses = jnp.where(supported, jnp.exp(tau * centered), 0.0)
        total = jnp.sum(masses)
        probs = masses / jnp.where(total > 0, total, 1.0)
        square_sum = jnp.sum(probs**2)
        ef = jnp.where(total > 0, 1.0 / (n * jnp.where(square_sum > 0, square_sum, 1.0)), 0.0)
        uniform_supported = jnp.all(~supported | (tau * centered == 0))
        ef = jnp.where(uniform_supported, jnp.sum(supported) / n, ef)
        return (probs, ef)

    _, raw_ess = distribution(jnp.asarray(1.0, lw.dtype))
    _, minimum_ess = distribution(safe_min)
    raw_ok = raw_ess >= floor
    feasible = valid & ((minimum_ess >= floor) | raw_ok)

    def body(_, endpoints):
        lo, hi = endpoints
        mid = lo + (hi - lo) * 0.5
        _, achieved = distribution(mid)
        good = achieved >= floor
        return (jnp.where(good, mid, lo), jnp.where(good, hi, mid))

    lo, _ = jax.lax.fori_loop(0, iterations, body, (safe_min, jnp.asarray(1.0, lw.dtype)))
    tau = jnp.where(raw_ok, 1.0, jnp.where(feasible, lo, safe_min))
    probabilities, achieved = distribution(tau)
    uniform = jnp.all(supported) & jnp.all(centered == 0)
    mean_one = jnp.where(uniform, jnp.ones_like(lw), probabilities * n)
    final_log = jnp.where(supported, tau * centered, -jnp.inf)
    return _detached(
        IWWeights(
            jnp.where(valid, final_log, -jnp.inf),
            jnp.where(valid, mean_one, 0.0),
            jnp.where(valid, probabilities, 0.0),
            jnp.where(valid, tau, 0.0),
            jnp.where(valid, raw_ess, 0.0),
            jnp.where(valid, achieved, 0.0),
            feasible & (achieved >= floor),
            supported & valid,
            jnp.where(valid, jnp.sum(supported), 0),
            jnp.where(valid, jnp.sum(mean_one > 0), 0),
            valid,
        )
    )


def tilt_bootstrap_weights(prior, iw):
    prior = jnp.asarray(prior)
    prior = prior.astype(jnp.result_type(prior, iw.log_weights, jnp.float32))
    if prior.ndim not in (1, 2) or prior.shape[-1] != iw.log_weights.size:
        raise ValueError("prior must have shape (n,) or (draws,n) matching IW")
    if prior.size == 0:
        raise ValueError("prior must contain at least one draw")
    n = prior.shape[-1]
    sums = jnp.sum(prior, axis=-1, keepdims=True)
    row_valid = jnp.all(jnp.isfinite(prior) & (prior >= 0), axis=-1) & (
        jnp.abs(sums[..., 0] - 1.0) <= 32 * jnp.finfo(prior.dtype).eps
    )
    safe_prior = jnp.where(jnp.isfinite(prior) & (prior >= 0), prior, 0.0)
    prior_support = safe_prior > 0
    product_support = prior_support & iw.support_mask
    combined = jnp.log(jnp.where(prior_support, safe_prior, 1.0)) + jnp.where(
        iw.support_mask, iw.log_weights, 0.0
    )
    row_valid = (
        row_valid
        & iw.inputs_valid
        & jnp.any(product_support, axis=-1)
        & jnp.all(~product_support | jnp.isfinite(combined), axis=-1)
    )
    log_product = jnp.where(product_support, combined, -jnp.inf)
    maximum = jnp.max(log_product, axis=-1, keepdims=True)
    safe_max = jnp.where(jnp.isfinite(maximum), maximum, 0.0)
    masses = jnp.where(product_support, jnp.exp(log_product - safe_max), 0.0)
    totals = jnp.sum(masses, axis=-1, keepdims=True)
    product = masses / jnp.where(totals > 0, totals, 1.0)
    uniform = jnp.all(iw.support_mask) & jnp.all(iw.log_weights == iw.log_weights[0])
    result = jnp.where(uniform, prior, product)
    valid = jnp.all(row_valid)
    normalized_prior = safe_prior / jnp.where(sums > 0, sums, 1.0)
    prior_squares = jnp.sum(normalized_prior**2, axis=-1)
    product_squares = jnp.sum(product**2, axis=-1)
    prior_ess = 1.0 / (n * jnp.where(prior_squares > 0, prior_squares, 1.0))
    product_ess = 1.0 / (n * jnp.where(product_squares > 0, product_squares, 1.0))
    product_ess = jnp.where(uniform, prior_ess, product_ess)
    return _detached(
        BootstrapIW(
            jnp.where(valid, result, 0.0),
            jnp.where(row_valid, prior_ess, 0.0),
            iw.ess_fraction,
            jnp.where(row_valid, product_ess, 0.0),
            row_valid,
            valid,
        )
    )
