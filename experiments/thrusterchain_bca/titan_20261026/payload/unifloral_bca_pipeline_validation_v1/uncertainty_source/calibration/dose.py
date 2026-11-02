"""Detached frozen-width multipliers for actor BC or the CQL critic gap."""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class CQLLevelDose(NamedTuple):
    width: jax.Array
    dose: jax.Array
    support_mask: jax.Array
    inputs_valid: jax.Array
    has_support: jax.Array


def tree_finite(tree):
    valid = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        valid = valid & jnp.all(jnp.isfinite(leaf))
    return valid


def level_critic_dose(widths, usable, residual_scale, blend, inputs_valid=True):
    widths, usable = (jnp.asarray(widths), jnp.asarray(usable))
    unit, blend, upstream = map(jnp.asarray, (residual_scale, blend, inputs_valid))
    if widths.ndim != 1 or usable.shape != widths.shape:
        raise ValueError("widths and usable must be aligned vectors")
    if unit.ndim or blend.ndim or upstream.ndim:
        raise ValueError("residual_scale, blend and inputs_valid must be scalars")
    if usable.dtype != jnp.bool_ or upstream.dtype != jnp.bool_:
        raise TypeError("usable and inputs_valid must be boolean")
    for value in (widths, unit, blend):
        if not (
            jnp.issubdtype(value.dtype, jnp.floating)
            or jnp.issubdtype(value.dtype, jnp.integer)
        ):
            raise TypeError(
                "widths, residual_scale and blend must be real numeric values"
            )
    dtype = jnp.result_type(widths, unit, blend, jnp.float32)
    widths, unit, blend = (
        jax.lax.stop_gradient(v.astype(dtype)) for v in (widths, unit, blend)
    )
    valid_unit = jnp.isfinite(unit) & (unit > 0)
    valid_blend = jnp.isfinite(blend) & (blend >= 0) & (blend <= 1)
    valid = (
        upstream
        & jnp.all(~jnp.isnan(widths) & (widths >= 0))
        & valid_unit
        & valid_blend
    )
    support = valid & usable & jnp.isfinite(widths)
    safe_u, safe_c = (jnp.where(support, widths, 0.0), jnp.where(valid_unit, unit, 1.0))
    normalizer = jnp.maximum(safe_u, safe_c)
    rescale = jnp.where(normalizer > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
    denominator_scale = normalizer * rescale
    u_norm = safe_u * rescale / denominator_scale
    c_norm = safe_c * rescale / denominator_scale
    ratio = u_norm / (u_norm + c_norm)
    dose = 1.0 + jnp.where(valid_blend, blend, 0.0) * ratio
    result = CQLLevelDose(widths, dose, support, valid, jnp.any(support))
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


def frozen_level_dose(reference, predictions, blend):
    """Width = WBCP threshold x frozen scale at the recorded rows; not ready means width 0.

    A threshold of +inf (nothing certifiable) gives infinite widths, which
    level_critic_dose treats as unsupported rows with the native multiplier 1.
    """
    predictions = jnp.asarray(predictions)
    if predictions.ndim != 1 or reference.threshold.ndim != 0 or reference.residual_scale.ndim != 0:
        raise ValueError("level dose requires vector predictions, one threshold and a scalar unit")
    threshold, unit = reference.threshold, reference.residual_scale
    scale = jnp.maximum(predictions, 1e-06) * unit
    width = threshold * scale
    valid = (
        tree_finite(reference.cal_params)
        & jnp.all(jnp.isfinite(predictions) & (predictions > 0))
        & jnp.isfinite(unit)
        & (unit > 0)
        & jnp.all(jnp.isfinite(scale) & (scale > 0))
        & ~jnp.isnan(threshold)
        & (threshold >= 0)
        & (
            jnp.isposinf(threshold)
            | jnp.all(jnp.isfinite(width) & ((threshold == 0) | (width > 0)))
        )
    )
    width = jnp.where(reference.ready, width, jnp.zeros_like(width))
    valid = jnp.where(reference.ready, valid, True)
    unit = jnp.where(reference.ready, unit, 1.0)
    return level_critic_dose(
        width, jnp.ones_like(width, dtype=bool), unit, blend, valid
    )
