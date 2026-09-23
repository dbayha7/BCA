"""BCA bca level.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class LevelActorWeights(NamedTuple):
    weights: jax.Array
    used_advantage: jax.Array
    support_mask: jax.Array
    inputs_valid: jax.Array
    has_support: jax.Array


def level_actor_weights(advantages, widths, usable, beta, max_weight, inputs_valid=True):
    return _actor_weights(
        advantages, widths, usable, beta, max_weight, inputs_valid, smooth=False
    )


def smooth_level_actor_weights(advantages, widths, usable, beta, max_weight, inputs_valid=True):
    return _actor_weights(
        advantages, widths, usable, beta, max_weight, inputs_valid, smooth=True
    )


def postcap_level_actor_weights(
    advantages, widths, usable, beta, max_weight, inputs_valid=True
):
    base = level_actor_weights(
        advantages, widths, usable, beta, max_weight, inputs_valid=inputs_valid
    )
    dtype = base.weights.dtype
    advantages, widths, beta, max_weight = (
        jax.lax.stop_gradient(jnp.asarray(value, dtype=dtype))
        for value in (advantages, widths, beta, max_weight)
    )
    valid = base.inputs_valid & (max_weight >= 1.0)
    support = base.support_mask & valid
    safe_adv = jnp.where(support, advantages, 0.0)
    safe_width = jnp.where(support, widths, 0.0)
    safe_beta = jnp.where(valid, beta, 0.0)
    safe_cap = jnp.where(valid, max_weight, 1.0)
    positive = jnp.maximum(safe_adv, 0.0)
    scale = jnp.maximum(positive, safe_width)
    safe_scale = jnp.where(scale > 0.0, scale, 1.0)
    rescale = jnp.where(safe_scale > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
    denominator_scale = safe_scale * rescale
    a = positive * rescale / denominator_scale
    b = safe_width * rescale / denominator_scale
    fraction = jnp.clip(a / jnp.where(a + b > 0.0, a + b, 1.0), 0.0, 1.0)
    native = jnp.minimum(
        jnp.exp(jnp.minimum(safe_beta * safe_adv, jnp.log(safe_cap))), safe_cap
    )
    bonus = fraction * (native - 1.0)
    positive_weight = jnp.minimum(jnp.maximum(1.0 + bonus, 1.0), native)
    bounded = jnp.where(safe_adv > 0.0, positive_weight, native)
    weights = jnp.where(support, bounded, 0.0)
    effective = jnp.log1p(jnp.maximum(bounded - 1.0, 0.0)) / jnp.where(
        safe_beta > 0.0, safe_beta, 1.0
    )
    effective = jnp.minimum(positive, jnp.maximum(effective, 0.0))
    used = jnp.where(safe_adv > 0.0, effective, safe_adv)
    result = LevelActorWeights(weights, used, support, valid, jnp.any(support))
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)


def _actor_weights(advantages, widths, usable, beta, max_weight, inputs_valid, *, smooth):
    advantages, widths = (jnp.asarray(advantages), jnp.asarray(widths))
    usable, upstream_valid = (jnp.asarray(usable), jnp.asarray(inputs_valid))
    beta, max_weight = (jnp.asarray(beta), jnp.asarray(max_weight))
    if (
        advantages.ndim != 1
        or widths.shape != advantages.shape
        or usable.shape != advantages.shape
    ):
        raise ValueError("advantages, widths and usable must be aligned vectors")
    if beta.ndim != 0 or max_weight.ndim != 0 or upstream_valid.ndim != 0:
        raise ValueError("beta, max_weight and inputs_valid must be scalars")
    if usable.dtype != jnp.bool_ or upstream_valid.dtype != jnp.bool_:
        raise TypeError("usable and inputs_valid must have boolean dtype")
    for value in (advantages, widths, beta, max_weight):
        if not (
            jnp.issubdtype(value.dtype, jnp.floating)
            or jnp.issubdtype(value.dtype, jnp.integer)
        ):
            raise TypeError(
                "advantages, widths, beta and max_weight must be real numeric values"
            )
    dtype = jnp.result_type(advantages, widths, beta, max_weight, jnp.float32)
    advantages, widths, beta, max_weight = (
        jax.lax.stop_gradient(value.astype(dtype))
        for value in (advantages, widths, beta, max_weight)
    )
    finite_adv = jnp.isfinite(advantages)
    valid_width = ~jnp.isnan(widths) & (widths >= 0)
    valid_beta = jnp.isfinite(beta) & (beta >= 0)
    valid_cap = jnp.isfinite(max_weight) & (max_weight > 0)
    valid = upstream_valid & jnp.all(finite_adv) & jnp.all(valid_width) & valid_beta & valid_cap
    support = valid & usable & jnp.isfinite(widths)
    safe_adv = jnp.where(support, advantages, 0.0)
    safe_width = jnp.where(support, widths, 0.0)
    if smooth:
        positive = jnp.maximum(safe_adv, 0.0)
        scale = jnp.maximum(positive, safe_width)
        safe_scale = jnp.where(scale > 0.0, scale, 1.0)
        rescale = jnp.where(safe_scale > jnp.finfo(dtype).max / 4.0, 0.25, 1.0)
        denominator_scale = safe_scale * rescale
        a = positive * rescale / denominator_scale
        b = safe_width * rescale / denominator_scale
        denominator = a + b
        fraction = a / jnp.where(denominator > 0.0, denominator, 1.0)
        used = jnp.minimum(safe_adv, 0.0) + positive * fraction
    else:
        used = jnp.minimum(safe_adv, 0.0) + jnp.maximum(
            jnp.maximum(safe_adv, 0.0) - safe_width, 0.0
        )
    safe_beta = jnp.where(valid_beta, beta, 1.0)
    safe_cap = jnp.where(valid_cap, max_weight, 1.0)
    exponent = safe_beta * used
    bounded = jnp.minimum(jnp.exp(jnp.minimum(exponent, jnp.log(safe_cap))), safe_cap)
    weights = jnp.where(support, bounded, 0.0)
    result = LevelActorWeights(weights, used, support, valid, jnp.any(support))
    return jax.tree_util.tree_map(jax.lax.stop_gradient, result)
