"""Finite-rank conformal and Bayesian-bootstrap quantile radii."""

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from typing import NamedTuple
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PosteriorConfig:
    alpha: float = 0.1
    credibility: float = 0.95
    draws: int = 128

    def __post_init__(self):
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if not 0.0 < self.credibility < 1.0:
            raise ValueError("credibility must be strictly between zero and one")
        if self.draws < 2:
            raise ValueError("at least two Bayesian bootstrap draws are required")


class PosteriorRadius(NamedTuple):
    radius: jax.Array
    conformal_radius: jax.Array
    bayesian_radius: jax.Array
    posterior_quantiles: jax.Array
    effective_sample_size: jax.Array
    supported_count: jax.Array
    inputs_valid: jax.Array
    finite: jax.Array


class AppliedRadius(NamedTuple):
    width: jax.Array
    usable: jax.Array


def _rank(count, probability):
    return int((Decimal(count) * probability).to_integral_value(rounding=ROUND_CEILING))


def posterior_radius(
    scores,
    rng,
    config=PosteriorConfig(),
    *,
    weights=None,
    query_weight=None,
    bootstrap_weights=None
):
    scores = jnp.asarray(scores)
    scores = scores.astype(jnp.result_type(scores, jnp.float32))
    if scores.ndim != 1 or scores.shape[0] == 0:
        raise ValueError("scores must be a nonempty one-dimensional array")
    n = scores.shape[0]
    unweighted = weights is None
    if unweighted:
        if query_weight is not None:
            raise ValueError("query_weight requires explicit calibration weights")
        weights = jnp.ones_like(scores)
        query_weight = jnp.asarray(1.0, scores.dtype)
    else:
        if query_weight is None:
            raise ValueError("weighted calibration requires an explicit query_weight")
        weights = jnp.asarray(weights, dtype=scores.dtype)
        query_weight = jnp.asarray(query_weight, dtype=scores.dtype)
        if weights.shape != scores.shape or query_weight.ndim != 0:
            raise ValueError(
                "weights must match scores and query_weight must be scalar"
            )
    valid = (
        jnp.all(jnp.isfinite(scores) & (scores >= 0))
        & jnp.all(jnp.isfinite(weights) & (weights >= 0))
        & jnp.isfinite(query_weight)
        & (query_weight >= 0)
        & jnp.any(weights > 0)
    )
    safe_scores = jnp.where(jnp.isfinite(scores) & (scores >= 0), scores, 0.0)
    w = jnp.where(jnp.isfinite(weights) & (weights >= 0), weights, 0.0)
    qw = jnp.where(jnp.isfinite(query_weight) & (query_weight >= 0), query_weight, 0.0)
    original_w = w
    uniform = (qw > 0) & jnp.all((w == 0) | (w == qw))
    supported = w > 0
    scale = jnp.maximum(jnp.maximum(jnp.max(w), qw), jnp.finfo(scores.dtype).tiny)
    w, qw = (w / scale, qw / scale)
    calibration_w = jnp.where(jnp.isfinite(weights) & (weights >= 0), weights, 0.0)
    calibration_w = calibration_w / jnp.maximum(
        jnp.max(calibration_w), jnp.finfo(scores.dtype).tiny
    )
    tail_level = Decimal(1) - Decimal(str(config.alpha))
    order = jnp.argsort(safe_scores)
    sorted_scores, sorted_w = (safe_scores[order], w[order])
    if unweighted:
        rank = _rank(n + 1, tail_level)
        floor = jnp.concatenate([sorted_scores, jnp.array([jnp.inf], scores.dtype)])[
            min(rank - 1, n)
        ]
    else:
        cumulative = jnp.cumsum(sorted_w)
        cutoff = (1.0 - config.alpha) * (cumulative[-1] + qw)
        count = jnp.sum(supported)
        ranks = jnp.asarray([_rank(k + 1, tail_level) for k in range(n + 1)])
        integer_cutoff = ranks[count]
        integer_cdf = jnp.cumsum((sorted_w > 0).astype(jnp.int32))
        margin = 8 * (n + 1) * jnp.finfo(scores.dtype).eps * (cumulative[-1] + qw)
        index = jnp.where(
            uniform,
            jnp.sum(integer_cdf < integer_cutoff),
            jnp.sum(cumulative < cutoff + margin),
        )
        floor = jnp.concatenate([sorted_scores, jnp.array([jnp.inf], scores.dtype)])[
            index
        ]
    if bootstrap_weights is None:
        bootstrap_weights = jax.random.exponential(
            rng, (config.draws, n), dtype=scores.dtype
        )
    else:
        bootstrap_weights = jnp.asarray(bootstrap_weights, dtype=scores.dtype)
        if bootstrap_weights.shape != (config.draws, n):
            raise ValueError("bootstrap_weights must have shape (config.draws, n)")
    draw_valid = jnp.all(
        jnp.isfinite(bootstrap_weights) & (bootstrap_weights >= 0)
    ) & jnp.all(jnp.any((bootstrap_weights > 0) & supported[None, :], axis=1))
    valid = valid & draw_valid
    raw = jnp.where(
        jnp.isfinite(bootstrap_weights) & (bootstrap_weights >= 0),
        bootstrap_weights,
        0.0,
    )
    log_tilted = jnp.log(raw) + jnp.log(original_w)[None, :]
    maximum = jnp.max(log_tilted, axis=1, keepdims=True)
    valid = valid & jnp.all(jnp.isfinite(maximum))
    tilted = jnp.exp(log_tilted - jnp.where(jnp.isfinite(maximum), maximum, 0.0))[
        :, order
    ]
    cumulative = jnp.cumsum(tilted, axis=1)
    cutoff = (1.0 - config.alpha) * cumulative[:, -1]
    indices = jnp.sum(cumulative < cutoff[:, None], axis=1)
    qs = sorted_scores[jnp.minimum(indices, n - 1)]
    posterior_rank = _rank(config.draws, Decimal(str(config.credibility))) - 1
    bayes = jnp.sort(qs)[posterior_rank]
    floor = jnp.where(valid, floor, jnp.inf)
    bayes = jnp.where(valid, bayes, jnp.inf)
    radius = jnp.maximum(floor, bayes)
    ess = jnp.where(
        valid,
        jnp.sum(calibration_w) ** 2
        / jnp.maximum(jnp.sum(calibration_w**2), jnp.finfo(scores.dtype).tiny),
        0.0,
    )
    return PosteriorRadius(
        radius,
        floor,
        bayes,
        jnp.where(valid, qs, jnp.inf),
        ess,
        jnp.sum(supported),
        valid,
        valid & jnp.isfinite(radius),
    )


def partitioned_posterior(
    scores, group_ids, rng, config=PosteriorConfig(), *, num_groups
):
    scores, group_ids = (jnp.asarray(scores), jnp.asarray(group_ids))
    if not isinstance(num_groups, int) or num_groups < 1:
        raise ValueError("num_groups must be a positive static integer")
    if group_ids.shape != scores.shape or not jnp.issubdtype(
        group_ids.dtype, jnp.integer
    ):
        raise ValueError("integer group_ids must match the score vector")
    labels_valid = jnp.all((group_ids >= 0) & (group_ids < num_groups))

    def one(g):
        result = posterior_radius(
            scores,
            jax.random.fold_in(rng, g),
            config,
            weights=(group_ids == g).astype(jnp.float32),
            query_weight=1.0,
        )
        return result._replace(
            radius=jnp.where(labels_valid, result.radius, jnp.inf),
            inputs_valid=result.inputs_valid & labels_valid,
            finite=result.finite & labels_valid,
        )

    return jax.vmap(one)(jnp.arange(num_groups))


def apply_partitioned_radius(result: PosteriorRadius, query_groups, positive_scale):
    query_groups = jnp.asarray(query_groups)
    if not jnp.issubdtype(query_groups.dtype, jnp.integer):
        raise ValueError("query_groups must be integer labels")
    valid = (query_groups >= 0) & (query_groups < result.radius.shape[0])
    selected = jax.tree.map(
        lambda x: x[jnp.clip(query_groups, 0, result.radius.shape[0] - 1)], result
    )
    selected = selected._replace(finite=selected.finite & valid)
    return apply_radius(selected, positive_scale)


def apply_radius(result: PosteriorRadius, positive_scale):
    scale = jnp.asarray(positive_scale)
    if (
        scale.ndim > 0
        and result.radius.ndim > 0
        and (scale.shape != result.radius.shape)
    ):
        raise ValueError(
            "nonscalar scale and selected radius must have matching shapes"
        )
    valid_scale = jnp.isfinite(scale) & (scale > 0)
    computed = result.radius * jnp.where(valid_scale, scale, 1.0)
    underflow = (result.radius > 0) & (computed == 0)
    usable = result.finite & valid_scale & jnp.isfinite(computed) & ~underflow
    width = jnp.where(usable, computed, jnp.inf)
    return AppliedRadius(jax.lax.stop_gradient(width), usable)
