"""Block reservation and frozen scale/radius reference construction."""

import hashlib
from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
from calibration.posterior import (
    PosteriorConfig,
    PosteriorRadius,
    partitioned_posterior,
    apply_partitioned_radius,
)


class IQLPosteriorState(NamedTuple):
    cal_params: Any
    residual_scale: jax.Array
    group_edges: jax.Array
    radii: PosteriorRadius
    ready: jax.Array


def qlearning_episode_ids(raw):
    if "timeouts" not in raw:
        return None
    ends = np.asarray(raw["terminals"], bool) | np.asarray(raw["timeouts"], bool)
    ids = np.concatenate([[0], np.cumsum(ends[:-1], dtype=np.int64)])
    return ids[:-1][~np.asarray(raw["timeouts"][:-1], bool)]


def reserve_calibration(
    obs, next_obs, done, target_size, seed, max_fraction=0.25, episode_ids=None
):
    obs, next_obs, done = (np.asarray(obs), np.asarray(next_obs), np.asarray(done))
    n = len(obs)
    if obs.ndim != 2 or next_obs.shape != obs.shape or done.shape != (n,):
        raise ValueError("expected aligned obs/next_obs matrices and done vector")
    if not 0 < target_size < n or not 0 < max_fraction < 1:
        raise ValueError("calibration target must leave a nonempty training pool")
    rule = "terminal or exact next_obs/obs discontinuity"
    if episode_ids is not None:
        ids = np.asarray(episode_ids)
        if ids.shape != (n,) or np.any(ids[1:] < ids[:-1]):
            raise ValueError("episode IDs must be aligned and nondecreasing")
        ends = np.concatenate([ids[1:] != ids[:-1], [True]])
        rule = "raw terminal/timeout IDs mapped through default D4RL filtering"
    else:
        ends = (done != 0).copy()
        ends[:-1] |= np.any(next_obs[:-1] != obs[1:], axis=1)
    ends[-1] = True
    boundaries = np.concatenate([[0], np.flatnonzero(ends) + 1])
    lengths = np.diff(boundaries)
    order = np.random.default_rng(seed).permutation(len(lengths))
    count = int(np.searchsorted(np.cumsum(lengths[order]), target_size)) + 1
    chosen = order[:count]
    cal = np.sort(
        np.concatenate([np.arange(boundaries[i], boundaries[i + 1]) for i in chosen])
    )
    if len(cal) > max_fraction * n:
        raise ValueError("whole-block calibration exceeds the reserved fraction limit")
    train = np.flatnonzero(~np.isin(np.arange(n), cal))
    digest = hashlib.sha256(cal.astype("<i8").tobytes()).hexdigest()
    return (
        train.astype(np.int32),
        cal.astype(np.int32),
        {
            "boundary_rule": rule,
            "target_size": int(target_size),
            "calibration_size": len(cal),
            "training_size": len(train),
            "reserved_blocks": count,
            "total_blocks": len(lengths),
            "seed": int(seed),
            "calibration_indices_sha256": digest,
        },
    )


def initialize_posterior(cal_params, groups, draws):
    if (
        not isinstance(groups, int)
        or groups < 1
        or (not isinstance(draws, int))
        or (draws < 2)
    ):
        raise ValueError("groups must be positive and draws must be at least two")
    inf = jnp.full((groups,), jnp.inf)
    radii = PosteriorRadius(
        inf,
        inf,
        inf,
        jnp.full((groups, draws), jnp.inf),
        jnp.zeros(groups),
        jnp.zeros(groups, jnp.int32),
        jnp.zeros(groups, bool),
        jnp.zeros(groups, bool),
    )
    return IQLPosteriorState(
        cal_params, jnp.asarray(1.0), jnp.zeros(groups - 1), radii, jnp.asarray(False)
    )


def positive_scale(predictions, residual_scale):
    return jnp.maximum(predictions, 1e-06) * residual_scale


def fit_posterior(
    cal_params,
    residual_scale,
    fit_predictions,
    cal_predictions,
    residuals,
    rng,
    config=PosteriorConfig(),
    *,
    groups=4
):
    residuals, residual_scale = (jnp.asarray(residuals), jnp.asarray(residual_scale))
    if residual_scale.ndim != 0 or not isinstance(groups, int) or groups < 1:
        raise ValueError("residual_scale must be scalar and groups positive")
    fit_scale = positive_scale(jnp.asarray(fit_predictions), residual_scale)
    cal_scale = positive_scale(jnp.asarray(cal_predictions), residual_scale)
    if (
        fit_scale.ndim != 1
        or fit_scale.size == 0
        or residuals.ndim != 1
        or (residuals.size == 0)
        or (cal_scale.shape != residuals.shape)
    ):
        raise ValueError("predictions and residuals must be aligned vectors")
    edges = jnp.quantile(fit_scale, jnp.arange(1, groups) / groups)
    labels = jnp.searchsorted(edges, cal_scale, side="right")
    radii = partitioned_posterior(
        jnp.abs(residuals) / cal_scale, labels, rng, config, num_groups=groups
    )
    valid = jnp.all(jnp.isfinite(fit_scale) & (fit_scale > 0)) & jnp.all(
        jnp.isfinite(cal_scale) & (cal_scale > 0)
    )
    radii = radii._replace(
        radius=jnp.where(valid, radii.radius, jnp.inf),
        finite=radii.finite & valid,
        inputs_valid=radii.inputs_valid & valid,
    )
    return IQLPosteriorState(
        cal_params, residual_scale, edges, radii, jnp.asarray(True)
    )


def apply_posterior(state, predictions, mode="full"):
    if mode not in ("full", "floor"):
        raise ValueError("posterior mode must be full or floor")
    scale = positive_scale(predictions, state.residual_scale)
    groups = jnp.searchsorted(state.group_edges, scale, side="right")
    radii = state.radii
    if mode == "floor":
        radii = radii._replace(radius=radii.conformal_radius)
    return apply_partitioned_radius(radii, groups, scale)
