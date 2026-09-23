"""Policy/affinity fitting weights with a separate bootstrap-product ESS."""

from dataclasses import dataclass
from numbers import Real
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np
import calibration.weights as IW

ACTION_EPS = 1e-06
ASSIGNMENT_FOLD = 1129531735


@dataclass(frozen=True)
class ScaleIWConfig:
    mode: str = "off"
    ess_floor: float = 0.25
    tau_min: float = 0.05
    iterations: int = 32

    def __post_init__(self):
        if self.mode not in ("off", "policy", "policy_permuted"):
            raise ValueError("posterior_scale_iw must be off/policy/policy_permuted")
        for name in ("ess_floor", "tau_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(name + " must be a finite real scalar")
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value32 = np.float32(value)
            if not np.isfinite(value32):
                raise ValueError(name + " must be finite in float32")
            if name == "ess_floor" and (not (0 < value32 and value <= 1)):
                raise ValueError("ess_floor must be in (0,1]")
            if name == "tau_min" and (not 0 <= value <= 1):
                raise ValueError("tau_min must be in [0,1]")
        if (
            isinstance(self.iterations, bool)
            or not isinstance(self.iterations, int)
            or self.iterations < 1
        ):
            raise ValueError("iterations must be a positive integer")


class PolicyLogWeights(NamedTuple):
    log_weights: jax.Array
    inputs_valid: jax.Array
    action_clip_fraction: jax.Array


class FitWeighting(NamedTuple):
    prior: jax.Array
    iw: IW.IWWeights
    product: IW.BootstrapIW
    permutation: jax.Array
    inputs_valid: jax.Array
    fit_feasible: jax.Array


def _detach(value):
    return jax.tree_util.tree_map(jax.lax.stop_gradient, value)


def policy_log_weights(actor_apply_fn, actor_params, obs, action):
    obs, action = (jnp.asarray(obs), jnp.asarray(action))
    if (
        obs.ndim != 2
        or action.ndim != 2
        or obs.shape[0] != action.shape[0]
        or (not obs.shape[0])
    ):
        raise ValueError("obs and action must be nonempty aligned matrices")
    valid = jnp.all(jnp.isfinite(obs)) & jnp.all(jnp.isfinite(action))
    for leaf in jax.tree_util.tree_leaves(actor_params):
        valid = valid & jnp.all(jnp.isfinite(leaf))
    clipped = jnp.clip(action, -1.0 + ACTION_EPS, 1.0 - ACTION_EPS)
    logp = actor_apply_fn(_detach(actor_params), _detach(obs)).log_prob(
        _detach(clipped)
    )
    if logp.shape != action.shape:
        raise ValueError(
            "CQL actor must return one log probability per action dimension"
        )
    logp = jnp.sum(logp, axis=-1)
    valid = valid & jnp.all(jnp.isfinite(logp))
    fraction = jnp.mean(jnp.any(clipped != action, axis=-1).astype(jnp.float32))
    return _detach(PolicyLogWeights(logp, valid, fraction))


def fit_weighting(log_weights, original_prior, permutation_key, config):
    if not isinstance(config, ScaleIWConfig) or config.mode == "off":
        raise ValueError("fit_weighting needs an enabled ScaleIWConfig")
    iw = IW.stabilize_log_weights(
        log_weights,
        ess_floor=config.ess_floor,
        tau_min=config.tau_min,
        iterations=config.iterations,
    )
    prior = jnp.asarray(original_prior)
    if prior.ndim != 1 or prior.shape != iw.log_weights.shape:
        raise ValueError("original_prior must be one aligned Bayesian simplex")
    permutation = jnp.arange(prior.size)
    if config.mode == "policy_permuted":
        permutation = jax.random.permutation(permutation_key, permutation)
        iw = iw._replace(
            log_weights=iw.log_weights[permutation],
            mean_one_weights=iw.mean_one_weights[permutation],
            probabilities=iw.probabilities[permutation],
            support_mask=iw.support_mask[permutation],
        )
    product = IW.tilt_bootstrap_weights(prior, iw)
    valid = iw.inputs_valid & product.inputs_valid
    return _detach(
        FitWeighting(prior, iw, product, permutation, valid, valid & iw.ess_feasible)
    )
