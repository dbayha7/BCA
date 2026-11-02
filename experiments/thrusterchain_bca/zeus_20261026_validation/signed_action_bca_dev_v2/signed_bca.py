"""Frozen-checkpoint signed Q1 TD-residual candidate.

The target is the pinned native one-step TD target, including target twin-min,
policy smoothing and bootstrap mask.  The candidate predicts e=Q1-y, not
the error of Q1 against Q^pi.  It never updates the native actor or critic.
Calibration is on separately supplied, unique episode IDs and only estimates
a residual threshold for that supplied distribution.
"""
from pathlib import Path
import math
import sys

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

_PREVIOUS = Path(__file__).resolve().parent.parent / 'action_sensitive_bca_dev_v1'
if str(_PREVIOUS) not in sys.path:
    sys.path.insert(0, str(_PREVIOUS))
import q1_bca


class SignedResidual(nn.Module):
    """Two 64-unit layers with signed-location and positive-scale heads."""
    obs_mean: object
    obs_std: object

    @nn.compact
    def __call__(self, obs, action):
        if obs.ndim != 2 or action.ndim != 2 or obs.shape[0] != action.shape[0]:
            raise ValueError('expected aligned observation/action matrices')
        x = jnp.concatenate(((obs - self.obs_mean) / (self.obs_std + .001), action), axis=-1)
        x = nn.relu(nn.Dense(64)(x))
        x = nn.relu(nn.Dense(64)(x))
        loc = nn.Dense(1, name='location')(x).squeeze(-1)
        raw_scale = nn.Dense(1, name='scale')(x).squeeze(-1)
        scale = jax.nn.softplus(raw_scale) + .01
        return loc, scale


def initialize(context, saved_agent, aux_seed, *, expected_mean, expected_std):
    """Fresh independent auxiliary after validating native normalizers."""
    original_models, state, config, args = q1_bca.fresh_auxiliary_exact_normalization(
        context, saved_agent, aux_seed, expected_mean=expected_mean, expected_std=expected_std)
    p = context.pipeline
    fixture = context.data.normalization_fixture
    model = SignedResidual(jnp.asarray(expected_mean, jnp.float32), jnp.asarray(expected_std, jnp.float32))
    obs = jnp.zeros((1, fixture.obs.shape[1]), dtype=jnp.asarray(fixture.obs).dtype)
    action = jnp.zeros((1, fixture.action.shape[1]), dtype=jnp.asarray(fixture.action).dtype)
    key = jax.random.fold_in(jax.random.PRNGKey(aux_seed), 1397311300)
    state = state._replace(calibrator=p.TrainState.create(apply_fn=model.apply,
        params=model.init(key, obs, action), tx=p.optax.adam(config.cal_lr)))
    return (original_models[0], original_models[1], model), state, config, args


def q1_and_target(pipeline, args, models, state, batch, noise_key):
    """Stop gradients through both Q1 and the native bootstrapped target."""
    return q1_bca.make_q1_pipeline(pipeline).q1_and_target(
        args, models, state.native, batch, noise_key)


def fit_step(pipeline, args, models, state, batch, rng):
    """One fresh signed Gaussian-NLL auxiliary update, with native state fixed."""
    q, target = q1_and_target(pipeline, args, models, state, batch,
                              jax.random.fold_in(rng, 1179210836))
    residual = jax.lax.stop_gradient(q - target)

    def objective(params):
        mu, sigma = models[2].apply(params, batch.obs, batch.action)
        if mu.shape != residual.shape or sigma.shape != residual.shape:
            raise ValueError('signed heads must have one scalar per batch row')
        loss = jnp.mean(.5 * jnp.square((residual - mu) / sigma) + jnp.log(sigma))
        return loss, (mu, sigma)

    (loss, (mu, sigma)), grad = jax.value_and_grad(objective, has_aux=True)(state.calibrator.params)
    proposed = state.calibrator.apply_gradients(grads=grad)
    valid = (pipeline.tree_finite((batch, state.native, state.calibrator, q, target,
                                   loss, grad, proposed, mu, sigma))
             & jnp.all(sigma >= .01))
    result = jax.lax.cond(valid, lambda _: state._replace(calibrator=proposed),
                          lambda _: state, operand=None)
    return result, {'signed_fit_accepted': valid, 'signed_loss': loss,
                    'residual_mean': jnp.mean(residual), 'sigma_mean': jnp.mean(sigma)}


def predict(models, state, obs, action, threshold):
    """Return clipped one-sided penalty and its components; action gradients live."""
    mu, sigma = models[2].apply(state.calibrator.params, obs, action)
    penalty = jnp.maximum(0., mu + jnp.asarray(threshold, sigma.dtype) * sigma)
    return penalty, mu, sigma


def refresh(pipeline, args, models, state, heldout, rng, *, heldout_ids,
            alpha=.1, beta=.95, draws=1000):
    """Calibrate signed normalized residual scores on unique held-out episodes.

    The caller guarantees episode-level independence from fitting and selection.
    Both Bayesian and split thresholds are returned, without claiming OOD or
    true-Q coverage.  A non-finite Bayesian threshold fails closed.
    """
    # The bridge deliberately loads its hash-verified sources in a private
    # namespace, rather than installing a global ``calibration`` package.
    wbcp = sys.modules[pipeline.freeze_reference.__module__].wbcp
    ids = np.asarray(heldout_ids)
    n = len(heldout.obs)
    if (n == 0 or ids.shape != (n,) or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or len(np.unique(ids)) != n):
        raise ValueError('held-out episode IDs must be unique nonnegative integers')
    if not (0 < alpha < 1 and 0 < beta < 1 and isinstance(draws, int) and draws > 0):
        raise ValueError('invalid calibration configuration')
    noise_key = jax.random.fold_in(rng, 1213156420)
    q, target = q1_and_target(pipeline, args, models, state, heldout, noise_key)
    mu, sigma = models[2].apply(state.calibrator.params, heldout.obs, heldout.action)
    q, target, mu, sigma = (np.asarray(x, np.float64) for x in (q, target, mu, sigma))
    if (q.shape != (n,) or target.shape != (n,) or mu.shape != (n,) or sigma.shape != (n,)
            or not all(np.isfinite(x).all() for x in (q, target, mu, sigma))
            or np.any(sigma < .01)):
        raise ValueError('invalid signed calibration values')
    residual = q - target
    scores = (residual - mu) / sigma
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite signed calibration scores')
    posterior_key = jax.random.fold_in(rng, 1347375956)
    result = wbcp.calibrate(scores, np.random.default_rng(np.asarray(posterior_key, np.uint32).ravel()),
                            alpha=alpha, beta=beta, draws=draws)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n or not result.certified or not np.isfinite(result.threshold):
        raise ValueError('signed calibration cannot certify a finite threshold')
    return {'q_bayes': float(result.threshold), 'q_split': float(np.sort(scores)[rank-1]),
            'lambda_hat': float(result.lambda_hat), 'lambda_hpd': float(result.lambda_hpd),
            'alpha': float(alpha), 'beta': float(beta), 'draws': draws, 'split_rank_1based': rank,
            'n': n, 'scores': scores.tolist(), 'q1': q.tolist(), 'target': target.tolist(),
            'mu': mu.tolist(), 'sigma': sigma.tolist(), 'residual': residual.tolist(),
            'noise_key': np.asarray(noise_key).tolist(), 'posterior_key': np.asarray(posterior_key).tolist()}
