"""IQL positive scale networks and coverage/width objective."""

from dataclasses import dataclass
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from calibration.network import bayesian_bootstrap_weights, hard_coverage, soft_coverage


class ScalarCalibrator(nn.Module):

    @nn.compact
    def __call__(self, obs=None, action=None, residual=None):
        log_eta = self.param("log_eta", nn.initializers.zeros, ())
        eta = jax.nn.softplus(log_eta) + 0.01
        if obs is not None:
            return jnp.broadcast_to(eta, (obs.shape[0],))
        return eta


class StateDependentCalibrator(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs, action=None, residual=None):
        x = (obs - self.obs_mean) / (self.obs_std + 0.001)
        if action is not None:
            x = jnp.concatenate([x, action], axis=-1)
        if residual is not None:
            r = residual if residual.ndim == 2 else residual[..., None]
            x = jnp.concatenate([x, r], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        log_eta = nn.Dense(1)(x).squeeze(-1)
        return jax.nn.softplus(log_eta) + 0.01


def create_cal_train_state(rng, calibrator, dummy_inputs, lr):
    return TrainState.create(
        apply_fn=calibrator.apply,
        params=calibrator.init(rng, *dummy_inputs),
        tx=optax.adam(lr, eps=1e-05),
    )


def make_cal_loss_fn(
    cal_apply_fn,
    cal_mode,
    cal_beta,
    cal_alpha,
    cal_lambda_width,
    cal_cond_mode="scalar",
    balance_coverage=False,
    balance_weight=0.5,
    weight_width=False,
    cal_loss_mode="coverage",
):
    target_cov = 1.0 - cal_alpha
    assert cal_loss_mode in ("coverage", "pinball"), cal_loss_mode
    assert (
        cal_loss_mode == "coverage" or cal_mode == "soft"
    ), "cal_loss_mode='pinball' requires cal_mode='soft' (trained by gradient)"

    def cal_loss_fn(
        cal_params, targets, q_for_cov, dir_w, batch_obs, batch_action, residual=None
    ):
        if cal_cond_mode == "scalar":
            eta = cal_apply_fn(cal_params)
            eta = jnp.broadcast_to(eta, targets.shape)
        elif cal_cond_mode == "state":
            eta = cal_apply_fn(cal_params, batch_obs)
        elif cal_cond_mode == "residual":
            eta = cal_apply_fn(cal_params, batch_obs, batch_action, residual)
        else:
            eta = cal_apply_fn(cal_params, batch_obs, batch_action)
        if cal_mode == "soft":
            cov = soft_coverage(targets, q_for_cov, eta, cal_beta)
        else:
            cov = hard_coverage(targets, q_for_cov, eta)
        cov_mean = (cov * dir_w).sum()
        cov_var = ((cov - cov_mean) ** 2 * dir_w).sum()
        if cal_loss_mode == "pinball":
            _u = jnp.abs(targets - q_for_cov) - eta
            _pin = jnp.maximum(target_cov * _u, (target_cov - 1.0) * _u)
            _loss = (_pin * dir_w).sum() / (dir_w.sum() + 1e-12)
            return (
                _loss,
                (jnp.mean(eta), cov_mean, cov_var, jnp.mean(jnp.square(eta))),
            )
        coverage_loss = jnp.square(cov_mean - target_cov)
        if balance_coverage:
            coverage_loss = coverage_loss + balance_weight * cov_var
        if weight_width:
            width_loss = (jnp.square(eta) * dir_w).sum()
        else:
            width_loss = jnp.mean(jnp.square(eta))
        total = coverage_loss + cal_lambda_width * width_loss
        return (total, (jnp.mean(eta), cov_mean, cov_var, width_loss))

    return cal_loss_fn
