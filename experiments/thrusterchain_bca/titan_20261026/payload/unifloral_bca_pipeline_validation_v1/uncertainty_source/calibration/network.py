"""Positive residual-scale network, coverage functions and Bayesian simplex draws."""

import flax.linen as nn
import jax
import jax.numpy as jnp


class Calibrator(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    state_dep: bool = False
    hidden: int = 64
    depth: int = 2

    @nn.compact
    def __call__(self, obs=None, action=None):
        if self.state_dep and obs is not None:
            obs_n = (obs - self.obs_mean) / (self.obs_std + 0.001)
            x = (
                jnp.concatenate([obs_n, action], axis=-1)
                if action is not None
                else obs_n
            )
            for _ in range(self.depth):
                x = nn.Dense(self.hidden)(x)
                x = nn.relu(x)
            log_eta = nn.Dense(1)(x).squeeze(-1)
        else:
            log_eta = self.param("log_eta", nn.initializers.zeros, ())
        eta = jax.nn.softplus(log_eta) + 0.01
        return eta


def soft_coverage(y, q, eta, beta=20.0):
    lower = jax.nn.sigmoid(beta * (y - (q - eta)))
    upper = jax.nn.sigmoid(beta * (q + eta - y))
    return lower * upper


def hard_coverage(y, q, eta):
    return ((y >= q - eta) & (y <= q + eta)).astype(jnp.float32)


def bayesian_bootstrap_weights(rng, batch_size):
    w_raw = jax.random.exponential(rng, (batch_size,))
    return w_raw / w_raw.sum()
