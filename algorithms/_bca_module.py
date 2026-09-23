"""BCA bca module.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from dataclasses import dataclass
import flax.linen as nn
from flax.linen.initializers import constant, uniform
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri


@dataclass
class BCAArgs:
    num_heads: int = 5
    prior_scale: float = 0.0
    use_head_bootstrap: bool = False
    head_bootstrap_p: float = 0.5
    use_cal: bool = True
    cal_mode: str = "soft"
    cal_lr: float = 0.001
    cal_beta: float = 20.0
    cal_alpha: float = 0.1
    cal_lambda_width: float = 0.005
    cal_warmup: int = 0
    cal_state_dep: bool = False
    cal_eta_max: float = 10.0
    cal_min_alpha_ratio: float = 0.0
    use_trigger: bool = False
    cal_trig_signal: str = "resid"
    cal_trig_kappa: float = 4.0
    cal_trig_tau: float = 0.0
    cal_loss_mode: str = "coverage"
    use_bootstrap: bool = True
    use_reliability_weighting: bool = True
    head_weight_warmup: int = 50000
    min_head_weight: float = 0.04


def sym(scale):

    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool
    final_init_scale: float = 0.003
    arch: str = "bca"
    width: int = 256
    depth: int = 3

    @nn.compact
    def __call__(self, obs, action):
        if self.norm_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 0.001)
        x = jnp.concatenate([obs, action], axis=-1)
        if self.arch in ("iql", "td3"):
            for _ in range(2):
                x = nn.Dense(256)(x)
                x = nn.relu(x)
            q = nn.Dense(1)(x)
            return q.squeeze(-1)
        for _ in range(self.depth):
            x = nn.Dense(self.width, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.use_ln else x
        s = self.final_init_scale
        q = nn.Dense(1, bias_init=sym(s), kernel_init=sym(s))(x)
        return q.squeeze(-1)


class MultiHeadQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool
    num_heads: int = 5
    prior_scale: float = 0.0
    prior_final_scale: float = 0.1
    arch: str = "bca"
    width: int = 256
    depth: int = 3

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_heads,
        )
        q = vmap_critic(
            self.obs_mean,
            self.obs_std,
            self.use_ln,
            self.norm_obs,
            arch=self.arch,
            width=self.width,
            depth=self.depth,
        )(obs, action)
        if self.prior_scale > 0.0:
            prior_critic = nn.vmap(
                SoftQNetwork,
                variable_axes={"params": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=None,
                out_axes=-1,
                axis_size=self.num_heads,
            )
            prior = prior_critic(
                self.obs_mean,
                self.obs_std,
                self.use_ln,
                self.norm_obs,
                self.prior_final_scale,
                width=self.width,
                depth=self.depth,
            )(obs, action)
            q = q + self.prior_scale * jax.lax.stop_gradient(prior)
        return q


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
            x = jnp.concatenate([obs_n, action], axis=-1) if action is not None else obs_n
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


def reliability_weights(per_head_err, min_w):
    centered = per_head_err - per_head_err.mean()
    scale = jnp.maximum(per_head_err.std(), 1e-06)
    log_w = -(centered / scale)
    log_w = log_w - log_w.max()
    w = jnp.exp(log_w)
    w = w / w.sum()
    w = jnp.maximum(w, min_w)
    w = w / w.sum()
    return w


def aggregate_q(q_heads, head_weights, use_weighted):
    q_min = q_heads.min(-1)
    q_weighted = (q_heads * head_weights).sum(-1)
    return jnp.where(use_weighted, q_weighted, q_min)


def bayesian_bootstrap_weights(rng, batch_size):
    w_raw = jax.random.exponential(rng, (batch_size,))
    return w_raw / w_raw.sum()


def make_cal_loss_fn(
    cal_apply_fn, cal_mode, cal_beta, cal_alpha, cal_lambda_width, cal_state_dep
):

    def cal_loss_fn(cal_params, targets, q_for_cov, dir_w, batch_obs=None, batch_action=None):
        if cal_state_dep:
            eta = cal_apply_fn(cal_params, batch_obs, batch_action)
        else:
            eta = cal_apply_fn(cal_params)
        if cal_mode == "soft":
            cov = soft_coverage(targets, q_for_cov, eta, cal_beta)
        else:
            cov = hard_coverage(targets, q_for_cov, eta)
        target_cov = 1.0 - cal_alpha
        cov_weighted_mean = (cov * dir_w).sum()
        coverage_loss = jnp.square(cov_weighted_mean - target_cov)
        if cal_state_dep:
            width_loss = jnp.square(eta).mean()
        else:
            width_loss = jnp.square(eta)
        total = coverage_loss + cal_lambda_width * width_loss
        eta_scalar = eta if not cal_state_dep else eta.mean()
        return (total, (eta_scalar, cov_weighted_mean))

    return cal_loss_fn


def rank_gauss(x, a):
    _rank = jnp.argsort(jnp.argsort(x)) + 1
    _pp = (_rank - a) / (x.shape[0] + 1.0 - 2.0 * a)
    _live = (x.std() > 0).astype(x.dtype)
    return ndtri(_pp) * _live


def diff_rank(eta, spread, a):
    n = eta.shape[0]
    d = (jnp.argsort(jnp.argsort(eta)) - jnp.argsort(jnp.argsort(spread))).astype(eta.dtype) / n
    _live = ((eta.std() > 0) & (spread.std() > 0)).astype(eta.dtype)
    return rank_gauss(d, a) * _live


__all__ = [
    "BCAArgs",
    "SoftQNetwork",
    "MultiHeadQNetwork",
    "Calibrator",
    "soft_coverage",
    "hard_coverage",
    "reliability_weights",
    "aggregate_q",
    "bayesian_bootstrap_weights",
    "make_cal_loss_fn",
    "sym",
    "rank_gauss",
    "diff_rank",
]
