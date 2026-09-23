"""JAX iql host. BCA is isolated in the corresponding *_bca.py extension."""

import collections
import collections.abc
from dataclasses import dataclass
from typing import Optional
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax.training.train_state import TrainState
import runtime.networks as C

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class Args:
    seed: int = 0
    dataset: str = "halfcheetah-medium-expert-v2"
    algorithm: str = "corl_iql"
    num_updates: int = 1000000
    reward_transform: str = C.RT_DEFAULT_MODE
    reward_scale: float = C.RT_DEFAULT_SCALE
    reward_bias: float = C.RT_DEFAULT_BIAS
    allow_off_config: bool = False
    eval_interval: int = 5000
    eval_workers: int = 10
    eval_final_episodes: int = 1000
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    vf_lr: float = 0.0003
    qf_lr: float = 0.0003
    actor_lr: float = 0.0003
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    normalize: bool = True
    normalize_reward: bool = False
    beta: float = 3.0
    iql_tau: float = 0.7
    iql_deterministic: bool = False
    actor_dropout: Optional[float] = None


AgentTrainState = collections.namedtuple("AgentTrainState", "actor qf qf_target vf")


def asymmetric_l2_loss(u, tau):
    return jnp.mean(jnp.abs(tau - (u < 0.0).astype(jnp.float32)) * u**2)


def gaussian_log_prob(mean, std, value):
    log_scale = jnp.log(std)
    var = std**2
    return (
        -((value - mean) ** 2) / (2.0 * var) - log_scale - 0.5 * jnp.log(2.0 * jnp.pi)
    )


class MLPTrunk(nn.Module):
    in_dim: int
    out_dim: int
    hidden_dim: int = 256
    n_hidden: int = 2
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        fan_in = self.in_dim
        for _ in range(self.n_hidden):
            x = C.torch_dense(self.hidden_dim, fan_in)(x)
            x = nn.relu(x)
            if self.dropout > 0.0:
                x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
            fan_in = self.hidden_dim
        return C.torch_dense(self.out_dim, fan_in)(x)


class GaussianPolicy(nn.Module):
    obs_dim: int
    action_dim: int
    max_action: float
    dropout: float = 0.0

    @nn.compact
    def __call__(self, obs, deterministic=True):
        x = MLPTrunk(self.obs_dim, self.action_dim, dropout=self.dropout)(
            obs, deterministic=deterministic
        )
        mean = jnp.tanh(x)
        log_std = self.param(
            "log_std", lambda key: jnp.zeros(self.action_dim, dtype=jnp.float32)
        )
        std = jnp.exp(jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX))
        return (mean, jnp.broadcast_to(std, mean.shape))

    def act(self, params, obs):
        mean, _ = self.apply(params, obs, deterministic=True)
        return jnp.clip(self.max_action * mean, -self.max_action, self.max_action)


class DeterministicPolicy(nn.Module):
    obs_dim: int
    action_dim: int
    max_action: float
    dropout: float = 0.0

    @nn.compact
    def __call__(self, obs, deterministic=True):
        x = MLPTrunk(self.obs_dim, self.action_dim, dropout=self.dropout)(
            obs, deterministic=deterministic
        )
        return jnp.tanh(x)

    def act(self, params, obs):
        a = self.apply(params, obs, deterministic=True)
        return jnp.clip(a * self.max_action, -self.max_action, self.max_action)


class Critic(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        return MLPTrunk(self.obs_dim + self.action_dim, 1)(x).squeeze(-1)


class TwinQ(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=None,
            out_axes=-1,
            axis_size=2,
        )
        return vmap_critic(self.obs_dim, self.action_dim)(obs, action)


class ValueFunction(nn.Module):
    obs_dim: int

    @nn.compact
    def __call__(self, obs):
        return MLPTrunk(self.obs_dim, 1)(obs).squeeze(-1)


def iql_update(
    args,
    actor_apply_fn,
    q_apply_fn,
    value_apply_fn,
    agent_state,
    batch,
    it,
    rng_dropout,
):
    agent_state, adv, target, nuisance_metrics = nuisance_update(
        args, q_apply_fn, value_apply_fn, agent_state, batch
    )
    value_loss = nuisance_metrics["value_loss"]
    q_loss = nuisance_metrics["q_loss"]
    exp_adv = jnp.minimum(jnp.exp(args.beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)

    def _actor_loss_fn(params):
        out = actor_apply_fn(
            params, batch.obs, deterministic=False, rngs={"dropout": rng_dropout}
        )
        if args.iql_deterministic:
            bc_losses = jnp.sum((out - batch.action) ** 2, axis=1)
        else:
            mean, std = out
            bc_losses = -gaussian_log_prob(mean, std, batch.action).sum(-1)
        return jnp.mean(exp_adv * bc_losses)

    actor_loss, a_grad = jax.value_and_grad(_actor_loss_fn)(agent_state.actor.params)
    agent_state = agent_state._replace(
        actor=agent_state.actor.apply_gradients(grads=a_grad)
    )
    return (
        agent_state,
        {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
            "adv_mean": adv.mean(),
            "exp_adv_mean": exp_adv.mean(),
        },
    )


def make_train_step(args, actor_apply_fn, q_apply_fn, value_apply_fn, dataset):
    n = dataset.obs.shape[0]

    def _train_step(runner_state, _):
        rng, agent_state, it = runner_state
        it = it + 1
        rng, rng_batch, rng_dropout = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (args.batch_size,), 0, n)
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        agent_state, loss = iql_update(
            args,
            actor_apply_fn,
            q_apply_fn,
            value_apply_fn,
            agent_state,
            batch,
            it,
            rng_dropout,
        )
        return ((rng, agent_state, it), loss)

    return _train_step


def initialize_agent(args, obs_dim, action_dim, max_action):
    key = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1414676809)
    key, ka, kq, kv = jax.random.split(key, 4)
    dropout = 0.0 if args.actor_dropout is None else float(args.actor_dropout)
    actor = (DeterministicPolicy if args.iql_deterministic else GaussianPolicy)(
        obs_dim, action_dim, max_action, dropout
    )
    q = TwinQ(obs_dim, action_dim)
    v = ValueFunction(obs_dim)
    obs = jnp.zeros(obs_dim)
    action = jnp.zeros(action_dim)

    def make(k, net, inputs, tx):
        return TrainState.create(apply_fn=net.apply, params=net.init(k, *inputs), tx=tx)

    state = AgentTrainState(
        make(
            ka,
            actor,
            [obs],
            C.torch_adam(optax.cosine_decay_schedule(args.actor_lr, args.num_updates)),
        ),
        make(kq, q, [obs, action], C.torch_adam(args.qf_lr)),
        make(kq, q, [obs, action], C.torch_adam(args.qf_lr)),
        make(kv, v, [obs], C.torch_adam(args.vf_lr)),
    )
    return (state, key, actor)


def nuisance_update(args, q_apply_fn, value_apply_fn, state, batch):
    next_v = jax.lax.stop_gradient(value_apply_fn(state.vf.params, batch.next_obs))
    target_q = jax.lax.stop_gradient(
        jnp.min(q_apply_fn(state.qf_target.params, batch.obs, batch.action), axis=-1)
    )

    def value_loss(params):
        advantage = target_q - value_apply_fn(params, batch.obs)
        return (asymmetric_l2_loss(advantage, args.iql_tau), advantage)

    (v_loss, adv), grad = jax.value_and_grad(value_loss, has_aux=True)(state.vf.params)
    state = state._replace(vf=state.vf.apply_gradients(grads=grad))
    target = batch.reward + (1.0 - batch.done) * args.discount * next_v

    def q_loss(params):
        q = q_apply_fn(params, batch.obs, batch.action)
        return jnp.square(q - target[:, None]).mean()

    q_loss, grad = jax.value_and_grad(q_loss)(state.qf.params)
    state = state._replace(qf=state.qf.apply_gradients(grads=grad))
    state = state._replace(
        qf_target=state.qf_target.replace(
            step=state.qf_target.step + 1,
            params=optax.incremental_update(
                state.qf.params, state.qf_target.params, args.tau
            ),
        )
    )
    return (
        state,
        jax.lax.stop_gradient(adv),
        target,
        {"value_loss": v_loss, "q_loss": q_loss},
    )
