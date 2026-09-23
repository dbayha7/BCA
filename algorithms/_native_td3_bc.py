"""BCA native td3 bc.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

import collections
import collections.abc

if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
    collections.MutableMapping = collections.abc.MutableMapping
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
from flax.training.train_state import TrainState
import _common as C
import _published_config

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    seed: int = 0
    dataset: str = "halfcheetah-medium-expert-v2"
    algorithm: str = "corl_td3_bc"
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
    lr: float = 0.0003
    batch_size: int = 256
    discount: float = 0.99
    tau: float = 0.005
    normalize: bool = True
    normalize_reward: bool = False
    alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2


AgentTrainState = collections.namedtuple(
    "AgentTrainState", "actor actor_target critic critic_target"
)


class Actor(nn.Module):
    obs_dim: int
    action_dim: int
    max_action: float

    @nn.compact
    def __call__(self, obs):
        x = nn.relu(C.torch_dense(256, self.obs_dim)(obs))
        x = nn.relu(C.torch_dense(256, 256)(x))
        x = C.torch_dense(self.action_dim, 256)(x)
        return self.max_action * jnp.tanh(x)


class Critic(nn.Module):
    obs_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.relu(C.torch_dense(256, self.obs_dim + self.action_dim)(x))
        x = nn.relu(C.torch_dense(256, 256)(x))
        return C.torch_dense(1, 256)(x).squeeze(-1)


class DualCritic(nn.Module):
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


def td3_bc_update(
    args,
    actor_apply_fn,
    critic_apply_fn,
    agent_state,
    batch,
    it,
    rng_noise,
    max_action,
    bc_multiplier=None,
):
    if bc_multiplier is not None:
        bc_multiplier = jnp.asarray(bc_multiplier)
        if bc_multiplier.shape != (batch.action.shape[0],):
            raise ValueError("bc_multiplier must be one value per batch row")
    noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    next_action = actor_apply_fn(agent_state.actor_target.params, batch.next_obs)
    next_action = jnp.clip(next_action + noise, -max_action, max_action)
    target_q = critic_apply_fn(agent_state.critic_target.params, batch.next_obs, next_action)
    target_q = jnp.min(target_q, axis=-1)
    target = batch.reward + (1.0 - batch.done) * args.discount * target_q
    target = jax.lax.stop_gradient(target)

    def _critic_loss_fn(params):
        q = critic_apply_fn(params, batch.obs, batch.action)
        return jnp.square(q - target[:, None]).mean(axis=0).sum()

    critic_loss, critic_grad = jax.value_and_grad(_critic_loss_fn)(agent_state.critic.params)
    agent_state = agent_state._replace(
        critic=agent_state.critic.apply_gradients(grads=critic_grad)
    )

    def _actor_loss_fn(params, critic_params):
        pi = actor_apply_fn(params, batch.obs)
        q = critic_apply_fn(critic_params, batch.obs, pi)[..., 0]
        lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        if bc_multiplier is None:
            bc = jnp.square(pi - batch.action).mean()
        else:
            bc = (
                jax.lax.stop_gradient(bc_multiplier)
                * jnp.square(pi - batch.action).mean(axis=-1)
            ).mean()
        return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc))

    def _do_actor_update(state):
        grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        (a_loss, aux), a_grad = grad_fn(state.actor.params, state.critic.params)
        state = state._replace(actor=state.actor.apply_gradients(grads=a_grad))

        def _soft(src, tgt):
            return tgt.replace(
                step=tgt.step + 1,
                params=optax.incremental_update(src.params, tgt.params, args.tau),
            )

        state = state._replace(
            actor_target=_soft(state.actor, state.actor_target),
            critic_target=_soft(state.critic, state.critic_target),
        )
        return (state, (a_loss, *aux))

    def _skip_actor_update(state):
        zero = jnp.float32(0.0)
        return (state, (zero, zero, zero, zero))

    agent_state, (actor_loss, q_mean, lmbda, bc_loss) = jax.lax.cond(
        it % args.policy_freq == 0, _do_actor_update, _skip_actor_update, agent_state
    )
    return (
        agent_state,
        {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "q_mean": q_mean,
            "lambda": lmbda,
            "bc_loss": bc_loss,
        },
    )


def make_train_step(args, actor_apply_fn, critic_apply_fn, dataset, max_action):
    n = dataset.obs.shape[0]

    def _train_step(runner_state, _):
        rng, agent_state, it = runner_state
        it = it + 1
        rng, rng_batch, rng_noise = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (args.batch_size,), 0, n)
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        agent_state, loss = td3_bc_update(
            args, actor_apply_fn, critic_apply_fn, agent_state, batch, it, rng_noise, max_action
        )
        return ((rng, agent_state, it), loss)

    return _train_step
