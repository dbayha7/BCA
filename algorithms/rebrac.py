"""JAX rebrac host. BCA is isolated in the corresponding *_bca.py extension."""

import collections
import collections.abc
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import runtime.networks as C


@dataclass
class Args:
    seed: int = 0
    dataset: str = "halfcheetah-medium-v2"
    algorithm: str = "corl_rebrac"
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
    actor_learning_rate: float = 0.001
    critic_learning_rate: float = 0.001
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    normalize_reward: bool = False
    normalize_states: bool = False
    hidden_dim: int = 256
    actor_n_hiddens: int = 3
    critic_n_hiddens: int = 3
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_ln: bool = False
    critic_ln: bool = True
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    normalize_q: bool = True


class ActorTrainState(TrainState):
    target_params: dict


class CriticTrainState(TrainState):
    target_params: dict


AgentTrainState = collections.namedtuple("AgentTrainState", "actor critic")


class DetActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state):
        s_d, h_d = (state.shape[-1], self.hidden_dim)
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=C.pytorch_init(s_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else C.identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=C.pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else C.identity,
            ]
        layers += [
            nn.Dense(
                self.action_dim,
                kernel_init=C.uniform_init(0.001),
                bias_init=C.uniform_init(0.001),
            ),
            nn.tanh,
        ]
        net = nn.Sequential(layers)
        actions = net(state)
        return actions


class Critic(nn.Module):
    hidden_dim: int = 256
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        s_d, a_d, h_d = (state.shape[-1], action.shape[-1], self.hidden_dim)
        layers = [
            nn.Dense(
                self.hidden_dim,
                kernel_init=C.pytorch_init(s_d + a_d),
                bias_init=nn.initializers.constant(0.1),
            ),
            nn.relu,
            nn.LayerNorm() if self.layernorm else C.identity,
        ]
        for _ in range(self.n_hiddens - 1):
            layers += [
                nn.Dense(
                    self.hidden_dim,
                    kernel_init=C.pytorch_init(h_d),
                    bias_init=nn.initializers.constant(0.1),
                ),
                nn.relu,
                nn.LayerNorm() if self.layernorm else C.identity,
            ]
        layers += [
            nn.Dense(
                1, kernel_init=C.uniform_init(0.003), bias_init=C.uniform_init(0.003)
            )
        ]
        network = nn.Sequential(layers)
        state_action = jnp.hstack([state, action])
        out = network(state_action).squeeze(-1)
        return out


class EnsembleCritic(nn.Module):
    hidden_dim: int = 256
    num_critics: int = 10
    layernorm: bool = True
    n_hiddens: int = 3

    @nn.compact
    def __call__(self, state, action):
        ensemble = nn.vmap(
            target=Critic,
            in_axes=None,
            out_axes=0,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            axis_size=self.num_critics,
        )
        q_values = ensemble(self.hidden_dim, self.layernorm, self.n_hiddens)(
            state, action
        )
        return q_values


def rebrac_update(
    args,
    actor_apply_fn,
    critic_apply_fn,
    agent_state,
    batch,
    it,
    rng,
    *,
    actor_bc_multiplier=None
):
    if actor_bc_multiplier is not None:
        actor_bc_multiplier = jnp.asarray(actor_bc_multiplier)
        if actor_bc_multiplier.shape != (batch.action.shape[0],):
            raise ValueError("actor_bc_multiplier must be an aligned (B,) vector")
        if not (
            jnp.issubdtype(actor_bc_multiplier.dtype, jnp.floating)
            or jnp.issubdtype(actor_bc_multiplier.dtype, jnp.integer)
        ):
            raise TypeError("actor_bc_multiplier must be real numeric")

    def _update_critic(carry):
        agent_state, key = carry
        key, actions_key = jax.random.split(key)
        next_actions = actor_apply_fn(agent_state.actor.target_params, batch.next_obs)
        noise = jnp.clip(
            jax.random.normal(actions_key, next_actions.shape) * args.policy_noise,
            -args.noise_clip,
            args.noise_clip,
        )
        next_actions = jnp.clip(next_actions + noise, -1, 1)
        bc_penalty = ((next_actions - batch.next_action) ** 2).sum(-1)
        next_q = critic_apply_fn(
            agent_state.critic.target_params, batch.next_obs, next_actions
        ).min(0)
        next_q = next_q - args.critic_bc_coef * bc_penalty
        target_q = batch.reward + (1 - batch.done) * args.gamma * next_q

        def critic_loss_fn(critic_params):
            q = critic_apply_fn(critic_params, batch.obs, batch.action)
            q_min = q.min(0).mean()
            loss = ((q - target_q[None, ...]) ** 2).mean(1).sum(0)
            return (loss, q_min)

        (loss, q_min), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(
            agent_state.critic.params
        )
        new_critic = agent_state.critic.apply_gradients(grads=grads)
        return (
            (agent_state._replace(critic=new_critic), key),
            {"critic_loss": loss, "q_min": q_min},
        )

    def _update_actor(carry):
        agent_state, key = carry
        key, random_action_key = jax.random.split(key, 2)

        def actor_loss_fn(params):
            actions = actor_apply_fn(params, batch.obs)
            bc_penalty = ((actions - batch.action) ** 2).sum(-1)
            q_values = critic_apply_fn(
                agent_state.critic.params, batch.obs, actions
            ).min(0)
            lmbda = 1
            if args.normalize_q:
                lmbda = jax.lax.stop_gradient(1 / jnp.abs(q_values).mean())
            if actor_bc_multiplier is None:
                loss = (args.actor_bc_coef * bc_penalty - lmbda * q_values).mean()
            else:
                loss = (
                    args.actor_bc_coef
                    * jax.lax.stop_gradient(actor_bc_multiplier)
                    * bc_penalty
                    - lmbda * q_values
                ).mean()
            random_actions = jax.random.uniform(
                random_action_key, shape=batch.action.shape, minval=-1.0, maxval=1.0
            )
            aux = {
                "bc_mse_policy": bc_penalty.mean(),
                "bc_mse_random": ((random_actions - batch.action) ** 2).sum(-1).mean(),
                "action_mse": ((actions - batch.action) ** 2).mean(),
            }
            return (loss, aux)

        (loss, aux), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            agent_state.actor.params
        )
        new_actor = agent_state.actor.apply_gradients(grads=grads)
        new_actor = new_actor.replace(
            target_params=optax.incremental_update(
                agent_state.actor.params, agent_state.actor.target_params, args.tau
            )
        )
        new_critic = agent_state.critic.replace(
            target_params=optax.incremental_update(
                agent_state.critic.params, agent_state.critic.target_params, args.tau
            )
        )
        carry = (AgentTrainState(actor=new_actor, critic=new_critic), key)
        return (carry, {"actor_loss": loss, **aux})

    def _full_update(carry):
        carry, critic_metrics = _update_critic(carry)
        carry, actor_metrics = _update_actor(carry)
        return (carry, {**critic_metrics, **actor_metrics})

    def _delayed_update(carry):
        carry, critic_metrics = _update_critic(carry)
        zero = jnp.float32(0.0)
        return (
            carry,
            {
                **critic_metrics,
                "actor_loss": zero,
                "bc_mse_policy": zero,
                "bc_mse_random": zero,
                "action_mse": zero,
            },
        )

    (agent_state, rng), metrics = jax.lax.cond(
        it % args.policy_freq == 0, _full_update, _delayed_update, (agent_state, rng)
    )
    return (agent_state, rng, metrics)


def make_train_step(args, actor_apply_fn, critic_apply_fn, dataset):
    n = dataset.obs.shape[0]

    def _train_step(runner_state, _):
        rng, agent_state, it = runner_state
        rng, batch_key = jax.random.split(rng)
        indices = jax.random.randint(
            batch_key, shape=(args.batch_size,), minval=0, maxval=n
        )
        batch = jax.tree_util.tree_map(lambda arr: arr[indices], dataset)
        agent_state, rng, loss = rebrac_update(
            args, actor_apply_fn, critic_apply_fn, agent_state, batch, it, rng
        )
        return ((rng, agent_state, it + 1), loss)

    return _train_step


def initialize(args, training):
    obs, action = (training.obs[:1], training.action[:1])
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1414676809)
    rng, ka, kc = jax.random.split(rng, 3)
    actor = DetActor(
        action.shape[-1], args.hidden_dim, args.actor_ln, args.actor_n_hiddens
    )
    critic = EnsembleCritic(args.hidden_dim, 2, args.critic_ln, args.critic_n_hiddens)
    native = AgentTrainState(
        ActorTrainState.create(
            apply_fn=actor.apply,
            params=actor.init(ka, obs),
            target_params=actor.init(ka, obs),
            tx=C.torch_adam(args.actor_learning_rate),
        ),
        CriticTrainState.create(
            apply_fn=critic.apply,
            params=critic.init(kc, obs, action),
            target_params=critic.init(kc, obs, action),
            tx=C.torch_adam(args.critic_learning_rate),
        ),
    )
    return (rng, native, (actor, critic))
