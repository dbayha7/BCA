"""JAX cql host. BCA is isolated in the corresponding *_bca.py extension."""

import collections
import collections.abc
from dataclasses import dataclass
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as onp
import optax
from flax.training.train_state import TrainState
import runtime.networks as C


@dataclass
class Args:
    seed: int = 0
    dataset: str = "halfcheetah-medium-expert-v2"
    algorithm: str = "corl_cql"
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
    batch_size: int = 256
    discount: float = 0.99
    policy_lr: float = 3e-05
    qf_lr: float = 0.0003
    soft_target_update_rate: float = 0.005
    target_update_period: int = 1
    alpha_multiplier: float = 1.0
    use_automatic_entropy_tuning: bool = True
    backup_entropy: bool = False
    policy_log_std_multiplier: float = 1.0
    cql_n_actions: int = 10
    cql_importance_sample: bool = True
    cql_lagrange: bool = False
    cql_target_action_gap: float = -1.0
    cql_temp: float = 1.0
    cql_alpha: float = 10.0
    cql_max_target_backup: bool = False
    cql_clip_diff_min: float = -onp.inf
    cql_clip_diff_max: float = onp.inf
    orthogonal_init: bool = True
    q_n_hidden_layers: int = 3
    bc_steps: int = 0
    normalize: bool = True
    normalize_reward: bool = False
    cql_reward_scale: float = 5.0
    cql_reward_bias: float = -1.0


AgentTrainState = collections.namedtuple(
    "AgentTrainState",
    "actor critic1 critic2 critic1_target critic2_target log_alpha log_alpha_prime",
)
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def scalar_value(state):
    return state.params["constant"]


def corl_dense(features, fan_in, fan_out, orthogonal, is_last):
    zeros = nn.initializers.zeros
    if is_last:
        if orthogonal:
            return nn.Dense(
                features, kernel_init=C.orthogonal_init(0.01), bias_init=zeros
            )
        bound = 0.01 * float(onp.sqrt(6.0 / (fan_in + fan_out)))
        return nn.Dense(features, kernel_init=C.uniform_init(bound), bias_init=zeros)
    if orthogonal:
        return nn.Dense(
            features,
            kernel_init=C.orthogonal_init(float(onp.sqrt(2.0))),
            bias_init=zeros,
        )
    return C.torch_dense(features, fan_in)


class TanhGaussianPolicy(nn.Module):
    obs_dim: int
    action_dim: int
    max_action: float
    orthogonal_init: bool = False
    log_std_multiplier_init: float = 1.0
    log_std_offset_init: float = -1.0

    @nn.compact
    def __call__(self, obs):
        x = obs
        fan = self.obs_dim
        for _ in range(3):
            x = nn.relu(corl_dense(256, fan, 256, False, False)(x))
            fan = 256
        out = corl_dense(2 * self.action_dim, fan, 2 * self.action_dim, False, True)(x)
        mean, log_std = jnp.split(out, 2, axis=-1)
        mult = self.param(
            "log_std_multiplier",
            lambda k: jnp.asarray(self.log_std_multiplier_init, jnp.float32),
        )
        off = self.param(
            "log_std_offset",
            lambda k: jnp.asarray(self.log_std_offset_init, jnp.float32),
        )
        return (mean, mult * log_std + off)


class FullyConnectedQFunction(nn.Module):
    obs_dim: int
    action_dim: int
    orthogonal_init: bool = False
    n_hidden_layers: int = 3

    @nn.compact
    def __call__(self, obs, action):
        if action.ndim == obs.ndim + 1:
            obs = jnp.broadcast_to(
                obs[..., None, :], action.shape[:-1] + (obs.shape[-1],)
            )
        x = jnp.concatenate([obs, action], axis=-1)
        fan = self.obs_dim + self.action_dim
        for _ in range(self.n_hidden_layers):
            x = nn.relu(corl_dense(256, fan, 256, self.orthogonal_init, False)(x))
            fan = 256
        return corl_dense(1, fan, 1, self.orthogonal_init, True)(x).squeeze(-1)


def tanh_gaussian_sample(mean, log_std, eps, max_action):
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    std = jnp.exp(log_std)
    x = mean + eps * std
    base_log_prob = (
        -jnp.square(x - mean) / (2.0 * jnp.square(std))
        - jnp.log(std)
        - float(onp.log(onp.sqrt(2.0 * onp.pi)))
    )
    log_det = 2.0 * (float(onp.log(2.0)) - x - jax.nn.softplus(-2.0 * x))
    return (max_action * jnp.tanh(x), jnp.sum(base_log_prob - log_det, axis=-1))


def cql_update(
    args,
    actor_apply_fn,
    critic1_apply_fn,
    critic2_apply_fn,
    agent_state,
    batch,
    it,
    rng,
    max_action,
    target_entropy,
    conservative_multiplier=None,
):
    batch_size = batch.action.shape[0]
    action_dim = batch.action.shape[-1]
    n_act = args.cql_n_actions
    rng_pi, rng_bc, rng_next, rng_rand, rng_cur, rng_nxt = jax.random.split(rng, 6)
    eps_pi = jax.random.normal(rng_pi, (batch_size, action_dim))
    eps_bc = jax.random.normal(rng_bc, (batch_size, action_dim))
    next_shape = (
        (batch_size, n_act, action_dim)
        if args.cql_max_target_backup
        else (batch_size, action_dim)
    )
    eps_next = jax.random.normal(rng_next, next_shape)
    u_rand = jax.random.uniform(
        rng_rand, (batch_size, n_act, action_dim), minval=-1.0, maxval=1.0
    )
    eps_cur = jax.random.normal(rng_cur, (batch_size, n_act, action_dim))
    eps_nxt = jax.random.normal(rng_nxt, (batch_size, n_act, action_dim))

    def _repeat(obs):
        return jnp.broadcast_to(obs[:, None, :], (batch_size, n_act, obs.shape[-1]))

    _, log_pi_now = tanh_gaussian_sample(
        *actor_apply_fn(agent_state.actor.params, batch.obs), eps_pi, max_action
    )
    if args.use_automatic_entropy_tuning:

        def _alpha_loss_fn(params):
            return -(
                params["constant"] * jax.lax.stop_gradient(log_pi_now + target_entropy)
            ).mean()

        alpha_loss, alpha_grad = jax.value_and_grad(_alpha_loss_fn)(
            agent_state.log_alpha.params
        )
        alpha = jnp.exp(scalar_value(agent_state.log_alpha)) * args.alpha_multiplier
    else:
        alpha_loss = jnp.float32(0.0)
        alpha_grad = jax.tree_util.tree_map(
            jnp.zeros_like, agent_state.log_alpha.params
        )
        alpha = jnp.asarray(args.alpha_multiplier, jnp.float32)

    def _policy_loss_fn(actor_params):
        mean, log_std = actor_apply_fn(actor_params, batch.obs)
        new_actions, log_pi = tanh_gaussian_sample(mean, log_std, eps_pi, max_action)
        _, bc_log_probs = tanh_gaussian_sample(mean, log_std, eps_bc, max_action)
        bc_loss = (alpha * log_pi - bc_log_probs).mean()
        q_new = jnp.minimum(
            critic1_apply_fn(agent_state.critic1.params, batch.obs, new_actions),
            critic2_apply_fn(agent_state.critic2.params, batch.obs, new_actions),
        )
        q_loss = (alpha * log_pi - q_new).mean()
        return jnp.where(it <= args.bc_steps, bc_loss, q_loss)

    policy_loss, actor_grad = jax.value_and_grad(_policy_loss_fn)(
        agent_state.actor.params
    )
    if args.cql_max_target_backup:
        next_mean, next_log_std = actor_apply_fn(
            agent_state.actor.params, _repeat(batch.next_obs)
        )
        next_actions, next_log_pi = tanh_gaussian_sample(
            next_mean, next_log_std, eps_next, max_action
        )
        pair_min = jnp.minimum(
            critic1_apply_fn(
                agent_state.critic1_target.params, batch.next_obs, next_actions
            ),
            critic2_apply_fn(
                agent_state.critic2_target.params, batch.next_obs, next_actions
            ),
        )
        max_idx = jnp.argmax(pair_min, axis=-1)
        target_q_values = jnp.max(pair_min, axis=-1)
        next_log_pi = jnp.take_along_axis(
            next_log_pi, max_idx[:, None], axis=-1
        ).squeeze(-1)
    else:
        next_mean, next_log_std = actor_apply_fn(
            agent_state.actor.params, batch.next_obs
        )
        next_actions, next_log_pi = tanh_gaussian_sample(
            next_mean, next_log_std, eps_next, max_action
        )
        target_q_values = jnp.minimum(
            critic1_apply_fn(
                agent_state.critic1_target.params, batch.next_obs, next_actions
            ),
            critic2_apply_fn(
                agent_state.critic2_target.params, batch.next_obs, next_actions
            ),
        )
    if args.backup_entropy:
        target_q_values = target_q_values - alpha * next_log_pi
    td_target = jax.lax.stop_gradient(
        batch.reward + (1.0 - batch.done) * args.discount * target_q_values
    )
    cur_actions, cur_log_pis = tanh_gaussian_sample(
        *actor_apply_fn(agent_state.actor.params, _repeat(batch.obs)),
        eps_cur,
        max_action
    )
    nxt_actions, nxt_log_pis = tanh_gaussian_sample(
        *actor_apply_fn(agent_state.actor.params, _repeat(batch.next_obs)),
        eps_nxt,
        max_action
    )
    cur_actions = jax.lax.stop_gradient(cur_actions)
    cur_log_pis = jax.lax.stop_gradient(cur_log_pis)
    nxt_actions = jax.lax.stop_gradient(nxt_actions)
    nxt_log_pis = jax.lax.stop_gradient(nxt_log_pis)
    alpha_prime = jnp.clip(
        jnp.exp(scalar_value(agent_state.log_alpha_prime)), 0.0, 1000000.0
    )

    def _one_critic(apply_fn, params):
        q_pred = apply_fn(params, batch.obs, batch.action)
        td_loss = jnp.mean(jnp.square(q_pred - td_target))
        q_rand = apply_fn(params, batch.obs, u_rand)
        q_cur = apply_fn(params, batch.obs, cur_actions)
        q_nxt = apply_fn(params, batch.obs, nxt_actions)
        cat_plain = jnp.concatenate([q_rand, q_pred[:, None], q_nxt, q_cur], axis=1)
        std_q = jnp.std(cat_plain, axis=1, ddof=1)
        if args.cql_importance_sample:
            random_density = float(onp.log(0.5**action_dim))
            cat = jnp.concatenate(
                [q_rand - random_density, q_nxt - nxt_log_pis, q_cur - cur_log_pis],
                axis=1,
            )
        else:
            cat = cat_plain
        q_ood = jax.scipy.special.logsumexp(cat / args.cql_temp, axis=1) * args.cql_temp
        q_diff = jnp.mean(
            jnp.clip(q_ood - q_pred, args.cql_clip_diff_min, args.cql_clip_diff_max)
        )
        critic_gap = q_diff
        if conservative_multiplier is not None:
            dose = jnp.asarray(conservative_multiplier)
            if dose.shape != q_pred.shape:
                raise ValueError("conservative_multiplier must have shape (batch,)")
            dose = jax.lax.stop_gradient(
                jnp.where(jnp.all(jnp.isfinite(dose) & (dose >= 0)), dose, jnp.nan)
            )
            critic_gap = jnp.mean(
                dose
                * jnp.clip(
                    q_ood - q_pred, args.cql_clip_diff_min, args.cql_clip_diff_max
                )
            )
        if args.cql_lagrange:
            min_q_loss = (
                alpha_prime * args.cql_alpha * (critic_gap - args.cql_target_action_gap)
            )
        else:
            min_q_loss = critic_gap * args.cql_alpha
        return (
            td_loss + min_q_loss,
            (td_loss, q_diff, min_q_loss, q_pred, q_rand, q_cur, q_nxt, std_q),
        )

    def _critic1_loss_fn(params):
        return _one_critic(critic1_apply_fn, params)

    def _critic2_loss_fn(params):
        return _one_critic(critic2_apply_fn, params)

    (loss1, aux1), grad1 = jax.value_and_grad(_critic1_loss_fn, has_aux=True)(
        agent_state.critic1.params
    )
    (loss2, aux2), grad2 = jax.value_and_grad(_critic2_loss_fn, has_aux=True)(
        agent_state.critic2.params
    )
    qf1_loss, qf1_diff, cql_min_qf1_loss, q1_pred, q1_rand, q1_cur, q1_nxt, std_q1 = (
        aux1
    )
    qf2_loss, qf2_diff, cql_min_qf2_loss, q2_pred, q2_rand, q2_cur, q2_nxt, std_q2 = (
        aux2
    )
    if args.cql_lagrange:

        def _alpha_prime_loss_fn(params):
            ap = jnp.clip(jnp.exp(params["constant"]), 0.0, 1000000.0)
            l1 = ap * args.cql_alpha * (qf1_diff - args.cql_target_action_gap)
            l2 = ap * args.cql_alpha * (qf2_diff - args.cql_target_action_gap)
            return (-l1 - l2) * 0.5

        alpha_prime_loss, alpha_prime_grad = jax.value_and_grad(_alpha_prime_loss_fn)(
            agent_state.log_alpha_prime.params
        )
    else:
        alpha_prime_loss = jnp.float32(0.0)
        alpha_prime_grad = jax.tree_util.tree_map(
            jnp.zeros_like, agent_state.log_alpha_prime.params
        )
        alpha_prime = jnp.float32(0.0)
    agent_state = agent_state._replace(
        actor=agent_state.actor.apply_gradients(grads=actor_grad),
        critic1=agent_state.critic1.apply_gradients(grads=grad1),
        critic2=agent_state.critic2.apply_gradients(grads=grad2),
        log_alpha=agent_state.log_alpha.apply_gradients(grads=alpha_grad),
        log_alpha_prime=agent_state.log_alpha_prime.apply_gradients(
            grads=alpha_prime_grad
        ),
    )

    def _soft(src, tgt):
        return tgt.replace(
            step=tgt.step + 1,
            params=optax.incremental_update(
                src.params, tgt.params, args.soft_target_update_rate
            ),
        )

    def _do_polyak(state):
        return state._replace(
            critic1_target=_soft(state.critic1, state.critic1_target),
            critic2_target=_soft(state.critic2, state.critic2_target),
        )

    agent_state = jax.lax.cond(
        it % args.target_update_period == 0, _do_polyak, lambda s: s, agent_state
    )
    return (
        agent_state,
        {
            "log_pi": log_pi_now.mean(),
            "policy_loss": policy_loss,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "qf1_loss": qf1_loss,
            "qf2_loss": qf2_loss,
            "average_qf1": q1_pred.mean(),
            "average_qf2": q2_pred.mean(),
            "average_target_q": target_q_values.mean(),
            "cql_std_q1": std_q1.mean(),
            "cql_std_q2": std_q2.mean(),
            "cql_q1_rand": q1_rand.mean(),
            "cql_q2_rand": q2_rand.mean(),
            "cql_min_qf1_loss": cql_min_qf1_loss,
            "cql_min_qf2_loss": cql_min_qf2_loss,
            "cql_qf1_diff": qf1_diff,
            "cql_qf2_diff": qf2_diff,
            "cql_q1_current_actions": q1_cur.mean(),
            "cql_q2_current_actions": q2_cur.mean(),
            "cql_q1_next_actions": q1_nxt.mean(),
            "cql_q2_next_actions": q2_nxt.mean(),
            "alpha_prime_loss": alpha_prime_loss,
            "alpha_prime": alpha_prime,
            "qf_loss": loss1 + loss2,
        },
    )


def make_train_step(
    args,
    actor_apply_fn,
    critic1_apply_fn,
    critic2_apply_fn,
    dataset,
    max_action,
    target_entropy,
):
    n = dataset.obs.shape[0]

    def _train_step(runner_state, _):
        rng, agent_state, it = runner_state
        it = it + 1
        rng, rng_batch, rng_update = jax.random.split(rng, 3)
        idx = jax.random.randint(rng_batch, (args.batch_size,), 0, n)
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        agent_state, loss = cql_update(
            args,
            actor_apply_fn,
            critic1_apply_fn,
            critic2_apply_fn,
            agent_state,
            batch,
            it,
            rng_update,
            max_action,
            target_entropy,
        )
        return ((rng, agent_state, it), loss)

    return _train_step


def initialize(args, obs_dim, action_dim, max_action=1.0):
    rng = jax.random.fold_in(jax.random.PRNGKey(args.seed), 1414676809)
    rng, ka, k1, k2 = jax.random.split(rng, 4)
    actor = TanhGaussianPolicy(
        obs_dim,
        action_dim,
        max_action,
        args.orthogonal_init,
        args.policy_log_std_multiplier,
        -1.0,
    )
    c1 = FullyConnectedQFunction(
        obs_dim, action_dim, args.orthogonal_init, args.q_n_hidden_layers
    )
    c2 = FullyConnectedQFunction(obs_dim, action_dim, args.orthogonal_init, 3)
    obs, action = (jnp.zeros(obs_dim), jnp.zeros(action_dim))

    def make(key, net, ins, lr):
        return TrainState.create(
            apply_fn=net.apply, params=net.init(key, *ins), tx=C.torch_adam(lr)
        )

    def scalar(v, lr):
        return TrainState.create(
            apply_fn=C.identity,
            params={"constant": jnp.asarray(v, jnp.float32)},
            tx=C.torch_adam(lr),
        )

    native = AgentTrainState(
        make(ka, actor, [obs], args.policy_lr),
        make(k1, c1, [obs, action], args.qf_lr),
        make(k2, c2, [obs, action], args.qf_lr),
        make(k1, c1, [obs, action], args.qf_lr),
        make(k2, c2, [obs, action], args.qf_lr),
        scalar(0.0, args.policy_lr),
        scalar(1.0, args.qf_lr),
    )
    return (rng, native, (actor, c1, c2))
