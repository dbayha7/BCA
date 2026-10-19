"""Micro-benchmark on MADE-UP arrays (no ThrusterChain data, no repository import).

Restates Unifloral TD3+BC's make_train_step (baselines/unifloral/algorithms/td3_bc.py:174-279, revision f2dc127)
with the same network sizes (13 obs, 4 actions, 2x256 ReLU MLPs, dual Q via nn.vmap), batch 256, two critic updates
per outer step, Adam(3e-4, eps 1e-5), Polyak 0.005, gamma 0.98, scanned in chunks of 2,500 outer steps exactly as the
training block scans it (td3_bc.py:332-337). The only change: distrax.Transformed(Deterministic(x), Tanh()).sample()
is written as jnp.tanh(x) (the same value; distrax is not installed in the scratch venv).

Variants:
  alone   the restated Unifloral step
  bca     alone + a BCA-like per-step fit: one more own-target computation on the batch (actor_target + dual_q_target
          forward at next obs with fresh noise), q = min(Q1,Q2)(s,a), and one Adam step of a 17-64-64-1 softplus scale
          network (calibration/network.Calibrator size) on a Bayesian-bootstrap-weighted soft-coverage loss, plus a dose
          read of that network on the batch (the shift_hosts TD3+BC+BCA step's per-step work, approximately)
  lower   alone + the value lower edge in the actor loss: -lambda*mean[Q1(s, pi(s)+xi) - k*R*eta(s, pi(s)+xi)*unit]
          with a frozen 17-64-64-1 eta network (gradient flows through eta's command input), xi ~ clip(N(0,0.2^2),+-0.5)

Usage: python -B bench_td3bc.py <variant> <chunks> [n_rows]
"""
import sys
import time
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

jax.config.update("jax_default_matmul_precision", "highest")

VARIANT = sys.argv[1]
CHUNKS = int(sys.argv[2])
N_ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 42_500
EVAL_INTERVAL = 2_500


class Args:
    lr = 3e-4
    batch_size = 256
    gamma = 0.98
    polyak_step_size = 0.005
    td3_alpha = 2.5
    noise_clip = 0.5
    policy_noise = 0.2
    num_critic_updates_per_step = 2


args = Args()
AgentTrainState = namedtuple("AgentTrainState", "actor actor_target dual_q dual_q_target")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return nn.Dense(1)(x).squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(SoftQNetwork, variable_axes={"params": 0}, split_rngs={"params": True, "dropout": True},
                              in_axes=None, out_axes=-1, axis_size=2)
        return vmap_critic(self.obs_mean, self.obs_std)(obs, action)


class TanhActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        return jnp.tanh(nn.Dense(self.num_actions)(x))


class Eta(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([(obs - self.obs_mean) / (self.obs_std + 0.001), action], axis=-1)
        for _ in range(2):
            x = nn.relu(nn.Dense(64)(x))
        return jax.nn.softplus(nn.Dense(1)(x).squeeze(-1)) + 0.01


def create_train_state(rng, network, dummy_input, eps=1e-5):
    return TrainState.create(apply_fn=network.apply, params=network.init(rng, *dummy_input),
                             tx=optax.adam(args.lr, eps=eps))


def make_train_step(actor_apply_fn, q_apply_fn, eta_apply_fn, dataset, variant):
    def _train_step(runner_state, _):
        rng, agent_state, eta_state = runner_state
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(rng_batch, (args.batch_size,), 0, len(dataset.obs))
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        def _targets(rng, agent_state):
            def _compute_target(rng, transition):
                next_obs = transition.next_obs
                action = actor_apply_fn(agent_state.actor_target.params, next_obs)
                rng, rng_noise = jax.random.split(rng)
                noise = jnp.clip(jax.random.normal(rng_noise, shape=action.shape) * args.policy_noise,
                                 -args.noise_clip, args.noise_clip)
                action = jnp.clip(action + noise, -1, 1)
                target_q = q_apply_fn(agent_state.dual_q_target.params, next_obs, action)
                return transition.reward + args.gamma * (1.0 - transition.done) * jnp.min(target_q)

            keys = jax.random.split(rng, args.batch_size)
            return jax.vmap(_compute_target)(keys, batch)

        dose_mean = jnp.float32(1.0)
        if variant == "bca":
            # BCA's fit: own target (state at the start of the step), q = min(Q1, Q2) at (s, a), one scale-net step
            fit_key = jax.random.fold_in(rng, 1179210836)
            y = _targets(fit_key, agent_state)
            q = jnp.min(q_apply_fn(agent_state.dual_q.params, batch.obs, batch.action), axis=-1)
            r = jax.lax.stop_gradient(jnp.abs(y - q))
            wb = jax.random.exponential(jax.random.fold_in(fit_key, 1128352850), (args.batch_size,))
            wb = wb / wb.sum()

            def _eta_loss(p):
                eta = eta_apply_fn(p, batch.obs, batch.action)
                cov = jax.nn.sigmoid(20.0 * (eta - r))
                return jnp.square((wb * cov).sum() - 0.9)

            g = jax.grad(_eta_loss)(eta_state.params)
            eta_state = eta_state.apply_gradients(grads=g)
            u = eta_apply_fn(eta_state.params, batch.obs, batch.action)
            dose = 1.0 + 0.5 * u / (u + 1.0)
            dose_mean = dose.mean()

        def _update_critics(runner_state, _):
            rng, agent_state = runner_state
            rng, rng_targets = jax.random.split(rng)
            targets = _targets(rng_targets, agent_state)

            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                return jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            agent_state = agent_state._replace(dual_q=agent_state.dual_q.apply_gradients(grads=q_grad))
            return (rng, agent_state), q_loss

        (rng, agent_state), q_loss = jax.lax.scan(_update_critics, (rng, agent_state), None,
                                                  length=args.num_critic_updates_per_step)
        rng, rng_xi = jax.random.split(rng)
        xi = jnp.clip(0.2 * jax.random.normal(rng_xi, batch.action.shape), -0.5, 0.5)

        def _actor_loss_function(params):
            def _transition_loss(transition, xi_row):
                pi_action = actor_apply_fn(params, transition.obs)
                if variant == "lower":
                    a = jnp.clip(pi_action + xi_row, -1.0, 1.0)
                    q = q_apply_fn(agent_state.dual_q.params, transition.obs, a)[0]
                    width = 1.7 * eta_apply_fn(eta_state.params, transition.obs, a) * 0.5
                    q = q - 1.0 * width
                else:
                    q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)[0]
                bc_loss = jnp.square(pi_action - transition.action).mean()
                return q, bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch, xi)
            lambda_ = jax.lax.stop_gradient(args.td3_alpha / (jnp.abs(q).mean() + 1e-7))
            actor_loss = -lambda_ * q.mean() + (dose_mean * bc_loss).mean()
            return actor_loss.mean(), (q.mean(), lambda_.mean(), bc_loss.mean())

        (actor_loss, aux), actor_grad = jax.value_and_grad(_actor_loss_function, has_aux=True)(
            agent_state.actor.params)
        agent_state = agent_state._replace(actor=agent_state.actor.apply_gradients(grads=actor_grad))

        def _update_target(state, target_state):
            new = optax.incremental_update(state.params, target_state.params, args.polyak_step_size)
            return target_state.replace(step=target_state.step + 1, params=new)

        agent_state = agent_state._replace(
            actor_target=_update_target(agent_state.actor, agent_state.actor_target),
            dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target))
        return (rng, agent_state, eta_state), (actor_loss, q_loss.mean())

    return _train_step


def main():
    rs = np.random.default_rng(12345)          # made-up arrays only
    obs = rs.standard_normal((N_ROWS, 13)).astype(np.float32)
    act = np.tanh(rs.standard_normal((N_ROWS, 4))).astype(np.float32)
    rew = -rs.random(N_ROWS).astype(np.float32)
    nxt = (obs + 0.1 * rs.standard_normal((N_ROWS, 13))).astype(np.float32)
    dataset = Transition(jnp.array(obs), jnp.array(act), jnp.array(rew), jnp.array(nxt),
                         jnp.zeros(N_ROWS, jnp.float32))
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    actor_net = TanhActor(4, obs_mean, obs_std)
    q_net = DualQNetwork(obs_mean, obs_std)
    eta_net = Eta(obs_mean, obs_std)
    rng = jax.random.PRNGKey(0)
    rng, ra, rq, re = jax.random.split(rng, 4)
    d_obs, d_act = jnp.zeros((13,)), jnp.zeros(4)
    agent_state = AgentTrainState(
        actor=create_train_state(ra, actor_net, [d_obs]), actor_target=create_train_state(ra, actor_net, [d_obs]),
        dual_q=create_train_state(rq, q_net, [d_obs, d_act]),
        dual_q_target=create_train_state(rq, q_net, [d_obs, d_act]))
    eta_state = create_train_state(re, eta_net, [d_obs, d_act], eps=1e-8)
    step = make_train_step(actor_net.apply, q_net.apply, eta_net.apply, dataset, VARIANT)
    chunk = jax.jit(lambda carry: jax.lax.scan(step, carry, None, EVAL_INTERVAL))
    carry = (rng, agent_state, eta_state)
    t0 = time.perf_counter()
    carry, out = chunk(carry)
    jax.block_until_ready(out)
    first = time.perf_counter() - t0
    times = []
    for _ in range(CHUNKS):
        t0 = time.perf_counter()
        carry, out = chunk(carry)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    ms = 1000.0 * np.median(times) / EVAL_INTERVAL
    print(f"variant={VARIANT} rows={N_ROWS} first_chunk_s={first:.2f} steady_ms_per_outer_step={ms:.3f} "
          f"chunks={[round(t, 2) for t in times]} finite={bool(np.isfinite(np.asarray(out[0])).all())} "
          f"min_per_100k={ms * 100_000 / 60_000:.1f} xla_flags={__import__('os').environ.get('XLA_FLAGS')!r} "
          f"affinity={sorted(__import__('os').sched_getaffinity(0))}", flush=True)


if __name__ == "__main__":
    main()
