"""Bounded real-host TD3+BC check on an immutable ThrusterChain train split.

This intentionally performs exactly two optimizer updates and does not save a
policy, select a checkpoint, or evaluate returns. It is not a training runner.
"""
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from algorithms import td3_bc as host
from runtime.networks import Transition
from thrusterchain.training_runner import train_transitions


def prepare(dataset, indices):
    """Normalize using the complete train split, then select explicit rows."""
    train = train_transitions(dataset)
    obs = np.asarray(train.observations, np.float32)
    nxt = np.asarray(train.next_observations, np.float32)
    act = np.asarray(train.actions, np.float32)
    reward = np.asarray(train.rewards, np.float32)
    term = np.asarray(train.terminated, bool)
    mask = np.asarray(train.bootstrap_mask)
    n = len(obs)
    if (obs.ndim != 2 or nxt.shape != obs.shape or act.shape != (n, 4)
            or reward.shape != (n,) or term.shape != (n,) or mask.shape != (n,)
            or not np.array_equal(mask, ~term) or np.any(np.abs(act) > 1)
            or not all(np.all(np.isfinite(x)) for x in (obs, nxt, act, reward))):
        raise ValueError("invalid train transition shape, values, action range, or bootstrap mask")
    idx = np.asarray(indices)
    if idx.ndim != 1 or idx.dtype.kind not in "iu" or len(idx) == 0 or np.any(idx >= n):
        raise ValueError("indices must be a nonempty in-range integer vector")
    if idx.dtype.kind == "i" and np.any(idx < 0):
        raise ValueError("indices must be nonnegative")
    mean, scale = obs.mean(axis=0), obs.std(axis=0) + np.float32(0.001)
    batch = Transition(jnp.asarray((obs[idx] - mean) / scale), jnp.asarray(act[idx]),
                       jnp.asarray(reward[idx]), jnp.asarray((nxt[idx] - mean) / scale),
                       jnp.asarray(term[idx], dtype=jnp.float32))
    return batch, mean, scale


def _manual_critic_loss(args, state, actor, critic, batch, noise_key):
    """Independent NumPy reduction of host network outputs and sampled noise."""
    noise = np.asarray(jax.random.normal(noise_key, batch.action.shape)) * args.policy_noise
    noise = np.clip(noise, -args.noise_clip, args.noise_clip)
    next_action = np.clip(np.asarray(actor.apply(state.actor_target.params, batch.next_obs)) + noise, -1, 1)
    target_heads = np.asarray(critic.apply(state.critic_target.params, batch.next_obs, jnp.asarray(next_action)))
    current_heads = np.asarray(critic.apply(state.critic.params, batch.obs, batch.action))
    target = np.asarray(batch.reward) + (1 - np.asarray(batch.done)) * args.discount * np.min(target_heads, axis=-1)
    assert target.shape == (len(batch.reward),) and current_heads.shape == (len(batch.reward), 2)
    return float(np.square(current_heads - target[:, None]).mean(axis=0).sum())


def _manual_actor_loss(args, before, after, actor, critic, batch):
    actions = np.asarray(actor.apply(before.actor.params, batch.obs))
    q = np.asarray(critic.apply(after.critic.params, batch.obs, jnp.asarray(actions)))[:, 0]
    return float(-args.alpha / np.abs(q).mean() * q.mean() + np.square(actions - np.asarray(batch.action)).mean())


def run_probe(dataset, indices, seed=13):
    """Execute two updates, check losses and delayed counters; return a receipt."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    batch, mean, scale = prepare(dataset, indices)
    args = replace(host.Args(), seed=seed, discount=.98, batch_size=len(batch.reward))
    rng, initial, (actor, critic) = host.initialize(args, batch.obs.shape[-1], 4, 1.0)
    update = jax.jit(lambda state, b, step, key: host.td3_bc_update(
        args, actor.apply, critic.apply, state, b, step, key, 1.0))
    rng, noise1 = jax.random.split(rng)
    expected1 = _manual_critic_loss(args, initial, actor, critic, batch, noise1)
    first, metrics1 = update(initial, batch, jnp.int32(1), noise1)
    rng, noise2 = jax.random.split(rng)
    expected2 = _manual_critic_loss(args, first, actor, critic, batch, noise2)
    second, metrics2 = update(first, batch, jnp.int32(2), noise2)
    actor_expected = _manual_actor_loss(args, first, second, actor, critic, batch)
    measured = {"critic_loss_1": float(metrics1["critic_loss"]),
                "critic_loss_2": float(metrics2["critic_loss"]),
                "actor_loss_2": float(metrics2["actor_loss"])}
    expected = {"critic_loss_1": expected1, "critic_loss_2": expected2, "actor_loss_2": actor_expected}
    for name in measured:
        if not np.isfinite(measured[name]) or not np.isclose(measured[name], expected[name], rtol=2e-5, atol=2e-5):
            raise AssertionError(f"{name} does not match independent reduction")
    counts = {"critic": int(second.critic.step), "actor": int(second.actor.step),
              "critic_target": int(second.critic_target.step), "actor_target": int(second.actor_target.step)}
    if counts != {"critic": 2, "actor": 1, "critic_target": 1, "actor_target": 1}:
        raise AssertionError(f"unexpected update counts: {counts}")
    source = Path(host.__file__)
    return {"purpose": "bounded_two_update_plumbing_and_loss_check_not_a_trained_policy",
            "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset_digest": dataset.digest, "selected_rows": np.asarray(indices).tolist(),
            "batch_size": len(batch.reward), "observation_dim": batch.obs.shape[-1],
            "action_dim": 4, "gamma": args.discount, "reward_transform": "identity",
            "normalization": "train observations mean and std + 0.001, applied to both observation fields",
            "normalization_mean": mean.tolist(), "normalization_scale": scale.tolist(),
            "terminal_rows": int(np.asarray(batch.done).sum()), "counts": counts,
            "metrics": measured, "independent_reductions": expected,
            "jax_version": jax.__version__, "backend": jax.default_backend()}
