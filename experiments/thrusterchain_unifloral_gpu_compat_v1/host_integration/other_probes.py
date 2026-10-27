"""Bounded actual IQL, CQL and ReBRAC host checks; not training adapters."""
from dataclasses import replace
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algorithms import iql, cql, rebrac
from runtime.networks import TransitionNA
from thrusterchain.training_runner import train_transitions
from .td3bc_probe import prepare


def _source(module):
    path = Path(module.__file__)
    return {"source": str(path), "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "jax_version": jax.__version__, "backend": jax.default_backend()}


def _check(name, actual, expected):
    if not np.isfinite(float(actual)) or not np.isclose(float(actual), float(expected), rtol=3e-5, atol=3e-5):
        raise AssertionError(f"{name}: host {actual}, independent reduction {expected}")


def iql_probe(dataset, indices, seed=13):
    """One IQL update; NumPy checks expectile, Bellman, and likelihood losses."""
    batch, _, _ = prepare(dataset, indices)
    args = replace(iql.Args(), seed=seed, discount=.98, batch_size=len(batch.reward))
    state, key, actor = iql.initialize_agent(args, batch.obs.shape[-1], 4, 1.0)
    q, v = iql.TwinQ(batch.obs.shape[-1], 4), iql.ValueFunction(batch.obs.shape[-1])
    heads = np.asarray(q.apply(state.qf.params, batch.obs, batch.action))
    target_heads = np.asarray(q.apply(state.qf_target.params, batch.obs, batch.action))
    values = np.asarray(v.apply(state.vf.params, batch.obs))
    next_values = np.asarray(v.apply(state.vf.params, batch.next_obs))
    adv = target_heads.min(-1) - values
    expectile = np.mean(np.where(adv < 0, 1 - args.iql_tau, args.iql_tau) * adv ** 2)
    target = np.asarray(batch.reward) + (1 - np.asarray(batch.done)) * args.discount * next_values
    q_loss = np.square(heads - target[:, None]).mean()
    mean, std = actor.apply(state.actor.params, batch.obs, deterministic=True)
    mean, std, actions = np.asarray(mean), np.asarray(std), np.asarray(batch.action)
    log_prob = -((actions - mean) ** 2) / (2 * std ** 2) - np.log(std) - .5 * np.log(2 * np.pi)
    actor_loss = np.mean(np.minimum(np.exp(args.beta * adv), iql.EXP_ADV_MAX) * -log_prob.sum(-1))
    updated, metrics = jax.jit(lambda st, k: iql.iql_update(
        args, actor.apply, q.apply, v.apply, st, batch, 1, k))(state, key)
    for name, expected in (("value_loss", expectile), ("q_loss", q_loss), ("actor_loss", actor_loss)):
        _check(name, metrics[name], expected)
    counts = {name: int(getattr(updated, name).step) for name in ("actor", "qf", "qf_target", "vf")}
    if counts != {"actor": 1, "qf": 1, "qf_target": 1, "vf": 1}:
        raise AssertionError(counts)
    return {**_source(iql), "variant": "repository corl_iql Gaussian actor, beta multiplier, capped weights",
            "counts": counts, "losses": {k: float(metrics[k]) for k in ("value_loss", "q_loss", "actor_loss")},
            "expected": {"value_loss": float(expectile), "q_loss": float(q_loss), "actor_loss": float(actor_loss)}}


def _rebrac_batch(dataset, indices):
    batch, _, _ = prepare(dataset, indices)
    full = train_transitions(dataset)
    idx = np.asarray(indices)
    valid = np.asarray(full.next_action_valid)[idx]
    masks = np.asarray(full.bootstrap_mask)[idx]
    if np.any(~valid & (masks != 0)):
        raise ValueError("ReBRAC requires a dataset next action on every bootstrapped row")
    # Missing next actions on true terminal rows are inert because the entire
    # successor expression is multiplied by a zero bootstrap mask.
    next_actions = np.asarray(full.next_actions)[idx].copy()
    next_actions[~valid] = 0
    return TransitionNA(*batch, jnp.asarray(next_actions)), valid


def rebrac_probe(dataset, indices, seed=13):
    """Two ReBRAC updates with an independent next-action target calculation."""
    batch, valid = _rebrac_batch(dataset, indices)
    args = replace(rebrac.Args(), seed=seed, gamma=.98, batch_size=len(batch.reward))
    rng, initial, (actor, critic) = rebrac.initialize(args, batch)
    _, key = jax.random.split(rng)
    a = np.asarray(actor.apply(initial.actor.target_params, batch.next_obs))
    noise = np.asarray(jax.random.normal(key, a.shape)) * args.policy_noise
    a = np.clip(a + np.clip(noise, -args.noise_clip, args.noise_clip), -1, 1)
    qs = np.asarray(critic.apply(initial.critic.target_params, batch.next_obs, jnp.asarray(a)))
    penalty = np.square(a - np.asarray(batch.next_action)).sum(-1)
    target = np.asarray(batch.reward) + (1 - np.asarray(batch.done)) * args.gamma * (qs.min(0) - args.critic_bc_coef * penalty)
    pred = np.asarray(critic.apply(initial.critic.params, batch.obs, batch.action))
    expected = np.square(pred - target[None, :]).mean(axis=1).sum()
    update = jax.jit(lambda st, it, k: rebrac.rebrac_update(args, actor.apply, critic.apply, st, batch, it, k))
    first, rng, metrics = update(initial, jnp.int32(1), rng)
    second, rng, _ = update(first, jnp.int32(2), rng)
    _check("critic_loss", metrics["critic_loss"], expected)
    counts = {"critic": int(second.critic.step), "actor": int(second.actor.step)}
    if counts != {"critic": 2, "actor": 1}:
        raise AssertionError(counts)
    return {**_source(rebrac), "variant": "repository corl_rebrac; dataset next-action critic penalty",
            "counts": counts, "valid_next_actions": int(valid.sum()), "terminal_placeholders": int((~valid).sum()),
            "critic_loss": float(metrics["critic_loss"]), "expected_critic_loss": float(expected)}


def validate_cql_settings(args):
    """Refuse the repository CQL warmup branch for benchmark use.

    Its sampled-action log probability is not the dataset-action likelihood
    expected of a behavior-cloning warmup. The no-warmup branch is checked here.
    """
    if type(args.bc_steps) is not int or args.bc_steps != 0:
        raise ValueError("CQL bc_steps must be 0: repository warmup does not score dataset actions")


def cql_probe(dataset, indices, seed=13):
    """One CQL update; independently reconstruct TD and proposal-gap reductions."""
    batch, _, _ = prepare(dataset, indices)
    args = replace(cql.Args(), seed=seed, discount=.98, batch_size=len(batch.reward))
    validate_cql_settings(args)
    rng, state, (actor, c1, c2) = cql.initialize(args, batch.obs.shape[-1], 4, 1.)
    _, _, knext, krand, kcur, knxt = jax.random.split(rng, 6)
    b, k, d = len(batch.reward), args.cql_n_actions, 4
    next_a, _ = cql.tanh_gaussian_sample(*actor.apply(state.actor.params, batch.next_obs),
                                         jax.random.normal(knext, (b, d)), 1.)
    qnext = np.minimum(np.asarray(c1.apply(state.critic1_target.params, batch.next_obs, next_a)),
                       np.asarray(c2.apply(state.critic2_target.params, batch.next_obs, next_a)))
    target = np.asarray(batch.reward) + (1 - np.asarray(batch.done)) * args.discount * qnext
    repeated = lambda obs: jnp.broadcast_to(obs[:, None, :], (b, k, obs.shape[-1]))
    random = jax.random.uniform(krand, (b, k, d), minval=-1., maxval=1.)
    cur, logcur = cql.tanh_gaussian_sample(*actor.apply(state.actor.params, repeated(batch.obs)),
                                           jax.random.normal(kcur, (b, k, d)), 1.)
    nxt, lognxt = cql.tanh_gaussian_sample(*actor.apply(state.actor.params, repeated(batch.next_obs)),
                                           jax.random.normal(knxt, (b, k, d)), 1.)
    expected = {}
    for name, critic, critic_state in (("1", c1, state.critic1), ("2", c2, state.critic2)):
        pred = np.asarray(critic.apply(critic_state.params, batch.obs, batch.action))
        rand_q = np.asarray(critic.apply(critic_state.params, batch.obs, random))
        cur_q = np.asarray(critic.apply(critic_state.params, batch.obs, cur))
        nxt_q = np.asarray(critic.apply(critic_state.params, batch.obs, nxt))
        logits = np.concatenate([rand_q - np.log(.5 ** d), nxt_q - np.asarray(lognxt),
                                 cur_q - np.asarray(logcur)], axis=1) / args.cql_temp
        maximum = logits.max(axis=1)
        gap = np.mean(args.cql_temp * (maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))) - pred)
        expected["qf" + name + "_loss"] = float(np.square(pred - target).mean())
        expected["cql_qf" + name + "_diff"] = float(gap)
    updated, metrics = jax.jit(lambda st, key: cql.cql_update(
        args, actor.apply, c1.apply, c2.apply, st, batch, 1, key, 1., -4.))(state, rng)
    for name, value in expected.items():
        _check(name, metrics[name], value)
    counts = {name: int(getattr(updated, name).step) for name in
              ("actor", "critic1", "critic2", "critic1_target", "critic2_target", "log_alpha", "log_alpha_prime")}
    if any(v != 1 for v in counts.values()):
        raise AssertionError(counts)
    return {**_source(cql), "variant": "repository corl_cql, importance sampled, temperature 1, no Lagrange",
            "counts": counts, "losses": {k: float(metrics[k]) for k in expected}, "expected": expected,
            "note": "log_alpha_prime optimizer is stepped with zero gradient when Lagrange is disabled"}
