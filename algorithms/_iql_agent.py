"""BCA iql agent.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from __future__ import annotations
import _bca_host as H
from _bca_host import C, jax, jnp
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import NamedTuple
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
import _native_iql as BASE
import _published_config
import _iql_posterior as P
from _native_iql import (
    AgentTrainState,
    GaussianPolicy,
    DeterministicPolicy,
    TwinQ,
    ValueFunction,
    EXP_ADV_MAX,
    asymmetric_l2_loss,
    gaussian_log_prob,
)

BASE_HOST = "corl_iql"


@dataclass
class Args(BASE.Args):
    algorithm: str = "corl_iql_posterior"
    posterior: P.PosteriorArgs = field(default_factory=P.PosteriorArgs)


class PosteriorTrainState(NamedTuple):
    calibration: object
    posterior: object


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
            params=optax.incremental_update(state.qf.params, state.qf_target.params, args.tau),
        )
    )
    return (state, jax.lax.stop_gradient(adv), target, {"value_loss": v_loss, "q_loss": q_loss})


def actor_update(args, actor_state, batch, advantage, dropout_key, weights):
    del advantage
    safe_obs = jnp.where(weights.support_mask[:, None], batch.obs, 0.0)
    safe_action = jnp.where(weights.support_mask[:, None], batch.action, 0.0)

    def propose(_):

        def loss(params):
            out = actor_state.apply_fn(
                params, safe_obs, deterministic=False, rngs={"dropout": dropout_key}
            )
            if args.iql_deterministic:
                terms = jnp.sum((out - safe_action) ** 2, axis=1)
            else:
                mean, std = out
                terms = -gaussian_log_prob(mean, std, safe_action).sum(-1)
            return jnp.mean(weights.weights * terms)

        value, grad = jax.value_and_grad(loss)(actor_state.params)
        proposed = actor_state.apply_gradients(grads=grad)
        valid = jnp.isfinite(value) & P.finite_tree(grad) & P.finite_tree(proposed)
        return (
            jax.lax.cond(valid, lambda _: proposed, lambda _: actor_state, None),
            value,
            valid,
        )

    def skip(_):
        return (
            actor_state,
            jnp.asarray(0.0),
            weights.inputs_valid & P.finite_tree(actor_state),
        )

    return jax.lax.cond(weights.has_support & weights.inputs_valid, propose, skip, None)


def component_diagnostics(a, reference, predictions, adv, beta):
    candidates = {}
    valid = jnp.asarray(True)
    for mode in ("full", "floor", "bayes"):
        width, support, ok = P.posterior_width(reference, predictions, mode)
        candidates[mode] = P.gain_actor_weights(
            adv, width, support, beta, EXP_ADV_MAX, a.decision_gain, ok
        )
        valid = valid & candidates[mode].inputs_valid
    active = reference.ready & valid
    full = candidates["full"].weights
    out = {}
    for name, control in [("bayes_effect", "floor"), ("floor_effect", "bayes")]:
        diff = jnp.abs(full - candidates[control].weights)
        out[name + "_absdiff"] = jnp.where(active, jnp.mean(diff), 0.0)
        out[name + "_changed_fraction"] = jnp.where(
            active, jnp.mean((diff > 0).astype(jnp.float32)), 0.0
        )
    return out


def iql_posterior_update(args, state, batch, it, dropout_key, fitter=None, extra=None):
    if args.posterior.mode == "off":
        updated, loss = BASE.iql_update(
            args,
            state.actor.apply_fn,
            state.qf.apply_fn,
            state.vf.apply_fn,
            state,
            batch,
            it,
            dropout_key,
        )
        return (updated, None, loss)
    state, adv, target, loss = nuisance_update(
        args, state.qf.apply_fn, state.vf.apply_fn, state, batch
    )
    calibration, readout, cal_valid = fitter.update(
        extra.calibration, state, batch, target, adv, dropout_key, it
    )
    predictions = fitter.predictions(extra.calibration, extra.posterior.cal_params, batch)
    weights = P.weights_at_reference(
        args.posterior, extra.posterior, predictions, adv, args.beta, EXP_ADV_MAX
    )
    native = jnp.minimum(jnp.exp(args.beta * jax.lax.stop_gradient(adv)), EXP_ADV_MAX)
    weights = weights._replace(
        weights=jnp.where(extra.posterior.ready, weights.weights, native)
    )
    valid = cal_valid & P.finite_tree(state) & P.finite_tree(loss) & weights.inputs_valid
    weights = weights._replace(inputs_valid=valid, has_support=weights.has_support & valid)
    actor, actor_loss, actor_valid = actor_update(
        args, state.actor, batch, adv, dropout_key, weights
    )
    state = state._replace(actor=actor)
    width, support, width_valid = P.posterior_width(
        extra.posterior, predictions, args.posterior.mode
    )
    ready = extra.posterior.ready
    finite_width = jnp.where(support, width, 0.0)
    n_supported = jnp.sum(support)
    mass = jnp.sum(weights.weights)
    diag = {
        "actor_loss": actor_loss,
        "adv_mean": adv.mean(),
        "exp_adv_mean": weights.weights.mean(),
        "posterior_ready": ready.astype(jnp.float32),
        "inputs_valid": (valid & actor_valid).astype(jnp.float32),
        "actor_updated": (weights.has_support & actor_valid).astype(jnp.float32),
        "actor_ess_fraction": mass**2 / (len(adv) * jnp.sum(weights.weights**2) + 1e-30),
        "actor_cap_fraction": jnp.mean((weights.weights >= EXP_ADV_MAX).astype(jnp.float32)),
        "native_cap_fraction": jnp.mean((native >= EXP_ADV_MAX).astype(jnp.float32)),
        "used_advantage_effective": weights.used_advantage.mean(),
        "calibrated_width_mean_finite": jnp.where(
            ready, jnp.sum(finite_width) / jnp.maximum(n_supported, 1), 0.0
        ),
        "width_supported_fraction": jnp.where(
            ready, jnp.mean(support.astype(jnp.float32)), 1.0
        ),
        "scale_cov": readout.coverage,
        "scale_loss": readout.diagnostics["cal_loss"],
        **component_diagnostics(args.posterior, extra.posterior, predictions, adv, args.beta),
    }
    return (state, PosteriorTrainState(calibration, extra.posterior), {**loss, **diag})


def make_train_step(args, state, dataset, fitter=None):
    if args.posterior.mode == "off":
        return BASE.make_train_step(
            args, state.actor.apply_fn, state.qf.apply_fn, state.vf.apply_fn, dataset
        )
    n = len(dataset.reward)

    def step(carry, _):
        rng, agent, it, extra = carry
        it = it + 1
        rng, batch_key, dropout_key = jax.random.split(rng, 3)
        idx = jax.random.randint(batch_key, (args.batch_size,), 0, n)
        batch = jax.tree_util.tree_map(lambda x: x[idx], dataset)
        agent, extra, loss = iql_posterior_update(
            args, agent, batch, it, dropout_key, fitter, extra
        )
        return ((rng, agent, it, extra), loss)

    return step


def load_data(args):
    if args.posterior.mode == "off" and args.posterior.reserve_size == 0:
        train, mean, std = C.load_dataset(
            args.dataset,
            BASE_HOST,
            args.reward_transform,
            args.reward_scale,
            args.reward_bias,
            normalize_reward=args.normalize_reward,
            normalize=args.normalize,
        )
        return P.PreparedData(
            train,
            None,
            mean,
            std,
            None,
            None,
            None,
            {
                "preprocessing": "native C.load_dataset direct call; full-data control",
                "training_size": len(train.reward),
                "calibration_size": 0,
                "obs_mean": np.asarray(mean).tolist(),
                "obs_std_with_epsilon": np.asarray(std).tolist(),
                "training_numeric_sha256": P.fingerprint(train),
            },
        )
    env = C.gym.make(args.dataset)
    try:
        raw = env.get_dataset()
        converted = C.d4rl.qlearning_dataset(env, dataset=raw)
    finally:
        env.close()
    return P.prepare_dataset(args, converted, raw)


def source_provenance():
    alg = Path(BASE.__file__).resolve().parent
    paths = list(alg.glob("*.py")) + list((alg / "corl_bca").glob("*.py"))
    paths += [
        Path(_published_config.REFERENCE_CSV),
        Path(_published_config.REFERENCE_PROVENANCE),
    ]
    return {
        str(path.relative_to(alg.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }
