"""IQL data binding, supported actor updates and posterior-state records."""

from __future__ import annotations
from algorithms.iql import initialize_agent, nuisance_update
import calibration.iql_imports as H
from calibration.iql_imports import C, jax, jnp
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
import algorithms.iql as BASE
import runtime.published as _published_config
import calibration.iql_reference as P
from algorithms.iql import (
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
    calibration: object  # BCAState: the scale network and residual unit still being fit
    posterior: object  # calibration.reference.FrozenReference from the last refresh


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
