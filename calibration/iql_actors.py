"""Stacked host/BCA actors using the same nuisance values and minibatches."""

from dataclasses import dataclass, replace
from typing import NamedTuple
import calibration.iql_state as H
from calibration.iql_state import jax, jnp, P
from calibration.advantage import LevelActorWeights
import numpy as np


@dataclass(frozen=True)
class Arm:
    name: str
    mode: str
    beta: float
    gain: float


class SharedCarry(NamedTuple):
    rng: object
    nuisance: object
    step: object
    extra: object
    actors: object


def stack_actors(actor, count):
    if count < 1:
        raise ValueError("at least one actor required")
    return jax.tree_util.tree_map(lambda x: jnp.stack([x] * count), actor)


def actor_at(actors, index):
    return jax.tree_util.tree_map(lambda x: x[index], actors)


def native_weights(advantage, beta):
    adv = jax.lax.stop_gradient(advantage)
    values = jnp.minimum(jnp.exp(beta * adv), H.EXP_ADV_MAX)
    valid = jnp.all(jnp.isfinite(adv)) & jnp.all(jnp.isfinite(values))
    return LevelActorWeights(values, adv, jnp.ones_like(adv, dtype=bool), valid, valid)


def update_actors(args, actors, batch, advantage, dropout_key, weights):
    return jax.vmap(
        lambda actor, weight: H.actor_update(
            args, actor, batch, advantage, dropout_key, weight
        )
    )(actors, weights)
