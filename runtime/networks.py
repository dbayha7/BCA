"""Shared JAX initialization and optimizer conventions; no simulator or BCA imports."""

import math
from collections import namedtuple
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from runtime.rewards import (
    DEFAULT_BIAS as RT_DEFAULT_BIAS,
    DEFAULT_MODE as RT_DEFAULT_MODE,
    DEFAULT_SCALE as RT_DEFAULT_SCALE,
    check_reward_transform,
    is_antmaze,
)

Transition = namedtuple("Transition", "obs action reward next_obs done")
TransitionNA = namedtuple("TransitionNA", "obs action reward next_obs done next_action")


def pytorch_init(fan_in):
    bound = math.sqrt(1.0 / fan_in)

    def _init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def uniform_init(bound):

    def _init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(
            key, shape=shape, minval=-bound, maxval=bound, dtype=dtype
        )

    return _init


def orthogonal_init(scale=math.sqrt(2.0)):
    return nn.initializers.orthogonal(scale)


def torch_dense(features, fan_in, **kwargs):
    return nn.Dense(
        features,
        kernel_init=pytorch_init(fan_in),
        bias_init=pytorch_init(fan_in),
        **kwargs
    )


def identity(x):
    return x


def torch_adam(learning_rate, eps=1e-08, b1=0.9, b2=0.999):
    return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, eps_root=0.0)
