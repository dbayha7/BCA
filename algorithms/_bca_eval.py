"""BCA bca eval.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

from numbers import Integral
import jax
import jax.numpy as jnp


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def evaluation_key(rng, seed, evaluation_id):
    seed = _integer(seed, "seed")
    if seed < 0:
        return rng
    evaluation_id = _integer(evaluation_id, "evaluation_id")
    if seed > 4294967295 or not 0 <= evaluation_id <= 4294967295:
        raise ValueError("enabled seed and evaluation_id must be in [0, 2**32 - 1]")
    return jax.random.fold_in(jax.random.PRNGKey(seed), evaluation_id)


def reset_environment(env, reset_key, workers, seeded):
    if not seeded:
        return env.reset()
    workers = _integer(workers, "workers")
    if workers < 1 or getattr(env, "num_envs", workers) != workers:
        raise ValueError("workers must be positive and match the vector environment")
    seeds = jax.random.randint(reset_key, (workers,), 0, 2**31 - 1, dtype=jnp.int32)
    env.seed(jax.device_get(seeds).tolist())
    return env.reset()
