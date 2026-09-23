"""BCA iw module.

Scientific definitions extracted from the recorded BCA source; see configs/sources.json.
"""

import jax.numpy as jnp


def iw_gauss_logp(mean, action, sigma):
    return -0.5 * (
        jnp.square((action - mean) / sigma) + 2.0 * jnp.log(sigma) + jnp.log(2.0 * jnp.pi)
    ).sum(axis=-1)
