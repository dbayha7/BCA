"""IQL fitting target vintage: pre-update V target, post-update target Q on recorded actions.

IQL's actor weights read min_k Qbar_k(s, a) - V(s), where Qbar is the Polyak target copy
(algorithms/iql.py nuisance_update). The residual therefore uses Qbar's heads, not the
online heads; after the host update they are the copy the next actor advantage reads.
"""

from dataclasses import dataclass
from typing import Callable
import jax
import jax.numpy as jnp
from calibration.iql_loss import CalibrationInputs


@dataclass(frozen=True)
class CorlIQLAdapter:
    q_apply_fn: Callable
    num_heads: int = 2

    def __post_init__(self):
        if self.num_heads != 2:
            raise ValueError("IQL uses its twin critic.")

    def calibration_inputs(self, state, batch, rng, step, host=None):
        if host is None or "target" not in host:
            raise ValueError("Supply the host pre-update V target.")
        heads = self.q_apply_fn(state.qf_target.params, batch.obs, batch.action)
        return CalibrationInputs(
            targets=jax.lax.stop_gradient(host["target"]),
            q_at_data=jax.lax.stop_gradient(jnp.min(heads, axis=-1)),
            q_heads=jax.lax.stop_gradient(heads),
            obs=batch.obs,
            action=batch.action,
            vintage={
                "targets": "pre_update_value_net",
                "q_at_data": "post_update_target",
                "q_heads": "post_update_target",
            },
        )
