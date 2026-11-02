"""Bind Q1-aligned width and action compatibility into the pinned actor loss.

No optimizers, checkpoint writes, simulator calls, or training are initiated by
this module.  The supplied native factory owns the unchanged critic/target path.
"""
import math


def make_combined_width(pipeline, models, neighbor_lookup, compatibility_cost,
                        bandwidth, *, kappa=3.0, eta=0.0):
    """Return native width_fn representing kappa*U + eta*D.

    The native factory multiplies the returned scalar by kappa.  Therefore the
    returned width is U + (eta/kappa)*D.  The pipeline posterior, neighbors and
    bandwidth are frozen; actor actions remain differentiable through both terms.
    The lookup must be a deterministic JAX-traceable function obs -> [B,K,A].
    """
    import jax
    import jax.numpy as jnp

    kappa, eta = float(kappa), float(eta)
    if not math.isfinite(kappa) or kappa <= 0 or not math.isfinite(eta) or eta < 0:
        raise ValueError('kappa must be positive and eta finite and nonnegative')
    if not callable(neighbor_lookup) or not callable(compatibility_cost):
        raise ValueError('Frozen neighbor lookup and compatibility cost are required')
    frozen_h = jax.lax.stop_gradient(jnp.asarray(bandwidth))

    def width(reference, obs, live_actions):
        reference = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
        values, usable, valid = pipeline.live_width(models, reference, obs, live_actions)
        values = jnp.asarray(values)
        if values.shape != (obs.shape[0],) or values.dtype != live_actions.dtype:
            raise ValueError('Uncertainty must be one action-dtype scalar per row')
        if eta != 0.0:
            neighbors = jax.lax.stop_gradient(neighbor_lookup(jax.lax.stop_gradient(obs)))
            cost = compatibility_cost(live_actions, neighbors, frozen_h)
            if cost.shape != values.shape or cost.dtype != values.dtype:
                raise ValueError('Compatibility must be one action-dtype scalar per row')
            values = values + jnp.asarray(eta / kappa, values.dtype) * cost
        return jnp.where(usable & valid & jnp.isfinite(values), values, jnp.full_like(values, jnp.nan))

    return width


def make_actor_microsteps(candidate, args, actor_apply_fn, q_apply_fn, dataset,
                          pipeline, models, neighbor_lookup, compatibility_cost,
                          bandwidth, *, kappa=3.0, eta=0.0):
    """Use the pinned candidate without changing critic loss, targets or schedule."""
    width = make_combined_width(pipeline, models, neighbor_lookup, compatibility_cost,
                                bandwidth, kappa=kappa, eta=eta)
    return candidate.make_microsteps(args, actor_apply_fn, q_apply_fn, dataset,
                                     width_fn=width, kappa_penalty=kappa)


def make_combined_width_from_cost(pipeline, models, cost_fn, *, kappa=3.0, eta=0.0):
    """Bind JIT-compatible ``cost_fn(frozen_observations, live_actions)``.

    This form accepts compatibility_from_observations_jax directly as a closure
    over a frozen, appropriately cross-fitted reference. The caller must ensure
    training queries use a fold-specific or episode-excluding reference; raw
    observations alone cannot establish episode identity. Evaluation anchors
    must be episode-disjoint from the retained fitting data.
    """
    import jax
    import jax.numpy as jnp
    kappa, eta = float(kappa), float(eta)
    if not math.isfinite(kappa) or kappa <= 0 or not math.isfinite(eta) or eta < 0:
        raise ValueError('kappa must be positive and eta finite and nonnegative')
    if not callable(cost_fn):
        raise ValueError('A frozen compatibility cost is required')

    def width(reference, obs, live_actions):
        ref = jax.tree_util.tree_map(jax.lax.stop_gradient, reference)
        values, usable, valid = pipeline.live_width(models, ref, obs, live_actions)
        values = jnp.asarray(values)
        if values.shape != (obs.shape[0],) or values.dtype != live_actions.dtype:
            raise ValueError('Uncertainty must be one action-dtype scalar per row')
        if eta != 0.0:
            cost = cost_fn(jax.lax.stop_gradient(obs), live_actions)
            if cost.shape != values.shape or cost.dtype != values.dtype:
                raise ValueError('Compatibility must be one action-dtype scalar per row')
            values = values + jnp.asarray(eta / kappa, values.dtype) * cost
        return jnp.where(usable & valid & jnp.isfinite(values), values, jnp.full_like(values, jnp.nan))
    return width


def make_actor_microsteps_from_cost(candidate, args, actor_apply_fn, q_apply_fn, dataset,
                                    pipeline, models, cost_fn, *, kappa=3.0, eta=0.0):
    """Bind the full state-to-cost path into the unchanged native candidate."""
    width = make_combined_width_from_cost(pipeline, models, cost_fn, kappa=kappa, eta=eta)
    return candidate.make_microsteps(args, actor_apply_fn, q_apply_fn, dataset,
                                     width_fn=width, kappa_penalty=kappa)
