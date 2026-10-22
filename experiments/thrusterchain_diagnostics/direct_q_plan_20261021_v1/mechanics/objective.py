"""Proposed simple direct-width actor objective; not Option C or a Q certificate.

Mechanical primitive only. Call validate_inputs on concrete forward outputs before
differentiation. All candidate widths must be finite and nonnegative; statistical
support decisions belong to a separate caller and are not inferred here.
"""
from typing import NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

CANDIDATE_ID = "simple-direct-width-mechanics-v1"
MODES = ("live", "detached", "constant", "none")


class Terms(NamedTuple):
    loss: jax.Array
    raw_q_mean: jax.Array
    penalty_mean: jax.Array
    bc_mean: jax.Array
    host_scale: jax.Array


def freeze_parameters(params):
    """Freeze model parameter derivatives without detaching its action input."""
    return jax.tree_util.tree_map(jax.lax.stop_gradient, params)


def validate_inputs(actions, logged_actions, q_values, widths, *,
                    strength=1.0, host_alpha=2.5, constant=0.0, mode="live"):
    """Concrete eager preflight, not a JIT/grad-time predicate or support test."""
    a, b, q, u = map(np.asarray, (actions, logged_actions, q_values, widths))
    if a.ndim != 2 or b.shape != a.shape or not a.shape[0] or not a.shape[1]:
        raise ValueError("actions and logged_actions must be nonempty [batch, action]")
    if q.shape != (len(a),) or u.shape != (len(a),):
        raise ValueError("Q and width must be [batch], without broadcasting")
    if mode not in MODES:
        raise ValueError("unknown penalty mode")
    consume_width = strength != 0 and mode in ("live", "detached")
    if not all(np.isfinite(v).all() for v in ((a, b, q, u) if consume_width else (a,b,q))):
        raise ValueError("nonfinite input: refuse this width before objective evaluation")
    if (consume_width and (u < 0).any()) or not np.isfinite([strength, host_alpha, constant]).all():
        raise ValueError("width and coefficients must be finite and nonnegative")
    if strength < 0 or host_alpha <= 0 or constant < 0:
        raise ValueError("invalid coefficient")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        denominator = np.abs(q).mean()
        scale = np.asarray(host_alpha,dtype=q.dtype) / denominator
        penalty = (u if consume_width else np.full_like(q,constant)
                   if mode == "constant" and strength != 0 else np.zeros_like(q))
        score = q-strength*penalty
        loss = -scale*score.mean()+np.square(a-b).mean()
    if denominator <= 0 or not all(np.isfinite(v).all() for v in (denominator,scale,score,loss)):
        raise ValueError("nonfinite derived objective or invalid raw-Q normalization; no silent floor")


def raw_q_scale(q_values, host_alpha=2.5):
    """Match host normalization: raw Q1, detached coefficient, no added epsilon."""
    return host_alpha / jax.lax.stop_gradient(jnp.abs(q_values).mean())


def candidate_scores(q_values, widths, *, strength=1.0, mode="live", constant=0.0):
    """Differentiate live width only; constant is a stopped scalar placebo."""
    if mode not in MODES:
        raise ValueError("unknown penalty mode")
    if mode == "none" or strength == 0:
        return q_values, jnp.zeros_like(q_values)
    if mode == "live":
        penalty = widths
    elif mode == "detached":
        penalty = jax.lax.stop_gradient(widths)
    else:
        penalty = jnp.ones_like(q_values) * jax.lax.stop_gradient(jnp.asarray(constant))
    penalty = strength * penalty
    return q_values - penalty, penalty


def actor_objective(actions, logged_actions, q_values, widths, *,
                    strength=1.0, host_alpha=2.5, mode="live", constant=0.0,
                    fixed_host_scale=None):
    """Finite-input actor objective; differentiation remains live through actions.

    fixed_host_scale is for a local derivative/finite-difference fixture. It is
    computed at the unperturbed action and held numerically fixed. Recomputing
    stop-gradient coefficients under perturbation tests a different derivative.
    strength/mode are static configuration, not learned or traced arguments.
    """
    if actions.ndim != 2 or logged_actions.shape != actions.shape:
        raise ValueError("action shape mismatch")
    if q_values.shape != (actions.shape[0],) or widths.shape != q_values.shape:
        raise ValueError("Q and width must be [batch]")
    scale = (raw_q_scale(q_values, host_alpha) if fixed_host_scale is None
             else jax.lax.stop_gradient(jnp.asarray(fixed_host_scale)))
    score, penalty = candidate_scores(q_values, widths, strength=strength,
                                      mode=mode, constant=constant)
    bc = jnp.square(actions - logged_actions).mean()
    return Terms(-scale * score.mean() + bc, q_values.mean(),
                 penalty.mean(), bc, scale)
