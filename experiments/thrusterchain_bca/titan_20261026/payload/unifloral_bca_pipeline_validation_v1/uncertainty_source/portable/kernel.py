"""Repaired-Regular neural TD3+BC and audited direct-width update kernel.

Ported from the read-only historical ``direct_extension.py`` (SHA256
6af068817b3e3ca8b3c183d8a203ad88c44d3f18ab19046ba03e0f8894f1f462).
The inherited network initialization, auxiliary fitting, and frozen-reference
formulas are reused from authenticated source modules. The only new algorithmic
branches are an explicit fixed stronger-BC control and ``fit_auxiliary=False``
for the versioned frozen-offline-reference online strategy. This module does
not load historical weights, collect transitions, select data, or evaluate.
"""

from dataclasses import dataclass
import math

from .dependencies import BASE, NATIVE, DIRECT

import jax
import jax.numpy as jnp
import optax
from calibration.dose import tree_finite
from calibration.reference import reference_valid


METHODS = ("host", "twin_disagreement", "bca_direct", "stronger_bc")


@dataclass(frozen=True)
class Config(DIRECT.Config):
    """All arms carry identical offline auxiliary work for matched attribution.

    ``bc_multiplier`` affects only ``method='stronger_bc'``. Direct BCA always
    retains coefficient-one mean-over-action BC and raw-Q normalization.
    """

    bc_multiplier: float = 2.0

    def __post_init__(self):
        # DIRECT's parent validates native calibration coefficients. DIRECT's
        # method whitelist predates the newly declared stronger-BC control.
        NATIVE.Config.__post_init__(self)
        if self.arm != "bca" or self.method not in METHODS:
            raise ValueError("Every comparison arm requires identical auxiliary calibration")
        if isinstance(self.kappa, bool) or not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")
        if (isinstance(self.bc_multiplier, bool)
                or not math.isfinite(self.bc_multiplier) or self.bc_multiplier <= 1):
            raise ValueError("stronger-BC multiplier must be finite and greater than one")


State = NATIVE.State
fit_scale = NATIVE.fit_scale
refresh = NATIVE.refresh
native_target = NATIVE.native_target
live_width = DIRECT.live_width
require_valid = NATIVE.require_valid


def initialize(args, config, obs_dim=14, action_dim=4, max_action=1.0):
    """Fresh initialization; 14D physical-plus-time is distinct from history."""
    if obs_dim != 14 or action_dim != 4:
        raise ValueError("Repaired-Regular requires 14 observations and 4 actions")
    if float(args.discount) != 0.98:
        raise ValueError("Repaired-Regular requires discount 0.98")
    return NATIVE.initialize(args, config, obs_dim, action_dim, max_action)


def penalty(config, models, reference, obs, action, q):
    """Identical auxiliary forward; only declared actor-score consumption differs."""
    width, ready, valid = live_width(models, reference, obs, action)
    disagreement = .5 * jnp.abs(q[..., 0] - q[..., 1])
    if config.method in ("host", "stronger_bc"):
        used = jnp.zeros_like(width)
    elif config.method == "twin_disagreement":
        used = jnp.where(reference.ready, config.kappa * disagreement,
                         jnp.zeros_like(disagreement))
    else:
        used = config.kappa * width
    return used, dict(width_mean=width.mean(), disagreement_mean=disagreement.mean(),
                     penalty_mean=used.mean(), width_usable=ready,
                     width_inputs_valid=valid)


def update(args, config, models, state, batch, it, rng, max_action=1.0,
           fit_auxiliary=True):
    """One primary critic step, with native delay and atomic validity rejection.

    Auxiliary scale fitting precedes the host update but the actor consumes the
    previous frozen reference, as in the audited code. With fit_auxiliary=False
    neither calibrator nor reference nor residual unit changes. Reference refresh
    remains an explicit adapter operation outside this kernel.
    """
    if fit_auxiliary:
        fitted, diag = fit_scale(args, config, models, state, batch, rng, max_action)
    else:
        fitted = state
        diag = dict(scale_inputs_valid=jnp.bool_(True),
                    scale_fit_accepted=jnp.bool_(False), scale_loss=jnp.float32(0))
    agent = state.native
    actor_apply, critic_apply = models[0].apply, models[1].apply
    # Exact native TD3 target smoothing, detached twin-min target, and two-head
    # mean-over-batch then sum-over-head loss. No uncertainty penalty here.
    noise = jax.random.normal(rng, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    next_action = actor_apply(agent.actor_target.params, batch.next_obs)
    next_action = jnp.clip(next_action + noise, -max_action, max_action)
    next_q = critic_apply(agent.critic_target.params, batch.next_obs, next_action).min(axis=-1)
    target = jax.lax.stop_gradient(batch.reward + (1.-batch.done)*args.discount*next_q)

    def critic_loss(params):
        q = critic_apply(params, batch.obs, batch.action)
        return jnp.square(q-target[:, None]).mean(axis=0).sum()

    c_loss, c_grad = jax.value_and_grad(critic_loss)(agent.critic.params)
    agent = agent._replace(critic=agent.critic.apply_gradients(grads=c_grad))

    def actor_loss(params, critic_params):
        action = actor_apply(params, batch.obs)
        # Differentiate actor params only; critic/scale parameters are constants
        # while the Q and eta action-input derivatives remain live.
        q = critic_apply(critic_params, batch.obs, action)
        raw = q[..., 0]
        denominator = jax.lax.stop_gradient(jnp.abs(raw).mean())
        used, pdiag = penalty(config, models, state.posterior, batch.obs, action, q)
        valid = pdiag["width_inputs_valid"] & jnp.isfinite(denominator) & (denominator > 0)

        def value(_):
            weight = args.alpha / denominator
            bc = jnp.square(action-batch.action).mean()
            if config.method == "stronger_bc":
                bc = config.bc_multiplier * bc
            return -weight*(raw-used).mean()+bc, weight, bc

        loss, weight, bc = jax.lax.cond(valid, value,
            lambda _: (jnp.float32(0), jnp.float32(0), jnp.float32(0)), None)
        return loss, (raw.mean(), weight, bc, pdiag["penalty_mean"], pdiag["width_mean"],
                      pdiag["disagreement_mean"], pdiag["width_usable"], valid)

    def do_actor(native):
        (loss, aux), grad = jax.value_and_grad(actor_loss, has_aux=True)(
            native.actor.params, native.critic.params)
        native = native._replace(actor=native.actor.apply_gradients(grads=grad))

        def soft(src, tgt):
            return tgt.replace(step=tgt.step+1,
                               params=optax.incremental_update(src.params, tgt.params, args.tau))

        native = native._replace(actor_target=soft(native.actor, native.actor_target),
                                 critic_target=soft(native.critic, native.critic_target))
        return native, (loss, *aux)

    def skip_actor(native):
        z = jnp.float32(0)
        return native, (z, z, z, z, z, z, z, jnp.bool_(False), jnp.bool_(True))

    agent, values = jax.lax.cond(it % args.policy_freq == 0, do_actor, skip_actor, agent)
    a_loss, qmean, weight, bc, pmean, wmean, dmean, usable, actor_valid = values
    valid = (tree_finite((batch, state.native, agent, c_loss, c_grad, target,
                          a_loss, qmean, weight, bc, pmean, wmean, dmean))
             & diag["scale_inputs_valid"] & reference_valid(state.posterior) & actor_valid)
    result = jax.lax.cond(valid, lambda _: fitted._replace(native=agent), lambda _: state, None)
    actor_attempt = it % args.policy_freq == 0
    metrics = dict(critic_loss=c_loss, actor_loss=a_loss, q_mean=qmean,
                   **{"lambda": weight}, bc_loss=bc, penalty_mean=pmean, width_mean=wmean,
                   disagreement_mean=dmean, actor_attempt=actor_attempt,
                   width_usable=actor_attempt & usable, width_fallback=actor_attempt & ~usable,
                   inputs_valid=valid, **diag)
    return result, metrics
