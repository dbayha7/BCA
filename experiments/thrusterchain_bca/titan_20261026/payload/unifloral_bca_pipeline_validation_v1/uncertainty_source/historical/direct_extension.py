"""Explicit Q1-minus-native-BCA-width actor objective, version 1.

This is an approved direct-Q candidate, not a reconstruction of an absent
original thesis equation. Native residual calibration is deliberately retained;
its one-step twin-min target is NOT a Q1 policy-value uncertainty guarantee.
All compared training arms fit/refresh the auxiliary calibrator identically.
Only the direct actor-score term differs. Critic and bootstrap targets do not
receive an extra uncertainty penalty.
"""
from dataclasses import dataclass
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax
import algorithms.td3_bc as BASE
import algorithms.td3_bc_bca as NATIVE
from calibration.dose import tree_finite
from calibration.reference import reference_valid


METHODS = ("host", "twin_disagreement", "bca_direct")


@dataclass(frozen=True)
class Config(NATIVE.Config):
    method: str = "host"
    kappa: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.arm != "bca" or self.method not in METHODS:
            raise ValueError("Every comparison arm carries the identical auxiliary calibration work")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")


State = NATIVE.State
initialize = NATIVE.initialize
fit_scale = NATIVE.fit_scale
refresh = NATIVE.refresh
reserve_pool = NATIVE.reserve_pool
select_training_pool = NATIVE.select_training_pool
require_valid = NATIVE.require_valid


def live_width(models, reference, observations, actions):
    """Branch before infinite arithmetic; freeze parameters, keep d eta / d a.

    Returned usability is numerical reference usability only. It says nothing
    about categorical-logger support or statistical validity at a query.
    Finite-reference overflow is invalid; it is never called a normal refusal.
    """
    params = jax.lax.stop_gradient(reference.cal_params)
    eta = models[2].apply(params, observations, actions)
    valid = (reference_valid(reference) & jnp.all(jnp.isfinite(eta))
             & jnp.all(eta > 0))
    radius = jax.lax.stop_gradient(reference.threshold)
    unit = jax.lax.stop_gradient(reference.residual_scale)
    scale = jnp.maximum(eta, 1e-6) * unit
    valid = valid & jnp.all(jnp.isfinite(scale)) & jnp.all(scale > 0)
    usable = reference.ready & jnp.isfinite(radius)

    def evaluate(_):
        return radius * scale

    width = jax.lax.cond(usable, evaluate, lambda _: jnp.zeros_like(eta), None)
    valid = valid & jnp.all(jnp.isfinite(width)) & jnp.all(width >= 0)
    valid = valid & (~usable | (radius == 0) | jnp.all(width > 0))
    return width, usable, valid


def penalty(config, models, reference, obs, action, q):
    """Same inputs/auxiliary forward for all arms; method selects consumption."""
    width, ready, valid = live_width(models, reference, obs, action)
    disagreement = .5 * jnp.abs(q[..., 0] - q[..., 1])
    if config.method == "host":
        used = jnp.zeros_like(width)
    elif config.method == "twin_disagreement":
        used = jnp.where(reference.ready, config.kappa * disagreement, jnp.zeros_like(disagreement))
    else:
        used = config.kappa * width
    return used, dict(width_mean=width.mean(), disagreement_mean=disagreement.mean(),
                     penalty_mean=used.mean(), width_usable=ready,
                     width_inputs_valid=valid)


def update(args, config, models, state, batch, it, rng, max_action=1.0):
    # Identical auxiliary fit for every arm. Actor consumes previous frozen
    # reference, exactly as native co-training; no current-update leakage.
    fitted, diag = fit_scale(args, config, models, state, batch, rng, max_action)
    agent = state.native
    actor_apply, critic_apply = models[0].apply, models[1].apply
    noise = jax.random.normal(rng, batch.action.shape) * args.policy_noise
    noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
    next_action = actor_apply(agent.actor_target.params, batch.next_obs)
    next_action = jnp.clip(next_action + noise, -max_action, max_action)
    next_q = critic_apply(agent.critic_target.params, batch.next_obs, next_action).min(axis=-1)
    target = jax.lax.stop_gradient(batch.reward + (1.-batch.done)*args.discount*next_q)

    def critic_loss(params):
        q=critic_apply(params,batch.obs,batch.action)
        return jnp.square(q-target[:,None]).mean(axis=0).sum()

    c_loss,c_grad=jax.value_and_grad(critic_loss)(agent.critic.params)
    agent=agent._replace(critic=agent.critic.apply_gradients(grads=c_grad))

    def actor_loss(params, critic_params):
        action=actor_apply(params,batch.obs)
        # The caller differentiates only actor params; critic/calibration
        # parameters are constants but action derivatives remain live.
        q=critic_apply(critic_params,batch.obs,action)
        raw=q[...,0]
        denominator=jax.lax.stop_gradient(jnp.abs(raw).mean())
        used,pdiag=penalty(config,models,state.posterior,batch.obs,action,q)
        valid=(pdiag["width_inputs_valid"] & jnp.isfinite(denominator) & (denominator>0))
        def value(_):
            weight=args.alpha/denominator
            bc=jnp.square(action-batch.action).mean()
            return (-weight*(raw-used).mean()+bc, weight, bc)
        loss,weight,bc=jax.lax.cond(valid,value,
            lambda _: (jnp.float32(0),jnp.float32(0),jnp.float32(0)),None)
        return loss,(raw.mean(),weight,bc,pdiag["penalty_mean"],pdiag["width_mean"],
                     pdiag["disagreement_mean"],pdiag["width_usable"],valid)

    def do_actor(native):
        (loss,aux),grad=jax.value_and_grad(actor_loss,has_aux=True)(native.actor.params,native.critic.params)
        native=native._replace(actor=native.actor.apply_gradients(grads=grad))
        def soft(src,tgt):
            return tgt.replace(step=tgt.step+1,params=optax.incremental_update(src.params,tgt.params,args.tau))
        native=native._replace(actor_target=soft(native.actor,native.actor_target),
                               critic_target=soft(native.critic,native.critic_target))
        return native,(loss,*aux)

    def skip_actor(native):
        z=jnp.float32(0)
        return native,(z,z,z,z,z,z,z,jnp.bool_(False),jnp.bool_(True))

    agent,values=jax.lax.cond(it%args.policy_freq==0,do_actor,skip_actor,agent)
    a_loss,qmean,weight,bc,pmean,wmean,dmean,usable,actor_valid=values
    valid=(tree_finite((batch,state.native,agent,c_loss,c_grad,target,a_loss,qmean,weight,bc,pmean,wmean,dmean))
           & diag["scale_inputs_valid"] & reference_valid(state.posterior) & actor_valid)
    result=jax.lax.cond(valid,lambda _: fitted._replace(native=agent),lambda _:state,None)
    actor_attempt=it%args.policy_freq==0
    metrics=dict(critic_loss=c_loss,actor_loss=a_loss,q_mean=qmean,**{"lambda":weight},bc_loss=bc,
                 penalty_mean=pmean,width_mean=wmean,disagreement_mean=dmean,
                 actor_attempt=actor_attempt,width_usable=actor_attempt & usable,
                 width_fallback=actor_attempt & ~usable,inputs_valid=valid,**diag)
    return result,metrics


def make_train_step(args,config,models,dataset,max_action=1.0):
    if not len(dataset.obs):
        raise ValueError("Training pool must be nonempty")
    def step(carry,_):
        rng,state,it=carry
        kr,kb,kn=jax.random.split(rng,3)
        idx=jax.random.randint(kb,(args.batch_size,),0,len(dataset.obs))
        batch=jax.tree_util.tree_map(lambda x:x[idx],dataset)
        result,metrics=update(args,config,models,state,batch,it+1,kn,max_action)
        proposed=(kr,result,it+1)
        accepted=jax.lax.cond(metrics["inputs_valid"],lambda _:proposed,lambda _:carry,None)
        return accepted,metrics
    return step
