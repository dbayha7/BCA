"""Opt-in actor-loss prototype for signed BCA plus revised compatibility.

Standalone only: this module is not connected to the production learner and
does not create an optimizer or execute any update.  The existing candidate's
actor/critic update schedule and twin-min Bellman target are untouched.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

import revised_compat


def bind_training_batch(reference, observations, timesteps, episode_ids, *,
                        signed_params, threshold):
    """Build batch-aligned episode-excluding neighbors on the host.

    Reference must be fit on retained training episodes.  Observation/order and
    episode IDs are checked again by the loss; changing a sampled minibatch
    requires a fresh binding.  The caller owns calibration/training split
    identity and must never fit or select a reference using evaluation labels.
    """
    obs = np.asarray(observations, np.float32)
    ids = np.asarray(episode_ids)
    if (obs.ndim != 2 or obs.shape[1] != 14 or not len(obs)
            or not np.isfinite(obs).all() or ids.shape != (len(obs),)
            or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int32).max)):
        raise ValueError('batch must have aligned finite observations and int32-range episode IDs')
    if not math.isfinite(float(threshold)):
        raise ValueError('signed calibration threshold must be finite')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(signed_params)):
        raise ValueError('signed parameters must be finite')
    query = revised_compat.query_neighbors(reference,obs,timesteps,exclude_episode_ids=ids)
    neighbor_ids = np.asarray(query['episode_ids'])
    if (np.any(neighbor_ids < 0) or np.any(neighbor_ids > np.iinfo(np.int32).max)
            or np.any(neighbor_ids == ids[:,None])):
        raise ValueError('neighbor episode leakage')
    bandwidth = np.asarray(reference.base.action_bandwidth,np.float32)
    if (bandwidth.shape != (4,) or not np.isfinite(bandwidth).all() or np.any(bandwidth <= 0)):
        raise ValueError('invalid action bandwidth')
    return {'observations': jnp.asarray(obs), 'episode_ids': jnp.asarray(ids,dtype=jnp.int32),
            'neighbors': jnp.asarray(query['actions'],dtype=jnp.float32),
            'neighbor_episode_ids': jnp.asarray(neighbor_ids,dtype=jnp.int32),
            'bandwidth': jnp.asarray(bandwidth), 'reliable': jnp.asarray(query['reliable']),
            'signed_params': jax.tree_util.tree_map(jnp.asarray,signed_params),
            'threshold': jnp.asarray(threshold,dtype=jnp.float32)}


def make_actor_loss(candidate, args, actor_apply_fn, q_apply_fn, dataset, *,
                    enabled=False, signed_model=None, eta=0.0):
    """Return opt-in pure actor loss; default is the pinned unmodified actor loss.

    Signature is ``loss(params, agent_state, batch, bound, episode_ids)``.
    With enabled=True, bound must come from bind_training_batch for that exact
    minibatch.  The score is Q1 - max(0,mu+q*sigma) - eta*D when the state
    distance guard is reliable, and omits D otherwise.  The original detached
    TD3+BC Q coefficient and BC term are unchanged.  No training is launched.
    """
    if type(enabled) is not bool:
        raise ValueError('enabled must be an explicit bool')
    eta = float(eta)
    if not math.isfinite(eta) or eta < 0:
        raise ValueError('eta must be finite and nonnegative')
    native = candidate.make_microsteps(args,actor_apply_fn,q_apply_fn,dataset,
                                       width_fn=None,kappa_penalty=0.)
    if not enabled:
        def baseline(params, agent_state, batch, bound=None, episode_ids=None):
            return native.actor_loss(params,agent_state,batch,None)
        return baseline
    if signed_model is None or not callable(getattr(signed_model,'apply',None)):
        raise ValueError('enabled signed BCA requires the frozen signed model')

    def loss(params, agent_state, batch, bound, episode_ids):
        if bound is None or episode_ids is None:
            raise ValueError('explicit episode-excluding binding and batch episode IDs required')
        n = batch.obs.shape[0]
        ids = jnp.asarray(episode_ids)
        if (ids.shape != (n,) or bound['observations'].shape != batch.obs.shape
                or bound['episode_ids'].shape != (n,) or bound['neighbors'].ndim != 3
                or bound['neighbors'].shape[0] != n or bound['neighbors'].shape[2] != 4
                or bound['neighbor_episode_ids'].shape != bound['neighbors'].shape[:2]
                or bound['reliable'].shape != (n,)):
            raise ValueError('bound reference and minibatch shapes differ')
        frozen = jax.tree_util.tree_map(jax.lax.stop_gradient,bound)
        def transition(transition):
            action = actor_apply_fn(params,transition.obs).sample(seed=None)
            q = q_apply_fn(agent_state.dual_q.params,transition.obs,action)[0]
            bc = jnp.square(action-transition.action).mean()
            return q,bc,action
        q,bc,actions = jax.vmap(transition)(batch)
        scale = jax.lax.stop_gradient(args.td3_alpha/(jnp.abs(q).mean()+1e-7))
        mu,sigma = signed_model.apply(frozen['signed_params'],batch.obs,actions)
        if mu.shape != (n,) or sigma.shape != (n,):
            raise ValueError('signed outputs must be per-row vectors')
        bound_penalty = jnp.maximum(0.,mu+frozen['threshold']*sigma)
        compatibility = revised_compat.compatibility_cost_jax(actions,frozen['neighbors'],frozen['bandwidth'])
        compatibility = jnp.where(frozen['reliable'],compatibility,0.)
        penalty = bound_penalty + jnp.asarray(eta,q.dtype)*compatibility
        valid = (jnp.all(batch.obs == frozen['observations']) & jnp.all(ids == frozen['episode_ids'])
                 & jnp.all(frozen['neighbor_episode_ids'] != ids[:,None])
                 & jnp.all(jnp.isfinite(penalty)) & jnp.all(jnp.isfinite(mu))
                 & jnp.all(jnp.isfinite(sigma) & (sigma > 0)))
        value = -scale*q.mean()+bc.mean()+scale*penalty.mean()
        value = jnp.where(valid,value,jnp.asarray(jnp.nan,q.dtype))
        return value, {'q_mean':q.mean(), 'q_scale_coefficient':scale,
                       'bc_loss':bc.mean(), 'signed_penalty_mean':bound_penalty.mean(),
                       'compatibility_cost_mean':compatibility.mean(), 'binding_valid':valid}
    return loss
