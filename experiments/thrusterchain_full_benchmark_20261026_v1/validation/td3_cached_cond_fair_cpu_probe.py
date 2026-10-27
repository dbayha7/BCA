"""Bounded TD3+BC host update probe: reuse the delayed-update branch closures.

Exploratory only. This is a direct structural transcription of the native host
update, keeping all operations and optimizer state; no production imports use it.
"""
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from algorithms import td3_bc as host
from compiled.diagnose_parity import compare
from compiled.engine import _factory, _native_dataset
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions


def make_update(model):
    args, actor_apply_fn, critic_apply_fn = model._args, model._actor.apply, model._critic.apply
    def _do_actor_update(operand):
        state, batch = operand
        def _actor_loss_fn(params, critic_params):
            pi = actor_apply_fn(params, batch.obs)
            q = critic_apply_fn(critic_params, batch.obs, pi)[..., 0]
            lmbda = args.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
            bc = jnp.square(pi - batch.action).mean()
            return (-lmbda * q.mean() + bc, (q.mean(), lmbda, bc))
        grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        (a_loss, aux), a_grad = grad_fn(state.actor.params, state.critic.params)
        state = state._replace(actor=state.actor.apply_gradients(grads=a_grad))
        def _soft(src, tgt):
            return tgt.replace(step=tgt.step + 1, params=optax.incremental_update(src.params, tgt.params, args.tau))
        state = state._replace(actor_target=_soft(state.actor, state.actor_target),
                               critic_target=_soft(state.critic, state.critic_target))
        return (state, (a_loss, *aux))
    def _skip_actor_update(operand):
        state, _ = operand
        zero = jnp.float32(0.0)
        return (state, (zero, zero, zero, zero))
    def update(agent_state, batch, it, rng_noise):
        noise = jax.random.normal(rng_noise, batch.action.shape) * args.policy_noise
        noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
        next_action = actor_apply_fn(agent_state.actor_target.params, batch.next_obs)
        next_action = jnp.clip(next_action + noise, -1., 1.)
        target_q = critic_apply_fn(agent_state.critic_target.params, batch.next_obs, next_action)
        target_q = jnp.min(target_q, axis=-1)
        target = batch.reward + (1.0 - batch.done) * args.discount * target_q
        target = jax.lax.stop_gradient(target)
        def _critic_loss_fn(params):
            q = critic_apply_fn(params, batch.obs, batch.action)
            return jnp.square(q - target[:, None]).mean(axis=0).sum()
        critic_loss, critic_grad = jax.value_and_grad(_critic_loss_fn)(agent_state.critic.params)
        agent_state = agent_state._replace(critic=agent_state.critic.apply_gradients(grads=critic_grad))
        agent_state, (actor_loss, q_mean, lmbda, bc_loss) = jax.lax.cond(
            it % args.policy_freq == 0, _do_actor_update, _skip_actor_update, (agent_state, batch))
        return agent_state, {'critic_loss':critic_loss, 'actor_loss':actor_loss,
                             'q_mean':q_mean, 'lambda':lmbda, 'bc_loss':bc_loss}
    return update


records = []
for data_seed, train_seed in ((733,834),(744,845)):
    dataset = generate_behavior_dataset(behavior='mixed_random_strong', episodes=10, seed=data_seed, provenance='smoke')
    batch = train_transitions(dataset)
    seed = int(_rng(train_seed,'learner_initialization').integers(0, 2**63-1))
    eager, optimized = _factory('td3bc')(), _factory('td3bc')()
    eager.initialize(batch, seed=seed, learning_settings={})
    optimized.initialize(batch, seed=seed, learning_settings={})
    replay = _native_dataset(optimized,'td3bc',batch)
    update = make_update(optimized)
    sampler = _rng(train_seed,'offline_batches')
    steps = []
    for i in range(4):
        indices = sampler.integers(0,len(batch),size=8)
        selected = jax.tree_util.tree_map(lambda x:x[indices],replay)
        begin=time.perf_counter()
        em=eager.update(batch.select(indices))
        elapsed_eager=time.perf_counter()-begin
        begin=time.perf_counter()
        rng,noise=jax.random.split(optimized._rng)
        state,cm=update(optimized._state,selected,int(optimized._state.critic.step)+1,noise)
        jax.block_until_ready((state,cm))
        elapsed_optimized=None
        optimized._rng,optimized._state=rng,state
        all_finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((state,cm)))
        if not all_finite: raise FloatingPointError('nonfinite')
        cm = {k: float(v) for k, v in cm.items()}
        elapsed_optimized = time.perf_counter() - begin
        steps.append({'step':i+1,'strict_mismatches':compare(eager._state,state),
                      'rng_equal':bool(np.array_equal(np.asarray(eager._rng),np.asarray(rng))),
                      'metrics':{k:{'eager':em[k],'optimized':float(cm[k])} for k in em},
                      'eager_seconds':elapsed_eager,'optimized_seconds':elapsed_optimized})
    records.append({'data_seed':data_seed,'training_seed':train_seed,'steps':steps})
    print(json.dumps(records[-1]),flush=True)
with open('validation/td3_cached_cond_fair_cpu.json','x',encoding='utf-8') as fp:
    json.dump({'jax':jax.__version__,'device':str(jax.devices()[0]),'fixtures':records},fp,indent=2,allow_nan=False)
