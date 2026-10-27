"""Exploratory eager CQL with reusable native Polyak cond branches, CPU only."""
import argparse
import inspect
import json
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from algorithms import cql as host
from runtime.networks import Transition
from compiled.diagnose_parity import compare
from compiled.engine import _factory
from host_integration import td3bc_adapter as helpers
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng,train_transitions

def make_update(model):
    # Preserve every host line except hoisting the two cond branch closures.
    # Inputs/target timing/optimizer operations are unchanged.
    rate=model._args.soft_target_update_rate
    def soft(src,tgt):
        return tgt.replace(step=tgt.step+1,params=optax.incremental_update(src.params,tgt.params,rate))
    def cached_do(state):
        return state._replace(critic1_target=soft(state.critic1,state.critic1_target),
                              critic2_target=soft(state.critic2,state.critic2_target))
    def cached_skip(state): return state
    source=inspect.getsource(host.cql_update)
    assert source.count('    def _soft(src, tgt):')==1
    i=source.index('    def _soft(src, tgt):'); j=source.index('    agent_state = jax.lax.cond(',i)
    source=source[:i]+source[j:]
    old='it % args.target_update_period == 0, _do_polyak, lambda s: s, agent_state'
    new='it % args.target_update_period == 0, _cached_do, _cached_skip, agent_state'
    assert source.count(old)==1
    source=source.replace(old,new)
    ns=dict(host.__dict__); ns.update(_cached_do=cached_do,_cached_skip=cached_skip)
    exec(compile(source,'<cql-polyak-closure-probe>','exec'),ns)
    return ns['cql_update']

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,required=True); ap.add_argument('--batch-size',type=int,required=True); ap.add_argument('--output',required=True)
    args=ap.parse_args()
    if jax.default_backend()!='cpu': raise RuntimeError('CPU only')
    if not 1<=args.steps<=16 or not 1<=args.batch_size<=256: raise ValueError('bounded diagnostic')
    records=[]
    for data_seed,train_seed in ((733,834),(744,845)):
        data=generate_behavior_dataset(behavior='mixed_random_strong',episodes=10,seed=data_seed,provenance='smoke')
        batch=train_transitions(data); seed=int(_rng(train_seed,'learner_initialization').integers(0,2**63-1))
        native,cached=_factory('cql')(),_factory('cql')()
        native.initialize(batch,seed=seed,learning_settings={}); cached.initialize(batch,seed=seed,learning_settings={})
        update=make_update(cached); sampler=_rng(train_seed,'offline_batches'); steps=[]
        for i in range(args.steps):
            indices=sampler.integers(0,len(batch),size=args.batch_size)
            begin=time.perf_counter(); em=native.update(batch.select(indices)); et=time.perf_counter()-begin
            begin=time.perf_counter()
            selected=batch.select(indices); obs,nxt,action,reward=helpers._arrays(selected)
            transition=Transition(jnp.asarray((obs-cached._mean)/cached._scale),jnp.asarray(action),jnp.asarray(reward),
                                  jnp.asarray((nxt-cached._mean)/cached._scale),jnp.asarray(selected.terminated,jnp.float32))
            rng,key=jax.random.split(cached._rng)
            state,cm=update(cached._args,cached._actor.apply,cached._critic1.apply,cached._critic2.apply,
                            cached._state,transition,int(cached._state.critic1.step)+1,key,1.,-4.)
            if not helpers._finite_tree((state,cm,rng)): raise FloatingPointError('nonfinite update')
            cm={k:float(v) for k,v in cm.items()}; cached._state,cached._rng=state,rng
            ct=time.perf_counter()-begin
            metric_mismatch={k:{'native':em[k],'cached':cm[k]} for k in em if not np.isclose(em[k],cm[k],atol=1e-6,rtol=1e-5)}
            counts_equal=native.optimizer_step_counts==cached.optimizer_step_counts
            steps.append({'step':i+1,'strict_mismatches':compare(native._state,state),
                          'rng_equal':bool(np.array_equal(native._rng,rng)),'counts_equal':counts_equal,
                          'metric_mismatch':metric_mismatch,'native_seconds':et,'cached_seconds':ct})
        records.append({'data_seed':data_seed,'training_seed':train_seed,'steps':steps})
        print(json.dumps({'data_seed':data_seed,'steps':len(steps),'mismatched_steps':sum(bool(x['strict_mismatches'] or x['metric_mismatch'] or not x['rng_equal'] or not x['counts_equal']) for x in steps)}),flush=True)
    result={'jax':jax.__version__,'device':str(jax.devices()[0]),'batch_size':args.batch_size,'steps':args.steps,'fixtures':records}
    with open(args.output,'x',encoding='utf-8') as fp: json.dump(result,fp,indent=2,allow_nan=False)

if __name__=='__main__': main()
