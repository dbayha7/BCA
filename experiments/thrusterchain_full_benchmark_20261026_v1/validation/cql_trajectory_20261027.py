"""Prospective descriptive eager/scan probe; never used for study training."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from algorithms import cql
from compiled.engine import _factory, _native_dataset, _scan
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions

def diff(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape: raise ValueError('shape mismatch')
    if not (np.isfinite(a).all() and np.isfinite(b).all()): raise FloatingPointError('nonfinite value')
    d=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'max_abs':float(d.max(initial=0)),'mean_abs':float(d.mean()),'rms':float(np.sqrt(np.mean(d*d)))}

def state_diff(a,b):
    left=jax.tree_util.tree_flatten_with_path(a)[0]; right=jax.tree_util.tree_flatten_with_path(b)[0]
    if [jax.tree_util.keystr(p) for p,_ in left]!=[jax.tree_util.keystr(p) for p,_ in right]:
        raise ValueError('state leaf paths differ')
    return [{'path':jax.tree_util.keystr(p),'shape':list(np.shape(x)),**diff(x,y)}
            for (p,x),(_,y) in zip(left,right)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); raw=args.protocol.read_bytes(); p=json.loads(raw)
    if p['schema']!='thrusterchain.cql_numerics_diagnostic.v1': raise ValueError('unsupported protocol')
    if jax.default_backend()!='gpu': raise RuntimeError('diagnostic requires GPU')
    results={'protocol_sha256':hashlib.sha256(raw).hexdigest(),'protocol':p,
             'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'jax':jax.__version__,'numpy':np.__version__,'device':str(jax.devices()[0]),'fixtures':[]}
    for data_seed,train_seed in p['fixture_pairs']:
        start=time.perf_counter()
        dataset=generate_behavior_dataset(behavior=p['behavior'],episodes=p['episodes_per_dataset'],seed=data_seed,provenance='smoke')
        batch=train_transitions(dataset)
        init_seed=int(_rng(train_seed,'learner_initialization').integers(0,2**63-1))
        eager,compiled=_factory('cql')(),_factory('cql')()
        eager.initialize(batch,seed=init_seed,learning_settings={}); compiled.initialize(batch,seed=init_seed,learning_settings={})
        if eager._args.cql_max_target_backup or eager._args.backup_entropy: raise ValueError('unsupported target branch')
        replay=_native_dataset(compiled,'cql',batch)
        split=dataset.split(p['action_observation_split'])
        horizon=split['actions'].shape[1]
        held=np.asarray(split['observations'][:,:horizon,:]).reshape(-1,batch.observations.shape[-1])[:p['action_observation_count']]
        if len(held)!=p['action_observation_count']: raise ValueError('not enough validation observations')
        normalized=jnp.asarray((held-eager._mean)/eager._scale)
        @jax.jit
        def actions(params):
            mean,_=eager._actor.apply(params,normalized)
            return jnp.tanh(mean)
        def target(rng,state,transitions):
            _,updatekey=jax.random.split(rng)
            _,_,nextkey,_,_,_=jax.random.split(updatekey,6)
            noise=jax.random.normal(nextkey,transitions.action.shape)
            mean,logstd=eager._actor.apply(state.actor.params,transitions.next_obs)
            next_action,_=cql.tanh_gaussian_sample(mean,logstd,noise,1.)
            q1=eager._critic1.apply(state.critic1_target.params,transitions.next_obs,next_action)
            q2=eager._critic2.apply(state.critic2_target.params,transitions.next_obs,next_action)
            return transitions.reward+(1.-transitions.done)*eager._args.discount*jnp.minimum(q1,q2)
        target_compiled=jax.jit(target)
        sampler=_rng(train_seed,'offline_batches')
        fixture={'dataset_seed':data_seed,'training_seed':train_seed,'dataset_sha256':dataset.digest,
                 'validation_split_sha256':dataset.split_digest(p['action_observation_split']),'steps':[],'checkpoints':[]}
        def checkpoint(step):
            ea,ca=np.asarray(actions(eager._state.actor.params)),np.asarray(actions(compiled._state.actor.params))
            fixture['checkpoints'].append({'step':step,'state_leaves':state_diff(eager._state,compiled._state),
                'action_difference':diff(ea,ca),'eager_actions':ea.tolist(),'compiled_actions':ca.tolist()})
        checkpoint(0)
        for step in range(1,p['critic_steps']+1):
            rows=sampler.integers(0,len(batch),size=(1,p['batch_size']))
            selected=jax.tree.map(lambda a:a[rows[0]],replay)
            eager_target=np.asarray(target(eager._rng,eager._state,selected))
            same_target=np.asarray(target_compiled(eager._rng,eager._state,selected))
            path_target=np.asarray(target_compiled(compiled._rng,compiled._state,selected))
            em=eager.update(batch.select(rows[0]))
            rng,state,cm=_scan(compiled,'cql',replay,rows)
            compiled._rng,compiled._state=rng,state
            samekey=bool(np.array_equal(np.asarray(eager._rng),np.asarray(rng)))
            ec={k:int(v) for k,v in eager.optimizer_step_counts.items()}; cc={k:int(v) for k,v in compiled.optimizer_step_counts.items()}
            if not samekey or ec!=cc: raise AssertionError('RNG or optimizer counters diverged')
            metrics={k:{'eager':float(em[k]),'compiled':float(cm[k]),'absolute_difference':abs(float(em[k])-float(cm[k]))} for k in em}
            if not all(np.isfinite(v) for m in metrics.values() for v in m.values()): raise FloatingPointError('nonfinite metrics')
            fixture['steps'].append({'step':step,'rng_equal':samekey,'counters':ec,
                'same_state_target_difference':diff(eager_target,same_target),
                'trajectory_target_difference':diff(same_target,path_target),'metrics':metrics})
            if step in p['checkpoints']: checkpoint(step)
        fixture['wall_seconds']=time.perf_counter()-start; results['fixtures'].append(fixture)
        print(json.dumps({'completed_data_seed':data_seed,'wall_seconds':fixture['wall_seconds']}),flush=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(results,stream,indent=2,allow_nan=False)
    print(json.dumps({'output':str(args.output),'protocol_sha256':results['protocol_sha256']}),flush=True)

if __name__=='__main__': main()
