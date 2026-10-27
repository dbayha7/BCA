"""Bounded CQL scan variants, no source changes or tolerance adjustments."""
import json
from contextlib import contextmanager
import jax
import jax.numpy as jnp
import numpy as np
from algorithms import cql
from compiled.engine import _factory,_native_dataset
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import train_transitions

@contextmanager
def patches(names):
    normal,uniform,sample=jax.random.normal,jax.random.uniform,cql.tanh_gaussian_sample
    try:
        if 'random' in names:
            def n(*a,**k): return jax.lax.optimization_barrier(normal(*a,**k))
            def u(*a,**k): return jax.lax.optimization_barrier(uniform(*a,**k))
            jax.random.normal,jax.random.uniform=n,u
        if 'sample' in names:
            def s(*a,**k): return jax.tree.map(jax.lax.optimization_barrier,sample(*a,**k))
            cql.tanh_gaussian_sample=s
        yield
    finally:
        jax.random.normal,jax.random.uniform,cql.tanh_gaussian_sample=normal,uniform,sample

data=generate_behavior_dataset(behavior='mixed_random_strong',episodes=10,seed=744,provenance='smoke')
batch=train_transitions(data)
rows=np.array([[0,99,102,199,2,2,1],[199,2,99,0,102,2,1]],np.int32)
eager=_factory('cql')(); eager.initialize(batch,seed=20,learning_settings={})
for row in rows: eager.update(batch.select(row))
results=[]
for name,flags in [('baseline',()),('random',('random',)),('sample',('sample',)),
                   ('networks',('networks',)),('all',('random','sample','networks'))]:
    model=_factory('cql')(); model.initialize(batch,seed=20,learning_settings={})
    replay=_native_dataset(model,'cql',batch)
    def actor(*a):
        x=model._actor.apply(*a)
        return jax.tree.map(jax.lax.optimization_barrier,x) if 'networks' in flags else x
    def critic1(*a):
        x=model._critic1.apply(*a)
        return jax.lax.optimization_barrier(x) if 'networks' in flags else x
    def critic2(*a):
        x=model._critic2.apply(*a)
        return jax.lax.optimization_barrier(x) if 'networks' in flags else x
    def execute(rng,state,replay,indices):
        def body(carry,row):
            key,state=carry; key,subkey=jax.random.split(key)
            selected=jax.tree.map(lambda x:x[row],replay)
            state,metrics=cql.cql_update(model._args,actor,critic1,critic2,state,selected,state.critic1.step+1,subkey,1.,-4.)
            return (key,state),metrics
        return jax.lax.scan(body,(rng,state),indices)
    with patches(flags):
        (key,state),metrics=jax.jit(execute)(model._rng,model._state,replay,jnp.asarray(rows))
        jax.block_until_ready(state)
    fail=[]; max_abs=0.
    la=jax.tree_util.tree_flatten_with_path(eager._state)[0]; lb=jax.tree_util.tree_flatten_with_path(state)[0]
    for (p,x),(q,y) in zip(la,lb):
        assert jax.tree_util.keystr(p)==jax.tree_util.keystr(q)
        x,y=np.asarray(x),np.asarray(y)
        if x.dtype.kind in 'biu': assert np.array_equal(x,y); continue
        d=np.abs(x-y); max_abs=max(max_abs,float(d.max(initial=0)))
        nfail=int((~np.isclose(x,y,atol=1e-6,rtol=1e-5)).sum())
        if nfail: fail.append({'path':jax.tree_util.keystr(p),'n':nfail,'max_abs':float(d.max())})
    result={'variant':name,'rng_equal':bool(np.array_equal(eager._rng,key)),'max_abs':max_abs,'failing_leaves':fail}
    results.append(result); print(json.dumps(result),flush=True)
print(json.dumps({'jax':jax.__version__,'device':str(jax.devices()[0]),'results':results},indent=2),flush=True)
