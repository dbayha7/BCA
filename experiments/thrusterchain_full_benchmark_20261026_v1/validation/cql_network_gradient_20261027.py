"""Prospectively fixed full-network CQL finite differences; see protocol."""
import copy
import json
import jax
import jax.numpy as jnp
import numpy as np
from algorithms import cql
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import train_transitions
from compiled.engine import _factory, _native_dataset

data = generate_behavior_dataset(behavior='feedback_weak', episodes=10, seed=2026102703, provenance='smoke')
batch = train_transitions(data)
learner = _factory('cql')()
learner.initialize(batch, seed=2026102803, learning_settings={})
rows = np.array([11,29,70,99,145,199,225,299],dtype=np.int32)
native = jax.tree.map(lambda x: x[rows], _native_dataset(learner,'cql',batch))
assert np.array_equal(np.asarray(native.done), np.array([0,0,0,1,0,1,0,1],np.float32))
nextkey, updatekey = jax.random.split(learner._rng)
def update(state):
    return cql.cql_update(learner._args,learner._actor.apply,learner._critic1.apply,
                          learner._critic2.apply,state,native,1,updatekey,1.,-4.)
execute=jax.jit(update)
updated,metrics=execute(learner._state)
jax.block_until_ready(updated)
for name in ('actor','critic1','critic2','log_alpha','log_alpha_prime'):
    assert int(getattr(updated,name).step)==1,name
assert np.array_equal(np.asarray(nextkey),np.asarray(jax.random.split(learner._rng)[0]))
coords={
    'Dense_0':[(1,2),(3,4),(5,6),(7,8),(9,10)],
    'Dense_1':[(1,2),(23,34),(45,56),(67,78),(89,100)],
    'Dense_2':[(1,2),(23,34),(45,56),(67,78),(89,100)],
    'actor_Dense_3':[(1,0),(23,1),(45,2),(67,3),(89,4)],
    'critic1_Dense_3':[(1,0),(23,0),(45,0),(67,0),(89,0)],
}
records=[]
failures=[]
for modelname in ('actor','critic1'):
    initial=getattr(learner._state,modelname)
    new=getattr(updated,modelname)
    adam_mu=new.opt_state[0].mu['params']
    for layer in ('Dense_0','Dense_1','Dense_2','Dense_3'):
        selected=coords[modelname+'_'+layer] if layer=='Dense_3' else coords[layer]
        values=[]
        for idx in selected:
            estimated=float(np.asarray(adam_mu[layer]['kernel'][idx]))/.1
            result=[]
            for sign in (1,-1):
                params=copy.deepcopy(initial.params)
                params['params'][layer]['kernel']=params['params'][layer]['kernel'].at[idx].add(sign*.001)
                perturbed=learner._state._replace(**{modelname:initial.replace(params=params)})
                _,m=execute(perturbed)
                loss=m['policy_loss'] if modelname=='actor' else m['qf1_loss']+m['cql_min_qf1_loss']
                result.append(float(np.asarray(loss)))
            finite=(result[0]-result[1])/.002
            error=abs(estimated-finite)
            passed=bool(np.isclose(estimated,finite,atol=.003,rtol=.03))
            item={'model':modelname,'layer':layer,'index':list(idx),'adam_gradient':estimated,
                  'finite_difference':finite,'absolute_error':error,'pass':passed}
            records.append(item); values.append(abs(estimated))
            if not passed: failures.append(item)
        if max(values)<=1e-5: failures.append({'model':modelname,'layer':layer,'reason':'no active selected coordinates'})
output={'jax':jax.__version__,'devices':list(map(str,jax.devices())),'rows':rows.tolist(),
        'terminal':np.asarray(native.done).tolist(),'checks':records,'failures':failures,
        'status':'pass' if not failures else 'fail'}
print(json.dumps(output,indent=2,allow_nan=False))
if failures: raise SystemExit(1)
