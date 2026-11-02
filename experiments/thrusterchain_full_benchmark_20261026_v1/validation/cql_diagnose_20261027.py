"""Bounded CQL eager/scan comparison, preserving the original tolerance."""
import json
import jax
import numpy as np
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import train_transitions
from compiled.engine import _factory, _native_dataset, _scan

data = generate_behavior_dataset(behavior='mixed_random_strong', episodes=10, seed=744, provenance='smoke')
batch = train_transitions(data)
rows = np.array([[0,99,102,199,2,2,1],[199,2,99,0,102,2,1]], dtype=np.int64)
eager, scanned = _factory('cql')(), _factory('cql')()
eager.initialize(batch, seed=20, learning_settings={})
scanned.initialize(batch, seed=20, learning_settings={})
native = _native_dataset(scanned,'cql',batch)
out = {'devices':list(map(str,jax.devices())), 'stages':[]}
for step, row in enumerate(rows, 1):
    em = eager.update(batch.select(row))
    key,state,sm = _scan(scanned,'cql',native,rows[:step])
    stage = {'step':step,'differences':[],'metric_differences':{},'rng_equal':bool(np.array_equal(eager._rng,key))}
    for (p,a),(q,b) in zip(jax.tree_util.tree_flatten_with_path(eager._state)[0],jax.tree_util.tree_flatten_with_path(state)[0]):
        assert jax.tree_util.keystr(p) == jax.tree_util.keystr(q)
        a,b=np.asarray(a),np.asarray(b)
        if a.dtype.kind not in 'fc':
            assert np.array_equal(a,b)
            continue
        d=np.abs(a-b)
        mask=~np.isclose(a,b,atol=1e-6,rtol=1e-5)
        if np.any(d):
            idx=np.unravel_index(np.argmax(d),d.shape)
            stage['differences'].append({'path':jax.tree_util.keystr(p),'max_abs':float(d[idx]),'index':[int(x) for x in idx],
                                         'eager':float(a[idx]),'scan':float(b[idx]),'failed':int(mask.sum())})
    stage['metric_differences']={k:float(abs(em[k]-sm[k])) for k in em if em[k]!=sm[k]}
    out['stages'].append(stage)
print(json.dumps(out,indent=2,allow_nan=False))
