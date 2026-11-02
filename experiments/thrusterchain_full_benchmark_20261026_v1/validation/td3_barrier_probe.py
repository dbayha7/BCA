"""Bounded exploratory eager-boundary lowering; not an experiment runner."""
import json
import time
import jax
import numpy as np
from algorithms import td3_bc
from compiled.diagnose_parity import compare
from compiled.engine import _factory, _native_dataset
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions

data = generate_behavior_dataset(behavior='mixed_random_strong', episodes=10, seed=733, provenance='smoke')
batch = train_transitions(data)
init_seed = int(_rng(834, 'learner_initialization').integers(0, 2**63 - 1))
eager, alternative = _factory('td3bc')(), _factory('td3bc')()
eager.initialize(batch, seed=init_seed, learning_settings={})
alternative.initialize(batch, seed=init_seed, learning_settings={})
native = _native_dataset(alternative, 'td3bc', batch)
def update(rng, state, selected):
    rng, noise = jax.random.split(rng)
    state, metrics = td3_bc.td3_bc_update(alternative._args, alternative._actor.apply, alternative._critic.apply,
                                           state, selected, state.critic.step + 1, noise, 1.)
    return rng, state, metrics

selected = jax.tree_util.tree_map(lambda x: x[np.zeros(8, dtype=np.int32)], native)
initial = (alternative._rng, alternative._state, selected)
closed = jax.make_jaxpr(update)(*initial)
output_tree = jax.tree_util.tree_structure(jax.eval_shape(update, *initial))

def barrier_interpret(*args):
    flat, _ = jax.tree_util.tree_flatten(args)
    env = dict(zip(closed.jaxpr.constvars, closed.consts))
    env.update(zip(closed.jaxpr.invars, flat))
    def read(v):
        return v.val if isinstance(v, jax.core.Literal) else env[v]
    for eqn in closed.jaxpr.eqns:
        try:
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            value = eqn.primitive.bind(*subfuns, *(read(v) for v in eqn.invars), **params)
        except Exception:
            print('primitive failed:', eqn.primitive, 'params:', eqn.params, flush=True)
            raise
        values = list(value) if eqn.primitive.multiple_results else [value]
        values = jax.lax.optimization_barrier(values)
        for v, x in zip(eqn.outvars, values):
            if not isinstance(v, jax.core.DropVar): env[v] = x
    return jax.tree_util.tree_unflatten(output_tree, [read(v) for v in closed.jaxpr.outvars])

fn = jax.jit(barrier_interpret)
sampler = _rng(834, 'offline_batches')
records = []
for step in range(4):
    rows = sampler.integers(0, len(batch), size=8)
    before = time.perf_counter()
    em = eager.update(batch.select(rows))
    t_eager = time.perf_counter() - before
    selected = jax.tree_util.tree_map(lambda x: x[rows], native)
    before = time.perf_counter()
    rng, state, metrics = fn(alternative._rng, alternative._state, selected)
    jax.block_until_ready((rng, state, metrics))
    t_compiled = time.perf_counter() - before
    alternative._rng, alternative._state = rng, state
    records.append({'step':step+1, 'mismatches':compare(eager._state, state),
                    'rng_equal':bool(np.array_equal(np.asarray(eager._rng), np.asarray(rng))),
                    'eager_seconds':t_eager, 'barrier_seconds':t_compiled,
                    'metric_differences':{k:float(abs(em[k]-float(metrics[k]))) for k in em}})
    print(json.dumps(records[-1]), flush=True)
print(json.dumps({'eqns':len(closed.jaxpr.eqns), 'records':records},indent=2),flush=True)
