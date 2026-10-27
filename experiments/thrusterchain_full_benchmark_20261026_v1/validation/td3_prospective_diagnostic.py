"""Prospective descriptive eager/scan probe; never used for study training."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from compiled.engine import _factory, _native_dataset, _scan
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions


def diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f'difference shape mismatch: {a.shape}, {b.shape}')
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise FloatingPointError('nonfinite value in diagnostic')
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'max_abs': float(d.max(initial=0)), 'mean_abs': float(d.mean()),
            'rms': float(np.sqrt(np.mean(d * d)))}


def state_diff(a, b):
    la = jax.tree_util.tree_flatten_with_path(a)[0]
    lb = jax.tree_util.tree_flatten_with_path(b)[0]
    if [jax.tree_util.keystr(x[0]) for x in la] != [jax.tree_util.keystr(x[0]) for x in lb]:
        raise ValueError('state leaf paths differ')
    rows = []
    for (pa, x), (_, y) in zip(la, lb):
        d = diff(x, y)
        rows.append({'path': jax.tree_util.keystr(pa), 'shape': list(np.shape(x)), **d})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    raw = args.protocol.read_bytes()
    p = json.loads(raw)
    if p['schema'] != 'thrusterchain.td3bc_numerics_diagnostic.v1':
        raise ValueError('unsupported protocol')
    if jax.default_backend() != 'gpu':
        raise RuntimeError('diagnostic requires GPU')
    results = {'protocol_sha256': hashlib.sha256(raw).hexdigest(), 'protocol': p,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'jax': jax.__version__, 'numpy': np.__version__,
               'device': str(jax.devices()[0]), 'fixtures': []}
    for data_seed, train_seed in p['fixture_pairs']:
        start = time.perf_counter()
        dataset = generate_behavior_dataset(behavior=p['behavior'], episodes=p['episodes_per_dataset'],
                                            seed=data_seed, provenance='smoke')
        batch = train_transitions(dataset)
        init_seed = int(_rng(train_seed, 'learner_initialization').integers(0, 2**63 - 1))
        eager, compiled = _factory('td3bc')(), _factory('td3bc')()
        eager.initialize(batch, seed=init_seed, learning_settings={})
        compiled.initialize(batch, seed=init_seed, learning_settings={})
        replay = _native_dataset(compiled, 'td3bc', batch)
        held = dataset.split(p['action_observation_split'])['observations']
        horizon = dataset.split(p['action_observation_split'])['actions'].shape[1]
        held = np.asarray(held[:, :horizon, :]).reshape(-1, batch.observations.shape[-1])[:p['action_observation_count']]
        if len(held) != p['action_observation_count']:
            raise ValueError('not enough validation observations')
        normalized = jnp.asarray((held - eager._mean) / eager._scale)
        @jax.jit
        def actions(params):
            return eager._actor.apply(params, normalized)
        def target(rng, state, transitions):
            _, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, transitions.action.shape) * eager._args.policy_noise
            noise = jnp.clip(noise, -eager._args.noise_clip, eager._args.noise_clip)
            next_action = eager._actor.apply(state.actor_target.params, transitions.next_obs)
            next_action = jnp.clip(next_action + noise, -1., 1.)
            q = eager._critic.apply(state.critic_target.params, transitions.next_obs, next_action)
            return transitions.reward + (1. - transitions.done) * eager._args.discount * jnp.min(q, axis=-1)
        target_compiled = jax.jit(target)
        sampler = _rng(train_seed, 'offline_batches')
        fixture = {'dataset_seed': data_seed, 'training_seed': train_seed, 'dataset_sha256': dataset.digest,
                   'validation_split_sha256': dataset.split_digest(p['action_observation_split']),
                   'steps': [], 'checkpoints': []}
        def checkpoint(step):
            ea, ca = np.asarray(actions(eager._state.actor.params)), np.asarray(actions(compiled._state.actor.params))
            fixture['checkpoints'].append({'step': step, 'state_leaves': state_diff(eager._state, compiled._state),
                                           'action_difference': diff(ea, ca),
                                           'eager_actions': ea.tolist(), 'compiled_actions': ca.tolist()})
        if 0 in p['checkpoints']:
            checkpoint(0)
        for step in range(1, p['critic_steps'] + 1):
            rows = sampler.integers(0, len(batch), size=(1, p['batch_size']))
            selected = jax.tree_util.tree_map(lambda a: a[rows[0]], replay)
            eager_target = np.asarray(target(eager._rng, eager._state, selected))
            same_state_target = np.asarray(target_compiled(eager._rng, eager._state, selected))
            path_target = np.asarray(target_compiled(compiled._rng, compiled._state, selected))
            em = eager.update(batch.select(rows[0]))
            rng, state, cm = _scan(compiled, 'td3bc', replay, rows)
            compiled._rng, compiled._state = rng, state
            matching_keys = bool(np.array_equal(np.asarray(eager._rng), np.asarray(rng)))
            ec = {key:int(value) for key,value in eager.optimizer_step_counts.items()}
            cc = {key:int(value) for key,value in compiled.optimizer_step_counts.items()}
            if not matching_keys or ec != cc:
                raise AssertionError('RNG or optimizer counters diverged')
            metrics = {k: {'eager':float(em[k]), 'compiled':float(cm[k]), 'absolute_difference':abs(float(em[k])-float(cm[k]))} for k in em}
            if not all(np.isfinite(v) for m in metrics.values() for v in m.values()):
                raise FloatingPointError('nonfinite metrics')
            fixture['steps'].append({'step':step, 'rng_equal':matching_keys, 'counters':ec,
                                     'same_state_target_difference':diff(eager_target,same_state_target),
                                     'trajectory_target_difference':diff(same_state_target,path_target),
                                     'metrics':metrics})
            if step in p['checkpoints']:
                checkpoint(step)
        fixture['wall_seconds'] = time.perf_counter() - start
        results['fixtures'].append(fixture)
        print(json.dumps({'completed_data_seed':data_seed,'wall_seconds':fixture['wall_seconds']}),flush=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
    print(json.dumps({'output':str(args.output),'protocol_sha256':results['protocol_sha256']}),flush=True)


if __name__ == '__main__':
    main()
