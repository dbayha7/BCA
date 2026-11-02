"""Bounded eager/scan discrepancy evidence; never used for study training."""
import argparse
import json
import math

import jax
import numpy as np
from compiled.engine import _factory, _native_dataset, _scan
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions


def compare(left, right):
    la, _ = jax.tree_util.tree_flatten_with_path(left)
    lb, _ = jax.tree_util.tree_flatten_with_path(right)
    rows = []
    for (pa, a), (pb, b) in zip(la, lb):
        if jax.tree_util.keystr(pa) != jax.tree_util.keystr(pb):
            raise RuntimeError('state paths differ')
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise RuntimeError('state shapes differ')
        if a.dtype.kind in 'biu':
            if a.dtype != b.dtype or not np.array_equal(a, b):
                rows.append({'path': jax.tree_util.keystr(pa), 'dtype_eager': str(a.dtype),
                             'dtype_scan': str(b.dtype), 'values_equal': bool(np.array_equal(a,b)),
                             'eager': a.tolist(), 'scan': b.tolist()})
            continue
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        bad = delta > 1e-6 + 1e-5 * np.abs(b.astype(np.float64))
        if np.any(bad):
            ind = np.unravel_index(np.argmax(delta), delta.shape)
            denom = max(abs(float(b[ind])), np.finfo(np.float64).tiny)
            rows.append({'path': jax.tree_util.keystr(pa), 'shape': list(a.shape),
                         'dtype_eager': str(a.dtype), 'dtype_scan': str(b.dtype),
                         'bad_elements': int(bad.sum()), 'size': int(a.size),
                         'max_abs': float(delta[ind]), 'eager_at_max': float(a[ind]),
                         'scan_at_max': float(b[ind]), 'relative_at_max': float(delta[ind]/denom),
                         'max_abs_eager': float(np.max(np.abs(a)))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--method', default='td3bc')
    args = ap.parse_args()
    results = {'method': args.method, 'jax_version': jax.__version__,
               'device': str(jax.devices()[0]), 'tolerance': {'atol': 1e-6, 'rtol': 1e-5},
               'fixtures': []}
    for dataset_seed, run_seed in ((733, 834), (744, 845)):
        dataset = generate_behavior_dataset(behavior='mixed_random_strong', episodes=10,
                                            seed=dataset_seed, provenance='smoke')
        batch = train_transitions(dataset)
        init_seed = int(_rng(run_seed, 'learner_initialization').integers(0, 2**63 - 1))
        eager, scan = _factory(args.method)(), _factory(args.method)()
        eager.initialize(batch, seed=init_seed, learning_settings={})
        scan.initialize(batch, seed=init_seed, learning_settings={})
        replay = _native_dataset(scan, args.method, batch)
        sampler = _rng(run_seed, 'offline_batches')
        four_scan = _factory(args.method)()
        four_scan.initialize(batch, seed=init_seed, learning_settings={})
        steps, all_indices = [], []
        for step in range(4):
            indices = sampler.integers(0, len(batch), size=(1, 8))
            all_indices.append(indices)
            em = eager.update(batch.select(indices[0]))
            rng, state, sm = _scan(scan, args.method, replay, indices)
            scan._rng, scan._state = rng, state
            steps.append({'step': step + 1, 'mismatches': compare(eager._state, state),
                          'rng_equal': bool(np.array_equal(np.asarray(eager._rng), np.asarray(rng))),
                          'metrics': {key: {'eager': float(em[key]), 'scan': float(sm[key])}
                                      for key in em if not np.isclose(em[key], sm[key], atol=1e-6, rtol=1e-5)}})
        four_rng, four_state, _ = _scan(four_scan, args.method,
                                       _native_dataset(four_scan, args.method, batch),
                                       np.concatenate(all_indices, axis=0))
        results['fixtures'].append({'dataset_seed': dataset_seed, 'run_seed': run_seed,
                                    'dataset_sha256': dataset.digest, 'steps': steps,
                                    'four_step_eager_mismatches': compare(eager._state, four_state),
                                    'four_step_rng_equal': bool(np.array_equal(np.asarray(eager._rng), np.asarray(four_rng)))})
    with open(args.output, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2, allow_nan=False)
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
