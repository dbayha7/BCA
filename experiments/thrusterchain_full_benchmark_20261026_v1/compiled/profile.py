"""Short synchronized GPU throughput probe. This is not a study run."""
import argparse
import json
import sys
import time

import jax
import numpy as np
from compiled.engine import METHODS, _PROGRAMS, _engine_sha256, train_chunk
from thrusterchain.quality_datasets import generate_behavior_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    ap.add_argument('--steps', type=int, default=64)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()
    if args.steps < 1 or args.steps > 256:
        raise ValueError('probe must use 1 to 256 primary steps')
    data = generate_behavior_dataset(behavior='mixed_random_strong', episodes=10, seed=733,
                                     provenance='smoke')
    out = {'engine_sha256': _engine_sha256(), 'jax': jax.__version__, 'numpy': np.__version__,
           'device': str(jax.devices()[0]), 'dataset_sha256': data.digest, 'episodes': 10,
           'batch_size': args.batch_size, 'primary_steps_per_chunk': args.steps, 'methods': {}}
    for name in METHODS:
        times = []
        cp = None
        for unused in range(2):
            t0 = time.perf_counter()
            cp, receipt = train_chunk(data, name, 834, args.batch_size, args.steps, cp)
            times.append(time.perf_counter() - t0)
        cache_sizes = [fn._cache_size() for key, (fn, template) in _PROGRAMS.items() if key[0] == name]
        out['methods'][name] = {'first_chunk_seconds_including_compile': times[0],
                                'resumed_chunk_seconds': times[1],
                                'resumed_seconds_per_primary_step': times[1] / args.steps,
                                'optimizer_step_deltas_second_chunk': receipt['optimizer_step_deltas'],
                                'compiled_cache_sizes': cache_sizes}
        print(name, out['methods'][name], file=sys.stderr, flush=True)
    with open(args.output, 'w', encoding='utf-8') as fp:
        json.dump(out, fp, indent=2, allow_nan=False)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
