"""Bounded current-runner profiling. No study outputs or checkpoints are written."""
import argparse
import importlib
import json
import time

import jax
import numpy as np

from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _counters, _rng, _update, train_transitions

FACTORIES = {
    "td3bc": ("host_integration.td3bc_adapter", "TD3BCAdapter"),
    "iql": ("host_integration.iql_adapter", "IQLAdapter"),
    "cql": ("host_integration.cql_adapter", "CQLAdapter"),
    "rebrac": ("host_integration.rebrac_adapter", "ReBRACAdapter"),
    "unified": ("host_integration.unifloral_adapter", "UnifiedTD3BCAdapter"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=FACTORIES)
    args = parser.parse_args()
    if jax.default_backend() != "gpu" or str(jax.config.jax_default_matmul_precision) != "highest":
        raise RuntimeError("profiling requires GPU backend and highest matmul precision")
    module, name = FACTORIES[args.method]
    factory = getattr(importlib.import_module(module), name)
    dataset = generate_behavior_dataset(behavior="feedback_weak", episodes=10, seed=331,
                                        provenance="smoke")
    batch = train_transitions(dataset)
    init_seed = int(_rng(332, "learner_initialization").integers(0, 2**63 - 1))
    t0 = time.perf_counter()
    model = factory()
    model.initialize(batch, seed=init_seed, learning_settings={})
    init_s = time.perf_counter() - t0
    counts = _counters(model)
    rng = np.random.default_rng(333)
    latencies = []
    for _ in range(8):
        t0 = time.perf_counter()
        picked = batch.select(rng.integers(0, len(batch), size=256))
        counts, metrics = _update(model, picked, counts)
        latencies.append(time.perf_counter() - t0)
    print(json.dumps({"method": args.method, "device": str(jax.devices()[0]),
        "jax": jax.__version__, "numpy": np.__version__,
        "matmul_precision": str(jax.config.jax_default_matmul_precision),
        "dataset_hash": dataset.digest, "dataset_episodes": 10, "batch_size": 256,
        "source": "v6 NumPy-1 compatibility copy; actual adapter and runner counter path",
        "primary_optimizer": model.capabilities.primary_optimizer,
        "optimizer_counts_after_8": counts.optimizer_step_counts, "initialization_seconds": init_s,
        "warmup_step_seconds": latencies[:3], "timed_step_seconds": latencies[3:],
        "timed_mean_seconds_per_primary_update": float(np.mean(latencies[3:])),
        "scope": "8 bounded primary updates, includes minibatch sampling/selection, transfer, sync and counters; no checkpoint/evaluation, no training study"}, allow_nan=False))


if __name__ == "__main__":
    main()
