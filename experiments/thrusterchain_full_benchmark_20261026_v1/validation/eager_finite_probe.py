"""Bounded proof-of-concept: device-side finite checks, native eager update.

This file is NOT a training engine and does not modify any adapter or study.
It locally replaces the read-only finite predicate during a tiny comparison.
"""
import argparse
import json
import time
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from host_integration import td3bc_adapter as adapter
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_runner import _rng, train_transitions


@jax.jit
def device_all_finite(*leaves):
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))


def device_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return True if not leaves else bool(np.asarray(device_all_finite(*leaves)))


def identical_tree(a, b):
    aa, _ = jax.tree_util.tree_flatten_with_path(a)
    bb, _ = jax.tree_util.tree_flatten_with_path(b)
    return ([jax.tree_util.keystr(p) for p, _ in aa] == [jax.tree_util.keystr(p) for p, _ in bb]
            and all(np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)
                    for (_, x), (_, y) in zip(aa, bb)))


def make_model(batch):
    model = adapter.TD3BCAdapter()
    model.initialize(batch, seed=19, learning_settings={})
    return model


def check_rollback(model, batch, where, predicate):
    original_update = adapter.host.td3_bc_update
    before = model.checkpoint().to_bytes()
    before_rng = np.asarray(model._rng).copy()

    def corrupted(*args, **kwargs):
        state, metrics = original_update(*args, **kwargs)
        if where == "state":
            params = jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), state.actor.params)
            state = state._replace(actor=state.actor.replace(params=params))
        else:
            metrics = dict(metrics)
            metrics[next(iter(metrics))] = jnp.asarray(jnp.nan, dtype=jnp.float32)
        return state, metrics

    raised = False
    with patch.object(adapter, "_finite_tree", predicate), patch.object(adapter.host, "td3_bc_update", corrupted):
        try:
            model.update(batch)
        except FloatingPointError:
            raised = True
    return {"raised": raised, "checkpoint_unchanged": before == model.checkpoint().to_bytes(),
            "rng_unchanged": bool(np.array_equal(before_rng, np.asarray(model._rng)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--timed-steps", type=int, default=12)
    args = parser.parse_args()
    if jax.default_backend() != "gpu" or str(jax.config.jax_default_matmul_precision) != "highest":
        raise RuntimeError("probe requires CUDA and highest matmul precision")
    dataset = generate_behavior_dataset(behavior="mixed_random_strong", episodes=10,
                                        seed=733, provenance="smoke")
    batch = train_transitions(dataset)
    sampler = _rng(834, "offline_batches")
    rows = sampler.integers(0, len(batch), size=(4, 256))
    reference, candidate = make_model(batch), make_model(batch)
    original = adapter._finite_tree
    comparisons = []
    for row in rows:
        selected = batch.select(row)
        reference_metrics = reference.update(selected)
        with patch.object(adapter, "_finite_tree", device_finite):
            actual_metrics = candidate.update(selected)
        comparisons.append({"state_bitwise_equal": identical_tree(reference._state, candidate._state),
            "rng_equal": bool(np.array_equal(np.asarray(reference._rng), np.asarray(candidate._rng))),
            "metrics_equal": reference_metrics == actual_metrics,
            "counters_equal": reference.optimizer_step_counts == candidate.optimizer_step_counts})
    selected = batch.select(rows[-1])
    rollback = {name: {where: check_rollback(make_model(batch), selected, where, predicate)
                       for where in ("state", "metrics")}
                for name, predicate in (("original", original), ("device", device_finite))}
    # Time alternating native eager updates on separately initialized learners,
    # including adapter input processing and finite checks; no checkpoints/eval.
    # The first four parity steps are warmup; output includes every timing.
    timings = {"original": [], "device": []}
    for step in range(args.timed_steps):
        selected = batch.select(sampler.integers(0, len(batch), size=256))
        for name, model, predicate in (("original", reference, original),
                                       ("device", candidate, device_finite)):
            t0 = time.perf_counter()
            with patch.object(adapter, "_finite_tree", predicate):
                model.update(selected)
            timings[name].append(time.perf_counter() - t0)
    result = {"device": str(jax.devices()[0]), "jax": jax.__version__, "numpy": np.__version__,
        "precision": str(jax.config.jax_default_matmul_precision), "dataset_sha256": dataset.digest,
        "batch_size": 256, "parity_steps": comparisons, "nonfinite_rollback": rollback,
        "timed_steps_each": args.timed_steps, "timings_seconds": timings,
        "mean_seconds": {name: float(np.mean(values)) for name, values in timings.items()},
        "scope": "eager adapter.update only, native learner arithmetic unchanged; includes finite checks and host metrics synchronization, excludes runner counters, checkpointing and evaluation"}
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not all(all(step.values()) for step in comparisons):
        raise AssertionError("eager finite-check optimization changed native update")
    if not all(all(all(result.values()) for result in cases.values()) for cases in rollback.values()):
        raise AssertionError("nonfinite detection or rollback differs")


if __name__ == "__main__":
    main()
