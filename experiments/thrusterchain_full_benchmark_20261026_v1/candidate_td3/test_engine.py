"""Bounded CPU checks for the separate TD3+BC candidate; GPU remains pending."""
import os
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from candidate_td3.engine import (ENGINE, _native_dataset, dependency_hashes, make_update,
                                  runtime_signature, source_sha256, train_chunk)
from host_integration.td3bc_adapter import TD3BCAdapter
from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_protocol import OfflineBudget
from thrusterchain.training_runner import RunnerError, _rng, train_offline, train_transitions


class CandidateTD3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.datasets = [generate_behavior_dataset(behavior="mixed_random_strong", episodes=10,
                                               seed=seed, provenance="smoke")
                        for seed in (733, 744)]

    def assert_state_close(self, first, second):
        left = jax.tree_util.tree_flatten_with_path(first)[0]
        right = jax.tree_util.tree_flatten_with_path(second)[0]
        self.assertEqual([jax.tree_util.keystr(x[0]) for x in left],
                         [jax.tree_util.keystr(x[0]) for x in right])
        for (_, a), (_, b) in zip(left, right):
            a, b = np.asarray(a), np.asarray(b)
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.dtype, b.dtype)
            if a.dtype.kind in "biu":
                np.testing.assert_array_equal(a, b)
            else:
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)

    def test_native_state_metrics_rng_counters_and_pcg64(self):
        for dataset, seed in zip(self.datasets, (834, 845)):
            for batch_size in (8, 256):
                with self.subTest(dataset_seed=dataset.metadata["seed"], batch_size=batch_size):
                    reference, native_receipt = train_offline(
                        dataset, TD3BCAdapter, budget=OfflineBudget(4, batch_size), seed=seed)
                    candidate, receipt = train_chunk(dataset, seed, batch_size, 4)
                    left, right = TD3BCAdapter(), TD3BCAdapter()
                    left.restore(reference)
                    right.restore(candidate)
                    self.assert_state_close(left._state, right._state)
                    np.testing.assert_array_equal(np.asarray(left._rng), np.asarray(right._rng))
                    self.assertEqual(dict(reference.optimizer_step_counts), dict(candidate.optimizer_step_counts))
                    self.assertEqual(dict(candidate.optimizer_step_counts), {"critic": 4, "actor": 2})
                    self.assertEqual(reference.metadata["runner_state"]["sampler_rng_state"],
                                     candidate.metadata["runner_state"]["sampler_rng_state"])
                    self.assertEqual(set(native_receipt["last_update_metrics"]), set(receipt["last_update_metrics"]))
                    for key, expected in native_receipt["last_update_metrics"].items():
                        np.testing.assert_allclose(expected, receipt["last_update_metrics"][key], atol=1e-6, rtol=1e-5)
                    self.assertEqual(receipt["dataset_draws"], 4 * batch_size)
                    self.assertEqual(receipt["environment_interactions"], 0)
                    self.assertEqual(receipt["engine"], ENGINE)
                    self.assertEqual(receipt["engine_sha256"], source_sha256())
                    self.assertEqual(receipt["source_dependencies_sha256"], dependency_hashes())
                    self.assertEqual(receipt["runtime_signature"], runtime_signature())
                    manual = _rng(seed, "offline_batches")
                    # The train split has six 100-step episodes in these fixtures.
                    count = dataset.split("train")["actions"].shape[0] * dataset.split("train")["actions"].shape[1]
                    for _ in range(4):
                        manual.integers(0, count, size=batch_size)
                    self.assertEqual(manual.bit_generator.state, candidate.metadata["runner_state"]["sampler_rng_state"])

    def test_fixed_terminal_batches_every_step(self):
        batch = train_transitions(self.datasets[0])
        eager, candidate = TD3BCAdapter(), TD3BCAdapter()
        eager.initialize(batch, seed=19, learning_settings={})
        candidate.initialize(batch, seed=19, learning_settings={})
        native, update = _native_dataset(candidate, batch), make_update(candidate)
        rows = (np.array([0, 99, 102, 199, 2, 2, 1]),
                np.array([199, 2, 99, 0, 102, 2, 1]))
        self.assertTrue(np.asarray(batch.terminated)[[99, 199]].all())
        for row in rows * 2:
            expected = eager.update(batch.select(row))
            selected = jax.tree_util.tree_map(lambda value: value[row], native)
            rng, noise = jax.random.split(candidate._rng)
            state, metrics = update(candidate._state, selected, int(candidate._state.critic.step) + 1, noise)
            candidate._state, candidate._rng = state, rng
            self.assert_state_close(eager._state, candidate._state)
            np.testing.assert_array_equal(np.asarray(eager._rng), np.asarray(candidate._rng))
            self.assertEqual(eager.optimizer_step_counts, candidate.optimizer_step_counts)
            self.assertEqual(set(expected), set(metrics))
            for key in expected:
                np.testing.assert_allclose(expected[key], metrics[key], atol=1e-6, rtol=1e-5)

    def test_exact_resume_and_identity(self):
        dataset = self.datasets[0]
        for batch_size in (8, 256):
            with self.subTest(batch_size=batch_size):
                full, receipt = train_chunk(dataset, 834, batch_size, 4)
                first, first_receipt = train_chunk(dataset, 834, batch_size, 2)
                resumed, second_receipt = train_chunk(dataset, 834, batch_size, 2, first)
                self.assertEqual(full.to_bytes(), resumed.to_bytes())
                self.assertEqual(second_receipt["resume_from_sha256"], first.digest)
                self.assertEqual(second_receipt["checkpoint_sha256"], resumed.digest)
                self.assertIsNone(first_receipt["resume_from_sha256"])
                self.assertEqual(receipt["optimizer_step_deltas"], {"critic": 4, "actor": 2})
                with self.assertRaises(RunnerError):
                    train_chunk(dataset, 835, batch_size, 2, first)
                with self.assertRaises(RunnerError):
                    train_chunk(dataset, 834, batch_size + 1, 2, first)

    def test_nonfinite_update_fails_without_checkpoint(self):
        def invalid_factory(model):
            original = make_update(model)
            def invalid(state, batch, step, noise):
                state, metrics = original(state, batch, step, noise)
                metrics = dict(metrics)
                metrics["critic_loss"] = jnp.asarray(np.nan, dtype=jnp.float32)
                return state, metrics
            return invalid
        with patch("candidate_td3.engine.make_update", side_effect=invalid_factory):
            with self.assertRaises(FloatingPointError):
                train_chunk(self.datasets[0], 834, 8, 1)

    def test_changed_source_during_chunk_is_refused(self):
        actual = source_sha256()
        with patch("candidate_td3.engine.source_sha256", side_effect=(actual, "changed")):
            with self.assertRaisesRegex(RunnerError, "changed during chunk"):
                train_chunk(self.datasets[0], 834, 8, 1)

    def test_nonfinite_state_does_not_modify_resumed_checkpoint(self):
        first, _ = train_chunk(self.datasets[0], 834, 8, 2)
        before = first.to_bytes()
        def invalid_factory(model):
            original = make_update(model)
            def invalid(state, batch, step, noise):
                state, metrics = original(state, batch, step, noise)
                invalid_params = jax.tree_util.tree_map(lambda value: jnp.full_like(value, jnp.nan), state.critic.params)
                return state._replace(critic=state.critic.replace(params=invalid_params)), metrics
            return invalid
        with patch("candidate_td3.engine.make_update", side_effect=invalid_factory):
            with self.assertRaises(FloatingPointError):
                train_chunk(self.datasets[0], 834, 8, 1, first)
        self.assertEqual(first.to_bytes(), before)

    def test_runtime_environment_drift_refuses_resume(self):
        first, _ = train_chunk(self.datasets[0], 834, 8, 2)
        before = first.to_bytes()
        original = first.metadata["runner_state"]["runtime_signature"]
        with patch.dict(os.environ, {"XLA_FLAGS": "--xla_enable_fast_math=false"}):
            self.assertNotEqual(runtime_signature(), original)
            with self.assertRaisesRegex(RunnerError, "resume identity mismatch"):
                train_chunk(self.datasets[0], 834, 8, 2, first)
        self.assertEqual(first.to_bytes(), before)


if __name__ == "__main__":
    unittest.main()
