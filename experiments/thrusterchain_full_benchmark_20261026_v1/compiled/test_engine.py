"""Numerical parity, sampling and exact-resume gates for compiled study updates."""
import unittest

import jax
import numpy as np

from thrusterchain.quality_datasets import generate_behavior_dataset
from thrusterchain.training_protocol import OfflineBudget
from thrusterchain.training_runner import train_offline, train_transitions
from compiled.engine import METHODS, _PROGRAMS, _factory, _native_dataset, _program, _program_key, _scan, train_chunk
from thrusterchain.training_runner import _identity


class CompiledEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_behavior_dataset(behavior="mixed_random_strong", episodes=10, seed=733,
                                                provenance="smoke")
        cls.batch = train_transitions(cls.dataset)

    def assert_states_close(self, first, second):
        # Flax PyTreeDef carries static optimizer function objects. Equivalent
        # independent optimizer factories have different Python identities, so
        # compare numeric leaf paths and values rather than function identity.
        a, _ = jax.tree_util.tree_flatten_with_path(first)
        b, _ = jax.tree_util.tree_flatten_with_path(second)
        self.assertEqual([jax.tree_util.keystr(path) for path, _ in a],
                         [jax.tree_util.keystr(path) for path, _ in b])
        for (_, left), (_, right) in zip(a, b):
            left, right = np.asarray(left), np.asarray(right)
            self.assertEqual(left.shape, right.shape)
            self.assertEqual(left.dtype, right.dtype)
            if left.dtype.kind in "biu":
                np.testing.assert_array_equal(left, right)
            else:
                np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-5)

    def test_fixed_terminal_batches_match_eager_losses_state_and_keys(self):
        for name in METHODS:
            with self.subTest(method=name):
                eager, accelerated = _factory(name)(), _factory(name)()
                eager.initialize(self.batch, seed=19, learning_settings={})
                accelerated.initialize(self.batch, seed=19, learning_settings={})
                # Rows 99 and 199 are true terminal transitions; no next action
                # is invented. Repeated row 2 also tests arbitrary batch order.
                rows = np.array([[0, 99, 102, 199, 2, 2, 1],
                                 [199, 2, 99, 0, 102, 2, 1]], dtype=np.int64)
                metrics = None
                for row in rows:
                    metrics = eager.update(self.batch.select(row))
                key, state, actual = _scan(accelerated, name, _native_dataset(accelerated, name, self.batch), rows)
                self.assert_states_close(eager._state, state)
                np.testing.assert_array_equal(np.asarray(eager._rng), np.asarray(key))
                self.assertEqual(set(metrics), set(actual))
                for metric in metrics:
                    np.testing.assert_allclose(metrics[metric], actual[metric], atol=1e-6, rtol=1e-5)

    def test_real_runner_sampling_and_compiled_resume(self):
        for name in METHODS:
            with self.subTest(method=name):
                factory = _factory(name)
                standard, _ = train_offline(self.dataset, factory, budget=OfflineBudget(4, 8), seed=834)
                full, rec = train_chunk(self.dataset, name, 834, 8, 4)
                first, _ = train_chunk(self.dataset, name, 834, 8, 2)
                resumed, _ = train_chunk(self.dataset, name, 834, 8, 2, first)
                self.assertEqual(full.to_bytes(), resumed.to_bytes())
                self.assertEqual(dict(full.optimizer_step_counts), dict(standard.optimizer_step_counts))
                self.assertEqual(full.metadata["runner_state"]["sampler_rng_state"],
                                 standard.metadata["runner_state"]["sampler_rng_state"])
                left, right = factory(), factory()
                left.restore(standard)
                right.restore(full)
                self.assert_states_close(left._state, right._state)
                np.testing.assert_array_equal(np.asarray(left._rng), np.asarray(right._rng))
                self.assertEqual(rec["dataset_draws"], 32)
                self.assertEqual(rec["environment_interactions"], 0)

    def test_cache_cross_dataset_and_seed(self):
        other = generate_behavior_dataset(behavior="mixed_random_strong", episodes=10, seed=744,
                                          provenance="smoke")
        other_batch = train_transitions(other)
        rows = np.array([[0, 99, 102, 199, 2, 2, 1],
                         [199, 2, 99, 0, 102, 2, 1]], dtype=np.int64)
        for name in METHODS:
            with self.subTest(method=name):
                first, second, eager = (_factory(name)() for _ in range(3))
                first.initialize(self.batch, seed=19, learning_settings={})
                second.initialize(other_batch, seed=20, learning_settings={})
                eager.initialize(other_batch, seed=20, learning_settings={})
                fn1, _ = _program(first, name, 7)
                _scan(first, name, _native_dataset(first, name, self.batch), rows)
                size1 = fn1._cache_size()
                fn2, _ = _program(second, name, 7)
                if name == 'unifloral-unified':
                    # Its networks close over data normalization.
                    self.assertIsNot(fn1, fn2)
                else:
                    self.assertIs(fn1, fn2)
                for row in rows:
                    eager.update(other_batch.select(row))
                key, state, _ = _scan(second, name, _native_dataset(second, name, other_batch), rows)
                if fn1 is fn2:
                    self.assertEqual(size1, fn1._cache_size())
                self.assert_states_close(eager._state, state)
                np.testing.assert_array_equal(np.asarray(eager._rng), np.asarray(key))

    def test_receipt_verifier_for_first_and_resumed_chunks(self):
        from study import primary_steps, verify_native_receipt
        identity = _identity(self.dataset)
        data_receipt = {"dataset_sha256": self.dataset.digest,
                        "train_split_sha256": identity["split_sha256"]["train"]}
        spec = {"batch_size": 8, "chunk_critic_rounds": 2}
        for name in METHODS:
            with self.subTest(method=name):
                steps = primary_steps(name, spec["chunk_critic_rounds"])
                first, rec1 = train_chunk(self.dataset, name, 834, 8, steps)
                verify_native_receipt(rec1, first, None, data_receipt, name, spec)
                second, rec2 = train_chunk(self.dataset, name, 834, 8, steps, first)
                verify_native_receipt(rec2, second, first, data_receipt, name, spec)

    def test_cql_cross_dataset_cached_program_equals_fresh(self):
        other = generate_behavior_dataset(behavior="mixed_random_strong", episodes=10, seed=744,
                                          provenance="smoke")
        other_batch = train_transitions(other)
        rows = np.array([[0, 99, 102, 199, 2, 2, 1],
                         [199, 2, 99, 0, 102, 2, 1]], dtype=np.int64)
        source, second, fresh = (_factory("cql")() for _ in range(3))
        source.initialize(self.batch, seed=19, learning_settings={})
        second.initialize(other_batch, seed=20, learning_settings={})
        fresh.initialize(other_batch, seed=20, learning_settings={})
        _scan(source, "cql", _native_dataset(source, "cql", self.batch), rows)
        cached_key, cached_state, cached_metrics = _scan(
            second, "cql", _native_dataset(second, "cql", other_batch), rows)
        _PROGRAMS.pop(_program_key(second, "cql", 7))
        fresh_key, fresh_state, fresh_metrics = _scan(
            fresh, "cql", _native_dataset(fresh, "cql", other_batch), rows)
        np.testing.assert_array_equal(np.asarray(cached_key), np.asarray(fresh_key))
        for a, b in zip(jax.tree_util.tree_leaves(cached_state), jax.tree_util.tree_leaves(fresh_state)):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        self.assertEqual(cached_metrics, fresh_metrics)


if __name__ == "__main__":
    unittest.main()
