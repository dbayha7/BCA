"""Small, task-specific adapter and exact-resumption checks."""
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from algorithms import td3_bc as native_host
from host_integration.td3bc_adapter import TD3BCAdapter, _initialize
from host_integration.td3bc_probe import _manual_critic_loss
from runtime.networks import Transition
from thrusterchain.datasets import DatasetError, generate_dataset, inference_evaluation_guard
from thrusterchain.training_protocol import EvaluationConfig, LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import run_mode, train_offline, train_transitions


class TD3BCAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=793, provenance="smoke")

    def test_runner_resume_and_frozen_evaluation(self):
        budget = OfflineBudget(updates=3, batch_size=7)
        full, receipt = train_offline(self.dataset, TD3BCAdapter, budget=budget, seed=842)
        first, _ = train_offline(self.dataset, TD3BCAdapter, budget=OfflineBudget(1, 7), seed=842)
        first = LearnerCheckpoint.from_bytes(first.to_bytes())
        resumed, _ = train_offline(self.dataset, TD3BCAdapter, budget=OfflineBudget(2, 7),
                                   seed=842, resume_from=first)
        self.assertEqual(full.to_bytes(), resumed.to_bytes())
        self.assertEqual(dict(full.optimizer_step_counts), {"critic": 3, "actor": 1})
        self.assertGreater(full.metadata["adapter"]["args"]["seed"], 2**32)
        self.assertEqual(full.metadata["adapter"]["args"]["seed"], full.metadata["adapter"]["seed"])
        self.assertEqual(receipt["environment_interactions"], 0)
        with TemporaryDirectory() as tmp:
            results = run_mode(self.dataset, full, TD3BCAdapter,
                               config=EvaluationConfig(frozen_policy=True, episodes=1, seed=843),
                               families=("recorded",), checkpoint_directory=Path(tmp) / "saved")
            self.assertEqual(results["receipt"]["adaptation_updates"], 0)
            self.assertEqual(results["receipt"]["frozen_evaluation_steps"], 100)
            self.assertEqual(results["frozen_evaluation"]["recorded"]["episodes"], 1)
            self.assertEqual(results["receipt"]["evaluation_checkpoint_source"], "reloaded_verified_checkpoint")

    def test_transform_and_cooperative_guard(self):
        batch = train_transitions(self.dataset)
        model = TD3BCAdapter()
        model.initialize(batch, seed=9, learning_settings={})
        np.testing.assert_array_equal(model._mean, np.asarray(batch.observations, np.float32).mean(0))
        frozen = model.freeze()
        action = frozen.act(batch.observations[0])
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((np.abs(action) <= 1).all())
        normalized = (np.asarray(batch.observations[0], np.float32) - model._mean) / model._scale
        expected = np.asarray(model._actor.apply(model._state.actor.params, jnp.asarray(normalized)))
        np.testing.assert_allclose(action, expected, atol=1e-6)
        with inference_evaluation_guard():
            with self.assertRaises(DatasetError):
                model.update(batch.select([0, 1]))
        with self.assertRaises(AttributeError):
            frozen._params = b"modified"
        with self.assertRaises(ValueError):
            model.act(batch.observations[0], explore=True)
        with self.assertRaises(ValueError):
            TD3BCAdapter().initialize(batch, 9, {"discount": .99})
        with self.assertRaisesRegex(ValueError, "seed must be"):
            TD3BCAdapter().initialize(batch, 2**64, {})

        before = model.checkpoint()
        metadata = before.metadata
        metadata["adapter"]["source_sha256"]["host"] = "0" * 64
        wrong_source = LearnerCheckpoint(before.adapter_id, before.components, before.optimizer_steps,
                                         before.optimizer_step_counts, metadata)
        with self.assertRaisesRegex(ValueError, "source or runtime version"):
            TD3BCAdapter().restore(wrong_source)

        too_large = replace(batch.select([0]), rewards=np.array([3e38], dtype=np.float32))
        with self.assertRaises(FloatingPointError):
            model.update(too_large)
        self.assertEqual(before.to_bytes(), model.checkpoint().to_bytes())

    def test_time_limit_bootstraps_and_termination_does_not(self):
        train = train_transitions(self.dataset)
        selected = train.select([0, 99])
        model = TD3BCAdapter()
        model.initialize(train, seed=9, learning_settings={})
        obs, next_obs = np.asarray(selected.observations, np.float32), np.asarray(selected.next_observations, np.float32)
        native = Transition(jnp.asarray((obs - model._mean) / model._scale), jnp.asarray(selected.actions),
                            jnp.asarray(selected.rewards, dtype=jnp.float32),
                            jnp.asarray((next_obs - model._mean) / model._scale),
                            jnp.asarray(selected.terminated, dtype=jnp.float32))
        _, noise = jax.random.split(model._rng)
        expected = _manual_critic_loss(model._args, model._state, model._actor, model._critic, native, noise)
        wrong = _manual_critic_loss(model._args, model._state, model._actor, model._critic,
                                    native._replace(done=jnp.ones(2)), noise)
        # Set the first row's external timeout while retaining its true successor and bootstrap.
        timed = replace(selected, truncated=np.array([True, False]),
                        next_action_valid=np.array([False, False]),
                        next_actions=np.zeros_like(selected.next_actions))
        measured = model.update(timed)["critic_loss"]
        self.assertAlmostEqual(measured, expected, places=4)
        # Suppressing the timeout's continuation is a different target.
        self.assertGreater(abs(expected - wrong), 1e-7)

    def test_full_seed_words_and_native_32bit_initialization(self):
        args = replace(native_host.Args(), seed=19, discount=.98)
        key, state, _ = _initialize(args, 14)
        native_key, native_state, _ = native_host.initialize(args, 14, 4, 1.)
        np.testing.assert_array_equal(key, native_key)
        for actual, expected in zip(jax.tree_util.tree_leaves(state), jax.tree_util.tree_leaves(native_state)):
            np.testing.assert_array_equal(actual, expected)
        high_key, high_state, _ = _initialize(replace(args, seed=19 + 2**32), 14)
        self.assertFalse(np.array_equal(key, high_key))
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
            jax.tree_util.tree_leaves(state.actor.params), jax.tree_util.tree_leaves(high_state.actor.params))))
        batch = train_transitions(self.dataset)
        high = TD3BCAdapter()
        high.initialize(batch, 19 + 2**32, {})
        checkpoint = high.checkpoint()
        restored = TD3BCAdapter()
        restored.restore(checkpoint)
        self.assertEqual(restored.checkpoint().to_bytes(), checkpoint.to_bytes())
        self.assertEqual(checkpoint.metadata["adapter"]["seed"], 19 + 2**32)


if __name__ == "__main__":
    unittest.main()
