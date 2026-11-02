"""Bounded runner, numerical target, and reproducibility checks for CQL."""
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from algorithms import cql as native_cql
from host_integration.cql_adapter import CQLAdapter, _initialize
from thrusterchain.datasets import DatasetError, generate_dataset, inference_evaluation_guard
from thrusterchain.training_protocol import EvaluationConfig, LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import run_mode, train_offline, train_transitions


class CQLAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=1976, provenance="smoke")
        cls.train = train_transitions(cls.dataset)

    def test_runner_exact_resume_and_reloaded_frozen_evaluation(self):
        full, receipt = train_offline(self.dataset, CQLAdapter, budget=OfflineBudget(2, 4), seed=875)
        first, _ = train_offline(self.dataset, CQLAdapter, budget=OfflineBudget(1, 4), seed=875)
        resumed, _ = train_offline(self.dataset, CQLAdapter, budget=OfflineBudget(1, 4), seed=875, resume_from=first)
        self.assertEqual(full.to_bytes(), resumed.to_bytes())
        self.assertEqual(set(full.optimizer_step_counts.values()), {2})
        self.assertEqual(receipt["environment_interactions"], 0)
        with TemporaryDirectory() as tmp:
            result = run_mode(self.dataset, full, CQLAdapter,
                config=EvaluationConfig(frozen_policy=True, episodes=1, seed=947),
                families=("recorded",), checkpoint_directory=Path(tmp) / "saved")
            self.assertEqual(result["receipt"]["evaluation_checkpoint_source"], "reloaded_verified_checkpoint")
            self.assertEqual(result["receipt"]["adaptation_updates"], 0)
            self.assertEqual(result["receipt"]["frozen_evaluation_steps"], 100)

    def test_numerical_target_boundary_and_deterministic_inference(self):
        model = CQLAdapter()
        model.initialize(self.train, 9, {})
        selected = self.train.select([0, 99])
        obs, nxt = np.asarray(selected.observations, np.float32), np.asarray(selected.next_observations, np.float32)
        obs, nxt = (obs - model._mean) / model._scale, (nxt - model._mean) / model._scale
        _, update_key = jax.random.split(model._rng)
        _, _, knext, _, _, _ = jax.random.split(update_key, 6)
        next_a, _ = native_cql.tanh_gaussian_sample(*model._actor.apply(model._state.actor.params, jnp.asarray(nxt)),
                                                     jax.random.normal(knext, (2, 4)), 1.)
        qnext = np.minimum(np.asarray(model._critic1.apply(model._state.critic1_target.params, nxt, next_a)),
                           np.asarray(model._critic2.apply(model._state.critic2_target.params, nxt, next_a)))
        target = np.asarray(selected.rewards) + (1 - np.asarray(selected.terminated)) * .98 * qnext
        pred = np.asarray(model._critic1.apply(model._state.critic1.params, obs, selected.actions))
        expected = np.square(pred - target).mean()
        frozen = model.freeze()
        mu, _ = model._actor.apply(model._state.actor.params, jnp.asarray(obs[0]))
        np.testing.assert_allclose(frozen.act(selected.observations[0]), np.tanh(np.asarray(mu)), atol=1e-6)
        np.testing.assert_array_equal(frozen.act(selected.observations[0]), frozen.act(selected.observations[0]))
        with inference_evaluation_guard():
            with self.assertRaises(DatasetError): model.update(selected)
        timed = replace(selected, truncated=np.array([True, False]),
                        next_action_valid=np.array([False, False]), next_actions=np.zeros_like(selected.next_actions))
        result = model.update(timed)
        self.assertAlmostEqual(result["qf1_loss"], float(expected), places=4)
        self.assertTrue(all(v == 1 for v in model.optimizer_step_counts.values()))
        self.assertEqual(model._args.bc_steps, 0)
        before = model.checkpoint()
        wrong_metadata = before.metadata
        wrong_metadata["adapter"]["source_sha256"]["host"] = "0" * 64
        changed = LearnerCheckpoint(before.adapter_id, before.components, before.optimizer_steps,
                                    before.optimizer_step_counts, wrong_metadata)
        with self.assertRaisesRegex(ValueError, "source or runtime version"):
            CQLAdapter().restore(changed)
        huge = replace(selected.select([0]), rewards=np.asarray([3e38], np.float32))
        with self.assertRaises(FloatingPointError): model.update(huge)
        self.assertEqual(before.to_bytes(), model.checkpoint().to_bytes())

    def test_seed_width_and_host_initialization_parity(self):
        args = replace(native_cql.Args(), seed=13, dataset="thrusterchain-v5", discount=.98, bc_steps=0, cql_reward_scale=1., cql_reward_bias=0.)
        native_rng, native_state, _ = native_cql.initialize(args, 14, 4, 1.)
        actual_rng, actual_state, _ = _initialize(args, 14)
        np.testing.assert_array_equal(native_rng, actual_rng)
        for original, actual in zip(jax.tree_util.tree_leaves(native_state), jax.tree_util.tree_leaves(actual_state)):
            np.testing.assert_array_equal(original, actual)
        high = replace(args, seed=2**32 + 13)
        high_rng, _, _ = _initialize(high, 14)
        self.assertFalse(np.array_equal(actual_rng, high_rng))
        with self.assertRaises(ValueError): CQLAdapter().initialize(self.train, 2**64, {})
        with self.assertRaises(ValueError): CQLAdapter().initialize(self.train, 9, {"bc_steps": 1})


if __name__ == "__main__":
    unittest.main()
