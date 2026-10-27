"""Bounded ReBRAC runner, boundary and checkpoint tests."""
import unittest
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from algorithms import rebrac as native_rebrac
from host_integration.rebrac_adapter import ReBRACAdapter, _batch, _initialize, _pack
from thrusterchain.datasets import generate_dataset
from thrusterchain.training_protocol import EvaluationConfig, LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import run_mode, train_offline, train_transitions


class ReBRACAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=1937, provenance="smoke")

    def test_runner_resume_and_frozen_rollout(self):
        full, receipt = train_offline(self.dataset, ReBRACAdapter, budget=OfflineBudget(3, 6), seed=842)
        first, _ = train_offline(self.dataset, ReBRACAdapter, budget=OfflineBudget(1, 6), seed=842)
        resumed, _ = train_offline(self.dataset, ReBRACAdapter, budget=OfflineBudget(2, 6), seed=842,
                                   resume_from=LearnerCheckpoint.from_bytes(first.to_bytes()))
        self.assertEqual(full.to_bytes(), resumed.to_bytes())
        self.assertEqual(dict(full.optimizer_step_counts), {"critic": 3, "actor": 2})
        self.assertEqual(receipt["environment_interactions"], 0)
        result = run_mode(self.dataset, full, ReBRACAdapter,
                          config=EvaluationConfig(frozen_policy=True, episodes=1, seed=843),
                          families=("recorded",))
        self.assertEqual(result["receipt"]["adaptation_updates"], 0)
        self.assertEqual(result["receipt"]["frozen_evaluation_steps"], 100)
        self.assertEqual(result["frozen_evaluation"]["recorded"]["episodes"], 1)

    def test_delayed_updates_and_fail_fast_continuing(self):
        batch = train_transitions(self.dataset)
        model = ReBRACAdapter()
        model.initialize(batch, seed=9, learning_settings={})
        selected = batch.select(np.array([0, 99]))
        native = _batch(selected, model._mean, model._scale)
        _, noise_key = jax.random.split(model._rng)
        proposals = np.asarray(model._actor.apply(model._state.actor.target_params, native.next_obs))
        noise = np.asarray(jax.random.normal(noise_key, proposals.shape)) * model._args.policy_noise
        proposals = np.clip(proposals + np.clip(noise, -model._args.noise_clip, model._args.noise_clip), -1, 1)
        qnext = np.asarray(model._critic.apply(model._state.critic.target_params, native.next_obs, jnp.asarray(proposals)))
        penalty = np.square(proposals - np.asarray(native.next_action)).sum(-1)
        target = np.asarray(native.reward) + (1 - np.asarray(native.done)) * model._args.gamma * (qnext.min(0) - model._args.critic_bc_coef * penalty)
        predicted = np.asarray(model._critic.apply(model._state.critic.params, native.obs, native.action))
        expected_loss = np.square(predicted - target[None, :]).mean(axis=1).sum()
        metrics = model.update(selected)
        self.assertTrue(np.isclose(metrics["critic_loss"], expected_loss, atol=3e-5, rtol=3e-5))
        self.assertEqual(target[-1], float(native.reward[-1]))
        self.assertEqual(model.optimizer_step_counts, {"critic": 1, "actor": 1})
        model.update(batch.select(np.array([1, 100])))
        self.assertEqual(model.optimizer_step_counts, {"critic": 2, "actor": 1})
        frozen = model.freeze()
        self.assertEqual(frozen.act(batch.observations[0]).shape, (4,))
        with self.assertRaises(AttributeError):
            frozen._state = b"mutated"
        continuing = generate_dataset(episodes=10, seed=1938, provenance="smoke",
                                      task_semantics="continuing_time_limit")
        with self.assertRaisesRegex(ValueError, "recorded next actions"):
            ReBRACAdapter().initialize(train_transitions(continuing), seed=9, learning_settings={})

    def test_full_width_seed_and_native_low_seed_initialization(self):
        batch = train_transitions(self.dataset)
        low = ReBRACAdapter()
        high = ReBRACAdapter()
        low.initialize(batch, seed=13, learning_settings={})
        high.initialize(batch, seed=2**32 + 13, learning_settings={})
        self.assertNotEqual(low.checkpoint().components["model"], high.checkpoint().components["model"])
        sample = _batch(batch.select(np.array([0])), low._mean, low._scale)
        args = replace(native_rebrac.Args(), seed=13, gamma=.98)
        key_native, state_native, _ = native_rebrac.initialize(args, sample)
        key_actual, state_actual, _ = _initialize(args, sample)
        np.testing.assert_array_equal(key_native, key_actual)
        self.assertEqual(_pack(state_native.actor.params), _pack(state_actual.actor.params))
        self.assertEqual(_pack(state_native.critic.params), _pack(state_actual.critic.params))
        with self.assertRaises(ValueError):
            ReBRACAdapter().initialize(batch, seed=2**64, learning_settings={})


if __name__ == "__main__":
    unittest.main()
