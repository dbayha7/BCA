"""Bounded repository-native IQL adapter checks on v5 task data."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from algorithms import iql as native_iql
from host_integration.iql_adapter import IQLAdapter, _batch, _initialize, _pack
from thrusterchain.datasets import DatasetError, generate_dataset, inference_evaluation_guard
from thrusterchain.training_protocol import EvaluationConfig, LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import run_mode, train_offline, train_transitions


class IQLAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=2872, provenance="smoke")

    def test_runner_resume_and_frozen_rollout(self):
        full, receipt = train_offline(self.dataset, IQLAdapter, budget=OfflineBudget(3, 6), seed=842)
        first, _ = train_offline(self.dataset, IQLAdapter, budget=OfflineBudget(1, 6), seed=842)
        resumed, _ = train_offline(self.dataset, IQLAdapter, budget=OfflineBudget(2, 6), seed=842,
                                   resume_from=LearnerCheckpoint.from_bytes(first.to_bytes()))
        self.assertEqual(full.to_bytes(), resumed.to_bytes())
        self.assertEqual(dict(full.optimizer_step_counts), {"critic": 3, "actor": 3, "value": 3})
        self.assertEqual(receipt["environment_interactions"], 0)
        with TemporaryDirectory() as tmp:
            result = run_mode(self.dataset, full, IQLAdapter,
                              config=EvaluationConfig(frozen_policy=True, episodes=1, seed=843),
                              families=("recorded",), checkpoint_directory=Path(tmp) / "saved")
            self.assertEqual(result["receipt"]["adaptation_updates"], 0)
            self.assertEqual(result["receipt"]["frozen_evaluation_steps"], 100)
            self.assertEqual(result["receipt"]["evaluation_checkpoint_source"], "reloaded_verified_checkpoint")

    def test_update_losses_and_termination_against_numpy(self):
        batch = train_transitions(self.dataset)
        model = IQLAdapter()
        model.initialize(batch, seed=9, learning_settings={})
        # The target critics are deliberately different from online critics, so
        # the advantage check distinguishes the two parameter copies.
        import jax
        target_params = jax.tree_util.tree_map(lambda x: x + .005, model._state.qf_target.params)
        model._state = model._state._replace(qf_target=model._state.qf_target.replace(params=target_params))
        selected = batch.select(np.array([0, 1, 99, 102, 199]))
        native = _batch(selected, model._mean, model._scale)
        q = np.asarray(model._q.apply(model._state.qf.params, native.obs, native.action))
        target_q = np.asarray(model._q.apply(model._state.qf_target.params, native.obs, native.action)).min(-1)
        self.assertGreater(np.max(np.abs(target_q - q.min(-1))), 1e-5)
        values = np.asarray(model._value.apply(model._state.vf.params, native.obs))
        next_values = np.asarray(model._value.apply(model._state.vf.params, native.next_obs))
        advantage = target_q - values
        expectile = np.mean(np.where(advantage < 0, 1 - model._args.iql_tau, model._args.iql_tau) * advantage ** 2)
        target = np.asarray(native.reward) + (1 - np.asarray(native.done)) * model._args.discount * next_values
        q_loss = np.square(q - target[:, None]).mean()
        mean, std = model._actor.apply(model._state.actor.params, native.obs, deterministic=True)
        mean, std, actions = np.asarray(mean), np.asarray(std), np.asarray(native.action)
        log_prob = -((actions - mean) ** 2) / (2 * std ** 2) - np.log(std) - .5 * np.log(2 * np.pi)
        actor_loss = np.mean(np.minimum(np.exp(model._args.beta * advantage), native_iql.EXP_ADV_MAX) * -log_prob.sum(-1))
        actual = model.update(selected)
        for name, expected in (("value_loss", expectile), ("q_loss", q_loss), ("actor_loss", actor_loss)):
            self.assertTrue(np.isclose(actual[name], expected, rtol=3e-5, atol=3e-5), name)
        np.testing.assert_array_equal(target[[2, 4]], np.asarray(native.reward)[[2, 4]])
        self.assertEqual(model.optimizer_step_counts, {"critic": 1, "actor": 1, "value": 1})
        self.assertEqual(int(model._state.qf_target.step), 1)
        frozen = model.freeze()
        expected_mean, _ = model._actor.apply(model._state.actor.params, native.obs[0], deterministic=True)
        np.testing.assert_allclose(frozen.act(selected.observations[0]), np.asarray(expected_mean), atol=1e-6)
        with self.assertRaises(AttributeError):
            frozen._state = b"changed"

    def test_time_limit_bootstrap_and_update_guards(self):
        batch = train_transitions(self.dataset)
        model = IQLAdapter()
        model.initialize(batch, seed=9, learning_settings={})
        selected = batch.select(np.array([0, 99]))
        selected = replace(selected, truncated=np.array([True, False]),
                           next_action_valid=np.array([False, False]),
                           next_actions=np.zeros_like(selected.next_actions))
        native = _batch(selected, model._mean, model._scale)
        q = np.asarray(model._q.apply(model._state.qf.params, native.obs, native.action))
        next_v = np.asarray(model._value.apply(model._state.vf.params, native.next_obs))
        reward, done = np.asarray(native.reward), np.asarray(native.done)
        expected = np.square(q - (reward + (1 - done) * model._args.discount * next_v)[:, None]).mean()
        wrong = np.square(q - reward[:, None]).mean()
        self.assertGreater(abs(expected - wrong), 1e-7)
        actual = model.update(selected)
        self.assertTrue(np.isclose(actual["q_loss"], expected, rtol=3e-5, atol=3e-5))
        with inference_evaluation_guard():
            with self.assertRaises(DatasetError):
                model.update(selected)
        before = model.checkpoint()
        too_large = replace(batch.select([0]), rewards=np.array([3e38], dtype=np.float32))
        with self.assertRaises(FloatingPointError):
            model.update(too_large)
        self.assertEqual(before.to_bytes(), model.checkpoint().to_bytes())

    def test_seed_mapping_and_native_initialization(self):
        batch = train_transitions(self.dataset)
        low, high = IQLAdapter(), IQLAdapter()
        low.initialize(batch, seed=13, learning_settings={})
        high.initialize(batch, seed=2**32 + 13, learning_settings={})
        self.assertNotEqual(low.checkpoint().components["model"], high.checkpoint().components["model"])
        args = replace(native_iql.Args(), seed=13, discount=.98)
        state_native, key_native, _ = native_iql.initialize_agent(args, batch.observations.shape[-1], 4, 1.)
        state_actual, key_actual, _, _, _ = _initialize(args, batch.observations.shape[-1])
        np.testing.assert_array_equal(key_native, key_actual)
        for name in ("actor", "qf", "qf_target", "vf"):
            self.assertEqual(_pack(getattr(state_native, name).params), _pack(getattr(state_actual, name).params))
        with self.assertRaises(ValueError):
            IQLAdapter().initialize(batch, seed=2**64, learning_settings={})


if __name__ == "__main__":
    unittest.main()
