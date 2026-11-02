"""Actual gradients, complete deterministic resume and detached inference."""
import json
import unittest

import numpy as np

from thrusterchain.datasets import DatasetError, inference_evaluation_guard
from thrusterchain.sgd_adapter import ADAPTER_ID, AdapterError, AffineSGDBCAdapter
from thrusterchain.training_protocol import LearnerCheckpoint, TransitionBatch


def batch(observations=None, actions=None, *, reward=0.0, origin=0):
    if observations is None:
        observations = np.random.default_rng(1842).normal(size=(12, 14))
        observations[:, -1] = 7.0
    n = len(observations)
    if actions is None:
        actions = np.random.default_rng(91).uniform(-0.8, 0.8, (n, 4))
    return TransitionBatch(
        observations=observations, actions=actions, rewards=np.full(n, reward),
        next_observations=np.asarray(observations) + 1,
        terminated=np.zeros(n, dtype=bool), truncated=np.zeros(n, dtype=bool),
        bootstrap_mask=np.ones(n), origin=np.full(n, origin, dtype=np.uint8),
        group_ids=np.arange(n))


def component(checkpoint, name):
    return json.loads(checkpoint.components[name])


def altered_checkpoint(checkpoint, *, name=None, value=None, metadata=None):
    components = dict(checkpoint.components)
    if name is not None:
        components[name] = json.dumps(value, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()
    return LearnerCheckpoint(checkpoint.adapter_id, components, checkpoint.optimizer_steps,
                             checkpoint.optimizer_step_counts,
                             checkpoint.metadata if metadata is None else metadata)


class AffineSGDBCAdapterTests(unittest.TestCase):
    def initialized(self, **settings):
        adapter = AffineSGDBCAdapter()
        adapter.initialize(batch(), seed=192, learning_settings=settings)
        return adapter

    def test_capabilities_zero_initial_updates_and_genuine_parameter_change(self):
        adapter = self.initialized()
        before = adapter.checkpoint()
        self.assertEqual(adapter.capabilities.adapter_id, ADAPTER_ID)
        self.assertEqual(adapter.capabilities.fit_semantics, "optimizer_steps")
        self.assertTrue(adapter.capabilities.supports_online)
        self.assertEqual(adapter.capabilities.primary_optimizer, "affine_sgd")
        self.assertEqual(before.optimizer_steps, 0)
        np.testing.assert_array_equal(component(before, "model")["coefficients"], np.zeros((15, 4)))
        result = adapter.update(batch())
        after = adapter.checkpoint()
        self.assertEqual(result["optimizer_steps_performed"], 1)
        self.assertEqual(after.optimizer_steps, 1)
        self.assertEqual(dict(after.optimizer_step_counts), {"affine_sgd": 1})
        self.assertEqual(after.optimizer_step_counts[adapter.capabilities.primary_optimizer], 1)
        self.assertEqual(after.optimizer_steps, sum(after.optimizer_step_counts.values()))
        self.assertEqual(after.primary_optimizer, "affine_sgd")
        self.assertEqual(after.primary_optimizer_steps, 1)
        self.assertEqual(after.metadata["primary_optimizer"], "affine_sgd")
        self.assertEqual(after.metadata["adapter"]["primary_optimizer"], "affine_sgd")
        self.assertNotEqual(before.components["model"], after.components["model"])
        self.assertNotEqual(before.components["optimizer"], after.components["optimizer"])
        self.assertEqual(before.components["rng"], after.components["rng"])
        self.assertEqual(component(after, "target_networks")["applicable"], False)
        self.assertEqual(after.metadata["not_applicable_components"], ["target_networks"])
        self.assertIn("without_reward_learning", after.metadata["adapter"]["online_semantics"])

    def test_multistep_momentum_matches_independent_scalar_gradient(self):
        adapter = self.initialized(learning_rate=0.37, momentum=0.72)
        train = batch()
        prep = component(adapter.checkpoint(), "preprocessing")
        weights, velocity = np.zeros((15, 4)), np.zeros((15, 4))
        for step, indices in enumerate(([0, 2, 4], [7, 3, 8, 9], [11, 0], list(range(12)))):
            subset = train.select(indices)
            x = np.column_stack(((subset.observations - prep["mean"]) / prep["scale"],
                                 np.ones(len(subset))))
            residual = x @ weights - subset.actions
            expected_loss = float(np.mean(residual ** 2))
            gradient = np.zeros_like(weights)
            for row in range(len(subset)):
                for feature in range(15):
                    for action in range(4):
                        gradient[feature, action] += 2 * x[row, feature] * residual[row, action] / (len(subset) * 4)
            velocity = 0.72 * velocity + gradient
            weights -= 0.37 * velocity
            result = adapter.update(subset)
            self.assertAlmostEqual(result["loss"], expected_loss, places=14)
            saved = adapter.checkpoint()
            np.testing.assert_allclose(component(saved, "model")["coefficients"], weights, rtol=1e-14, atol=1e-15)
            np.testing.assert_allclose(component(saved, "optimizer")["velocity"], velocity, rtol=1e-14, atol=1e-15)
            self.assertEqual(saved.optimizer_steps, step + 1)

    def test_normalization_fits_train_observations_once_and_online_ignores_rewards(self):
        first, second = self.initialized(), self.initialized()
        prep = first.checkpoint().components["preprocessing"]
        decoded = json.loads(prep)
        np.testing.assert_array_equal(decoded["mean"], batch().observations.mean(axis=0))
        self.assertEqual(decoded["scale"][-1], 1.0)
        online_x = batch().observations + 10.0
        first.update(batch(online_x, reward=100.0, origin=1))
        second.update(batch(online_x, reward=-500.0, origin=1))
        self.assertEqual(first.checkpoint().to_bytes(), second.checkpoint().to_bytes())
        self.assertEqual(first.checkpoint().components["preprocessing"], prep)

    def test_serialized_resume_matches_uninterrupted_updates_and_exploration_exactly(self):
        first = self.initialized(learning_rate=0.018, momentum=0.83, exploration_std=0.21)
        train = batch()
        for indices in ([0, 2, 5], [8, 3, 1], [11, 9]):
            first.update(train.select(indices))
            first.act(train.observations[indices[0]], explore=True)
        before = first.checkpoint()
        metadata = before.metadata
        metadata["runner_state"] = {"sampling_rng": {"state": 1234567}, "batches": 3}
        before = altered_checkpoint(before, metadata=metadata)
        first.restore(before)
        resumed = AffineSGDBCAdapter()
        resumed.restore(LearnerCheckpoint.from_bytes(before.to_bytes()))
        self.assertEqual(before.digest, resumed.checkpoint().digest)
        self.assertEqual(before.to_bytes(), resumed.checkpoint().to_bytes())
        for step, indices in enumerate(([4, 1, 2], [0, 0, 8], [10, 11, 9], [3, 7])):
            if step == 2:
                for adapter in (first, resumed):
                    adapter.configure_learning_settings({"learning_rate": 0.025, "exploration_std": 0.35})
            self.assertEqual(first.update(train.select(indices)), resumed.update(train.select(indices)))
            for explore in (False, True, True, False):
                np.testing.assert_array_equal(first.act(train.observations[step], explore=explore),
                                              resumed.act(train.observations[step], explore=explore))
            self.assertEqual(first.checkpoint().to_bytes(), resumed.checkpoint().to_bytes())
        self.assertEqual(resumed.checkpoint().metadata["runner_state"], metadata["runner_state"])

    def test_configuration_preserves_all_nonsetting_state(self):
        adapter = self.initialized(momentum=0.5, exploration_std=0.2)
        adapter.update(batch())
        adapter.act(np.ones(14), explore=True)
        before = adapter.checkpoint()
        adapter.configure_learning_settings({"learning_rate": 0.03})
        after = adapter.checkpoint()
        for name in ("model", "preprocessing", "rng", "target_networks"):
            self.assertEqual(before.components[name], after.components[name])
        self.assertEqual(component(before, "optimizer")["velocity"], component(after, "optimizer")["velocity"])
        self.assertEqual(before.optimizer_steps, after.optimizer_steps)
        self.assertEqual(dict(before.optimizer_step_counts), dict(after.optimizer_step_counts))
        self.assertEqual(before.primary_optimizer_steps, after.primary_optimizer_steps)
        self.assertEqual(before.metadata["primary_optimizer"], after.metadata["primary_optimizer"])
        self.assertEqual(before.metadata["adapter"]["primary_optimizer"],
                         after.metadata["adapter"]["primary_optimizer"])
        self.assertEqual(adapter.learning_settings, {"learning_rate": 0.03, "momentum": 0.5, "exploration_std": 0.2})

    def test_frozen_state_is_complete_immutable_and_detached(self):
        adapter = self.initialized(learning_rate=10.0, exploration_std=3.0)
        adapter.update(batch())
        frozen = adapter.freeze()
        before = frozen.state_bytes()
        expected = frozen.act(batch().observations)
        self.assertEqual(expected.dtype, np.float32)
        self.assertTrue(np.all(np.abs(expected) <= 1))
        self.assertTrue(np.any(np.abs(expected) == 1))
        for name in ("mean", "scale", "coefficients"):
            with self.assertRaises(AttributeError):
                setattr(frozen, name, np.zeros(1))
            with self.assertRaises(AttributeError):
                delattr(frozen, name)
            with self.assertRaises(ValueError):
                getattr(frozen, name).setflags(write=True)
            with self.assertRaises(ValueError):
                getattr(frozen, name).flat[0] = 4.0
            exposed = getattr(frozen, name)
            exposed.shape = (exposed.size, 1)
            np.testing.assert_array_equal(expected, frozen.act(batch().observations))
        adapter.update(batch())
        adapter.act(batch().observations, explore=True)
        self.assertEqual(before, frozen.state_bytes())
        np.testing.assert_array_equal(expected, frozen.act(batch().observations))
        state = json.loads(before)
        self.assertEqual(state["transform"]["action_clip"], [-1.0, 1.0])
        self.assertEqual(state["transform"]["action_dtype"], "float32")
        coefficients = np.array(state["coefficients"])
        reproduced = np.clip(((batch().observations - state["mean"]) / state["scale"]) @ coefficients[:-1]
                             + coefficients[-1], -1, 1).astype(np.float32)
        np.testing.assert_array_equal(reproduced, expected)

    def test_invalid_settings_and_observations_fail_without_mutating(self):
        adapter = self.initialized()
        saved = adapter.checkpoint().to_bytes()
        invalid = ({"learning_rate": True}, {"learning_rate": "0.1"}, {"learning_rate": 0},
                   {"learning_rate": float("inf")}, {"momentum": -0.1}, {"momentum": 1.0},
                   {"momentum": complex(0.2)}, {"exploration_std": -0.1},
                   {"exploration_std": np.array(0.1)}, {"exploration_std": float("nan")},
                   {"weight_decay": 0.0}, [])
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(AdapterError):
                adapter.configure_learning_settings(settings)
            self.assertEqual(saved, adapter.checkpoint().to_bytes())
        for observation in (np.ones(13), np.ones((2, 4)), np.ones(14, dtype=bool),
                            np.full(14, "0"), np.full(14, 1j), np.full(14, np.nan)):
            with self.assertRaises(AdapterError):
                adapter.act(observation, explore=True)
            self.assertEqual(saved, adapter.checkpoint().to_bytes())
        with self.assertRaises(AdapterError):
            adapter.act(np.ones(14), explore=1)
        for observations, actions in ((np.ones((2, 13)), np.zeros((2, 4))),
                                      (np.ones((2, 14), dtype=bool), np.zeros((2, 4))),
                                      (np.ones((2, 14)), np.ones((2, 4)) * 1.1),
                                      (np.ones((0, 14)), np.zeros((0, 4)))):
            with self.assertRaises(AdapterError):
                adapter.initialize(batch(observations, actions), 0, {})
            self.assertEqual(saved, adapter.checkpoint().to_bytes())

    def test_nonfinite_gradient_and_bad_restore_are_atomic(self):
        adapter = self.initialized()
        saved = adapter.checkpoint()
        with self.assertRaises(AdapterError):
            adapter.update(batch(np.full((2, 14), 1e308), np.ones((2, 4))))
        self.assertEqual(saved.to_bytes(), adapter.checkpoint().to_bytes())
        invalid = []
        prep = component(saved, "preprocessing")
        prep["scale"][0] = 0.0
        invalid.append(altered_checkpoint(saved, name="preprocessing", value=prep))
        optimizer = component(saved, "optimizer")
        optimizer["optimizer_steps"] = 1
        invalid.append(altered_checkpoint(saved, name="optimizer", value=optimizer))
        model = component(saved, "model")
        model["coefficients"] = [[0.0]]
        invalid.append(altered_checkpoint(saved, name="model", value=model))
        rng = component(saved, "rng")
        rng["state"]["bit_generator"] = "MT19937"
        invalid.append(altered_checkpoint(saved, name="rng", value=rng))
        metadata = saved.metadata
        metadata["not_applicable_components"] = []
        invalid.append(altered_checkpoint(saved, metadata=metadata))
        metadata = saved.metadata
        metadata["adapter"]["learning_settings"]["learning_rate"] = 0.5
        invalid.append(altered_checkpoint(saved, metadata=metadata))
        for bad in invalid:
            with self.assertRaises(AdapterError):
                adapter.restore(bad)
            self.assertEqual(saved.to_bytes(), adapter.checkpoint().to_bytes())

    def test_frozen_guard_refuses_all_learner_mutations(self):
        adapter = self.initialized()
        saved, frozen = adapter.checkpoint(), adapter.freeze()
        with inference_evaluation_guard() as ledger:
            for operation in (lambda: adapter.initialize(batch(), 1, {}),
                              lambda: adapter.configure_learning_settings({}),
                              lambda: adapter.update(batch()),
                              lambda: adapter.restore(saved),
                              lambda: adapter.act(np.zeros(14), explore=True)):
                with self.assertRaises(DatasetError):
                    operation()
            frozen.act(np.zeros(14))
            adapter.act(np.zeros(14), explore=False)
        self.assertEqual(ledger["forbidden_api_calls"], 5)
        self.assertEqual(saved.to_bytes(), adapter.checkpoint().to_bytes())


if __name__ == "__main__":
    unittest.main()
