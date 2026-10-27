"""Bounded tests for the actual pinned unified computational definitions."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from thrusterchain.datasets import generate_dataset
from thrusterchain.training_protocol import EvaluationConfig, LearnerCheckpoint, OfflineBudget
from thrusterchain.training_runner import train_offline, train_transitions, run_mode
from host_integration.unifloral_adapter import (HOST, UnifiedTD3BCAdapter, _args, _initialize,
    _native, _run_step, _runtime, _transition)


SETTINGS = {"allow_outside_upstream_constraints": True}


class UnifiedAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=1097, provenance="smoke")
        cls.train = train_transitions(cls.dataset)

    def test_native_internal_sampling_vs_external_selected_batch(self):
        model = UnifiedTD3BCAdapter()
        model.initialize(self.train, 13, SETTINGS)
        pool = self.train.select([0, 1, 2, 3, 4, 98, 99])
        batch_size = 5
        args = replace(model._args, batch_size=batch_size)
        _, dataset_key, _ = jax.random.split(model._rng, 3)
        indices = np.asarray(jax.random.randint(dataset_key, (batch_size,), 0, len(pool)))
        selected = pool.select(indices)
        original = _native(False)
        (ref_rng, ref_state, _), ref_losses = _run_step(original, args, model._actor, model._critic,
                                                        model._rng, model._state, _transition(pool))

        obs = jnp.asarray(selected.observations, jnp.float32)
        actions = np.asarray(selected.actions, np.float32)
        pi = jax.vmap(lambda o: model._actor.apply(model._state.actor.params, o).mean())(obs)
        pi_np = np.asarray(pi)
        q = np.asarray(model._critic.apply(model._state.vec_q.params, obs, pi))[:, 0]
        expected_actor = np.mean(np.sum((pi_np - actions) ** 2, axis=-1)) - 2.5 * np.mean(q) / (np.mean(np.abs(q)) + 1e-7)
        actual_losses = model.update(selected)
        np.testing.assert_allclose(actual_losses["actor_loss"], expected_actor, rtol=3e-5, atol=3e-5)
        for name in ref_losses:
            np.testing.assert_allclose(actual_losses[name], np.asarray(ref_losses[name]), rtol=3e-5, atol=3e-5)
        for actual, expected in zip(jax.tree_util.tree_leaves(model._state), jax.tree_util.tree_leaves(ref_state)):
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(model._rng, ref_rng)
        self.assertEqual(model.optimizer_step_counts, {"actor": 1, "critic": 2})
        self.assertEqual(int(model._state.actor_target.step), 1)
        self.assertEqual(int(model._state.vec_q_target.step), 1)

    def test_runner_exact_resume_and_frozen_evaluation(self):
        full, receipt = train_offline(self.dataset, UnifiedTD3BCAdapter, budget=OfflineBudget(2, 7),
                                      seed=842, learning_settings=SETTINGS)
        one, _ = train_offline(self.dataset, UnifiedTD3BCAdapter, budget=OfflineBudget(1, 7),
                               seed=842, learning_settings=SETTINGS)
        one = LearnerCheckpoint.from_bytes(one.to_bytes())
        resumed, _ = train_offline(self.dataset, UnifiedTD3BCAdapter, budget=OfflineBudget(1, 7),
                                   seed=842, learning_settings=SETTINGS, resume_from=one)
        self.assertEqual(full.to_bytes(), resumed.to_bytes())
        self.assertEqual(dict(full.optimizer_step_counts), {"actor": 2, "critic": 4})
        self.assertEqual(receipt["primary_optimizer"], "actor")
        self.assertEqual(receipt["environment_interactions"], 0)
        with TemporaryDirectory() as tmp:
            result = run_mode(self.dataset, full, UnifiedTD3BCAdapter,
                config=EvaluationConfig(frozen_policy=True, episodes=1, seed=843),
                families=("recorded",), checkpoint_directory=Path(tmp) / "frozen")
            self.assertEqual(result["receipt"]["frozen_evaluation_steps"], 100)
            self.assertEqual(result["receipt"]["adaptation_updates"], 0)
            self.assertEqual(result["receipt"]["evaluation_checkpoint_source"], "reloaded_verified_checkpoint")

    def test_seed_width_runtime_gate_and_terminal_masks(self):
        if not _runtime()["upstream_constraints_met"]:
            with self.assertRaisesRegex(ValueError, "outside upstream"):
                UnifiedTD3BCAdapter().initialize(self.train, 13, {})
        a, b = UnifiedTD3BCAdapter(), UnifiedTD3BCAdapter()
        a.initialize(self.train, 13, SETTINGS)
        b.initialize(self.train, 2**32 + 13, SETTINGS)
        self.assertFalse(np.array_equal(a._rng, b._rng))
        low_key, low_state, _, _ = _initialize(_args(13), a._mean, a._std)
        np.testing.assert_array_equal(low_key, a._rng)
        for actual, expected in zip(jax.tree_util.tree_leaves(a._state), jax.tree_util.tree_leaves(low_state)):
            np.testing.assert_array_equal(actual, expected)
        batch = self.train.select([0, 99])
        tr = _transition(batch)
        np.testing.assert_array_equal(tr.done, [0, 1])
        timed = replace(batch, truncated=np.array([True, False]),
                        next_action_valid=np.array([False, False]), next_actions=np.zeros_like(batch.next_actions))
        np.testing.assert_array_equal(_transition(timed).done, [0, 1])
        with self.assertRaises(ValueError):
            UnifiedTD3BCAdapter().initialize(self.train, 2**64, SETTINGS)

    def test_independent_critic_target_terminal_and_time_limit(self):
        model = UnifiedTD3BCAdapter()
        model.initialize(self.train, 17, SETTINGS)
        selected = self.train.select([0, 99])
        # The first transition is an external time limit with a true successor;
        # the second is a true task termination and must have no continuation.
        selected = replace(selected, truncated=np.array([True, False]),
                           next_action_valid=np.array([False, False]),
                           next_actions=np.zeros_like(selected.next_actions))
        batch = _transition(selected)
        args = replace(model._args, batch_size=2, num_critic_updates_per_step=1)
        rng, _, _ = jax.random.split(model._rng, 3)
        rng, _ = jax.random.split(rng)  # actor update
        _, target_rng = jax.random.split(rng)  # first critic update
        target_keys = jax.random.split(target_rng, 2)

        def target_action(key, next_obs):
            _, action_key, noise_key = jax.random.split(key, 3)
            pi = model._actor.apply(model._state.actor_target.params, next_obs)
            action = pi.sample(seed=action_key)
            noise = jnp.clip(jax.random.normal(noise_key, action.shape) * args.policy_noise,
                             -args.noise_clip, args.noise_clip)
            return jnp.clip(action + noise, -1, 1)

        target_actions = jax.vmap(target_action)(target_keys, batch.next_obs)
        heads = np.asarray(model._critic.apply(model._state.vec_q_target.params, batch.next_obs, target_actions))
        self.assertGreater(np.max(np.abs(heads[:, 0] - heads[:, 1])), 1e-8)
        bootstrap = 1.0 - np.asarray(batch.done)
        target = np.asarray(batch.reward) + args.gamma * bootstrap * heads.min(axis=-1)
        self.assertEqual(target[1], float(batch.reward[1]))
        self.assertNotEqual(target[0], float(batch.reward[0]))
        predicted = np.asarray(model._critic.apply(model._state.vec_q.params, batch.obs, batch.action))
        expected_loss = np.mean(np.sum(np.square(predicted - target[:, None]), axis=-1))
        # One critic step exposes the independently reconstructed first loss;
        # the supported adapter remains the declared two-critic-step composition.
        (_, _, _), losses = _run_step(HOST, args, model._actor, model._critic,
                                      model._rng, model._state, batch)
        np.testing.assert_allclose(losses["critic_loss"], expected_loss, rtol=3e-5, atol=3e-5)


if __name__ == "__main__":
    unittest.main()
