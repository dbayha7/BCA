"""Real CPU updates, leakage contracts, and exact optimizer/PRNG resume checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.thrusterchain_diagnostics.continuous_data import generate_continuous_data
from experiments.thrusterchain_diagnostics.continuous import Objective
from experiments.thrusterchain_diagnostics import training as T


class TrainingContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = generate_continuous_data(n_episodes=44, seed=74).learner_rows
        cls.settings = T.TrainingSettings(updates=6, chunk=2, batch_size=8,
            calibration_size=36, bca_warmup=2, bca_refresh=2)
        cls.reserved = np.arange(36)
        cls.prepared = T.prepare_training(cls.rows, cls.settings,
                                         calibration_episode_ids=cls.reserved,
                                         data_metadata={"objective": Objective().as_dict()})

    def test_reserved_trajectories_normalization_and_horizon(self):
        prepared, rows = self.prepared, self.rows
        train, bank = prepared.training_indices, prepared.calibration_indices
        self.assertFalse(prepared.bca_refusal)
        self.assertEqual(len(bank), 36)
        self.assertEqual(len(np.unique(rows["episode_id"][bank])), 36)
        self.assertFalse(np.intersect1d(rows["episode_id"][train], self.reserved).size)
        np.testing.assert_array_equal(prepared.mean[13:], [0, 0])
        np.testing.assert_array_equal(prepared.std[13:], [1, 1])
        np.testing.assert_array_equal(np.asarray(prepared.train.obs)[:, 13:], rows["observation"][train, 13:].astype(np.float32))
        np.testing.assert_array_equal(prepared.mean[:13], rows["observation"][train, :13].astype(np.float32).mean(0))
        np.testing.assert_array_equal(prepared.train.done, rows["time"][train] == 99)
        self.assertEqual(T.make_args(self.settings, 7).discount, 1)

    def test_holdout_oracle_and_broken_transition_rejected(self):
        for change in (dict(split=np.full(len(self.rows["reward"]), "holdout")),
                       dict(oracle_q=np.zeros(len(self.rows["reward"]))),
                       dict(bootstrap_mask=np.ones(len(self.rows["reward"])))):
            with self.assertRaises(ValueError):
                T.prepare_training(dict(self.rows, **change), self.settings,
                                   calibration_episode_ids=self.reserved)
        broken = self.rows["next_observation"].copy()
        broken[0, 0] += 1
        with self.assertRaises(ValueError):
            T.validate_learner_rows(dict(self.rows, next_observation=broken), self.settings)
        with self.assertRaises(ValueError):
            T.prepare_training(self.rows, self.settings, data_metadata={"budget_topups": 1})

    def test_reservation_ids_survive_retention_without_topups(self):
        zero = generate_continuous_data(n_episodes=44, seed=74, retention="zero").learner_rows
        prepared = T.prepare_training(zero, self.settings, calibration_episode_ids=self.reserved)
        np.testing.assert_array_equal(prepared.manifest["reservation"]["reserved_episode_ids"], self.reserved)
        self.assertFalse(np.isin(zero["episode_id"][prepared.training_indices], self.reserved).any())
        self.assertEqual(prepared.manifest["budget_topups"], 0)
        # Removing every row of a reserved trajectory does not create replacements.
        keep = ~np.isin(zero["episode_id"], np.arange(5))
        reduced = {k: v[keep] for k, v in zero.items()}
        prepared = T.prepare_training(reduced, self.settings, calibration_episode_ids=self.reserved)
        self.assertEqual(len(prepared.bank.obs), 31)
        self.assertTrue(prepared.bca_refusal)
        self.assertEqual(prepared.manifest["reservation"]["absent_reserved_episode_ids"], list(range(5)))

    def test_native_host_update_trace_and_delayed_counters(self):
        import jax
        import jax.numpy as jnp
        import algorithms.td3_bc as base
        import algorithms.td3_bc_bca as posterior
        carry, models, args, config = T.initialize(self.settings, "td3_bc", 7)
        original = (carry[0], carry[1].native, carry[2])
        host_step = posterior.make_train_step(args, config, models, self.prepared.train)
        base_step = base.make_train_step(args, models[0].apply, models[1].apply, self.prepared.train, 1.)
        got, _ = jax.jit(lambda x: jax.lax.scan(host_step, x, None, length=2))(carry)
        expected, _ = jax.jit(lambda x: jax.lax.scan(base_step, x, None, length=2))(original)
        self.assertEqual(T.tree_sha256(got[1].native), T.tree_sha256(expected[1]))
        np.testing.assert_array_equal(got[0], expected[0])
        self.assertEqual(int(got[1].native.actor.step), 1)
        self.assertEqual(int(got[1].native.critic.step), 2)
        # A horizon terminal cannot read a target critic through the bootstrap.
        idx = np.flatnonzero(np.asarray(self.prepared.train.done))[:3]
        batch = jax.tree_util.tree_map(lambda x: x[idx], self.prepared.train)
        target = posterior.native_target(args, models, carry[1].native, batch, jax.random.PRNGKey(42))
        np.testing.assert_array_equal(target, batch.reward)
        bca, _, _, _ = T.initialize(self.settings, "bca", 7)
        self.assertEqual(T.tree_sha256(bca[1].native), T.tree_sha256(carry[1].native))
        np.testing.assert_array_equal(bca[0], carry[0])

    def test_bca_refresh_checkpoint_resume_and_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            full = Path(directory) / "full"
            split = Path(directory) / "split"
            result = T.train(self.prepared, self.settings, arm="bca", seed=8, out_dir=full)
            partial = T.train(self.prepared, self.settings, arm="bca", seed=8, out_dir=split, stop_after=2)
            self.assertEqual(partial.record["status"], "paused")
            resumed = T.train(self.prepared, self.settings, arm="bca", seed=8, out_dir=split)
            self.assertEqual(T.tree_sha256(result.carry), T.tree_sha256(resumed.carry))
            self.assertEqual(resumed.record["status"], "completed")
            self.assertEqual(resumed.record["counters"], dict(actor=3, critic=6, actor_target=3, critic_target=3, calibrator=6))
            self.assertEqual([x["update"] for x in resumed.record["refreshes"]], [2, 4])
            self.assertTrue(all(x["posterior_inputs_valid"] for x in resumed.record["refreshes"]))
            loaded = T.load_trained(split)
            raw = self.rows["observation"][:3]
            np.testing.assert_array_equal(loaded.policy(raw), result.policy(raw))
            self.assertEqual(loaded.critic(raw, loaded.policy(raw)).shape, (3, 2))
            self.assertTrue(np.all(np.abs(loaded.policy(raw)) <= 1))
            midpoint = T.load_trained(split, update=3)
            self.assertEqual(int(midpoint.carry[2]), 3)
            self.assertEqual(midpoint.record["status"], "snapshot")
            self.assertEqual(midpoint.record["final_update"], 3)
            self.assertEqual(midpoint.record["counters"]["critic"], 3)
            with self.assertRaises(ValueError):
                T.train(self.prepared, self.settings, arm="bca", seed=9, out_dir=split)
            checkpoint = split / resumed.record["checkpoint"]["file"]
            checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
            with self.assertRaises(ValueError):
                T.load_trained(split)

    def test_refused_bca_does_not_skip_host(self):
        settings = replace(self.settings, updates=2)
        prepared = T.prepare_training(self.rows, settings, calibration_episode_ids=np.arange(2))
        self.assertTrue(prepared.bca_refusal)
        with tempfile.TemporaryDirectory() as directory:
            bca = T.train(prepared, settings, arm="bca", seed=9, out_dir=Path(directory) / "bca")
            host = T.train(prepared, settings, arm="td3_bc", seed=9, out_dir=Path(directory) / "host")
            self.assertEqual(bca.record["status"], "refused")
            self.assertIsNone(bca.carry)
            self.assertEqual(host.record["status"], "completed")
            self.assertEqual(host.record["final_update"], 2)

    def test_native_invalid_update_preserves_optimizer_rng_and_counter(self):
        import jax
        import jax.numpy as jnp
        import algorithms.td3_bc_bca as posterior
        carry, models, args, config = T.initialize(self.settings, "td3_bc", 71)
        # Canonicalize Python integer TrainState counters before JIT converts
        # them to int32; unchanged values should compare with the same dtype.
        carry = jax.tree_util.tree_map(jnp.asarray, carry)
        invalid = self.prepared.train._replace(reward=jnp.full_like(self.prepared.train.reward, jnp.nan))
        step = jax.jit(posterior.make_train_step(args, config, models, invalid))
        after, metrics = step(carry, None)
        self.assertEqual(T.tree_sha256(after), T.tree_sha256(carry))
        with self.assertRaises(FloatingPointError):
            posterior.require_valid(metrics)


if __name__ == "__main__":
    unittest.main()
