"""Bounded checks of real IQL, CQL, and ReBRAC task-batch updates."""
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from algorithms import cql
from thrusterchain.datasets import generate_dataset
from host_integration.td3bc_probe import prepare
from host_integration.other_probes import (_rebrac_batch, iql_probe, cql_probe,
                                          rebrac_probe, validate_cql_settings)


class OtherHostProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset(episodes=10, seed=992, provenance="smoke")
        cls.indices = [0, 1, 99, 102, 199]

    def test_iql_real_update_and_numerical_losses(self):
        result = iql_probe(self.data, self.indices)
        self.assertEqual(result["counts"], {"actor": 1, "qf": 1, "qf_target": 1, "vf": 1})

    def test_cql_real_update_and_numerical_td_and_gaps(self):
        result = cql_probe(self.data, self.indices)
        self.assertTrue(all(v == 1 for v in result["counts"].values()))

    def test_cql_warmup_guard_and_demonstrated_data_independence(self):
        batch, _, _ = prepare(self.data, self.indices)
        args = replace(cql.Args(), bc_steps=1, discount=.98, batch_size=len(batch.reward))
        with self.assertRaisesRegex(ValueError, "does not score dataset actions"):
            validate_cql_settings(args)
        rng, state, (actor, c1, c2) = cql.initialize(args, batch.obs.shape[-1], 4, 1.)
        update = jax.jit(lambda sample: cql.cql_update(
            args, actor.apply, c1.apply, c2.apply, state, sample, 1, rng, 1., -4.)[1])
        original = update(batch)
        replacement = update(batch._replace(action=jnp.clip(-batch.action, -1, 1)))
        self.assertGreater(np.max(np.abs(np.asarray(batch.action) - np.asarray(-batch.action))), .1)
        self.assertEqual(float(original["policy_loss"]), float(replacement["policy_loss"]))

    def test_rebrac_true_next_actions_and_terminal_rows(self):
        batch, valid = _rebrac_batch(self.data, self.indices)
        episodes = self.data.split("train")["actions"]
        np.testing.assert_array_equal(np.asarray(batch.next_action)[0], episodes[0, 1])
        np.testing.assert_array_equal(np.asarray(batch.next_action)[3], episodes[1, 3])
        np.testing.assert_array_equal(valid, [True, True, False, True, False])
        result = rebrac_probe(self.data, self.indices)
        self.assertEqual(result["valid_next_actions"], 3)
        self.assertEqual(result["terminal_placeholders"], 2)
        self.assertEqual(result["counts"], {"critic": 2, "actor": 1})

    def test_rebrac_refuses_missing_bootstrapped_next_action(self):
        fake = SimpleNamespace(next_action_valid=np.array([False]),
                               bootstrap_mask=np.array([1.0]), next_actions=np.zeros((1, 4)))
        with patch("host_integration.other_probes.train_transitions", return_value=fake):
            with self.assertRaisesRegex(ValueError, "every bootstrapped row"):
                _rebrac_batch(self.data, [0])


if __name__ == "__main__":
    unittest.main()
