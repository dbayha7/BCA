"""Episode-linked actions survive flattening, sampling, and online replay."""
import unittest

import numpy as np

from thrusterchain.datasets import generate_dataset
from thrusterchain.sgd_adapter import AffineSGDBCAdapter
from thrusterchain.training_protocol import EvaluationConfig, OfflineBudget, OnlineBudget, TransitionBatch
from thrusterchain.training_runner import train_transitions, train_offline, run_mode


class NextActionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = generate_dataset(episodes=10, seed=17991, provenance="smoke")

    def test_offline_pairs_before_flattening_and_random_sampling(self):
        source = self.data.split("train")
        batch = train_transitions(self.data)
        actions = source["actions"]
        n, horizon, dim = actions.shape
        np.testing.assert_array_equal(batch.actions.reshape(n, horizon, dim), actions)
        np.testing.assert_array_equal(batch.next_actions.reshape(n, horizon, dim)[:, :-1], actions[:, 1:])
        mask = batch.next_action_valid.reshape(n, horizon)
        self.assertTrue(mask[:, :-1].all())
        self.assertFalse(mask[:, -1].any())
        np.testing.assert_array_equal(batch.next_actions.reshape(n, horizon, dim)[:, -1], 0)
        self.assertFalse(batch.bootstrap_mask.reshape(n, horizon)[:, -1].any())
        indices = np.array([horizon - 1, 1, horizon, 1, 0])
        picked = batch.select(indices)
        np.testing.assert_array_equal(picked.next_actions, batch.next_actions[indices])
        np.testing.assert_array_equal(picked.next_action_valid, batch.next_action_valid[indices])
        mixed = picked.select([0, 1]).concat(picked.select([2, 3, 4]))
        np.testing.assert_array_equal(mixed.next_action_valid, picked.next_action_valid)
        np.testing.assert_array_equal(mixed.next_actions, picked.next_actions)
        with self.assertRaises(ValueError):
            picked.next_action_valid.setflags(write=True)

    def test_unknown_actions_and_boundary_validity_are_not_inferred(self):
        batch = train_transitions(self.data).select([0, 99])
        fields = {name: getattr(batch, name) for name in batch.__dataclass_fields__}
        fields.pop("next_actions")
        fields.pop("next_action_valid")
        missing = TransitionBatch(**fields)
        self.assertFalse(missing.next_action_valid.any())
        np.testing.assert_array_equal(missing.next_actions, 0)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            TransitionBatch(**fields, next_actions=batch.next_actions)
        with self.assertRaisesRegex(ValueError, "boundaries"):
            TransitionBatch(**fields, next_actions=batch.next_actions,
                            next_action_valid=np.array([True, True]))
        with self.assertRaises(TypeError):
            TransitionBatch(**fields, next_actions=batch.next_actions,
                            next_action_valid=np.array([1, 0]))

    def test_online_actions_link_only_after_next_command_is_executed(self):
        checkpoint, _ = train_offline(self.data, AffineSGDBCAdapter,
                                       budget=OfflineBudget(1, 4), seed=815)
        config = EvaluationConfig(False, 1, 872, OnlineBudget(101, 1, 0., 4, {}))
        result = run_mode(self.data, checkpoint, AffineSGDBCAdapter, config, families=("recorded",))
        rows = result["raw_adaptation"]["recorded"]["transition_records"]
        for i in range(99):
            self.assertTrue(rows[i]["next_action_valid"])
            np.testing.assert_array_equal(rows[i]["next_actions"], rows[i + 1]["actions"])
        self.assertFalse(rows[99]["next_action_valid"])
        self.assertFalse(rows[100]["next_action_valid"])
        np.testing.assert_array_equal(rows[99]["next_actions"], 0)
        np.testing.assert_array_equal(rows[100]["next_actions"], 0)
        self.assertNotEqual(rows[99]["group_ids"], rows[100]["group_ids"])

    def test_replay_does_not_expose_a_future_action_before_collection(self):
        checkpoint, _ = train_offline(self.data, AffineSGDBCAdapter,
                                       budget=OfflineBudget(1, 4), seed=816)
        instances = []
        class Capture(AffineSGDBCAdapter):
            def __init__(self):
                super().__init__()
                self.observed_batches = []
                instances.append(self)

            def update(self, batch):
                self.observed_batches.append(batch)
                return super().update(batch)

        config = EvaluationConfig(False, 1, 873, OnlineBudget(2, 4, 0., 32, {}))
        result = run_mode(self.data, checkpoint, Capture, config, families=("recorded",))
        rows = result["raw_adaptation"]["recorded"]["transition_records"]
        batches = instances[0].observed_batches
        self.assertEqual(len(batches), 4)
        for batch in batches[:2]:
            self.assertFalse(batch.next_action_valid.any())
            np.testing.assert_array_equal(batch.next_actions, 0)
        for batch in batches[2:]:
            self.assertTrue(batch.next_action_valid.any())
            self.assertTrue((~batch.next_action_valid).any())
            for action in batch.next_actions[batch.next_action_valid]:
                np.testing.assert_array_equal(action, rows[1]["actions"])
            np.testing.assert_array_equal(batch.next_actions[~batch.next_action_valid], 0)

    def test_continuing_cutoff_keeps_bootstrap_and_marks_next_action_unknown(self):
        data = generate_dataset(episodes=10, seed=17992, provenance="smoke",
                                task_semantics="continuing_time_limit")
        batch = train_transitions(data)
        self.assertTrue(batch.truncated[99])
        self.assertEqual(batch.bootstrap_mask[99], 1.)
        self.assertFalse(batch.next_action_valid[99])
        self.assertEqual(batch.next_observations[99, -1], 0.)
        checkpoint, _ = train_offline(data, AffineSGDBCAdapter,
                                       budget=OfflineBudget(1, 4), seed=817)
        config = EvaluationConfig(False, 1, 874, OnlineBudget(1, 1, 0., 4, {}))
        result = run_mode(data, checkpoint, AffineSGDBCAdapter, config, families=("recorded",))
        row = result["raw_adaptation"]["recorded"]["transition_records"][0]
        self.assertEqual(row["observations"][-1], 0.)
        self.assertEqual(row["next_observations"][-1], 0.)
        self.assertFalse(row["next_action_valid"])
        self.assertEqual(result["frozen_evaluation"]["recorded"]["environment_steps"], 100)


if __name__ == "__main__":
    unittest.main()
