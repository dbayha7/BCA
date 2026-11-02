"""V5 finite versus continuing semantics, including the actual cutoff successor."""
import unittest

import numpy as np

from thrusterchain.datasets import DatasetError, TrajectoryDataset, generate_dataset
from thrusterchain.environment import ThrusterChainEnv, dynamics
from thrusterchain.regular import evaluate_frozen_policy, train_behavior_cloning


class ContinuingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = generate_dataset(episodes=10, seed=777, task_semantics="continuing_time_limit")

    def test_dataset_cutoff_is_actual_stationary_successor(self):
        data = self.dataset.arrays
        self.assertEqual(self.dataset.metadata["schema"], "thrusterchain.continuous_trajectories.v3")
        self.assertEqual(self.dataset.metadata["observation_mode"], "physical_padded")
        np.testing.assert_array_equal(data["observations"][..., 13], 0)
        np.testing.assert_array_equal(data["observations"][..., :13], data["states"].astype(np.float32))
        self.assertFalse(data["terminated"].any())
        self.assertTrue(data["truncated"][:, -1].all())
        nxt, reward = dynamics(data["states"][:, -2], data["actions"][:, -1])
        np.testing.assert_allclose(nxt, data["states"][:, -1], rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(reward, data["rewards"][:, -1], rtol=1e-13, atol=1e-13)
        # One more physical step has the same representation and a defined successor.
        after, _ = dynamics(nxt, np.zeros((len(nxt), 4)))
        self.assertTrue(np.isfinite(after).all())
        self.assertEqual(np.column_stack((after.astype(np.float32), np.zeros(len(nxt), np.float32))).shape,
                         data["observations"][:, -1].shape)

    def test_incompatible_or_corrupt_versions_are_refused(self):
        arrays = {k: np.array(v) for k, v in self.dataset.arrays.items()}
        arrays["observations"][0, -1, 13] = 1
        with self.assertRaisesRegex(DatasetError, "padding coordinate"):
            TrajectoryDataset(arrays, self.dataset.metadata)
        metadata = self.dataset.metadata
        metadata["schema"] = "thrusterchain.continuous_trajectories.v2"
        with self.assertRaisesRegex(DatasetError, "unknown dataset schema"):
            TrajectoryDataset(self.dataset.arrays, metadata)
        with self.assertRaisesRegex(ValueError, "stationary physical"):
            ThrusterChainEnv(task_semantics="continuing_time_limit", observation_mode="time_aware")

    def test_environment_and_bc_evaluation_use_matching_observations(self):
        env = ThrusterChainEnv(task_semantics="continuing_time_limit", observation_mode="physical_padded")
        obs, _ = env.reset(seed=5)
        self.assertEqual(obs.shape, (14,))
        for _ in range(100):
            obs, _, term, trunc, info = env.step(np.zeros(4))
            self.assertEqual(obs[-1], 0.)
            self.assertTrue(env.observation_space.contains(obs))
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(info["bootstrap_mask"], 1.)
        np.testing.assert_array_equal(obs[:13], info["state"].astype(np.float32))
        policy, receipt = train_behavior_cloning(self.dataset)
        result = evaluate_frozen_policy(policy, receipt, episodes_per_family=1, seed=19)
        self.assertEqual(result["environment_episodes"], 4)
        self.assertEqual(result["parameter_updates"], 0)

    def test_claimed_final_and_research_labels_do_not_attest_holdout(self):
        dataset = generate_dataset(episodes=10, seed=778, provenance="research")
        policy, receipt = train_behavior_cloning(dataset)
        result = evaluate_frozen_policy(policy, receipt, episodes_per_family=1, seed=19)
        self.assertIsNone(result["eligible_strict_offline_final_reporting"])
        self.assertIsNone(result["eligible_research_evidence"])
        self.assertEqual(result["eligibility_status"], "unverified_prior_exposure")
        self.assertTrue(result["families"]["big_waves"]["held_out_from_dataset_generation"])
        tuned = evaluate_frozen_policy(policy, receipt, episodes_per_family=1, seed=19,
                                       purpose="simulator_assisted_tuning")
        self.assertFalse(tuned["eligible_strict_offline_final_reporting"])
        self.assertFalse(tuned["eligible_research_evidence"])


if __name__ == "__main__":
    unittest.main()
