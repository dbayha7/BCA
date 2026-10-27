import unittest

from thrusterchain.comparison import summarize_final, validate_partitions


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.parts = validate_partitions([1, 2], [], [10, 11, 12])
        self.a, self.b, self.dataset = "a" * 64, "b" * 64, "d" * 64

    def row(self, train, episode, value, policy=None):
        return {"algorithm": "td3bc", "training_seed": train,
                "dataset_sha256": self.dataset, "policy_sha256": policy or (self.a if train == 1 else self.b),
                "family": "big_kicks", "episode_seed": episode, "discounted_return": value}

    def plan(self, train, episodes):
        return {"algorithm": "td3bc", "training_seed": train,
                "family": "big_kicks", "episode_seeds": episodes}

    def test_unequal_episodes_do_not_reweight_trained_policies(self):
        rows = [self.row(1, 10, 0), self.row(2, 10, 10), self.row(2, 11, 10), self.row(2, 12, 10)]
        expected = [self.plan(1, [10]), self.plan(2, [10, 11, 12])]
        result = summarize_final(rows, self.parts, expected=expected, selection_mode="strict_offline", selection_interactions=0)
        summary = result["summaries"][0]
        self.assertEqual(summary["mean_of_run_means"], 5)
        self.assertEqual(summary["observed_training_runs"], 2)
        self.assertEqual(result["holdout_exposure"], "not established by seed separation")

    def test_leakage_duplicates_and_policy_mutation_refused(self):
        with self.assertRaises(ValueError): validate_partitions([1], [2], [2])
        with self.assertRaises(ValueError): validate_partitions([1], [], [10, 10])
        row = self.row(1, 10, 0)
        kw = dict(expected=[self.plan(1, [10, 11])], selection_mode="strict_offline", selection_interactions=0)
        with self.assertRaises(ValueError): summarize_final([row, row], self.parts, **kw)
        with self.assertRaises(ValueError): summarize_final([row, self.row(1, 11, 0, self.b)], self.parts, **kw)
        with self.assertRaises(ValueError): summarize_final([self.row(1, 2, 0)], self.parts, **kw)
        result = summarize_final([row], self.parts, **kw)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["summaries"][0]["mean_of_run_means"])
        self.assertEqual(result["summaries"][0]["missing_episodes"], [{"training_seed": 1, "episode_seed": 11}])

    def test_selection_budget_and_single_seed_uncertainty(self):
        row = self.row(1, 10, 2)
        expected = [self.plan(1, [10])]
        self.assertIsNone(summarize_final([row], self.parts, expected=expected, selection_mode="strict_offline", selection_interactions=0)["summaries"][0]["sample_sd_across_training_runs"])
        with self.assertRaises(ValueError): summarize_final([row], self.parts, expected=expected, selection_mode="simulator_assisted", selection_interactions=100)
        selected = validate_partitions([1], [3], [10])
        out = summarize_final([row], selected, expected=expected, selection_mode="simulator_assisted", selection_interactions=100)
        self.assertEqual(out["selection_interactions"], 100)


if __name__ == "__main__":
    unittest.main()
