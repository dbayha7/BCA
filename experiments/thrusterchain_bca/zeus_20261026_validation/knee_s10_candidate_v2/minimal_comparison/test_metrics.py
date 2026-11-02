"""Explicit synthetic arrays only: no production data/model/simulator access."""
import math
import unittest

from metrics import across_seeds, paired_returns, score_summary


class MetricTests(unittest.TestCase):
    def test_help_and_harm_use_same_truth(self):
        q = [[3, 2, 0, 9], [3, 2, 0, 9]]
        truth = [[0, 2, 0, 8], [3, 1, 0, 8]]
        result = score_summary(q, truth, [[1, 0, 0, 0], [1, 0, 0, 0]])
        self.assertEqual(result["common_bank"]["raw_indices"], [0, 0])
        self.assertEqual(result["common_bank"]["adjusted_indices"], [1, 1])
        self.assertEqual(result["common_bank"]["beneficial_switches"], 1)
        self.assertEqual(result["common_bank"]["harmful_switches"], 1)
        self.assertEqual(result["common_bank"]["mean_selected_value_gain"], 0)
        self.assertEqual(result["actor"]["raw"]["count"], 2)
        self.assertEqual(result["common"]["raw"]["count"], 6)
        self.assertEqual(result["all"]["raw"]["count"], 8)

    def test_zero_penalty_exact_parity(self):
        result = score_summary([[3, 2, 1, 4]], [[1, 2, 3, 4]], [[99, 2, 1, 0]], kappa=0)
        self.assertEqual(result["all"]["raw"], result["all"]["adjusted"])
        self.assertEqual(result["common_bank"]["switches"], 0)

    def test_zero_width_parity(self):
        result = score_summary([[3, 2, 1, 4]], [[1, 2, 3, 4]], [[0, 0, 0, 0]])
        self.assertEqual(result["all"]["raw"], result["all"]["adjusted"])

    def test_ties_first_index_and_actor_excluded(self):
        result = score_summary([[1, 1 + 5e-11, 0, 100]], [[1, 1, 0, 100]], [[0, 0, 0, 0]])
        self.assertEqual(result["common_bank"]["raw_indices"], [0])
        self.assertEqual(result["common_bank"]["raw_optimal_count"], 1)

    def test_tolerance_and_crossing(self):
        result = score_summary([[.15, -.15, .2, -.2]], [[0, 0, 0, 0]], [[0, 0, .2, 0]])
        self.assertEqual(result["all"]["raw"]["over_count"], 1)
        self.assertEqual(result["all"]["raw"]["under_count"], 1)
        self.assertEqual(result["all"]["raw"]["near_zero_count"], 2)
        self.assertEqual(result["all"]["adjustment"]["over_to_under"], 1)
        self.assertEqual(result["all"]["adjustment"]["absolute_error_worsened"], 1)

    def test_score_scaling_identity(self):
        a = score_summary([[4, 3, 0, 0]], [[2, 3, 0, 0]], [[.5, 0, 0, 0]], kappa=3)
        b = score_summary([[4, 3, 0, 0]], [[2, 3, 0, 0]], [[1.5, 0, 0, 0]], kappa=1)
        self.assertEqual(a["common_bank"], b["common_bank"])
        self.assertEqual(a["all"], b["all"])

    def test_nonfinite_and_negative_width_refuse(self):
        for x in (math.nan, math.inf, -math.inf, -1, True):
            with self.subTest(x=x), self.assertRaises(ValueError):
                score_summary([[1, 2, 3, 4]], [[1, 2, 3, 4]], [[x, 0, 0, 0]])

    def test_shapes_and_empty_refuse(self):
        for q, truth, width in (([], [], []), ([[1, 2]], [[1, 2]], [[0, 0]]),
                                ([[1, 2, 3, 4]], [[1, 2, 3]], [[0, 0, 0, 0]]),
                                ([[1], [1, 2]], [[1], [1, 2]], [[0], [0, 0]])):
            with self.subTest(q=q), self.assertRaises(ValueError):
                score_summary(q, truth, width)

    def test_arithmetic_overflow_refuse(self):
        with self.assertRaises(ValueError):
            score_summary([[1e308] * 4], [[-1e308] * 4], [[0] * 4])

    def test_paired_returns_identity(self):
        result = paired_returns([1, 3], [2, 2], [10, 11], [10, 11], expected_count=2)
        self.assertEqual(result["differences"], [1, -1])
        self.assertEqual(result["mean_difference"], 0)

    def test_missing_reordered_duplicate_episodes_refuse(self):
        for ids in ([11, 10], [10, 10], [10], [True, 11]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                paired_returns([1, 3], [2, 2], [10, 11], ids, expected_count=2)
        with self.assertRaises(ValueError):
            paired_returns([1], [2], [1], [1], expected_count=2)

    def test_seed_aggregation_keeps_negative_seed(self):
        result = across_seeds({1: {"mean_difference": 2}, 2: {"mean_difference": 1},
                               3: {"mean_difference": -6}}, expected_seeds=[1, 2, 3])
        self.assertEqual(result["mean_seed_difference"], -1)
        self.assertEqual(result["negative_seeds"], 1)
        with self.assertRaises(ValueError):
            across_seeds({1: {"mean_difference": 2}}, expected_seeds=[1, 2, 3])


if __name__ == "__main__":
    unittest.main()
